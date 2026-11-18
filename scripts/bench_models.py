"""Benchmark extraction models: latency, throughput, yield, and memory.

Runs the real `phase2.v1` combined-strategy call (same prompt/pipeline as
`horizon gather --scan`) and records per-item wall time, token throughput, parse
success, and peak process memory. Works on CPU (Intel-Mac dev), MPS
(Apple/Metal), and CUDA (a GPU cluster) — pass `--device cuda` there for real
GPU numbers.

This is the tool that picks the production model: it sweeps a grid of
(model x batch size x quantization) cells and reports **seconds per validated
analysis** for each. That, not raw speed, is the decision metric — a faster model
that needs repair retries costs more per usable analysis.

Two item sources:

- **corpus mode** (`--records`) — a length-stratified, deterministic sample of
  built dataset records (`horizon collection-build` shards). At batch 1 each item
  goes through `extract.analyze_text`, so timings include the repair retry and
  the measurement reports how often extraction actually validated. This is the
  mode to use for "how long would the whole corpus take".
- **legacy mode** (default) — one short and one long item, timing raw
  `client.generate` only. Defaults to committed `examples/` texts.

Each (model, quant) combo runs in its own subprocess so peak RSS isolates that
model; batch sizes are swept *inside* the child, so a large model is loaded once.
Per-item rows are appended to `--out` as they finish, so a run that is cut short
still leaves usable data — including a row for a cell that ran out of memory.

Usage:
    python scripts/bench_models.py --device cpu
    python scripts/bench_models.py --probe            # VRAM + which variants fit
    python scripts/bench_models.py --records datasets/wos-scopus/records \
        --items 8 --out /tmp/bench.jsonl
    # The GPU sweep: every Gemma 4 variant that fits, across batch sizes.
    python scripts/bench_models.py --device cuda --preset fit \
        --records datasets/wos-scopus/records --items 8 \
        --batch 1 4 8 16 --quant none 4bit --out bench-cuda.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-1.7B"]
# Legacy-mode fallbacks: committed, so a fresh clone/cluster checkout works. The old
# defaults pointed into gitignored datasets/tech/corpus and broke outside this machine.
SHORT = "examples/sample.txt"
LONG = "examples/sample_policy.txt"
DEFAULT_MAX_NEW_TOKENS = 512
# Set per-run from --max-new-tokens; the child reads it to flag truncated replies.
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS

# Gemma 4 variants with their *estimated* VRAM.
# These are parameter-count arithmetic, NOT measurements: activations, fragmentation
# and the KV cache at batch>1 are not in them, so expect them to run low. The number
# to trust is the measured gpu_peak_gb this script reports.
GEMMA4_VRAM_GB = {
    "google/gemma-4-E2B-it": {"params_b": 5, "bf16": 12, "4bit": 4},
    "google/gemma-4-E4B-it": {"params_b": 8, "bf16": 18, "4bit": 6},
    "google/gemma-4-12B-it": {"params_b": 12, "bf16": 28, "4bit": 9},
    "google/gemma-4-26B-A4B-it": {"params_b": 27, "bf16": 62, "4bit": 18},
    "google/gemma-4-31B-it": {"params_b": 33, "bf16": 74, "4bit": 22},
}
GEMMA4_MODELS = list(GEMMA4_VRAM_GB)
# Leave headroom for activations and the KV cache; the estimates above cover weights.
VRAM_HEADROOM = 0.9
# Candidates drawn per shard before stratifying — big enough to cover the length
# distribution, small enough to keep selection to a fraction of a second.
POOL_PER_SHARD = 2000


# --- item selection -----------------------------------------------------------


def record_text(record: dict) -> str:
    """The scanned text of a built record — the same `title. abstract` `WosSource` emits."""
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"{title}. {abstract}" if abstract else title


def select_items(records: str, count: int, seed: int) -> list[dict]:
    """Pick ``count`` records spread across the corpus length distribution.

    Deterministic in ``(records, count, seed)``: a seeded pool is drawn from every
    shard, sorted by text length, and sampled at evenly spaced quantile midpoints.
    Stratifying by length (rather than taking a plain random sample) keeps a small
    n from over- or under-stating mean latency, which scales with token count.
    """
    from horizon.analysis import content_hash

    root = Path(records)
    shards = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    rng = random.Random(seed)
    pool: list[dict] = []
    for shard in shards:
        rows = []
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                text = record_text(record)
                if not record.get("abstract"):
                    continue  # title-only records are not representative of scan cost
                rows.append(
                    {
                        "shard": shard.stem,
                        "chars": len(text),
                        "text": text,
                        # Built shards carry no hash (WosSource computes it at
                        # gather time); recompute it so a benchmarked item can be
                        # joined to its store analysis.
                        "content_hash": content_hash(text),
                        "year": record.get("year"),
                        "doi": record.get("doi"),
                    }
                )
                if len(rows) >= POOL_PER_SHARD:
                    break
        rng.shuffle(rows)
        pool.extend(rows[: POOL_PER_SHARD // max(len(shards), 1) or 1])
    # Stable sort: equal-length records keep the seeded pool order, so the
    # selection stays reproducible without depending on record identity.
    pool.sort(key=lambda r: r["chars"])
    if not pool:
        raise SystemExit(f"no records with an abstract found under {records}")
    picked = []
    for i in range(count):
        idx = min(int((i + 0.5) / count * len(pool)), len(pool) - 1)
        picked.append(pool[idx])
    return picked


# --- GPU probe / variant selection --------------------------------------------


def gpu_info() -> dict | None:
    """Total/free VRAM and driver versions for device 0, or ``None`` without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "name": torch.cuda.get_device_properties(0).name,
            "count": torch.cuda.device_count(),
            "total_gb": round(total / 1e9, 1),
            "free_gb": round(free / 1e9, 1),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"warning: could not probe the GPU: {exc}", file=sys.stderr)
        return None


def fits(budget_gb: float) -> list[tuple[str, bool, bool]]:
    """Per Gemma 4 variant: ``(model_id, fits_bf16, fits_4bit)`` under ``budget_gb``."""
    return [
        (model, est["bf16"] <= budget_gb, est["4bit"] <= budget_gb)
        for model, est in GEMMA4_VRAM_GB.items()
    ]


def print_probe() -> None:
    """Report the card and which Gemma 4 variants fit it, then how to sweep them."""
    info = gpu_info()
    if info is None:
        print("no CUDA device visible — the fit table below assumes nothing.")
        budget = 0.0
    else:
        print(
            f"device 0: {info['name']}  {info['total_gb']} GB total, "
            f"{info['free_gb']} GB free  (compute {info['capability']}, "
            f"{info['count']} device(s))"
        )
        print(f"torch {info['torch']} / cuda {info['cuda']}")
        budget = info["free_gb"] * VRAM_HEADROOM
        print(f"usable budget ({VRAM_HEADROOM:.0%} of free): {budget:.1f} GB\n")
    print(
        f"{'variant':<28} {'params':>7} {'bf16':>6} {'fits':>5} {'4-bit':>6} {'fits':>5}"
    )
    for model, bf16_ok, q4_ok in fits(budget):
        est = GEMMA4_VRAM_GB[model]
        print(
            f"{model:<28} {est['params_b']:>6}B {est['bf16']:>5} GB "
            f"{'\u2713' if bf16_ok else '\u2717':>5} {est['4bit']:>4} GB "
            f"{'\u2713' if q4_ok else '\u2717':>5}"
        )
    print(
        "\nEstimates — predictions, not measurements; they\n"
        "exclude activations and the KV cache, which is what actually OOMs at batch>1.\n"
        "Sweep the cells that fit:  --preset fit --batch 1 4 8 16 --quant none 4bit"
    )


def resolve_models(preset: str, explicit: list[str] | None) -> list[str]:
    """The model list for this run: explicit ``--models`` wins over ``--preset``."""
    if explicit:
        return explicit
    if preset == "gemma4":
        return GEMMA4_MODELS
    if preset == "fit":
        info = gpu_info()
        budget = (info["free_gb"] * VRAM_HEADROOM) if info else 0.0
        chosen = [m for m, bf16_ok, q4_ok in fits(budget) if bf16_ok or q4_ok]
        if not chosen:
            raise SystemExit(
                "no Gemma 4 variant fits the visible VRAM even 4-bit; run --probe "
                "and pick --models explicitly."
            )
        skipped = [m for m in GEMMA4_MODELS if m not in chosen]
        if skipped:
            print(f"--preset fit skipped (won't fit): {', '.join(skipped)}")
        return chosen
    return DEFAULT_MODELS


def load_text(path: str) -> str:
    """Read a bench item: a corpus JSON record, or a plain .txt file."""
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(raw)["text"]
    return raw


# --- child: one (model, device) combo -----------------------------------------


def _instrumented(client_cls):
    """Wrap a client so every model call records its own tokens and wall time."""

    class Instrumented(client_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls: list[dict] = []

        def generate(self, messages, *, config=None):
            start = time.perf_counter()
            out = super().generate(messages, config=config)
            self._record([out], time.perf_counter() - start)
            return out

        def generate_batch(self, conversations, *, config=None):
            # Without this, a batched cell's timings would be invisible: the whole
            # point of the sweep is per-batch throughput.
            start = time.perf_counter()
            outs = super().generate_batch(conversations, config=config)
            self._record(outs, time.perf_counter() - start)
            return outs

        def _record(self, outputs, elapsed):
            for out in outputs:
                n_out = len(self._tokenizer(out)["input_ids"])
                self.calls.append(
                    {
                        # Wall time is per *call*, so a batched call attributes the
                        # same elapsed to each row; aggregate at the cell level.
                        "s": round(elapsed, 2),
                        "out_tok": n_out,
                        "truncated": n_out >= MAX_NEW_TOKENS,
                        "batch": len(outputs),
                    }
                )

    return Instrumented


def _cell(client, settings, items, batch: int) -> dict:
    """Measure one (model, quant, batch) cell over ``items``.

    Batch 1 runs the real ``analyze_text`` path so the numbers stay comparable to the
    existing CPU baseline (repairs included, yield measured). Batch >1 issues the same
    combined prompt through ``generate_batch`` and validates each reply with the shared
    evaluator, which measures yield without the per-item repair loop.
    """
    import torch

    from horizon import prompts
    from horizon.llm.base import Message
    from horizon.training.evaluate import _safe_fields

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = len(client.calls)
    rows: list[dict] = []
    started = time.perf_counter()

    if batch == 1:
        from horizon.extract import AnalysisError, analyze_text

        for i, item in enumerate(items):
            call_start = len(client.calls)
            t0 = time.perf_counter()
            error = None
            try:
                analyze_text(
                    item["text"], client, settings=settings, config=settings.generation
                )
            except (AnalysisError, RuntimeError) as exc:
                error = str(exc)[:200]
            calls = client.calls[call_start:]
            rows.append(
                {
                    "i": i,
                    "content_hash": item.get("content_hash"),
                    "year": item.get("year"),
                    "chars": item["chars"],
                    "s": round(time.perf_counter() - t0, 2),
                    "calls": len(calls),
                    "out_tok": sum(c["out_tok"] for c in calls),
                    "truncated": any(c["truncated"] for c in calls),
                    "ok": error is None,
                    "error": error,
                    "path": "analyze_text",
                }
            )
    else:
        conversations = [
            [
                Message("system", prompts.EXTRACT_SYSTEM_PROMPT),
                Message(
                    "user",
                    prompts.COMBINED_PROMPT_TEMPLATE.format(text=item["text"].strip()),
                ),
            ]
            for item in items
        ]
        outputs = client.generate_batch(conversations, config=settings.generation)
        calls = client.calls[before:]
        for i, (item, out, call) in enumerate(zip(items, outputs, calls)):
            rows.append(
                {
                    "i": i,
                    "content_hash": item.get("content_hash"),
                    "year": item.get("year"),
                    "chars": item["chars"],
                    "s": round(call["s"] / max(len(items), 1), 2),
                    "calls": 1,
                    "out_tok": call["out_tok"],
                    "truncated": call["truncated"],
                    "ok": _safe_fields(out) is not None,
                    "error": None,
                    "path": "batch_raw",
                }
            )

    total_s = time.perf_counter() - started
    ok = [r for r in rows if r["ok"]]
    cell = {
        "batch": batch,
        "n": len(rows),
        "n_ok": len(ok),
        "total_s": round(total_s, 1),
        "mean_s": round(total_s / max(len(rows), 1), 1),
        # The decision metric: a fast model that needs repairs costs more per usable
        # analysis than a slower one that validates first time.
        "s_per_ok_item": (round(total_s / len(ok), 1) if ok else None),
        "tok_s": round(sum(r["out_tok"] for r in rows) / max(total_s, 1e-9), 2),
        "repairs": sum(1 for r in rows if r["calls"] > 1),
        "rows": rows,
    }
    if torch.cuda.is_available():
        cell["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return cell


def _is_oom(exc: BaseException) -> bool:
    """Whether a failure is an out-of-memory condition rather than a real error."""
    return "out of memory" in str(exc).lower()


def _run_child(
    model_id: str,
    device: str,
    items_path: str | None,
    out_path: str | None,
    *,
    quant: str = "none",
    batches: tuple[int, ...] = (1,),
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    global MAX_NEW_TOKENS

    import torch

    from horizon import prompts
    from horizon.config import GenerationConfig, Settings
    from horizon.llm.base import Message
    from horizon.llm.transformers_client import TransformersClient

    MAX_NEW_TOKENS = max_new_tokens
    settings = Settings(
        model_id=model_id,
        device=device,
        load_in_4bit=(quant == "4bit"),
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
    )
    client = _instrumented(TransformersClient)(settings)
    t = time.time()
    client._ensure_loaded()
    load_s = time.time() - t
    params = sum(p.numel() for p in client._model.parameters())

    res: dict = {
        "model": model_id,
        "device": device,
        "quant": quant,
        "max_new_tokens": max_new_tokens,
        "load_s": round(load_s, 1),
        "params_m": round(params / 1e6, 1),
        "torch": torch.__version__,
        "cells": [],
        "items": {},
    }
    sink = open(out_path, "a", encoding="utf-8") if out_path else None

    def emit(row: dict) -> None:
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

    if items_path:  # corpus mode: sweep the batch axis over the sampled items
        items = json.loads(Path(items_path).read_text(encoding="utf-8"))
        for batch in batches:
            settings = replace(
                settings, batch_size=batch
            )  # client reads batch_size at call time
            client.settings = settings
            print(f"  -- batch {batch} --", flush=True)
            try:
                cell = _cell(client, settings, items, batch)
            except Exception as exc:
                oom = _is_oom(exc)
                emit(
                    {
                        "model": model_id,
                        "device": device,
                        "quant": quant,
                        "batch": batch,
                        "ok": False,
                        "oom": oom,
                        "error": str(exc)[:300],
                    }
                )
                print(
                    f"  batch {batch}: {'OOM' if oom else 'FAILED'} — {str(exc)[:160]}",
                    flush=True,
                )
                if oom:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    # Larger batches cannot help once one has run out of memory.
                    break
                continue
            for row in cell.pop("rows"):
                emit(
                    {
                        "model": model_id,
                        "device": device,
                        "quant": quant,
                        **row,
                        "batch": batch,
                    }
                )
                print(
                    f"  item {row['i'] + 1}/{cell['n']}  {row['chars']:>5} ch  "
                    f"{row['s']:6.1f}s  {row['out_tok']:>3} tok  "
                    f"{'ok' if row['ok'] else 'FAIL'}",
                    flush=True,
                )
            res["cells"].append(cell)
            print(
                f"  batch {batch}: {cell['n_ok']}/{cell['n']} ok, "
                f"{cell['mean_s']}s/item, {cell['tok_s']} tok/s, "
                f"s_per_ok_item={cell['s_per_ok_item']}"
                + (
                    f", gpu_peak={cell['gpu_peak_gb']}GB"
                    if "gpu_peak_gb" in cell
                    else ""
                ),
                flush=True,
            )
    else:  # legacy mode: raw generate on one short and one long item
        for name, path in [("short", SHORT), ("long", LONG)]:
            txt = load_text(path)
            prompt = prompts.COMBINED_PROMPT_TEMPLATE.format(text=txt.strip())
            msgs = [
                Message("system", prompts.EXTRACT_SYSTEM_PROMPT),
                Message("user", prompt),
            ]
            t = time.time()
            client.generate(msgs, config=settings.generation)
            dt = time.time() - t
            n_out = client.calls[-1]["out_tok"]
            res["items"][name] = {
                "s": round(dt, 1),
                "out_tok": n_out,
                "tok_s": round(n_out / dt, 1),
            }

    # macOS ru_maxrss is bytes; Linux (the cluster) reports kilobytes.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res["peak_rss_gb"] = round(maxrss / (1e9 if sys.platform == "darwin" else 1e6), 2)
    if device == "cuda":
        res["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    elif device == "mps":
        try:
            res["mps_alloc_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        except Exception as exc:  # pragma: no cover
            res["mps_alloc_gb"] = f"n/a ({exc})"
    line = "RESULT " + json.dumps(res, ensure_ascii=False)
    if sink:
        sink.write(line + "\n")
        sink.close()
    print(line)


def main() -> None:
    # Keep the tokenizer's own threads out of the measurement, and don't phone home.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model ids (wins over --preset).",
    )
    ap.add_argument(
        "--preset",
        choices=["fallback", "gemma4", "fit"],
        default="fallback",
        help="Model set: fallback (the small stand-ins), gemma4 (every variant), or "
        "fit (only the Gemma 4 variants the visible VRAM can hold).",
    )
    ap.add_argument(
        "--quant",
        nargs="+",
        choices=["none", "4bit"],
        default=["none"],
        help="Quantization axis. 4bit is CUDA-only and is what makes a large model "
        "fit a small card.",
    )
    ap.add_argument(
        "--batch",
        nargs="+",
        type=int,
        default=[1],
        help="Batch-size axis — the GPU throughput lever. Batch 1 uses the real "
        "analyze_text path; larger batches measure raw batched generation.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation cap (default {DEFAULT_MAX_NEW_TOKENS}). Cost is set by output "
        "tokens, so this is the single biggest knob on total runtime.",
    )
    ap.add_argument("--records", help="dataset records dir/shard to sample items from")
    ap.add_argument(
        "--items", type=int, default=8, help="items to sample in corpus mode"
    )
    ap.add_argument("--seed", type=int, default=7, help="selection seed (corpus mode)")
    ap.add_argument("--out", help="append per-item JSONL results here")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Print VRAM and which Gemma 4 variants fit, then exit.",
    )
    ap.add_argument(
        "--legacy-short", help="Short item for legacy mode (txt or corpus json)."
    )
    ap.add_argument(
        "--legacy-long", help="Long item for legacy mode (txt or corpus json)."
    )
    ap.add_argument(
        "--child", nargs=2, metavar=("MODEL", "DEVICE"), help=argparse.SUPPRESS
    )
    ap.add_argument("--child-items", help=argparse.SUPPRESS)
    ap.add_argument("--child-quant", default="none", help=argparse.SUPPRESS)
    ap.add_argument("--child-batches", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.probe:
        print_probe()
        return

    global SHORT, LONG
    if args.legacy_short:
        SHORT = args.legacy_short
    if args.legacy_long:
        LONG = args.legacy_long

    if args.child:
        _run_child(
            args.child[0],
            args.child[1],
            args.child_items,
            args.out,
            quant=args.child_quant,
            batches=tuple(int(b) for b in (args.child_batches or "1").split(",")),
            max_new_tokens=args.max_new_tokens,
        )
        return

    models = resolve_models(args.preset, args.models)
    quants = [q for q in args.quant if q != "4bit" or args.device == "cuda"]
    if len(quants) != len(args.quant):
        print("note: 4-bit needs CUDA — dropped from the sweep on this device.")
    if not quants:
        raise SystemExit("no runnable quantization setting for this device")

    items_path = None
    if args.records:
        items = select_items(args.records, args.items, args.seed)
        items_path = str(Path(args.out or "bench").with_suffix(".items.json"))
        Path(items_path).write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            f"{len(items)} items from {args.records} "
            f"({min(i['chars'] for i in items)}-{max(i['chars'] for i in items)} ch, "
            f"mean {sum(i['chars'] for i in items) // len(items)}) -> {items_path}"
        )

    batches = ",".join(str(b) for b in sorted(set(args.batch)))
    # One subprocess per (model, quant) so peak RSS isolates the model; batch sizes are
    # swept inside it, so a 27B model is loaded once rather than once per batch size.
    for model in models:
        for quant in quants:
            print(f"\n===== {model} on {args.device} [{quant}] =====", flush=True)
            cmd = [sys.executable, __file__, "--child", model, args.device]
            cmd += ["--child-quant", quant, "--child-batches", batches]
            cmd += ["--max-new-tokens", str(args.max_new_tokens)]
            if items_path:
                cmd += ["--child-items", items_path]
            if args.out:
                cmd += ["--out", args.out]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
            for line in proc.stdout.splitlines():
                if line.startswith(("  item", "  batch", "  --", "RESULT")):
                    print(line, flush=True)
            if "RESULT" not in proc.stdout:
                tail = (proc.stderr.strip().splitlines() or ["?"])[-1]
                print(f"FAILED: {tail}", flush=True)
                # A host OOM-kill leaves no other trace, so the death must reach the
                # data file — otherwise the gap reads as "never attempted".
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "model": model,
                                    "device": args.device,
                                    "quant": quant,
                                    "ok": False,
                                    "died": True,
                                    "returncode": proc.returncode,
                                    "stderr_tail": tail[:300],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )


if __name__ == "__main__":
    main()
