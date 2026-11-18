"""Evaluate a model's structured output against gold analyses.

Pure, deterministic metrics that reuse the inference-time parser/validator
(:mod:`horizon.extract`), so a fine-tuned model is judged on exactly the contract
the pipeline relies on. The model-running part (:func:`run_predictions`) just
calls an ``LLMClient`` over the exported eval prompts — pass a fake client to test
it; everything else runs on CPU with no model.

Metrics:
- **valid-JSON rate** — fraction of outputs that parse *and* validate into the
  ``Analysis`` schema;
- **enum accuracy** — maturity / impact / time_horizon vs gold (over parseable);
- **entity / keyword F1** — set overlap vs gold (over parseable).

``evaluate_outputs`` returns a metrics dict; ``compare`` lays base vs fine-tuned
side by side so the swap only happens when the fine-tune wins.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from horizon.analysis import Analysis, Entities
from horizon.config import GenerationConfig
from horizon.extract import AnalysisError, _build_combined, parse_json
from horizon.llm.base import LLMClient

# A (raw model output, gold fields) pair. "gold fields" is the dict shape
# returned by ``extract._build_combined`` (summary/signals/classification/
# entities/keywords), so prediction and gold are compared like-for-like.
Pair = tuple[str, dict]


def gold_fields_from_completion(completion: str) -> dict:
    """Parse an exported ``completion`` JSON string into combined gold fields."""
    return _build_combined(parse_json(completion))


def gold_fields_from_analysis(analysis: Analysis) -> dict:
    """Combined-field view of a stored :class:`Analysis` (for ad-hoc eval)."""
    return {
        "summary": analysis.summary,
        "signals": analysis.signals,
        "classification": analysis.classification,
        "entities": analysis.entities,
        "keywords": analysis.keywords,
    }


def _safe_fields(raw: str) -> dict | None:
    """Parse+validate a raw model reply into combined fields, or ``None``."""
    try:
        return _build_combined(parse_json(raw))
    except AnalysisError:
        return None


def valid_json_rate(raw_outputs: Sequence[str]) -> float:
    """Fraction of outputs that parse and validate into the ``Analysis`` schema."""
    if not raw_outputs:
        return 0.0
    ok = sum(1 for r in raw_outputs if _safe_fields(r) is not None)
    return ok / len(raw_outputs)


def enum_accuracy(pairs: Iterable[Pair]) -> float:
    """Per-field enum match rate (maturity/impact/time_horizon) over parseable preds."""
    total = correct = 0
    for raw, gold in pairs:
        fields = _safe_fields(raw)
        if fields is None:
            continue
        pc = fields["classification"]
        gc = gold["classification"]
        for got, want in (
            (pc.maturity, gc.maturity),
            (pc.impact, gc.impact),
            (pc.time_horizon, gc.time_horizon),
        ):
            total += 1
            correct += int(got == want)
    return correct / total if total else 0.0


def set_f1(pred: Iterable[str], gold: Iterable[str]) -> float:
    """F1 of two string sets (case-insensitive). Empty vs empty scores 1.0."""
    p = {x.strip().lower() for x in pred if x.strip()}
    g = {x.strip().lower() for x in gold if x.strip()}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def _all_entities(entities: Entities) -> list[str]:
    """Flatten all entity types (orgs/people/tech/locations) into one list."""
    return [
        *entities.organizations,
        *entities.people,
        *entities.technologies,
        *entities.locations,
    ]


def _mean_f1(pairs: Iterable[Pair], select) -> float:
    """Mean set-F1 of ``select(pred)`` vs ``select(gold)`` over parseable preds."""
    scores = []
    for raw, gold in pairs:
        fields = _safe_fields(raw)
        if fields is None:
            continue
        scores.append(set_f1(select(fields), select(gold)))
    return sum(scores) / len(scores) if scores else 0.0


def entity_f1(pairs: Iterable[Pair]) -> float:
    """Mean F1 over the union of all entity mentions, over parseable preds."""
    return _mean_f1(pairs, lambda f: _all_entities(f["entities"]))


def keyword_f1(pairs: Iterable[Pair]) -> float:
    """Mean keyword-set F1 over parseable preds."""
    return _mean_f1(pairs, lambda f: f["keywords"])


def evaluate_outputs(pairs: Sequence[Pair]) -> dict:
    """Compute the full metrics dict for ``(raw_output, gold_fields)`` pairs."""
    raws = [r for r, _ in pairs]
    return {
        "n": len(pairs),
        "valid_json_rate": valid_json_rate(raws),
        "enum_accuracy": enum_accuracy(pairs),
        "entity_f1": entity_f1(pairs),
        "keyword_f1": keyword_f1(pairs),
    }


def compare(base: dict, tuned: dict) -> dict:
    """Lay base vs fine-tuned metrics side by side with deltas (float metrics)."""
    out: dict[str, dict] = {}
    for key, b in base.items():
        if isinstance(b, float):
            t = tuned.get(key, 0.0)
            out[key] = {"base": b, "tuned": t, "delta": round(t - b, 4)}
    return out


def run_predictions(
    client: LLMClient,
    eval_examples: Iterable[dict],
    *,
    config: GenerationConfig | None = None,
) -> list[Pair]:
    """Run ``client`` over exported eval examples; return ``(raw, gold)`` pairs.

    ``eval_examples`` are dicts loaded from ``eval.jsonl`` (each with ``messages``
    and ``completion``). The prompt is the already-folded single user turn, so it
    reproduces the inference input exactly.
    """
    pairs: list[Pair] = []
    for ex in eval_examples:
        prompt = ex["messages"][-1]["content"]
        raw = client.complete(prompt, config=config)
        gold = gold_fields_from_completion(ex["completion"])
        pairs.append((raw, gold))
    return pairs


# --- base-vs-tuned report -----------------------------------------------------

_METRIC_LABELS = (
    ("valid_json_rate", "valid JSON"),
    ("enum_accuracy", "enum accuracy"),
    ("entity_f1", "entity F1"),
    ("keyword_f1", "keyword F1"),
)


def format_comparison(base: dict, tuned: dict | None = None) -> str:
    """Render a base(-vs-tuned) metrics table with a swap verdict."""
    lines = [f"Evaluated on {base['n']} held-out example(s).", ""]
    if tuned is None:
        lines.append(f"{'metric':<16}{'base':>8}")
        lines += [f"{label:<16}{base[key]:>8.3f}" for key, label in _METRIC_LABELS]
        return "\n".join(lines)

    lines.append(f"{'metric':<16}{'base':>8}{'tuned':>8}{'delta':>9}")
    for key, label in _METRIC_LABELS:
        b, t = base[key], tuned[key]
        lines.append(f"{label:<16}{b:>8.3f}{t:>8.3f}{t - b:>+9.3f}")
    won = sum(1 for k, _ in _METRIC_LABELS if tuned[k] > base[k])
    lost = sum(1 for k, _ in _METRIC_LABELS if tuned[k] < base[k])
    lines += [
        "",
        f"Fine-tuned wins on {won}/{len(_METRIC_LABELS)} metrics "
        f"({lost} regressed). Swap only if it wins.",
    ]
    return "\n".join(lines)


def _build_client(
    base_model: str,
    *,
    adapter_path: str | None = None,
    tuned_model: str | None = None,
    device: str | None = None,
):
    """Build a ``TransformersClient`` for one eval side (base, or adapter/tuned).

    ``transformers`` imports lazily here so the module stays model-free. An
    ``adapter_path`` loads the LoRA on top of ``base_model``; ``tuned_model`` selects
    a standalone merged model instead.
    """
    from horizon.config import Settings
    from horizon.llm.transformers_client import TransformersClient

    return TransformersClient(
        Settings(
            model_id=tuned_model or base_model,
            adapter_path=adapter_path,
            device=device,
        )
    )


def evaluate_models(
    base_model: str,
    eval_examples: list[dict],
    *,
    adapter_path: str | None = None,
    tuned_model: str | None = None,
    device: str | None = None,
    config: GenerationConfig | None = None,
) -> tuple[dict, dict | None]:
    """Run the base — and, if an adapter/tuned model is given, the fine-tune — over
    the eval set. Returns ``(base_metrics, tuned_metrics_or_None)``."""
    base_client = _build_client(base_model, device=device)
    base_metrics = evaluate_outputs(
        run_predictions(base_client, eval_examples, config=config)
    )
    tuned_metrics = None
    if adapter_path or tuned_model:
        tuned_client = _build_client(
            base_model,
            adapter_path=adapter_path,
            tuned_model=tuned_model,
            device=device,
        )
        tuned_metrics = evaluate_outputs(
            run_predictions(tuned_client, eval_examples, config=config)
        )
    return base_metrics, tuned_metrics


# --- CLI ----------------------------------------------------------------------


def build_parser():
    """Argument parser for the evaluate CLI (base vs fine-tuned on the eval split)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m horizon.training.evaluate",
        description="Evaluate base vs fine-tuned on the held-out eval split.",
    )
    p.add_argument("--base", dest="base_model", required=True, help="Base model id.")
    p.add_argument(
        "--adapter",
        dest="adapter_path",
        default=None,
        help="LoRA adapter dir for the tuned side (applied on top of --base).",
    )
    p.add_argument(
        "--tuned-model",
        dest="tuned_model",
        default=None,
        help="Standalone tuned/merged model id for the tuned side (instead of --adapter).",
    )
    p.add_argument(
        "--data",
        dest="data_dir",
        type=Path,
        default=Path("data") / "sft",
        help="Exported dataset dir containing eval.jsonl.",
    )
    p.add_argument("--device", default=None, help="Force device (default: auto).")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature; 0 (default) = greedy/deterministic.",
    )
    p.add_argument("--json", dest="as_json", action="store_true", help="Emit raw JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Evaluate the base (and any tuned side) on ``eval.jsonl`` and print the report.

    Errors out if the export's eval split is missing or empty. Prints a metrics
    table (or raw JSON with ``--json``); returns a process exit code.
    """
    args = build_parser().parse_args(argv)
    from horizon.training.train import load_examples

    eval_path = args.data_dir / "eval.jsonl"
    if not eval_path.is_file():
        print(
            f"error: no eval split at {eval_path} — run `horizon export` first.",
            file=sys.stderr,
        )
        return 1
    eval_examples = load_examples(eval_path)
    if not eval_examples:
        print(f"error: eval split is empty: {eval_path}", file=sys.stderr)
        return 1

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )
    base_metrics, tuned_metrics = evaluate_models(
        args.base_model,
        eval_examples,
        adapter_path=args.adapter_path,
        tuned_model=args.tuned_model,
        device=args.device,
        config=config,
    )

    if args.as_json:
        out: dict = {"base": base_metrics}
        if tuned_metrics is not None:
            out["tuned"] = tuned_metrics
            out["delta"] = compare(base_metrics, tuned_metrics)
        print(json.dumps(out, indent=2))
    else:
        print(format_comparison(base_metrics, tuned_metrics))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
