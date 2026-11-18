"""Merge a LoRA adapter into its base model (T6 — edge/weak-machine inference).

``train.py`` emits a LoRA *adapter* that loads on top of the base at runtime
(``HORIZON_ADAPTER``). On a weak machine it's simpler and faster to load a single
*merged* model instead of base+adapter every time. This folds the adapter weights
into the base and saves a standalone ``transformers`` model (safetensors) — still
the one runtime, just selected with ``HORIZON_MODEL_ID=<merged dir>`` and no
``HORIZON_ADAPTER``.

Run as ``python -m horizon.training.merge --base <model> --adapter runs/x/adapter
--out models/x-merged``. ``torch``/``transformers``/``peft`` import lazily inside
:func:`merge_adapter`, so the pure helpers/tests need no ML stack. Behind the
``[train]`` extra (``peft``).
"""

from __future__ import annotations

import json
from pathlib import Path

_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json")


def _has_tokenizer(path: Path | str) -> bool:
    """True if ``path`` is a dir that carries its own tokenizer files."""
    p = Path(path)
    return p.is_dir() and any((p / m).is_file() for m in _TOKENIZER_MARKERS)


def _read_run_card(adapter_path: Path) -> dict | None:
    """The adapter's run-card if present (``train.py`` writes it beside the adapter)."""
    card = adapter_path.parent / "run_card.json"
    if card.is_file():
        return json.loads(card.read_text(encoding="utf-8"))
    return None


def build_merge_card(
    base_model: str,
    adapter_path: Path | str,
    *,
    merged_at: str,
    adapter_run_card: dict | None = None,
) -> dict:
    """Provenance record written next to the merged model."""
    card = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "merged_at": merged_at,
    }
    if adapter_run_card is not None:
        card["adapter_run_card"] = adapter_run_card
    return card


def merge_adapter(
    base_model: str,
    adapter_path: Path | str,
    out_dir: Path | str,
    *,
    device: str | None = None,
) -> Path:
    """Merge the LoRA adapter at ``adapter_path`` into ``base_model``, save to ``out_dir``.

    Loads the tokenizer from the adapter dir when it carries one (so any tokens the
    fine-tune added survive); resizes the base embeddings to match before applying
    the adapter (a no-op when sizes already agree, as for Gemma/Qwen). Returns the
    output directory.
    """
    from datetime import datetime, timezone

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            'merging needs the [train] extra: pip install -e ".[dev,train]"'
        ) from exc

    adapter_path = Path(adapter_path)
    out = Path(out_dir)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok_src = str(adapter_path) if _has_tokenizer(adapter_path) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if base.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))

    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    card = build_merge_card(
        base_model,
        adapter_path,
        merged_at=datetime.now(timezone.utc).isoformat(),
        adapter_run_card=_read_run_card(adapter_path),
    )
    (out / "merge_card.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    return out


# --- CLI ----------------------------------------------------------------------


def build_parser():
    """Argument parser for the merge CLI (fold a LoRA adapter into its base)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m horizon.training.merge",
        description="Merge a LoRA adapter into its base for standalone (edge) loading.",
    )
    p.add_argument("--base", dest="base_model", required=True, help="Base model id.")
    p.add_argument(
        "--adapter", dest="adapter_path", type=Path, required=True, help="Adapter dir."
    )
    p.add_argument("--out", dest="out_dir", type=Path, required=True)
    p.add_argument("--device", default=None, help="Load device (default: cpu).")
    return p


def main(argv: list[str] | None = None) -> int:
    """Merge the adapter into its base and print how to load the standalone model."""
    args = build_parser().parse_args(argv)
    out = merge_adapter(
        args.base_model, args.adapter_path, args.out_dir, device=args.device
    )
    print(f"Saved merged model to {out}")
    print(f"Use it with: HORIZON_MODEL_ID={out}  (and no HORIZON_ADAPTER)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
