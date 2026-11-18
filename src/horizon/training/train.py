"""LoRA supervised fine-tuning — one code path, three tiers.

Reads an exported SFT dataset (``horizon export`` -> ``train.jsonl`` /
``eval.jsonl``) and fine-tunes a base model with ``trl``'s ``SFTTrainer`` + a
``peft`` LoRA adapter. Device, model size and quantization are config knobs, so
the *same* function runs:

- **tier A** — a tiny non-gated model, a few steps on CPU (loop validation);
- **tier B** — the small Gemma, full LoRA run on CPU (slow; an idle/overnight job);
- **tier C** — the large Gemma on the A100, LoRA or QLoRA (4-bit) when CUDA is present.

Training uses **completion-only loss**: the prompt (the exact inference prompt the
export reproduces) is masked, so the model learns to emit the JSON, not echo the
instructions. The response template is derived from the tokenizer's own chat
template, so it works across model families (Gemma ``<start_of_turn>model`` vs a
ChatML stand-in's ``<|im_start|>assistant``) with no per-model config.

Run as ``python -m horizon.training.train --base <model> --data data/sft --out
runs/<name>``. ``torch``/``trl``/``peft``/``datasets`` are imported lazily inside
:func:`train`, so importing this module (for the pure helpers/tests) needs none of
the ML stack. Lives behind the ``[train]`` extra.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from horizon import prompts


@dataclass
class TrainConfig:
    """Knobs for a LoRA SFT run. Defaults suit a small CPU prototype."""

    base_model: str
    data_dir: Path
    output_dir: Path
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] | None = None  # None -> peft infers
    # SFT
    max_seq_length: int = 1024
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1  # -1 -> use epochs
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 5
    seed: int = 0
    # device / quantization
    device: str | None = None  # None -> auto (cuda else cpu; never mps for training)
    use_qlora: bool | None = None  # None -> auto (only if cuda + bitsandbytes)

    def __post_init__(self) -> None:
        """Coerce the dir fields to ``Path`` and normalize target modules to a tuple."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.lora_target_modules is not None:
            self.lora_target_modules = tuple(self.lora_target_modules)


# --- pure helpers (no ML imports) --------------------------------------------


def load_examples(path: Path | str) -> list[dict]:
    """Read an exported JSONL split into a list of records."""
    path = Path(path)
    examples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def data_hash(examples: list[dict]) -> str:
    """Stable digest of a dataset's content hashes (order-sensitive)."""
    h = hashlib.sha256()
    for ex in examples:
        h.update(ex.get("content_hash", "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def response_template(tokenizer) -> str:
    """The assistant-turn opening the chat template adds for generation.

    Derived by diffing the template with and without ``add_generation_prompt`` —
    e.g. ``<start_of_turn>model\\n`` (Gemma) or ``<|im_start|>assistant\\n``
    (ChatML). Used as the completion-only-loss marker, so it's model-agnostic.
    """
    probe = [{"role": "user", "content": "x"}]
    with_gen = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True
    )
    without_gen = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False
    )
    if with_gen.startswith(without_gen):
        suffix = with_gen[len(without_gen) :]
        if suffix.strip():
            return suffix
    raise ValueError(
        "could not derive a response template from the tokenizer's chat template; "
        "pass one explicitly for this model family"
    )


def to_text(tokenizer, record: dict) -> str:
    """Render one SFT record into the full training string.

    ``prompt`` (the inference prompt + the assistant-generation prefix) followed by
    the canonical JSON completion and EOS. The collator masks everything up to and
    including the response template, so loss falls only on the completion + EOS.
    """
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True
    )
    eos = tokenizer.eos_token or ""
    return prompt + record["completion"] + eos


def build_run_card(
    cfg: TrainConfig,
    *,
    n_train: int,
    n_eval: int,
    dataset_digest: str,
    prompt_version: str,
    device: str,
    qlora: bool,
    created_at: str,
) -> dict:
    """Reproducibility record written next to the adapter."""
    hp = asdict(cfg)
    # Paths aren't reproducibility-relevant and don't serialize cleanly.
    for key in ("data_dir", "output_dir"):
        hp[key] = str(hp[key])
    return {
        "base_model": cfg.base_model,
        "prompt_version": prompt_version,
        "dataset_hash": dataset_digest,
        "n_train": n_train,
        "n_eval": n_eval,
        "device": device,
        "qlora": qlora,
        "created_at": created_at,
        "hyperparams": hp,
    }


def resolve_train_device(preferred: str | None = None) -> str:
    """Training device: explicit > cuda > cpu. Never ``mps``.

    MPS has incomplete training-op coverage on torch 2.2.2 and too little VRAM for
    even a 2B model on this hardware, so local
    training is CPU-bound and the A100's CUDA is the only real accelerator.
    """
    if preferred:
        return preferred
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_qlora(use_qlora: bool | None, device: str) -> bool:
    """Whether to load the base in 4-bit. Only on CUDA with ``bitsandbytes``.

    QLoRA's ``bitsandbytes`` is CUDA-only, so on CPU this is always ``False``.
    ``None`` means auto-enable when the requirements are present.
    """
    if device != "cuda":
        return False
    import importlib.util

    has_bnb = importlib.util.find_spec("bitsandbytes") is not None
    if use_qlora is None:
        return has_bnb
    if use_qlora and not has_bnb:
        raise RuntimeError(
            "use_qlora=True but bitsandbytes is not installed "
            '(install the cluster extra: pip install -e ".[train,train-gpu]")'
        )
    return use_qlora


# --- training entry point (lazy ML imports) ----------------------------------


def train(cfg: TrainConfig) -> Path:
    """Run the LoRA SFT and write the adapter + run-card to ``cfg.output_dir``.

    Returns the adapter directory path.
    """
    from datetime import datetime, timezone

    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            'training needs the [train] extra: pip install -e ".[dev,train]"'
        ) from exc

    device = resolve_train_device(cfg.device)
    qlora = resolve_qlora(cfg.use_qlora, device)

    train_examples = load_examples(cfg.data_dir / "train.jsonl")
    eval_path = cfg.data_dir / "eval.jsonl"
    eval_examples = load_examples(eval_path) if eval_path.is_file() else []
    if not train_examples:
        raise RuntimeError(f"no training examples in {cfg.data_dir / 'train.jsonl'}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {}
    if qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(model)
    elif device == "cuda":
        model = model.to(device)

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=(
            list(cfg.lora_target_modules) if cfg.lora_target_modules else None
        ),
    )

    train_ds = Dataset.from_list(
        [{"text": to_text(tokenizer, r)} for r in train_examples]
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template(tokenizer), tokenizer=tokenizer
    )

    sft = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        bf16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        gradient_checkpointing=(device == "cuda"),
        report_to=[],
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        data_collator=collator,
        peft_config=lora,
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_dir = cfg.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    manifest = _read_manifest(cfg.data_dir)
    run_card = build_run_card(
        cfg,
        n_train=len(train_examples),
        n_eval=len(eval_examples),
        dataset_digest=data_hash(train_examples),
        prompt_version=manifest.get("prompt_version", prompts.PROMPT_VERSION),
        device=device,
        qlora=qlora,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (cfg.output_dir / "run_card.json").write_text(
        json.dumps(run_card, indent=2) + "\n", encoding="utf-8"
    )
    return adapter_dir


def _read_manifest(data_dir: Path) -> dict:
    """The export's ``manifest.json`` as a dict, or ``{}`` if it's absent."""
    path = data_dir / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# --- CLI ----------------------------------------------------------------------


def build_parser():
    """Argument parser for the train CLI (LoRA SFT on an exported dataset)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m horizon.training.train",
        description="LoRA SFT on an exported dataset (CPU prototype or A100).",
    )
    p.add_argument("--base", dest="base_model", required=True, help="Base model id.")
    p.add_argument(
        "--data",
        dest="data_dir",
        type=Path,
        default=Path("data") / "sft",
        help="Exported dataset dir (train.jsonl / eval.jsonl).",
    )
    p.add_argument("--out", dest="output_dir", type=Path, required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lr", dest="learning_rate", type=float, default=2e-4)
    p.add_argument("--epochs", dest="num_train_epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument(
        "--batch-size", dest="per_device_train_batch_size", type=int, default=1
    )
    p.add_argument(
        "--grad-accum", dest="gradient_accumulation_steps", type=int, default=8
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="Force device (default: auto).")
    qg = p.add_mutually_exclusive_group()
    qg.add_argument("--qlora", dest="use_qlora", action="store_true", default=None)
    qg.add_argument("--no-qlora", dest="use_qlora", action="store_false")
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse args into a ``TrainConfig``, run the LoRA SFT, and print the adapter path."""
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        device=args.device,
        use_qlora=args.use_qlora,
    )
    adapter = train(cfg)
    print(f"Saved adapter to {adapter}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
