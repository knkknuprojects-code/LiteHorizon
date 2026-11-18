"""Configuration: model selection and generation parameters.

The model is pinned here (and overridable via env) so the same code path serves
the small local-dev Gemma and the large fine-tuned/production Gemma — only the
``model_id`` changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Small instruction-tuned Gemma that runs on the MacBook (dev default).
DEFAULT_MODEL_ID = "google/gemma-2-2b-it"

# Extraction strategy: "combined" (one structured call) or "staged" (one call per
# aspect). See horizon.extract.
DEFAULT_STRATEGY = "combined"

# Phase 3 trend detection. Multilingual embeddings so
# sources in different languages cluster by meaning; small model for local dev.
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Clustering: "greedy" (dependency-free, local default) or "agglomerative" (sklearn).
DEFAULT_CLUSTER_METHOD = "greedy"
# Cosine-similarity threshold to treat two analyses as the same trend. Verified
# margin: same-meaning ~0.90, unrelated ~0, so anything in 0.5..0.7 separates them.
# Picked inside that band by sweeping (scripts/threshold_sweep.py) on the count of
# clusters big enough to read, the one non-monotone column. Confirmed on the full corpus,
# and the confirmation moved the reasoning: at a 50k sample 0.62 gives 542 such clusters
# against 423 at 0.55 and 403 at 0.68, where the first sweep had chosen it by a margin of
# three. That margin was fragile and a 20k sample of the same corpus does prefer 0.55
# (230 against 211) — the criterion moves with density, because more publications per
# topic afford finer distinctions without dropping below twenty members.
#
# The sweep clusters globally; this threshold is applied per subject-area block, and
# blocks run from a handful of publications to ~56k. So no single sweep answers it
# exactly, and the 50k row is the one closest to the blocks that carry the corpus.
DEFAULT_CLUSTER_THRESHOLD = 0.62
# Velocity window for rising-trend scoring: recent vs prior window of this size. Carries
# a unit — "7d" (days) or "2y" (calendar years). A year window snaps to January 1st, so a
# corpus dated only by publication year buckets by cohort instead of by leap-day drift
# (a 730-day window spans 2, 3 or 6 year-cohorts depending on the anchor).
DEFAULT_VELOCITY_WINDOW = "7d"
# Recency decay length in days: recency = 1 / (1 + days_since / scale). 1 reproduces the
# original 1/(1+days_since); 365 is the right scale for year-granularity corpora.
DEFAULT_RECENCY_SCALE_DAYS = 1
# Weight of the attention factor: 1 + w * (percentile - 0.5) * 2, so the factor spans
# [1-w, 1+w] and is exactly 1.0 when an item carries no attention signal. Must be < 1.0,
# or a zero percentile would zero the score outright.
DEFAULT_ATTENTION_WEIGHT = 0.5


# Batched generation is off by default: 1 keeps the single-sequence path, which is
# the only sensible setting on CPU (generation there is memory-bandwidth bound, so a
# batch buys nothing). Raise it on a GPU — that is where batching is the whole lever.
DEFAULT_BATCH_SIZE = 1


def _env_flag(name: str) -> bool:
    """Read a boolean ``HORIZON_*`` var: ``1/true/yes/on`` (any case) is true."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def parse_max_memory(spec: str) -> dict[int | str, str]:
    """Parse ``"0=38GiB,cpu=64GiB"`` into accelerate's ``max_memory`` map.

    GPU keys must be ``int`` and ``cpu``/``disk`` must be ``str``: accelerate silently
    ignores a string ``"0"``, so a device budget that looks applied would not be. Raises
    ValueError on a malformed spec rather than dropping the entry.
    """
    budget: dict[int | str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"malformed max-memory entry {part!r}: expected DEVICE=SIZE, "
                'e.g. "0=38GiB,cpu=64GiB"'
            )
        device, _, size = part.partition("=")
        device, size = device.strip(), size.strip()
        if not size:
            raise ValueError(f"max-memory entry {part!r} has no size")
        budget[int(device) if device.isdigit() else device] = size
    return budget


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters passed to the model's ``generate`` call."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


@dataclass(frozen=True)
class Settings:
    """Runtime settings for an LLM client."""

    model_id: str = DEFAULT_MODEL_ID
    # Device for inference; ``None`` means auto-detect (cuda > mps > cpu).
    device: str | None = None
    # Optional LoRA adapter directory (Phase 4). When set, it's applied on top of
    # ``model_id`` at load time — this is how a fine-tuned tier-B/C model is
    # selected: HORIZON_MODEL_ID=<base> + HORIZON_ADAPTER=<adapter dir>. ``None``
    # means plain base model (unchanged behaviour).
    adapter_path: str | None = None
    # Extraction strategy for the structured path: "combined" or "staged".
    strategy: str = DEFAULT_STRATEGY
    # --- model loading / throughput knobs (GPU cluster) ---
    # Every one of these defaults to today's behaviour, so an unset field means the
    # CPU dev path is untouched.
    # Weight dtype name ("bfloat16"/"float16"/"float32"). ``None`` keeps the built-in
    # rule: bf16 on CUDA, fp32 elsewhere. Explicit fp16 is the escape hatch for
    # pre-Ampere cards, which have no bf16.
    dtype: str | None = None
    # 4-bit (nf4) weight quantization via bitsandbytes — CUDA-only. This is what makes
    # a large Gemma fit a small card.
    load_in_4bit: bool = False
    # 8-bit is the middle rung, and it exists because the top one does not fit: the 12B
    # weights are 22.3 GiB in bf16 against a 24 GiB card, leaving nothing for the KV
    # cache. int8 halves that to ~11 GiB, which leaves room to batch.
    load_in_8bit: bool = False
    # accelerate placement, e.g. "auto" — shards a model too big for one GPU. When set,
    # the model is dispatched rather than ``.to()``-moved.
    device_map: str | None = None
    # Per-device budget for ``device_map``, e.g. "0=38GiB,cpu=64GiB". See
    # ``parse_max_memory``.
    max_memory: str | None = None
    # Attention kernel: "sdpa" | "flash_attention_2" | "eager". ``None`` leaves the
    # transformers default; an unavailable choice falls back with a warning.
    attn_implementation: str | None = None
    # Conversations per ``generate`` call. 1 keeps the single-sequence path.
    batch_size: int = DEFAULT_BATCH_SIZE
    # Truncate prompts to this many tokens (from the left, so the generation prompt
    # survives). ``None`` means no truncation.
    max_input_tokens: int | None = None
    # Trend detection (Phase 3).
    embed_model: str = DEFAULT_EMBED_MODEL
    cluster_method: str = DEFAULT_CLUSTER_METHOD
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD
    velocity_window: str = DEFAULT_VELOCITY_WINDOW
    recency_scale_days: int = DEFAULT_RECENCY_SCALE_DAYS
    attention_weight: float = DEFAULT_ATTENTION_WEIGHT
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``HORIZON_*`` environment variables, with defaults."""
        return cls(
            model_id=os.environ.get("HORIZON_MODEL_ID", DEFAULT_MODEL_ID),
            device=os.environ.get("HORIZON_DEVICE") or None,
            adapter_path=os.environ.get("HORIZON_ADAPTER") or None,
            strategy=os.environ.get("HORIZON_STRATEGY", DEFAULT_STRATEGY),
            dtype=os.environ.get("HORIZON_DTYPE") or None,
            load_in_4bit=_env_flag("HORIZON_LOAD_4BIT"),
            load_in_8bit=_env_flag("HORIZON_LOAD_8BIT"),
            device_map=os.environ.get("HORIZON_DEVICE_MAP") or None,
            max_memory=os.environ.get("HORIZON_MAX_MEMORY") or None,
            attn_implementation=os.environ.get("HORIZON_ATTN") or None,
            batch_size=int(os.environ.get("HORIZON_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            max_input_tokens=(
                int(os.environ["HORIZON_MAX_INPUT_TOKENS"])
                if os.environ.get("HORIZON_MAX_INPUT_TOKENS")
                else None
            ),
            embed_model=os.environ.get("HORIZON_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            cluster_method=os.environ.get(
                "HORIZON_CLUSTER_METHOD", DEFAULT_CLUSTER_METHOD
            ),
            cluster_threshold=float(
                os.environ.get("HORIZON_CLUSTER_THRESHOLD", DEFAULT_CLUSTER_THRESHOLD)
            ),
            # HORIZON_VELOCITY_WINDOW_DAYS is the pre-unit spelling, kept as an alias so
            # an existing shell profile doesn't break: it means days.
            velocity_window=os.environ.get("HORIZON_VELOCITY_WINDOW")
            or (
                f"{os.environ['HORIZON_VELOCITY_WINDOW_DAYS']}d"
                if "HORIZON_VELOCITY_WINDOW_DAYS" in os.environ
                else DEFAULT_VELOCITY_WINDOW
            ),
            recency_scale_days=int(
                os.environ.get("HORIZON_RECENCY_SCALE_DAYS", DEFAULT_RECENCY_SCALE_DAYS)
            ),
            attention_weight=float(
                os.environ.get("HORIZON_ATTENTION_WEIGHT", DEFAULT_ATTENTION_WEIGHT)
            ),
        )
