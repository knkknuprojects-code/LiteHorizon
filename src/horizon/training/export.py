"""Export stored analyses as supervised fine-tuning (SFT) data.

Turns persisted :class:`~horizon.analysis.Analysis` records (plus their original
source text from the :class:`~horizon.sources.SourceStore`) into (prompt,
completion) pairs that match the **inference** prompt distribution exactly:

- the prompt is the same ``EXTRACT_SYSTEM_PROMPT`` + ``COMBINED_PROMPT_TEMPLATE``,
  folded into a single user turn by the same ``to_chat`` the client uses, so the
  training script only has to apply the model's chat template — no train/inference
  skew;
- the completion is the canonical *content* JSON (no provenance — the model learns
  the structure, not ``generated_at``).

All of this is pure logic (no model), so it's fully testable on CPU. Filtering
drops degenerate base-model outputs; the train/eval split is deterministic and
leak-free (held out by ``content_hash``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from horizon import prompts
from horizon.analysis import Analysis
from horizon.llm.base import Message
from horizon.llm.transformers_client import to_chat

# Strings lifted straight from the prompt templates. A weak base/stand-in model
# sometimes echoes these verbatim instead of analysing — such records are noise,
# so we drop them rather than teach the placeholder back to the fine-tune.
_PLACEHOLDERS = frozenset(
    {
        "",
        "...",
        "2-3 sentence summary",
        "2-3 sentences",
        "short domain-agnostic theme label",
        "key signal",
        "keyword",
    }
)


def _is_placeholder(value: str) -> bool:
    """True if ``value`` is empty or a verbatim prompt-template placeholder string."""
    return value.strip().lower() in _PLACEHOLDERS


def is_trainable(analysis: Analysis) -> bool:
    """True if the analysis looks like real content worth training on.

    A conservative structural filter (T1): the summary, at least one signal, at
    least one keyword and the theme must be present and not a verbatim prompt
    placeholder. Keeps the obviously degenerate outputs out of the training set.
    """
    if _is_placeholder(analysis.summary):
        return False
    if not any(not _is_placeholder(s) for s in analysis.signals):
        return False
    if not any(not _is_placeholder(k) for k in analysis.keywords):
        return False
    if _is_placeholder(analysis.classification.theme):
        return False
    return True


def build_prompt_messages(text: str) -> list[dict[str, str]]:
    """The exact prompt the model sees at inference, as a chat message list.

    Mirrors ``extract.analyze_text`` (combined strategy): the EXTRACT system prompt
    + COMBINED template, folded into a single user turn by the same ``to_chat`` the
    client uses. The trainer applies the tokenizer's chat template to this.
    """
    clean = text.strip()
    messages = [
        Message("system", prompts.EXTRACT_SYSTEM_PROMPT),
        Message("user", prompts.COMBINED_PROMPT_TEMPLATE.format(text=clean)),
    ]
    return to_chat(messages)


def completion_dict(analysis: Analysis) -> dict:
    """Canonical content JSON (no provenance) — the SFT target the model learns.

    Key order mirrors ``COMBINED_PROMPT_TEMPLATE`` so the target matches what the
    prompt asks for.
    """
    c = analysis.classification
    e = analysis.entities
    return {
        "summary": analysis.summary,
        "signals": list(analysis.signals),
        "classification": {
            "theme": c.theme,
            "maturity": c.maturity.value,
            "impact": c.impact.value,
            "time_horizon": c.time_horizon.value,
        },
        "entities": {
            "organizations": list(e.organizations),
            "people": list(e.people),
            "technologies": list(e.technologies),
            "locations": list(e.locations),
        },
        "keywords": list(analysis.keywords),
    }


@dataclass(frozen=True)
class SFTExample:
    """One supervised example: the prompt-side chat turns and the target JSON."""

    content_hash: str
    messages: list[dict[str, str]]
    completion: str

    def to_dict(self) -> dict:
        """The example as a JSONL-serializable dict (one line per record)."""
        return {
            "content_hash": self.content_hash,
            "messages": self.messages,
            "completion": self.completion,
        }


def build_examples(
    analyses: Iterable[Analysis],
    sources: Mapping[str, str],
    *,
    filter_trainable: bool = True,
    approved_only: bool = False,
    source: str | None = None,
) -> tuple[list[SFTExample], dict[str, int]]:
    """Pair each analysis with its source text into an :class:`SFTExample`.

    ``sources`` maps ``content_hash`` -> original text. Analyses with no stored
    source (pre-Phase-4 records) or that fail :func:`is_trainable` are skipped.
    ``approved_only`` keeps only records vetted via ``horizon approve``; ``source``
    (if set) keeps only records with that provenance label (e.g. ``"teacher"``).
    Returns ``(examples, skipped)`` where ``skipped`` counts each drop reason.
    """
    examples: list[SFTExample] = []
    skipped = {"not_trainable": 0, "no_source": 0, "not_approved": 0, "wrong_source": 0}
    for a in analyses:
        if filter_trainable and not is_trainable(a):
            skipped["not_trainable"] += 1
            continue
        if approved_only and not a.approved:
            skipped["not_approved"] += 1
            continue
        if source is not None and a.source != source:
            skipped["wrong_source"] += 1
            continue
        text = sources.get(a.content_hash)
        if not text:
            skipped["no_source"] += 1
            continue
        examples.append(
            SFTExample(
                content_hash=a.content_hash,
                messages=build_prompt_messages(text),
                completion=json.dumps(completion_dict(a), ensure_ascii=False),
            )
        )
    return examples, skipped


def _eval_bucket(content_hash: str) -> float:
    """Deterministic ``[0, 1)`` bucket from the hash — same text, same side."""
    return int(content_hash[:8], 16) / 0x100000000


def split_examples(
    examples: list[SFTExample], *, eval_fraction: float = 0.2
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Partition into ``(train, eval)``, held out by ``content_hash``.

    The same text never lands in both splits (no leakage), and the split is
    deterministic (no RNG). With >=2 examples both splits are guaranteed
    non-empty, so even a tiny smoke set exercises train *and* eval.
    """
    train: list[SFTExample] = []
    held: list[SFTExample] = []
    for ex in examples:
        (held if _eval_bucket(ex.content_hash) < eval_fraction else train).append(ex)
    if len(examples) >= 2:
        if not held:
            ex = min(train, key=lambda e: _eval_bucket(e.content_hash))
            train.remove(ex)
            held.append(ex)
        elif not train:
            ex = max(held, key=lambda e: _eval_bucket(e.content_hash))
            held.remove(ex)
            train.append(ex)
    return train, held


def _write_jsonl(path: Path, examples: list[SFTExample]) -> None:
    """Write examples to ``path`` as UTF-8 JSONL, one example per line."""
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def export_dataset(
    analyses: Iterable[Analysis],
    sources: Mapping[str, str],
    out_dir: Path | str,
    *,
    smoke: int | None = None,
    eval_fraction: float = 0.2,
    filter_trainable: bool = True,
    approved_only: bool = False,
    source: str | None = None,
) -> dict:
    """Write ``train.jsonl`` / ``eval.jsonl`` / ``manifest.json`` to ``out_dir``.

    ``smoke`` caps the dataset to the first N trainable examples (T4's tiny
    loop-validation set). ``approved_only`` / ``source`` narrow the set to vetted
    or specific-provenance records. Returns the manifest dict.
    """
    examples, skipped = build_examples(
        analyses,
        sources,
        filter_trainable=filter_trainable,
        approved_only=approved_only,
        source=source,
    )
    if smoke is not None:
        examples = examples[:smoke]
    train, held = split_examples(examples, eval_fraction=eval_fraction)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "eval.jsonl", held)

    manifest = {
        "prompt_version": prompts.PROMPT_VERSION,
        "n_train": len(train),
        "n_eval": len(held),
        "eval_fraction": eval_fraction,
        "smoke": smoke,
        "approved_only": approved_only,
        "source": source,
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
