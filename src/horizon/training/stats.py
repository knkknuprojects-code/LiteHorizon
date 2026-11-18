"""Audit accumulated analyses to gauge training-set readiness.

Pure logic over stored :class:`~horizon.analysis.Analysis` records plus their
source texts — runs on CPU, no model. Answers the T4 "when to train" question in
concrete numbers: how many examples are training-ready, how far from the target,
whether the set is balanced (enum / theme distribution), and whether it's
cross-lingual (a rough writing-system check that flags an English-only corpus).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from horizon.analysis import Analysis
from horizon.training.export import is_trainable

DEFAULT_TARGET = 500


def _script_bucket(text: str) -> str:
    """Dominant writing system of ``text`` — a rough, zero-dependency proxy.

    Not real language detection: it buckets characters by Unicode range and
    returns the most common. Enough to notice that a corpus meant to be
    cross-lingual is in fact all Latin/English.
    """
    counts: Counter[str] = Counter()
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
            counts["cjk"] += 1
        elif ch.isalpha() and o < 0x0250:
            counts["latin"] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def dataset_stats(
    analyses: Iterable[Analysis],
    sources: Mapping[str, str],
    *,
    target: int = DEFAULT_TARGET,
) -> dict:
    """Summarize the store's training readiness.

    "Ready" = passes :func:`is_trainable` *and* has its source text stored (both
    are required to export an SFT example). Distributions are computed over the
    ready set, since that is what would actually train. ``source_breakdown`` is
    over all records so unvetted vs teacher/human provenance is visible.
    """
    analyses = list(analyses)
    total = len(analyses)
    trainable = [a for a in analyses if is_trainable(a)]
    ready = [a for a in trainable if sources.get(a.content_hash)]
    approved_ready = [a for a in ready if a.approved]

    maturity = Counter(a.classification.maturity.value for a in ready)
    impact = Counter(a.classification.impact.value for a in ready)
    time_horizon = Counter(a.classification.time_horizon.value for a in ready)
    themes = Counter(a.classification.theme.strip().lower() for a in ready)
    scripts = Counter(_script_bucket(sources[a.content_hash]) for a in ready)

    return {
        "total": total,
        "trainable": len(trainable),
        "not_trainable": total - len(trainable),
        "ready": len(ready),
        "trainable_no_source": len(trainable) - len(ready),
        "approved": len(approved_ready),
        "target": target,
        "gap": max(0, target - len(ready)),
        "source_breakdown": dict(Counter(a.source for a in analyses)),
        # Ingestion source (which feed/site/dir), over the ready set — the C6
        # per-source view. Blank origins (directly-scanned texts) are grouped as "-".
        "origin_breakdown": dict(Counter(a.origin or "-" for a in ready)),
        "maturity": dict(maturity),
        "impact": dict(impact),
        "time_horizon": dict(time_horizon),
        "themes_top": themes.most_common(10),
        "scripts": dict(scripts),
    }


def _fmt_dist(dist: dict) -> str:
    """Indented ``label count (pct)`` lines for a distribution, most-frequent first."""
    if not dist:
        return "    (none)"
    total = sum(dist.values()) or 1
    rows = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    return "\n".join(f"    {k:<14} {n:>4}  ({n / total:.0%})" for k, n in rows)


def format_stats(stats: dict) -> str:
    """Render :func:`dataset_stats` output as a readable CLI report."""
    ready, target, gap = stats["ready"], stats["target"], stats["gap"]
    verdict = (
        f"READY — {ready} ≥ target {target}"
        if gap == 0
        else f"need {gap} more to reach target {target}"
    )
    themes = stats["themes_top"]
    themes_str = (
        "\n".join(f"    {t:<24} {n:>4}" for t, n in themes) if themes else "    (none)"
    )
    return "\n".join(
        [
            "Dataset readiness",
            "=================",
            f"  total records      {stats['total']}",
            f"  trainable          {stats['trainable']}"
            f"  (dropped {stats['not_trainable']} as degenerate)",
            f"  ready (w/ source)  {ready}"
            f"  (missing source for {stats['trainable_no_source']} trainable)",
            f"  approved           {stats['approved']}",
            f"  >> {verdict}",
            "",
            "  source provenance:",
            _fmt_dist(stats["source_breakdown"]),
            "",
            "  ingestion source (over ready set):",
            _fmt_dist(stats["origin_breakdown"]),
            "",
            "  maturity:",
            _fmt_dist(stats["maturity"]),
            "  impact:",
            _fmt_dist(stats["impact"]),
            "  time_horizon:",
            _fmt_dist(stats["time_horizon"]),
            "",
            "  writing system (rough, over ready set):",
            _fmt_dist(stats["scripts"]),
            "",
            f"  top themes (of {len(themes)} shown):",
            themes_str,
        ]
    )
