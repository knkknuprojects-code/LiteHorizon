"""Compare two scans of the same corpus — e.g. small stand-in vs API teacher.

Pure logic over stored :class:`~horizon.analysis.Analysis` records (no model). It
pairs analyses of the *same text* produced by two different models and measures how
much they agree: enum agreement per axis, entity/keyword-set F1 between the two, and
theme match — plus each side's label distribution and trainable yield. This answers
the question that justifies a teacher pass: *did the stronger model actually change
(and hopefully improve) the labels, and how much?*

The two label sets both live in the append-only store after a ``--rescan`` teacher
pass (same key, different ``model_id``); read them with
:meth:`~horizon.store.JsonlStore.list_all_records` and split by model here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from horizon.analysis import Analysis
from horizon.training.evaluate import _all_entities, set_f1
from horizon.training.export import is_trainable

_ENUM_AXES = ("maturity", "impact", "time_horizon")


def distinct_models(records: Iterable[Analysis]) -> list[str]:
    """Model ids present, most-frequent first."""
    counts = Counter(a.model_id for a in records)
    return [m for m, _ in counts.most_common()]


def select_by_model(records: Iterable[Analysis], substring: str) -> dict[str, Analysis]:
    """Latest analysis per ``content_hash`` whose ``model_id`` contains ``substring``.

    Case-insensitive substring match (so ``gemini`` picks ``gemini-3.5-flash``).
    Later records win, matching the store's last-write-wins semantics per model.
    """
    needle = substring.lower()
    chosen: dict[str, Analysis] = {}
    for a in records:
        if needle in (a.model_id or "").lower():
            chosen[a.content_hash] = a
    return chosen


def _axis(analysis: Analysis, axis: str) -> str:
    """The enum value of one classification axis (e.g. ``maturity``) as a string."""
    return getattr(analysis.classification, axis).value


def _distribution(analyses: Iterable[Analysis], axis: str) -> dict[str, int]:
    """Count of each enum value along ``axis`` across the analyses."""
    return dict(Counter(_axis(a, axis) for a in analyses))


def _side(chosen: Mapping[str, Analysis], paired: list[str]) -> dict:
    """Per-side stats. ``n``/``trainable`` are over the side's full coverage; the
    distributions are over the *paired* texts only, so the two sides are directly
    comparable (same texts) rather than mixing different coverage counts."""
    values = list(chosen.values())
    paired_values = [chosen[h] for h in paired]
    return {
        "n": len(values),
        "trainable": sum(1 for a in values if is_trainable(a)),
        "distributions": {
            axis: _distribution(paired_values, axis) for axis in _ENUM_AXES
        },
    }


def compare_scans(
    baseline: Mapping[str, Analysis], teacher: Mapping[str, Analysis]
) -> dict:
    """Agreement metrics between two ``content_hash -> Analysis`` maps.

    Agreement is measured over the *paired* texts (scanned by both). Per side we
    also report coverage, trainable yield and label distribution, since a teacher
    that scans more texts or collapses fewer labels matters even where the two agree.
    """
    paired = sorted(set(baseline) & set(teacher))
    n = len(paired)

    enum_agreement: dict[str, float] = {}
    for axis in _ENUM_AXES:
        matches = sum(
            1 for h in paired if _axis(baseline[h], axis) == _axis(teacher[h], axis)
        )
        enum_agreement[axis] = matches / n if n else 0.0
    enum_agreement["overall"] = (
        sum(enum_agreement[a] for a in _ENUM_AXES) / len(_ENUM_AXES) if n else 0.0
    )

    theme_match = (
        sum(
            1
            for h in paired
            if baseline[h].classification.theme.strip().lower()
            == teacher[h].classification.theme.strip().lower()
        )
        / n
        if n
        else 0.0
    )
    entity_f1 = (
        sum(
            set_f1(
                _all_entities(baseline[h].entities), _all_entities(teacher[h].entities)
            )
            for h in paired
        )
        / n
        if n
        else 0.0
    )
    keyword_f1 = (
        sum(set_f1(baseline[h].keywords, teacher[h].keywords) for h in paired) / n
        if n
        else 0.0
    )

    return {
        "n_baseline": len(baseline),
        "n_teacher": len(teacher),
        "n_paired": n,
        "enum_agreement": enum_agreement,
        "theme_match": theme_match,
        "entity_f1": entity_f1,
        "keyword_f1": keyword_f1,
        "baseline": _side(baseline, paired),
        "teacher": _side(teacher, paired),
    }


# --- formatting ---------------------------------------------------------------


def _dist_line(dist: Mapping[str, int]) -> str:
    """One-line ``label pct`` rendering of a distribution, most-frequent first."""
    total = sum(dist.values()) or 1
    rows = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k} {v / total:.0%}" for k, v in rows) or "-"


def format_comparison(result: dict, *, baseline_label: str, teacher_label: str) -> str:
    """Render a stand-in-vs-teacher comparison as a plain-text report."""
    ea = result["enum_agreement"]
    b, t = result["baseline"], result["teacher"]
    lines = [
        f"Scan comparison: {baseline_label} (baseline) vs {teacher_label} (teacher)",
        "=" * 64,
        f"  coverage        baseline {result['n_baseline']}, "
        f"teacher {result['n_teacher']}, paired {result['n_paired']}",
        "",
        "  agreement on paired texts",
        f"    maturity        {ea['maturity']:.0%}",
        f"    impact          {ea['impact']:.0%}",
        f"    time_horizon    {ea['time_horizon']:.0%}",
        f"    enum overall    {ea['overall']:.0%}",
        f"    theme (exact)   {result['theme_match']:.0%}",
        f"    entity F1       {result['entity_f1']:.3f}",
        f"    keyword F1      {result['keyword_f1']:.3f}",
        "",
        "  label distribution (paired texts)",
    ]
    for axis in _ENUM_AXES:
        lines.append(f"    {axis}")
        lines.append(f"      baseline: {_dist_line(b['distributions'][axis])}")
        lines.append(f"      teacher:  {_dist_line(t['distributions'][axis])}")
    lines += [
        "",
        f"  trainable yield  baseline {b['trainable']}/{b['n']}, "
        f"teacher {t['trainable']}/{t['n']}",
    ]
    return "\n".join(lines)
