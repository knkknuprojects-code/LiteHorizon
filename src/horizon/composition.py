"""Split each topic across the values of a record field, and trend each part separately.

A corpus is rarely one population. The patent register mixes academic applicants with
foreign corporations and independent inventors, and their technology profiles barely
overlap — pharmaceuticals are 28% of non-resident filings and 8.7% of resident ones. A
share computed over the union describes none of them, and the failure is not a matter of
precision: measured on this register, four of the ten largest fields **change the
direction** of their 2012–2020 movement depending on whether the corporate half is in.

The fix is to decompose, not to filter. Filtering answers one population's question and
throws the rest away; this keeps every record in the clustering — so the topics are built
from the whole corpus and a topic spanning two populations is still one topic — and
reports the split underneath. Where the groups disagree, the disagreement is the finding.

Nothing here is patent-specific: ``key`` names any categorical record field, and the same
pass would split a bibliographic corpus by document type or by provenance.

**Each group is rated against itself.** A group's velocity divides its recent window by
its prior window and then normalises by the same ratio for that group across the whole
corpus — not the corpus total. Without that, a group that is shrinking everywhere would
show every one of its topics as declining, which says nothing about any of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from horizon.analysis import Analysis
from horizon.corpus import RecordIndex
from horizon.trends import Trend

#: Groups smaller than this get no velocity. A ratio over two or three records is noise
#: with a decimal point, and it lands in a table beside figures that are not.
MIN_GROUP = 5

#: Records a group needs in the *prior* window before a ratio is reported. Below it the
#: denominator carries the result: the first version used ``recent / max(prior, 1)``,
#: which returns the recent count itself whenever the prior window is empty, so a group
#: with twenty recent records scored 20 and one with five scored 5 — size wearing the
#: name of growth. It put 12% of cells above 5.0 against a median of 1.12, and the
#: largest read 31.4.
MIN_PRIOR = 2


@dataclass(frozen=True)
class Group:
    """One population's presence inside one topic."""

    name: str
    size: int
    share: float  # of the topic's resolved members
    velocity: float | None  # against this group's own corpus-wide rate; None if small
    first_year: int | None
    last_year: int | None
    is_new: bool = False  # present in the recent window and absent from the prior one

    @property
    def measurable(self) -> bool:
        """Whether this group is large enough for its velocity to mean anything."""
        return self.velocity is not None


@dataclass(frozen=True)
class Composition:
    """A topic's split across a record field, with each part's own trend."""

    label: str
    seed_hash: str
    size: int
    resolved: int  # members whose record was found; ratios use this, not ``size``
    groups: tuple[Group, ...]  # largest first

    @property
    def divergence(self) -> float | None:
        """Spread between the fastest and slowest measurable group, or ``None``.

        This is what makes the section worth reading. A topic where every population
        moves together carries no information the overall row did not already have; one
        where the academic half is rising while the corporate half withdraws is a
        different claim from either, and neither the total nor a filtered run would show
        it.
        """
        rates = [g.velocity for g in self.groups if g.velocity is not None]
        return max(rates) - min(rates) if len(rates) > 1 else None


def _window(years: Sequence[int], counts: Mapping[int, int]) -> tuple[int, int]:
    """``(recent, prior)`` counts over the two halves of the covered span.

    Split at the midpoint of the *corpus* span rather than the topic's own, so every
    topic and every group is measured over the same two periods and the numbers can be
    compared down a column.
    """
    if len(years) < 2:
        return 0, 0
    mid = years[len(years) // 2]
    recent = sum(n for y, n in counts.items() if y >= mid)
    prior = sum(n for y, n in counts.items() if y < mid)
    return recent, prior


def compose(
    trend: Trend,
    analyses: Mapping[str, Analysis],
    index: RecordIndex,
    *,
    key: str,
    years: Sequence[int],
    baselines: Mapping[str, Mapping[int, int]],
    min_group: int = MIN_GROUP,
) -> Composition:
    """Split one topic by ``key``; ``baselines`` gives each group's corpus-wide cohorts.

    A member whose record is missing from ``index`` counts toward ``size`` and toward no
    group, so partial coverage is visible in ``resolved`` instead of quietly reweighting
    the shares.
    """
    members = [analyses[h] for h in trend.members if h in analyses]
    pairs = list(index.pairs(members))
    per_group: dict[str, dict[int, int]] = {}
    for _, record in pairs:
        name = record.get(key)
        year = record.get("year")
        if not name or year is None:
            continue
        per_group.setdefault(str(name), {})
        per_group[str(name)][int(year)] = per_group[str(name)].get(int(year), 0) + 1

    resolved = sum(sum(c.values()) for c in per_group.values())
    groups = []
    for name, counts in per_group.items():
        n = sum(counts.values())
        rate = None
        is_new = False
        if n >= min_group:
            recent, prior = _window(years, counts)
            base_recent, base_prior = _window(years, baselines.get(name, {}))
            # A group absent from the prior window has no growth rate — it is new to this
            # topic, which is a different statement and is reported as one.
            is_new = prior == 0 and recent > 0
            if prior >= MIN_PRIOR and base_prior:
                # The group's rate against the group's own rate, so a population that is
                # contracting everywhere does not make all of its topics look like
                # decline.
                group_rate = base_recent / base_prior
                rate = (recent / prior) / group_rate if group_rate else None
        covered = sorted(y for y, c in counts.items() if c)
        groups.append(
            Group(
                name=name,
                size=n,
                share=n / resolved if resolved else 0.0,
                velocity=rate,
                first_year=covered[0] if covered else None,
                last_year=covered[-1] if covered else None,
                is_new=is_new,
            )
        )
    groups.sort(key=lambda g: (-g.size, g.name))
    return Composition(
        label=trend.label,
        seed_hash=trend.seed_hash,
        size=trend.size,
        resolved=resolved,
        groups=tuple(groups),
    )


def baselines_for(
    records: Iterable[Mapping], *, key: str, years: Sequence[int]
) -> dict[str, dict[int, int]]:
    """Corpus-wide cohort counts per group — the denominator each group is rated against."""
    wanted = set(years)
    out: dict[str, dict[int, int]] = {}
    for record in records:
        name, year = record.get(key), record.get("year")
        if not name or year is None or int(year) not in wanted:
            continue
        bucket = out.setdefault(str(name), {})
        bucket[int(year)] = bucket.get(int(year), 0) + 1
    return out


def gather(
    trends: Iterable[Trend],
    analyses: Iterable[Analysis],
    index: RecordIndex,
    *,
    key: str,
    years: Sequence[int],
    baselines: Mapping[str, Mapping[int, int]],
    min_group: int = MIN_GROUP,
) -> list[Composition]:
    """A composition per trend, in input order."""
    by_hash = {a.content_hash: a for a in analyses}
    return [
        compose(
            t,
            by_hash,
            index,
            key=key,
            years=years,
            baselines=baselines,
            min_group=min_group,
        )
        for t in trends
    ]


@dataclass(frozen=True)
class FieldSplit:
    """One field's share of each population, at the ends of the reported span.

    This is the pair of numbers that decides whether an aggregate movement is a fact
    about the subject or about the mix. On the patent register four of the ten largest
    fields move one way over the union and the other way inside the domestic population,
    because the two populations have nearly disjoint profiles — so a reader given only
    the aggregate concludes that activity in a field slowed when the domestic half of it
    was growing.
    """

    name: str
    groups: tuple[tuple[str, float, float], ...]  # (group, share_first, share_last)

    @property
    def disagrees(self) -> bool:
        """Whether two groups moved in opposite directions."""
        deltas = [last - first for _, first, last in self.groups]
        return bool(deltas) and max(deltas) > 0 > min(deltas)


def field_splits(
    records: Iterable[Mapping],
    *,
    key: str,
    field_key: str = "subject_areas",
    years: Sequence[int],
    min_group: int = 50,
) -> list[FieldSplit]:
    """Per-field share of each population at the first and last reported cohort.

    Shares are of the group's own cohort, not of the corpus: the question is what each
    population was working on, and dividing by the union would fold the changing mix back
    into the answer this exists to separate.
    """
    if not years:
        return []
    first, last = years[0], years[-1]
    totals: dict[str, dict[int, int]] = {}
    counts: dict[str, dict[str, dict[int, int]]] = {}
    for record in records:
        group, year = record.get(key), record.get("year")
        if not group or year is None or int(year) not in (first, last):
            continue
        group, year = str(group), int(year)
        totals.setdefault(group, {}).setdefault(year, 0)
        totals[group][year] += 1
        for field in record.get(field_key) or []:
            if field:
                bucket = counts.setdefault(str(field), {}).setdefault(group, {})
                bucket[year] = bucket.get(year, 0) + 1

    out = []
    for field, per_group in counts.items():
        rows = []
        for group, by_year in per_group.items():
            base = totals.get(group, {})
            if min(base.get(first, 0), base.get(last, 0)) < min_group:
                continue  # a cohort too small for a share to mean anything
            rows.append(
                (
                    group,
                    by_year.get(first, 0) / base[first],
                    by_year.get(last, 0) / base[last],
                )
            )
        if len(rows) > 1:
            rows.sort(key=lambda r: -r[2])
            out.append(FieldSplit(name=field, groups=tuple(rows)))
    out.sort(key=lambda f: (not f.disagrees, f.name))
    return out


def population_totals(
    records: Iterable[Mapping], *, key: str, years: Sequence[int]
) -> dict[str, dict[str, int]]:
    """Size of each population overall and at the ends of the span.

    The corpus prompt needs these to say which populations are worth reading: a group
    that is 2% of the register at the end carries little of whatever it is doing, and a
    reading that treats all four as equals mistakes a rounding error for a movement.
    """
    if not years:
        return {}
    first, last = years[0], years[-1]
    out: dict[str, dict[str, int]] = {}
    for record in records:
        group, year = record.get(key), record.get("year")
        if not group or year is None:
            continue
        bucket = out.setdefault(str(group), {"total": 0, "first": 0, "last": 0})
        bucket["total"] += 1
        if int(year) == first:
            bucket["first"] += 1
        elif int(year) == last:
            bucket["last"] += 1
    return out


def by_population(splits: Sequence[FieldSplit], *, top: int = 6) -> dict[str, dict]:
    """Pivot the field splits into a per-population view of what each is moving toward.

    :func:`field_splits` is organised by field because that is how the report's section 1
    is organised. The prompt wants the other axis: a population is a coherent actor with
    priorities that shifted, and "what are universities moving toward" is a question a
    reader has, while "how did the four populations divide field 17" is not.
    """
    per: dict[str, list[tuple[str, float, float]]] = {}
    for split in splits:
        for group, first, last in split.groups:
            per.setdefault(group, []).append((split.name, first, last))
    out = {}
    for group, rows in per.items():
        rows.sort(key=lambda r: r[2] - r[1])
        out[group] = {
            "falling": [r for r in rows[:top] if r[2] < r[1]],
            "rising": [r for r in reversed(rows[-top:]) if r[2] > r[1]],
        }
    return out
