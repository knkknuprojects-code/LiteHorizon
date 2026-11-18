"""The taxonomy layer: what the bibliographic databases already classified.

The records carry curated subject classification from WoS and Scopus — ``subject_areas``
on 80% of the WoS/Scopus corpus (27 values) and ``wos_categories`` on 49% (252). Unlike
the model's free-text ``theme``, these do not drift between models, do not depend on the
prompt, and need no agreement measurement. Nothing here loads a model or an embedder.

Two things come out of that, and the second is the reason this module exists rather than
being a report script:

* **Field development** — records per field per year, as share of the corpus and as a
  least-squares slope. It needs no clustering and no threshold, which makes it the most
  robust signal available: it stands whatever the trend clustering does.
* **A denominator.** Velocity measured against the whole corpus cannot tell a theme that
  grows with its field from one that grows against it — and the fields moved a long way
  over 2015–2025, Physics from 22.8% to 12.8% while Medicine went 4.9% to 17.7%. The
  right divisor is the field, and :func:`Tabulation.field_totals` supplies it.

Cross-field pairs fall out of the same pass. 45% of records carry more than one subject
area, which reads as a complication for blocking and as the measurement for
interdisciplinarity: the co-occurrence series says which seams are opening.

Everything is a share of *this corpus*, which is an export with its own scope — never of
world science.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

DEFAULT_KEY = "subject_areas"


@dataclass(frozen=True)
class Series:
    """One field (or field pair) counted per year, with its derived trend terms."""

    name: str
    counts: tuple[tuple[int, int], ...]  # (year, records), oldest first
    total: int
    slope: float  # least-squares records/year over the covered years
    share_first: float  # share of the corpus in the earliest covered year
    share_last: float  # share in the latest
    share_delta: float  # share_last - share_first

    def count(self, year: int) -> int:
        """Records in ``year``, or 0 if the field is absent from that cohort."""
        return dict(self.counts).get(year, 0)

    @classmethod
    def from_dict(cls, data: dict) -> "Series":
        """Rebuild a series from the shape ``report._series_dict`` writes.

        Counts come back keyed by string, because JSON has no integer keys; the derived
        terms are stored rather than refitted, so a report re-rendered from its facts
        file shows the numbers that were measured rather than a second estimate of them.
        """
        return cls(
            name=data["name"],
            counts=tuple(
                sorted((int(year), n) for year, n in (data.get("counts") or {}).items())
            ),
            total=data["total"],
            slope=data["slope"],
            share_first=data["share_first"],
            share_last=data["share_last"],
            share_delta=data["share_delta"],
        )


@dataclass(frozen=True)
class Tabulation:
    """One pass over the records: cohort sizes, field counts and pair counts.

    Built by :func:`tabulate`. Holds counts rather than derived statistics so the same
    pass can serve both the report tables and the velocity denominator.
    """

    year_totals: Mapping[int, int]
    field_counts: Mapping[str, Mapping[int, int]]
    pair_counts: Mapping[tuple[str, str], Mapping[int, int]]
    records: int
    unclassified: int

    @property
    def years(self) -> tuple[int, ...]:
        """Every year present, oldest first."""
        return tuple(sorted(self.year_totals))

    @property
    def classified_years(self) -> tuple[int, ...]:
        """Years in which at least one record carries a subject area.

        The share columns have to be read at these endpoints rather than at the corpus's
        own, because a cohort can exist without the taxonomy existing for it. Measured on
        the WoS/Scopus corpus: the 2026 cohort holds 482 records and **none** of them
        carries a subject area, so every field's share there is 0 and every field reads
        as having lost exactly the share it started with — a table of confident negative
        numbers describing nothing but a gap in the export.
        """
        classified: set[int] = set()
        for counts in self.field_counts.values():
            classified.update(year for year, n in counts.items() if n)
        return tuple(sorted(classified))

    def field_totals(self, field: str) -> Mapping[int, int]:
        """Records per year in ``field`` — the denominator for a field-relative rate.

        Empty for an unknown field, which callers should treat as "no basis to
        normalise" rather than as zero: dividing by it would be worse than not
        normalising at all.
        """
        return self.field_counts.get(field, {})


def _slope(counts: Mapping[int, float], years: Sequence[int]) -> float:
    """Least-squares slope of ``counts`` over ``years``, per year.

    Missing years count as zero rather than being skipped: a field that stops appearing
    has declined, and dropping the gap would hide exactly that.

    Fitted against the years themselves rather than against their positions, so a gap in
    the cohorts is a gap in the fit. Values are counts here and shares in ``figures``,
    which is why the annotation is not ``int``: the arithmetic never cared.
    """
    n = len(years)
    if n < 2:
        return 0.0
    mean_x = sum(years) / n
    ys = [counts.get(y, 0) for y in years]
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in years)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(years, ys)) / denom


def tabulate(
    records: Iterable[Mapping],
    *,
    key: str = DEFAULT_KEY,
    years: Iterable[int] | None = None,
) -> Tabulation:
    """Count cohort sizes, per-field and per-pair records in a single pass.

    ``key`` selects the taxonomy (``subject_areas`` or ``wos_categories``). ``years``
    restricts the cohorts — pass the complete ones, since a partial final year depresses
    every rate computed from it.

    A record with no value under ``key`` still counts toward its cohort total: it is part
    of the corpus even though it cannot be attributed, and leaving it out would inflate
    every share.
    """
    wanted = None if years is None else set(years)
    year_totals: Counter[int] = Counter()
    field_counts: dict[str, Counter[int]] = defaultdict(Counter)
    pair_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    total = 0
    unclassified = 0

    for record in records:
        year = record.get("year")
        if not isinstance(year, int) or (wanted is not None and year not in wanted):
            continue
        total += 1
        year_totals[year] += 1
        # Sorted and de-duplicated so a pair has one canonical order and a record that
        # repeats an area is not counted twice.
        areas = sorted({a for a in (record.get(key) or []) if a})
        if not areas:
            unclassified += 1
            continue
        for area in areas:
            field_counts[area][year] += 1
        for pair in combinations(areas, 2):
            pair_counts[pair][year] += 1

    return Tabulation(
        year_totals=dict(year_totals),
        field_counts={k: dict(v) for k, v in field_counts.items()},
        pair_counts={k: dict(v) for k, v in pair_counts.items()},
        records=total,
        unclassified=unclassified,
    )


def _series(
    name: str,
    counts: Mapping[int, int],
    years: Sequence[int],
    totals: Mapping[int, int],
    fit_years: Sequence[int] | None = None,
) -> Series:
    """Assemble one :class:`Series` from raw per-year counts.

    ``fit_years`` separates what is *shown* from what is *fitted*. The counts series keeps
    every cohort in ``years``, so a reader sees the whole span; the endpoint shares and the
    slope are taken over ``fit_years`` alone.

    They come apart when the trailing cohorts are incomplete, and then both derived terms
    are wrong in ways a reader cannot see. The slope is dragged toward the stub — the
    failure the bibliographic corpus already hit. The shares are subtler and worse: a
    partial cohort is not a uniform sample of itself, because the groups inside it are
    granted at different speeds, so its composition is biased toward whatever resolves
    fastest. On the patent register the last cohort over-represents resident, mechanical
    fields and under-represents the slow non-resident pharmaceutical ones, which would
    read as a shift between technologies rather than as an artefact of when the register
    was cut.

    ``None`` fits over ``years``, which is the behaviour every existing caller had.
    """
    fit = list(fit_years) if fit_years else list(years)
    first, last = fit[0], fit[-1]
    share_first = counts.get(first, 0) / totals[first] if totals.get(first) else 0.0
    share_last = counts.get(last, 0) / totals[last] if totals.get(last) else 0.0
    return Series(
        name=name,
        counts=tuple((y, counts.get(y, 0)) for y in years),
        total=sum(counts.values()),
        slope=_slope(counts, fit),
        share_first=share_first,
        share_last=share_last,
        share_delta=share_last - share_first,
    )


def field_series(
    tab: Tabulation, *, min_total: int = 0, fit_years: Sequence[int] | None = None
) -> list[Series]:
    """Per-field series, largest share gain first.

    ``min_total`` drops fields below that many records — a slope fitted to a handful of
    papers is noise wearing a number. ``fit_years`` restricts the derived statistics to
    the cohorts worth deriving from while the counts keep the full span (see
    :func:`_series`).
    """
    years = tab.classified_years or tab.years
    out = [
        _series(name, counts, years, tab.year_totals, fit_years)
        for name, counts in tab.field_counts.items()
        if sum(counts.values()) >= min_total
    ]
    return sorted(out, key=lambda s: s.share_delta, reverse=True)


def pair_series(
    tab: Tabulation, *, min_total: int = 0, fit_years: Sequence[int] | None = None
) -> list[Series]:
    """Per-pair series named ``"A + B"``, steepest slope first.

    Ranked by slope rather than by share change: a pair's share of the corpus is tiny by
    construction, so absolute growth is the readable quantity.
    """
    years = tab.classified_years or tab.years
    out = [
        _series(f"{a} + {b}", counts, years, tab.year_totals, fit_years)
        for (a, b), counts in tab.pair_counts.items()
        if sum(counts.values()) >= min_total
    ]
    return sorted(out, key=lambda s: s.slope, reverse=True)
