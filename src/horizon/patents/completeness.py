"""Which cohorts a patent register can be measured over, and on which time axis.

A publication corpus has one date and one way to be incomplete: the year it was gathered
in is a stub. A patent register has neither property, and both differences change what a
fitted rate means.

**Three axes.** A patent carries a priority date (the invention's claim to precedence), a
filing date, and a grant date. In this register they sit a median 2.5 years apart, mean
3.4, with a 10% tail past 6.2. A trend measured on grant year and the same trend measured
on priority year are different statements — the first is about the office's output, the
second about when the inventing happened — and neither is wrong, so the axis is a named
choice rather than a default buried in a parser.

**Truncation reaches back years, not one year.** The register was cut on a date. A patent
whose priority year is recent has had less time to be granted than one from a decade ago,
so recent cohorts hold only the fraction of their eventual grants that have issued. That
fraction is not small for one year and fine for the next — it decays smoothly, and by the
last cohorts it is severe:

    priority year  2012  2013  2014 … 2019  2020  2021  2022
    grants seen    3065  3120  2488 … 1936  1143   768   340

Read as counts, that is a collapse in Ukrainian inventing after 2019. Read against the
lag distribution, most of it is the register not having caught up. Fitting a rate across
the untrimmed range measures the truncation, not the trend — the same failure the
bibliographic corpus hit with a single partial cohort (see ``_complete_years`` in the
CLI), one order of magnitude wider.

The guard here is therefore not "drop the last cohort" but an estimate from the
register's own lag distribution: for priority year ``Y`` and cut date ``C``, the share of
eventual grants already visible is the empirical CDF of the grant lag at ``C - Y``. A
cohort is reported when that share clears :data:`DEFAULT_COVERAGE`.

That CDF must not be measured on the whole register, and the reason is the defect it is
meant to correct. A recent cohort has had no time to accumulate long-lag grants, so its
*observed* lags are short — p90 falls from 6.25 years for 2019 to 3.61 for 2022 — and
pooling them in shortens the very distribution used to judge how much is missing, making
an incomplete cohort look complete. So the reference set is the cohorts old enough to have
resolved: those whose elapsed time exceeds :data:`REFERENCE_QUANTILE` of the pooled lag.
Measured both ways on this register the two agree closely (2020: 0.893 against 0.888),
which is worth knowing — the bias is real but small here because the recent cohorts are a
small share of the pool — but the pooled form is only correct by accident, and on a
register with a heavier recent tail it would not be.

**What the guard does not do is explain the decline.** Corrected for truncation, the
2020–2022 cohorts still imply final sizes of 0.51, 0.41 and 0.23 of the 2012–2019 mean.
Truncation accounts for part of the fall in the raw counts and nothing like all of it; the
rest is a real contraction in filings, and trimming the cohorts must not be allowed to
read as having explained it away. The estimate is stated in the report rather than applied
silently, so a reader sees which cohorts were removed, how close each came, and that the
ones removed were also genuinely smaller.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

# The register's three dates, and the record key each one lands on. ``priority`` is the
# default: it dates the invention rather than the office's handling of it, which is what
# a trend over technologies is about. The others are here because a question about
# backlog or about filing behaviour wants a different one, not because they are fallbacks.
YEAR_AXES = {
    "priority": "priority_year",
    "filing": "filing_year",
    "grant": "grant_year",
}
DEFAULT_AXIS = "priority"

# A cohort is reported when at least this share of its eventual grants is estimated to
# have issued. Deliberately high: the whole point of the axis is to date inventions, and a
# cohort missing a third of itself distorts every share and every slope drawn through it.
DEFAULT_COVERAGE = 0.85

# Below this many records a lag distribution is not worth estimating from and the guard
# falls back to reporting every cohort — better to report the raw range and say so than
# to trim on a distribution built from a handful of rows.
MIN_LAG_SAMPLE = 200

# A cohort is treated as fully resolved, and so usable as a reference for the lag
# distribution, once its elapsed time exceeds this quantile of the pooled lag. Not the
# maximum: one 13.8-year outlier would disqualify every cohort in the register and leave
# nothing to estimate from.
REFERENCE_QUANTILE = 0.95


def lag_quantiles(
    lags: Sequence[float], quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95)
) -> dict[float, float]:
    """Empirical quantiles of the grant lag, in years.

    Plain nearest-rank on the sorted sample: the lag is reported in the register to
    several decimals but is only meaningful to about a month, so interpolating between
    neighbours would dress up a precision the source does not have.
    """
    ordered = sorted(float(x) for x in lags)
    if not ordered:
        return {}
    out = {}
    for q in quantiles:
        idx = min(len(ordered) - 1, max(0, int(round(q * len(ordered))) - 1))
        out[q] = ordered[idx]
    return out


def coverage_at(lags: Sequence[float], elapsed: float) -> float:
    """Share of grants whose lag was at most ``elapsed`` years — the empirical CDF.

    This is the estimate of how much of a cohort is visible: a cohort ``elapsed`` years
    old has had time to accumulate exactly the grants whose lag is shorter than that.
    Returns 1.0 for a cohort older than every observed lag, 0.0 for a sample of nothing.
    """
    if not lags:
        return 0.0
    if elapsed < 0:
        return 0.0
    return sum(1 for x in lags if float(x) <= elapsed) / len(lags)


def cohort_coverage(
    records: Iterable[dict],
    *,
    axis: str = DEFAULT_AXIS,
    cut: date | None = None,
    by: str | None = None,
) -> dict[int, float]:
    """Estimated completeness per cohort on ``axis``, keyed by year.

    ``cut`` is the register's slice date; without one the latest grant date in the
    records stands in for it, which is the same thing whenever the export was taken the
    day it was cut.

    ``by`` names a record key whose groups grant at different speeds, and on this register
    that is not a refinement but a correction. Residents reach grant in a median 1.57
    years; non-residents, arriving through the international route, take 5.37. A single
    pooled distribution describes neither, and the error is largest exactly where it
    matters: for the 2022 cohort the resident population is 99% visible and the
    non-resident one 11%, while pooling reports 59% for both. Stratified, each group is
    extrapolated on its own lag and the cohort's coverage is what is observed over what is
    implied. Leaving ``by`` unset keeps the pooled estimate.
    """
    rows = [r for r in records if r.get(YEAR_AXES[axis]) is not None]
    lags = [r["lag_years"] for r in rows if r.get("lag_years") is not None]
    years = sorted({int(r[YEAR_AXES[axis]]) for r in rows})
    if not years:
        return {}

    if cut is None:
        stamps = [r.get("granted_at") for r in rows if r.get("granted_at")]
        cut = (
            date.fromisoformat(max(stamps)[:10])
            if stamps
            else date(max(years) + 1, 1, 1)
        )
    # Grant year needs no lag correction: a patent granted in year Y is in the register
    # the moment it is granted, so the only incomplete cohort is the one the cut falls
    # inside, and its coverage is simply the fraction of the year elapsed.
    if axis == "grant":
        elapsed_in_cut = (cut - date(cut.year, 1, 1)).days / 365.25
        return {y: (1.0 if y < cut.year else min(1.0, elapsed_in_cut)) for y in years}

    if len(lags) < MIN_LAG_SAMPLE:
        return dict.fromkeys(years, 1.0)

    # Mid-year as the cohort's centre of mass: a cohort's filings are spread across its
    # year, so the average member has had half a year less than the year boundary implies.
    def elapsed(year: int) -> float:
        return (cut - date(year, 7, 1)).days / 365.25

    if by is None:
        reference = reference_lags(rows, lags, axis=axis, cut=cut)
        return {y: coverage_at(reference, elapsed(y)) for y in years}

    groups: dict[Any, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(by), []).append(r)
    # A group too small to estimate from falls back to the pooled reference rather than
    # being dropped: its records are still in the cohort's numerator, so excluding it from
    # the denominator would report a coverage above 1.
    pooled = reference_lags(rows, lags, axis=axis, cut=cut)
    references = {}
    for key, members in groups.items():
        own = [r["lag_years"] for r in members if r.get("lag_years") is not None]
        references[key] = (
            reference_lags(members, own, axis=axis, cut=cut)
            if len(own) >= MIN_LAG_SAMPLE
            else pooled
        )

    out = {}
    for year in years:
        observed = implied = 0.0
        for key, members in groups.items():
            n = sum(1 for r in members if int(r[YEAR_AXES[axis]]) == year)
            if not n:
                continue
            share = coverage_at(references[key], elapsed(year))
            observed += n
            if share <= 0:
                implied = float("inf")  # a group with nothing visible yet
                break
            implied += n / share
        out[year] = observed / implied if implied else 0.0
    return out


def lag_drift(
    records: Iterable[dict], *, axis: str = DEFAULT_AXIS, by: str | None = None
) -> dict[Any, dict[int, float]]:
    """Median grant lag per cohort — the check that the estimator's assumption holds.

    Every coverage figure here rests on the lag distribution being the same for a recent
    cohort as for an old one. On this register it is not: the resident median runs 1.55
    years for 2012 and 2.51 for 2022, while the office's annual grant count halves after
    2021. A lengthening lag means coverage is **overstated** for the newest cohorts, and
    since the reference set is by construction the oldest ones, the estimator cannot see
    this itself.

    The drift is reported rather than corrected for, because correcting it would need the
    lag of grants that have not happened. What it supports is the weaker and defensible
    statement: the trailing cohorts are *at best* as complete as the estimate says.

    ``by`` matters here for the same reason it matters to the coverage, and the pooled
    figure is actively misleading without it: pooled, this register's median lag is flat
    at 2.6 years and then *falls* to 2.5, which reads as an office speeding up. It is
    not — the resident median rises from 1.55 to 2.51 throughout. What moves the pooled
    number is the mix, as the slow-granting non-resident population thins out of the
    recent cohorts. A drift check that a change in composition can cancel is not a check.

    Note also what the drift is measured on — grants already issued. A cohort's slow half
    is missing from its own median, so this understates the lengthening as well.
    """
    groups: dict[Any, dict[int, list[float]]] = {}
    for r in records:
        year, lag = r.get(YEAR_AXES[axis]), r.get("lag_years")
        if year is None or lag is None:
            continue
        key = r.get(by) if by is not None else None
        groups.setdefault(key, {}).setdefault(int(year), []).append(float(lag))
    out: dict[Any, dict[int, float]] = {}
    for key, per_year in groups.items():
        medians = {}
        for year, lags in sorted(per_year.items()):
            lags.sort()
            medians[year] = lags[len(lags) // 2]
        out[key] = medians
    return out


def reference_lags(
    rows: Sequence[dict],
    pooled: Sequence[float],
    *,
    axis: str = DEFAULT_AXIS,
    cut: date,
) -> list[float]:
    """Lags drawn only from cohorts old enough to have resolved.

    Estimating the lag distribution from the whole register would use the truncated
    cohorts to judge their own truncation, and they are short precisely because they are
    truncated (see the module docstring). The cutoff is :data:`REFERENCE_QUANTILE` of the
    pooled lag: a cohort that has had longer than that to grant has, by definition, seen
    all but the last few percent of its grants arrive.

    Falls back to the pooled lags when no cohort is that old, or when the ones that are
    hold too few records to estimate from — a smaller sample of the right cohorts stops
    being the better choice at some point, and this is where it does.
    """
    horizon = lag_quantiles(pooled, (REFERENCE_QUANTILE,)).get(REFERENCE_QUANTILE)
    if horizon is None:
        return list(pooled)
    resolved = [
        r["lag_years"]
        for r in rows
        if r.get("lag_years") is not None
        and (cut - date(int(r[YEAR_AXES[axis]]), 7, 1)).days / 365.25 >= horizon
    ]
    return resolved if len(resolved) >= MIN_LAG_SAMPLE else list(pooled)


def complete_years(
    records: Iterable[dict],
    *,
    axis: str = DEFAULT_AXIS,
    coverage: float = DEFAULT_COVERAGE,
    cut: date | None = None,
    by: str | None = None,
) -> tuple[list[int], dict[int, float]]:
    """The cohorts worth fitting a trend across, and the coverage estimate behind them.

    Returns the reportable years oldest first, together with every cohort's estimated
    coverage — the second value is what the report prints, so a reader sees which years
    were dropped and how close each one came, rather than a range that arrived unexplained.

    Only a *trailing* run is trimmed. A gap in the middle of a register is a fact about
    the register, and removing cohorts from the interior would quietly reshape the series
    into one no data supports.
    """
    records = list(records)
    est = cohort_coverage(records, axis=axis, cut=cut, by=by)
    years = sorted(est)
    keep = list(years)
    while keep and est[keep[-1]] < coverage:
        keep.pop()
    return keep, est
