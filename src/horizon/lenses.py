"""Indicators: ways of reading the same topics.

The report is scientometric, so the vocabulary here follows that field rather than the
software one. A **topic** is a cluster of semantically similar publications. An
**indicator** is a *selection rule plus a ranking quantity* over those topics — formally
a selection followed by an ordering, computing nothing new. This module was called
"lenses" first; the word is a metaphor with no standing in scientometrics, and it collides
with the bidirectional get/put lens of functional programming, which this is not.

:func:`~horizon.trends.build_trends` answers one question — what is emerging — and ranks
by ``velocity * size * recency * attention``. The topics answer several more, and none of
them needs a second model pass or even a re-clustering: every indicator here is a filter
and a sort over fields :class:`~horizon.trends.Trend` already carries.

Where the name is an established term it is used as the field uses it. ``sleeping-beauty``
is van Raan's: a body of work dormant for years that then takes off. ``journal-dispersion``
is Bradford scattering. Two names were deliberately *not* borrowed — this module's
post-peak durability is not the citation *half-life*, and declining output is not
literature *obsolescence*; both of those denote citation ageing, and reusing them would
claim a measurement that was not made.

Three design choices worth stating, because all three could have gone the other way.

**Where a question is "large on one axis, small on another" — impact against output — the
indicator is a filter plus a sort, not an invented product.** A composite score would have
to weight the axes against each other, that weight would be arbitrary, and it would be
buried in a number nobody could audit. Splitting on the median and ranking within the half
is cruder and honest. The two halves are the familiar output-versus-impact quadrants.

**An indicator that cannot apply returns nothing rather than a default.** A topic with no
citation data is not a topic with low citations, and a one-year topic has no growth curve;
ranking them as if they scored zero would put absent data at the bottom of every table as
though it had been measured.

**An indicator needs more publications than a topic table does, so it carries its own
floor.** Every indicator ranks on a per-topic statistic — a citation percentile, years held
after a peak, output going quiet — and below a handful of publications those are properties
of one or two papers rather than of a body of work. Measured on the 2015-2018 sample at
threshold 0.62, the high-output-low-impact table ranked topics of 12-38 publications and
the sustained table topics of 5-10: the report's own ``--min-size`` had let them through
because it exists to keep a *topic table* readable, which is a different question.
:data:`DEFAULT_LENS_MIN_SIZE` gates every indicator, and the median that the size-split
indicators compare against is taken over the survivors — so "low output" means low among
topics big enough to measure, not small enough to be noise.

Three planned indicators are not here — journal dispersion, new authors
and script distribution — because they read author, venue and script from the
*records* rather than from the topic, which needs a join this module deliberately does not
do. They live in :mod:`horizon.metadata_lenses` and honour the same floor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from horizon.trends import Trend


@dataclass(frozen=True)
class Lens:
    """A named reading: which clusters qualify, and how they rank."""

    name: str
    question: str
    applies: Callable[[Trend, "Context"], bool]
    key: Callable[[Trend], float]
    descending: bool = True
    #: Label key for the quantity this lens ranks on, resolved against
    #: ``report.LABELS``. A table that ranks by citation percentile and shows velocity
    #: leaves the reader guessing at the criterion — and guessing wrongly, since the
    #: two can point opposite ways.
    column: str = "score"


#: Members a cluster needs before a lens will rank it. Distinct from the report's
#: ``--min-size``, which only decides what is worth tabulating: a percentile over five
#: records is not a measurement of anything, however legible the row looks. Provisional
#: on the 2015-2018 sample, like the clustering threshold — re-check on the full corpus.
DEFAULT_LENS_MIN_SIZE = 20


@dataclass(frozen=True)
class Context:
    """Population statistics a lens needs to say what "large" means here."""

    median_size: float

    @classmethod
    def of(cls, trends: Sequence[Trend]) -> "Context":
        """Summarise a trend population; an empty one has a median size of 0."""
        return cls(median_size=median([t.size for t in trends]) if trends else 0.0)


def peak_year(trend: Trend) -> int | None:
    """The calendar year holding the most members, or ``None`` without a curve.

    Ties go to the earliest year: a cluster that matched its peak again later has not
    peaked twice, it has stayed at its peak since the first time.
    """
    if not trend.cohort_counts:
        return None
    best_year, best_n = None, -1
    for year, n in trend.cohort_counts:
        if n > best_n:
            best_year, best_n = int(year), n
    return best_year


def _dormancy(trend: Trend) -> float:
    """Years between a cluster's first record and its peak."""
    peak = peak_year(trend)
    if peak is None or not trend.first_seen:
        return 0.0
    return float(peak - int(trend.first_seen[:4]))


def _sustain(trend: Trend) -> float:
    """Years after the peak in which the cluster still produced half of it.

    A cluster that spikes once and stops differs from one that reaches a level and holds
    it, and size alone cannot tell them apart.
    """
    peak = peak_year(trend)
    if peak is None:
        return 0.0
    counts = {int(y): n for y, n in trend.cohort_counts}
    threshold = counts[peak] / 2
    return float(sum(1 for y, n in counts.items() if y > peak and n >= threshold))


def _decaying_volume(trend: Trend) -> float:
    """How much membership is going quiet: size scaled by the shortfall in velocity.

    Bounded on purpose. The obvious form — ``size / velocity`` — is unbounded as velocity
    approaches zero, so a two-member cluster with no recent records would outrank a
    hundred-member one that halved.
    """
    return trend.size * max(0.0, 1.0 - trend.velocity)


LENSES: tuple[Lens, ...] = (
    Lens(
        name="emerging",
        column="score",
        question="what is gaining, relative to its own past",
        applies=lambda t, c: True,
        key=lambda t: t.score,
    ),
    Lens(
        name="declining",
        column="decay",
        question="where established volume is going quiet",
        # Below the median it is not established enough for its decline to mean anything.
        applies=lambda t, c: t.velocity < 1.0 and t.size >= c.median_size,
        key=_decaying_volume,
    ),
    Lens(
        name="sleeping-beauty",
        column="dormancy",
        question="what lay dormant for years and then took off",
        applies=lambda t, c: _dormancy(t) >= 2 and t.velocity >= 1.0,
        # Sleep duration alone, not duration x publications. The product ranked a topic
        # of 2140 publications with a three-year sleep at "6420.00" under a column headed
        # "sleep duration, years", and it is a composite carrying an unstated weight
        # between two axes — the thing this module says it does not build. Publications
        # are already a column of their own, so a reader can see both quantities and
        # weigh them without the table deciding on their behalf.
        key=_dormancy,
    ),
    Lens(
        name="high-impact-low-output",
        column="attention",
        question="small bodies of work drawing outsized citation",
        applies=lambda t, c: t.attention_pctile is not None and t.size <= c.median_size,
        key=lambda t: t.attention_pctile or 0.0,
    ),
    Lens(
        name="high-output-low-impact",
        column="attention",
        question="much work, little uptake",
        applies=lambda t, c: t.attention_pctile is not None and t.size > c.median_size,
        key=lambda t: t.attention_pctile or 0.0,
        descending=False,
    ),
    Lens(
        name="sustained",
        column="sustain",
        question="what held its level after peaking, rather than spiking once",
        applies=lambda t, c: _sustain(t) > 0,
        key=_sustain,
    ),
)

BY_NAME: dict[str, Lens] = {lens.name: lens for lens in LENSES}


def apply(
    lens: Lens,
    trends: Sequence[Trend],
    *,
    top: int | None = None,
    min_size: int = DEFAULT_LENS_MIN_SIZE,
) -> list[Trend]:
    """The trends this lens applies to, ranked by it.

    Clusters below ``min_size`` are dropped before anything else, including before the
    median that :class:`Context` carries. Filtering first is the point: a floor applied
    after the split would leave ``influential-narrow`` comparing against a median pulled
    down by the very clusters too small to rank, and the two size-split lenses would stop
    partitioning the same population.

    Ties break on ``content_hash`` of the seed so the order is stable across runs — two
    clusters with the same score would otherwise swap places between invocations and make
    a diff of two reports unreadable.
    """
    eligible = [t for t in trends if t.size >= min_size]
    ctx = Context.of(eligible)
    chosen = [t for t in eligible if lens.applies(t, ctx)]
    # Two stable passes: publications first, then the indicator's own quantity. A tuple
    # key would reverse both, and the tie-break has to stay largest-first whichever way
    # the indicator sorts. Ties are common on the integer-valued indicators — every
    # topic that slept the full span reports the same sleep duration — and inside a tie
    # the larger body of work is the one worth naming.
    chosen.sort(key=lambda t: (-t.size, t.seed_hash))
    chosen.sort(key=lens.key, reverse=lens.descending)
    return chosen[:top] if top is not None else chosen
