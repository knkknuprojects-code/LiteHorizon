"""Trend detection: cluster analyses, then score how rising each cluster is.

``build_trends`` groups analyses into trends with a :class:`Clusterer` over their
embeddings, labels each cluster from its dominant theme/keywords, and scores it as
``velocity * size * recency * attention`` — recent activity against the prior window,
decayed by recency, weighted by size, and nudged by how much attention the members drew
relative to their own cohorts. It also reports each cluster's **lineage**: its earliest
member, its per-year growth curve, and how widely the idea spread.

Every time term is **scale-relative**, because items do not all arrive at the same
granularity. A feed delivers items by the day; bibliographic records carry no date finer
than the publication year. So the velocity window carries a unit (``"7d"`` / ``"2y"``,
see :class:`Window`) and recency decays over ``recency_scale_days``. The defaults
(``7d``, scale 1) reproduce the original day-scale arithmetic exactly.

Scoring is anchored at a date, not the wall clock: :func:`resolve_anchor` uses the newest
``item_date`` in the set unless told otherwise, so a fixed store scores identically today
and next month, and a corpus whose newest item predates the window still scores at all.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from horizon.analysis import Analysis
from horizon.clustering import Clusterer
from horizon.config import DEFAULT_ATTENTION_WEIGHT, DEFAULT_RECENCY_SCALE_DAYS

_WINDOW_SPEC = re.compile(r"^(\d+)([dy]?)$")
_BARE_YEAR = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class Window:
    """A velocity window: ``n`` days (``"d"``) or calendar years (``"y"``).

    Both units resolve to the same comparison convention — recent is
    ``[start(anchor), anchor]``, prior is ``[start(anchor, back=2), start(anchor))``.
    Only the start differs: days subtract elapsed days, years snap to January 1st of the
    n-th calendar year back.

    That snap is the point. Items dated only by publication year all land on
    ``<year>-01-01``, so a day-expressed window puts its boundary *on* a cohort stamp and
    leap-day drift decides whether a whole year of records counts as recent: a 730-day
    window spans {2025, 2026} anchored at 2026, but {2021…2026} anchored at 2023. A
    ``2y`` window spans exactly two cohorts at every anchor.
    """

    n: int
    unit: str  # "d" | "y"

    def start(self, anchor: date, *, back: int = 1) -> date:
        """The inclusive lower bound of the ``back``-th window ending at ``anchor``."""
        if self.unit == "y":
            return date(anchor.year - self.n * back + 1, 1, 1)
        return anchor - timedelta(days=self.n * back)

    def __str__(self) -> str:
        """The spec that would parse back to this window (e.g. ``"2y"``)."""
        return f"{self.n}{self.unit}"


def parse_window(spec: str | Window) -> Window:
    """Parse ``"7d"`` / ``"2y"`` / a bare ``"7"`` (days) into a :class:`Window`.

    Passes an existing :class:`Window` through. Raises ``ValueError`` on anything else,
    including a zero or negative count.
    """
    if isinstance(spec, Window):
        return spec
    match = _WINDOW_SPEC.match(str(spec).strip())
    if not match:
        raise ValueError(f"window must look like 7d or 2y, got {spec!r}")
    n = int(match.group(1))
    if n < 1:
        raise ValueError(f"window must be at least 1, got {spec!r}")
    return Window(n, match.group(2) or "d")


@dataclass(frozen=True)
class Trend:
    """A cluster of analyses with rising-trend scoring, lineage and a derived label."""

    label: str
    members: tuple[str, ...]  # content_hashes
    size: int
    first_seen: str  # ISO date
    last_seen: str  # ISO date
    velocity: float  # recent_count / max(prior_count, 1)
    score: float  # ranking score (rising trends first)
    top_keywords: tuple[str, ...]
    # Scoring terms, reported so the score is auditable from its printed parts.
    recency: float = 1.0  # 1 / (1 + days_since / recency_scale_days)
    attention: float = 1.0  # the multiplicative factor; exactly 1.0 without data
    attention_pctile: float | None = None  # mean cohort percentile of contributors
    attention_n: int = 0  # members that carried a percentile (coverage is partial)
    # Lineage: where the idea started and how far it spread. Correlational — see
    # ``build_trends``.
    seed_hash: str = ""  # earliest member
    seed_date: str = ""  # its item_date
    cohort_counts: tuple[tuple[str, int], ...] = ()  # (year, members), oldest first
    growth: float = 0.0  # least-squares slope of cohort_counts, members/year
    breadth: int = 0  # distinct organizations across members

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the trend."""
        return {
            "label": self.label,
            "members": list(self.members),
            "size": self.size,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "velocity": self.velocity,
            "score": self.score,
            "top_keywords": list(self.top_keywords),
            "recency": self.recency,
            "attention": self.attention,
            "attention_pctile": self.attention_pctile,
            "attention_n": self.attention_n,
            "seed_hash": self.seed_hash,
            "seed_date": self.seed_date,
            "cohort_counts": {year: n for year, n in self.cohort_counts},
            "growth": self.growth,
            "breadth": self.breadth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trend":
        """Rebuild a trend from what :meth:`to_dict` wrote.

        The inverse exists so a report can be re-rendered from its ``-facts.json`` without
        re-clustering the corpus: the clustering is the expensive half and the printed
        tables are the half that gets revised. Every float is stored unrounded, so the
        round trip is exact; only the container types have to be restored.
        """
        return cls(
            label=data["label"],
            members=tuple(data.get("members") or ()),
            size=data["size"],
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            velocity=data["velocity"],
            score=data["score"],
            top_keywords=tuple(data.get("top_keywords") or ()),
            recency=data.get("recency", 1.0),
            attention=data.get("attention", 1.0),
            attention_pctile=data.get("attention_pctile"),
            attention_n=data.get("attention_n", 0),
            seed_hash=data.get("seed_hash", ""),
            seed_date=data.get("seed_date", ""),
            cohort_counts=tuple(
                (y, n) for y, n in (data.get("cohort_counts") or {}).items()
            ),
            growth=data.get("growth", 0.0),
            breadth=data.get("breadth", 0),
        )


def _to_date(iso: str) -> date:
    """Parse an ISO date or datetime string to a plain ``date``."""
    return datetime.fromisoformat(iso).date()


def resolve_anchor(analyses: Sequence[Analysis], *, now: str | None = None) -> date:
    """The date scoring is anchored at: ``now`` when given, else the newest ``item_date``.

    Anchoring to the data rather than the clock makes ``trends`` reproducible over a fixed
    store, and keeps velocity meaningful for a corpus whose newest item is older than the
    window — bibliographic records dated by publication year, for instance, would
    otherwise fall outside every window and score zero.

    ``now`` accepts a bare year (``"2025"`` → ``2025-01-01``, which is what you type on a
    year-dated corpus), a date, or a full ISO datetime. Raises ``ValueError`` on an
    unparseable ``now`` or an empty ``analyses``.
    """
    if now is not None:
        text = now.strip()
        if _BARE_YEAR.match(text):
            return date(int(text), 1, 1)
        try:
            return _to_date(text)
        except ValueError:
            raise ValueError(
                f"--now must be a year, date or ISO datetime, got {now!r}"
            ) from None
    if not analyses:
        raise ValueError("cannot resolve an anchor from no analyses")
    # Parse before max: string comparison over mixed ISO forms orders by byte, which
    # gets timezone offsets wrong ("…-05:00" sorts above "…+00:00").
    return max(_to_date(a.item_date) for a in analyses)


def build_trends(
    analyses: Sequence[Analysis],
    vectors: Sequence[Sequence[float]],
    clusterer: Clusterer,
    *,
    window: Window | str,
    now: str | None = None,
    recency_scale_days: int = DEFAULT_RECENCY_SCALE_DAYS,
    attention_weight: float = DEFAULT_ATTENTION_WEIGHT,
    min_size: int = 1,
    baseline: Mapping[int, int] | None = None,
) -> list[Trend]:
    """Cluster ``analyses`` (aligned with ``vectors``) into scored, ranked trends.

    ``score = velocity * size * recency * attention``:

    * ``velocity`` — members in ``[window.start(anchor), anchor]`` over those in the
      preceding window of equal size. The upper bound matters: without it a backdated
      ``now`` sweeps every later item into "recent".

      With ``baseline`` — corpus (or field) records per year, e.g.
      ``fields.Tabulation.year_totals`` or ``field_totals(name)`` — this becomes a ratio
      of *shares* instead of counts, so a cluster that merely keeps pace with a growing
      corpus scores 1.0 rather than rising with it. The WoS/Scopus corpus grows from
      15 883 to 29 388 records a year over 2015–2025, which is enough to lift roughly
      half of everything without the correction; and passing a single field's totals
      answers the sharper question, since a theme in Physics holding a constant share of
      the *corpus* is growing inside a field that fell from 22.8% to 12.8%. A flat
      baseline reduces to the count form exactly.
    * ``recency`` — ``1 / (1 + days_since / recency_scale_days)`` off the newest member.
    * ``attention`` — ``1 + attention_weight * (pctile - 0.5) * 2`` over the members that
      carry an :attr:`Analysis.attention` percentile, and exactly ``1.0`` when none do.
      Absent data is never a penalty, and the factor never reaches zero, so a genuinely
      new cluster with no attention yet cannot be erased.

    Also reports lineage per cluster: ``seed_hash``/``seed_date`` (earliest member),
    ``cohort_counts`` (members per year), ``growth`` (slope of that curve) and ``breadth``
    (distinct organizations). This is **topical propagation, not proven descent** —
    clusters are embedding similarity, so the seed is the earliest similar analysis rather
    than a demonstrated origin, and ``breadth`` rests on model-extracted organization
    names, making it a noisy lower bound.
    """
    if not analyses:
        return []
    window = parse_window(window)
    if recency_scale_days < 1:
        raise ValueError("recency_scale_days must be >= 1")
    if not 0.0 <= attention_weight < 1.0:
        # At w == 1.0 a percentile of 0 gives a factor of exactly 0.0, which would zero
        # the score — the one thing the bounded factor exists to prevent.
        raise ValueError("attention_weight must be in [0, 1)")

    labels = clusterer.cluster(vectors)
    # Parse each item_date once: a full corpus run puts ~269k analyses through here.
    groups: dict[int, list[tuple[Analysis, date]]] = {}
    for label, analysis in zip(labels, analyses):
        groups.setdefault(label, []).append((analysis, _to_date(analysis.item_date)))

    # Anchor over every analysis, before min_size filtering, so clusters stay mutually
    # comparable and dropping small ones cannot move the windows.
    anchor = resolve_anchor(analyses, now=now)
    recent_start = window.start(anchor)
    prior_start = window.start(anchor, back=2)
    # Once per run, not per cluster: the windows are the same for every group.
    base_recent = (
        _baseline_in(baseline, recent_start, anchor, inclusive_hi=True)
        if baseline
        else 0
    )
    base_prior = (
        _baseline_in(baseline, prior_start, recent_start, inclusive_hi=False)
        if baseline
        else 0
    )

    trends: list[Trend] = []
    for group in groups.values():
        if len(group) < min_size:
            continue
        members = [a for a, _ in group]
        dates = [d for _, d in group]
        first_seen, last_seen = min(dates), max(dates)

        recent = sum(1 for d in dates if recent_start <= d <= anchor)
        prior = sum(1 for d in dates if prior_start <= d < recent_start)
        if base_recent and base_prior:
            # Shares, not counts: a cluster that holds its share of a growing corpus is
            # not rising. ``max(prior, 1)`` keeps the guard the count form has — a
            # cluster with no prior members is treated as having one rather than
            # dividing by zero — and with a flat baseline the two forms are identical.
            velocity = (recent / base_recent) * (base_prior / max(prior, 1))
        else:
            velocity = recent / max(prior, 1)

        days_since = max((anchor - last_seen).days, 0)
        recency = 1.0 / (1.0 + days_since / recency_scale_days)

        rated = [m.attention for m in members if m.attention is not None]
        if rated:
            pctile = sum(rated) / len(rated)
            attention = 1.0 + attention_weight * (pctile - 0.5) * 2.0
        else:
            pctile, attention = None, 1.0

        score = velocity * len(members) * recency * attention
        seed, seed_date = min(
            zip(members, dates), key=lambda p: (p[1], p[0].content_hash)
        )
        cohorts = _cohort_counts(dates)

        trends.append(
            Trend(
                label=_label(members),
                members=tuple(m.content_hash for m in members),
                size=len(members),
                first_seen=first_seen.isoformat(),
                last_seen=last_seen.isoformat(),
                velocity=round(velocity, 3),
                score=round(score, 3),
                top_keywords=_top_keywords(members),
                recency=round(recency, 3),
                attention=round(attention, 3),
                attention_pctile=None if pctile is None else round(pctile, 3),
                attention_n=len(rated),
                seed_hash=seed.content_hash,
                seed_date=seed_date.isoformat(),
                cohort_counts=cohorts,
                growth=round(_growth(cohorts), 2),
                breadth=_breadth(members),
            )
        )
    trends.sort(key=lambda t: t.score, reverse=True)
    return trends


def _baseline_in(
    baseline: Mapping[int, int], lo: date, hi: date, *, inclusive_hi: bool
) -> int:
    """Corpus records dated inside a window, from per-year cohort sizes.

    Cohort stamps are ``<year>-01-01``, so testing that stamp against the window uses
    exactly the predicate the members themselves are counted with — this is an equality,
    not an approximation, for a corpus dated by publication year.
    """
    total = 0
    for year, count in baseline.items():
        stamp = date(year, 1, 1)
        if lo <= stamp and (stamp <= hi if inclusive_hi else stamp < hi):
            total += count
    return total


def _cohort_counts(dates: Sequence[date]) -> tuple[tuple[str, int], ...]:
    """Members per calendar year, oldest first — a cluster's growth curve."""
    counter = Counter(d.year for d in dates)
    return tuple((str(year), counter[year]) for year in sorted(counter))


def _growth(cohorts: Sequence[tuple[str, int]]) -> float:
    """Least-squares slope of a growth curve, in members per year (0.0 for one point).

    Fitted over the years actually present rather than a filled range, so a gap year
    neither invents a zero nor breaks the fit.
    """
    if len(cohorts) < 2:
        return 0.0
    xs = [int(year) for year, _ in cohorts]
    ys = [float(n) for _, n in cohorts]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def _breadth(members: Sequence[Analysis]) -> int:
    """Distinct organizations across the cluster, casefolded.

    How widely an idea has been taken up, not just how many papers repeat it: forty papers
    from one group is a different phenomenon from forty across thirty organizations. Names
    are model-extracted, so spelling variants inflate this — treat it as a lower bound.
    """
    names = {
        org.strip().casefold()
        for m in members
        for org in m.entities.organizations
        if org.strip()
    }
    return len(names)


def _top_keywords(members: Sequence[Analysis], n: int = 5) -> tuple[str, ...]:
    """Return the ``n`` most common keywords across the cluster, lowercased."""
    counter: Counter[str] = Counter()
    for m in members:
        counter.update(k.lower() for k in m.keywords)
    return tuple(k for k, _ in counter.most_common(n))


def _label(members: Sequence[Analysis]) -> str:
    """Most common theme across the cluster, falling back to the top keyword."""
    themes = Counter(
        m.classification.theme.strip()
        for m in members
        if m.classification.theme.strip()
    )
    if themes:
        return themes.most_common(1)[0][0]
    kws = _top_keywords(members, 1)
    return kws[0] if kws else "(unlabeled)"
