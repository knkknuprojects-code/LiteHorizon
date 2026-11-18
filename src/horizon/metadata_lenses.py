"""Lenses that read the records rather than the trend.

:mod:`horizon.lenses` ranks clusters on what :class:`~horizon.trends.Trend` already
carries. These three need the bibliographic metadata behind the members — venue, authors,
script — which lives in the built records and reaches them through
:class:`~horizon.corpus.RecordIndex`.

Each asks a question the trend fields cannot:

* **broadening** — is the work spreading across more venues than its volume alone
  explains? A cluster doubling inside one journal is a research group; a cluster
  doubling across twenty is a field opening.
* **new entrants** — is the growth new people, or the same people publishing more?
* **language asymmetry** — does a theme exist in only one of the corpus's two script
  segments? 8.7% of these records are Cyrillic, and a theme present in one segment and
  absent from the other is a direct signal about where a local research agenda diverges
  from the global one.

All three split a cluster's own timeline at its median year rather than using the
velocity window. The window is a global setting chosen for scoring; these questions are
about a cluster's internal history, and a cluster whose whole life predates the window
would otherwise have nothing to compare.

Author names are compared as written. The corpus has no identifiers, ``authors`` is one
delimited string, and 2 966 records had names truncated at build time, so the "new
entrants" figure is a floor: a name spelled two ways counts as two people and inflates
novelty. It is reported to be read as a comparison between clusters, not as a count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from horizon.analysis import Analysis
from horizon.corpus import RecordIndex
from horizon.lenses import DEFAULT_LENS_MIN_SIZE
from horizon.trends import Trend

#: Minimum members per half before a ratio between the halves means anything.
MIN_PER_HALF = 3


def split_authors(raw: str | None) -> frozenset[str]:
    """Author names from the corpus's semicolon-delimited ``authors`` field."""
    if not raw:
        return frozenset()
    return frozenset(name.strip() for name in raw.split(";") if name.strip())


@dataclass(frozen=True)
class Facts:
    """What the records say about one cluster, split early against late."""

    label: str
    size: int
    resolved: int  # members whose record was found
    early_n: int
    late_n: int
    early_venues: int
    late_venues: int
    new_author_share: float | None
    cyrillic: int
    latin: int

    @property
    def venue_spread(self) -> float | None:
        """Venues per record late, over the same early — above 1.0 is broadening.

        ``None`` when either half is too small for the ratio to mean anything.
        """
        if self.early_n < MIN_PER_HALF or self.late_n < MIN_PER_HALF:
            return None
        early = self.early_venues / self.early_n
        return (self.late_venues / self.late_n) / early if early else None

    @property
    def script_imbalance(self) -> float | None:
        """0.0 when a theme lives in one script only, 1.0 when evenly split.

        ``None`` when no member's script is known.
        """
        total = self.cyrillic + self.latin
        if not total:
            return None
        return min(self.cyrillic, self.latin) / max(self.cyrillic, self.latin, 1)

    @property
    def cyrillic_share(self) -> float | None:
        """The Cyrillic share of this topic's publications, ``None`` if none are known.

        Named for the script rather than for its size: "minority share" reads as a
        judgement and stops being true the moment a corpus is gathered elsewhere, while
        the two counts this divides are called ``cyrillic`` and ``latin`` throughout.

        Ranking by ``script_imbalance`` instead answers nothing on a corpus this
        one-sided — 91.3% of publications here are Latin-script, hundreds of topics are
        Latin-only, they all score exactly 0.0, and the table filled with rows reading
        ``0 / N`` that differed only in the topic's name.
        """
        total = self.cyrillic + self.latin
        if not total:
            return None
        return self.cyrillic / total


def facts_for(
    trend: Trend, analyses: Mapping[str, Analysis], index: RecordIndex
) -> Facts:
    """Gather the record-side facts for one cluster.

    ``analyses`` maps content hash to analysis for the whole population; a member whose
    record is missing from ``index`` is counted in ``size`` and excluded from every
    ratio, so partial coverage shows up rather than silently skewing.
    """
    members = [analyses[h] for h in trend.members if h in analyses]
    pairs = [(a, r) for a, r in index.pairs(members)]
    years = sorted(int(str(r.get("year"))) for _, r in pairs if r.get("year"))
    cut = median(years) if years else 0

    early: list[Mapping] = []
    late: list[Mapping] = []
    cyr = lat = 0
    for _, record in pairs:
        year = record.get("year")
        (late if year and year >= cut else early).append(record)
        script = record.get("script")
        if script == "cyrillic":
            cyr += 1
        elif script == "latin":
            lat += 1

    def venues(rows: Sequence[Mapping]) -> int:
        return len({(r.get("venue") or "").strip().lower() for r in rows} - {""})

    early_authors: set[str] = set()
    for r in early:
        early_authors |= split_authors(r.get("authors"))
    late_authors: set[str] = set()
    for r in late:
        late_authors |= split_authors(r.get("authors"))
    share = (
        len(late_authors - early_authors) / len(late_authors)
        if late_authors and early and len(early) >= MIN_PER_HALF
        else None
    )

    return Facts(
        label=trend.label,
        size=trend.size,
        resolved=len(pairs),
        early_n=len(early),
        late_n=len(late),
        early_venues=venues(early),
        late_venues=venues(late),
        new_author_share=share,
        cyrillic=cyr,
        latin=lat,
    )


def gather(
    trends: Iterable[Trend], analyses: Iterable[Analysis], index: RecordIndex
) -> list[Facts]:
    """Facts for every trend, in input order."""
    by_hash = {a.content_hash: a for a in analyses}
    return [facts_for(t, by_hash, index) for t in trends]


def _rank(rows: list, key, *, descending: bool) -> list:
    """Rank by ``key``, breaking ties on publication count, largest first.

    Two stable passes rather than a composite sort key, because the tie-break has to run
    the same direction whichever way the primary sorts — a tuple would reverse both.

    The tie-break is load-bearing here, not cosmetic. ``script_imbalance`` is 0.0 for
    *every* topic living in a single script, and there are hundreds of those, so the
    whole visible table was one tie ordered alphabetically: it opened with Abstract
    Algebra and Advanced Ceramic Materials because of the letter A. The same applied to
    the other two, with 100% author renewal and a flat 3.00x dispersion producing ties
    resolved by spelling.
    """
    rows.sort(key=lambda f: (-f.size, f.label))
    rows.sort(key=key, reverse=descending)
    return rows


def discriminates(rows: Sequence[Facts], key) -> bool:
    """Whether a lens's ranking separates anything on this corpus.

    A lens that cannot discriminate is not evidence of nothing — it is evidence of
    nothing *printed as though it were something*. On a single-script corpus every topic
    scores identically on the script indicator, so the order is arbitrary and the table
    is rows of "N / 0" sorted by nothing at all. The patent register is 100% Cyrillic and
    produced exactly that: the mirror image of the defect the script lens's own note
    warns about on a 91%-Latin corpus.

    Deliberately not inside the ranking functions. They are pure orderings and are tested
    as such — including a tie-break over two single-script topics, which is a legitimate
    question about ordering even though such a pair says nothing. Whether a lens is worth
    printing is a property of the corpus, so the caller that has the corpus decides.
    """
    return len({round(key(r), 3) for r in rows}) > 1


def broadening(
    facts: Iterable[Facts],
    *,
    min_size: int = DEFAULT_LENS_MIN_SIZE,
    top: int | None = None,
) -> list[Facts]:
    """Clusters spreading across venues faster than their volume explains."""
    rows = [f for f in facts if f.size >= min_size and (f.venue_spread or 0) > 1.0]
    _rank(rows, lambda f: f.venue_spread or 0, descending=True)
    return rows[:top] if top is not None else rows


def new_entrants(
    facts: Iterable[Facts],
    *,
    min_size: int = DEFAULT_LENS_MIN_SIZE,
    top: int | None = None,
) -> list[Facts]:
    """Clusters whose recent work comes from names absent from their earlier half."""
    rows = [f for f in facts if f.size >= min_size and f.new_author_share is not None]
    _rank(rows, lambda f: f.new_author_share or 0, descending=True)
    return rows[:top] if top is not None else rows


def language_asymmetry(
    facts: Iterable[Facts],
    *,
    min_size: int = DEFAULT_LENS_MIN_SIZE,
    top: int | None = None,
) -> list[Facts]:
    """Clusters ranked by the Cyrillic share of their publications.

    ``min_size`` gates on the *scripted* records rather than on ``size``: a cluster whose
    records mostly failed to join carries no evidence about scripts, however large it is.
    """
    rows = [
        f
        for f in facts
        if f.cyrillic_share is not None and f.cyrillic + f.latin >= min_size
    ]
    _rank(rows, lambda f: f.cyrillic_share or 0.0, descending=True)
    return rows[:top] if top is not None else rows
