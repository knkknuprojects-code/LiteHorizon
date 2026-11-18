"""Joining analyses back to the records they were produced from.

An :class:`~horizon.analysis.Analysis` carries a ``content_hash`` and nothing else about
its source. Everything bibliographic — subject area, venue, authors, citation counts,
script — stays in the built records. Several things want both at once: clustering blocked
by field, lenses that read the author or venue set, the language-asymmetry question.

The join is exactly one line of arithmetic, and getting it wrong is quiet: an ad-hoc
``sha256(f"{title}. {abstract}")`` matched only 92% of a sample, because
:func:`~horizon.ingest.wos.item_text` picks its joiner based on the title's final
character and normalises before hashing. So the rule lives here once, expressed as a call
to the same two functions the scan used, rather than re-derived at each call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from horizon.analysis import Analysis, content_hash
from horizon.ingest.wos import item_text


def record_hash(record: Mapping) -> str:
    """The ``content_hash`` the scan would have stored for this record.

    Empty when the record has no usable text — those never reached the model, so they
    have no analysis to join to.
    """
    text = item_text(dict(record))
    return content_hash(text) if text else ""


@dataclass(frozen=True)
class RecordIndex:
    """Records keyed by the hash their analyses carry."""

    by_hash: Mapping[str, Mapping]

    @classmethod
    def build(cls, records: Iterable[Mapping]) -> "RecordIndex":
        """Index an iterable of built records; unusable ones are skipped."""
        out: dict[str, Mapping] = {}
        for record in records:
            key = record_hash(record)
            if key:
                out[key] = record
        return cls(by_hash=out)

    def __len__(self) -> int:
        """How many records are indexed."""
        return len(self.by_hash)

    def get(self, analysis: Analysis) -> Mapping | None:
        """The record behind ``analysis``, or ``None`` if it is not in this index.

        ``None`` is normal rather than exceptional: an index built from one shard will
        not hold analyses from another, and callers filter on it.
        """
        return self.by_hash.get(analysis.content_hash)

    def pairs(self, analyses: Iterable[Analysis]) -> Iterator[tuple[Analysis, Mapping]]:
        """Yield ``(analysis, record)`` for the analyses this index can resolve."""
        for analysis in analyses:
            record = self.by_hash.get(analysis.content_hash)
            if record is not None:
                yield analysis, record

    def coverage(self, analyses: Iterable[Analysis]) -> tuple[int, int]:
        """``(resolved, total)`` — worth printing before trusting anything joined.

        A join that silently drops a fifth of the analyses produces tables that look
        complete and are not.
        """
        total = resolved = 0
        for analysis in analyses:
            total += 1
            resolved += analysis.content_hash in self.by_hash
        return resolved, total


#: Taxonomies to block on, in order. The two are complementary rather than redundant:
#: 20% of the corpus has no ``subject_areas``, but 18.1 points of that 20 carry
#: ``wos_categories``, so falling through cuts the unblockable remainder from 20% to
#: 2.1% — and that remainder would otherwise be a single 54 000-record block, larger
#: than any real field and defeating the point of blocking.
BLOCK_KEYS = ("subject_areas", "wos_categories")


def primary_field(record: Mapping, *, keys: Sequence[str] = BLOCK_KEYS) -> str | None:
    """The field a record is blocked under, or ``None`` when no taxonomy has one.

    The first of a sorted set, so a record listing its areas in a different order still
    lands in the same block — measured on the corpus the records are already in sorted
    order, so this costs nothing and removes a dependency on the export's ordering.

    45% of the corpus carries more than one area, so this is a choice with a cost:
    cross-disciplinary work is assigned somewhere rather than everywhere, and clustering
    inside blocks cannot see across the seam. A later two-stage merge
    is what recovers it.
    """
    for key in keys:
        areas = sorted({a for a in (record.get(key) or []) if a})
        if areas:
            return areas[0]
    return None


def block(
    pairs: Iterable[tuple[Analysis, Mapping]], *, keys: Sequence[str] = BLOCK_KEYS
) -> dict[str, list[Analysis]]:
    """Group joined analyses by primary field, unclassified ones under ``""``.

    Blocking is exact rather than approximate — every analysis lands in exactly one
    block and none is dropped — which is why it is preferred to sampling for cutting the
    clustering cost. On the corpus this yields 261 blocks with the largest at 12.5%,
    about 21x cheaper than clustering everything at once.
    """
    out: dict[str, list[Analysis]] = {}
    for analysis, record in pairs:
        out.setdefault(primary_field(record, keys=keys) or "", []).append(analysis)
    return out
