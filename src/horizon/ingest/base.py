"""The ``Source`` interface and the ``RawItem`` it yields.

A ``Source`` is one place texts come from (an RSS feed, a folder, later arXiv/URLs).
Its ``fetch()`` yields :class:`RawItem`\\ s — normalized text plus the provenance we
want to carry forward (origin URL, source label, publication date, declared language).
Concrete adapters live in sibling modules and lazy-import their own dependencies, so
importing this package stays cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from horizon.analysis import as_attention

# The one *conventional* key in ``RawItem.meta``. Everything else in meta is free-form
# per-adapter payload; this key has a contract (see :class:`RawItem`), because the
# pipeline promotes it onto ``Analysis.attention``. Named, not inlined, so the three
# modules that touch it cannot drift.
ATTENTION_KEY = "attention"


@dataclass(frozen=True)
class RawItem:
    """One gathered text plus its provenance.

    ``text`` is already normalized (see ``normalize``). ``published_at`` is an ISO-8601
    string when the source provides a date (feeds ``item_date`` under ``--scan``); the
    typed fields (``source``/``url``/``published_at``/``lang``) are the *promoted*
    provenance the pipeline uses directly.

    ``meta`` is an open, JSON-safe dict for **everything else the source gave us** —
    title, author, tags/categories, guid, the alternate body (summary vs. full content),
    DOI/journal-ref for arXiv, sitename/description for pages, … Adapters funnel their
    full native payload here so nothing is discarded: it can't be recovered from the
    normalized text later, and it's often what makes the corpus reusable elsewhere. It's
    persisted verbatim into each item's ``corpus/<slug>/<hash>.json`` record.

    ``meta`` may carry one **conventional** key, :data:`ATTENTION_KEY` (``"attention"``):
    a float in ``(0, 1)`` for how much attention this item has drawn **relative to its
    own comparable cohort** — a peer group the adapter defines (WoS/Scopus: the citation
    percentile within the publication year; a repo or feed adapter could rank stars or
    engagement within its own window). Higher means more. It is deliberately a
    *normalized rank* rather than a raw count, so values from different adapters and eras
    stay commensurable — a raw count is age- and platform-confounded (in the WoS corpus,
    mean citations run 13.9 for 2015 against 1.9 for 2025, with a max of 15 185).
    **Absent means unknown, not zero** — omit the key rather than writing ``0.0``.
    Adapters keep their raw domain-specific numbers in ``meta`` too; this key is only the
    one the pipeline promotes onto ``Analysis.attention``.

    ``raw`` is the pre-normalization content (e.g. the fetched HTML) when the source can
    supply it — retained under ``raw/`` so the corpus can be *re-normalized* later
    without re-downloading; ``raw_suffix`` is the file extension to store it under. Left
    ``None`` when there's nothing worth keeping beyond ``text`` (e.g. local files, whose
    originals you already hold).
    """

    text: str
    source: str
    url: str | None = None
    published_at: str | None = None
    lang: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None
    raw_suffix: str = ".txt"


def attention_of(item: RawItem) -> float | None:
    """The item's conventional ``meta`` attention signal, or ``None``.

    ``None`` when the key is absent, non-numeric or NaN — "unknown" and "no attention"
    are different things here, so a malformed value must not silently become ``0.0``.
    """
    return as_attention((item.meta or {}).get(ATTENTION_KEY))


class Source(ABC):
    """A batch source of texts. ``fetch`` yields normalized :class:`RawItem`\\ s."""

    @abstractmethod
    def fetch(self) -> Iterable[RawItem]:
        """Yield the source's items. May hit the network; callers handle errors."""
        raise NotImplementedError
