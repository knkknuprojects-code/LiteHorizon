"""arXiv source — recent papers matching a search query, one :class:`RawItem` each.

The arXiv API (``export.arxiv.org/api/query``) returns an Atom feed that ``feedparser``
already parses, so this adapter reuses the ``[ingest]`` extra with **no new dependency**.
It's a C1 follow-on: arXiv skews sci/tech, so it complements — rather than replaces — the
broader RSS/local sources. Each result's title + abstract become the item text; the
paper's abstract page is the ``url``.

Be a good API citizen: arXiv asks callers to keep request rates modest and to page with
``max_results`` rather than pulling everything at once. This adapter issues one query per
``fetch()`` and leaves paging to the caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlencode

from horizon.ingest.base import RawItem, Source
from horizon.ingest.feedparse import published_iso
from horizon.ingest.normalize import normalize_text

API_URL = "http://export.arxiv.org/api/query"


def _entry_meta(entry) -> dict:
    """The structured arXiv fields the text blob doesn't preserve — authors, categories,
    DOI, journal-ref, links — kept JSON-safe. These aren't recoverable from the abstract
    later and are exactly what makes arXiv data reusable elsewhere.
    """
    meta: dict = {}
    for key in ("title", "summary", "id", "published", "updated"):
        val = entry.get(key)
        if val:
            meta[key] = val
    authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
    if authors:
        meta["authors"] = authors
    cats = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
    if cats:
        meta["categories"] = cats
    primary = entry.get("arxiv_primary_category")
    if primary and primary.get("term"):
        meta["primary_category"] = primary["term"]
    for key in ("arxiv_doi", "arxiv_journal_ref", "arxiv_comment"):
        val = entry.get(key)
        if val:
            meta[key.removeprefix("arxiv_")] = val
    links = {
        (ln.get("type") or ln.get("rel")): ln.get("href")
        for ln in entry.get("links", [])
        if ln.get("href")
    }
    if links:
        meta["links"] = links
    return meta


class ArxivSource(Source):
    """Fetch recent arXiv results for a search query, newest first.

    ``query`` is an arXiv search string — e.g. ``"cat:cs.CL"`` (a category) or
    ``"all:quantum error correction"`` (a full-text search); see the arXiv API docs for
    the field syntax. ``max_results`` caps the batch. ``source`` labels the batch in the
    manifest / as ``origin`` (default ``"arxiv"``); each item's specific paper lives in
    its ``url``.
    """

    def __init__(
        self, query: str, max_results: int = 50, source: str = "arxiv"
    ) -> None:
        """Store the arXiv search query, batch cap, and manifest source label."""
        self.query = query
        self.max_results = max_results
        self._source = source

    def _url(self) -> str:
        """The arXiv API query URL (separated out so tests can point ``fetch`` offline)."""
        return f"{API_URL}?" + urlencode(
            {
                "search_query": self.query,
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )

    def fetch(self) -> Iterable[RawItem]:
        """Query the arXiv API and yield a :class:`RawItem` per result entry.

        Each item's text is the title joined to the abstract; entries that normalize to
        empty are skipped. ``feedparser`` is imported lazily, only needed here.
        """
        import feedparser  # lazy: only needed when actually querying arXiv

        parsed = feedparser.parse(self._url())
        items: list[RawItem] = []
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            # Title gives the headline, abstract the detail; normalize collapses the
            # newlines/whitespace arXiv wraps both in.
            raw = f"{title}. {summary}" if title else summary
            text = normalize_text(raw)
            if not text:
                continue
            items.append(
                RawItem(
                    text=text,
                    source=self._source,
                    url=entry.get("link"),
                    published_at=published_iso(entry),
                    lang="en",
                    meta=_entry_meta(entry),
                    raw=raw,
                )
            )
        return items
