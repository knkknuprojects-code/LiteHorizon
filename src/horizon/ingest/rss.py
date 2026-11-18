"""RSS/Atom source — one :class:`RawItem` per feed entry.

Lazy-imports ``feedparser`` (the ``[ingest]`` extra) so the core package and the
local-dir path need no new dependency. ``feedparser.parse`` accepts a URL, a local
file path, or a raw string, so this adapter is testable fully offline by pointing it
at a fixture file.
"""

from __future__ import annotations

from collections.abc import Iterable

from horizon.ingest.base import RawItem, Source
from horizon.ingest.feedparse import published_iso
from horizon.ingest.normalize import normalize_text


def _entry_text(entry) -> str:
    """Richest text a feed entry offers: full content if present, else summary."""
    content = entry.get("content")
    if content:
        return content[0].get("value", "")
    return entry.get("summary", "") or entry.get("title", "")


def _entry_meta(entry, feed) -> dict:
    """Everything the feed entry carries beyond text/url/date — kept JSON-safe.

    feedparser entries hold ``struct_time`` objects and nested dicts, so we whitelist
    the plainly-serializable, plainly-useful fields rather than dumping the raw entry.
    Whatever isn't present is simply omitted.
    """
    meta: dict = {}
    for key in ("title", "author", "id", "published", "updated", "summary"):
        val = entry.get(key)
        if val:
            meta[key] = val
    tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
    if tags:
        meta["tags"] = tags
    # Keep the alternate body: if text came from ``summary`` we still keep full content,
    # and vice-versa — the reader chose one, we retain both.
    content = entry.get("content")
    if content and content[0].get("value"):
        meta["content"] = content[0]["value"]
    links = [ln.get("href") for ln in entry.get("links", []) if ln.get("href")]
    if links:
        meta["links"] = links
    if feed and feed.get("title"):
        meta["feed_title"] = feed["title"]
    return meta


class RssSource(Source):
    """Fetch and normalize the entries of one RSS/Atom feed.

    ``location`` is anything ``feedparser`` accepts — a feed URL for real use, or a
    local file path / raw XML string in tests. Malformed entries are skipped, not
    fatal; an empty-after-normalization entry is dropped. ``source`` overrides the
    label recorded in the manifest (e.g. the ``name`` from ``feeds.toml``); when unset
    it falls back to the feed's own title, then the location.
    """

    def __init__(self, location: str, source: str | None = None) -> None:
        """Store the feed location and the optional manifest source-label override."""
        self.location = location
        self._source = source

    def fetch(self) -> Iterable[RawItem]:
        """Parse the feed and yield a :class:`RawItem` per entry.

        Empty-after-normalization entries are dropped; the feed body (HTML) is kept as
        ``raw`` for later re-normalization. ``feedparser`` is imported lazily here.
        """
        import feedparser  # lazy: only needed when actually gathering from a feed

        parsed = feedparser.parse(self.location)
        feed_title = parsed.feed.get("title") if parsed.feed else None
        source = self._source or feed_title or self.location
        items: list[RawItem] = []
        for entry in parsed.entries:
            raw = _entry_text(entry)
            text = normalize_text(raw)
            if not text:
                continue
            items.append(
                RawItem(
                    text=text,
                    source=source,
                    url=entry.get("link"),
                    published_at=published_iso(entry),
                    lang=parsed.feed.get("language") if parsed.feed else None,
                    meta=_entry_meta(entry, parsed.feed),
                    raw=raw,  # feed body (HTML) kept for later re-normalization
                    raw_suffix=".html",
                )
            )
        return items
