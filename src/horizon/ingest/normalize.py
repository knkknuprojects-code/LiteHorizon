"""Minimal, dependency-free text normalization for gathered items.

C3 (locked slice): *minimal* cleanup for feeds and local files — strip HTML tags,
unescape entities, collapse whitespace, and reject empty/near-empty text. Boilerplate
/ nav removal (readability-style) is deferred with ``UrlSource``, where that noise
actually lives, and would come behind an optional dependency.

Uses only the standard library (``html.parser`` + ``html.unescape``) so the ingest
package adds no dependency for the local-dir path and only ``feedparser`` for RSS.
"""

from __future__ import annotations

from html.parser import HTMLParser

# Text shorter than this after cleanup is treated as empty (nav crumbs, "…", etc.).
MIN_LENGTH = 20


class _TextExtractor(HTMLParser):
    """Collect text nodes, dropping the content of script/style elements."""

    _SKIP = {"script", "style"}

    def __init__(self) -> None:
        """Set up the collected-text buffer and the skip-depth counter."""
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        """Enter skip mode when opening a script/style element."""
        if tag in self._SKIP:
            self._skipping += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave skip mode when closing a script/style element."""
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data: str) -> None:
        """Collect a text node unless inside a skipped script/style element."""
        if not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        """The collected text nodes joined with single spaces."""
        return " ".join(self._parts)


def strip_html(raw: str) -> str:
    """Return the text content of ``raw``, dropping tags and script/style bodies.

    ``convert_charrefs=True`` unescapes entities in element text; anything the parser
    misses (e.g. an already-plaintext string) is passed through unchanged.
    """
    if "<" not in raw and ">" not in raw:
        return raw  # plain text — nothing to strip
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    return parser.text()


def normalize_text(raw: str) -> str:
    """Clean ``raw`` into training-ready plaintext: de-HTML + collapse whitespace.

    Returns ``""`` for text that is empty or too short to be a real item (the caller
    treats that as skip-empty). Whitespace is collapsed the same way ``content_hash``
    normalizes, so a normalized text hashes stably for dedup.
    """
    text = " ".join(strip_html(raw).split())
    return text if len(text) >= MIN_LENGTH else ""
