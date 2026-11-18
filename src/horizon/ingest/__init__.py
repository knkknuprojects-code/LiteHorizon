"""Corpus gathering (Phase 5) — batch source adapters for building training data.

One ``Source`` interface with lazy backends (mirrors ``LLMClient`` / ``Embedder``):
each adapter fetches items from a real source and yields normalized
:class:`RawItem`\\ s. Kept separate from the ``SourceStore`` sidecar in
``horizon.sources`` (which persists the *input text* of a scan) — this package is
about *where texts come from*, at volume, with provenance.

First slice (C1 locked): RSS/Atom + local dir; arXiv is a
sci/tech-skewed follow-on that reuses the same ``feedparser`` extra. ``WosSource`` reads
the ``collection/`` WoS+Scopus workbooks once built (see ``horizon.ingest.collection``).
"""

from __future__ import annotations

from horizon.ingest.arxiv import ArxivSource
from horizon.ingest.base import RawItem, Source
from horizon.ingest.corpus import CorpusSource
from horizon.ingest.feeds import Feed, load_feeds
from horizon.ingest.local import LocalDirSource
from horizon.ingest.rss import RssSource
from horizon.ingest.url import UrlSource
from horizon.ingest.wos import WosSource

__all__ = [
    "RawItem",
    "Source",
    "LocalDirSource",
    "RssSource",
    "ArxivSource",
    "UrlSource",
    "CorpusSource",
    "WosSource",
    "Feed",
    "load_feeds",
]
