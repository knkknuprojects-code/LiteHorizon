"""Shared helpers for ``feedparser``-based adapters (RSS/Atom and arXiv).

Both the RSS and arXiv sources parse a feed via ``feedparser`` and need an entry's
publication date in the same ``YYYY-MM-DD`` shape ``scan --date`` expects. Text
extraction differs per source (an RSS entry's richest body vs. an arXiv title +
abstract), so only the date helper is shared here. Pure stdlib — no ``feedparser``
import — so it operates on an already-parsed entry mapping and adds no dependency.
"""

from __future__ import annotations

import time


def published_iso(entry) -> str | None:
    """An entry's publication date as ``YYYY-MM-DD`` (matches ``scan --date``), or None."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return time.strftime("%Y-%m-%d", parsed)
