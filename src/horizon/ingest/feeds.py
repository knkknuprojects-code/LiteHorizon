"""Load a feeds config for ``horizon gather feeds``.

A small TOML file (stdlib ``tomllib`` — no new dependency) listing RSS/Atom feeds so a
whole set is gathered in one run instead of one ``--url`` at a time. See ``feeds.toml``
at the repo root for the shipped example. Malformed entries are skipped, not fatal.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FEEDS_FILE = Path("feeds.toml")


@dataclass(frozen=True)
class Feed:
    """One configured RSS/Atom feed: a human label and its URL."""

    name: str
    url: str


def load_feeds(path: Path | str = DEFAULT_FEEDS_FILE) -> list[Feed]:
    """Read ``[[feed]]`` tables from a TOML file into :class:`Feed`\\ s.

    Entries missing a ``url`` are skipped; a missing ``name`` falls back to the URL.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"feeds config not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    feeds: list[Feed] = []
    for entry in data.get("feed", []):
        url = entry.get("url")
        if not url:
            continue
        feeds.append(Feed(name=entry.get("name") or url, url=url))
    return feeds
