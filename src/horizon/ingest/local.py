"""Local directory source — ingest a folder of prepared text files.

The zero-risk adapter: text you already hold, no network, no licensing burden. Reads
``*.txt`` / ``*.md`` (recursively), normalizes each, and dates items by file mtime so
they carry a plausible ``published_at`` until a real one is supplied. ``.pdf`` is left
as a follow-on (needs a parser dependency).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from horizon.ingest.base import RawItem, Source
from horizon.ingest.normalize import normalize_text

DEFAULT_SUFFIXES = (".txt", ".md")


class LocalDirSource(Source):
    """Yield a :class:`RawItem` per readable text file under ``path``."""

    def __init__(
        self, path: Path | str, suffixes: tuple[str, ...] = DEFAULT_SUFFIXES
    ) -> None:
        """Store the directory to walk and the file suffixes to accept."""
        self.path = Path(path)
        self.suffixes = suffixes

    def fetch(self) -> Iterable[RawItem]:
        """Yield a :class:`RawItem` per readable text file under ``path`` (recursive).

        Raises ``NotADirectoryError`` if ``path`` is not a directory; unreadable/binary
        files and empty-after-normalization files are skipped. Items are dated by mtime.
        """
        if not self.path.is_dir():
            raise NotADirectoryError(f"not a directory: {self.path}")
        # Resolve so file.as_uri() works regardless of a relative --path.
        base = self.path.resolve()
        items: list[RawItem] = []
        for file in sorted(base.rglob("*")):
            if file.suffix.lower() not in self.suffixes or not file.is_file():
                continue
            try:
                raw = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # skip unreadable/binary files rather than fail the batch
            text = normalize_text(raw)
            if not text:
                continue
            mtime = time.strftime("%Y-%m-%d", time.gmtime(file.stat().st_mtime))
            items.append(
                RawItem(
                    text=text,
                    source="local",
                    url=file.as_uri(),
                    published_at=mtime,
                    meta={"filename": file.name, "path": str(file), "mtime": mtime},
                )
            )
        return items
