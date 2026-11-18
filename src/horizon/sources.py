"""Sidecar store mapping ``content_hash`` -> original source text.

Kept separate from ``analyses.jsonl`` so the (possibly large) raw text doesn't
bloat the analysis store, and so Phase-4 export can recover the *exact* inputs a
fine-tuning prompt was built from. Mirrors the embeddings sidecar: an append-only
JSONL keyed by ``content_hash`` (last write wins), written during ``scan`` and
read during ``export``.
"""

from __future__ import annotations

import json
from pathlib import Path

from horizon.analysis import content_hash

DEFAULT_SOURCE_PATH = Path("data") / "sources.jsonl"


class SourceStore:
    """Append-only JSONL of ``{content_hash, text}`` records (last write wins)."""

    def __init__(self, path: Path | str = DEFAULT_SOURCE_PATH) -> None:
        """Store the JSONL path (its parent dir is created lazily on first ``put``)."""
        self.path = Path(path)

    def put(self, text: str) -> str:
        """Persist ``text`` keyed by its content hash; return the hash."""
        digest = content_hash(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"content_hash": digest, "text": text}, ensure_ascii=False)
                + "\n"
            )
        return digest

    def get(self, digest: str) -> str | None:
        """Return the stored text for ``digest``, or ``None`` if unknown."""
        return self.all().get(digest)

    def all(self) -> dict[str, str]:
        """Every stored source as ``{content_hash: text}`` (last write wins)."""
        if not self.path.is_file():
            return {}
        out: dict[str, str] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out[data["content_hash"]] = data["text"]
        return out
