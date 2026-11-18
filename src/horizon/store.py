"""Local persistence for analyses.

Phase 2 uses a simple append-only JSONL file, keyed by
``(content_hash, prompt_version)`` so a re-run of the same text under the same
prompt version is a cache hit (skipping the model call). The :class:`Store`
protocol lets Phase 3 swap in a queryable backend (e.g. SQLite) without touching
callers. ``strategy`` is intentionally **not** part of the key — a ``combined`` and
a ``staged`` analysis of the same text are interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from horizon.analysis import Analysis

DEFAULT_STORE_PATH = Path("data") / "analyses.jsonl"


def _key(content_hash: str, prompt_version: str) -> tuple[str, str]:
    """The store's cache key: ``(content_hash, prompt_version)``."""
    return (content_hash, prompt_version)


class Store(Protocol):
    """Persist and look up analyses by content hash + prompt version."""

    def get(self, content_hash: str, prompt_version: str) -> Analysis | None:
        """Return the analysis for this content hash + prompt version, or ``None``."""
        ...

    def put(self, analysis: Analysis) -> None:
        """Persist ``analysis``."""
        ...

    def list_all(self) -> list[Analysis]:
        """Return every stored analysis (one per key)."""
        ...


class JsonlStore:
    """Append-only JSONL store. Last write for a key wins on read."""

    def __init__(self, path: Path | str = DEFAULT_STORE_PATH) -> None:
        """Store the JSONL path (its parent dir is created lazily on first ``put``)."""
        self.path = Path(path)

    def get(self, content_hash: str, prompt_version: str) -> Analysis | None:
        """Return the last-written analysis for this key, or ``None`` if absent.

        Scans the whole file; later lines override earlier ones for the same key.
        """
        if not self.path.is_file():
            return None
        target = _key(content_hash, prompt_version)
        found: Analysis | None = None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if _key(data["content_hash"], data["prompt_version"]) == target:
                    found = Analysis.from_dict(data)  # later lines override earlier
        return found

    def put(self, analysis: Analysis) -> None:
        """Append ``analysis`` as one JSON line, creating the parent dir if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(analysis.to_dict(), ensure_ascii=False) + "\n")

    def list_all(self) -> list[Analysis]:
        """Every stored analysis, one per key (last write wins), in first-seen order."""
        if not self.path.is_file():
            return []
        by_key: dict[tuple[str, str], Analysis] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                by_key[_key(data["content_hash"], data["prompt_version"])] = (
                    Analysis.from_dict(data)
                )
        return list(by_key.values())

    def list_all_records(self) -> list[Analysis]:
        """Every appended record, uncollapsed, in write order.

        Unlike :meth:`list_all` (which keeps only the last write per key), this
        preserves overwritten versions — so a teacher re-scan and the stand-in
        analysis it overwrote both survive here. That's what the stand-in-vs-teacher
        comparison reads: same key, different ``model_id``.
        """
        if not self.path.is_file():
            return []
        records: list[Analysis] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(Analysis.from_dict(json.loads(line)))
        return records
