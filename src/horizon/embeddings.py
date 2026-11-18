"""Embeddings for trend detection.

Mirrors the ``LLMClient`` pattern: an ``Embedder`` interface plus a lazy local
backend, so the package imports without the ML stack and the model loads on first
use. The default model is **multilingual** (``paraphrase-multilingual-MiniLM-L12-v2``)
so analyses written in different languages cluster by meaning.

``EmbeddingCache`` persists vectors in a sidecar JSONL keyed by
``(content_hash, model)`` — load the embedder once, never recompute a seen text.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from horizon.analysis import Analysis
from horizon.config import Settings

DEFAULT_EMBED_CACHE_PATH = Path("data") / "embeddings.jsonl"


def embed_text(analysis: Analysis) -> str:
    """The text we embed for an analysis: its summary plus the signal phrases."""
    return " ".join([analysis.summary, *analysis.signals]).strip()


class Embedder(ABC):
    """Turns texts into dense vectors. Backends decide the model/runtime."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text."""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Local ``sentence-transformers`` backend; the model loads on first use."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Record settings and model id; the model itself loads on first ``embed``."""
        self.settings = settings or Settings()
        self.model_id = self.settings.embed_model
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load the sentence-transformers model on first call, raising a clear error
        if the package or the model itself is unavailable."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "sentence-transformers is required for trend detection. "
                'Install it with: pip install -e ".[dev]"'
            ) from exc
        try:
            self._model = SentenceTransformer(
                self.model_id, device=self.settings.device
            )
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                f"Could not load embedding model {self.model_id!r}: {exc}."
            ) from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode ``texts`` into L2-normalized vectors, loading the model on first
        call; empty input returns no vectors."""
        if not texts:
            return []
        self._ensure_loaded()
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vectors]


class EmbeddingCache:
    """Caches analysis embeddings in a sidecar JSONL keyed by content hash + model."""

    def __init__(
        self, embedder: Embedder, path: Path | str = DEFAULT_EMBED_CACHE_PATH
    ) -> None:
        """Bind an embedder to its sidecar JSONL path, keyed by the embedder's
        model id (``"fake"`` when it has none)."""
        self.embedder = embedder
        self.path = Path(path)
        self.model = getattr(embedder, "model_id", "fake")
        self._cache: dict[str, list[float]] | None = None

    def _load(self) -> dict[str, list[float]]:
        """This model's cached vectors as ``content_hash -> vector``, read once.

        Memoised because the caller is a loop over blocks, and re-parsing the sidecar per
        block is quadratic in a way that only shows up at corpus scale: the full-corpus
        sidecar is 2.2 GB, and a 250-block report was re-reading all of it 250 times —
        slower with a warm cache than the run that computed the vectors from scratch.

        The consequence is that a sidecar written by another process mid-run is not
        picked up. That is the right trade here: one report owns its store for the length
        of a run, and every write this instance makes updates the map in step.
        """
        if self._cache is not None:
            return self._cache
        if not self.path.is_file():
            self._cache = {}
            return self._cache
        cache: dict[str, list[float]] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("model") == self.model:  # ignore other models' vectors
                    cache[rec["content_hash"]] = rec["vector"]
        self._cache = cache
        return cache

    def _append(self, content_hash: str, vector: list[float]) -> None:
        """Append one ``content_hash``/``model``/``vector`` record to the sidecar."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "content_hash": content_hash,
                        "model": self.model,
                        "vector": vector,
                    }
                )
                + "\n"
            )

    def embed_analyses(self, analyses: Sequence[Analysis]) -> dict[str, list[float]]:
        """Return ``content_hash -> vector`` for ``analyses``, computing only misses."""
        cache = self._load()
        # De-duplicate missing hashes while preserving order.
        missing: list[Analysis] = []
        seen: set[str] = set()
        for a in analyses:
            if a.content_hash not in cache and a.content_hash not in seen:
                missing.append(a)
                seen.add(a.content_hash)
        if missing:
            vectors = self.embedder.embed([embed_text(a) for a in missing])
            for a, vec in zip(missing, vectors):
                cache[a.content_hash] = vec
                self._append(a.content_hash, vec)
        return {a.content_hash: cache[a.content_hash] for a in analyses}
