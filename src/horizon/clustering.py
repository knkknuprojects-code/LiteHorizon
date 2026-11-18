"""Clustering backends for trend detection.

A ``Clusterer`` takes a list of (L2-normalized) vectors and returns an integer
cluster label per vector. Two interchangeable backends:

- :class:`GreedyClusterer` — dependency-free cosine-threshold, the local default.
- :class:`AgglomerativeClusterer` — ``scikit-learn`` (present transitively via
  ``sentence-transformers``), better quality at volume, for the cluster.

Both interpret the threshold as a **cosine similarity**: two items join the same
cluster when their similarity is at least ``threshold``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Clusterer(ABC):
    """Assigns an integer cluster label per vector. Backends choose the algorithm."""

    @abstractmethod
    def cluster(self, vectors: Sequence[Sequence[float]]) -> list[int]:
        """Return a cluster label for each input vector."""
        raise NotImplementedError


class GreedyClusterer(Clusterer):
    """Online cosine-threshold clustering.

    Each vector joins the most-similar existing cluster if their cosine similarity
    is ``>= threshold``, else it seeds a new cluster. Centroids are kept normalized
    and updated incrementally. Deterministic given input order.
    """

    def __init__(self, threshold: float) -> None:
        """Store the cosine-similarity threshold at which a vector joins a cluster."""
        self.threshold = threshold

    def cluster(self, vectors: Sequence[Sequence[float]]) -> list[int]:
        """Label vectors online: each joins its most-similar cluster if that
        similarity is ``>= threshold``, else seeds a new one; centroids stay
        normalized and are updated incrementally.

        The centroids live in one array so each item costs a single matrix-vector
        product rather than one ``np.dot`` per centroid. The arithmetic and the
        tie-breaking are unchanged — ``argmax`` returns the lowest index on a tie, as
        the ascending loop did — but the constant is a BLAS call instead of a Python
        iteration. That matters at corpus scale: the loop form measured 1.1 / 3.1 /
        6.6 ms per item at n = 2 000 / 5 000 / 10 000, i.e. quadratic overall, which
        extrapolates to roughly 13 hours over 269 065 analyses.

        The array grows by doubling. Appending with ``vstack`` would copy every
        centroid on every new cluster, which is the same O(k²) cost in a different
        place.
        """
        labels: list[int] = []
        counts: list[int] = []
        capacity, k = 0, 0
        centroids: np.ndarray | None = None
        for vec in vectors:
            v = _unit(np.asarray(vec, dtype=float))
            best, best_sim = -1, -1.0
            if k:
                sims = centroids[:k] @ v
                best = int(np.argmax(sims))
                best_sim = float(sims[best])
            if best >= 0 and best_sim >= self.threshold:
                n = counts[best]
                centroids[best] = _unit(centroids[best] * n + v)
                counts[best] = n + 1
                labels.append(best)
            else:
                if centroids is None:
                    capacity = 8
                    centroids = np.empty((capacity, v.shape[0]), dtype=float)
                elif k == capacity:
                    capacity *= 2
                    grown = np.empty((capacity, centroids.shape[1]), dtype=float)
                    grown[:k] = centroids[:k]
                    centroids = grown
                centroids[k] = v
                counts.append(1)
                labels.append(k)
                k += 1
        return labels


class AgglomerativeClusterer(Clusterer):
    """Bottom-up agglomerative clustering via scikit-learn (cosine distance)."""

    def __init__(self, threshold: float) -> None:
        """Store the cosine-similarity threshold at which a vector joins a cluster."""
        self.threshold = threshold

    def cluster(self, vectors: Sequence[Sequence[float]]) -> list[int]:
        """Cluster all vectors at once via scikit-learn average-linkage over cosine
        distance (``1 - threshold``). Inputs of ``<= 1`` vector get a single label;
        raises if scikit-learn is unavailable."""
        n = len(vectors)
        if n <= 1:
            return [0] * n
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:  # pragma: no cover - sklearn ships with s-t
            raise RuntimeError(
                "scikit-learn is required for --cluster-method agglomerative. "
                'Install it with: pip install -e ".[dev]"'
            ) from exc
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - self.threshold,  # cosine distance
        )
        return model.fit_predict(np.asarray(vectors, dtype=float)).tolist()


def make_clusterer(method: str, threshold: float) -> Clusterer:
    """Build a clusterer by name: ``greedy`` or ``agglomerative``."""
    if method == "greedy":
        return GreedyClusterer(threshold)
    if method == "agglomerative":
        return AgglomerativeClusterer(threshold)
    raise ValueError(f"unknown cluster method: {method!r} (use greedy|agglomerative)")


def _unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm; a zero vector is returned unchanged."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm
