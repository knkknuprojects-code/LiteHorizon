import pytest

from horizon.clustering import (
    AgglomerativeClusterer,
    GreedyClusterer,
    make_clusterer,
)

# Orthogonal unit vectors (cosine 0) and a 45-degree vector (cosine ~0.707).
A = [1.0, 0.0]
B = [0.0, 1.0]
DIAG = [1.0, 1.0]


def _n_clusters(labels):
    return len(set(labels))


# --- greedy -------------------------------------------------------------------


def test_greedy_separates_orthogonal():
    labels = GreedyClusterer(0.5).cluster([A, A, B])
    assert labels[0] == labels[1] != labels[2]
    assert _n_clusters(labels) == 2


def test_greedy_threshold_boundary():
    # cos(A, DIAG) ~ 0.707: merges below the threshold, splits above it.
    assert _n_clusters(GreedyClusterer(0.6).cluster([A, DIAG])) == 1
    assert _n_clusters(GreedyClusterer(0.8).cluster([A, DIAG])) == 2


def test_greedy_empty():
    assert GreedyClusterer(0.5).cluster([]) == []


# --- agglomerative (sklearn ships with sentence-transformers) -----------------


def test_agglomerative_separates_orthogonal():
    labels = AgglomerativeClusterer(0.5).cluster([A, A, B])
    assert _n_clusters(labels) == 2


def test_agglomerative_single_vector():
    assert AgglomerativeClusterer(0.5).cluster([A]) == [0]


# --- factory ------------------------------------------------------------------


def test_make_clusterer_dispatch():
    assert isinstance(make_clusterer("greedy", 0.5), GreedyClusterer)
    assert isinstance(make_clusterer("agglomerative", 0.5), AgglomerativeClusterer)


def test_make_clusterer_unknown():
    with pytest.raises(ValueError):
        make_clusterer("dbscan", 0.5)


def _greedy_reference(vectors, threshold):
    """The straightforward loop the vectorised implementation replaced.

    Kept in the tests rather than in the module: it is the definition of what
    ``GreedyClusterer`` must do, and the fast version is only worth having if it
    agrees with it exactly, tie-breaking included.
    """
    import numpy as np

    from horizon.clustering import _unit

    labels, centroids, counts = [], [], []
    for vec in vectors:
        v = _unit(np.asarray(vec, dtype=float))
        best, best_sim = -1, -1.0
        for i, c in enumerate(centroids):
            sim = float(np.dot(v, c))
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and best_sim >= threshold:
            n = counts[best]
            centroids[best] = _unit(centroids[best] * n + v)
            counts[best] = n + 1
            labels.append(best)
        else:
            centroids.append(v)
            counts.append(1)
            labels.append(len(centroids) - 1)
    return labels


@pytest.mark.parametrize("threshold", [0.3, 0.55, 0.8])
def test_greedy_matches_the_reference_loop(threshold):
    """Vectorising must not change a single label, at any threshold."""
    import numpy as np

    rng = np.random.default_rng(11)
    # Enough vectors to cross the capacity-doubling boundary (8, 16, 32, ...) several
    # times, and enough structure that some join and some seed.
    centers = rng.normal(size=(7, 24))
    vectors = [
        (centers[rng.integers(0, 7)] + rng.normal(scale=0.6, size=24)).tolist()
        for _ in range(120)
    ]
    assert GreedyClusterer(threshold).cluster(vectors) == _greedy_reference(
        vectors, threshold
    )


def test_greedy_ties_go_to_the_lowest_index():
    """Two equally similar centroids: the first one wins, as the old loop did."""
    # A and B are orthogonal; DIAG sits at 45 degrees from both, so the similarities
    # are equal and the tie-break is the only thing deciding the label.
    labels = GreedyClusterer(0.7).cluster([A, B, DIAG])
    assert labels == [0, 1, 0]


def test_greedy_zero_vector_seeds_its_own_cluster():
    """A zero vector has similarity 0 with everything, so it never joins above 0."""
    assert GreedyClusterer(0.5).cluster([A, [0.0, 0.0]]) == [0, 1]
