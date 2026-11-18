"""Tests for the pure halves of scripts/threshold_sweep.py.

The clustering and embedding need the ML stack and a GPU to be worth running; the
sampling and the partition summary are plain data handling and are where the mistakes
would be silent.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "threshold_sweep",
    Path(__file__).resolve().parents[1] / "scripts" / "threshold_sweep.py",
)
sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["threshold_sweep"] = sweep
_SPEC.loader.exec_module(sweep)


def _items(spec):
    """``{year: n}`` to analyses carrying only what the sampler reads."""
    return [
        {"item_date": f"{year}-01-01", "content_hash": f"{year}-{i}"}
        for year, n in spec.items()
        for i in range(n)
    ]


def test_sample_keeps_each_year_in_proportion():
    """Velocity is read across years, so a sample must not re-weight the cohorts."""
    items = _items({2020: 100, 2021: 300})
    picked = sweep.stratified_sample(items, 40, seed=1)
    years = [it["item_date"][:4] for it in picked]
    assert len(picked) == 40
    assert years.count("2020") == 10 and years.count("2021") == 30


def test_sample_returns_everything_when_n_covers_the_input():
    items = _items({2020: 5})
    assert len(sweep.stratified_sample(items, 99, seed=1)) == 5


def test_sample_is_deterministic_for_a_seed():
    items = _items({2020: 50, 2021: 50})
    a = sweep.stratified_sample(items, 20, seed=3)
    b = sweep.stratified_sample(items, 20, seed=3)
    assert [x["content_hash"] for x in a] == [x["content_hash"] for x in b]


def test_sample_makes_up_a_rounding_shortfall():
    """Proportional rounding can land under the target; the total still has to match."""
    items = _items({2020: 3, 2021: 3, 2022: 3})
    assert len(sweep.stratified_sample(items, 5, seed=1)) == 5


def test_summarise_counts_clusters_and_coverage():
    # Sizes 3, 2, 1, 1 — only the 3 counts as covering at min_size=3.
    labels = [0, 0, 0, 1, 1, 2, 3]
    s = sweep.summarise(labels, min_size=3)
    assert s["clusters"] == 4 and s["singletons"] == 2
    assert s["largest"] == 3 and s["covered"] == pytest.approx(3 / 7)


def test_summarise_of_an_empty_partition():
    assert sweep.summarise([], min_size=2)["covered"] == 0


def test_load_analyses_groups_by_hash_and_model(tmp_path):
    root = tmp_path / "ds"
    (root / "data").mkdir(parents=True)
    rows = [
        {"content_hash": "h1", "model_id": "google/gemma-4-E4B-it"},
        {"content_hash": "h1", "model_id": "google/gemma-4-12B-it"},
        {"content_hash": "h2", "model_id": "google/gemma-4-E4B-it"},
    ]
    (root / "data" / "analyses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    both = sweep.load_analyses(root, ["E4B", "12B"])
    assert set(both["h1"]) == {"E4B", "12B"} and set(both["h2"]) == {"E4B"}
    # Without a model list everything collapses under one tag, last write winning.
    single = sweep.load_analyses(root, None)
    assert set(single["h1"]) == {"*"} and len(single) == 2


def _captured(a, b):
    """Stand in for the ARI, returning what it was handed instead of a score."""
    return (a, b)


def test_core_ari_reads_only_records_inside_a_real_cluster_on_both_sides():
    # Two triples survive on both sides; the trailing record is alone on both.
    a = [0, 0, 0, 1, 1, 1, 2]
    b = [9, 9, 9, 8, 8, 8, 7]
    assert sweep.core_ari(a, b, _captured, min_size=3) == (
        [0, 0, 0, 1, 1, 1],
        [9, 9, 9, 8, 8, 8],
    )


def test_core_ari_drops_records_in_a_real_cluster_on_only_one_side():
    a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    b = [9, 9, 9, 8, 8, 8, 7, 6, 5]  # the third triple is split into singletons
    assert sweep.core_ari(a, b, _captured, min_size=3) == (
        [0, 0, 0, 1, 1, 1],
        [9, 9, 9, 8, 8, 8],
    )


def test_core_ari_is_undefined_once_the_partition_has_degenerated():
    """Doubletons agree between orderings for free — the reason min_size exists."""
    a = [0, 0, 1, 1, 2, 2]
    b = [0, 0, 1, 1, 2, 2]
    assert sweep.core_ari(a, b, _captured, min_size=20) != sweep.core_ari(
        a, b, _captured, min_size=20
    )  # nan != nan


def test_core_ari_is_undefined_when_one_cluster_survives():
    """A single surviving group agrees with itself; that is not stability."""
    a, b = [0, 0, 0, 1], [9, 9, 9, 8]
    assert sweep.core_ari(a, b, _captured, min_size=3) != sweep.core_ari(
        a, b, _captured, min_size=3
    )
