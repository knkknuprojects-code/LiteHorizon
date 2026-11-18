"""Tests for folding a split interpret run back into one reading."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_interpretation import merge_readings  # noqa: E402


def _reading(**blocks):
    base = {"model_id": "google/gemma-4-12B-it", "label": "12B (8bit)", "lang": "uk"}
    base.update(blocks)
    return base


def test_passes_from_one_model_become_one_reading():
    """A run split across two boxes writes one file per pass. Unmerged they would be
    treated as two models and printed side by side under identical names — the layout
    reserved for showing genuine disagreement between models."""
    merged = merge_readings(
        [
            _reading(topics={"a": {"text": "t"}}),
            _reading(areas={"x": {"text": "a"}}, corpus={"text": "c"}),
        ]
    )
    assert len(merged) == 1
    assert set(merged[0]["topics"]) == {"a"}
    assert set(merged[0]["areas"]) == {"x"}
    assert merged[0]["corpus"]["text"] == "c"


def test_different_models_stay_apart():
    """Two models are the case the side-by-side layout exists for."""
    a = _reading(topics={"t": {"text": "1"}})
    b = dict(a, model_id="google/gemma-4-E4B-it", label="E4B")
    assert len(merge_readings([a, b])) == 2


def test_the_same_model_at_two_quantizations_stays_apart():
    """Same weights, different generator — that is why the label carries the quant."""
    a = _reading(label="12B (8bit)")
    b = _reading(label="12B (4bit)")
    assert len(merge_readings([a, b])) == 2


def test_a_rerun_does_not_replace_what_the_other_pass_wrote():
    """Within a key the first block wins, so re-running one pass cannot quietly drop
    readings the other produced."""
    merged = merge_readings(
        [
            _reading(topics={"a": {"text": "first"}}),
            _reading(topics={"a": {"text": "second"}, "b": {"text": "new"}}),
        ]
    )
    assert merged[0]["topics"]["a"]["text"] == "first"
    assert merged[0]["topics"]["b"]["text"] == "new"


def test_languages_are_never_folded_together():
    a = _reading(lang="uk")
    b = _reading(lang="en")
    assert len(merge_readings([a, b])) == 2
