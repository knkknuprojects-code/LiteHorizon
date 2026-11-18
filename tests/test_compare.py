"""Tests for stand-in-vs-teacher scan comparison (pure logic)."""

from __future__ import annotations

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.training.compare import (
    compare_scans,
    distinct_models,
    format_comparison,
    select_by_model,
)


def _analysis(
    h,
    model,
    *,
    maturity="emerging",
    impact="high",
    time_horizon="near",
    theme="open models",
    technologies=("LLM",),
    keywords=("open-weights",),
):
    return Analysis(
        summary="s",
        signals=("sig",),
        classification=Classification(
            theme=theme,
            maturity=Maturity(maturity),
            impact=Impact(impact),
            time_horizon=TimeHorizon(time_horizon),
        ),
        entities=Entities(technologies=tuple(technologies)),
        keywords=tuple(keywords),
        content_hash=h,
        model_id=model,
        prompt_version="phase2.v1",
        strategy="combined",
        generated_at="2026-01-01T00:00:00+00:00",
        item_date="2026-01-01T00:00:00+00:00",
    )


def test_distinct_models_orders_by_frequency():
    records = [
        _analysis("a", "Qwen/Qwen2.5-0.5B-Instruct"),
        _analysis("b", "Qwen/Qwen2.5-0.5B-Instruct"),
        _analysis("a", "gemini-3.5-flash"),
    ]
    assert distinct_models(records) == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "gemini-3.5-flash",
    ]


def test_select_by_model_matches_substring_and_takes_latest():
    records = [
        _analysis("a", "gemini-3.5-flash", theme="old"),
        _analysis("a", "gemini-3.5-flash", theme="new"),  # later wins
        _analysis("b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ]
    chosen = select_by_model(records, "gemini")
    assert set(chosen) == {"a"}
    assert chosen["a"].classification.theme == "new"


def test_compare_scans_measures_agreement_and_distributions():
    baseline = {
        "a": _analysis("a", "qwen", maturity="emerging", theme="ai", keywords=("x",)),
        "b": _analysis("b", "qwen", maturity="emerging", theme="policy"),
        "c": _analysis("c", "qwen", maturity="emerging"),  # teacher lacks c
    }
    teacher = {
        "a": _analysis(
            "a", "gemini", maturity="growing", theme="ai", keywords=("x", "y")
        ),
        "b": _analysis("b", "gemini", maturity="emerging", theme="markets"),
    }

    r = compare_scans(baseline, teacher)

    assert r["n_baseline"] == 3
    assert r["n_teacher"] == 2
    assert r["n_paired"] == 2  # a and b

    # maturity: a differs (emerging vs growing), b agrees -> 0.5
    assert r["enum_agreement"]["maturity"] == 0.5
    # impact + time_horizon identical across all -> 1.0 each
    assert r["enum_agreement"]["impact"] == 1.0
    assert r["enum_agreement"]["time_horizon"] == 1.0
    assert r["enum_agreement"]["overall"] == (0.5 + 1.0 + 1.0) / 3

    # theme: a agrees ("ai"), b differs ("policy" vs "markets") -> 0.5
    assert r["theme_match"] == 0.5

    # keyword F1: b identical (1.0); a is {x} vs {x,y} -> P=1,R=0.5 -> F1=0.667
    assert round(r["keyword_f1"], 3) == round((1.0 + 2 / 3) / 2, 3)

    # distributions are over the paired texts (a, b) only, so both sides compare
    # like-for-like — c (baseline-only) is excluded even though n counts it.
    assert r["baseline"]["distributions"]["maturity"] == {"emerging": 2}
    assert r["teacher"]["distributions"]["maturity"] == {"growing": 1, "emerging": 1}
    assert r["baseline"]["n"] == 3  # coverage still counts the unpaired record


def test_compare_scans_empty_pairing_is_safe():
    baseline = {"a": _analysis("a", "qwen")}
    teacher = {"b": _analysis("b", "gemini")}
    r = compare_scans(baseline, teacher)
    assert r["n_paired"] == 0
    assert r["enum_agreement"]["overall"] == 0.0
    assert r["keyword_f1"] == 0.0


def test_format_comparison_renders_key_lines():
    r = compare_scans(
        {"a": _analysis("a", "qwen")},
        {"a": _analysis("a", "gemini")},
    )
    out = format_comparison(r, baseline_label="qwen", teacher_label="gemini")
    assert "Scan comparison: qwen (baseline) vs gemini (teacher)" in out
    assert "enum overall" in out
    assert "label distribution" in out
    assert "trainable yield" in out
