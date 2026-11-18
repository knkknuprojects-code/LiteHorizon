"""Tests for the pure half of scripts/build_deck.py — facts in, pandoc markdown out."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_deck", Path(__file__).resolve().parents[1] / "scripts" / "build_deck.py"
)
deck = importlib.util.module_from_spec(_SPEC)
sys.modules["build_deck"] = deck
_SPEC.loader.exec_module(deck)


def _facts(**over):
    base = {
        "records": 100,
        "unclassified": 10,
        "years": [2020, 2021, 2022],
        "classified_years": [2020, 2021],
        "fields": [
            {
                "name": "Physics",
                "total": 60,
                "slope": 3.0,
                "share_first": 0.5,
                "share_last": 0.7,
                "share_delta": 0.2,
                "counts": {"2020": 30, "2021": 30, "2022": 0},
            },
            {
                "name": "Chemistry",
                "total": 40,
                "slope": -2.0,
                "share_first": 0.5,
                "share_last": 0.3,
                "share_delta": -0.2,
                "counts": {"2020": 20, "2021": 20, "2022": 0},
            },
        ],
        "pairs": [{"name": "Physics + Chemistry", "slope": 1.0, "total": 12}],
        "field_trends": {},
        "overall": [
            {
                "label": "Topic A",
                "size": 30,
                "velocity": 1.2,
                "growth": 4.0,
                "score": 9.0,
                "first_seen": "2020-01-01",
                "cohort_counts": {"2020": 10, "2021": 20},
                "attention_pctile": 0.8,
            }
        ],
        "lenses": {},
        "meta_lenses": {},
    }
    base.update(over)
    return base


def test_the_deck_names_the_cohorts_the_shares_were_measured_at():
    out = deck.build_deck(_facts())
    assert "2015" not in out
    assert "· 2020–2021" in out


def test_classified_years_are_derived_when_the_facts_predate_the_field():
    """An older facts.json must not silently label the corpus span instead.

    They differ exactly when the last cohort carries no taxonomy — the case the label
    exists to name — so a fallback to `years` would print a span the numbers were never
    measured over.
    """
    facts = _facts()
    del facts["classified_years"]
    assert deck._classified_years(facts) == [2020, 2021]
    assert "· 2020–2021" in deck.build_deck(facts)


def test_a_missing_indicator_is_skipped_rather_than_rendered_empty():
    out = deck.build_deck(_facts())
    assert "Відкладене визнання" not in out


def test_an_indicator_present_in_the_facts_gets_its_own_slide_and_column():
    facts = _facts(
        lenses={
            "sleeping-beauty": [
                {
                    "label": "Dormant topic",
                    "size": 25,
                    "first_seen": "2020-01-01",
                    "cohort_counts": {"2020": 2, "2023": 23},
                }
            ]
        }
    )
    out = deck.build_deck(facts)
    assert "## Відкладене визнання" in out
    assert "тривалість латентного періоду, років" in out
    assert "| Dormant topic | 25 | 3 |" in out


def test_dormancy_and_sustain_survive_a_topic_with_no_curve():
    bare = {"first_seen": "2020-01-01", "cohort_counts": {}}
    assert deck._dormancy(bare) == 0.0 and deck._sustain(bare) == 0.0


def test_tables_are_pipe_tables_pandoc_can_turn_into_slides():
    rows = deck._table(["a", "b"], [["1", "2"]])
    assert rows[0] == "| a | b |" and rows[1] == "|---|---|"


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_both_languages_render(lang):
    assert deck.build_deck(_facts(), lang=lang).count("\n## ") == 7
