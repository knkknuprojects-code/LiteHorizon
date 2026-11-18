"""Tests for the comparison's reading pass.

No model runs here. What is checked is the shape of what a model would be given: a prompt
that silently lost a placeholder in one language, or that listed topics the comparison
never tested, produces prose that reads perfectly well and is about the wrong thing.
"""

import importlib.util
import re
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_interpret",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_interpret.py",
)
ci = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_interpret"] = ci
_SPEC.loader.exec_module(ci)

FACTS = {
    "kind": "comparison",
    "present_years": [2015, 2016, 2017],
    "measured_years": [2015, 2016],
    "a": {
        "name": {"en": "Register", "uk": "Реєстр"},
        "taxonomy": {"en": "WIPO fields", "uk": "Поля WIPO"},
        "records": {"2015": 100, "2016": 80, "2017": 40},
        "coverage": {"2015": 1.0, "2016": 0.9, "2017": 0.5},
    },
    "b": {
        "name": {"en": "Corpus", "uk": "Корпус"},
        "taxonomy": {"en": "Subject areas", "uk": "Предметні області"},
        "records": {"2015": 1000, "2016": 1200, "2017": 1300},
        "coverage": {},
    },
    "volume": {
        "a": [
            {"year": 2015, "n": 100, "index": 100.0, "measured": True},
            {"year": 2016, "n": 80, "index": 80.0, "measured": True},
            {"year": 2017, "n": 40, "index": 40.0, "measured": False},
        ],
        "b": [
            {"year": 2015, "n": 1000, "index": 100.0, "measured": True},
            {"year": 2016, "n": 1200, "index": 120.0, "measured": True},
            {"year": 2017, "n": 1300, "index": 130.0, "measured": False},
        ],
    },
    "fields": {
        "a": [{"name": "Metals", "first": 0.1, "last": 0.2, "delta": 0.1, "n": 30}],
        "b": [{"name": "Medicine", "first": 0.2, "last": 0.1, "delta": -0.1, "n": 300}],
    },
    "shape": {
        "a": {
            "fields": 35,
            "multi": 1.55,
            "top5": 0.357,
            "effective": 20.9,
            "pairs": 5,
        },
        "b": {
            "fields": 27,
            "multi": 1.48,
            "top5": 0.546,
            "effective": 12.7,
            "pairs": 3,
        },
    },
    "pairs": {
        "a": [{"name": "Metals + Chemistry", "first": 0.01, "last": 0.02}],
        "b": [],
    },
    "matches": [
        {
            "label": "Tested Topic",
            "a_n": 40,
            "b_n": 50,
            "a_slope": 1.0,
            "b_slope": -1.0,
            "a_keywords": ["alpha", "beta"],
            "b_keywords": ["gamma"],
        },
        {
            "label": "Untested Topic",
            "a_n": 2,
            "b_n": 3,
            "a_slope": 9.0,
            "b_slope": 9.0,
            "a_keywords": [],
            "b_keywords": [],
        },
    ],
    "agreement": {
        "n": 1,
        "same": 0,
        "all_n": 2,
        "all_same": 1,
        "r": 0.12,
        "rows": ["Tested Topic"],
    },
}


def _placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{(\w+)\}", template))


def test_both_languages_of_every_prompt_take_the_same_placeholders():
    """A placeholder dropped from one language raises at format time — on that run only.

    The Ukrainian prompt is the one that runs; the English one would go untested until
    someone asked for it, by which point the pass is hours long and the failure is a
    KeyError at the end of it.
    """
    for name, prompt in (
        ("comparison", ci.COMPARISON_PROMPT),
        ("taxonomy", ci.TAXONOMY_PROMPT),
        ("matches", ci.MATCHES_PROMPT),
    ):
        assert _placeholders(prompt["uk"]) == _placeholders(prompt["en"]), name


def test_every_prompt_renders_in_both_languages():
    for lang in ("uk", "en"):
        assert FACTS["a"]["name"][lang] in ci.comparison_prompt(FACTS, lang, 20)
        assert FACTS["a"]["taxonomy"][lang] in ci.taxonomy_prompt(FACTS, "a", lang)
        assert "Tested Topic" in ci.matches_prompt(FACTS, lang, 20)


def test_the_matches_prompt_lists_only_what_the_direction_test_ran_on():
    """A pair carried by two records has a share slope, and it is not worth reading.

    Listing it would put a +9.0 against a +9.0 in front of the model as though it agreed
    with something.
    """
    prompt = ci.matches_prompt(FACTS, "uk", 20)
    assert "Tested Topic" in prompt
    assert "Untested Topic" not in prompt


def test_the_volume_block_marks_the_cohorts_that_were_not_measured():
    """Otherwise the model reads an excluded cohort as the end of the series.

    Marked in the prompt's own language: a Ukrainian aside inside an English prompt is
    read as noise rather than as a qualifier.
    """
    for lang, marker in (("uk", "не виміряно"), ("en", "not measured")):
        block = ci._volume_block(FACTS, lang)
        assert "2016: A 80, B 120" in block
        assert marker in block.split("2017")[1]
        assert marker not in block.split("2016")[0]


def test_the_comparison_prompt_carries_the_agreement_result():
    """The null result is the finding; a prompt without it invites the opposite reading."""
    prompt = ci.comparison_prompt(FACTS, "uk", 20)
    assert "r = +0.12" in prompt
    assert "129" not in prompt  # nothing from the real run leaks into a toy
