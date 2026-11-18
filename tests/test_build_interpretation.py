"""Tests for the merge that produces the two commentary documents."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_interpretation",
    Path(__file__).resolve().parents[1] / "scripts" / "build_interpretation.py",
)
merge = importlib.util.module_from_spec(_SPEC)
sys.modules["build_interpretation"] = merge
_SPEC.loader.exec_module(merge)

FACTS = {
    "records": 100,
    "years": [2015, 2025],
    "topics_total": 3,
    "fields": [],
    "overall": [],
    "declining": [],
    "largest": [
        {
            "label": "Big Topic",
            "size": 900,
            "velocity": 0.99,
            "growth": 12.0,
            "seed_hash": "h1",
        }
    ],
}


def _reading(model, text, unsupported=()):
    return {
        "model_id": model,
        "lang": "uk",
        "corpus": {"text": f"{text} corpus", "unsupported": []},
        "topics": {"h1": {"text": text, "unsupported": list(unsupported)}},
    }


def test_both_models_appear_under_the_same_topic():
    """Averaging two readings into one voice hides what a reader needs to discount."""
    out = merge.build_interpretation(
        FACTS, [_reading("e4b", "first"), _reading("12b", "second")], "uk", 200
    )
    section = out[out.index("### Big Topic") :]
    assert "e4b" in section and "12b" in section
    assert section.index("first") < section.index("second")


def test_the_measured_figures_sit_above_the_reading():
    """A judgement read without the numbers it is about is not checkable."""
    out = merge.build_interpretation(FACTS, [_reading("e4b", "text")], "uk", 200)
    section = out[out.index("### Big Topic") :]
    assert section.index("900 · 0.99 · +12.0") < section.index("text")


def test_a_figure_the_facts_do_not_contain_is_named_under_the_paragraph():
    """It is the one thing about a generated reading that reading it cannot check."""
    out = merge.build_interpretation(
        FACTS, [_reading("e4b", "the topic holds 99999 publications")], "uk", 200
    )
    assert "Числа, яких немає у фактах: 99999" in out


def test_the_flagged_figures_are_recomputed_rather_than_trusted():
    """The check improves — its first version reported an enumeration of years as one
    invented number — and recomputing applies the fix to readings already generated."""
    out = merge.build_interpretation(
        FACTS, [_reading("e4b", "grew steadily", ["2023.2024"])], "uk", 200
    )
    assert "2023.2024" not in out


def test_a_heading_the_model_added_is_stripped():
    """The model adds its own despite being asked not to, and a stray ## inside a
    section reorders the document's outline."""
    out = merge.build_interpretation(
        FACTS, [_reading("e4b", "## Аналіз\n\nthe body")], "uk", 200
    )
    assert "## Аналіз" not in out and "the body" in out


def test_a_topic_no_model_read_is_skipped_rather_than_left_empty():
    facts = dict(
        FACTS,
        largest=FACTS["largest"]
        + [
            {
                "label": "Unread",
                "size": 800,
                "velocity": 1.0,
                "growth": 1.0,
                "seed_hash": "h2",
            }
        ],
    )
    out = merge.build_interpretation(facts, [_reading("e4b", "text")], "uk", 200)
    assert "### Unread" not in out


def test_a_single_reading_names_no_model_anywhere():
    """Which model wrote a document belongs in the technical report that describes the
    run, not in the middle of the analysis."""
    report = "# Report\n\nintro\n\n## 1. Fields\n\n| a | b |\n"
    out = merge.build_annotated(report, FACTS, [_reading("e4b", "text")], "uk", 200)
    assert "## 1. Fields" in out and "| a | b |" in out
    assert out.index("Корпус у цілому") < out.index("## 1. Fields")
    assert "e4b" not in out


def test_several_models_are_named_on_every_block():
    """Without the per-block name there is no telling which model said what."""
    out = merge.build_interpretation(
        FACTS, [_reading("e4b", "first"), _reading("12b", "second")], "uk", 200
    )
    assert out.count("Прочитання — e4b") >= 1
    assert out.count("Прочитання — 12b") >= 1


def test_the_corpus_reading_goes_before_the_first_section_not_after_the_last():
    report = "# Report\n\n## 1. Fields\n\nx\n\n## 2. Pairs\n\ny\n"
    out = merge.build_annotated(report, FACTS, [_reading("e4b", "text")], "uk", 200)
    assert out.index("Корпус у цілому") < out.index("## 1. Fields")


def _with_areas(model, text):
    r = _reading(model, text)
    r["areas"] = {"Physics": f"{text} physics", "Chemistry": f"{text} chemistry"}
    r["areas"] = {k: {"text": v, "unsupported": []} for k, v in r["areas"].items()}
    return r


REPORT = (
    "# Report\n\nintro\n\n"
    "## 1. Fields\n\n| a | b |\n\n"
    "## 3. Topics\n\n"
    "### Physics\n\n| p | q |\n\n"
    "### Chemistry\n\n| c | d |\n\n"
    "## 4. Corpus-wide\n\n"
    "### 4.1. By score\n\n| x | y |\n"
)


def test_each_area_reading_lands_under_its_own_heading():
    out = merge.build_annotated(REPORT, FACTS, [_with_areas("e4b", "R")], "uk", 200)
    physics = out.index("### Physics")
    chemistry = out.index("### Chemistry")
    assert physics < out.index("R physics") < chemistry
    assert chemistry < out.index("R chemistry")


def test_an_area_reading_stays_on_its_own_page():
    """A reading is buffered until the next heading, and a page break comes before it.

    Flushed at the break rather than at the heading, or the Physics commentary is torn
    off the Physics table and printed at the top of the Chemistry page.
    """
    from horizon.report import PAGE_BREAK

    brk = "\n".join(PAGE_BREAK)
    report = REPORT.replace("### Physics", brk + "### Physics").replace(
        "### Chemistry", brk + "### Chemistry"
    )
    out = merge.build_annotated(report, FACTS, [_with_areas("e4b", "R")], "uk", 200)
    assert out.index("R physics") < out.index("### Chemistry")
    assert out.index("R physics") < out.rindex(PAGE_BREAK[1])


def test_a_subheading_outside_section_three_gets_no_reading():
    """Sections 4 and 5 use ### for their own views and indicators."""
    reading = _with_areas("e4b", "R")
    reading["areas"]["4.1. By score"] = {"text": "wrong place", "unsupported": []}
    out = merge.build_annotated(REPORT, FACTS, [reading], "uk", 200)
    assert "wrong place" not in out


def test_the_annotated_document_carries_no_per_topic_appendix():
    """Two hundred topic readings appended would be read by nobody; they stay in the
    commentary document where they can be looked up."""
    out = merge.build_annotated(REPORT, FACTS, [_with_areas("e4b", "R")], "uk", 200)
    assert "### Big Topic" not in out
