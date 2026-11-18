"""Tests for the commentary layer's checkable parts.

The generation itself needs a model; what is worth pinning is the guard that makes the
commentary trustworthy — a numeral in the prose that the facts do not contain — and the
sampling that decides which publications the model gets to read.
"""

import pytest

from horizon.interpret import (
    corpus_prompt,
    facts_numbers,
    sample_summaries,
    topic_prompt,
    unsupported_numbers,
)


def _facts(**over):
    base = {
        "records": 268583,
        "years": [2015, 2025],
        "topics_total": 1575,
        "fields": [
            {
                "name": "Medicine",
                "share_first": 0.049,
                "share_last": 0.177,
                "share_delta": 0.128,
            },
            {
                "name": "Physics",
                "share_first": 0.228,
                "share_last": 0.128,
                "share_delta": -0.1,
            },
        ],
        "overall": [
            {"label": "A", "size": 900, "velocity": 1.06, "growth": 12.0},
        ],
        "declining": [
            {"label": "B", "size": 480, "velocity": 0.33, "growth": 7.2},
        ],
        "largest": [
            {"label": "C", "size": 14606, "velocity": 0.99, "growth": 194.7},
        ],
    }
    base.update(over)
    return base


# --- the guard ----------------------------------------------------------------


def test_a_number_absent_from_the_facts_is_reported():
    allowed = facts_numbers(_facts())
    assert unsupported_numbers("the corpus holds 99999 publications", allowed) == [
        "99999"
    ]


def test_numbers_present_in_the_facts_pass():
    allowed = facts_numbers(_facts())
    assert (
        unsupported_numbers("268583 publications, growth 1.06, 900 in topic A", allowed)
        == []
    )


def test_a_rounded_figure_counts_as_supported():
    """Prose that writes 1.1 for a velocity of 1.06 is rounding, not inventing, and a
    check that flagged it would be useless by the first sentence."""
    allowed = facts_numbers({"velocity": 1.06})
    assert unsupported_numbers("velocity of about 1.1", allowed) == []


def test_a_share_written_as_a_percentage_counts_as_supported():
    allowed = facts_numbers({"share": 0.128})
    assert unsupported_numbers("gained 12.8% of the corpus", allowed) == []


def test_each_unsupported_number_is_reported_once_in_order():
    allowed = facts_numbers({"a": 1})
    assert unsupported_numbers("7 then 9 then 7 again", allowed) == ["7", "9"]


# --- what the model gets to read ----------------------------------------------


def _store(years):
    return {
        f"h{i}": {"item_date": f"{y}-01-01", "summary": f"summary {i}"}
        for i, y in enumerate(years)
    }


def test_summaries_are_spread_across_the_topics_lifetime():
    """The question a reading answers is what changed, and that shows at the ends."""
    years = [2015 + (i // 10) for i in range(100)]
    store = _store(years)
    picked = sample_summaries(list(store), store, limit=5)
    assert len(picked) == 5
    assert picked[0][0] == "2015" and picked[-1][0] == "2024"


def test_a_small_topic_gives_every_summary_it_has():
    store = _store([2015, 2016, 2017])
    assert len(sample_summaries(list(store), store, limit=9)) == 3


def test_members_with_no_stored_summary_are_skipped():
    store = _store([2015])
    assert sample_summaries(["h0", "missing"], store) == [("2015", "summary 0")]


def test_a_topic_with_nothing_readable_yields_nothing():
    assert sample_summaries(["h0"], {"h0": {"summary": ""}}) == []


# --- the prompts ---------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_the_topic_prompt_carries_the_figures_and_the_summaries(lang):
    topic = {
        "label": "Public Sector Governance",
        "size": 14606,
        "velocity": 0.99,
        "growth": 194.7,
        "attention_pctile": 0.43,
        "first_seen": "2015-01-01",
        "cohort_counts": {"2015": 280, "2025": 2000},
        "top_keywords": ["ukraine", "higher education"],
    }
    out = topic_prompt(topic, [("2015", "an early paper")], _facts(), lang=lang)
    assert "Public Sector Governance" in out
    assert "14606" in out and "0.99" in out and "+194.7" in out
    assert "an early paper" in out


def test_the_corpus_prompt_reports_the_topics_found_not_the_topics_listed():
    """`overall` is capped at --top-trends, so its length is a reporting decision."""
    out = corpus_prompt(_facts(), lang="en")
    assert "1575" in out
    assert "into 1 topics" not in out


def test_the_corpus_prompt_lists_only_areas_that_actually_lost_share():
    """The tail of the ranking is whatever came last, which on the real corpus included
    areas that gained."""
    out = corpus_prompt(_facts(), lang="en")
    losing = out[out.index("losing the most share") :]
    assert "Physics" in losing.split("Fastest-growing")[0]
    assert "Medicine" not in losing.split("Fastest-growing")[0]


def test_large_topics_are_drawn_from_every_view_not_just_the_score_ranking():
    """Score hides a class: on the real corpus 33 topics of 500+ publications are named
    in no table, sitting slightly below the corpus rate and extreme on no axis."""
    from horizon.interpret import large_topics

    facts = _facts()
    picked = large_topics(facts, min_size=200)
    assert [t["label"] for t in picked] == ["C", "A", "B"]


def test_a_topic_appearing_in_two_views_is_read_once():
    from horizon.interpret import large_topics

    topic = {
        "label": "A",
        "size": 900,
        "velocity": 1.0,
        "growth": 1.0,
        "seed_hash": "h",
    }
    facts = _facts(overall=[topic], largest=[topic], declining=[topic])
    assert len(large_topics(facts, min_size=200)) == 1


def test_topics_below_the_threshold_are_left_out():
    from horizon.interpret import large_topics

    facts = _facts(
        largest=[{"label": "small", "size": 10, "velocity": 1.0, "growth": 0.0}]
    )
    assert [t["label"] for t in large_topics(facts, min_size=200)] == ["A", "B"]


def test_an_enumeration_of_years_is_not_read_as_one_number():
    """ "у 2023, 2024 та 2025 роках" was swallowed whole and reported as 2023.2024 —
    the check firing on prose that had invented nothing."""
    allowed = facts_numbers({"years": [2023, 2024, 2025]})
    assert unsupported_numbers("у 2023, 2024 та 2025 роках", allowed) == []


def test_a_trailing_zero_does_not_make_a_figure_unsupported():
    """0.50 and 0.5 are the same number; treating them as different fired on ten
    readings that had invented nothing."""
    allowed = facts_numbers({"attention_pctile": 0.5})
    assert unsupported_numbers("перцентиль цитувань на рівні 0.50", allowed) == []


def test_a_decimal_comma_still_parses():
    allowed = facts_numbers({"velocity": 1.06})
    assert unsupported_numbers("зростання 1,06", allowed) == []


def test_a_thousands_space_still_parses():
    allowed = facts_numbers({"size": 14606})
    assert unsupported_numbers("14 606 публікацій", allowed) == []


# --- generations that collapsed ------------------------------------------------


def test_a_repetition_loop_is_recognised():
    """12B emits middle dots until the token budget runs out on about 2% of prompts."""
    from horizon.interpret import is_degenerate

    assert is_degenerate("Тема зростає. " + "·" * 2048)
    assert is_degenerate("·" * 20)


def test_ordinary_prose_is_not_degenerate():
    from horizon.interpret import is_degenerate

    assert not is_degenerate("Тема зростає, її частка падає — це не суперечність.")
    assert not is_degenerate("Крапки ... і тире --- трапляються в тексті.")


def test_an_empty_generation_counts_as_degenerate():
    from horizon.interpret import is_degenerate

    assert is_degenerate("   \n  ")


def test_a_reading_that_stops_mid_sentence_is_recognised():
    """At the package default of 512 tokens, 29% of readings ended this way — and a
    truncated one looks like a short one until someone reads to the end."""
    from horizon.interpret import is_truncated

    assert is_truncated("Найбільші теми корпусу ви")
    assert not is_truncated("Найбільші теми корпусу видно в таблиці.")
    assert not is_truncated("Так каже модель «обережно»")
    assert not is_truncated("")


# --- what a document should not carry -----------------------------------------


def test_a_lead_in_line_is_dropped():
    """ "Ось наукометричний аналіз наданого корпусу:" is an assistant answering a
    request, not a sentence of the analysis."""
    from horizon.interpret import clean_reading

    out = clean_reading("Ось наукометричний аналіз корпусу:\n\nКорпус охоплює 268583.")
    assert out == "Корпус охоплює 268583."


def test_a_sentence_ending_in_a_colon_inside_the_text_survives():
    """A real sentence can end in a colon and introduce the paragraph it belongs to."""
    from horizon.interpret import clean_reading

    text = "Тема зростає, і причина одна: корпус росте повільніше."
    assert clean_reading(text) == text


def test_a_lead_in_with_nothing_after_it_is_kept():
    from horizon.interpret import clean_reading

    assert clean_reading("Ось аналіз:") == "Ось аналіз:"


def test_headings_the_model_added_are_dropped():
    from horizon.interpret import clean_reading

    assert clean_reading("## Аналіз\n\nтіло тексту") == "тіло тексту"
