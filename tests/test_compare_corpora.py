"""Tests for the two-corpus comparison.

The things checked here are the ones that would still produce a clean-looking document
if they were wrong: a window that quietly includes a cohort one corpus calls incomplete,
a topic counted once per block it appears in, and a direction read off counts rather than
shares in a corpus that is shrinking under all of them.
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_corpora",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_corpora.py",
)
cmp = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_corpora"] = cmp
_SPEC.loader.exec_module(cmp)


def _facts(years, totals, *, fit=(), fields=(), field_trends=None):
    return {
        "records": sum(totals),
        "unclassified": 0,
        "years": list(years),
        "fit_years": list(fit),
        "year_totals": {str(y): n for y, n in zip(years, totals)},
        "fields": list(fields),
        "pairs": [],
        "field_trends": field_trends or {},
    }


def _side(facts, **kw):
    side = cmp.Side(
        facts=facts,
        name={"en": "A", "uk": "А"},
        unit={
            "en": ("record", "records", "records"),
            "uk": ("запис", "записи", "записів"),
        },
        taxonomy={"en": "fields", "uk": "поля"},
        **kw,
    )
    side.topics = cmp._index_topics(facts)
    return side


def test_the_window_is_intersected_on_presence_and_again_on_completeness():
    """A cohort both corpora hold, but one of them declines to fit, is present and unmeasured.

    This is the whole reason the document has two windows: the overlap of the calendars
    runs further than the overlap of what either corpus is willing to derive from.
    """
    a = _side(_facts(range(2012, 2023), [10] * 11, fit=range(2012, 2021)))
    b = _side(_facts(range(2015, 2026), [10] * 11))
    present, measured = cmp.windows(a, b)
    assert present == list(range(2015, 2023))
    assert measured == list(range(2015, 2021))


def test_an_empty_fit_years_means_every_cohort_was_fitted():
    """The facts omit the key when the derived columns used everything shown."""
    a = _side(_facts([2015, 2016], [10, 10]))
    assert a.fitted == [2015, 2016]


def test_a_topic_is_counted_once_however_many_blocks_list_it():
    """An unblocked clustering lists a topic under every field its members carry.

    Summing per block would multiply such a topic by its own interdisciplinarity, and the
    corpora differ in how multiply classified they are — so the inflation would land on
    one side of the comparison only.
    """
    trend = {
        "label": "Shared Topic",
        "seed_hash": "h1",
        "size": 30,
        "cohort_counts": {"2015": 10, "2016": 20},
        "top_keywords": ["a", "b"],
    }
    facts = _facts(
        [2015, 2016], [100, 100], field_trends={"Field A": [trend], "Field B": [trend]}
    )
    topics = cmp._index_topics(facts)
    assert topics["shared topic"]["counts"] == {2015: 10, 2016: 20}
    assert set(topics["shared topic"]["blocks"]) == {"Field A", "Field B"}


def test_direction_is_read_off_the_share_not_the_count():
    """A topic that falls slower than its corpus has gained ground, and must read as rising.

    The register this was written against halves across the window, which drags every
    count in it downward; a count slope would report every one of its topics as declining
    and the comparison would measure the corpus rather than the topics.
    """
    trend = {
        "label": "Holding On",
        "seed_hash": "h1",
        "size": 15,
        "cohort_counts": {"2015": 10, "2016": 8},
        "top_keywords": [],
    }
    shrinking = _side(
        _facts([2015, 2016], [1000, 400], field_trends={"F": [trend]}),
    )
    steady = _side(_facts([2015, 2016], [1000, 1000], field_trends={"F": [trend]}))
    rows = cmp.matches(shrinking, steady, [2015, 2016])
    assert len(rows) == 1
    assert rows[0]["a_slope"] > 0  # 1.0% -> 2.0% of a corpus that shrank
    assert rows[0]["b_slope"] < 0  # 1.0% -> 0.8% of one that did not


def test_labels_match_across_casing_but_print_as_they_were_written():
    """The analysis layer's casing is not stable; the label a reader sees should be."""

    def trend(label):
        return {
            "label": label,
            "seed_hash": "h1",
            "size": 5,
            "cohort_counts": {"2015": 5},
            "top_keywords": [],
        }

    a = _side(_facts([2015], [100], field_trends={"F": [trend("Gas Separation")]}))
    b = _side(_facts([2015], [100], field_trends={"F": [trend("gas separation")]}))
    rows = cmp.matches(a, b, [2015])
    assert [r["label"] for r in rows] == ["Gas Separation"]


def test_a_ukrainian_numeral_chooses_the_form_of_the_word_it_counts():
    """1 patent, 2 patents, 5 patents and 11 patents are four different words.

    English needs one form and Ukrainian three, so the unit is carried as a form set and
    the renderer picks; a single form would be right for the English document and wrong
    in about half the sentences of the Ukrainian one.
    """
    forms = ("патент", "патенти", "патентів")
    assert [cmp._unit(forms, n) for n in (1, 2, 5, 11, 21, 104, 2460)] == [
        "патент",
        "патенти",
        "патентів",
        "патентів",  # the teens take the last form, not the one their last digit would
        "патент",
        "патенти",
        "патентів",
    ]
    assert cmp._forms("publications") == ("publications",) * 3


def test_both_languages_define_the_same_labels():
    """The bilingual rule, enforced here rather than by memory — as in ``report.LABELS``."""
    for name, mapping in (
        ("LABELS", cmp.LABELS),
        ("FIGURE_TEXT", cmp.FIGURE_TEXT),
        ("HEAD", cmp.HEAD),
    ):
        assert set(mapping["en"]) == set(mapping["uk"]), name


def test_a_coverage_column_appears_only_for_the_corpus_that_has_one():
    """An unattached percentage column in a two-corpus table reads as belonging to either."""
    fields = [
        {
            "name": "F",
            "total": 20,
            "slope": 0.0,
            "share_first": 0.1,
            "share_last": 0.1,
            "counts": {"2015": 10, "2016": 10},
        }
    ]
    a = _side(
        _facts([2015, 2016], [100, 100], fields=fields),
        coverage={2015: 1.0, 2016: 0.9},
    )
    b = _side(_facts([2015, 2016], [100, 100], fields=fields))
    present = measured = [2015, 2016]
    data = {
        "volume_a": cmp.volume(a, present, measured),
        "volume_b": cmp.volume(b, present, measured),
        "fields_a": cmp.fields(a, measured),
        "fields_b": cmp.fields(b, measured),
        "pairs_a": [],
        "pairs_b": [],
        "matches": [],
        "match_a_n": 0,
        "match_b_n": 0,
        "agreement": cmp.agreement([], cmp.MIN_MATCH_SIZE),
        "crosswalk": [],
    }
    data["headline"] = cmp.headline(data["volume_a"], data["volume_b"], measured)
    data["shape_a"] = cmp.shape(a, data["fields_a"], measured)
    data["shape_b"] = cmp.shape(b, data["fields_b"], measured)
    data["limits"] = cmp.limits(a, b, present, measured)
    md = cmp.render(a, b, present, measured, data, "en", "figures/x", [], top=5)
    header = next(line for line in md.splitlines() if line.startswith("| year |"))
    assert header.count("est. coverage") == 1
    assert "so the coverage column belongs to it" in md


def _rendered(readings, lang="uk"):
    """The report and the annotated report over a two-year toy comparison."""
    fields = [
        {
            "name": "F",
            "total": 20,
            "slope": 0.0,
            "share_first": 0.1,
            "share_last": 0.1,
            "counts": {"2015": 10, "2016": 10},
        },
        {
            "name": "Спадна галузь",
            "total": 30,
            "slope": 0.0,
            "share_first": 0.2,
            "share_last": 0.1,
            "counts": {"2015": 20, "2016": 10},
        },
    ]
    a = _side(_facts([2015, 2016], [100, 100], fields=fields))
    b = _side(_facts([2015, 2016], [100, 100], fields=fields))
    a.taxonomy = {"en": "Fields of A", "uk": "Поля A"}
    b.taxonomy = {"en": "Fields of B", "uk": "Поля B"}
    present = measured = [2015, 2016]
    data = {
        "volume_a": cmp.volume(a, present, measured),
        "volume_b": cmp.volume(b, present, measured),
        "fields_a": cmp.fields(a, measured),
        "fields_b": cmp.fields(b, measured),
        "pairs_a": [],
        "pairs_b": [],
        "matches": [],
        "match_a_n": 0,
        "match_b_n": 0,
        "agreement": cmp.agreement([], cmp.MIN_MATCH_SIZE),
        "crosswalk": [],
    }
    data["headline"] = cmp.headline(data["volume_a"], data["volume_b"], measured)
    data["shape_a"] = cmp.shape(a, data["fields_a"], measured)
    data["shape_b"] = cmp.shape(b, data["fields_b"], measured)
    data["limits"] = cmp.limits(a, b, present, measured)
    annotated = cmp.render(
        a, b, present, measured, data, lang, "figures/x", [], top=5, readings=readings
    )
    commentary = cmp.render_commentary(a, b, present, measured, data, readings, lang)
    return annotated, commentary


def _reading(model, **buckets):
    return {"model_id": model, "label": model, "lang": "uk", **buckets}


def test_one_reading_is_printed_unattributed_and_two_are_named():
    """A model's name is there to tell several apart, not to record provenance.

    Which model wrote a document belongs in the technical report describing the run, so a
    single reading carries no name at all — the rule ``build_interpretation._block``
    follows, applied to the same documents built from a comparison.
    """
    one = [_reading("model-a", comparison={"text": "Один.", "unsupported": []})]
    two = one + [_reading("model-b", comparison={"text": "Два.", "unsupported": []})]
    solo, _ = _rendered(one)
    pair, _ = _rendered(two)
    assert "Один." in solo and "model-a" not in solo
    assert "model-a" in pair and "model-b" in pair


def test_a_truncated_reading_says_so_where_it_is_printed():
    """A reading cut off mid-sentence is the failure nobody notices in a long document."""
    readings = [
        _reading(
            "m",
            comparison={"text": "Почав і не", "unsupported": [], "truncated": True},
        )
    ]
    annotated, commentary = _rendered(readings)
    for doc in (annotated, commentary):
        assert "Обірвано на середині речення" in doc


def test_a_figure_the_facts_do_not_contain_is_named_under_the_paragraph():
    """Re-checked at merge time, so a later fix to the check reaches old readings."""
    readings = [
        _reading("m", comparison={"text": "Зросло на 4242%.", "unsupported": []})
    ]
    annotated, _ = _rendered(readings)
    assert "4242" in annotated.split("Зросло")[1].split("##")[0]
    assert "Числа, яких немає у фактах" in annotated


def test_each_taxonomy_reading_lands_under_its_own_taxonomy():
    """The two sides' readings are interchangeable-looking prose; the placement is not."""
    readings = [
        _reading(
            "m",
            taxonomies={
                "a": {"text": "Про поля A.", "unsupported": []},
                "b": {"text": "Про поля B.", "unsupported": []},
            },
        )
    ]
    annotated, _ = _rendered(readings)
    section = annotated.split("## 3.")[1]
    first, second = section.split("### Поля B")
    assert "Про поля A." in first and "Про поля A." not in second
    assert "Про поля B." in second


def test_the_annotated_document_is_the_report_plus_readings():
    """Rendered by the same function, so it cannot drift from what it annotates.

    The single-corpus merge re-parses the rendered markdown to fold readings in; this one
    does not, and the guard is that removing the readings gives the report back.
    """
    readings = [_reading("m", comparison={"text": "Прочитання.", "unsupported": []})]
    annotated, _ = _rendered(readings)
    plain, _ = _rendered([])
    assert "Прочитання." in annotated and "Прочитання." not in plain
    for heading in ("## 1.", "## 4.", "## 6."):
        assert heading in annotated and heading in plain


def test_the_companions_say_what_the_checks_cannot_catch():
    """A direction claim contains no wrong number, so nothing in the pipeline sees it.

    Observed in the run these documents were built from: a reading said a field rose
    while it fell, and passed every mechanical check. A reader who is told the numerals
    were verified will assume the sentences were too.
    """
    readings = [_reading("m", comparison={"text": "Прочитання.", "unsupported": []})]
    annotated, commentary = _rendered(readings)
    for doc in (annotated, commentary):
        assert "Напрямок не перевіряється" in doc


def test_the_facts_contain_the_window_totals_the_documents_print():
    """The commentary check reads this file; a printed figure missing from it is flagged.

    Caught by the check itself during the run these documents were built from: the
    blockquote and the reading prompt both state the window total, and the facts held
    only the per-year counts it is summed from.
    """
    a = _side(_facts([2015, 2016], [100, 80]))
    b = _side(_facts([2015, 2016], [1000, 1200]))
    present = measured = [2015, 2016]
    data = {
        "volume_a": cmp.volume(a, present, measured),
        "volume_b": cmp.volume(b, present, measured),
        "fields_a": [],
        "fields_b": [],
        "pairs_a": [],
        "pairs_b": [],
        "matches": [],
        "match_a_n": 0,
        "match_b_n": 0,
        "agreement": cmp.agreement([], cmp.MIN_MATCH_SIZE),
        "crosswalk": [],
    }
    data["shape_a"] = cmp.shape(a, [], measured)
    data["shape_b"] = cmp.shape(b, [], measured)
    data["headline"] = cmp.headline(data["volume_a"], data["volume_b"], measured)
    facts = cmp.facts_dict(a, b, present, measured, data)
    assert facts["a"]["records_in_window"] == 180
    assert facts["b"]["records_in_window"] == 2200

    from horizon.interpret import facts_numbers, unsupported_numbers

    assert unsupported_numbers("180 and 2200", facts_numbers(facts)) == []


def test_a_reading_that_moves_a_field_the_wrong_way_is_flagged():
    """The failure the numeral check cannot see, and the one both models made.

    gemma-3-4b wrote «значне зростання частки полів "Фармацевтика" та "Біотехнології"»
    over a table in which those two fell hardest, −5.4 and −6.9 points. No numeral in the
    sentence is wrong, so every other check passes it.
    """
    fields = [
        {"name": "Фармацевтика", "delta": -0.054},
        {"name": "Біотехнології", "delta": -0.069},
        {"name": "Матеріали та металургія", "delta": +0.048},
    ]
    text = (
        'Найбільш помітним є значне зростання частки полів "Фармацевтика" та '
        '"Біотехнології", а також зростання частки "Матеріали та металургія".'
    )
    assert cmp.diverging_directions(text, fields, "uk") == [
        "Фармацевтика",
        "Біотехнології",
    ]


def test_the_check_reads_through_ukrainian_inflection():
    """The table says «Медичні технології»; generated prose says «Медичних технологій»."""
    fields = [{"name": "Медичні технології", "delta": -0.02}]
    assert cmp.diverging_directions(
        "Спостерігається зростання частки медичних технологій.", fields, "uk"
    ) == ["Медичні технології"]


def test_a_field_that_did_not_move_is_not_second_guessed():
    """A sector named in passing inside a sentence about growth is not a claim.

    Без цього the check fires on «промисловість, будівництво та сільське господарство»
    because Будівництво happens to carry a share delta of about zero.
    """
    fields = [{"name": "Будівництво", "delta": -0.0001}]
    text = "Це зростання пов'язано з зростанням впливу таких галузей, як будівництво."
    assert cmp.diverging_directions(text, fields, "uk") == []


def test_a_negated_direction_is_left_alone():
    """ "Did not grow" is a claim this check cannot read, so it declines to read it."""
    fields = [{"name": "Фармацевтика", "delta": -0.054}]
    assert (
        cmp.diverging_directions(
            "Частка не зростала для полів Фармацевтика.", fields, "uk"
        )
        == []
    )


def test_a_correct_direction_passes():
    fields = [{"name": "Фармацевтика", "delta": -0.054}]
    assert (
        cmp.diverging_directions(
            "Помітне зниження частки полів Фармацевтика.", fields, "uk"
        )
        == []
    )


def test_the_flag_is_printed_under_the_paragraph_it_belongs_to():
    """And only there: a reading that reads the table right carries no flag."""
    wrong = [
        _reading(
            "m",
            taxonomies={
                "a": {
                    "text": "Помітне зростання частки Спадної галузі.",
                    "unsupported": [],
                }
            },
        )
    ]
    right = [
        _reading(
            "m",
            taxonomies={
                "a": {
                    "text": "Помітне зниження частки Спадної галузі.",
                    "unsupported": [],
                }
            },
        )
    ]
    for doc in _rendered(wrong):
        assert "Напрямок розходиться з таблицею: Спадна галузь" in doc
    for doc in _rendered(right):
        assert "Напрямок розходиться з таблицею" not in doc
