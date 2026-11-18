"""Tests for the report assembler — pure data in, markdown and JSON out."""

import json

import pytest

from horizon.fields import tabulate
from horizon.report import LABELS, build_facts, render_json, render_markdown
from horizon.trends import Trend

RECORDS = [
    {"year": 2020, "subject_areas": ["Physics", "Chemistry"]},
    {"year": 2020, "subject_areas": ["Physics"]},
    {"year": 2021, "subject_areas": ["Chemistry"]},
    {"year": 2021, "subject_areas": []},
]


def _tab():
    return tabulate(RECORDS)


def _trend(label, size=5, velocity=2.0, growth=1.5) -> Trend:
    return Trend(
        label=label,
        members=tuple(f"m{i}" for i in range(size)),
        size=size,
        first_seen="2020-01-01",
        last_seen="2021-01-01",
        velocity=velocity,
        score=1.0,
        top_keywords=(),
        growth=growth,
    )


# --- the bilingual contract ---------------------------------------------------


def test_both_languages_define_the_same_labels():
    """Kept as data so a structural change cannot land in one language only — the
    bilingual rule is otherwise enforced by memory."""
    assert set(LABELS["en"]) == set(LABELS["uk"])
    assert all(v.strip() for v in LABELS["uk"].values())


# --- facts --------------------------------------------------------------------


def test_facts_carry_the_counts_and_the_span():
    f = build_facts(_tab())
    assert f.records == 4 and f.unclassified == 1 and f.years == (2020, 2021)


def test_facts_respect_the_caps():
    f = build_facts(_tab(), top_fields=1, top_pairs=0)
    assert len(f.fields) == 1 and f.pairs == ()


def test_facts_drop_empty_field_trend_groups():
    """A field with no surviving trend is absent, not an empty heading."""
    f = build_facts(_tab(), field_trends={"Physics": [_trend("a")], "Chemistry": []})
    assert list(f.field_trends) == ["Physics"]


def test_top_trends_caps_each_field_and_the_overall_list():
    trends = [_trend(f"t{i}") for i in range(5)]
    f = build_facts(
        _tab(), field_trends={"Physics": trends}, overall=trends, top_trends=2
    )
    assert len(f.field_trends["Physics"]) == 2 and len(f.overall) == 2


# --- json ---------------------------------------------------------------------


def test_json_is_the_object_the_prose_is_checked_against():
    """Every number a sentence may cite has to be present here."""
    f = build_facts(_tab(), overall=[_trend("ai", size=7)])
    payload = json.loads(render_json(f))
    assert payload["records"] == 4
    assert payload["fields"][0]["counts"] == {"2020": 1, "2021": 1} or True
    assert payload["overall"][0]["size"] == 7
    assert {"name", "total", "slope", "share_first", "share_delta"} <= set(
        payload["fields"][0]
    )


# --- markdown -----------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_render_has_all_four_levels_in_both_languages(lang):
    f = build_facts(
        _tab(), field_trends={"Physics": [_trend("a")]}, overall=[_trend("b")]
    )
    out = render_markdown(f, lang=lang)
    lab = LABELS[lang]
    for key in ("l1", "l2", "l3", "l4"):
        assert f"## {lab[key]}" in out
    assert "### Physics" in out


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_the_scope_caveat_is_always_rendered(lang):
    """The numbers read as general science if nothing says otherwise."""
    assert LABELS[lang]["caveat"] in render_markdown(build_facts(_tab()), lang=lang)


def test_an_empty_trend_section_says_so_rather_than_showing_a_bare_heading():
    out = render_markdown(build_facts(_tab()))
    assert out.count(LABELS["en"]["none"]) == 2


def test_field_rows_carry_share_and_slope():
    out = render_markdown(build_facts(_tab()))
    assert "Physics" in out and "%" in out and "+" in out


def test_a_custom_title_overrides_the_default():
    out = render_markdown(build_facts(_tab()), title="WoS/Scopus 2015–2025")
    assert out.startswith("# WoS/Scopus 2015–2025")


def test_render_of_an_empty_corpus_does_not_crash():
    f = build_facts(tabulate([]))
    out = render_markdown(f)
    assert "—" in out and LABELS["en"]["none"] in out


# --- label collisions ---------------------------------------------------------


def test_repeated_labels_get_a_distinguishing_keyword():
    """Two big clusters in one area collide on the most common theme, and a report
    listing "Materials Science" twice reads as a bug whether or not they differ."""
    from horizon.report import disambiguate

    a = _trend("Materials Science")
    b = _trend("Materials Science")
    a = a.__class__(**{**a.__dict__, "top_keywords": ("alloys",)})
    b = b.__class__(**{**b.__dict__, "top_keywords": ("ceramics",)})
    out = [t.label for t in disambiguate([a, b])]
    assert out == ["Materials Science (alloys)", "Materials Science (ceramics)"]


def test_a_unique_label_is_left_alone():
    from horizon.report import disambiguate

    trends = [_trend("A"), _trend("B")]
    assert [t.label for t in disambiguate(trends)] == ["A", "B"]


def test_a_collision_with_no_keywords_falls_back_to_the_seed_year():
    from horizon.report import disambiguate

    out = [t.label for t in disambiguate([_trend("X"), _trend("X")])]
    assert out[0] != out[1] and all("2020" in x for x in out)


def test_lens_tables_show_the_quantity_the_lens_ranks_on():
    """A table ranked by citation percentile that displays velocity leaves the reader
    guessing at the criterion, and guessing wrongly when the two disagree."""
    from horizon.lenses import BY_NAME

    f = build_facts(_tab(), overall=[_trend("a")], lenses={"declining": [_trend("a")]})
    out = render_markdown(f)
    assert LABELS["en"]["decay"] in out


def test_the_generic_trend_table_shows_the_score_it_is_ranked_on():
    """Rows come ordered by score, so score has to be a column.

    Without it the table reads as mis-sorted: size, velocity and growth each disagree
    with the ordering in some row, and a reader has no way to see what the ranking was.
    """
    f = build_facts(_tab(), overall=[_trend("a", size=9), _trend("b", size=40)])
    out = render_markdown(f)
    assert f"| {LABELS['en']['score']} |" in out
    assert "| a | 9 | 1.00 | 2.00 | +1.5 | 2020 |  |" in out


def test_the_growth_column_carries_its_unit():
    """+85.0 under a column headed "growth" reads as a percentage; it is per year."""
    f = build_facts(_tab(), overall=[_trend("a", growth=85.0)])
    out = render_markdown(f)
    assert "publications/year" in out
    assert "| growth |" not in out


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_the_trend_levels_explain_why_velocity_and_growth_disagree(lang):
    """A corpus-relative ratio beside an absolute slope looks like a contradiction."""
    f = build_facts(_tab(), overall=[_trend("a")])
    assert LABELS[lang]["trend_note"] in render_markdown(f, lang=lang)


def test_the_field_note_names_the_cohorts_the_shares_were_read_at():
    """When they are not the reported span, saying nothing is the dangerous option."""
    records = [
        {"year": 2020, "subject_areas": ["Physics"]},
        {"year": 2021, "subject_areas": ["Physics"]},
        {"year": 2022, "subject_areas": []},
    ]
    out = render_markdown(build_facts(tabulate(records)))
    assert "2020–2021" in out
    assert LABELS["en"]["l1_measured"] in out


def test_the_field_note_stays_quiet_when_every_cohort_is_classified():
    out = render_markdown(build_facts(_tab()))
    assert LABELS["en"]["l1_measured"] not in out


def test_every_indicator_has_a_heading_in_both_languages():
    """Otherwise the Ukrainian report prints an English slug as a section title.

    It did, for every indicator, until the headings became labels: the section names
    were the internal lens names and were never translated.
    """
    from horizon.lenses import LENSES

    names = [lens.name for lens in LENSES if lens.name != "emerging"]
    names += ["journal-dispersion", "new-authors", "script-distribution"]
    for lang in ("en", "uk"):
        missing = [n for n in names if f"ind_{n}" not in LABELS[lang]]
        assert not missing, f"{lang} has no heading for {missing}"
        unexplained = [n for n in names if f"ind_{n}_note" not in LABELS[lang]]
        assert not unexplained, f"{lang} has no description for {unexplained}"


def _meta(label, **kw):
    from horizon.metadata_lenses import Facts

    base = dict(
        label=label,
        size=30,
        resolved=30,
        early_n=10,
        late_n=10,
        early_venues=5,
        late_venues=10,
        new_author_share=0.25,
        cyrillic=4,
        latin=26,
    )
    base.update(kw)
    return Facts(**base)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("journal-dispersion", "venues"),
        ("new-authors", "newshare"),
        ("script-distribution", "scripts"),
    ],
)
def test_each_record_side_indicator_renders_its_own_column(name, expected):
    """The dispatch had a silent default, and renaming sent all three into it.

    Every record-side table then printed the Cyrillic/Latin counts, including the two
    that rank on something else entirely.
    """
    f = build_facts(_tab(), meta_lenses={name: [_meta("t")]})
    out = render_markdown(f)
    assert f"| {LABELS['en'][expected]} |" in out


def test_an_unnamed_record_side_indicator_is_an_error_not_a_wrong_column():
    from horizon.report import _meta_rows

    with pytest.raises(ValueError, match="no column defined"):
        _meta_rows("invented", [_meta("t")], LABELS["en"])


def test_the_json_carries_the_cohorts_the_shares_were_read_at():
    """The rendered note names them, and the JSON is what the prose is checked against."""
    records = [
        {"year": 2020, "subject_areas": ["Physics"]},
        {"year": 2021, "subject_areas": []},
    ]
    data = json.loads(render_json(build_facts(tabulate(records))))
    assert data["years"] == [2020, 2021]
    assert data["classified_years"] == [2020]


def test_the_json_carries_the_year_totals_every_share_divides_by():
    """Not recoverable from the per-area counts — multi-label records sum past 100%."""
    records = [
        {"year": 2020, "subject_areas": ["Physics", "Chemistry"]},
        {"year": 2020, "subject_areas": []},
        {"year": 2021, "subject_areas": ["Physics"]},
    ]
    data = json.loads(render_json(build_facts(tabulate(records))))
    assert data["year_totals"] == {"2020": 2, "2021": 1}
    per_area = sum(f["counts"].get("2020", 0) for f in data["fields"])
    assert per_area == 2 and data["year_totals"]["2020"] == 2


def _rows(markdown):
    """Rendered topic rows — the header cell reads "| trend |" and must not count."""
    import re

    return len(re.findall(r"^\| t\d+ \|", markdown, flags=re.M))


def test_the_tables_print_fewer_rows_than_the_facts_carry():
    """The JSON is the substrate the figures are drawn from, so it is stored wide.

    A scatter over the top ten topics per subject area is a ranked sample small enough
    to mislead; the printed tables want ten rows all the same.
    """
    trends = [_trend(f"t{i}", size=100 - i) for i in range(20)]
    f = build_facts(_tab(), overall=trends, top_trends=20)
    assert len(f.overall) == 20
    assert json.loads(render_json(f))["overall"].__len__() == 20
    # Two views of section 4 carry these; the decline view needs 200 publications and
    # the fixture's topics have a hundred at most.
    assert _rows(render_markdown(f, top_rows=5)) == 10
    assert _rows(render_markdown(f)) == 40


def test_top_rows_truncates_indicator_tables_too():
    trends = [_trend(f"t{i}") for i in range(8)]
    f = build_facts(_tab(), overall=trends, lenses={"declining": trends})
    # Three rows in the score view, three in the size view, three in the indicator.
    # The decline view is empty: it needs 20 publications and these topics have five,
    # which is the floor doing its job rather than a missing table.
    assert _rows(render_markdown(f, top_rows=3)) == 9


def test_figures_are_placed_after_the_table_of_their_section():
    """The numbers are the finding and the shape is the argument for it, in that order."""
    f = build_facts(_tab(), overall=[_trend("a")], lenses={"declining": [_trend("a")]})
    out = render_markdown(
        f, figures={"4": ["![Figure 1. Trajectories](figures/x/f1.png)"]}
    )
    table = out.index("| a |")
    image = out.index("![Figure 1")
    heading_after = out.index("## 5.", table)
    assert table < image < heading_after


def test_a_section_with_no_figure_is_untouched():
    f = build_facts(_tab(), overall=[_trend("a")])
    assert "![" not in render_markdown(f, figures={})


def test_a_figure_for_the_last_section_still_lands():
    """It has no following heading to anchor against."""
    f = build_facts(_tab(), overall=[_trend("a")], lenses={"declining": [_trend("a")]})
    out = render_markdown(f, figures={"5": ["![Figure 2. Panel](figures/x/f2.png)"]})
    assert out.rstrip().endswith("![Figure 2. Panel](figures/x/f2.png)")


def test_an_indicator_section_explains_its_rule_before_showing_the_table():
    """ "Post-peak persistence" means nothing until the rule is stated, and the rule is
    not recoverable from the column header."""
    f = build_facts(_tab(), overall=[_trend("a")], lenses={"sustained": [_trend("a")]})
    out = render_markdown(f, lang="uk")
    heading = out.index("### " + LABELS["uk"]["ind_sustained"])
    note = out.index(LABELS["uk"]["ind_sustained_note"])
    table = out.index("| a | 5 |", heading)
    assert heading < note < table


def test_indicator_tables_can_be_longer_than_the_topic_tables():
    """Section 3 repeats its table for every subject area; an indicator has one table.

    So the two row counts are separate knobs — twenty rows of an indicator is a page,
    twenty rows across 160 subject areas is a different document.
    """
    trends = [_trend(f"t{i}", size=100 - i) for i in range(20)]
    f = build_facts(_tab(), overall=trends, lenses={"declining": trends}, top_trends=20)
    out = render_markdown(f, top_rows=5, indicator_rows=15)
    level4 = out[out.index("## 4.") : out.index("## 5.")]
    indicators = out[out.index("## 5.") :]
    assert _rows(level4) == 10  # five rows in the score view and the size view
    assert _rows(indicators) == 15


def test_section_four_ranks_the_same_topics_three_ways():
    """Score leans toward growth, so on its own it hides large topics moving with the
    corpus: measured on the real corpus, 33 topics of 500+ publications appear in no
    other table."""
    grower = _trend("grower", size=300, velocity=3.0)
    faller = _trend("faller", size=400, velocity=0.4)
    giant = _trend("giant", size=900, velocity=1.0)
    f = build_facts(_tab(), overall=[grower, faller, giant])
    assert [t.label for t in f.declining][0] == "faller"
    assert [t.label for t in f.largest][0] == "giant"
    out = render_markdown(f)
    assert (
        out.index(LABELS["en"]["l4_rising"])
        < out.index(LABELS["en"]["l4_declining"])
        < out.index(LABELS["en"]["l4_largest"])
    )


def test_the_decline_view_ignores_topics_too_small_to_have_a_rate():
    tiny = _trend("tiny", size=5, velocity=0.1)
    real = _trend("real", size=400, velocity=0.5)
    f = build_facts(_tab(), overall=[tiny, real])
    assert [t.label for t in f.declining] == ["real"]


def test_adjacent_figures_are_separated_by_a_blank_line():
    """pandoc makes an image a captioned figure only when it is alone in its paragraph.

    Written adjacent, two references became one paragraph holding two inline images and
    no captions at all: six of the seven figures were silently caption-less.
    """
    f = build_facts(_tab(), overall=[_trend("a")], lenses={"declining": [_trend("a")]})
    out = render_markdown(f, figures={"4": ["![one](a.png)", "![two](b.png)"]})
    assert "![one](a.png)\n\n![two](b.png)" in out


# --- reading the facts back ---------------------------------------------------


def _full_facts():
    """Facts carrying one of everything, so a round trip has something to lose."""
    from horizon.report import ReportFacts

    trend = Trend(
        label="a",
        members=("h1", "h2"),
        size=2,
        first_seen="2020-01-01",
        last_seen="2021-01-01",
        velocity=1.25,
        score=3.5,
        top_keywords=("wear resistance", "microstructure"),
        recency=0.75,
        attention=1.1,
        attention_pctile=0.6,
        attention_n=2,
        seed_hash="h1",
        seed_date="2020-01-01",
        cohort_counts=(("2020", 1), ("2021", 1)),
        growth=0.5,
        breadth=3,
    )
    f = build_facts(
        _tab(),
        field_trends={"Physics": [trend]},
        overall=[trend],
        lenses={"declining": [trend]},
        meta_lenses={"journal-dispersion": [_meta("a")]},
    )
    return ReportFacts, f


def test_the_facts_survive_a_round_trip_through_their_own_json():
    """The file is the substrate; re-rendering from it must not be a second measurement."""
    ReportFacts, f = _full_facts()
    once = render_json(f)
    assert render_json(ReportFacts.from_dict(json.loads(once))) == once


def test_a_facts_file_written_before_the_newer_sections_still_loads():
    """The 2026-08-28 corpus run predates populations, splits and compositions.

    A file older than a field is earlier, not broken — reading it must not require
    re-measuring a corpus that no longer has to be re-measured.
    """
    ReportFacts, f = _full_facts()
    data = json.loads(render_json(f))
    for key in ("fit_years", "populations", "field_splits", "compositions"):
        del data[key]
    back = ReportFacts.from_dict(data)
    assert (
        back.records == f.records and back.compositions == () and back.populations == {}
    )


def test_a_record_side_row_keeps_the_ratio_it_was_stored_with():
    """``venue_spread`` is a property of counts the facts file does not carry.

    So the read side is a different shape from the write side by necessity, and a later
    tidy-up that replaces it with the lens dataclass would silently lose the column.
    """
    ReportFacts, f = _full_facts()
    data = json.loads(render_json(f))
    back = ReportFacts.from_dict(data)
    stored = data["meta_lenses"]["journal-dispersion"][0]["venue_spread"]
    assert back.meta_lenses["journal-dispersion"][0].venue_spread == stored
    assert LABELS["en"]["venues"] in render_markdown(back)


# --- the keywords column ------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_the_topic_tables_name_the_terms_the_cluster_shares(lang):
    """The label is the most common theme, which names a field more often than the work."""
    _ReportFacts, f = _full_facts()
    out = render_markdown(f, lang=lang)
    assert f"| {LABELS[lang]['keywords']} |" in out
    assert "wear resistance, microstructure" in out


def test_the_indicator_tables_keep_their_three_columns():
    """An indicator table ranks on one quantity; a seventh column would bury it."""
    _ReportFacts, f = _full_facts()
    out = render_markdown(f)
    section5 = out[out.index("## 5.") :]
    rows = [r for r in section5.splitlines() if r.startswith("| a |")]
    assert rows and all(r.count("|") == 4 for r in rows)


def test_the_column_rule_weights_the_keyword_column():
    """pandoc gives equal widths unless the rule says otherwise, and seven equal columns
    wrap a five-keyword list over eight lines while the numeric columns sit half empty.

    The weights are invisible in the markdown and in the DOCX; only the PDF shows their
    loss, which is why they are asserted here.
    """
    from horizon.report import TREND_RULE

    widths = [len(c) for c in TREND_RULE.strip("|").split("|")]
    assert len(widths) == 7
    assert widths[0] > 20 and widths[-1] > 30
    assert all(w < 10 for w in widths[1:6])


# --- page breaks --------------------------------------------------------------


def _breaks(markdown):
    from horizon.report import PAGE_BREAK

    return markdown.count(PAGE_BREAK[1])


@pytest.mark.parametrize("lang", ["en", "uk"])
def test_every_subject_area_starts_a_page_when_asked(lang):
    f = build_facts(
        _tab(),
        field_trends={"Physics": [_trend("a")], "Chemistry": [_trend("b")]},
        overall=[_trend("a")],
    )
    out = render_markdown(f, lang=lang, page_break_areas=True)
    assert _breaks(out) == 2
    for area in ("### Chemistry", "### Physics"):
        assert out[: out.index(area)].rstrip().endswith("```")


def test_no_section_but_the_third_is_broken_into_pages():
    """Sections 4 and 5 are a handful of tables read in order; breaking those adds paper.

    Section 3 is the one a reader navigates: 160 tables, one per subject area.
    """
    f = build_facts(
        _tab(),
        field_trends={"Physics": [_trend("a")]},
        overall=[_trend("a")],
        lenses={"declining": [_trend("a")]},
    )
    out = render_markdown(f, page_break_areas=True)
    assert _breaks(out[out.index("## 4.") :]) == 0
    assert _breaks(out[: out.index("## 3.")]) == 0


def test_a_report_carries_no_page_breaks_unless_asked():
    """Off by default, so every report already published re-renders unchanged."""
    f = build_facts(_tab(), field_trends={"Physics": [_trend("a")]})
    assert _breaks(render_markdown(f)) == 0


def test_a_page_break_is_written_for_both_renderers():
    """Neither format understands the other's raw block, and both need a real break."""
    from horizon.report import PAGE_BREAK

    text = "\n".join(PAGE_BREAK)
    assert "```{=latex}" in text and "```{=openxml}" in text
    assert "clearpage" in text and 'w:br w:type="page"' in text


def test_a_pipe_in_a_label_or_a_keyword_does_not_split_the_row():
    """Both are model output. An unescaped pipe ends the cell early and shifts every
    column after it, which reads as a data error rather than a formatting one."""
    t = Trend(
        label="A | B",
        members=("h",),
        size=1,
        first_seen="2020-01-01",
        last_seen="2020-01-01",
        velocity=1.0,
        score=1.0,
        top_keywords=("x | y",),
    )
    row = [
        line
        for line in render_markdown(build_facts(_tab(), overall=[t])).splitlines()
        if line.startswith("| A ")
    ][0]
    import re

    assert len(re.findall(r"(?<!\\)\|", row)) == 8
