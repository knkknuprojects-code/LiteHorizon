"""Tests for the figure catalogue and the decisions inside the drawing code.

The drawing itself produces pixels; what is worth pinning is which cohorts a curve covers,
which population the scatters use, that a facts file too old to carry the denominator
fails instead of drawing a wrong share, and that every figure the catalogue promises has a
place in the report.
"""

import pytest

from horizon import figures as figs


def _facts(**over):
    base = {
        "records": 100,
        "unclassified": 10,
        "years": [2020, 2021, 2022],
        "classified_years": [2020, 2021],
        "year_totals": {"2020": 40, "2021": 40, "2022": 20},
        "fields": [
            {
                "name": "Physics",
                "total": 50,
                "slope": 1.0,
                "share_first": 0.5,
                "share_last": 0.6,
                "share_delta": 0.1,
                "counts": {"2020": 20, "2021": 24, "2022": 0},
            }
        ],
        "pairs": [],
        "field_trends": {
            "Physics": [
                {
                    "label": "A",
                    "size": 30,
                    "velocity": 1.4,
                    "growth": 3.0,
                    "attention_pctile": 0.8,
                },
                {
                    "label": "B",
                    "size": 8,
                    "velocity": 0.6,
                    "growth": -1.0,
                    "attention_pctile": None,
                },
            ]
        },
        "overall": [],
        "lenses": {},
        "meta_lenses": {},
    }
    base.update(over)
    return base


def test_curves_are_drawn_over_classified_cohorts_only():
    """Including a cohort with no taxonomy drops every area to zero in its last point."""
    assert figs._classified_years(_facts()) == [2020, 2021]


def test_classified_years_are_derived_when_the_facts_predate_the_field():
    facts = _facts()
    del facts["classified_years"]
    assert figs._classified_years(facts) == [2020, 2021]


def test_a_facts_file_without_the_denominator_refuses_to_draw_a_share(tmp_path):
    """The per-area counts sum past their cohort, so a share from them looks right."""
    facts = _facts()
    del facts["year_totals"]
    with pytest.raises(SystemExit, match="year_totals"):
        figs.f5_share_over_time(facts, figs.TEXT["uk"], tmp_path / "f5.png")


def test_the_scatters_use_the_per_area_topics_not_the_capped_corpus_list():
    """`overall` is truncated to --top-trends and has no shape to plot."""
    assert [t["label"] for t in figs._topics(_facts())] == ["A", "B"]


def test_both_languages_define_the_same_figure_text():
    assert set(figs.TEXT["en"]) == set(figs.TEXT["uk"])


def test_the_indicator_panel_covers_every_indicator_the_report_prints():
    """A panel that quietly omits one would read as "this indicator found nothing"."""
    from horizon.lenses import LENSES

    panelled = {name for name, _, _ in figs.PANEL}
    expected = {lens.name for lens in LENSES if lens.name != "emerging"}
    expected |= {"journal-dispersion", "new-authors", "script-distribution"}
    assert panelled == expected


def test_the_one_sided_indicator_is_drawn_stacked_not_as_a_share():
    """Its leading rows are 100% one script, so a share bar renders an empty panel."""
    entry = next(e for e in figs.PANEL if e[0] == "script-distribution")
    assert entry[2] is None


def test_every_figure_has_a_file_name_a_function_a_section_and_a_caption():
    names = [name for name, _fn, _section, _caption in figs.FIGURES]
    assert names == sorted(names), "figure order should follow the f1..fN prefix"
    for name, fn, section, caption in figs.FIGURES:
        assert callable(fn)
        assert section in {"1", "2", "3", "4", "5"}, name
        assert caption in figs.TEXT["en"] and caption in figs.TEXT["uk"], name


def test_references_name_only_the_figures_that_were_drawn():
    """A caption for a picture that does not exist is worse than a missing figure."""
    refs = figs.references(["f4-topic-trajectories"], "uk", "figures/x-ua")
    assert list(refs) == ["4"]
    assert refs["4"] == [
        "![Діаграма 4. Теми по всьому корпусу, за роками]"
        "(figures/x-ua/f4-topic-trajectories.png)"
    ]


def test_references_are_empty_when_nothing_was_drawn():
    assert figs.references([], "en", "figures/x") == {}


def test_the_figures_render_end_to_end(tmp_path):
    """A smoke test: five files, both languages, no exception on real-shaped facts."""
    written = figs.draw(_facts(), tmp_path / "uk", "uk")
    assert set(written) >= {
        "f1-area-share-over-time",
        "f2-share-change",
        "f5-growth-measures",
        "f8-output-vs-impact",
    }
    assert all((tmp_path / "uk" / f"{n}.png").stat().st_size > 1000 for n in written)


def test_the_catalogue_is_in_the_order_the_report_shows_the_figures():
    """A reader meeting "Діаграма 5" on the first page and "Діаграма 1" near the end is
    being numbered by a catalogue they cannot see."""
    sections = [section for _name, _fn, section, _caption in figs.FIGURES]
    assert sections == sorted(sections)


def test_the_number_in_the_caption_matches_the_position_in_the_document():
    names = [name for name, *_ in figs.FIGURES]
    refs = figs.references(names, "uk", "figures/x")
    numbered = [
        int(line.split(".")[0].split()[-1])
        for section in sorted(refs)
        for line in refs[section]
    ]
    assert numbered == list(range(1, len(names) + 1))


def _vocab_facts(n_words=16, years=(2020, 2021, 2022, 2023)):
    """A corpus that quadruples, with words that gain share and words that lose it."""
    totals = {str(y): 1000 * (i + 1) for i, y in enumerate(years)}
    topics = []
    for w in range(n_words):
        # Rising: the count outruns the corpus, so the share climbs. Falling: the count
        # holds exactly, which against a growing corpus is a share cut to a quarter.
        rising = w % 2 == 0
        counts = {
            str(y): (60 * (i + 1) ** 2 if rising else 200) for i, y in enumerate(years)
        }
        topics.append(
            {
                "label": f"T{w}",
                "seed_hash": f"h{w}",
                "size": sum(counts.values()),
                "velocity": 1.2,
                "growth": 1.0,
                "attention_pctile": None,
                "cohort_counts": counts,
                "top_keywords": [f"word{w}", "shared"],
            }
        )
    return _facts(
        years=list(years),
        classified_years=list(years),
        year_totals=totals,
        field_trends={"Physics": topics},
    )


def test_the_vocabulary_tree_is_silent_without_the_inputs_it_needs(tmp_path):
    """A report without cohort counts or keywords loses the figure, not the report."""
    import matplotlib.pyplot as plt

    figs._style(plt)
    out = tmp_path / "f.png"
    figs.f8_rising_vocabulary(_facts(), figs.TEXT["en"], out)
    assert not out.exists()


def test_the_vocabulary_tree_draws_from_growing_topics(tmp_path):
    out = tmp_path / "f.png"
    import matplotlib.pyplot as plt

    figs._style(plt)
    figs.f8_rising_vocabulary(_vocab_facts(), figs.TEXT["uk"], out)
    assert out.exists() and out.stat().st_size > 1000


def test_growing_is_share_growing_not_count_growing():
    """The corpus adds records every year, so a count filter would keep everything.

    The falling topics here hold their count exactly; against a corpus that doubles, that
    is a halving share, and they must not reach the figure.
    """
    facts = _vocab_facts()
    years = figs._classified_years(facts)
    grown = figs._growing_topics(facts, years)
    assert len(grown) == 8  # the eight rising of sixteen
    assert all(t["counts"][-1] > t["counts"][0] for t in grown)


def test_a_topic_listed_under_several_blocks_is_counted_once():
    """An unblocked clustering repeats a topic per field its members carry."""
    facts = _vocab_facts(n_words=1)
    only = facts["field_trends"]["Physics"]
    facts["field_trends"] = {"Physics": only, "Chemistry": list(only)}
    years = figs._classified_years(facts)
    assert len(figs._growing_topics(facts, years)) == 1
