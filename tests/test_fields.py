"""Tests for the taxonomy layer — counting only, no model and no embedder."""

import pytest

from horizon.fields import Series, field_series, pair_series, tabulate


def _rec(year, areas=(), key="subject_areas"):
    return {"year": year, key: list(areas)}


RECORDS = [
    _rec(2015, ["Physics"]),
    _rec(2015, ["Physics", "Chemistry"]),
    _rec(2015, []),  # unclassified, but still part of the cohort
    _rec(2016, ["Physics", "Chemistry"]),
    _rec(2016, ["Chemistry"]),
    _rec(2016, ["Chemistry"]),
]


def test_cohort_totals_include_unclassified_records():
    """A record with no subject area is still part of its year, or shares inflate."""
    tab = tabulate(RECORDS)
    assert tab.year_totals == {2015: 3, 2016: 3}
    assert tab.records == 6
    assert tab.unclassified == 1


def test_multi_label_records_count_in_every_field():
    tab = tabulate(RECORDS)
    assert tab.field_counts["Physics"] == {2015: 2, 2016: 1}
    assert tab.field_counts["Chemistry"] == {2015: 1, 2016: 3}


def test_pairs_are_canonical_and_counted_once():
    """A pair has one order, so ('Chemistry', 'Physics') is the only key."""
    tab = tabulate(RECORDS)
    assert list(tab.pair_counts) == [("Chemistry", "Physics")]
    assert tab.pair_counts[("Chemistry", "Physics")] == {2015: 1, 2016: 1}


def test_a_repeated_area_is_not_double_counted():
    """Deduplication matters: a record listing an area twice is still one record."""
    tab = tabulate([_rec(2020, ["Physics", "Physics"])])
    assert tab.field_counts["Physics"] == {2020: 1}
    assert tab.pair_counts == {}


def test_years_filter_excludes_whole_cohorts():
    """A partial final year depresses every rate, so callers must be able to drop it."""
    tab = tabulate(RECORDS, years=[2015])
    assert tab.years == (2015,)
    assert tab.field_counts["Chemistry"] == {2015: 1}


def test_records_without_a_usable_year_are_skipped():
    tab = tabulate([_rec(None, ["Physics"]), {"subject_areas": ["Physics"]}])
    assert tab.records == 0 and tab.year_totals == {}


def test_alternative_taxonomy_key():
    tab = tabulate([_rec(2015, ["Optics"], key="wos_categories")], key="wos_categories")
    assert tab.field_counts["Optics"] == {2015: 1}


# --- derived series -----------------------------------------------------------


def _series_named(rows, name):
    return next(s for s in rows if s.name == name)


def test_shares_are_measured_against_the_cohort():
    tab = tabulate(RECORDS)
    physics = _series_named(field_series(tab), "Physics")
    assert physics.share_first == pytest.approx(2 / 3)
    assert physics.share_last == pytest.approx(1 / 3)
    assert physics.share_delta == pytest.approx(-1 / 3)


def test_slope_is_records_per_year():
    """Two cohorts one year apart: the slope is just the difference."""
    tab = tabulate(RECORDS)
    assert _series_named(field_series(tab), "Chemistry").slope == pytest.approx(2.0)
    assert _series_named(field_series(tab), "Physics").slope == pytest.approx(-1.0)


def test_a_missing_year_counts_as_zero_not_as_a_gap():
    """A field that stops appearing has declined; skipping the gap would hide it."""
    tab = tabulate([_rec(2015, ["A"]), _rec(2015, ["A"]), _rec(2016, ["B"])])
    a = _series_named(field_series(tab), "A")
    assert a.counts == ((2015, 2), (2016, 0))
    assert a.slope == pytest.approx(-2.0)


def test_single_year_has_no_slope():
    tab = tabulate([_rec(2015, ["A"])])
    assert _series_named(field_series(tab), "A").slope == 0.0


def test_field_series_ranks_by_share_gain_and_honours_min_total():
    tab = tabulate(RECORDS)
    assert [s.name for s in field_series(tab)] == ["Chemistry", "Physics"]
    assert [s.name for s in field_series(tab, min_total=4)] == ["Chemistry"]


def test_pair_series_names_and_ranks_pairs():
    tab = tabulate(RECORDS)
    (pair,) = pair_series(tab)
    assert pair.name == "Chemistry + Physics"
    assert pair.total == 2


def test_count_reads_a_year_off_a_series():
    tab = tabulate(RECORDS)
    physics = _series_named(field_series(tab), "Physics")
    assert physics.count(2015) == 2 and physics.count(1999) == 0


def test_field_totals_are_the_normalising_denominator():
    """The point of the module: a field-relative rate divides by this."""
    tab = tabulate(RECORDS)
    assert tab.field_totals("Chemistry") == {2015: 1, 2016: 3}
    assert tab.field_totals("Astrology") == {}


def test_empty_input():
    tab = tabulate([])
    assert tab.records == 0 and field_series(tab) == [] and pair_series(tab) == []


def test_series_is_frozen():
    tab = tabulate(RECORDS)
    with pytest.raises(Exception):
        field_series(tab)[0].name = "x"
    assert isinstance(field_series(tab)[0], Series)


# --- a cohort can exist without the taxonomy existing for it -------------------


_STUB = [
    {"year": 2015, "subject_areas": ["Physics"]},
    {"year": 2015, "subject_areas": ["Medicine"]},
    {"year": 2016, "subject_areas": ["Physics"]},
    # The final cohort is real but unclassified — the shape of the WoS/Scopus 2026 stub.
    {"year": 2017, "subject_areas": []},
    {"year": 2017, "subject_areas": []},
]


def test_classified_years_excludes_a_cohort_the_taxonomy_never_reached():
    tab = tabulate(_STUB)
    assert tab.years == (2015, 2016, 2017)
    assert tab.classified_years == (2015, 2016)


def test_shares_are_read_at_the_last_classified_cohort_not_the_last_cohort():
    """Otherwise every field reads as having lost exactly the share it started with.

    Measured on the real corpus: the 2026 cohort holds 482 records and none carries a
    subject area, so section 1 ranked every field by a decline that was a gap in the
    export.
    """
    physics = next(s for s in field_series(tabulate(_STUB)) if s.name == "Physics")
    assert physics.share_first == pytest.approx(0.5)
    assert physics.share_last == pytest.approx(1.0)
    assert physics.share_delta == pytest.approx(0.5)


def test_a_wholly_unclassified_corpus_falls_back_to_its_own_years():
    """No classified cohort at all is a different failure, and must not divide by zero."""
    tab = tabulate([{"year": 2015, "subject_areas": []}])
    assert tab.classified_years == ()
    assert field_series(tab) == []
