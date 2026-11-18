"""Tests for the trend lenses — pure ranking over Trend fields, no model or clustering."""

import pytest

from horizon.lenses import BY_NAME, LENSES, Context, apply, peak_year
from horizon.trends import Trend


def _t(
    label,
    *,
    size=10,
    velocity=1.0,
    score=1.0,
    first_seen="2015-01-01",
    last_seen="2025-01-01",
    cohorts=(),
    attention_pctile=None,
    seed_hash="h",
) -> Trend:
    return Trend(
        label=label,
        members=tuple(f"m{i}" for i in range(size)),
        size=size,
        first_seen=first_seen,
        last_seen=last_seen,
        velocity=velocity,
        score=score,
        top_keywords=(),
        attention_pctile=attention_pctile,
        attention_n=size if attention_pctile is not None else 0,
        seed_hash=seed_hash,
        seed_date=first_seen,
        cohort_counts=tuple((str(y), n) for y, n in cohorts),
    )


def _names(rows):
    return [t.label for t in rows]


def _apply(lens, trends, **kw):
    """``apply`` with the size floor lifted, so a test exercises the lens, not the floor.

    The lens semantics below are about ranking and eligibility rules; the floor gets its
    own tests further down rather than silently gating every fixture.
    """
    kw.setdefault("min_size", 1)
    return apply(lens, trends, **kw)


# --- peak and its helpers -----------------------------------------------------


def test_peak_year_picks_the_largest_cohort():
    assert peak_year(_t("a", cohorts=[(2018, 3), (2020, 9), (2021, 4)])) == 2020


def test_peak_year_ties_go_to_the_earliest():
    """A cluster that matched its peak again has not peaked twice."""
    assert peak_year(_t("a", cohorts=[(2018, 5), (2021, 5)])) == 2018


def test_peak_year_is_none_without_a_curve():
    assert peak_year(_t("a")) is None


# --- rising / fading ----------------------------------------------------------


def test_rising_ranks_by_the_built_in_score():
    trends = [_t("low", score=1.0), _t("high", score=9.0)]
    assert _names(_apply(BY_NAME["emerging"], trends)) == ["high", "low"]


def test_fading_needs_both_decline_and_established_size():
    trends = [
        _t("big-quiet", size=100, velocity=0.2),
        _t("big-busy", size=100, velocity=2.0),  # not declining
        _t("small-quiet", size=1, velocity=0.1),  # below median size
    ]
    assert _names(_apply(BY_NAME["declining"], trends)) == ["big-quiet"]


def test_fading_is_bounded_so_volume_wins_over_a_tiny_stalled_cluster():
    """size / velocity would be unbounded and rank the two-member cluster first."""
    trends = [
        _t("halved", size=100, velocity=0.5),
        _t("stopped", size=60, velocity=0.0),
        _t("also-stopped", size=59, velocity=0.0),
    ]
    assert _names(_apply(BY_NAME["declining"], trends))[0] == "stopped"


# --- sleeping beauty ----------------------------------------------------------


def test_sleeping_beauty_wants_a_long_dormancy_and_a_live_cluster():
    trends = [
        _t(
            "dormant-then-up",
            first_seen="2015-01-01",
            velocity=2.0,
            cohorts=[(2015, 1), (2024, 30)],
        ),
        _t(
            "dormant-but-dead",
            first_seen="2015-01-01",
            velocity=0.2,
            cohorts=[(2015, 1), (2024, 30)],
        ),
        _t("immediate", first_seen="2024-01-01", velocity=2.0, cohorts=[(2024, 30)]),
    ]
    assert _names(_apply(BY_NAME["sleeping-beauty"], trends)) == ["dormant-then-up"]


# --- attention splits ---------------------------------------------------------


def test_influential_narrow_takes_the_small_half_by_citation():
    trends = [
        _t("small-cited", size=2, attention_pctile=0.95),
        _t("small-ignored", size=2, attention_pctile=0.10),
        _t("big-cited", size=50, attention_pctile=0.99),
    ]
    assert _names(_apply(BY_NAME["high-impact-low-output"], trends)) == [
        "small-cited",
        "small-ignored",
    ]


def test_crowded_takes_the_large_half_ascending():
    # Four clusters, so the median (26) falls between sizes rather than on one.
    trends = [
        _t("big-ignored", size=50, attention_pctile=0.05),
        _t("big-cited", size=60, attention_pctile=0.90),
        _t("small-a", size=2, attention_pctile=0.01),
        _t("small-b", size=2, attention_pctile=0.99),
    ]
    assert _names(_apply(BY_NAME["high-output-low-impact"], trends)) == [
        "big-ignored",
        "big-cited",
    ]


def test_the_two_attention_lenses_partition_the_population():
    """Every rated cluster belongs to exactly one half — the split is on the median,
    with the boundary going to the smaller side."""
    trends = [_t(f"t{i}", size=i + 1, attention_pctile=0.5) for i in range(7)]
    narrow = set(_names(_apply(BY_NAME["high-impact-low-output"], trends)))
    crowded = set(_names(_apply(BY_NAME["high-output-low-impact"], trends)))
    assert not (narrow & crowded)
    assert narrow | crowded == {t.label for t in trends}


def test_a_cluster_without_citation_data_is_absent_not_last():
    """No attention is not low attention; ranking it as zero would fake a measurement."""
    trends = [_t("rated", size=2, attention_pctile=0.5), _t("unrated", size=2)]
    assert _names(_apply(BY_NAME["high-impact-low-output"], trends)) == ["rated"]
    assert _names(_apply(BY_NAME["high-output-low-impact"], trends)) == []


# --- persistence --------------------------------------------------------------


def test_persistent_counts_years_holding_half_the_peak():
    trends = [
        _t("held", cohorts=[(2020, 10), (2021, 8), (2022, 6), (2023, 1)]),
        _t("spiked", cohorts=[(2020, 10), (2021, 1), (2022, 1)]),
    ]
    ranked = _apply(BY_NAME["sustained"], trends)
    assert _names(ranked) == ["held"]


# --- the machinery ------------------------------------------------------------


def test_apply_respects_top():
    trends = [_t(f"t{i}", score=float(i)) for i in range(5)]
    assert len(_apply(BY_NAME["emerging"], trends, top=2)) == 2


def test_ties_break_stably_on_the_seed_hash():
    """Equal scores must not swap between runs, or a diff of two reports is unreadable."""
    a = _t("a", score=1.0, seed_hash="aaa")
    b = _t("b", score=1.0, seed_hash="bbb")
    assert _names(_apply(BY_NAME["emerging"], [a, b])) == _names(
        _apply(BY_NAME["emerging"], [b, a])
    )


def test_context_of_an_empty_population():
    assert Context.of([]).median_size == 0.0


def test_every_lens_survives_an_empty_and_a_degenerate_population():
    degenerate = [_t("only")]
    for lens in LENSES:
        assert _apply(lens, []) == []
        _apply(lens, degenerate)  # must not raise


def test_lens_names_are_unique():
    assert len(BY_NAME) == len(LENSES)


# --- the size floor -----------------------------------------------------------


def test_a_lens_drops_clusters_below_the_size_floor():
    """A percentile over a handful of records is a property of those records."""
    trends = [
        _t("tiny", size=4, attention_pctile=0.99),
        _t("solid", size=40, attention_pctile=0.9),
    ]
    assert _names(apply(BY_NAME["high-impact-low-output"], trends, min_size=20)) == [
        "solid"
    ]


def test_the_floor_is_applied_before_the_median_the_split_compares_against():
    """Otherwise the two size-split lenses stop partitioning the same population.

    With clusters of 4, 4, 30 and 60 the raw median is 17, so 30 and 60 both read as
    "large" and ``influential-narrow`` comes back empty. Filtering first leaves 30 and 60,
    median 45, and the split falls between them where it belongs.
    """
    trends = [
        _t("a", size=4, attention_pctile=0.5),
        _t("b", size=4, attention_pctile=0.5),
        _t("c", size=30, attention_pctile=0.8),
        _t("d", size=60, attention_pctile=0.2),
    ]
    assert _names(apply(BY_NAME["high-impact-low-output"], trends, min_size=20)) == [
        "c"
    ]
    assert _names(apply(BY_NAME["high-output-low-impact"], trends, min_size=20)) == [
        "d"
    ]


def test_the_floor_partitions_every_survivor_between_the_two_attention_lenses():
    trends = [_t(f"t{i}", size=20 + i * 10, attention_pctile=i / 10) for i in range(6)]
    trends.append(_t("dropped", size=3, attention_pctile=0.5))
    narrow = set(_names(apply(BY_NAME["high-impact-low-output"], trends, min_size=20)))
    crowded = set(_names(apply(BY_NAME["high-output-low-impact"], trends, min_size=20)))
    assert narrow | crowded == {f"t{i}" for i in range(6)}
    assert not narrow & crowded


def test_sleeping_beauty_ranks_on_sleep_duration_not_duration_times_size():
    """The column is headed "sleep duration, years" and must contain exactly that.

    Ranking on the product printed 6420.00 in that column for a 2140-publication topic
    that slept three years.
    """
    long_sleep_small = _t(
        "long-small", size=5, cohorts=((2015, 1), (2016, 0), (2020, 4))
    )
    short_sleep_huge = _t("short-huge", size=500, cohorts=((2015, 1), (2017, 499)))
    lens = BY_NAME["sleeping-beauty"]
    assert lens.key(long_sleep_small) == 5.0
    assert _names(_apply(lens, [short_sleep_huge, long_sleep_small])) == [
        "long-small",
        "short-huge",
    ]


def test_a_tie_on_the_indicator_is_broken_by_publications():
    """Every topic that slept the full span reports the same sleep duration; inside a
    tie the larger body of work is the one worth naming."""
    small = _t("small", size=30, cohorts=((2015, 1), (2025, 29)))
    big = _t("big", size=400, cohorts=((2015, 1), (2025, 399)))
    lens = BY_NAME["sleeping-beauty"]
    assert lens.key(small) == lens.key(big)
    assert _names(_apply(lens, [small, big])) == ["big", "small"]
    assert _names(_apply(lens, [big, small])) == ["big", "small"]
