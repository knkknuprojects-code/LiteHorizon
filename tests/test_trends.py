from datetime import date

import pytest

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.clustering import GreedyClusterer
from horizon.trends import (
    Window,
    build_trends,
    parse_window,
    resolve_anchor,
)

NOW = "2026-06-30T00:00:00+00:00"


def _mk(
    content_hash,
    *,
    item_date,
    theme,
    keywords=("k",),
    attention=None,
    organizations=(),
) -> Analysis:
    return Analysis(
        summary=f"summary {content_hash}",
        signals=("s",),
        classification=Classification(
            theme, Maturity.EMERGING, Impact.HIGH, TimeHorizon.NEAR
        ),
        entities=Entities(organizations=tuple(organizations)),
        keywords=tuple(keywords),
        content_hash=content_hash,
        model_id="m",
        prompt_version="phase2.v1",
        strategy="combined",
        generated_at=item_date,
        item_date=item_date,
        attention=attention,
    )


def test_build_trends_clusters_and_scores():
    # Two recent "ai" items (one cluster) + one old "econ" item (another).
    analyses = [
        _mk("a1", item_date="2026-06-29", theme="ai", keywords=("llm", "open")),
        _mk("a2", item_date="2026-06-29", theme="ai", keywords=("llm",)),
        _mk("b1", item_date="2026-06-10", theme="econ"),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    trends = build_trends(analyses, vectors, GreedyClusterer(0.5), now=NOW, window="7d")

    assert len(trends) == 2
    top = trends[0]  # rising cluster ranks first
    assert top.label == "ai"
    assert top.size == 2
    assert top.velocity == 2.0  # 2 recent / max(0 prior, 1)
    assert top.score > trends[1].score
    assert top.first_seen == "2026-06-29"
    assert "llm" in top.top_keywords


def test_build_trends_min_size_filters_singletons():
    analyses = [
        _mk("a1", item_date="2026-06-29", theme="ai"),
        _mk("a2", item_date="2026-06-29", theme="ai"),
        _mk("b1", item_date="2026-06-29", theme="econ"),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    trends = build_trends(
        analyses, vectors, GreedyClusterer(0.5), now=NOW, window="7d", min_size=2
    )
    assert len(trends) == 1
    assert trends[0].label == "ai"


def test_build_trends_empty():
    assert build_trends([], [], GreedyClusterer(0.5), now=NOW, window="7d") == []


# --- window parsing -----------------------------------------------------------


def test_parse_window_accepts_days_years_and_bare_counts():
    assert parse_window("7d") == Window(7, "d")
    assert parse_window("7") == Window(7, "d")  # bare count means days
    assert parse_window("2y") == Window(2, "y")
    assert parse_window(Window(3, "y")) == Window(3, "y")  # passes through
    assert str(parse_window("2y")) == "2y"


@pytest.mark.parametrize("spec", ["", "0d", "-1y", "2m", "abc", "7dd"])
def test_parse_window_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_window(spec)


def test_window_year_start_snaps_to_january():
    window = Window(2, "y")
    assert window.start(date(2026, 1, 1)) == date(2025, 1, 1)
    assert window.start(date(2026, 1, 1), back=2) == date(2023, 1, 1)
    # A mid-year anchor still snaps — the cohort, not the day, is the unit.
    assert window.start(date(2026, 6, 30)) == date(2025, 1, 1)


def test_window_day_start_matches_plain_subtraction():
    assert Window(7, "d").start(date(2026, 6, 30)) == date(2026, 6, 23)


# --- anchor -------------------------------------------------------------------


def test_resolve_anchor_defaults_to_the_newest_item_date():
    analyses = [
        _mk("a", item_date="2019-01-01", theme="t"),
        _mk("b", item_date="2025-01-01", theme="t"),
    ]
    assert resolve_anchor(analyses) == date(2025, 1, 1)


def test_resolve_anchor_parses_before_comparing():
    # String max over mixed ISO forms orders by byte and gets offsets wrong.
    analyses = [
        _mk("a", item_date="2026-06-30T00:00:00-05:00", theme="t"),
        _mk("b", item_date="2026-07-01", theme="t"),
    ]
    assert resolve_anchor(analyses) == date(2026, 7, 1)


def test_resolve_anchor_accepts_a_bare_year():
    assert resolve_anchor([], now="2025") == date(2025, 1, 1)


def test_resolve_anchor_rejects_junk_and_an_empty_set():
    with pytest.raises(ValueError):
        resolve_anchor([], now="nonsense")
    with pytest.raises(ValueError):
        resolve_anchor([])


def test_build_trends_anchors_to_the_data_not_the_clock():
    # Every item is years old; with a wall-clock anchor these would all score 0.
    analyses = [
        _mk("a", item_date="2025-01-01", theme="ai"),
        _mk("b", item_date="2025-01-01", theme="ai"),
    ]
    (trend,) = build_trends(
        analyses, [[1.0, 0.0]] * 2, GreedyClusterer(0.5), window="2y"
    )
    assert trend.recency == 1.0
    assert trend.velocity > 0
    assert trend.score > 0


# --- year-scale windows (D1) --------------------------------------------------


def _cohort_store(counts):
    """One cluster with ``counts[year]`` members in each publication year."""
    analyses, vectors = [], []
    for year, n in counts.items():
        for i in range(n):
            analyses.append(_mk(f"{year}-{i}", item_date=f"{year}-01-01", theme="ai"))
            vectors.append([1.0, 0.0])
    return analyses, vectors


COUNTS = {2023: 8, 2024: 4, 2025: 2, 2026: 1}


def test_year_window_buckets_by_calendar_cohort():
    analyses, vectors = _cohort_store(COUNTS)
    (trend,) = build_trends(
        analyses, vectors, GreedyClusterer(0.5), window="2y", now="2026-01-01"
    )
    # recent {2025, 2026} = 3, prior {2023, 2024} = 12
    assert trend.velocity == round(3 / 12, 3)


def test_year_window_spans_the_same_cohort_count_at_every_anchor():
    # The defect a day-expressed window has: 730 days spans 2, 3 or 6 cohorts
    # depending on leap-day drift. A year window must always span exactly n.
    analyses, vectors = _cohort_store(COUNTS)
    at_2026 = build_trends(
        analyses, vectors, GreedyClusterer(0.5), window="2y", now="2026-01-01"
    )[0]
    at_2024 = build_trends(
        analyses, vectors, GreedyClusterer(0.5), window="2y", now="2024-01-01"
    )[0]
    assert at_2026.velocity == round(3 / 12, 3)  # {2025,2026} vs {2023,2024}
    assert at_2024.velocity == round(12 / 1, 3)  # {2023,2024} vs {2021,2022} -> empty


def test_recent_window_excludes_items_after_the_anchor():
    # D2: `recent` was unbounded above, so a backdated --now swept later items in.
    analyses, vectors = _cohort_store({2024: 1, 2025: 1, 2026: 5})
    (trend,) = build_trends(
        analyses, vectors, GreedyClusterer(0.5), window="1y", now="2025-01-01"
    )
    assert trend.velocity == round(1 / 1, 3)  # the five 2026 items must not count


# --- recency scale ------------------------------------------------------------


def test_recency_scale_gives_a_year_ladder():
    scale = 365
    ladder = {}
    for year in (2026, 2025, 2024):
        analyses = [_mk("a", item_date=f"{year}-01-01", theme="ai")]
        (trend,) = build_trends(
            analyses,
            [[1.0, 0.0]],
            GreedyClusterer(0.5),
            window="2y",
            now="2026-01-01",
            recency_scale_days=scale,
        )
        ladder[year] = trend.recency
    assert ladder[2026] == 1.0
    assert ladder[2025] == 0.5
    assert ladder[2024] == pytest.approx(0.333, abs=0.001)


def test_recency_default_scale_reproduces_the_day_formula():
    analyses = [_mk("a", item_date="2026-06-29", theme="ai")]
    (trend,) = build_trends(
        analyses, [[1.0, 0.0]], GreedyClusterer(0.5), window="7d", now=NOW
    )
    assert trend.recency == 0.5  # 1 / (1 + 1 day)


def test_score_arithmetic_is_pinned_for_a_daily_fixture():
    # Nothing pinned the absolute score before; this is the anti-drift anchor.
    analyses = [
        _mk("a", item_date="2026-06-29", theme="ai"),
        _mk("b", item_date="2026-06-29", theme="ai"),
    ]
    (trend,) = build_trends(
        analyses, [[1.0, 0.0]] * 2, GreedyClusterer(0.5), window="7d", now=NOW
    )
    # velocity 2.0 (2 recent / max(0 prior, 1)) * size 2 * recency 0.5 * attention 1.0
    assert (trend.velocity, trend.recency, trend.attention) == (2.0, 0.5, 1.0)
    assert trend.score == 2.0


# --- attention ----------------------------------------------------------------


def _one_cluster(attentions):
    analyses = [
        _mk(f"h{i}", item_date="2026-06-29", theme="ai", attention=a)
        for i, a in enumerate(attentions)
    ]
    return build_trends(
        analyses,
        [[1.0, 0.0]] * len(analyses),
        GreedyClusterer(0.5),
        window="7d",
        now=NOW,
    )[0]


def test_attention_is_neutral_when_no_member_carries_one():
    trend = _one_cluster([None, None])
    assert trend.attention == 1.0
    assert trend.attention_pctile is None
    assert trend.attention_n == 0
    assert trend.score == 2.0  # identical to the pinned no-attention fixture


def test_attention_averages_over_contributors_only():
    trend = _one_cluster([0.9, 0.5, None])
    assert trend.attention_pctile == 0.7  # mean of 0.9 and 0.5, not of three
    assert trend.attention_n == 2
    assert trend.attention == 1.2  # 1 + 0.5 * (0.7 - 0.5) * 2


def test_attention_spans_half_to_one_and_a_half():
    assert _one_cluster([0.0, 0.0]).attention == 0.5
    assert _one_cluster([0.5, 0.5]).attention == 1.0
    assert _one_cluster([1.0, 1.0]).attention == 1.5


def test_attention_never_zeroes_a_score():
    # An uncited cluster is demoted, never erased — it may be genuinely new.
    assert _one_cluster([0.0, 0.0]).score > 0


def test_attention_weight_zero_is_neutral():
    analyses = [_mk("a", item_date="2026-06-29", theme="ai", attention=1.0)]
    (trend,) = build_trends(
        analyses,
        [[1.0, 0.0]],
        GreedyClusterer(0.5),
        window="7d",
        now=NOW,
        attention_weight=0.0,
    )
    assert trend.attention == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_weight": 1.0},  # would make a zero percentile zero the score
        {"attention_weight": -0.1},
        {"recency_scale_days": 0},
    ],
)
def test_invalid_scoring_knobs_raise(kwargs):
    analyses = [_mk("a", item_date="2026-06-29", theme="ai")]
    with pytest.raises(ValueError):
        build_trends(
            analyses, [[1.0, 0.0]], GreedyClusterer(0.5), window="7d", **kwargs
        )


# --- lineage ------------------------------------------------------------------


def test_lineage_reports_seed_curve_growth_and_breadth():
    analyses, vectors = [], []
    for year, n in ((2021, 1), (2023, 2), (2025, 4)):
        for i in range(n):
            analyses.append(
                _mk(
                    f"{year}-{i}",
                    item_date=f"{year}-01-01",
                    theme="ai",
                    organizations=(f"Org {year}{i}", "Shared Lab"),
                )
            )
            vectors.append([1.0, 0.0])
    (trend,) = build_trends(
        analyses, vectors, GreedyClusterer(0.5), window="2y", now="2026-01-01"
    )
    assert trend.seed_date == "2021-01-01"
    assert trend.seed_hash == "2021-0"
    assert trend.cohort_counts == (("2021", 1), ("2023", 2), ("2025", 4))
    assert trend.growth == 0.75  # slope over the years present
    assert trend.breadth == 8  # 7 distinct orgs + the shared lab


def test_lineage_seed_is_deterministic_on_a_date_tie():
    analyses = [
        _mk("zzz", item_date="2025-01-01", theme="ai"),
        _mk("aaa", item_date="2025-01-01", theme="ai"),
    ]
    (trend,) = build_trends(
        analyses, [[1.0, 0.0]] * 2, GreedyClusterer(0.5), window="2y"
    )
    assert trend.seed_hash == "aaa"  # lowest hash breaks the tie


def test_lineage_growth_is_zero_for_a_single_cohort():
    trend = _one_cluster([None])
    assert trend.cohort_counts == (("2026", 1),)
    assert trend.growth == 0.0


def test_lineage_breadth_deduplicates_organization_spellings():
    analyses = [
        _mk("a", item_date="2026-06-29", theme="ai", organizations=("KPI", " kpi ")),
        _mk("b", item_date="2026-06-29", theme="ai", organizations=("KPI", "")),
    ]
    (trend,) = build_trends(
        analyses, [[1.0, 0.0]] * 2, GreedyClusterer(0.5), window="7d", now=NOW
    )
    assert trend.breadth == 1


# --- serialization ------------------------------------------------------------


def test_to_dict_carries_the_scoring_and_lineage_fields():
    trend = _one_cluster([0.9, 0.5])
    payload = trend.to_dict()
    for key in (
        "recency",
        "attention",
        "attention_pctile",
        "attention_n",
        "seed_hash",
        "seed_date",
        "cohort_counts",
        "growth",
        "breadth",
    ):
        assert key in payload
    assert payload["cohort_counts"] == {"2026": 2}  # a dict in JSON, not pairs


# --- cohort-normalised velocity -----------------------------------------------

# One cluster, two members in 2025 and one in 2024, anchored at 2025 with a 1-year
# window: recent = 2, prior = 1.
_NORM = [
    _mk("n1", item_date="2025-01-01", theme="ai"),
    _mk("n2", item_date="2025-01-01", theme="ai"),
    _mk("n3", item_date="2024-01-01", theme="ai"),
]
_NORM_VECS = [[1.0, 0.0]] * 3


def _velocity(baseline):
    (trend,) = build_trends(
        _NORM,
        _NORM_VECS,
        GreedyClusterer(0.5),
        now="2025-01-01T00:00:00+00:00",
        window="1y",
        baseline=baseline,
    )
    return trend.velocity


def test_velocity_without_a_baseline_is_the_count_ratio():
    assert _velocity(None) == pytest.approx(2.0)


def test_a_flat_baseline_reproduces_the_count_ratio_exactly():
    """The normalised form must not move numbers when there is nothing to correct for."""
    assert _velocity({2024: 500, 2025: 500}) == pytest.approx(2.0)


def test_growing_with_the_corpus_scores_one():
    """The acceptance criterion: holding a constant share is not rising.

    The cluster doubles (1 -> 2) and so does the corpus (500 -> 1000), so its share is
    flat and velocity is 1.0 — where the count form reports 2.0 and calls it a trend.
    """
    assert _velocity({2024: 500, 2025: 1000}) == pytest.approx(1.0)


def test_growing_against_a_shrinking_field_scores_higher():
    """Doubling while the field halves is the case the count form cannot see."""
    assert _velocity({2024: 1000, 2025: 500}) == pytest.approx(4.0)


def test_a_baseline_that_misses_the_windows_falls_back_to_counts():
    """Years outside both windows leave no denominator; counts are better than zero."""
    assert _velocity({2001: 10, 2002: 20}) == pytest.approx(2.0)


def test_no_prior_members_does_not_divide_by_zero():
    """The count form's max(prior, 1) guard has to survive normalisation."""
    analyses = [_mk("p1", item_date="2025-01-01", theme="ai")]
    (trend,) = build_trends(
        analyses,
        [[1.0, 0.0]],
        GreedyClusterer(0.5),
        now="2025-01-01T00:00:00+00:00",
        window="1y",
        baseline={2024: 100, 2025: 100},
    )
    assert trend.velocity == pytest.approx(1.0)
