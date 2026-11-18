"""Tests for the lenses that read records rather than trends."""

import pytest

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.corpus import RecordIndex, record_hash
from horizon.metadata_lenses import (
    broadening,
    facts_for,
    gather,
    language_asymmetry,
    new_entrants,
    split_authors,
)
from horizon.trends import Trend

BODY = "An abstract long enough to survive normalization and be kept."


def _rec(n, *, year, venue="J", authors="A, X", script="latin"):
    return {
        "title": f"Record number {n} title",
        "abstract": BODY,
        "year": year,
        "venue": venue,
        "authors": authors,
        "script": script,
    }


def _ana(rec) -> Analysis:
    return Analysis(
        summary="s",
        signals=("x",),
        classification=Classification(
            "t", Maturity.EMERGING, Impact.LOW, TimeHorizon.NEAR
        ),
        entities=Entities(),
        keywords=("k",),
        content_hash=record_hash(rec),
        model_id="m",
        prompt_version="p",
        strategy="combined",
        generated_at=f"{rec['year']}-01-01",
        item_date=f"{rec['year']}-01-01",
    )


def _trend(recs, label="t") -> Trend:
    return Trend(
        label=label,
        members=tuple(record_hash(r) for r in recs),
        size=len(recs),
        first_seen="2015-01-01",
        last_seen="2025-01-01",
        velocity=1.0,
        score=1.0,
        top_keywords=(),
    )


def _facts(recs, label="t"):
    idx = RecordIndex.build(recs)
    return facts_for(
        _trend(recs, label), {a.content_hash: a for a in map(_ana, recs)}, idx
    )


# --- author parsing -----------------------------------------------------------


def test_authors_split_on_semicolons():
    assert split_authors("Vanyorek, L; Ilosvai, AM") == {"Vanyorek, L", "Ilosvai, AM"}


def test_missing_authors_is_an_empty_set_not_an_error():
    assert split_authors(None) == frozenset() and split_authors("") == frozenset()


# --- coverage -----------------------------------------------------------------


def test_a_member_with_no_record_is_counted_but_not_ranked():
    """Partial coverage has to show, not skew: size includes it, resolved does not."""
    recs = [_rec(i, year=2020) for i in range(4)]
    idx = RecordIndex.build(recs[:2])
    trend = _trend(recs)
    f = facts_for(trend, {a.content_hash: a for a in map(_ana, recs)}, idx)
    assert f.size == 4 and f.resolved == 2


# --- broadening ---------------------------------------------------------------


def test_broadening_needs_venue_diversity_to_rise_not_just_volume():
    same = _facts(
        [_rec(i, year=2020, venue="J") for i in range(4)]
        + [_rec(i + 10, year=2024, venue="J") for i in range(8)],
        label="one-journal",
    )
    spread = _facts(
        [_rec(i, year=2020, venue="J") for i in range(4)]
        + [_rec(i + 10, year=2024, venue=f"J{i}") for i in range(8)],
        label="many-journals",
    )
    assert [f.label for f in broadening([same, spread], min_size=1)] == [
        "many-journals"
    ]


def test_a_half_too_small_yields_no_ratio():
    """Two records cannot establish a trend in venue diversity."""
    f = _facts([_rec(1, year=2020), _rec(2, year=2024)])
    assert f.venue_spread is None
    assert broadening([f], min_size=1) == []


# --- new entrants -------------------------------------------------------------


def test_new_entrants_measures_names_absent_from_the_earlier_half():
    fresh = _facts(
        [_rec(i, year=2020, authors="Old, A") for i in range(3)]
        + [_rec(i + 10, year=2024, authors=f"New{i}, B") for i in range(3)],
        label="fresh",
    )
    same = _facts(
        [_rec(i, year=2020, authors="Old, A") for i in range(3)]
        + [_rec(i + 10, year=2024, authors="Old, A") for i in range(3)],
        label="same",
    )
    assert fresh.new_author_share == pytest.approx(1.0)
    assert same.new_author_share == pytest.approx(0.0)
    assert [f.label for f in new_entrants([same, fresh], min_size=1)] == [
        "fresh",
        "same",
    ]


def test_new_entrants_is_absent_without_an_earlier_half():
    f = _facts([_rec(i, year=2024) for i in range(4)])
    assert f.new_author_share is None and new_entrants([f], min_size=1) == []


# --- language asymmetry -------------------------------------------------------


def test_asymmetry_ranks_one_sided_themes_first():
    one_sided = _facts(
        [_rec(i, year=2020, script="cyrillic") for i in range(5)], "local"
    )
    mixed = _facts(
        [_rec(i, year=2020, script="cyrillic") for i in range(3)]
        + [_rec(i + 10, year=2020, script="latin") for i in range(3)],
        "shared",
    )
    assert one_sided.script_imbalance == pytest.approx(0.0)
    assert mixed.script_imbalance == pytest.approx(1.0)
    assert [f.label for f in language_asymmetry([mixed, one_sided], min_size=1)] == [
        "local",
        "shared",
    ]


def test_asymmetry_is_absent_when_no_script_is_known():
    f = _facts([_rec(i, year=2020, script=None) for i in range(4)])
    assert f.script_imbalance is None and language_asymmetry([f], min_size=1) == []


# --- the gatherer -------------------------------------------------------------


def test_gather_returns_one_fact_per_trend_in_order():
    recs = [_rec(i, year=2020) for i in range(4)]
    idx = RecordIndex.build(recs)
    analyses = [_ana(r) for r in recs]
    out = gather([_trend(recs, "a"), _trend(recs, "b")], analyses, idx)
    assert [f.label for f in out] == ["a", "b"]


# --- the shared size floor ----------------------------------------------------


def _raw(label, size, *, spread=2.0):
    """A Facts row built directly, to vary size independently of the record fixtures.

    ``late_venues``/``early_venues`` are set so ``venue_spread`` is ``spread`` and
    ``new_author_share`` is present, which is all the two lenses read besides size.
    """
    from horizon.metadata_lenses import Facts

    return Facts(
        label=label,
        size=size,
        resolved=size,
        early_n=10,
        late_n=10,
        early_venues=10,
        late_venues=int(10 * spread),
        new_author_share=0.5,
        cyrillic=0,
        latin=size,
    )


def test_the_record_side_lenses_honour_the_same_floor_as_the_trend_lenses():
    """They were left out when the floor landed, and the sample report showed it.

    Its broadening table ranked clusters of 6 and 10 members at a flat 3.00x venue
    spread — a ratio over a handful of records, the same defect the floor exists to
    prevent on the trend side.
    """
    from horizon.lenses import DEFAULT_LENS_MIN_SIZE

    small = _raw("small", 6, spread=3.0)
    big = _raw("big", DEFAULT_LENS_MIN_SIZE, spread=1.5)
    assert [f.label for f in broadening([small, big])] == ["big"]
    assert [f.label for f in new_entrants([small, big])] == ["big"]


def test_language_asymmetry_gates_on_scripted_records_not_on_cluster_size():
    """A big cluster whose records did not join carries no evidence about scripts."""
    from horizon.metadata_lenses import Facts

    unjoined = Facts(
        label="big-but-unjoined",
        size=500,
        resolved=2,
        early_n=1,
        late_n=1,
        early_venues=1,
        late_venues=1,
        new_author_share=None,
        cyrillic=2,
        latin=0,
    )
    assert language_asymmetry([unjoined], min_size=20) == []
    assert [f.label for f in language_asymmetry([unjoined], min_size=1)] == [
        "big-but-unjoined"
    ]


def test_a_tie_is_broken_by_publications_not_by_spelling():
    """script_imbalance is 0.0 for every single-script topic, and there are hundreds.

    The whole visible table was one tie ordered alphabetically, so it opened with
    Abstract Algebra because of the letter A.
    """
    small = _raw("Abstract Algebra", 30)
    big = _raw("Zoology", 400)
    for f in (small, big):
        object.__setattr__(f, "cyrillic", 0)
        object.__setattr__(f, "latin", f.size)
    assert [f.label for f in language_asymmetry([small, big], min_size=20)] == [
        "Zoology",
        "Abstract Algebra",
    ]


def test_the_tie_break_survives_an_ascending_indicator_and_a_descending_one():
    """A composite sort key would reverse the tie-break along with the primary."""
    a = _raw("A", 30, spread=2.0)
    b = _raw("B", 400, spread=2.0)
    assert [f.label for f in broadening([a, b])] == ["B", "A"]
    assert [f.label for f in new_entrants([a, b])] == ["B", "A"]


def test_the_script_indicator_ranks_by_the_cyrillic_share_not_by_lopsidedness():
    """91.3% of the corpus is Latin, so "most lopsided" is every Latin-only topic.

    They tie at exactly 0.0 and the table filled with rows reading 0 / N that differed
    only in the topic's name — no information at all.
    """
    latin_only = _raw("latin-only", 400)
    object.__setattr__(latin_only, "cyrillic", 0)
    object.__setattr__(latin_only, "latin", 400)
    mixed = _raw("mixed", 100)
    object.__setattr__(mixed, "cyrillic", 70)
    object.__setattr__(mixed, "latin", 30)
    ranked = language_asymmetry([latin_only, mixed], min_size=20)
    assert [f.label for f in ranked] == ["mixed", "latin-only"]
    assert mixed.cyrillic_share == pytest.approx(0.7)


def test_a_topic_with_no_scripted_publications_has_no_cyrillic_share():
    f = _raw("unknown", 50)
    object.__setattr__(f, "cyrillic", 0)
    object.__setattr__(f, "latin", 0)
    assert f.cyrillic_share is None
    assert language_asymmetry([f], min_size=1) == []
