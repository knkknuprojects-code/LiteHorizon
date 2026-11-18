"""Tests for joining analyses back to their records."""

import pytest

from horizon.analysis import Analysis, Classification, Entities, Impact, Maturity
from horizon.analysis import TimeHorizon
from horizon.corpus import RecordIndex, block, primary_field, record_hash


def _rec(title, abstract="", **extra):
    return {"title": title, "abstract": abstract, **extra}


def _ana(content_hash) -> Analysis:
    return Analysis(
        summary="s",
        signals=("x",),
        classification=Classification(
            "t", Maturity.EMERGING, Impact.LOW, TimeHorizon.NEAR
        ),
        entities=Entities(),
        keywords=("k",),
        content_hash=content_hash,
        model_id="m",
        prompt_version="p",
        strategy="combined",
        generated_at="2025-01-01",
        item_date="2025-01-01",
    )


# --- the hash rule ------------------------------------------------------------


def test_the_hash_matches_what_the_scan_would_have_stored():
    """The whole point of the module: one rule, not re-derived per call site."""
    from horizon.analysis import content_hash
    from horizon.ingest.wos import item_text

    rec = _rec("A title", "An abstract long enough to survive normalization.")
    assert record_hash(rec) == content_hash(item_text(rec))


def test_a_title_ending_in_punctuation_joins_differently():
    """The joiner depends on the title's last character — an ad-hoc f-string gets this
    wrong, which is how a naive join lost 8% of a sample."""
    body = "An abstract long enough to survive normalization."
    assert record_hash(_rec("Ends in a question?", body)) != record_hash(
        _rec("Ends in a question", body)
    )


def test_a_record_with_no_usable_text_has_no_hash():
    assert record_hash(_rec("", "")) == ""


# --- the index ----------------------------------------------------------------


def test_index_resolves_an_analysis_to_its_record():
    rec = _rec("A title", "An abstract long enough to survive normalization.")
    idx = RecordIndex.build([rec, _rec("", "")])
    assert len(idx) == 1
    assert idx.get(_ana(record_hash(rec))) == rec


def test_a_missing_analysis_is_none_not_an_error():
    """An index built from one shard will not hold another shard's analyses."""
    idx = RecordIndex.build([_rec("A title", "An abstract long enough to be kept.")])
    assert idx.get(_ana("nope")) is None


def test_coverage_reports_what_the_join_could_not_resolve():
    """A join that silently drops a fifth produces tables that look complete."""
    rec = _rec("A title", "An abstract long enough to survive normalization.")
    idx = RecordIndex.build([rec])
    assert idx.coverage([_ana(record_hash(rec)), _ana("nope")]) == (1, 2)


def test_pairs_yields_only_resolved_analyses():
    rec = _rec("A title", "An abstract long enough to survive normalization.")
    idx = RecordIndex.build([rec])
    got = list(idx.pairs([_ana(record_hash(rec)), _ana("nope")]))
    assert len(got) == 1 and got[0][1] == rec


# --- blocking -----------------------------------------------------------------


def test_primary_field_is_stable_under_reordering():
    assert primary_field(_rec("t", subject_areas=["Physics", "Chemistry"])) == (
        primary_field(_rec("t", subject_areas=["Chemistry", "Physics"]))
    )


def test_primary_field_falls_back_to_the_second_taxonomy():
    """20% of the corpus has no subject_areas and 18 points of that carry categories."""
    rec = _rec("t", subject_areas=[], wos_categories=["Optics"])
    assert primary_field(rec) == "Optics"


def test_primary_field_is_none_when_no_taxonomy_applies():
    assert primary_field(_rec("t")) is None


def test_block_puts_every_analysis_in_exactly_one_group():
    """Blocking is exact, not approximate: nothing is dropped or duplicated."""
    recs = [
        _rec(
            "A title one here", "Abstract long enough one.", subject_areas=["Physics"]
        ),
        _rec(
            "A title two here", "Abstract long enough two.", subject_areas=["Physics"]
        ),
        _rec(
            "A title three h", "Abstract long enough three.", wos_categories=["Optics"]
        ),
        _rec("A title four ok", "Abstract long enough four."),
    ]
    idx = RecordIndex.build(recs)
    analyses = [_ana(record_hash(r)) for r in recs]
    groups = block(idx.pairs(analyses))
    assert sorted(groups) == ["", "Optics", "Physics"]
    assert sum(len(v) for v in groups.values()) == len(analyses)


def test_unclassified_records_share_one_block():
    recs = [
        _rec("A title one here", "Abstract long enough one."),
        _rec("A title two here", "Abstract long enough two."),
    ]
    idx = RecordIndex.build(recs)
    groups = block(idx.pairs([_ana(record_hash(r)) for r in recs]))
    assert list(groups) == [""] and len(groups[""]) == 2
