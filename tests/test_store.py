from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.store import JsonlStore


def _analysis(prompt_version="phase2.v1", summary="S") -> Analysis:
    return Analysis(
        summary=summary,
        signals=("a",),
        classification=Classification(
            "t", Maturity.EMERGING, Impact.HIGH, TimeHorizon.NEAR
        ),
        entities=Entities(),
        keywords=("k",),
        content_hash="hash-1",
        model_id="m",
        prompt_version=prompt_version,
        strategy="combined",
        generated_at="2026-01-01T00:00:00+00:00",
        item_date="2026-01-01T00:00:00+00:00",
    )


def test_put_then_get_roundtrips(tmp_path):
    store = JsonlStore(tmp_path / "a.jsonl")
    a = _analysis()
    store.put(a)
    assert store.get("hash-1", "phase2.v1") == a


def test_get_misses_on_unknown_key(tmp_path):
    store = JsonlStore(tmp_path / "a.jsonl")
    store.put(_analysis())
    assert store.get("hash-1", "phase2.v2") is None  # different prompt version
    assert store.get("other", "phase2.v1") is None


def test_get_returns_none_when_no_file(tmp_path):
    assert JsonlStore(tmp_path / "missing.jsonl").get("h", "v") is None


def test_last_write_wins(tmp_path):
    store = JsonlStore(tmp_path / "a.jsonl")
    store.put(_analysis(summary="old"))
    store.put(_analysis(summary="new"))
    assert store.get("hash-1", "phase2.v1").summary == "new"


def test_list_all_one_per_key_last_wins(tmp_path):
    store = JsonlStore(tmp_path / "a.jsonl")
    store.put(_analysis(summary="old"))
    store.put(_analysis(summary="new"))  # same key
    store.put(_analysis(prompt_version="phase2.v2", summary="other"))  # new key
    all_ = store.list_all()
    assert len(all_) == 2
    summaries = {a.summary for a in all_}
    assert summaries == {"new", "other"}


def test_list_all_empty_when_no_file(tmp_path):
    assert JsonlStore(tmp_path / "missing.jsonl").list_all() == []
