from horizon.analysis import content_hash
from horizon.sources import SourceStore


def test_put_returns_hash_and_get_roundtrips(tmp_path):
    store = SourceStore(tmp_path / "sources.jsonl")
    text = "a development worth watching"
    digest = store.put(text)
    assert digest == content_hash(text)
    assert store.get(digest) == text


def test_get_missing_returns_none(tmp_path):
    store = SourceStore(tmp_path / "sources.jsonl")
    assert store.get("nope") is None


def test_all_returns_mapping(tmp_path):
    store = SourceStore(tmp_path / "sources.jsonl")
    store.put("first text")
    store.put("second text")
    allsrc = store.all()
    assert set(allsrc.values()) == {"first text", "second text"}


def test_last_write_wins_dedupes_by_hash(tmp_path):
    # Same normalized text -> same hash; the latest record wins, no duplicate key.
    store = SourceStore(tmp_path / "sources.jsonl")
    store.put("same   text")
    store.put("same text")
    allsrc = store.all()
    assert len(allsrc) == 1
