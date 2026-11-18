import json

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
)
from horizon.embeddings import EmbeddingCache, embed_text


def _mk(content_hash, summary="a summary", signals=("sig",)) -> Analysis:
    return Analysis(
        summary=summary,
        signals=tuple(signals),
        classification=Classification(
            "t", Maturity.EMERGING, Impact.HIGH, TimeHorizon.NEAR
        ),
        entities=Entities(),
        keywords=("k",),
        content_hash=content_hash,
        model_id="m",
        prompt_version="phase2.v1",
        strategy="combined",
        generated_at="2026-01-01T00:00:00+00:00",
        item_date="2026-01-01T00:00:00+00:00",
    )


def test_embed_text_joins_summary_and_signals():
    a = _mk("h", summary="S", signals=("one", "two"))
    assert embed_text(a) == "S one two"


def test_cache_computes_misses_then_reuses(tmp_path, make_embedder):
    path = tmp_path / "emb.jsonl"
    analyses = [_mk("h1"), _mk("h2")]

    e1 = make_embedder(lambda t: [1.0, 0.0])
    vecs = EmbeddingCache(e1, path).embed_analyses(analyses)
    assert set(vecs) == {"h1", "h2"}
    assert len(e1.calls) == 1 and len(e1.calls[0]) == 2  # both computed once
    assert path.is_file()

    # A fresh cache over the same file (same model) recomputes nothing.
    e2 = make_embedder(lambda t: [9.0, 9.0])  # would differ if it recomputed
    vecs2 = EmbeddingCache(e2, path).embed_analyses(analyses)
    assert e2.calls == []
    assert vecs2 == vecs


def test_cache_only_computes_new_hashes(tmp_path, make_embedder):
    path = tmp_path / "emb.jsonl"
    e = make_embedder(lambda t: [1.0, 0.0])
    cache = EmbeddingCache(e, path)
    cache.embed_analyses([_mk("h1")])
    e.calls.clear()

    cache2 = EmbeddingCache(e, path)
    cache2.embed_analyses([_mk("h1"), _mk("h2")])  # h1 cached, only h2 is new
    assert e.calls == [[embed_text(_mk("h2"))]]


def test_the_sidecar_is_read_once_per_instance(tmp_path, make_embedder):
    """A report is a loop over blocks, and re-parsing the sidecar per block is quadratic
    at corpus scale: 2.2 GB re-read 250 times made a warm cache slower than the run that
    computed the vectors from scratch.

    Asserted through the documented consequence — a line appended by someone else after
    the first read is not picked up, because the map is no longer rebuilt from the file.
    """
    path = tmp_path / "emb.jsonl"
    cache = EmbeddingCache(make_embedder(), path)
    cache.embed_analyses([_mk("h1")])

    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"content_hash": "h9", "model": "fake-embedder", "vector": [7.0, 7.0]}
            )
            + "\n"
        )
    assert "h9" not in cache._load()

    # A new instance reads the file again and does see it.
    assert "h9" in EmbeddingCache(make_embedder(), path)._load()


def test_a_vector_written_this_run_is_visible_without_a_reread(tmp_path, make_embedder):
    path = tmp_path / "emb.jsonl"
    cache = EmbeddingCache(make_embedder(), path)
    cache.embed_analyses([_mk("h1")])
    second = cache.embed_analyses([_mk("h1"), _mk("h2")])
    assert set(second) == {"h1", "h2"}
    third = cache.embed_analyses([_mk("h2")])
    assert third["h2"] == second["h2"]
