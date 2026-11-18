import argparse
import json
from dataclasses import replace

import pytest

import horizon.cli as cli
from horizon.analysis import Analysis, content_hash
from horizon.prompts import PROMPT_VERSION
from horizon.sources import SourceStore
from horizon.store import JsonlStore

COMBINED_JSON = json.dumps(
    {
        "summary": "A new open model was released.",
        "signals": ["state-of-the-art reasoning", "startups fine-tuning"],
        "classification": {
            "theme": "open models",
            "maturity": "emerging",
            "impact": "high",
            "time_horizon": "near",
        },
        "entities": {
            "organizations": ["A cloud provider"],
            "people": [],
            "technologies": ["LLM"],
            "locations": [],
        },
        "keywords": ["open-weights", "reasoning"],
    }
)


def _patch_model(monkeypatch, fake):
    monkeypatch.setattr(
        "horizon.llm.transformers_client.TransformersClient",
        lambda *a, **k: fake,
    )


def _provenance(**over):
    base = {
        **json.loads(COMBINED_JSON),
        "content_hash": "h",
        "model_id": "fake",
        "prompt_version": PROMPT_VERSION,
        "strategy": "combined",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "item_date": "2026-01-01T00:00:00+00:00",
    }
    base.update(over)
    return base


# --- scan ---------------------------------------------------------------------


def test_cli_missing_file_returns_error(capsys, tmp_path):
    rc = cli.main(["scan", str(tmp_path / "nope.txt")])
    assert rc == 1
    assert "file not found" in capsys.readouterr().err


def test_cli_scans_file_and_prints_report(monkeypatch, capsys, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development worth watching", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["scan", str(text_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Summary" in out
    assert "A new open model was released." in out
    assert "open models" in out


def test_cli_json_flag_emits_valid_json(monkeypatch, capsys, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["scan", str(text_file), "--json", "--no-save"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"]["impact"] == "high"
    assert payload["prompt_version"] == PROMPT_VERSION


def test_cli_scan_date_sets_item_date(monkeypatch, capsys, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(
        ["scan", str(text_file), "--json", "--no-save", "--date", "2026-05-01"]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["item_date"] == "2026-05-01"


def test_cli_saves_to_store(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["scan", str(text_file)])
    assert rc == 0
    assert (tmp_path / "data" / "analyses.jsonl").is_file()


def test_cli_scan_persists_source_text(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text = "a development worth watching"
    text_file = tmp_path / "input.txt"
    text_file.write_text(text, encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["scan", str(text_file)])
    assert rc == 0
    assert SourceStore(tmp_path / "data" / "sources.jsonl").get(content_hash(text)) == (
        text
    )


def test_cli_no_save_writes_nothing(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["scan", str(text_file), "--no-save"])
    assert rc == 0
    assert not (tmp_path / "data").exists()


def test_cli_cache_hit_skips_model(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    text = "a development"
    text_file = tmp_path / "input.txt"
    text_file.write_text(text, encoding="utf-8")

    cached = Analysis.from_dict(_provenance(content_hash=content_hash(text)))
    JsonlStore(tmp_path / "data" / "analyses.jsonl").put(cached)

    monkeypatch.setattr(
        "horizon.llm.transformers_client.TransformersClient",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("model used on cache hit")
        ),
    )

    rc = cli.main(["scan", str(text_file)])
    assert rc == 0
    assert "A new open model was released." in capsys.readouterr().out


def test_cli_applies_generation_overrides(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development", encoding="utf-8")

    captured = {}

    def fake_analyze(
        text, client, *, settings=None, config=None, strategy=None, item_date=None
    ):
        captured["config"] = config
        return Analysis.from_dict(_provenance())

    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    monkeypatch.setattr("horizon.extract.analyze_text", fake_analyze)

    rc = cli.main(
        [
            "scan",
            str(text_file),
            "--max-new-tokens",
            "128",
            "--temperature",
            "0.2",
            "--no-save",
        ]
    )
    assert rc == 0
    assert captured["config"].max_new_tokens == 128
    assert captured["config"].temperature == 0.2


def test_cli_reports_runtime_error_cleanly(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("x", encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        "horizon.llm.transformers_client.TransformersClient", lambda *a, **k: object()
    )
    monkeypatch.setattr("horizon.extract.analyze_text", boom)

    rc = cli.main(["scan", str(text_file)])
    assert rc == 1
    assert "model unavailable" in capsys.readouterr().err


# --- trends -------------------------------------------------------------------


def test_cli_trends_empty_store(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["trends"])
    assert rc == 1
    assert "no analyses" in capsys.readouterr().err.lower()


def test_cli_trends_reports_clusters(monkeypatch, capsys, tmp_path, make_embedder):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    for i in range(2):
        store.put(
            Analysis.from_dict(
                _provenance(content_hash=f"h{i}", summary=f"open model release {i}")
            )
        )

    # All texts map to the same vector -> one cluster.
    embedder = make_embedder(lambda t: [1.0, 0.0])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder", lambda *a, **k: embedder
    )

    rc = cli.main(["trends", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["size"] == 2


# --- export -------------------------------------------------------------------


def test_cli_export_empty_store(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["export"])
    assert rc == 1
    assert "no analyses" in capsys.readouterr().err.lower()


def test_cli_export_writes_sft_dataset(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    sources = SourceStore(tmp_path / "data" / "sources.jsonl")
    for i in range(2):
        text = f"a development worth watching {i}"
        store.put(Analysis.from_dict(_provenance(content_hash=content_hash(text))))
        sources.put(text)  # keyed by the same content hash as the analysis

    rc = cli.main(["export"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Exported" in out
    assert (tmp_path / "data" / "sft" / "train.jsonl").is_file()
    assert (tmp_path / "data" / "sft" / "eval.jsonl").is_file()
    assert (tmp_path / "data" / "sft" / "manifest.json").is_file()


# --- dataset-stats ------------------------------------------------------------


def test_cli_dataset_stats_empty_store(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["dataset-stats"])
    assert rc == 1
    assert "no analyses" in capsys.readouterr().err.lower()


def test_cli_dataset_stats_reports(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    sources = SourceStore(tmp_path / "data" / "sources.jsonl")
    text = "a development worth watching"
    store.put(Analysis.from_dict(_provenance(content_hash=content_hash(text))))
    sources.put(text)

    rc = cli.main(["dataset-stats", "--target", "5", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] == 1
    assert payload["gap"] == 4


# --- approve ------------------------------------------------------------------


def test_cli_approve_requires_target(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    store.put(Analysis.from_dict(_provenance(content_hash="abc123")))
    rc = cli.main(["approve"])
    assert rc == 1
    assert "--all" in capsys.readouterr().err


def test_cli_approve_all_marks_trainable(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    store.put(Analysis.from_dict(_provenance(content_hash="abc123")))
    store.put(Analysis.from_dict(_provenance(content_hash="def456")))

    rc = cli.main(["approve", "--all"])
    assert rc == 0
    assert "Approved 2" in capsys.readouterr().out
    assert all(a.approved for a in store.list_all())


def test_cli_approve_by_hash_prefix_and_set_source(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    store.put(Analysis.from_dict(_provenance(content_hash="abc123")))
    store.put(Analysis.from_dict(_provenance(content_hash="def456")))

    rc = cli.main(["approve", "abc", "--set-source", "human"])
    assert rc == 0
    by_hash = {a.content_hash: a for a in store.list_all()}
    assert by_hash["abc123"].approved and by_hash["abc123"].source == "human"
    assert not by_hash["def456"].approved


# --- gather -------------------------------------------------------------------


def test_cli_gather_local_lands_corpus(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("a development worth watching now", encoding="utf-8")

    rc = cli.main(["gather", "local", "--path", str(docs)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 new" in out
    # Split by source: local files land under corpus/local/ as self-contained records.
    assert (tmp_path / "corpus" / "local" / "manifest.jsonl").is_file()
    records = list((tmp_path / "corpus").glob("*/*.json"))
    assert len(records) == 1


def test_cli_gather_local_requires_path(capsys):
    rc = cli.main(["gather", "local"])
    assert rc == 1
    assert "--path" in capsys.readouterr().err


def test_cli_gather_rss_requires_url(capsys):
    rc = cli.main(["gather", "rss"])
    assert rc == 1
    assert "--url" in capsys.readouterr().err


def test_cli_gather_missing_dir_errors(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["gather", "local", "--path", str(tmp_path / "nope")])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().err.lower()


_FEED_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Origin Title</title>
  <item>
    <title>One</title>
    <link>http://example.org/1</link>
    <description>A development worth watching in the tech landscape</description>
    <pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_cli_gather_feeds_fans_over_config(monkeypatch, capsys, tmp_path):
    pytest.importorskip("feedparser")
    monkeypatch.chdir(tmp_path)
    feed_a = tmp_path / "a.xml"
    feed_a.write_text(_FEED_XML, encoding="utf-8")
    feed_b = tmp_path / "b.xml"
    feed_b.write_text(
        _FEED_XML.replace("worth watching in the tech", "worth tracking in the policy"),
        encoding="utf-8",
    )
    config = tmp_path / "feeds.toml"
    config.write_text(
        f'[[feed]]\nname = "Feed A"\nurl = "{feed_a}"\n'
        f'[[feed]]\nname = "Feed B"\nurl = "{feed_b}"\n',
        encoding="utf-8",
    )

    rc = cli.main(["gather", "feeds", "--config", str(config)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Feed A" in out and "Feed B" in out  # per-feed lines in multi-source mode
    assert "2 new" in out
    # Each feed becomes its own source subdir; the config name is the recorded source.
    lines = [
        json.loads(line)
        for m in (tmp_path / "corpus").glob("*/manifest.jsonl")
        for line in m.read_text().strip().splitlines()
    ]
    assert {rec["source"] for rec in lines} == {"Feed A", "Feed B"}
    assert {d.name for d in (tmp_path / "corpus").iterdir()} == {"feed-a", "feed-b"}


def test_cli_gather_feeds_missing_config_errors(capsys, tmp_path):
    rc = cli.main(["gather", "feeds", "--config", str(tmp_path / "nope.toml")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <title>A development worth watching in the science landscape</title>
    <link href="http://arxiv.org/abs/2606.00001v1" rel="alternate" type="text/html"/>
    <summary>A new result on emerging methods that could matter across fields.</summary>
    <published>2026-06-01T00:00:00Z</published>
  </entry>
</feed>
"""


def test_cli_gather_arxiv_requires_query(capsys):
    rc = cli.main(["gather", "arxiv"])
    assert rc == 1
    assert "--query" in capsys.readouterr().err


def test_cli_gather_arxiv_lands_corpus(monkeypatch, capsys, tmp_path):
    pytest.importorskip("feedparser")
    monkeypatch.chdir(tmp_path)
    feed = tmp_path / "arxiv.xml"
    feed.write_text(_ARXIV_XML, encoding="utf-8")
    # Point the query URL at the local fixture — no network.
    monkeypatch.setattr("horizon.ingest.arxiv.ArxivSource._url", lambda self: str(feed))

    rc = cli.main(["gather", "arxiv", "--query", "cat:cs.CL"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 new" in out
    manifest = (
        (tmp_path / "corpus" / "arxiv" / "manifest.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )
    rec = json.loads(manifest[0])
    assert rec["source"] == "arxiv"
    assert rec["url"] == "http://arxiv.org/abs/2606.00001v1"
    assert rec["published_at"] == "2026-06-01"


_URL_HTML = (
    "<html><body><nav>menu</nav>"
    "<article><p>A development worth watching in the tech landscape.</p></article>"
    "</body></html>"
)


def test_cli_gather_url_requires_input(capsys):
    rc = cli.main(["gather", "url"])
    assert rc == 1
    assert "--url" in capsys.readouterr().err


def test_cli_gather_url_missing_file_errors(capsys, tmp_path):
    rc = cli.main(["gather", "url", "--url-file", str(tmp_path / "nope.txt")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_cli_gather_url_lands_corpus(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    from horizon.ingest.url import UrlSource

    url = "http://x.example/story"
    # Serve the page; robots.txt (any other url) -> None -> allowed.
    monkeypatch.setattr(
        UrlSource, "_get", lambda self, u: _URL_HTML if u == url else None
    )

    rc = cli.main(["gather", "url", "--url", url])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 new" in out
    # source = domain -> corpus/x-example/.
    manifest = (
        (tmp_path / "corpus" / "x-example" / "manifest.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )
    rec = json.loads(manifest[0])
    assert rec["source"] == "x.example"
    assert rec["url"] == url


# --- data-dir rooting + raw retention + corpus replay -------------------------


def test_cli_gather_data_dir_roots_corpus_and_raw(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    from horizon.ingest.url import UrlSource

    url = "http://x.example/story"
    monkeypatch.setattr(
        UrlSource, "_get", lambda self, u: _URL_HTML if u == url else None
    )

    rc = cli.main(["gather", "url", "--url", url, "--data-dir", "datasets/tech"])
    assert rc == 0
    root = tmp_path / "datasets" / "tech"
    assert (root / "corpus" / "x-example" / "manifest.jsonl").is_file()
    assert list((root / "raw").glob("*/*.html"))  # raw HTML retained per source
    assert not (tmp_path / "corpus").exists()  # nothing landed at the default root


def test_cli_gather_no_raw_skips_raw(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    from horizon.ingest.url import UrlSource

    url = "http://x.example/story"
    monkeypatch.setattr(
        UrlSource, "_get", lambda self, u: _URL_HTML if u == url else None
    )

    rc = cli.main(["gather", "url", "--url", url, "--data-dir", "d", "--no-raw"])
    assert rc == 0
    assert not (tmp_path / "d" / "raw").exists()


def test_cli_scan_and_stats_respect_data_dir(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development worth watching", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    assert cli.main(["scan", str(text_file), "--data-dir", "datasets/tech"]) == 0
    assert (tmp_path / "datasets" / "tech" / "data" / "analyses.jsonl").is_file()
    assert not (tmp_path / "data").exists()  # not written to the default root
    capsys.readouterr()
    rc = cli.main(["dataset-stats", "--data-dir", "datasets/tech"])
    assert rc == 0
    assert "Dataset readiness" in capsys.readouterr().out


def test_cli_trends_embedding_cache_respects_data_dir(
    monkeypatch, capsys, tmp_path, make_embedder
):
    """The vector sidecar lands under --data-dir, not the current dir.

    A full-corpus run writes one vector per analysis; a cwd-relative sidecar would
    accumulate outside the dataset it describes (and fill $HOME on a cluster).
    """
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "datasets" / "tech"
    store = JsonlStore(root / "data" / "analyses.jsonl")
    for i in range(2):
        store.put(
            Analysis.from_dict(
                _provenance(content_hash=f"h{i}", summary=f"open model release {i}")
            )
        )
    embedder = make_embedder(lambda t: [1.0, 0.0])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder", lambda *a, **k: embedder
    )

    rc = cli.main(["trends", "--json", "--data-dir", "datasets/tech"])
    assert rc == 0
    assert (root / "data" / "embeddings.jsonl").is_file()
    assert not (tmp_path / "data").exists()  # nothing written to the default root


def test_cli_data_dir_from_env(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HORIZON_DATA_DIR", "datasets/policy")
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development worth watching", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    assert cli.main(["scan", str(text_file)]) == 0
    assert (tmp_path / "datasets" / "policy" / "data" / "analyses.jsonl").is_file()


def test_cli_gather_corpus_requires_path(capsys):
    rc = cli.main(["gather", "corpus"])
    assert rc == 1
    assert "--path" in capsys.readouterr().err


def test_cli_corpus_stats_by_source(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    from horizon.ingest import RawItem
    from horizon.ingest.corpus import write_corpus

    corpus = tmp_path / "datasets" / "tech" / "corpus"
    write_corpus(
        [
            RawItem(text="a development worth watching in the markets", source="Wired"),
            RawItem(
                text="another tech signal worth tracking closely now", source="Wired"
            ),
            RawItem(
                text="Через масовану атаку можливі перебої в роботі", source="AIN.ua"
            ),
        ],
        corpus,
    )

    rc = cli.main(["corpus-stats", "--data-dir", "datasets/tech", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 3
    assert payload["by_source"]["Wired"]["count"] == 2
    assert payload["by_source"]["AIN.ua"]["count"] == 1
    # Language mix is model-independent (read from the landed text).
    assert payload["by_source"]["AIN.ua"]["scripts"] == {"cyrillic": 1}
    assert payload["scripts"] == {"latin": 2, "cyrillic": 1}


def test_cli_corpus_stats_missing_corpus_errors(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["corpus-stats", "--data-dir", "datasets/nope"])
    assert rc == 1
    assert "no corpus" in capsys.readouterr().err.lower()


def test_cli_gather_corpus_replay_and_scan(monkeypatch, capsys, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "a development worth watching in ukraine", encoding="utf-8"
    )
    # 1) gather into a per-topic dataset (no scan, no model).
    assert (
        cli.main(
            ["gather", "local", "--path", str(docs), "--data-dir", "datasets/tech"]
        )
        == 0
    )
    capsys.readouterr()
    corpus = tmp_path / "datasets" / "tech" / "corpus"

    # 2) replay the landed corpus offline and scan it into the same dataset.
    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    rc = cli.main(
        [
            "gather",
            "corpus",
            "--path",
            str(corpus),
            "--scan",
            "--data-dir",
            "datasets/tech",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Scanned 1 new" in out
    store = JsonlStore(tmp_path / "datasets" / "tech" / "data" / "analyses.jsonl")
    analyses = store.list_all()
    assert len(analyses) == 1
    assert analyses[0].origin == "local"  # ingestion provenance replayed from manifest


# --- gather --scan (C5: publication date -> item_date) ------------------------


class _FakeSrc:
    """A Source yielding canned RawItems — no network, no feedparser."""

    def __init__(self, items):
        self._items = items

    def fetch(self):
        return list(self._items)


def _fake_items():
    from horizon.ingest import RawItem

    return [
        RawItem(
            text="a development worth watching in the markets landscape",
            source="feedX",
            url="http://x/1",
            published_at="2026-06-01",
        ),
        RawItem(  # no publication date -> falls back to scan time
            text="another signal worth tracking across the policy landscape",
            source="feedX",
            url="http://x/2",
            published_at=None,
        ),
    ]


def test_cli_gather_scan_stamps_published_date(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _fake_items()
    monkeypatch.setattr(
        cli, "_gather_sources", lambda args: [("feedX", _FakeSrc(items))]
    )
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["gather", "rss", "--url", "x", "--scan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 new" in out  # landed into corpus/
    assert "Scanned 2 new" in out

    stored = JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()
    by_hash = {a.content_hash: a for a in stored}
    dated = by_hash[content_hash(items[0].text)]
    assert dated.item_date == "2026-06-01"  # C5: real publication date
    # §4: ingestion provenance recorded without touching the training `source` axis.
    assert dated.origin == "feedX" and dated.url == "http://x/1"
    assert dated.source == "model"
    # Missing published_at -> scan-time fallback (an ISO timestamp, not "2026-06-01").
    fallback = by_hash[content_hash(items[1].text)]
    assert fallback.item_date and fallback.item_date != "2026-06-01"


def test_cli_gather_scan_is_idempotent(monkeypatch, capsys, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    items = _fake_items()
    monkeypatch.setattr(
        cli, "_gather_sources", lambda args: [("feedX", _FakeSrc(items))]
    )
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    capsys.readouterr()
    # Second run: corpus + store already hold everything — nothing new scanned.
    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    out = capsys.readouterr().out
    assert "Scanned 0 new, 2 already in store" in out
    # No duplicate analyses accreted in the store.
    assert len(JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()) == 2


def test_cli_compare_auto_detects_sides(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    # Same text scanned twice: stand-in then teacher (overwrites on read, both on disk).
    store.put(Analysis.from_dict(_provenance(content_hash="h1", model_id="Qwen/x")))
    store.put(
        Analysis.from_dict(
            _provenance(
                content_hash="h1",
                model_id="gemini-3.5-flash",
                classification={
                    "theme": "open models",
                    "maturity": "growing",  # teacher disagrees with the stand-in
                    "impact": "high",
                    "time_horizon": "near",
                },
            )
        )
    )

    rc = cli.main(["compare", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_model"] == "Qwen/x"
    assert payload["teacher_model"] == "gemini-3.5-flash"
    assert payload["n_paired"] == 1
    assert payload["enum_agreement"]["maturity"] == 0.0  # emerging vs growing
    assert payload["enum_agreement"]["impact"] == 1.0


def test_cli_compare_needs_two_distinguishable_models(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    store.put(Analysis.from_dict(_provenance(content_hash="h1", model_id="Qwen/x")))

    rc = cli.main(["compare"])
    assert rc == 1
    assert "could not tell the two scans apart" in capsys.readouterr().err


def test_cli_gather_rescan_reanalyses_and_overwrites_labels(
    monkeypatch, capsys, tmp_path, make_client
):
    # Teacher pass: re-scan the same items with a stronger (API) model, replacing
    # the earlier stand-in labels in place — same store keys, no duplicates.
    monkeypatch.chdir(tmp_path)
    items = _fake_items()
    monkeypatch.setattr(
        cli, "_gather_sources", lambda args: [("feedX", _FakeSrc(items))]
    )
    # A gemini-* id routes through make_client to GeminiClient; patch make_client
    # itself so no SDK/key is needed and every call returns canned JSON.
    monkeypatch.setattr(
        "horizon.llm.make_client", lambda settings=None: make_client(COMBINED_JSON)
    )

    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    capsys.readouterr()
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    first = store.list_all()
    assert len(first) == 2
    assert all(a.model_id != "gemini-3.5-flash" for a in first)

    # Teacher re-scan: same items, --rescan overwrites; model id now the teacher's.
    rc = cli.main(
        [
            "gather",
            "rss",
            "--url",
            "x",
            "--scan",
            "--rescan",
            "--model",
            "gemini-3.5-flash",
        ]
    )
    assert rc == 0
    assert "Scanned 2 new" in capsys.readouterr().out
    after = store.list_all()
    assert len(after) == 2  # overwritten, not duplicated
    assert all(a.model_id == "gemini-3.5-flash" for a in after)


def test_cli_gather_local_scan_dates_by_mtime(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("a development worth watching now", encoding="utf-8")
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["gather", "local", "--path", str(docs), "--scan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Scanned 1 new" in out
    stored = JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()
    assert len(stored) == 1
    assert stored[0].item_date  # LocalDirSource dates by file mtime


# --- attention: from meta into the store --------------------------------------


def _rated_items():
    from horizon.ingest import RawItem

    return [
        RawItem(
            text="a development worth watching in the markets landscape",
            source="wos-scopus-2019",
            url="http://x/1",
            published_at="2019-01-01",
            meta={"attention": 0.87, "cited_max": 12},
        ),
        RawItem(  # no attention signal at all — most sources give none
            text="another signal worth tracking across the policy landscape",
            source="feedX",
            url="http://x/2",
            published_at="2026-06-01",
            meta={},
        ),
    ]


def test_cli_gather_scan_records_the_attention_signal(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _rated_items()
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("wos", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    by_hash = {
        a.content_hash: a
        for a in JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()
    }
    assert by_hash[content_hash(items[0].text)].attention == 0.87
    assert by_hash[content_hash(items[1].text)].attention is None


def test_cli_gather_scan_refreshes_a_stale_attention_without_a_model(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _rated_items()
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("wos", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    capsys.readouterr()

    # A rebuild moves every cohort-relative percentile. Re-running must re-stamp the
    # store from meta — and must not load a model to do it.
    items[0] = replace(items[0], meta={"attention": 0.42})
    fresh = make_client(COMBINED_JSON)
    _patch_model(monkeypatch, fresh)
    assert cli.main(["gather", "rss", "--url", "x", "--scan"]) == 0
    out = capsys.readouterr().out

    assert "1 attention refreshed" in out
    by_hash = {
        a.content_hash: a
        for a in JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()
    }
    assert by_hash[content_hash(items[0].text)].attention == 0.42
    assert fresh.calls == []  # the whole point: no generation on the refresh path


# --- trends flags -------------------------------------------------------------


def _year_store(tmp_path, years):
    """Seed a store with one analysis per entry in ``years`` (a list of ints)."""
    store = JsonlStore(tmp_path / "data" / "analyses.jsonl")
    for i, year in enumerate(years):
        store.put(
            Analysis.from_dict(
                _provenance(content_hash=f"h{i}", item_date=f"{year}-01-01")
            )
        )
    return store


def test_cli_trends_anchors_to_the_newest_item(
    monkeypatch, capsys, tmp_path, make_embedder
):
    monkeypatch.chdir(tmp_path)
    _year_store(tmp_path, [2019, 2019])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder",
        lambda *a, **k: make_embedder(),
    )
    assert cli.main(["trends", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # Years-old items used to score 0.0 against a wall-clock anchor.
    assert payload[0]["velocity"] > 0
    assert payload[0]["score"] > 0
    assert payload[0]["attention"] == 1.0
    assert payload[0]["attention_n"] == 0


def test_cli_trends_header_only_in_human_mode(
    monkeypatch, capsys, tmp_path, make_embedder
):
    monkeypatch.chdir(tmp_path)
    _year_store(tmp_path, [2024, 2025])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder",
        lambda *a, **k: make_embedder(),
    )
    assert cli.main(["trends", "--window", "2y", "--recency-scale", "365"]) == 0
    human = capsys.readouterr().out
    assert "Trends as of 2025-01-01" in human
    assert "window 2y" in human
    assert "recency scale 365d" in human

    assert cli.main(["trends", "--json"]) == 0
    # JSON stays a bare list so existing consumers can index payload[0].
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_cli_trends_now_flag_reanchors(monkeypatch, capsys, tmp_path, make_embedder):
    monkeypatch.chdir(tmp_path)
    _year_store(tmp_path, [2024, 2025])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder",
        lambda *a, **k: make_embedder(),
    )
    assert cli.main(["trends", "--now", "2024", "--window", "1y", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # Anchored at 2024, the 2025 item must not count as recent (the D2 clamp).
    assert payload[0]["velocity"] == 1.0


@pytest.mark.parametrize(
    "flags", [["--window", "2m"], ["--now", "nonsense"], ["--recency-scale", "0"]]
)
def test_cli_trends_rejects_bad_scoring_flags(
    monkeypatch, capsys, tmp_path, make_embedder, flags
):
    monkeypatch.chdir(tmp_path)
    _year_store(tmp_path, [2025])
    monkeypatch.setattr(
        "horizon.embeddings.SentenceTransformerEmbedder",
        lambda *a, **k: make_embedder(),
    )
    assert cli.main(["trends", *flags]) == 1
    assert "error:" in capsys.readouterr().err


# --- gather --scan: batching, progress, manifest, circuit breaker -------------


def _n_items(n):
    from horizon.ingest import RawItem

    return [
        RawItem(
            text=f"signal number {i} worth watching in this landscape",
            source="feedX",
            url=f"http://x/{i}",
            published_at="2026-06-01",
        )
        for i in range(n)
    ]


def test_cli_gather_scan_batches_model_calls(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(5)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    fake = make_client(COMBINED_JSON)
    _patch_model(monkeypatch, fake)

    rc = cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "2"])
    assert rc == 0
    assert fake.batches == [2, 2, 1]  # chunked, nothing dropped
    assert "Scanned 5 new" in capsys.readouterr().out
    assert len(JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()) == 5


def test_cli_gather_scan_stays_idempotent_with_batching(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(4)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    assert cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "2"]) == 0
    capsys.readouterr()

    second = make_client(COMBINED_JSON)
    _patch_model(monkeypatch, second)
    assert cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "2"]) == 0
    assert "Scanned 0 new, 4 already in store" in capsys.readouterr().out
    assert second.calls == []  # nothing re-analysed


def test_cli_gather_scan_progress_goes_to_stderr(
    monkeypatch, capsys, tmp_path, make_client
):
    """stdout stays machine-readable; progress must not pollute the summary."""
    monkeypatch.chdir(tmp_path)
    items = _n_items(4)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(
        [
            "gather",
            "rss",
            "--url",
            "x",
            "--scan",
            "--batch-size",
            "1",
            "--progress-every",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "/4" in captured.err and "it/s" in captured.err
    assert "eta" in captured.err
    assert "it/s" not in captured.out  # summary line untouched
    assert "Scanned 4 new" in captured.out


def test_cli_gather_scan_progress_can_be_disabled(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(3)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    cli.main(["gather", "rss", "--url", "x", "--scan", "--progress-every", "0"])
    assert "it/s" not in capsys.readouterr().err


def test_cli_gather_scan_writes_a_run_manifest(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(2)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(
        [
            "gather",
            "rss",
            "--url",
            "x",
            "--scan",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    runs = sorted((tmp_path / "data" / "runs").glob("*.json"))
    assert len(runs) == 1
    manifest = json.loads(runs[0].read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "scanned": 2,
        "skipped": 0,
        "refreshed": 0,
        "failed": 0,
    }
    assert manifest["device"] == "cpu"
    assert manifest["batch_size"] == 2
    assert manifest["prompt_version"] == PROMPT_VERSION
    assert manifest["items"] == 2
    assert manifest["aborted"] is None
    assert "torch" in manifest["versions"]
    assert manifest["run_id"] == runs[0].stem


def test_cli_gather_scan_circuit_breaker_stops_a_doomed_run(
    monkeypatch, capsys, tmp_path, make_client
):
    """An OOM fragments the allocator, so every later item fails too — stop, don't grind."""
    monkeypatch.chdir(tmp_path)
    items = _n_items(40)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    fake = make_client("not json at all")  # every item fails validation, twice
    _patch_model(monkeypatch, fake)

    rc = cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "1"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "consecutive failures" in captured.err
    # Stopped at the limit rather than burning through all 40 items. Each item costs
    # an attempt plus one repair, so the breaker trips after LIMIT items / 2xLIMIT calls.
    limit = cli._GatheredScanner.FAILURE_LIMIT
    first_attempts = [
        c[-1].content for c in fake.calls if "could not be used" not in c[-1].content
    ]
    assert len(first_attempts) == limit < len(items)
    assert len(fake.batches) == limit * 2  # one attempt + one repair batch per item


def test_cli_gather_scan_reports_a_whole_batch_failure(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(2)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])

    class _Exploding:
        def generate(self, messages, *, config=None):
            raise RuntimeError("CUDA out of memory")

        def generate_batch(self, conversations, *, config=None):
            raise RuntimeError("CUDA out of memory")

        def complete_batch(self, prompts, *, system=None, config=None):
            raise RuntimeError("CUDA out of memory")

    _patch_model(monkeypatch, _Exploding())
    rc = cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "2"])
    captured = capsys.readouterr()
    assert rc == 0  # 2 failures is under the breaker limit
    assert "batch of 2 failed" in captured.err
    assert "2 failed" in captured.out


def test_cli_gather_scan_survives_a_keyboard_interrupt(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(4)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])

    calls = {"n": 0}

    def reply(messages):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return COMBINED_JSON

    _patch_model(monkeypatch, make_client(reply))
    rc = cli.main(["gather", "rss", "--url", "x", "--scan", "--batch-size", "1"])
    captured = capsys.readouterr()
    assert rc == 130
    assert "Interrupted" in captured.err
    # Everything finished before the interrupt is durable.
    assert len(JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()) == 2


def test_cli_gather_no_corpus_skips_corpus_records(
    monkeypatch, capsys, tmp_path, make_client
):
    monkeypatch.chdir(tmp_path)
    items = _n_items(3)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))

    rc = cli.main(["gather", "rss", "--url", "x", "--scan", "--no-corpus"])
    assert rc == 0
    assert not (tmp_path / "corpus").exists()  # no per-item JSON files
    assert len(JsonlStore(tmp_path / "data" / "analyses.jsonl").list_all()) == 3


def test_cli_model_flags_thread_into_settings(monkeypatch, tmp_path, make_client):
    """Every model flag must reach Settings — asserted without loading a model."""
    monkeypatch.chdir(tmp_path)
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development worth watching", encoding="utf-8")
    seen = {}

    def capture(settings):
        seen["settings"] = settings
        return make_client(COMBINED_JSON)

    monkeypatch.setattr("horizon.llm.make_client", capture)
    rc = cli.main(
        [
            "scan",
            str(text_file),
            "--no-save",
            "--device",
            "cuda:1",
            "--dtype",
            "float16",
            "--load-4bit",
            "--device-map",
            "auto",
            "--max-memory",
            "0=38GiB",
            "--attn",
            "sdpa",
            "--batch-size",
            "8",
            "--max-input-tokens",
            "1024",
            "--max-new-tokens",
            "256",
            "--temperature",
            "0.3",
        ]
    )
    assert rc == 0
    s = seen["settings"]
    assert s.device == "cuda:1"
    assert s.dtype == "float16"
    assert s.load_in_4bit is True
    assert s.device_map == "auto"
    assert s.max_memory == "0=38GiB"
    assert s.attn_implementation == "sdpa"
    assert s.batch_size == 8
    assert s.max_input_tokens == 1024
    assert s.generation.max_new_tokens == 256
    assert s.generation.temperature == 0.3


def test_cli_absent_flags_do_not_override_env(monkeypatch, tmp_path, make_client):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HORIZON_DEVICE", "cuda")
    monkeypatch.setenv("HORIZON_BATCH_SIZE", "4")
    text_file = tmp_path / "input.txt"
    text_file.write_text("a development worth watching", encoding="utf-8")
    seen = {}

    def capture(settings):
        seen["settings"] = settings
        return make_client(COMBINED_JSON)

    monkeypatch.setattr("horizon.llm.make_client", capture)
    assert cli.main(["scan", str(text_file), "--no-save"]) == 0
    assert seen["settings"].device == "cuda"
    assert seen["settings"].batch_size == 4


# --- trends velocity baseline -------------------------------------------------


def _records_shard(tmp_path, year, rows):
    """Write one built record shard, as collection-build would."""
    d = tmp_path / "ds" / "records"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{year}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path / "ds"


def test_velocity_baseline_accepts_the_dataset_root_or_the_records_dir(tmp_path):
    """Both spellings of --records work: the ambiguity has cost a run already."""
    from horizon.cli import _velocity_baseline

    root = _records_shard(
        tmp_path, 2020, [{"year": 2020, "subject_areas": ["Physics"]}] * 3
    )
    for path in (root, root / "records"):
        args = argparse.Namespace(records=path, baseline_field=None)
        assert _velocity_baseline(args) == {2020: 3}


def test_velocity_baseline_is_none_without_records():
    from horizon.cli import _velocity_baseline

    assert _velocity_baseline(argparse.Namespace(records=None)) is None


def test_velocity_baseline_can_select_one_field(tmp_path):
    from horizon.cli import _velocity_baseline

    root = _records_shard(
        tmp_path,
        2020,
        [
            {"year": 2020, "subject_areas": ["Physics"]},
            {"year": 2020, "subject_areas": ["Physics", "Chemistry"]},
            {"year": 2020, "subject_areas": ["Chemistry"]},
        ],
    )
    args = argparse.Namespace(records=root, baseline_field="Physics")
    assert _velocity_baseline(args) == {2020: 2}


def test_velocity_baseline_rejects_an_unknown_field(tmp_path, capsys):
    """Silently normalising by nothing would be worse than refusing."""
    from horizon.cli import _velocity_baseline

    root = _records_shard(
        tmp_path, 2020, [{"year": 2020, "subject_areas": ["Physics"]}]
    )
    args = argparse.Namespace(records=root, baseline_field="Astrology")
    with pytest.raises(SystemExit):
        _velocity_baseline(args)
    assert "Astrology" in capsys.readouterr().err


def test_velocity_baseline_rejects_a_path_with_no_shards(tmp_path, capsys):
    from horizon.cli import _velocity_baseline

    args = argparse.Namespace(records=tmp_path, baseline_field=None)
    with pytest.raises(SystemExit):
        _velocity_baseline(args)
    assert "collection-build" in capsys.readouterr().err


# --- horizon fields -----------------------------------------------------------


def test_fields_reports_development_and_pairs(tmp_path, capsys):
    root = _records_shard(
        tmp_path,
        2020,
        [
            {"year": 2020, "subject_areas": ["Physics"]},
            {"year": 2020, "subject_areas": ["Physics", "Chemistry"]},
        ],
    )
    _records_shard(tmp_path, 2021, [{"year": 2021, "subject_areas": ["Chemistry"]}])
    assert cli.main(["fields", "--records", str(root)]) == 0
    out = capsys.readouterr().out
    assert "3 records, 2020-2021" in out
    assert "Physics" in out and "Chemistry + Physics" in out
    assert "not of world science" in out


def test_fields_json_carries_counts_and_shares(tmp_path, capsys):
    root = _records_shard(
        tmp_path, 2020, [{"year": 2020, "subject_areas": ["Physics"]}] * 2
    )
    assert cli.main(["fields", "--records", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2 and payload["year_totals"] == {"2020": 2}
    (physics,) = payload["fields"]
    assert physics["name"] == "Physics" and physics["counts"] == {"2020": 2}


def test_fields_honours_the_years_spec(tmp_path, capsys):
    root = _records_shard(tmp_path, 2020, [{"year": 2020, "subject_areas": ["A"]}])
    _records_shard(tmp_path, 2021, [{"year": 2021, "subject_areas": ["B"]}])
    assert cli.main(["fields", "--records", str(root), "--years", "2021"]) == 0
    out = capsys.readouterr().out
    assert "1 records, 2021-2021" in out and "\n  A " not in out


def test_fields_rejects_a_bad_year_spec(tmp_path, capsys):
    root = _records_shard(tmp_path, 2020, [{"year": 2020, "subject_areas": ["A"]}])
    assert cli.main(["fields", "--records", str(root), "--years", "2021-2019"]) == 1
    assert "bad year range" in capsys.readouterr().err


def test_fields_rejects_a_path_without_shards(tmp_path, capsys):
    assert cli.main(["fields", "--records", str(tmp_path)]) == 1
    assert "collection-build" in capsys.readouterr().err


def test_progress_rate_ignores_skipped_items(
    monkeypatch, capsys, tmp_path, make_client
):
    """A resumed shard is mostly skips, and a skip costs a dict lookup. Counting them in
    the rate makes the shard look several times faster than it is and understates the
    wait by hours across twelve of them."""
    monkeypatch.chdir(tmp_path)
    items = _n_items(4)
    monkeypatch.setattr(cli, "_gather_sources", lambda args: [("f", _FakeSrc(items))])
    _patch_model(monkeypatch, make_client(COMBINED_JSON))
    argv = [
        "gather",
        "rss",
        "--url",
        "x",
        "--scan",
        "--batch-size",
        "1",
        "--progress-every",
        "1",
    ]

    assert cli.main(argv) == 0
    first = capsys.readouterr().err

    # Second pass over the same items: everything skips, nothing is worked.
    assert cli.main(argv) == 0
    second = capsys.readouterr().err
    assert "4 skipped" in second
    # No work done means no basis for a rate, so it must not inherit the first pass's.
    assert "0.00 it/s" in second
    assert "it/s" in first


# --- report: cross-block merge ------------------------------------------------


def test_medoid_returns_an_actual_member_not_an_average():
    """The distinction is load-bearing: averaging cancels the topic-specific directions
    and leaves what every document shares, so centroids collapse toward each other and
    one threshold stops meaning one thing across the block boundary."""
    import numpy as np

    from horizon.cli import _medoid

    vectors = [
        np.array([1.0, 0.0]),
        np.array([0.9, 0.44]),  # closest to the mean of the three
        np.array([0.8, 0.6]),
    ]
    got = _medoid(vectors, [0, 1, 2])
    assert any(np.allclose(got, v) for v in vectors)
    assert np.allclose(got, vectors[1])


def test_medoid_of_a_single_member_is_that_member():
    import numpy as np

    from horizon.cli import _medoid

    v = np.array([0.6, 0.8])
    assert np.allclose(_medoid([v], [0]), v)


def test_fixed_clusterer_hands_back_the_labels_it_was_given():
    """build_trends clusters as part of scoring; levels 3 and 4 decide membership first
    and the scoring pass has to accept that rather than redo it."""
    from horizon.cli import _FixedClusterer

    labels = [0, 0, 1]
    assert _FixedClusterer(labels).cluster([[1.0], [1.0], [0.0]]) == labels


# --- the velocity window a corpus needs ---------------------------------------


def _dated(*dates):
    """Analyses carrying only the field the window rule reads."""
    from horizon.analysis import Analysis, Classification, Entities

    return [
        Analysis(
            summary="s",
            signals=(),
            classification=Classification(
                theme="t", maturity="emerging", impact="low", time_horizon="near"
            ),
            entities=Entities(),
            keywords=(),
            content_hash=f"h{i}",
            model_id="m",
            prompt_version="v",
            strategy="combined",
            generated_at=d,
            item_date=d,
        )
        for i, d in enumerate(dates)
    ]


def test_a_year_dated_corpus_gets_a_year_window_without_being_asked():
    """7d against January-1st stamps compares a whole cohort with nothing at all.

    The failure is silent — velocity fills with ratios against zero — so the rule is
    detected from the dates rather than left to whoever writes the command line.
    """
    from dataclasses import replace

    from horizon.cli import _default_window
    from horizon.config import Settings

    settings = replace(Settings.from_env(), velocity_window="7d")
    assert _default_window(_dated("2015-01-01", "2016-01-01"), settings) == "2y"


def test_a_finer_dated_corpus_keeps_the_configured_window():
    from dataclasses import replace

    from horizon.cli import _default_window
    from horizon.config import Settings

    settings = replace(Settings.from_env(), velocity_window="7d")
    assert _default_window(_dated("2015-01-01", "2016-04-17"), settings) == "7d"


def test_an_empty_store_keeps_the_configured_window():
    from dataclasses import replace

    from horizon.cli import _default_window
    from horizon.config import Settings

    settings = replace(Settings.from_env(), velocity_window="7d")
    assert _default_window([], settings) == "7d"


# --- the partial final cohort -------------------------------------------------


def _tab_with(counts):
    """A tabulation carrying only the cohort sizes the rule reads."""
    from horizon.fields import tabulate

    records = [
        {"year": year, "subject_areas": ["Physics"]}
        for year, n in counts.items()
        for _ in range(n)
    ]
    return tabulate(records)


def test_a_stub_final_cohort_is_dropped_from_the_reported_years(capsys):
    """A corpus gathered mid-year ends in a stub, and every fitted rate leans on it.

    Measured on the real corpus before this guard: the largest topic reported +92
    publications a year against several hundred over the complete cohorts.
    """
    from horizon.cli import _complete_years

    years = _complete_years(_tab_with({2020: 1000, 2021: 1100, 2022: 900, 2023: 20}))
    assert years == {2020, 2021, 2022}
    assert "dropping 2023" in capsys.readouterr().out


def test_a_merely_small_final_cohort_is_kept():
    """The threshold is blunt on purpose — this is not a borderline judgement."""
    from horizon.cli import _complete_years

    assert (
        _complete_years(_tab_with({2020: 1000, 2021: 1100, 2022: 900, 2023: 600}))
        is None
    )


def test_too_few_cohorts_to_judge_leaves_the_years_alone():
    from horizon.cli import _complete_years

    assert _complete_years(_tab_with({2020: 1000, 2021: 5})) is None


# --- rendered copies ----------------------------------------------------------


def test_pandoc_is_given_the_report_directory_as_its_resource_path(
    monkeypatch, tmp_path
):
    """Images are written relative to the report; pandoc resolves relative to the cwd.

    Without this the figures are silently absent from the PDF: the build succeeds, the
    page count is unchanged, and only pdfimages shows that nothing was embedded.
    """
    import subprocess

    from horizon import cli

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    md = tmp_path / "reports" / "r-ua.md"
    md.parent.mkdir()
    md.write_text("x", encoding="utf-8")
    cli._pandoc(md, md.with_suffix(".pdf"), pdf=True)
    assert f"--resource-path={md.parent}" in seen["cmd"]


def test_a_pandoc_warning_on_success_is_printed_not_swallowed(monkeypatch, tmp_path):
    """A report shipped with seven figure references and no figures because this was
    discarded — pandoc had said so."""
    import subprocess

    from horizon import cli

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, "", "[WARNING] Could not fetch resource figures/f1.png\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    md = tmp_path / "r-ua.md"
    md.write_text("x", encoding="utf-8")
    out = cli._pandoc(md, md.with_suffix(".pdf"), pdf=True)
    assert "Could not fetch resource" in out


def test_pass_all_selects_every_reading_and_a_named_pass_selects_only_it():
    """The first version built {"all", asked}, which contains "all" whichever pass was
    asked for — so every gate passed and --pass selected nothing."""
    from horizon.cli import _passes

    assert _passes("all") == {"corpus", "topics", "areas"}
    assert _passes("areas") == {"areas"}
    assert _passes("corpus") == {"corpus"}


def test_the_reading_filename_carries_the_quantization():
    """The same weights at 4-bit and 8-bit are different generators. Sharing a filename,
    the second run would open the first one's file, find every topic present and write
    nothing."""
    from dataclasses import replace

    from horizon.cli import _reading_identity
    from horizon.config import Settings

    base = replace(Settings(), model_id="google/gemma-4-12B-it")
    assert _reading_identity(replace(base, load_in_4bit=True)) == (
        "gemma-4-12B-it-4bit",
        "google/gemma-4-12B-it (4bit)",
    )
    assert _reading_identity(replace(base, load_in_8bit=True))[0] == (
        "gemma-4-12B-it-8bit"
    )
    assert _reading_identity(base) == ("gemma-4-12B-it", "google/gemma-4-12B-it")


# --- report: re-rendering an earlier run --------------------------------------


def _facts_file(tmp_path):
    """A facts file of the shape an earlier run wrote, without the newer sections."""
    from horizon.fields import tabulate
    from horizon.report import build_facts, render_json
    from horizon.trends import Trend

    trend = Trend(
        label="a",
        members=("h",),
        size=5,
        first_seen="2020-01-01",
        last_seen="2021-01-01",
        velocity=2.0,
        score=1.0,
        top_keywords=("wear resistance",),
    )
    facts = build_facts(
        tabulate([{"year": 2020, "subject_areas": ["Physics"]}]),
        field_trends={"Physics": [trend]},
        overall=[trend],
    )
    data = json.loads(render_json(facts))
    for key in ("fit_years", "populations", "field_splits", "compositions"):
        del data[key]
    path = tmp_path / "old-facts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_report_re_renders_from_a_facts_file_without_a_corpus(tmp_path, monkeypatch):
    """The measurement is the expensive half and the tables are the cheap one.

    Re-rendering reads no records, loads no embedder and clusters nothing — which is
    also what keeps a change to the presentation from arriving with numbers that moved.
    """
    monkeypatch.setattr(cli, "_pandoc", lambda *a, **k: "skipped")
    out = tmp_path / "out"
    code = cli.main(
        [
            "report",
            "--from-facts",
            str(_facts_file(tmp_path)),
            "--out",
            str(out),
            "--slug",
            "r",
            "--date",
            "2026-09-28",
            "--no-figures",
            "--page-break-areas",
        ]
    )
    assert code == 0
    uk = (out / "r-2026-09-28-ua.md").read_text(encoding="utf-8")
    assert "### Physics" in uk and "wear resistance" in uk
    assert "\\clearpage" in uk
    assert (out / "r-2026-09-28.md").exists()
    assert (out / "r-2026-09-28-facts.json").exists()


def test_report_wants_exactly_one_of_records_and_from_facts(tmp_path, capsys):
    """Neither is a report of nothing; both is two answers to where the numbers came
    from."""
    assert cli.main(["report", "--out", str(tmp_path)]) == 1
    assert "either --records" in capsys.readouterr().err
    code = cli.main(
        [
            "report",
            "--records",
            str(tmp_path),
            "--from-facts",
            str(_facts_file(tmp_path)),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1


def test_a_flag_that_shapes_the_facts_is_refused_on_a_re_render(tmp_path, capsys):
    """Honouring it would mean re-measuring; ignoring it would mean a table that
    disagrees with the flag that asked for it."""
    code = cli.main(
        [
            "report",
            "--from-facts",
            str(_facts_file(tmp_path)),
            "--out",
            str(tmp_path),
            "--top-trends",
            "300",
        ]
    )
    assert code == 1
    assert "--top-trends" in capsys.readouterr().err
