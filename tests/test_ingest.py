import json

import pytest

from horizon.analysis import content_hash
from horizon.ingest import LocalDirSource, RawItem
from horizon.ingest.base import ATTENTION_KEY, attention_of
from horizon.ingest.corpus import GatherResult, write_corpus
from horizon.ingest.feeds import Feed, load_feeds
from horizon.ingest.normalize import MIN_LENGTH, normalize_text, strip_html


class FakeSource:
    """A Source that yields canned items — no network, no dependency."""

    def __init__(self, items):
        self._items = items

    def fetch(self):
        return list(self._items)


# --- normalization ------------------------------------------------------------


def test_strip_html_extracts_text_and_drops_script():
    html = "<p>Hello <b>world</b></p><script>evil()</script>"
    assert "Hello" in strip_html(html)
    assert "world" in strip_html(html)
    assert "evil" not in strip_html(html)


def test_strip_html_passes_through_plain_text():
    assert strip_html("just plain text") == "just plain text"


def test_normalize_collapses_whitespace_and_unescapes():
    raw = "<p>A   new\n\nmodel &amp; a signal worth watching</p>"
    out = normalize_text(raw)
    assert out == "A new model & a signal worth watching"


def test_normalize_drops_too_short():
    assert normalize_text("tiny") == ""
    assert normalize_text("x" * MIN_LENGTH) != ""


def test_normalize_matches_content_hash_whitespace_rule():
    # Normalized text hashes the same as its own content_hash (stable dedup key).
    raw = "<div>Some   development    worth   watching closely</div>"
    out = normalize_text(raw)
    assert content_hash(out) == content_hash(" ".join(out.split()))


# --- local dir source ---------------------------------------------------------


def test_local_dir_reads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text(
        "first development worth watching", encoding="utf-8"
    )
    (tmp_path / "b.md").write_text("second signal worth tracking now", encoding="utf-8")
    (tmp_path / "c.log").write_text("ignored non-text file entirely", encoding="utf-8")

    items = list(LocalDirSource(tmp_path).fetch())
    assert {i.text for i in items} == {
        "first development worth watching",
        "second signal worth tracking now",
    }
    assert all(i.source == "local" for i in items)
    assert all(i.published_at for i in items)  # dated by mtime


def test_local_dir_handles_relative_path(tmp_path, monkeypatch):
    # Regression: as_uri() needs an absolute path; a relative --path must still work.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("a development worth watching now", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    items = list(LocalDirSource("docs").fetch())
    assert len(items) == 1
    assert items[0].url.startswith("file:///")


def test_local_dir_skips_empty_and_missing_dir(tmp_path):
    (tmp_path / "empty.txt").write_text("hi", encoding="utf-8")  # too short -> dropped
    assert list(LocalDirSource(tmp_path).fetch()) == []
    with pytest.raises(NotADirectoryError):
        list(LocalDirSource(tmp_path / "nope").fetch())


# --- corpus writer ------------------------------------------------------------


def test_write_corpus_splits_by_source(tmp_path):
    from horizon.ingest.corpus import source_slug

    items = [
        RawItem(text="a development worth watching", source="local", url="file:///a"),
        RawItem(text="another signal to track here", source="feedX", url="http://x"),
    ]
    result = write_corpus(items, tmp_path)
    assert result == GatherResult(fetched=2, new=2, duplicate=0)

    # Each source landed in its own subdir with its own manifest index + <hash>.json.
    for item in items:
        sub = tmp_path / source_slug(item.source)
        manifest = (sub / "manifest.jsonl").read_text().strip().splitlines()
        assert len(manifest) == 1
        rec = json.loads(manifest[0])
        assert set(rec) == {"content_hash", "url", "source", "published_at", "lang"}
        assert rec["source"] == item.source  # real label preserved in the manifest
        record_file = sub / f"{content_hash(item.text)}.json"
        assert record_file.is_file()
        record = json.loads(record_file.read_text(encoding="utf-8"))
        assert record["text"] == item.text  # full text lives in the record now
        assert record["source"] == item.source and record["url"] == item.url


def test_write_corpus_dedupes_per_source_across_runs(tmp_path):
    item = RawItem(text="a development worth watching closely", source="local")
    write_corpus([item], tmp_path)
    result = write_corpus([item], tmp_path)  # same text again
    assert result == GatherResult(fetched=1, new=0, duplicate=1)
    manifest = tmp_path / "local" / "manifest.jsonl"
    assert len(manifest.read_text().strip().splitlines()) == 1


def test_write_corpus_retains_raw(tmp_path):
    raw = "<p>a development worth watching in the markets</p>"
    item = RawItem(
        text="a development worth watching in the markets",
        source="feedX",
        raw=raw,
        raw_suffix=".html",
    )
    raw_dir = tmp_path / "raw"
    write_corpus([item], tmp_path / "corpus", raw_dir=raw_dir)
    digest = content_hash(item.text)
    record_file = tmp_path / "corpus" / "feedx" / f"{digest}.json"
    assert record_file.is_file()
    # The record points at the retained raw file, which holds the pre-normalization body.
    record = json.loads(record_file.read_text(encoding="utf-8"))
    assert record["raw_file"] == f"{digest}.html"
    assert (raw_dir / "feedx" / f"{digest}.html").read_text(encoding="utf-8") == raw


def test_write_corpus_skips_raw_when_absent(tmp_path):
    # No raw content -> nothing written under raw/ (dir not even created); raw_file null.
    item = RawItem(text="another signal worth tracking now", source="local")
    write_corpus([item], tmp_path / "corpus", raw_dir=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()
    digest = content_hash(item.text)
    record = json.loads(
        (tmp_path / "corpus" / "local" / f"{digest}.json").read_text(encoding="utf-8")
    )
    assert record["raw_file"] is None


def test_write_corpus_persists_full_meta(tmp_path):
    # Everything a source hands us beyond the promoted fields is retained in the record.
    item = RawItem(
        text="a development worth watching in the markets landscape",
        source="feedX",
        url="http://x/1",
        meta={"title": "Big news", "author": "A. Writer", "tags": ["ai", "policy"]},
    )
    write_corpus([item], tmp_path / "corpus")
    digest = content_hash(item.text)
    record = json.loads(
        (tmp_path / "corpus" / "feedx" / f"{digest}.json").read_text(encoding="utf-8")
    )
    assert record["meta"] == {
        "title": "Big news",
        "author": "A. Writer",
        "tags": ["ai", "policy"],
    }


# --- corpus replay (CorpusSource) ---------------------------------------------


def test_corpus_source_replays_manifest(tmp_path):
    from horizon.ingest import CorpusSource

    items = [
        RawItem(
            text="a development worth watching in the markets",
            source="feedX",
            url="http://x/1",
            published_at="2026-06-01",
            lang="en",
        ),
        RawItem(
            text="another signal worth tracking in the policy space",
            source="feedY",
            url="http://x/2",
            published_at="2026-06-02",
        ),
    ]
    corpus = tmp_path / "corpus"
    write_corpus(items, corpus)

    replayed = {r.text: r for r in CorpusSource(corpus).fetch()}
    assert len(replayed) == 2
    a = replayed["a development worth watching in the markets"]
    assert a.source == "feedX" and a.url == "http://x/1"
    assert a.published_at == "2026-06-01" and a.lang == "en"


def test_corpus_source_replays_single_source_subdir(tmp_path):
    from horizon.ingest import CorpusSource
    from horizon.ingest.corpus import source_slug

    items = [
        RawItem(
            text="a development worth watching in the markets today", source="feedX"
        ),
        RawItem(
            text="another signal worth tracking in the policy space", source="feedY"
        ),
    ]
    corpus = tmp_path / "corpus"
    write_corpus(items, corpus)
    # Whole corpus root -> both sources; a single source subdir -> just that source.
    assert len(list(CorpusSource(corpus).fetch())) == 2
    only = list(CorpusSource(corpus / source_slug("feedX")).fetch())
    assert len(only) == 1 and only[0].source == "feedX"


def test_corpus_source_missing_manifest_raises(tmp_path):
    from horizon.ingest import CorpusSource

    with pytest.raises(FileNotFoundError):
        list(CorpusSource(tmp_path / "nope").fetch())


def test_corpus_source_skips_missing_text(tmp_path):
    from horizon.ingest import CorpusSource

    item = RawItem(text="a development worth watching closely today", source="s")
    corpus = tmp_path / "corpus"
    write_corpus([item], corpus)
    (
        corpus / "s" / f"{content_hash(item.text)}.json"
    ).unlink()  # manifest entry, no record
    assert list(CorpusSource(corpus).fetch()) == []


def test_corpus_source_carries_meta(tmp_path):
    from horizon.ingest import CorpusSource

    item = RawItem(
        text="a development worth watching across the markets landscape",
        source="feedX",
        meta={"title": "Headline", "tags": ["ai"]},
    )
    corpus = tmp_path / "corpus"
    write_corpus([item], corpus)
    (replayed,) = CorpusSource(corpus).fetch()
    assert replayed.meta == {"title": "Headline", "tags": ["ai"]}


def test_corpus_source_reads_legacy_txt(tmp_path):
    # Corpora written before the JSON format (bare <hash>.txt + manifest) still replay.
    from horizon.ingest import CorpusSource
    from horizon.ingest.corpus import MANIFEST_NAME, source_slug

    text = "a legacy development worth watching in the markets"
    digest = content_hash(text)
    sub = tmp_path / "corpus" / source_slug("feedX")
    sub.mkdir(parents=True)
    (sub / f"{digest}.txt").write_text(text, encoding="utf-8")
    (sub / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "content_hash": digest,
                "url": "http://x/1",
                "source": "feedX",
                "published_at": "2026-06-01",
                "lang": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (item,) = CorpusSource(tmp_path / "corpus").fetch()
    assert item.text == text and item.source == "feedX"
    assert item.published_at == "2026-06-01" and item.lang == "en"
    assert item.meta == {}  # nothing more was recorded back then


# --- feeds config -------------------------------------------------------------


def test_load_feeds_reads_tables(tmp_path):
    config = tmp_path / "feeds.toml"
    config.write_text(
        "[[feed]]\n"
        'name = "TechCrunch"\n'
        'url = "https://techcrunch.com/feed/"\n'
        "[[feed]]\n"
        'name = "The Verge"\n'
        'url = "https://www.theverge.com/rss/index.xml"\n',
        encoding="utf-8",
    )
    feeds = load_feeds(config)
    assert feeds == [
        Feed(name="TechCrunch", url="https://techcrunch.com/feed/"),
        Feed(name="The Verge", url="https://www.theverge.com/rss/index.xml"),
    ]


def test_load_feeds_skips_urlless_and_defaults_name(tmp_path):
    config = tmp_path / "feeds.toml"
    config.write_text(
        '[[feed]]\nname = "no url"\n'  # skipped: no url
        '[[feed]]\nurl = "https://x.example/feed"\n',  # name defaults to url
        encoding="utf-8",
    )
    feeds = load_feeds(config)
    assert feeds == [Feed(name="https://x.example/feed", url="https://x.example/feed")]


def test_load_feeds_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feeds(tmp_path / "nope.toml")


def test_shipped_feeds_toml_is_loadable():
    # The example config ships with the repo; keep it parseable.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    feeds = load_feeds(repo_root / "feeds.toml")
    assert len(feeds) >= 5
    assert all(f.url.startswith("http") for f in feeds)


# --- rss source (offline: feedparser parses a fixture file) -------------------

RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <language>en</language>
  <item>
    <title>Item one</title>
    <link>http://example.org/1</link>
    <description>A development worth watching in the markets landscape</description>
    <pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Item two</title>
    <link>http://example.org/2</link>
    <description>Another signal worth tracking across the policy landscape</description>
    <pubDate>Tue, 02 Jun 2026 12:00:00 GMT</pubDate>
  </item>
</channel></rss>
"""


def test_rss_source_parses_entries(tmp_path):
    pytest.importorskip("feedparser")
    from horizon.ingest import RssSource

    feed = tmp_path / "feed.xml"
    feed.write_text(RSS_FIXTURE, encoding="utf-8")

    items = list(RssSource(str(feed)).fetch())
    assert len(items) == 2
    assert items[0].source == "Test Feed"
    assert items[0].url == "http://example.org/1"
    assert items[0].published_at == "2026-06-01"
    assert items[0].lang == "en"
    assert "development worth watching" in items[0].text
    # Native fields beyond text/url/date are retained in meta.
    assert items[0].meta["title"] == "Item one"
    assert items[0].meta["feed_title"] == "Test Feed"
    assert "http://example.org/1" in items[0].meta.get("links", [])


# --- arxiv source (offline: feedparser parses an Atom fixture) ----------------

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: cat:cs.CL</title>
  <entry>
    <title>A quantum development worth watching closely</title>
    <id>http://arxiv.org/abs/2606.00001v1</id>
    <link href="http://arxiv.org/abs/2606.00001v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2606.00001v1" rel="related" type="application/pdf"/>
    <author><name>A. Researcher</name></author>
    <category term="cs.CL"/>
    <summary>We present a new approach to quantum error correction that may matter.</summary>
    <published>2026-06-01T00:00:00Z</published>
    <updated>2026-06-01T00:00:00Z</updated>
  </entry>
  <entry>
    <title>Another signal in machine learning research</title>
    <id>http://arxiv.org/abs/2606.00002v1</id>
    <link href="http://arxiv.org/abs/2606.00002v1" rel="alternate" type="text/html"/>
    <summary>A method for tracking emerging trends across large corpora of papers.</summary>
    <published>2026-06-02T00:00:00Z</published>
    <updated>2026-06-02T00:00:00Z</updated>
  </entry>
</feed>
"""


def test_arxiv_url_encodes_query():
    from horizon.ingest import ArxivSource

    url = ArxivSource("cat:cs.CL", max_results=7)._url()
    assert "search_query=cat%3Acs.CL" in url
    assert "max_results=7" in url
    assert "sortBy=submittedDate" in url and "sortOrder=descending" in url


def test_arxiv_source_parses_entries(tmp_path, monkeypatch):
    pytest.importorskip("feedparser")
    from horizon.ingest import ArxivSource

    feed = tmp_path / "arxiv.xml"
    feed.write_text(ARXIV_FIXTURE, encoding="utf-8")
    src = ArxivSource("cat:cs.CL", max_results=5)
    monkeypatch.setattr(src, "_url", lambda: str(feed))  # point fetch at the fixture

    items = list(src.fetch())
    assert len(items) == 2
    assert items[0].source == "arxiv"
    assert items[0].url == "http://arxiv.org/abs/2606.00001v1"
    assert items[0].published_at == "2026-06-01"
    assert items[0].lang == "en"
    # title + abstract both fold into the item text
    assert "quantum development worth watching" in items[0].text
    assert "error correction" in items[0].text
    # Structured fields the text blob loses are retained in meta.
    assert items[0].meta["authors"] == ["A. Researcher"]
    assert items[0].meta["categories"] == ["cs.CL"]
    assert "application/pdf" in items[0].meta.get("links", {})


# --- url source (offline: _get is stubbed; stdlib-fallback extraction) ---------

_PAGE_HTML = (
    "<html><body><nav>Home About Contact</nav>"
    "<article><p>A development worth watching in the markets landscape today.</p>"
    "</article><footer>Copyright</footer></body></html>"
)


def _stub_get(pages):
    """Return a fake ``_get(self, url)`` serving ``pages`` (dict url->text), else None."""

    def _get(self, url):
        return pages.get(url)

    return _get


def test_url_domain_helper():
    from horizon.ingest.url import _domain

    assert _domain("https://example.org/a/b?c=1") == "example.org"
    assert _domain("not-a-url") == "url"


def test_url_source_fallback_extracts_and_labels_by_domain(monkeypatch):
    from horizon.ingest import UrlSource

    url = "http://x.example/story"
    src = UrlSource([url], min_delay=0)
    # robots.txt unreachable (None) -> allowed; page returns HTML.
    monkeypatch.setattr(UrlSource, "_get", _stub_get({url: _PAGE_HTML}))

    items = list(src.fetch())
    assert len(items) == 1
    assert items[0].source == "x.example"
    assert items[0].url == url
    assert items[0].published_at is None  # no trafilatura -> no date
    assert "development worth watching" in items[0].text


def test_url_source_honors_robots_disallow(monkeypatch):
    from horizon.ingest import UrlSource

    pages = {
        "http://x.example/robots.txt": "User-agent: *\nDisallow: /private",
        "http://x.example/private/p": _PAGE_HTML,
        "http://x.example/ok": _PAGE_HTML,
    }
    src = UrlSource(["http://x.example/private/p", "http://x.example/ok"], min_delay=0)
    monkeypatch.setattr(UrlSource, "_get", _stub_get(pages))

    items = list(src.fetch())
    # /private is disallowed and skipped; /ok is fetched.
    assert [i.url for i in items] == ["http://x.example/ok"]


def test_url_source_ignore_robots_fetches_anyway(monkeypatch):
    from horizon.ingest import UrlSource

    pages = {
        "http://x.example/robots.txt": "User-agent: *\nDisallow: /",
        "http://x.example/blocked": _PAGE_HTML,
    }
    src = UrlSource(["http://x.example/blocked"], min_delay=0, obey_robots=False)
    monkeypatch.setattr(UrlSource, "_get", _stub_get(pages))

    assert len(list(src.fetch())) == 1  # robots ignored


def test_url_source_skips_unfetchable(monkeypatch):
    from horizon.ingest import UrlSource

    src = UrlSource(["http://x.example/gone"], min_delay=0)
    monkeypatch.setattr(UrlSource, "_get", _stub_get({}))  # everything -> None
    assert list(src.fetch()) == []


def test_url_source_uses_trafilatura_when_present(monkeypatch):
    pytest.importorskip("trafilatura")
    from horizon.ingest import UrlSource

    url = "http://x.example/story"
    src = UrlSource([url], min_delay=0, obey_robots=False)
    monkeypatch.setattr(UrlSource, "_get", _stub_get({url: _PAGE_HTML}))
    items = list(src.fetch())
    assert len(items) == 1
    assert "development worth watching" in items[0].text


# --- the conventional attention key -------------------------------------------


def _item(meta):
    return RawItem(text="a development worth watching", source="s", meta=meta)


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({}, None),
        (None, None),
        ({"other": 0.5}, None),
        ({ATTENTION_KEY: 0.2875}, 0.2875),
        ({ATTENTION_KEY: None}, None),
        ({ATTENTION_KEY: "0.5"}, None),  # a string is malformed, not zero
        ({ATTENTION_KEY: True}, None),  # bool is an int in Python; reject it
        ({ATTENTION_KEY: float("nan")}, None),
        ({ATTENTION_KEY: -0.5}, 0.0),  # clamped
        ({ATTENTION_KEY: 1.5}, 1.0),
    ],
)
def test_attention_of_reads_the_conventional_key(meta, expected):
    assert attention_of(_item(meta)) == expected
