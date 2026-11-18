"""Write gathered items into a ``corpus/`` directory, and replay them back.

C4 (locked slice): land normalized text to files first — inspectable and re-runnable,
keeping the fetch step separate from the model step.

The corpus is **split by source**: each item lands under ``corpus/<source-slug>/`` as a
self-contained ``<content_hash>.json`` record — the normalized ``text`` plus every scrap
of provenance the source gave us (``source``/``url``/``published_at``/``lang`` and the
open ``meta`` blob: title, author, tags, guid, …). One file travels with everything, so
an item is reusable on its own (in this or another project) without re-joining sidecars.
Each subdir also keeps a lightweight ``manifest.jsonl`` index — one line per item — used
for fast dedup and iteration without opening every record.

Keeping each source in its own folder makes per-source work trivial — count/inspect a
source, drop one (delete its folder), or reuse a subset (point :class:`CorpusSource` at a
single subdir). Dedup is **per source** (a story syndicated across two feeds is kept
under each); global dedup for *training* happens later in the store, keyed by content
hash. When a ``raw_dir`` is given, each item's pre-normalization content is mirrored to
``raw/<source-slug>/<content_hash><suffix>`` (the record's ``raw_file`` points to it) so
the corpus can be re-normalized without re-downloading.

:class:`CorpusSource` reads a landed corpus back as :class:`RawItem`\\ s (all sources, or
one subdir), so gathered text can be re-scanned (e.g. with a better model) offline.
Records written before the JSON format (a bare ``<content_hash>.txt``) are still read via
:func:`load_record`'s fallback, so older corpora keep working.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from horizon.analysis import content_hash
from horizon.ingest.base import RawItem, Source

DEFAULT_CORPUS_DIR = Path("corpus")
MANIFEST_NAME = "manifest.jsonl"

# The lightweight index fields mirrored into manifest.jsonl (the full record lives in
# each item's <hash>.json). Kept small so dedup/iteration needn't open every record.
_MANIFEST_FIELDS = ("content_hash", "url", "source", "published_at", "lang")


def source_slug(source: str) -> str:
    """Filesystem-safe folder name for a source label (real label kept in the manifest)."""
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    return slug or "unknown"


@dataclass(frozen=True)
class GatherResult:
    """Counts from one gather run, for the CLI report."""

    fetched: int
    new: int
    duplicate: int


def _existing_hashes(manifest: Path) -> set[str]:
    """The ``content_hash`` values already recorded in ``manifest`` (empty if none)."""
    if not manifest.is_file():
        return set()
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["content_hash"])
    return seen


def write_corpus(
    items: Iterable[RawItem], out_dir: Path | str, *, raw_dir: Path | str | None = None
) -> GatherResult:
    """Write new items under ``out_dir/<source-slug>/`` (deduped per source); count them.

    Each new item becomes a self-contained ``<content_hash>.json`` record (text + all
    provenance + ``meta`` + a ``raw_file`` pointer) and a one-line entry in the subdir's
    ``manifest.jsonl`` index. When ``raw_dir`` is set, the item's ``raw`` content is
    mirrored to ``raw_dir/<source-slug>/<content_hash><raw_suffix>`` (skipped when it has
    no raw), and the record's ``raw_file`` names that file.
    """
    out = Path(out_dir)
    raw_out = Path(raw_dir) if raw_dir is not None else None

    # Group by source so each subdir's manifest is opened once, even when a single
    # fetch mixes sources (e.g. UrlSource labels by domain, CorpusSource replays many).
    by_slug: dict[str, list[RawItem]] = defaultdict(list)
    for item in items:
        by_slug[source_slug(item.source)].append(item)

    fetched = new = duplicate = 0
    for slug, group in by_slug.items():
        subdir = out / slug
        subdir.mkdir(parents=True, exist_ok=True)
        manifest = subdir / MANIFEST_NAME
        seen = _existing_hashes(manifest)
        raw_sub = (raw_out / slug) if raw_out is not None else None
        with manifest.open("a", encoding="utf-8") as mf:
            for item in group:
                fetched += 1
                digest = content_hash(item.text)
                if digest in seen:
                    duplicate += 1
                    continue
                seen.add(digest)

                raw_file = None
                if raw_sub is not None and item.raw is not None:
                    raw_sub.mkdir(parents=True, exist_ok=True)
                    raw_file = f"{digest}{item.raw_suffix}"
                    (raw_sub / raw_file).write_text(item.raw, encoding="utf-8")

                record = {
                    "content_hash": digest,
                    "text": item.text,
                    "source": item.source,
                    "url": item.url,
                    "published_at": item.published_at,
                    "lang": item.lang,
                    "meta": item.meta or {},
                    "raw_file": raw_file,
                }
                (subdir / f"{digest}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                mf.write(
                    json.dumps(
                        {k: record[k] for k in _MANIFEST_FIELDS}, ensure_ascii=False
                    )
                    + "\n"
                )
                new += 1
    return GatherResult(fetched=fetched, new=new, duplicate=duplicate)


def load_record(entry: dict) -> dict | None:
    """Full item record for a manifest ``entry`` (as yielded by :func:`iter_corpus_entries`).

    Prefers the self-contained ``<content_hash>.json``; falls back to a legacy
    ``<content_hash>.txt`` (older corpora), reconstructing the record from the manifest
    fields and an empty ``meta``. Returns ``None`` when neither text file exists (a
    dangling manifest entry), so callers can skip-with-warning.
    """
    subdir = entry["_dir"]
    digest = entry["content_hash"]
    jf = subdir / f"{digest}.json"
    if jf.is_file():
        return json.loads(jf.read_text(encoding="utf-8"))
    tf = subdir / f"{digest}.txt"
    if tf.is_file():
        return {
            "content_hash": digest,
            "text": tf.read_text(encoding="utf-8"),
            "source": entry.get("source"),
            "url": entry.get("url"),
            "published_at": entry.get("published_at"),
            "lang": entry.get("lang"),
            "meta": {},
            "raw_file": None,
        }
    return None


def iter_corpus_entries(corpus_dir: Path | str) -> Iterator[dict]:
    """Yield manifest records across a corpus, tagged with their subdir.

    Accepts either a corpus root (walks every ``<source-slug>/manifest.jsonl``) or a
    single source subdir (its own ``manifest.jsonl``). Each yielded dict is the manifest
    record plus ``_dir`` (the ``Path`` holding its ``<hash>.txt``).
    """
    root = Path(corpus_dir)
    if (root / MANIFEST_NAME).is_file():
        manifests = [root / MANIFEST_NAME]
    else:
        manifests = sorted(root.glob(f"*/{MANIFEST_NAME}"))
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_dir"] = manifest.parent
                yield rec


class CorpusSource(Source):
    """Replay a landed corpus: yield a :class:`RawItem` per manifest entry.

    Point it at a corpus root to replay every source, or at one ``<source-slug>/`` subdir
    to reuse just that source. Reconstructs each :class:`RawItem` from its
    ``<content_hash>.json`` record — text and the full provenance (source, url,
    published_at, lang, and the ``meta`` blob), so gathered text can be re-scanned later
    (e.g. with a teacher model, stamping the real publication dates) with no network and
    nothing lost. A manifest entry whose record file is missing is skipped with a warning.
    """

    def __init__(self, path: Path | str) -> None:
        """Store the corpus root (or a single ``<source-slug>/`` subdir) to replay."""
        self.path = Path(path)

    def fetch(self) -> Iterable[RawItem]:
        """Reconstruct a :class:`RawItem` per manifest entry from its stored record.

        Raises ``FileNotFoundError`` if no manifest is found under the path; entries
        whose record file is missing are skipped with a warning.
        """
        has_manifest = (self.path / MANIFEST_NAME).is_file() or any(
            self.path.glob(f"*/{MANIFEST_NAME}")
        )
        if not has_manifest:
            raise FileNotFoundError(f"no corpus manifest under {self.path}")
        items: list[RawItem] = []
        for entry in iter_corpus_entries(self.path):
            rec = load_record(entry)
            if rec is None:
                print(
                    f"warning: corpus record missing: {entry['content_hash'][:12]}…",
                    file=sys.stderr,
                )
                continue
            items.append(
                RawItem(
                    text=rec["text"],
                    source=rec.get("source") or "corpus",
                    url=rec.get("url"),
                    published_at=rec.get("published_at"),
                    lang=rec.get("lang"),
                    meta=rec.get("meta") or {},
                )
            )
        return items
