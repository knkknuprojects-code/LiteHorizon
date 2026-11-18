"""Read the built ``collection/`` dataset back as gatherable items.

The other half of the two-stage collection pipeline (see
:mod:`horizon.ingest.collection`): :class:`WosSource` streams the per-year JSONL shards a
build produced and yields :class:`~horizon.ingest.base.RawItem`\\ s, so ~300 k WoS/Scopus
bibliographic records flow into the ordinary ``gather`` → ``corpus/`` → ``scan`` path with
no Excel parsing and no network.

**Partial use is the point.** ``years`` selects publication years (one, a range, a list),
``limit`` takes the first N, and ``sample``/``seed`` draws N *deterministically* — a
20-record smoke test off the same corpus a full run uses. Each item is labelled
``wos-scopus-<year>``, so the corpus lands one folder per year and
``gather corpus --path corpus/wos-scopus-2019`` replays a single year unchanged.

Item text follows the arXiv adapter's convention — title joined to abstract — with every
other field (DOI, venue, subject areas, citations, provenance stratum, …) carried in
``meta`` — including the record's citation percentile within its publication year, under
the conventional ``attention`` key the pipeline promotes onto ``Analysis.attention``.
Nothing is lost and scan prompts stay shaped like the rest of the corpus.
Stdlib only: no dependency, and no ``openpyxl``.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from horizon.ingest.base import ATTENTION_KEY, RawItem, Source
from horizon.ingest.collection import RECORDS_DIRNAME
from horizon.ingest.normalize import normalize_text

# `--years 2019`, `--years 2015-2018`, `--years 2019,2023`, or any mix of those.
_YEAR = re.compile(r"^\d{4}$")
_RANGE = re.compile(r"^(\d{4})\s*-\s*(\d{4})$")


def parse_years(spec: str) -> list[int]:
    """Expand a year spec into a sorted list of years.

    Accepts a single year (``"2019"``), a closed range (``"2015-2018"``), a
    comma-separated list, or a mix (``"2015-2017,2023"``). Raises ``ValueError`` on
    anything else, including a reversed range.
    """
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if _YEAR.match(part):
            years.add(int(part))
            continue
        span = _RANGE.match(part)
        if not span:
            raise ValueError(
                f"bad year spec: {part!r} (want 2019, 2015-2018, or a list)"
            )
        start, end = int(span.group(1)), int(span.group(2))
        if end < start:
            raise ValueError(f"bad year range: {part!r} (end before start)")
        years.update(range(start, end + 1))
    if not years:
        raise ValueError("empty year spec")
    return sorted(years)


def shard_years(records_dir: Path | str) -> list[int]:
    """The publication years a built dataset has shards for, oldest first."""
    root = Path(records_dir)
    years = []
    for path in root.glob("*.jsonl"):
        if path.stem.isdigit():
            years.append(int(path.stem))
    return sorted(years)


def iter_records(
    records_dir: Path | str, years: Iterable[int] | None = None
) -> Iterator[dict]:
    """Stream built records from a ``records/`` directory, oldest shard first.

    The public form of what :class:`WosSource` does internally, without its selection
    or ``RawItem`` conversion — for callers that want the bibliographic metadata
    (``subject_areas``, ``venue``, ``authors``, citation counts) rather than scannable
    text. ``years`` restricts to those shards; ``None`` reads all of them.
    """
    root = Path(records_dir)
    wanted = None if years is None else set(years)
    for year in shard_years(root):
        if wanted is not None and year not in wanted:
            continue
        with (root / f"{year}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def records_dir(dataset_dir: Path | str) -> Path:
    """The ``records/`` subdir of a built dataset directory."""
    return Path(dataset_dir) / RECORDS_DIRNAME


def item_text(record: dict) -> str:
    """The scannable text for a record: title joined to abstract, normalized.

    Mirrors :class:`~horizon.ingest.arxiv.ArxivSource` — the title gives the headline,
    the abstract the detail. Returns ``""`` when the pair normalizes to nothing usable
    (the caller skips it), which is how a stub title with no abstract drops out.
    """
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    if title and abstract:
        joiner = " " if title[-1] in ".?!" else ". "
        return normalize_text(f"{title}{joiner}{abstract}")
    return normalize_text(title or abstract)


def _meta(record: dict) -> dict:
    """Everything about a record that isn't its scannable text.

    ``date_precision`` flags that ``published_at`` is a year stamped to January 1st —
    these records carry no finer date than the publication year.
    """
    keys = (
        "title",
        "abstract",
        "authors",
        "author_count",
        "year",
        "venue",
        "doi",
        "doc_type",
        "subject_areas",
        "wos_categories",
        "keywords",
        "cited_by_scopus",
        "cited_wos_core",
        "cited_wos_all",
        "cited_max",
        "citation_percentile",
        "provenance",
        "script",
        "source_files",
        "flags",
    )
    meta = {k: record.get(k) for k in keys}
    meta["date_precision"] = "year"
    # The conventional attention signal (see RawItem.meta): for bibliographic records
    # it's the citation percentile within the publication-year cohort. ``None`` for the
    # 23.3% with no citation count at all — never 0.0, which would read as "no attention"
    # rather than "not measurable". ``.get`` also makes shards built before percentiles
    # existed yield None, so older datasets keep working.
    meta[ATTENTION_KEY] = record.get("citation_percentile")
    return meta


class WosSource(Source):
    """Yield a :class:`RawItem` per record from a built collection dataset.

    ``dataset_dir`` is the directory a build wrote (holding ``records/``). ``years``
    restricts which shards are read; ``limit`` keeps the first N records in year order;
    ``sample`` draws N at random, reproducibly for a given ``seed``, via reservoir
    sampling so a 20-record draw over 273 k records costs no memory. ``limit`` and
    ``sample`` are mutually exclusive.
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        *,
        years: list[int] | None = None,
        limit: int | None = None,
        sample: int | None = None,
        seed: int = 0,
    ) -> None:
        """Store the dataset dir and the selection; validate the knob combination."""
        if limit is not None and sample is not None:
            raise ValueError("use --limit or --sample, not both.")
        if limit is not None and limit <= 0:
            raise ValueError("--limit must be positive.")
        if sample is not None and sample <= 0:
            raise ValueError("--sample must be positive.")
        self.dir = Path(dataset_dir)
        self.years = years
        self.limit = limit
        self.sample = sample
        self.seed = seed

    def _shards(self) -> list[Path]:
        """The shard files to read, oldest year first.

        Raises ``FileNotFoundError`` when the dataset hasn't been built, or when
        ``years`` selects nothing that exists (naming the years that do).
        """
        root = records_dir(self.dir)
        available = shard_years(root) if root.is_dir() else []
        if not available:
            raise FileNotFoundError(
                f"no built collection dataset under {root} "
                "(run `horizon collection-build` first)"
            )
        wanted = (
            available
            if self.years is None
            else [y for y in available if y in self.years]
        )
        if not wanted:
            have = ", ".join(str(y) for y in available)
            raise FileNotFoundError(f"no shards for the requested years (have {have})")
        return [root / f"{year}.jsonl" for year in wanted]

    def _iter_records(self) -> Iterator[dict]:
        """Stream records from the selected shards in year order."""
        for shard in self._shards():
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def _selected(self) -> Iterator[dict]:
        """Apply ``limit`` / ``sample`` to the streamed records.

        Sampling is a seeded reservoir over the stream, so the draw is deterministic and
        independent of how the records are sharded, and the whole dataset never has to be
        held in memory.
        """
        if self.sample is None:
            stream = self._iter_records()
            if self.limit is None:
                yield from stream
                return
            for index, record in enumerate(stream):
                if index >= self.limit:
                    return
                yield record
            return

        rng = random.Random(self.seed)
        reservoir: list[dict] = []
        for seen, record in enumerate(self._iter_records()):
            if seen < self.sample:
                reservoir.append(record)
                continue
            slot = rng.randrange(seen + 1)
            if slot < self.sample:
                reservoir[slot] = record
        yield from reservoir

    def fetch(self) -> Iterable[RawItem]:
        """Reconstruct a :class:`RawItem` per selected record.

        Labels each item ``wos-scopus-<year>`` so the corpus splits by publication year,
        dates it ``<year>-01-01`` (the records carry no finer date — see
        ``meta["date_precision"]``), and sets ``lang="en"`` only for Latin-script records,
        leaving it unset where Cyrillic appears rather than guessing the language.
        Records whose title and abstract normalize to nothing are skipped.
        """
        items: list[RawItem] = []
        for record in self._selected():
            text = item_text(record)
            if not text:
                continue
            year = record.get("year")
            items.append(
                RawItem(
                    text=text,
                    source=f"wos-scopus-{year}",
                    url=record.get("url"),
                    published_at=f"{year}-01-01" if year else None,
                    lang="en" if record.get("script") == "latin" else None,
                    meta=_meta(record),
                )
            )
        return items
