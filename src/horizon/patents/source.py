"""Read a built patent dataset back as gatherable items.

The other half of the two-stage build (see :mod:`horizon.patents.workbooks`), and the
exact counterpart of :class:`~horizon.ingest.wos.WosSource`: it streams the per-cohort
JSONL shards and yields :class:`~horizon.ingest.base.RawItem`\\ s, so the register flows
into the ordinary ``gather`` → ``corpus/`` → ``scan`` path with no Excel parsing.

The shard reading itself is *not* reimplemented here — ``horizon.ingest.wos`` already
streams ``records/<year>.jsonl`` and knows nothing about bibliography, so this module
imports it. The parts that genuinely differ are the item text, which joins a patent's
title to its abstract, and the label each item lands under.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from horizon.ingest.base import ATTENTION_KEY, RawItem, Source
from horizon.ingest.normalize import normalize_text
from horizon.ingest.wos import records_dir, shard_years

SOURCE_PREFIX = "patents-ua"

# What travels into ``meta`` with every item. The scannable text is dropped from it (it is
# already ``RawItem.text``); everything else the register knew is kept, because it cannot
# be recovered from the normalized text later and it is what makes the corpus reusable.
_DROP_FROM_META = frozenset({"url", "script"})


def item_text(record: dict) -> str:
    """The scannable text: title joined to abstract, normalized.

    A patent title is a formal claim phrase rather than a summary — "СПОСІБ РОЗКРИТТЯ ТА
    ПІДЗЕМНОЇ РОЗРОБКИ РУДНИХ РОДОВИЩ" — so it carries the subject and the abstract
    carries the substance, which is the same division the bibliographic adapter relies on.
    Returns ``""`` when the pair normalizes to nothing, and the caller skips the record.
    """
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    if title and abstract:
        joiner = " " if title[-1] in ".?!" else ". "
        return normalize_text(f"{title}{joiner}{abstract}")
    return normalize_text(title or abstract)


def _meta(record: dict) -> dict:
    """Everything about a record that is not its scannable text.

    ``date_precision`` says the stamp is a year pinned to January 1st. The register does
    hold exact dates — they ride along in ``filed_at``/``granted_at``/``priority_at`` —
    but the item is *dated* by its cohort, because that is the axis every trend is fitted
    on and a finer stamp on the item would invite the two to disagree.
    """
    meta = {k: v for k, v in record.items() if k not in _DROP_FROM_META}
    meta["date_precision"] = "year"
    # The conventional attention signal: citations ranked within the cohort. ``None`` for
    # the ~98% with no international sibling — unknown, not zero (see
    # ``attach_citation_percentiles``).
    meta[ATTENTION_KEY] = record.get("citation_percentile")
    return meta


class PatentSource(Source):
    """Yield a :class:`RawItem` per record from a built patent dataset.

    ``dataset_dir`` is the directory a build wrote (holding ``records/``). ``years``
    restricts which cohorts are read; ``limit`` keeps the first N in cohort order;
    ``sample`` draws N reproducibly for a given ``seed`` by reservoir sampling, so a
    20-record smoke test costs no memory over the whole register. ``limit`` and ``sample``
    are mutually exclusive.
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
        """The shard files to read, oldest cohort first.

        Raises ``FileNotFoundError`` when the dataset has not been built, or when
        ``years`` selects nothing that exists — naming the cohorts that do, since the
        usual cause is asking for a year the register's axis does not reach.
        """
        root = records_dir(self.dir)
        available = shard_years(root) if root.is_dir() else []
        if not available:
            raise FileNotFoundError(
                f"no built patent dataset under {root} "
                "(run `horizon patents-build` first)"
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
        """Stream records from the selected shards in cohort order."""
        import json

        for shard in self._shards():
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def _selected(self) -> Iterator[dict]:
        """Apply ``limit`` / ``sample`` to the streamed records."""
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

        Labels each item ``patents-ua-<year>`` so the corpus lands one folder per cohort
        and a single cohort can be replayed with ``gather corpus``. ``lang`` comes from
        the register's own abstract-language column rather than from guessing at the
        script: 99.3% of it says ``UA``, and the handful that say ``EN`` are not
        transliterations but genuinely English abstracts.
        """
        items: list[RawItem] = []
        for record in self._selected():
            text = item_text(record)
            if not text:
                continue
            year = record.get("year")
            lang = (record.get("abstract_lang") or "").strip().lower() or None
            items.append(
                RawItem(
                    text=text,
                    source=f"{SOURCE_PREFIX}-{year}",
                    url=record.get("url"),
                    published_at=f"{year}-01-01" if year else None,
                    lang={"ua": "uk"}.get(lang, lang),
                    meta=_meta(record),
                )
            )
        return items
