"""Build a normalized dataset from the ``collection/`` WoS + Scopus workbooks.

The ``collection/`` corpus is 11 hand-curated Excel exports (one per publication year,
2015–2025) of Web-of-Science and Scopus bibliographic records — ~300 k rows across
**11 different column layouts**, with two stacked-block/column-shift defects and a
deduplication that was only partially applied.

This module is the **parse-once** half of a two-stage pipeline: it reads the workbooks,
maps every layout onto one canonical record, repairs the known defects, dedupes globally
(DOI first, then normalized title, *merging* the Scopus-side and WoS-side copies of the
same paper), and writes one JSONL shard **per publication year** plus a
``build-report.json``. Reading Excel is the expensive step (~10 min, 32 MB shared-string
tables); everything downstream reads the shards, so slicing a year — or 20 records for a
smoke test — is instant. :class:`~horizon.ingest.wos.WosSource` is the other half.

The build also stamps each record's **citation percentile within its publication-year
cohort** (:func:`attach_citation_percentiles`), which is what makes citation counts
usable as an importance signal: raw counts are age-confounded (mean 13.9 for 2015 vs 1.9
for 2025) and heavy-tailed (max 15 185 against a per-cohort p99 of 22–151). The
percentile is a snapshot of one build, not a stable identifier — see that function.

Column resolution is **header-driven**, never index-driven: headers carry irregular
whitespace and 2019 renames every column to a ``(Scopus) …`` prefix, so indices shift
between years while the underlying fields don't. ``openpyxl`` is imported lazily inside
:func:`iter_rows`, so this module (and the package) imports without it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_COLLECTION_DIR = Path("collection")
RECORDS_DIRNAME = "records"
REPORT_NAME = "build-report.json"

# Workbook file names are Дедублікація_WoS_Scopus_<YYYY>.xlsx; the year is the only
# part we need (it dates rows whose own year cell is unusable).
_FILE_YEAR = re.compile(r"(\d{4})\.xlsx$")
_YEAR_CELL = re.compile(r"^(\d{4})(\.0+)?$")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.I)

# Sentinel Scopus writes instead of leaving the abstract empty (~4.5 k rows).
_NO_ABSTRACT = "[no abstract available]"

# Author lists hit Excel's 32 767-char cell cap (large physics collaborations), and one
# 2018 abstract is a concatenation artefact. Capping keeps a full-corpus build inside a
# few hundred MB of RAM; both caps are recorded as flags on the affected record.
MAX_AUTHOR_CHARS = 2000
MAX_ABSTRACT_CHARS = 12000

# Plausible publication years; anything outside this dates from the workbook's filename.
_YEAR_MIN, _YEAR_MAX = 1990, 2035

# The three raw citation tallies. The derived ``cited_max`` is their maximum — a
# *coverage-union heuristic, not a citation count*: Scopus and WoS count different
# citing populations (where both report, they disagree 62% of the time), and the WoS
# pair is not even internally consistent (39 records have all-databases < core), so no
# single field can be preferred.
CITED_KEYS = ("cited_by_scopus", "cited_wos_core", "cited_wos_all")

# Below this many *counted* records a cohort percentile is sampling noise rather than a
# statistic, so none is emitted and every member is flagged instead. Only ever fires on
# partial builds: shards leak across years (a ``--years 2019`` build drops 36 records
# into the 2022 shard), while the full build's smallest cohort has 482 counted records.
MIN_CITATION_COHORT = 50
SMALL_COHORT_FLAG = "small-citation-cohort"


def norm_header(value: Any) -> str:
    """Collapse a header cell to a single-spaced, trimmed string (``""`` when empty).

    Headers in these workbooks carry leading, trailing and doubled spaces
    (``" Прізвище Автора / Authors"``, ``"Назва публікації /  Title "``, ``" DOI"``), so
    every lookup normalizes first and matches on substrings.
    """
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


@dataclass(frozen=True)
class ColumnMap:
    """Resolved column indices for one workbook's header row.

    Every field is the 0-based index of that logical field in the sheet, or ``None``
    when this year's layout omits it (keywords vanish in 2019–2021, ``Cited by`` in 2015
    and 2022–2025, …). ``export_*`` are 2015's extra WoS-side columns, used to fill gaps
    in the Scopus-side cells. ``has_export_block`` marks that 18-column 2015 layout — the
    one file with a stacked, column-shifted second block.
    """

    authors: int | None = None
    title: int | None = None
    year: int | None = None
    source_title: int | None = None
    cited_by: int | None = None
    doi: int | None = None
    link: int | None = None
    abstract: int | None = None
    keywords: int | None = None
    doc_type: int | None = None
    subject_area: int | None = None
    wos_core: int | None = None
    wos_all: int | None = None
    wos_categories: int | None = None
    export_title: int | None = None
    export_abstract: int | None = None
    export_doi: int | None = None

    @property
    def has_export_block(self) -> bool:
        """True for 2015's 18-column layout (the only one with a shifted tail block)."""
        return self.export_title is not None


def resolve_columns(header: list[str]) -> ColumnMap:
    """Map a normalized ``header`` row onto :class:`ColumnMap` field indices.

    Matches on lowercase substrings so one rule covers every year: the Ukrainian/English
    dual headers (``"Назва видання / Source title"``), 2019's ``(Scopus) …`` renames, and
    ``(Scopus) Branch`` standing in for ``Галузь науки``. ``(Export) …`` columns are
    excluded from the primary fields and picked up separately, and ``(Export) DOI Link``
    (a lone ``Y`` flag in 2022–2024) is never mistaken for the DOI.
    """
    low = [h.lower() for h in header]

    def find(*needles: str, exclude: tuple[str, ...] = ()) -> int | None:
        for i, h in enumerate(low):
            if not h or any(x in h for x in exclude):
                continue
            if all(n in h for n in needles):
                return i
        return None

    def exact(*names: str) -> int | None:
        for i, h in enumerate(low):
            if h in names:
                return i
        return None

    def first(*candidates: int | None) -> int | None:
        """The first resolved index — ``or`` would discard a legitimate index 0."""
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    return ColumnMap(
        authors=first(
            find("authors", exclude=("export",)),
            find("прізвище автора", exclude=("export",)),
        ),
        title=find("title", exclude=("export", "source")),
        year=first(exact("рік", "(scopus) year"), find("year", exclude=("export",))),
        source_title=find("source title", exclude=("export",)),
        cited_by=find("cited by"),
        # Exact match: 2015 also carries "DOI / (Export) DOI" and 2022–2024 an
        # "(Export) DOI Link" flag column, neither of which is the record's DOI.
        doi=exact("doi", "(scopus) doi"),
        link=find("link", exclude=("doi",)),
        abstract=find("abstract", exclude=("export",)),
        keywords=find("keywords", exclude=("export",)),
        # 2020–2021 drop the Scopus-side document type and keep only the WoS one.
        doc_type=first(
            find("document type", exclude=("export",)), find("document type")
        ),
        subject_area=first(find("галузь"), find("branch")),
        wos_core=find("wos core"),
        wos_all=find("all databases"),
        wos_categories=find("wos categories"),
        export_title=find("(export) article title"),
        export_abstract=find("(export) abstract"),
        export_doi=find("(export) doi", exclude=("link",)),
    )


# 2015's rows 11 201–17 522 are a WoS-only block appended under the header with a
# *different* layout: the year column holds the source title and everything after it
# slides by one. Indices are fixed because this block has no header of its own.
_SHIFTED = ColumnMap(
    authors=0,
    title=1,
    source_title=2,
    abstract=3,
    doi=4,
    wos_categories=9,
    wos_core=14,
    wos_all=15,
)


def _cell(row: tuple, idx: int | None) -> str | None:
    """Stripped string value at ``idx``, or ``None`` when absent/blank.

    Whitespace-only cells (the unnamed spacer column, some ``Cited by`` cells) hold
    ``"  "``, which is truthy in Python — so every read goes through here.
    """
    if idx is None or idx >= len(row):
        return None
    value = row[idx]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: str | None) -> int | None:
    """Citation counts and years arrive as ``"5"`` or ``"5.0"``; coerce, else ``None``."""
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _split_list(value: str | None) -> list[str]:
    """Split a ``;``-separated cell (keywords, subject areas, WoS categories)."""
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def normalize_doi(value: str | None) -> str | None:
    """Casefold a DOI and strip any ``https://doi.org/`` / ``doi:`` prefix."""
    if not value:
        return None
    doi = _DOI_PREFIX.sub("", value.strip()).strip().lower()
    return doi or None


def title_key(title: str | None) -> str | None:
    """Dedup key for a title: casefolded, stripped of everything but alphanumerics.

    Used only for the ~51 k rows with no DOI; short residues are rejected so a stub
    title can't collapse unrelated records.
    """
    if not title:
        return None
    key = _NON_ALNUM.sub("", title.casefold())
    return key if len(key) >= 12 else None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` chars at a word boundary; flag whether it was cut."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rsplit(" ", 1)[0], True


def normalize_row(row: tuple, cols: ColumnMap, file_year: int) -> dict | None:
    """Turn one sheet row into a canonical record, or ``None`` if unusable.

    Applies the repairs the corpus needs: 2015's shifted block is detected (a non-year
    in the year column of the 18-column layout) and re-mapped; rows whose title is an
    unparsed TSV line (a literal tab, ~6 rows in 2019) or missing entirely are dropped;
    ``[No abstract available]`` becomes a missing abstract; oversized author lists and
    abstracts are capped. A row's own year wins when plausible, else the workbook's.

    The returned dict is the JSONL record: bibliographic fields, the ``provenance``
    stratum (``scopus`` / ``wos`` / ``both``, read from the Scopus link and the WoS
    citation columns), and ``flags`` naming every repair applied to it.
    """
    flags: list[str] = []

    # 2015's stacked tail: the year column holds a source title, so re-read the row
    # through the shifted map. Only that layout has the defect.
    year_cell = _cell(row, cols.year)
    shifted = (
        cols.has_export_block
        and cols.year is not None
        and (year_cell is None or not _YEAR_CELL.match(year_cell))
    )
    if shifted:
        cols = _SHIFTED
        year_cell = None
        flags.append("shifted-block")

    title = _cell(row, cols.title)
    if title is None and cols.export_title is not None:
        title = _cell(row, cols.export_title)
    if title is None:
        return None  # no title, no record (0–8 rows per file)
    if "\t" in title:
        return None  # unparsed TSV line collapsed into one cell (2019)

    year = _as_int(year_cell)
    if year is None or not (_YEAR_MIN <= year <= _YEAR_MAX):
        year = file_year
        if not shifted:
            flags.append("year-from-filename")

    abstract = _cell(row, cols.abstract)
    if abstract is None and cols.export_abstract is not None:
        abstract = _cell(row, cols.export_abstract)
    if abstract and abstract.strip().lower() == _NO_ABSTRACT:
        abstract = None
    if abstract:
        abstract, cut = _truncate(abstract, MAX_ABSTRACT_CHARS)
        if cut:
            flags.append("abstract-truncated")

    authors = _cell(row, cols.authors)
    author_count = len(_split_list(authors)) if authors else 0
    if authors:
        authors, cut = _truncate(authors, MAX_AUTHOR_CHARS)
        if cut:
            flags.append("authors-truncated")

    doi = normalize_doi(_cell(row, cols.doi))
    if doi is None and cols.export_doi is not None:
        doi = normalize_doi(_cell(row, cols.export_doi))

    link = _cell(row, cols.link)
    wos_core = _as_int(_cell(row, cols.wos_core))
    wos_all = _as_int(_cell(row, cols.wos_all))
    from_scopus = link is not None
    from_wos = wos_core is not None or wos_all is not None
    provenance = (
        "both" if from_scopus and from_wos else "scopus" if from_scopus else "wos"
    )

    text = f"{title} {abstract or ''}"
    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "author_count": author_count,
        "year": year,
        "venue": _cell(row, cols.source_title),
        "doi": doi,
        "url": link or (f"https://doi.org/{doi}" if doi else None),
        "doc_type": _cell(row, cols.doc_type),
        "subject_areas": _split_list(_cell(row, cols.subject_area)),
        "wos_categories": _split_list(_cell(row, cols.wos_categories)),
        "keywords": _split_list(_cell(row, cols.keywords)),
        "cited_by_scopus": _as_int(_cell(row, cols.cited_by)),
        "cited_wos_core": wos_core,
        "cited_wos_all": wos_all,
        # Derived at write time, once the whole publication-year cohort is in hand
        # (see attach_citation_percentiles) — a row alone cannot know its own rank.
        "cited_max": None,
        "citation_percentile": None,
        "provenance": provenance,
        "script": "cyrillic" if _CYRILLIC.search(text) else "latin",
        "source_files": [file_year],
        "flags": flags,
    }


def merge_records(kept: dict, other: dict) -> dict:
    """Fold a duplicate ``other`` into ``kept`` in place; return ``kept``.

    The workbooks' deduplication missed ~2 700 papers per year that are present as *both*
    a Scopus-only row and a WoS-only row, so a duplicate is usually the other database's
    view of the same paper: merging recovers a fuller record than either row alone.
    Missing scalars are filled, list fields unioned (order-preserving), citation counts
    take the maximum, and the provenance stratum widens to ``both`` when the two rows
    came from different databases.
    """
    for key in ("abstract", "authors", "venue", "doi", "url", "doc_type"):
        if not kept.get(key) and other.get(key):
            kept[key] = other[key]
    for key in ("subject_areas", "wos_categories", "keywords"):
        merged = list(kept.get(key) or [])
        merged += [v for v in (other.get(key) or []) if v not in merged]
        kept[key] = merged
    for key in CITED_KEYS:
        values = [v for v in (kept.get(key), other.get(key)) if v is not None]
        kept[key] = max(values) if values else None
    if kept["provenance"] != other["provenance"]:
        kept["provenance"] = "both"
    kept["author_count"] = max(kept["author_count"], other["author_count"])
    kept["source_files"] = sorted(
        set(kept["source_files"]) | set(other["source_files"])
    )
    kept["flags"] = sorted(set(kept["flags"]) | set(other["flags"]))
    if kept["abstract"] and kept["script"] == "latin":
        kept["script"] = "cyrillic" if _CYRILLIC.search(kept["abstract"]) else "latin"
    return kept


def cited_max(record: dict) -> int | None:
    """The record's citation count: the max of its Scopus and WoS tallies, or ``None``.

    ``None`` means **unknown**, never zero. The ``Cited by`` column is absent from the
    2015 and 2022–2025 workbooks, so 51.3% of Scopus-only records carry no count at all
    (against 1.6% of WoS-only and 0% of ``both``); collapsing that to 0 would rank a
    whole provenance stratum at the bottom of its cohort. See :data:`CITED_KEYS` for why
    the maximum rather than a preferred field.
    """
    values = [v for v in (record.get(key) for key in CITED_KEYS) if v is not None]
    return max(values) if values else None


def rank_percentiles(values: Sequence[int]) -> dict[int, float]:
    """Mid-rank percentile in ``(0, 1)`` for every distinct value in ``values``.

    ``(below + equal / 2) / n`` — tied values share the *average* of the ranks they span.
    Strict less-than would be wrong here: 57.5% of the 2025 cohort has zero citations,
    and strict-less-than pins that entire mode at 0.0, i.e. claims the median paper drew
    no attention *relative to its peers*. The mid-rank puts it at 0.2875 instead. Neither
    endpoint is attainable, so a real value can never be confused with an absent one.
    """
    tally = Counter(values)
    total = sum(tally.values())
    out: dict[int, float] = {}
    below = 0
    for value in sorted(tally):
        equal = tally[value]
        out[value] = (below + equal / 2) / total
        below += equal
    return out


def attach_citation_percentiles(
    records: list[dict], *, min_cohort: int | None = None
) -> int:
    """Stamp one publication-year cohort's ``cited_max``/``citation_percentile`` in place.

    The cohort is the records *that have a count*: the rest are excluded from the ranking
    — so the missing ``Cited by`` column cannot distort the distribution — and keep
    ``citation_percentile = None``. Cohorts below ``min_cohort`` counted records get no
    percentile at all and every member is flagged :data:`SMALL_COHORT_FLAG`.

    The percentile is a **snapshot of this build**: it is relative to its cohort, so it
    shifts whenever the cohort does (a new workbook year, or a ``years``-restricted
    build), and values are only comparable to others from the same build. Returns the
    number of counted records in the cohort.
    """
    floor = MIN_CITATION_COHORT if min_cohort is None else min_cohort
    counted: list[tuple[dict, int]] = []
    for record in records:
        count = cited_max(record)
        record["cited_max"] = count
        # Written unconditionally, so a rebuild clears a percentile that no longer
        # qualifies instead of leaving the previous run's value behind.
        record["citation_percentile"] = None
        if count is not None:
            counted.append((record, count))
    if not counted:
        return 0
    if len(counted) < floor:
        for record in records:
            if SMALL_COHORT_FLAG not in record["flags"]:
                record["flags"].append(SMALL_COHORT_FLAG)
        return len(counted)
    percentiles = rank_percentiles([count for _, count in counted])
    for record, count in counted:
        record["citation_percentile"] = round(percentiles[count], 6)
    return len(counted)


@dataclass
class BuildReport:
    """Counts from one :func:`build_dataset` run — printed by the CLI, saved as JSON."""

    files: list[str] = field(default_factory=list)
    rows: int = 0
    records: int = 0
    merged: int = 0
    dropped_no_title: int = 0
    with_abstract: int = 0
    with_citations: int = 0
    per_year: dict[str, int] = field(default_factory=dict)
    per_year_citations: dict[str, int] = field(default_factory=dict)
    per_provenance_citations: dict[str, int] = field(default_factory=dict)
    per_provenance: dict[str, int] = field(default_factory=dict)
    per_script: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-safe view, with year/flag tallies sorted for a stable report file."""
        return {
            "files": self.files,
            "rows": self.rows,
            "records": self.records,
            "merged": self.merged,
            "dropped_no_title": self.dropped_no_title,
            "with_abstract": self.with_abstract,
            "with_citations": self.with_citations,
            "per_year": dict(sorted(self.per_year.items())),
            "per_year_citations": dict(sorted(self.per_year_citations.items())),
            "per_provenance": dict(sorted(self.per_provenance.items())),
            "per_provenance_citations": dict(
                sorted(self.per_provenance_citations.items())
            ),
            "per_script": dict(sorted(self.per_script.items())),
            "flags": dict(sorted(self.flags.items())),
        }


def workbook_year(path: Path) -> int | None:
    """The publication year in a workbook's filename, or ``None`` if it has none."""
    match = _FILE_YEAR.search(path.name)
    return int(match.group(1)) if match else None


def find_workbooks(
    collection_dir: Path | str, years: list[int] | None = None
) -> list[Path]:
    """The ``.xlsx`` workbooks under ``collection_dir``, oldest first.

    ``years`` keeps only those filename years. Excel lock files (``~$…``) are skipped.
    Raises ``NotADirectoryError`` when the directory is missing.
    """
    root = Path(collection_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    found = [
        p
        for p in sorted(root.glob("*.xlsx"))
        if not p.name.startswith("~$") and (years is None or workbook_year(p) in years)
    ]
    return found


def iter_rows(path: Path) -> Iterator[tuple[list[str], tuple]]:
    """Stream ``(header, row)`` pairs from a workbook's first sheet.

    Always the *first* sheet by index — the sheet name varies (``Лист1``, ``все_2017``,
    ``все 2018``, …). Read-only and values-only so a 26 MB workbook streams instead of
    loading; blank rows are skipped. ``openpyxl`` is imported here, not at module level.
    """
    import openpyxl  # lazy: only the build step needs the Excel reader

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book[book.sheetnames[0]]
        rows = sheet.iter_rows(values_only=True)
        header = [norm_header(h) for h in next(rows, ())]
        for row in rows:
            if all(v is None or (isinstance(v, str) and not v.strip()) for v in row):
                continue
            yield header, row
    finally:
        book.close()


def build_dataset(
    collection_dir: Path | str,
    out_dir: Path | str,
    *,
    years: list[int] | None = None,
    on_file=None,
) -> BuildReport:
    """Parse the workbooks into ``out_dir/records/<year>.jsonl`` + ``build-report.json``.

    Records are sharded by **publication year** (a paper the 2024 workbook dates 2025
    lands in ``2025.jsonl``), deduped globally across every workbook in the run — DOI
    first, then normalized title for the rows without one — and duplicates are merged
    via :func:`merge_records`. ``on_file(path, rows)`` is called after each workbook so
    the CLI can report progress on a run that takes minutes.

    Records are held in memory until every workbook is read (cross-file duplicates can
    only be merged that way): a full 11-year build peaks a few hundred MB. Build a
    subset with ``years`` to bound it — dedup then spans only the years built.
    """
    out = Path(out_dir)
    report = BuildReport()
    by_key: dict[tuple[str, str], dict] = {}
    ordered: list[dict] = []

    for path in find_workbooks(collection_dir, years):
        file_year = workbook_year(path) or 0
        rows = 0
        cols: ColumnMap | None = None
        for header, row in iter_rows(path):
            if cols is None:
                cols = resolve_columns(header)
            rows += 1
            record = normalize_row(row, cols, file_year)
            if record is None:
                report.dropped_no_title += 1
                continue
            doi = record["doi"]
            key = ("doi", doi) if doi else None
            if key is None:
                tkey = title_key(record["title"])
                key = ("title", tkey) if tkey else None
            if key is not None and key in by_key:
                merge_records(by_key[key], record)
                report.merged += 1
                continue
            if key is not None:
                by_key[key] = record
            ordered.append(record)
        report.rows += rows
        report.files.append(path.name)
        if on_file is not None:
            on_file(path, rows)

    # Shard by publication year, writing each file once.
    shards: dict[int, list[dict]] = {}
    for record in ordered:
        shards.setdefault(record["year"], []).append(record)

    records_dir = out / RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)
    for year, group in sorted(shards.items()):
        # Percentiles are cohort-relative, and this shard *is* the publication-year
        # cohort — the only point in the pipeline where a whole one is in hand. The
        # dicts are shared with ``ordered``, so the report loop below sees the stamps.
        attach_citation_percentiles(group)
        with (records_dir / f"{year}.jsonl").open("w", encoding="utf-8") as fh:
            for record in group:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        report.per_year[str(year)] = len(group)

    report.records = len(ordered)
    for record in ordered:
        if record["abstract"]:
            report.with_abstract += 1
        if record["cited_max"] is not None:
            report.with_citations += 1
            year_key = str(record["year"])
            report.per_year_citations[year_key] = (
                report.per_year_citations.get(year_key, 0) + 1
            )
            report.per_provenance_citations[record["provenance"]] = (
                report.per_provenance_citations.get(record["provenance"], 0) + 1
            )
        report.per_provenance[record["provenance"]] = (
            report.per_provenance.get(record["provenance"], 0) + 1
        )
        report.per_script[record["script"]] = (
            report.per_script.get(record["script"], 0) + 1
        )
        for flag in record["flags"]:
            report.flags[flag] = report.flags.get(flag, 0) + 1

    (out / REPORT_NAME).write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
