"""Parse the patent workbooks into the canonical record the analysis layer already reads.

Three files, one register. The main sheet is the Ukrainian grant micro-base — every
record, with the classification and the abstract. The two satellites cover the small
fraction of those families that went abroad, and they carry what the national register
does not: citation counts, and for the IP5 filings a set of valuation scores.

    b26_070_FinalResult_20260813.xlsx  22 225 grants   the register
    b26_070_final_WIPO.xlsx             1 391 WO pubs  joins on PatBase family number
    b26_070_ip5.xlsx                      238 filings  joins on UA patent number

The satellites are **enrichment, never rows**. Joining them in as records would double-count
an invention that reached three offices, and letting them define the corpus would replace a
register with the 2% of it that was filed internationally. So they attach to the record they
match and nothing else: 480 of 503 WO-linked families resolve, and 225 IP5 filings.

The emitted record carries patent-native fields *and* the canonical keys
``horizon.ingest.collection`` emits, because the analysis layer reads those names and
teaching it a second vocabulary would fork every module downstream for no gain. The
mapping is a deliberate reading of what each key means rather than a rename:

    year            the chosen time axis (see horizon.patents.completeness)
    subject_areas   WIPO technologies — 35 values, multi-valued on 46% of records, which
                    is what ``subject_areas`` is on the bibliographic corpus (27, 45%)
    venue           the principal applicant. The dispersion lens counts distinct venues to
                    ask how concentrated a topic's output is; for patents the same
                    question is asked of applicants, and the answer means the same thing.
    authors         inventors — the new-entrant lens reads exactly this
    citation_percentile   forward citations ranked within the cohort, where a satellite
                    supplied them. Absent on 98% of the register, and absent means
                    unknown: a national grant with no international sibling has no
                    citation count, which is not a count of zero.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from horizon.patents.completeness import (
    DEFAULT_AXIS,
    YEAR_AXES,
    complete_years,
    lag_drift,
)

DEFAULT_PATENTS_DIR = Path("patents")
RECORDS_DIRNAME = "records"
REPORT_NAME = "build-report.json"

MAIN_SHEET = "Агрегатовані дані UA"
SATELLITE_SHEET = "Publications"
# The main sheet opens with a title banner and a provenance note, so its header is the
# fourth row. Located by content rather than by index: a re-export that gains or loses a
# note line would otherwise shift every column silently.
HEADER_ANCHOR = "Номер патенту"
MAX_HEADER_SCAN = 20

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Semicolon only, never comma. The export uses ";" throughout, and commas occur *inside*
# values: 1 907 records carry a WIPO technology whose own name has one ("Електричні
# машини, апарати та енергетика"), as do 9 023 inventor lists. Splitting on comma parses
# those names into fragments and then reports the fragments as a cross-technology pair —
# it produced "Двигуни + насоси й турбіни", which is one technology, not two.
_SPLIT = re.compile(r"\s*;\s*")
_YEAR_MIN, _YEAR_MAX = 1900, 2100

MAX_ABSTRACT_CHARS = 12000
MAX_AUTHOR_CHARS = 2000

# Main-sheet columns, by header text. Header text is the key because the sheet is a
# hand-curated export: its column *order* is not a contract, its headings are.
MAIN_COLUMNS = {
    "patent_number": "Номер патенту",
    "application_number": "Номер заявки",
    "filed_at": "Дата подання",
    "granted_at": "Дата видачі",
    "priority_at": "Пріоритет / дата подання",
    "priority_year": "Рік пріоритету",
    "priority_source": "Джерело пріоритету",
    "filing_year": "Рік подання",
    "grant_year": "Рік видачі",
    "lag_years": "Лаг, років (оцінка)",
    "title": "Назва",
    "applicant": "Основний заявник",
    "owner": "Основний власник",
    "inventors": "Винахідники",
    "abstract_lang": "Мова реферату",
    "attorney": "Представник (патентний повірений)",
    "applicant_countries": "Країни заявників",
    "origin": "Походження",
    "ua_participation": "Участь UA",
    "applicant_type": "Тип заявника",
    "ipc_codes": "Коди МПК",
    "ipc_subclasses": "Підкласи МПК",
    "wipo_field_codes": "Поля WIPO",
    "wipo_technologies": "Технології WIPO",
    "wipo_sectors": "Сектори WIPO",
    "family_number": "PatBase Family number",
    "family_members": "PatBase Family Member List",
    "wo_publication": "UA2WIPO Application",
    "is_ip5": "IP5",
    "abstract": "Реферат",
}
LIST_FIELDS = (
    "inventors",
    "applicant_countries",
    "ipc_codes",
    "ipc_subclasses",
    "wipo_field_codes",
    "wipo_technologies",
    "wipo_sectors",
    "family_members",
)


def _text(value: Any) -> str | None:
    """A trimmed cell, or ``None`` when it holds nothing usable."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_int(value: Any) -> int | None:
    """An integer from a cell that may be a float, a float-shaped string, or blank.

    The export writes years as ``2012.0``; ``int("2012.0")`` raises, so the float route
    is taken first and only then narrowed.
    """
    s = _text(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    """A float from a cell, or ``None``."""
    s = _text(value)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _as_bool(value: Any) -> bool | None:
    """The export's booleans, written as ``True``/``False`` text."""
    s = _text(value)
    if s is None:
        return None
    return s.lower() in {"true", "1", "yes", "так"}


def _split(value: Any) -> list[str]:
    """A delimited cell as a list, order preserved, blanks and repeats dropped.

    Family member lists are newline-delimited and every other list is semicolon-delimited,
    so both separators are honoured — and only those two (see :data:`_SPLIT`). Order is
    kept rather than sorted: the first IPC code of a patent is its principal
    classification, and sorting would throw that away.
    """
    s = _text(value)
    if s is None:
        return []
    parts = [p.strip() for chunk in s.splitlines() for p in _SPLIT.split(chunk)]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _iso(value: Any) -> str | None:
    """A date cell as ``YYYY-MM-DD``, whether it arrived typed or as text."""
    if isinstance(value, date):
        return value.isoformat()
    s = _text(value)
    if s is None:
        return None
    s = s.split(" ")[0].replace("/", "-").replace(".", "-")
    parts = s.split("-")
    if len(parts) == 3 and len(parts[0]) == 4:
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2])).isoformat()
        except ValueError:
            return None
    if len(parts) == 3 and len(parts[2]) == 4:  # DD-MM-YYYY, as the note line uses
        try:
            return date(int(parts[2]), int(parts[1]), int(parts[0])).isoformat()
        except ValueError:
            return None
    return None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cap a field, reporting whether it was cut so the record can flag itself."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def _find_header(rows: list[tuple], anchor: str) -> int:
    """Index of the header row, located by the anchor heading it must contain."""
    for i, row in enumerate(rows[:MAX_HEADER_SCAN]):
        if any(_text(v) == anchor for v in row):
            return i
    raise ValueError(f"no header row containing {anchor!r} in the first rows")


def _sheet_rows(path: Path, sheet: str) -> list[tuple]:
    """Every row of a sheet, blanks dropped. ``openpyxl`` is imported here, not at top.

    The export writes no dimension record, so ``read_only`` mode cannot size the sheet
    and trailing padding rows come through as all-``None`` — hence the filter rather than
    a row count.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            raise ValueError(f"{path.name} has no sheet {sheet!r} ({wb.sheetnames})")
        rows = [
            row
            for row in wb[sheet].iter_rows(values_only=True)
            if any(v is not None for v in row)
        ]
    finally:
        wb.close()
    return rows


def read_main(path: Path | str) -> Iterator[dict]:
    """Yield one raw field dict per register row, keyed by :data:`MAIN_COLUMNS` names.

    Raw here means "cells, addressed by name" — parsing into the canonical record is
    :func:`normalize_row`'s job, so the two can be tested apart.
    """
    rows = _sheet_rows(Path(path), MAIN_SHEET)
    head_at = _find_header(rows, HEADER_ANCHOR)
    header = [_text(v) for v in rows[head_at]]
    index = {
        name: header.index(col) for name, col in MAIN_COLUMNS.items() if col in header
    }
    missing = sorted(set(MAIN_COLUMNS) - set(index))
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {', '.join(missing)}")
    for row in rows[head_at + 1 :]:
        yield {name: (row[i] if i < len(row) else None) for name, i in index.items()}


def read_satellite(path: Path | str) -> list[dict]:
    """The WO or IP5 sheet as dicts keyed by its own English headers.

    One shape serves both: they are the same export with different column sets, so the
    reader takes whatever columns are present and the join decides what to use.
    """
    rows = _sheet_rows(Path(path), SATELLITE_SHEET)
    header = [_text(v) for v in rows[0]]
    return [
        {h: row[i] for i, h in enumerate(header) if h and i < len(row)}
        for row in rows[1:]
    ]


def normalize_row(raw: dict, *, axis: str = DEFAULT_AXIS) -> dict | None:
    """One register row as a canonical record, or ``None`` if it cannot carry analysis.

    A row is dropped only when it has no title *and* no abstract: without either there is
    nothing to embed, and a record that cannot be clustered would still be counted in
    every share and denominator. Everything else that is merely missing is kept and
    flagged — 0.7% of the register has no title, and those rows are still grants.
    """
    flags: list[str] = []
    title = _text(raw.get("title"))
    abstract = _text(raw.get("abstract"))
    if not title and not abstract:
        return None
    if not title:
        flags.append("no-title")
    if not abstract:
        flags.append("no-abstract")
    if abstract:
        abstract, cut = _truncate(abstract, MAX_ABSTRACT_CHARS)
        if cut:
            flags.append("abstract-truncated")

    inventors = _split(raw.get("inventors"))
    inventor_text = "; ".join(inventors)
    if len(inventor_text) > MAX_AUTHOR_CHARS:
        inventor_text, _ = _truncate(inventor_text, MAX_AUTHOR_CHARS)
        flags.append("inventors-truncated")

    years = {
        key: _as_int(raw.get(key))
        for key in ("priority_year", "filing_year", "grant_year")
    }
    for key, value in years.items():
        if value is not None and not (_YEAR_MIN <= value <= _YEAR_MAX):
            years[key] = None
            flags.append(f"{key.replace('_', '-')}-out-of-range")
    year = years[YEAR_AXES[axis]]
    if year is None:
        flags.append("no-year-on-axis")

    technologies = _split(raw.get("wipo_technologies"))
    if not technologies:
        flags.append("unclassified")

    members = _split(raw.get("family_members"))
    text = f"{title or ''} {abstract or ''}"
    return {
        # --- canonical keys the analysis layer reads ---
        "title": title,
        "abstract": abstract,
        "authors": inventor_text or None,
        "author_count": len(inventors),
        "year": year,
        "venue": _text(raw.get("applicant")),
        "subject_areas": technologies,
        "keywords": _split(raw.get("ipc_codes")),
        "url": None,
        "doc_type": "patent",
        "citation_percentile": None,  # attached from a satellite, per cohort
        "script": "cyrillic" if _CYRILLIC.search(text) else "latin",
        "flags": flags,
        # --- patent-native fields ---
        "patent_number": _text(raw.get("patent_number")),
        "application_number": _text(raw.get("application_number")),
        "filed_at": _iso(raw.get("filed_at")),
        "granted_at": _iso(raw.get("granted_at")),
        "priority_at": _iso(raw.get("priority_at")),
        "priority_year": years["priority_year"],
        "filing_year": years["filing_year"],
        "grant_year": years["grant_year"],
        "lag_years": _as_float(raw.get("lag_years")),
        "priority_source": _text(raw.get("priority_source")),
        "applicant": _text(raw.get("applicant")),
        "owner": _text(raw.get("owner")),
        "inventors": inventors,
        "attorney": _text(raw.get("attorney")),
        "applicant_countries": _split(raw.get("applicant_countries")),
        "applicant_type": _text(raw.get("applicant_type")),
        "origin": _text(raw.get("origin")),
        "ua_participation": _as_bool(raw.get("ua_participation")),
        "abstract_lang": _text(raw.get("abstract_lang")),
        "ipc_codes": _split(raw.get("ipc_codes")),
        "ipc_subclasses": _split(raw.get("ipc_subclasses")),
        "wipo_field_codes": _split(raw.get("wipo_field_codes")),
        "wipo_technologies": technologies,
        "wipo_sectors": _split(raw.get("wipo_sectors")),
        "family_number": _text(raw.get("family_number")),
        "family_members": members,
        "family_size": len(members),
        "wo_publication": _text(raw.get("wo_publication")),
        "is_ip5": bool(_as_bool(raw.get("is_ip5"))),
        # --- filled by the satellite join ---
        "forward_citations": None,
        "backward_citations": None,
        "international_offices": [],
        "valuation": None,
    }


def _offices(members: list[str]) -> list[str]:
    """The patent offices a family reached, from its member publication numbers.

    A member reads ``EP4025006 A1``; the office is its leading letters. This is the only
    international-reach signal available for the 98% of the register with no satellite
    row, and it costs nothing — the family member list is already there on every record.
    """
    out: list[str] = []
    for member in members:
        code = re.match(r"\s*([A-Z]{2})", member.strip())
        if code and code.group(1) not in out:
            out.append(code.group(1))
    return out


def attach_satellites(
    records: list[dict], wo: list[dict], ip5: list[dict]
) -> dict[str, int]:
    """Fold the two international sheets onto the records they belong to.

    They join on different keys because they are indexed differently at source: the WO
    sheet shares the register's PatBase family numbering, while the IP5 sheet carries no
    PatBase number at all and is matched on the UA patent number it names. That asymmetry
    was measured, not assumed — joining IP5 on family number resolves *zero* rows.

    Citation counts are summed across a family's satellite rows rather than taken from
    one: a family with a WO publication and an EP grant has two rows, and either alone
    understates the invention's citation record.
    """
    by_family: dict[str, list[dict]] = {}
    by_number: dict[str, dict] = {}
    for rec in records:
        if rec.get("family_number"):
            by_family.setdefault(rec["family_number"], []).append(rec)
        if rec.get("patent_number"):
            by_number[rec["patent_number"]] = rec

    stats = {"wo_matched": 0, "wo_unmatched": 0, "ip5_matched": 0, "ip5_unmatched": 0}

    def _cite(rec: dict, row: dict) -> None:
        for key, col in (
            ("forward_citations", "Number of Forward Citations"),
            ("backward_citations", "Number of Backward Citations"),
        ):
            value = _as_int(row.get(col))
            if value is not None:
                rec[key] = (rec[key] or 0) + value

    for row in wo:
        family = _text(row.get("Family Number"))
        targets = by_family.get(family or "", [])
        if not targets:
            stats["wo_unmatched"] += 1
            continue
        stats["wo_matched"] += 1
        for rec in targets:
            _cite(rec, row)
            if "WO" not in rec["international_offices"]:
                rec["international_offices"].append("WO")

    for row in ip5:
        number = _text(row.get("Patent Number")) or _text(
            row.get("DOCDB Patent Number")
        )
        rec = by_number.get(number or "")
        if rec is None:
            stats["ip5_unmatched"] += 1
            continue
        stats["ip5_matched"] += 1
        _cite(rec, row)
        scores = {
            name: _as_float(row.get(f"Patent Valuation Score {col}"))
            for name, col in (
                ("overall", "Overall"),
                ("citation", "Citation"),
                ("technology", "Technology"),
                ("legal", "Legal"),
                ("commercial", "Commercial"),
            )
        }
        if any(v is not None for v in scores.values()):
            rec["valuation"] = scores
        for office in _offices(_split(row.get("Member List"))):
            if office not in rec["international_offices"]:
                rec["international_offices"].append(office)

    for rec in records:
        for office in _offices(rec.get("family_members") or []):
            if office not in rec["international_offices"]:
                rec["international_offices"].append(office)
    return stats


def attach_citation_percentiles(records: list[dict]) -> int:
    """Rank forward citations within each cohort and store the percentile.

    Ranked within the cohort rather than across the register, for the reason the
    bibliographic corpus ranks within the publication year: a 2013 patent has had a decade
    to be cited and a 2022 one has not, so a raw count compares age as much as impact.

    Only records that *have* a count are ranked, and the percentile is their rank among
    that set. Records with no international sibling keep ``None``: they were never
    observed by a citing index at all, which is missing data, and calling it a zero
    percentile would rank the whole national register below its own international tail.
    """
    cohorts: dict[int, list[dict]] = {}
    for rec in records:
        if rec.get("forward_citations") is not None and rec.get("year") is not None:
            cohorts.setdefault(int(rec["year"]), []).append(rec)
    ranked = 0
    for rows in cohorts.values():
        rows.sort(key=lambda r: r["forward_citations"])
        n = len(rows)
        if n == 1:
            rows[0]["citation_percentile"] = 0.5
            ranked += 1
            continue
        for i, rec in enumerate(rows):
            rec["citation_percentile"] = i / (n - 1)
            ranked += 1
    return ranked


def build_dataset(
    data_dir: Path | str = DEFAULT_PATENTS_DIR,
    out_dir: Path | str = Path("datasets/patents"),
    *,
    axis: str = DEFAULT_AXIS,
    coverage: float | None = None,
    stratify: str | None = "origin",
) -> dict:
    """Parse the workbooks into per-cohort JSONL shards and a build report.

    The shard layout is ``records/<year>.jsonl`` — the same one the bibliographic
    datasets use, which is what lets ``horizon.ingest.wos.iter_records`` and every
    analysis command read a patent dataset without knowing it is one.

    **Every cohort is written, including the incomplete ones.** The completeness estimate
    is recorded in the build report and applied when the analysis selects years, not by
    deleting rows here: the trailing cohorts are real grants, and a question about the
    register's recent filings needs them even though a fitted trend must not use them.
    """
    from horizon.patents.completeness import DEFAULT_COVERAGE

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if axis not in YEAR_AXES:
        raise ValueError(f"unknown axis {axis!r} (want {', '.join(YEAR_AXES)})")
    coverage = DEFAULT_COVERAGE if coverage is None else coverage

    main = _one(data_dir, "*FinalResult*.xlsx", "the register")
    records: list[dict] = []
    dropped = 0
    for raw in read_main(main):
        rec = normalize_row(raw, axis=axis)
        if rec is None:
            dropped += 1
            continue
        records.append(rec)

    wo_path = _one(data_dir, "*WIPO*.xlsx", "the WO sheet", required=False)
    ip5_path = _one(data_dir, "*ip5*.xlsx", "the IP5 sheet", required=False)
    joins = attach_satellites(
        records,
        read_satellite(wo_path) if wo_path else [],
        read_satellite(ip5_path) if ip5_path else [],
    )
    ranked = attach_citation_percentiles(records)
    keep, estimates = complete_years(records, axis=axis, coverage=coverage, by=stratify)
    drift = lag_drift(records, axis=axis, by=stratify)

    shard_dir = out_dir / RECORDS_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in shard_dir.glob("*.jsonl"):
        stale.unlink()
    counts: dict[int, int] = {}
    handles: dict[int, Any] = {}
    try:
        for rec in records:
            year = rec.get("year")
            if year is None:
                continue
            year = int(year)
            if year not in handles:
                handles[year] = (shard_dir / f"{year}.jsonl").open(
                    "w", encoding="utf-8"
                )
            handles[year].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[year] = counts.get(year, 0) + 1
    finally:
        for fh in handles.values():
            fh.close()

    report = {
        "records": len(records),
        "dropped": dropped,
        "axis": axis,
        "shards": dict(sorted(counts.items())),
        "coverage": {str(y): round(c, 4) for y, c in sorted(estimates.items())},
        "reportable_years": keep,
        "coverage_floor": coverage,
        "citation_percentiles": ranked,
        # The assumption every coverage figure rests on, reported so it can be checked:
        # a lengthening lag means those figures are upper bounds, not estimates.
        "median_lag": {
            str(group): {str(y): round(v, 2) for y, v in sorted(series.items())}
            for group, series in sorted(drift.items(), key=lambda kv: str(kv[0]))
        },
        "stratified_by": stratify,
        "joins": joins,
        "sources": [p.name for p in (main, wo_path, ip5_path) if p],
    }
    (out_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _one(root: Path, pattern: str, what: str, *, required: bool = True) -> Path | None:
    """The single workbook matching ``pattern``, or a readable failure.

    Matching by pattern rather than by exact name so a re-export with a new date stamp
    still builds; refusing to guess when several match, because picking one silently is
    how a build quietly analyses last month's register.
    """
    found = sorted(p for p in root.glob(pattern) if not p.name.startswith("~$"))
    if not found:
        if required:
            raise FileNotFoundError(f"no workbook for {what} ({root}/{pattern})")
        return None
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise ValueError(f"several workbooks match {what} ({pattern}): {names}")
    return found[0]
