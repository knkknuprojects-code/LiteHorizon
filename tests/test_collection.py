"""Tests for the collection/ WoS+Scopus pipeline: build (xlsx -> shards) and WosSource.

No ``openpyxl`` and no real workbooks: the sheet reader (``iter_rows``) is monkeypatched
with canned header/row tuples, so the layout resolution, defect repairs, dedup/merge and
sharding are all exercised on synthetic rows that mirror the real files' quirks.
"""

import json

import pytest

import horizon.cli as cli
import horizon.ingest.collection as collection
from horizon.ingest.base import attention_of
from horizon.ingest.collection import (
    MAX_AUTHOR_CHARS,
    SMALL_COHORT_FLAG,
    ColumnMap,
    attach_citation_percentiles,
    cited_max,
    build_dataset,
    merge_records,
    norm_header,
    normalize_doi,
    normalize_row,
    rank_percentiles,
    resolve_columns,
    title_key,
)
from horizon.ingest.wos import WosSource, item_text, parse_years, shard_years

# The real 2016–2018 header, verbatim (irregular spaces included).
HEADER_2016 = [
    " Прізвище Автора / Authors",
    "Назва публікації /  Title ",
    "Рік",
    "Назва видання / Source title",
    "Цитування (Scopus) / Cited by",
    " DOI",
    "(Scopus) Link",
    "Резюме / Abstract",
    "Ключові слова / Author Keywords",
    "Тип документу / Document Type",
    " Галузь науки (Scopus/WoS)",
    None,
    "(Export) Times Cited, WoS Core",
    "(Export) Times Cited, All Databases",
    "(Export) WoS Categories",
]

# 2019 renames every column with a "(Scopus) " prefix and calls the subject area "Branch".
HEADER_2019 = [
    "(Scopus) Authors",
    "(Scopus) Title",
    "(Scopus) Year",
    "(Scopus) Source title",
    "(Scopus) Cited by",
    "(Scopus) DOI",
    "(Scopus) Link",
    "(Scopus) Abstract",
    "(Scopus) Document Type",
    "(Scopus) Branch",
    "",
    "(Export) Times Cited, WoS Core",
    "(Export) Times Cited, All Databases",
    "(Export) WoS Categories",
]

# 2015's 18-column layout: no Cited by, five extra "(Export)" WoS-side columns.
HEADER_2015 = [
    " Прізвище Автора / Authors",
    "Назва публікації /  Title ",
    "Рік",
    "Назва видання / Source title",
    " DOI",
    "(Scopus) Link",
    "Резюме / Abstract",
    "Ключові слова / Author Keywords",
    "Тип документу / Document Type",
    " Галузь науки (Scopus/WoS)",
    " Прізвище Автора / (Export) Authors",
    "Назва публікації /  (Export) Article Title",
    "Назва видання / (Export) Source Title",
    "Резюме / (Export) Abstract",
    "Цитування WoS Core / (Export) Times Cited, WoS Core",
    "Цитування Всі бази / (Export) Times Cited, All Databases",
    "DOI / (Export) DOI",
    "Категорії WoS  /  (Export) WoS Categories",
]

# 2022–2024: no Cited by, plus an "(Export) DOI Link" flag column that is not a DOI.
HEADER_2023 = [
    " Прізвище Автора / Authors",
    "Назва публікації /  Title ",
    "Рік",
    "Назва видання / Source title",
    " DOI",
    "(Scopus) Link",
    "Резюме / Abstract",
    "Ключові слова / Author Keywords",
    "Тип документу / Document Type",
    " Галузь науки (Scopus/WoS)",
    "(Export) Times Cited, WoS Core",
    "(Export) Times Cited, All Databases",
    "(Export) DOI Link",
    "(Export) WoS Categories",
]


def row_2016(**over):
    """A plausible 2016 row (Scopus + WoS, i.e. the 'both' stratum)."""
    row = [
        "Pirog T.; Shulyakova M.",
        "Biosurfactant synthesis on a byproduct of biodiesel production",
        2016,
        "Food and Bioproducts Processing",
        6,
        "10.1016/j.fbp.2013.09.003",
        "https://www.scopus.com/pages/publications/84922260096?origin=resultslist",
        "The possibility of using technical glycerol as a substrate was investigated.",
        "Surfactant; Waste glycerol",
        "Article",
        "Chemical Engineering; Energy",
        "  ",
        10,
        21,
        "Biotechnology & Applied Microbiology",
    ]
    for idx, value in over.items():
        row[int(idx)] = value
    return tuple(row)


# --- header + column resolution -----------------------------------------------


def test_norm_header_collapses_irregular_whitespace():
    assert norm_header("Назва публікації /  Title ") == "Назва публікації / Title"
    assert norm_header(None) == ""


def test_resolve_columns_on_the_2016_layout():
    cols = resolve_columns([norm_header(h) for h in HEADER_2016])
    assert (cols.authors, cols.title, cols.year, cols.source_title) == (0, 1, 2, 3)
    assert (cols.cited_by, cols.doi, cols.link, cols.abstract) == (4, 5, 6, 7)
    assert (cols.keywords, cols.doc_type, cols.subject_area) == (8, 9, 10)
    assert (cols.wos_core, cols.wos_all, cols.wos_categories) == (12, 13, 14)
    assert not cols.has_export_block


def test_resolve_columns_handles_2019_scopus_prefixes():
    cols = resolve_columns([norm_header(h) for h in HEADER_2019])
    # Index 0 must survive resolution — an `or` chain would drop it as falsy.
    assert cols.authors == 0
    assert cols.year == 2  # "(Scopus) Year", not the bare "Рік"
    assert cols.doi == 5  # "(Scopus) DOI"
    assert cols.abstract == 7
    assert cols.subject_area == 9  # "(Scopus) Branch" stands in for Галузь науки
    assert cols.keywords is None  # 2019 has no keywords column at all


def test_resolve_columns_marks_the_2015_export_block():
    cols = resolve_columns([norm_header(h) for h in HEADER_2015])
    assert cols.has_export_block
    assert cols.doi == 4  # the bare " DOI", not "DOI / (Export) DOI" at 16
    assert (cols.export_title, cols.export_abstract, cols.export_doi) == (11, 13, 16)
    assert cols.cited_by is None  # 2015 has no Cited by column


def test_resolve_columns_ignores_the_export_doi_link_flag():
    cols = resolve_columns([norm_header(h) for h in HEADER_2023])
    assert cols.doi == 4
    assert cols.wos_categories == 13  # not the "(Export) DOI Link" column at 12


def test_resolve_columns_keeps_index_zero_for_every_layout():
    for header in (HEADER_2015, HEADER_2016, HEADER_2019, HEADER_2023):
        cols = resolve_columns([norm_header(h) for h in header])
        assert cols.authors == 0, header[0]


# --- row normalization --------------------------------------------------------


def _cols(header):
    return resolve_columns([norm_header(h) for h in header])


def test_normalize_row_maps_every_field():
    rec = normalize_row(row_2016(), _cols(HEADER_2016), 2016)
    assert rec["year"] == 2016
    assert rec["venue"] == "Food and Bioproducts Processing"
    assert rec["doi"] == "10.1016/j.fbp.2013.09.003"
    assert rec["url"].startswith("https://www.scopus.com/")
    assert rec["doc_type"] == "Article"
    assert rec["subject_areas"] == ["Chemical Engineering", "Energy"]
    assert rec["wos_categories"] == ["Biotechnology & Applied Microbiology"]
    assert rec["keywords"] == ["Surfactant", "Waste glycerol"]
    assert (rec["cited_by_scopus"], rec["cited_wos_core"], rec["cited_wos_all"]) == (
        6,
        10,
        21,
    )
    assert rec["provenance"] == "both"
    assert rec["author_count"] == 2
    assert rec["script"] == "latin"
    assert rec["flags"] == []


def test_normalize_row_reads_provenance_from_link_and_wos_counts():
    scopus_only = normalize_row(
        row_2016(**{"12": None, "13": None}), _cols(HEADER_2016), 2016
    )
    wos_only = normalize_row(row_2016(**{"6": None}), _cols(HEADER_2016), 2016)
    assert scopus_only["provenance"] == "scopus"
    assert wos_only["provenance"] == "wos"


def test_normalize_row_falls_back_to_doi_org_url():
    rec = normalize_row(row_2016(**{"6": None}), _cols(HEADER_2016), 2016)
    assert rec["url"] == "https://doi.org/10.1016/j.fbp.2013.09.003"


def test_normalize_row_treats_whitespace_only_cells_as_empty():
    # The unnamed spacer column holds "  ", which is truthy in Python.
    rec = normalize_row(row_2016(**{"4": "   "}), _cols(HEADER_2016), 2016)
    assert rec["cited_by_scopus"] is None


def test_normalize_row_coerces_float_year_and_counts():
    rec = normalize_row(row_2016(**{"2": 2025.0, "12": 5.0}), _cols(HEADER_2016), 2016)
    assert rec["year"] == 2025
    assert rec["cited_wos_core"] == 5


def test_normalize_row_dates_by_filename_when_year_unusable():
    rec = normalize_row(row_2016(**{"2": "Article"}), _cols(HEADER_2016), 2018)
    assert rec["year"] == 2018
    assert "year-from-filename" in rec["flags"]


def test_normalize_row_drops_no_abstract_sentinel():
    rec = normalize_row(
        row_2016(**{"7": "[No abstract available]"}), _cols(HEADER_2016), 2016
    )
    assert rec["abstract"] is None


def test_normalize_row_drops_untitled_and_tab_corrupted_rows():
    cols = _cols(HEADER_2016)
    assert normalize_row(row_2016(**{"1": None}), cols, 2016) is None
    # 2019 holds a few unparsed TSV lines collapsed into the title cell.
    tsv = "Geospatial modelling of an iron ore deposit\t2019\tConference paper"
    assert normalize_row(row_2016(**{"1": tsv}), cols, 2016) is None


def test_normalize_row_caps_oversized_author_lists():
    rec = normalize_row(
        row_2016(**{"0": "A. Author; " * 5000}), _cols(HEADER_2016), 2016
    )
    assert len(rec["authors"]) <= MAX_AUTHOR_CHARS
    assert "authors-truncated" in rec["flags"]
    assert rec["author_count"] == 5000  # counted before truncation


def test_normalize_row_flags_cyrillic_script():
    rec = normalize_row(
        row_2016(**{"1": "Гістерезисні властивості ЕМГ-активності"}),
        _cols(HEADER_2016),
        2016,
    )
    assert rec["script"] == "cyrillic"


def test_normalize_row_fills_gaps_from_the_2015_export_columns():
    row = [
        "Author A.",
        None,  # Scopus-side title missing
        2015,
        "Some Journal",
        None,  # Scopus-side DOI missing
        None,
        None,  # abstract missing
        None,
        None,
        None,
        "Author, A.",
        "AN EXPORTED TITLE",
        "SOME JOURNAL",
        "The exported abstract text.",
        3,
        4,
        "10.1000/exported",
        "Zoology",
    ]
    rec = normalize_row(tuple(row), _cols(HEADER_2015), 2015)
    assert rec["title"] == "AN EXPORTED TITLE"
    assert rec["abstract"] == "The exported abstract text."
    assert rec["doi"] == "10.1000/exported"
    assert rec["provenance"] == "wos"


def test_normalize_row_repairs_the_2015_shifted_block():
    # Rows 11 201+ of the 2015 file are a WoS-only block written one column over:
    # the year column holds the source title and everything after it slides.
    row = (
        "Striuk, Andrii M.; Semerikov, Serhiy O.",
        "BACHELOR OF INFORMATICS COMPETENCE IN PROGRAMMING",
        "INFORMATION TECHNOLOGIES AND LEARNING TOOLS",
        "A comprehensive study on the problem of competence.",
        "10.33407/itlt.v46i2.1225",
        None,
        None,
        None,
        None,
        "Education & Educational Research",
        None,
        None,
        None,
        None,
        0,
        4,
        None,
        None,
    )
    rec = normalize_row(row, _cols(HEADER_2015), 2015)
    assert "shifted-block" in rec["flags"]
    assert rec["year"] == 2015  # from the filename, not the source title
    assert rec["venue"] == "INFORMATION TECHNOLOGIES AND LEARNING TOOLS"
    assert rec["abstract"] == "A comprehensive study on the problem of competence."
    assert rec["doi"] == "10.33407/itlt.v46i2.1225"
    assert rec["wos_categories"] == ["Education & Educational Research"]
    assert (rec["cited_wos_core"], rec["cited_wos_all"]) == (0, 4)
    assert rec["provenance"] == "wos"


def test_shifted_repair_only_applies_to_the_2015_layout():
    # A stray non-year in another layout must not re-map the whole row.
    rec = normalize_row(row_2016(**{"2": "Conference paper"}), _cols(HEADER_2016), 2016)
    assert "shifted-block" not in rec["flags"]
    assert rec["venue"] == "Food and Bioproducts Processing"


def test_normalize_row_reads_the_2019_layout():
    row = (
        "Wadi M.; Ivanik O.; Third A.",
        "Geospatial modelling of an iron ore deposit",
        2019,
        "Journal of Geology",
        4,
        "10.1000/xyz",
        "https://www.scopus.com/pages/publications/1?origin=resultslist",
        "A geospatial model of the deposit is presented.",
        "Conference paper",
        "Environmental Science",
        None,
        2,
        3,
        "Geology",
    )
    rec = normalize_row(row, _cols(HEADER_2019), 2019)
    assert rec["authors"] == "Wadi M.; Ivanik O.; Third A."
    assert rec["author_count"] == 3
    assert rec["year"] == 2019
    assert rec["doc_type"] == "Conference paper"
    assert rec["subject_areas"] == ["Environmental Science"]  # from "(Scopus) Branch"
    assert rec["cited_by_scopus"] == 4
    assert rec["provenance"] == "both"


# --- dedup keys + merge -------------------------------------------------------


def test_normalize_doi_strips_prefixes_and_case():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi: 10.1000/abc") == "10.1000/abc"
    assert normalize_doi("  ") is None


def test_title_key_ignores_case_and_punctuation_but_rejects_stubs():
    assert title_key("Mobility: A Systems Approach!") == title_key(
        "mobility a systems approach"
    )
    assert title_key("Errata") is None  # too short to be a safe key


def test_merge_records_recovers_a_fuller_record():
    cols = _cols(HEADER_2016)
    scopus = normalize_row(row_2016(**{"7": None, "12": None, "13": None}), cols, 2016)
    wos = normalize_row(
        row_2016(**{"6": None, "8": None, "10": "Energy; Physics", "12": 12, "13": 30}),
        cols,
        2016,
    )
    wos["source_files"] = [2017]
    merged = merge_records(scopus, wos)
    assert merged["abstract"]  # filled from the WoS row
    assert merged["provenance"] == "both"  # two strata widen to both
    assert merged["subject_areas"] == ["Chemical Engineering", "Energy", "Physics"]
    assert (merged["cited_wos_core"], merged["cited_wos_all"]) == (12, 30)
    assert merged["source_files"] == [2016, 2017]


# --- build_dataset ------------------------------------------------------------


@pytest.fixture
def fake_workbooks(tmp_path, monkeypatch):
    """Two empty .xlsx files whose rows are served by a patched ``iter_rows``."""
    src = tmp_path / "collection"
    src.mkdir()
    (src / "Дедублікація_WoS_Scopus_2016.xlsx").touch()
    (src / "Дедублікація_WoS_Scopus_2017.xlsx").touch()
    (src / "~$lockfile.xlsx").touch()  # Excel lock file — must be skipped

    def serve(rows_by_year):
        def fake_iter_rows(path):
            year = int(path.stem[-4:])
            header = [norm_header(h) for h in HEADER_2016]
            for row in rows_by_year.get(year, []):
                yield header, row

        monkeypatch.setattr("horizon.ingest.collection.iter_rows", fake_iter_rows)
        return src

    return serve


def test_build_dataset_shards_by_publication_year(fake_workbooks, tmp_path):
    src = fake_workbooks(
        {
            2016: [row_2016(), row_2016(**{"1": "Second paper title", "5": "10.1/b"})],
            # The 2017 workbook holds a paper dated 2018 — it belongs in 2018.jsonl.
            2017: [row_2016(**{"1": "Third paper", "2": 2018, "5": "10.1/c"})],
        }
    )
    report = build_dataset(src, tmp_path / "ds")
    shards = sorted(p.name for p in (tmp_path / "ds" / "records").glob("*.jsonl"))
    assert shards == ["2016.jsonl", "2018.jsonl"]
    assert report.per_year == {"2016": 2, "2018": 1}
    assert report.records == 3
    assert report.files == [
        "Дедублікація_WoS_Scopus_2016.xlsx",
        "Дедублікація_WoS_Scopus_2017.xlsx",
    ]


def test_build_dataset_merges_duplicates_across_workbooks(fake_workbooks, tmp_path):
    # The same DOI in two workbooks, once Scopus-only and once WoS-only — the
    # residue the workbooks' own deduplication missed.
    src = fake_workbooks(
        {
            2016: [row_2016(**{"7": None, "12": None, "13": None})],
            2017: [row_2016(**{"6": None, "12": 12, "13": 30})],
        }
    )
    report = build_dataset(src, tmp_path / "ds")
    assert (report.records, report.merged) == (1, 1)
    record = json.loads((tmp_path / "ds" / "records" / "2016.jsonl").read_text())
    assert record["provenance"] == "both"
    assert record["abstract"]
    assert record["cited_wos_core"] == 12
    assert record["source_files"] == [2016, 2017]


def test_build_dataset_dedupes_by_title_without_a_doi(fake_workbooks, tmp_path):
    src = fake_workbooks(
        {
            2016: [row_2016(**{"5": None})],
            2017: [row_2016(**{"5": None, "2": 2016})],
        }
    )
    report = build_dataset(src, tmp_path / "ds")
    assert (report.records, report.merged) == (1, 1)


def test_build_dataset_years_filter_selects_workbooks(fake_workbooks, tmp_path):
    src = fake_workbooks({2016: [row_2016()], 2017: [row_2016(**{"5": "10.1/b"})]})
    report = build_dataset(src, tmp_path / "ds", years=[2017])
    assert report.files == ["Дедублікація_WoS_Scopus_2017.xlsx"]
    assert report.records == 1


def test_build_dataset_writes_a_report_file(fake_workbooks, tmp_path):
    src = fake_workbooks({2016: [row_2016(), row_2016(**{"1": None})]})
    report = build_dataset(src, tmp_path / "ds")
    saved = json.loads((tmp_path / "ds" / "build-report.json").read_text())
    assert saved["rows"] == 2
    assert saved["records"] == 1
    assert saved["dropped_no_title"] == 1
    assert saved["with_abstract"] == 1
    assert saved["per_provenance"] == {"both": 1}
    assert saved["per_script"] == {"latin": 1}
    assert report.as_dict() == saved


def test_build_dataset_rejects_a_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_dataset(tmp_path / "nope", tmp_path / "ds")


# --- WosSource ----------------------------------------------------------------


def _shard(dataset, year, count, **over):
    """Write ``count`` synthetic records into ``dataset/records/<year>.jsonl``."""
    records_dir = dataset / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    with (records_dir / f"{year}.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(count):
            record = {
                "title": f"Paper {year}-{i} on emerging methods",
                "abstract": f"An abstract for paper {i} describing the method.",
                "authors": "Author A.; Author B.",
                "author_count": 2,
                "year": year,
                "venue": "Some Journal",
                "doi": f"10.1000/{year}-{i}",
                "url": f"https://doi.org/10.1000/{year}-{i}",
                "doc_type": "Article",
                "subject_areas": ["Computer Science"],
                "wos_categories": [],
                "keywords": [],
                "cited_by_scopus": i,
                "cited_wos_core": None,
                "cited_wos_all": None,
                "provenance": "scopus",
                "script": "latin",
                "source_files": [year],
                "flags": [],
            }
            record.update(over)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dataset


def test_parse_years_accepts_single_range_and_list():
    assert parse_years("2019") == [2019]
    assert parse_years("2015-2018") == [2015, 2016, 2017, 2018]
    assert parse_years("2015-2016,2023") == [2015, 2016, 2023]


@pytest.mark.parametrize("spec", ["", "20xx", "2018-2015", "2015..2018"])
def test_parse_years_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_years(spec)


def test_shard_years_lists_built_years(tmp_path):
    _shard(tmp_path / "ds", 2016, 1)
    _shard(tmp_path / "ds", 2019, 1)
    (tmp_path / "ds" / "records" / "notes.jsonl").write_text("", encoding="utf-8")
    assert shard_years(tmp_path / "ds" / "records") == [2016, 2019]


def test_item_text_joins_title_and_abstract():
    text = item_text({"title": "A title worth watching", "abstract": "The detail."})
    assert text == "A title worth watching. The detail."
    # A title already ending in punctuation isn't given a second full stop.
    question = {"title": "Why does this matter now?", "abstract": "Because of scale."}
    assert item_text(question) == "Why does this matter now? Because of scale."
    # Abstract-less records fall back to the title alone.
    assert item_text(
        {"title": "A standalone title with enough length", "abstract": None}
    )


def test_wos_source_yields_items_labelled_by_year(tmp_path):
    _shard(tmp_path / "ds", 2019, 2)
    items = list(WosSource(tmp_path / "ds").fetch())
    assert [i.source for i in items] == ["wos-scopus-2019"] * 2
    assert items[0].published_at == "2019-01-01"
    assert items[0].lang == "en"
    assert items[0].url == "https://doi.org/10.1000/2019-0"
    assert items[0].text.startswith("Paper 2019-0 on emerging methods.")
    assert items[0].meta["doi"] == "10.1000/2019-0"
    assert items[0].meta["date_precision"] == "year"
    assert items[0].meta["subject_areas"] == ["Computer Science"]


def test_wos_source_leaves_lang_unset_for_cyrillic_records(tmp_path):
    _shard(tmp_path / "ds", 2019, 1, script="cyrillic")
    (item,) = list(WosSource(tmp_path / "ds").fetch())
    assert item.lang is None


def test_wos_source_selects_years(tmp_path):
    _shard(tmp_path / "ds", 2016, 3)
    _shard(tmp_path / "ds", 2019, 2)
    items = list(WosSource(tmp_path / "ds", years=[2019]).fetch())
    assert len(items) == 2
    assert {i.source for i in items} == {"wos-scopus-2019"}


def test_wos_source_limit_takes_the_first_n_in_year_order(tmp_path):
    _shard(tmp_path / "ds", 2016, 5)
    _shard(tmp_path / "ds", 2019, 5)
    items = list(WosSource(tmp_path / "ds", limit=7).fetch())
    assert len(items) == 7
    assert [i.source for i in items[:5]] == ["wos-scopus-2016"] * 5


def test_wos_source_sample_is_deterministic_per_seed(tmp_path):
    _shard(tmp_path / "ds", 2016, 40)
    _shard(tmp_path / "ds", 2019, 40)

    def draw(seed):
        return [i.url for i in WosSource(tmp_path / "ds", sample=10, seed=seed).fetch()]

    assert len(draw(1)) == 10
    assert draw(1) == draw(1)  # same seed, same draw
    assert draw(1) != draw(2)  # a different seed draws differently


def test_wos_source_sample_larger_than_the_dataset_yields_everything(tmp_path):
    _shard(tmp_path / "ds", 2019, 3)
    assert len(list(WosSource(tmp_path / "ds", sample=50).fetch())) == 3


def test_wos_source_skips_records_with_no_usable_text(tmp_path):
    _shard(tmp_path / "ds", 2019, 1, title="tiny", abstract=None)
    assert list(WosSource(tmp_path / "ds").fetch()) == []


def test_wos_source_rejects_limit_with_sample(tmp_path):
    with pytest.raises(ValueError):
        WosSource(tmp_path / "ds", limit=5, sample=5)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"sample": -1}])
def test_wos_source_rejects_non_positive_selection(tmp_path, kwargs):
    with pytest.raises(ValueError):
        WosSource(tmp_path / "ds", **kwargs)


def test_wos_source_reports_an_unbuilt_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="collection-build"):
        list(WosSource(tmp_path / "ds").fetch())


def test_wos_source_reports_years_it_has_no_shards_for(tmp_path):
    _shard(tmp_path / "ds", 2019, 1)
    with pytest.raises(FileNotFoundError, match="2019"):
        list(WosSource(tmp_path / "ds", years=[2020]).fetch())


# --- CLI ----------------------------------------------------------------------


def test_cli_gather_wos_lands_a_corpus_slice(tmp_path, capsys):
    _shard(tmp_path / "ds", 2019, 5)
    code = cli.main(
        [
            "gather",
            "wos",
            "--data-dir",
            str(tmp_path / "ds"),
            "--years",
            "2019",
            "--limit",
            "2",
            "--no-raw",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2 item(s)" in out
    landed = tmp_path / "ds" / "corpus" / "wos-scopus-2019"
    assert len(list(landed.glob("*.json"))) == 2
    assert (landed / "manifest.jsonl").is_file()


def test_cli_gather_wos_rejects_limit_with_sample(tmp_path, capsys):
    _shard(tmp_path / "ds", 2019, 5)
    code = cli.main(
        [
            "gather",
            "wos",
            "--data-dir",
            str(tmp_path / "ds"),
            "--limit",
            "2",
            "--sample",
            "2",
        ]
    )
    assert code == 1
    assert "not both" in capsys.readouterr().err


def test_cli_gather_wos_reports_an_unbuilt_dataset(tmp_path, capsys):
    code = cli.main(["gather", "wos", "--data-dir", str(tmp_path / "ds")])
    assert code == 1
    assert "collection-build" in capsys.readouterr().err


def test_cli_collection_build_reports_counts(fake_workbooks, tmp_path, capsys):
    src = fake_workbooks({2016: [row_2016()], 2017: [row_2016(**{"5": "10.1/b"})]})
    code = cli.main(
        ["collection-build", "--path", str(src), "--out", str(tmp_path / "ds")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2 records" in out
    assert (tmp_path / "ds" / "records" / "2016.jsonl").is_file()
    assert (tmp_path / "ds" / "build-report.json").is_file()


def test_cli_collection_build_rejects_a_bad_year_spec(tmp_path, capsys):
    code = cli.main(["collection-build", "--path", str(tmp_path), "--years", "20xx"])
    assert code == 1
    assert "bad year spec" in capsys.readouterr().err


# --- citation counts + cohort percentiles -------------------------------------


def _rec(**over):
    """A minimal record dict for the citation helpers."""
    base = {
        "cited_by_scopus": None,
        "cited_wos_core": None,
        "cited_wos_all": None,
        "flags": [],
    }
    base.update(over)
    return base


def test_cited_max_takes_the_highest_tally():
    assert cited_max(_rec(cited_by_scopus=9, cited_wos_all=4)) == 9
    assert cited_max(_rec(cited_by_scopus=4, cited_wos_all=9)) == 9
    # 39 real records have all-databases < core, so neither WoS field can be preferred.
    assert cited_max(_rec(cited_wos_core=7, cited_wos_all=3)) == 7


def test_cited_max_distinguishes_zero_from_unknown():
    assert cited_max(_rec(cited_wos_core=0)) == 0  # genuinely uncited
    assert cited_max(_rec()) is None  # not measurable


def test_rank_percentiles_averages_ties():
    # [0,0,0,5]: the three zeros share ranks 0-2 -> (0 + 3/2)/4; the 5 -> (3 + 1/2)/4.
    assert rank_percentiles([0, 0, 0, 5]) == {0: 0.375, 5: 0.875}


def test_rank_percentiles_never_reaches_zero_or_one():
    # A real value must never be confusable with an absent one.
    out = rank_percentiles([0] * 50 + [1, 2, 900])
    assert 0.0 < min(out.values())
    assert max(out.values()) < 1.0


def test_rank_percentiles_single_value_is_the_midpoint():
    assert rank_percentiles([7]) == {7: 0.5}


def test_attach_citation_percentiles_ranks_only_counted_records():
    counted = [_rec(cited_wos_core=n) for n in range(60)]
    uncounted = [_rec() for _ in range(40)]
    cohort = counted + uncounted
    assert attach_citation_percentiles(cohort) == 60
    # The uncounted records are excluded from the ranking, not ranked as zero-cited.
    assert all(r["citation_percentile"] is None for r in uncounted)
    assert all(r["cited_max"] is None for r in uncounted)
    expected = rank_percentiles(list(range(60)))
    assert counted[0]["citation_percentile"] == round(expected[0], 6)
    assert counted[-1]["citation_percentile"] == round(expected[59], 6)


def test_attach_citation_percentiles_skips_a_small_cohort():
    # A "percentile" over a handful of records is noise dressed as a statistic.
    cohort = [_rec(cited_wos_core=n) for n in (0, 5, 40)]
    assert attach_citation_percentiles(cohort) == 3
    assert all(r["citation_percentile"] is None for r in cohort)
    assert all(r["cited_max"] is not None for r in cohort)
    assert all(SMALL_COHORT_FLAG in r["flags"] for r in cohort)


def test_attach_citation_percentiles_handles_a_cohort_with_no_counts():
    cohort = [_rec(), _rec()]
    assert attach_citation_percentiles(cohort) == 0
    assert all(r["citation_percentile"] is None for r in cohort)


def _cited(title, doi, count, *, year=2016):
    """A row whose only citation tally is ``count`` (the base row carries three)."""
    return row_2016(
        **{"1": title, "2": year, "5": doi, "4": None, "12": count, "13": None}
    )


def test_build_dataset_ranks_each_publication_year_separately(
    fake_workbooks, tmp_path, monkeypatch
):
    monkeypatch.setattr(collection, "MIN_CITATION_COHORT", 2)
    # Same raw count (5) in two cohorts with different distributions: in 2016 it is the
    # low value, in 2017 the high one, so the percentiles must differ.
    src = fake_workbooks(
        {
            2016: [
                _cited("A", "10.1/a", 5),
                _cited("B", "10.1/b", 90),
            ],
            2017: [
                _cited("C", "10.1/c", 5, year=2017),
                _cited("D", "10.1/d", 1, year=2017),
            ],
        }
    )
    build_dataset(src, tmp_path / "ds")
    by_year = {}
    for year in (2016, 2017):
        rows = [
            json.loads(line)
            for line in (tmp_path / "ds" / "records" / f"{year}.jsonl")
            .read_text()
            .splitlines()
        ]
        by_year[year] = {r["cited_max"]: r["citation_percentile"] for r in rows}
    assert by_year[2016][5] == 0.25  # the lower of the 2016 pair
    assert by_year[2017][5] == 0.75  # the higher of the 2017 pair


def test_build_dataset_ranks_after_merging_duplicates(
    fake_workbooks, tmp_path, monkeypatch
):
    monkeypatch.setattr(collection, "MIN_CITATION_COHORT", 1)
    # The same DOI at 5 and at 40 must rank as one record holding 40 — i.e. the cohort
    # pass runs after merge_records, not per row.
    src = fake_workbooks(
        {
            2016: [row_2016(**{"4": None, "12": 5, "13": None})],
            2017: [row_2016(**{"2": 2016, "4": None, "12": 40, "13": None})],
        }
    )
    report = build_dataset(src, tmp_path / "ds")
    record = json.loads((tmp_path / "ds" / "records" / "2016.jsonl").read_text())
    assert report.merged == 1
    assert record["cited_max"] == 40


def test_build_dataset_reports_citation_coverage(fake_workbooks, tmp_path):
    src = fake_workbooks(
        {
            2016: [
                row_2016(),
                row_2016(
                    **{"1": "B", "5": "10.1/b", "4": None, "12": None, "13": None}
                ),
            ]
        }
    )
    report = build_dataset(src, tmp_path / "ds")
    saved = json.loads((tmp_path / "ds" / "build-report.json").read_text())
    assert saved["with_citations"] == 1  # the second row has no tally at all
    assert saved["per_year_citations"] == {"2016": 1}
    assert saved["per_provenance_citations"] == {"both": 1}
    assert report.as_dict() == saved


# --- the attention convention -------------------------------------------------


def test_wos_source_meta_carries_the_attention_signal(tmp_path):
    _shard(tmp_path / "ds", 2019, 1, citation_percentile=0.87, cited_max=12)
    (item,) = list(WosSource(tmp_path / "ds").fetch())
    assert item.meta["attention"] == 0.87
    assert item.meta["cited_max"] == 12
    assert attention_of(item) == 0.87


def test_wos_source_meta_attention_is_none_for_a_legacy_shard(tmp_path):
    # _shard's default record has neither key — i.e. a shard built before percentiles.
    _shard(tmp_path / "ds", 2019, 1)
    (item,) = list(WosSource(tmp_path / "ds").fetch())
    assert item.meta["attention"] is None
    assert attention_of(item) is None
