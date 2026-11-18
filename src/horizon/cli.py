"""Command-line entry point. Subcommands:

    horizon scan path/to/text.txt          # scan one text into a structured analysis
    horizon scan file.txt --strategy staged --json --date 2026-06-30
    horizon trends                          # cluster stored analyses into rising trends
    horizon trends --cluster-method agglomerative --min-size 2 --json
    horizon export --approved-only          # write SFT (prompt, completion) data
    horizon dataset-stats                   # audit training-set readiness
    horizon corpus-stats                    # per-source corpus counts + language mix
    horizon compare                         # stand-in vs teacher: label-agreement report
    horizon approve <hash>... | --all       # mark records vetted for training
    horizon gather rss --url ... --out corpus/    # gather texts into a corpus
    horizon gather local --path ./docs
    horizon gather feeds --config feeds.toml      # gather every feed listed in a config
    horizon gather arxiv --query "cat:cs.CL"      # gather recent arXiv results for a query
    horizon gather url --url-file urls.txt        # gather main text of web pages (robots-aware)
    horizon gather corpus --path datasets/tech/corpus --scan   # re-scan a landed corpus (offline)
    horizon gather feeds --scan --data-dir datasets/tech       # gather+scan into a per-topic dataset
    horizon collection-build --data-dir datasets/wos    # collection/*.xlsx -> per-year shards
    horizon gather wos --data-dir datasets/wos --years 2019 --limit 20   # slice 20 records
    horizon trends --window 2y --recency-scale 365   # score a corpus dated by year only

`scan` caches/persists each analysis to ``data/analyses.jsonl``; `trends` reads the
whole store, embeds + clusters the analyses, and prints the rising trends. `export`,
`dataset-stats` and `approve` are the Phase-4 dataset-building tools (pure, no model);
`gather` is the Phase-5 corpus-gathering front end (lands text into ``corpus/``);
`collection-build` parses the ``collection/`` WoS+Scopus workbooks into per-year JSONL
shards that `gather wos` then slices (all of it, one year, or N records).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from horizon.analysis import Analysis, content_hash
from horizon.config import Settings
from horizon import figures
from horizon.interpret import VOCAB as _INTERPRET_VOCAB
from horizon.lenses import DEFAULT_LENS_MIN_SIZE

# Pure stdlib (no openpyxl, no model), so the constants a --help needs cost nothing.
from horizon.patents.completeness import DEFAULT_AXIS, DEFAULT_COVERAGE, YEAR_AXES
from horizon.report import LARGE_TOPIC_SIZE
from horizon.prompts import PROMPT_VERSION
from horizon.sources import SourceStore
from horizon.store import JsonlStore
from horizon.trends import Trend

# --- dataset rooting ----------------------------------------------------------
# A "dataset" is one directory holding data/ (analyses + source texts), corpus/
# (landed normalized text + manifest) and raw/ (pre-normalization content). This
# lets you keep separate per-topic datasets (datasets/tech, datasets/policy, …) and
# later export them merged. The root comes from --data-dir, else $HORIZON_DATA_DIR,
# else the current dir (which reproduces the original data/ + corpus/ layout).


def _root(args) -> Path:
    """Dataset root from --data-dir, else $HORIZON_DATA_DIR, else the current dir."""
    return Path(
        getattr(args, "data_dir", None) or os.environ.get("HORIZON_DATA_DIR") or "."
    )


def _store(args) -> JsonlStore:
    """Analyses store (``<root>/data/analyses.jsonl``) for the resolved dataset root."""
    return JsonlStore(_root(args) / "data" / "analyses.jsonl")


def _source_store(args) -> SourceStore:
    """Source-text store (``<root>/data/sources.jsonl``) for the resolved root."""
    return SourceStore(_root(args) / "data" / "sources.jsonl")


def _embed_cache_path(args) -> Path:
    """Embedding sidecar (``<root>/data/embeddings.jsonl``) for the resolved root.

    Rooted like the stores rather than at the current dir: a full-corpus run writes
    one vector per analysis, so a cwd-relative sidecar would pile up outside the
    dataset it belongs to (and fill $HOME on a cluster).
    """
    return _root(args) / "data" / "embeddings.jsonl"


def _settings_from(args) -> Settings:
    """Env settings with every model/generation CLI override applied.

    One place, so `scan` and `gather --scan` cannot drift. A flag left at its ``None``
    default never overrides the config/env value.
    """
    settings = Settings.from_env()
    overrides: dict[str, object] = {}
    for flag, field in (
        ("model_id", "model_id"),
        ("strategy", "strategy"),
        ("device", "device"),
        ("dtype", "dtype"),
        ("device_map", "device_map"),
        ("max_memory", "max_memory"),
        ("attn_implementation", "attn_implementation"),
        ("batch_size", "batch_size"),
        ("max_input_tokens", "max_input_tokens"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[field] = value
    # store_true flags: only an explicit True is an override.
    if getattr(args, "load_in_4bit", False):
        overrides["load_in_4bit"] = True
    if getattr(args, "load_in_8bit", False):
        overrides["load_in_8bit"] = True
    if overrides:
        settings = replace(settings, **overrides)

    generation: dict[str, object] = {}
    if getattr(args, "max_new_tokens", None) is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    if getattr(args, "temperature", None) is not None:
        generation["temperature"] = args.temperature
    if generation:
        settings = replace(
            settings, generation=replace(settings.generation, **generation)
        )
    return settings


def interpret_domains():
    """Domain names the interpret prompts can be written in."""
    return set(_INTERPRET_VOCAB)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``horizon`` argparse tree: the shared --data-dir option plus the
    scan/trends/export/dataset-stats/corpus-stats/compare/approve/collection-build/gather
    subcommands."""
    parser = argparse.ArgumentParser(
        prog="horizon",
        description="Horizon-scanning over prepared texts (any landscape).",
    )
    # Shared across subcommands: which dataset directory to read/write.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root holding data/, corpus/, raw/ (default $HORIZON_DATA_DIR or "
        "current dir). Use per-topic roots, e.g. datasets/tech.",
    )
    # Shared by every subcommand that may load a model. Each flag defaults to None/
    # False so an unset flag never overrides the config/env value.
    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument(
        "--device",
        default=None,
        help="Inference device: cpu | cuda | cuda:N | mps (default: auto-detect).",
    )
    model_opts.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default=None,
        help="Weight dtype (default: bfloat16 on CUDA, float32 otherwise). Use "
        "float16 on pre-Ampere cards, which have no bfloat16.",
    )
    model_opts.add_argument(
        "--load-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load 4-bit (nf4) via bitsandbytes — CUDA only. Fits a large model on a "
        'small card; needs pip install -e ".[infer-gpu]".',
    )
    model_opts.add_argument(
        "--load-8bit",
        dest="load_in_8bit",
        action="store_true",
        help="Load 8-bit via bitsandbytes — CUDA only. The middle rung: the 12B weights "
        "are 22.3 GiB in bfloat16 against a 24 GiB card, which leaves nothing for the "
        "KV cache, and int8 halves that to about 11 GiB with room to batch.",
    )
    model_opts.add_argument(
        "--device-map",
        default=None,
        help='accelerate placement, e.g. "auto" — shards a model across GPUs/CPU.',
    )
    model_opts.add_argument(
        "--max-memory",
        default=None,
        help='Per-device budget for --device-map, e.g. "0=38GiB,cpu=64GiB".',
    )
    model_opts.add_argument(
        "--attn",
        dest="attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default=None,
        help="Attention kernel (default: the transformers default). An unavailable "
        "choice falls back with a warning.",
    )
    model_opts.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Conversations per model call (default 1). Raise on a GPU — batching is "
        "the throughput lever there; on CPU it buys nothing.",
    )
    model_opts.add_argument(
        "--max-new-tokens", type=int, default=None, help="Cap generated length."
    )
    model_opts.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature."
    )
    model_opts.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Truncate prompts to this many tokens, from the left (default: no limit).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan",
        parents=[common, model_opts],
        help="Scan one text file into a structured analysis.",
    )
    scan.add_argument("text_file", type=Path, help="Path to a UTF-8 text file.")
    scan.add_argument(
        "--model", dest="model_id", default=None, help="Override model id."
    )
    scan.add_argument(
        "--strategy",
        choices=["combined", "staged"],
        default=None,
        help="Extraction strategy (default from config/env: combined).",
    )
    scan.add_argument(
        "--date",
        dest="item_date",
        default=None,
        help="Item date (YYYY-MM-DD) for trend velocity; default is scan time.",
    )
    scan.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )
    scan.add_argument(
        "--no-cache", dest="use_cache", action="store_false", help="Ignore cache."
    )
    scan.add_argument(
        "--no-save", dest="save", action="store_false", help="Don't persist."
    )

    trends = sub.add_parser(
        "trends",
        parents=[common],
        help="Cluster stored analyses into rising trends.",
    )
    trends.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )
    trends.add_argument(
        "--cluster-method",
        choices=["greedy", "agglomerative"],
        default=None,
        help="Clustering backend (default from config/env: greedy).",
    )
    trends.add_argument(
        "--threshold", type=float, default=None, help="Cosine similarity threshold."
    )
    trends.add_argument(
        "--min-size", type=int, default=1, help="Drop clusters smaller than this."
    )
    trends.add_argument(
        "--now",
        default=None,
        help="Anchor scoring at this date (YYYY, YYYY-MM-DD or ISO datetime). "
        "Default: the newest item_date in the store.",
    )
    trends.add_argument(
        "--window",
        default=None,
        help="Velocity window: 7d (days) or 2y (calendar years). A year window buckets "
        "by publication-year cohort (default from config/env: 7d).",
    )
    trends.add_argument(
        "--recency-scale",
        type=int,
        default=None,
        help="Recency decay length in days: recency = 1/(1 + days_since/scale). "
        "Use 365 for a corpus dated only by publication year (default 1).",
    )
    trends.add_argument(
        "--top", type=int, default=None, help="Show only the top N trends."
    )
    trends.add_argument(
        "--records",
        type=Path,
        default=None,
        metavar="PATH",
        help="Built records to normalise velocity against, so a cluster that merely "
        "keeps pace with a growing corpus does not read as rising. Accepts either the "
        "dataset root or the records/ dir inside it.",
    )
    trends.add_argument(
        "--baseline-field",
        default=None,
        metavar="NAME",
        help="With --records, normalise against one subject area (e.g. Physics and "
        "Astronomy) instead of the whole corpus — the sharper comparison, since a theme "
        "holding a constant share of the corpus may be rising inside a shrinking field.",
    )

    export = sub.add_parser(
        "export",
        parents=[common],
        help="Export stored analyses as SFT (prompt, completion) data for fine-tuning.",
    )
    export.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "sft",
        help="Output directory for train.jsonl / eval.jsonl / manifest.json.",
    )
    export.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Emit only the first N trainable examples (tiny loop-validation set).",
    )
    export.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction held out for the eval split (by content hash; default 0.2).",
    )
    export.add_argument(
        "--no-filter",
        dest="filter_trainable",
        action="store_false",
        help="Keep degenerate/placeholder analyses (default: drop them).",
    )
    export.add_argument(
        "--approved-only",
        action="store_true",
        help="Export only records vetted via `horizon approve`.",
    )
    export.add_argument(
        "--source",
        default=None,
        metavar="SRC",
        help="Export only records with this provenance (e.g. teacher, human).",
    )

    stats = sub.add_parser(
        "dataset-stats",
        parents=[common],
        help="Audit the store's training-set readiness (counts, balance, gap).",
    )
    stats.add_argument(
        "--target",
        type=int,
        default=None,
        help="Ready-example target to measure the gap against (default 500).",
    )
    stats.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )

    corpus_stats = sub.add_parser(
        "corpus-stats",
        parents=[common],
        help="Summarize a landed corpus by source (counts + language mix, no model).",
    )
    corpus_stats.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus directory to summarize (default <data-dir>/corpus).",
    )
    corpus_stats.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )

    patents = sub.add_parser(
        "patents-build",
        parents=[common],
        help="Parse the patents/ workbooks into per-cohort JSONL shards.",
    )
    patents.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory holding the register workbook plus the WO and IP5 sheets "
        "(default patents/).",
    )
    patents.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dataset root to write records/<year>.jsonl + build-report.json into "
        "(default datasets/patents).",
    )
    patents.add_argument(
        "--axis",
        choices=sorted(YEAR_AXES),
        default=DEFAULT_AXIS,
        help="Which of a patent's three dates becomes the cohort year: priority "
        "(when the invention was claimed — the default), filing, or grant. This is a "
        "measurement decision, not a detail: a trend on grant year describes the "
        "office's output, one on priority year describes when the inventing happened.",
    )
    patents.add_argument(
        "--coverage",
        type=float,
        default=None,
        help="Estimated share of a cohort's eventual grants that must already have "
        f"issued for it to be reportable (default {DEFAULT_COVERAGE}). Grant lag "
        "truncates a run of recent cohorts, so they are written but excluded from the "
        "reportable range; see the build report.",
    )

    collection = sub.add_parser(
        "collection-build",
        parents=[common],
        help="Parse the collection/ WoS+Scopus workbooks into per-year JSONL shards.",
    )
    collection.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory of Дедублікація_WoS_Scopus_<YYYY>.xlsx workbooks "
        "(default collection/).",
    )
    collection.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dataset root to write records/<year>.jsonl + build-report.json into "
        "(default the --data-dir root).",
    )
    collection.add_argument(
        "--years",
        default=None,
        help="Only build these workbook years: 2019, 2015-2018, or 2019,2023. "
        "Bounds memory, but dedup then spans only the years built.",
    )

    report_p = sub.add_parser(
        "report",
        parents=[common],
        help="Build the corpus report: field development, cross-field interaction, "
        "per-field trends and corpus-wide trends, in both languages.",
    )
    report_p.add_argument(
        "--records",
        type=Path,
        default=None,
        metavar="PATH",
        help="Built records: the dataset root or the records/ dir inside it.",
    )
    report_p.add_argument(
        "--from-facts",
        type=Path,
        default=None,
        metavar="PATH",
        help="Re-render an earlier run from its facts.json instead of measuring a corpus "
        "again. The counting and the clustering are most of an hour on a full corpus and "
        "the tables are a minute, so a change to what is printed should not cost a "
        "re-measurement — and a re-measurement would arrive with numbers that had moved, "
        "mixed in with the formatting change. Only the flags that decide what is printed "
        "still apply; the ones that shape the facts were spent when the file was written.",
    )
    report_p.add_argument(
        "--page-break-areas",
        action="store_true",
        help="Start each subject area of section 3 on its own page. Section 3 is looked "
        "things up in rather than read through, and its tables otherwise run into one "
        "another; sections 4 and 5 are read in order and are left alone.",
    )
    report_p.add_argument(
        "--out", type=Path, default=Path("reports"), help="Directory to write into."
    )
    report_p.add_argument("--slug", default="corpus", help="Filename stem.")
    report_p.add_argument("--title", default=None, help="Report title.")
    report_p.add_argument(
        "--years",
        default=None,
        metavar="SPEC",
        help="Cohorts to include, e.g. 2015-2025.",
    )
    report_p.add_argument("--min-field-total", type=int, default=0)
    report_p.add_argument("--min-pair-total", type=int, default=0)
    report_p.add_argument("--top-fields", type=int, default=None)
    report_p.add_argument("--top-pairs", type=int, default=None)
    report_p.add_argument(
        "--top-trends",
        type=int,
        default=50,
        help="Topics kept per section in facts.json (default 50). Higher than what the "
        "tables print on purpose: the figures are drawn from the JSON, and a scatter "
        "over the top ten per subject area is a ranked sample small enough to mislead.",
    )
    report_p.add_argument(
        "--table-rows",
        type=int,
        default=10,
        help="Rows printed per topic table (default 10).",
    )
    report_p.add_argument(
        "--large-topic-size",
        type=int,
        default=LARGE_TOPIC_SIZE,
        metavar="N",
        help=f"Publications a topic needs to be ranked by decline (default "
        f"{LARGE_TOPIC_SIZE}). Below about a hundred that ranking is nothing but topics "
        "that stopped publishing outright, which crowds out the large ones losing "
        "share.",
    )
    report_p.add_argument(
        "--indicator-rows",
        type=int,
        default=20,
        help="Rows printed per indicator table (default 20). Separate from "
        "--table-rows because section 3 repeats its table for every subject area, "
        "while an indicator has exactly one table and can afford to be longer.",
    )
    report_p.add_argument(
        "--min-size", type=int, default=5, help="Drop clusters smaller than this."
    )
    report_p.add_argument(
        "--lens-min-size",
        type=int,
        default=None,
        metavar="N",
        help=f"Members a cluster needs before a lens ranks it (default "
        f"{DEFAULT_LENS_MIN_SIZE}). Higher than --min-size on purpose: a lens ranks on a "
        "per-cluster statistic, and a citation percentile over five records measures "
        "those five records.",
    )
    report_p.add_argument("--threshold", type=float, default=None)
    report_p.add_argument(
        "--cluster-method", default=None, choices=["greedy", "agglomerative"]
    )
    report_p.add_argument(
        "--window", default=None, metavar="SPEC", help="Velocity window, e.g. 2y."
    )
    report_p.add_argument("--anchor", default=None, metavar="DATE")
    report_p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the figures and their references in the report text.",
    )
    report_p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="The date stamped into the filenames (default: today). A run is identified "
        "by when its data was gathered, not by when a rebuild happened to be launched — "
        "without this, rebuilding a report after midnight writes a second, differently "
        "named set beside the first and the run appears to be two.",
    )
    report_p.add_argument(
        "--compose-by",
        default=None,
        metavar="FIELD",
        help="Split every topic across the values of this record field and rate each "
        "part against its own corpus-wide rate (e.g. applicant_type). Adds section 6, "
        "ranked by how far the groups disagree. Use where the corpus is not one "
        "population: an aggregate over groups with different profiles describes none of "
        "them, and decomposing keeps what filtering would throw away.",
    )
    report_p.add_argument(
        "--block-by",
        choices=["field", "none"],
        default="field",
        help="Whether clustering is blocked by subject area (default) or runs over the "
        "whole population at once. Blocking buys speed and costs a rule that a topic "
        "belongs to exactly one field; below roughly ten thousand records that trade "
        "stops paying, and where records carry several fields the rule is wrong. With "
        "'none' the clusters are global and section 3 scores each one inside every field "
        "its members belong to.",
    )
    report_p.add_argument(
        "--fit-years",
        default=None,
        metavar="SPEC",
        help="Cohorts the shares and slopes are derived from, when that is narrower than "
        "--years. The counts keep every cohort in --years, so an incomplete tail stays "
        "visible instead of being deleted; only the derived statistics avoid it. Needed "
        "wherever the trailing cohorts are still filling up — a partial cohort drags a "
        "fitted rate toward its stub, and is not a uniform sample of itself either, so "
        "it distorts shares toward whatever part of the corpus resolves fastest.",
    )
    report_p.add_argument(
        "--vocabulary",
        choices=["bibliographic", "patents"],
        default="bibliographic",
        help="Wording for the report's headings, columns and caveat. The arithmetic is "
        "the same either way; this decides whether the tables say 'publications' and "
        "name WoS and Scopus, or say 'patents' and name the register and its cut date.",
    )
    # The parser itself, so ``_run_report`` can ask what a flag's default was instead of
    # keeping a second copy of the defaults that would drift from these.
    report_p.set_defaults(_report_parser=report_p)
    report_p.add_argument(
        "--no-trends",
        action="store_true",
        help="Levels 1 and 2 only — no embedder, no clustering, seconds instead of "
        "minutes.",
    )

    interpret_p = sub.add_parser(
        "interpret",
        parents=[common, model_opts],
        help="A scientometrician's reading of a report, written by a local model. "
        "One model per run, writing its own file — merge them afterwards.",
    )
    interpret_p.add_argument(
        "--domain",
        choices=sorted(interpret_domains()),
        default="bibliographic",
        help="The nouns the prompt describes the corpus with. The machinery is the same "
        "either way; this decides whether the model is told it is reading publication "
        "abstracts from a WoS/Scopus corpus or patent abstracts from a register. Getting "
        "it wrong is invisible in the output — a generated reading is the one part of a "
        "report that cannot be checked by reading it.",
    )
    interpret_p.add_argument(
        "--facts",
        type=Path,
        required=True,
        metavar="PATH",
        help="a report's facts.json",
    )
    interpret_p.add_argument(
        "--model", dest="model_id", default=None, help="Override model id."
    )
    interpret_p.add_argument(
        "--out", type=Path, default=Path("reports"), help="directory to write into."
    )
    interpret_p.add_argument("--slug", default="corpus", help="Filename stem.")
    interpret_p.add_argument(
        "--lang", default="uk", choices=["en", "uk"], help="Language to write in."
    )
    interpret_p.add_argument(
        "--min-size",
        type=int,
        default=LARGE_TOPIC_SIZE,
        metavar="N",
        help=f"Publications a topic needs to get its own reading (default "
        f"{LARGE_TOPIC_SIZE}) — the same threshold section 4 uses to decide what is "
        "large enough to rank by decline.",
    )
    interpret_p.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Stop after N topics."
    )
    interpret_p.add_argument(
        "--pass",
        dest="passes",
        default="all",
        choices=["all", "corpus", "topics", "areas"],
        help="Which readings to generate (default all). Each is stored separately and "
        "resumable on its own, so a pass can be added to a finished run without "
        "regenerating the others.",
    )

    fields_p = sub.add_parser(
        "fields",
        parents=[common],
        help="Field development and cross-field interaction, from the taxonomy WoS and "
        "Scopus already assigned. Needs no model and no clustering.",
    )
    fields_p.add_argument(
        "--records",
        type=Path,
        required=True,
        metavar="PATH",
        help="Built records: the dataset root or the records/ dir inside it.",
    )
    fields_p.add_argument(
        "--key",
        default="subject_areas",
        choices=["subject_areas", "wos_categories"],
        help="Which taxonomy to count (default: subject_areas).",
    )
    fields_p.add_argument(
        "--years",
        default=None,
        metavar="SPEC",
        help="Cohorts to include, e.g. 2015-2025. Exclude a partial final year: it "
        "depresses every rate computed from it.",
    )
    fields_p.add_argument(
        "--min-total",
        type=int,
        default=0,
        help="Drop entries below this many records — a slope fitted to a handful of "
        "papers is noise wearing a number.",
    )
    fields_p.add_argument(
        "--top", type=int, default=None, help="Show only the top N of each table."
    )
    fields_p.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )

    compare = sub.add_parser(
        "compare",
        parents=[common],
        help="Compare two scans of the same corpus (e.g. stand-in vs teacher).",
    )
    compare.add_argument(
        "--baseline",
        default=None,
        metavar="SUBSTR",
        help="model_id substring for the baseline side (default: the non-Gemini "
        "model present).",
    )
    compare.add_argument(
        "--teacher",
        default=None,
        metavar="SUBSTR",
        help="model_id substring for the teacher side (default: the Gemini model "
        "present).",
    )
    compare.add_argument(
        "--json", dest="as_json", action="store_true", help="Emit raw JSON."
    )

    approve = sub.add_parser(
        "approve",
        parents=[common],
        help="Mark stored analyses as approved (vetted) for the training set.",
    )
    approve.add_argument(
        "hashes",
        nargs="*",
        help="content_hash values to approve (prefix match allowed).",
    )
    approve.add_argument(
        "--all",
        dest="approve_all",
        action="store_true",
        help="Approve every trainable record in the store.",
    )
    approve.add_argument(
        "--set-source",
        default=None,
        metavar="SRC",
        help="Also set provenance on the approved records (e.g. human, teacher).",
    )

    gather = sub.add_parser(
        "gather",
        parents=[common, model_opts],
        help="Gather texts from a source into a corpus/ directory (for training).",
    )
    gather.add_argument(
        "kind",
        choices=["rss", "local", "feeds", "arxiv", "url", "corpus", "wos", "patents"],
        help="Source: rss feed (--url), local dir (--path), feeds (--config), "
        "arxiv (--query), url (--url / --url-file), corpus (--path; replay a "
        "landed corpus, e.g. to re-scan with --scan), wos (the built "
        "collection/ dataset), or patents (the built patents/ dataset) — the "
        "last two slice with --years / --limit / --sample.",
    )
    gather.add_argument(
        "--url",
        default=None,
        help="Feed URL for `gather rss`, or a page URL for `gather url`.",
    )
    gather.add_argument(
        "--path", type=Path, default=None, help="Directory for `gather local`."
    )
    gather.add_argument(
        "--query", default=None, help="arXiv search query for `gather arxiv`."
    )
    gather.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Max results for `gather arxiv` (default 50).",
    )
    gather.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="File of page URLs (one per line, # comments ok) for `gather url`.",
    )
    gather.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between page fetches for `gather url` (rate limit; default 1.0).",
    )
    gather.add_argument(
        "--config",
        type=Path,
        default=Path("feeds.toml"),
        help="Feeds TOML file for `gather feeds` (default feeds.toml).",
    )
    gather.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Dataset root holding the built records/ shards for `gather wos` "
        "(default the --data-dir root).",
    )
    gather.add_argument(
        "--years",
        default=None,
        help="Publication years for `gather wos`: 2019, 2015-2018, or 2019,2023 "
        "(default: every built year).",
    )
    gather.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For `gather wos`: keep only the first N records (fast, year-ordered).",
    )
    gather.add_argument(
        "--sample",
        type=int,
        default=None,
        help="For `gather wos`: draw N records at random, reproducibly per --seed "
        "(use instead of --limit for a representative smoke test).",
    )
    gather.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for `gather wos --sample` (default 0 — same draw every run).",
    )
    gather.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Corpus directory for texts + manifest.jsonl (default <data-dir>/corpus).",
    )
    gather.add_argument(
        "--no-raw",
        dest="keep_raw",
        action="store_false",
        help="Don't retain raw pre-normalization content under raw/ (default: keep it).",
    )
    gather.add_argument(
        "--scan",
        dest="scan",
        action="store_true",
        help="Also scan each item into the store, dating it by the item's "
        "publication date (needs a model; idempotent — already-scanned items skip).",
    )
    gather.add_argument(
        "--model",
        dest="model_id",
        default=None,
        help="Model id for --scan (default from config/env). Use a gemini-* id "
        "(e.g. gemini-3.5-flash) to scan with the hosted API teacher.",
    )
    gather.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="With --scan, report progress to stderr every N items (0 disables). A "
        "multi-hour run must be distinguishable from a hung one.",
    )
    gather.add_argument(
        "--no-corpus",
        dest="write_corpus",
        action="store_false",
        help="With --scan, skip writing corpus/ records. A full pass would create one "
        "small file per item — an inode problem on a shared filesystem — and for "
        "`gather wos` it only duplicates the built JSONL shards.",
    )
    gather.add_argument(
        "--rescan",
        action="store_true",
        help="With --scan, re-analyse items already in the store, overwriting "
        "them (last write wins). Use for a teacher pass that replaces earlier "
        "stand-in labels with a stronger model's.",
    )
    return parser


# --- formatting ---------------------------------------------------------------


def format_report(analysis: Analysis) -> str:
    """Render one analysis as the human-readable CLI report (summary, signals,
    classification, entities, keywords)."""
    c = analysis.classification
    e = analysis.entities
    lines = ["Summary", f"  {analysis.summary}", "", "Key signals"]
    lines += [f"  - {s}" for s in analysis.signals] or ["  (none)"]
    lines += [
        "",
        "Classification",
        f"  theme:        {c.theme}",
        f"  maturity:     {c.maturity.value}",
        f"  impact:       {c.impact.value}",
        f"  time horizon: {c.time_horizon.value}",
        "",
        "Entities",
        f"  organizations: {', '.join(e.organizations) or '-'}",
        f"  people:        {', '.join(e.people) or '-'}",
        f"  technologies:  {', '.join(e.technologies) or '-'}",
        f"  locations:     {', '.join(e.locations) or '-'}",
        "",
        f"Keywords: {', '.join(analysis.keywords) or '-'}",
    ]
    return "\n".join(lines)


def format_trends(trends: list[Trend]) -> str:
    """Render ranked trends as the human-readable CLI listing (one block per trend
    with score/velocity/size, seen range, and top keywords)."""
    if not trends:
        return "No trends found."
    lines: list[str] = []
    for i, t in enumerate(trends, 1):
        seen = f"   seen: {t.first_seen} → {t.last_seen}   recency {t.recency}"
        if t.attention_n:
            seen += (
                f"   attention {t.attention} "
                f"(p{round((t.attention_pctile or 0) * 100)}, "
                f"n={t.attention_n}/{t.size})"
            )
        lines += [
            f"{i}. {t.label}  (score {t.score}, velocity {t.velocity}, size {t.size})",
            seen,
        ]
        # Lineage only says something once a cluster spans more than one cohort.
        if len(t.cohort_counts) > 1:
            curve = " ".join(f"{year}:{n}" for year, n in t.cohort_counts)
            lines.append(
                f"   seed: {t.seed_date}   curve: {curve}   "
                f"{t.growth:+.1f}/yr   breadth: {t.breadth} orgs"
            )
        lines += [
            f"   keywords: {', '.join(t.top_keywords) or '-'}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (the default 'now' for trend velocity)."""
    return datetime.now(timezone.utc).isoformat()


# --- scan ---------------------------------------------------------------------


def _run_scan(args) -> int:
    """`scan`: analyse one text file into a structured analysis and print it.

    Applies --model/--strategy/--max-new-tokens/--temperature overrides, serves a
    cache hit without loading a model (backfilling the source text so ``export`` sees
    it), and unless --no-save persists the analysis and its source text to the store."""
    if not args.text_file.is_file():
        print(f"error: file not found: {args.text_file}", file=sys.stderr)
        return 1
    text = args.text_file.read_text(encoding="utf-8")

    settings = _settings_from(args)
    generation = settings.generation

    store = _store(args)
    sources = _source_store(args)
    if args.use_cache:
        cached = store.get(content_hash(text), PROMPT_VERSION)
        if cached is not None:
            if args.save:
                sources.put(text)  # backfill source so export has the input
            _emit_analysis(cached, as_json=args.as_json)
            return 0

    # Imported here so `horizon --help` and cache hits work without a model stack.
    from horizon.extract import analyze_text
    from horizon.llm import make_client

    client = make_client(settings)
    try:
        analysis = analyze_text(
            text, client, settings=settings, config=generation, item_date=args.item_date
        )
    except RuntimeError as exc:  # includes AnalysisError
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.save:
        store.put(analysis)
        sources.put(text)
    _emit_analysis(analysis, as_json=args.as_json)
    return 0


def _emit_analysis(analysis: Analysis, *, as_json: bool) -> None:
    """Print an analysis to stdout — raw JSON when ``as_json``, else the CLI report."""
    if as_json:
        print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(analysis))


# --- trends -------------------------------------------------------------------


class _FixedClusterer:
    """A :class:`~horizon.clustering.Clusterer` that returns labels decided elsewhere.

    ``build_trends`` clusters as part of scoring, which is right when the clusters are
    its own to make. Levels 3 and 4 decide membership first — per field, then by merging
    across fields — so the scoring pass has to accept that decision rather than redo it.
    """

    def __init__(self, labels: list[int]) -> None:
        """Store the labels this clusterer will hand back."""
        self._labels = labels

    def cluster(self, vectors) -> list[int]:  # noqa: ARG002 - fixed by construction
        """Return the stored labels, ignoring the vectors."""
        return list(self._labels)


def _medoid(vectors, indices):
    """The member closest to the cluster's mean — its most representative document.

    A medoid rather than a centroid, and the difference is not stylistic. Averaging
    cancels the topic-specific directions and leaves what every document shares, so
    centroids collapse toward each other: measured on this corpus, random pairs of
    16-document centroids have a mean cosine of 0.809 and **100%** of them clear a 0.55
    threshold, against 0.204 and 1.1% for random pairs of documents. Merging blocks on
    centroids at the clustering threshold therefore merges nearly everything — it
    produced a single 301-member cluster out of 999 analyses before this changed.

    A medoid is an actual document, so one threshold keeps one meaning on either side of
    the block boundary.
    """
    import numpy as np

    from horizon.clustering import _unit

    rows = np.asarray([vectors[i] for i in indices], dtype=float)
    mean = _unit(rows.mean(axis=0))
    return rows[int(np.argmax(rows @ mean))]


def _pandoc(md_path, out_path, *, pdf: bool, lang: str = "uk") -> str:
    """Render one copy through the shared recipe in :mod:`horizon.render`."""
    from horizon.render import render

    return render(md_path, out_path, pdf=pdf, lang=lang)


def _reading_identity(settings) -> tuple[str, str]:
    """The filename tag and the human label for one generator.

    The quantization belongs in both. The same weights at 4-bit and at 8-bit are
    different generators, and without this they share a filename: the second run would
    open the first one's file, find every topic already present, and write nothing.
    """
    quant = "4bit" if settings.load_in_4bit else "8bit" if settings.load_in_8bit else ""
    stem = settings.model_id.rsplit("/", 1)[-1]
    if not quant:
        return stem, settings.model_id
    return f"{stem}-{quant}", f"{settings.model_id} ({quant})"


def _passes(asked: str) -> set[str]:
    """The readings a ``--pass`` value selects.

    A function because the first version was written inline as ``{"all", asked}``, which
    contains "all" whichever pass was asked for — so every gate downstream passed and the
    flag selected nothing.
    """
    return {"corpus", "topics", "areas"} if asked == "all" else {asked}


def _measure_facts(args):
    """Levels 1 to 6 over a built corpus: the expensive half of a report.

    Split out from :func:`_run_report` so the cheap half — rendering the tables — can
    also be reached from a facts file written by an earlier run. Returns ``None`` after
    printing the reason when the corpus cannot be read.
    """
    from horizon.fields import tabulate
    from horizon.ingest.wos import iter_records, parse_years, records_dir, shard_years
    from horizon.report import build_facts, disambiguate

    root = Path(args.records)
    shards = root if shard_years(root) else records_dir(root)
    if not shard_years(shards):
        print(
            f"error: no built record shards under {root} (nor in {root}/records).",
            file=sys.stderr,
        )
        return None
    try:
        years = parse_years(args.years) if args.years else None
        fit_years = parse_years(args.fit_years) if args.fit_years else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    if fit_years and years and not set(fit_years) <= set(years):
        outside = sorted(set(fit_years) - set(years))
        print(
            f"error: --fit-years names cohorts outside --years: "
            f"{', '.join(map(str, outside))}",
            file=sys.stderr,
        )
        return None

    tab = tabulate(iter_records(shards, years), years=years)
    if not tab.records:
        print("error: no records in the requested years.", file=sys.stderr)
        return None
    if years is None:
        years = _complete_years(tab)
        if years is not None:
            tab = tabulate(iter_records(shards, years), years=years)
    print(f"records {tab.records}, years {tab.years[0]}-{tab.years[-1]}")
    if fit_years:
        # Said out loud, because a reader of the tables cannot tell which cohorts a slope
        # was fitted over, and the two ranges differing is the whole point of the flag.
        shown = sorted(set(tab.years) - set(fit_years))
        print(
            f"  shares and slopes fitted over {fit_years[0]}-{fit_years[-1]}; "
            f"{', '.join(map(str, shown))} shown in the counts only"
        )

    field_trends: dict = {}
    overall: list = []
    if not args.no_trends:
        field_trends, overall = _build_levels_3_and_4(args, shards, tab, years)

    # Before the lenses, so every section names a cluster the same way — they share
    # these objects rather than copies.
    overall = disambiguate(overall)
    field_trends = {k: disambiguate(v) for k, v in field_trends.items()}
    lenses, meta_lenses = _apply_lenses(args, shards, overall, years)
    splits, pops = _field_splits(args, shards, fit_years or list(tab.years))
    facts = build_facts(
        tab,
        field_trends=field_trends,
        overall=overall,
        lenses=lenses,
        meta_lenses=meta_lenses,
        min_field_total=args.min_field_total,
        min_pair_total=args.min_pair_total,
        top_fields=args.top_fields,
        top_pairs=args.top_pairs,
        top_trends=args.top_trends,
        large_topic_size=args.large_topic_size,
        fit_years=fit_years,
        compositions=_compositions(args, shards, overall, tab),
        field_splits=splits,
        populations=pops,
    )

    return facts


def _run_report(args) -> int:
    """`report`: assemble and write the corpus report in both languages.

    Levels 1 and 2 are counting and need nothing else. Levels 3 and 4 cluster inside
    each field — blocking, which is exact where sampling is lossy — and then merge the
    per-field clusters by their medoids before ranking them together. The merge is not
    optional: without it a cross-disciplinary theme is counted once per field it spans
    and ranks spuriously high, which is precisely the class that matters most here.

    With ``--from-facts`` none of that runs: the facts are read back from an earlier
    run's JSON and only the rendering happens. The two paths meet at the same object, so
    a report re-rendered from a file is the same report with different tables, not a
    second measurement that happens to agree.
    """
    from horizon.report import ReportFacts, render_json, render_markdown

    if bool(args.records) == bool(args.from_facts):
        print(
            "error: pass either --records (measure a corpus) or --from-facts "
            "(re-render an earlier run), not both and not neither.",
            file=sys.stderr,
        )
        return 1
    if args.from_facts:
        # Every one of these was spent when the facts file was written; honouring one now
        # would mean re-measuring, and ignoring one would mean a table that quietly
        # disagrees with the flag that asked for it. So neither.
        spent = [
            "years",
            "fit_years",
            "min_field_total",
            "min_pair_total",
            "top_fields",
            "top_pairs",
            "top_trends",
            "large_topic_size",
            "min_size",
            "lens_min_size",
            "threshold",
            "cluster_method",
            "window",
            "anchor",
            "block_by",
            "compose_by",
            "no_trends",
        ]
        given = [
            d
            for d in spent
            if getattr(args, d, None) != args._report_parser.get_default(d)
        ]
        if given:
            flags = ", ".join("--" + d.replace("_", "-") for d in given)
            print(
                f"error: {flags} shape the facts, and --from-facts reads facts that are "
                "already shaped. Re-run with --records to change them.",
                file=sys.stderr,
            )
            return 1
        # The measurement already happened and is on disk; only the flags that decide
        # what is printed still have anything to act on.
        facts = ReportFacts.from_dict(
            json.loads(Path(args.from_facts).read_text(encoding="utf-8"))
        )
        print(
            f"re-rendering {args.from_facts}: {facts.records} records, "
            f"{facts.topics_total or len(facts.overall)} topics"
        )
    else:
        facts = _measure_facts(args)
    if facts is None:
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.date or date.today().isoformat()
    en = out_dir / f"{args.slug}-{stamp}.md"
    uk = out_dir / f"{args.slug}-{stamp}-ua.md"
    rows = args.table_rows
    # Drawn before the markdown is rendered, so the report references the figures that
    # exist rather than the ones that were planned. Without matplotlib it references
    # none, which is the honest outcome: a caption for a missing picture is worse than
    # no picture.
    slug = f"{args.slug}-{stamp}"
    drawn = _draw_figures(
        facts, out_dir, slug, enabled=not args.no_figures, vocabulary=args.vocabulary
    )
    for path, lang, folder in ((en, "en", slug), (uk, "uk", f"{slug}-ua")):
        path.write_text(
            render_markdown(
                facts,
                lang=lang,
                title=args.title,
                top_rows=rows,
                indicator_rows=args.indicator_rows,
                page_break_areas=args.page_break_areas,
                figures=figures.references(
                    drawn.get(lang, ()),
                    lang,
                    f"figures/{folder}",
                    _figure_text(args.vocabulary, lang),
                ),
                labels=_vocabulary(args.vocabulary, lang),
            ),
            encoding="utf-8",
        )
    facts_path = out_dir / f"{args.slug}-{stamp}-facts.json"
    facts_path.write_text(render_json(facts), encoding="utf-8")
    print(f"  wrote {en}\n  wrote {uk}\n  wrote {facts_path}")
    print(_pandoc(uk, uk.with_suffix(".pdf"), pdf=True))
    print(_pandoc(uk, uk.with_suffix(".docx"), pdf=False))
    return 0


def _draw_figures(
    facts, out_dir, slug: str, *, enabled: bool, vocabulary: str = "bibliographic"
) -> dict[str, list[str]]:
    """Draw both language sets under ``out_dir/figures/``; returns what was written.

    Optional in the same way pandoc is: a box without matplotlib produces the report and
    says the figures were skipped, rather than failing a run that took hours to feed.
    """
    if not enabled:
        return {}
    if not figures.available():
        print("  skipped figures (matplotlib not installed)", flush=True)
        return {}
    from horizon.report import render_json  # noqa: F401 - keeps the import list honest

    data = json.loads(render_json(facts))
    drawn = {}
    for lang, folder in (("en", slug), ("uk", f"{slug}-ua")):
        target = Path(out_dir) / "figures" / folder
        drawn[lang] = figures.draw(data, target, lang, _figure_text(vocabulary, lang))
        print(f"  wrote {len(drawn[lang])} figures to {target}", flush=True)
    return drawn


def _run_interpret(args) -> int:
    """Generate one model's reading of a report and write it beside the report.

    One model per run by design, each writing its own file. The commentary is a judgement
    and two models disagree about judgements — that disagreement is the honest part of the
    document, so it is preserved rather than averaged, and merging is a separate step that
    costs nothing to redo.

    Resumable in the same way the scan is: a topic already present in the output file is
    not regenerated.
    """
    import json as _json

    from horizon import interpret
    from horizon.llm.transformers_client import TransformersClient

    facts = _json.loads(Path(args.facts).read_text(encoding="utf-8"))
    settings = _settings_from(args)
    tag, label = _reading_identity(settings)
    out_path = Path(args.out) / f"{args.slug}-reading-{tag}-{args.lang}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = (
        _json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.is_file()
        else {
            "model_id": settings.model_id,
            "label": label,
            "lang": args.lang,
            "topics": {},
        }
    )
    done.setdefault("label", label)

    client = TransformersClient(settings)
    allowed = interpret.facts_numbers(facts)
    # Length is set here rather than left to the package default, which is sized for a
    # summary: at 512 tokens 29% of readings ended mid-sentence.
    from dataclasses import replace as _replace

    def budget(tokens: int):
        if args.max_new_tokens is not None:
            return None
        return _replace(settings.generation, max_new_tokens=tokens)

    def retry_if_collapsed(text: str, prompt: str) -> str:
        """One more attempt when a generation collapsed into a repetition loop.

        Retried rather than kept, because 2048 identical characters are worse in a report
        than an invented figure: obviously broken to a reader, and an unbreakable 5000pt
        line that the typesetter spends hours on. Sampling makes the second attempt a
        different draw; if it collapses too the block is left as it is and the merge will
        show it, which is better than looping.
        """
        if not interpret.is_degenerate(text):
            return text
        print("    collapsed into a repetition loop — retrying once", flush=True)
        return client.complete(prompt)

    def write(text: str, key: str | None = None, *, bucket: str = "topics") -> None:
        flagged = interpret.unsupported_numbers(text, allowed)
        if key is None:
            done["corpus"] = {"text": text, "unsupported": flagged}
        else:
            done.setdefault(bucket, {})[key] = {"text": text, "unsupported": flagged}
        out_path.write_text(
            _json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if flagged:
            # Printed, not silently recorded: a figure the facts do not contain is the
            # one thing about a generated reading that cannot be checked by reading it.
            print(f"    unsupported figures: {', '.join(flagged[:6])}", flush=True)

    # Not {"all", args.passes}: that set contains "all" whichever pass was asked for,
    # so every gate below would pass and --pass would select nothing.
    wanted = _passes(args.passes)
    if "corpus" not in done and "corpus" in wanted:
        print("  corpus-wide reading", flush=True)
        prompt = interpret.corpus_prompt(facts, lang=args.lang, domain=args.domain)
        write(
            retry_if_collapsed(
                client.complete(prompt, config=budget(interpret.CORPUS_TOKENS)), prompt
            )
        )
    if not wanted & {"topics", "areas"}:
        print(f"  wrote {out_path}")
        return 0

    by_hash = _summaries_by_hash(args)
    if "areas" in wanted:
        done.setdefault("areas", {})
        names = [a for a in interpret.areas(facts) if a not in done["areas"]]
        if args.limit:
            names = names[: args.limit]
        print(f"  {len(names)} subject areas still to read", flush=True)
        size = max(1, settings.batch_size)
        for start in range(0, len(names), size):
            chunk = names[start : start + size]
            prompts = [
                interpret.area_prompt(
                    a, facts, by_hash, lang=args.lang, domain=args.domain
                )
                for a in chunk
            ]
            print(
                f"  area {start + len(chunk)}/{len(names)} … {chunk[0][:44]}",
                flush=True,
            )
            for area, prompt, text in zip(
                chunk,
                prompts,
                client.complete_batch(prompts, config=budget(interpret.AREA_TOKENS)),
            ):
                write(retry_if_collapsed(text, prompt), area, bucket="areas")
    if "topics" not in wanted:
        print(f"  wrote {out_path}")
        return 0
    topics = interpret.large_topics(facts, min_size=args.min_size)
    if args.limit:
        topics = topics[: args.limit]
    pending = [
        t for t in topics if (t["seed_hash"] or t["label"]) not in done["topics"]
    ]
    print(
        f"  {len(topics)} topics at or above {args.min_size} publications, "
        f"{len(pending)} still to read",
        flush=True,
    )
    # Batched, because the forward pass is the throughput lever on a GPU and these
    # prompts are long. Written after each batch, so an interrupted run resumes at a
    # batch boundary rather than at the start.
    size = max(1, settings.batch_size)
    for start in range(0, len(pending), size):
        chunk = pending[start : start + size]
        prompts = [
            interpret.topic_prompt(
                t,
                interpret.sample_summaries(t.get("members") or [], by_hash),
                facts,
                lang=args.lang,
                domain=args.domain,
            )
            for t in chunk
        ]
        print(
            f"  {start + len(chunk)}/{len(pending)} … {chunk[0]['label'][:44]}",
            flush=True,
        )
        generated = client.complete_batch(
            prompts, config=budget(interpret.TOPIC_TOKENS)
        )
        for topic, prompt, text in zip(chunk, prompts, generated):
            write(
                retry_if_collapsed(text, prompt), topic["seed_hash"] or topic["label"]
            )
    blocks = [done.get("corpus")] + list(done.get("topics", {}).values())
    blocks += list(done.get("areas", {}).values())
    cut = sum(1 for b in blocks if b and interpret.is_truncated(b["text"]))
    if cut:
        # Printed rather than left to be noticed in the PDF: a reading that stops
        # mid-sentence looks like a short reading until someone reads to the end of it.
        print(f"  {cut} readings end mid-sentence — raise --max-new-tokens", flush=True)
    print(f"  wrote {out_path}")
    return 0


def _summaries_by_hash(args) -> dict:
    """``content_hash -> {item_date, summary}`` for every analysis in the store.

    Only the two fields the sampler reads: the full analyses are several gigabytes and
    the commentary needs a date and a paragraph.
    """
    store = _store(args)
    out = {}
    for a in store.list_all():
        out[a.content_hash] = {"item_date": a.item_date, "summary": a.summary}
    print(f"  {len(out)} summaries available", flush=True)
    return out


def _apply_lenses(args, shards, overall, years):
    """Read the corpus-wide clusters through every lens.

    Applied to the merged level-4 trends rather than per field: a lens asking "what is
    fading" wants one answer over the corpus, and the per-field tables above already give
    the within-field view.

    Nothing here re-clusters or re-scores — each lens is a filter and a sort over rows
    already reported, which is what keeps the section auditable against the JSON.
    """
    if not overall:
        return {}, {}
    from horizon import lenses as trend_lenses
    from horizon import metadata_lenses as meta
    from horizon.corpus import RecordIndex
    from horizon.ingest.wos import iter_records

    floor = getattr(args, "lens_min_size", None) or trend_lenses.DEFAULT_LENS_MIN_SIZE
    picked = {
        lens.name: trend_lenses.apply(
            lens, overall, top=args.top_trends, min_size=floor
        )
        for lens in trend_lenses.LENSES
        # "rising" is what level 4 already ranks by; repeating it would pad the report
        # with a copy of the table above it.
        if lens.name != "emerging"
    }

    analyses = _store(args).list_all()
    index = RecordIndex.build(iter_records(shards, years))
    facts = meta.gather(overall, analyses, index)
    meta_picked = {
        "journal-dispersion": meta.broadening(
            facts, min_size=floor, top=args.top_trends
        ),
        "new-authors": meta.new_entrants(facts, min_size=floor, top=args.top_trends),
        "script-distribution": meta.language_asymmetry(
            facts, min_size=floor, top=args.top_trends
        ),
    }
    # Two guards, and they answer different questions. The domain names indicators that
    # are meaningless by construction — a patent's script is set by the filing rules, so
    # ranking topics by it ranks an administrative requirement. The generic check then
    # drops anything that merely came out flat on this data, which is the right test for
    # a lens that could have varied and did not.
    for name in _suppressed(getattr(args, "vocabulary", None)):
        meta_picked.pop(name, None)
        picked.pop(name, None)
    if "script-distribution" in meta_picked and not meta.discriminates(
        meta_picked["script-distribution"], lambda f: f.cyrillic_share or 0.0
    ):
        meta_picked.pop("script-distribution")
    return picked, meta_picked


def _suppressed(vocabulary: str | None) -> frozenset:
    """Indicators the report's domain never reports. Empty for the default wording."""
    if vocabulary == "patents":
        from horizon.patents.vocabulary import SUPPRESSED_INDICATORS

        return SUPPRESSED_INDICATORS
    return frozenset()


def _complete_years(tab, *, floor: float = 0.25) -> set[int] | None:
    """Cohorts to report over, dropping a final one that is obviously partial.

    A corpus gathered mid-year ends in a stub: this one holds 482 publications for 2026
    against a median of 27 000. Left in, it is not merely a small bar — every rate fitted
    across cohorts is dragged toward it. Measured before this guard existed, the largest
    topic in the corpus reported +92 publications a year where the fit over the complete
    cohorts gives several hundred, and a topic that plainly grew reported +0.8.

    ``--years`` still wins; this only decides what happens when nothing was said. The
    threshold is deliberately blunt — a quarter of the median cohort is not a borderline
    call, and a genuinely small final year would have to be four times smaller than
    typical to trip it.
    """
    from statistics import median

    counts = [tab.year_totals[y] for y in tab.years]
    if len(counts) < 3 or counts[-1] >= floor * median(counts[:-1]):
        return None
    dropped = tab.years[-1]
    print(
        f"  dropping {dropped}: {counts[-1]} publications against a median cohort of "
        f"{median(counts[:-1]):.0f} — a partial year depresses every rate fitted across "
        f"cohorts (pass --years to override)",
        flush=True,
    )
    return set(tab.years[:-1])


def _default_window(analyses, settings) -> str:
    """The velocity window to use when ``--window`` was not given.

    ``Settings`` defaults to ``7d``, which is right for a feed and silently wrong for a
    corpus dated by publication year: every record is stamped January 1st, so a
    seven-day window either catches a whole cohort or nothing at all, and velocity comes
    back as a ratio against zero. The failure is invisible in the output — the column
    fills with plausible numbers.

    So detect the case rather than trusting the caller to remember: if every analysis
    falls on January 1st the corpus is year-precision, and ``2y`` is the smallest window
    that has a prior cohort to compare against.
    """
    if not analyses:
        return settings.velocity_window
    year_precision = all(a.item_date[5:10] == "01-01" for a in analyses if a.item_date)
    if not year_precision:
        return settings.velocity_window
    print(
        f"  window {settings.velocity_window} -> 2y (records are dated by year; a "
        "shorter window has no prior cohort)",
        flush=True,
    )
    return "2y"


def _levels_unblocked(args, index, analyses, tab, cache, window, settings):
    """Cluster the whole population at once, then score each cluster inside every field.

    Blocking exists to make clustering affordable: on 269 000 records it is 21x cheaper
    and the cost is a hard rule that a topic belongs to one field. Below roughly ten
    thousand records that trade stops paying, and on a patent register the rule is
    actively wrong — 46% of grants carry more than one WIPO technology, and combining
    technologies is a large part of what patenting *is*.

    So the clusters here are global and the statistics are per field. A topic whose
    patents fall in three fields appears under all three, each time counted and rated
    against that field's own cohort sizes rather than the register's. That is a different
    claim from the blocked version and a truer one: section 3 stops meaning "topics we
    assigned to this field" and starts meaning "how this topic behaves inside this field".
    """
    from horizon.clustering import make_clusterer
    from horizon.trends import build_trends

    pairs = list(index.pairs(analyses))
    if not pairs:
        return {}, []
    members = [a for a, _ in pairs]
    by_hash = cache.embed_analyses(members)
    vectors = [by_hash[m.content_hash] for m in members]
    labels = make_clusterer(
        settings.cluster_method, settings.cluster_threshold
    ).cluster(vectors)
    print(f"  clustered {len(members)} analyses unblocked", flush=True)

    overall = build_trends(
        members,
        vectors,
        _FixedClusterer(labels),
        window=window,
        now=args.anchor,
        attention_weight=settings.attention_weight,
        min_size=args.min_size,
        baseline=dict(tab.year_totals),
    )

    # Every field each analysis belongs to, not just a primary one — that choice is the
    # whole point of dropping the blocking.
    fields: dict[str, list[int]] = {}
    for i, (_, record) in enumerate(pairs):
        for name in record.get("subject_areas") or []:
            if name:
                fields.setdefault(name, []).append(i)

    field_trends: dict = {}
    for name in sorted(fields):
        idx = fields[name]
        if len(idx) < args.min_size:
            continue
        trends = build_trends(
            [members[i] for i in idx],
            [vectors[i] for i in idx],
            _FixedClusterer([labels[i] for i in idx]),
            window=window,
            now=args.anchor,
            attention_weight=settings.attention_weight,
            min_size=args.min_size,
            # The field's own cohort sizes: a topic holding a constant share of a field
            # that shrank is rising inside it.
            baseline=dict(tab.field_totals(name)) or dict(tab.year_totals),
        )
        if trends:
            field_trends[name] = trends
        print(f"  {name[:34]:<34} {len(idx):>7} -> {len(trends)} trends", flush=True)
    return field_trends, overall


def _build_levels_3_and_4(args, shards, tab, years):
    """Cluster within each field, then merge across fields and rank together."""
    import numpy as np

    from horizon.analysis import Analysis  # noqa: F401 - typing clarity
    from horizon.clustering import make_clusterer
    from horizon.corpus import RecordIndex, block
    from horizon.embeddings import EmbeddingCache, SentenceTransformerEmbedder
    from horizon.ingest.wos import iter_records
    from horizon.trends import build_trends

    settings = Settings.from_env()
    if args.cluster_method:
        settings = replace(settings, cluster_method=args.cluster_method)
    if args.threshold is not None:
        settings = replace(settings, cluster_threshold=args.threshold)

    analyses = _store(args).list_all()
    if not analyses:
        print("  no analyses in the store — levels 3 and 4 skipped.", file=sys.stderr)
        return {}, []

    index = RecordIndex.build(iter_records(shards, years))
    resolved, total = index.coverage(analyses)
    # Printed rather than assumed: a join that drops a fifth produces tables that look
    # complete and are not.
    print(f"  joined {resolved}/{total} analyses to records ({resolved / total:.0%})")

    cache = EmbeddingCache(
        SentenceTransformerEmbedder(settings), _embed_cache_path(args)
    )
    window = args.window or _default_window(analyses, settings)
    if args.block_by == "none":
        return _levels_unblocked(args, index, analyses, tab, cache, window, settings)

    blocks = block(index.pairs(analyses))

    field_trends: dict = {}
    merged_members: list[list] = []
    merged_medoids: list = []
    for name in sorted(blocks):
        members = blocks[name]
        if len(members) < args.min_size:
            continue
        by_hash = cache.embed_analyses(members)
        vectors = [by_hash[m.content_hash] for m in members]
        labels = make_clusterer(
            settings.cluster_method, settings.cluster_threshold
        ).cluster(vectors)
        trends = build_trends(
            members,
            vectors,
            _FixedClusterer(labels),
            window=window,
            now=args.anchor,
            attention_weight=settings.attention_weight,
            min_size=args.min_size,
            # The field's own growth, not the corpus's: a theme holding a constant share
            # of the corpus is rising inside a field that shrank.
            baseline=dict(tab.field_totals(name)) or dict(tab.year_totals),
        )
        if trends:
            field_trends[name or "(unclassified)"] = trends
        groups: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(i)
        for indices in groups.values():
            if len(indices) >= args.min_size:
                merged_members.append([members[i] for i in indices])
                merged_medoids.append(_medoid(vectors, indices))
        # Flushed: this loop is most of a ~40-minute run, and redirected stdout is
        # block-buffered, so without it a backgrounded report looks hung for its first
        # several thousand lines of work.
        print(
            f"  {name or '(unclassified)':<34} {len(members):>7} -> {len(trends)} trends",
            flush=True,
        )

    overall = []
    if merged_medoids:
        # Merge across fields before ranking: blocking put each record in one field, so a
        # theme spanning two would otherwise appear twice and rank as though it were two.
        merge = make_clusterer(
            settings.cluster_method, settings.cluster_threshold
        ).cluster(merged_medoids)
        flat, labels = [], []
        for cluster_id, group in zip(merge, merged_members):
            flat.extend(group)
            labels.extend([cluster_id] * len(group))
        by_hash = cache.embed_analyses(flat)
        vectors = [by_hash[a.content_hash] for a in flat]
        overall = build_trends(
            flat,
            vectors,
            _FixedClusterer(labels),
            window=window,
            now=args.anchor,
            attention_weight=settings.attention_weight,
            min_size=args.min_size,
            baseline=dict(tab.year_totals),
        )
        print(
            f"  merged {len(merged_medoids)} field clusters -> {len(overall)} trends",
            flush=True,
        )
    return field_trends, overall


def _run_fields(args) -> int:
    """`fields`: report field development and cross-field interaction from the records.

    Level 1 and 2 of the report structure: the part that
    needs no model, no embedder, no clustering and no threshold, and therefore stands
    whatever the trend clustering turns out to do. Pure counting — see
    :mod:`horizon.fields`."""
    from horizon.fields import field_series, pair_series, tabulate
    from horizon.ingest.wos import iter_records, parse_years, records_dir, shard_years

    root = Path(args.records)
    shards = root if shard_years(root) else records_dir(root)
    if not shard_years(shards):
        print(
            f"error: no built record shards under {root} (nor in {root}/records) — "
            "run `horizon collection-build` first.",
            file=sys.stderr,
        )
        return 1
    try:
        years = parse_years(args.years) if args.years else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tab = tabulate(iter_records(shards, years), key=args.key, years=years)
    if not tab.records:
        print("error: no records in the requested years.", file=sys.stderr)
        return 1

    fields = field_series(tab, min_total=args.min_total)
    pairs = pair_series(tab, min_total=args.min_total)
    if args.top is not None:
        fields, pairs = fields[: args.top], pairs[: args.top]

    if args.as_json:
        print(
            json.dumps(
                {
                    "records": tab.records,
                    "unclassified": tab.unclassified,
                    "years": list(tab.years),
                    "year_totals": {
                        str(y): n for y, n in sorted(tab.year_totals.items())
                    },
                    "fields": [_series_dict(x) for x in fields],
                    "pairs": [_series_dict(x) for x in pairs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    span = f"{tab.years[0]}-{tab.years[-1]}" if tab.years else "-"
    print(
        f"{tab.records} records, {span}, {args.key} "
        f"({tab.unclassified} unclassified, {tab.unclassified / tab.records:.0%})"
    )
    print("\nField development — share of the corpus, and records per year")
    print(f"  {'field':<34}{'first':>8}{'last':>8}{'delta':>9}{'slope':>10}{'n':>9}")
    for x in fields:
        print(
            f"  {x.name[:33]:<34}{x.share_first:>7.1%}{x.share_last:>8.1%}"
            f"{x.share_delta:>+9.1%}{x.slope:>+9.1f}/y{x.total:>9}"
        )
    # Pairs are ranked by slope: a pair's share of the corpus is tiny by construction,
    # so absolute growth is the readable quantity.
    print("\nCross-field interaction — records carrying both, by slope")
    print(f"  {'pair':<52}{'slope':>10}{'first':>8}{'last':>8}{'n':>9}")
    for x in pairs:
        first, last = x.counts[0][1], x.counts[-1][1]
        print(f"  {x.name[:51]:<52}{x.slope:>+9.1f}/y{first:>8}{last:>8}{x.total:>9}")
    print("\nShares are of this corpus, not of world science.")
    return 0


def _series_dict(series) -> dict:
    """A :class:`horizon.fields.Series` as JSON-ready data."""
    return {
        "name": series.name,
        "total": series.total,
        "slope": round(series.slope, 3),
        "share_first": round(series.share_first, 5),
        "share_last": round(series.share_last, 5),
        "share_delta": round(series.share_delta, 5),
        "counts": {str(y): n for y, n in series.counts},
    }


def _velocity_baseline(args) -> dict[int, int] | None:
    """Records per year for ``--records``, or ``None`` when it was not given.

    Accepts the dataset root or the ``records/`` directory inside it. The two spellings
    have already cost one failed run — `gather wos --records` wants the root while
    `bench_models.py --records` wants the shard dir — so this one takes either rather
    than adding a third convention to remember.
    """
    if not getattr(args, "records", None):
        return None

    from horizon.fields import tabulate
    from horizon.ingest.wos import iter_records, records_dir, shard_years

    root = Path(args.records)
    shards = root if shard_years(root) else records_dir(root)
    if not shard_years(shards):
        print(
            f"error: no built record shards under {root} (nor in {root}/records) — "
            "run `horizon collection-build` first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tab = tabulate(iter_records(shards))
    field = getattr(args, "baseline_field", None)
    if not field:
        return dict(tab.year_totals)
    totals = tab.field_totals(field)
    if not totals:
        known = ", ".join(sorted(tab.field_counts)[:8])
        print(
            f"error: no records carry subject area {field!r}. Known areas include: "
            f"{known}…",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return dict(totals)


def _run_trends(args) -> int:
    """`trends`: embed and cluster every stored analysis into rising trends and print
    them. Honours --cluster-method/--threshold/--min-size/--top; errors if the store
    is empty. Loads the embedder/ML stack only here."""
    settings = Settings.from_env()
    if args.cluster_method:
        settings = replace(settings, cluster_method=args.cluster_method)
    if args.threshold is not None:
        settings = replace(settings, cluster_threshold=args.threshold)

    analyses = _store(args).list_all()
    if not analyses:
        print(
            "error: no analyses in the store yet — run `horizon scan` first.",
            file=sys.stderr,
        )
        return 1

    # Resolved before the ML stack loads: a bad --records path should fail in a second,
    # not after the embedder has been pulled into memory.
    baseline = _velocity_baseline(args)

    # Imported here so the embedder/ML stack loads only when building trends.
    from horizon.clustering import make_clusterer
    from horizon.embeddings import EmbeddingCache, SentenceTransformerEmbedder
    from horizon.trends import build_trends, parse_window, resolve_anchor

    embedder = SentenceTransformerEmbedder(settings)
    cache = EmbeddingCache(embedder, _embed_cache_path(args))
    try:
        vectors_by_hash = cache.embed_analyses(analyses)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    vectors = [vectors_by_hash[a.content_hash] for a in analyses]

    clusterer = make_clusterer(settings.cluster_method, settings.cluster_threshold)
    try:
        window = parse_window(args.window or settings.velocity_window)
        scale = (
            args.recency_scale
            if args.recency_scale is not None
            else settings.recency_scale_days
        )
        anchor = resolve_anchor(analyses, now=args.now)
        # Validate the scoring knobs here rather than widening the try around
        # build_trends, so a genuine internal ValueError isn't reported as user error.
        if scale < 1:
            raise ValueError("--recency-scale must be at least 1")
        if not 0.0 <= settings.attention_weight < 1.0:
            raise ValueError("HORIZON_ATTENTION_WEIGHT must be in [0, 1)")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    trends = build_trends(
        analyses,
        vectors,
        clusterer,
        window=window,
        now=anchor.isoformat(),
        recency_scale_days=scale,
        attention_weight=settings.attention_weight,
        min_size=args.min_size,
        baseline=baseline,
    )
    if args.top is not None:
        trends = trends[: args.top]

    if args.as_json:
        # A bare list, no envelope — existing consumers index payload[0].
        print(json.dumps([t.to_dict() for t in trends], indent=2, ensure_ascii=False))
    else:
        source = "--now" if args.now else "newest item_date — override with --now"
        print(f"Trends as of {anchor.isoformat()} ({source})")
        print(
            f"  window {window}   "
            f"recent {window.start(anchor).isoformat()} → {anchor.isoformat()}   "
            f"vs prior {window.start(anchor, back=2).isoformat()} → "
            f"{window.start(anchor).isoformat()}"
        )
        print(
            f"  recency scale {scale}d   "
            f"attention weight {settings.attention_weight}   "
            f"({len(trends)} clusters from {len(analyses)} analyses)"
        )
        print()
        print(format_trends(trends))
    return 0


# --- export -------------------------------------------------------------------


def _run_export(args) -> int:
    """`export`: write stored analyses as SFT (prompt, completion) train/eval splits
    plus a manifest for fine-tuning. Applies --smoke/--eval-fraction/--no-filter/
    --approved-only/--source, and prints per-reason skip counts. Pure — no model."""
    analyses = _store(args).list_all()
    if not analyses:
        print(
            "error: no analyses in the store yet — run `horizon scan` first.",
            file=sys.stderr,
        )
        return 1

    # Pure logic — no model. Imported here to keep the training subpackage off the
    # core import path.
    from horizon.training.export import export_dataset

    sources = _source_store(args).all()
    manifest = export_dataset(
        analyses,
        sources,
        args.out,
        smoke=args.smoke,
        eval_fraction=args.eval_fraction,
        filter_trainable=args.filter_trainable,
        approved_only=args.approved_only,
        source=args.source,
    )
    skipped = manifest["skipped"]
    print(
        f"Exported {manifest['n_train']} train + {manifest['n_eval']} eval "
        f"examples to {args.out} "
        f"(skipped: {skipped['not_trainable']} degenerate, "
        f"{skipped['no_source']} without source text, "
        f"{skipped['not_approved']} unapproved, "
        f"{skipped['wrong_source']} other-source)."
    )
    if manifest["n_train"] + manifest["n_eval"] == 0:
        print(
            "warning: no usable examples — scan more texts (and re-scan old ones so "
            "their source text is persisted).",
            file=sys.stderr,
        )
    return 0


# --- dataset-stats ------------------------------------------------------------


def _run_stats(args) -> int:
    """`dataset-stats`: audit the store's training-set readiness (counts, balance, gap
    to --target) and print it. Pure — no model."""
    analyses = _store(args).list_all()
    if not analyses:
        print(
            "error: no analyses in the store yet — run `horizon scan` first.",
            file=sys.stderr,
        )
        return 1

    from horizon.training.stats import DEFAULT_TARGET, dataset_stats, format_stats

    sources = _source_store(args).all()
    target = args.target if args.target is not None else DEFAULT_TARGET
    stats = dataset_stats(analyses, sources, target=target)
    if args.as_json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print(format_stats(stats))
    return 0


# --- corpus-stats -------------------------------------------------------------


def _run_corpus_stats(args) -> int:
    """`corpus-stats`: summarize a landed corpus by source — item counts and
    writing-system mix — reading the landed text, so it works before any scanning.
    Corpus dir is --corpus, else ``<data-dir>/corpus``. Pure — no model."""
    from collections import Counter

    from horizon.ingest.corpus import iter_corpus_entries, load_record
    from horizon.training.stats import _script_bucket

    corpus_dir = args.corpus if args.corpus is not None else _root(args) / "corpus"
    if not corpus_dir.is_dir():
        print(f"error: no corpus directory at {corpus_dir}", file=sys.stderr)
        return 1

    # Per source: item count + writing-system mix (model-independent — read from the
    # landed text, so it works before any scanning).
    counts: Counter[str] = Counter()
    scripts: dict[str, Counter[str]] = {}
    for entry in iter_corpus_entries(corpus_dir):
        rec = load_record(entry)
        if rec is None:
            continue
        src = rec.get("source") or "-"
        counts[src] += 1
        scripts.setdefault(src, Counter())[_script_bucket(rec["text"])] += 1

    if not counts:
        print(f"error: corpus at {corpus_dir} is empty.", file=sys.stderr)
        return 1

    total_scripts: Counter[str] = Counter()
    for c in scripts.values():
        total_scripts.update(c)
    payload = {
        "corpus": str(corpus_dir),
        "total": sum(counts.values()),
        "by_source": {
            s: {"count": counts[s], "scripts": dict(scripts[s])}
            for s in sorted(counts, key=lambda k: (-counts[k], k))
        },
        "scripts": dict(total_scripts),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_corpus_stats(payload))
    return 0


def _script_mix(scripts: dict) -> str:
    """Format a writing-system counter as descending percentages (e.g. ``latin 60%,
    cyrillic 40%``)."""
    total = sum(scripts.values()) or 1
    rows = sorted(scripts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k} {n / total:.0%}" for k, n in rows)


def _format_corpus_stats(p: dict) -> str:
    """Render the corpus-stats payload as an aligned per-source table with a TOTAL row."""
    lines = [
        f"Corpus by source ({p['corpus']})",
        "=" * 40,
        f"  {'source':<26}{'items':>6}   scripts",
    ]
    for src, d in p["by_source"].items():
        lines.append(f"  {src:<26}{d['count']:>6}   {_script_mix(d['scripts'])}")
    lines += [
        "  " + "-" * 38,
        f"  {'TOTAL':<26}{p['total']:>6}   {_script_mix(p['scripts'])}",
    ]
    return "\n".join(lines)


# --- compare ------------------------------------------------------------------


def _resolve_compare_sides(args, models: list[str]) -> tuple[str, str] | None:
    """Pick (baseline, teacher) model substrings from flags, else auto-detect.

    Auto: the teacher defaults to the Gemini model present, the baseline to any
    other model. Returns ``None`` when the sides can't be determined (the caller
    prints the available model ids so the user can pass --baseline/--teacher).
    """
    teacher = args.teacher
    if teacher is None:
        gemini = [m for m in models if "gemini" in m.lower()]
        teacher = gemini[0] if gemini else None
    baseline = args.baseline
    if baseline is None and teacher is not None:
        others = [m for m in models if teacher.lower() not in m.lower()]
        baseline = others[0] if others else None
    if not baseline or not teacher:
        return None
    return baseline, teacher


def _run_compare(args) -> int:
    """`compare`: report label agreement between two scans of the same corpus (e.g.
    stand-in vs teacher). Resolves the baseline/teacher sides from --baseline/--teacher
    or auto-detects them, and errors with the available model ids if it can't. Pure."""
    from horizon.training.compare import (
        compare_scans,
        distinct_models,
        format_comparison,
        select_by_model,
    )

    records = _store(args).list_all_records()
    if not records:
        print(
            "error: no analyses in the store yet — run `horizon scan` first.",
            file=sys.stderr,
        )
        return 1

    models = distinct_models(records)
    sides = _resolve_compare_sides(args, models)
    if sides is None:
        print(
            "error: could not tell the two scans apart. Models present: "
            f"{', '.join(models)}. Pass --baseline and --teacher (model_id "
            "substrings).",
            file=sys.stderr,
        )
        return 1
    baseline_sub, teacher_sub = sides

    baseline = select_by_model(records, baseline_sub)
    teacher = select_by_model(records, teacher_sub)
    if not baseline or not teacher:
        missing = baseline_sub if not baseline else teacher_sub
        print(
            f"error: no records match model {missing!r}. Models present: "
            f"{', '.join(models)}.",
            file=sys.stderr,
        )
        return 1

    result = compare_scans(baseline, teacher)
    if args.as_json:
        # Note: result already has "baseline"/"teacher" (per-side stats), so the
        # selected model substrings go under distinct *_model keys.
        payload = {
            "baseline_model": baseline_sub,
            "teacher_model": teacher_sub,
            **result,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            format_comparison(
                result, baseline_label=baseline_sub, teacher_label=teacher_sub
            )
        )
    return 0


# --- approve ------------------------------------------------------------------


def _run_approve(args) -> int:
    """`approve`: mark stored analyses as vetted for the training set — the records
    named by content_hash prefix, or every trainable one with --all. Optionally sets
    provenance via --set-source; skips records already in the desired state. Pure."""
    if not args.approve_all and not args.hashes:
        print(
            "error: pass content_hash values or --all.",
            file=sys.stderr,
        )
        return 1

    from horizon.training.export import is_trainable

    store = _store(args)
    analyses = store.list_all()
    if not analyses:
        print(
            "error: no analyses in the store yet — run `horizon scan` first.",
            file=sys.stderr,
        )
        return 1

    if args.approve_all:
        targets = [a for a in analyses if is_trainable(a)]
    else:
        targets = [
            a
            for a in analyses
            if any(a.content_hash.startswith(h) for h in args.hashes)
        ]
    if not targets:
        print("error: no matching records to approve.", file=sys.stderr)
        return 1

    n = 0
    for a in targets:
        if a.approved and (args.set_source is None or a.source == args.set_source):
            continue  # already in the desired state; don't append a no-op record
        updated = replace(
            a, approved=True, source=args.set_source if args.set_source else a.source
        )
        store.put(updated)
        n += 1
    suffix = f" (source={args.set_source})" if args.set_source else ""
    print(f"Approved {n} record(s){suffix}.")
    return 0


# --- collection ---------------------------------------------------------------


def _figure_text(name: str, lang: str):
    """The figure-text overlay for a report's domain, or ``None`` for the defaults.

    Must agree with :func:`_vocabulary`: a report whose tables say "patents" over figures
    whose axes say "publications" is worse than either wording used consistently.
    """
    if name == "patents":
        from horizon.patents.vocabulary import figure_text

        return figure_text(lang)
    return None


def _field_splits(args, shards, years):
    """Per-field share of each population at the ends of the span, or ``()``.

    Shares the ``--compose-by`` key: a report that splits its topics by applicant type
    and its fields by something else would be two analyses in one document.
    """
    key = getattr(args, "compose_by", None)
    if not key or not years:
        return (), {}
    from horizon.composition import field_splits, population_totals
    from horizon.ingest.wos import iter_records

    records = list(iter_records(shards, set(years)))
    rows = field_splits(records, key=key, years=sorted(years))
    pops = population_totals(records, key=key, years=sorted(years))
    n = sum(1 for f in rows if f.disagrees)
    print(
        f"  split {len(rows)} fields by {key} across {len(pops)} populations "
        f"({n} where the groups disagree)",
        flush=True,
    )
    return tuple(rows), pops


def _compositions(args, shards, overall, tab):
    """Per-topic split by ``--compose-by``, ranked by disagreement between groups.

    Returns ``()`` when the flag is unset, so the section is absent rather than empty —
    a heading over nothing reads as a finding that there was nothing to find.
    """
    key = getattr(args, "compose_by", None)
    if not key or not overall:
        return ()
    from horizon.composition import baselines_for, gather
    from horizon.corpus import RecordIndex
    from horizon.ingest.wos import iter_records

    years = list(tab.years)
    records = list(iter_records(shards, set(years)))
    index = RecordIndex.build(records)
    rows = gather(
        overall,
        _store(args).list_all(),
        index,
        key=key,
        years=years,
        baselines=baselines_for(records, key=key, years=years),
    )
    # Ranked here rather than in the renderer: the facts file is the substrate the prose
    # is checked against, and it should hold the same order the tables print.
    rows.sort(key=lambda c: (c.divergence is None, -(c.divergence or 0.0), -c.size))
    named = {g.name for c in rows for g in c.groups}
    print(f"  composed {len(rows)} topics by {key} ({len(named)} groups)", flush=True)
    return tuple(rows)


def _vocabulary(name: str, lang: str):
    """The label overlay for a report's domain, or ``None`` for the default wording.

    ``None`` rather than a copy of the bibliographic labels, so the default path through
    ``render_markdown`` is byte-for-byte the one it had before domains existed.
    """
    if name == "patents":
        from horizon.patents.vocabulary import labels

        return labels(lang)
    return None


def _run_patents_build(args) -> int:
    """`patents-build`: parse the patent workbooks into per-cohort shards.

    Reads the register workbook, folds the WO and IP5 sheets onto the records they match
    (they enrich; they never add rows), ranks forward citations within each cohort, and
    writes ``records/<year>.jsonl`` plus ``build-report.json``. Pure: no model.

    Every cohort is written, including the ones grant lag leaves incomplete — the
    reportable range is *reported*, not enforced by deleting rows, because a question
    about recent filings needs those cohorts even though a fitted trend must not use them.
    """
    from horizon.patents.workbooks import DEFAULT_PATENTS_DIR, build_dataset

    src = args.path if args.path is not None else DEFAULT_PATENTS_DIR
    out = args.out if args.out is not None else Path("datasets/patents")
    print(f"Building {out}/ from {src}/ on the {args.axis} axis …")
    try:
        report = build_dataset(src, out, axis=args.axis, coverage=args.coverage)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "error: reading .xlsx needs openpyxl — install with "
            'pip install -e ".[ingest]"',
            file=sys.stderr,
        )
        return 1

    print(
        f"\n{report['records']} records "
        f"({report['dropped']} dropped with neither title nor abstract)."
    )
    print(
        "  joined:        "
        + ", ".join(f"{k}={v}" for k, v in sorted(report["joins"].items()))
    )
    print(f"  ranked by citations: {report['citation_percentiles']}")
    print(
        "  per cohort:    "
        + ", ".join(f"{y}={n}" for y, n in sorted(report["shards"].items()))
    )
    # The coverage line is the one a reader has to see. Printed for every cohort, with
    # the excluded ones marked, so the reportable range never arrives unexplained.
    keep = set(report["reportable_years"])
    print(
        "  coverage:      "
        + ", ".join(
            f"{y}={float(c):.2f}" + ("" if int(y) in keep else "*")
            for y, c in sorted(report["coverage"].items())
        )
    )
    # Per group, because pooled this register's median lag is flat and then falls, which
    # reads as an office speeding up while the resident series rises by 62%. What moves
    # the pooled number is the mix, not the speed.
    for group, series in sorted((report.get("median_lag") or {}).items()):
        print(
            f"  lag {group:<12} "
            + ", ".join(f"{y}={float(v):.1f}" for y, v in sorted(series.items()))
        )
    if len(keep) < len(report["coverage"]):
        dropped = sorted(set(map(int, report["coverage"])) - keep)
        print(
            f"  * below the {report['coverage_floor']:.0%} floor — {len(dropped)} "
            f"cohort(s) written but outside the reportable range "
            f"({', '.join(map(str, dropped))})."
        )
        # Deliberately not "grant lag rather than a real fall". The register holds only
        # granted patents, so an application filed and not yet granted is invisible, and
        # the trailing cohorts confound three causes that cannot be separated from inside
        # it: fewer filings, an office granting more slowly, and the lag itself. The
        # lengthening median above is the evidence for the middle one, and it also means
        # every coverage figure here is an upper bound.
        print(
            "    These counts confound three causes — fewer applications, slower "
            "granting, and the lag — which a register of grants alone cannot separate. "
            "A rising median lag means the coverage above is an upper bound."
        )
    span = report["reportable_years"]
    print(
        f"\nWrote {out}/records/*.jsonl and {out}/build-report.json "
        f"(next: horizon gather patents --years {span[0]}-{span[-1]} --limit 20)."
        if span
        else f"\nWrote {out}/records/*.jsonl and {out}/build-report.json."
    )
    return 0


def _run_collection_build(args) -> int:
    """`collection-build`: parse the ``collection/`` workbooks into per-year shards.

    Reads each ``Дедублікація_WoS_Scopus_<YYYY>.xlsx``, maps its layout onto one canonical
    record, repairs the known defects, dedupes globally (DOI, then title) merging the
    Scopus-side and WoS-side copies of a paper, and writes ``records/<year>.jsonl`` plus
    ``build-report.json`` under --out (else ``<data-dir>/collection``). Slow by nature —
    Excel is the cost — so progress is printed per workbook. Pure: no model.
    """
    from horizon.ingest.collection import DEFAULT_COLLECTION_DIR, build_dataset
    from horizon.ingest.wos import parse_years

    src = args.path if args.path is not None else DEFAULT_COLLECTION_DIR
    out = args.out if args.out is not None else _root(args)
    try:
        years = parse_years(args.years) if args.years else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def progress(path: Path, rows: int) -> None:
        print(f"  {path.name}: {rows} rows", flush=True)

    print(f"Building {out}/ from {src}/ …")
    try:
        report = build_dataset(src, out, years=years, on_file=progress)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print(
            "error: reading .xlsx needs openpyxl — install with "
            'pip install -e ".[ingest]"',
            file=sys.stderr,
        )
        return 1

    if not report.records:
        print(f"error: no records found under {src}", file=sys.stderr)
        return 1

    dropped = report.dropped_no_title
    print(
        f"\n{report.rows} rows from {len(report.files)} workbook(s) → "
        f"{report.records} records "
        f"({report.merged} merged as duplicates, {dropped} dropped as unusable)."
    )
    print(f"  with abstract: {report.with_abstract}")
    print(
        f"  with citations: {report.with_citations} "
        f"({100 * report.with_citations / report.records:.1f}%)"
    )
    # Citation coverage is structurally uneven — the Cited by column is missing from the
    # 2015 and 2022-2025 workbooks — so print it per stratum; it is the fact that stops
    # anyone treating this signal as uniformly available.
    print(
        "  cited by prov: "
        + ", ".join(
            f"{k}={v}/{report.per_provenance.get(k, 0)}"
            for k, v in sorted(report.per_provenance_citations.items())
        )
    )
    print(
        "  provenance:    "
        + ", ".join(f"{k}={v}" for k, v in sorted(report.per_provenance.items()))
    )
    print(
        "  script:        "
        + ", ".join(f"{k}={v}" for k, v in sorted(report.per_script.items()))
    )
    print(
        "  per year:      "
        + ", ".join(f"{y}={n}" for y, n in sorted(report.per_year.items()))
    )
    if report.flags:
        print(
            "  repairs:       "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.flags.items()))
        )
    print(
        f"\nWrote {out}/records/*.jsonl and {out}/build-report.json "
        "(next: horizon gather wos --years <year> --limit 20)."
    )
    return 0


# --- gather -------------------------------------------------------------------


def _collect_urls(args) -> list[str]:
    """URLs for `gather url` from --url and/or --url-file (blank/`#` lines skipped).

    Raises ``ValueError`` when neither is given (or the file is empty of URLs) and
    ``FileNotFoundError`` for a missing --url-file — both surfaced by the caller.
    """
    urls: list[str] = []
    if args.url:
        urls.append(args.url)
    if args.url_file:
        if not args.url_file.is_file():
            raise FileNotFoundError(f"url file not found: {args.url_file}")
        for line in args.url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        raise ValueError("`gather url` needs --url or a non-empty --url-file.")
    return urls


def _gather_sources(args) -> list[tuple[str, object]]:
    """Resolve ``args.kind`` into a list of ``(label, Source)`` to gather in one run.

    Raises ``ValueError`` for a missing kind-specific arg / empty config, and
    ``FileNotFoundError`` for a missing feeds file — both surfaced by the caller.
    """
    if args.kind == "rss":
        if not args.url:
            raise ValueError("`gather rss` needs --url.")
        from horizon.ingest import RssSource

        return [(args.url, RssSource(args.url))]
    if args.kind == "local":
        if not args.path:
            raise ValueError("`gather local` needs --path.")
        from horizon.ingest import LocalDirSource

        return [(str(args.path), LocalDirSource(args.path))]
    if args.kind == "arxiv":
        if not args.query:
            raise ValueError("`gather arxiv` needs --query.")
        from horizon.ingest import ArxivSource

        label = f"arxiv:{args.query}"
        return [(label, ArxivSource(args.query, max_results=args.max_results))]
    if args.kind == "url":
        urls = _collect_urls(args)  # raises ValueError/FileNotFoundError on bad input
        from horizon.ingest import UrlSource

        return [("url", UrlSource(urls, min_delay=args.delay))]
    if args.kind == "corpus":
        if not args.path:
            raise ValueError("`gather corpus` needs --path (a landed corpus dir).")
        from horizon.ingest import CorpusSource

        return [(str(args.path), CorpusSource(args.path))]
    if args.kind == "patents":
        # Same shard layout, same slicing knobs — the difference is the record shape,
        # which PatentSource knows and this function does not need to.
        from horizon.ingest.wos import parse_years
        from horizon.patents import PatentSource

        dataset = args.records if args.records is not None else _root(args)
        years = parse_years(args.years) if args.years else None
        source = PatentSource(
            dataset,
            years=years,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
        return [(f"patents:{args.years}" if args.years else "patents", source)]
    if args.kind == "wos":
        from horizon.ingest import WosSource
        from horizon.ingest.wos import parse_years

        dataset = args.records if args.records is not None else _root(args)
        years = parse_years(args.years) if args.years else None
        source = WosSource(
            dataset,
            years=years,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
        label = f"wos:{args.years}" if args.years else "wos"
        return [(label, source)]
    # feeds: every [[feed]] in the TOML config, one RssSource each.
    from horizon.ingest import RssSource, load_feeds

    feeds = load_feeds(args.config)
    if not feeds:
        raise ValueError(f"no feeds found in {args.config}.")
    return [(f.name, RssSource(f.url, source=f.name)) for f in feeds]


def _run_gather(args) -> int:
    """`gather`: land texts from the requested source(s) into a corpus/ directory
    (and, unless --no-raw, keep raw content under raw/). Fetches each resolved source,
    dedups via ``write_corpus``, prints per-source and total fetched/new/duplicate
    counts, and with --scan runs the gathered items through the model. Corpus dir is
    --out, else ``<data-dir>/corpus``. A bad source in a multi-source run warns and is
    skipped; a lone bad source is fatal. Needs a model only for --scan."""
    from horizon.ingest.corpus import write_corpus

    try:
        sources = _gather_sources(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Resolve corpus + raw dirs from the dataset root (--out overrides the corpus dir).
    out_dir = args.out if args.out is not None else _root(args) / "corpus"
    raw_dir = (_root(args) / "raw") if args.keep_raw else None

    # Gather each source into the same corpus; dedup is handled by write_corpus.
    # With --scan we also keep the fetched items to run through the model below.
    multi = len(sources) > 1
    fetched = new = duplicate = 0
    gathered: list = []
    for label, source in sources:
        try:
            items = list(source.fetch())
        except ImportError:
            print(
                "error: RSS gathering needs feedparser — install with "
                'pip install -e ".[ingest]"',
                file=sys.stderr,
            )
            return 1
        except (OSError, NotADirectoryError) as exc:
            # One bad feed shouldn't sink a whole batch; a lone source is fatal.
            if multi:
                print(f"warning: {label}: {exc}", file=sys.stderr)
                continue
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "write_corpus", True):
            r = write_corpus(items, out_dir, raw_dir=raw_dir)
            fetched += r.fetched
            new += r.new
            duplicate += r.duplicate
        else:
            fetched += len(items)
        if args.scan:
            gathered.extend(items)
        if multi:
            print(f"  {label}: {r.fetched} fetched, {r.new} new, {r.duplicate} dup")

    tail = (
        "(next: horizon dataset-stats)."
        if args.scan
        else "(next: horizon dataset-stats, after scanning the corpus)."
    )
    print(
        f"Gathered {fetched} item(s) → {new} new, {duplicate} duplicate, "
        f"into {out_dir}/ {tail}"
    )
    if fetched == 0:
        print(
            "warning: source yielded nothing usable (empty or all-filtered).",
            file=sys.stderr,
        )
        return 0

    if args.scan:
        return _scan_gathered(gathered, args)
    return 0


class _ScanAborted(RuntimeError):
    """Raised to stop a scan run early (circuit breaker or interrupt)."""


class _GatheredScanner:
    """Scans gathered items into the store in batches, resumably.

    Idempotent: items already analysed under the current prompt version are skipped
    (their source text is still backfilled so ``export`` sees it) — unless
    ``--rescan`` is set, which re-analyses them and overwrites the store entry (the
    teacher pass: replace stand-in labels with a stronger model's). Items without a
    publication date fall back to scan time, matching ``scan`` with no ``--date``.
    A single item that fails to analyse warns and is skipped — one bad entry must not
    sink the batch. The model loads lazily on the first item that actually needs it.

    A skipped item's ``attention`` is still refreshed from its ``meta`` when it changed:
    a cohort-relative signal moves whenever its dataset is rebuilt, and this path needs
    no model, so re-stamping a whole corpus is a pure-CPU JSONL pass.

    Holds the state a multi-hour run needs across batches: resolved settings, the
    stores, the already-scanned index, the lazily-loaded client, the counters, and the
    consecutive-failure count that trips the circuit breaker.
    """

    # An OOM leaves the CUDA allocator fragmented, so every later item fails too: the
    # job would burn hours producing nothing while looking healthy. Stop instead.
    FAILURE_LIMIT = 10

    def __init__(self, args) -> None:
        self.args = args
        self.settings = _settings_from(args)
        self.rescan = getattr(args, "rescan", False)
        self.store = _store(args)
        self.sources = _source_store(args)
        self.batch_size = max(1, self.settings.batch_size)
        self.progress_every = max(0, getattr(args, "progress_every", 0) or 0)
        # With --rescan we don't pre-seed from the store, so every item is re-analysed;
        # digests still accumulate below to dedup identical texts *within* this run.
        # Keyed by digest, holding the record itself so the skip path can refresh a
        # stale attention value; ``list_all()`` already materialises every Analysis, so
        # keeping them costs nothing over a set of hashes.
        self.already = (
            {}
            if self.rescan
            else {
                a.content_hash: a
                for a in self.store.list_all()
                if a.prompt_version == PROMPT_VERSION
            }
        )
        self._client = None
        self.scanned = self.skipped = self.refreshed = self.failed = 0
        self.consecutive_failures = 0
        self.total = 0
        self.seen = 0
        self.started = time.monotonic()
        self.run_id = _run_id()
        self.aborted: str | None = None

    # --- model ------------------------------------------------------------

    def client(self):
        """The LLM client, constructed on the first item that actually needs it."""
        if self._client is None:
            # Imported here so `gather` without --scan needs no model stack.
            from horizon.llm import make_client

            self._client = make_client(self.settings)
        return self._client

    # --- the run ----------------------------------------------------------

    def scan(self, items: Sequence) -> None:
        """Scan ``items``, batching the model calls and skipping what's already stored."""
        from horizon.ingest.base import attention_of

        self.total = len(items)
        pending: list = []
        queued: set[str] = set()
        try:
            for item in items:
                digest = content_hash(item.text)
                cached = self.already.get(digest)
                if cached is not None or digest in queued:
                    self.sources.put(item.text)  # backfill so export has the input
                    if cached is not None:
                        attention = attention_of(item)
                        if attention != cached.attention:
                            updated = replace(cached, attention=attention)
                            self.store.put(updated)
                            self.already[digest] = updated
                            self.refreshed += 1
                    self.skipped += 1
                    self._tick(1)
                    continue
                pending.append((item, digest))
                queued.add(digest)
                if len(pending) >= self.batch_size:
                    self._flush(pending)
                    pending = []
            if pending:
                self._flush(pending)
        except _ScanAborted as exc:
            self.aborted = str(exc)
        except KeyboardInterrupt:
            self.aborted = "interrupted"

    def _flush(self, pending: list) -> None:
        """Analyse one batch of not-yet-stored items and persist the results."""
        from horizon.extract import AnalysisError, analyze_batch
        from horizon.ingest.base import attention_of

        texts = [item.text for item, _ in pending]
        try:
            results = analyze_batch(
                texts,
                self.client(),
                settings=self.settings,
                config=self.settings.generation,
                item_dates=[item.published_at for item, _ in pending],
            )
        except RuntimeError as exc:
            # A whole-batch failure: a model that won't load, or a hard device error.
            print(f"warning: batch of {len(pending)} failed: {exc}", file=sys.stderr)
            self.failed += len(pending)
            self.consecutive_failures += len(pending)
            self._tick(len(pending))
            self._check_circuit_breaker()
            return

        for (item, digest), result in zip(pending, results):
            if isinstance(result, AnalysisError):
                print(f"warning: skipped an item: {result}", file=sys.stderr)
                self.failed += 1
                self.consecutive_failures += 1
                continue
            self.consecutive_failures = 0
            # Record ingestion provenance (§4) — the feed/dir the text came from and
            # its link — without touching the training-`source` axis.
            analysis = replace(
                result,
                origin=item.source,
                url=item.url,
                attention=attention_of(item),
            )
            self.store.put(analysis)
            self.sources.put(item.text)
            self.already[digest] = analysis
            self.scanned += 1
        self._tick(len(pending))
        self._check_circuit_breaker()

    def _check_circuit_breaker(self) -> None:
        if self.consecutive_failures >= self.FAILURE_LIMIT:
            raise _ScanAborted(
                f"{self.consecutive_failures} consecutive failures — stopping. "
                "This usually means the model cannot run here (out of memory, or a "
                "device/dtype mismatch), not that the items are bad."
            )

    # --- reporting --------------------------------------------------------

    def _tick(self, n: int) -> None:
        """Advance the progress counter, reporting to stderr every N items."""
        self.seen += n
        if not self.progress_every or self.seen % self.progress_every >= n:
            return
        elapsed = time.monotonic() - self.started
        # Rate and ETA count only the items the model actually ran on. A skipped item
        # costs a dict lookup, so mixing the two makes a resumed shard look five times
        # faster than it is and understates the wait by hours across twelve shards.
        worked = self.scanned + self.failed
        rate = worked / elapsed if elapsed > 0 and worked else 0.0
        remaining = (self.total - self.seen) / rate if rate > 0 else 0.0
        print(
            f"  {self.seen}/{self.total}  {self.scanned} scanned, "
            f"{self.skipped} skipped, {self.failed} failed  "
            f"{rate:.2f} it/s  elapsed {_hms(elapsed)}  eta {_hms(remaining)}",
            file=sys.stderr,
            flush=True,
        )

    def counts(self) -> dict:
        """Final tallies, for the summary line and the run manifest."""
        return {
            "scanned": self.scanned,
            "skipped": self.skipped,
            "refreshed": self.refreshed,
            "failed": self.failed,
        }

    def report(self) -> int:
        """Print the summary line and return the process exit code."""
        msg = f"Scanned {self.scanned} new, {self.skipped} already in store"
        if self.refreshed:
            msg += f" ({self.refreshed} attention refreshed)"
        if self.failed:
            msg += f", {self.failed} failed"
        print(msg + ".")
        if self.aborted == "interrupted":
            print("Interrupted — partial results are saved.", file=sys.stderr)
            return 130
        if self.aborted:
            print(f"error: {self.aborted}", file=sys.stderr)
            return 2
        return 0

    def write_manifest(self) -> None:
        """Record what this run was, so shard outputs are attributable afterwards."""
        path = _root(self.args) / "data" / "runs" / f"{self.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "argv": sys.argv[1:],
            "model_id": self.settings.model_id,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "load_in_4bit": self.settings.load_in_4bit,
            "device_map": self.settings.device_map,
            "attn_implementation": self.settings.attn_implementation,
            "batch_size": self.settings.batch_size,
            "strategy": self.settings.strategy,
            "max_new_tokens": self.settings.generation.max_new_tokens,
            "prompt_version": PROMPT_VERSION,
            "adapter_path": self.settings.adapter_path,
            "host": platform.node(),
            "versions": _versions(),
            "env": {
                k: v for k, v in sorted(os.environ.items()) if k.startswith("HORIZON_")
            },
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "items": self.total,
            "counts": self.counts(),
            "aborted": self.aborted,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _run_id() -> str:
    """A run identifier that is unique across concurrent shard jobs on one cluster."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{platform.node().split('.')[0]}-{os.getpid()}"


def _hms(seconds: float) -> str:
    """Format a duration as ``H:MM:SS``, for progress and ETA lines."""
    total = int(max(0.0, seconds))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _versions() -> dict[str, str]:
    """Versions of the libraries that decide what a run's numbers mean."""
    found: dict[str, str] = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes"):
        try:
            found[name] = __import__(name).__version__
        except Exception:
            found[name] = "absent"
    return found


def _scan_gathered(items: list, args) -> int:
    """Scan gathered items into the store, dating each by its ``published_at`` (C5)."""
    scanner = _GatheredScanner(args)
    scanner.started_at = datetime.now(timezone.utc).isoformat()
    scanner.scan(items)
    try:
        scanner.write_manifest()
    except OSError as exc:  # a manifest must never lose a completed scan
        print(f"warning: could not write the run manifest: {exc}", file=sys.stderr)
    return scanner.report()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse ``argv`` and dispatch to the subcommand handler,
    returning its process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return _run_scan(args)
    if args.command == "trends":
        return _run_trends(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "dataset-stats":
        return _run_stats(args)
    if args.command == "corpus-stats":
        return _run_corpus_stats(args)
    if args.command == "interpret":
        return _run_interpret(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "fields":
        return _run_fields(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "approve":
        return _run_approve(args)
    if args.command == "patents-build":
        return _run_patents_build(args)
    if args.command == "collection-build":
        return _run_collection_build(args)
    if args.command == "gather":
        return _run_gather(args)
    return 1  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":
    raise SystemExit(main())
