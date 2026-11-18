# Reproducibility and experiment provenance

## What this archive can reproduce

It includes the current Python implementation, configuration, scripts and tests.
A new user can install the package, exercise deterministic components, run a
small public/example scan with their own model, build small corpora, and compute
reports over **newly gathered** inputs.

## What it cannot reproduce by itself

**Exact E1–E5/D1 numerical outputs cannot be regenerated from this archive alone.**
Missing assets include original source snapshots and paid/externally supplied
WoS/Scopus books, most recorded per-item JSONL outputs, experiment logs,
training artifacts, dependency lockfiles for every historical host, and historical
versions of prompts/code. `PROMPT_VERSION` in this snapshot is `phase2.v4`; the
July 2026 E2/E3 pilot reports used `phase2.v1`. A common version label is not a
substitute for the actual prompt contents at that time.

## Reported counts (documented, not independently rerun here)

| Evidence | Reported observations | Caveat |
|---|---|---|
| E1 | 250 Tech/AI items from 11 sources | Source fetch snapshots not supplied. |
| E2 | 245 arXiv items, 238 valid analyses (7 failures) | `gemma-3-1b-it`, CPU, `phase2.v1` for the July run. |
| E3 | Five paired inputs (3 English, 2 Ukrainian) | Too small to establish population-level model superiority. |
| E4 | 16 training + 4 evaluation items, five LoRA steps | Smoke test of mechanics, not reliable improvement in task quality. |
| E5 | 269,065 bibliographic records rebuilt; 268,237 analyzable unique texts; 268,216 valid analyses | Missing original workbooks and complete production outputs. |
| E5 longitudinal | 268,583 records from complete 2015–2025 cohorts; 1,575 topic clusters at the chosen threshold | Partial 2026 cohort excluded from rate calculations; selected-corpus statistics only. |
| D1 | Small arXiv-derived trend grouping | Very narrow observation dates; velocity is not a validated emergence signal. |

The manuscript and accompanying experiment reports are the source of these
numbers. The counts are not claimed to have been rechecked by a fresh run of
this release archive.

## Minimal experiment provenance record

For every repeat or extension, archive a manifest containing at least:

```yaml
experiment_id: "REPLACE_WITH_ID"
source_snapshot: "REPLACE_WITH_REFERENCE_OR_CHECKSUM"
source_license: "REPLACE_WITH_ACCESS_TERMS"
code_commit: "REPLACE_WITH_GIT_SHA"
python_version: "REPLACE"
resolved_dependencies: "REPLACE_WITH_LOCKFILE_REFERENCE"
model_id: "REPLACE_WITH_EXACT_REVISION"
model_weights_revision: "REPLACE_WITH_REVISION_OR_CHECKSUM"
prompt_version: "REPLACE_WITH_LABEL"
prompt_file_checksum: "REPLACE_WITH_SHA256"
strategy: "combined"
decoding_configuration: "REPLACE_WITH_JSON_OR_PATH"
hardware: "REPLACE_WITH_DEVICE_DETAILS"
seed: null
clustering_configuration: "REPLACE_WITH_JSON_OR_PATH"
temporal_window: "REPLACE_WITH_EXACT_WINDOW"
produced_outputs_sha256: "REPLACE_WITH_CHECKSUM_OR_MANIFEST"
```

This YAML is an **empty schema example**, not a record of a performed run.

## How to verify the published claim chain

Use [paper/EVIDENCE_MATRIX.md](../paper/EVIDENCE_MATRIX.md) to map E1–E5/D1 to code and
report names. For each scientific sentence, distinguish (i) directly observed
measurements, (ii) report-only observations, and (iii) unvalidated interpretation.

If outputs change after modifying prompts, code, models, cluster thresholds,
record scope, or temporal windows, report the change as a **new experiment**. Do
not silently overwrite figures associated with a manuscript version.
