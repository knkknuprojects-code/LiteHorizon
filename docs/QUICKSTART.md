# Quickstart: from a text to an analysis

Start with [INSTALL](INSTALL.md). These steps are designed for a first-time user and
require no licensed bibliographic corpus.

## 1. Check the package without a language model

```bash
horizon --help
python -m pytest -q tests/test_analysis.py tests/test_extract.py
```

This confirms CLI import and the structured `Analysis` machinery. It does **not**
benchmark, fine-tune or evaluate a language model.

## 2. Scan one public example

```bash
horizon scan examples/sample.txt \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 256 --json
```

The example is a prepared local input, not one of the private WoS/Scopus records.
If the first model generation produces invalid JSON, try a larger
`--max-new-tokens` value and inspect local hardware limits. Successful JSON
validation does not establish that labels or summary are correct.

To run without writing to the default data store, add `--no-save --no-cache`.

## 3. Collect a small public arXiv batch (optional)

```bash
python -m pip install -e ".[ingest]"
horizon gather arxiv --query "cat:cs.AI" --max-results 10 --data-dir datasets/demo
horizon corpus-stats --data-dir datasets/demo
```

The command fetches currently available arXiv metadata and abstracts, so its
output will **not** match the specific July 2026 snapshot in the manuscript.

## 4. Analyze, group and report a small local dataset

```bash
horizon gather corpus --path datasets/demo/corpus --scan \
  --data-dir datasets/demo \
  --model Qwen/Qwen2.5-0.5B-Instruct
horizon trends --data-dir datasets/demo
```

This runs multiple model generations and embeddings. Runtime, format yield, and
topic assignments depend on the exact versions and inputs. Avoid interpreting a
small short-date batch as a validated measure of longitudinal change.

## 5. Next steps

- [Architecture](ARCHITECTURE.md): what each stage does.
- [Datasets](DATASETS.md): where local inputs and outputs go.
- [Reproducibility](REPRODUCIBILITY.md): which experiments are and are not reproducible
  from this export.
- [Full developer guide](DEVELOPER_GUIDE_UK.md): additional commands and backends.
