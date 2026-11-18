# Examples

The `sample*.txt` files are small prepared inputs for a first CLI scan. They are
not the externally supplied WoS/Scopus records and do not reproduce E1–E5 as
reported. The commands use a newly downloaded or already installed local model
and produce **unreviewed** analyses.

```bash
horizon scan examples/sample.txt --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 256 --json
```

To exercise parsing and data-model invariants without downloading a model:

```bash
python -m pytest -q tests/test_analysis.py tests/test_extract.py
```
