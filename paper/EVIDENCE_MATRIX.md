# Evidence matrix: manuscript ↔ software ↔ historical Sources

This table links the *LiteHorizon* manuscript to the provided 2026 project files.
It documents **traceability**, not independent re-execution of the experiments.
Report files named below were supplied separately in the research project and
are **not automatically copied into the public GitHub repository**.

| Study | Main source files in `src/horizon/` and scripts | Historical project report / source | Evidence and limits |
|---|---|---|---|
| E1 — acquisition and replay | `ingest/rss.py`, `ingest/feeds.py`, `ingest/corpus.py`, `ingest/normalize.py` | `dataset-creation-tech-pilot-2026-07-02-ua.pdf`, `PHASE_5_PLAN(1).md` | Reported 250 items from 11 English/Ukrainian feeds; original landed data absent. |
| E2 — arXiv structured extraction | `analysis.py`, `extract.py`, `prompts.py`, `llm/transformers_client.py` | `arxiv-scan-cs-ai-2026-07-04-ua.pdf`, `arxiv-dataset-characterization-2026-08-20-ua.md` | 245 items, 238 schema-valid analyses; June/July `phase2.v1` prompts not preserved as historical code snapshot here. |
| E3 — paired model comparison | `llm/gemini_client.py`, `training/compare.py`, `scripts/compare_models.sh` | `teacher-comparison-tech-pilot-2026-07-03-ua.pdf` | Five paired texts, mixed English/Ukrainian. Small engineering comparison, not population-level ranking. |
| E4 — LoRA smoke loop | `training/export.py`, `training/train.py`, `training/evaluate.py` | `finetuning-smoke-loop-tech-pilot-2026-07-02-ua.pdf`, `TRAINING.md` | 16 train, 4 eval, five steps; demonstrates mechanics, not reliable semantic gain. |
| E5 — WoS/Scopus pipeline and time-series report | `ingest/collection.py`, `ingest/wos.py`, `clustering.py`, `trends.py`, `fields.py`, `report.py`; GPU and merge scripts | `gemma4-gpu-throughput-wos-scopus-2026-08-26.md`, `PHASE_8_PLAN.md`, `wos-scopus-2026-08-28-annotated-ua.pdf` | 269,065 reconstructed records; 268,237 unique analyzable texts; 268,216 schema-valid outputs; 268,583 full-year (2015–2025) records; 1,575 selected-threshold topics. Workbooks and production JSONL absent. |
| D1 — downstream topic display | `embeddings.py`, `clustering.py`, `trends.py`, `cli.py` | `res_multy.txt`, `arxiv_scan.md`, July arXiv scan report | Illustrative topic grouping; short observational span precludes independent validation of long-term emergence. |

## Related but **not** E1–E5 evidence

- `res_single.txt`: an illustrative structured per-document output.
- `DEVELOPMENT_PLAN.md`, `PHASE_2_PLAN(1).md` … `PHASE_5_PLAN(1).md`: historical
  plans; not proof of a run at a specified date.
- `IT-DistilGPT-regular.ipynb` and `IT-DistilGPT-distil.ipynb`: experiments in
  ordinary GPT-style tuning and teacher/student distillation on a different
  task/dataset. They are **not integrated into Horizon E4**.
- `LiteHorizon_15p2_from_original_revised.docx`: a shorter editorial version of
  the main 30 October manuscript, not another independent experiment.

## Critical distinctions

1. Archive date 2026-11-17 does **not** imply that every current code path existed
   during the July/August experiments.
2. The manuscript's `LiteHorizon` and CLI package `horizon` refer to the same
   project family, but code/experiment revisions must be specified precisely.
3. Schema-valid outputs do not imply human-verified semantic correctness.
4. Open-label extraction can be more stable than subjective enum judgments.
5. The WoS/Scopus collection's coverage is an observation-frame property. Its
   changes over time must not be generalized to global science without validation.
