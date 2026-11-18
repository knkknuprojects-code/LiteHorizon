# System architecture

## Scope

LiteHorizon implements a configurable **automated support chain** for horizon
scanning. Human interpretation and downstream foresight decisions remain outside
the demonstrated automated boundary.

1. **Source acquisition:** `src/horizon/ingest/` handles RSS/Atom, arXiv, URLs,
   local files, and bibliographic workbooks; provenance accompanies records.
2. **Normalization and persistence:** `src/horizon/corpus.py`,
   `ingest/normalize.py`, and `store.py` separate normalized inputs from analyses.
3. **Structured extraction:** `llm/base.py`, `llm/transformers_client.py`,
   `llm/gemini_client.py`, `prompts.py`, `extract.py`, and `analysis.py` create a
   schema-validated `Analysis` including summary, signals, classifications,
   entities, keywords, dates, and model provenance.
4. **Representation and grouping:** `embeddings.py`, `clustering.py` group document
   analyses with threshold-sensitive multilingual semantic representations.
5. **Temporal characterization:** `trends.py`, `fields.py`, `metadata_lenses.py`
   compute trends and cohort-aware statistics where suitable metadata exist.
6. **Reporting and interpretation:** `report.py`, `figures.py`, `render.py`, and
   `interpret.py` separate measurement from model-assisted reading.
7. **Optional adaptation:** `training/export.py`, `training/train.py`,
   `training/evaluate.py` handle supervised adaptation and evaluation. The LoRA
   pathway is separate from the base analysis CLI.

## Interfaces and reproducibility boundaries

`LLMClient` lets the same analysis interface call a local model or an optional
hosted model. `Analysis` stores `model_id`, `prompt_version`, `strategy`,
`generated_at`, `item_date`, `content_hash`, `origin`, `url`, training provenance,
and an optional normalized attention score.

The **presence of metadata** enables auditing; it does not in itself make the
system or its results reproducible. Exact repeatability also requires the same
source snapshots, code commit, model weights/tokenizer, prompt contents,
decoding settings, hardware constraints, and downstream parameters.

## Limitations that matter in publications

- A valid JSON object can still contain wrong or unstable scientific judgments.
- Cross-document similarity and temporal co-variation do not by themselves prove
  that two texts report the same developing phenomenon.
- Temporal metrics describe **the observed corpus**, not all global activity.
- The current repository contains functionality added after some reported pilots;
  backdating it to those pilots without historic commits would be incorrect.
