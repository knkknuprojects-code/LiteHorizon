# Data inventory and placement

The GitHub release is deliberately a **code-and-methods distribution**. Do not
commit the underlying licensed or confidential workbooks and raw corpora.

## Directory conventions

```text
collection/               externally supplied WoS/Scopus workbooks (local only)
patents/                  externally supplied patent workbooks (local only)
datasets/<topic>/
  corpus/                 normalized gathered records and manifest(s)
  raw/                    originals retained for provenance
  data/analyses.jsonl     model-generated per-item analyses
  data/embeddings.jsonl   embedding cache
  records/<year>.jsonl    reconstructed bibliographic year shards
reports/                  generated scientific reports (review before sharing)
```

Only the directory README files are tracked in the data-containing root folders;
`.gitignore` excludes data within them. There may also be root-local `data/`,
`corpus/`, `raw/`, `runs/` and `models/` during development, also ignored.

## Status of the paper datasets

| Dataset | Role | Included in the export? | Notes |
|---|---|---|---|
| Tech/AI feed sample | E1 and small E3/E4 pilots | No original corpus | Reported 250 acquired items from 11 feeds. Source content and API results can change. |
| arXiv `cat:cs.AI` July 2026 | E2 and D1 | No snapshot | Reported 245 records, 238 valid analyses. Querying arXiv today is a different sample. |
| WoS/Scopus-derived workbooks | E5 | No | 11 supplied workbooks, reported 269,065 reconstructed records (2015–2026); primary inputs may have redistribution restrictions. |
| Large-scale model analyses | E5 | No | Reported 268,237 analyzable unique texts, 268,216 valid objects, 21 failures; exact production outputs are absent. |
| Adaptation datasets and adapters | E4 | No | Only the smoke-loop report and source code exist in the external project materials; trained weights are absent. |

There is **no permission or factual basis to synthesize missing real-world
records** merely to make the archive appear complete. Request legal access from
the data owners and validate checksums when rehydrating an experiment. Separate
examples can be synthetic and must be labeled as such.

## Before adding a dataset publicly

Record its source, access terms, download/export date, selection protocol,
canonical identifier, checksum, language, deduplication rules, missing-value
handling, software version, ethical restrictions, and permitted redistribution.
Do not infer global scientific prevalence from an externally selected sample.
