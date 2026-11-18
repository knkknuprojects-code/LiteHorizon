# datasets/

Корені наборів даних за темами (порожні — створюються командами). Кожен корінь має структуру

    <тема>/
    ├── data/analyses.jsonl     сховище аналізів
    ├── data/embeddings.jsonl   кеш ембедингів
    ├── corpus/                 зібрані тексти + manifest.jsonl
    ├── raw/                    сирий вміст до нормалізації
    └── records/<year>.jsonl    побудовані шарди + build-report.json

Корінь передається через `--data-dir datasets/<тема>` або `HORIZON_DATA_DIR`.
Типові теми: `tech`, `arxiv`, `wos-scopus`, `patents`.
