#!/usr/bin/env python3
"""Export the topics, their figures and the models' readings to a spreadsheet.

    python scripts/build_workbook.py reports/wos-scopus-2026-08-28-facts.json \\
        reports/wos-scopus-*-reading-*-uk.json

One row per topic, one column per model. The markdown documents are for reading; this is
for sorting, filtering and pasting into something else — the two are different jobs and a
table that reads well is rarely one that sorts well.

Readings are recomputed against the facts on the way out, like the merged documents do.
Figures the facts do not contain are reported once on the console rather than given a
column: in a clean run that column is empty in every row, and an empty column is noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from horizon.interpret import (
    clean_reading,
    facts_numbers,
    large_topics,
    unsupported_numbers,
)  # noqa: E402

HEAD = {
    "uk": [
        "тема",
        "публікацій",
        "відносне зростання",
        "публікацій/рік",
        "перцентиль цитувань",
        "перший рік",
        "останній рік",
        "ключові слова",
    ],
    "en": [
        "topic",
        "publications",
        "relative growth",
        "publications/year",
        "citation percentile",
        "first year",
        "last year",
        "keywords",
    ],
}
SHEETS = {"uk": ("Теми", "Предметні області"), "en": ("Topics", "Subject areas")}
FIELD_HEAD = {
    "uk": [
        "предметна область",
        "частка на початку",
        "частка в кінці",
        "зміна",
        "публікацій/рік",
        "публікацій",
    ],
    "en": [
        "subject area",
        "share first",
        "share last",
        "change",
        "publications/year",
        "publications",
    ],
}
READING = {"uk": "Огляд", "en": "Overview"}


def build(facts: dict, readings: list[dict], lang: str, min_size: int):
    """The workbook: topics with both readings, and the subject-area table beside them."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    allowed = facts_numbers(facts)
    flagged: list[str] = []
    wb = Workbook()
    ws = wb.active
    ws.title = SHEETS[lang][0]

    # One model gets a plain heading: which model wrote a reading is a property of the
    # run and belongs in the technical report, not in a column title. Several models need
    # naming, and then the name is there to tell them apart.
    models = [r.get("label") or r["model_id"] for r in readings]
    single = len(models) == 1
    ws.append(
        HEAD[lang]
        + [READING[lang] if single else f"{READING[lang]} — {m}" for m in models]
    )

    for topic in large_topics(facts, min_size=min_size):
        key = topic.get("seed_hash") or topic["label"]
        texts = []
        for r in readings:
            block = r.get("topics", {}).get(key)
            text = clean_reading(block["text"]) if block else ""
            texts.append(text)
            flagged.extend(unsupported_numbers(text, allowed) if text else [])
        ws.append(
            [
                topic["label"],
                topic["size"],
                round(topic["velocity"], 2),
                round(topic["growth"], 1),
                (
                    round(topic["attention_pctile"], 2)
                    if topic.get("attention_pctile") is not None
                    else None
                ),
                int(str(topic.get("first_seen", "0"))[:4] or 0) or None,
                int(str(topic.get("last_seen", "0"))[:4] or 0) or None,
                ", ".join(topic.get("top_keywords") or []),
                *texts,
            ]
        )

    fields = wb.create_sheet(SHEETS[lang][1])
    fields.append(FIELD_HEAD[lang])
    for f in facts.get("fields") or []:
        fields.append(
            [
                f["name"],
                round(f["share_first"], 4),
                round(f["share_last"], 4),
                round(f["share_delta"], 4),
                round(f["slope"], 1),
                f["total"],
            ]
        )

    for sheet, widths in (
        (
            ws,
            [46, 12, 14, 14, 14, 10, 10, 40] + [90] * len(models),
        ),
        (fields, [40, 14, 14, 12, 16, 14]),
    ):
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for i, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(i)].width = width
        # Wrapped, top-aligned: the reading columns hold paragraphs, and a spreadsheet
        # that clips them is a spreadsheet nobody reads the readings in.
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    if flagged:
        # Not a column: it would be empty in all but a row or two, and an empty column in
        # a spreadsheet is noise. Said once here instead, so a run that starts citing
        # figures the facts do not contain is still noticed.
        print(f"  figures absent from the facts: {', '.join(sorted(set(flagged)))}")
    return wb


def main() -> int:
    """Write the workbook beside the facts it was built from."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("facts", type=Path)
    ap.add_argument("readings", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-size", type=int, default=200)
    args = ap.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_interpretation import merge_readings  # noqa: E402

    readings = merge_readings(
        [json.loads(p.read_text(encoding="utf-8")) for p in args.readings]
    )
    lang = (readings[0].get("lang") if readings else "uk") or "uk"
    stem = args.facts.name.replace("-facts.json", "")
    out = args.out or args.facts.parent / f"{stem}-topics-{lang}.xlsx"
    build(facts, readings, lang, args.min_size).save(out)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
