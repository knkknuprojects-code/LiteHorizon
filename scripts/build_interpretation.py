#!/usr/bin/env python3
"""Merge the models' readings into the two commentary documents.

    python scripts/build_interpretation.py reports/wos-scopus-2026-08-28-facts.json \\
        reports/wos-scopus-*-reading-*-uk.json --report reports/wos-scopus-2026-08-28-ua.md

Produces two files, because they answer different questions and mixing them would let an
inference borrow the authority of a measurement:

* ``<slug>-interpretation-ua.md`` — the readings alone, each model under its own name,
  side by side per topic. Where two models disagree the disagreement is on the page: the
  judgements a model makes about what a movement means are largely its own prior, and
  averaging them into one voice would hide exactly what a reader needs to discount.
* ``<slug>-annotated-ua.md`` — the report with the readings folded in, for reading start
  to finish. Every generated paragraph is marked with the model that wrote it, so the
  boundary between what was measured and what was inferred survives the merge.

Figures a model used that the facts do not contain are listed under the paragraph rather
than quietly dropped. A generated reading is the one part of this report that cannot be
checked by reading it.
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
from horizon.report import PAGE_BREAK  # noqa: E402

HEAD = {
    "uk": {
        "title": "Міркування до звіту",
        "lede": (
            "Це прочитання моделей, а не вимірювання. Звіт стверджує, що виміряно, і "
            "кожне його число простежується до `facts.json`; цей документ каже, що рух "
            "може означати, а це вже судження. Дві моделі наведено поруч навмисно: "
            "оцінки того, що означає рух, великою мірою є пріором конкретної моделі, "
            "тож розбіжність між ними є частиною свідчення, а не шумом, який варто "
            "усереднити."
        ),
        "corpus": "Корпус у цілому",
        "topics": "Окремі теми",
        "unsupported": "Числа, яких немає у фактах",
        "annotated_title": "Звіт із міркуваннями",
        "annotated_lede": (
            "Той самий звіт, доповнений прочитаннями моделей. Виміряне лишається "
            "виміряним: кожен згенерований абзац підписано моделлю, яка його написала."
        ),
        "reading": "Прочитання",
        "figures": "публікацій · відносне зростання · публікацій/рік",
    },
    "en": {
        "title": "Commentary on the report",
        "lede": (
            "This is a reading by models, not a measurement. The report states what was "
            "measured and every number in it is traceable to `facts.json`; this document "
            "says what a movement might mean, which is a judgement. Two models appear "
            "side by side on purpose: what a model takes a movement to mean is largely "
            "its own prior, so disagreement between them is part of the evidence rather "
            "than noise to average away."
        ),
        "corpus": "The corpus as a whole",
        "topics": "Individual topics",
        "unsupported": "Figures absent from the facts",
        "annotated_title": "The report, annotated",
        "annotated_lede": (
            "The same report with the models' readings folded in. What was measured "
            "stays measured: every generated paragraph carries the model that wrote it."
        ),
        "reading": "Reading",
        "figures": "publications · relative growth · publications/year",
    },
}


def merge_readings(readings: list[dict]) -> list[dict]:
    """Fold readings from the same model into one, keyed by (model, label, lang).

    A run split across machines writes one file per pass — topics on one box, areas and
    the corpus on another — and they are the same reading, not two. Left unmerged they
    would be treated as two models and shown side by side under identical names, which
    is exactly the presentation reserved for showing genuine disagreement.

    Within a key the blocks are unioned and an existing block wins, so re-running one
    pass cannot silently replace readings the other produced.
    """
    out: dict[tuple, dict] = {}
    for r in readings:
        key = (r.get("model_id"), r.get("label"), r.get("lang", "uk"))
        into = out.setdefault(
            key, {k: v for k, v in r.items() if not isinstance(v, dict)}
        )
        for section in ("topics", "areas"):
            merged = into.setdefault(section, {})
            for name, block in (r.get(section) or {}).items():
                merged.setdefault(name, block)
        if "corpus" in r:
            into.setdefault("corpus", r["corpus"])
    return list(out.values())


def _block(
    reading: dict, model: str, lab: dict, allowed: set, *, attribute: bool = True
) -> list[str]:
    """One model's paragraph, with any unsupported figures named under it.

    ``attribute`` adds the model's name above the paragraph. It is there to tell several
    models apart, not to record provenance: which model wrote a document belongs in the
    technical report that describes the run, not in the middle of the analysis, so with a
    single reading nothing is named at all.

    The figures are recomputed here rather than taken from the stored run. The check
    improves — its first version reported an enumeration of years as one invented number
    and flagged 0.50 against a stored 0.5 — and recomputing means a fixed check applies
    to readings already generated, instead of only to the next run.
    """
    text = clean_reading(reading["text"])
    out = ([f"**{lab['reading']} — {model}**", ""] if attribute else []) + [text, ""]
    flagged = unsupported_numbers(text, allowed)
    if flagged:
        out += [f"> {lab['unsupported']}: {', '.join(flagged)}", ""]
    return out


def _topic_line(topic: dict, lab: dict) -> str:
    """The measured figures, so a reading is never read without them."""
    return (
        f"*{topic['size']} · {topic['velocity']:.2f} · {topic['growth']:+.1f}* "
        f"({lab['figures']})"
    )


def build_interpretation(facts: dict, readings: list[dict], lang: str, min_size: int):
    """The readings alone, corpus first, then a section per topic."""
    lab = HEAD[lang]
    allowed = facts_numbers(facts)
    attribute = len(readings) > 1
    out = [f"# {lab['title']}", "", lab["lede"], "", f"## {lab['corpus']}", ""]
    for r in readings:
        if "corpus" in r:
            out += _block(
                r["corpus"],
                r.get("label") or r["model_id"],
                lab,
                allowed,
                attribute=attribute,
            )
    out += [f"## {lab['topics']}", ""]
    for topic in large_topics(facts, min_size=min_size):
        key = topic.get("seed_hash") or topic["label"]
        blocks = [r for r in readings if key in r.get("topics", {})]
        if not blocks:
            continue
        out += [f"### {topic['label']}", "", _topic_line(topic, lab), ""]
        for r in blocks:
            out += _block(
                r["topics"][key],
                r.get("label") or r["model_id"],
                lab,
                allowed,
                attribute=attribute,
            )
    return "\n".join(out).rstrip() + "\n"


def build_annotated(report: str, facts: dict, readings: list, lang: str, min_size: int):
    """The report with the corpus reading up front and each area's reading in place.

    Per subject area rather than per topic, and that is a decision about reading rather
    than about coverage: section 3 already repeats a table for every area, and a comment
    under each one is read in the flow. Two hundred topic readings appended as an
    appendix would be read by nobody, so they stay in the commentary document where they
    can be looked up.
    """
    lab = HEAD[lang]
    allowed = facts_numbers(facts)
    attribute = len(readings) > 1
    corpus: list[str] = []
    for r in readings:
        if "corpus" in r:
            corpus += _block(
                r["corpus"],
                r.get("label") or r["model_id"],
                lab,
                allowed,
                attribute=attribute,
            )

    out: list[str] = []
    placed = False
    section = ""
    pending: list[str] = []
    for line in report.splitlines():
        if line.startswith("## "):
            out += pending
            pending = []
            section = line[3:].split(".", 1)[0].strip()
            if not placed:
                out += [f"## {lab['corpus']}", "", *corpus]
                placed = True
        elif line.startswith(PAGE_BREAK[0]):
            # A page break opens the next area. The reading buffered for the previous one
            # has to be flushed before it, or the commentary is torn off its own table
            # and printed at the top of the following area's page.
            out += pending
            pending = []
        elif line.startswith("### ") and section == "3":
            out += pending
            area = line[4:].strip()
            pending = []
            for r in readings:
                block = r.get("areas", {}).get(area)
                if block:
                    pending += _block(
                        block,
                        r.get("label") or r["model_id"],
                        lab,
                        allowed,
                        attribute=attribute,
                    )
            if pending:
                pending = ["", *pending]
        out.append(line)
    out += pending
    if not placed:
        out += [f"## {lab['corpus']}", "", *corpus]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    """Write both documents beside the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("facts", type=Path, help="the report's facts.json")
    ap.add_argument("readings", type=Path, nargs="+", help="one JSON per model")
    ap.add_argument("--report", type=Path, default=None, help="the report markdown")
    ap.add_argument(
        "--out", type=Path, default=None, help="directory (default: facts')"
    )
    ap.add_argument("--min-size", type=int, default=200)
    ap.add_argument(
        "--render",
        action="store_true",
        help="Also write the PDF and DOCX, through the same recipe the report uses.",
    )
    args = ap.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    readings = merge_readings(
        [json.loads(p.read_text(encoding="utf-8")) for p in args.readings]
    )
    langs = {r.get("lang", "uk") for r in readings}
    if len(langs) > 1:
        raise SystemExit(f"error: readings are in different languages: {sorted(langs)}")
    lang = langs.pop()
    out_dir = args.out or args.facts.parent
    stem = args.facts.name.replace("-facts.json", "")
    suffix = "-ua" if lang == "uk" else ""

    written = []
    doc = build_interpretation(facts, readings, lang, args.min_size)
    path = out_dir / f"{stem}-interpretation{suffix}.md"
    path.write_text(doc, encoding="utf-8")
    written.append(path)
    print(f"  wrote {path}")

    if args.report:
        annotated = build_annotated(
            args.report.read_text(encoding="utf-8"),
            facts,
            readings,
            lang,
            args.min_size,
        )
        path = out_dir / f"{stem}-annotated{suffix}.md"
        path.write_text(annotated, encoding="utf-8")
        written.append(path)
        print(f"  wrote {path}")

    if args.render:
        # Through horizon.render rather than a copy of the flags: the annotated PDF
        # carried floating, double-numbered figures for a day because this script
        # repeated the recipe by hand and the fix went into the CLI.
        from horizon.render import render

        for md in written:
            print(render(md, md.with_suffix(".pdf"), pdf=True))
            print(render(md, md.with_suffix(".docx"), pdf=False))

    allowed = facts_numbers(facts)
    flagged = sum(
        1
        for r in readings
        for block in [r.get("corpus"), *r.get("topics", {}).values()]
        if block and unsupported_numbers(clean_reading(block["text"]), allowed)
    )
    print(f"  {flagged} readings cite a figure the facts do not contain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
