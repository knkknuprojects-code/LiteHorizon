#!/usr/bin/env python3
"""Build a slide deck from a report's ``facts.json``.

    python scripts/build_deck.py out/sample/sci3-2026-08-27-facts.json --lang uk

The deck is a *selection* from the report, not a copy of it: 65 pages of tables do not
become slides by being cut into slide-sized pieces. Each slide carries one claim and the
smallest table that supports it, and the speaker notes carry the definition the audience
would otherwise have to be told out loud.

Facts come from ``facts.json`` and nowhere else, so the same invariant the report holds
applies here — every number on a slide is traceable to the file, and the deck regenerates
from a later corpus without being re-authored.

Markdown goes through pandoc to ``.pptx``. ``--reference-doc`` accepts a template deck if
one is ever wanted; without it pandoc uses its default master, which is plain and legible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from horizon.report import LABELS  # noqa: E402

#: Rows per slide. Eight is about what stays legible projected; the report has the rest.
ROWS = 8

#: Indicator -> (question for the slide, label key for its column, how to render it).
#: Mirrors ``report._META_COLUMNS`` and the indicator definitions, as a table rather than
#: a chain of conditionals so a renamed indicator fails loudly instead of falling into a
#: default and printing the wrong column.
INDICATORS: Mapping[str, tuple[str, str, str, Callable[[Mapping], str]]] = {
    "declining": (
        "Згасання публікаційної активності",
        "Where established volume is going quiet",
        "decay",
        lambda r: f"{r['size'] * max(0.0, 1 - r['velocity']):.2f}",
    ),
    "high-impact-low-output": (
        "Високий цитувальний вплив за малого обсягу публікацій",
        "Small bodies of work drawing outsized citation",
        "attention",
        lambda r: f"{r['attention_pctile']:.2f}",
    ),
    "high-output-low-impact": (
        "Великий обсяг публікацій за низького цитувального впливу",
        "Much work, little uptake",
        "attention",
        lambda r: f"{r['attention_pctile']:.2f}",
    ),
    "sleeping-beauty": (
        "Відкладене визнання",
        "Dormant for years, then taking off",
        "dormancy",
        lambda r: f"{_dormancy(r):.0f}",
    ),
    "sustained": (
        "Постпікова стійкість",
        "Held its level after peaking",
        "sustain",
        lambda r: f"{_sustain(r):.0f}",
    ),
    "journal-dispersion": (
        "Розсіювання за виданнями",
        "Venue set growing faster than output explains",
        "venues",
        lambda r: f"{r['venue_spread']:.2f}×",
    ),
    "new-authors": (
        "Оновлення авторського складу",
        "Share of members from authors absent earlier",
        "newshare",
        lambda r: f"{r['new_author_share']:.0%}",
    ),
    "script-distribution": (
        "Зосередження кирилиці",
        "Share of publications written in Cyrillic",
        "scripts",
        lambda r: f"{r['cyrillic']} / {r['latin']}",
    ),
}


def _dormancy(row: Mapping) -> float:
    """Years between a topic's first publication and its peak cohort."""
    counts = row.get("cohort_counts") or {}
    if not counts:
        return 0.0
    peak = max(counts.items(), key=lambda kv: (kv[1], -int(kv[0])))[0]
    return float(int(peak) - int(row["first_seen"][:4]))


def _sustain(row: Mapping) -> float:
    """Years after the peak still producing at least half of it."""
    counts = {int(y): n for y, n in (row.get("cohort_counts") or {}).items()}
    if not counts:
        return 0.0
    peak = max(counts, key=lambda y: (counts[y], -y))
    return float(
        sum(1 for y, n in counts.items() if y > peak and n >= counts[peak] / 2)
    )


def _num(n: int) -> str:
    """Thousands separated by a narrow space, as Ukrainian typography sets them."""
    return f"{n:,}".replace(",", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A pipe table pandoc turns into a native pptx table."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def _notes(text: str) -> list[str]:
    """Speaker notes: what the slide asserts, and what the number means."""
    return ["::: notes", text, ":::", ""]


def _classified_years(facts: Mapping[str, Any]) -> list[int]:
    """The cohorts the share columns were read at.

    Derived from the per-area counts when ``facts.json`` predates the field carrying it,
    rather than falling back to the corpus years: those differ exactly when the last
    cohort has no taxonomy, which is the case the label exists to name. Falling back
    would print a span the numbers were not measured over — silently, and plausibly.
    """
    stored = facts.get("classified_years")
    if stored:
        return [int(y) for y in stored]
    seen = {int(year) for f in facts["fields"] for year, n in f["counts"].items() if n}
    return sorted(seen) or [int(y) for y in facts["years"]]


def build_deck(facts: Mapping[str, Any], *, lang: str = "uk", title: str = "") -> str:
    """The whole deck as pandoc markdown."""
    lab = LABELS[lang]
    years = facts["years"]
    classified = _classified_years(facts)
    fields = facts["fields"]
    out: list[str] = []

    out += [
        "---",
        f'title: "{title or "Тематичний зріз корпусу WoS+Scopus"}"',
        f'subtitle: "{years[0]}–{years[-1]} · {_num(facts["records"])} публікацій"',
        "---",
        "",
    ]

    share = facts["unclassified"] / max(facts["records"], 1)
    out += [f"## Корпус", ""]
    out += _table(
        ["Показник", "Значення"],
        [
            ["Публікацій", _num(facts["records"])],
            ["Роки", f"{years[0]}–{years[-1]}"],
            ["Предметних областей", str(len(fields))],
            ["Роки з класифікацією", f"{classified[0]}–{classified[-1]}"],
            ["Без предметної області", f"{_num(facts['unclassified'])} ({share:.0%})"],
            ["Пар областей", _num(len(facts["pairs"]))],
        ],
    )
    out += _notes(
        "Об'єднане вивантаження WoS і Scopus, зведене до унікальних записів за DOI, "
        "потім за назвою. Дублікати злито, а не відкинуто: запис успадковує показники "
        "цитування з обох баз."
    )

    out += ["## Що і на чому виміряно", ""]
    out += _table(
        ["Рівень", "Основа", "Потребує моделі"],
        [
            ["Розвиток областей", "таксономія WoS/Scopus", "ні"],
            ["Співпоява областей", "та сама таксономія", "ні"],
            ["Теми", "кластеризація ембедингів", "так"],
            ["Індикатори", "відбір і впорядкування тем", "ні"],
        ],
    )
    out += _notes(
        "Перші два рівні не залежать ані від моделі, ані від порога кластеризації, і "
        "тому лишаються чинними, хоч би що робила кластеризація. Індикатори не "
        "обчислюють нових величин — це селекція з упорядкуванням."
    )

    gain = fields[:ROWS]
    loss = [f for f in fields if f["share_delta"] < 0][-ROWS:]
    for heading, rows, note in (
        (
            f"Області, що набирають частку · {classified[0]}–{classified[-1]}",
            gain,
            "Частки взято за перший і останній роки, для яких є класифікація, а не за "
            "краями корпусу: останній рік може існувати без призначеної йому таксономії.",
        ),
        (
            f"Області, що втрачають частку · {classified[0]}–{classified[-1]}",
            list(reversed(loss)),
            "Нахил рахується по всіх роках: область, що перестала з'являтися, справді "
            "впала, і пропуск року тут є вимірюванням, а не діркою.",
        ),
    ):
        out += [f"## {heading}", ""]
        out += _table(
            [lab["field"], lab["first"], lab["last"], lab["delta"], lab["n"]],
            [
                [
                    f["name"],
                    f"{f['share_first']:.1%}",
                    f"{f['share_last']:.1%}",
                    f"{f['share_delta'] * 100:+.1f} п.п.",
                    _num(f["total"]),
                ]
                for f in rows
            ],
        )
        out += _notes(note)

    out += ["## Міждисциплінарні шви", ""]
    out += _table(
        [lab["pair"], lab["slope"], lab["n"]],
        [
            [p["name"], f"{p['slope']:+.1f}", _num(p["total"])]
            for p in facts["pairs"][:ROWS]
        ],
    )
    out += _notes(
        "Понад 45% записів віднесено більш ніж до однієї предметної області, тож "
        "співпоява є прямим вимірюванням міждисциплінарності, а не непрямою ознакою."
    )

    out += ["## Теми по всьому корпусу", ""]
    out += _table(
        [lab["trend"], lab["size"], lab["velocity"], lab["growth"]],
        [
            [
                t["label"],
                _num(t["size"]),
                f"{t['velocity']:.2f}",
                f"{t['growth']:+.1f}",
            ]
            for t in facts["overall"][:ROWS]
        ],
    )
    out += _notes(
        "Відносне зростання — відношення кількості публікацій теми за останній період "
        "до кількості за попередній період тієї самої тривалості, нормалізоване на "
        "аналогічне відношення для корпусу. Значення 1.0 відповідає темпові корпусу, "
        "менші — відставанню, навіть якщо тема щороку збільшує кількість публікацій. "
        "Публікації/рік — коефіцієнт нахилу лінійної регресії річних кількостей самої "
        "теми, без нормалізації на корпус."
    )

    for name, (heading, question, key, fmt) in INDICATORS.items():
        rows = facts["lenses"].get(name) or facts["meta_lenses"].get(name)
        if not rows:
            continue
        out += [f"## {heading}", ""]
        out += _table(
            [lab["trend"], lab["size"], lab[key]],
            [[r["label"], _num(r["size"]), fmt(r)] for r in rows[:ROWS]],
        )
        out += _notes(
            f"{question}. Поріг — 20 публікацій: нижче нього ця величина "
            "описує одну-дві праці, а не корпус робіт."
        )

    out += ["## Межі тверджень", ""]
    out += [
        f"- {lab['caveat']}",
        "- Теми залежать від порога кластеризації 0,62, обраного вимірюванням на "
        "вибірці; рівні 1–2 від нього не залежать.",
        "- Цитованість подано перцентилем у межах року публікації, тож ранні й пізні "
        "роки порівнянні між собою.",
        "",
    ]
    out += _notes(
        "Поріг переглядається за хвилини на вже наявних аналізах, без повторного "
        "інференсу, тому це налаштовуваний параметр, а не властивість даних."
    )
    return "\n".join(out)


def main() -> int:
    """Render ``facts.json`` to markdown and hand it to pandoc."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("facts", type=Path, help="a report's -facts.json")
    ap.add_argument("--lang", default="uk", choices=sorted(LABELS))
    ap.add_argument("--out", type=Path, default=None, help="output .pptx")
    ap.add_argument("--title", default="", help="deck title")
    ap.add_argument(
        "--reference-doc", type=Path, default=None, help="template .pptx for the master"
    )
    ap.add_argument(
        "--keep-md", action="store_true", help="leave the markdown beside it"
    )
    args = ap.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    deck = build_deck(facts, lang=args.lang, title=args.title)

    out = args.out or args.facts.with_name(
        args.facts.name.replace("-facts.json", f"-deck-{args.lang}.pptx")
    )
    md = out.with_suffix(".md")
    md.write_text(deck, encoding="utf-8")
    cmd = ["pandoc", str(md), "-o", str(out), "--slide-level=2"]
    if args.reference_doc:
        cmd += ["--reference-doc", str(args.reference_doc)]
    subprocess.run(cmd, check=True)
    if not args.keep_md:
        md.unlink()
    slides = deck.count("\n## ")
    print(f"  wrote {out} ({slides} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
