#!/usr/bin/env python3
"""Compare two corpora over the years both of them cover.

Horizon's report is written for one corpus at a time, and rightly so: every share in it
is a share of that corpus. Putting two corpora in one document is therefore not a bigger
report but a different one, and this script is the whole of it — it reads two
``-facts.json`` files, keeps only what both can be asked, and writes the comparison in
both languages.

Three rules decide everything below.

**The window is intersected twice.** Once on the cohorts each corpus contains, which is
what "the years present in both" means, and again on the cohorts each corpus *fits* its
own derived columns over. A patent register trims a trailing run of cohorts that are
still filling up with grants (``patents.completeness``), and a bibliographic corpus trims
a gathering stub; those trims are already recorded in ``fit_years``, so honouring them
costs nothing and ignoring them would put an admittedly incomplete cohort at the end of
every series here. The presence window is shown, the fitted window is measured over, and
the difference between the two is printed rather than quietly applied.

**Nothing is compared in absolute units.** One corpus is a twelfth the size of the other,
and both are exports with their own scope, so a count from one against a count from the
other says only which export is bigger. Everything derived is a share of its own corpus
and a change in that share — the same discipline the single-corpus report applies to
fields, applied here to the corpora themselves.

**The taxonomies are not bridged by hand.** WIPO technology fields and Scopus subject
areas do not map onto one another, and writing down a mapping would be a judgement
dressed as a measurement. What the two corpora do share is the analysis layer: the same
model, the same prompt and the same embedder produced topic labels for both, in English,
without being told which was which. A label that occurs in both is therefore a measured
join, and §4 and §5 are built on it — with its weakness stated, because a shared label is
evidence that two clusters were described alike, not proof that they hold the same work.

Usage::

    python scripts/compare_corpora.py A-facts.json B-facts.json \\
        --out reports/<slug>-<date> --slug <slug> --date <date> \\
        --name-a "Ukrainian patent register" --name-a-uk "Патентний реєстр України" \\
        --unit-a patents --unit-a-uk патентів \\
        --taxonomy-a "WIPO technology fields" --taxonomy-a-uk "технологічні поля WIPO" \\
        --coverage-a datasets/patents/build-report.json \\
        ... the same six for --*-b ... \\
        --render
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Topics smaller than this on either side are matched and counted, but not used to test
#: whether the two landscapes move together. A six-year share slope fitted over four
#: records is noise, and at 129 matched labels enough of them are that small to decide the
#: test on their own.
MIN_MATCH_SIZE = 20

#: Everything the two languages differ by. One mapping rather than two renderers, for the
#: reason ``report.LABELS`` gives: a change to the structure cannot then land in one
#: language only.
LABELS: dict[str, dict[str, str]] = {
    "en": {
        "window": "Years",
        "measured_over": "Measured over",
        "present_in_both": "present in both",
        "l1": "1. The window, and why it is shorter than the overlap",
        "l1_note": (
            "The cohorts both corpora contain, and the cohorts both report as complete. "
            "A corpus trims a trailing run of cohorts that are still filling up; this "
            "table is where those trims meet."
        ),
        "l2": "2. Volume",
        "l2_note": (
            "Records per cohort, and the same series indexed to the first year of the "
            "measured window. The counts are not comparable to one another — the exports "
            "differ in size and in scope — so the index is the column to read across."
        ),
        "l3": "3. Each taxonomy over the window",
        "l3_note": (
            "Share of its own corpus at the ends of the measured window, per field. The "
            "two taxonomies are not mapped onto one another anywhere in this document; "
            "what is comparable is the shape of each list and which way its ends moved."
        ),
        "l3_shape": "3.1 The shape of the two taxonomies",
        "l3_pairs": "3.2 Co-classification",
        "l3_pairs_note": (
            "Pairs of fields a record carries together, ranked by how fast that pair's "
            "share of its own corpus rose — the seams each landscape opened over the "
            "window. Only the fastest-rising are listed; a pair is not a topic, and the "
            "two lists are not matched to one another."
        ),
        "l4": "4. Topics that carry the same label in both corpora",
        "l4_note": (
            "The analysis layer was never told which corpus it was reading, and it "
            "labels in English from either. A label occurring in both is therefore a "
            "join neither taxonomy provides — and a weak one: it says two clusters were "
            "described alike, not that they hold the same work. Keywords are printed so "
            "that can be judged rather than assumed."
        ),
        "l5": "5. Where the matched topics sit",
        "l5_note": (
            "Each matched label carries a field on one side and a classification block "
            "on the other, so the matches induce a crosswalk instead of asserting one. "
            "Only links carried by at least two distinct labels are listed."
        ),
        "l6": "6. What this document does not compare",
        "year": "year",
        "records": "records",
        "index": "index",
        "coverage": "est. coverage",
        "corrected": "corrected",
        "status": "status",
        "measured": "measured",
        "excluded": "present, not measured",
        "field": "field",
        "share_first": "first",
        "share_last": "last",
        "change": "change",
        "per_year": "pp/year",
        "in_window": "in window",
        "metric": "metric",
        "fields_n": "fields in the taxonomy",
        "multi": "field assignments per record",
        "top5": "five largest fields",
        "effective": "effective number of fields",
        "pairs_n": "co-occurring pairs",
        "pair": "pair of fields",
        "topic": "topic",
        "keywords": "keywords",
        "size": "size",
        "share_slope": "share slope",
        "rising": "4.1 Rising on both sides",
        "falling": "4.2 Falling on both sides",
        "diverging": "4.3 Moving opposite ways",
        "largest": "4.4 The largest matched topics",
        "link": "field ↔ block",
        "labels_n": "labels",
        "weight": "records",
    },
    "uk": {
        "window": "Роки",
        "measured_over": "Вимірювання за",
        "present_in_both": "присутні в обох",
        "l1": "1. Вікно, і чому воно коротше за перетин",
        "l1_note": (
            "Когорти, які містять обидва корпуси, і когорти, які обидва вважають "
            "повними. Корпус відтинає кінцевий відрізок когорт, що ще наповнюються; ця "
            "таблиця — місце, де ці відтинання зустрічаються."
        ),
        "l2": "2. Обсяг",
        "l2_note": (
            "Записів на когорту і той самий ряд, індексований до першого року "
            "виміряного вікна. Самі кількості між корпусами незіставні — експорти "
            "різняться і розміром, і охопленням, — тож читати впоперек треба індекс."
        ),
        "l3": "3. Кожна таксономія на вікні",
        "l3_note": (
            "Частка власного корпусу на кінцях виміряного вікна, за полями. Ніде в "
            "цьому документі дві таксономії не відображаються одна на одну; зіставними "
            "є форма кожного списку і напрямок руху його кінців."
        ),
        "l3_shape": "3.1 Форма двох таксономій",
        "l3_pairs": "3.2 Спільна класифікація",
        "l3_pairs_note": (
            "Пари полів, які запис несе разом, упорядковані за тим, як швидко зростала "
            "частка цієї пари у власному корпусі, — шви, що їх відкрив кожен ландшафт на "
            "вікні. Наведено лише ті, що зростають найшвидше; пара — це не тема, і два "
            "списки не зіставлено один з одним."
        ),
        "l4": "4. Теми з однаковою назвою в обох корпусах",
        "l4_note": (
            "Аналітичному шару ніколи не повідомляли, який корпус він читає, і назви він "
            "дає англійською з будь-якого. Тож назва, що трапилася в обох, — це "
            "зчеплення, якого не дає жодна таксономія, і зчеплення слабке: воно каже, що "
            "два кластери описано однаково, а не що вони містять ту саму роботу. Ключові "
            "слова наведено, щоб це можна було оцінити, а не припустити."
        ),
        "l5": "5. Де опиняються зіставлені теми",
        "l5_note": (
            "Кожна зіставлена назва несе поле з одного боку і класифікаційний блок з "
            "іншого, тож зіставлення породжують відповідність замість того, щоб її "
            "постулювати. Наведено лише зв’язки, які тримають щонайменше дві різні назви."
        ),
        "l6": "6. Що цей документ не порівнює",
        "year": "рік",
        "records": "записів",
        "index": "індекс",
        "coverage": "оцінка повноти",
        "corrected": "з поправкою",
        "status": "статус",
        "measured": "виміряно",
        "excluded": "присутній, не виміряно",
        "field": "поле",
        "share_first": "початок",
        "share_last": "кінець",
        "change": "зміна",
        "per_year": "в.п./рік",
        "in_window": "у вікні",
        "metric": "показник",
        "fields_n": "полів у таксономії",
        "multi": "класифікацій на запис",
        "top5": "п’ять найбільших полів",
        "effective": "ефективна кількість полів",
        "pairs_n": "пар, що трапляються разом",
        "pair": "пара полів",
        "topic": "тема",
        "keywords": "ключові слова",
        "size": "розмір",
        "share_slope": "нахил частки",
        "rising": "4.1 Зростають з обох боків",
        "falling": "4.2 Спадають з обох боків",
        "diverging": "4.3 Рухаються в різні боки",
        "largest": "4.4 Найбільші зіставлені теми",
        "link": "поле ↔ блок",
        "labels_n": "назв",
        "weight": "записів",
    },
}


def slope(values: list[float]) -> float:
    """Least-squares slope per step over evenly spaced points."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(values) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / den if den else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    """Correlation of two equal-length series; 0.0 when either is constant."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if not sx or not sy:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


@dataclass
class Side:
    """One corpus, with the names it is called by and the years it can be read over."""

    facts: dict
    name: dict[str, str]
    #: Language -> the three numeral forms of the word for one record.
    unit: dict[str, tuple[str, str, str]]
    taxonomy: dict[str, str]
    #: Cohort -> estimated share of its eventual records already present. Read from the
    #: ingest's build report when there is one; empty when the corpus has no such notion.
    coverage: dict[int, float] = field(default_factory=dict)
    #: Topic label (lowercased) -> the per-year counts and the blocks it was clustered in.
    topics: dict[str, dict] = field(default_factory=dict)

    @property
    def years(self) -> list[int]:
        return sorted(int(y) for y in self.facts["year_totals"])

    @property
    def fitted(self) -> list[int]:
        fit = self.facts.get("fit_years") or self.facts["years"]
        return sorted(int(y) for y in fit)

    def total(self, year: int) -> int:
        return self.facts["year_totals"][str(year)]


def load_side(path: Path, args, which: str) -> Side:
    """Read a facts file and the display names it is to be called by."""
    facts = json.loads(path.read_text(encoding="utf-8"))

    def both(key: str) -> dict[str, str]:
        """``--name-a`` / ``--name-a-uk`` as the two-language mapping the renderer wants."""
        return {
            "en": getattr(args, f"{key}_{which}"),
            "uk": getattr(args, f"{key}_{which}_uk"),
        }

    coverage: dict[int, float] = {}
    build = getattr(args, f"coverage_{which}")
    if build:
        raw = json.loads(Path(build).read_text(encoding="utf-8")).get("coverage") or {}
        coverage = {int(y): float(c) for y, c in raw.items()}
    side = Side(
        facts=facts,
        name=both("name"),
        unit={k: _forms(v) for k, v in both("unit").items()},
        taxonomy=both("taxonomy"),
        coverage=coverage,
    )
    side.topics = _index_topics(facts)
    return side


def _index_topics(facts: dict) -> dict[str, dict]:
    """Level-3 topics, folded by label.

    Level 3 is used rather than the corpus-wide level because it is the one both corpora
    hold in the same form: the corpus-wide lists are capped at a reporting limit that was
    set per run, so their lengths are a decision about the report rather than a property
    of the landscape.

    Within level 3 a topic can be listed under more than one block, because one corpus was
    clustered unblocked and a record there carries several fields. The two halves of that
    are counted differently on purpose. **Volume** is deduplicated by label and seed, or a
    topic would be multiplied by its own interdisciplinarity — and the corpora differ in
    how multiply classified they are, so the inflation would land on one side of the
    comparison only. **Blocks** are collected from every listing, because §5 asks which
    fields a topic touches and dropping all but the first would answer a different
    question. The block weight is therefore the topic's size under each block it appears
    in, not a partition of that size.
    """
    out: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for block, trends in (facts.get("field_trends") or {}).items():
        for trend in trends:
            label = trend["label"].lower()
            row = out.setdefault(
                label,
                {
                    "label": trend["label"],
                    "counts": Counter(),
                    "size": 0,
                    "blocks": Counter(),
                    "keywords": Counter(),
                },
            )
            row["blocks"][block] += trend["size"]
            key = (trend["label"], trend.get("seed_hash", ""))
            if key in seen:
                continue
            seen.add(key)
            row["counts"].update(
                {int(y): n for y, n in (trend.get("cohort_counts") or {}).items()}
            )
            row["size"] += trend["size"]
            for word in trend.get("top_keywords") or ():
                row["keywords"][word] += 1
    return out


# --------------------------------------------------------------------------- measuring


def windows(a: Side, b: Side) -> tuple[list[int], list[int]]:
    """The cohorts both corpora contain, and the subset both of them fit over."""
    present = sorted(set(a.years) & set(b.years))
    measured = [y for y in present if y in set(a.fitted) and y in set(b.fitted)]
    return present, measured


def volume(side: Side, present: list[int], measured: list[int]) -> list[dict]:
    """Per-cohort counts, a coverage-corrected count where one can be had, and an index."""
    base = side.total(measured[0])
    rows = []
    for year in present:
        n = side.total(year)
        cov = side.coverage.get(year)
        corrected = n / cov if cov else None
        rows.append(
            {
                "year": year,
                "n": n,
                "coverage": cov,
                "corrected": corrected,
                "index": 100 * n / base,
                "measured": year in measured,
            }
        )
    return rows


def fields(side: Side, measured: list[int]) -> list[dict]:
    """Every field's share of its own corpus at the ends of the measured window."""
    totals = {y: side.total(y) for y in measured}
    rows = []
    for f in side.facts["fields"]:
        counts = {int(y): n for y, n in f["counts"].items()}
        shares = [counts.get(y, 0) / totals[y] for y in measured]
        rows.append(
            {
                "name": f["name"],
                "n": sum(counts.get(y, 0) for y in measured),
                "first": shares[0],
                "last": shares[-1],
                "delta": shares[-1] - shares[0],
                "slope": slope(shares),
            }
        )
    rows.sort(key=lambda r: -r["delta"])
    return rows


def shape(side: Side, rows: list[dict], measured: list[int]) -> dict:
    """The taxonomy's own geometry: how many fields, how skewed, how multiply applied."""
    assignments = sum(r["n"] for r in rows)
    records = sum(side.total(y) for y in measured)
    largest = sorted(rows, key=lambda r: -r["n"])[:5]
    hhi = sum((r["n"] / assignments) ** 2 for r in rows) if assignments else 0.0
    return {
        "fields": len(rows),
        "multi": assignments / records if records else 0.0,
        "top5": sum(r["n"] for r in largest) / assignments if assignments else 0.0,
        "effective": 1 / hhi if hhi else 0.0,
        "pairs": len(side.facts.get("pairs") or ()),
    }


def pairs(side: Side, measured: list[int], top: int) -> list[dict]:
    """Field pairs by how fast their share of the corpus moved inside the window."""
    totals = {y: side.total(y) for y in measured}
    rows = []
    for p in side.facts.get("pairs") or ():
        counts = {int(y): n for y, n in p["counts"].items()}
        n = sum(counts.get(y, 0) for y in measured)
        if not n:
            continue
        shares = [counts.get(y, 0) / totals[y] for y in measured]
        rows.append(
            {
                "name": p["name"],
                "n": n,
                "first": shares[0],
                "last": shares[-1],
                "slope": slope(shares),
            }
        )
    rows.sort(key=lambda r: -r["slope"])
    return rows[:top]


def matches(a: Side, b: Side, measured: list[int]) -> list[dict]:
    """Topics whose label occurs in both corpora, with each side's window trajectory."""
    ta = {y: a.total(y) for y in measured}
    tb = {y: b.total(y) for y in measured}
    rows = []
    for label in sorted(set(a.topics) & set(b.topics)):
        ra, rb = a.topics[label], b.topics[label]
        na = [ra["counts"].get(y, 0) for y in measured]
        nb = [rb["counts"].get(y, 0) for y in measured]
        sa = [n / ta[y] for n, y in zip(na, measured)]
        sb = [n / tb[y] for n, y in zip(nb, measured)]
        rows.append(
            {
                "label": ra["label"],
                "a_n": sum(na),
                "b_n": sum(nb),
                # Shares per ten thousand: the raw figures run to five leading zeros and
                # a table of them is unreadable.
                "a_first": sa[0] * 1e4,
                "a_last": sa[-1] * 1e4,
                "b_first": sb[0] * 1e4,
                "b_last": sb[-1] * 1e4,
                "a_slope": slope(sa) * 1e4,
                "b_slope": slope(sb) * 1e4,
                # Six rather than the three the table prints: the commentary layer has
                # no summaries to read on either side (see scripts/compare_interpret.py),
                # so the keywords are the whole of what says what a cluster is about.
                "a_keywords": [w for w, _ in ra["keywords"].most_common(6)],
                "b_keywords": [w for w, _ in rb["keywords"].most_common(6)],
                "a_blocks": ra["blocks"],
                "b_blocks": rb["blocks"],
            }
        )
    return rows


def headline(va: list[dict], vb: list[dict], measured: list[int]) -> dict:
    """The one comparison that needs no bridge: how far apart the two volumes ended.

    ``crossed`` is the first measured cohort after the base in which one index sits above
    100 and the other below — the year the two series stop being the same story, stated as
    a year rather than as an adjective.
    """
    last = measured[-1]
    ia = {r["year"]: r for r in va}
    ib = {r["year"]: r for r in vb}
    base = ia[measured[0]]["corrected"]
    corrected = (
        100 * (ia[last]["corrected"] / base - 1)
        if ia[last]["corrected"] and base
        else 0.0
    )
    crossed = next(
        (y for y in measured[1:] if (ia[y]["index"] > 100) != (ib[y]["index"] > 100)),
        last,
    )
    return {
        "a": ia[last]["index"] - 100,
        "a_corrected": corrected,
        "b": ib[last]["index"] - 100,
        "crossed": crossed,
    }


def agreement(rows: list[dict], floor: int) -> dict:
    """Whether the matched topics move the same way on both sides.

    Directions are read off the share slope, not the count slope. A register that halves
    drags every count in it downward, so counts would measure the corpus rather than the
    topic; the share says whether a topic gained ground inside its own landscape.
    """
    big = [r for r in rows if r["a_n"] >= floor and r["b_n"] >= floor]
    same = sum(1 for r in big if (r["a_slope"] > 0) == (r["b_slope"] > 0))
    return {
        "n": len(big),
        "same": same,
        "all_n": len(rows),
        "all_same": sum(1 for r in rows if (r["a_slope"] > 0) == (r["b_slope"] > 0)),
        "r": pearson([r["a_slope"] for r in big], [r["b_slope"] for r in big]),
        "rows": big,
    }


def crosswalk(rows: list[dict], min_labels: int) -> list[dict]:
    """The field-to-block links the matched labels induce, weighted by the smaller side.

    The weight is the topic's window count on whichever side is smaller — the same
    restriction every other number in the document is under, and the conservative choice
    of the two, since a link is only as well evidenced as its thinner end. The label count
    is what the list is ranked by: one large topic landing in an odd block should not
    outrank a link four separate topics agree on.
    """
    weight: Counter = Counter()
    labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for fa in row["a_blocks"]:
            for fb in row["b_blocks"]:
                weight[(fa, fb)] += min(row["a_n"], row["b_n"])
                labels[(fa, fb)].add(row["label"])
    out = [
        {"a": fa, "b": fb, "weight": w, "labels": len(labels[(fa, fb)])}
        for (fa, fb), w in weight.items()
        if len(labels[(fa, fb)]) >= min_labels
    ]
    out.sort(key=lambda r: (-r["labels"], -r["weight"]))
    return out


def facts_dict(a: Side, b: Side, present, measured, data) -> dict:
    """Everything the two documents are drawn from, JSON-ready.

    Written beside the comparison for the reason the single-corpus report writes one: the
    commentary layer is checked against it. :func:`horizon.interpret.facts_numbers` walks
    whatever shape it is given, so a figure invented by a model is caught by the same
    mechanism here as there — which only works if every figure the documents print is in
    this file.
    """

    def side(x: Side) -> dict:
        return {
            "name": x.name,
            "unit": list(x.unit["en"]),
            "taxonomy": x.taxonomy,
            "records": {str(y): x.total(y) for y in present},
            # The window total, stored rather than left to be summed. The blockquote
            # prints it and the reading prompt hands it to the model, so a facts file
            # without it makes the commentary's own check flag a figure this code
            # supplied — which is how it was found.
            "records_in_window": sum(x.total(y) for y in measured),
            "coverage": {str(y): c for y, c in sorted(x.coverage.items())},
            "topics": len(x.topics),
        }

    def match(r: dict) -> dict:
        out = {k: v for k, v in r.items() if k not in ("a_blocks", "b_blocks")}
        out["a_blocks"] = dict(r["a_blocks"])
        out["b_blocks"] = dict(r["b_blocks"])
        return out

    agr = dict(data["agreement"])
    agr["rows"] = [r["label"] for r in agr["rows"]]
    return {
        "kind": "comparison",
        "present_years": present,
        "measured_years": measured,
        "a": side(a),
        "b": side(b),
        "volume": {"a": data["volume_a"], "b": data["volume_b"]},
        "fields": {"a": data["fields_a"], "b": data["fields_b"]},
        "shape": {"a": data["shape_a"], "b": data["shape_b"]},
        "pairs": {"a": data["pairs_a"], "b": data["pairs_b"]},
        "headline": data["headline"],
        "matches": [match(r) for r in data["matches"]],
        "agreement": agr,
        "crosswalk": data["crosswalk"],
        "match_totals": {"a": data["match_a_n"], "b": data["match_b_n"]},
        # Recorded so the reading pass cannot describe the direction test with a floor
        # the comparison did not use: it only labels a sentence in the prompt, which is
        # exactly the kind of number nobody checks.
        "match_floor": MIN_MATCH_SIZE,
    }


# ------------------------------------------------------------------------- commentary

#: Everything the two companion documents say in their own voice. Both languages carried
#: for the reason :data:`LABELS` is, even though the companions are produced in Ukrainian
#: only: a structure that exists in one language cannot be added to the other later if it
#: was never written down.
HEAD: dict[str, dict[str, str]] = {
    "uk": {
        "title": "Міркування до порівняння",
        "lede": (
            "Це прочитання моделей, а не вимірювання. Порівняння стверджує, що виміряно, "
            "і кожне його число простежується до `-facts.json`; цей документ каже, що "
            "рух може означати, а це вже судження."
        ),
        "thin": (
            "Ці прочитання тонші за коментарі до окремих корпусів, і причина названа: "
            "сховище аналізів більшого з двох масивів на цій машині відсутнє — його "
            "сканували на кластері, а повернулися звіт і факти. Тож моделі не отримали "
            "жодного резюме публікації чи патента, лише ключові слова, кількості, "
            "траєкторії часток і блоки. Дати резюме одному боку й не дати іншому "
            "змістило б кожне судження «чи це про одне й те саме» на бік із даними."
        ),
        "checked": (
            "Перевірено механічно лише те, що можна: кожен числівник звірено з фактами, "
            "обірвані та зациклені генерації позначено. Напрямок не перевіряється — "
            "речення, яке каже, що поле зростало, тоді як воно спадало, не містить "
            "жодного хибного числа і проходить усі перевірки. У цьому запуску таке "
            "траплялося. Читайте ці абзаци поряд із таблицями, а не замість них."
        ),
        "annotated_title": "Порівняння з міркуваннями",
        "annotated_lede": (
            "Те саме порівняння, доповнене прочитаннями моделей. Виміряне лишається "
            "виміряним: кожен згенерований абзац стоїть окремо від таблиць, на яких він "
            "заснований."
        ),
        "h_comparison": "Порівняння в цілому",
        "h_taxonomies": "Таксономії, кожна окремо",
        "h_matches": "Зіставлені теми як множина",
        "reading": "Прочитання",
        "unsupported": "Числа, яких немає у фактах",
        "truncated": "Обірвано на середині речення — збережено як є.",
        "diverging": "Напрямок розходиться з таблицею",
    },
    "en": {
        "title": "Commentary on the comparison",
        "lede": (
            "This is a reading by models, not a measurement. The comparison states what "
            "was measured and every number in it is traceable to `-facts.json`; this "
            "document says what a movement might mean, which is a judgement."
        ),
        "thin": (
            "These readings are thinner than a single-corpus commentary's, and the "
            "reason is on the record: the analyses store of the larger of the two bodies "
            "is not on this machine — it was scanned on the cluster and the report and "
            "facts are what came back. So the models were given no record summaries at "
            "all, only keywords, counts, share trajectories and blocks. Giving summaries "
            "to one side and not the other would bias every judgement about whether two "
            "clusters are the same thing toward the side with the evidence."
        ),
        "checked": (
            "Only what can be checked mechanically has been: every numeral is matched "
            "against the facts, and truncated or collapsed generations are marked. "
            "Direction is not checked — a sentence saying a field rose while it fell "
            "contains no wrong number and passes every check. That happened in this run. "
            "Read these paragraphs beside the tables, not instead of them."
        ),
        "annotated_title": "The comparison, annotated",
        "annotated_lede": (
            "The same comparison with the models' readings folded in. What was measured "
            "stays measured: every generated paragraph stands apart from the tables it "
            "was written from."
        ),
        "h_comparison": "The comparison as a whole",
        "h_taxonomies": "Each taxonomy on its own",
        "h_matches": "The matched topics as a set",
        "reading": "Reading",
        "unsupported": "Figures absent from the facts",
        "truncated": "Ended mid-sentence — kept as written.",
        "diverging": "Direction at odds with the table",
    },
}


#: Word stems that claim a direction. Stems rather than words because both languages
#: inflect them and the prose is generated, so the surface form is not predictable.
RISE = {
    "uk": ("зроста", "зросл", "зріс", "збільш", "підвищ", "приріст", "нарост"),
    "en": ("rise", "rose", "rising", "grow", "grew", "increase", "gain"),
}
FALL = {
    "uk": ("знижу", "зниже", "знижен", "скороч", "спад", "зменш", "падін", "втрат"),
    "en": ("fall", "fell", "falling", "decline", "decrease", "shrink", "loss"),
}

#: How far back from a field's name a direction word is taken to govern it. Measured
#: against the readings this was written for: at 80 characters it catches "significant
#: growth in the share of fields X and Y" and stops short of the previous clause.
DIRECTION_REACH = 80

#: Share movement, in share units, below which a field has no direction worth
#: contradicting. Without it the check fires on fields that did not move: a sector named
#: in passing inside a sentence about growth is not a claim that a flat field rose.
DIRECTION_FLOOR = 0.005


def _stem_pattern(name: str) -> str:
    """A field name as a regex tolerant of Ukrainian inflection.

    "Медичні технології" appears in generated prose as "Медичних технологій", so an exact
    match finds neither. Each word keeps a prefix and loses its ending, which is crude
    and enough: the names are distinctive well before their last two letters.

    A word of four letters or fewer keeps all of it, so an inflected short word is missed.
    That is the direction this should fail in — the check reports candidates, and one it
    never raises costs a reader nothing beyond what they had before it existed.
    """
    words = [w for w in re.findall(r"\w+", name.lower()) if len(w) > 2]
    # ``\w*`` after each stem, or a prefix would have to be followed by a space:
    # "спеціалізова" sits inside "спеціалізовані" and the next character is a letter.
    return r"\w*\W+".join(re.escape(w[: max(4, len(w) - 2)]) for w in words)


def diverging_directions(text: str, fields: Sequence[Mapping], lang: str) -> list[str]:
    """Fields the prose moves the opposite way from the table, in order of appearance.

    Blunt in the way :func:`horizon.interpret.unsupported_numbers` is blunt, and for the
    same reason: it reports candidates for a person to look at rather than deciding
    anything. What it catches is the failure that check cannot see — a sentence with no
    wrong number in it saying a field rose when it fell. Both models used for these
    documents did exactly that, on the two largest movements in the table printed
    immediately above the paragraph.

    A direction word governs the nearest field name within :data:`DIRECTION_REACH`
    characters after it, and a negated one governs nothing: "did not grow" is a claim
    this check is not equipped to read, so it declines to.
    """
    low = text.lower()
    negated = re.compile(r"\b(?:не|not)\W+\w*$")
    flagged: list[str] = []
    for field in fields:
        if abs(field.get("delta") or 0.0) < DIRECTION_FLOOR:
            continue
        pattern = _stem_pattern(field["name"])
        if not pattern:
            continue
        for hit in re.finditer(pattern, low):
            window = low[max(0, hit.start() - DIRECTION_REACH) : hit.start()]
            claims = [
                (window.rfind(stem), stem in RISE[lang])
                for stem in RISE[lang] + FALL[lang]
                if window.rfind(stem) >= 0
                and not negated.search(window[: window.rfind(stem)])
            ]
            if not claims:
                continue
            _at, claims_rise = max(claims)
            if claims_rise != (field["delta"] > 0):
                flagged.append(field["name"])
            break
    return flagged


def load_readings(paths: Sequence[Path]) -> list[dict]:
    """The reading files, in the order given, newest-written last."""
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def _reading_block(
    entry: Mapping,
    model: str,
    lab: Mapping[str, str],
    allowed: set,
    *,
    attribute: bool,
    fields: Sequence[Mapping] = (),
    lang: str = "uk",
) -> list[str]:
    """One model's paragraphs, with any unsupported figure named under them.

    ``attribute`` names the model above the paragraph. It is there to tell several models
    apart and not to record provenance — which model wrote a document belongs in the
    technical report describing the run — so a single reading is printed unattributed.

    The figures are re-checked here rather than trusted from the stored run, so a later
    improvement to the check reaches readings already generated.
    """
    from horizon.interpret import clean_reading, unsupported_numbers

    text = clean_reading(entry["text"])
    out = ([f"**{lab['reading']} — {model}**", ""] if attribute else []) + [text, ""]
    if entry.get("truncated"):
        out += [f"> {lab['truncated']}", ""]
    flagged = unsupported_numbers(text, allowed)
    if flagged:
        out += [f"> {lab['unsupported']}: {', '.join(flagged)}", ""]
    wrong_way = diverging_directions(text, fields, lang)
    if wrong_way:
        out += [f"> {lab['diverging']}: {', '.join(wrong_way)}", ""]
    return out


def _readings_for(
    readings: Sequence[Mapping],
    bucket: str,
    key: str | None,
    lab,
    allowed,
    fields: Sequence[Mapping] = (),
    lang: str = "uk",
) -> list[str]:
    """Every model's reading of one unit, or nothing when none of them wrote it."""
    attribute = len(readings) > 1
    out: list[str] = []
    for r in readings:
        entry = r.get(bucket) if key is None else (r.get(bucket) or {}).get(key)
        if entry:
            out += _reading_block(
                entry,
                r.get("label") or r.get("model_id", "?"),
                lab,
                allowed,
                attribute=attribute,
                fields=fields,
                lang=lang,
            )
    return out


def render_commentary(
    a: Side, b: Side, present, measured, data, readings: Sequence[Mapping], lang: str
) -> str:
    """The readings alone, in the order the comparison raises the questions."""
    from horizon.interpret import facts_numbers

    lab = HEAD[lang]
    allowed = facts_numbers(facts_dict(a, b, present, measured, data))
    out = [
        f"# {lab['title']}",
        "",
        lab["lede"],
        "",
        lab["thin"],
        "",
        lab["checked"],
        "",
    ]
    out += [f"## {lab['h_comparison']}", ""]
    out += _readings_for(
        readings,
        "comparison",
        None,
        lab,
        allowed,
        data["fields_a"] + data["fields_b"],
        lang,
    )
    out += [f"## {lab['h_taxonomies']}", ""]
    for side, key in ((a, "a"), (b, "b")):
        blocks = _readings_for(
            readings, "taxonomies", key, lab, allowed, data[f"fields_{key}"], lang
        )
        if blocks:
            out += [f"### {side.taxonomy[lang]}", ""] + blocks
    out += [f"## {lab['h_matches']}", ""]
    out += _readings_for(readings, "matches", None, lab, allowed, (), lang)
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- rendering


#: Figure stem -> the key its caption is looked up under, in the order they are shown.
#: Numbered by position, so a figure that could not be drawn does not leave a gap in the
#: numbering of the ones that could.
FIGURE_CAPTIONS = (
    ("f1-volume-index", "f1"),
    ("f2-share-change", "f2"),
    ("f3-matched-topics", "f3"),
)

#: The word that introduces a caption, per language — as in ``figures.CAPTION``.
CAPTION = {"en": "Figure", "uk": "Діаграма"}


def _cell(text: str) -> str:
    return text.replace("|", r"\|")


def _forms(raw: str) -> tuple[str, str, str]:
    """``"патент|патенти|патентів"`` as the three forms a Ukrainian numeral chooses between.

    One form given means one form used — which is right for English and wrong for
    Ukrainian, so the flag's help asks for three and this is the fallback, not the plan.
    The last form is also the bare genitive plural, which is what a phrase with no numeral
    in front of it wants.

    The three are the **nominative** forms, so every Ukrainian sentence that puts a count
    in front of one is written to a position where nominative (or, for these inanimate
    nouns, the accusative that matches it) is what the grammar asks for. Carrying the full
    case paradigm instead would be a morphology table in a reporting script.
    """
    parts = tuple(x.strip() for x in raw.split("|"))
    return parts if len(parts) == 3 else (parts[0], parts[0], parts[0])


def _unit(forms: tuple[str, str, str], n: float) -> str:
    """The form the numeral ``n`` governs: 1, 2–4, or 5 and up — teens taking the last."""
    n = abs(int(n))
    if n % 100 // 10 == 1:
        return forms[2]
    last = n % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


def _num(n: float) -> str:
    """A thousands-separated count. A thin space, so the number is not read as two."""
    return f"{round(n):,}".replace(",", "\u2009")


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def render(
    a: Side,
    b: Side,
    present: list[int],
    measured: list[int],
    data: dict,
    lang: str,
    folder: str,
    figures_drawn: list[str],
    *,
    top: int,
    readings: Sequence[Mapping] = (),
) -> str:
    """The whole comparison, in one language.

    With ``readings`` this is the annotated document rather than the report: the same
    renderer, so the annotated version cannot drift from the report it annotates. The
    single-corpus merge re-parses the rendered markdown to fold its readings in, which
    works and has to be kept in step by hand; here the readings are simply another thing
    the renderer emits.
    """
    lab = LABELS[lang]
    head = HEAD[lang]
    allowed: set = set()
    if readings:
        from horizon.interpret import facts_numbers

        allowed = facts_numbers(facts_dict(a, b, present, measured, data))
    na, nb = a.name[lang], b.name[lang]
    ua, ub = a.unit[lang][2], b.unit[lang][2]  # the bare, no-numeral form
    out: list[str] = []
    txt = FIGURE_TEXT[lang]
    fig: dict[str, str] = {}
    for number, (name, key) in enumerate(
        [c for c in FIGURE_CAPTIONS if c[0] in figures_drawn], start=1
    ):
        fig[name] = f"![{CAPTION[lang]} {number}. {txt[key]}]({folder}/{name}.png)"

    title = {
        "en": f"{na} and {nb} over their common years, {measured[0]}–{measured[-1]}",
        "uk": f"{na} і {nb} на спільних роках, {measured[0]}–{measured[-1]}",
    }[lang]
    if readings:
        title = f"{title} — {head['annotated_title'].lower()}"
    out += [f"# {title}", ""]
    out += [
        f"**{lab['window']}:** {present[0]}–{present[-1]} ({lab['present_in_both']})  ",
        f"**{lab['measured_over']}:** {measured[0]}–{measured[-1]}  ",
        "",
    ]
    n_a = sum(a.total(y) for y in measured)
    n_b = sum(b.total(y) for y in measured)
    out += [
        {
            "en": (
                f"> Every share below is a share of its own corpus. Over the measured "
                f"window {na} holds {_num(n_a)} {_unit(a.unit[lang], n_a)} and "
                f"{nb} holds {_num(n_b)} {_unit(b.unit[lang], n_b)}; those two "
                "counts are not a finding about either landscape, only about the two "
                "exports, and no figure in this document divides one by the other."
            ),
            "uk": (
                f"> Кожна частка нижче — це частка власного корпусу. На виміряному "
                f"вікні {na} містить {_num(n_a)} {_unit(a.unit[lang], n_a)}, а "
                f"{nb} — {_num(n_b)} {_unit(b.unit[lang], n_b)}; ці два числа не "
                "є висновком про жоден із ландшафтів, лише про два експорти, і жодна "
                "цифра в цьому документі не ділить одне на одне."
            ),
        }[lang],
        "",
    ]
    if readings:
        out += [head["annotated_lede"], "", head["thin"], "", head["checked"], ""]

    # 1 -------------------------------------------------------------------- the window
    # A coverage column exists only for a corpus that estimates its own completeness, and
    # it sits immediately right of that corpus's count — with a sentence saying so when
    # only one of the two has one, because an unattached percentage column would read as
    # belonging to either.
    cov_a = data["volume_a"]
    cov_b = {r["year"]: r for r in data["volume_b"]}
    heads, align = [lab["year"]], ["---"]
    for side, name in ((a, na), (b, nb)):
        heads.append(_cell(name))
        align.append("--:")
        if side.coverage:
            heads.append(lab["coverage"])
            align.append("--:")
    heads.append(lab["status"])
    align.append("---")
    note = [lab["l1_note"]]
    if bool(a.coverage) != bool(b.coverage):
        owner = na if a.coverage else nb
        note.append(
            {
                "en": f"Only {owner} estimates its own completeness, so the coverage column belongs to it.",
                "uk": f"Лише {owner} оцінює власну повноту, тож стовпець повноти належить йому.",
            }[lang]
        )
    out += [f"## {lab['l1']}", "", " ".join(note), ""]
    out += ["| " + " | ".join(heads) + " |", "|" + "|".join(align) + "|"]
    for row in cov_a:
        other = cov_b[row["year"]]
        cells = [str(row["year"])]
        for side, r in ((a, row), (b, other)):
            cells.append(str(r["n"]))
            if side.coverage:
                cells.append(f"{r['coverage']:.1%}" if r["coverage"] else "—")
        cells.append(lab["measured"] if row["measured"] else lab["excluded"])
        out += ["| " + " | ".join(cells) + " |"]
    out += [""]

    # 2 -------------------------------------------------------------------- volume
    out += [f"## {lab['l2']}", "", lab["l2_note"], ""]
    out += [
        f"| {lab['year']} | {_cell(na)} | {lab['index']} | {lab['corrected']} "
        f"| {_cell(nb)} | {lab['index']} |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    base_a = next(r for r in cov_a if r["measured"])["corrected"]
    for row in cov_a:
        other = cov_b[row["year"]]
        corr = (
            f"{100 * row['corrected'] / base_a:.0f}"
            if row["corrected"] and base_a
            else "—"
        )
        out += [
            f"| {row['year']} | {row['n']} | {row['index']:.0f} | {corr} "
            f"| {other['n']} | {other['index']:.0f} |"
        ]
    out += [""]
    h = data["headline"]
    verbs = {"en": ("rises", "falls"), "uk": ("зростає", "спадає")}[lang]
    va, vb = verbs[h["a"] < 0], verbs[h["b"] < 0]
    out += [
        {
            "en": (
                f"Across the {len(measured)} measured cohorts {na} {va} "
                f"{abs(h['a']):.0f}% — {abs(h['a_corrected']):.0f}% once each cohort is "
                f"divided by the share of its eventual {ua} the register estimates it "
                f"already holds — while {nb} {vb} {abs(h['b']):.0f}%. The two series part "
                f"in {h['crossed']} and do "
                "not meet again inside the window. That divergence is the one finding "
                "here that needs no bridge between the taxonomies, and it is also the "
                "one most exposed to scope: neither export states what it selects on."
            ),
            "uk": (
                f"За {len(measured)} виміряних когорт {na} {va} на {abs(h['a']):.0f}% "
                f"— на {abs(h['a_corrected']):.0f}%, якщо поділити кожну когорту на "
                f"оцінену реєстром частку її остаточних {ua}, які вже наявні, — тоді як "
                f"{nb} {vb} на {abs(h['b']):.0f}%. Два ряди розходяться у "
                f"{h['crossed']} році і "
                "більше не сходяться всередині вікна. Це розходження — єдиний висновок "
                "тут, якому не потрібен місток між таксономіями, і водночас найвразливіший "
                "до охоплення: жоден з експортів не повідомляє, за чим він відбирає."
            ),
        }[lang],
        "",
    ]
    for name in ("f1-volume-index",):
        if name in fig:
            out += [fig[name], ""]
    out += _readings_for(
        readings,
        "comparison",
        None,
        head,
        allowed,
        data["fields_a"] + data["fields_b"],
        lang,
    )

    # 3 -------------------------------------------------------------------- taxonomies
    out += [f"## {lab['l3']}", "", lab["l3_note"], ""]
    for side, key, rows in ((a, "a", data["fields_a"]), (b, "b", data["fields_b"])):
        out += [f"### {side.taxonomy[lang]}", ""]
        out += [
            f"| {lab['field']} | {lab['share_first']} | {lab['share_last']} "
            f"| {lab['change']} | {lab['per_year']} | {lab['in_window']} |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        # Both ends of the ranking, and an elision where the middle was dropped.
        top_rows, bottom_rows = rows[:top], rows[max(top, len(rows) - top) :]
        shown = (
            top_rows
            + ([None] if len(rows) > len(top_rows) + len(bottom_rows) else [])
            + bottom_rows
        )
        for r in shown:
            if r is None:
                out += ["| … | | | | | |"]
                continue
            out += [
                f"| {_cell(r['name'])} | {_pct(r['first'])} | {_pct(r['last'])} "
                f"| {_pp(r['delta'])} | {r['slope'] * 100:+.2f} | {r['n']} |"
            ]
        out += [""]
        out += _readings_for(readings, "taxonomies", key, head, allowed, rows, lang)
    for name in ("f2-share-change",):
        if name in fig:
            out += [fig[name], ""]

    out += [f"### {lab['l3_shape']}", ""]
    sa, sb = data["shape_a"], data["shape_b"]
    out += [
        f"| {lab['metric']} | {_cell(na)} | {_cell(nb)} |",
        "|---|--:|--:|",
        f"| {lab['fields_n']} | {sa['fields']} | {sb['fields']} |",
        f"| {lab['multi']} | {sa['multi']:.2f} | {sb['multi']:.2f} |",
        f"| {lab['top5']} | {sa['top5']:.1%} | {sb['top5']:.1%} |",
        f"| {lab['effective']} | {sa['effective']:.1f} | {sb['effective']:.1f} |",
        f"| {lab['pairs_n']} | {sa['pairs']} | {sb['pairs']} |",
        "",
    ]

    out += [f"### {lab['l3_pairs']}", "", lab["l3_pairs_note"], ""]
    for side, rows in ((a, data["pairs_a"]), (b, data["pairs_b"])):
        out += [f"**{side.name[lang]}**", ""]
        out += [
            f"| {lab['pair']} | {lab['share_first']} | {lab['share_last']} "
            f"| {lab['per_year']} | {lab['in_window']} |",
            "|---|--:|--:|--:|--:|",
        ]
        for r in rows:
            out += [
                f"| {_cell(r['name'])} | {_pct(r['first'])} | {_pct(r['last'])} "
                f"| {r['slope'] * 100:+.2f} | {r['n']} |"
            ]
        out += [""]

    # 4 -------------------------------------------------------------------- matches
    agr = data["agreement"]
    out += [f"## {lab['l4']}", "", lab["l4_note"], ""]
    out += [
        {
            "en": (
                f"{agr['all_n']} labels occur in both, covering "
                f"{data['match_a_n']} {_unit(a.unit[lang], data['match_a_n'])} and "
                f"{data['match_b_n']} {_unit(b.unit[lang], data['match_b_n'])} of the measured "
                f"window. Of those, {agr['n']} are carried by at least {MIN_MATCH_SIZE} "
                f"records on both sides, which is the set the direction test below is run "
                f"on: {agr['same']} of {agr['n']} move the same way, and the correlation "
                f"between the two share slopes is r = {agr['r']:+.2f}. Across all "
                f"{agr['all_n']}, {agr['all_same']} agree. Both figures sit where chance "
                "would put them, and that is the result of this section."
            ),
            "uk": (
                f"{agr['all_n']} назв трапляються в обох; на них припадає "
                f"{data['match_a_n']} {_unit(a.unit[lang], data['match_a_n'])} і "
                f"{data['match_b_n']} {_unit(b.unit[lang], data['match_b_n'])} "
                f"виміряного вікна. "
                f"З них {agr['n']} тримаються щонайменше на {MIN_MATCH_SIZE} записах з "
                f"кожного боку — саме на цій множині виконано перевірку напрямку нижче: "
                f"{agr['same']} з {agr['n']} рухаються в один бік, а кореляція між "
                f"нахилами часток становить r = {agr['r']:+.2f}. На всіх {agr['all_n']} "
                f"збігаються {agr['all_same']}. Обидві цифри лежать там, де їх поставив "
                "би випадок, і це і є результат цього розділу."
            ),
        }[lang],
        "",
    ]
    for name in ("f3-matched-topics",):
        if name in fig:
            out += [fig[name], ""]
    out += _readings_for(readings, "matches", None, head, allowed, (), lang)

    def topic_table(rows: list[dict]) -> list[str]:
        lines = [
            f"| {lab['topic']} | {_cell(na)} | {lab['share_slope']} | {_cell(nb)} "
            f"| {lab['share_slope']} | {lab['keywords']} |",
            "|---|--:|--:|--:|--:|---|",
        ]
        for r in rows:
            words = ", ".join(r["a_keywords"][:3])
            lines.append(
                f"| {_cell(r['label'])} | {r['a_n']} | {r['a_slope']:+.1f} "
                f"| {r['b_n']} | {r['b_slope']:+.1f} | {_cell(words)} |"
            )
        return lines + [""]

    big = agr["rows"]
    out += [f"### {lab['rising']}", ""]
    out += topic_table(
        sorted(
            [r for r in big if r["a_slope"] > 0 and r["b_slope"] > 0],
            key=lambda r: -r["a_slope"],
        )[:top]
    )
    out += [f"### {lab['falling']}", ""]
    out += topic_table(
        sorted(
            [r for r in big if r["a_slope"] < 0 and r["b_slope"] < 0],
            key=lambda r: r["a_slope"],
        )[:top]
    )
    out += [f"### {lab['diverging']}", ""]
    out += topic_table(
        sorted(
            [r for r in big if (r["a_slope"] > 0) != (r["b_slope"] > 0)],
            key=lambda r: -abs(r["a_slope"] - r["b_slope"]),
        )[:top]
    )
    out += [f"### {lab['largest']}", ""]
    out += [
        {
            "en": (
                f"Ranked by the {na} side alone, and with no floor on the other — which "
                "is why several rows carry a handful of records opposite a hundred. Those "
                "are the matches to distrust, and they are printed rather than filtered "
                "so that the join's failure rate is visible in the same table as its "
                "successes."
            ),
            "uk": (
                f"Упорядковано лише за боком «{na}», без порога з іншого боку — тому "
                "кілька рядків несуть жменю записів проти сотні. Саме таким зіставленням "
                "не варто довіряти, і їх наведено, а не відфільтровано, щоб частота "
                "невдач цього зчеплення була видна в тій самій таблиці, що й його успіхи."
            ),
        }[lang],
        "",
    ]
    out += topic_table(sorted(data["matches"], key=lambda r: -r["a_n"])[:top])

    # 5 -------------------------------------------------------------------- crosswalk
    out += [f"## {lab['l5']}", "", lab["l5_note"], ""]
    out += [
        f"| {lab['link']} | {lab['labels_n']} | {lab['weight']} |",
        "|---|--:|--:|",
    ]
    for r in data["crosswalk"][:top]:
        out += [
            f"| {_cell(r['a'])} ↔ {_cell(r['b'])} | {r['labels']} | {r['weight']} |"
        ]
    out += [""]

    # 6 -------------------------------------------------------------------- limits
    out += [f"## {lab['l6']}", "", *data["limits"][lang], ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- figures


#: Titles and axis labels. Beside the code that draws rather than in :data:`LABELS`,
#: for the reason ``figures.TEXT`` gives: those are table headers, a figure wants sentences.
FIGURE_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "f1": "Volume, indexed to the first measured cohort",
        "f1y": "index (first measured year = 100)",
        "f1corr": "corrected for coverage",
        "f2": "Change in share across the measured window",
        "f2x": "percentage points",
        "f3": "Matched topics: how each side's share moved",
        "f3ax": "share slope, per 10 000 of the corpus per year",
        "f3key": "topic (records: register / corpus)",
        "unmeasured": "present, not measured",
    },
    "uk": {
        "f1": "Обсяг, індексований до першої виміряної когорти",
        "f1y": "індекс (перший виміряний рік = 100)",
        "f1corr": "з поправкою на повноту",
        "f2": "Зміна частки на виміряному вікні",
        "f2x": "відсоткові пункти",
        "f3": "Зіставлені теми: як рухалася частка з кожного боку",
        "f3ax": "нахил частки, на 10 000 корпусу за рік",
        "f3key": "тема (записів: реєстр / корпус)",
        "unmeasured": "присутній, не виміряно",
    },
}


def draw(
    a: Side, b: Side, present, measured, data, out_dir: Path, lang: str
) -> list[str]:
    """Three figures: the two volumes, the two taxonomies, and the matched topics."""
    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from horizon.figures import (
        DOWN,
        INK,
        MUTED,
        SERIES,
        UP,
        _integer_years,
        _short,
        _style,
    )

    _style(plt)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = FIGURE_TEXT[lang]
    drawn = []

    # f1 — the two volumes, indexed.
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for side, rows, colour in (
        (a, data["volume_a"], SERIES[0]),
        (b, data["volume_b"], SERIES[3]),
    ):
        xs = [r["year"] for r in rows]
        ys = [r["index"] for r in rows]
        ax.plot(xs, ys, color=colour, lw=2.2, label=_short(side.name[lang], 28))
        corrected = [r for r in rows if r["corrected"]]
        if len(corrected) == len(rows):
            base = next(r for r in rows if r["measured"])["corrected"]
            ax.plot(
                [r["year"] for r in corrected],
                [100 * r["corrected"] / base for r in corrected],
                color=colour,
                lw=1.2,
                ls=(0, (4, 2)),
                label=f"{_short(side.name[lang], 28)} — {txt['f1corr']}",
            )
    if measured[-1] < present[-1]:
        # The band opens between the last measured cohort and the first excluded one, so
        # a reader does not have to decide which side of the line 2020 is on.
        ax.axvspan(measured[-1] + 0.5, present[-1] + 0.2, color="#EEF1F4", zorder=0)
        ax.annotate(
            txt["unmeasured"],
            xy=((measured[-1] + 0.5 + present[-1]) / 2, ax.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=7,
            color=MUTED,
        )
    ax.axhline(100, color="#C9CFD6", lw=0.8, zorder=0)
    ax.set_title(txt["f1"])
    ax.set_ylabel(txt["f1y"])
    ax.grid(axis="y")
    _integer_years(ax)
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(out_dir / "f1-volume-index.png")
    plt.close(fig)
    drawn.append("f1-volume-index")

    # f2 — the two taxonomies' share change, one panel each.
    span = max(
        abs(r["delta"]) for rows in (data["fields_a"], data["fields_b"]) for r in rows
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.8), gridspec_kw={"wspace": 0.62})
    for ax, side, rows in (
        (axes[0], a, data["fields_a"]),
        (axes[1], b, data["fields_b"]),
    ):
        rows = sorted(rows, key=lambda r: r["delta"])
        ys = list(range(len(rows)))
        ax.barh(
            ys,
            [r["delta"] * 100 for r in rows],
            color=[UP if r["delta"] >= 0 else DOWN for r in rows],
            height=0.7,
        )
        ax.set_yticks(ys)
        ax.set_yticklabels([_short(r["name"], 26) for r in rows], fontsize=6)
        ax.axvline(0, color=INK, lw=0.8)
        # One scale across both panels: two auto-scaled axes would draw a 4.8-point move
        # and a 7.6-point move the same length.
        ax.set_xlim(-span * 105, span * 105)
        ax.set_title(_short(side.name[lang], 40))
        ax.set_xlabel(txt["f2x"])
        ax.grid(axis="x")
    fig.suptitle(txt["f2"], fontsize=9.5, fontweight="semibold", color=INK)
    fig.savefig(out_dir / "f2-share-change.png")
    plt.close(fig)
    drawn.append("f2-share-change")

    # f3 — the matched topics, one point each, named in a key beside the plot.
    #
    # Labelled in place first, and that was the wrong answer: sixteen topics in a cloud
    # this tight need either leader lines long enough to cross the cloud or a font too
    # small to read, and the version with both was neither. A key costs the reader one
    # saccade and costs the picture nothing.
    #
    # Area still carries the two sides' combined size, so the key says which topic and
    # the marker says how much of it there is.
    big = data["agreement"]["rows"]
    if big:
        from matplotlib.lines import Line2D

        ranked = sorted(big, key=lambda r: -(r["a_n"] + r["b_n"]))
        colours = [plt.get_cmap("tab20")(i % 20) for i in range(len(ranked))]
        # Sized for the page rather than for the file. The rendered width is fixed at
        # the 17 cm text block, so every extra inch here shrinks the type that arrives:
        # at 11 inches wide a 9-point key prints at about 5.5 points.
        # Sized for the page rather than for the file: the rendered width is fixed at
        # the 17 cm text block, so every extra inch here shrinks the type that arrives.
        #
        # The key's column is reserved inside the figure rather than left to hang off the
        # axes, because ``savefig.bbox="tight"`` grows the canvas to take in anything
        # outside it — with the legend simply anchored past the right spine, figsize
        # stopped deciding the width at all and a 10-point key printed at 6.3.
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        fig.subplots_adjust(left=0.095, right=0.545, top=0.90, bottom=0.135)
        ax.axhline(0, color="#C9CFD6", lw=0.8)
        ax.axvline(0, color="#C9CFD6", lw=0.8)
        for row, colour in zip(ranked, colours):
            ax.scatter(
                row["a_slope"],
                row["b_slope"],
                s=max(45, min(560, 3 * (row["a_n"] + row["b_n"]))),
                color=colour,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        ax.margins(0.12)
        ax.set_title(txt["f3"], fontsize=13)
        ax.set_xlabel(f"{a.name[lang]} — {txt['f3ax']}", fontsize=10)
        ax.set_ylabel(f"{b.name[lang]} — {txt['f3ax']}", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True)
        # Proxy handles: a scatter's own handle inherits the marker area, so the key
        # would show the same size differences as the plot and read as a scale.
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=colour,
                    markersize=7,
                    markeredgecolor="white",
                    label=f"{row['label']} · {row['a_n']}/{row['b_n']}",
                )
                for row, colour in zip(ranked, colours)
            ],
            title=txt["f3key"],
            title_fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            labelspacing=0.62,
            borderpad=0.0,
        )
        fig.savefig(out_dir / "f3-matched-topics.png")
        plt.close(fig)
        drawn.append("f3-matched-topics")
    return drawn


# ------------------------------------------------------------------------------- main


def limits(a: Side, b: Side, present, measured) -> dict[str, list[str]]:
    """The comparisons this document refuses, each with the number that refuses it."""
    ua, ub = a.unit["en"][2], b.unit["en"][2]
    dropped = [y for y in present if y not in measured]
    en = [
        (
            f"- **The cohorts {dropped[0]}–{dropped[-1]} are in both corpora and in no "
            f"measurement here.** They are present, so the overlap of the two calendars runs "
            f"to {present[-1]}; one of the two reports them as still filling up, so every "
            f"derived column stops at {measured[-1]}. The counts are printed in §1 and §2 "
            "and used nowhere else."
            if dropped
            else "- **The presence window and the measured window coincide.**"
        ),
        "- **The two taxonomies are never mapped onto one another.** §5 shows which "
        "fields and which classification blocks the matched topics happen to sit in, "
        "which is a consequence of the matches rather than a crosswalk applied to them. "
        "It is also visibly noisy: a block is the classification the export carries, and "
        "materials topics sit inside blocks whose names are nothing to do with materials.",
        "- **A shared label is not a shared subject.** It says the same model described "
        "two clusters alike from summaries it wrote itself. The keyword columns in §4 "
        "are there so a reader can reject a match; several deserve rejecting.",
        f"- **Topic counts are not compared.** The number of topics a corpus yields is "
        f"set by the clustering parameters of the run that produced it, and the two runs "
        f"did not share them — so {len(a.topics)} labels against {len(b.topics)} is a "
        "fact about two commands, not about two landscapes.",
        "- **Citation-derived indicators are left out entirely.** Coverage differs by "
        "more than an order of magnitude between the two corpora, and an indicator "
        "computed on 2% of one register against 77% of the other would compare the "
        "indexes rather than the work.",
        "- **The record-side lenses are left out for the same kind of reason.** Both "
        "corpora carry a venue-dispersion and a new-entrant table, but each holds only "
        "the topics its own lens ranked highest, and the two runs kept different numbers "
        "of them. Two differently-sized top slices are not two samples of the same thing.",
        f"- **The time axes are not the same clock.** A cohort of {ua} is dated by one "
        f"event and a cohort of {ub} by another, so records the two corpora file under "
        "the same year did not happen at the same time.",
    ]
    uk = [
        (
            f"- **Когорти {dropped[0]}–{dropped[-1]} є в обох корпусах і в жодному "
            f"вимірюванні тут.** Вони присутні, тож перетин двох календарів сягає "
            f"{present[-1]}; один із двох звітує, що вони ще наповнюються, тож кожен "
            f"похідний стовпець зупиняється на {measured[-1]}. Кількості наведено в §1 і §2 "
            "і більше ніде не використано."
            if dropped
            else "- **Вікно присутності й виміряне вікно збігаються.**"
        ),
        "- **Дві таксономії ніде не відображаються одна на одну.** §5 показує, у які "
        "поля і які класифікаційні блоки випадають зіставлені теми, — це наслідок "
        "зіставлень, а не відповідність, застосована до них. Він і помітно шумний: блок "
        "— це класифікація, яку несе експорт, і теми про матеріали сидять у блоках, чиї "
        "назви до матеріалів не мають стосунку.",
        "- **Спільна назва — не спільний предмет.** Вона каже, що одна модель описала "
        "два кластери однаково за резюме, які написала сама. Стовпці ключових слів у §4 "
        "існують, щоб читач міг відхилити зіставлення; кілька з них варто відхилити.",
        f"- **Кількості тем не порівнюються.** Скільки тем дає корпус, визначають "
        f"параметри кластеризації того запуску, що його породив, а два запуски їх не "
        f"поділяли — тож {len(a.topics)} назв проти {len(b.topics)} є фактом про дві "
        "команди, а не про два ландшафти.",
        "- **Показники на основі цитувань повністю вилучено.** Покриття різниться між "
        "двома корпусами більш ніж на порядок, і показник, обчислений на 2% одного "
        "реєстру проти 77% іншого, порівнював би індекси, а не роботу.",
        "- **Лінзи з боку запису вилучено з подібної причини.** Обидва корпуси несуть "
        "таблицю розпорошення майданчиків і таблицю нових учасників, але кожна містить "
        "лише теми, які найвище проранжувала власна лінза, і два запуски зберегли різну "
        "їх кількість. Два зрізи різного розміру — це не дві вибірки того самого.",
        "- **Осі часу — не той самий годинник.** Одна когорта — це коли запис "
        "опубліковано, інша — коли заявлено винахід; запис і патент, датовані тим самим "
        "роком, сталися не одночасно.",
    ]
    return {"en": en, "uk": uk}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("facts_a", type=Path)
    ap.add_argument("facts_b", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--date", required=True)
    for which in ("a", "b"):
        ap.add_argument(f"--name-{which}", required=True)
        ap.add_argument(f"--name-{which}-uk", required=True)
        ap.add_argument(
            f"--unit-{which}",
            required=True,
            help="the word for one record, e.g. 'publications'",
        )
        ap.add_argument(
            f"--unit-{which}-uk",
            required=True,
            help=(
                "the same word in the three forms a Ukrainian numeral governs, "
                "pipe-separated: 'публікація|публікації|публікацій'"
            ),
        )
        ap.add_argument(f"--taxonomy-{which}", required=True)
        ap.add_argument(f"--taxonomy-{which}-uk", required=True)
        ap.add_argument(f"--coverage-{which}", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-labels", type=int, default=2)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--reading",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="a reading file from scripts/compare_interpret.py; repeat for each model. "
        "Given any, the commentary and the annotated comparison are written too.",
    )
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args(argv)

    a = load_side(args.facts_a, args, "a")
    b = load_side(args.facts_b, args, "b")
    present, measured = windows(a, b)
    if len(measured) < 2:
        print(
            "error: the two corpora share fewer than two measurable cohorts.",
            file=sys.stderr,
        )
        return 1
    print(
        f"present in both: {present[0]}–{present[-1]}; measured over: {measured[0]}–{measured[-1]}"
    )

    rows = matches(a, b, measured)
    data: dict[str, Any] = {
        "volume_a": volume(a, present, measured),
        "volume_b": volume(b, present, measured),
        "fields_a": fields(a, measured),
        "fields_b": fields(b, measured),
        "pairs_a": pairs(a, measured, args.top),
        "pairs_b": pairs(b, measured, args.top),
        "matches": rows,
        "match_a_n": sum(r["a_n"] for r in rows),
        "match_b_n": sum(r["b_n"] for r in rows),
        "agreement": agreement(rows, MIN_MATCH_SIZE),
        "crosswalk": crosswalk(rows, args.min_labels),
    }
    data["headline"] = headline(data["volume_a"], data["volume_b"], measured)
    data["shape_a"] = shape(a, data["fields_a"], measured)
    data["shape_b"] = shape(b, data["fields_b"], measured)
    data["limits"] = limits(a, b, present, measured)

    stem = f"{args.slug}-{args.date}"
    args.out.mkdir(parents=True, exist_ok=True)
    facts_path = args.out / f"{stem}-facts.json"
    facts_path.write_text(
        json.dumps(
            facts_dict(a, b, present, measured, data), ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    print(f"  wrote {facts_path}")
    for lang, suffix in (("en", ""), ("uk", "-ua")):
        folder = f"figures/{stem}{suffix}"
        drawn = (
            []
            if args.no_figures
            else draw(a, b, present, measured, data, args.out / folder, lang)
        )
        md = args.out / f"{stem}{suffix}.md"
        md.write_text(
            render(a, b, present, measured, data, lang, folder, drawn, top=args.top)
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {md} ({len(drawn)} figures)")

    companions = [f"{stem}-ua.md"]
    if args.reading:
        readings = load_readings(args.reading)
        lang = "uk"
        folder = f"figures/{stem}-ua"
        drawn = (
            []
            if args.no_figures
            else sorted(q.stem for q in (args.out / folder).glob("*.png"))
        )
        for name, text in (
            (
                f"{stem}-interpretation-ua.md",
                render_commentary(a, b, present, measured, data, readings, lang),
            ),
            (
                f"{stem}-annotated-ua.md",
                render(
                    a,
                    b,
                    present,
                    measured,
                    data,
                    lang,
                    folder,
                    drawn,
                    top=args.top,
                    readings=readings,
                )
                + "\n",
            ),
        ):
            (args.out / name).write_text(text, encoding="utf-8")
            print(f"  wrote {args.out / name}")
            companions.append(name)

    if args.render:
        from horizon.render import render as render_doc

        for name in companions:
            md = args.out / name
            for pdf, ext in ((True, "pdf"), (False, "docx")):
                print(render_doc(md, md.with_suffix(f".{ext}"), pdf=pdf, lang="uk"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
