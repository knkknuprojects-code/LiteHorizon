#!/usr/bin/env python3
"""One model's reading of a two-corpus comparison, written beside it.

``horizon interpret`` reads a corpus: its topics, its subject areas, and the corpus as a
whole. A comparison has none of those three. Its units are the ones §2 to §5 of the
comparison actually measure — the two volumes against each other, each taxonomy's own
movement over the shared window, and the topics whose label occurs in both — so the
prompts are written here rather than bent out of the single-corpus ones.

Everything else is deliberately the same, because it is the part that keeps a generated
reading honest:

* **Every numeral must exist in the facts.** The comparison writes its own
  ``-facts.json`` for exactly this, and :func:`horizon.interpret.unsupported_numbers`
  flags the rest.
* **A collapsed generation is retried, a truncated one is recorded.** Both are invisible
  to a reader of the finished document until it is too late.
* **One model per run, each writing its own file.** What a movement means is largely a
  model's own prior, so the merge shows them side by side rather than averaging them.
* **Local only.** The commentary layer never reaches an external service.

**There are no summaries in these prompts, and that is a limit rather than a choice.**
The single-corpus pass samples publication summaries per topic, which is what lets a
reading say what a topic studies rather than restating its counts. Here one of the two
analyses stores is the one this machine holds and the other is not — the larger corpus
was scanned on the cluster and only its report and facts came back. Giving one side
summaries and the other only keywords would bias every "are these the same thing?"
judgement toward the side with evidence, so both sides are described the same way: by
keywords, counts, share trajectories and the blocks they were clustered in. The readings
are thinner than a single-corpus commentary's for that reason, and the merged document
says so.

That also decides the units. The matched topics are read **as a set** rather than one by
one: §4 of the comparison measures the set — 129 labels, of which a handful are carried
by enough records on both sides to test — and its result is about the set, not about any
member. Sixteen separate hedged paragraphs would read as sixteen findings.

    python scripts/compare_interpret.py \\
        reports/<run>/<slug>-<date>-facts.json \\
        --out reports/<run> --slug <slug>-<date> --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from horizon import interpret  # noqa: E402

#: Generated length per reading, in tokens. Smaller than the single-corpus pass's 1024
#: because each of these answers a narrower question — two paragraphs about one pair of
#: series rather than about a whole subject area — and because this pass is expected to
#: run on a CPU, where every token is paid for in wall-clock.
READING_TOKENS = 768

#: The whole-comparison reading covers five sections and is given room accordingly.
COMPARISON_TOKENS = 1024

#: Matched topics listed in the set prompt. The ones the direction test runs on, which is
#: every one carried by enough records on both sides to have a direction worth reading.
MATCHES_LISTED = 20

COMPARISON_PROMPT = {
    "uk": (
        "Ви — наукометрист, який пише коментар до порівняння двох масивів за спільні "
        "роки {span}.\n\n"
        "A: {name_a} — {records_a} одиниць у вікні, таксономія «{tax_a}».\n"
        "B: {name_b} — {records_b} одиниць у вікні, таксономія «{tax_b}».\n\n"
        "Обсяг, індекс до {first_year} = 100:\n{volume}\n\n"
        "Найбільші зрушення часток у кожній таксономії (відсоткові пункти за вікно):\n"
        "{fields}\n\n"
        "Зчеплення через однакові назви тем: {matched} назв трапилися в обох; "
        "{testable} з них тримаються щонайменше на {floor} записах з кожного боку; "
        "з цих {testable} в один бік рухаються {same}, кореляція нахилів часток "
        "r = {r}.\n\n"
        "Напишіть два абзаци. Перший — що може означати розходження обсягів і чого воно "
        "не доводить: обидва масиви є експортами з власним охопленням, і жоден не "
        "повідомляє, за чим відбирає. Другий — як читати те, що на рівні тем збігу "
        "напрямків немає: це може означати і що ландшафти справді не пов’язані, і що "
        "зчеплення через назву заслабке, щоб щось виявити; скажіть, що саме розрізнило "
        "б ці дві відповіді.\n\n"
        "Правила: користуйтеся лише наведеними вище показниками — не вигадуйте "
        "кількостей, відсотків чи років. Пишіть прозою, а не переліком. Позначайте "
        "припущення як припущення. Без заголовків."
    ),
    "en": (
        "You are a scientometrician writing a commentary on a comparison of two bodies "
        "of records over the years {span} that both of them cover.\n\n"
        "A: {name_a} — {records_a} records in the window, taxonomy '{tax_a}'.\n"
        "B: {name_b} — {records_b} records in the window, taxonomy '{tax_b}'.\n\n"
        "Volume, indexed to {first_year} = 100:\n{volume}\n\n"
        "The largest share movements in each taxonomy (percentage points over the "
        "window):\n{fields}\n\n"
        "The join through identical topic labels: {matched} labels occur in both; "
        "{testable} of them are carried by at least {floor} records on each side; of "
        "those {testable}, {same} move the same way, and the correlation of the share "
        "slopes is r = {r}.\n\n"
        "Write two paragraphs. The first: what the divergence in volume might mean and "
        "what it does not establish — both bodies are exports with their own coverage, "
        "and neither states what it selects on. The second: how to read the absence of "
        "agreement at topic level, which could mean the landscapes are genuinely "
        "unrelated or that a join through labels is too weak to detect anything; say "
        "what would tell those two answers apart.\n\n"
        "Rules: use only the figures given above — do not invent counts, percentages or "
        "years. Write prose, not a list. Mark inference as inference. No headings."
    ),
}

TAXONOMY_PROMPT = {
    "uk": (
        "Ви — наукометрист, який пише короткий коментар до однієї таксономії за "
        "{span}.\n\n"
        "Масив: {name} — {records} одиниць у вікні.\n"
        "Таксономія: {tax} — {n_fields} полів, {multi} класифікацій на запис, "
        "{top5} припадає на п’ять найбільших полів.\n\n"
        "Частка масиву на початку і в кінці вікна, за полями:\n{fields}\n\n"
        "Пари полів, чия спільна частка зростала найшвидше:\n{pairs}\n\n"
        "Напишіть два абзаци. Перший — що саме зрушилося в цій таксономії за вікно. "
        "Другий — чого ця картина не доводить: це частки цього масиву, а не світові, і "
        "зростання частки сумісне зі скороченням кількості, якщо масив скорочується "
        "швидше.\n\n"
        "Правила: користуйтеся лише наведеними вище показниками — не вигадуйте "
        "кількостей, відсотків чи років. Пишіть прозою, а не переліком. Позначайте "
        "припущення як припущення. Без заголовків."
    ),
    "en": (
        "You are a scientometrician writing a short commentary on one taxonomy over "
        "{span}.\n\n"
        "Body: {name} — {records} records in the window.\n"
        "Taxonomy: {tax} — {n_fields} fields, {multi} classifications per record, "
        "{top5} of them in the five largest fields.\n\n"
        "Share of the body at the ends of the window, by field:\n{fields}\n\n"
        "The field pairs whose joint share rose fastest:\n{pairs}\n\n"
        "Write two paragraphs. The first: what moved inside this taxonomy over the "
        "window. The second: what the picture does not establish — these are shares of "
        "this body and not of the world, and a rising share is compatible with a falling "
        "count when the body shrinks faster.\n\n"
        "Rules: use only the figures given above — do not invent counts, percentages or "
        "years. Write prose, not a list. Mark inference as inference. No headings."
    ),
}

MATCHES_PROMPT = {
    "uk": (
        "Ви — наукометрист. Дві незалежні кластеризації — одна над масивом "
        "«{name_a}», друга над «{name_b}» — дали кластери з однаковими назвами. "
        "Модель, що давала назви, не знала, який масив читає, тож однакова назва — це "
        "єдине зчеплення між двома таксономіями, яке не довелося вигадувати.\n\n"
        "Усього таких назв {matched}. З них {testable} тримаються щонайменше на {floor} "
        "записах з кожного боку; саме вони наведені нижче. У цій підмножині в один бік "
        "рухаються {same} з {testable}, кореляція нахилів часток r = {r}.\n\n"
        "Назва — записи A, нахил частки A (на 10 000 за рік) — записи B, нахил B — "
        "ключові слова A — ключові слова B:\n{rows}\n\n"
        "Напишіть два абзаци. Перший — чи ці пари взагалі про одне й те саме: судіть за "
        "ключовими словами з обох боків, назвіть ті пари, які радше збіг назви, ніж "
        "спільний предмет. Другий — що означає відсутність спільного напрямку в цій "
        "підмножині: це однаково сумісно з тим, що ландшафти не пов’язані, і з тим, що "
        "зчеплення через назву заслабке; скажіть, які дані розрізнили б ці дві "
        "відповіді.\n\n"
        "Зауважте: резюме публікацій і патентів вам не надано — лише ключові слова, "
        "тож судження про предмет є судженням за ключовими словами. Не вигадуйте "
        "кількостей, відсотків чи років. Пишіть прозою, а не переліком. Позначайте "
        "припущення як припущення. Без заголовків."
    ),
    "en": (
        "You are a scientometrician. Two independent clusterings — one over '{name_a}', "
        "one over '{name_b}' — produced clusters carrying identical labels. The model "
        "that wrote the labels was not told which body it was reading, so an identical "
        "label is the one join between the two taxonomies that nobody had to invent.\n\n"
        "There are {matched} such labels. {testable} of them are carried by at least "
        "{floor} records on each side, and those are listed below. Within that subset "
        "{same} of {testable} move the same way, and the correlation of the share slopes "
        "is r = {r}.\n\n"
        "Label — A records, A share slope (per 10 000 per year) — B records, B slope — "
        "A keywords — B keywords:\n{rows}\n\n"
        "Write two paragraphs. The first: whether these pairs are about the same thing "
        "at all — judge from the keywords on each side, and name the pairs that look "
        "like a coincidence of naming rather than a shared subject. The second: what the "
        "absence of a shared direction in this subset means, which is equally compatible "
        "with the landscapes being unrelated and with the join being too weak to detect "
        "anything; say what data would tell those apart.\n\n"
        "Note: you have not been given summaries of the records, only keywords, so a "
        "judgement about subject is a judgement about keywords. Do not invent counts, "
        "percentages or years. Write prose, not a list. Mark inference as inference. No "
        "headings."
    ),
}


#: How an excluded cohort is marked inside a prompt. In the prompt's own language: a
#: Ukrainian note in an English prompt is a small thing that a model reads as noise, and
#: an unmarked excluded cohort is read as the end of the series.
UNMEASURED = {"uk": "(присутній, не виміряно)", "en": "(present, not measured)"}


def _volume_block(facts: Mapping, lang: str = "uk") -> str:
    rows = []
    for a, b in zip(facts["volume"]["a"], facts["volume"]["b"]):
        mark = "" if a["measured"] else f"  {UNMEASURED[lang]}"
        rows.append(f"- {a['year']}: A {a['index']:.0f}, B {b['index']:.0f}{mark}")
    return "\n".join(rows)


def _fields_block(facts: Mapping, side: str, limit: int) -> str:
    rows = facts["fields"][side]
    ends = rows[:limit] + rows[-limit:] if len(rows) > 2 * limit else rows
    return "\n".join(
        f"- {r['name']}: {r['first']:.1%} → {r['last']:.1%} ({r['delta'] * 100:+.1f})"
        for r in ends
    )


def _pairs_block(facts: Mapping, side: str, limit: int) -> str:
    return (
        "\n".join(
            f"- {r['name']}: {r['first']:.1%} → {r['last']:.1%}"
            for r in facts["pairs"][side][:limit]
        )
        or "—"
    )


def comparison_prompt(facts: Mapping, lang: str, floor: int) -> str:
    measured = facts["measured_years"]
    agr = facts["agreement"]
    return COMPARISON_PROMPT[lang].format(
        span=f"{measured[0]}–{measured[-1]}",
        first_year=measured[0],
        name_a=facts["a"]["name"][lang],
        name_b=facts["b"]["name"][lang],
        records_a=sum(facts["a"]["records"][str(y)] for y in measured),
        records_b=sum(facts["b"]["records"][str(y)] for y in measured),
        tax_a=facts["a"]["taxonomy"][lang],
        tax_b=facts["b"]["taxonomy"][lang],
        volume=_volume_block(facts, lang),
        fields="A:\n"
        + _fields_block(facts, "a", 5)
        + "\nB:\n"
        + _fields_block(facts, "b", 5),
        matched=agr["all_n"],
        testable=agr["n"],
        floor=floor,
        same=agr["same"],
        r=f"{agr['r']:+.2f}",
    )


def taxonomy_prompt(facts: Mapping, side: str, lang: str) -> str:
    measured = facts["measured_years"]
    return TAXONOMY_PROMPT[lang].format(
        span=f"{measured[0]}–{measured[-1]}",
        name=facts[side]["name"][lang],
        records=sum(facts[side]["records"][str(y)] for y in measured),
        tax=facts[side]["taxonomy"][lang],
        n_fields=facts["shape"][side]["fields"],
        multi=f"{facts['shape'][side]['multi']:.2f}",
        top5=f"{facts['shape'][side]['top5']:.1%}",
        fields=_fields_block(facts, side, 8),
        pairs=_pairs_block(facts, side, 6),
    )


def matches_prompt(facts: Mapping, lang: str, floor: int) -> str:
    """The matched topics as one list: what the set shows, if anything.

    Capped at :data:`MATCHES_LISTED`, ordered by the two sides' combined size, and drawn
    from the direction-test subset rather than from all 129 — a pair carried by four
    records on one side has a share slope, but not one worth putting in front of a model.
    """
    agr = facts["agreement"]
    testable = set(agr["rows"])
    rows = sorted(
        (r for r in facts["matches"] if r["label"] in testable),
        key=lambda r: -(r["a_n"] + r["b_n"]),
    )[:MATCHES_LISTED]
    return MATCHES_PROMPT[lang].format(
        name_a=facts["a"]["name"][lang],
        name_b=facts["b"]["name"][lang],
        matched=agr["all_n"],
        testable=agr["n"],
        floor=floor,
        same=agr["same"],
        r=f"{agr['r']:+.2f}",
        rows="\n".join(
            f"- {r['label']} — {r['a_n']}, {r['a_slope']:+.1f} — {r['b_n']}, "
            f"{r['b_slope']:+.1f} — {', '.join(r['a_keywords']) or '—'} — "
            f"{', '.join(r['b_keywords']) or '—'}"
            for r in rows
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("facts", type=Path, help="the comparison's own -facts.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slug", required=True, help="filename stem, e.g. <slug>-<date>")
    ap.add_argument("--model", dest="model_id", required=True)
    ap.add_argument("--lang", default="uk", choices=["uk", "en"])
    ap.add_argument(
        "--pass",
        dest="passes",
        default="all",
        choices=["all", "comparison", "taxonomies", "matches"],
        help="Which readings to generate. Each is stored separately and resumable on "
        "its own, so a pass can be added to a finished run.",
    )
    ap.add_argument(
        "--match-floor",
        type=int,
        default=None,
        help="records a matched topic needs on each side to enter the direction test. "
        "Taken from the facts file, which records the floor the comparison actually "
        "used; this overrides it, and getting it wrong only mislabels a sentence in the "
        "prompt, which is the kind of number nobody checks.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    from horizon.config import Settings
    from horizon.llm.transformers_client import TransformersClient

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    if facts.get("kind") != "comparison":
        print("error: that facts file is not a comparison's.", file=sys.stderr)
        return 1
    lang = args.lang
    floor = args.match_floor or facts.get("match_floor") or 20
    wanted = (
        {"comparison", "taxonomies", "matches"}
        if args.passes == "all"
        else {args.passes}
    )

    settings = Settings(model_id=args.model_id)
    if args.device:
        settings = replace(settings, device=args.device)
    if args.max_new_tokens is not None:
        settings = replace(
            settings,
            generation=replace(settings.generation, max_new_tokens=args.max_new_tokens),
        )
    tag = settings.model_id.rsplit("/", 1)[-1]
    out_path = args.out / f"{args.slug}-reading-{tag}-{lang}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.is_file()
        else {"model_id": settings.model_id, "label": settings.model_id, "lang": lang}
    )

    client = TransformersClient(settings)
    allowed = interpret.facts_numbers(facts)

    def budget(tokens: int):
        """The per-reading token budget, unless the caller pinned one for the whole run.

        Pinned is what a smoke test wants: the point of it is to prove the pipeline, and
        at CPU speed a full-length reading costs a quarter of an hour to prove nothing
        extra.
        """
        if args.max_new_tokens is not None:
            return settings.generation
        return replace(settings.generation, max_new_tokens=tokens)

    def ask(prompt: str, tokens: int) -> str:
        text = client.complete(prompt, config=budget(tokens))
        if interpret.is_degenerate(text):
            print("    collapsed into a repetition loop — retrying once", flush=True)
            text = client.complete(prompt, config=budget(tokens))
        return interpret.clean_reading(text)

    def store(text: str, bucket: str, key: str | None = None) -> None:
        flagged = interpret.unsupported_numbers(text, allowed)
        entry = {"text": text, "unsupported": flagged}
        if interpret.is_truncated(text):
            entry["truncated"] = True
            print("    ended mid-sentence — kept, and marked", flush=True)
        if key is None:
            done[bucket] = entry
        else:
            done.setdefault(bucket, {})[key] = entry
        out_path.write_text(
            json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if flagged:
            print(f"    unsupported figures: {', '.join(flagged[:6])}", flush=True)

    if "comparison" in wanted and "comparison" not in done:
        print("  the comparison as a whole", flush=True)
        store(
            ask(comparison_prompt(facts, lang, floor), COMPARISON_TOKENS),
            "comparison",
        )

    if "taxonomies" in wanted:
        for side in ("a", "b"):
            if side in (done.get("taxonomies") or {}):
                continue
            print(f"  taxonomy {side}: {facts[side]['taxonomy'][lang]}", flush=True)
            store(
                ask(taxonomy_prompt(facts, side, lang), READING_TOKENS),
                "taxonomies",
                side,
            )

    if "matches" in wanted and "matches" not in done:
        print("  the matched topics, as a set", flush=True)
        store(
            ask(matches_prompt(facts, lang, floor), READING_TOKENS),
            "matches",
        )

    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
