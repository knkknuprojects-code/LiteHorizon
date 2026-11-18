"""The commentary layer: a scientometrician's reading of what the report measured.

Kept apart from the report on purpose, and the separation is the point rather than a
filing convention. A report states what was measured and every number in it is traceable
to ``facts.json``; a reading says what a movement might mean, and that is a judgement.
Blending the two would let an inference borrow the authority of a measurement.

Two rules follow from that, and both are enforced here rather than requested in the
prompt:

* **Every numeral in the commentary must exist in the facts.** :func:`unsupported_numbers`
  extracts the figures from generated prose and returns the ones the facts do not
  contain, so a hallucinated count is caught mechanically instead of being believed.
* **Two models, side by side.** The enum-style judgements a model makes about maturity or
  impact are largely its own prior, so showing where two models disagree is honest where
  averaging them into one voice would not be. Both run locally; neither reaches an
  external service.

The prompt carries three things: the measured figures for the topic, a sample of the
publication summaries the scan produced, and the vocabulary the report uses. The summaries
matter more than they look — without them a model can only restate the table, because the
label and the counts say how a topic moved and nothing at all about what it studies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Publication summaries sampled per topic. Enough to show what the topic studies and how
#: that shifted; small enough that a topic of 14 000 publications costs the same as one of
#: 200, which keeps the pass time proportional to topics rather than to the corpus.
SAMPLE_SUMMARIES = 9

#: Generated length for a reading, in tokens. The package default of 512 truncated 29% of
#: blocks mid-sentence — measured, 106 of 366, the longest surviving block ending at 1831
#: characters. Ukrainian costs more tokens per character than English in this tokenizer,
#: so two paragraphs of it do not fit in a budget sized for a summary.
TOPIC_TOKENS = 1024

#: The corpus reading asks for four paragraphs rather than two.
CORPUS_TOKENS = 2048

#: An area reading is two paragraphs over a whole technology field, so it needs more room
#: than a topic and less than the corpus. It had none: the area pass called the model
#: without a budget and fell through to the 512-token default, which is half what a topic
#: gets for a smaller job. One reading in 35 ran out mid-sentence, and a reading cut off
#: mid-sentence is the failure mode nobody notices in a 67-page document.
AREA_TOKENS = 1024

#: The domain nouns the prompts are written around. A patent register runs through this
#: machinery unchanged and would otherwise be described to the model as a corpus of
#: publications — the one part of a generated reading a reader cannot check by reading it.
#:
#: Placeholders rather than a second copy of each prompt: three prompts times two
#: languages is six long strings, and duplicating them for a domain guarantees the copies
#: drift. A rule added to one and not the other is invisible until the prose is already
#: written.
VOCAB: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "bibliographic": {
        "en": {
            "corpus": "a WoS/Scopus corpus",
            "items": "publications",
            "abstracts": "publication abstracts",
            "summaries_of": "publication summaries",
            "field": "subject area",
            "fields_cap": "Subject areas",
            "per_year": "publications/year",
            "universe": "world science",
            "topic_kind": "research topic",
        },
        "uk": {
            "corpus": "корпусі WoS/Scopus",
            "items": "публікацій",
            "abstracts": "анотацій публікацій",
            "summaries_of": "стислих викладів публікацій",
            "field": "предметна область",
            "fields_cap": "Предметні області",
            "per_year": "публікацій/рік",
            "universe": "світовою наукою",
            "topic_kind": "дослідницької теми",
        },
    },
    "patents": {
        "en": {
            "corpus": "the Ukrainian patent register",
            "items": "patents",
            "abstracts": "patent abstracts",
            "summaries_of": "patent summaries",
            "field": "technology field",
            "fields_cap": "Technology fields",
            "per_year": "patents/year",
            "universe": "world invention",
            "topic_kind": "technology topic",
        },
        "uk": {
            "corpus": "патентному реєстрі України",
            "items": "патентів",
            "abstracts": "рефератів патентів",
            "summaries_of": "стислих викладів патентів",
            "field": "технологічний напрям",
            "fields_cap": "Технологічні напрями",
            "per_year": "патентів/рік",
            "universe": "світовим винахідництвом",
            "topic_kind": "технологічної теми",
        },
    },
}
DEFAULT_DOMAIN = "bibliographic"


def vocab(lang: str, domain: str = DEFAULT_DOMAIN) -> Mapping[str, str]:
    """The domain nouns for one language. Raises on an unknown domain rather than
    falling back: prose describing a patent register as a corpus of papers is worse
    than a run that refuses to start."""
    return VOCAB[domain][lang]


#: Figures in generated prose. Matches 1.05, 12%, 1 234 and 2015.
#:
#: The comma is deliberately *not* a general separator inside a number, which it was in
#: the first version: prose reading "у 2023, 2024 та 2025 роках" was swallowed whole and
#: reported as the invented figure 2023.2024. A comma is read as a decimal point only
#: when one to three digits follow it, so an enumeration splits into its members while
#: 1,06 still parses. Thousands are separated by a space, as the report writes them.
_NUMBER = re.compile(
    r"(?<![\w.,])\d+(?:[\u00a0\u202f ]\d{3})*(?:[.,]\d{1,3})?%?(?![\w])"
)


def _canon(text: str) -> str:
    """A number as it will be compared: digits and a decimal point, nothing else.

    Trailing zeros go, on both sides of the comparison. Prose writing 0.50 for a value
    the facts hold as 0.5 is using the same number, and treating the two as different
    made the check fire on ten readings that had invented nothing.
    """
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ".")).rstrip(".")
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


#: A run of one character this long is not prose. Measured: 12B falls into a repetition
#: loop on about 2% of prompts and emits middle dots until the token budget runs out —
#: 2048 of them in one block — while E4B in bfloat16 never did it once in 365.
DEGENERATE_RUN = 20

_RUN = re.compile(r"(.)\1{" + str(DEGENERATE_RUN - 1) + r",}")


#: A generation-channel marker Gemma sometimes emits before the answer proper. Anchored
#: to the start and allowed one leading stray character, because it arrived both as
#: ``thought\n`` and as ``.thought\n``.
_CHANNEL = re.compile(r"\A[.\s]*thought\s*\n", re.IGNORECASE)


def clean_reading(text: str) -> str:
    """Generated prose as a document should carry it.

    Three things go. **Headings**, because the model adds its own despite being asked not
    to, and a stray ``##`` inside a section reorders the whole document's outline.
    **A lead-in line** — a short opening that ends in a colon, of the "Ось наукометричний
    аналіз наданого корпусу:" kind — because that is an assistant answering a request,
    not a sentence of the analysis. Only when something follows it, and only when it is
    short: a real sentence can end in a colon and introduce the paragraph it belongs to.

    And **a leaked channel marker**: Gemma sometimes emits a bare ``thought`` line before
    the answer, which is the model's own scaffolding and not prose. It reached the
    rendered documents as a first word — three blocks in 87 — because it is lower-case,
    carries no markup, and reads as a typo rather than as machinery.
    """
    text = _CHANNEL.sub("", text.strip(), count=1)
    lines = [ln for ln in text.strip().splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(lines).strip()
    parts = body.split("\n", 1)
    if len(parts) == 2 and parts[0].rstrip().endswith(":") and len(parts[0]) < 200:
        body = parts[1].strip()
    return body


def is_truncated(text: str) -> bool:
    """Whether a generation stopped mid-sentence — it ran out of token budget.

    Cheap and slightly blunt: prose that ends without terminal punctuation. A reading
    cut at "Найбільші теми корпусу ви" is not a shorter reading, it is a broken one, and
    at the package default of 512 tokens 29% of them ended that way.
    """
    return bool(text.strip()) and not re.search(r"[.!?»)\]\"'’]\s*$", text.rstrip())


def is_degenerate(text: str) -> bool:
    """Whether a generation collapsed into a repetition loop.

    Checked mechanically for the same reason the numbers are: a reading that is 2048
    identical characters is worse in a report than an invented figure, because it is
    obviously broken to a reader and quietly breaks the typesetter — a single such run
    is an unbreakable 5000pt line that TeX spends hours on.
    """
    return bool(_RUN.search(text)) or not text.strip()


def facts_numbers(facts: Mapping[str, Any]) -> set[str]:
    """Every figure the facts contain, canonicalised for comparison.

    Rounded forms count as present: prose that writes 1.1 for a velocity of 1.06 is
    rounding, not inventing, and treating it as unsupported would make the check useless
    on the first sentence.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, int):
            out.add(str(node))
        elif isinstance(node, float):
            out.add(_canon(f"{node:.6f}"))
            for places in (0, 1, 2, 3):
                out.add(_canon(f"{node:.{places}f}"))
                out.add(_canon(f"{node * 100:.{places}f}"))
        elif isinstance(node, str) and node.replace(".", "").isdigit():
            out.add(node)

    walk(facts)
    return {n for n in out if n}


def unsupported_numbers(prose: str, allowed: set[str]) -> list[str]:
    """Figures in ``prose`` that the facts do not contain, in order of appearance.

    The check is deliberately blunt: it cannot tell a wrong number from a right one used
    in a context the facts do not cover, so it reports candidates for a human to look at
    rather than deciding anything. What it does catch reliably is invention — a count, a
    percentage or a year that exists nowhere in the data.
    """
    seen: list[str] = []
    for match in _NUMBER.finditer(prose):
        value = _canon(match.group(0))
        if value and value not in allowed and value not in seen:
            seen.append(value)
    return seen


def sample_summaries(
    members: Sequence[str],
    by_hash: Mapping[str, Mapping[str, Any]],
    *,
    limit: int = SAMPLE_SUMMARIES,
) -> list[tuple[str, str]]:
    """``(year, summary)`` for publications spread across the topic's lifetime.

    Spread rather than sampled at random: the question a reading has to answer is what
    changed, and the earliest and latest work in a topic is where that shows. Members are
    ordered by date first, then taken at even intervals, so a topic that turned over
    completely reads as one and a topic that did not also reads as one.
    """
    dated = [
        (str(by_hash[h].get("item_date", ""))[:4], by_hash[h].get("summary", ""))
        for h in members
        if h in by_hash and by_hash[h].get("summary")
    ]
    if not dated:
        return []
    dated.sort(key=lambda pair: pair[0])
    if len(dated) <= limit:
        return dated
    step = (len(dated) - 1) / (limit - 1)
    return [dated[round(i * step)] for i in range(limit)]


def _figures(
    topic: Mapping[str, Any], lang: str, domain: str = DEFAULT_DOMAIN
) -> list[str]:
    """The measured figures for one topic, as lines the prompt can quote from."""
    curve = ", ".join(
        f"{year}: {n}" for year, n in sorted((topic.get("cohort_counts") or {}).items())
    )
    pctile = topic.get("attention_pctile")
    words = vocab(lang, domain)
    labels = {
        "en": (
            words["items"],
            "relative growth",
            words["per_year"],
            "citation percentile",
            "first year",
            "annual counts",
            "keywords",
        ),
        "uk": (
            words["items"],
            "відносне зростання",
            words["per_year"],
            "перцентиль цитувань",
            "перший рік",
            "по роках",
            "ключові слова",
        ),
    }[lang]
    rows = [
        f"- {labels[0]}: {topic['size']}",
        f"- {labels[1]}: {topic['velocity']:.2f}",
        f"- {labels[2]}: {topic['growth']:+.1f}",
        f"- {labels[3]}: {pctile:.2f}" if pctile is not None else "",
        f"- {labels[4]}: {str(topic.get('first_seen', ''))[:4]}",
        f"- {labels[5]}: {curve}",
        f"- {labels[6]}: {', '.join(topic.get('top_keywords') or [])}",
    ]
    return [r for r in rows if r]


TOPIC_PROMPT = {
    "en": (
        "You are a scientometrician writing a short commentary on one {topic_kind} "
        "found by clustering {abstracts} in {corpus} of {records} "
        "{items} covering {span}.\n\n"
        "Topic: {label}\n\n"
        "Measured figures:\n{figures}\n\n"
        "A sample of {summaries_of} from this topic, oldest first:\n{summaries}\n\n"
        "Write two short paragraphs. The first says what this topic studies and whether "
        "its subject matter shifted over the period, drawing on the summaries. The "
        "second reads the figures: what the trajectory suggests, and what it does not "
        "establish. Relative growth is measured against the corpus, so a value below "
        "1.00 means the topic is losing share even while it may be adding {items}.\n\n"
        "Rules: use only the figures given above — do not invent counts, percentages or "
        "years. Do not repeat the figures as a list; write prose. Mark inference as "
        "inference. No headings, no bullet points."
    ),
    "uk": (
        "Ви — наукометрист, який пише короткий коментар до однієї {topic_kind}, "
        "виявленої кластеризацією {abstracts} у {corpus} обсягом "
        "{records} {items} за {span}.\n\n"
        "Тема: {label}\n\n"
        "Виміряні показники:\n{figures}\n\n"
        "Вибірка {summaries_of} цієї теми, від найдавніших:\n{summaries}\n\n"
        "Напишіть два короткі абзаци. Перший — про що ця тема і чи змінювався її "
        "предмет протягом періоду, спираючись на стислі виклади. Другий — прочитання "
        "показників: на що вказує траєкторія і чого вона не доводить. Відносне "
        "зростання вимірюється відносно корпусу, тож значення нижче 1.00 означає втрату "
        "частки навіть тоді, коли тема додає {items}.\n\n"
        "Правила: користуйтеся лише наведеними вище показниками — не вигадуйте "
        "кількостей, відсотків чи років. Не переказуйте показники списком, пишіть "
        "прозою. Позначайте припущення як припущення. Без заголовків і без переліків."
    ),
}

CORPUS_PROMPT = {
    "en": (
        "You are a scientometrician summarising {corpus} of {records} "
        "{items} covering {span}, clustered into {topics} topics.\n\n"
        "{fields_view}"
        "Fastest-growing topics relative to the corpus:\n{rising}\n\n"
        "Topics falling furthest behind the corpus:\n{declining}\n\n"
        "The largest topics:\n{largest}\n\n"
        "Write four short paragraphs: what the corpus covers and how its composition "
        "changed; {fields_task}; and what the largest topics show that the growth "
        "rankings do not.\n\n"
        "Rules: use only the figures given above — do not invent counts, percentages or "
        "years. Mark inference as inference, and say plainly where the data cannot "
        "settle a question. This corpus has its own coverage and is not {universe}.\n\n"
        "Where populations are given, every share is of its own population's cohort, so "
        "the rows compare down a population and not across them; and outside the fields "
        "listed as moving together the corpus has no single direction to report."
    ),
    "uk": (
        "Ви — наукометрист, який підсумовує {corpus} обсягом {records} "
        "{items} за {span}, кластеризований у {topics} тем.\n\n"
        "{fields_view}"
        "Теми, що найшвидше зростають відносно корпусу:\n{rising}\n\n"
        "Теми, що найбільше відстають від корпусу:\n{declining}\n\n"
        "Найбільші теми:\n{largest}\n\n"
        "Напишіть чотири короткі абзаци: що охоплює цей корпус і як змінювався його "
        "склад; {fields_task}; і що показують найбільші теми такого, чого не показують "
        "ранжування за зростанням.\n\n"
        "Правила: користуйтеся лише наведеними вище показниками — не вигадуйте "
        "кількостей, відсотків чи років. Позначайте припущення як припущення і прямо "
        "кажіть, де дані не дають відповіді. Цей корпус має власне покриття і не є "
        "{universe}.\n\n"
        "Там, де наведено сукупності, кожна частка є часткою власної когорти своєї "
        "сукупності, тож рядки порівнюються всередині сукупності, а не між ними; а поза "
        "напрямами, переліченими як спільні, корпус не має єдиного напрямку руху."
    ),
}


POP_HEADING = {
    "en": "Populations in this corpus, and what each moved toward and away from",
    "uk": "Сукупності в цьому корпусі, і куди кожна рухалася",
}
AGREED_HEADING = {
    "en": "Fields where every population moved the same way",
    "uk": "Напрями, де всі сукупності рухалися в один бік",
}
AGG_HEADING = {
    "en": ("Fields gaining the most share", "Fields losing the most share"),
    "uk": (
        "Напрями, що найбільше нарощують частку",
        "Напрями, що найбільше втрачають частку",
    ),
}
WORDS = {
    "en": {"rising": "into", "falling": "out of", "records": "records", "none": "—"},
    "uk": {"rising": "у", "falling": "з", "records": "записів", "none": "—"},
}

#: The middle paragraphs, in the two shapes the data can take. Returned together with the
#: data they describe (see :func:`fields_view`) so the instruction cannot ask for a
#: paragraph about populations that were never supplied — a mismatch whose symptom is
#: fluent prose about nothing.
FIELDS_TASK = {
    "populations": {
        "en": (
            "what each population moved toward and away from, and what that suggests "
            "about its priorities; where the populations diverge, and what a divergence "
            "means that a single number could not have said"
        ),
        "uk": (
            "куди рухалася кожна сукупність і на що це вказує щодо її пріоритетів; де "
            "сукупності розходяться і що означає таке розходження, чого не могло б "
            "сказати одне число"
        ),
    },
    "aggregate": {
        "en": (
            "what is gaining share and what that suggests; what is losing share and "
            "what that suggests"
        ),
        "uk": "що нарощує частку і на що це вказує; що втрачає частку і на що це вказує",
    },
}


def populations_block(
    facts: Mapping[str, Any], *, lang: str = "uk", top: int = 5
) -> str:
    """What each population moved toward and away from — the corpus prompt's spine.

    This replaced an aggregate-plus-caveat arrangement that did not work. Given the
    union's share movements and a rule saying not to read them as a field changing, the
    model read them that way anyway: the rule contradicted the output specification
    ("what is receding and what that suggests") and the specification won. Putting the
    populations in the primary position removes the question rather than arguing with it.

    Shares are of each population's own cohort, so rows compare down a population and not
    across them. Populations are ordered by size: a group holding 8% of the register
    carries correspondingly little of whatever it is doing.
    """
    splits = facts.get("field_splits") or []
    totals = facts.get("populations") or {}
    if not splits or not totals:
        return ""
    w = WORDS[lang]
    per: dict[str, list] = {}
    for f in splits:
        for g in f["groups"]:
            per.setdefault(g["name"], []).append(
                (f["name"], g["share_first"], g["share_last"])
            )

    def fmt(items):
        return (
            ", ".join(f"{n} {a * 100:.1f}%→{b * 100:.1f}%" for n, a, b in items)
            or w["none"]
        )

    lines = []
    for name in sorted(totals, key=lambda n: -totals[n].get("total", 0)):
        rows = sorted(per.get(name) or [], key=lambda r: r[2] - r[1])
        if not rows:
            continue
        fell = [r for r in rows[:top] if r[2] < r[1]]
        rose = [r for r in reversed(rows[-top:]) if r[2] > r[1]]
        t = totals[name]
        lines.append(
            f"- {name} ({t.get('total', 0)} {w['records']}; "
            f"{t.get('first', 0)}→{t.get('last', 0)})\n"
            f"    {w['rising']}: {fmt(rose)}\n"
            f"    {w['falling']}: {fmt(fell)}"
        )
    return f"{POP_HEADING[lang]}:\n" + "\n".join(lines) + "\n\n" if lines else ""


def agreed_block(facts: Mapping[str, Any], *, lang: str = "uk", limit: int = 8) -> str:
    """The fields whose populations all moved the same way.

    The only fields where a statement about "the field" is a statement about anything.
    On this register that is 6 of 35, which is itself worth the model knowing.
    """
    splits = [f for f in (facts.get("field_splits") or []) if not f.get("disagrees")]
    if not splits:
        return ""

    def move(f):
        return sum(g["share_last"] - g["share_first"] for g in f["groups"])

    splits.sort(key=lambda f: -abs(move(f)))
    lines = [
        f"- {f['name']}: "
        + ", ".join(
            f"{g['name']} {g['share_first'] * 100:.1f}%→{g['share_last'] * 100:.1f}%"
            for g in f["groups"]
        )
        for f in splits[:limit]
    ]
    return f"{AGREED_HEADING[lang]}:\n" + "\n".join(lines) + "\n\n"


def fields_view(facts: Mapping[str, Any], *, lang: str = "uk") -> tuple[str, str]:
    """``(data, task)`` for the field half of the corpus prompt.

    Two shapes, chosen by whether the report carries a population split. With one, the
    populations lead and the aggregate survives only for the fields they agree on.
    Without one this is the aggregate prompt it always was, so a bibliographic corpus is
    unaffected.
    """
    if facts.get("field_splits") and facts.get("populations"):
        data = populations_block(facts, lang=lang) + agreed_block(facts, lang=lang)
        if data:
            return data, FIELDS_TASK["populations"][lang]
    gain_head, lose_head = AGG_HEADING[lang]
    fields = facts.get("fields") or []
    gaining = "\n".join(
        f"- {f['name']}: {f['share_first']:.1%} → {f['share_last']:.1%}"
        for f in fields[:8]
        if f["share_delta"] > 0
    )
    # Only fields that actually lost share: the tail of the list is whatever ranked last,
    # which on the bibliographic corpus included fields that gained.
    losing = "\n".join(
        f"- {f['name']}: {f['share_first']:.1%} → {f['share_last']:.1%}"
        for f in reversed(fields)
        if f["share_delta"] < 0
    )
    data = f"{gain_head}:\n{gaining or '—'}\n\n{lose_head}:\n{losing or '—'}\n\n"
    return data, FIELDS_TASK["aggregate"][lang]


def large_topics(facts: Mapping[str, Any], *, min_size: int) -> list[Mapping[str, Any]]:
    """Topics big enough for a reading of their own, largest first, deduplicated.

    Drawn from every corpus-wide view rather than from the score ranking alone. Measured
    on this corpus, score hides a whole class: 33 topics of 500 publications or more are
    named in no table, because they sit slightly below the corpus rate and are extreme on
    no indicator axis. Selecting by size instead has no such blind spot and needs no
    argument about weighting.
    """
    seen: dict[str, Mapping[str, Any]] = {}
    for view in ("largest", "overall", "declining"):
        for topic in facts.get(view) or []:
            if topic["size"] < min_size:
                continue
            seen.setdefault(topic.get("seed_hash") or topic["label"], topic)
    return sorted(seen.values(), key=lambda t: -t["size"])


AREA_PROMPT = {
    "en": (
        "You are a scientometrician writing a short commentary on one {field} of "
        "{corpus} of {records} {items} covering {span}.\n\n"
        "{field}: {area} — {topic_count} topics, {publications} {items}.\n\n"
        "Its topics, with {items}, relative growth and {per_year}:\n"
        "{topics}\n\n"
        "A sample of {summaries_of} from this area, oldest first:\n{summaries}\n\n"
        "Write two short paragraphs. The first says what this area works on and whether "
        "its focus shifted over the period. The second reads the figures: which topics "
        "carry the area, which are growing or receding, and what the pattern does not "
        "establish. Relative growth is measured against the whole corpus, so a value "
        "below 1.00 means losing share even while {items} may be rising.\n\n"
        "Rules: use only the figures given above — do not invent counts, percentages or "
        "years. Write prose, not a list. Mark inference as inference. No headings."
    ),
    "uk": (
        "Ви — наукометрист, який пише короткий коментар до однієї одиниці «{field}» "
        "у {corpus} обсягом {records} {items} за {span}.\n\n"
        "{field}: {area} — {topic_count} тем, {publications} {items}.\n\n"
        "Її теми, з кількістю ({items}), відносним зростанням і показником {per_year}:\n"
        "{topics}\n\n"
        "Вибірка {summaries_of} цієї області, від найдавніших:\n"
        "{summaries}\n\n"
        "Напишіть два короткі абзаци. Перший — чим займається ця область і чи "
        "змінювався її фокус протягом періоду. Другий — прочитання показників: які теми "
        "несуть область, які зростають чи згасають, і чого ця картина не доводить. "
        "Відносне зростання вимірюється відносно всього корпусу, тож значення нижче "
        "1.00 означає втрату частки навіть тоді, коли одиниць ({items}) більшає.\n\n"
        "Правила: користуйтеся лише наведеними вище показниками — не вигадуйте "
        "кількостей, відсотків чи років. Пишіть прозою, а не переліком. Позначайте "
        "припущення як припущення. Без заголовків."
    ),
}


def areas(facts: Mapping[str, Any], *, min_topics: int = 1) -> list[str]:
    """Subject areas with enough topics to say anything about, largest first."""
    groups = facts.get("field_trends") or {}
    ranked = sorted(
        (
            (name, sum(t["size"] for t in topics), len(topics))
            for name, topics in groups.items()
            if len(topics) >= min_topics
        ),
        key=lambda row: -row[1],
    )
    return [name for name, _pubs, _n in ranked]


def area_prompt(
    area: str,
    facts: Mapping[str, Any],
    by_hash: Mapping[str, Mapping[str, Any]],
    *,
    lang: str = "uk",
    top: int = 15,
    domain: str = DEFAULT_DOMAIN,
) -> str:
    """The prompt for one subject area: its topics, its records, and the rules.

    Summaries are drawn from the area's largest topics rather than from the area at
    large: those are what the figures below are mostly about, and a sample spread over
    every one-publication cluster would describe the tail instead of the subject.
    """
    topics = sorted(
        (facts.get("field_trends") or {}).get(area, []), key=lambda t: -t["size"]
    )
    years = facts.get("years") or []
    span = f"{years[0]}–{years[-1]}" if years else "—"
    members: list[str] = []
    for topic in topics[:5]:
        members.extend(topic.get("members") or [])
    picked = sample_summaries(members, by_hash)
    return AREA_PROMPT[lang].format(
        records=facts.get("records", 0),
        span=span,
        area=area,
        topic_count=len(topics),
        publications=sum(t["size"] for t in topics),
        topics="\n".join(
            f"- {t['label']}: {t['size']}, {t['velocity']:.2f}, {t['growth']:+.1f}"
            for t in topics[:top]
        )
        or "—",
        summaries="\n".join(f"- {y}: {text.strip()}" for y, text in picked) or "—",
        **vocab(lang, domain),
    )


def topic_prompt(
    topic: Mapping[str, Any],
    summaries: Sequence[tuple[str, str]],
    facts: Mapping[str, Any],
    *,
    lang: str = "uk",
    domain: str = DEFAULT_DOMAIN,
) -> str:
    """The prompt for one topic: its figures, its records, and the rules."""
    years = facts.get("years") or []
    span = f"{years[0]}–{years[-1]}" if years else "—"
    body = "\n".join(f"- {year}: {text.strip()}" for year, text in summaries)
    return TOPIC_PROMPT[lang].format(
        records=facts.get("records", 0),
        span=span,
        label=topic.get("label", ""),
        figures="\n".join(_figures(topic, lang, domain)),
        summaries=body or "—",
        **vocab(lang, domain),
    )


def _rows(topics: Sequence[Mapping[str, Any]], limit: int) -> str:
    """Topic lines for the corpus prompt: label, size, both growth measures."""
    return "\n".join(
        f"- {t['label']}: {t['size']}, {t['velocity']:.2f}, {t['growth']:+.1f}"
        for t in topics[:limit]
    )


def corpus_prompt(
    facts: Mapping[str, Any],
    *,
    lang: str = "uk",
    limit: int = 12,
    domain: str = DEFAULT_DOMAIN,
) -> str:
    """The prompt for the corpus-wide reading."""
    years = facts.get("years") or []
    span = f"{years[0]}–{years[-1]}" if years else "—"
    view, task = fields_view(facts, lang=lang)
    return CORPUS_PROMPT[lang].format(
        records=facts.get("records", 0),
        span=span,
        topics=facts.get("topics_total") or len(facts.get("overall") or []),
        fields_view=view,
        fields_task=task,
        rising=_rows(facts.get("overall") or [], limit),
        declining=_rows(facts.get("declining") or [], limit),
        largest=_rows(facts.get("largest") or [], limit),
        **vocab(lang, domain),
    )
