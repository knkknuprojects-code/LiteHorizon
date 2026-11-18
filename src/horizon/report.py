"""Assembling a corpus report: facts first, prose after.

The report is built bottom-up, from the layer that can fail least to the layer that can
fail most:

1. **Field development** — records per subject area per year. No model, no embedder, no
   clustering, no threshold, so it stands whatever the clustering does.
2. **Cross-field interaction** — the same counting over field pairs.
3. **Trends within each field** — clusters, scored against their own field's growth.
4. **Corpus-wide trends** — the per-field results re-ranked together.

Each level stands alone, so a failure at one does not invalidate the ones above it.

This module is deliberately pure: it takes counts and already-built trends and returns
data, then renders that data. Clustering and embedding live in the caller, which keeps
every table here testable without a GPU and makes the intermediate JSON — the thing the
prose layer is checked against — the same object the markdown is rendered from.

**Every figure is a share of this corpus, not of world science.** The corpus is an export
with its own scope, and the numbers read as general science if nothing says so. The
renderer emits that line; it is not decoration.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from horizon.fields import Series, Tabulation, field_series, pair_series
from horizon.lenses import BY_NAME
from horizon.metadata_lenses import Facts as MetaFacts
from horizon.trends import Trend

#: Report language codes and the strings that differ between them. Kept as data rather
#: than as two renderers so a change to the structure cannot land in one language only —
#: the bilingual rule is otherwise enforced by memory.
#: Publications a topic needs before it is discussed on its own terms — the decline
#: ranking's floor, and the scope of the commentary that reads this report.
#:
#: Measured rather than picked: below 100 the decline ranking is nothing but topics that
#: stopped publishing outright, four of the worst five sitting at exactly 0.00 relative
#: growth on 21 to 50 publications. That is a different phenomenon from a large topic
#: losing share, and it crowds the latter out entirely. At 200 the same table reads
#: Educational Technology Adoption at 0.33 over 480 publications and Infectious Disease
#: Surveillance at 0.45 over 947 — recognisable retreats rather than small extinctions.
LARGE_TOPIC_SIZE = 200

#: A page break, written once for each output format pandoc has to produce it in.
#:
#: Two raw blocks rather than one, because neither format understands the other's: the
#: LaTeX writer emits the ``\clearpage`` and drops the OpenXML, the DOCX writer does the
#: reverse, and a plain-markdown reader sees two short fenced blocks. That is the price of
#: keeping the break in the ``.md``, and it has to be there — the ``.md`` is the source of
#: record, and a break applied in ``render.py``'s preamble would reach the PDF only.
PAGE_BREAK = (
    "```{=latex}",
    r"\clearpage",
    "```",
    "",
    "```{=openxml}",
    '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
    "```",
    "",
)

LABELS: Mapping[str, Mapping[str, str]] = {
    "en": {
        "title": "Scientometric corpus report",
        "generated": "Publications",
        "span": "Years",
        "unclassified": "without a subject area",
        "l1": "1. Subject-area development",
        "l1_note": (
            "Share of the corpus, and publications per year, from the subject areas "
            "WoS and Scopus assigned. No clustering involved."
        ),
        "l1_measured": "Shares are read at the first and last classified cohorts,",
        "l2": "2. Subject-area co-occurrence",
        "l2_note": (
            "Publications classified into both areas, ranked by least-squares slope."
        ),
        "l3": "3. Topics within each subject area",
        "l4": "4. Corpus-wide topics",
        "l4_rising": "4.1. Ranked by score",
        "l4_declining": "4.2. Falling furthest behind the corpus",
        "l4_largest": "4.3. The largest topics",
        "l4_rising_note": (
            "Score is velocity x publications x recency x citation percentile, so this "
            "ranking leans toward topics that are both large and recently growing."
        ),
        "l4_declining_note": (
            "The same topics ranked by relative growth, lowest first, among those with "
            "at least 200 publications. A topic here can still be adding publications "
            "every year — it is losing share, not necessarily volume."
        ),
        "l4_largest_note": (
            "Ranked by publications alone, the one criterion with nothing composite in "
            "it. It exists because the two rankings above are both about change, and a "
            "large topic moving at the corpus rate appears in neither: measured on this "
            "corpus, 33 topics of 500 publications or more are named in no other table."
        ),
        "trend_note": (
            "Topics are clusters of semantically similar publications, ranked by "
            "score. The two growth measures answer different questions and are "
            "expected to disagree.\n\n"
            "**Relative growth** is the ratio of the topic's publications over the "
            "most recent period to its publications over a preceding period of equal "
            "length, normalised by the same ratio computed for the whole corpus. A "
            "value of 1.0 matches the corpus rate; above 1.0 the topic is outgrowing "
            "the corpus, below it is falling behind. A topic can add publications every "
            "year and still fall below 1.0 if the corpus grows faster.\n\n"
            "**Publications/year** is the slope coefficient of a linear regression of "
            "the topic's annual publication counts, estimated by least squares over its "
            "whole life. It is an absolute measure and is not normalised against the "
            "corpus."
        ),
        "field": "subject area",
        "pair": "pair of areas",
        "first": "first",
        "last": "last",
        "delta": "change",
        "slope": "per year",
        "n": "publications",
        "trend": "topic",
        "size": "publications",
        "velocity": "relative growth",
        "growth": "publications/year",
        "seen": "first year",
        "keywords": "keywords",
        "l6": "6. Who is working on each topic",
        "l6_note": (
            "Every topic split across the populations that filed it, each rated against "
            "its own corpus-wide rate rather than against the total. A population that "
            "is contracting everywhere would otherwise make all of its topics read as "
            "declining, which says something about the population and nothing about any "
            "topic.\n\n"
            "Ranked by spread, because that is what this section adds. A topic where "
            "every population moves together carries nothing the overall row did not; "
            "one where a group is rising while another withdraws is a third claim, "
            "visible in neither the total nor a run filtered to one population.\n\n"
            "Read the direction, not the magnitude. A ratio needs only two records in "
            "the prior window to be reported, so a large value can rest on a very small "
            "denominator: the largest here divides thirty recent records by two earlier "
            "ones. That a group grew is robust; how many times over is not. A group "
            "absent from the prior window entirely is marked new rather than given a "
            "number, because it has no rate at all."
        ),
        "group": "population",
        "spread": "spread",
        "new": "new",
        "l5": "5. Indicators",
        "l5_note": (
            "The same topics read several ways. Each indicator is a selection rule "
            "plus a ranking quantity over rows already reported above — no new "
            "measurement, and every figure is reproducible from the accompanying JSON."
        ),
        "score": "score",
        "decay": "volume going quiet",
        "dormancy": "latency period, years",
        "attention": "citation percentile",
        "sustain": "years above half-peak",
        "value": "value",
        "venues": "journals per publication",
        "newshare": "share of new authors",
        "scripts": "Cyrillic / Latin",
        "none": "_No topic met the reporting threshold._",
        "caveat": (
            "All figures are shares of this corpus — a WoS/Scopus export with its own "
            "coverage — and not of world science."
        ),
        # Indicator headings. Kept as labels rather than as the internal names so the
        # Ukrainian report stops printing English slugs as its section titles.
        "ind_declining": "Fading publication activity",
        "ind_sleeping-beauty": "Delayed recognition",
        "ind_high-impact-low-output": "High citation impact at a small body of work",
        "ind_high-output-low-impact": "A large body of work at low citation impact",
        "ind_sustained": "Post-peak persistence",
        "ind_journal-dispersion": "Journal dispersion",
        "ind_new-authors": "Renewal of the author set",
        "ind_script-distribution": "Concentration of Cyrillic",
        "ind_declining_note": (
            "Topics publishing less than they used to, ranked by publications "
            "weighted by the shortfall in relative growth — the volume that is going "
            "quiet, not a rate per author. Only topics at or above the median size "
            "qualify: below it a decline has no established body of work to be a "
            "decline from."
        ),
        "ind_sleeping-beauty_note": (
            "Topics whose peak year came at least two years after their first "
            "publication and which are still growing relative to the corpus, ranked by "
            "the length of that latency. Publications are a column of their own, so a "
            "long latency and a large body of work stay separable."
        ),
        "ind_high-impact-low-output_note": (
            "Topics below the median in publications and ranked by citation percentile, "
            "highest first — small bodies of work drawing citation out of proportion to "
            "their size. Topics with no citation data are absent rather than ranked "
            "last."
        ),
        "ind_high-output-low-impact_note": (
            "The other half of the same split: topics above the median in publications, "
            "ranked by citation percentile lowest first. The two indicators partition "
            "one population, so a topic appears in exactly one of them."
        ),
        "ind_sustained_note": (
            "Topics that held their level after peaking rather than spiking once, "
            "ranked by how many years after the peak still produced at least half of "
            "it."
        ),
        "ind_journal-dispersion_note": (
            "Journals per publication in the topic's later half over the same figure in "
            "its earlier half. Above 1.0 the topic is spreading across venues faster "
            "than its own volume explains — Bradford scattering, read forward in time."
        ),
        "ind_new-authors_note": (
            "The share of authors in the topic's later half who are absent from its "
            "earlier half. Author names are compared as written, so a name spelled two "
            "ways counts as two people and the figure is an upper bound."
        ),
        "ind_script-distribution_note": (
            "Topics ranked by the share of their publications written in Cyrillic, "
            "which is 8.7% of this corpus against 91.3% Latin. Ranking by lopsidedness "
            "instead answers nothing on a corpus this "
            "one-sided: hundreds of topics are Latin-only, they share the same extreme "
            "value, and the table reads 0 / N throughout. Restricted to publications "
            "whose script is known rather than to a topic size, since a topic whose "
            "records could not be matched to bibliographic ones carries no evidence "
            "about script however large it is."
        ),
    },
    "uk": {
        "title": "Наукометричний звіт по корпусу",
        "generated": "Публікацій",
        "span": "Роки",
        "unclassified": "без предметної області",
        "l1": "1. Розвиток предметних областей",
        "l1_note": (
            "Частка корпусу і публікацій на рік за предметними областями, які "
            "призначили WoS і Scopus. Кластеризація не задіяна."
        ),
        "l1_measured": "Частки взято за перший і останній роки, для яких є класифікація,",
        "l2": "2. Співпоява предметних областей",
        "l2_note": (
            "Публікації, віднесені до обох областей, за нахилом найменших квадратів."
        ),
        "l3": "3. Теми в межах предметних областей",
        "l4": "4. Теми по всьому корпусу",
        "l4_rising": "4.1. За оцінкою",
        "l4_declining": "4.2. Найбільше відстають від корпусу",
        "l4_largest": "4.3. Найбільші теми",
        "l4_rising_note": (
            "Оцінка є добутком відносного зростання, обсягу, свіжості та перцентиля "
            "цитувань, тож це впорядкування схиляється до тем, які водночас великі й "
            "нещодавно зростали."
        ),
        "l4_declining_note": (
            "Ті самі теми, впорядковані за відносним зростанням від найнижчого, серед "
            "тих, що мають щонайменше 200 публікацій. Тема звідси може й далі щороку "
            "додавати публікації: вона втрачає частку, а не обов'язково обсяг."
        ),
        "l4_largest_note": (
            "Впорядкування за кількістю публікацій — єдиний критерій без жодної "
            "композитності. Він потрібен, бо два попередні стосуються зміни, а велика "
            "тема, що рухається в темпі корпусу, не потрапляє в жоден із них: на цьому "
            "корпусі 33 теми обсягом від 500 публікацій не названі в жодній іншій "
            "таблиці."
        ),
        "trend_note": (
            "Теми — це кластери семантично подібних публікацій, упорядковані за "
            "оцінкою. Дві міри зростання відповідають на різні питання, і розбіжність "
            "між ними очікувана.\n\n"
            "**Відносне зростання** — це відношення кількості публікацій теми за "
            "останній період до кількості за попередній період тієї самої тривалості, "
            "нормалізоване на аналогічне відношення для всього корпусу. Значення 1.0 "
            "відповідає темпові корпусу, більші за 1.0 — випередженню, менші — "
            "відставанню. Тема може щороку збільшувати кількість публікацій і водночас "
            "мати значення нижче 1.0, якщо корпус зростає швидше.\n\n"
            "**Публікації/рік** — це коефіцієнт нахилу лінійної регресії річних "
            "кількостей публікацій теми, оцінений методом найменших квадратів за весь "
            "період її існування. Показник абсолютний і на динаміку корпусу не "
            "нормалізується."
        ),
        "field": "предметна область",
        "pair": "пара областей",
        "first": "початок",
        "last": "кінець",
        "delta": "зміна",
        "slope": "на рік",
        "n": "публікацій",
        "trend": "тема",
        "size": "публікацій",
        "velocity": "відносне зростання",
        "growth": "публікацій/рік",
        "seen": "перший рік",
        "keywords": "ключові слова",
        "l6": "6. Хто працює над кожною темою",
        "l6_note": (
            "Кожну тему розкладено за сукупностями заявників, і кожну з них оцінено "
            "проти її власного темпу по всьому корпусу, а не проти підсумку. Інакше "
            "сукупність, яка стискається всюди, показала б спад у кожній своїй темі — "
            "а це твердження про сукупність і ні про яку тему.\n\n"
            "Впорядковано за розкидом, бо саме це додає розділ. Тема, де всі "
            "сукупності рухаються разом, не несе нічого понад загальний рядок; тема, де "
            "одна група зростає, а інша виходить, є третім твердженням, якого не видно "
            "ні в підсумку, ні у звіті, відфільтрованому до однієї сукупності.\n\n"
            "Читати слід напрям, а не величину. Для оцінки достатньо двох записів у "
            "ранньому вікні, тож велике значення може спиратися на дуже малий "
            "знаменник: найбільше тут ділить тридцять свіжих записів на два ранні. Те, "
            "що група зросла, є надійним; у скільки разів — ні. Групу, цілком відсутню "
            "в ранньому вікні, позначено як нову, а не оцінено числом, бо ставки в неї "
            "немає взагалі."
        ),
        "group": "сукупність",
        "spread": "розкид",
        "new": "нова",
        "l5": "5. Індикатори",
        "l5_note": (
            "Ті самі теми, прочитані кількома способами. Кожен індикатор — це правило "
            "відбору плюс величина ранжування над рядками, вже наведеними вище: "
            "жодного нового вимірювання, і кожне число відтворюється із супровідного файлу "
            "фактів."
        ),
        "score": "оцінка",
        "decay": "обсяг, що згасає",
        "dormancy": "тривалість латентного періоду, років",
        "attention": "перцентиль цитувань",
        "sustain": "років понад половину піку",
        "value": "значення",
        "venues": "видань на публікацію",
        "newshare": "частка нових авторів",
        "scripts": "кирилиця / латиниця",
        "none": "_Жодна тема не подолала поріг звітування._",
        "caveat": (
            "Усі цифри є частками цього корпусу — експорту WoS/Scopus із власним "
            "покриттям — а не світової науки."
        ),
        "ind_declining": "Згасання публікаційної активності",
        "ind_sleeping-beauty": "Відкладене визнання",
        "ind_high-impact-low-output": "Високий цитувальний вплив за малого обсягу публікацій",
        "ind_high-output-low-impact": "Великий обсяг публікацій за низького цитувального впливу",
        "ind_sustained": "Постпікова стійкість",
        "ind_journal-dispersion": "Розсіювання за виданнями",
        "ind_new-authors": "Оновлення авторського складу",
        "ind_script-distribution": "Зосередження кирилиці",
        "ind_declining_note": (
            "Теми, що публікують менше, ніж раніше, упорядковані за обсягом, "
            "зваженим на нестачу відносного зростання, — тобто за кількістю публікацій, "
            "що згасає, а не за виробітком на автора. Проходять лише теми не менші за "
            "медіанну: нижче неї спад не має усталеного масиву робіт, від якого був би "
            "спадом."
        ),
        "ind_sleeping-beauty_note": (
            "Теми, чий піковий рік настав щонайменше через два роки після першої "
            "публікації і які досі зростають відносно корпусу, впорядковані за "
            "тривалістю цього латентного періоду. Обсяг є окремою колонкою, тож довга "
            "латентність і великий масив робіт лишаються окремими величинами."
        ),
        "ind_high-impact-low-output_note": (
            "Теми, менші за медіанний обсяг, упорядковані за перцентилем цитувань від "
            "найвищого — малі масиви робіт, що збирають цитування непропорційно до "
            "свого розміру. Теми без даних про цитування відсутні, а не поставлені в "
            "кінець."
        ),
        "ind_high-output-low-impact_note": (
            "Друга половина того самого поділу: теми, більші за медіанний обсяг, "
            "упорядковані за перцентилем цитувань від найнижчого. Два індикатори ділять "
            "одну сукупність, тож тема потрапляє рівно до одного з них."
        ),
        "ind_sustained_note": (
            "Теми, що втримали рівень після піку, а не дали разовий сплеск, "
            "упорядковані за кількістю років після піку, у які кількість публікацій "
            "становила щонайменше половину пікової."
        ),
        "ind_journal-dispersion_note": (
            "Видань на публікацію в пізнішій половині життя теми, поділене на той самий "
            "показник у ранішій. Вище 1.0 тема розходиться виданнями швидше, ніж "
            "пояснює її власний обсяг, — розсіювання за Бредфордом у динаміці."
        ),
        "ind_new-authors_note": (
            "Частка авторів пізнішої половини, відсутніх у ранішій. Імена порівнюються "
            "як записані, тож ім'я, написане двома способами, рахується за двох людей, "
            "і величина є верхньою межею."
        ),
        "ind_script-distribution_note": (
            "Теми, впорядковані за часткою публікацій, писаних кирилицею; вона "
            "становить 8.7% цього корпусу проти 91.3% латиниці. "
            "Упорядкування за однобокістю не відповідає ні на що на настільки "
            "несиметричному корпусі: сотні тем є суто латинськими, усі вони дають "
            "однакове крайнє значення, і таблиця складається з рядків «0 / N». Відбір "
            "обмежено публікаціями з відомою системою письма, а не обсягом теми: тема, "
            "чиї записи не вдалося зіставити з бібліографічними, не несе свідчень про "
            "письмо, хоч би якою великою вона була."
        ),
    },
}


@dataclass(frozen=True)
class ReportFacts:
    """Every row the report draws on, before any of it becomes a sentence."""

    records: int
    unclassified: int
    years: tuple[int, ...]
    fields: tuple[Series, ...]
    pairs: tuple[Series, ...]
    field_trends: Mapping[str, tuple[Trend, ...]] = field(default_factory=dict)
    overall: tuple[Trend, ...] = ()
    #: Lens name -> the trends it selects, ranked by it. Same objects as ``overall``.
    lenses: Mapping[str, tuple[Trend, ...]] = field(default_factory=dict)
    #: Lens name -> record-side facts, for the three that read venue, authors or script.
    meta_lenses: Mapping[str, tuple[MetaFacts, ...]] = field(default_factory=dict)
    #: The cohorts the share columns are read at — the first and last years in which any
    #: record carries a subject area. Equal to the ends of ``years`` unless the taxonomy
    #: is missing from an end cohort, which the WoS/Scopus 2026 stub is.
    classified_years: tuple[int, ...] = ()
    #: The cohorts the shares and slopes were fitted over, when narrower than ``years``.
    #: Empty means they were fitted over everything shown, which is the usual case. It is
    #: set where the trailing cohorts are still filling up: they stay in ``counts`` so the
    #: span remains visible, and the derived columns avoid them.
    fit_years: tuple[int, ...] = ()
    #: Per-topic split across a record field, ranked by how far its groups disagree.
    #: Empty unless the report was asked for one (``--compose-by``).
    compositions: tuple = ()
    #: Per-field share of each population at the ends of the span. Section 1's aggregate
    #: is a weighted blend where the populations differ, and this is what says so — the
    #: prose layer reads it to avoid attributing a change in the mix to a change in a
    #: field.
    field_splits: tuple = ()
    #: Size of each population overall and at the ends of the span. The prose layer reads
    #: it to order the populations and to know which are large enough to be worth reading.
    populations: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    #: The same corpus-wide topics ranked two other ways. ``overall`` ranks by score, a
    #: composite that leans toward growth, so on its own it hides a whole class: measured
    #: here, 33 topics of 500+ publications appear in no table at all, because they sit
    #: slightly below the corpus rate and are extreme on no indicator axis. Ranking by
    #: decline and by size answers what score cannot.
    declining: tuple[Trend, ...] = ()
    largest: tuple[Trend, ...] = ()
    #: How many corpus-wide topics were found, as against how many are listed. The lists
    #: above are capped at ``--top-trends``, so their length is a reporting decision and
    #: not a measurement — anything reading the facts and wanting the second number was
    #: silently getting the first.
    topics_total: int = 0
    #: Publications per year. The denominator under every share in level 1, and not
    #: recoverable from the per-area counts: 45% of publications carry more than one
    #: subject area, so those sum past the cohort they came from.
    year_totals: Mapping[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-ready facts — the object the prose layer is checked against.

        Written out beside the report so a generated sentence can be verified against
        the rows it was given, and so a later re-analysis reads these rather than
        re-deriving them.
        """
        return {
            "records": self.records,
            "unclassified": self.unclassified,
            "years": list(self.years),
            # The cohorts the share columns were read at. In the JSON because the
            # rendered note names them, and every numeral in the prose has to be
            # traceable to this file.
            "classified_years": list(self.classified_years),
            # The cohorts the shares and slopes were actually fitted over, when that is
            # narrower than the span shown. The figures read it to mark the rest: a
            # partial cohort swings hard for reasons that are not about the subject, and
            # a reader who cannot see which cohorts those are will read the swing.
            "fit_years": list(self.fit_years) if self.fit_years else [],
            "populations": {k: dict(v) for k, v in self.populations.items()},
            "field_splits": [
                {
                    "name": f.name,
                    "disagrees": f.disagrees,
                    "groups": [
                        {"name": g, "share_first": a, "share_last": b}
                        for g, a, b in f.groups
                    ],
                }
                for f in self.field_splits
            ],
            "compositions": [
                {
                    "label": c.label,
                    "seed_hash": c.seed_hash,
                    "size": c.size,
                    "resolved": c.resolved,
                    "divergence": c.divergence,
                    "groups": [
                        {
                            "name": g.name,
                            "size": g.size,
                            "share": g.share,
                            "velocity": g.velocity,
                            "first_year": g.first_year,
                            "last_year": g.last_year,
                            # Written because the table prints it: a group with no
                            # velocity is either new or too thin to rate, and without
                            # this the two collapse into one dash on a re-render.
                            "is_new": g.is_new,
                        }
                        for g in c.groups
                    ],
                }
                for c in self.compositions
            ],
            "year_totals": {str(y): n for y, n in sorted(self.year_totals.items())},
            "fields": [_series_dict(s) for s in self.fields],
            "pairs": [_series_dict(s) for s in self.pairs],
            "field_trends": {
                name: [t.to_dict() for t in trends]
                for name, trends in self.field_trends.items()
            },
            "overall": [t.to_dict() for t in self.overall],
            "topics_total": self.topics_total,
            "declining": [t.to_dict() for t in self.declining],
            "largest": [t.to_dict() for t in self.largest],
            "lenses": {
                name: [t.to_dict() for t in rows] for name, rows in self.lenses.items()
            },
            "meta_lenses": {
                name: [_meta_dict(f) for f in rows]
                for name, rows in self.meta_lenses.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ReportFacts":
        """Rebuild the facts from what :meth:`to_dict` wrote.

        The point of the inverse is that the two halves of a report cost different
        amounts. Counting, embedding and clustering a corpus of a quarter-million records
        is most of an hour; deciding that a table should print forty rows instead of ten
        is a minute. Without this, the cheap half could only be revised by paying for the
        expensive one again — and paying for it again re-measures, so a formatting change
        would arrive mixed with numbers that had quietly moved.

        Tolerant of a file older than the fields it is read into: ``fit_years``,
        ``populations``, ``field_splits`` and ``compositions`` all post-date the first
        corpus runs, and a facts file written before them is not broken, only earlier.
        Each falls back to its own default rather than to an error.
        """
        from horizon.composition import Composition, FieldSplit, Group

        return cls(
            records=data["records"],
            unclassified=data.get("unclassified", 0),
            years=tuple(data.get("years") or ()),
            fields=tuple(Series.from_dict(d) for d in data.get("fields") or ()),
            pairs=tuple(Series.from_dict(d) for d in data.get("pairs") or ()),
            field_trends={
                name: tuple(Trend.from_dict(t) for t in rows)
                for name, rows in (data.get("field_trends") or {}).items()
            },
            overall=tuple(Trend.from_dict(t) for t in data.get("overall") or ()),
            lenses={
                name: tuple(Trend.from_dict(t) for t in rows)
                for name, rows in (data.get("lenses") or {}).items()
            },
            meta_lenses={
                name: tuple(
                    _MetaRow(
                        label=r["label"],
                        size=r["size"],
                        resolved=r.get("resolved", 0),
                        venue_spread=r.get("venue_spread"),
                        new_author_share=r.get("new_author_share"),
                        cyrillic=r.get("cyrillic", 0),
                        latin=r.get("latin", 0),
                    )
                    for r in rows
                )
                for name, rows in (data.get("meta_lenses") or {}).items()
            },
            classified_years=tuple(data.get("classified_years") or ()),
            fit_years=tuple(data.get("fit_years") or ()),
            compositions=tuple(
                Composition(
                    label=c["label"],
                    seed_hash=c.get("seed_hash", ""),
                    size=c["size"],
                    resolved=c.get("resolved", 0),
                    groups=tuple(
                        Group(
                            name=g["name"],
                            size=g["size"],
                            share=g["share"],
                            velocity=g.get("velocity"),
                            first_year=g.get("first_year"),
                            last_year=g.get("last_year"),
                            is_new=g.get("is_new", False),
                        )
                        for g in c.get("groups") or ()
                    ),
                )
                for c in data.get("compositions") or ()
            ),
            field_splits=tuple(
                FieldSplit(
                    name=f["name"],
                    groups=tuple(
                        (g["name"], g["share_first"], g["share_last"])
                        for g in f.get("groups") or ()
                    ),
                )
                for f in data.get("field_splits") or ()
            ),
            populations={
                k: dict(v) for k, v in (data.get("populations") or {}).items()
            },
            declining=tuple(Trend.from_dict(t) for t in data.get("declining") or ()),
            largest=tuple(Trend.from_dict(t) for t in data.get("largest") or ()),
            topics_total=data.get("topics_total", 0),
            # JSON has no integer keys, so the cohorts came back as strings; every
            # reader of this mapping indexes it by year.
            year_totals={int(y): n for y, n in (data.get("year_totals") or {}).items()},
        )


@dataclass(frozen=True)
class _MetaRow:
    """A record-side lens row as the facts file recorded it — not the lens behind it.

    :func:`_meta_dict` writes the derived ``venue_spread`` and drops the early/late venue
    and record counts it was computed from, so a row read back from the JSON cannot be a
    :class:`MetaFacts`. It can be exactly this: the fields the file holds, which is also
    everything :func:`_meta_rows` prints and everything :func:`_meta_dict` needs to write
    the same file again.
    """

    label: str
    size: int
    resolved: int
    venue_spread: float | None
    new_author_share: float | None
    cyrillic: int
    latin: int


def _meta_dict(facts: MetaFacts | _MetaRow) -> dict:
    """One record-side lens row as JSON-ready data."""
    return {
        "label": facts.label,
        "size": facts.size,
        "resolved": facts.resolved,
        "venue_spread": (
            round(facts.venue_spread, 3) if facts.venue_spread is not None else None
        ),
        "new_author_share": (
            round(facts.new_author_share, 3)
            if facts.new_author_share is not None
            else None
        ),
        "cyrillic": facts.cyrillic,
        "latin": facts.latin,
    }


def _series_dict(series: Series) -> dict:
    """One field or pair series as JSON-ready data."""
    return {
        "name": series.name,
        "total": series.total,
        "slope": round(series.slope, 3),
        "share_first": round(series.share_first, 5),
        "share_last": round(series.share_last, 5),
        "share_delta": round(series.share_delta, 5),
        "counts": {str(y): n for y, n in series.counts},
    }


def build_facts(
    tab: Tabulation,
    *,
    field_trends: Mapping[str, Sequence[Trend]] | None = None,
    overall: Sequence[Trend] = (),
    lenses: Mapping[str, Sequence[Trend]] | None = None,
    meta_lenses: Mapping[str, Sequence[MetaFacts]] | None = None,
    min_field_total: int = 0,
    min_pair_total: int = 0,
    top_fields: int | None = None,
    top_pairs: int | None = None,
    top_trends: int | None = None,
    large_topic_size: int = LARGE_TOPIC_SIZE,
    fit_years: Sequence[int] | None = None,
    compositions: Sequence = (),
    field_splits: Sequence = (),
    populations: Mapping[str, Mapping[str, int]] | None = None,
) -> ReportFacts:
    """Assemble the report's rows from a tabulation and already-built trends.

    Trends are passed in rather than computed here: clustering needs an embedder and a
    GPU, and keeping it outside means every table below is testable without either.

    ``fit_years`` keeps every cohort in the counts while restricting the shares and slopes
    to the cohorts complete enough to derive from — see :func:`horizon.fields._series` for
    why a partial cohort corrupts a share as well as a rate.
    """
    fields = field_series(tab, min_total=min_field_total, fit_years=fit_years)
    pairs = pair_series(tab, min_total=min_pair_total, fit_years=fit_years)
    fitted = tuple(fit_years) if fit_years else ()
    trends_by_field = {
        name: tuple(rows[:top_trends] if top_trends is not None else rows)
        for name, rows in (field_trends or {}).items()
        if rows
    }
    ranked = list(overall)
    by_decline = sorted(
        (t for t in ranked if t.size >= large_topic_size),
        key=lambda t: (t.velocity, -t.size),
    )
    by_size = sorted(ranked, key=lambda t: (-t.size, t.seed_hash))
    cap = top_trends if top_trends is not None else None
    return ReportFacts(
        records=tab.records,
        topics_total=len(ranked),
        declining=tuple(by_decline[:cap] if cap else by_decline),
        largest=tuple(by_size[:cap] if cap else by_size),
        unclassified=tab.unclassified,
        years=tab.years,
        classified_years=tab.classified_years,
        fit_years=fitted,
        compositions=tuple(compositions),
        field_splits=tuple(field_splits),
        populations=dict(populations or {}),
        year_totals=dict(tab.year_totals),
        fields=tuple(fields[:top_fields] if top_fields is not None else fields),
        pairs=tuple(pairs[:top_pairs] if top_pairs is not None else pairs),
        field_trends=trends_by_field,
        overall=tuple(overall[:top_trends] if top_trends is not None else overall),
        lenses={
            name: tuple(rows[:top_trends] if top_trends is not None else rows)
            for name, rows in (lenses or {}).items()
            if rows
        },
        meta_lenses={
            name: tuple(rows[:top_trends] if top_trends is not None else rows)
            for name, rows in (meta_lenses or {}).items()
            if rows
        },
    )


#: Relative column widths for the topic table, as the dash counts pandoc reads them from.
#:
#: Not decoration. A pipe table wider than the page gets *equal* columns unless the rule
#: says otherwise, and seven equal columns put the keyword list in 2.4 cm — eight wrapped
#: lines per row, in a table of forty rows — while four numeric columns sat half empty.
#: These weights give the label 27% of the text width and the keywords 40%.
TREND_RULE = (
    "|"
    + "|".join(
        ("-" * (width - 1) + ":") if right else "-" * width
        for width, right in (
            (26, False),  # topic
            (5, True),  # publications
            (5, True),  # score
            (8, True),  # relative growth
            (8, True),  # publications/year
            (6, False),  # first year
            (38, False),  # keywords
        )
    )
    + "|"
)


def _cell(text: str) -> str:
    """One table cell's text, with anything that would split the row escaped.

    Labels and keywords are model output. A single ``|`` in either silently ends the cell
    early and shifts every column after it, which reads as a data error rather than as a
    formatting one.
    """
    return text.replace("|", r"\|")


def _pct(x: float) -> str:
    """A share as a signed-free percentage with one decimal."""
    return f"{x:.1%}"


def disambiguate(trends: Sequence[Trend]) -> list[Trend]:
    """Give repeated labels a distinguishing keyword.

    ``trends._label`` names a cluster by the most common ``theme`` among its members, and
    two large clusters in the same area collide — a report listing "Materials Science"
    twice reads as a bug whether or not the clusters are genuinely different. The first
    top keyword not shared with the colliding cluster is appended; failing that, the
    seed year, which is always available.
    """
    seen = Counter(t.label for t in trends)
    used: set[str] = set()
    out: list[Trend] = []
    for trend in trends:
        if seen[trend.label] == 1:
            out.append(trend)
            used.add(trend.label)
            continue
        for keyword in trend.top_keywords:
            candidate = f"{trend.label} ({keyword})"
            if candidate not in used:
                break
        else:
            candidate = f"{trend.label} ({trend.first_seen[:4]})"
        suffix = 2
        while candidate in used:
            candidate = f"{trend.label} ({trend.first_seen[:4]}, {suffix})"
            suffix += 1
        used.add(candidate)
        out.append(replace(trend, label=candidate))
    return out


def _trend_rows(
    trends: Sequence[Trend],
    lab: Mapping[str, str],
    *,
    lens: object | None = None,
) -> list[str]:
    """A markdown table for a list of trends, or the empty-set note.

    With ``lens``, the table shows the quantity that lens ranks on instead of the
    generic velocity/growth pair."""
    if not trends:
        return [lab["none"], ""]
    if lens is not None:
        out = [
            f"| {lab['trend']} | {lab['size']} | {lab[lens.column]} |",
            "|---|--:|--:|",
        ]
        out += [f"| {t.label} | {t.size} | {lens.key(t):.2f} |" for t in trends]
        out.append("")
        return out
    # Score is the column the rows are ranked on, so it has to be visible: without it
    # the table looks mis-sorted, since size, velocity and growth each disagree with the
    # order in some row.
    head = (
        f"| {lab['trend']} | {lab['size']} | {lab['score']} | {lab['velocity']} | "
        f"{lab['growth']} | {lab['seen']} | {lab['keywords']} |"
    )
    out = [head, TREND_RULE]
    for t in trends:
        # The label is the cluster's most common theme, which names a field more often
        # than it names the work: four tables in a row headed "Materials Science" say
        # nothing about what separates them. The terms the members actually share do.
        out.append(
            f"| {_cell(t.label)} | {t.size} | {t.score:.2f} | {t.velocity:.2f} | "
            f"{t.growth:+.1f} | {t.first_seen[:4]} | "
            f"{_cell(', '.join(t.top_keywords))} |"
        )
    out.append("")
    return out


def _indicator(name: str, lab: Mapping[str, str]) -> str:
    """The heading for one indicator, in the report's language.

    Falls back to the internal name so a newly added indicator still renders — visibly
    untranslated, which is the failure mode to prefer over a missing section.
    """
    return lab.get(f"ind_{name}", name)


def _indicator_note(name: str, lab: Mapping[str, str]) -> list[str]:
    """What an indicator selects and what it ranks by, in the report's language.

    Under the heading and above the table: a reader meeting "post-peak persistence" needs
    the rule before the numbers mean anything, and the rule is not recoverable from the
    column header. Absent for an indicator with no note, rather than a placeholder.
    """
    note = lab.get(f"ind_{name}_note")
    return [note, ""] if note else []


def _l1_note(facts: "ReportFacts", lab: Mapping[str, str]) -> str:
    """The level-1 note, naming the cohorts the share columns were read at.

    Only when they differ from the reported span, and silence would be the dangerous
    option: a cohort can exist without the taxonomy existing for it, and then every field
    reads as having lost exactly the share it started with.
    """
    note = lab["l1_note"]
    measured = facts.classified_years
    if not measured or not facts.years:
        return note
    if (measured[0], measured[-1]) == (facts.years[0], facts.years[-1]):
        return note
    return f"{note} {lab['l1_measured']} {measured[0]}–{measured[-1]}."


def _top(rows: Sequence, top_rows: int | None) -> Sequence:
    """The first ``top_rows`` of a list, or all of it."""
    return rows if top_rows is None else rows[:top_rows]


def render_markdown(
    facts: ReportFacts,
    *,
    lang: str = "en",
    title: str | None = None,
    top_rows: int | None = None,
    indicator_rows: int | None = None,
    figures: Mapping[str, Sequence[str]] | None = None,
    labels: Mapping[str, str] | None = None,
    page_break_areas: bool = False,
) -> str:
    """Render the facts as a markdown report in ``lang`` (``"en"`` or ``"uk"``).

    ``top_rows`` truncates each topic and indicator table for reading. The facts object
    deliberately holds more than the tables print: it is the substrate the figures are
    drawn from, and a scatter over the top ten topics per subject area is a ranked sample
    small enough to mislead. Storing wide and printing narrow keeps both honest without
    running the report twice.

    ``page_break_areas`` starts every subject area of section 3 on its own page. Section
    3 alone, because it is the only one a reader navigates rather than reads: it repeats a
    table for each of a hundred and sixty areas, and with them running into one another
    looking one up means finding a heading mid-page. Sections 4 and 5 are a handful of
    tables read in order, and breaking those would only add blank paper.

    ``labels`` replaces the wording without touching the structure. The arithmetic in this
    module is domain-agnostic — it counts records in cohorts and fits rates — but every
    string in :data:`LABELS` says "publication" and names WoS and Scopus, so a patent
    register rendered through them reads as a corpus of papers. A caller supplies the
    vocabulary of its own domain (see ``horizon.patents.vocabulary``) and gets the same
    tables under the right nouns. ``None`` keeps :data:`LABELS`, so existing callers are
    unaffected; a partial mapping is completed from the base, so an overlay names only the
    keys that actually differ and cannot silently drop the rest.
    """
    lab = LABELS[lang] if labels is None else {**LABELS[lang], **labels}
    span = f"{facts.years[0]}–{facts.years[-1]}" if facts.years else "—"
    unclassified = (
        f" ({facts.unclassified} {lab['unclassified']}, "
        f"{facts.unclassified / facts.records:.0%})"
        if facts.records
        else ""
    )
    out: list[str] = [
        f"# {title or lab['title']}",
        "",
        f"**{lab['generated']}:** {facts.records}{unclassified}  ",
        f"**{lab['span']}:** {span}",
        "",
        # Every line prefixed, not just the first: a caveat that carries footnotes is
        # several paragraphs, and marking only the opening one drops the rest out of the
        # quote and into the body, where they read as part of the report's own text.
        "\n".join(f"> {line}" if line else ">" for line in lab["caveat"].split("\n")),
        "",
        f"## {lab['l1']}",
        "",
        _l1_note(facts, lab),
        "",
        f"| {lab['field']} | {lab['first']} | {lab['last']} | {lab['delta']} "
        f"| {lab['slope']} | {lab['n']} |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for s in facts.fields:
        out.append(
            f"| {s.name} | {_pct(s.share_first)} | {_pct(s.share_last)} "
            f"| {s.share_delta:+.1%} | {s.slope:+.1f} | {s.total} |"
        )
    out += ["", f"## {lab['l2']}", "", lab["l2_note"], ""]
    out += [
        f"| {lab['pair']} | {lab['slope']} | {lab['first']} | {lab['last']} "
        f"| {lab['n']} |",
        "|---|--:|--:|--:|--:|",
    ]
    for s in facts.pairs:
        first = s.counts[0][1] if s.counts else 0
        last = s.counts[-1][1] if s.counts else 0
        out.append(f"| {s.name} | {s.slope:+.1f} | {first} | {last} | {s.total} |")

    out += ["", f"## {lab['l3']}", "", lab["trend_note"], ""]
    if not facts.field_trends:
        out += [lab["none"], ""]
    for name in sorted(facts.field_trends):
        if page_break_areas:
            out += list(PAGE_BREAK)
        out += [f"### {name}", ""] + _trend_rows(
            _top(facts.field_trends[name], top_rows), lab
        )

    out += [f"## {lab['l4']}", "", lab["trend_note"], ""]
    views = (
        ("l4_rising", facts.overall),
        ("l4_declining", facts.declining),
        ("l4_largest", facts.largest),
    )
    if not any(rows for _key, rows in views):
        # A bare heading reads as a section that was forgotten rather than one with
        # nothing in it.
        out += [lab["none"], ""]
    for key, rows in views:
        if not rows:
            continue
        out += [f"### {lab[key]}", "", lab[f"{key}_note"], ""]
        out += _trend_rows(_top(rows, top_rows), lab)

    if facts.lenses or facts.meta_lenses:
        out += [f"## {lab['l5']}", "", lab["l5_note"], ""]
        for name in sorted(facts.lenses):
            out += [f"### {_indicator(name, lab)}", ""]
            out += _indicator_note(name, lab)
            out += _trend_rows(
                _top(facts.lenses[name], indicator_rows or top_rows),
                lab,
                lens=BY_NAME.get(name),
            )
        for name in sorted(facts.meta_lenses):
            out += [f"### {_indicator(name, lab)}", ""]
            out += _indicator_note(name, lab)
            out += _meta_rows(
                name, _top(facts.meta_lenses[name], indicator_rows or top_rows), lab
            )
    if facts.compositions:
        out += [f"## {lab['l6']}", "", lab["l6_note"], ""]
        out += _composition_rows(_top(facts.compositions, top_rows), lab)
    if figures:
        out = _place_figures(out, figures)
    return "\n".join(out).rstrip() + "\n"


def _composition_rows(rows, lab) -> list[str]:
    """One row per topic, one column per population, ranked by spread.

    A group too small to rate shows its count and a dash rather than a number: an empty
    cell reads as absent and a ratio over four records reads as measured, and it is
    neither.
    """
    if not rows:
        return []
    names: list[str] = []
    for comp in rows:
        for g in comp.groups:
            if g.name not in names:
                names.append(g.name)
    head = f"| {lab['trend']} | {lab['size']} | " + " | ".join(names)
    out = [head + f" | {lab['spread']} |", "|---|--:|" + "--:|" * (len(names) + 1)]
    for comp in rows:
        by_name = {g.name: g for g in comp.groups}
        cells = []
        for name in names:
            g = by_name.get(name)
            if g is None:
                cells.append("—")
            elif g.velocity is None:
                # "new" and "too thin to rate" are different states and a single dash
                # would merge them: the first is a finding about the topic, the second
                # is a fact about the sample.
                cells.append(f"{g.size} · {lab['new'] if g.is_new else '—'}")
            else:
                cells.append(f"{g.size} · {g.velocity:.2f}")
        spread = f"{comp.divergence:.2f}" if comp.divergence is not None else "—"
        out.append(
            f"| {comp.label} | {comp.size} | " + " | ".join(cells) + f" | {spread} |"
        )
    return out + [""]


def _place_figures(lines: list[str], figures: Mapping[str, Sequence[str]]) -> list[str]:
    """Put each figure at the end of the section it illustrates.

    After the table rather than before it: the numbers are the finding and the shape is
    the argument for it, in that order. Keyed by the number the heading opens with, so
    rewording a heading does not silently strand a figure.
    """
    out: list[str] = []
    pending: list[str] = []
    for line in lines:
        if line.startswith("## "):
            out += pending
            found = list(figures.get(line[3:].split(".", 1)[0].strip(), ()))
            # A blank line between them, because pandoc turns an image into a captioned
            # figure only when it is alone in its paragraph. Written adjacent, two
            # references became one paragraph holding two inline images and no captions
            # at all — six of the seven figures were silently caption-less, and the one
            # that stood alone showed a doubled number.
            pending = []
            for ref in found:
                pending += [ref, ""]
            if pending:
                pending = ["", *pending]
        out.append(line)
    return out + pending


#: Record-side indicator -> (label key, how to render its ranking quantity). A table
#: rather than a chain of string comparisons with a trailing ``else``: that shape had a
#: silent default, so renaming the indicators sent all three through it and every one of
#: them printed the script counts instead of what it ranks on. An unknown name now says
#: so instead of rendering a plausible wrong column.
_META_COLUMNS: Mapping[str, tuple[str, Callable[[MetaFacts], str]]] = {
    "journal-dispersion": ("venues", lambda f: f"{f.venue_spread:.2f}×"),
    "new-authors": ("newshare", lambda f: f"{f.new_author_share:.0%}"),
    "script-distribution": ("scripts", lambda f: f"{f.cyrillic} / {f.latin}"),
}


def _meta_rows(
    name: str, rows: Sequence[MetaFacts], lab: Mapping[str, str]
) -> list[str]:
    """A markdown table for one record-side indicator, with the column it ranks on."""
    if not rows:
        return [lab["none"], ""]
    if name not in _META_COLUMNS:
        raise ValueError(f"no column defined for record-side indicator {name!r}")
    key, value = _META_COLUMNS[name]
    col = lab[key]
    out = [f"| {lab['trend']} | {lab['size']} | {col} |", "|---|--:|--:|"]
    out += [f"| {f.label} | {f.size} | {value(f)} |" for f in rows]
    out.append("")
    return out


def render_json(facts: ReportFacts) -> str:
    """The facts as indented JSON, for the companion ``-facts.json`` file."""
    return json.dumps(facts.to_dict(), ensure_ascii=False, indent=2) + "\n"
