"""Patent wording for the report, as an overlay on the bibliographic labels.

The report's arithmetic is domain-agnostic — it counts records in cohorts and fits rates —
but its *strings* are not: every one of them says "publication", and the caveat names WoS
and Scopus. Rendered through them a patent register reads as a corpus of papers, which is
wrong in a way a reader cannot see, because the tables are otherwise correct.

So this is an overlay rather than a second label table. It names only the keys whose
wording actually differs, ``render_markdown`` completes the rest from the base, and a key
added to the report in future appears here in its bibliographic form rather than vanishing.
That is the failure mode worth designing against: a forked copy of all 59 strings would
drift silently, and the drift would show up as a missing heading in one language only.

Three substantive changes ride along with the nouns, and none is cosmetic:

* **The caveat** names this register and its cut date. The bibliographic caveat exists to
  stop a reader treating an export as world science; the patent one has to stop them
  treating a national register as world invention, which is a stronger claim to guard.
* **The per-year column** is footnoted where the bibliographic report is not, because on a
  contracting register the least-squares slope is negative for 32 of 34 technologies and
  mostly measures the common contraction. Left unexplained it reads as every technology
  declining at once.
* **The citation labels** say what they rest on. Only ~2% of the register has a citation
  count at all, so an indicator ranked on the citation percentile is ranking that fraction
  and the report has to say which.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Only the keys whose wording differs from the bibliographic report. Everything else is
#: completed from :data:`horizon.report.LABELS`.
PATENT_LABELS: Mapping[str, Mapping[str, str]] = {
    "en": {
        "title": "Patent register report",
        "generated": "Patents",
        "unclassified": "without a technology class",
        "l1": "1. Technology-field development",
        "l1_note": (
            "Share of the register, and patents per year, by the WIPO technology fields "
            "assigned to each grant. No clustering involved."
        ),
        "l2": "2. Technology-field co-occurrence",
        "l2_note": (
            "Patents classified into both fields, ranked by least-squares slope. A "
            "patent carries more than one field in 46% of the register, so these pairs "
            "are the seams where technologies are being combined."
        ),
        "l3": "3. Topics within each technology field",
        "l4": "4. Register-wide topics",
        "l4_declining_note": (
            "The same topics ranked by relative growth, lowest first, among those with "
            "at least 200 patents. A topic here can still be granting patents every "
            "year — it is losing share, not necessarily volume."
        ),
        "field": "technology field",
        "pair": "pair of fields",
        "n": "patents",
        "size": "patents",
        "growth": "patents/year",
        "slope": "per year†",
        "attention": "citation percentile‡",
        "venues": "applicants per patent",
        "newshare": "share of new inventors",
        "caveat": (
            "All figures are shares of this register — grants issued in Ukraine, cut on "
            "2026-08-12 — and not of world invention. Cohorts are priority years: the "
            "date the invention was claimed, not the date it was granted.\n\n"
            "**The reported range is not a stationary period.** It opens at 3 083 grants "
            "a year (2012–2013 mean), falls 20.2% between the 2013 and 2014 cohorts, "
            "holds near that level to 2019, then falls a further 40.8% into 2020. Rates "
            "fitted across 2012–2020 are therefore fitted across a step and two levels, "
            "not a trend, and the per-year column should be read as a summary of that "
            "shape rather than as a rate of change.\n\n"
            "**The range ends before February 2022.** The 2021 and 2022 cohorts are "
            "excluded as incomplete — 72% and 48% of their eventual grants are estimated "
            "to have issued — and they are also the only cohorts whose priority dates "
            "fall in the full-scale war. This register therefore cannot measure the "
            "war's effect on invention: the two cohorts that would carry it are the two "
            "the completeness guard removes, and in a register of grants a fall in the "
            "count confounds fewer applications, an office granting more slowly, and the "
            "grant lag itself. The office's annual output halved after 2021 (2 265 "
            "grants in 2021 against 1 153 in 2023) and the resident median lag rose from "
            "1.55 years to 2.51, so the second cause is present and is not separable "
            "from the first inside this data.\n\n"
            "† Per year is a least-squares slope in patents/year. The register contracts "
            "sharply across the reported range (3 057 grants in 2012 against 1 141 in "
            "2020), so this column is negative for almost every field and largely "
            "measures that common contraction. Share of cohort is the column that "
            "separates fields from one another.\n\n"
            "‡ Citation percentiles rest on the ~2% of the register with an "
            "international sibling, which is the only part of it a citation index "
            "observes. The rest are unmeasured, not uncited, and are excluded from these "
            "rankings rather than ranked as zero."
        ),
        "l4_rising_note": (
            "Score is velocity x patents x recency x citation percentile, so this "
            "ranking leans toward topics that are both large and recently growing."
        ),
        "l4_largest_note": (
            "Ranked by patents alone, the one criterion with nothing composite in it. It "
            "exists because the two rankings above are both about change, and a large "
            "topic moving at the register's rate appears in neither."
        ),
        "trend_note": (
            "Topics are clusters of semantically similar patents, ranked by score. The "
            "two growth measures answer different questions and are expected to "
            "disagree.\n\n"
            "**Relative growth** is the ratio of the topic's patents over the most recent "
            "period to its patents over a preceding period of equal length, normalised by "
            "the same ratio for the whole register. A value of 1.0 matches the register's "
            "rate; above 1.0 the topic is outgrowing it, below it is falling behind. On a "
            "register that contracts this is the measure that still separates topics from "
            "one another.\n\n"
            "**Patents/year** is the slope of a least-squares fit to the topic's annual "
            "counts over its whole life. It is absolute and is not normalised, so on this "
            "register it is negative for most topics and largely measures the common "
            "contraction."
        ),
        "ind_declining": "Fading patenting activity",
        "ind_declining_note": (
            "Topics granted fewer patents than before, ranked by size weighted by the "
            "shortfall in relative growth. Only topics at or above the median size "
            "qualify: below it a decline has no established body of work to be a decline "
            "from."
        ),
        "ind_high-impact-low-output": "High citation impact on few patents",
        "ind_high-output-low-impact": "Many patents at low citation impact",
        "ind_high-impact-low-output_note": (
            "Topics below the median in patents, ranked by citation percentile highest "
            "first. Rests on the ~2% of the register with an international sibling, so it "
            "ranks that fraction and not the register."
        ),
        "ind_high-output-low-impact_note": (
            "The other half of the same split: topics above the median in patents, ranked "
            "by citation percentile lowest first. The two indicators divide one population, "
            "so a topic falls in exactly one."
        ),
        "ind_sleeping-beauty_note": (
            "Topics whose peak year came at least two years after their first patent and "
            "which are still growing against the register, ranked by the length of that "
            "latency. Size is a separate column, so a long latency and a large body of "
            "work stay separate quantities."
        ),
        "ind_sustained_note": (
            "Topics that held their level after the peak rather than producing a single "
            "burst, ranked by the number of post-peak years in which patents stayed at "
            "least half the peak."
        ),
        "ind_journal-dispersion_note": (
            "Applicants per patent in the topic's later half over the same figure in its "
            "earlier half. Above 1.0 the topic is spreading across applicants faster than "
            "its own volume explains — concentration giving way to a broader field."
        ),
        "ind_new-authors_note": (
            "The share of inventors in the topic's later half who are absent from its "
            "earlier half. Names are compared as written, so one person recorded two ways "
            "counts as two and the figure is an upper bound."
        ),
        "ind_script-distribution_note": (
            "Not reported for this register. Filing with the Ukrainian office requires "
            "Ukrainian, so a patent's script is set by the filing rules rather than by "
            "its subject and can say nothing about topics."
        ),
        "ind_journal-dispersion": "Applicant dispersion",
        "ind_new-authors": "Turnover among inventors",
    },
    "uk": {
        "title": "Звіт по патентному реєстру",
        "generated": "Патентів",
        "unclassified": "без технологічного класу",
        "l1": "1. Розвиток технологічних напрямів",
        "l1_note": (
            "Частка реєстру і патентів на рік за технологічними напрямами WIPO, "
            "призначеними кожному патенту. Кластеризація не задіяна."
        ),
        "l2": "2. Співпоява технологічних напрямів",
        "l2_note": (
            "Патенти, віднесені до обох напрямів, за нахилом найменших квадратів. "
            "У 46% реєстру патент має більше ніж один напрям, тож ці пари є тими швами, "
            "де технології поєднуються."
        ),
        "l3": "3. Теми в межах технологічних напрямів",
        "l4": "4. Теми по всьому реєстру",
        "l4_declining_note": (
            "Ті самі теми, впорядковані за відносним зростанням від найнижчого, серед "
            "тих, що мають щонайменше 200 патентів. Тема звідси може й далі отримувати "
            "патенти щороку — вона втрачає частку, а не обов'язково обсяг."
        ),
        "field": "технологічний напрям",
        "pair": "пара напрямів",
        "n": "патентів",
        "size": "патентів",
        "growth": "патентів/рік",
        "slope": "на рік†",
        "attention": "перцентиль цитувань‡",
        "venues": "заявників на патент",
        "newshare": "частка нових винахідників",
        "caveat": (
            "Усі цифри є частками цього реєстру — патентів, виданих в Україні, зріз "
            "станом на 12.08.2026 — а не світового винахідництва. Когорти — це роки "
            "пріоритету: дата, на яку заявлено винахід, а не дата видачі патенту.\n\n"
            "**Звітний діапазон не є стаціонарним періодом.** Він починається з 3 083 "
            "виданих патентів на рік (середнє за 2012–2013), падає на 20,2% між "
            "когортами 2013 і 2014, тримається близько цього рівня до 2019, а далі "
            "падає ще на 40,8% до 2020. Тому темпи, підігнані на 2012–2020, підігнано "
            "через сходинку і два рівні, а не через тренд, і колонку «на рік» слід "
            "читати як підсумок цієї форми, а не як швидкість зміни.\n\n"
            "**Діапазон завершується до лютого 2022 року.** Когорти 2021 і 2022 "
            "виключено як неповні — за оцінкою, видано 72% і 48% їхніх майбутніх "
            "патентів — і вони ж є єдиними когортами, чиї дати пріоритету припадають на "
            "повномасштабну війну. Тож цей реєстр не може виміряти вплив війни на "
            "винахідництво: дві когорти, які його несли б, — це саме ті дві, які "
            "прибирає запобіжник повноти, а в реєстрі виданих патентів падіння лічби "
            "змішує меншу кількість заявок, повільнішу видачу і сам лаг. Річний виробіток "
            "відомства скоротився удвічі після 2021 року (2 265 патентів у 2021 проти "
            "1 153 у 2023), а медіанний лаг резидентів зріс з 1,55 до 2,51 року — тобто "
            "друга причина присутня і всередині цих даних не відокремлюється від "
            "першої.\n\n"
            "† «На рік» — це нахил найменших квадратів у патентах на рік. Реєстр різко "
            "стискається впродовж звітного діапазону (3 057 виданих патентів у 2012 "
            "проти 1 141 у 2020), тож ця колонка є від'ємною майже для кожного напряму і "
            "значною мірою вимірює саме це спільне стискання. Колонкою, яка відрізняє "
            "напрями один від одного, є частка когорти.\n\n"
            "‡ Перцентилі цитувань спираються на ті ~2% реєстру, що мають закордонний "
            "відповідник, — це єдина його частина, яку бачить індекс цитувань. Решта є "
            "невиміряною, а не нецитованою, і виключена з цих впорядкувань, а не "
            "ранжована як нуль."
        ),
        "l4_rising_note": (
            "Оцінка є добутком відносного зростання, обсягу, свіжості та перцентиля "
            "цитувань, тож це впорядкування схиляється до тем, які водночас великі й "
            "нещодавно зростали."
        ),
        "l4_largest_note": (
            "Впорядкування за кількістю патентів — єдиний критерій без жодної "
            "композитності. Він потрібен, бо два попередні стосуються зміни, а велика "
            "тема, що рухається в темпі реєстру, не потрапляє в жоден із них."
        ),
        "trend_note": (
            "Теми — це кластери семантично подібних патентів, упорядковані за оцінкою. "
            "Дві міри зростання відповідають на різні питання, і розбіжність між ними "
            "очікувана.\n\n"
            "**Відносне зростання** — це відношення кількості патентів теми за останній "
            "період до кількості за попередній період тієї самої тривалості, "
            "нормалізоване на аналогічне відношення для всього реєстру. Значення 1.0 "
            "відповідає темпові реєстру, більші за 1.0 — випередженню, менші — "
            "відставанню. На реєстрі, що стискається, саме ця міра ще розрізняє теми "
            "між собою.\n\n"
            "**Патентів/рік** — це коефіцієнт нахилу лінійної регресії річних "
            "кількостей теми за весь період її існування. Показник абсолютний і не "
            "нормалізується, тож на цьому реєстрі він від'ємний для більшості тем і "
            "значною мірою вимірює спільне стискання."
        ),
        "ind_declining": "Згасання патентної активності",
        "ind_declining_note": (
            "Теми, що отримують менше патентів, ніж раніше, упорядковані за обсягом, "
            "зваженим на нестачу відносного зростання. Проходять лише теми не менші за "
            "медіанну: нижче неї спад не має усталеного масиву робіт, від якого був би "
            "спадом."
        ),
        "ind_high-impact-low-output": "Високий цитувальний вплив за малої кількості патентів",
        "ind_high-output-low-impact": "Велика кількість патентів за низького цитувального впливу",
        "ind_high-impact-low-output_note": (
            "Теми, менші за медіанний обсяг, упорядковані за перцентилем цитувань від "
            "найвищого. Спирається на ті ~2% реєстру, що мають закордонний відповідник, "
            "тож упорядковує саме цю частку, а не реєстр."
        ),
        "ind_high-output-low-impact_note": (
            "Друга половина того самого поділу: теми, більші за медіанний обсяг, "
            "упорядковані за перцентилем цитувань від найнижчого. Два індикатори ділять "
            "одну сукупність, тож тема потрапляє рівно до одного з них."
        ),
        "ind_sleeping-beauty_note": (
            "Теми, чий піковий рік настав щонайменше через два роки після першого "
            "патенту і які досі зростають відносно реєстру, упорядковані за тривалістю "
            "цього латентного періоду. Обсяг є окремою колонкою, тож довга латентність і "
            "великий масив робіт лишаються окремими величинами."
        ),
        "ind_sustained_note": (
            "Теми, що втримали рівень після піку, а не дали разовий сплеск, упорядковані "
            "за кількістю років після піку, у які кількість патентів становила "
            "щонайменше половину пікової."
        ),
        "ind_journal-dispersion_note": (
            "Заявників на патент у пізнішій половині життя теми, поділене на той самий "
            "показник у ранішій. Вище 1.0 тема розходиться заявниками швидше, ніж "
            "пояснює її власний обсяг, — концентрація поступається ширшому полю."
        ),
        "ind_new-authors_note": (
            "Частка винахідників пізнішої половини, відсутніх у ранішій. Імена "
            "порівнюються як записані, тож ім'я, написане двома способами, рахується за "
            "двох людей, і величина є верхньою межею."
        ),
        "ind_script-distribution_note": (
            "Для цього реєстру не наводиться. Подання до українського відомства "
            "вимагає української мови, тож система письма патенту задана правилами "
            "подання, а не його предметом, і нічого не може сказати про теми."
        ),
        "ind_journal-dispersion": "Розсіювання за заявниками",
        "ind_new-authors": "Оновлення складу винахідників",
    },
}


def labels(lang: str) -> Mapping[str, str]:
    """The patent overlay for ``lang``, to hand to ``render_markdown(labels=…)``.

    Raises ``KeyError`` on an unknown language rather than falling back to English: a
    report silently rendered in the wrong language is worse than one that fails to build.
    """
    return PATENT_LABELS[lang]


#: Figure titles and axis labels. Separate from :data:`PATENT_LABELS` because
#: ``horizon.figures`` keeps its own text table — a table header is a noun, a figure axis
#: is a phrase — and the two are handed to different functions.
PATENT_FIGURE_TEXT: Mapping[str, Mapping[str, str]] = {
    "en": {
        "f1_title": "Register-wide topics, year by year",
        "f1_y": "patents in the topic",
        "f5_title": "Technology field's share of the register",
        "f5_y": "share of patents that year",
        "f6_title": "Change in share, first to last classified cohort",
        "f3_title": "Output against citation percentile, by topic",
        "f3_x": "patents in the topic",
        "f4_y": "patents per year",
        "f4_x": "relative growth (against the register)",
        "f7_title": "Technology-field co-classification",
        "reported": "topics the report names (top per technology field)",
        "parity": "keeping pace",
    },
    "uk": {
        "f1_title": "Теми по всьому реєстру, за роками",
        "f1_y": "патентів у темі",
        "f5_title": "Частка технологічного напряму в реєстрі",
        "f5_y": "частка патентів року",
        "f6_title": "Зміна частки, від першої до останньої когорти з класифікацією",
        "f3_title": "Обсяг і перцентиль цитувань, за темами",
        "f3_x": "патентів у темі",
        "f4_y": "патентів на рік",
        "f4_x": "відносне зростання (проти реєстру)",
        "f7_title": "Спільна класифікація технологічних напрямів",
        "reported": "теми, які звіт називає (найбільші в кожному напрямі)",
        "parity": "нарівні з реєстром",
    },
}


def figure_text(lang: str) -> Mapping[str, str]:
    """The patent figure overlay for ``lang``, to hand to ``figures.draw(text=…)``.

    The same mapping must go to ``figures.references``, or a caption will describe
    something other than the picture above it.
    """
    return PATENT_FIGURE_TEXT[lang]


#: Indicators this domain never reports, and why. Structural rather than empirical: the
#: generic guard in ``horizon.metadata_lenses.discriminates`` drops a lens whose values
#: happen to be flat, which is the right check on a corpus that could have varied. Here
#: the register could not. Filing with the Ukrainian office requires Ukrainian, so the
#: script of a patent is set by the filing rules and not by its subject — it carries no
#: information about topics, and would carry none even if a handful of records varied.
#: (Ten do have an English abstract; all ten have a Ukrainian title, so even the data
#: agrees.) A guard that reasons from the data would re-enable the indicator the moment a
#: slice happened to be mixed, and print a ranking of an administrative requirement.
SUPPRESSED_INDICATORS = frozenset({"script-distribution"})
