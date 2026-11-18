"""The report's figures: what is drawn, where it belongs, and in which language.

In the package rather than in a script because the report writes the reference and this
module draws the picture — a catalogue in two places would let a caption outlive the
figure it names, and nothing would notice. ``scripts/build_figures.py`` is a thin way to
redraw from an existing ``facts.json`` without re-running the report.

matplotlib is imported inside the drawing functions, like torch and transformers
elsewhere: the package must import without it, and a report without matplotlib loses its
figures rather than failing.

Seven figures, numbered and named in the order the report shows them:

* **f1** — subject-area share over time. The tables give first and last; the curve shows
  whether the move was a slope or a step.
* **f2** — change in share across every area at once, which no table ordering shows.
* **f3** — the co-classification matrix, of which the pair table is the top of one ranking.
* **f4** — how the leading corpus-wide topics moved year by year. A rank in a table says
  which topics matter; the curves say what happened to them.
* **f5** — the two growth measures against each other. They disagree by construction, and
  the disagreement is easier to accept as a shape than as a sentence.
* **f6** — all eight indicators at once, so the indicator section reads as one page.
* **f7** — output against citation percentile, the plane the two quadrant indicators cut
  in half. The median lines are drawn, so those indicator tables become regions.

The first three characterise the corpus; the last four are the analysis over it. The
drawing functions keep their original names — after what each draws rather than where it
lands — so renumbering the report never silently renames the code.

f5 and f7 are drawn on the topics the report kept, not on every cluster found: the facts
hold the top ``--top-trends`` per subject area. That is a ranked sample, not a random one,
so the scatters show where the reported topics sit and are not a distribution of all
clustering output.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from horizon.fields import _slope

#: Axis and title text, per language. Kept beside the code that draws rather than in
#: ``report.LABELS``: those are table headers, and a figure needs sentences.
TEXT: Mapping[str, Mapping[str, str]] = {
    "en": {
        "f1_title": "Corpus-wide topics, year by year",
        "f1_y": "publications in the topic",
        "f2_title": "Every indicator on one page",
        "f5_title": "Subject-area share of the corpus",
        "f5_y": "share of publications that year",
        "f6_title": "Change in share, first to last classified cohort",
        "f6_x": "percentage points",
        "f3_title": "Output against citation percentile, by topic",
        "f3_x": "publications in the topic",
        "f3_y": "citation percentile",
        "f4_title": "The two growth measures disagree by construction",
        "f4_x": "relative growth (against the corpus)",
        "f4_y": "publications per year",
        "f7_title": "Subject-area co-classification",
        "f8_title": "Words gaining ground in the topics that are gaining ground",
        "f8_x": "1 − correlation of yearly shares",
        "reported": "topics the report names (top per subject area)",
        "median": "median",
        "parity": "keeping pace",
        "flat": "flat",
        "year": "year",
        "partial": "incomplete cohorts (counts only)",
    },
    "uk": {
        "f1_title": "Теми по всьому корпусу, за роками",
        "f1_y": "публікацій у темі",
        "f2_title": "Усі індикатори на одній сторінці",
        "f5_title": "Частка предметної області в корпусі",
        "f5_y": "частка публікацій року",
        "f6_title": "Зміна частки, від першого до останнього року з класифікацією",
        "f6_x": "відсоткові пункти",
        "f3_title": "Обсяг і перцентиль цитувань, за темами",
        "f3_x": "публікацій у темі",
        "f3_y": "перцентиль цитувань",
        "f4_title": "Дві міри зростання розходяться за побудовою",
        "f4_x": "відносне зростання (проти корпусу)",
        "f4_y": "публікацій на рік",
        "f7_title": "Спільна класифікація предметних областей",
        "f8_title": "Слова, що набирають вагу в темах, які набирають вагу",
        "f8_x": "1 − кореляція річних часток",
        "reported": "теми, які звіт називає (найкращі в кожній області)",
        "median": "медіана",
        "parity": "нарівні з корпусом",
        "flat": "без зміни",
        "year": "рік",
        "partial": "неповні когорти (лише лічба)",
    },
}

#: Muted, print-safe, and readable in greyscale by lightness order. Not a rainbow: the
#: series are ranked, so the palette runs cool-to-warm rather than categorical.
SERIES = [
    "#1F4E79",
    "#2E7D9A",
    "#4F9D8A",
    "#8FAE5B",
    "#C79A3E",
    "#C1663E",
    "#A6402E",
    "#7B4B6B",
]
UP, DOWN, INK, MUTED = "#1B6E4B", "#A6402E", "#151A21", "#5D6773"


def _style(plt) -> None:
    """One restrained house style, applied once."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",  # carries Cyrillic; matplotlib ships it
            "font.size": 8,
            "axes.edgecolor": "#C9CFD6",
            "axes.labelcolor": MUTED,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#E8EBEF",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _classified_years(facts: Mapping[str, Any]) -> list[int]:
    """Cohorts in which the taxonomy exists, derived when the facts predate the field.

    Not a fallback to the corpus years: those differ exactly when the last cohort carries
    no subject areas, and a curve drawn to that cohort would drop every area to zero in
    its final point.
    """
    stored = facts.get("classified_years")
    if stored:
        return [int(y) for y in stored]
    return sorted(
        {int(y) for f in facts["fields"] for y, n in f["counts"].items() if n}
    )


def _topics(facts: Mapping[str, Any]) -> list[dict]:
    """Every per-area topic, flattened. The population with a shape."""
    return [t for rows in facts["field_trends"].values() for t in rows]


#: Indicator -> (label key, value to rank by). The same eight the report prints, in the
#: order they appear there. A table rather than conditionals, so a renamed indicator
#: fails loudly instead of dropping into a default and plotting the wrong quantity.
PANEL: tuple[tuple[str, str, Any], ...] = (
    ("declining", "decay", lambda r: r["size"] * max(0.0, 1 - r["velocity"])),
    ("high-impact-low-output", "attention", lambda r: r["attention_pctile"]),
    ("high-output-low-impact", "attention", lambda r: r["attention_pctile"]),
    ("sleeping-beauty", "dormancy", lambda r: _dormancy(r)),
    ("sustained", "sustain", lambda r: _sustain(r)),
    ("journal-dispersion", "venues", lambda r: r["venue_spread"]),
    ("new-authors", "newshare", lambda r: r["new_author_share"]),
    # None marks the one indicator a single bar cannot carry: it ranks by how one-sided
    # a topic is, so its leading rows are 100% one script and a share bar renders as an
    # empty panel. Drawn stacked instead, where the imbalance is the whole picture.
    ("script-distribution", "scripts", None),
)


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


def _integer_years(ax) -> None:
    """Whole years on the x axis. A four-year span otherwise ticks at 2015.5."""
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))


def _short(label: str, width: int = 30) -> str:
    """Topic labels run long; the axis has room for about thirty characters."""
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"


def f1_topic_trajectories(facts, txt, path) -> None:
    """The named corpus-wide topics, publication counts year by year.

    A rank in a table says which topics matter. This says what happened to them, and the
    shapes differ in a way no single growth number carries: a topic that stepped up once
    and held is not a topic that has been climbing since the first cohort.
    """
    import matplotlib.pyplot as plt

    topics = [t for t in facts["overall"] if t.get("cohort_counts")][: len(SERIES)]
    if not topics:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for colour, t in zip(SERIES, topics):
        counts = {int(y): n for y, n in t["cohort_counts"].items()}
        xs = sorted(counts)
        ax.plot(
            xs,
            [counts[y] for y in xs],
            color=colour,
            linewidth=1.8,
            label=_short(t["label"], 34),
        )
        ax.plot(xs[-1:], [counts[xs[-1]]], "o", color=colour, markersize=3.5)
    _mark_unfitted(ax, facts, txt)
    ax.set_title(txt["f1_title"])
    ax.set_ylabel(txt["f1_y"])
    ax.set_xlabel(txt["year"])
    _integer_years(ax)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def f2_indicator_panel(facts, txt, path) -> None:
    """All eight indicators as small multiples — section 5 on one page.

    Each panel keeps its own ranking quantity, because that is what the indicator means;
    a shared axis would put citation percentiles and years-held on one scale and say
    something false about both.
    """
    import matplotlib.pyplot as plt

    from horizon.report import LABELS

    lab = LABELS["uk" if txt is TEXT["uk"] else "en"]
    present = [
        (name, key, fn)
        for name, key, fn in PANEL
        if facts["lenses"].get(name) or facts["meta_lenses"].get(name)
    ]
    if not present:
        return
    cols = 2
    rows = (len(present) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 1.55 * rows + 0.9))
    for ax, (name, key, fn) in zip(axes.ravel(), present):
        data = (facts["lenses"].get(name) or facts["meta_lenses"].get(name))[:5]
        labels = [_short(r["label"]) for r in data][::-1]
        if fn is None:
            rows_ = data[::-1]
            totals = [max(r["cyrillic"] + r["latin"], 1) for r in rows_]
            cyr = [r["cyrillic"] / t for r, t in zip(rows_, totals)]
            ax.barh(labels, cyr, color=SERIES[6], height=0.62)
            ax.barh(
                labels, [1 - c for c in cyr], left=cyr, color=SERIES[1], height=0.62
            )
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        else:
            ax.barh(
                labels, [float(fn(r)) for r in data][::-1], color=SERIES[0], height=0.62
            )
        # Wrapped: the formal indicator names run to sixty characters, and a small
        # multiple is about four inches wide.
        ax.set_title(
            "\n".join(textwrap.wrap(lab[f"ind_{name}"], 46)),
            fontsize=8,
            loc="left",
        )
        ax.set_xlabel(lab[key], fontsize=6.5)
        ax.tick_params(axis="y", length=0, labelsize=6.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(present) :]:
        ax.set_visible(False)
    fig.suptitle(
        txt["f2_title"],
        fontsize=9.5,
        fontweight="semibold",
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _mark_unfitted(ax, facts, txt) -> None:
    """Shade the cohorts the shares and slopes were not fitted over.

    Only the counts cover those years. A cohort still filling up swings hard for reasons
    that have nothing to do with its subject — it over-represents whatever part of the
    corpus resolves fastest — and on the patent register the last two cohorts move several
    percentage points on that alone. Unmarked, a reader takes the swing for a finding, and
    a figure is the part of a report that gets read without its caveat.
    """
    fit = [int(y) for y in facts.get("fit_years") or []]
    if not fit:
        return
    shown = [int(y) for y in facts.get("years") or []]
    excluded = [y for y in shown if y > max(fit)]
    if not excluded:
        return
    ax.axvspan(
        max(fit) + 0.5,
        max(excluded) + 0.4,
        color="#000000",
        alpha=0.06,
        zorder=0,
        label=txt.get("partial", "incomplete cohorts"),
    )


def f5_share_over_time(facts, txt, path) -> None:
    """Share of each leading area, year by year."""
    import matplotlib.pyplot as plt

    if "year_totals" not in facts:
        # No silent fallback: the per-area counts sum past the cohort they came from,
        # so a share drawn from them would be wrong and would look right.
        raise SystemExit(
            "error: this facts.json predates 'year_totals' — regenerate the report "
            "before drawing figures."
        )
    totals = {int(y): n for y, n in facts["year_totals"].items()}
    classified = _classified_years(facts)
    ranked = sorted(facts["fields"], key=lambda f: -abs(f["share_delta"]))[
        : len(SERIES)
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for colour, f in zip(SERIES, ranked):
        counts = {int(y): n for y, n in f["counts"].items()}
        xs = [y for y in classified if totals.get(y)]
        ys = [counts.get(y, 0) / totals[y] for y in xs]
        ax.plot(xs, ys, color=colour, linewidth=1.8, label=f["name"])
        ax.plot(xs[-1:], ys[-1:], "o", color=colour, markersize=3.5)
    _mark_unfitted(ax, facts, txt)
    ax.set_title(txt["f5_title"])
    ax.set_ylabel(txt["f5_y"])
    ax.set_xlabel(txt["year"])
    _integer_years(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def f6_share_change(facts, txt, path) -> None:
    """Every area's change in share, as a diverging bar."""
    import matplotlib.pyplot as plt

    rows = sorted(facts["fields"], key=lambda f: f["share_delta"])
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.19 * len(rows))))
    values = [f["share_delta"] * 100 for f in rows]
    ax.barh(
        [f["name"] for f in rows],
        values,
        color=[UP if v >= 0 else DOWN for v in values],
        height=0.68,
    )
    ax.axvline(0, color="#C9CFD6", linewidth=0.8)
    ax.set_title(txt["f6_title"])
    ax.set_xlabel(txt["f6_x"])
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def f3_output_vs_impact(facts, txt, path) -> None:
    """The plane the two quadrant indicators split, with the split drawn."""
    import matplotlib.pyplot as plt
    from statistics import median

    rows = [t for t in _topics(facts) if t.get("attention_pctile") is not None]
    if not rows:
        return
    sizes = [t["size"] for t in rows]
    mid = median(sizes)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.scatter(
        sizes,
        [t["attention_pctile"] for t in rows],
        s=9,
        alpha=0.45,
        linewidths=0,
        color=SERIES[0],
    )
    ax.axvline(mid, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate(
        f"{txt['median']} {mid:.0f}",
        (mid, 1.0),
        xytext=(4, -2),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    ax.set_xscale("log")
    ax.set_title(txt["f3_title"])
    ax.set_xlabel(f'{txt["f3_x"]} · {txt["reported"]}')
    ax.set_ylabel(txt["f3_y"])
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def f4_growth_measures(facts, txt, path) -> None:
    """Relative growth against absolute slope, with both neutral lines."""
    import matplotlib.pyplot as plt

    rows = _topics(facts)
    if not rows:
        return
    xs = [t["velocity"] for t in rows]
    ys = [t["growth"] for t in rows]
    colours = [UP if x >= 1.0 else DOWN for x in xs]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.scatter(xs, ys, s=9, alpha=0.45, linewidths=0, c=colours)
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate(
        txt["parity"],
        (1.0, max(ys)),
        xytext=(4, -4),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    ax.set_xscale("log")
    ax.set_title(txt["f4_title"])
    ax.set_xlabel(f'{txt["f4_x"]} · {txt["reported"]}')
    ax.set_ylabel(txt["f4_y"])
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)


def f7_cooccurrence(facts, txt, path) -> None:
    """The co-classification matrix the pair table ranks the top of."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = [f["name"] for f in sorted(facts["fields"], key=lambda f: -f["total"])]
    index = {n: i for i, n in enumerate(names)}
    m = np.zeros((len(names), len(names)))
    for p in facts["pairs"]:
        a, _, b = p["name"].partition(" + ")
        if a in index and b in index:
            m[index[a], index[b]] = m[index[b], index[a]] = p["total"]
    np.fill_diagonal(m, np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(np.sqrt(m), cmap="BuPu", interpolation="nearest")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    ax.set_title(txt["f7_title"])
    ax.tick_params(length=0)
    bar = fig.colorbar(im, ax=ax, shrink=0.7)
    # sqrt only to keep the long tail visible; the ticks stay in publications.
    ticks = bar.get_ticks()
    bar.set_ticks(ticks, labels=[f"{t**2:,.0f}".replace(",", " ") for t in ticks])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=6)
    fig.savefig(path)
    plt.close(fig)


#: Records a word needs across the growing topics before its curve is fitted. Measured
#: on the WoS/Scopus corpus: below this the ranking fills with words carried by one or
#: two topics, whose "rise" is that topic's rise under another name.
VOCAB_MIN_RECORDS = 200

#: Words in the tree. Bounded by the page rather than by the data — 502 words clear
#: :data:`VOCAB_MIN_RECORDS` on the WoS/Scopus corpus. A figure taller than the text block
#: is scaled to fit its height, and then its width, so every extra row shrinks every
#: label: at fifty leaves a 9-point label still prints near 8, at sixty it is under 7.
VOCAB_LEAVES = 50


def _growing_topics(facts: Mapping[str, Any], years: Sequence[int]) -> list[dict]:
    """Topics whose share of the cohort rose, deduplicated across the blocks listing them.

    Share rather than count: this corpus adds records every year, so a count-based filter
    keeps almost everything and selects nothing. Deduplicated because an unblocked
    clustering lists one topic under every field its members carry, and a word would
    otherwise be weighted by its topic's interdisciplinarity.
    """
    totals = {y: facts["year_totals"][str(y)] for y in years}
    seen: set[tuple[str, str]] = set()
    out = []
    for rows in facts["field_trends"].values():
        for t in rows:
            key = (t["label"], t.get("seed_hash", ""))
            if key in seen:
                continue
            seen.add(key)
            cohort = {int(y): n for y, n in (t.get("cohort_counts") or {}).items()}
            counts = [cohort.get(y, 0) for y in years]
            shares = {y: c / totals[y] for c, y in zip(counts, years)}
            if _slope(shares, years) > 0:
                out.append({"counts": counts, "keywords": t.get("top_keywords") or ()})
    return out


def f8_rising_vocabulary(facts, txt, path) -> None:
    """The words gaining ground inside the topics that are gaining ground.

    Built from the topics whose share of the cohort rose, and from the keywords those
    topics carry: a word's curve is the summed cohort counts of the growing topics that
    name it, read as a share of the corpus so the corpus's own growth is divided out.

    **The tree groups by the shape of the rise, not by subject**, and that is a finding
    about the data rather than a preference. The obvious distance — how often two words
    appear in the same topic — is unusable at this granularity: measured on the
    WoS/Scopus corpus, 70% of the pairs among the leading words never share a topic and
    the median Jaccard distance is exactly 1.0, which draws a comb rather than a tree.
    Correlation between the words' yearly share curves does spread (0.003 to 0.688 over
    the same pairs) and answers a question a horizon scan actually asks: which words
    surged together. It does not answer what they mean, so two words with nothing in
    common can share a branch because they rose on one schedule. The heatmap beside the
    tree is there so a reader can see which of the two it is looking at.

    Silent when the inputs are missing: no scipy, too few qualifying words, or a corpus
    with no growing topics all leave the figure undrawn rather than failing the report.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except ImportError:  # pragma: no cover - scipy rides in with scikit-learn
        return

    years = _classified_years(facts)
    if len(years) < 3 or not facts.get("field_trends"):
        return
    totals = {y: facts["year_totals"][str(y)] for y in years}
    series: dict[str, list[int]] = {}
    for topic in _growing_topics(facts, years):
        for word in dict.fromkeys(w.lower().strip() for w in topic["keywords"]):
            if not word:
                continue
            row = series.setdefault(word, [0] * len(years))
            for i, c in enumerate(topic["counts"]):
                row[i] += c
    ranked = sorted(
        (
            (w, c, _slope({y: x / totals[y] for x, y in zip(c, years)}, years))
            for w, c in series.items()
            if sum(c) >= VOCAB_MIN_RECORDS
        ),
        key=lambda r: -r[2],
    )[:VOCAB_LEAVES]
    if len(ranked) < 6:
        return

    shares = np.array(
        [[x / totals[y] for x, y in zip(c, years)] for _w, c, _s in ranked]
    )
    centred = shares - shares.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    if not norms.all():
        return
    unit = centred / norms
    distance = np.clip(1 - unit @ unit.T, 0, 2)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")

    n = len(ranked)
    fig, (left, right) = plt.subplots(
        1,
        2,
        # Rows kept tight for the same reason: height is what the page rations here.
        figsize=(7.8, 0.19 * n + 1.4),
        gridspec_kw={"width_ratios": [0.72, 1.3], "wspace": 0.035},
    )
    drawn = dendrogram(
        tree,
        ax=left,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color=MUTED,
    )
    order = drawn["leaves"]
    left.set_xlabel(txt["f8_x"], fontsize=8)
    # Not inverted: ``orientation="left"`` already puts the leaves against the right
    # edge, which is the edge the heatmap is on. Flipping it parts the tree from the rows
    # it labels and leaves a corridor of white down the middle of the figure.
    for spine in left.spines.values():
        spine.set_visible(False)
    left.tick_params(axis="y", length=0, labelleft=False)
    left.tick_params(axis="x", labelsize=7)

    # Each row scaled to its own peak: the figure is about when a word rose, and the
    # levels differ by two orders of magnitude, so a shared scale would show one row.
    heat = shares[order]
    heat = heat / heat.max(axis=1, keepdims=True)
    right.imshow(
        heat,
        aspect="auto",
        origin="lower",
        cmap=LinearSegmentedColormap.from_list("rise", ["#FFFFFF", SERIES[0]]),
        extent=(0, len(years), 0, 10 * n),
        interpolation="nearest",
    )
    right.set_yticks([10 * i + 5 for i in range(n)])
    right.set_yticklabels([ranked[i][0] for i in order], fontsize=9)
    right.yaxis.tick_right()
    right.set_xticks([i + 0.5 for i in range(len(years))])
    right.set_xticklabels([str(y) for y in years], fontsize=7, rotation=90)
    right.tick_params(length=0)
    for spine in right.spines.values():
        spine.set_visible(False)
    left.set_ylim(0, 10 * n)
    fig.suptitle(txt["f8_title"], fontsize=11, fontweight="semibold", color=INK)
    fig.savefig(path)
    plt.close(fig)


#: (file stem, drawing function, the report section it belongs under, caption key).
#: The section lives here because the report asks this module where a figure goes.
#: In the order the report shows them, which is what the numbering has to follow: a
#: reader meeting "Діаграма 5" on the first page and "Діаграма 1" near the end is being
#: numbered by a catalogue they cannot see. The file stems carry the same order, so the
#: name on disk, the caption and the position in the document all agree.
FIGURES: tuple[tuple[str, Callable, str, str], ...] = (
    ("f1-area-share-over-time", f5_share_over_time, "1", "f5_title"),
    ("f2-share-change", f6_share_change, "1", "f6_title"),
    ("f3-cooccurrence", f7_cooccurrence, "2", "f7_title"),
    ("f4-topic-trajectories", f1_topic_trajectories, "4", "f1_title"),
    ("f5-growth-measures", f4_growth_measures, "4", "f4_title"),
    ("f6-rising-vocabulary", f8_rising_vocabulary, "4", "f8_title"),
    ("f7-indicator-panel", f2_indicator_panel, "5", "f2_title"),
    ("f8-output-vs-impact", f3_output_vs_impact, "5", "f3_title"),
)

#: The word that introduces a caption, per language.
CAPTION: Mapping[str, str] = {"en": "Figure", "uk": "Діаграма"}


def available() -> bool:
    """Whether figures can be drawn at all — matplotlib is optional."""
    from importlib.util import find_spec

    return find_spec("matplotlib") is not None


def draw(
    facts: Mapping[str, Any],
    out_dir: Path,
    lang: str,
    text: Mapping[str, str] | None = None,
) -> list[str]:
    """Draw every figure for one language, returning the stems actually written.

    A figure whose input is missing draws nothing and is simply absent from the return,
    so the report references what exists rather than what was planned.

    ``text`` overlays the titles and axis labels the way ``labels`` overlays the report's
    table headers — a figure axis reading "publications" over a patent register is the
    same defect as a column heading that does, and harder to notice because nobody
    proof-reads a picture. Partial mappings are completed from :data:`TEXT`; ``None``
    keeps it exactly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = TEXT[lang] if text is None else {**TEXT[lang], **text}
    written = []
    for name, fn, _section, _caption in FIGURES:
        path = out_dir / f"{name}.png"
        fn(facts, txt, path)
        if path.exists():
            written.append(name)
    return written


def references(
    drawn: Sequence[str],
    lang: str,
    folder: str,
    text: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Markdown image lines per report section, for the figures that exist.

    ``text`` must be the same overlay :func:`draw` was given, or the caption under a
    figure will describe something other than what the figure was drawn as.
    """
    txt = TEXT[lang] if text is None else {**TEXT[lang], **text}
    out: dict[str, list[str]] = {}
    for name, _fn, section, caption_key in FIGURES:
        if name not in drawn:
            continue
        number = name.split("-", 1)[0].lstrip("f")
        alt = f"{CAPTION[lang]} {number}. {txt[caption_key]}"
        out.setdefault(section, []).append(f"![{alt}]({folder}/{name}.png)")
    return out
