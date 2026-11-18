"""Versioned prompts, kept in code.

``PROMPT_VERSION`` covers the whole set below: bump it whenever any prompt changes
so cached analyses produced by an older prompt are re-generated rather than reused
(the store keys on ``(content_hash, prompt_version)``).

Both extraction strategies live here:
- ``COMBINED_*`` — one call returns the whole ``Analysis`` JSON.
- the four ``STAGED_*`` prompts — one call each for summary / classification /
  entities / keywords, merged into one ``Analysis``.

Every extraction prompt instructs the model to emit **only** a JSON object — Gemma
has no native JSON mode, so the shape is prompt-driven and parsed tolerantly. The
templates are plain ``str.format`` strings: literal JSON braces are doubled, and
``{text}`` is the only substitution.
"""

from __future__ import annotations

PROMPT_VERSION = "phase2.v4"

# --- Phase-1 freeform scan (kept for the scan_text path) ----------------------

SCAN_SYSTEM_PROMPT = (
    "You are a horizon-scanning analyst. Given a text, identify emerging signals "
    "about the landscape it describes: notable developments, who is involved, and "
    "why they might matter. Work across any domain. Be concise and factual."
)

SCAN_PROMPT_TEMPLATE = """\
Analyze the following text for horizon-scanning signals.

Produce:
1. Summary (2-3 sentences).
2. Key signals (bullet points).
3. Why it matters (1-2 sentences).

TEXT:
{text}
"""

# --- Structured extraction (Phase 2) ------------------------------------------

EXTRACT_SYSTEM_PROMPT = (
    "You are a horizon-scanning analyst working across any domain. You extract "
    "structured signals from a text. Respond with ONLY a single JSON object — no "
    "prose, no markdown fences, no commentary. Use double quotes for all keys and "
    "string values."
)

# Granularity anchor for the free-text `theme`. Note what this does NOT do: clustering
# runs on embeddings of `summary` + `signals` (embeddings.embed_text) and never reads
# this field, so the anchor changes no grouping at all. What it changes is readability —
# `trends._label` names each cluster with the most common `theme` string among its
# members, so labels pitched at different levels of generality make the winner arbitrary.
# It is also deliberately NOT offered as a fix for `maturity` — the granularity account
# of that disagreement was tested and refuted (report §8.5 (ii)): 12B writes the narrowest
# themes of the three and still answers `mature` for 77% of records, and two models have
# produced the identical theme string with opposite maturity.
THEME_RUBRIC = """\
`theme` names the line of work the text belongs to, at a consistent level: narrower than
a discipline, broader than this text's own contribution — the level at which a review
article or a conference session is defined. "Formal logic" is too broad, "entailment in
three-valued semantics" too narrow. Two or three words. Use the same level for every
text: the judgements below are made relative to it."""

# Rubric for `impact`. It had no rubric in phase2.v2 and was never mentioned there, yet
# specifying its two neighbours moved it: agreement fell on two of three pairs and E2B
# flipped its modal label from `medium` 49% to `high` 46% (report §8.5 (iv)). Enum fields
# in one prompt are not independent, so leaving one unspecified does not leave it alone.
IMPACT_RUBRIC = """\
`impact` is how consequential that theme becoming established would be, measured by how
far its effects reach beyond its own specialty — not by the ambition of this text.
  low: effects stay within the specialty; refinement of existing practice.
  medium: changes practice across neighbouring areas, or across one sector.
  high: would change decisions or capabilities well outside its own field.
The text's own claims of significance are not evidence. Judge the theme's reach, not the
quality of the work. Do not use `medium` as a hedge."""

# Rubric for the `maturity` enum. Without it, three Gemma 4 variants scanning the same
# 997 records agreed at Cohen's kappa 0.04-0.20 — at or near chance — because each
# collapsed onto a different constant: 70% "growing", 70% "emerging", 58% "mature"
# respectively (reports/gemma4-gpu-throughput-wos-scopus-2026-08-26.md §8). They agreed
# far better on extraction (entity F1 0.58-0.62 for every pair), which located the defect
# in the specification rather than in any model: nothing said WHAT was being judged.
# Naming the unit of judgement is the substance; the two override lines counteract the
# defaults actually observed.
#
# Kept terse on purpose. Input length is cheap in *time* (the August 21 measurements put
# cost correlation at r = +0.85..1.00 with output tokens against -0.08..0.55 with input
# length) but it is not cheap in *memory*: under batching every input token occupies KV
# cache, and cache size is what caps the batch, which is the throughput lever. An earlier,
# three-times-longer draft of these rubrics pushed gemma-4-12B-it at batch 16 from a
# measured 16.5 GB to ~20.6 GB and out of memory. Length here is paid for in batch size.
MATURITY_RUBRIC = """\
`maturity` judges that theme — not this text's own contribution. A new result in an old field is `mature`; a routine study of
last year's invention is `emerging`.
  emerging: the theme itself is new; evidence preliminary; little established practice.
  growing: established enough to build on, and actively expanding.
  mature: settled and used as a given — standard tool, survey, or routine application.
  declining: superseded, abandoned, or contested by negative results.
Novelty of THIS text is not maturity. Age or formality is not maturity. Choose
`declining` only on explicit evidence of decline."""

# Rubric for the `time_horizon` enum, and the same defect: kappa 0.14-0.23 across the
# three variants, with `E2B` and `E4B` answering "mid" for 78% and 86% of records while
# `12B` answered "long" for 53%. Horizon of WHAT was never stated — until publication,
# until adoption, until consequence — so each model picked one and stayed with it. The
# year bands are what turn this from a vague judgement into a checkable one; they are
# deliberately rough, since the enum only has three values.
TIME_HORIZON_RUBRIC = """\
`time_horizon` is how soon that theme is likely to matter in practice —
to affect decisions, deployment, or outcomes beyond the research literature itself —
judged from the text's own vantage point, not from today.
  - near (roughly 0-2 years): already in use, or deployable with means that exist now;
    its effects are being felt or are imminent.
  - mid (roughly 2-5 years): a plausible path to practical effect exists, but further
    development, validation, or adoption has to happen first.
  - long (roughly 5+ years): effects depend on advances, infrastructure, or conditions
    not yet in place, or the work is foundational with no near path to application.
These override the impressions above:
  - The text's own timeframe is not the horizon. Work completed today can concern
    something whose consequences are decades away, and vice versa.
  - Abstraction is not distance. If a theme is already applied routinely, it is `near`
    however theoretical the text looks.
  - Do not use `mid` as a hedge. Choose it when the evidence points there, not to avoid
    committing.
  - `time_horizon` and `maturity` are separate judgements: a mature theme can still have
    long-horizon consequences, and an emerging one can matter immediately."""

# Allowed enum values (mirror analysis.Maturity / Impact / TimeHorizon), inlined
# into the prompts so the model stays in range.
COMBINED_PROMPT_TEMPLATE = (
    """\
Extract a horizon-scanning analysis of the TEXT as a JSON object with exactly these keys:

{{
  "summary": "2-3 sentence summary",
  "signals": ["key signal", "..."],
  "classification": {{
    "theme": "short domain-agnostic theme label",
    "maturity": "emerging",
    "impact": "high",
    "time_horizon": "long"
  }},
  "entities": {{
    "organizations": [], "people": [], "technologies": [], "locations": []
  }},
  "keywords": ["keyword", "..."]
}}

The enum values above are EXAMPLES — replace each with the single value that fits:
  - maturity: one of emerging, growing, mature, declining
  - impact: one of low, medium, high
  - time_horizon: one of near, mid, long
Each enum field must be exactly one of those words — never a list, never "a | b".
Use empty lists where nothing applies.

"""
    + THEME_RUBRIC
    + """

"""
    + MATURITY_RUBRIC
    + """

"""
    + TIME_HORIZON_RUBRIC
    + """

"""
    + IMPACT_RUBRIC
    + """

TEXT:
{text}
"""
)

STAGED_SUMMARY_PROMPT = """\
Summarize the TEXT for horizon scanning. Return ONLY JSON:
{{"summary": "2-3 sentences", "signals": ["key signal", "..."]}}

TEXT:
{text}
"""

STAGED_CLASSIFICATION_PROMPT = (
    """\
Classify the TEXT for horizon scanning. Return ONLY JSON with exactly these keys
(the enum values shown are EXAMPLES — replace each with the one value that fits):
{{"theme": "short domain-agnostic theme label", "maturity": "emerging", "impact": "high", "time_horizon": "long"}}

Each enum field must be exactly one word — never a list, never "a | b":
  - maturity: one of emerging, growing, mature, declining
  - impact: one of low, medium, high
  - time_horizon: one of near, mid, long

"""
    + THEME_RUBRIC
    + """

"""
    + MATURITY_RUBRIC
    + """

"""
    + TIME_HORIZON_RUBRIC
    + """

"""
    + IMPACT_RUBRIC
    + """

TEXT:
{text}
"""
)

STAGED_ENTITIES_PROMPT = """\
Extract named entities from the TEXT. Return ONLY JSON with exactly these keys
(use empty lists where nothing applies):
{{"organizations": [], "people": [], "technologies": [], "locations": []}}

TEXT:
{text}
"""

STAGED_KEYWORDS_PROMPT = """\
Extract the salient keywords/phrases from the TEXT. Return ONLY JSON:
{{"keywords": ["keyword", "..."]}}

TEXT:
{text}
"""
