"""Structured extraction: text -> validated :class:`Analysis`.

Gemma has no native JSON mode, so we prompt for JSON, parse it tolerantly, and
validate into the dataclasses. Two strategies,
both producing the same :class:`Analysis`:

- ``combined`` — one model call returns the whole object.
- ``staged``   — four calls (summary / classification / entities / keywords),
  each a small JSON fragment, merged into one :class:`Analysis`.

Each model call gets **one** repair retry on a parse/validation failure (the error
is fed back to the model); a second failure raises :class:`AnalysisError`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum

from horizon import prompts
from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
    content_hash,
)
from horizon.config import GenerationConfig, Settings
from horizon.llm.base import LLMClient


class Strategy(str, Enum):
    """Extraction strategy: one combined model call or four staged calls."""

    COMBINED = "combined"
    STAGED = "staged"


class AnalysisError(RuntimeError):
    """Raised when the model output can't be parsed/validated into an Analysis."""


# --- tolerant JSON parsing ----------------------------------------------------


_VALID_ESCAPE = set('"\\/bfnrtu')


def _escape_stray_backslashes(text: str) -> str:
    r"""Double any backslash that is not opening a valid JSON escape.

    Abstracts in mathematics and physics carry LaTeX, the model copies it into its
    summary, and ``\alpha`` is not a JSON escape — ``json.loads`` rejects the whole
    object over one character. Measured on the 2015 shard this was every single parse
    failure, about 0.5% of items, and concentrated by construction in the fields whose
    decline the report is trying to measure: a random 0.5% is noise, a 0.5% drawn from
    Physics and Mathematics is bias.

    Only backslashes are touched, and only those that would otherwise be a syntax error,
    so a reply that was already valid JSON parses identically. ``\u`` is left alone —
    a malformed one is a different error, and guessing at it would corrupt a real escape.
    """
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if ch != "\\":
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt not in _VALID_ESCAPE:
            out.append("\\")
    return "".join(out)


# Every key the extraction prompts ask for, in the order they are asked for. Used only
# as boundaries when repairing a broken reply: knowing which key must come next is what
# separates "the value ends here" from a guess.
_SCHEMA_KEYS = (
    "summary",
    "signals",
    "classification",
    "theme",
    "maturity",
    "impact",
    "time_horizon",
    "entities",
    "organizations",
    "people",
    "technologies",
    "locations",
    "keywords",
)
# The subset whose value is a bare string, so a stray quote inside it breaks the parse.
_STRING_KEYS = frozenset({"summary", "theme", "maturity", "impact", "time_horizon"})
_KEY_RE = re.compile(r'"(' + "|".join(_SCHEMA_KEYS) + r')"\s*:')


def _escape_quotes_in_string_values(text: str) -> str:
    r"""Escape interior quotes inside the string values the schema declares.

    The failure this addresses: a title quoted inside a summary — ``so-called "smart"
    grids`` — ends the JSON string early and the parse dies on the next character.
    Unlike a stray backslash, an interior quote is not self-evidently wrong: telling it
    from the quote that closes the string needs to know where the value ends.

    So this does not guess. It uses the schema we ourselves asked for: a value runs until
    the next key the prompt declares, and every quote between the first and last one in
    that span is interior by construction. A reply that puts the keys in another order,
    or omits them, simply fails to match and falls through to the normal error.

    Measured on the corpus, this matters more than 0.2% of items suggests: 37.5% of Arts
    and Humanities records carry a quote character against 6.9% of Mathematics ones, a
    5.4x spread, and the quote-heavy fields are the fastest-growing ones in the report.
    """
    marks = [(m.start(), m.group(1)) for m in _KEY_RE.finditer(text)]
    if not marks:
        return text
    out = text
    # Right to left, so earlier offsets stay valid as the string is rewritten.
    for i in range(len(marks) - 1, -1, -1):
        start, key = marks[i]
        if key not in _STRING_KEYS:
            continue
        colon = out.find(":", start)
        end = marks[i + 1][0] if i + 1 < len(marks) else len(out)
        span = out[colon + 1 : end]
        opening = span.find('"')
        closing = span.rfind('"')
        if opening == -1 or closing <= opening:
            continue
        inner = span[opening + 1 : closing]
        if '"' not in inner:
            continue
        fixed = span[: opening + 1] + inner.replace('"', '\\"') + span[closing:]
        out = out[: colon + 1] + fixed + out[end:]
    return out


def _around(text: str, pos: int, width: int = 60) -> str:
    """The text either side of ``pos``, for an error message.

    Centred rather than truncated from the start: the fault is what needs reading, and
    in a 2 000-character reply the first 120 characters never contain it.
    """
    lo = max(0, pos - width // 2)
    return (
        ("…" if lo else "")
        + text[lo : pos + width // 2]
        + ("…" if pos + width // 2 < len(text) else "")
    )


def _not_an_object() -> dict:
    """Raise the shared "not an object" error; a repaired non-dict is still a failure."""
    raise AnalysisError("expected a JSON object, got a non-object")


def parse_json(raw: str) -> dict:
    """Extract a JSON object from a model reply that may wrap it in prose/fences."""
    text = raw.strip()
    # Try the clean case first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Otherwise pull out the first balanced {...}, ignoring braces inside strings.
    snippet = _first_json_object(text)
    if snippet is None:
        raise AnalysisError(f"no JSON object found in model output: {raw[:120]!r}")
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as exc:
        # One more attempt with stray backslashes doubled, before spending a repair call
        # on what is a transcription problem rather than a comprehension one.
        repaired = _escape_stray_backslashes(snippet)
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            try:
                obj = json.loads(_escape_quotes_in_string_values(repaired))
            except json.JSONDecodeError:
                pass
            else:
                return obj if isinstance(obj, dict) else _not_an_object()
            # Carry the text around the fault, not just the class of fault. The
            # "no JSON object found" branch below already does this, and the asymmetry
            # cost a diagnosis: a log full of "Expecting ',' delimiter" tells you
            # something is wrong with a string and leaves you guessing which string.
            raise AnalysisError(
                f"malformed JSON in model output: {exc} — near "
                f"{_around(snippet, exc.pos)!r}"
            ) from exc
    if not isinstance(obj, dict):
        raise AnalysisError("expected a JSON object, got a non-object")
    return obj


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces inside strings;
    ``None`` if there is no complete object."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# --- validation helpers -------------------------------------------------------


def _require(d: dict, key: str):
    """Return ``d[key]`` or raise :class:`AnalysisError` if the field is missing."""
    if key not in d:
        raise AnalysisError(f"missing required field: {key!r}")
    return d[key]


def _str_tuple(value) -> tuple[str, ...]:
    """Coerce a model value into a tuple of non-empty strings, leniently."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(str(x).strip() for x in value if str(x).strip())


def _enum(enum_cls, value, field: str):
    """Coerce a model value into an ``enum_cls`` member, leniently. A partial
    option-list echo (a strict subset of the allowed values) is salvaged by taking
    its first valid token; a whole-list echo or unknown value raises
    :class:`AnalysisError`."""
    valid = {e.value for e in enum_cls}
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(e.value for e in enum_cls)
        got = str(value).strip()
        # Small models sometimes echo the enum options ("a | b | …") instead of
        # choosing one. Salvage a *partial* echo (a strict subset of the allowed
        # values) by taking its first valid token — the model narrowed the choice
        # even if it didn't commit. A *whole-list* echo is a non-answer: no signal
        # to salvage, so we surface it (and let the repair retry try again).
        parts = [p.strip().lower() for p in got.split("|") if p.strip()]
        chosen = [p for p in parts if p in valid]
        if len(parts) > 1 and chosen and set(chosen) != valid:
            return enum_cls(chosen[0])
        if set(parts) == valid:
            detail = "model returned the whole options list instead of one value"
        else:
            detail = f"got {value!r}"
        raise AnalysisError(
            f"invalid {field}: {detail}; must be one of: {allowed}"
        ) from exc


def build_classification(d: dict) -> Classification:
    """Validate a dict into a :class:`Classification` (required theme + three enums)."""
    return Classification(
        theme=str(_require(d, "theme")).strip(),
        maturity=_enum(Maturity, _require(d, "maturity"), "maturity"),
        impact=_enum(Impact, _require(d, "impact"), "impact"),
        time_horizon=_enum(TimeHorizon, _require(d, "time_horizon"), "time_horizon"),
    )


def build_entities(d: dict) -> Entities:
    """Build an :class:`Entities` from a dict, coercing each field to a string tuple
    (all four fields optional)."""
    return Entities(
        organizations=_str_tuple(d.get("organizations", [])),
        people=_str_tuple(d.get("people", [])),
        technologies=_str_tuple(d.get("technologies", [])),
        locations=_str_tuple(d.get("locations", [])),
    )


def _build_combined(d: dict) -> dict:
    """Validate a whole-object reply into the :class:`Analysis` field dict."""
    return {
        "summary": str(_require(d, "summary")).strip(),
        "signals": _str_tuple(_require(d, "signals")),
        "classification": build_classification(_require(d, "classification")),
        "entities": build_entities(d.get("entities", {})),
        "keywords": _str_tuple(_require(d, "keywords")),
    }


# --- stages: one model call each, with a shared repair retry -------------------


@dataclass(frozen=True)
class _Stage:
    """One model call in a strategy: its prompt template, validator and merge step.

    Making the strategies data rather than straight-line code is what lets a stage be
    issued for many items at once (``analyze_batch``) without duplicating the
    parse/validate/repair contract per strategy.
    """

    template: str
    build: Callable[[dict], object]
    merge: Callable[[dict, object], None]


def _merge_update(fields: dict, result) -> None:
    """Merge a stage that returns a partial field dict."""
    fields.update(result)


def _merge_into(key: str) -> Callable[[dict, object], None]:
    """Merge a stage whose result is a single field."""

    def merge(fields: dict, result) -> None:
        fields[key] = result

    return merge


_COMBINED_STAGES = (
    _Stage(prompts.COMBINED_PROMPT_TEMPLATE, _build_combined, _merge_update),
)

_STAGED_STAGES = (
    _Stage(
        prompts.STAGED_SUMMARY_PROMPT,
        lambda d: {
            "summary": str(_require(d, "summary")).strip(),
            "signals": _str_tuple(_require(d, "signals")),
        },
        _merge_update,
    ),
    _Stage(
        prompts.STAGED_CLASSIFICATION_PROMPT,
        build_classification,
        _merge_into("classification"),
    ),
    _Stage(prompts.STAGED_ENTITIES_PROMPT, build_entities, _merge_into("entities")),
    _Stage(
        prompts.STAGED_KEYWORDS_PROMPT,
        lambda d: _str_tuple(_require(d, "keywords")),
        _merge_into("keywords"),
    ),
)


def _stages(strategy: Strategy) -> tuple[_Stage, ...]:
    """The ordered model calls for ``strategy``."""
    return _COMBINED_STAGES if strategy is Strategy.COMBINED else _STAGED_STAGES


def _repair_prompt(prompt: str, error: Exception) -> str:
    """The one retry we allow: the original prompt plus what was wrong with the reply."""
    return (
        f"{prompt}\n\nYour previous reply could not be used: {error}\n"
        "Return ONLY a valid JSON object matching the schema above."
    )


# --- public API ---------------------------------------------------------------


def analyze_text(
    text: str,
    client: LLMClient,
    *,
    settings: Settings | None = None,
    config: GenerationConfig | None = None,
    strategy: Strategy | str | None = None,
    item_date: str | None = None,
) -> Analysis:
    """Scan ``text`` into a validated :class:`Analysis` using the chosen strategy.

    A batch of one, so the single-item and batched paths cannot drift: same prompts,
    same call order, same repair retry.
    """
    (result,) = analyze_batch(
        [text],
        client,
        settings=settings,
        config=config,
        strategy=strategy,
        item_dates=[item_date],
    )
    if isinstance(result, AnalysisError):
        raise result
    return result


def analyze_batch(
    texts: Sequence[str],
    client: LLMClient,
    *,
    settings: Settings | None = None,
    config: GenerationConfig | None = None,
    strategy: Strategy | str | None = None,
    item_dates: Sequence[str | None] | None = None,
) -> list[Analysis | AnalysisError]:
    """Analyse many texts, batching each stage across items.

    Returns one entry per input, in input order: an :class:`Analysis`, or the
    :class:`AnalysisError` that item failed with — one bad item never sinks the batch.

    Stage-major: every pending item's stage-N prompt goes out in one
    ``client.complete_batch`` call, then a *second* call repairs only the replies that
    failed to validate. That preserves the per-item contract exactly — at most two model
    calls per item per stage, with the same prompts — and changes only the order the
    calls are interleaved in. An item that fails twice drops out and issues no further
    stage calls, matching the single-item behaviour where an exception ends that item.
    """
    settings = settings or Settings()
    strategy = Strategy(strategy or settings.strategy)
    system = prompts.EXTRACT_SYSTEM_PROMPT
    cleaned = [t.strip() for t in texts]
    dates = list(item_dates or [None] * len(cleaned))

    # Structured extraction is a "pick the right tokens" task, not creative
    # generation. Greedy decoding is far more reliable here: with sampling, small
    # models (e.g. Gemma-3-1B) drift and echo the enum spec ("emerging | growing |
    # …") back as a value. Force it regardless of the caller's sampling config.
    config = replace(config or GenerationConfig(), do_sample=False)

    fields: list[dict] = [{} for _ in cleaned]
    errors: list[AnalysisError | None] = [None] * len(cleaned)
    pending = list(range(len(cleaned)))

    for stage in _stages(strategy):
        if not pending:
            break
        stage_prompts = [stage.template.format(text=cleaned[i]) for i in pending]
        replies = client.complete_batch(stage_prompts, system=system, config=config)
        retry: list[tuple[int, str]] = []
        for index, prompt, raw in zip(pending, stage_prompts, replies):
            try:
                stage.merge(fields[index], stage.build(parse_json(raw)))
            except AnalysisError as first:
                retry.append((index, _repair_prompt(prompt, first)))
        if retry:
            repairs = client.complete_batch(
                [prompt for _, prompt in retry], system=system, config=config
            )
            for (index, _), raw in zip(retry, repairs):
                try:
                    stage.merge(fields[index], stage.build(parse_json(raw)))
                except AnalysisError as second:
                    errors[index] = AnalysisError(
                        "model did not return valid JSON after one repair retry: "
                        f"{second}"
                    )
        pending = [i for i in pending if errors[i] is None]

    generated_at = datetime.now(timezone.utc).isoformat()
    results: list[Analysis | AnalysisError] = []
    for index, text in enumerate(texts):
        if errors[index] is not None:
            results.append(errors[index])
            continue
        f = fields[index]
        results.append(
            Analysis(
                summary=f["summary"],
                signals=f["signals"],
                classification=f["classification"],
                entities=f["entities"],
                keywords=f["keywords"],
                content_hash=content_hash(text),
                model_id=settings.model_id,
                prompt_version=prompts.PROMPT_VERSION,
                strategy=strategy.value,
                generated_at=generated_at,
                item_date=dates[index] or generated_at,
            )
        )
    return results
