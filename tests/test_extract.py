import json

import pytest

from horizon.analysis import Impact, Maturity, TimeHorizon
from horizon.extract import (
    AnalysisError,
    Strategy,
    analyze_batch,
    analyze_text,
    parse_json,
)
from horizon.prompts import PROMPT_VERSION

COMBINED = {
    "summary": "A new open model was released.",
    "signals": ["sota reasoning", "startups fine-tuning"],
    "classification": {
        "theme": "open models",
        "maturity": "emerging",
        "impact": "high",
        "time_horizon": "near",
    },
    "entities": {
        "organizations": ["CloudCo"],
        "people": [],
        "technologies": ["LLM"],
        "locations": [],
    },
    "keywords": ["open-weights", "reasoning"],
}

# Staged fragments, in call order: summary, classification, entities, keywords.
STAGED = [
    json.dumps(
        {
            "summary": "A new open model was released.",
            "signals": ["sota reasoning", "startups fine-tuning"],
        }
    ),
    json.dumps(
        {
            "theme": "open models",
            "maturity": "emerging",
            "impact": "high",
            "time_horizon": "near",
        }
    ),
    json.dumps(
        {
            "organizations": ["CloudCo"],
            "people": [],
            "technologies": ["LLM"],
            "locations": [],
        }
    ),
    json.dumps({"keywords": ["open-weights", "reasoning"]}),
]


# --- parse_json tolerance -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(COMBINED),
        f"```json\n{json.dumps(COMBINED)}\n```",
        f"Here is the analysis:\n{json.dumps(COMBINED)}\nHope that helps!",
    ],
)
def test_parse_json_tolerates_fences_and_prose(raw):
    assert parse_json(raw)["summary"] == COMBINED["summary"]


def test_parse_json_raises_when_no_object():
    with pytest.raises(AnalysisError):
        parse_json("no json here")


# --- combined strategy --------------------------------------------------------


def test_combined_builds_analysis_with_provenance(make_client):
    client = make_client(json.dumps(COMBINED))
    a = analyze_text("some text", client, strategy="combined")

    assert a.summary == COMBINED["summary"]
    assert a.signals == ("sota reasoning", "startups fine-tuning")
    assert a.classification.maturity is Maturity.EMERGING
    assert a.classification.impact is Impact.HIGH
    assert a.classification.time_horizon is TimeHorizon.NEAR
    assert a.entities.organizations == ("CloudCo",)
    assert a.keywords == ("open-weights", "reasoning")
    assert a.prompt_version == PROMPT_VERSION
    assert a.strategy == "combined"
    assert a.content_hash  # stamped
    assert len(client.calls) == 1


# --- staged strategy ----------------------------------------------------------


def test_staged_matches_combined(make_client):
    combined_client = make_client(json.dumps(COMBINED))
    staged_client = make_client(STAGED)

    a = analyze_text("some text", combined_client, strategy="combined")
    b = analyze_text("some text", staged_client, strategy="staged")

    assert len(staged_client.calls) == 4
    # Same content modulo provenance (strategy + generated_at differ).
    assert (a.summary, a.signals, a.classification, a.entities, a.keywords) == (
        b.summary,
        b.signals,
        b.classification,
        b.entities,
        b.keywords,
    )
    assert b.strategy == "staged"
    assert a.content_hash == b.content_hash


# --- validation ---------------------------------------------------------------


def test_missing_field_raises(make_client):
    bad = {k: v for k, v in COMBINED.items() if k != "summary"}
    client = make_client(json.dumps(bad))
    with pytest.raises(AnalysisError):
        analyze_text("t", client, strategy="combined")


def test_bad_enum_value_raises(make_client):
    bad = json.loads(json.dumps(COMBINED))
    bad["classification"]["impact"] = "catastrophic"
    client = make_client(json.dumps(bad))
    with pytest.raises(AnalysisError):
        analyze_text("t", client, strategy="combined")


def test_enum_error_reports_bad_value():
    from horizon.analysis import Maturity
    from horizon.extract import _enum

    with pytest.raises(AnalysisError) as exc:
        _enum(Maturity, "Mid-term", "maturity")
    msg = str(exc.value)
    assert "got 'Mid-term'" in msg
    assert "must be one of: emerging, growing, mature, declining" in msg


def test_enum_error_flags_echoed_options_list():
    # The model parroting the whole allowed list gets a clear message, not the
    # baffling "invalid X (allowed: X)" that repeats an identical string.
    from horizon.analysis import Maturity
    from horizon.extract import _enum

    with pytest.raises(AnalysisError) as exc:
        _enum(Maturity, "emerging | growing | mature | declining", "maturity")
    assert "whole options list" in str(exc.value)


# --- repair retry -------------------------------------------------------------


def test_repair_retry_recovers(make_client):
    client = make_client(["not json at all", json.dumps(COMBINED)])
    a = analyze_text("t", client, strategy="combined")
    assert a.summary == COMBINED["summary"]
    assert len(client.calls) == 2  # one bad + one repair


def test_repair_retry_gives_up(make_client):
    client = make_client(["nope", "still nope"])
    with pytest.raises(AnalysisError):
        analyze_text("t", client, strategy="combined")
    assert len(client.calls) == 2


# --- analyze_batch ------------------------------------------------------------


def _by_prompt(mapping, default):
    """A prompt-keyed reply function: batching reorders calls, indices don't survive."""

    def reply(messages):
        prompt = messages[-1].content
        for needle, answer in mapping.items():
            if needle in prompt:
                return answer
        return default

    return reply


def _content(a):
    return (a.summary, a.signals, a.classification, a.entities, a.keywords)


def test_analyze_batch_matches_sequential(make_client):
    """Batch of N must equal N sequential calls, field for field."""
    good = json.dumps(COMBINED)
    texts = ["alpha text", "beta text", "gamma text"]

    batched = make_client(_by_prompt({}, good))
    results = analyze_batch(texts, batched, strategy="combined")
    sequential = [
        analyze_text(t, make_client(_by_prompt({}, good)), strategy="combined")
        for t in texts
    ]
    assert [_content(r) for r in results] == [_content(s) for s in sequential]
    assert [r.content_hash for r in results] == [s.content_hash for s in sequential]


def test_analyze_batch_groups_one_call_per_stage(make_client):
    client = make_client(_by_prompt({}, json.dumps(COMBINED)))
    analyze_batch(["a", "b", "c"], client, strategy="combined")
    assert client.batches == [3]  # one combined stage, one batch, no repairs


def test_analyze_batch_staged_batches_every_stage(make_client):
    replies = {
        "Summarize the TEXT": STAGED[0],
        "Classify the TEXT": STAGED[1],
        "Extract named entities": STAGED[2],
        "salient keywords": STAGED[3],
    }
    client = make_client(_by_prompt(replies, STAGED[0]))
    results = analyze_batch(["a", "b"], client, strategy="staged")
    assert all(not isinstance(r, AnalysisError) for r in results)
    assert client.batches == [2, 2, 2, 2]  # four stages, each batched across both items


def test_analyze_batch_repairs_only_the_failing_item(make_client):
    good = json.dumps(COMBINED)

    def reply(messages):
        prompt = messages[-1].content
        # "beta" fails its first attempt, then succeeds on the repair.
        if "beta" in prompt and "could not be used" not in prompt:
            return "not json at all"
        return good

    client = make_client(reply)
    results = analyze_batch(["alpha", "beta", "gamma"], client, strategy="combined")
    assert all(not isinstance(r, AnalysisError) for r in results)
    assert client.batches == [3, 1]  # one attempt batch, one repair batch of one
    repair_prompts = [
        c[-1].content for c in client.calls if "could not be used" in c[-1].content
    ]
    assert len(repair_prompts) == 1 and "beta" in repair_prompts[0]


def test_analyze_batch_isolates_a_permanent_failure(make_client):
    good = json.dumps(COMBINED)

    def reply(messages):
        return "nope" if "beta" in messages[-1].content else good

    client = make_client(reply)
    results = analyze_batch(["alpha", "beta", "gamma"], client, strategy="combined")
    assert isinstance(results[1], AnalysisError)
    assert not isinstance(results[0], AnalysisError)
    assert not isinstance(results[2], AnalysisError)
    assert "after one repair retry" in str(results[1])


def test_analyze_batch_at_most_two_calls_per_item_per_stage(make_client):
    """The invariant that bounds cost: an attempt plus at most one repair."""
    client = make_client(lambda messages: "never valid")
    results = analyze_batch(["a", "b"], client, strategy="combined")
    assert all(isinstance(r, AnalysisError) for r in results)
    assert len(client.calls) == 4  # 2 items x (1 attempt + 1 repair)


def test_analyze_batch_staged_drops_a_failed_item_from_later_stages(make_client):
    replies = {
        "Classify the TEXT": STAGED[1],
        "Extract named entities": STAGED[2],
        "salient keywords": STAGED[3],
    }

    def reply(messages):
        prompt = messages[-1].content
        # "beta" can never pass stage 1, so it must issue no stage 2-4 calls.
        if "Summarize the TEXT" in prompt:
            return "garbage" if "beta" in prompt else STAGED[0]
        for needle, answer in replies.items():
            if needle in prompt:
                return answer
        return STAGED[0]

    client = make_client(reply)
    results = analyze_batch(["alpha", "beta"], client, strategy="staged")
    assert isinstance(results[1], AnalysisError)
    assert not isinstance(results[0], AnalysisError)
    later = [
        c[-1].content for c in client.calls if "Summarize the TEXT" not in c[-1].content
    ]
    assert later and all("beta" not in p for p in later)


def test_analyze_batch_empty(make_client):
    client = make_client("x")
    assert analyze_batch([], client, strategy="combined") == []
    assert client.calls == []


def test_analyze_batch_stamps_item_dates_positionally(make_client):
    """Dates must stay aligned to inputs even after a failure shifts the pending list."""
    good = json.dumps(COMBINED)

    def reply(messages):
        return "nope" if "beta" in messages[-1].content else good

    client = make_client(reply)
    results = analyze_batch(
        ["alpha", "beta", "gamma"],
        client,
        strategy="combined",
        item_dates=["2020-01-01", "2021-01-01", "2022-01-01"],
    )
    assert results[0].item_date == "2020-01-01"
    assert isinstance(results[1], AnalysisError)
    assert results[2].item_date == "2022-01-01"


def test_analyze_text_call_order_is_unchanged_for_staged(make_client):
    """Guards the wrapper refactor: still four calls, still in stage order."""
    client = make_client(STAGED)
    analyze_text("some text", client, strategy="staged")
    assert len(client.calls) == 4
    prompts_sent = [c[-1].content for c in client.calls]
    assert "Summarize the TEXT" in prompts_sent[0]
    assert "Classify the TEXT" in prompts_sent[1]
    assert "Extract named entities" in prompts_sent[2]
    assert "salient keywords" in prompts_sent[3]


# --- stray backslashes --------------------------------------------------------


def test_latex_in_a_summary_no_longer_kills_the_object():
    """Every parse failure on the 2015 shard was this: \\alpha is not a JSON escape."""
    out = parse_json(r'{"summary": "we compute \alpha and \pm 3", "signals": []}')
    assert out["summary"] == r"we compute \alpha and \pm 3"


def test_a_valid_escape_is_left_alone():
    assert parse_json(r'{"summary": "line\nbreak"}')["summary"] == "line\nbreak"
    assert parse_json(r'{"summary": "A"}')["summary"] == "A"
    assert parse_json(r'{"summary": "quote \" inside"}')["summary"] == 'quote " inside'


def test_a_real_escaped_backslash_survives():
    """Doubling must not turn one backslash into three."""
    assert parse_json(json.dumps({"summary": r"a \ backslash"}))["summary"] == (
        r"a \ backslash"
    )


def test_already_valid_json_parses_identically():
    """The repair runs only after a failure, so the clean path cannot be affected."""
    payload = {"summary": "plain", "signals": ["a", "b"]}
    assert parse_json(json.dumps(payload)) == payload


def test_a_trailing_backslash_does_not_crash():
    with pytest.raises(AnalysisError):
        parse_json('{"summary": "ends with \\')


def test_still_fails_on_json_that_is_broken_some_other_way():
    """The repair is narrow on purpose: it fixes escapes, not structure."""
    with pytest.raises(AnalysisError, match="malformed JSON"):
        parse_json('{"summary": "a", "signals": [1,,2]}')


def test_a_parse_error_carries_the_text_around_the_fault():
    """The class of fault is not enough: a delimiter error is true of many replies and
    identifies none of them. Uses damage no repair covers, so the message is what is
    under test rather than the recovery."""
    bad = '{"summary": "ok", "signals": ["a" "b"], "keywords": []}'
    with pytest.raises(AnalysisError) as exc:
        parse_json(bad)
    assert '"a" "b"' in str(exc.value)


def test_the_excerpt_is_centred_on_the_fault_not_the_start():
    """A fault 1 500 characters into a reply must still appear in the message."""
    filler = "x" * 1500
    bad = '{"summary": "' + filler + '", "signals": ["a" "needle"], "keywords": []}'
    with pytest.raises(AnalysisError) as exc:
        parse_json(bad)
    assert "needle" in str(exc.value)
    assert len(str(exc.value)) < 300


# --- schema-anchored quote repair ---------------------------------------------


def test_a_quoted_term_inside_a_summary_is_recovered():
    """The residual failure class after the backslash fix: an interior quote."""
    raw = '{"summary": "the study of so-called "smart" grids", "signals": []}'
    assert parse_json(raw)["summary"] == 'the study of so-called "smart" grids'


def test_the_two_repairs_compose():
    raw = r'{"summary": "we bound \alpha for "hard" cases", "signals": []}'
    assert parse_json(raw)["summary"] == r'we bound \alpha for "hard" cases'


def test_a_quote_in_a_nested_string_value_is_recovered():
    raw = (
        '{"summary": "ok", "classification": {"theme": "the "new" economy", '
        '"maturity": "growing", "impact": "low", "time_horizon": "mid"}}'
    )
    assert parse_json(raw)["classification"]["theme"] == 'the "new" economy'


def test_valid_json_is_never_rewritten():
    """The repair runs only after a failure, so a good reply cannot be altered."""
    payload = {"summary": 'he said "hi"', "signals": ["a"], "keywords": []}
    assert parse_json(json.dumps(payload)) == payload


def test_structural_damage_is_still_an_error():
    """The repair fixes quoting, not syntax: it must not paper over a broken object."""
    with pytest.raises(AnalysisError, match="malformed JSON"):
        parse_json('{"summary": "a", "signals": [1,,2]}')


def test_an_unrecognised_shape_falls_through_to_the_error():
    """Nothing in the schema to anchor on means no repair, not a wrong repair."""
    with pytest.raises(AnalysisError):
        parse_json('{"notes": "a "quoted" word", "other": 1}')


def test_repair_leaves_array_values_alone():
    """Only declared string values are touched; a quote inside signals is out of scope
    and must fail loudly rather than be silently mangled."""
    with pytest.raises(AnalysisError):
        parse_json('{"summary": "ok", "signals": ["a "quoted" item"]}')
