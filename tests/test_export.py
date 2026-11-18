import json

from horizon import prompts
from horizon.analysis import Analysis
from horizon.training import export

GOOD = {
    "summary": "A new open model was released.",
    "signals": ["state-of-the-art reasoning"],
    "classification": {
        "theme": "open models",
        "maturity": "emerging",
        "impact": "high",
        "time_horizon": "near",
    },
    "entities": {
        "organizations": ["A cloud provider"],
        "people": [],
        "technologies": ["LLM"],
        "locations": [],
    },
    "keywords": ["open-weights", "reasoning"],
}


def _analysis(content_hash="h", **content_over):
    data = {
        **GOOD,
        **content_over,
        "content_hash": content_hash,
        "model_id": "fake",
        "prompt_version": prompts.PROMPT_VERSION,
        "strategy": "combined",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "item_date": "2026-01-01T00:00:00+00:00",
    }
    return Analysis.from_dict(data)


# --- prompt / completion shape ------------------------------------------------


def test_build_prompt_messages_matches_inference_prompt():
    msgs = export.build_prompt_messages("some source text")
    # to_chat folds the system prompt into a single user turn.
    assert [m["role"] for m in msgs] == ["user"]
    content = msgs[0]["content"]
    assert prompts.EXTRACT_SYSTEM_PROMPT in content
    assert "some source text" in content


def test_completion_dict_drops_provenance_and_keeps_enum_values():
    comp = export.completion_dict(_analysis())
    assert set(comp) == {"summary", "signals", "classification", "entities", "keywords"}
    assert comp["classification"]["impact"] == "high"
    assert "content_hash" not in comp
    assert "generated_at" not in comp


# --- filtering ----------------------------------------------------------------


def test_is_trainable_accepts_real_content():
    assert export.is_trainable(_analysis())


def test_is_trainable_rejects_placeholder_summary():
    assert not export.is_trainable(_analysis(summary="2-3 sentence summary"))


def test_is_trainable_rejects_empty_signals():
    assert not export.is_trainable(_analysis(signals=[]))


def test_is_trainable_rejects_placeholder_only_keywords():
    assert not export.is_trainable(_analysis(keywords=["keyword", "..."]))


# --- example building ---------------------------------------------------------


def test_build_examples_skips_no_source_and_not_trainable():
    analyses = [
        _analysis(content_hash="a"),  # good + has source
        _analysis(content_hash="b"),  # good but no source
        _analysis(content_hash="c", summary=""),  # not trainable
    ]
    sources = {"a": "text a"}
    examples, skipped = export.build_examples(analyses, sources)
    assert [e.content_hash for e in examples] == ["a"]
    assert skipped == {
        "not_trainable": 1,
        "no_source": 1,
        "not_approved": 0,
        "wrong_source": 0,
    }
    # completion round-trips to the canonical content JSON.
    assert json.loads(examples[0].completion)["summary"] == GOOD["summary"]


def test_build_examples_approved_only_keeps_vetted():
    analyses = [
        _analysis(content_hash="a", approved=True),
        _analysis(content_hash="b"),  # unapproved
    ]
    sources = {"a": "text a", "b": "text b"}
    examples, skipped = export.build_examples(analyses, sources, approved_only=True)
    assert [e.content_hash for e in examples] == ["a"]
    assert skipped["not_approved"] == 1


def test_build_examples_source_filter():
    analyses = [
        _analysis(content_hash="a", source="teacher"),
        _analysis(content_hash="b", source="model"),
    ]
    sources = {"a": "text a", "b": "text b"}
    examples, skipped = export.build_examples(analyses, sources, source="teacher")
    assert [e.content_hash for e in examples] == ["a"]
    assert skipped["wrong_source"] == 1


def test_export_dataset_records_filters_in_manifest(tmp_path):
    analyses = [_analysis(content_hash="00000000a", approved=True)]
    sources = {"00000000a": "text one"}
    manifest = export.export_dataset(
        analyses, sources, tmp_path / "sft", approved_only=True, source="model"
    )
    assert manifest["approved_only"] is True
    assert manifest["source"] == "model"


# --- split --------------------------------------------------------------------


def _ex(content_hash):
    return export.SFTExample(content_hash=content_hash, messages=[], completion="{}")


def test_split_is_leak_free_and_deterministic():
    # "00.." buckets near 0 (eval), "ff.." near 1 (train).
    low = _ex("00000000" + "x")
    high = _ex("ffffffff" + "x")
    train, held = export.split_examples([low, high], eval_fraction=0.2)
    assert [e.content_hash for e in held] == [low.content_hash]
    assert [e.content_hash for e in train] == [high.content_hash]
    # deterministic: same input, same split
    assert export.split_examples([low, high], eval_fraction=0.2) == (train, held)


def test_split_guarantees_both_nonempty_for_smoke_set():
    # Two examples that both bucket to train; the guard moves one to eval.
    a = _ex("ffffffff" + "1")
    b = _ex("fffffffe" + "2")
    train, held = export.split_examples([a, b], eval_fraction=0.2)
    assert len(train) == 1 and len(held) == 1


# --- export_dataset -----------------------------------------------------------


def test_export_dataset_writes_splits_and_manifest(tmp_path):
    analyses = [
        _analysis(content_hash="00000000a"),
        _analysis(content_hash="ffffffffb"),
    ]
    sources = {"00000000a": "text one", "ffffffffb": "text two"}
    out = tmp_path / "sft"
    manifest = export.export_dataset(analyses, sources, out)

    assert (out / "train.jsonl").is_file()
    assert (out / "eval.jsonl").is_file()
    assert manifest["n_train"] == 1 and manifest["n_eval"] == 1
    assert manifest["prompt_version"] == prompts.PROMPT_VERSION
    written = json.loads((out / "manifest.json").read_text())
    assert written["n_eval"] == 1


def test_export_dataset_smoke_caps_examples(tmp_path):
    analyses = [_analysis(content_hash=f"{i:08x}c") for i in range(10)]
    sources = {a.content_hash: f"text {a.content_hash}" for a in analyses}
    manifest = export.export_dataset(analyses, sources, tmp_path / "sft", smoke=3)
    assert manifest["n_train"] + manifest["n_eval"] == 3
