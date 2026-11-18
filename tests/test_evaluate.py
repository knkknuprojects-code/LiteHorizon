import json

from horizon.training import evaluate
from horizon.training.export import completion_dict
from tests.test_export import GOOD, _analysis

GOLD_JSON = json.dumps(GOOD)


def _gold():
    return evaluate.gold_fields_from_completion(GOLD_JSON)


# --- valid-JSON rate ----------------------------------------------------------


def test_valid_json_rate_mixes_good_and_garbage():
    outputs = [GOLD_JSON, "not json at all", json.dumps({"summary": "x"})]
    # only the first parses AND validates into the full schema
    assert evaluate.valid_json_rate(outputs) == 1 / 3


def test_valid_json_rate_empty_is_zero():
    assert evaluate.valid_json_rate([]) == 0.0


def test_valid_json_rate_tolerates_prose_wrapped_json():
    wrapped = f"Here is the analysis:\n```json\n{GOLD_JSON}\n```"
    assert evaluate.valid_json_rate([wrapped]) == 1.0


# --- enum accuracy ------------------------------------------------------------


def test_enum_accuracy_all_correct():
    pairs = [(GOLD_JSON, _gold())]
    assert evaluate.enum_accuracy(pairs) == 1.0


def test_enum_accuracy_partial():
    wrong = dict(GOOD)
    wrong["classification"] = {**GOOD["classification"], "impact": "low"}
    pairs = [(json.dumps(wrong), _gold())]
    # 2 of 3 enum fields match
    assert evaluate.enum_accuracy(pairs) == 2 / 3


def test_enum_accuracy_ignores_unparseable():
    assert evaluate.enum_accuracy([("garbage", _gold())]) == 0.0


# --- set F1 -------------------------------------------------------------------


def test_set_f1_exact_match():
    assert evaluate.set_f1(["a", "B"], ["b", "A"]) == 1.0


def test_set_f1_both_empty_is_one():
    assert evaluate.set_f1([], []) == 1.0


def test_set_f1_one_empty_is_zero():
    assert evaluate.set_f1(["a"], []) == 0.0


def test_set_f1_partial_overlap():
    # pred {a,b}, gold {b,c}: precision 0.5, recall 0.5 -> F1 0.5
    assert evaluate.set_f1(["a", "b"], ["b", "c"]) == 0.5


# --- entity / keyword F1 ------------------------------------------------------


def test_keyword_f1_perfect_on_self():
    pairs = [(GOLD_JSON, _gold())]
    assert evaluate.keyword_f1(pairs) == 1.0


def test_entity_f1_perfect_on_self():
    pairs = [(GOLD_JSON, _gold())]
    assert evaluate.entity_f1(pairs) == 1.0


# --- aggregate + compare ------------------------------------------------------


def test_evaluate_outputs_self_consistency():
    pairs = [(GOLD_JSON, _gold())]
    metrics = evaluate.evaluate_outputs(pairs)
    assert metrics["n"] == 1
    assert metrics["valid_json_rate"] == 1.0
    assert metrics["enum_accuracy"] == 1.0
    assert metrics["entity_f1"] == 1.0
    assert metrics["keyword_f1"] == 1.0


def test_compare_reports_deltas():
    base = {"valid_json_rate": 0.5, "enum_accuracy": 0.4, "n": 2}
    tuned = {"valid_json_rate": 0.9, "enum_accuracy": 0.4, "n": 2}
    table = evaluate.compare(base, tuned)
    assert table["valid_json_rate"]["delta"] == 0.4
    assert table["enum_accuracy"]["delta"] == 0.0
    assert "n" not in table  # non-float metrics are skipped


# --- runner -------------------------------------------------------------------


def test_evaluate_models_base_only_without_tuned(monkeypatch, make_client):
    # No adapter/tuned model -> only the base is evaluated.
    ex = {
        "content_hash": "h",
        "messages": [{"role": "user", "content": "x"}],
        "completion": GOLD_JSON,
    }
    monkeypatch.setattr(
        evaluate, "_build_client", lambda *a, **k: make_client(GOLD_JSON)
    )
    base_m, tuned_m = evaluate.evaluate_models("base", [ex])
    assert tuned_m is None
    assert base_m["valid_json_rate"] == 1.0


def test_evaluate_models_runs_tuned_when_adapter_given(monkeypatch, make_client):
    ex = {
        "content_hash": "h",
        "messages": [{"role": "user", "content": "x"}],
        "completion": GOLD_JSON,
    }
    built = []

    def fake_build(base_model, *, adapter_path=None, tuned_model=None, device=None):
        built.append(adapter_path)
        return make_client(GOLD_JSON)

    monkeypatch.setattr(evaluate, "_build_client", fake_build)
    base_m, tuned_m = evaluate.evaluate_models("base", [ex], adapter_path="/a")
    assert tuned_m is not None
    assert built == [None, "/a"]  # base client, then tuned client with the adapter


# --- formatting ---------------------------------------------------------------


def test_format_comparison_base_only():
    base = {
        "n": 3,
        "valid_json_rate": 1.0,
        "enum_accuracy": 0.5,
        "entity_f1": 0.0,
        "keyword_f1": 1.0,
    }
    out = evaluate.format_comparison(base)
    assert "Evaluated on 3" in out
    assert "valid JSON" in out and "tuned" not in out


def test_format_comparison_table_with_verdict():
    base = {
        "n": 2,
        "valid_json_rate": 0.5,
        "enum_accuracy": 0.5,
        "entity_f1": 0.5,
        "keyword_f1": 0.5,
    }
    tuned = {
        "n": 2,
        "valid_json_rate": 0.9,
        "enum_accuracy": 0.5,
        "entity_f1": 0.4,
        "keyword_f1": 0.6,
    }
    out = evaluate.format_comparison(base, tuned)
    assert "delta" in out
    assert "+0.400" in out  # valid JSON improved
    assert "wins on 2/4 metrics" in out  # valid JSON + keyword up, entity down


# --- CLI ----------------------------------------------------------------------


def test_evaluate_main_errors_without_eval_split(monkeypatch, tmp_path, capsys):
    rc = evaluate.main(["--base", "m", "--data", str(tmp_path)])
    assert rc == 1
    assert "no eval split" in capsys.readouterr().err


def test_evaluate_main_prints_table(monkeypatch, tmp_path, capsys):
    (tmp_path / "eval.jsonl").write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "x"}], "completion": GOLD_JSON}
        )
        + "\n",
        encoding="utf-8",
    )
    base = {
        "n": 1,
        "valid_json_rate": 1.0,
        "enum_accuracy": 1.0,
        "entity_f1": 1.0,
        "keyword_f1": 1.0,
    }
    monkeypatch.setattr(evaluate, "evaluate_models", lambda *a, **k: (base, None))
    rc = evaluate.main(["--base", "m", "--data", str(tmp_path)])
    assert rc == 0
    assert "Evaluated on 1" in capsys.readouterr().out


def test_run_predictions_uses_folded_prompt(make_client):
    from horizon.training.export import build_prompt_messages

    ex = {
        "content_hash": "h",
        "messages": build_prompt_messages("source text here"),
        "completion": json.dumps(completion_dict(_analysis())),
    }
    client = make_client(GOLD_JSON)
    pairs = evaluate.run_predictions(client, [ex])
    assert len(pairs) == 1
    # the client saw the folded inference prompt
    (sent,) = client.calls
    assert "source text here" in sent[0].content
    # and the gold side parses
    metrics = evaluate.evaluate_outputs(pairs)
    assert metrics["valid_json_rate"] == 1.0
