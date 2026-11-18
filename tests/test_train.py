import json

import pytest
import torch

from horizon.training import train
from horizon.training.train import TrainConfig


class FakeTokenizer:
    """A Gemma-like chat template, enough to exercise the pure helpers."""

    eos_token = "<eos>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        text = "".join(
            f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        return text


class NoGenTokenizer:
    """A template that ignores add_generation_prompt (no derivable response template)."""

    eos_token = "</s>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        return "".join(m["content"] for m in messages)


# --- load / hash --------------------------------------------------------------


def test_load_examples_reads_jsonl(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(
        json.dumps({"content_hash": "a", "messages": [], "completion": "{}"}) + "\n\n",
        encoding="utf-8",
    )
    examples = train.load_examples(p)
    assert len(examples) == 1
    assert examples[0]["content_hash"] == "a"


def test_data_hash_is_deterministic_and_content_sensitive():
    a = [{"content_hash": "x"}, {"content_hash": "y"}]
    assert train.data_hash(a) == train.data_hash(a)
    assert train.data_hash(a) != train.data_hash([{"content_hash": "y"}])


# --- response template / text -------------------------------------------------


def test_response_template_derives_assistant_prefix():
    assert train.response_template(FakeTokenizer()) == "<start_of_turn>model\n"


def test_response_template_raises_without_generation_prompt():
    with pytest.raises(ValueError):
        train.response_template(NoGenTokenizer())


def test_to_text_appends_completion_and_eos_after_prompt():
    record = {"messages": [{"role": "user", "content": "hi"}], "completion": '{"a":1}'}
    text = train.to_text(FakeTokenizer(), record)
    # prompt (with the generation prefix) then completion then eos
    assert text.endswith('<start_of_turn>model\n{"a":1}<eos>')
    assert "<start_of_turn>user\nhi" in text


# --- device / quantization knobs ----------------------------------------------


def test_resolve_train_device_explicit_wins():
    assert train.resolve_train_device("cpu") == "cpu"


def test_resolve_train_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert train.resolve_train_device() == "cuda"


def test_resolve_train_device_never_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # No mps branch at all -> cpu, even on hardware where mps exists.
    assert train.resolve_train_device() == "cpu"


def test_resolve_qlora_false_on_cpu():
    assert train.resolve_qlora(True, "cpu") is False
    assert train.resolve_qlora(None, "cpu") is False


def test_resolve_qlora_auto_enables_with_bitsandbytes(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda n: object())
    assert train.resolve_qlora(None, "cuda") is True


def test_resolve_qlora_raises_when_requested_without_bnb(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None)
    with pytest.raises(RuntimeError):
        train.resolve_qlora(True, "cuda")


# --- run card -----------------------------------------------------------------


def test_build_run_card_records_provenance(tmp_path):
    cfg = TrainConfig(base_model="m", data_dir=tmp_path, output_dir=tmp_path / "out")
    card = train.build_run_card(
        cfg,
        n_train=3,
        n_eval=1,
        dataset_digest="deadbeef",
        prompt_version="phase2.v1",
        device="cpu",
        qlora=False,
        created_at="2026-06-30T00:00:00+00:00",
    )
    assert card["base_model"] == "m"
    assert card["dataset_hash"] == "deadbeef"
    assert card["prompt_version"] == "phase2.v1"
    assert card["device"] == "cpu" and card["qlora"] is False
    assert card["hyperparams"]["lora_r"] == 16
    # paths serialize as strings
    assert isinstance(card["hyperparams"]["output_dir"], str)


# --- CLI wiring ---------------------------------------------------------------


def test_main_builds_config_and_calls_train(monkeypatch, tmp_path):
    captured = {}

    def fake_train(cfg):
        captured["cfg"] = cfg
        return tmp_path / "out" / "adapter"

    monkeypatch.setattr(train, "train", fake_train)
    rc = train.main(
        [
            "--base",
            "google/gemma-2-2b-it",
            "--out",
            str(tmp_path / "out"),
            "--max-steps",
            "2",
        ]
    )
    assert rc == 0
    assert captured["cfg"].base_model == "google/gemma-2-2b-it"
    assert captured["cfg"].max_steps == 2
