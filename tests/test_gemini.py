"""Tests for the Gemini API teacher backend and the client factory.

No SDK or network needed: ``GeminiClient`` accepts an injected client object, and
``make_client`` only imports concrete backends lazily.
"""

from __future__ import annotations

import types

import pytest

from horizon.config import GenerationConfig, Settings
from horizon.llm import make_client
from horizon.llm.base import Message
from horizon.llm.gemini_client import (
    GeminiClient,
    _gen_config,
    _is_transient,
    to_gemini,
)


class FakeGenai:
    """Stand-in for a ``genai.Client``: records the last call, returns canned text.

    ``replies`` may be strings (returned in order) or exceptions (raised). Mirrors
    the ``client.models.generate_content(...)`` shape the backend calls.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.models = types.SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(item, Exception):
            raise item
        return types.SimpleNamespace(text=item, candidates=[])


class Boom(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.code = code


# --- message / config mapping -------------------------------------------------


def test_to_gemini_keeps_system_separate_and_maps_roles():
    system, contents = to_gemini(
        [
            Message("system", "be terse"),
            Message("user", "hi"),
            Message("assistant", "yo"),
            Message("user", "again"),
        ]
    )
    assert system == "be terse"
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "yo"}]},
        {"role": "user", "parts": [{"text": "again"}]},
    ]


def test_to_gemini_joins_multiple_system_turns():
    system, contents = to_gemini([Message("system", "a"), Message("system", "b")])
    assert system == "a\n\nb"
    assert contents == []


def test_gen_config_maps_generation_and_defaults_to_json_mode():
    cfg = _gen_config(
        GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.9),
        system="sys",
        json_mode=True,
        thinking_budget=0,
    )
    assert cfg["max_output_tokens"] == 128
    assert cfg["temperature"] == 0.3
    assert cfg["top_p"] == 0.9
    assert cfg["system_instruction"] == "sys"
    assert cfg["response_mime_type"] == "application/json"
    # thinking off by default so reasoning tokens don't truncate the JSON.
    assert cfg["thinking_config"] == {"thinking_budget": 0}


def test_gen_config_without_json_mode_or_system():
    cfg = _gen_config(
        GenerationConfig(), system=None, json_mode=False, thinking_budget=None
    )
    assert "response_mime_type" not in cfg
    assert "system_instruction" not in cfg
    # thinking_budget=None leaves the model default (no thinking_config key).
    assert "thinking_config" not in cfg


# --- generate -----------------------------------------------------------------


def test_generate_calls_api_and_returns_stripped_text():
    fake = FakeGenai(['  {"ok": true}  '])
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"), client=fake)

    out = client.complete("scan this", system="be a scanner")

    assert out == '{"ok": true}'
    call = fake.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "scan this"}]}]
    assert call["config"]["system_instruction"] == "be a scanner"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["thinking_config"] == {"thinking_budget": 0}


def test_generate_retries_transient_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(
        "horizon.llm.gemini_client.time.sleep", lambda s: slept.append(s)
    )
    fake = FakeGenai([Boom(503), Boom(429), "recovered"])
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"), client=fake)

    assert client.complete("x") == "recovered"
    assert len(fake.calls) == 3
    assert len(slept) == 2  # one sleep before each retry


def test_generate_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("horizon.llm.gemini_client.time.sleep", lambda s: None)
    fake = FakeGenai([Boom(400), "unreached"])
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"), client=fake)

    with pytest.raises(Boom):
        client.complete("x")
    assert len(fake.calls) == 1  # no retry on a 400


def test_generate_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("horizon.llm.gemini_client.time.sleep", lambda s: None)
    fake = FakeGenai([Boom(503), Boom(503), Boom(503), Boom(503), Boom(503)])
    client = GeminiClient(
        Settings(model_id="gemini-3.5-flash"), client=fake, max_retries=2
    )

    with pytest.raises(Boom):
        client.complete("x")
    assert len(fake.calls) == 3  # initial + 2 retries


def test_generate_raises_on_empty_text():
    fake = FakeGenai([""])
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"), client=fake)
    with pytest.raises(RuntimeError, match="no text"):
        client.complete("x")


def _inject_fake_sdk(monkeypatch, seen_keys):
    """Make ``from google import genai`` resolve to a fake, without the real SDK."""
    import sys

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = lambda api_key: seen_keys.append(api_key) or FakeGenai(["{}"])
    google_pkg = types.ModuleType("google")
    google_pkg.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _inject_fake_sdk(monkeypatch, [])  # SDK present, but no key configured
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"))
    with pytest.raises(RuntimeError, match="API key"):
        client.complete("x")


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key-123")
    seen: list[str] = []
    _inject_fake_sdk(monkeypatch, seen)
    client = GeminiClient(Settings(model_id="gemini-3.5-flash"))
    client.complete("x")
    assert seen == ["env-key-123"]


def test_is_transient_classification():
    assert _is_transient(Boom(429))
    assert _is_transient(Boom(503))
    assert not _is_transient(Boom(400))
    assert not _is_transient(Boom(401))


# --- factory ------------------------------------------------------------------


def test_make_client_routes_gemini_ids_to_gemini_backend():
    client = make_client(Settings(model_id="gemini-3.5-flash"))
    assert isinstance(client, GeminiClient)


def test_make_client_defaults_to_transformers_backend():
    # device set so construction doesn't import torch for auto-detection.
    client = make_client(Settings(model_id="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"))
    from horizon.llm.transformers_client import TransformersClient

    assert isinstance(client, TransformersClient)
