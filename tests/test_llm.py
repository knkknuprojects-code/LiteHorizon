import importlib.machinery
import platform
import sys
import types

import pytest
import torch

from horizon.config import Settings
from horizon.llm.base import LLMClient, Message
from horizon.llm.transformers_client import (
    TransformersClient,
    resolve_device,
    to_chat,
)


def test_complete_builds_system_and_user_messages(make_client):
    client = make_client("ok")
    out = client.complete("hello", system="be brief")
    assert out == "ok"
    (messages,) = client.calls
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "hello"


def test_complete_without_system(make_client):
    client = make_client()
    client.complete("hi")
    (messages,) = client.calls
    assert [m.role for m in messages] == ["user"]


def test_to_chat_folds_system_into_user():
    chat = to_chat([Message("system", "rules"), Message("user", "question")])
    assert chat == [{"role": "user", "content": "rules\n\nquestion"}]


def test_to_chat_maps_assistant_to_model():
    chat = to_chat([Message("user", "hi"), Message("assistant", "yo")])
    assert [turn["role"] for turn in chat] == ["user", "model"]


def test_fake_client_is_an_llm_client(make_client):
    assert isinstance(make_client(), LLMClient)


def test_resolve_device_explicit_preferred_wins():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"


def test_resolve_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == "cuda"


def test_resolve_device_mps_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert resolve_device() == "mps"


def test_resolve_device_intel_mac_falls_back_to_cpu(monkeypatch):
    # MPS is available (Metal/AMD) but on x86 it's slower than CPU for generation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert resolve_device() == "cpu"


# --- adapter loading (mock the heavy loads — no real model) -------------------


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.moved_to = None
        self.eval_called = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def _stub_loaders(monkeypatch, base, tokenizer="tok"):
    """Patch the lazily-imported transformers loaders to return ``base``.

    Returns a dict that captures the ``from_pretrained`` kwargs, so tests can assert
    on *how* the model was loaded (dtype, device_map, quantization) without a model.
    """
    import transformers

    captured: dict = {}

    def load_model(mid, **kwargs):
        captured["model_id"] = mid
        captured["kwargs"] = kwargs
        return base

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda mid: tokenizer),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=load_model),
    )
    return captured


@pytest.fixture
def peft_module(monkeypatch):
    """A module named ``peft`` for the adapter tests to patch ``PeftModel`` on.

    ``_apply_adapter`` does ``from peft import PeftModel`` lazily, so a stand-in in
    ``sys.modules`` drives the same code path the real package would. Importing the
    real one instead makes these tests fail wherever peft is absent — which is every
    GPU box, since requirements-gpu.txt leaves it out on purpose, and the suite is
    meant to run without the ML stack.
    """
    try:
        import peft

        return peft
    except ImportError:
        # Import transformers before the stub lands: it computes _peft_available via
        # importlib.util.find_spec("peft") at import time and caches it, so a stub
        # visible then would flip that flag for the whole session.
        import transformers  # noqa: F401

        stub = types.ModuleType("peft")
        # A module with __spec__ = None makes find_spec() raise ValueError rather
        # than report absence, which breaks any later caller that probes for peft.
        stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
        stub.PeftModel = types.SimpleNamespace(from_pretrained=None)
        monkeypatch.setitem(sys.modules, "peft", stub)
        return stub


def test_client_applies_adapter_when_set(monkeypatch, peft_module):
    base = _FakeModel("base")
    adapted = _FakeModel("adapted")
    _stub_loaders(monkeypatch, base)

    calls = {}

    def fake_from_pretrained(model, path):
        calls["model"] = model
        calls["path"] = path
        return adapted

    monkeypatch.setattr(
        peft_module,
        "PeftModel",
        types.SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    client = TransformersClient(
        Settings(model_id="base", device="cpu", adapter_path="/tmp/adapter")
    )
    client._ensure_loaded()

    assert calls["model"] is base  # adapter applied on top of the base
    assert calls["path"] == "/tmp/adapter"
    assert client._model is adapted
    assert adapted.moved_to == "cpu" and adapted.eval_called


def test_client_without_adapter_does_not_touch_peft(monkeypatch, peft_module):
    base = _FakeModel("base")
    _stub_loaders(monkeypatch, base)

    def boom(*a, **k):
        raise AssertionError("PeftModel used without an adapter set")

    monkeypatch.setattr(
        peft_module, "PeftModel", types.SimpleNamespace(from_pretrained=boom)
    )

    client = TransformersClient(Settings(model_id="base", device="cpu"))
    client._ensure_loaded()

    assert client._model is base
    assert base.moved_to == "cpu" and base.eval_called


def test_template_kwargs_disables_thinking_when_supported():
    # Qwen3/Qwen3.5-style template: their chat template branches on enable_thinking.
    client = TransformersClient(Settings(model_id="Qwen/Qwen3-1.7B", device="cpu"))
    client._tokenizer = types.SimpleNamespace(
        chat_template="{% if enable_thinking %}<think>{% endif %}"
    )
    assert client._template_kwargs() == {"enable_thinking": False}


def test_template_kwargs_noop_for_plain_templates():
    # Gemma / Qwen2.5 have no such flag — pass nothing so their templates are untouched.
    client = TransformersClient(Settings(model_id="google/gemma-2-2b-it", device="cpu"))
    client._tokenizer = types.SimpleNamespace(chat_template="{{ messages }}")
    assert client._template_kwargs() == {}
    client._tokenizer = types.SimpleNamespace(chat_template=None)
    assert client._template_kwargs() == {}


def test_settings_from_env_reads_adapter(monkeypatch):
    monkeypatch.setenv("HORIZON_ADAPTER", "/tmp/a")
    assert Settings.from_env().adapter_path == "/tmp/a"
    monkeypatch.delenv("HORIZON_ADAPTER", raising=False)
    assert Settings.from_env().adapter_path is None


# --- load kwargs / pad token (Phase 1 helpers) --------------------------------


def test_dtype_kwarg_follows_transformers_version(monkeypatch):
    import transformers

    from horizon.llm.transformers_client import _dtype_kwarg

    monkeypatch.setattr(transformers, "__version__", "4.46.3")
    assert _dtype_kwarg() == "torch_dtype"
    monkeypatch.setattr(transformers, "__version__", "4.57.6")
    assert _dtype_kwarg() == "dtype"
    # transformers 5.x removed torch_dtype outright; this is the cluster floor.
    monkeypatch.setattr(transformers, "__version__", "5.5.0")
    assert _dtype_kwarg() == "dtype"
    # Pre-release/dev suffixes must not crash the parse.
    monkeypatch.setattr(transformers, "__version__", "5.6.0.dev0")
    assert _dtype_kwarg() == "dtype"


def test_load_uses_the_versioned_dtype_kwarg(monkeypatch):
    from horizon.llm.transformers_client import _dtype_kwarg

    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    client = TransformersClient(Settings(model_id="base", device="cpu"))
    client._ensure_loaded()
    assert captured["kwargs"] == {_dtype_kwarg(): torch.float32}
    assert base.moved_to == "cpu" and base.eval_called


def test_load_on_cuda_uses_bfloat16(monkeypatch):
    from horizon.llm.transformers_client import _dtype_kwarg

    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    client = TransformersClient(Settings(model_id="base", device="cuda"))
    client._ensure_loaded()
    assert captured["kwargs"] == {_dtype_kwarg(): torch.bfloat16}


def test_pad_token_id_keeps_a_zero_pad_token():
    """Gemma's <pad> is id 0 — truthiness would silently swap in EOS."""
    from horizon.llm.transformers_client import _pad_token_id

    tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=1, unk_token_id=3)
    assert _pad_token_id(tok) == 0


def test_pad_token_id_falls_back_to_eos_then_unk():
    from horizon.llm.transformers_client import _pad_token_id

    assert _pad_token_id(types.SimpleNamespace(pad_token_id=None, eos_token_id=2)) == 2
    assert (
        _pad_token_id(
            types.SimpleNamespace(pad_token_id=None, eos_token_id=None, unk_token_id=5)
        )
        == 5
    )


def test_pad_token_id_errors_when_none_declared():
    import pytest

    from horizon.llm.transformers_client import _pad_token_id

    with pytest.raises(RuntimeError, match="no pad, eos or unk"):
        _pad_token_id(types.SimpleNamespace())


# --- batch interface defaults -------------------------------------------------


def test_generate_batch_defaults_to_a_sequential_loop(make_client):
    client = make_client(lambda msgs: msgs[-1].content.upper())
    out = client.generate_batch(
        [[Message("user", "a")], [Message("user", "b")], [Message("user", "c")]]
    )
    assert out == ["A", "B", "C"]  # one result per conversation, in input order
    assert len(client.calls) == 3


def test_generate_batch_empty(make_client):
    client = make_client()
    assert client.generate_batch([]) == []
    assert client.calls == []


def test_complete_batch_builds_the_same_shape_as_complete(make_client):
    """A batch of one must be indistinguishable from ``complete`` at the boundary."""
    batched = make_client("ok")
    batched.complete_batch(["hello"], system="be brief")
    single = make_client("ok")
    single.complete("hello", system="be brief")
    assert batched.calls == single.calls


def test_complete_batch_shares_one_system_across_prompts(make_client):
    client = make_client("ok")
    client.complete_batch(["one", "two"], system="rules")
    assert [[m.role for m in c] for c in client.calls] == [
        ["system", "user"],
        ["system", "user"],
    ]
    assert [c[1].content for c in client.calls] == ["one", "two"]


def test_complete_batch_without_system(make_client):
    client = make_client("ok")
    client.complete_batch(["one", "two"])
    assert [[m.role for m in c] for c in client.calls] == [["user"], ["user"]]


# --- load kwargs for an unknown card ------------------------------------------


def test_load_kwargs_cpu_default_is_exactly_todays_call(monkeypatch):
    """The byte-identical fence: no GPU knob set must mean the old CPU call."""
    from horizon.llm.transformers_client import _dtype_kwarg

    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    client = TransformersClient(Settings(model_id="base", device="cpu"))
    client._ensure_loaded()
    assert captured["kwargs"] == {_dtype_kwarg(): torch.float32}
    assert base.moved_to == "cpu"  # still .to()-moved, not dispatched
    assert client._dispatched is False
    assert client._input_device == "cpu"


def test_resolve_dtype_table():
    from horizon.llm.transformers_client import resolve_dtype

    assert resolve_dtype("cpu") is torch.float32
    assert resolve_dtype("cuda") is torch.bfloat16
    # Explicit wins on either device — fp16 is the pre-Ampere escape hatch.
    assert resolve_dtype("cuda", "float16") is torch.float16
    assert resolve_dtype("cpu", "bfloat16") is torch.bfloat16
    assert resolve_dtype("cuda", "auto") == "auto"


def test_resolve_dtype_rejects_a_typo():
    import pytest

    from horizon.llm.transformers_client import resolve_dtype

    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("cuda", "bflaot16")


def test_device_map_dispatches_and_parses_max_memory(monkeypatch):
    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    client = TransformersClient(
        Settings(
            model_id="base",
            device="cuda",
            device_map="auto",
            max_memory="0=38GiB,cpu=64GiB",
        )
    )
    client._ensure_loaded()
    kwargs = captured["kwargs"]
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: "38GiB", "cpu": "64GiB"}  # int GPU key
    assert kwargs["low_cpu_mem_usage"] is True
    assert base.moved_to is None  # dispatched models must not be .to()-moved
    assert client._dispatched is True


def test_load_4bit_quantizes_pins_the_device_and_skips_to(monkeypatch):
    import transformers

    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    monkeypatch.setattr(
        transformers, "BitsAndBytesConfig", lambda **kw: ("bnb", kw), raising=False
    )
    client = TransformersClient(
        Settings(model_id="base", device="cuda", load_in_4bit=True)
    )
    client._ensure_loaded()
    kwargs = captured["kwargs"]
    tag, conf = kwargs["quantization_config"]
    assert tag == "bnb"
    assert conf["load_in_4bit"] is True
    assert conf["bnb_4bit_quant_type"] == "nf4"
    assert conf["bnb_4bit_use_double_quant"] is True
    assert conf["bnb_4bit_compute_dtype"] is torch.bfloat16
    # bnb owns the weight dtype, so no dtype kwarg rides along.
    assert "dtype" not in kwargs and "torch_dtype" not in kwargs
    assert kwargs["device_map"] == {"": "cuda"}
    assert base.moved_to is None
    assert client._dispatched is True


def test_load_4bit_off_cuda_errors_clearly(monkeypatch):
    import pytest

    _stub_loaders(monkeypatch, _FakeModel("base"))
    client = TransformersClient(
        Settings(model_id="base", device="cpu", load_in_4bit=True)
    )
    with pytest.raises(RuntimeError, match="quantized loading needs CUDA"):
        client._quantization_config()


def test_attn_implementation_is_passed_through(monkeypatch):
    base = _FakeModel("base")
    captured = _stub_loaders(monkeypatch, base)
    client = TransformersClient(
        Settings(model_id="base", device="cuda", attn_implementation="sdpa")
    )
    client._ensure_loaded()
    assert captured["kwargs"]["attn_implementation"] == "sdpa"


def test_attn_falls_back_to_sdpa_then_default(monkeypatch, capsys):
    """flash-attn may not build on an unknown card; a warning beats a crash."""
    import transformers

    base = _FakeModel("base")
    seen: list = []

    def load_model(mid, **kwargs):
        attn = kwargs.get("attn_implementation")
        seen.append(attn)
        if attn == "flash_attention_2":
            raise ImportError("flash-attn not installed")
        return base

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda mid: "tok"),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=load_model),
    )
    client = TransformersClient(
        Settings(
            model_id="base", device="cuda", attn_implementation="flash_attention_2"
        )
    )
    client._ensure_loaded()
    assert seen == ["flash_attention_2", "sdpa"]
    assert "flash_attention_2" in capsys.readouterr().err
    assert client._model is base


def test_to_chat_folds_system_for_every_template():
    """One PROMPT_VERSION means one prompt shape: fold uniformly, even where a
    template (Gemma 3/4, Qwen) would accept a native system turn."""
    assert to_chat([Message("system", "rules"), Message("user", "q")]) == [
        {"role": "user", "content": "rules\n\nq"}
    ]


# --- batched generation (real tensors, no weights) ----------------------------


class _CharTokenizer:
    """A char-level tokenizer stub: enough of the real API to exercise padding.

    Token id is ``ord(char)``; 0 is the pad id, which is deliberately *falsy* so the
    pad-token bug can't come back unnoticed.
    """

    pad_token_id = 0
    eos_token_id = 1
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.padding_side = "right"
        self.truncation_side = "right"
        self.seen_padding_side: list[str] = []

    def apply_chat_template(self, chat, *, add_generation_prompt=True, **kwargs):
        text = "".join(turn["content"] for turn in chat)
        if kwargs.get("tokenize") is False:
            return text
        ids = torch.tensor([[ord(c) for c in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __call__(self, texts, *, return_tensors=None, padding=False, **kwargs):
        self.seen_padding_side.append(self.padding_side)
        rows = [[ord(c) for c in t] for t in texts]
        if kwargs.get("truncation"):
            limit = kwargs["max_length"]
            # truncation_side="left" keeps the tail, where the generation prompt lives
            rows = [
                r[-limit:] if self.truncation_side == "left" else r[:limit]
                for r in rows
            ]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [self.pad_token_id] * (width - len(r))
            if self.padding_side == "left":
                ids.append(pad + r)
                mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad)
                mask.append([1] * len(r) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(
            chr(int(i)) for i in ids if not (skip_special_tokens and int(i) <= 1)
        )


class _EchoModel(_FakeModel):
    """Appends each row's final real token twice — a per-row identifiable continuation.

    With left padding every row's last token is its own, so a correct implementation
    yields a distinct continuation per row; a wrong slice or a lost ordering shows up
    immediately.
    """

    def __init__(self) -> None:
        super().__init__("echo")
        self.batch_sizes: list[int] = []

    def generate(self, *, input_ids, attention_mask=None, **kwargs):
        self.batch_sizes.append(int(input_ids.shape[0]))
        tail = input_ids[:, -1:]
        return torch.cat([input_ids, tail, tail], dim=1)


def _batch_client(settings, model=None, tokenizer=None):
    client = TransformersClient(settings)
    client._model = model or _EchoModel()
    client._tokenizer = tokenizer or _CharTokenizer()
    client._input_device = "cpu"
    return client


def _convs(*texts):
    return [[Message("user", t)] for t in texts]


def test_generate_batch_slices_each_row_and_restores_input_order():
    """The single most likely batching bug: a wrong slice or a lost ordering."""
    model = _EchoModel()
    client = _batch_client(Settings(device="cpu", batch_size=8), model=model)
    out = client.generate_batch(_convs("aaaX", "bY", "ccZ"))
    assert out == ["XX", "YY", "ZZ"]  # input order, despite long-first sorting
    assert model.batch_sizes == [3]


def test_generate_batch_of_one_delegates_to_generate(monkeypatch):
    """Guards the CPU dev path: a batch of one must never touch the padding code."""
    client = _batch_client(Settings(device="cpu", batch_size=8))
    monkeypatch.setattr(client, "_generate_chunk", lambda *a, **k: 1 / 0)
    called = {}

    def fake_generate(messages, *, config=None):
        called["messages"] = list(messages)
        return "single"

    monkeypatch.setattr(client, "generate", fake_generate)
    assert client.generate_batch(_convs("only")) == ["single"]
    assert called["messages"][0].content == "only"


def test_generate_batch_empty_does_not_load_a_model():
    client = TransformersClient(Settings(device="cpu"))
    assert client.generate_batch([]) == []  # _ensure_loaded never reached


def test_generate_batch_pads_left_and_restores_the_tokenizer():
    tokenizer = _CharTokenizer()
    client = _batch_client(Settings(device="cpu", batch_size=8), tokenizer=tokenizer)
    client.generate_batch(_convs("aaaX", "bY"))
    assert tokenizer.seen_padding_side == ["left"]  # left during encode
    assert tokenizer.padding_side == "right"  # and restored afterwards
    assert tokenizer.truncation_side == "right"


def test_generate_batch_chunks_by_batch_size():
    model = _EchoModel()
    client = _batch_client(Settings(device="cpu", batch_size=2), model=model)
    out = client.generate_batch(_convs("aA", "bbB", "ccccC", "dD", "eE"))
    assert out == ["AA", "BB", "CC", "DD", "EE"]
    assert model.batch_sizes == [2, 2, 1]


def test_generate_batch_left_truncates_to_keep_the_generation_prompt():
    tokenizer = _CharTokenizer()
    client = _batch_client(
        Settings(device="cpu", batch_size=8, max_input_tokens=2), tokenizer=tokenizer
    )
    out = client.generate_batch(_convs("aaaaX", "bbY"))
    # Truncating from the left keeps the tail, so the final token still identifies the row.
    assert out == ["XX", "YY"]
    assert tokenizer.truncation_side == "right"  # restored


def test_generate_batch_halves_the_chunk_on_oom(capsys):
    class _OomOnceModel(_EchoModel):
        def generate(self, *, input_ids, attention_mask=None, **kwargs):
            if input_ids.shape[0] > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().generate(input_ids=input_ids, attention_mask=attention_mask)

    model = _OomOnceModel()
    client = _batch_client(Settings(device="cpu", batch_size=4), model=model)
    out = client.generate_batch(_convs("aA", "bB", "cC", "dD"))
    assert out == ["AA", "BB", "CC", "DD"]  # nothing lost
    assert model.batch_sizes == [2, 2]  # halved and carried forward
    assert "out of memory" in capsys.readouterr().err


def test_generate_batch_reraises_a_real_error():
    import pytest

    class _BrokenModel(_EchoModel):
        def generate(self, **kwargs):
            raise RuntimeError("shapes are wrong")

    client = _batch_client(Settings(device="cpu", batch_size=4), model=_BrokenModel())
    with pytest.raises(RuntimeError, match="shapes are wrong"):
        client.generate_batch(_convs("aA", "bB"))


def test_generate_batch_matches_single_generate():
    """Batch of N must agree with N sequential calls through the real client paths."""
    batched = _batch_client(Settings(device="cpu", batch_size=8))
    convs = _convs("aaaX", "bY", "ccZ")
    got = batched.generate_batch(convs)
    sequential = [
        _batch_client(Settings(device="cpu", batch_size=1)).generate(c) for c in convs
    ]
    assert got == sequential


def test_eight_bit_is_a_separate_rung_from_four_bit(monkeypatch):
    """The 12B weights are 22.3 GiB in bfloat16 against a 24 GiB card, which leaves
    nothing for the KV cache; int8 halves that with room to batch."""
    from dataclasses import replace

    from horizon.config import Settings
    from horizon.llm.transformers_client import TransformersClient

    client = TransformersClient(replace(Settings(), load_in_8bit=True))
    client.device = "cuda"
    config = client._quantization_config()
    assert config.load_in_8bit and not config.load_in_4bit


def test_asking_for_both_quantizations_is_an_error(monkeypatch):
    from dataclasses import replace

    import pytest

    from horizon.config import Settings
    from horizon.llm.transformers_client import TransformersClient

    client = TransformersClient(
        replace(Settings(), load_in_4bit=True, load_in_8bit=True)
    )
    client.device = "cuda"
    with pytest.raises(RuntimeError, match="not both"):
        client._quantization_config()


def test_no_quantization_flag_means_no_config():
    from horizon.config import Settings
    from horizon.llm.transformers_client import TransformersClient

    assert TransformersClient(Settings())._quantization_config() is None
