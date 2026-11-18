"""Hugging Face ``transformers`` backend for ``LLMClient``.

``torch`` and ``transformers`` are imported lazily inside methods so the package
(and its tests) can be imported without the heavy ML stack installed.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence

from horizon.config import GenerationConfig, Settings, parse_max_memory
from horizon.llm.base import LLMClient, Message


def resolve_device(preferred: str | None = None) -> str:
    """Pick an inference device: explicit > cuda > mps (Apple Silicon only) > cpu.

    MPS is preferred only on Apple-Silicon (``arm64``) Macs. On Intel (``x86_64``)
    Macs the Metal/AMD path is unreliable and was measured *slower* than CPU for
    autoregressive generation (~1.8x), so we fall back to CPU there. Override with
    ``HORIZON_DEVICE`` / ``--device``.
    """
    if preferred:
        return preferred
    import platform

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def _dtype_kwarg() -> str:
    """The ``from_pretrained`` dtype kwarg name for the installed transformers.

    Renamed ``torch_dtype`` -> ``dtype`` in 4.56 and the old name was **removed** in
    5.x, which is the floor Gemma 4 needs. We can't just hard-code the new name: the
    ``[train]`` extra pins ``trl<0.13``, which drags transformers back to ~4.46 where
    ``dtype=`` is rejected. So resolve it instead of guessing.
    """
    import transformers

    parts = []
    for chunk in transformers.__version__.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    major, minor = (parts + [0, 0])[:2]
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _pad_token_id(tokenizer) -> int:
    """The id to pad/fill generated sequences with.

    Explicitly ``is None``-checked, not truthiness: Gemma's ``<pad>`` is **id 0**, so
    ``pad_token_id or eos_token_id`` silently substitutes EOS. That is invisible at
    batch size 1 and corrupts every early-finishing row once generation is batched.
    """
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
    ):
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "tokenizer defines no pad, eos or unk token id — cannot pad generated "
        "sequences. Pick a model whose tokenizer declares one."
    )


_DTYPES = ("bfloat16", "float16", "float32")

# torch.cuda.OutOfMemoryError subclasses RuntimeError, and CPU/MPS allocators raise a
# plain RuntimeError, so the type alone can't identify an OOM — the message has to.
_OOM_ERRORS = (RuntimeError,)


def _is_oom(exc: BaseException) -> bool:
    """Whether an exception is an out-of-memory condition rather than a real error."""
    text = str(exc).lower()
    return "out of memory" in text or "alloc" in text and "fail" in text


def resolve_dtype(device: str, requested: str | None = None):
    """Pick the weight dtype: explicit > bf16 on CUDA > fp32.

    ``requested`` is a dtype name (``"bfloat16"``/``"float16"``/``"float32"``, or
    ``"auto"`` to let transformers decide from the checkpoint). ``None`` reproduces the
    built-in rule exactly, so an unset knob leaves the CPU path untouched. An unknown
    name raises rather than silently falling back — a typo on a cluster must not cost a
    30-minute reload at the wrong precision. fp16 is the escape hatch for pre-Ampere
    cards, which have no bf16.
    """
    import torch

    if requested is None:
        return torch.bfloat16 if device == "cuda" else torch.float32
    name = requested.strip().lower()
    if name == "auto":
        return "auto"
    if name not in _DTYPES:
        raise ValueError(
            f"unknown dtype {requested!r}: expected one of {', '.join(_DTYPES)} or auto"
        )
    return getattr(torch, name)


def to_chat(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Convert ``Message``s to a portable ``user``/``model`` chat list.

    Any ``system`` content is folded into the next user turn. Gemma 2's template
    *requires* this (it raises on a ``system`` turn), and we do it **uniformly** —
    including for the templates that would accept one (Gemma 3/4, Qwen) — so that one
    ``PROMPT_VERSION`` means exactly one prompt shape across every model. Emitting a
    native ``system`` turn where it's supported would silently change the effective
    prompt for models whose stored analyses were produced with the folded form, which
    is a measured change for its own phase, not a side effect of a backend choice.
    """
    chat: list[dict[str, str]] = []
    pending_system: str | None = None
    for message in messages:
        if message.role == "system":
            pending_system = message.content
            continue
        role = "model" if message.role in ("assistant", "model") else "user"
        content = message.content
        if pending_system and role == "user":
            content = f"{pending_system}\n\n{content}"
            pending_system = None
        chat.append({"role": role, "content": content})
    return chat


class TransformersClient(LLMClient):
    """Runs a local Gemma via ``transformers``. The model loads on first use."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Store settings and resolve the device; the model loads lazily on first use."""
        self.settings = settings or Settings()
        self.device = resolve_device(self.settings.device)
        self._model = None
        self._tokenizer = None
        # True when accelerate placed the weights (device_map) or bitsandbytes
        # quantized them: both forbid a later ``.to()``.
        self._dispatched = False
        # Where input tensors go. Equals ``self.device`` unless dispatched, in which
        # case the embedding shard decides.
        self._input_device: str | None = None

    def _quantization_config(self):
        """A 4-bit ``BitsAndBytesConfig``, or ``None`` when quantization is off.

        CUDA-only: bitsandbytes ships no CPU/MPS kernels, so a 4-bit request off CUDA is
        an error rather than a silent full-precision load — the operator asked for a
        model that only *fits* quantized. nf4 + double quantization is the settled
        default and is deliberately not exposed as a knob.
        """
        if not (self.settings.load_in_4bit or self.settings.load_in_8bit):
            return None
        if self.settings.load_in_4bit and self.settings.load_in_8bit:
            raise RuntimeError("choose one of --load-4bit and --load-8bit, not both")
        if self.device != "cuda":
            raise RuntimeError(
                "quantized loading needs CUDA (bitsandbytes has no CPU/MPS kernels); "
                f"resolved device is {self.device!r}. Drop --load-4bit/--load-8bit, or "
                "run on a GPU "
                'with: pip install -e ".[infer-gpu]"'
            )
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                'quantized loading needs bitsandbytes: pip install -e ".[infer-gpu]"'
            ) from exc
        compute = resolve_dtype(self.device, self.settings.dtype)
        if self.settings.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(torch.bfloat16 if compute == "auto" else compute),
        )

    def _load_kwargs(self) -> dict:
        """``from_pretrained`` kwargs for this settings/device combination.

        With no GPU knobs set this is exactly ``{dtype: float32}`` on CPU — the same
        call the pre-batching code made. ``low_cpu_mem_usage`` is added *only* alongside
        a device map or quantization, because it switches the load to meta-device init
        and would otherwise change the CPU path.
        """
        quant = self._quantization_config()
        kwargs: dict = {}
        if quant is not None:
            # bitsandbytes owns the weight dtype; the compute dtype rides in the config.
            kwargs["quantization_config"] = quant
        else:
            kwargs[_dtype_kwarg()] = resolve_dtype(self.device, self.settings.dtype)
        if self.settings.device_map:
            kwargs["device_map"] = self.settings.device_map
            if self.settings.max_memory:
                kwargs["max_memory"] = parse_max_memory(self.settings.max_memory)
        elif quant is not None:
            # A quantized load has to be placed at load time; pin it to this GPU.
            kwargs["device_map"] = {"": self.device}
        if "device_map" in kwargs:
            kwargs["low_cpu_mem_usage"] = True
        return kwargs

    def _attn_candidates(self) -> list[str | None]:
        """Attention implementations to try, best first, ending in the library default.

        ``flash_attention_2`` needs a separate package, Ampere+ and a half precision, so
        on an unknown card it may simply not build. Falling back with a warning beats
        crashing after a multi-gigabyte download.
        """
        requested = self.settings.attn_implementation
        if not requested:
            return [None]
        candidates: list[str | None] = [requested]
        if requested != "sdpa":
            candidates.append("sdpa")
        candidates.append(None)
        return candidates

    def _load_tokenizer(self):
        """The tokenizer, falling back to the processor for multimodal checkpoints.

        Gemma 4 is multimodal and its documented text path goes through
        ``AutoProcessor``; ``AutoTokenizer`` is still correct for text-only checkpoints
        (Gemma 2/3, Qwen), so try it first and fall back rather than branching on the
        model id.
        """
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(self.settings.model_id)
        except Exception:
            from transformers import AutoProcessor

            # padding_side matters here: batched decoder-only generation must pad left.
            return AutoProcessor.from_pretrained(
                self.settings.model_id, padding_side="left"
            )

    def _ensure_loaded(self) -> None:
        """Load the tokenizer and model on first call (a no-op afterwards).

        Imports ``torch``/``transformers`` lazily, resolves dtype/quantization/placement
        from settings (defaulting to today's bf16-on-CUDA-else-fp32), applies a LoRA
        adapter if one is configured, places the model, and switches it to eval mode.
        Raises a helpful error if the stack is missing or the model can't load.
        """
        if self._model is not None:
            return
        try:
            import transformers  # noqa: F401  (probed for the loaders below)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "transformers/torch are required to load a model. "
                'Install them with: pip install -e ".[dev]"'
            ) from exc
        from transformers import AutoModelForCausalLM

        load_kwargs = self._load_kwargs()
        self._dispatched = "device_map" in load_kwargs
        try:
            self._tokenizer = self._load_tokenizer()
            model = self._load_model(AutoModelForCausalLM, load_kwargs)
            if self.settings.adapter_path:
                model = self._apply_adapter(model, self.settings.adapter_path)
            # An accelerate-dispatched or bnb-quantized model raises on ``.to()``.
            self._model = model if self._dispatched else model.to(self.device)
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                f"Could not load model {self.settings.model_id!r}"
                + (
                    f" with adapter {self.settings.adapter_path!r}"
                    if self.settings.adapter_path
                    else ""
                )
                + f": {exc}. If it is gated (e.g. Gemma 2/3), accept its license on "
                "Hugging Face and run `huggingface-cli login`."
            ) from exc
        self._input_device = (
            str(getattr(self._model, "device", self.device))
            if self._dispatched
            else self.device
        )
        self._model.eval()

    def _load_model(self, auto_class, load_kwargs: dict):
        """``from_pretrained`` with the attention-implementation fallback chain."""
        last: Exception | None = None
        candidates = self._attn_candidates()
        for attn in candidates:
            kwargs = dict(load_kwargs)
            if attn is not None:
                kwargs["attn_implementation"] = attn
            try:
                return auto_class.from_pretrained(self.settings.model_id, **kwargs)
            except (ImportError, ValueError, RuntimeError, OSError) as exc:
                last = exc
                if attn is not candidates[-1]:
                    print(
                        f"warning: attn_implementation={attn!r} unavailable ({exc}); "
                        "falling back.",
                        file=sys.stderr,
                    )
        raise last if last else RuntimeError("model load failed with no exception")

    @staticmethod
    def _apply_adapter(model, adapter_path: str):
        """Wrap a base model with a PEFT LoRA adapter (Phase 4 fine-tunes)."""
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "loading an adapter (HORIZON_ADAPTER) needs the [train] extra: "
                'pip install -e ".[dev,train]"'
            ) from exc
        return PeftModel.from_pretrained(model, adapter_path)

    def _template_kwargs(self) -> dict[str, bool]:
        """Extra kwargs for ``apply_chat_template``, per the model's template.

        Hybrid *thinking* models (Qwen3/Qwen3.5) default to emitting a
        ``<think>…</think>`` reasoning trace, which both burns the
        ``max_new_tokens`` budget (truncating the JSON we ask for) and prefixes
        the reply with non-JSON. Their chat templates accept ``enable_thinking``;
        we turn it off for structured extraction — the same decision made for the
        Gemini teacher (``thinking_budget=0``). Templates without the flag (Gemma,
        Qwen2.5) are left untouched, so this is a no-op there.
        """
        template = getattr(self._tokenizer, "chat_template", None) or ""
        if "enable_thinking" in template:
            return {"enable_thinking": False}
        return {}

    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text locally with the loaded model (greedy unless sampling is on).

        Builds the chat prompt, encodes it (with attention mask), and decodes only
        the newly generated tokens. Sampling params are passed only when
        ``do_sample`` is set, to avoid transformers' per-call greedy-decode warning.
        """
        import torch

        self._ensure_loaded()
        config = config or self.settings.generation
        chat = to_chat(messages)
        # return_dict gives us the attention_mask too — required for reliable
        # generation when pad and eos tokens coincide.
        encoded = self._tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **self._template_kwargs(),
        )
        encoded = {
            k: v.to(self._input_device or self.device) for k, v in encoded.items()
        }
        input_len = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            output = self._model.generate(**encoded, **self._gen_kwargs(config))
        generated = output[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _gen_kwargs(self, config: GenerationConfig) -> dict:
        """Decoding kwargs for ``model.generate``, shared by the single and batch paths.

        One place, so the pad-token and sampling rules cannot drift between them.
        Greedy decoding takes no sampling params: passing temperature/top_p while
        ``do_sample=False`` makes transformers warn on every call.
        """
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": _pad_token_id(self._tokenizer),
        }
        if config.do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
        return gen_kwargs

    @contextlib.contextmanager
    def _padding(self, truncate: bool):
        """Temporarily switch the tokenizer to left padding (and left truncation).

        Decoder-only batched generation **must** pad on the left: it puts every row's
        real tokens flush against the generation position, so one ``input_ids.shape[1]``
        slices every row correctly. Truncation must also come off the left, or it would
        cut the trailing generation prompt and destroy the chat format. Restored in a
        ``finally`` so the single-item path and any shared tokenizer never inherit it.
        """
        tok = self._tokenizer
        previous = (
            getattr(tok, "padding_side", None),
            getattr(tok, "truncation_side", None),
        )
        try:
            tok.padding_side = "left"
            if truncate:
                tok.truncation_side = "left"
            yield tok
        finally:
            if previous[0] is not None:
                tok.padding_side = previous[0]
            if previous[1] is not None:
                tok.truncation_side = previous[1]

    def _render(self, messages: Sequence[Message]) -> str:
        """The prompt string for one conversation, per the model's chat template."""
        return self._tokenizer.apply_chat_template(
            to_chat(messages),
            add_generation_prompt=True,
            tokenize=False,
            **self._template_kwargs(),
        )

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate one completion per conversation, batching the forward passes.

        This is the GPU throughput lever: a card processes many sequences per forward
        pass, while CPU generation is memory-bandwidth bound and gains nothing. Sizing
        comes from ``Settings.batch_size``.

        A batch of one delegates to :meth:`generate`, so the single-item path — the CPU
        dev default and ``horizon scan`` — is untouched. Longer batches are sorted by
        length before chunking so one long item can't pad a chunk of short ones, then
        restored to input order.
        """
        if not conversations:
            return []
        if len(conversations) == 1:
            return [self.generate(conversations[0], config=config)]

        self._ensure_loaded()
        config = config or self.settings.generation
        prompts = [self._render(c) for c in conversations]
        # Sort long-first: chunks then hold similar lengths, so padding waste (and the
        # KV cache each chunk reserves) is bounded by the chunk, not the whole batch.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)

        results: list[str | None] = [None] * len(prompts)
        size = max(1, self.settings.batch_size)
        position = 0
        while position < len(order):
            indices = order[position : position + size]
            try:
                for index, text in zip(
                    indices, self._generate_chunk(prompts, indices, config)
                ):
                    results[index] = text
            except _OOM_ERRORS as exc:
                if not _is_oom(exc) or size == 1:
                    raise
                size = max(1, size // 2)
                print(
                    f"warning: out of memory at batch {len(indices)}; "
                    f"retrying at {size} for the rest of this run.",
                    file=sys.stderr,
                )
                self._empty_cache()
                continue
            position += len(indices)
        return [r if r is not None else "" for r in results]

    def _generate_chunk(
        self, prompts: Sequence[str], indices: Sequence[int], config: GenerationConfig
    ) -> list[str]:
        """Generate for one already-grouped chunk of prompt indices."""
        import torch

        texts = [prompts[i] for i in indices]
        truncate = self.settings.max_input_tokens is not None
        with self._padding(truncate) as tok:
            encoded = tok(
                texts,
                return_tensors="pt",
                padding=True,
                # apply_chat_template already emitted the template's own BOS.
                add_special_tokens=False,
                **(
                    {"truncation": True, "max_length": self.settings.max_input_tokens}
                    if truncate
                    else {}
                ),
            )
        device = self._input_device or self.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        # Left padding makes this one length correct for every row in the chunk.
        input_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            output = self._model.generate(**encoded, **self._gen_kwargs(config))
        return [
            self._tokenizer.decode(row[input_len:], skip_special_tokens=True).strip()
            for row in output
        ]

    @staticmethod
    def _empty_cache() -> None:
        """Release cached CUDA blocks after an OOM, so the retry has room."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - depends on env
            pass
