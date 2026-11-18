"""Google Gemini API backend for ``LLMClient`` (the distillation *teacher*).

Local generation (``TransformersClient``) is CPU-bound and noisy on the small
stand-in; a hosted Gemini model produces far cleaner structured output, which is
what we want when generating training labels to distil into a local Gemma. This
backend is the "API teacher" for Phase-4/5: point ``model_id`` at a Gemini model
(e.g. ``gemini-3.5-flash``) and the rest of Horizon is unchanged.

The ``google-genai`` SDK is imported **lazily** inside methods (like ``torch`` in
``TransformersClient``) so the package and tests import without it. Install with
``pip install -e ".[teacher]"`` and set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from horizon.config import GenerationConfig, Settings
from horizon.llm.base import LLMClient, Message

# Transient API failures worth a retry (rate limits / server hiccups). A 4xx like
# 400/401/403 is a caller error — retrying it just wastes calls, so we don't.
_TRANSIENT_HTTP = {429, 500, 502, 503, 504}
_TRANSIENT_STATUS = {
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
}


def to_gemini(messages: Sequence[Message]) -> tuple[str | None, list[dict]]:
    """Split ``Message``s into (system_instruction, contents) for the Gemini API.

    Unlike Gemma (see ``transformers_client.to_chat``), Gemini has a first-class
    ``system_instruction``, so we keep system text *separate* rather than folding
    it into the first user turn — the model treats it as a proper system prompt.
    Roles map to Gemini's ``user`` / ``model``.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
            continue
        role = "model" if message.role in ("assistant", "model") else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, contents


def _gen_config(
    config: GenerationConfig,
    *,
    system: str | None,
    json_mode: bool,
    thinking_budget: int | None,
) -> dict:
    """Map Horizon's ``GenerationConfig`` to a Gemini ``config`` dict.

    ``do_sample`` has no Gemini analogue (it always samples; ``temperature=0`` is
    effectively greedy). ``response_mime_type=application/json`` puts the model in
    JSON mode, which is why the teacher path rarely needs the extract repair-retry.

    ``thinking_budget`` caps the model's internal reasoning tokens. On the 3.x
    "thinking" models those tokens count against ``max_output_tokens``, so leaving
    it unset lets thinking (~400+ tokens) eat the budget and truncate the JSON. For
    structured extraction we don't need reasoning, so it defaults to ``0`` (off) —
    cheaper and no truncation. ``None`` leaves the model's default thinking on.
    """
    cfg: dict[str, object] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_output_tokens": config.max_new_tokens,
    }
    if system:
        cfg["system_instruction"] = system
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        cfg["thinking_config"] = {"thinking_budget": thinking_budget}
    return cfg


def _is_transient(exc: Exception) -> bool:
    """True if ``exc`` is a transient API failure worth retrying.

    Retries rate limits and 5xx server hiccups (matched by HTTP code, gRPC-style
    status string, or a timeout/connection error type); a 4xx caller error is not
    transient, so it falls through to be re-raised.
    """
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT_HTTP:
        return True
    if str(getattr(exc, "status", "")).upper() in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


class GeminiClient(LLMClient):
    """Calls the Gemini API. The SDK client is built on first ``generate``.

    ``json_mode`` defaults to ``True`` because this backend exists to produce the
    structured JSON that ``horizon.extract`` consumes; pass ``json_mode=False`` for
    free-form generation. ``thinking_budget`` defaults to ``0`` (reasoning off) so
    thinking tokens don't consume ``max_output_tokens`` and truncate the JSON; pass
    ``None`` to leave the model's default thinking on. A pre-built ``client`` can be
    injected (tests / a shared client) to bypass SDK construction and env lookup.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client=None,
        api_key: str | None = None,
        json_mode: bool = True,
        thinking_budget: int | None = 0,
        max_retries: int = 3,
    ) -> None:
        """Store settings, credentials, and the JSON-mode / thinking / retry options."""
        self.settings = settings or Settings()
        self.api_key = api_key
        self.json_mode = json_mode
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self._client = client

    def _ensure_client(self):
        """Return the SDK client, building it (and reading the API key) on first use.

        Raises if the ``google-genai`` SDK or an API key is missing; an injected
        client short-circuits construction and env lookup.
        """
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "the Gemini API teacher needs the google-genai SDK. "
                'Install it with: pip install -e ".[teacher]"'
            ) from exc
        api_key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "no Gemini API key found — set GEMINI_API_KEY (or GOOGLE_API_KEY). "
                "Get one from Google AI Studio (https://aistudio.google.com/apikey)."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before retry ``attempt`` — exponential, capped at 30s."""
        return min(2.0**attempt, 30.0)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text from the Gemini API, retrying transient failures.

        Splits messages into system + contents, maps the config (JSON mode,
        thinking budget), and retries up to ``max_retries`` on transient errors with
        backoff. An empty response (safety block / truncation) is raised, not retried.
        """
        config = config or self.settings.generation
        system, contents = to_gemini(messages)
        gen_config = _gen_config(
            config,
            system=system,
            json_mode=self.json_mode,
            thinking_budget=self.thinking_budget,
        )
        client = self._ensure_client()

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.settings.model_id,
                    contents=contents,
                    config=gen_config,
                )
                text = getattr(response, "text", None)
                if not text:
                    # Empty usually means a safety block or a truncated/blocked
                    # candidate — not transient, so surface it rather than retry.
                    raise RuntimeError(
                        "Gemini returned no text "
                        f"(finish reason: {_finish_reason(response)})"
                    )
                return text.strip()
            except Exception as exc:  # noqa: BLE001 - re-raised unless transient
                last_exc = exc
                if attempt < self.max_retries and _is_transient(exc):
                    time.sleep(self._backoff(attempt))
                    continue
                raise
        raise last_exc  # pragma: no cover - loop always returns or raises


def _finish_reason(response) -> str:
    """Best-effort finish reason of the first candidate, or ``"unknown"``.

    Defensive because the response shape varies across ``google-genai`` versions.
    """
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:  # pragma: no cover - defensive, shape varies by SDK version
        return "unknown"
