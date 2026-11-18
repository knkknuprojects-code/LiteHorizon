"""LLM client abstraction and backends."""

from __future__ import annotations

from horizon.llm.base import LLMClient, Message

__all__ = ["LLMClient", "Message", "make_client"]


def make_client(settings=None) -> LLMClient:
    """Pick a backend from ``settings.model_id`` — the one place that decides.

    A ``gemini*`` model id selects the hosted Gemini API teacher
    (:class:`~horizon.llm.gemini_client.GeminiClient`); anything else runs locally
    via :class:`~horizon.llm.transformers_client.TransformersClient`. Concrete
    backends are imported lazily so callers (and ``horizon --help``) don't pull in
    ``torch`` or the ``google-genai`` SDK until a model is actually needed.
    """
    from horizon.config import Settings

    settings = settings or Settings()
    if (settings.model_id or "").lower().startswith("gemini"):
        from horizon.llm.gemini_client import GeminiClient

        return GeminiClient(settings)
    from horizon.llm.transformers_client import TransformersClient

    return TransformersClient(settings)
