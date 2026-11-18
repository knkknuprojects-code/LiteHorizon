"""Shared test fixtures: a fake LLM client so tests never load a real model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from horizon.config import GenerationConfig
from horizon.embeddings import Embedder
from horizon.llm.base import LLMClient, Message


class FakeClient(LLMClient):
    """Records messages and returns canned replies.

    Three reply shapes:

    - a single string — a fixed reply;
    - a list of strings — one per call in order, the last repeating once the queue is
      exhausted (handy for the staged strategy's 4 calls and the repair-retry path);
    - a callable ``(messages) -> str`` — keyed on the *prompt* rather than the call
      index. Batching changes the order calls arrive in, so an index-keyed fake makes a
      batch-vs-sequential comparison meaningless; a prompt-keyed one makes it exact.

    ``batches`` records the size of each ``generate_batch`` call, so a test can assert
    that batching actually happened and how the work was grouped.
    """

    def __init__(self, reply: str | list[str] | Callable = "fake-report") -> None:
        self.reply_fn = reply if callable(reply) else None
        if self.reply_fn is not None:
            self.replies: list[str] = []
        else:
            self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[list[Message]] = []
        self.batches: list[int] = []

    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if self.reply_fn is not None:
            return self.reply_fn(list(messages))
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        self.batches.append(len(conversations))
        return super().generate_batch(conversations, config=config)


@pytest.fixture
def make_client():
    """Factory: ``make_client("reply")``, ``make_client([...])`` or ``make_client(fn)``."""

    def _make(reply: str | list[str] | Callable = "fake-report") -> FakeClient:
        return FakeClient(reply)

    return _make


class FakeEmbedder(Embedder):
    """Deterministic embedder for tests.

    Maps each text to a vector via ``fn`` (default: a fixed unit vector). Records
    the texts it was asked to embed so cache behaviour can be asserted.
    """

    model_id = "fake-embedder"

    def __init__(self, fn=None) -> None:
        self.fn = fn or (lambda t: [1.0, 0.0])
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [list(map(float, self.fn(t))) for t in texts]


@pytest.fixture
def make_embedder():
    """Factory fixture: ``make_embedder(fn)`` -> a ``FakeEmbedder``."""

    def _make(fn=None) -> FakeEmbedder:
        return FakeEmbedder(fn)

    return _make
