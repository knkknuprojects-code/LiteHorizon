"""The ``LLMClient`` interface.

Every backend (local dev, fine-tuned/production) implements this so the rest of
Horizon never depends on a concrete model or runtime. Swapping Gemma sizes — or
even families — is a matter of providing a different ``LLMClient``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from horizon.config import GenerationConfig


@dataclass(frozen=True)
class Message:
    """A single chat turn. ``role`` is one of ``system``, ``user``, ``model``."""

    role: str
    content: str


def _one_shot(prompt: str, system: str | None) -> list[Message]:
    """The message list for a one-shot prompt, shared by ``complete``/``complete_batch``."""
    messages: list[Message] = []
    if system:
        messages.append(Message("system", system))
    messages.append(Message("user", prompt))
    return messages


class LLMClient(ABC):
    """Abstract chat-completion client."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a single completion for a conversation."""
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Convenience: a one-shot prompt (with optional system instruction)."""
        return self.generate(_one_shot(prompt, system), config=config)

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """One completion per conversation, in input order.

        Deliberately **not** abstract: the default is a sequential loop over
        :meth:`generate`, so a hosted backend (or a test fake) needs no changes.
        Backends that can genuinely batch a forward pass — ``TransformersClient`` on a
        GPU, where batching is the throughput lever — override it.
        """
        return [self.generate(c, config=config) for c in conversations]

    def complete_batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Batch counterpart of :meth:`complete`: one system prompt, many user turns.

        Builds each conversation exactly as :meth:`complete` does, so a batch of one is
        indistinguishable from a ``complete`` call at the client boundary.
        """
        return self.generate_batch(
            [_one_shot(p, system) for p in prompts], config=config
        )
