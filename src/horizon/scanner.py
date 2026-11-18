"""Minimal horizon-scanning analysis over a single prepared text.

This is the Phase-1 freeform path: one model call that turns a raw text into a
short signal report. The structured Phase-2 path lives in ``extract.py``.

Horizon is domain-agnostic — it scans any landscape (technology, policy, markets,
science, geopolitics, …), not a single topic.
"""

from __future__ import annotations

from horizon.config import GenerationConfig
from horizon.llm.base import LLMClient
from horizon.prompts import SCAN_PROMPT_TEMPLATE, SCAN_SYSTEM_PROMPT

# Back-compat aliases (the prompts now live in horizon.prompts).
SYSTEM_PROMPT = SCAN_SYSTEM_PROMPT
PROMPT_TEMPLATE = SCAN_PROMPT_TEMPLATE


def scan_text(
    text: str,
    client: LLMClient,
    *,
    config: GenerationConfig | None = None,
) -> str:
    """Run the freeform horizon-scanning prompt over ``text`` and return the report."""
    prompt = PROMPT_TEMPLATE.format(text=text.strip())
    return client.complete(prompt, system=SYSTEM_PROMPT, config=config)
