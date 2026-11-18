"""Phase-4 fine-tuning tooling: dataset export, training, evaluation.

This subpackage is pure-Python at import time — ``export`` and ``evaluate`` reuse
the same prompts/parser as inference and need no ML stack. ``train`` (added later)
imports ``peft``/``trl`` lazily behind the ``[train]`` extra.
"""
