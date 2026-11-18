# Contributing

This repository is currently a research-release candidate. Public contribution
procedures should be activated only after the rights holders approve a license.

Once open, please:

1. Search existing issues and describe the desired behavior or reproducible bug.
2. Include the operating system, Python version, model identifier, prompt version,
   command, input type, and sanitized error trace.
3. Add or update a test for changes to parsing, provenance, clustering, or scoring.
4. Run `python -m pytest -q --ignore=tests/test_llm.py` for the model-independent
   suite; the model-dependent test module needs extra optional packages.
5. Keep data licensing and source provenance visible. Never submit raw restricted
   workbooks, API keys, model weights, passwords, or personal data.

Changes that could alter research output distributions must state whether existing
results need to be regenerated and whether `PROMPT_VERSION` or another version
marker should change.
