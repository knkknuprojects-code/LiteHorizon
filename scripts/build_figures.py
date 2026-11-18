#!/usr/bin/env python3
"""Redraw a report's figures from its ``facts.json``.

    python scripts/build_figures.py reports/wos-scopus-2026-08-28-facts.json

``horizon report`` draws these itself when matplotlib is installed, and writes the
references into the report as it goes. This exists for the case where it could not — the
GPU box has no matplotlib — or where a figure changed and re-running the whole report to
redraw it would be wasteful.

Writes ``reports/figures/<slug>/`` and ``reports/figures/<slug>-ua/``, the layout the
arXiv characterization report established: one directory per language, because axis labels
are prose. It does not touch the report text; if the references are missing there, the
report itself has to be re-rendered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from horizon.figures import draw  # noqa: E402


def main() -> int:
    """Render both language sets beside the report they belong to."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("facts", type=Path, help="a report's -facts.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/figures"),
        help="directory holding the per-report figure folders",
    )
    ap.add_argument("--slug", default=None, help="folder name (default from the facts)")
    args = ap.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    slug = args.slug or args.facts.name.replace("-facts.json", "")
    for lang, folder in (("en", slug), ("uk", f"{slug}-ua")):
        written = draw(facts, args.out / folder, lang)
        print(f"  {args.out / folder}: {len(written)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
