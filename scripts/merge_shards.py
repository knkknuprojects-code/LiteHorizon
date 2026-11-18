#!/usr/bin/env python3
"""Merge per-shard dataset roots into one store, refusing merges that are not one dataset.

Pure stdlib — runs on a box with no ML stack installed.

    python scripts/merge_shards.py merged/ ds/wos-2015 ds/wos-2016 ...
    python scripts/merge_shards.py merged/ ds/wos-*        # the usual invocation
    python scripts/merge_shards.py --dry-run merged/ ds/wos-*

Concatenating the shards is trivial and already documented; the reason this is a script
is the checking. The store is append-only with last-write-wins per
``(content_hash, prompt_version)``, so a merge cannot corrupt anything — it can only
produce a file that *looks* like one corpus while being two. Three ways that happens,
all silent:

* **Mixed ``prompt_version``.** Analyses under two prompt versions do not collide (the
  version is part of the key), so both survive and every downstream count doubles.
* **Mixed ``model_id``.** Half the corpus judged by one model and half by another is not
  one dataset. With the enum labels being largely model priors (see
  ``reports/gemma4-gpu-throughput-wos-scopus-2026-08-26.md`` §8), a mixed merge would
  put a discontinuity straight into any trend computed from it — and it would look like
  a finding.
* **Overlapping shards.** The same year scanned on both machines wastes half the compute
  and, worse, makes per-year counts wrong in a way no total will reveal.

So those are refusals, not warnings. ``--allow-mixed-model`` exists for the deliberate
case: a stand-in-vs-teacher comparison, which *wants* two model ids under one key.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file, skipping blank lines. Missing file yields ``[]``."""
    if not path.is_file():
        return []
    out = []
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{i}: {exc}") from exc
    return out


def main() -> int:
    """Merge the given roots into ``out``, after checking they form one dataset."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="destination dataset root")
    ap.add_argument("roots", type=Path, nargs="+", help="per-shard dataset roots")
    ap.add_argument("--dry-run", action="store_true", help="check only, write nothing")
    ap.add_argument(
        "--allow-mixed-model",
        action="store_true",
        help="permit several model_ids (for a stand-in-vs-teacher comparison)",
    )
    ap.add_argument(
        "--allow-overlap",
        type=int,
        default=0,
        metavar="N",
        help="tolerate up to N items analysed by the same model in two roots. The "
        "corpus has a few genuine ones -- a paper WoS and Scopus date to different "
        "years lands in two cohorts -- and the store collapses them per key anyway. A "
        "year scanned twice would show thousands, so the number is what tells the two "
        "apart; state it rather than ignoring the check.",
    )
    ap.add_argument(
        "--expect",
        type=int,
        default=None,
        metavar="N",
        help="fail unless exactly N distinct records are merged (e.g. 269065)",
    )
    args = ap.parse_args()

    per_root: dict[Path, list[dict]] = {}
    for root in args.roots:
        f = root / "data" / "analyses.jsonl"
        recs = read_jsonl(f)
        if not recs:
            print(f"  skip (empty or missing): {f}")
            continue
        per_root[root] = recs
    if not per_root:
        print("error: nothing to merge", file=sys.stderr)
        return 1

    records = [r for recs in per_root.values() for r in recs]
    versions = collections.Counter(r.get("prompt_version") for r in records)
    models = collections.Counter(r.get("model_id") for r in records)
    keys = collections.Counter(
        (r.get("content_hash"), r.get("prompt_version")) for r in records
    )
    # Duplicate work means the same item analysed by the same model in two roots. The
    # model id has to be part of that test: a duplicate inside one root is a re-scan and
    # harmless, and one *item* in several roots is the whole point of a comparison — what
    # is never intentional is the same (item, prompt, model) triple computed twice.
    where = collections.defaultdict(set)
    for root, recs in per_root.items():
        for r in recs:
            triple = (r.get("content_hash"), r.get("prompt_version"), r.get("model_id"))
            where[triple].add(root.name)
    cross = {k for k, v in where.items() if len(v) > 1}

    print(f"\nroots         {len(per_root)}")
    for root, recs in per_root.items():
        print(f"  {root.name:<24} {len(recs):>7} records")
    print(f"records       {len(records)}")
    print(f"distinct keys {len(keys)}")
    print(f"prompt_version{'':<1} {dict(versions)}")
    print(f"model_id       {dict(models)}")

    fatal = []
    if len(versions) > 1:
        fatal.append(
            f"{len(versions)} prompt versions in one merge: {sorted(versions)} — "
            "re-scan the older shards rather than merging across a prompt change"
        )
    if len(models) > 1 and not args.allow_mixed_model:
        fatal.append(
            f"{len(models)} model ids in one merge: {sorted(models)} — pass "
            "--allow-mixed-model only if a comparison is what you want"
        )
    if len(cross) > args.allow_overlap:
        sample = ", ".join(sorted(f"{h[:12]}…" for h, _, _ in list(cross)[:3]))
        share = len(cross) / max(len(keys), 1)
        fatal.append(
            f"{len(cross)} items ({share:.3%}) were analysed by the same model in more "
            f"than one root (e.g. {sample}). A handful are genuine — a paper the "
            f"databases date to different years lands in two cohorts — and pass "
            f"--allow-overlap {len(cross)}; thousands mean a year was scanned twice"
        )
    elif cross:
        print(f"  tolerating {len(cross)} cross-root duplicates (--allow-overlap)")
    if args.expect is not None and len(keys) != args.expect:
        fatal.append(f"expected {args.expect} distinct records, merged {len(keys)}")

    if fatal:
        print()
        for m in fatal:
            print(f"error: {m}", file=sys.stderr)
        return 1
    models_note = (
        "one model" if len(models) == 1 else f"{len(models)} models (comparison merge)"
    )
    print(f"\nchecks passed: one prompt version, {models_note}, no duplicated work")

    if args.dry_run:
        print("(dry run — nothing written)")
        return 0

    dest = args.out / "data"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "analyses.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Source texts travel with the analyses so `export` has its inputs; duplicates are
    # harmless, the sources store is itself last-write-wins per content_hash.
    n_src = 0
    with (dest / "sources.jsonl").open("w", encoding="utf-8") as fh:
        for root in per_root:
            for r in read_jsonl(root / "data" / "sources.jsonl"):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_src += 1
    # Manifests are the audit trail: argv, model, quant, batch, host, library versions.
    runs = dest / "runs"
    runs.mkdir(exist_ok=True)
    n_runs = 0
    for root in per_root:
        for m in sorted((root / "data" / "runs").glob("*.json")):
            shutil.copy2(m, runs / f"{root.name}-{m.name}")
            n_runs += 1

    print(
        f"wrote {args.out}: {len(records)} analyses, {n_src} sources, {n_runs} manifests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
