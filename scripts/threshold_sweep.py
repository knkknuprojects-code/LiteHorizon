#!/usr/bin/env python3
"""Pick the clustering threshold from measurement instead of taste.

    python scripts/threshold_sweep.py "$HORIZON_SCRATCH/ds/merged" --sample 20000
    python scripts/threshold_sweep.py out/cmp-v4/merged --models E4B 12B

The threshold decides how broad a theme is, and it is the one knob that can be turned
after the scan for free. This reports two things per candidate value, because either
alone is misleading.

**Readability** — cluster count, median size, and what share of records land in a
cluster of at least ``--min-size``. A value that leaves most records alone, or that
collapses everything into one blob, is unusable however stable it is. This half is a
matter of taste and can be argued with.

**Stability** — adjusted Rand index between two partitions of the same records.

Measured, this half **cannot choose a threshold**, and the sweep prints it as a diagnostic
rather than a criterion. Order-ARI rises monotonically up the column (0.361 -> 0.679, and
0.397 -> 1.000 restricted) because it scores how *determined* the greedy assignment is,
which is maximised by making no decisions: a singleton is assigned identically in every
ordering, so a partition that has dissolved into singletons agrees with itself perfectly.
Restricting to non-singletons did not fix it (doubletons are just as free), and
restricting to clusters of >= ``--min-size`` made it worse (1.000 over the three tight
groups left at 0.82) — every restriction removes exactly the ambiguous records whose
placement *is* the instability. Two rows are comparable only at comparable coverage, and
in a sweep no two are. So read the readability half down the column, and read stability
only across at one threshold. Two sources of it:

* *order* — the greedy clusterer is order-dependent by construction, so clustering the
  same vectors twice in different orders and comparing measures directly whether the
  groups are real or artefacts of where the boundary happened to fall. Available on any
  store, including a single-model corpus run.
* *model* — with ``--models A B``, the same comparison across two models' analyses of
  the same records. Measured at the current default of 0.55 it was 0.417, with a paired
  cosine of 0.907 between the two models' embeddings — the models agreed and the
  partitions did not, which is what motivated this sweep.

Embedding is the expensive part and the thresholds do not change it, so it runs once and
every candidate re-clusters the same vectors.

Needs the ML stack. Sampling is stratified by year, so the cross-year structure that
velocity and growth read is preserved — never partition *by* year.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path


def load_analyses(root: Path, models: list[str] | None) -> dict[str, dict]:
    """Analyses from ``root``, keyed by content hash then by model tag.

    A record analysed by several models keeps one entry per model, which is what the
    inter-model comparison needs; with ``models`` unset everything collapses under a
    single ``"*"`` tag.
    """
    out: dict[str, dict] = collections.defaultdict(dict)
    path = root / "data" / "analyses.jsonl"
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tag = "*"
        if models:
            mid = d.get("model_id", "").lower()
            tag = next((m for m in models if m.lower() in mid), None)
            if tag is None:
                continue
        out[d["content_hash"]][tag] = d
    return out


def stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Sample ``n`` items, keeping each year's share of the whole.

    Proportional rather than uniform because velocity and growth are read across years:
    a sample that over-weights one cohort would distort exactly what the clustering is
    meant to feed. Returns everything when ``n`` covers the input.
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    by_year: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_year[str(it.get("item_date", ""))[:4]].append(it)
    picked: list[dict] = []
    for year in sorted(by_year):
        group = by_year[year]
        take = round(n * len(group) / len(items))
        picked.extend(rng.sample(group, min(take, len(group))))
    # Rounding can leave the total a little under or over; correct against the remainder.
    if len(picked) < n:
        rest = [it for it in items if it not in picked]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    return picked[:n]


def summarise(labels: list[int], min_size: int) -> dict:
    """Readability of one partition: how many clusters, how big, how much is covered."""
    sizes = collections.Counter(labels)
    counts = sorted(sizes.values(), reverse=True)
    covered = sum(c for c in counts if c >= min_size)
    return {
        "clusters": len(counts),
        "real": sum(1 for c in counts if c >= min_size),
        "singletons": sum(1 for c in counts if c == 1),
        "median": statistics.median(counts) if counts else 0,
        "largest": counts[0] if counts else 0,
        "covered": covered / max(len(labels), 1),
    }


def core_ari(a: list[int], b: list[int], score, min_size: int = 2) -> float:
    """Adjusted Rand over records that are in a real group on *both* sides.

    Unrestricted, this metric rewards degeneracy: at a high enough threshold nearly
    every record is its own cluster, two orderings agree trivially, and the score climbs
    to 0.755 while the partition has stopped saying anything.

    Restricting to non-singletons is not enough — measured, the score still climbed
    monotonically to 0.804, because a *doubleton* agrees between orderings just as
    freely as a singleton does. ``min_size`` is therefore the same size that ``covered``
    uses: stability is read only over records that landed somewhere a reader would
    call a trend. Below two surviving groups the number is undefined, not zero.
    """
    sizes_a, sizes_b = collections.Counter(a), collections.Counter(b)
    core = [
        i
        for i in range(len(a))
        if sizes_a[a[i]] >= min_size and sizes_b[b[i]] >= min_size
    ]
    if len(core) < 2 or len({a[i] for i in core}) < 2:
        return float("nan")
    return score([a[i] for i in core], [b[i] for i in core])


def main() -> int:
    """Cluster one embedding set at several thresholds and report both criteria."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="dataset root holding data/analyses.jsonl")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.45, 0.50, 0.55, 0.60, 0.65],
        help="candidate cosine thresholds",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="records to draw, stratified by year (default 20000)",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="cluster size that counts as covering a record (default 20)",
    )
    ap.add_argument(
        "--models",
        nargs=2,
        default=None,
        metavar="SUBSTR",
        help="two model_id substrings, to add inter-model stability",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", dest="as_json", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    from horizon.analysis import Analysis
    from horizon.clustering import GreedyClusterer
    from horizon.config import Settings
    from horizon.embeddings import EmbeddingCache, SentenceTransformerEmbedder

    settings = Settings.from_env()
    by_hash_all = load_analyses(args.root, args.models)
    if args.models:
        a_tag, b_tag = args.models
        shared = [h for h, v in by_hash_all.items() if a_tag in v and b_tag in v]
        primary = [by_hash_all[h][a_tag] for h in shared]
    else:
        primary = [next(iter(v.values())) for v in by_hash_all.values()]
    if not primary:
        raise SystemExit(f"no analyses found under {args.root}")

    picked = stratified_sample(primary, args.sample, args.seed)
    print(f"{len(primary)} analyses, sampling {len(picked)} stratified by year")

    # Through the cache, keyed on content hash: a report run over the same store has
    # already paid for these, and the sweep exists to be cheap enough to re-run.
    embedder = EmbeddingCache(
        SentenceTransformerEmbedder(settings), args.root / "data" / "embeddings.jsonl"
    )
    primary_analyses = [Analysis.from_dict(d) for d in picked]
    by_hash = embedder.embed_analyses(primary_analyses)
    vectors = np.asarray(
        [by_hash[a.content_hash] for a in primary_analyses], dtype=float
    )
    order = list(range(len(picked)))
    rng = random.Random(args.seed)
    shuffled = order[:]
    rng.shuffle(shuffled)
    # Same vectors, different arrival order: greedy is order-dependent, so disagreement
    # here is the threshold's instability rather than anything about the data.
    inverse = {j: i for i, j in enumerate(shuffled)}

    other = None
    if args.models:
        hashes = [d["content_hash"] for d in picked]
        second = [Analysis.from_dict(by_hash_all[h][b_tag]) for h in hashes]
        vecs = embedder.embed_analyses(second)
        other = np.asarray([vecs[a.content_hash] for a in second], dtype=float)

    rows = []
    for t in args.thresholds:
        labels = GreedyClusterer(t).cluster(vectors)
        reordered = GreedyClusterer(t).cluster(vectors[shuffled])
        same_order = [reordered[inverse[i]] for i in order]
        row = {
            "threshold": t,
            **summarise(labels, args.min_size),
            "order_ari": adjusted_rand_score(labels, same_order),
            "core_ari": core_ari(
                labels, same_order, adjusted_rand_score, args.min_size
            ),
        }
        if other is not None:
            other_labels = GreedyClusterer(t).cluster(other)
            row["model_ari"] = adjusted_rand_score(labels, other_labels)
            row["model_core_ari"] = core_ari(
                labels, other_labels, adjusted_rand_score, args.min_size
            )
        rows.append(row)

    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0

    head = (
        f"\n  {'thr':>5}{'clusters':>10}{'>=' + str(args.min_size):>8}{'largest':>9}"
        f"{'covered':>9}{'ARI':>8}{'core ARI':>10}"
    )
    if other is not None:
        head += f"{'model':>8}{'m.core':>9}"
    print(head)
    for r in rows:
        line = (
            f"  {r['threshold']:>5.2f}{r['clusters']:>10}{r['real']:>8}"
            f"{r['largest']:>9}{r['covered']:>8.0%}{r['order_ari']:>8.3f}"
            f"{r['core_ari']:>10.3f}"
        )
        if other is not None:
            line += f"{r['model_ari']:>8.3f}{r['model_core_ari']:>9.3f}"
        print(line)
    print(
        f"\n  '>= {args.min_size}' is clusters that size or larger; 'covered' the share of"
        " records in one."
        f"\n  Read core ARI, not ARI: core ARI is measured only over records inside a"
        f" cluster of >= {args.min_size} on both"
        "\n  sides, because small groups agree between orderings for free and drag the"
        " unrestricted figure up as"
        "\n  the partition degenerates. Compare stability across models at one threshold,"
        " never down the column."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
