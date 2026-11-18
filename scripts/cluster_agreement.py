#!/usr/bin/env python3
"""Do two models' analyses of the same records produce the same trend clusters?


    python scripts/cluster_agreement.py "$HORIZON_SCRATCH/cmp-v4/merged"
    python scripts/cluster_agreement.py merged --models E4B 12B --threshold 0.6

`horizon compare` measures agreement on the classification fields. Those are not what
the product is built on: `trends` embeds ``summary`` + ``signals`` (``embeddings.embed_text``)
and never reads ``maturity``, ``impact`` or ``time_horizon``. So the question that decides
the model choice is not whether two models assign the same labels — it is whether they
put the same records in the same clusters.

This clusters each model's analyses separately, over the same items, with the same
backend and threshold, then compares the two partitions:

* **Adjusted Rand Index** — agreement on which pairs of records share a cluster,
  corrected for chance. 1.0 is identical partitions, 0.0 is chance.
* **paired cosine** — how close the two models' embeddings of the *same* record are.
  High pairwise similarity with low ARI would mean the models describe records alike but
  the clustering threshold sits somewhere unstable.

Needs the ML stack (sentence-transformers, sklearn) — it runs on a box, not on a laptop
without the extras.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def tag_of(model_id: str, wanted: list[str]) -> str | None:
    """The first requested tag that appears in ``model_id``, or ``None``."""
    for w in wanted:
        if w.lower() in model_id.lower():
            return w
    return None


def main() -> int:
    """Cluster each model's analyses over the shared records and compare partitions."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="dataset root holding data/analyses.jsonl")
    ap.add_argument(
        "--models",
        nargs=2,
        default=["E4B", "12B"],
        metavar="SUBSTR",
        help="two model_id substrings to compare (default: E4B 12B)",
    )
    ap.add_argument("--threshold", type=float, default=None, help="cosine threshold")
    ap.add_argument(
        "--cluster-method", default=None, choices=["greedy", "agglomerative"]
    )
    args = ap.parse_args()

    import numpy as np
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    from horizon.analysis import Analysis
    from horizon.clustering import make_clusterer
    from horizon.config import Settings
    from horizon.embeddings import SentenceTransformerEmbedder, embed_text

    settings = Settings.from_env()
    threshold = args.threshold or settings.cluster_threshold
    method = args.cluster_method or settings.cluster_method

    path = args.root / "data" / "analyses.jsonl"
    by_hash: dict[str, dict[str, Analysis]] = collections.defaultdict(dict)
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        t = tag_of(d.get("model_id", ""), args.models)
        if t:
            by_hash[d["content_hash"]][t] = Analysis.from_dict(d)

    a_tag, b_tag = args.models
    # Same records, same order, for both sides — the comparison is meaningless otherwise.
    shared = [h for h, v in by_hash.items() if a_tag in v and b_tag in v]
    if not shared:
        raise SystemExit(f"no records carry both {a_tag} and {b_tag} in {path}")
    print(f"records analysed by both: {len(shared)}")
    print(f"clusterer: {method} @ threshold {threshold}\n")

    embedder = SentenceTransformerEmbedder(settings)
    vectors = {}
    for t in (a_tag, b_tag):
        texts = [embed_text(by_hash[h][t]) for h in shared]
        vectors[t] = np.asarray(embedder.embed(texts), dtype=float)
        print(f"  {t}: embedded {len(texts)} analyses")

    clusterer = make_clusterer(method, threshold)
    labels = {t: clusterer.cluster(vectors[t]) for t in (a_tag, b_tag)}
    for t in (a_tag, b_tag):
        sizes = collections.Counter(labels[t])
        singles = sum(1 for c in sizes.values() if c == 1)
        print(
            f"  {t}: {len(sizes)} clusters, largest {max(sizes.values())}, "
            f"{singles} singletons"
        )

    # Paired cosine: both sides are L2-normalised by the embedder, so a dot is the cosine.
    paired = (vectors[a_tag] * vectors[b_tag]).sum(axis=1)
    ari = adjusted_rand_score(labels[a_tag], labels[b_tag])
    ami = adjusted_mutual_info_score(labels[a_tag], labels[b_tag])

    print(
        f"\n  paired cosine   mean {paired.mean():.3f}  median {np.median(paired):.3f}"
    )
    print(
        f"                  p10 {np.percentile(paired, 10):.3f}  min {paired.min():.3f}"
    )
    print(f"  adjusted Rand   {ari:.3f}   (all records)")
    print(f"  adjusted MI     {ami:.3f}")

    # A record alone in its cluster on either side says nothing about whether the two
    # models group things alike — it says the threshold left it alone. When most records
    # are singletons, ARI is dominated by them, so report the restricted figure too: it
    # is the one that answers "do the clusters that exist agree?"
    sizes = {t: collections.Counter(labels[t]) for t in (a_tag, b_tag)}
    core = [
        i
        for i in range(len(shared))
        if sizes[a_tag][labels[a_tag][i]] > 1 and sizes[b_tag][labels[b_tag][i]] > 1
    ]
    if core and len(core) < len(shared):
        ari_core = adjusted_rand_score(
            [labels[a_tag][i] for i in core], [labels[b_tag][i] for i in core]
        )
        print(
            f"  adjusted Rand   {ari_core:.3f}   (the {len(core)} records in a real "
            f"cluster on both sides, {len(core) / len(shared):.0%})"
        )
    reading = (
        "the two models would produce essentially the same trends"
        if ari >= 0.6
        else (
            "the partitions overlap substantially but not interchangeably"
            if ari >= 0.3
            else "the two models would produce materially different trends"
        )
    )
    print(f"\n  {reading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
