"""Step 0a: exploratory statistics that DECIDE the clustering design.

Run before committing clustering params. Reports, per dataset:
- action-type distribution + how many users have non-trivial action signal,
- behavioural feature distributions (mean/std/quantiles) and their spread,
- the two-stage clustering outcome (leaf count, sizes, silhouette, out-of-cluster
  fraction) and per-leaf behavioural tags.

The two-stage structure is a hypothesis; this report is what validates/tunes it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger

from src.clustering.clusterer import BehavioralClusterer
from src.clustering.features import FeatureExtractor, VECTOR_FIELD_NAMES
from src.data.schemas import Thread


def run_eda(threads: list[Thread], seed: int = 42,
            clusterer: BehavioralClusterer | None = None) -> dict:
    n_msgs = sum(len(t.messages) for t in threads)
    action_dist = Counter()
    tox_vals = []
    for t in threads:
        for m in t.messages:
            action_dist[m.action_type.value] += 1
            tv = m.metadata.get("toxicity")
            if isinstance(tv, (int, float)):
                tox_vals.append(tv)

    feats = FeatureExtractor().extract_all(threads)
    M = np.stack([f.to_vector() for f in feats.values()]) if feats else np.zeros((0, len(VECTOR_FIELD_NAMES)))

    feat_stats = {}
    for i, name in enumerate(VECTOR_FIELD_NAMES):
        col = M[:, i] if M.size else np.array([0.0])
        feat_stats[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p50": float(np.percentile(col, 50)), "p90": float(np.percentile(col, 90)),
            "nonzero_frac": float((col != 0).mean()),
        }

    # two-stage clustering outcome
    if clusterer is None:
        clusterer = BehavioralClusterer(random_state=seed)
    cr = clusterer.fit(threads)
    leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]
    sizes = {int(l): len(cr.get_cluster_members(l)) for l in leaf_ids}
    n_noise = sum(1 for v in cr.labels.values() if v == -1)

    # per-leaf tags (reuse profiler tag logic)
    from src.skill.cluster_profile import ArchetypeProfiler
    leaf_feat = {}
    for l in leaf_ids:
        vecs = [cr.user_features[u].to_vector() for u in cr.get_cluster_members(l)
                if u in cr.user_features]
        if vecs:
            leaf_feat[l] = np.mean(vecs, axis=0)
    tags = ArchetypeProfiler()._compute_tags(leaf_feat)

    return {
        "n_threads": len(threads),
        "n_messages": n_msgs,
        "n_users": len(feats),
        "action_distribution": dict(action_dist.most_common()),
        "toxicity": {
            "n_scored": len(tox_vals),
            "mean": float(np.mean(tox_vals)) if tox_vals else None,
            "frac_high_0.6": float(np.mean([v >= 0.6 for v in tox_vals])) if tox_vals else None,
        },
        "feature_stats": feat_stats,
        "clustering": {
            "n_leaves": len(leaf_ids),
            "leaf_sizes": sizes,
            "silhouette": cr.silhouette_score,
            "davies_bouldin": cr.davies_bouldin_score,
            "out_of_cluster_users": n_noise,
            "out_of_cluster_frac": n_noise / max(1, len(cr.labels)),
            "leaf_tags": {int(k): v for k, v in tags.items()},
        },
    }


def save_report(report: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"EDA report → {out_path}")
    c = report.get("clustering", {})
    ooc = c.get("out_of_cluster_frac")
    ooc_str = f"out-of-cluster={ooc:.1%} " if isinstance(ooc, float) else ""
    actions = report.get("action_distribution") or {}
    logger.info(
        f"  leaves={c.get('n_leaves')} silhouette={c.get('silhouette', 0):.3f} "
        f"{ooc_str}actions={list(actions.items())[:6]}"
    )
