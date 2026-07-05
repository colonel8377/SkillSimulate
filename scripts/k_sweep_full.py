"""Full K-sweep (3-15) on merged v6 caches with behavior + language metrics.

Loads yearly v6 caches once, merges users across years, then fits behavior-only
KMeans for K=3..15 on the full 593k user set.  Reports the same metrics used in
role_k_sweep.py (action silhouette, davies-bouldin, mean leaf language-centroid
nearest cosine) so the full-data numbers can be compared directly to the
120k-sample sweep.

Output: outputs/k_sweep_full/sweep_results.json
"""

from __future__ import annotations

import sys, pickle, glob, json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.streaming import accum_to_features, UserAccum
from src.clustering.features import UserFeatures
from src.clustering.clusterer import ClusterResult
from src.skill.quality_report import _mean_centered_centroids, _nearest_other_cosine, _random_pair_cosine

VECTOR_FIELDS = [
    "reply_rate", "mean_indentation", "verbosity", "activity",
    "topical_breadth", "tenure", "hostility_score", "question_rate",
    "namespace_focus", "exclaim_rate", "lexical_ttr", "burstiness_cv",
    "activity_density",
]
SEED = 42
MIN_MSGS = 5
K_GRID = list(range(3, 16))
N_INIT = 10


def merge_accum(a: UserAccum, b: UserAccum) -> UserAccum:
    """Merge two per-year accumulators for the same user."""
    c = UserAccum()
    c.n = a.n + b.n
    c.n_discuss = a.n_discuss + b.n_discuss
    c.n_edit = a.n_edit + b.n_edit
    c.n_delete = a.n_delete + b.n_delete
    c.n_restore = a.n_restore + b.n_restore
    c.n_modified_received = a.n_modified_received + b.n_modified_received
    c.n_deleted_received = a.n_deleted_received + b.n_deleted_received
    c.n_restored_received = a.n_restored_received + b.n_restored_received
    c.reply_n = a.reply_n + b.reply_n
    c.reply_to_toxic_n = a.reply_to_toxic_n + b.reply_to_toxic_n
    c.indent_sum = a.indent_sum + b.indent_sum
    c.text_len_sum = a.text_len_sum + b.text_len_sum
    c.tox_sum = a.tox_sum + b.tox_sum
    c.tox_n = a.tox_n + b.tox_n
    c.tox_max = max(a.tox_max, b.tox_max)
    c.sev_sum = a.sev_sum + b.sev_sum
    c.sev_n = a.sev_n + b.sev_n
    c.sev_max = max(a.sev_max, b.sev_max)
    c.exclaim_n = a.exclaim_n + b.exclaim_n
    c.question_n = a.question_n + b.question_n
    c.wp_n = a.wp_n + b.wp_n
    c.contested_n = a.contested_n + b.contested_n
    c.ns_interpersonal_n = a.ns_interpersonal_n + b.ns_interpersonal_n
    c.ns_content_n = a.ns_content_n + b.ns_content_n
    c.ns_project_n = a.ns_project_n + b.ns_project_n
    c.ts_min = min(a.ts_min, b.ts_min) if (a.ts_min and b.ts_min) else (a.ts_min or b.ts_min)
    c.ts_max = max(a.ts_max, b.ts_max) if (a.ts_max and b.ts_max) else (a.ts_max or b.ts_max)
    c.ts_samples = sorted(set(a.ts_samples + b.ts_samples))[:300]
    c.page_ids = a.page_ids | b.page_ids
    c.out_targets = a.out_targets | b.out_targets
    c.in_repliers = a.in_repliers | b.in_repliers
    c.replies_received = a.replies_received + b.replies_received
    c.tox_recv_sum = a.tox_recv_sum + b.tox_recv_sum
    c.tox_recv_n = a.tox_recv_n + b.tox_recv_n
    c.hi_tox_texts = (a.hi_tox_texts + b.hi_tox_texts)[:15]
    c.lo_tox_texts = (a.lo_tox_texts + b.lo_tox_texts)[:15]
    return c


def main():
    out_dir = Path("outputs/k_sweep_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(glob.glob("outputs/stream_cache/detail_v6_*.pkl"))
    print(f"Loading and merging {len(pkls)} yearly v6 caches...")
    merged: dict[str, UserAccum] = {}
    for p in pkls:
        print(f"  {p}")
        accums = pickle.load(open(p, "rb"))
        for uid, a in accums.items():
            merged[uid] = merge_accum(merged[uid], a) if uid in merged else a

    user_features = {uid: accum_to_features(uid, a) for uid, a in merged.items() if a.n >= MIN_MSGS}
    print(f"  merged users n>={MIN_MSGS}: {len(user_features)}")

    embeddings = pickle.load(open("outputs/stream_cache/embeddings.pkl", "rb"))
    user_ids = [u for u in user_features if u in embeddings]
    print(f"  joint with embeddings: {len(user_ids)}")

    X = np.stack([np.array([getattr(user_features[u], f) for f in VECTOR_FIELDS], dtype=np.float64) for u in user_ids])
    E = np.stack([embeddings[u].astype(np.float64) for u in user_ids])

    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X)
    random_pair_cos = _random_pair_cosine({u: embeddings[u] for u in user_ids})
    print(f"random_pair_cosine (K-independent floor): {random_pair_cos:.4f}\n")

    print(f"{'K':>3}{'n_leaves':>10}{'action_sil':>12}{'davies_bouldin':>16}{'mean_leaf_cos':>16}{'ratio_to_floor':>16}")
    print("=" * 77)

    results = []
    for k in K_GRID:
        Xs = scaler.transform(X)
        km = KMeans(n_clusters=k, random_state=SEED, n_init=N_INIT)
        labels = km.fit_predict(Xs)

        sil = silhouette_score(Xs, labels, sample_size=50_000, random_state=SEED)
        db = davies_bouldin_score(Xs, labels)
        per_sample = silhouette_samples(Xs, labels)
        leaf_sil = {int(c): float(per_sample[labels == c].mean()) for c in range(k)}

        lang_centroids = {int(c): E[labels == c].mean(axis=0) for c in range(k)}
        centered = _mean_centered_centroids(lang_centroids)
        cosines = [c for c in (_nearest_other_cosine(cid, centered) for cid in centered) if c is not None]
        mean_leaf_cos = float(np.mean(cosines)) if cosines else float("nan")
        ratio = mean_leaf_cos / random_pair_cos if random_pair_cos else float("nan")

        print(f"{k:>3}{k:>10}{sil:>12.4f}{db:>16.4f}{mean_leaf_cos:>16.4f}{ratio:>16.1f}")

        cluster_sizes = {int(c): int((labels == c).sum()) for c in range(k)}
        results.append({
            "k": k,
            "n_users": len(user_ids),
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "leaf_silhouette": leaf_sil,
            "mean_leaf_nearest_cosine": mean_leaf_cos,
            "random_pair_cosine": random_pair_cos,
            "ratio_to_floor": ratio,
            "cluster_sizes": cluster_sizes,
        })

    out_path = out_dir / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
