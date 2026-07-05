"""
Generate final K=8 cluster result from the locked clustering decision.

Cross-year merge: a user may appear in multiple yearly v6 caches; we accumulate
all their action counters across years to form a single behavioural profile.
This matches the fact that clustering is a user-level policy, not per-year.

Uses: v6 detail cache + BGE embeddings (for language centroids only).
Clustering: behaviour-only KMeans K=8, RobustScaler(5-95).

Output: outputs/stream_cache/clustering_k8_final.pkl
"""

import sys, pickle, glob
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.streaming import accum_to_features, UserAccum
from src.clustering.features import UserFeatures
from src.clustering.clusterer import ClusterResult

VECTOR_FIELDS = [
    "reply_rate", "mean_indentation", "verbosity", "activity",
    "topical_breadth", "tenure", "hostility_score", "question_rate",
    "namespace_focus", "exclaim_rate", "lexical_ttr", "burstiness_cv",
    "activity_density",
]
SEED = 42
MIN_MSGS = 5
K = 8


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
    pkls = sorted(glob.glob("outputs/stream_cache/detail_v6_*.pkl"))
    print(f"Loading and merging {len(pkls)} yearly v6 caches...")
    merged: dict[str, UserAccum] = {}
    for p in pkls:
        print(f"  {p}")
        accums = pickle.load(open(p, "rb"))
        for uid, a in accums.items():
            merged[uid] = merge_accum(merged[uid], a) if uid in merged else a

    user_features = {uid: accum_to_features(uid, a) for uid, a in merged.items() if a.n >= MIN_MSGS}
    print(f"  merged users n>=5: {len(user_features)}")

    embeddings = pickle.load(open("outputs/stream_cache/embeddings.pkl", "rb"))
    user_ids = [u for u in user_features if u in embeddings]
    print(f"  joint with embeddings: {len(user_ids)}")

    X = np.stack([np.array([getattr(user_features[u], f) for f in VECTOR_FIELDS], dtype=np.float64) for u in user_ids])
    E = np.stack([embeddings[u].astype(np.float64) for u in user_ids])

    Xs = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(X)
    km = KMeans(n_clusters=K, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)

    sil = silhouette_score(Xs, labels, sample_size=50_000, random_state=SEED)
    db = davies_bouldin_score(Xs, labels)
    per_sample = silhouette_samples(Xs, labels)
    leaf_sil = {int(c): float(per_sample[labels == c].mean()) for c in range(K)}

    print(f"K={K} sil={sil:.4f} DB={db:.4f}")
    for c in range(K):
        n = int((labels == c).sum())
        print(f"  cluster {c}: {n:,} ({n/len(labels)*100:.1f}%)  leaf_sil={leaf_sil[c]:.4f}")

    beh_centroids = {int(c): Xs[labels == c].mean(axis=0) for c in range(K)}
    lang_centroids = {int(c): E[labels == c].mean(axis=0) for c in range(K)}

    cr = ClusterResult(
        labels={u: int(labels[i]) for i, u in enumerate(user_ids)},
        n_clusters=K,
        centroids=np.stack([lang_centroids[c] for c in range(K)]),
        silhouette_score=float(sil),
        davies_bouldin_score=float(db),
        behavioral_weight=1.0,
        language_weight=0.0,
        user_features=user_features,
        leaf_behavior_centroids=beh_centroids,
        leaf_language_centroids=lang_centroids,
        leaf_silhouette=leaf_sil,
        pre_impute_orphans=0,
        n_orphans_kept=0,
    )

    out_path = Path("outputs/stream_cache/clustering_k8_final.pkl")
    pickle.dump(cr, open(out_path, "wb"), protocol=4)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
