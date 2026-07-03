"""
Step C: Style Embedding Ablation.

Uses the existing embeddings.pkl (BGE-large mean+std, 2048-dim, ~594k users)
to extract a low-dimensional style component via PCA, then appends it to the
13-dim behavior vector with varying weights w ∈ {0.02, 0.05, 0.10, 0.20}.

Key design choices:
  - PCA on the full 2048-dim embedding space; retain n_components=5 (style axes)
  - Only users present in BOTH the v6 feature set AND embeddings.pkl are included
  - Weights are applied BEFORE RobustScaler so the style PC magnitudes are
    comparable to the behavior features in the scaled space
  - We measure: overall silhouette/DB, and leaf-3 internal silhouette
    (the 52% "other" cluster that we want to break up)

Outputs:
  outputs/k3_diag/sweep_style.json
"""

import json, sys, pickle, glob
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clustering.streaming import accum_to_features

OUT_DIR = Path("outputs/k3_diag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_FIELDS = [
    "reply_rate", "mean_indentation", "verbosity", "activity",
    "topical_breadth", "tenure", "hostility_score", "question_rate",
    "namespace_focus", "exclaim_rate", "lexical_ttr", "burstiness_cv",
    "activity_density",
]

STYLE_COMPONENTS = 10
K = 4
WEIGHTS = [0.0, 0.02, 0.05, 0.10, 0.20]
SEED = 42
MIN_MSGS = 5


def load_behavior() -> tuple[list, np.ndarray]:
    """Returns (user_ids, feature_matrix) for active users from v6 cache."""
    pkls = sorted(glob.glob("outputs/stream_cache/detail_v6_*.pkl"))
    uids, rows = [], []
    for p in pkls:
        accums = pickle.load(open(p, "rb"))
        for uid, a in accums.items():
            if a.n >= MIN_MSGS:
                f = accum_to_features(uid, a)
                uids.append(uid)
                rows.append([getattr(f, field) for field in VECTOR_FIELDS])
    return uids, np.array(rows, dtype=np.float64)


def load_embeddings(path="outputs/stream_cache/embeddings_styledistance.pkl") -> dict:
    return pickle.load(open(path, "rb"))


def main():
    print("Loading behavior features from v6 cache…")
    uids, X_beh = load_behavior()
    print(f"  behavior N={len(uids)}, D={X_beh.shape[1]}")

    print("Loading style embeddings…")
    embed_dict = load_embeddings()
    print(f"  embedding N={len(embed_dict)}, dim={next(iter(embed_dict.values())).shape[0]}")

    # Intersect: only users with both behavior and embedding
    uid_set = set(embed_dict.keys())
    mask = np.array([u in uid_set for u in uids])
    uids_joint = [u for u, m in zip(uids, mask) if m]
    X_beh_joint = X_beh[mask]
    E = np.stack([embed_dict[u].astype(np.float64) for u in uids_joint])
    print(f"  joint N={len(uids_joint)} (behavior∩embedding)")

    # PCA on embedding space → style components
    print(f"  PCA(n={STYLE_COMPONENTS}) on {E.shape[1]}-dim embedding…")
    pca = PCA(n_components=STYLE_COMPONENTS, random_state=SEED)
    S = pca.fit_transform(E)  # (N, STYLE_COMPONENTS)
    print(f"  explained variance: {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%")

    results = []

    # Scale behavior features once (baseline scaler)
    beh_scaler = RobustScaler(quantile_range=(5, 95))
    X_beh_scaled = beh_scaler.fit_transform(X_beh_joint)

    # Normalize style PCs to unit std so weight is interpretable as a fraction
    # of behavior feature magnitude
    from sklearn.preprocessing import StandardScaler
    style_scaler = StandardScaler()
    S_scaled = style_scaler.fit_transform(S)  # each PC: mean=0, std=1

    for w in WEIGHTS:
        label = f"w={w:.2f}"
        if w == 0.0:
            Xs = X_beh_scaled
            label = "baseline (no style)"
        else:
            Xs = np.hstack([X_beh_scaled, w * S_scaled])

        km = KMeans(n_clusters=K, n_init=20, random_state=SEED)
        labels = km.fit_predict(Xs)

        sil_global = silhouette_score(Xs, labels, sample_size=50_000, random_state=SEED)
        db_global  = davies_bouldin_score(Xs, labels)

        # Cluster size distribution
        sizes = {int(c): int((labels == c).sum()) for c in range(K)}
        fracs = {c: round(v / len(labels), 4) for c, v in sizes.items()}
        largest_frac = max(fracs.values())

        # Per-leaf silhouette (using sampled silhouette for speed)
        samp_idx = np.random.RandomState(SEED).choice(len(labels), size=min(50_000, len(labels)), replace=False)
        samp_sil = silhouette_samples(Xs[samp_idx], labels[samp_idx])
        leaf_sil = {}
        for c in range(K):
            mask_c = labels[samp_idx] == c
            if mask_c.sum() > 0:
                leaf_sil[c] = round(float(samp_sil[mask_c].mean()), 4)

        print(f"  [{label}]  sil={sil_global:.4f}  DB={db_global:.4f}  "
              f"sizes={fracs}  leaf_sil={leaf_sil}")

        results.append({
            "weight": w, "label": label, "k": K,
            "silhouette": round(sil_global, 4),
            "davies_bouldin": round(db_global, 4),
            "cluster_fracs": fracs,
            "largest_cluster_frac": largest_frac,
            "leaf_silhouette": leaf_sil,
            "style_pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        })

    out_path = OUT_DIR / "sweep_style.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'label':<28} {'sil':>8} {'DB':>8} {'largest_frac':>14}")
    for r in results:
        print(f"{r['label']:<28} {r['silhouette']:>8.4f} {r['davies_bouldin']:>8.4f} {r['largest_cluster_frac']:>14.4f}")


if __name__ == "__main__":
    main()
