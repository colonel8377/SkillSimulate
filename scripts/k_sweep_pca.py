"""K sweep (3/4/5) on purified features, with and without PCA whitening.

Loads cached stream_counts / stream_features (no re-embedding), fits KMeans on
RobustScaler(5-95) features and on PCA-whitened features, and emits per-cluster
behavioral profiles for comparison.

Output: outputs/k3_diag/sweep_k.json and sweep_k_pca.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.features import VECTOR_FIELD_NAMES, UserFeatures  # noqa: E402
from src.clustering.streaming import (  # noqa: E402
    accum_to_features,
    find_year_dirs,
    stream_counts,
    stream_features,
)

DATA_DIR = "data/raw/wikiconv_en"
CACHE_DIR = "outputs/stream_cache"
OUT_DIR = Path("outputs/k3_diag")
MIN_MESSAGES = 5
CONTESTED_THRESHOLD = 3
TOXIC_THRESHOLD = 0.6
WORKERS = 8
K_GRID = [3, 4, 5]
RANDOM_STATE = 42
Z_THRESHOLD = 1.0

TAG_LEXICON: list[tuple[str, str, str]] = [
    ("hostility_score", "high", "hostile"),
    ("hostility_score", "low", "civil"),
    ("topical_breadth", "high", "generalist"),
    ("topical_breadth", "low", "specialist"),
    ("tenure", "high", "veteran"),
    ("tenure", "low", "newcomer"),
    ("mean_indentation", "high", "deep-threading"),
    ("reply_rate", "high", "responder"),
    ("reply_rate", "low", "thread-initiator"),
    ("question_rate", "high", "inquisitive"),
    ("verbosity", "high", "verbose"),
    ("verbosity", "low", "terse"),
    ("activity", "high", "high-activity"),
    ("activity_density", "high", "intense-burst"),
    ("burstiness_cv", "high", "bursty-poster"),
    ("namespace_focus", "high", "article-focused"),
    ("namespace_focus", "low", "interpersonal-space"),
    ("exclaim_rate", "high", "emphatic"),
    ("lexical_ttr", "high", "rich-vocabulary"),
]


def load_users() -> dict[str, UserFeatures]:
    year_dirs = find_year_dirs(DATA_DIR)
    counts, _ = stream_counts(year_dirs, workers=WORKERS, cache_dir=CACHE_DIR)
    active = {u for u, c in counts.items() if c >= MIN_MESSAGES}
    accums = stream_features(
        year_dirs, active, CONTESTED_THRESHOLD, TOXIC_THRESHOLD,
        workers=WORKERS, cache_dir=CACHE_DIR,
    )
    accums = {u: a for u, a in accums.items() if u in active and a.n > 0}
    return {u: accum_to_features(u, a) for u, a in accums.items()}


def compute_tags(leaf_feat: dict[int, np.ndarray]) -> dict[int, list[str]]:
    if len(leaf_feat) < 2:
        return {lid: [] for lid in leaf_feat}
    ids = list(leaf_feat)
    M = np.stack([leaf_feat[i] for i in ids])
    mu, sd = M.mean(0), M.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (M - mu) / sd
    field_idx = {f: i for i, f in enumerate(VECTOR_FIELD_NAMES)}
    out: dict[int, list[str]] = {}
    for row, lid in enumerate(ids):
        tags = []
        for feat, direction, label in TAG_LEXICON:
            if feat not in field_idx:
                continue
            z = Z[row, field_idx[feat]]
            if (direction == "high" and z >= Z_THRESHOLD) or (
                direction == "low" and z <= -Z_THRESHOLD
            ):
                tags.append(label)
        out[lid] = tags
    return out


def profile_k(X_raw: np.ndarray, X: np.ndarray, k: int) -> dict:
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    leaf_ids = sorted(set(labels.tolist()))
    leaf_size = {}
    leaf_feat = {}
    for lid in leaf_ids:
        mask = labels == lid
        leaf_size[lid] = int(mask.sum())
        leaf_feat[lid] = X_raw[mask].mean(axis=0)
    tags = compute_tags(leaf_feat)
    clusters = []
    for lid in leaf_ids:
        clusters.append({
            "leaf_id": lid,
            "size": leaf_size[lid],
            "frac": round(leaf_size[lid] / len(X_raw), 4),
            "tags": tags[lid],
            "raw_means": {
                name: round(float(val), 4)
                for name, val in zip(VECTOR_FIELD_NAMES, leaf_feat[lid])
            },
        })
    return {
        "k": k,
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "clusters": clusters,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached stream features...")
    user_features = load_users()
    user_ids = list(user_features)
    X_raw = np.stack([user_features[u].to_vector() for u in user_ids])
    print(f"N={len(user_ids)} users, D={X_raw.shape[1]}")

    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_scaled = scaler.fit_transform(X_raw)

    # PCA whitening: keep 95% variance, scale each component by 1/sqrt(eigval).
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca_unscaled = pca.fit_transform(X_scaled)
    eps = 1e-9
    X_pca = X_pca_unscaled / np.sqrt(pca.explained_variance_ + eps)
    print(f"PCA explains {pca.explained_variance_ratio_.sum():.3f} with {pca.n_components_} components")

    results_scaled = [profile_k(X_raw, X_scaled, k) for k in K_GRID]
    results_pca = [profile_k(X_raw, X_pca, k) for k in K_GRID]

    # PCA loadings for interpretability: which original features drive each PC.
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc_labels = []
    for pc in range(pca.n_components_):
        idx = np.argsort(np.abs(loadings[:, pc]))[::-1][:5]
        pc_labels.append({
            "pc": pc,
            "variance_ratio": round(float(pca.explained_variance_ratio_[pc]), 4),
            "top_features": [
                {"feature": VECTOR_FIELD_NAMES[i], "loading": round(float(loadings[i, pc]), 3)}
                for i in idx
            ],
        })

    report = {
        "n_users": len(user_ids),
        "d_original": X_raw.shape[1],
        "d_pca": int(pca.n_components_),
        "pca_loadings": pc_labels,
        "robust": results_scaled,
        "pca_whitened": results_pca,
    }

    out_path = OUT_DIR / "sweep_k.json"
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved sweep to {out_path}")

    print("\n--- RobustScaler only ---")
    for r in results_scaled:
        print(f"K={r['k']}: sil={r['silhouette']}, DB={r['davies_bouldin']}")
        for c in r["clusters"]:
            print(f"  leaf {c['leaf_id']}: n={c['size']} ({c['frac']:.1%}) tags={c['tags']}")

    print("\n--- PCA whitened ---")
    for r in results_pca:
        print(f"K={r['k']}: sil={r['silhouette']}, DB={r['davies_bouldin']}")
        for c in r["clusters"]:
            print(f"  leaf {c['leaf_id']}: n={c['size']} ({c['frac']:.1%}) tags={c['tags']}")


if __name__ == "__main__":
    main()
