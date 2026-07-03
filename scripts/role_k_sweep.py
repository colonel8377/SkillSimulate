"""How does role_k trade off action-cluster quality vs. language separability?

Each manual `export-corpus --role-k N` run pays the FULL pipeline cost (thread
loading, distillation encode() calls for every leaf's members) just to answer
"what's cr.silhouette_score and mean_leaf_nearest_cosine at this K" — that's
the only thing a K-sweep actually needs. This script pays the one-time cost of
loading cached per-year features (stream_counts/stream_features, already on
disk from prior runs — no GPU re-embedding) and cached embeddings.pkl ONCE,
then re-fits BehavioralClusterer fresh (KMeans only, seconds each) for every K
in ROLE_K_GRID and reuses quality_report.py's diagnostic functions — no
distillation, no file writes. That turns N manual reruns into one script run.

Mirrors the CLI defaults in main.py::_build_clusterer (method=two_stage,
role_method=kmeans, scaler=robust, target_min_leaves=30/max_leaves=80,
random_state=42) so results are directly comparable to the manual runs already
done at role_k=4/12/22.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.clusterer import BehavioralClusterer  # noqa: E402
from src.clustering.streaming import (  # noqa: E402
    find_year_dirs, stream_counts, stream_features, accum_to_features,
)
from src.skill.quality_report import (  # noqa: E402
    _mean_centered_centroids, _nearest_other_cosine, _random_pair_cosine,
)

DATA_DIR = "data/raw/wikiconv_en"
CACHE_DIR = "outputs/stream_cache"
MIN_MESSAGES = 5
CONTESTED_THRESHOLD = 3
TOXIC_THRESHOLD = 0.6
WORKERS = 8
ROLE_K_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 22]


def main():
    year_dirs = find_year_dirs(DATA_DIR)
    counts, _ = stream_counts(year_dirs, workers=WORKERS, cache_dir=CACHE_DIR)
    active = {u for u, c in counts.items() if c >= MIN_MESSAGES}
    accums = stream_features(
        year_dirs, active, CONTESTED_THRESHOLD, TOXIC_THRESHOLD,
        workers=WORKERS, cache_dir=CACHE_DIR,
    )
    accums = {u: a for u, a in accums.items() if u in active and a.n > 0}
    user_features = {u: accum_to_features(u, a) for u, a in accums.items()}

    import pickle
    with open(Path(CACHE_DIR) / "embeddings.pkl", "rb") as f:
        cached_emb = pickle.load(f)
    user_embeddings = {u: v for u, v in cached_emb.items() if u in accums}
    print(f"{len(user_features)} users, {len(user_embeddings)} with embeddings")

    random_pair_cos = _random_pair_cosine(user_embeddings)
    print(f"random_pair_cosine (K-independent floor): {random_pair_cos:.4f}\n")

    print(f"{'role_k':>7}{'n_leaves':>10}{'action_sil':>12}{'davies_bouldin':>16}{'mean_leaf_cos':>16}{'ratio_to_floor':>16}")
    print("=" * 77)
    for k in ROLE_K_GRID:
        clusterer = BehavioralClusterer(
            method="two_stage", role_method="kmeans", role_k=k,
            target_min_leaves=30, target_max_leaves=80,
            scaler="robust", impute_orphans=False,
            cluster_selection_method="eom", random_state=42,
        )
        cr = clusterer.fit_from_vectors(user_features, user_embeddings)
        leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]
        centered = _mean_centered_centroids(cr.leaf_language_centroids)
        cosines = [c for lid in leaf_ids if (c := _nearest_other_cosine(lid, centered)) is not None]
        mean_leaf_cos = float(np.mean(cosines)) if cosines else float("nan")
        ratio = mean_leaf_cos / random_pair_cos if random_pair_cos else float("nan")
        print(f"{k:>7}{len(leaf_ids):>10}{cr.silhouette_score:>12.4f}{cr.davies_bouldin_score:>16.4f}{mean_leaf_cos:>16.4f}{ratio:>16.1f}")


if __name__ == "__main__":
    main()
