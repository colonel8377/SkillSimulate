"""Cluster stability validation via bootstrap resampling.

Computes Adjusted Rand Index (ARI) across resampled datasets.
ARI variance < 0.2 indicates stable clustering.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.clustering.clusterer import BehavioralClusterer, ClusterResult
from src.data.schemas import Thread


class ClusterStabilityValidator:
    """Validates cluster stability via bootstrap resampling."""

    def __init__(
        self,
        n_iterations: int = 100,
        sample_ratio: float = 0.8,
        ari_variance_threshold: float = 0.2,
        random_state: int = 42,
    ):
        self.n_iterations = n_iterations
        self.sample_ratio = sample_ratio
        self.ari_variance_threshold = ari_variance_threshold
        self.random_state = random_state

    def validate(
        self,
        threads: list[Thread],
        clusterer: BehavioralClusterer,
    ) -> dict:
        """Run bootstrap validation.

        Returns:
            Dict with:
                - ari_scores: list of ARI per iteration
                - ari_mean: mean ARI
                - ari_variance: variance of ARI
                - is_stable: bool (variance < threshold)
        """
        rng = np.random.RandomState(self.random_state)

        # Original clustering
        original = clusterer.fit(threads)
        original_labels = original.labels

        ari_scores = []
        for i in range(self.n_iterations):
            # Resample threads with replacement
            n_sample = max(int(len(threads) * self.sample_ratio), 2)
            sampled_indices = rng.choice(len(threads), size=n_sample, replace=True)
            sampled_threads = [threads[idx] for idx in sampled_indices]

            # Re-cluster
            try:
                resampled = clusterer.fit(sampled_threads)
            except Exception:
                continue

            # Compute ARI on common users
            common_users = set(original_labels) & set(resampled.labels)
            if len(common_users) < 3:
                continue

            orig = [original_labels[u] for u in common_users]
            re = [resampled.labels[u] for u in common_users]

            ari = adjusted_rand_score(orig, re)
            ari_scores.append(ari)

        ari_array = np.array(ari_scores) if ari_scores else np.array([0.0])
        ari_mean = float(ari_array.mean())
        ari_variance = float(ari_array.var())

        return {
            "ari_scores": ari_scores,
            "ari_mean": ari_mean,
            "ari_variance": ari_variance,
            "is_stable": ari_variance < self.ari_variance_threshold,
            "n_successful_iterations": len(ari_scores),
        }

    @staticmethod
    def select_k(
        threads: list[Thread],
        clusterer: BehavioralClusterer,
        k_range: range,
    ) -> dict[int, dict[str, float]]:
        """Evaluate clustering quality across K values.

        Returns:
            Dict K → {silhouette, davies_bouldin} for each K.
        """
        results = {}
        for k in k_range:
            clusterer.n_clusters = k
            try:
                res = clusterer.fit(threads)
                results[k] = {
                    "silhouette": res.silhouette_score,
                    "davies_bouldin": res.davies_bouldin_score,
                }
            except Exception:
                results[k] = {"silhouette": -1.0, "davies_bouldin": float("inf")}
        return results
