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

    @staticmethod
    def validate_locked_vectors(
        cluster_result: ClusterResult,
        merge_map: dict[int, int] | None = None,
        n_iterations: int = 20,
        train_sample_size: int = 30_000,
        eval_sample_size: int = 10_000,
        random_state: int = 42,
        ari_mean_threshold: float = 0.80,
        ari_variance_threshold: float = 0.02,
    ) -> dict:
        """Bootstrap stability of a locked QuantileTransformer+KMeans partition.

        Each iteration resamples users with replacement, refits the same
        preprocessing and KMeans family used by the locked WikiConv artifact,
        and predicts a fixed evaluation sample. ARI is label-permutation
        invariant, so fitted cluster IDs need not match the canonical IDs.
        """
        from scipy.optimize import linear_sum_assignment
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import QuantileTransformer

        source_labels = getattr(cluster_result, "source_labels", cluster_result.labels)
        user_ids = [
            uid for uid in source_labels
            if uid in cluster_result.user_features and source_labels[uid] >= 0
        ]
        if len(user_ids) < 100:
            raise ValueError("Locked clustering has too few labeled user vectors")
        X = np.stack([
            cluster_result.user_features[uid].to_vector() for uid in user_ids
        ])
        y_source = np.asarray([source_labels[uid] for uid in user_ids], dtype=int)
        merge_map = dict(merge_map or {})
        y_final = np.asarray([merge_map.get(int(v), int(v)) for v in y_source], dtype=int)
        source_ids = sorted(set(y_source.tolist()))
        n_clusters = len(source_ids)
        if n_clusters < 2:
            raise ValueError("Locked clustering must contain at least two clusters")

        rng = np.random.default_rng(random_state)
        eval_n = min(eval_sample_size, len(user_ids))
        eval_idx = rng.choice(len(user_ids), size=eval_n, replace=False)
        X_eval = X[eval_idx]
        y_eval_source = y_source[eval_idx]
        y_eval_final = y_final[eval_idx]
        train_n = min(train_sample_size, len(user_ids))
        source_scores: list[float] = []
        final_scores: list[float] = []
        for iteration in range(n_iterations):
            train_idx = rng.choice(len(user_ids), size=train_n, replace=True)
            X_train = X[train_idx]
            y_train_source = y_source[train_idx]
            scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, len(X_train)),
                random_state=random_state + iteration,
            )
            X_train_scaled = scaler.fit_transform(X_train)
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state + iteration,
                n_init=5,
            ).fit(X_train_scaled)
            predicted_train = model.labels_
            predicted_eval = model.predict(scaler.transform(X_eval))
            source_scores.append(float(adjusted_rand_score(y_eval_source, predicted_eval)))

            # Map arbitrary fitted IDs to canonical source-K8 IDs using the
            # bootstrap training sample, then apply the pre-registered merge.
            fitted_ids = sorted(set(predicted_train.tolist()))
            contingency = np.zeros((len(fitted_ids), len(source_ids)), dtype=int)
            fitted_pos = {v: i for i, v in enumerate(fitted_ids)}
            source_pos = {v: i for i, v in enumerate(source_ids)}
            for pred, truth in zip(predicted_train, y_train_source):
                contingency[fitted_pos[int(pred)], source_pos[int(truth)]] += 1
            rows, cols = linear_sum_assignment(-contingency)
            aligned = {fitted_ids[r]: source_ids[c] for r, c in zip(rows, cols)}
            predicted_final = np.asarray([
                merge_map.get(aligned[int(v)], aligned[int(v)]) for v in predicted_eval
            ])
            final_scores.append(float(adjusted_rand_score(y_eval_final, predicted_final)))

        source_values = np.asarray(source_scores, dtype=float)
        values = np.asarray(final_scores, dtype=float)
        is_stable = bool(
            values.mean() >= ari_mean_threshold
            and values.var() <= ari_variance_threshold
        )
        return {
            "protocol": "locked_k8_merge_bootstrap_v2",
            "preprocessor": "QuantileTransformer(output_distribution=normal)",
            "clusterer": f"KMeans(k={n_clusters}, n_init=5) then canonical merge",
            "n_users_available": len(user_ids),
            "n_iterations": n_iterations,
            "train_sample_size": train_n,
            "eval_sample_size": eval_n,
            "source_k8_ari_scores": source_scores,
            "source_k8_ari_mean": float(source_values.mean()),
            "source_k8_ari_variance": float(source_values.var()),
            "ari_scores": final_scores,
            "ari_mean": float(values.mean()),
            "ari_variance": float(values.var()),
            "ari_ci_low": float(np.quantile(values, 0.025)),
            "ari_ci_high": float(np.quantile(values, 0.975)),
            "ari_mean_threshold": ari_mean_threshold,
            "ari_variance_threshold": ari_variance_threshold,
            "is_stable": is_stable,
        }
