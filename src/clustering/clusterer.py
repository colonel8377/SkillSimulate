"""Behavioral clustering with two-stage feature extraction.

Stage 1: behavioral signals (reply depth, edit freq, stance shift, conflict ratio)
Stage 2: language embeddings (Sentence-BERT)
Combined via adaptive weight concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.clustering.embeddings import EmbeddingExtractor
from src.clustering.features import FeatureExtractor, UserFeatures
from src.data.schemas import Message, Thread


@dataclass
class ClusterResult:
    """Result of behavioral clustering."""
    labels: dict[str, int]               # user_id → cluster_id
    n_clusters: int
    centroids: np.ndarray
    silhouette_score: float
    davies_bouldin_score: float
    behavioral_weight: float
    language_weight: float
    user_features: dict[str, UserFeatures]

    def get_cluster_members(self, cluster_id: int) -> list[str]:
        return [uid for uid, cid in self.labels.items() if cid == cluster_id]

    def get_cluster_ids(self) -> list[int]:
        return sorted(set(self.labels.values()))


class BehavioralClusterer:
    """Two-stage behavioral + language clustering."""

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 4,
        behavioral_weight: float | None = None,
        language_weight: float | None = None,
        max_k: int = 8,
        min_k: int = 2,
        random_state: int = 42,
    ):
        self.method = method
        self.n_clusters = n_clusters
        # None = adaptive (data-driven); explicit values = fixed weights
        self.behavioral_weight = behavioral_weight
        self.language_weight = language_weight
        self._adaptive = behavioral_weight is None or language_weight is None
        self.max_k = max_k
        self.min_k = min_k
        self.random_state = random_state

        self.feature_extractor = FeatureExtractor()
        self.embedding_extractor = EmbeddingExtractor()

    def fit(self, threads: list[Thread]) -> ClusterResult:
        """Cluster all users from threads.

        Args:
            threads: List of conversation threads.

        Returns:
            ClusterResult with assignments and quality metrics.
        """
        # Stage 1: behavioral features
        user_features = self.feature_extractor.extract_all(threads)

        # Stage 2: language embeddings
        user_messages = self._group_messages_by_user(threads)
        user_embeddings = self.embedding_extractor.embed_all_users(user_messages)

        # Combine features
        user_ids = list(user_features.keys())
        behavioral_matrix = np.stack([
            user_features[uid].to_vector() for uid in user_ids
        ])
        language_matrix = np.stack([
            user_embeddings[uid] for uid in user_ids
        ])

        # Normalize
        behavioral_scaler = StandardScaler()
        language_scaler = StandardScaler()
        behavioral_normed = behavioral_scaler.fit_transform(behavioral_matrix)
        language_normed = language_scaler.fit_transform(language_matrix)

        # Adaptive weight concatenation
        if self._adaptive:
            bw, lw = self._compute_adaptive_weights(behavioral_normed, language_normed)
        else:
            bw = self.behavioral_weight
            lw = self.language_weight

        combined = np.hstack([
            bw * behavioral_normed,
            lw * language_normed,
        ])

        # Cluster
        if self.method == "hdbscan":
            labels, centroids, k = self._cluster_hdbscan(combined)
        else:
            k, labels, centroids = self._select_k_and_cluster(combined)

        # Quality metrics
        sil = silhouette_score(combined, labels) if len(set(labels)) > 1 else 0.0
        db = davies_bouldin_score(combined, labels) if len(set(labels)) > 1 else 0.0

        label_dict = {uid: int(lbl) for uid, lbl in zip(user_ids, labels)}

        return ClusterResult(
            labels=label_dict,
            n_clusters=k,
            centroids=centroids,
            silhouette_score=sil,
            davies_bouldin_score=db,
            behavioral_weight=bw,
            language_weight=lw,
            user_features=user_features,
        )

    def _compute_adaptive_weights(
        self,
        behavioral: np.ndarray,
        language: np.ndarray,
    ) -> tuple[float, float]:
        """Compute data-driven weights based on feature variance ratio.

        Higher-variance feature space gets proportionally more weight,
        capturing which modality carries more discriminative signal.
        """
        # Mean per-feature variance (averaged across dimensions)
        beh_var = float(np.mean(np.var(behavioral, axis=0)))
        lang_var = float(np.mean(np.var(language, axis=0)))

        if beh_var + lang_var == 0:
            raise ValueError(
                "Behavioral and language feature variances are both zero; "
                "cannot compute adaptive clustering weights."
            )

        # Weight proportional to variance contribution
        bw = beh_var / (beh_var + lang_var)
        lw = 1.0 - bw

        # Clamp to reasonable range [0.2, 0.8] to avoid one modality dominating
        bw = max(0.2, min(0.8, bw))
        lw = 1.0 - bw

        return bw, lw

    def _select_k_and_cluster(self, X: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """Select optimal K via silhouette + Davies-Bouldin, then cluster."""
        if self.n_clusters > 0:
            k = min(self.n_clusters, len(X) - 1)
            labels, centroids = self._kmeans(X, k)
            return k, labels, centroids

        best_k = self.min_k
        best_score = -1.0
        for k in range(self.min_k, min(self.max_k + 1, len(X))):
            labels, _ = self._kmeans(X, k)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            # Combined score: maximize silhouette, minimize DB
            score = sil - 0.5 * db
            if score > best_score:
                best_score = score
                best_k = k

        labels, centroids = self._kmeans(X, best_k)
        return best_k, labels, centroids

    def _kmeans(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X)
        return labels, km.cluster_centers_

    def _cluster_hdbscan(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(X) // 10))
        labels = clusterer.fit_predict(X)
        # HDBSCAN may produce noise (-1); assign noise to nearest cluster
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            # All noise — fall back to KMeans
            return self._kmeans(X, self.min_k)
        k = len(unique_labels)
        # Compute centroids
        centroids = np.zeros((k, X.shape[1]))
        for i, lbl in enumerate(sorted(unique_labels)):
            centroids[i] = X[labels == lbl].mean(axis=0)
        # Assign noise points to nearest centroid
        noise_mask = labels == -1
        if noise_mask.any():
            from scipy.spatial.distance import cdist
            dists = cdist(X[noise_mask], centroids)
            nearest = dists.argmin(axis=1)
            labels[noise_mask] = nearest
        return labels, centroids, k

    def _group_messages_by_user(self, threads: list[Thread]) -> dict[str, list[Message]]:
        """Group all messages by user across threads."""
        from collections import defaultdict
        user_msgs: dict[str, list[Message]] = defaultdict(list)
        for thread in threads:
            for msg in thread.messages:
                user_msgs[msg.user_id].append(msg)
        return user_msgs


# ---------------------------------------------------------------------------
# State-level cluster-validity helpers (R4 persona-collapse stress test,
# outline §5.6.7 / docs/r4_persona_collapse_stress_test.md §4.2).
#
# These compute silhouette / Davies-Bouldin on pre-aggregated embeddings and
# a FROZEN cluster count K, which is required for the longitudinal drift
# measurement: re-optimizing K per turn would confound the drift signal.
# ---------------------------------------------------------------------------


def silhouette_at_state(
    embeddings: np.ndarray,
    labels: np.ndarray,
    K_frozen: int | None = None,
) -> float:
    """Silhouette score on a snapshot of agent states (R4 §4.2).

    Args:
        embeddings: (n_agents, embed_dim) array of per-agent state embeddings
            at the current measurement point.
        labels: (n_agents,) cluster assignment for each agent. Used as a
            clustering-sanity check; the silhouette is computed on the
            CURRENT re-clustering at frozen K.
        K_frozen: Frozen cluster count. If provided and labels were
            re-fit on this snapshot with this K, the silhouette is
            well-defined. If None, labels are taken as-is.

    Returns:
        Silhouette score in [-1, 1]. 0.0 if fewer than 2 clusters present.
    """
    if embeddings.shape[0] < 2 or len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) < 2:
        return 0.0
    return float(silhouette_score(embeddings, labels))


def davies_bouldin_at_state(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Davies-Bouldin index on a snapshot of agent states (R4 §4.2).

    Returns:
        DB index (≥0, lower is better). 0.0 if fewer than 2 clusters.
    """
    if embeddings.shape[0] < 2 or len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) < 2:
        return 0.0
    return float(davies_bouldin_score(embeddings, labels))


def refit_labels_frozen_k(
    embeddings: np.ndarray,
    K_frozen: int,
    random_state: int = 42,
) -> np.ndarray:
    """Re-fit K-Means with frozen K on a state snapshot (R4 §4.2).

    Critical: the R4 protocol requires K to be frozen from §5.2 Step 1.
    Allowing K to re-optimize per turn would confound the drift signal.

    Args:
        embeddings: (n_agents, embed_dim) state snapshot.
        K_frozen: Frozen K from the initial clustering (§5.2 Step 1).
        random_state: RNG seed for K-Means reproducibility.

    Returns:
        (n_agents,) integer cluster labels.
    """
    k = min(K_frozen, embeddings.shape[0])
    if k < 2:
        return np.zeros(embeddings.shape[0], dtype=int)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(embeddings)
