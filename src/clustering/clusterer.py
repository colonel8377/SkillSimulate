"""Behaviour-only archetype clustering.

Users are partitioned by their action-policy behaviour vector into interpretable
role groups — one role = one distillation unit. KMeans is the default: the
behavioural space is a near-continuous manifold with no density gaps, so density
clustering (HDBSCAN) rejects most users as noise, whereas KMeans gives a
balanced, zero-orphan partition. HDBSCAN stays selectable for corpora with
density-separated roles.

Language style is deliberately NOT clustered. Averaged per-leaf language
centroids do not separate (verified at N=593,871: max inter-leaf cosine 0.063
vs 0.158 for random user pairs — sub-prototypes collapse to their role mean), so
sub-clustering roles on language only fragments them without recovering real
structure. Language is captured downstream at compile time by sampling
representative utterances per leaf (Expression DNA, outline §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

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
    # Per-leaf centroids in both spaces (downstream skill compile / repr selection).
    leaf_behavior_centroids: dict = field(default_factory=dict)
    leaf_language_centroids: dict = field(default_factory=dict)
    pre_impute_orphans: int = 0           # HDBSCAN noise count before imputation
    n_orphans_kept: int = 0               # -1 users left un-imputed
    leaf_silhouette: dict = field(default_factory=dict)  # leaf_id → mean silhouette

    def get_cluster_members(self, cluster_id: int) -> list[str]:
        return [uid for uid, cid in self.labels.items() if cid == cluster_id]

    def get_cluster_ids(self) -> list[int]:
        return sorted(set(self.labels.values()))


class BehavioralClusterer:
    """Behaviour-only archetype clustering: users → role leaves.

    Partition users by their *action policy* (the behavioural feature vector)
    into interpretable role groups — KMeans by default (see ``role_method``),
    HDBSCAN selectable for density-separated corpora. A leaf = one role = one
    distillation unit.

    Language is not part of the clustering objective (averaged language
    centroids do not separate — see module docstring); it is captured at
    skill-compile time. Under HDBSCAN, stage-1 noise is kept as label ``-1``
    (out-of-archetype long-tail) and never force-merged; under KMeans there is
    no noise.
    """

    def __init__(
        self,
        method: str = "two_stage",
        n_clusters: int = -1,                 # back-compat single-stage only
        role_method: str = "kmeans",          # stage-1 role: "kmeans" | "hdbscan"
        role_k: int = 12,                     # number of role clusters when role_method="kmeans"
        role_min_cluster_size: int | None = None,
        role_min_samples: int | None = None,
        # legacy style-stage params — language is no longer clustered; kept
        # inert for back-compat with main.py / config / streaming cache callers.
        style_min_cluster_size: int | None = None,
        style_min_samples: int | None = None,
        min_style_silhouette: float = 0.10,
        target_min_leaves: int = 30,
        target_max_leaves: int = 80,
        scaler: str = "robust",               # "standard" | "robust"
        impute_orphans: bool = False,         # keep -1 by default
        cluster_selection_method: str = "eom",  # "eom" | "leaf"
        max_k: int = 8,
        min_k: int = 2,
        random_state: int = 42,
        max_msgs_per_user: int | None = 25,   # cap per-user msgs for embedding
        metric_sample: int = 10_000,          # silhouette/DB subsample cap
        style_method: str = "umap",           # legacy (inert; was style-stage reduction)
        style_umap_dim: int = 15,             # legacy (inert; was UMAP output dim)
        # back-compat (single-stage concat weights); unused by two_stage
        behavioral_weight: float | None = None,
        language_weight: float | None = None,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.role_method = role_method
        self.role_k = role_k
        self.role_min_cluster_size = role_min_cluster_size
        self.role_min_samples = role_min_samples
        self.style_min_cluster_size = style_min_cluster_size
        self.style_min_samples = style_min_samples
        self.min_style_silhouette = min_style_silhouette
        self.target_min_leaves = target_min_leaves
        self.target_max_leaves = target_max_leaves
        self.scaler = scaler
        self.impute_orphans = impute_orphans
        self.cluster_selection_method = cluster_selection_method
        self.max_k = max_k
        self.min_k = min_k
        self.random_state = random_state
        self.max_msgs_per_user = max_msgs_per_user
        self.metric_sample = metric_sample
        self.style_method = style_method
        self.style_umap_dim = style_umap_dim
        self.behavioral_weight = behavioral_weight
        self.language_weight = language_weight

        self.feature_extractor = FeatureExtractor()
        self.embedding_extractor = EmbeddingExtractor()

    def fit(self, threads: list[Thread]) -> ClusterResult:
        # ---- features (behaviour) + language embeddings ----
        user_features = self.feature_extractor.extract_all(threads)

        user_msgs = self._group_messages_by_user(threads)
        logger.info(f"Clustering {len(user_features)} users")
        user_embeddings = self.embedding_extractor.embed_all_users(user_msgs)

        return self._finish(user_features, user_embeddings)

    def fit_from_vectors(
        self,
        user_features: dict,
        user_embeddings: dict,
    ) -> ClusterResult:
        """Cluster from pre-computed features + embeddings (streaming path).

        ``user_features`` / ``user_embeddings`` share the same user-id keys.
        """
        return self._finish(user_features, user_embeddings)

    def _finish(self, user_features: dict, user_embeddings: dict) -> ClusterResult:
        user_ids = [u for u in user_features if u in user_embeddings]
        if len(user_ids) < self.min_k:
            labels = {u: 0 for u in user_features}
            return ClusterResult(
                labels=labels, n_clusters=1,
                centroids=np.zeros((1, 1)), silhouette_score=0.0,
                davies_bouldin_score=0.0, behavioral_weight=0.0,
                language_weight=0.0, user_features=user_features,
            )

        behavioral = self._get_scaler().fit_transform(
            np.stack([user_features[u].to_vector() for u in user_ids])
        )
        language = np.stack([user_embeddings[u] for u in user_ids])

        # Language is NOT clustered (averaged centroids don't separate — see
        # module docstring); raw embeddings are kept only for per-leaf language
        # centroids used in downstream representative-utterance selection at
        # compile time. No UMAP / dimensionality reduction is needed.
        if self.method in ("kmeans", "hdbscan"):
            # back-compat single-stage ablation path: clusters on concatenated
            # behaviour+language (the baseline against which behaviour-only was
            # shown to win). Receives the raw language embedding.
            leaf_labels = self._single_stage(behavioral, language)
        else:
            leaf_labels = self._two_stage(behavioral)

        n_pre_noise = int((leaf_labels == -1).sum())
        logger.info(
            f"Leaf labels: {(leaf_labels >= 0).sum()} assigned, "
            f"{n_pre_noise} orphans ({n_pre_noise/len(user_ids):.1%}) before imputation"
        )

        # Orphan handling: keep -1 by default; only impute when explicitly asked.
        leaf_ids = sorted(set(leaf_labels.tolist()) - {-1})
        n_orphans_kept = n_pre_noise
        if self.impute_orphans and leaf_ids:
            leaf_labels = self._impute_orphans(leaf_labels, behavioral, leaf_ids)
            n_orphans_kept = 0

        # sanity metrics on the behavioural space — the space we actually
        # clustered on (language is not part of the objective).
        mask = leaf_labels >= 0
        uniq = set(leaf_labels[mask].tolist())
        sil = db = 0.0
        leaf_sil: dict[int, float] = {}
        if len(uniq) > 1 and mask.sum() > len(uniq):
            idx = np.where(mask)[0]
            if len(idx) > self.metric_sample:
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(idx, self.metric_sample, replace=False)
            Xs, ys = behavioral[idx], leaf_labels[idx]
            if len(set(ys.tolist())) > 1:
                sil = float(silhouette_score(Xs, ys))
                db = float(davies_bouldin_score(Xs, ys))
                # per-leaf breakdown so an overlapping leaf isn't hidden by the
                # global average (same subsample as the global scores above).
                per_sample = silhouette_samples(Xs, ys)
                for lid in set(ys.tolist()):
                    leaf_sil[int(lid)] = float(per_sample[ys == lid].mean())

        # per-leaf centroids in BOTH spaces (kept for downstream skill compile /
        # representative selection). language centroids use the raw embedding.
        beh_centroids = {
            int(lid): behavioral[leaf_labels == lid].mean(axis=0) for lid in leaf_ids
        }
        lang_centroids = {
            int(lid): language[leaf_labels == lid].mean(axis=0) for lid in leaf_ids
        }
        centroids = (np.stack([lang_centroids[l] for l in leaf_ids])
                     if leaf_ids else np.zeros((0, language.shape[1])))

        label_dict = {u: int(lbl) for u, lbl in zip(user_ids, leaf_labels)}
        for u in user_features:
            label_dict.setdefault(u, leaf_ids[0] if leaf_ids else 0)

        logger.info(
            f"Behaviour-only roles: {len(leaf_ids)} leaves, "
            f"{n_pre_noise} pre-impute orphans ({n_pre_noise/len(user_ids):.1%}), "
            f"silhouette={sil:.3f}, db={db:.3f}"
        )

        cr = ClusterResult(
            labels=label_dict,
            n_clusters=len(leaf_ids),
            centroids=centroids,
            silhouette_score=float(sil),
            davies_bouldin_score=float(db),
            behavioral_weight=1.0,
            language_weight=1.0,
            user_features=user_features,
        )
        cr.leaf_behavior_centroids = beh_centroids
        cr.leaf_language_centroids = lang_centroids
        cr.leaf_silhouette = leaf_sil
        cr.pre_impute_orphans = n_pre_noise
        cr.n_orphans_kept = n_orphans_kept
        return cr

    def _impute_orphans(
        self, leaf_labels: np.ndarray, behavioral: np.ndarray,
        leaf_ids: list[int],
    ) -> np.ndarray:
        """Assign every -1 user to its nearest leaf on the behavioural space."""
        noise_idx = np.where(leaf_labels == -1)[0]
        if len(noise_idx) == 0 or not leaf_ids:
            return leaf_labels
        leaf_centroids = np.stack([behavioral[leaf_labels == lid].mean(axis=0) for lid in leaf_ids])
        from scipy.spatial.distance import cdist
        # cdist keeps memory O(n_orphans × n_leaves) instead of a 3-D broadcast
        dists = cdist(behavioral[noise_idx], leaf_centroids)
        nearest = dists.argmin(axis=1)
        for i, oidx in enumerate(noise_idx):
            leaf_labels[oidx] = leaf_ids[int(nearest[i])]
        logger.info(f"Imputed {len(noise_idx)} orphans → nearest leaf")
        return leaf_labels

    def _get_scaler(self):
        if self.scaler == "robust":
            return RobustScaler(quantile_range=(5.0, 95.0))
        return StandardScaler()

    def _compute_role_mcs(self, n: int) -> int:
        """Auto-tune role min_cluster_size to hit the target leaf range."""
        if self.role_min_cluster_size is not None:
            return self.role_min_cluster_size
        # Sub-linear growth so huge corpora don't force mega-clusters.
        mcs = max(200, int(n ** 0.65))
        min_mcs = n // (self.target_max_leaves * 2)
        max_mcs = n // self.target_min_leaves
        return int(np.clip(mcs, min_mcs, max_mcs))

    # ------------------------------------------------------------------
    def _stage1_roles(self, behavioral: np.ndarray, n: int) -> np.ndarray:
        """Stage-1 role partition of the behavioural space.

        KMeans (default): on full WikiConv the behavioural space is a near-
        continuous manifold with no density gaps — HDBSCAN rejects ~65% of users
        as noise at every parameter setting, and GMM components collapse (both
        verified at N=593,871). KMeans imposes a balanced partition with zero
        orphans, so every user receives a distillation unit. HDBSCAN is retained
        as a selectable method for corpora that DO have density-separated roles.
        """
        if self.role_method == "hdbscan":
            role_mcs = self._compute_role_mcs(n)
            role_min_samples = self.role_min_samples or max(1, role_mcs // 5)
            logger.info(
                f"Stage 1 roles (HDBSCAN): n={n}, min_cluster_size={role_mcs}, min_samples={role_min_samples}"
            )
            return self._hdbscan(behavioral, role_mcs, role_min_samples)
        k = max(2, min(int(self.role_k), n))
        km = MiniBatchKMeans(
            n_clusters=k, random_state=self.random_state, n_init=5, batch_size=4096,
        )
        roles = km.fit_predict(behavioral)
        logger.info(f"Stage 1 roles (KMeans): n={n}, k={k}")
        return roles

    def _two_stage(self, behavioral: np.ndarray) -> np.ndarray:
        """Behaviour-only role partition; the role id IS the leaf id.

        Formerly a two-stage role→style split. The language/style stage was
        removed because averaged per-role language centroids do not separate
        (verified at N=593,871 — see module docstring): sub-clustering on
        language only fragments roles without recovering real structure, and
        language is captured at compile time (outline §4.3) instead. With the
        style stage gone, a leaf = one role, so this is a single-stage partition.

        ``_stage1_roles`` returns contiguous non-negative ids (with ``-1`` for
        HDBSCAN noise), which are valid leaf ids as-is.
        """
        return self._stage1_roles(behavioral, behavioral.shape[0])

    def _hdbscan(self, X: np.ndarray, min_cluster_size: int,
                 min_samples: int | None = None) -> np.ndarray:
        import hdbscan
        mcs = max(2, int(min_cluster_size))
        ms = max(1, int(min_samples)) if min_samples is not None else mcs
        if X.shape[0] <= mcs:
            return np.zeros(X.shape[0], dtype=int)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=self.cluster_selection_method,
        )
        return clusterer.fit_predict(X)

    def _single_stage(self, behavioral: np.ndarray, language: np.ndarray) -> np.ndarray:
        """Back-compat single-stage path (kmeans / hdbscan) on concatenated space."""
        bw = self.behavioral_weight if self.behavioral_weight is not None else 0.5
        lw = self.language_weight if self.language_weight is not None else 0.5
        combined = np.hstack([bw * behavioral, lw * self._get_scaler().fit_transform(language)])
        if self.method == "hdbscan":
            mcs = self.role_min_cluster_size or max(5, combined.shape[0] // 20)
            min_samples = self.role_min_samples or max(1, mcs // 5)
            return self._hdbscan(combined, mcs, min_samples)
        # kmeans with fixed or auto k
        k = self.n_clusters if self.n_clusters and self.n_clusters > 0 else self.min_k
        k = min(k, combined.shape[0] - 1)
        return KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit_predict(combined)

    def _kmeans(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X)
        return labels, km.cluster_centers_

    def _group_messages_by_user(self, threads: list[Thread]) -> dict[str, list[Message]]:
        """Group messages by user, capping per-user to bound embedding cost.

        Encoding every message of a multi-million-message corpus is the dominant
        cost; a per-user sample of ``max_msgs_per_user`` messages yields a stable
        mean embedding at a fraction of the cost.
        """
        from collections import defaultdict
        user_msgs: dict[str, list[Message]] = defaultdict(list)
        cap = self.max_msgs_per_user
        for thread in threads:
            for msg in thread.messages:
                lst = user_msgs[msg.user_id]
                if cap is None or len(lst) < cap:
                    lst.append(msg)
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
