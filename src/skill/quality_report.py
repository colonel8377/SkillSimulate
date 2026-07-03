"""Joint action-clustering + language-distillation quality report.

A leaf can look good on one axis while being unusable on the other — tight
behavioral clustering with thin/repetitive distilled material, or rich
material for a leaf whose behavior barely separates from its neighbours.
This module scores both axes per leaf and surfaces a concerns punch list,
so neither axis is judged (or missed) in isolation.
"""

from __future__ import annotations

import numpy as np

from src.clustering.clusterer import ClusterResult
from src.skill.cluster_profile import LeafProfile
from src.skill.corpus_export import EVIDENCE_BUDGET

# Flag thresholds for the "concerns" list — soft signals for manual review,
# not hard failures.
MIN_SILHOUETTE = 0.05
MIN_CANDIDATES = 10
MIN_DEDUP_RATIO = 0.15
MIN_EVIDENCE_FILL_FRAC = 0.3
# ~random user-pair cosine baseline observed at N=593,871 (0.158, see
# clusterer.py module docstring) — a leaf whose language centroid gets this
# close to another leaf's is not distinguishable from it.
MAX_NEAREST_COSINE = 0.15
# a per-leaf k=2 split with silhouette above this is a real bimodal signal,
# not clustering noise (well below sklearn's own >0.5 "strong structure" bar,
# since we're asking a much narrower question than the earlier global attempt).
MIN_INTRA_LEAF_SPLIT_SILHOUETTE = 0.1


def _evidence_type(line: str) -> str | None:
    if "deleted another's comment" in line:
        return "delete"
    if "personal-attack flagged" in line:
        return "attack"
    if "high-conflict/flagged tox=" in line:
        return "tox"
    return None


def _evidence_fill(lines: list[str]) -> dict[str, dict[str, int]]:
    counts = {t: 0 for t in EVIDENCE_BUDGET}
    for line in lines:
        t = _evidence_type(line)
        if t:
            counts[t] += 1
    return {t: {"filled": counts[t], "budget": EVIDENCE_BUDGET[t]} for t in EVIDENCE_BUDGET}


def _mean_centered_centroids(lang_centroids: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Subtract the shared global-mean direction before comparing leaf centroids.

    Each leaf centroid is a raw mean over hundreds-to-tens-of-thousands of
    per-user embeddings. Averaging that many embeddings collapses individual
    variance toward the corpus-wide background direction (embedding
    anisotropy), so raw centroid-vs-centroid cosine saturates near 1.0
    regardless of real language distinctiveness. Mean-centering against the
    global centroid mean is the standard correction — cosine on the residuals
    reflects between-leaf distinctiveness instead of the shared component
    every leaf inherits.
    """
    if not lang_centroids:
        return {}
    global_mean = np.mean(list(lang_centroids.values()), axis=0)
    return {lid: c - global_mean for lid, c in lang_centroids.items()}


def _nearest_other_cosine(leaf_id: int, lang_centroids: dict[int, np.ndarray]) -> float | None:
    if len(lang_centroids) < 2 or leaf_id not in lang_centroids:
        return None
    c = lang_centroids[leaf_id]
    c = c / (np.linalg.norm(c) + 1e-9)
    best = None
    for other_id, oc in lang_centroids.items():
        if other_id == leaf_id:
            continue
        oc = oc / (np.linalg.norm(oc) + 1e-9)
        cos = float(np.dot(c, oc))
        if best is None or cos > best:
            best = cos
    return best


def _random_baseline_cosine(
    user_embeddings: dict[str, np.ndarray],
    leaf_sizes: list[int],
    n_trials: int = 3,
    seed: int = 42,
) -> float | None:
    """Null baseline: mean-centered nearest-leaf cosine under a label-free random
    grouping of the same users into groups matching the real leaf sizes.

    Averaging embeddings over a group collapses individual variance toward the
    shared background direction *regardless of whether the grouping means
    anything* — the group-size effect alone produces a nonzero cosine. This
    measures that effect in isolation, so the real per-leaf cosine can be
    judged against "what random grouping of this size produces" instead of
    against a threshold borrowed from a differently-scaled experiment.
    """
    users = list(user_embeddings)
    if not leaf_sizes or len(users) < sum(leaf_sizes):
        return None
    rng = np.random.default_rng(seed)
    trial_means = []
    for _ in range(n_trials):
        perm = rng.permutation(users)
        centroids = {}
        idx = 0
        for i, size in enumerate(leaf_sizes):
            group = perm[idx: idx + size]
            idx += size
            if len(group):
                centroids[i] = np.mean([user_embeddings[u] for u in group], axis=0)
        centered = _mean_centered_centroids(centroids)
        cosines = [
            cos for lid in centroids
            if (cos := _nearest_other_cosine(lid, centered)) is not None
        ]
        if cosines:
            trial_means.append(float(np.mean(cosines)))
    return float(np.mean(trial_means)) if trial_means else None


def _random_pair_cosine(
    user_embeddings: dict[str, np.ndarray],
    n_pairs: int = 3000,
    seed: int = 42,
) -> float | None:
    """Null baseline: mean-centered cosine between random INDIVIDUAL user pairs
    (not group centroids).

    ``_random_baseline_cosine`` measures the group-size averaging effect in
    isolation — it answers "is there any non-noise signal at all," and in
    high-dimensional embedding space a group-mean's residual is close to
    orthogonal-random almost by construction (concentration of measure), so it
    reads near-zero regardless of whether real leaves are well-separated. That
    makes "real > random-group" a very low bar: two real, non-noise but
    topically-overlapping leaves (e.g. both "article-focused") will clear it
    easily without actually being distinguishable from each other.

    This is the correct bar, matching the original validated methodology (22
    leaves, max centroid cosine 0.063 vs random individual-pair cosine 0.158):
    if leaf centroids — themselves an average over hundreds to tens of
    thousands of users — are still LESS similar to each other than two raw
    INDIVIDUAL users are to each other, that's real evidence of separation,
    because averaging normally pulls estimates together, not apart.
    """
    users = list(user_embeddings)
    if len(users) < 2:
        return None
    global_mean = np.mean(list(user_embeddings.values()), axis=0)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(users), n_pairs)
    idx_b = rng.integers(0, len(users), n_pairs)
    cosines = []
    for ia, ib in zip(idx_a, idx_b):
        if ia == ib:
            continue
        a = user_embeddings[users[ia]] - global_mean
        b = user_embeddings[users[ib]] - global_mean
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        cosines.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(cosines)) if cosines else None


def _intra_leaf_split_signal(
    members: list[str],
    user_embeddings: dict[str, np.ndarray],
    max_sample: int = 2000,
    seed: int = 42,
) -> dict | None:
    """Does this one leaf, on its own, contain two language-distinct voices?

    A local (single-leaf) k=2 split is a much narrower question than the
    earlier corpus-wide density-based sub-clustering attempt, which asked for
    a stable *universal* typology across all leaves and failed to find one
    (see cadp-clustering-decisions). This only asks whether this leaf's own
    members separate into two groups — a real signal even if no universal
    typology exists. The minority side's members are returned so a human can
    inspect who the "different voice" actually is.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as _sk_silhouette

    users = [u for u in members if u in user_embeddings]
    if len(users) < 20:
        return None
    rng = np.random.default_rng(seed)
    if len(users) > max_sample:
        users = list(rng.choice(users, max_sample, replace=False))
    X = np.stack([user_embeddings[u] for u in users])
    X = X - X.mean(axis=0)  # remove this leaf's own shared background direction
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(X)
    if len(set(labels.tolist())) < 2:
        return None
    sil = float(_sk_silhouette(X, labels))
    sizes = [int((labels == 0).sum()), int((labels == 1).sum())]
    minority_side = 0 if sizes[0] < sizes[1] else 1
    minority_members = [users[i] for i in range(len(labels)) if labels[i] == minority_side]
    return {"silhouette": sil, "split_sizes": sizes, "minority_side_members": minority_members[:10]}


def build_quality_report(
    cr: ClusterResult,
    profiles: dict[int, LeafProfile],
    leaf_evidence: dict[int, list[str]] | None = None,
    user_embeddings: dict[str, np.ndarray] | None = None,
) -> dict:
    """Combine action-clustering and language-distillation quality per leaf.

    ``profiles`` (from ``ArchetypeProfiler.build``) supplies the language side
    — representative members, typical utterances, tags. ``leaf_evidence``
    (from ``corpus_export._rejected_evidence`` / ``streaming.collect_rejection_evidence``)
    supplies rejection-evidence fill; omit it for callers (e.g. ``cluster_eda``)
    that don't build distillation packs — evidence_fill then reads as all-zero.
    ``user_embeddings`` (per-user language vectors, same ones used to build
    ``cr.leaf_language_centroids``) is optional — when given, a random-grouping
    null baseline is computed for `random_baseline_cosine` / `mean_leaf_nearest_cosine`
    below; omit it to skip that (both fields come back `None`).
    """
    leaf_evidence = leaf_evidence or {}
    leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]
    n_users = len(cr.labels)
    centered_centroids = _mean_centered_centroids(cr.leaf_language_centroids)

    leaves: dict[int, dict] = {}
    concerns: list[dict] = []
    for lid in leaf_ids:
        prof = profiles.get(lid)
        sil = cr.leaf_silhouette.get(lid)
        evidence = _evidence_fill(leaf_evidence.get(lid, []))
        nearest_cos = _nearest_other_cosine(lid, centered_centroids)

        n_typical = len(prof.typical_utterances) if prof else 0
        n_candidates = prof.n_candidates if prof else 0
        dedup_ratio = round(n_typical / n_candidates, 3) if n_candidates else 0.0
        n_tags = len(prof.tags) if prof else 0
        intra_split = (
            _intra_leaf_split_signal(cr.get_cluster_members(lid), user_embeddings)
            if user_embeddings else None
        )

        leaves[lid] = {
            "size": len(cr.get_cluster_members(lid)),
            "leaf_silhouette": sil,
            "n_representative_members": len(prof.members) if prof else 0,
            "n_typical_utterances": n_typical,
            "n_candidates": n_candidates,
            "dedup_ratio": dedup_ratio,
            "evidence_fill": evidence,
            "n_tags": n_tags,
            "tags": prof.tags if prof else [],
            "nearest_other_leaf_cosine": nearest_cos,
            "intra_leaf_split": intra_split,
        }

        issues: list[str] = []
        if sil is not None and sil < MIN_SILHOUETTE:
            issues.append(f"low action silhouette ({sil:.3f} < {MIN_SILHOUETTE})")
        if n_candidates < MIN_CANDIDATES:
            issues.append(f"too few candidate utterances ({n_candidates} < {MIN_CANDIDATES})")
        elif dedup_ratio < MIN_DEDUP_RATIO:
            issues.append(f"low dedup ratio ({dedup_ratio} < {MIN_DEDUP_RATIO}) — material may be repetitive")
        total_filled = sum(v["filled"] for v in evidence.values())
        total_budget = sum(v["budget"] for v in evidence.values())
        if total_budget:
            fill_frac = total_filled / total_budget
            if fill_frac < MIN_EVIDENCE_FILL_FRAC:
                issues.append(
                    f"low rejection-evidence fill ({total_filled}/{total_budget}) — weak anti-pattern grounding"
                )
        if n_tags == 0:
            issues.append("no behavioral tags — leaf may be generic/indistinguishable")
        if nearest_cos is not None and nearest_cos > MAX_NEAREST_COSINE:
            issues.append(
                f"language centroid close to another leaf (cosine {nearest_cos:.3f} > {MAX_NEAREST_COSINE})"
            )
        if intra_split is not None and intra_split["silhouette"] > MIN_INTRA_LEAF_SPLIT_SILHOUETTE:
            issues.append(
                f"leaf contains two distinguishable voices (intra-leaf split silhouette "
                f"{intra_split['silhouette']:.3f}, sizes {intra_split['split_sizes']}) — "
                "minority side listed in intra_leaf_split.minority_side_members"
            )

        if issues:
            concerns.append({"leaf_id": lid, "issues": issues})

    mean_leaf_cos = (
        float(np.mean([v["nearest_other_leaf_cosine"] for v in leaves.values()
                       if v["nearest_other_leaf_cosine"] is not None]))
        if leaves else None
    )
    leaf_sizes = [len(cr.get_cluster_members(lid)) for lid in leaf_ids]
    random_baseline_cos = (
        _random_baseline_cosine(user_embeddings, leaf_sizes)
        if user_embeddings else None
    )
    random_pair_cos = (
        _random_pair_cosine(user_embeddings)
        if user_embeddings else None
    )
    if mean_leaf_cos is not None and random_pair_cos is not None:
        # the real bar (matches original 22-leaf validation methodology): leaf
        # centroids, despite being averages over many users, should still be
        # LESS similar to each other than two raw individual users are — if
        # not, behavior-based leaves aren't carrying distinguishable language.
        if mean_leaf_cos > random_pair_cos * 0.9:
            concerns.append({
                "leaf_id": None,
                "issues": [
                    f"leaves' mean language-centroid cosine ({mean_leaf_cos:.3f}) is not "
                    f"meaningfully below the random individual-user-pair cosine ({random_pair_cos:.3f}) "
                    "— behavior-based leaves may not carry distinguishable language"
                ],
            })

    return {
        "n_leaves": len(leaf_ids),
        "n_users": n_users,
        "orphan_rate": round(cr.n_orphans_kept / n_users, 4) if n_users else 0.0,
        "silhouette": cr.silhouette_score,
        "davies_bouldin": cr.davies_bouldin_score,
        "mean_leaf_nearest_cosine": mean_leaf_cos,
        "random_baseline_cosine": random_baseline_cos,
        "random_pair_cosine": random_pair_cos,
        "leaves": leaves,
        "concerns": concerns,
    }
