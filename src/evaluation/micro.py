"""Micro-level evaluation metrics.

Action Matrix Similarity (Frobenius), RSA, Uniformity (entropy), Complexity (variance).
"""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from scipy.stats import entropy


def action_matrix_similarity(
    sim_matrix: np.ndarray,
    real_matrix: np.ndarray,
) -> float:
    """Frobenius norm similarity between agent×action matrices.

    Aligns row counts by sampling/padding and pads columns to the union of
    actions. Normalization uses the sum of both norms, guaranteeing a score
    in [0, 1] and avoiding the negative values produced by the old
    truncation-to-min-size logic.

    Args:
        sim_matrix: Simulated agent × action count matrix.
        real_matrix: Real agent × action count matrix.

    Returns:
        Similarity in [0, 1]. 1 = identical, 0 = maximally different.
    """
    if sim_matrix.size == 0 or real_matrix.size == 0:
        return 0.0

    n_sim, m_sim = sim_matrix.shape
    n_real, m_real = real_matrix.shape

    # Pad columns to the union action set. The matrices are built from the
    # same sorted action list, so right-padding zeros preserves alignment.
    max_cols = max(m_sim, m_real)
    if m_sim < max_cols:
        sim_matrix = np.pad(sim_matrix, ((0, 0), (0, max_cols - m_sim)))
    if m_real < max_cols:
        real_matrix = np.pad(real_matrix, ((0, 0), (0, max_cols - m_real)))

    # Align row counts deterministically. Prefer sampling real rows down to
    # the simulated population size; if the real matrix is smaller, pad it.
    target_rows = max(n_sim, n_real)
    if n_real > n_sim:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_real, size=n_sim, replace=False)
        real_matrix = real_matrix[idx]
    elif n_real < n_sim:
        real_matrix = np.pad(real_matrix, ((0, n_sim - n_real), (0, 0)))

    if sim_matrix.shape != real_matrix.shape:
        # Defensive fallback: truncate to common shape if padding failed.
        min_rows = min(sim_matrix.shape[0], real_matrix.shape[0])
        min_cols = min(sim_matrix.shape[1], real_matrix.shape[1])
        sim_matrix = sim_matrix[:min_rows, :min_cols]
        real_matrix = real_matrix[:min_rows, :min_cols]

    frob = np.linalg.norm(sim_matrix - real_matrix)
    max_possible = np.linalg.norm(sim_matrix) + np.linalg.norm(real_matrix) + 1e-10
    return float(1.0 - frob / max_possible)


def representational_similarity(
    sim_profiles: np.ndarray,
    real_profiles: np.ndarray,
) -> float:
    """RSA — correlation of behavioral profile similarity matrices.

    Args:
        sim_profiles: Agent behavioral profiles (agent × feature matrix).
        real_profiles: Real user behavioral profiles.

    Returns:
        RSA score in [-1, 1]. Higher = better representational match.
    """
    if len(sim_profiles) < 2 or len(real_profiles) < 2:
        return 0.0

    # Compute similarity matrices (cosine)
    def _sim_matrix(profiles: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        normalized = profiles / (norms + 1e-10)
        return normalized @ normalized.T

    sim_sim = _sim_matrix(sim_profiles)
    real_sim = _sim_matrix(real_profiles)

    # Min size match
    min_size = min(sim_sim.shape[0], real_sim.shape[0])
    sim_flat = sim_sim[:min_size, :min_size].flatten()
    real_flat = real_sim[:min_size, :min_size].flatten()

    if np.std(sim_flat) == 0 or np.std(real_flat) == 0:
        return 0.0

    corr = np.corrcoef(sim_flat, real_flat)[0, 1]
    return float(corr)


def behavior_uniformity(action_counts: Counter) -> float:
    """Entropy of action distribution — detects homogenization.

    Returns:
        Normalized entropy in [0, 1]. 1 = uniform (diverse), 0 = single action.
    """
    if not action_counts:
        return 0.0

    counts = np.array(list(action_counts.values()), dtype=float)
    probs = counts / counts.sum()
    ent = entropy(probs)
    max_ent = np.log(len(counts)) if len(counts) > 1 else 1
    return float(ent / max_ent) if max_ent > 0 else 0.0


def behavior_complexity(
    agent_action_counts: dict[str, Counter],
) -> float:
    """Cross-agent behavioral variance.

    Higher variance = more diverse agent behaviors.

    Returns:
        Mean variance of action distributions across agents.
    """
    if not agent_action_counts:
        return 0.0

    # Get all action types
    all_actions = sorted(set(
        a for counts in agent_action_counts.values() for a in counts
    ))
    if not all_actions:
        return 0.0

    # Build agent × action matrix
    n_agents = len(agent_action_counts)
    matrix = np.zeros((n_agents, len(all_actions)))
    for i, (agent, counts) in enumerate(agent_action_counts.items()):
        for j, action in enumerate(all_actions):
            matrix[i, j] = counts.get(action, 0)

    # Normalize per agent
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / (row_sums + 1e-10)

    # Variance across agents
    return float(np.mean(np.var(matrix_norm, axis=0)))


def caricature_index(
    agent_action_counts: dict[str, Counter],
    agent_clusters: dict[str, int],
) -> float:
    """Caricature Index — between-cluster behavioral Cohen's d.

    Measures whether agents in different clusters produce behaviorally
    distinct action distributions (high d = caricatured / stereotyped;
    low d = naturally overlapping). Responds to Chameleon's Limit
    §3.3 "fidelity breeds caricature" (Cohen's d > 6 in their report)
    and Promise-with-a-Catch (more LLM content → more bias).

    CADP's claim: filter-retry enforcement constrains output distribution
    without increasing caricature (§3.2). This metric tests that
    prediction directly.

    Args:
        agent_action_counts: {agent_id: {action: count}}.
        agent_clusters: {agent_id: cluster_id}.

    Returns:
        Mean pairwise Cohen's d between clusters. 0 = no separation,
        higher = more caricatured. NaN if < 2 clusters have data.
    """
    # Group agents by cluster
    cluster_profiles: dict[int, list[np.ndarray]] = defaultdict(list)
    all_actions = sorted(set(
        a for counts in agent_action_counts.values() for a in counts
    ))
    if not all_actions:
        return 0.0

    for agent_id, counts in agent_action_counts.items():
        cluster_id = agent_clusters.get(agent_id, -1)
        if cluster_id < 0:
            continue
        profile = np.array([counts.get(a, 0) for a in all_actions], dtype=float)
        total = profile.sum()
        if total > 0:
            profile /= total
        cluster_profiles[cluster_id].append(profile)

    # Need at least 2 clusters with ≥ 2 agents each for meaningful d
    valid_clusters = {k: v for k, v in cluster_profiles.items() if len(v) >= 2}
    if len(valid_clusters) < 2:
        return 0.0

    # Compute pairwise Cohen's d between cluster centroids
    cluster_ids = sorted(valid_clusters.keys())
    d_values = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            profiles_i = np.array(valid_clusters[cluster_ids[i]])
            profiles_j = np.array(valid_clusters[cluster_ids[j]])
            mean_i = profiles_i.mean(axis=0)
            mean_j = profiles_j.mean(axis=0)
            var_i = profiles_i.var(axis=0, ddof=1).mean()
            var_j = profiles_j.var(axis=0, ddof=1).mean()
            pooled_std = np.sqrt((var_i + var_j) / 2)
            if pooled_std > 1e-10:
                d = float(np.linalg.norm(mean_i - mean_j) / pooled_std)
                d_values.append(d)

    return float(np.mean(d_values)) if d_values else 0.0


class MicroMetrics:
    """Container for all micro-level metrics."""

    @staticmethod
    def compute(
        sim_matrix: np.ndarray | None = None,
        real_matrix: np.ndarray | None = None,
        sim_profiles: np.ndarray | None = None,
        real_profiles: np.ndarray | None = None,
        sim_action_counts: Counter | None = None,
        real_action_counts: Counter | None = None,
        sim_agent_counts: dict[str, Counter] | None = None,
        real_agent_counts: dict[str, Counter] | None = None,
    ) -> dict[str, float]:
        result = {}

        if sim_matrix is not None and real_matrix is not None:
            result["action_matrix_similarity"] = action_matrix_similarity(sim_matrix, real_matrix)

        if sim_profiles is not None and real_profiles is not None:
            result["rsa"] = representational_similarity(sim_profiles, real_profiles)

        if sim_action_counts is not None and real_action_counts is not None:
            result["uniformity_sim"] = behavior_uniformity(sim_action_counts)
            result["uniformity_real"] = behavior_uniformity(real_action_counts)
            result["uniformity_gap"] = abs(
                result["uniformity_sim"] - result["uniformity_real"]
            )

        if sim_agent_counts is not None and real_agent_counts is not None:
            result["complexity_sim"] = behavior_complexity(sim_agent_counts)
            result["complexity_real"] = behavior_complexity(real_agent_counts)

        return result
