"""Macro-level evaluation metrics.

ΔQ Modularity, E-I Polarization Index, NED, Coverage.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.stats import entropy


def delta_q_modularity(
    sim_graph: nx.Graph,
    real_graph: nx.Graph,
    sim_communities: dict[str, int],
    real_communities: dict[str, int],
) -> float:
    """Compute ΔQ (modularity difference between simulation and real).

    Args:
        sim_graph: Simulation interaction graph.
        real_graph: Real interaction graph.
        sim_communities: Node → cluster_id mapping for simulation.
        real_communities: Node → cluster_id mapping for real data.

    Returns:
        ΔQ = |Q_sim - Q_real| (lower is better).
        Returns 0.0 when either graph has no edges (modularity undefined).
    """
    # Modularity is undefined for edgeless graphs
    if sim_graph.number_of_edges() == 0 or real_graph.number_of_edges() == 0:
        return 0.0

    sim_partition = _communities_to_partitions(sim_graph, sim_communities)
    real_partition = _communities_to_partitions(real_graph, real_communities)

    if not sim_partition or not real_partition:
        return 0.0

    q_sim = nx.community.modularity(sim_graph, sim_partition)
    q_real = nx.community.modularity(real_graph, real_partition)
    return abs(q_sim - q_real)


def _communities_to_partitions(graph: nx.Graph, communities: dict[str, int]) -> list[set]:
    """Convert community mapping to partition list.

    Assigns nodes present in *communities* to their mapped cluster.
    Nodes in the graph but absent from *communities* are grouped into a
    single ``-1`` bucket so the partition always covers every node
    (required by ``nx.community.modularity``).
    """
    partitions: dict[int, set] = {}
    covered: set[str] = set()
    for node, cid in communities.items():
        if node in graph:
            partitions.setdefault(cid, set()).add(node)
            covered.add(node)
    # Group uncovered nodes into a single "unassigned" community
    uncovered = set(graph.nodes()) - covered
    if uncovered:
        partitions.setdefault(-1, set()).update(uncovered)
    result = list(partitions.values())
    # Edge case: empty graph — return empty partition
    return result if result else []


def ei_polarization_index(
    graph: nx.Graph,
    communities: dict[str, int],
) -> float:
    """Compute E-I Polarization Index.

    Measures ratio of external (cross-cluster) to internal (within-cluster) interactions.

    Returns:
        E-I index in [-1, 1]. +1 = all external (polarized), -1 = all internal.
    """
    external = 0
    internal = 0
    for u, v in graph.edges():
        cu = communities.get(u)
        cv = communities.get(v)
        if cu is not None and cv is not None and cu == cv:
            internal += 1
        else:
            external += 1

    total = external + internal
    if total == 0:
        return 0.0
    return (external - internal) / total


def normalized_entropy_distance(
    sim_action_dist: dict[str, float],
    real_action_dist: dict[str, float],
) -> float:
    """Compute Normalized Entropy Distance (NED).

    Measures divergence between action type distributions.

    Returns:
        NED in [0, 1]. 0 = identical, 1 = maximally different.
    """
    all_actions = sorted(set(sim_action_dist) | set(real_action_dist))
    p = np.array([sim_action_dist.get(a, 0) for a in all_actions])
    q = np.array([real_action_dist.get(a, 0) for a in all_actions])

    # Normalize
    p = p / (p.sum() or 1)
    q = q / (q.sum() or 1)

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    h_p = entropy(p, m) if p.sum() > 0 else 0
    h_q = entropy(q, m) if q.sum() > 0 else 0
    jsd = 0.5 * (h_p + h_q)

    return float(np.sqrt(jsd))


def behavior_coverage(
    sim_actions: set[str],
    real_actions: set[str],
) -> float:
    """Compute behavior space coverage.

    Returns:
        Fraction of real action types covered by simulation.
    """
    if not real_actions:
        return 1.0
    return len(sim_actions & real_actions) / len(real_actions)


class MacroMetrics:
    """Container for all macro-level metrics."""

    @staticmethod
    def compute(
        sim_graph: nx.Graph,
        real_graph: nx.Graph,
        sim_communities: dict[str, int],
        real_communities: dict[str, int],
        sim_action_dist: dict[str, float],
        real_action_dist: dict[str, float],
    ) -> dict[str, float]:
        return {
            "delta_q_modularity": delta_q_modularity(sim_graph, real_graph, sim_communities, real_communities),
            "ei_polarization_sim": ei_polarization_index(sim_graph, sim_communities),
            "ei_polarization_real": ei_polarization_index(real_graph, real_communities),
            "ned": normalized_entropy_distance(sim_action_dist, real_action_dist),
            "coverage": behavior_coverage(
                set(sim_action_dist.keys()),
                set(real_action_dist.keys()),
            ),
        }
