"""Meso-level evaluation metrics.

Cascade Length Fit (KS-test), DTW, Structural Fidelity.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.stats import ks_2samp

from dtw import dtw


def cascade_length_fit(
    sim_chain_lengths: list[int],
    real_chain_lengths: list[int],
) -> dict[str, float]:
    """KS-test on reply chain length distributions.

    Returns:
        Dict with KS statistic and p-value (lower KS = better fit).
    """
    if not sim_chain_lengths or not real_chain_lengths:
        return {"ks_statistic": 1.0, "p_value": 0.0}

    result = ks_2samp(sim_chain_lengths, real_chain_lengths)
    return {
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def dynamic_time_warping(
    sim_sequence: list[float],
    real_sequence: list[float],
) -> float:
    """DTW distance between temporal sequences.

    Args:
        sim_sequence: Simulation metric over time (e.g., polarization per round).
        real_sequence: Real metric over time.

    Returns:
        DTW distance (lower = better fit).
    """
    if not sim_sequence or not real_sequence:
        return float("inf")

    alignment = dtw(sim_sequence, real_sequence)
    return float(alignment.distance)


def structural_fidelity(
    sim_graph: nx.Graph,
    real_graph: nx.Graph,
) -> float:
    """Correlation of network structural features.

    Compares degree distribution, clustering coefficient, etc.

    Returns:
        Correlation coefficient in [-1, 1]. Higher = better fidelity.
    """
    def _graph_features(g: nx.Graph) -> np.ndarray:
        degrees = sorted([d for _, d in g.degree()])
        if not degrees:
            return np.zeros(5)

        clustering = list(nx.clustering(g).values())
        return np.array([
            np.mean(degrees),
            np.std(degrees),
            np.mean(clustering) if clustering else 0,
            g.number_of_edges(),
            nx.density(g),
        ])

    sim_feat = _graph_features(sim_graph)
    real_feat = _graph_features(real_graph)

    if np.all(sim_feat == 0) or np.all(real_feat == 0):
        return 0.0

    # Normalize features
    norm_factor = np.linalg.norm(real_feat)
    if norm_factor > 0:
        real_feat = real_feat / norm_factor
        sim_feat = sim_feat / (np.linalg.norm(sim_feat) or 1)

    return float(1.0 - np.linalg.norm(sim_feat - real_feat))


class MesoMetrics:
    """Container for all meso-level metrics."""

    @staticmethod
    def compute(
        sim_graph: nx.Graph,
        real_graph: nx.Graph,
        sim_chain_lengths: list[int] | None = None,
        real_chain_lengths: list[int] | None = None,
        sim_temporal: list[float] | None = None,
        real_temporal: list[float] | None = None,
    ) -> dict[str, float]:
        result = {
            "structural_fidelity": structural_fidelity(sim_graph, real_graph),
        }

        if sim_chain_lengths is not None and real_chain_lengths is not None:
            result.update(cascade_length_fit(sim_chain_lengths, real_chain_lengths))

        if sim_temporal is not None and real_temporal is not None:
            result["dtw_distance"] = dynamic_time_warping(sim_temporal, real_temporal)

        return result
