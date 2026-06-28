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

    Compares a richer set of graph descriptors (degree moments, clustering,
    size, density) on a log scale. This avoids the saturation problem of the
    previous unit-vector normalization, which produced ~0.934 for all
    conditions regardless of absolute graph scale.

    Returns:
        Similarity score in [0, 1]. Higher = better fidelity.
    """
    def _graph_features(g: nx.Graph) -> np.ndarray:
        degrees = np.array(sorted((d for _, d in g.degree()), reverse=True))
        if degrees.size == 0:
            return np.zeros(11)

        clustering = list(nx.clustering(g).values())
        clust = np.array(clustering) if clustering else np.zeros(1)

        # Rich feature vector; log1p stabilizes across huge scale differences
        # (real graph ~38k nodes vs. sim graph ~30 nodes).
        feats = np.array([
            g.number_of_nodes(),
            g.number_of_edges(),
            nx.density(g),
            float(np.mean(degrees)),
            float(np.std(degrees)),
            float(np.median(degrees)),
            float(np.max(degrees)),
            float(np.percentile(degrees, 75)),
            float(np.percentile(degrees, 25)),
            float(np.mean(clust)),
            float(np.std(clust)),
        ])
        return np.log1p(feats)

    sim_feat = _graph_features(sim_graph)
    real_feat = _graph_features(real_graph)

    if np.all(sim_feat == 0) or np.all(real_feat == 0):
        return 0.0

    # Pearson correlation on log-scaled features; mapped to [0, 1].
    if np.std(sim_feat) == 0 or np.std(real_feat) == 0:
        return 0.0

    corr = float(np.corrcoef(sim_feat, real_feat)[0, 1])
    if np.isnan(corr):
        return 0.0
    return float((corr + 1.0) / 2.0)


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
