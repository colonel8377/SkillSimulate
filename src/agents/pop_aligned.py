"""Cluster-Stat Aligned Persona — internal population-description baseline.

Inspired by population-alignment work, but intentionally does not claim to
reproduce arXiv:2509.10127's full generation/filtering/importance-sampling
pipeline. It samples attributes from the project's behavioral clusters.
Key distinction from CADP: matches *attribute distributions* (correct
*types* of people) without enforcing behavioral rules (correct *actions*).
"""

from __future__ import annotations

import random
from typing import Any

from src.agents.base import BaseAgent
from src.clustering.features import UserFeatures


class PopAlignedPersonaAgent(BaseAgent):
    """Cluster-stat aligned static persona baseline.

    Samples individual attributes from the cluster's population distribution
    to create a distribution-matched persona. Uses system prompt only —
    no enforcement harness, no behavioral constraints.
    """

    def __init__(
        self,
        *args,
        cluster_attributes: dict[str, Any] | None = None,
        sampled_attributes: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # cluster_attributes: population-level stats (mean, std, percentiles)
        # sampled_attributes: one draw from the distribution for this individual
        self.cluster_attributes = cluster_attributes or {}
        self.sampled_attributes = sampled_attributes or {}

    def get_role_description(self) -> str:
        desc = "You are a participant in an online community discussion.\n\n"
        desc += "Your demographic and attitudinal profile (sampled from population distribution):\n"

        for attr, value in self.sampled_attributes.items():
            desc += f"  - {attr}: {value}\n"

        if self.cluster_attributes:
            desc += "\nYour community's population characteristics:\n"
            for attr, stats in self.cluster_attributes.items():
                if isinstance(stats, dict):
                    desc += f"  - {attr}: mean={stats.get('mean', 'N/A'):.2f}, "
                    desc += f"std={stats.get('std', 'N/A'):.2f}\n"
                else:
                    desc += f"  - {attr}: {stats}\n"

        desc += "\nBehave as someone with this background would."
        return desc

    def get_constraints_text(self) -> str:
        return "Stay consistent with your population-aligned profile."


def compute_cluster_attributes(
    members_features: list[UserFeatures],
) -> dict[str, dict[str, float]]:
    """Compute population-level attribute statistics for a cluster.

    Returns per-attribute mean/std for distribution matching.
    """
    import numpy as np

    if not members_features:
        return {}

    # Never bind prompt semantics to ``to_vector()`` positions. That vector is
    # the mutable clustering space and currently begins with reply_rate,
    # mean_indentation, verbosity, activity.
    attr_accessors = {
        "reply_depth": "mean_indentation",
        "verbosity": "verbosity",
        "question_rate": "question_rate",
        "conflict_engagement_ratio": "conflict_engagement_ratio",
    }

    stats = {}
    for name, field_name in attr_accessors.items():
        col = np.array([getattr(f, field_name) for f in members_features], dtype=float)
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
        }

    # Add message_count and thread_count as discrete attributes
    msg_counts = [f.message_count for f in members_features]
    thread_counts = [f.thread_count for f in members_features]
    stats["activity_level"] = {
        "mean": float(np.mean(msg_counts)),
        "std": float(np.std(msg_counts)),
        "median": float(np.median(msg_counts)),
    }
    stats["breadth"] = {
        "mean": float(np.mean(thread_counts)),
        "std": float(np.std(thread_counts)),
        "median": float(np.median(thread_counts)),
    }

    return stats


def sample_individual_attributes(
    cluster_stats: dict[str, dict[str, float]],
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Sample one individual's attributes from the cluster distribution.

    Uses truncated normal sampling around mean±2σ for realism.
    """
    rng = rng or random.Random()
    sampled = {}

    attr_labels = {
        "reply_depth": "typical reply depth",
        "verbosity": "message verbosity",
        "question_rate": "inquisitiveness",
        "conflict_engagement_ratio": "conflict involvement",
        "activity_level": "activity level",
        "breadth": "topic breadth",
    }

    for attr_key, label in attr_labels.items():
        stats = cluster_stats.get(attr_key)
        if not stats:
            continue

        mean = stats["mean"]
        std = stats.get("std", 0.0)
        lo = stats.get("min", max(0.0, mean - 2 * std))
        hi = stats.get("max", mean + 2 * std)

        if std > 0:
            # Truncated normal sample
            val = rng.gauss(mean, std)
            val = max(lo, min(hi, val))
        else:
            val = mean

        # Quantize into descriptive categories for persona readability
        if attr_key in ("reply_depth", "verbosity", "activity_level", "breadth"):
            if val >= mean + 0.5 * std if std > 0 else val >= mean:
                level = "high"
            elif val <= mean - 0.5 * std if std > 0 else val <= mean:
                level = "low"
            else:
                level = "moderate"
            sampled[label] = f"{level} ({val:.2f})"
        else:
            sampled[label] = f"{val:.3f}"

    return sampled
