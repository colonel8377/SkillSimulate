"""Visualization for experiment results — radar charts, network viz, trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx


def radar_chart(
    df: pd.DataFrame,
    metrics: list[str],
    conditions: list[str],
    output_path: str | Path,
    title: str = "Condition Comparison",
) -> None:
    """Generate radar chart comparing conditions across metrics.

    Figure 3 in the paper.
    """
    # Normalize metrics to [0, 1]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    palette = sns.color_palette("husl", len(conditions))

    for i, condition in enumerate(conditions):
        cond_df = df[df["condition"] == condition]
        values = []
        for metric in metrics:
            vals = cond_df[metric].dropna().values
            values.append(float(np.mean(vals)) if len(vals) > 0 else 0)

        # Normalize per metric (0-1 based on max across conditions)
        all_vals = []
        for metric in metrics:
            all_vals.append(df[metric].dropna().values)
        max_vals = [max(v) if len(v) > 0 else 1 for v in all_vals]
        normalized = [min(v / (m + 1e-10), 1.0) for v, m in zip(values, max_vals)]
        normalized += normalized[:1]

        ax.plot(angles, normalized, "o-", linewidth=2, label=condition, color=palette[i])
        ax.fill(angles, normalized, alpha=0.1, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def interaction_network_viz(
    graph_data: dict,
    communities: dict[str, int] | None = None,
    output_path: str | Path = "",
    title: str = "Interaction Network",
) -> None:
    """Visualize interaction network.

    Figure 4 in the paper.
    """
    graph = nx.Graph()
    for node in graph_data.get("nodes", []):
        graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in graph_data.get("edges", []):
        graph.add_edge(
            edge["source"], edge["target"],
            weight=edge.get("weight", 1),
        )

    if len(graph) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(graph, k=1.5 / np.sqrt(max(len(graph), 1)), seed=42)

    # Color by community
    if communities:
        colors = [communities.get(node, 0) for node in graph.nodes()]
    else:
        colors = [graph.nodes[n].get("cluster_id", 0) for n in graph.nodes()]

    # Size by degree
    degrees = dict(graph.degree())
    sizes = [max(degrees.get(n, 0) * 50, 100) for n in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes,
                           cmap="tab10", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)

    if len(graph) <= 50:
        nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def trajectory_plot(
    trajectories: dict[str, list[float]],
    output_path: str | Path = "",
    title: str = "Polarization Index Over Time",
    ylabel: str = "Polarization Index",
) -> None:
    """Plot temporal trajectory comparison.

    Figure 5 in the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, values in trajectories.items():
        rounds = range(len(values))
        ax.plot(rounds, values, marker="o", label=label, linewidth=2)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def key_events_timeline(
    events: dict[str, list[int]],
    output_path: str | Path = "",
    title: str = "Key Event Timing Comparison",
) -> None:
    """Plot key event timing comparison.

    Figure 7 in the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    labels = list(events.keys())
    for i, (label, rounds) in enumerate(events.items()):
        for r in rounds:
            ax.scatter(r, i, s=200, zorder=5)
            ax.annotate(f"R{r}", (r, i), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def multi_timepoint_network_snapshots(
    snapshot_graphs: list[dict],
    snapshot_labels: list[str],
    output_path: str | Path = "",
    title: str = "Interaction Network Evolution",
) -> None:
    """Plot interaction network at multiple timepoints.

    Figure 6 in the paper.
    """
    n = len(snapshot_graphs)
    if n == 0:
        return

    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (graph_data, label) in enumerate(zip(snapshot_graphs, snapshot_labels)):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        graph = nx.Graph()
        for node in graph_data.get("nodes", []):
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in graph_data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        if len(graph) > 0:
            pos = nx.spring_layout(graph, k=1.5 / np.sqrt(max(len(graph), 1)), seed=42)
            colors = [graph.nodes[n].get("cluster_id", 0) for n in graph.nodes()]
            degrees = dict(graph.degree())
            sizes = [max(degrees.get(n, 0) * 50, 100) for n in graph.nodes()]
            nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes,
                                   cmap="tab10", alpha=0.8, ax=ax)
            nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)

        ax.set_title(label, fontsize=11)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(snapshot_graphs), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
