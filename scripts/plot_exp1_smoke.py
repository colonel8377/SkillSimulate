"""Plot Exp1 Wikipedia smoke-test results.

Reads the rebuilt per-cell result JSONs from
outputs/results/{experiment_name}/ and writes PNGs to
outputs/results/{experiment_name}/figures/.

Usage:
    python scripts/plot_exp1_smoke.py --name exp1_wikipedia_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "vanilla",
    "descriptive",
    "segmentation",
    "pop_aligned",
    "colleague_skill",
    "clustering_only",
    "cadp_full",
    "cadp_shuffled",
    "cadp_minus_edna",
    "cadp_minus_mm",
    "cadp_minus_ap",
    "cadp_constraint_only",
    "pop_aligned_cadp",
]

SHORT_LABELS = {
    "vanilla": "vanilla",
    "descriptive": "desc",
    "segmentation": "seg",
    "pop_aligned": "pop",
    "colleague_skill": "skill",
    "clustering_only": "clust",
    "cadp_full": "cadp",
    "cadp_shuffled": "shuf",
    "cadp_minus_edna": "-edna",
    "cadp_minus_mm": "-mm",
    "cadp_minus_ap": "-ap",
    "cadp_constraint_only": "constr",
    "pop_aligned_cadp": "pop+cadp",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*_result.json")):
        rows.append(json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    df["complexity_gap"] = (df["complexity_real"] - df["complexity_sim"]).abs()
    df["violation_rate_pct"] = df["enforcement_violation_rate"] * 100.0

    order_map = {c: i for i, c in enumerate(CONDITION_ORDER)}
    df["_order"] = df["condition"].map(order_map)
    df = df.sort_values("_order").reset_index(drop=True)
    df["short"] = df["condition"].map(SHORT_LABELS)
    return df


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_structural(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    metrics = [
        ("delta_q_modularity", "ΔQ Modularity (lower = better)"),
        ("ned", "NED (lower = better)"),
        ("structural_fidelity", "Structural Fidelity (higher = better)"),
        ("dtw_distance", "DTW Distance (lower = better)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(df["short"], df[col], color="steelblue")
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(df[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    savefig(fig, out_dir / "structural_metrics.png")


def plot_linguistic(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.2
    cols = [
        "discourse_marker_match_normal",
        "sentiment_trajectory_similarity_normal",
        "speech_act_similarity_normal",
        "sip_normal",
    ]
    labels = ["Discourse", "Sentiment", "Speech Act", "SIP"]
    for i, (col, label) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - 1.5) * width, df[col], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Linguistic Fidelity by Condition (higher = better)")
    ax.legend()
    savefig(fig, out_dir / "linguistic_metrics.png")


def plot_predictive(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.2
    cols = [
        "pred_conflict_prediction_f1",
        "pred_persuasion_prediction_f1",
        "pred_escalation_prediction_f1",
        "pred_predictive_fidelity",
    ]
    labels = ["Conflict F1", "Persuasion F1", "Escalation F1", "Composite"]
    for i, (col, label) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - 1.5) * width, df[col], width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=45, ha="right")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Score")
    ax.set_title("Predictive Fidelity by Condition (higher = better)")
    ax.legend()
    savefig(fig, out_dir / "predictive_metrics.png")


def plot_enforcement(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["short"], df["violation_rate_pct"], color="coral")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Enforcement Violation Rate by Condition")
    ax.tick_params(axis="x", rotation=45)
    for bar, v in zip(bars, df["violation_rate_pct"]):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    savefig(fig, out_dir / "enforcement_rate.png")


def plot_summary_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Z-score heatmap of selected key metrics across conditions."""
    cols = {
        "delta_q_modularity": "ΔQ↓",
        "ned": "NED↓",
        "structural_fidelity": "Struct.Fid↑",
        "dtw_distance": "DTW↓",
        "sip_normal": "SIP↑",
        "pred_predictive_fidelity": "Pred.Fid↑",
        "violation_rate_pct": "Viol%",
    }
    sub = df[list(cols.keys())].copy()
    sub["delta_q_modularity"] = -sub["delta_q_modularity"]  # invert so higher=better
    sub["ned"] = -sub["ned"]
    sub["dtw_distance"] = -sub["dtw_distance"]

    z = (sub - sub.mean()) / sub.std()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(df)))
    ax.set_xticklabels(cols.values(), rotation=45, ha="right")
    ax.set_yticklabels(df["short"])
    ax.set_title("Z-Score Summary (green = better, red = worse)")
    fig.colorbar(im, ax=ax)

    for i in range(len(df)):
        for j in range(len(cols)):
            ax.text(j, i, f"{z.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)

    savefig(fig, out_dir / "summary_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="exp1_wikipedia_smoke", help="Experiment sub-dir under outputs/results/")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    results_dir = base / "outputs" / "results" / args.name
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    full_csv = results_dir / "exp1_all_metrics_full.csv"
    if not full_csv.exists():
        df.drop(columns=["short", "_order", "complexity_gap", "violation_rate_pct"], errors="ignore").to_csv(
            full_csv, index=False
        )
        print(f"Saved {full_csv}")

    plot_structural(df, figures_dir)
    plot_linguistic(df, figures_dir)
    plot_predictive(df, figures_dir)
    plot_enforcement(df, figures_dir)
    plot_summary_heatmap(df, figures_dir)

    print(f"\nFigures written to {figures_dir}")


if __name__ == "__main__":
    main()
