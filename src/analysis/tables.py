"""LaTeX table generation for paper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.stats import compare_conditions


def format_mean_std(values: np.ndarray) -> str:
    """Format mean ± std for LaTeX."""
    return f"{np.mean(values):.3f} $\\pm$ {np.std(values):.3f}"


def generate_main_results_table(
    df: pd.DataFrame,
    metrics: list[str],
    conditions: list[str],
) -> str:
    """Generate Table 1: main results (13 conditions × metric layers).

    Returns:
        LaTeX table string.
    """
    n_cols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Main results across conditions and metric layers.}",
        "\\label{tab:main_results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Condition & " + " & ".join(m.replace("_", "\\_") for m in metrics) + " \\\\",
        "\\midrule",
    ]

    for condition in conditions:
        row_data = [condition.replace("_", "\\_")]
        cond_df = df[df["condition"] == condition]

        for metric in metrics:
            vals = cond_df[metric].dropna().values
            if len(vals) > 0:
                row_data.append(format_mean_std(vals))
            else:
                row_data.append("--")

        lines.append(" & ".join(row_data) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_ablation_table(
    df: pd.DataFrame,
    metrics: list[str],
) -> str:
    """Generate Table 2: ablation + COLLEAGUE.SKILL chain results."""
    ablation_conditions = [
        "colleague_skill",
        "cadp_minus_ap",
        "cadp_full",
    ]

    return generate_main_results_table(df, metrics, ablation_conditions)


def generate_clustering_contribution_table(
    df: pd.DataFrame,
    metrics: list[str],
) -> str:
    """Generate Table 2b: clustering contribution isolation.

    Compares Descriptive Persona, Clustering-Only Descriptive Persona,
    and CADP Full to isolate clustering structure vs. behavioral rule
    distillation contributions (outline §5.2 Condition 6).
    """
    conditions = [
        "descriptive",
        "clustering_only",
        "cadp_full",
    ]
    return generate_main_results_table(df, metrics, conditions)


def generate_comparison_table(
    df: pd.DataFrame,
    metrics: list[str],
    treatment: str = "cadp_full",
    baselines: list[str] | None = None,
) -> str:
    """Generate statistical comparison table (CADP vs baselines)."""
    if baselines is None:
        baselines = [
            "vanilla",
            "descriptive",
            "segmentation",
            "pop_aligned",
            "colleague_skill",
            "clustering_only",
            "pop_aligned_cadp",
        ]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Statistical comparison of CADP vs baselines.}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Metric & Baseline & Cliff's $\\delta$ & $p$-value & Sig. \\\\",
        "\\midrule",
    ]

    for metric in metrics:
        comparisons = compare_conditions(df, metric, treatment, baselines)
        for _, row in comparisons.iterrows():
            lines.append(
                f"{metric.replace('_', '\\_')} & {row['baseline']} & "
                f"{row['cliffs_delta']:.3f} & {row['p_value']:.4f} & "
                f"{row['sig_marker']} \\\\"
            )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def save_tables(tables: dict[str, str], output_dir: str) -> None:
    """Save LaTeX tables to files."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, tex in tables.items():
        path = out / f"{name}.tex"
        with open(path, "w") as f:
            f.write(tex)


def generate_provenance_table(df: pd.DataFrame) -> str:
    """Per-(dataset, model) provenance table for outline §7.4 / §5.1(d).

    Surfaces the four reproducibility/anti-circularity columns added by
    audit P5 + P10 so reviewers can read off, per cell:
      * model_snapshot_date / model_commit_hash (P5)
      * used_role_label_proxy / used_held_out_events_heuristic (§7.4)
    Cells with empty / missing provenance are reported as
    ``unrecorded`` rather than dropped.
    """
    needed = {
        "dataset", "model",
        "model_snapshot_date", "model_commit_hash",
        "used_role_label_proxy", "used_held_out_events_heuristic",
    }
    missing = needed - set(df.columns)
    if missing:
        return (
            "% generate_provenance_table: input df missing columns: "
            f"{sorted(missing)} — cannot emit provenance table."
        )

    grouped = (
        df.groupby(["dataset", "model"])[
            [
                "model_snapshot_date", "model_commit_hash",
                "used_role_label_proxy", "used_held_out_events_heuristic",
            ]
        ]
        .agg(lambda s: s.iloc[0])
        .reset_index()
    )

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Reproducibility \\& anti-circularity provenance per (dataset, model).}",
        "\\label{tab:provenance}",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Dataset & Model & Snapshot date & Commit hash & Role-label proxy & Held-out heuristic \\\\",
        "\\midrule",
    ]
    for _, row in grouped.iterrows():
        snap = row["model_snapshot_date"] or "unrecorded"
        commit = row["model_commit_hash"] or "unrecorded"
        role_proxy = "yes" if bool(row["used_role_label_proxy"]) else "no"
        held_heur = "yes" if bool(row["used_held_out_events_heuristic"]) else "no"
        underscore = "_"
        latex_underscore = "\\_"
        dataset_lx = str(row["dataset"]).replace(underscore, latex_underscore)
        model_lx = str(row["model"]).replace(underscore, latex_underscore)
        snap_lx = str(snap).replace(underscore, latex_underscore)
        commit_lx = str(commit).replace(underscore, latex_underscore)
        lines.append(
            f"{dataset_lx} & {model_lx} & {snap_lx} & {commit_lx} & "
            f"{role_proxy} & {held_heur} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)
