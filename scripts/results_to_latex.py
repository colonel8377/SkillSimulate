"""Convert exp1 smoke-test results to LaTeX tables.

Usage:
    conda run -n SkillSimulate python scripts/results_to_latex.py --name exp1_wikipedia_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    # escape LaTeX
    df["condition_tex"] = df["condition"].str.replace("_", r"\_", regex=False)
    return df


def fmt(v, d=3):
    if pd.isna(v):
        return "--"
    return f"{v:.{d}f}"


def bold_best(col, direction):
    """Return a dict {idx: bool} indicating the best value(s) in col."""
    if col.isna().all():
        return {}
    if direction == "min":
        best = col.min()
    else:
        best = col.max()
    return {i: abs(col.iloc[i] - best) < 1e-9 for i in range(len(col))}


def make_table(df, rows, columns, directions, caption, label):
    """Build a LaTeX table with bold best values.

    rows: list of (col_name, display_name)
    columns: list of (col_name, display_name, direction)
    directions: dict col -> 'min'/'max'
    """
    # Build best-marker dict
    best_markers = {}
    for col, _, direction in columns:
        best_markers[col] = bold_best(df[col], direction)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    colspec = "l" + "r" * len(columns)
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    header = "Condition" + "".join(f" & {disp}" for _, disp, _ in columns)
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [row["condition_tex"]]
        for col, _, _ in columns:
            v = row[col]
            s = fmt(v)
            if best_markers[col].get(row.name, False):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="exp1_wikipedia_smoke")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    results_dir = base / "outputs" / "results" / args.name
    df = load_results(results_dir)

    sections = []

    # Table 1: Structural / network
    cols1 = [
        ("delta_q_modularity", r"$\Delta Q$ $\downarrow$", "min"),
        ("ei_polarization_sim", r"E-I Polar. $\uparrow$", "max"),
        ("ned", r"NED $\downarrow$", "min"),
        ("coverage", r"Coverage $\uparrow$", "max"),
        ("structural_fidelity", r"Struct. Fid. $\uparrow$", "max"),
        ("ks_statistic", r"KS $\downarrow$", "min"),
        ("dtw_distance", r"DTW $\downarrow$", "min"),
        ("action_matrix_similarity", r"Act. Mat. Sim. $\uparrow$", "max"),
        ("rsa", r"RSA $\uparrow$", "max"),
    ]
    sections.append(("Structural and Network Fidelity", "tab:structural", cols1, [
        r"\textbf{Note:} Structural fidelity is nearly constant ($\approx$0.934) across conditions, indicating metric saturation. Action matrix similarity is negative because the simulated matrix (30 agents) is truncated to match the much larger real matrix ($\sim$38k users).",
    ]))

    # Table 2: Linguistic
    cols2 = [
        ("discourse_marker_match_normal", r"Discourse $\uparrow$", "max"),
        ("sentiment_trajectory_similarity_normal", r"Sentiment $\uparrow$", "max"),
        ("speech_act_similarity_normal", r"Speech Act $\uparrow$", "max"),
        ("sip_normal", r"SIP $\uparrow$", "max"),
    ]
    sections.append(("Linguistic Fidelity", "tab:linguistic", cols2, [
        r"\textbf{Note:} ``Normal'' stratum excludes safe-template fallback messages. In this run the safe-template count is zero for all conditions, so normal and pooled metrics coincide.",
    ]))

    # Table 3: Predictive
    cols3 = [
        ("pred_conflict_prediction_f1", r"Conflict F1 $\uparrow$", "max"),
        ("pred_persuasion_prediction_f1", r"Persuasion F1 $\uparrow$", "max"),
        ("pred_escalation_prediction_f1", r"Escalation F1 $\uparrow$", "max"),
        ("pred_predictive_fidelity", r"Pred. Fidelity $\uparrow$", "max"),
    ]
    sections.append(("Predictive Fidelity", "tab:predictive", cols3, [
        r"\textbf{Note:} All predictive tasks use heuristic ground-truth labels (no held-out $\kappa$-validated annotations), so F1 scores should be treated as exploratory.",
    ]))

    # Table 4: Enforcement
    cols4 = [
        ("enforcement_total_checks", r"Checks", "min"),
        ("enforcement_total_violations", r"Violations", "min"),
        ("violation_rate_pct", r"Viol. Rate (\%) $\downarrow$", "min"),
    ]
    sections.append(("Enforcement Violations", "tab:enforcement", cols4, [
        r"\textbf{Note:} Baselines without enforcement show zero checks/violations. `cadp\_shuffled' has the highest violation rate, confirming that mismatched skills break the constraint pipeline.",
    ]))

    # Table 5: Micro uniformity/complexity
    cols5 = [
        ("uniformity_sim", r"Uniformity Sim.", "max"),
        ("uniformity_real", r"Uniformity Real", "max"),
        ("uniformity_gap", r"Uniformity Gap $\downarrow$", "min"),
        ("complexity_sim", r"Complexity Sim.", "max"),
        ("complexity_real", r"Complexity Real", "max"),
        ("complexity_gap", r"Complexity Gap $\downarrow$", "min"),
    ]
    sections.append(("Behavioral Uniformity and Complexity", "tab:micro", cols5, [
        r"\textbf{Note:} Simulated complexity is near zero for most conditions, suggesting agents produce highly homogeneous action distributions. `cadp\_minus\_edna' has the smallest uniformity gap.",
    ]))

    out_lines = [
        r"% Auto-generated LaTeX tables for " + args.name,
        r"% Requires: \usepackage{booktabs}",
        "",
    ]

    for title, label, cols, notes in sections:
        out_lines.append(r"\subsection*{" + title + "}")
        out_lines.append(make_table(df, [], cols, {}, title, label))
        for note in notes:
            out_lines.append(note)
        out_lines.append("")

    out_path = results_dir / "results_tables.tex"
    out_path.write_text("\n".join(out_lines))
    print(f"Saved LaTeX tables to {out_path}")


if __name__ == "__main__":
    main()
