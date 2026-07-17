"""Pre-registered CADP viability verdict for the Exp1 feasibility gate.

The feasibility gate uses one distance from each non-redundant family:
action fidelity, interaction structure, and independent linguistics. All are
lower-is-better. Diagnostic metrics never receive separate votes.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def evaluate_viability(df: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Return a deterministic GO / CONDITIONAL_GO / STOP verdict.

    Pairing keys are dataset, model, and repeat. A family is a win only when
    the treatment beats the content-matched control by the configured minimum
    relative effect in enough paired repeats. GO additionally requires
    throughput, safe-template, action/text, and no-large-regression guards.
    """
    treatment = config.viability_treatment
    control = config.viability_control
    metrics = list(config.viability_primary_metrics)
    pair_keys = [c for c in ("dataset", "model", "repeat") if c in df.columns]
    if len(pair_keys) != 3:
        raise ValueError(f"Viability results lack pairing keys: {pair_keys}")

    t = df[df["condition"] == treatment]
    c = df[df["condition"] == control]
    paired = t.merge(c, on=pair_keys, suffixes=("_treatment", "_control"))
    if len(paired) < config.viability_min_pairs:
        return {
            "verdict": "CONDITIONAL_GO",
            "reason": "insufficient_complete_pairs",
            "n_pairs": int(len(paired)),
            "required_pairs": int(config.viability_min_pairs),
            "treatment": treatment,
            "control": control,
        }

    required_repeat_wins = math.ceil(
        # Decimal YAML values such as 0.6666666667 are intended to mean
        # two-thirds. A tiny representation excess must not turn 2/3 into
        # a 3/3 requirement when n=3.
        config.viability_min_repeat_win_fraction * len(paired) - 1e-9
    )
    metric_results: dict[str, dict[str, Any]] = {}
    metric_wins = 0
    any_large_regression = False
    min_relative = float(config.viability_min_relative_improvement)
    max_regression = float(config.viability_max_family_regression)
    for metric in metrics:
        tk, ck = f"{metric}_treatment", f"{metric}_control"
        if tk not in paired or ck not in paired:
            raise ValueError(f"Pre-registered viability metric missing: {metric}")
        valid = paired[[tk, ck]].dropna()
        denominators = valid[ck].abs().clip(lower=1e-12)
        relative_improvements = (valid[ck] - valid[tk]) / denominators
        wins = int((relative_improvements >= min_relative).sum())
        ties = int((valid[tk] == valid[ck]).sum())
        is_win = len(valid) == len(paired) and wins >= required_repeat_wins
        mean_control = float(valid[ck].mean())
        mean_treatment = float(valid[tk].mean())
        mean_regression = (
            (mean_treatment - mean_control) / max(abs(mean_control), 1e-12)
        )
        large_regression = mean_regression > max_regression
        any_large_regression = any_large_regression or large_regression
        metric_wins += int(is_win)
        metric_results[metric] = {
            "treatment_mean": mean_treatment,
            "control_mean": mean_control,
            "paired_mean_difference": float((valid[tk] - valid[ck]).mean()),
            "paired_mean_relative_improvement": float(relative_improvements.mean()),
            "repeat_wins": wins,
            "repeat_ties": ties,
            "required_repeat_wins": required_repeat_wins,
            "metric_win": is_win,
            "large_regression": large_regression,
        }

    msg_t = paired["simulation_message_count_treatment"].astype(float)
    msg_c = paired["simulation_message_count_control"].astype(float)
    message_ratios = msg_t / msg_c.where(msg_c > 0)
    min_message_ratio = float(message_ratios.min())

    safe_key = "enforcement_safe_template_rate_treatment"
    if safe_key not in paired:
        raise ValueError(f"Viability quality guard missing: {safe_key}")
    max_safe_rate = float(paired[safe_key].astype(float).max())
    consistency_key = "action_text_consistency_treatment"
    if consistency_key not in paired:
        raise ValueError(f"Viability quality guard missing: {consistency_key}")
    min_action_text_consistency = float(
        paired[consistency_key].astype(float).min()
    )
    throughput_pass = min_message_ratio >= config.viability_min_message_ratio
    safe_template_pass = max_safe_rate <= config.viability_max_safe_template_rate
    action_text_pass = (
        min_action_text_consistency
        >= config.viability_min_action_text_consistency
    )
    regression_pass = not any_large_regression
    quality_pass = (
        throughput_pass and safe_template_pass
        and action_text_pass and regression_pass
    )

    if quality_pass and metric_wins >= config.viability_min_metric_wins:
        verdict = "GO"
        reason = "primary_metric_and_quality_gates_passed"
    elif quality_pass and metric_wins == config.viability_min_metric_wins - 1:
        verdict = "CONDITIONAL_GO"
        reason = "borderline_primary_metric_support"
    else:
        verdict = "STOP"
        reason = "primary_metric_or_quality_gate_failed"

    return {
        "verdict": verdict,
        "reason": reason,
        "treatment": treatment,
        "control": control,
        "n_pairs": int(len(paired)),
        "primary_metrics": metrics,
        "metric_wins": metric_wins,
        "required_metric_wins": int(config.viability_min_metric_wins),
        "metric_results": metric_results,
        "quality_guards": {
            "min_message_ratio": min_message_ratio,
            "required_min_message_ratio": float(config.viability_min_message_ratio),
            "throughput_pass": throughput_pass,
            "max_safe_template_rate": max_safe_rate,
            "allowed_max_safe_template_rate": float(config.viability_max_safe_template_rate),
            "safe_template_pass": safe_template_pass,
            "min_action_text_consistency": min_action_text_consistency,
            "required_min_action_text_consistency": float(
                config.viability_min_action_text_consistency
            ),
            "action_text_pass": action_text_pass,
            "max_allowed_family_regression": max_regression,
            "no_large_family_regression_pass": regression_pass,
        },
        "minimum_relative_improvement": min_relative,
        "excluded_from_verdict": [
            "action_matrix_similarity", "rsa", "structural_fidelity",
            "sentiment_trajectory_similarity",
        ],
    }
