"""Ceiling analysis — remaining sim-to-real gap per method family per layer.

ARS review 2026-06-19 (Major Revision, panel mean 58.2, final 56/100):
the paper's contribution should not depend on CADP winning absolutely.
Whatever the 13-condition sweep produces, the paper has a defensible
finding in the form:

    "Across N method families, the remaining sim-to-real gap on layer L
     is X (95% CI [...]). The distillation+constraints family closes Y
     of the gap relative to persona prompting; Z remains unclosed by
     current methods."

This is the negative-result insurance (review Tier-1 fix C-C, and the
"Ceiling Analysis" section added under §5.X of the rewritten outline).
It runs against any results DataFrame — smoke pilot, framing pilot, or
full experiment — and produces a layer-by-layer ceiling report that
publishes regardless of CADP's absolute rank.

Usage:
    from src.analysis.ceiling import compute_ceiling, format_ceiling_table
    report = compute_ceiling(df)
    print(format_ceiling_table(report))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.stats import DEFAULT_LAYER_METRICS, DEFAULT_METRIC_DIRECTION


# Method-family registry. Maps each family name to the conditions whose
# best representative defines that family's ceiling. Conditions not
# listed here are silently ignored (callers can override).
#
# Family choice rationale (review 2026-06-19):
#   - ``none``: no persona injection — establishes the floor.
#   - ``persona_prompting``: identity/attribute injection without
#     enforcement. Includes length_matched_control so the token-mass
#     alternative explanation (DA-E1) is folded into the family ceiling.
#   - ``distillation_advisory``: rule distillation without hard
#     enforcement (COLLEAGUE.SKILL lineage).
#   - ``distillation_enforced``: rule distillation + three-tier
#     enforcement (CADP family, all ablations included so the family
#     ceiling reflects the best CADP variant, not just cadp_full).
#   - ``perfect_reference``: real_history self-similarity ceiling
#     (outline §6.2). Defines "zero gap".
DEFAULT_METHOD_FAMILIES: dict[str, list[str]] = {
    "none": ["vanilla"],
    "persona_prompting": [
        "descriptive",
        "segmentation",
        "pop_aligned",
        "rich_narrative",
        "clustering_only",
        "length_matched_control",
    ],
    "distillation_advisory": ["colleague_skill"],
    "distillation_filter_enforced": [
        "cadp_full_nuwa",
        "cadp_full_colleague",
        "cadp_shuffled",
        "cadp_minus_edna",
        "cadp_minus_mm",
        "cadp_minus_ap",
        "cadp_constraint_only",
        "pop_aligned_cadp",
    ],
    "perfect_reference": ["real_history"],
}


@dataclass
class CeilingReport:
    """Per-method-family, per-layer remaining sim-to-real gap.

    All gaps are non-negative; 0.0 means the family's best condition
    matched the perfect reference on that layer. NaN means no data for
    that (family, layer) cell.
    """

    gap_table: pd.DataFrame
    # rows = method_family, cols = metric_layer, values = remaining gap
    best_condition: pd.DataFrame
    # rows = method_family, cols = layer, values = condition name that
    # achieved the family's best on that layer
    family_best_fidelity: pd.DataFrame
    # rows = method_family, cols = layer, values = best fidelity ∈ [0,1]
    layer_metric_counts: dict[str, int]
    # number of metric columns that actually had data, per layer
    perfect_reference: str
    # name of the condition (or fallback label) used as "zero gap"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_table": self.gap_table.to_dict(),
            "best_condition": self.best_condition.to_dict(),
            "family_best_fidelity": self.family_best_fidelity.to_dict(),
            "layer_metric_counts": self.layer_metric_counts,
            "perfect_reference": self.perfect_reference,
        }


def _layer_fidelity(
    df: pd.DataFrame,
    condition: str,
    metrics: list[str],
    direction: dict[str, bool],
) -> float:
    """Mean normalized fidelity of one condition on one layer ∈ [0, 1].

    Per-metric fidelity is computed as the metric's mean across repeats;
    lower-is-better metrics are inverted via ``1 - value`` (assumes the
    metric is already bounded in [0,1]). Layer fidelity = mean of
    per-metric fidelities. Returns NaN if no metrics have data.
    """
    fids: list[float] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = df[df["condition"] == condition][metric].dropna().values
        if len(vals) == 0:
            continue
        v = float(np.mean(vals))
        higher_better = direction.get(metric, True)
        # Clip after direction inversion so out-of-range distances do not
        # produce negative or >1 fidelities. This is a normalization
        # convention, not a statistical claim.
        f = v if higher_better else (1.0 - v)
        fids.append(min(1.0, max(0.0, f)))
    if not fids:
        return float("nan")
    return float(np.mean(fids))


def _layer_fidelity_samples(
    df: pd.DataFrame,
    condition: str,
    metrics: list[str],
    direction: dict[str, bool],
) -> np.ndarray:
    """Flat array of per-observation normalized fidelities (for bootstrap)."""
    out: list[float] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = df[df["condition"] == condition][metric].dropna().values
        if len(vals) == 0:
            continue
        higher_better = direction.get(metric, True)
        for v in vals:
            f = float(v) if higher_better else (1.0 - float(v))
            out.append(min(1.0, max(0.0, f)))
    return np.asarray(out, dtype=float)


def compute_ceiling(
    df: pd.DataFrame,
    method_families: dict[str, list[str]] | None = None,
    layer_metrics: dict[str, list[str]] | None = None,
    metric_direction: dict[str, bool] | None = None,
    perfect_reference_condition: str = "real_history",
) -> CeilingReport:
    """Compute per-family per-layer remaining sim-to-real gap.

    Args:
        df: Long-form results DataFrame (as produced by
            :meth:`MetricsAggregator.to_dataframe`). Must have a
            ``condition`` column plus one column per metric.
        method_families: Override the family → conditions mapping.
            Defaults to :data:`DEFAULT_METHOD_FAMILIES`.
        layer_metrics: Override layer → metrics mapping. Defaults to
            :data:`DEFAULT_LAYER_METRICS`.
        metric_direction: Override per-metric direction. Defaults to
            :data:`DEFAULT_METRIC_DIRECTION`.
        perfect_reference_condition: Condition whose fidelity is the
            "zero gap" reference. Defaults to ``real_history`` (outline
            §6.2 self-similarity ceiling). If absent from ``df``, falls
            back to the per-layer maximum across all observed conditions
            and records this in :attr:`CeilingReport.perfect_reference`.

    Returns:
        :class:`CeilingReport`.
    """
    method_families = method_families or DEFAULT_METHOD_FAMILIES
    layer_metrics = layer_metrics or DEFAULT_LAYER_METRICS
    metric_direction = metric_direction or DEFAULT_METRIC_DIRECTION
    layers = list(layer_metrics.keys())

    # Determine the perfect reference fidelity per layer
    perfect_fidelity: dict[str, float] = {}
    used_explicit_reference = perfect_reference_condition in df["condition"].unique()
    for layer in layers:
        metrics_present = [m for m in layer_metrics[layer] if m in df.columns]
        if not metrics_present:
            perfect_fidelity[layer] = 1.0
            continue
        if used_explicit_reference:
            f = _layer_fidelity(
                df, perfect_reference_condition, metrics_present, metric_direction,
            )
            perfect_fidelity[layer] = f if not np.isnan(f) else 1.0
        else:
            best = float("nan")
            for cond in df["condition"].unique():
                f = _layer_fidelity(df, cond, metrics_present, metric_direction)
                if not np.isnan(f) and (np.isnan(best) or f > best):
                    best = f
            perfect_fidelity[layer] = best if not np.isnan(best) else 1.0

    reference_name = (
        perfect_reference_condition if used_explicit_reference
        else "<observed max — real_history absent>"
    )

    # Per-family, per-layer: find the best condition and its gap
    gap_rows: dict[str, dict[str, float]] = {}
    best_rows: dict[str, dict[str, str]] = {}
    fid_rows: dict[str, dict[str, float]] = {}
    layer_metric_counts: dict[str, int] = {}

    for layer in layers:
        layer_metric_counts[layer] = sum(
            1 for m in layer_metrics[layer] if m in df.columns
        )

    for family, conditions in method_families.items():
        gap_rows[family] = {}
        best_rows[family] = {}
        fid_rows[family] = {}
        for layer in layers:
            metrics_present = [m for m in layer_metrics[layer] if m in df.columns]
            if not metrics_present:
                gap_rows[family][layer] = float("nan")
                best_rows[family][layer] = ""
                fid_rows[family][layer] = float("nan")
                continue

            best_cond = ""
            best_fid = float("nan")
            for cond in conditions:
                if cond not in df["condition"].unique():
                    continue
                f = _layer_fidelity(df, cond, metrics_present, metric_direction)
                if np.isnan(f):
                    continue
                if np.isnan(best_fid) or f > best_fid:
                    best_fid = f
                    best_cond = cond

            if np.isnan(best_fid):
                gap_rows[family][layer] = float("nan")
                best_rows[family][layer] = ""
                fid_rows[family][layer] = float("nan")
                continue

            ref = perfect_fidelity[layer]
            gap_rows[family][layer] = max(0.0, ref - best_fid)
            best_rows[family][layer] = best_cond
            fid_rows[family][layer] = best_fid

    return CeilingReport(
        gap_table=pd.DataFrame(gap_rows).T,
        best_condition=pd.DataFrame(best_rows).T,
        family_best_fidelity=pd.DataFrame(fid_rows).T,
        layer_metric_counts=layer_metric_counts,
        perfect_reference=reference_name,
    )


def compute_ceiling_ci(
    df: pd.DataFrame,
    report: CeilingReport,
    layer_metrics: dict[str, list[str]] | None = None,
    metric_direction: dict[str, bool] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CI on each (family, layer) gap cell.

    Returns a DataFrame with columns ``family``, ``layer``, ``gap``,
    ``ci_low``, ``ci_high``. Cells with fewer than 3 observations are
    returned with NaN CIs.

    This is split from :func:`compute_ceiling` so the headline report
    stays cheap to compute; CI is added when reporting.
    """
    layer_metrics = layer_metrics or DEFAULT_LAYER_METRICS
    metric_direction = metric_direction or DEFAULT_METRIC_DIRECTION
    alpha = 1.0 - confidence
    rng = np.random.default_rng(seed)

    out: list[dict[str, Any]] = []
    for family in report.gap_table.index:
        for layer in report.gap_table.columns:
            gap = report.gap_table.loc[family, layer]
            best_cond = report.best_condition.loc[family, layer]
            if isinstance(gap, float) and np.isnan(gap):
                continue
            metrics_present = [m for m in layer_metrics[layer] if m in df.columns]
            samples = _layer_fidelity_samples(
                df, best_cond, metrics_present, metric_direction,
            ) if best_cond else np.asarray([], dtype=float)
            if samples.size < 3:
                out.append({
                    "family": family, "layer": layer, "gap": gap,
                    "ci_low": float("nan"), "ci_high": float("nan"),
                    "n": int(samples.size),
                })
                continue
            # Bootstrap: resample fidelity observations, compute gap = ref - mean
            ref = report.family_best_fidelity.loc[family, layer] + gap  # reconstruct ref
            # Note: ref reconstruction = best_fid + gap = perfect_fidelity[layer]
            resampled_gaps: list[float] = []
            for _ in range(n_resamples):
                idx = rng.integers(0, samples.size, samples.size)
                resampled_gaps.append(max(0.0, ref - float(np.mean(samples[idx]))))
            out.append({
                "family": family, "layer": layer, "gap": gap,
                "ci_low": float(np.percentile(resampled_gaps, 100 * alpha / 2)),
                "ci_high": float(np.percentile(resampled_gaps, 100 * (1 - alpha / 2))),
                "n": int(samples.size),
            })
    return pd.DataFrame(out)


def format_ceiling_table(
    report: CeilingReport,
    ci_df: pd.DataFrame | None = None,
) -> str:
    """Format :class:`CeilingReport` as Markdown for paper inclusion.

    Rows = method family, columns = metric layer, cells = "remaining
    gap [95% CI] (best: <condition>)" when ``ci_df`` is supplied,
    otherwise "remaining gap (best: <condition>)".

    This is the framing-(a) "Ceiling Analysis" table — publishes
    regardless of CADP's absolute ranking.
    """
    layers = list(report.gap_table.columns)
    lines = [
        "## Ceiling Analysis — remaining sim-to-real gap per method family",
        "",
        f"Perfect reference: `{report.perfect_reference}`.",
        "",
    ]

    header = "| Method family | " + " | ".join(layers) + " |"
    sep = "|---" * (len(layers) + 1) + "|"
    lines.extend([header, sep])

    # Build a (family, layer) → CI string lookup if provided
    ci_lookup: dict[tuple[str, str], str] = {}
    if ci_df is not None and len(ci_df) > 0:
        for _, row in ci_df.iterrows():
            lo, hi = row["ci_low"], row["ci_high"]
            if isinstance(lo, float) and np.isnan(lo):
                continue
            ci_lookup[(row["family"], row["layer"])] = (
                f" [{lo:.3f}, {hi:.3f}]"
            )

    for family, row in report.gap_table.iterrows():
        cells = [f"**{family}**"]
        for layer in layers:
            gap = row[layer]
            best = report.best_condition.loc[family, layer]
            if isinstance(gap, float) and np.isnan(gap):
                cells.append("--")
                continue
            ci_str = ci_lookup.get((family, layer), "")
            tag = f" ({best})" if best else ""
            cells.append(f"{gap:.3f}{ci_str}{tag}")
        lines.append("| " + " | ".join(cells) + " |")

    # Metric-count footer so readers can see which layers are sparse
    counts_str = ", ".join(
        f"{l}: {n}" for l, n in report.layer_metric_counts.items()
    )
    lines.extend(["", f"_Metrics with data per layer — {counts_str}_"])
    return "\n".join(lines)
