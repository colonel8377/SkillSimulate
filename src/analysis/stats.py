"""Statistical tests for experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, bootstrap


def paired_wilcoxon_test(
    treatment: np.ndarray,
    control: np.ndarray,
    alternative: str = "greater",
) -> dict[str, float]:
    """Wilcoxon signed-rank test (paired).

    Returns:
        Dict with statistic and p-value.
    """
    stat, pval = wilcoxon(treatment, control, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(pval)}


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10000,
) -> tuple[float, float]:
    """Bootstrap confidence interval."""
    result = bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def cliffs_delta(treatment: np.ndarray, control: np.ndarray) -> float:
    """Cliff's delta effect size."""
    n = len(treatment)
    m = len(control)
    greater = sum(1 for t in treatment for c in control if t > c)
    less = sum(1 for t in treatment for c in control if t < c)
    return (greater - less) / (n * m)


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> float:
    """Cohen's d (independent groups, pooled SD).

    Used as the parametric companion to :func:`cliffs_delta` so the paper
    can report both a non-parametric (Cliff's δ) and a parametric (Cohen's
    d) effect size, addressing P2 of the audit.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    n_t, n_c = len(t), len(c)
    if n_t < 2 or n_c < 2:
        return float("nan")
    var_t = float(np.var(t, ddof=1))
    var_c = float(np.var(c, ddof=1))
    pooled = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    if pooled == 0.0:
        return 0.0
    return float((np.mean(t) - np.mean(c)) / pooled)


def paired_d_with_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10000,
) -> dict[str, float]:
    """Paired Cohen's d (a.k.a. d_z = mean(Δ) / sd(Δ)) with bootstrap CI.

    Pairs are taken in order; if the two arrays differ in length they are
    truncated to the shorter (consistent with :func:`compare_conditions`).
    Returns ``{"paired_d": d_z, "ci_low": l, "ci_high": h, "n_pairs": n}``.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    n = min(len(t), len(c))
    if n < 2:
        return {"paired_d": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_pairs": float(n)}
    diffs = t[:n] - c[:n]
    sd = float(np.std(diffs, ddof=1))
    d_z = float(np.mean(diffs) / sd) if sd > 0 else 0.0

    def _d_stat(sample: np.ndarray) -> float:
        s = float(np.std(sample, ddof=1))
        return float(np.mean(sample) / s) if s > 0 else 0.0

    try:
        boot = bootstrap(
            (diffs,),
            _d_stat,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="percentile",
        )
        ci_low = float(boot.confidence_interval.low)
        ci_high = float(boot.confidence_interval.high)
    except Exception:
        # SciPy occasionally raises on degenerate inputs; degrade gracefully.
        ci_low = float("nan")
        ci_high = float("nan")
    return {"paired_d": d_z, "ci_low": ci_low, "ci_high": ci_high, "n_pairs": float(n)}


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg multiple comparison correction.

    Returns:
        Boolean array indicating which hypotheses are rejected.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]

    rejected = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        threshold = alpha * (i + 1) / n
        if sorted_pvals[i] <= threshold:
            rejected[:i + 1] = True
            break

    # Unsort
    result = np.zeros(n, dtype=bool)
    result[sorted_idx] = rejected
    return result


def compare_conditions(
    df: pd.DataFrame,
    metric: str,
    treatment: str = "cadp_full",
    baselines: list[str] | None = None,
) -> pd.DataFrame:
    """Compare CADP against all baselines on a specific metric.

    Returns:
        DataFrame with per-baseline comparison results.
    """
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

    treat_vals = df[df["condition"] == treatment][metric].dropna().values

    results = []
    for baseline in baselines:
        ctrl_vals = df[df["condition"] == baseline][metric].dropna().values

        if len(treat_vals) == 0 or len(ctrl_vals) == 0:
            continue

        min_len = min(len(treat_vals), len(ctrl_vals))
        t = treat_vals[:min_len]
        c = ctrl_vals[:min_len]

        test = paired_wilcoxon_test(t, c)
        delta = cliffs_delta(t, c)
        ci_low, ci_high = bootstrap_ci(t - c)
        # P2: Cohen's d (independent) + paired d_z with bootstrap CI for the
        # parametric companion to Cliff's δ.
        d_indep = cohens_d(t, c)
        d_paired = paired_d_with_ci(t, c)

        # Significance marker
        p = test["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        results.append({
            "baseline": baseline,
            "treatment_mean": float(np.mean(t)),
            "control_mean": float(np.mean(c)),
            "cliffs_delta": delta,
            "cohens_d": d_indep,
            "paired_d": d_paired["paired_d"],
            "paired_d_ci_low": d_paired["ci_low"],
            "paired_d_ci_high": d_paired["ci_high"],
            "p_value": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": p < 0.05,
            "sig_marker": sig,
        })

    return pd.DataFrame(results)


def benjamini_hochberg_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values (one per input p-value).

    Companion to :func:`benjamini_hochberg` (which only returns reject/keep
    booleans). Used by :func:`compare_conditions_layered` so callers can
    threshold q at any α post-hoc.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    # Standard BH q: q_i = min_{j>=i} (n / rank_j) * p_j
    raw = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the top (largest rank) downward and clip to 1.
    q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return q


def compare_conditions_layered(
    df: pd.DataFrame,
    treatment: str = "cadp_full",
    baselines: list[str] | None = None,
    layer_metrics: dict[str, list[str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Layer-wise comparison with within-layer Benjamini-Hochberg correction.

    Audit P3: outline §5.1 promises layer-internal multiple-testing
    correction, but :func:`compare_conditions` reports raw p-values only.
    This wrapper iterates :data:`DEFAULT_LAYER_METRICS` (or a caller-
    supplied mapping), runs ``compare_conditions`` per metric, then applies
    BH within each layer's (metric × baseline) test family. Result columns:
    ``layer``, ``metric``, plus all columns of ``compare_conditions``,
    plus ``q_value`` (BH-corrected) and ``rejected_at_alpha``.

    Args:
        df: Long-form results DataFrame with a ``condition`` column.
        treatment: Treatment condition name.
        baselines: Baselines list (default = same as ``compare_conditions``).
        layer_metrics: Override mapping ``{layer: [metric, ...]}``. Defaults
            to :data:`DEFAULT_LAYER_METRICS`.
        alpha: Family-wise α used for the ``rejected_at_alpha`` flag.
    """
    layer_metrics = layer_metrics or DEFAULT_LAYER_METRICS
    pieces: list[pd.DataFrame] = []
    for layer, metrics in layer_metrics.items():
        layer_rows: list[pd.DataFrame] = []
        for metric in metrics:
            if metric not in df.columns:
                continue
            sub = compare_conditions(df, metric, treatment, baselines)
            if sub.empty:
                continue
            sub = sub.copy()
            sub.insert(0, "metric", metric)
            sub.insert(0, "layer", layer)
            layer_rows.append(sub)
        if not layer_rows:
            continue
        layer_df = pd.concat(layer_rows, ignore_index=True)
        # Within-layer BH on the (metric × baseline) family.
        layer_df["q_value"] = benjamini_hochberg_qvalues(
            layer_df["p_value"].to_numpy()
        )
        layer_df["rejected_at_alpha"] = layer_df["q_value"] <= alpha
        pieces.append(layer_df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def behavioral_entropy(action_counts: dict[str, int] | list[str]) -> float:
    """Shannon entropy (base-2) over an action-type distribution.

    Used by the R4 persona-collapse stress test (outline §5.6.7) to detect
    homogenization: as agents converge to a modal action pattern, the entropy
    of the population's action distribution shrinks.

    Args:
        action_counts: Either a mapping {action_type: count} or a flat list
            of action-type strings (which will be counted internally).

    Returns:
        Entropy in bits. 0.0 if no actions.
    """
    from collections import Counter
    from scipy.stats import entropy as scipy_entropy

    if isinstance(action_counts, (list, tuple)):
        counts = np.array(list(Counter(action_counts).values()), dtype=float)
    else:
        counts = np.array(list(action_counts.values()), dtype=float)

    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Drop zeros to avoid 0*log(0) warnings (scipy handles this, but be explicit)
    probs = probs[probs > 0]
    return float(scipy_entropy(probs, base=2))


def drift_slope(time_series: np.ndarray | list[float]) -> float:
    """Ordinary-least-squares slope of a time series.

    Used by the R4 stress test to summarize per-turn drift as a single number
    (β₁ in silhouette(t) = β₀ + β₁·t + ε). Positive = increasing, negative =
    decreasing (collapse signature for silhouette).

    Args:
        time_series: 1D sequence of measurements (e.g. silhouette per turn).

    Returns:
        Per-step slope. 0.0 if fewer than 2 points.
    """
    y = np.asarray(time_series, dtype=float)
    if y.size < 2:
        return 0.0
    t = np.arange(y.size, dtype=float)
    t_mean = t.mean()
    y_mean = y.mean()
    num = float(np.sum((t - t_mean) * (y - y_mean)))
    den = float(np.sum((t - t_mean) ** 2))
    return num / den if den != 0 else 0.0


def fleiss_kappa(annotation_matrix: np.ndarray) -> float:
    """Fleiss' kappa for inter-rater reliability with 3+ annotators.

    Required by outline §5.3.5 (Trigger Calibration Experiment: 3 annotators,
    Fleiss' κ ≥ 0.6) and reusable for any multi-annotator agreement check.

    Args:
        annotation_matrix: Shape (n_items, n_categories). Each row i contains
            the count of annotators who assigned item i to each category.
            Each row must sum to the same value (= number of annotators).

    Returns:
        Fleiss' κ ∈ (-1, 1]. 1.0 = perfect agreement, 0.0 = chance-level,
        negative = systematic disagreement.

    Raises:
        ValueError: if rows have inconsistent sums or matrix is empty.
    """
    M = np.asarray(annotation_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] == 0:
        raise ValueError("annotation_matrix must be a non-empty 2D array")

    n_per_item = M.sum(axis=1)
    if not np.allclose(n_per_item, n_per_item[0]):
        raise ValueError(
            "All rows of annotation_matrix must sum to the same value "
            "(= number of annotators per item)"
        )
    n = float(n_per_item[0])  # annotators per item
    if n < 2:
        raise ValueError("Need at least 2 annotators per item")

    N, K = M.shape

    # P_i: per-item observed agreement
    # P_i = (1 / (n(n-1))) * sum_k m_ik * (m_ik - 1)
    P_i = (M * (M - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = float(P_i.mean())  # mean observed agreement

    # p_k: marginal proportion of category k
    p_k = M.sum(axis=0) / (N * n)

    # P_e: expected agreement by chance
    P_e = float(np.sum(p_k ** 2))

    if P_e == 1.0:
        # All annotators always agree trivially (single category used)
        return 1.0

    return (P_bar - P_e) / (1.0 - P_e)


def compare_ablations(
    df: pd.DataFrame,
    metric: str,
    full: str = "cadp_full",
    ablations: list[str] | None = None,
) -> pd.DataFrame:
    """Compare CADP-Full against each ablation condition on a metric.

    Args:
        df: Results dataframe with 'condition' column.
        metric: Metric column name.
        full: Full CADP condition name.
        ablations: Ablation condition names. Defaults to the three
            dimension-level ablations plus shuffled and constraint-only.

    Returns:
        DataFrame with per-ablation comparison results.
    """
    if ablations is None:
        ablations = [
            "cadp_minus_edna",
            "cadp_minus_mm",
            "cadp_minus_ap",
            "cadp_shuffled",
            "cadp_constraint_only",
            "colleague_skill",
            "clustering_only",
            "pop_aligned_cadp",
        ]

    full_vals = df[df["condition"] == full][metric].dropna().values

    results = []
    for ablation in ablations:
        abl_vals = df[df["condition"] == ablation][metric].dropna().values

        if len(full_vals) == 0 or len(abl_vals) == 0:
            continue

        min_len = min(len(full_vals), len(abl_vals))
        t = full_vals[:min_len]
        c = abl_vals[:min_len]

        test = paired_wilcoxon_test(t, c)
        delta = cliffs_delta(t, c)
        ci_low, ci_high = bootstrap_ci(t - c)
        # P2: parametric effect-size companions.
        d_indep = cohens_d(t, c)
        d_paired = paired_d_with_ci(t, c)

        p = test["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        results.append({
            "ablation": ablation,
            "full_mean": float(np.mean(t)),
            "ablation_mean": float(np.mean(c)),
            "cliffs_delta": delta,
            "cohens_d": d_indep,
            "paired_d": d_paired["paired_d"],
            "paired_d_ci_low": d_paired["ci_low"],
            "paired_d_ci_high": d_paired["ci_high"],
            "p_value": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": p < 0.05,
            "sig_marker": sig,
        })

    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# Outline §5.8 — three-dimensional ↔ metric-layer dissociation test
# ----------------------------------------------------------------------

# Default mapping from individual metric columns (produced by
# MetricsAggregator.evaluate) to the five evaluation layers (outline §5.3).
DEFAULT_LAYER_METRICS: dict[str, list[str]] = {
    "macro": [
        "delta_q_modularity", "ei_polarization_sim", "ei_polarization_real",
        "ned", "coverage",
    ],
    "meso": ["ks_statistic", "structural_fidelity"],
    "micro": [
        "action_matrix_similarity", "rsa",
        "uniformity_sim", "uniformity_real", "uniformity_gap",
        "complexity_sim", "complexity_real",
    ],
    "linguistics": [
        "discourse_marker_match", "sentiment_trajectory_similarity",
        "speech_act_similarity",
    ],
    "predictive": [
        "pred_conflict_accuracy", "pred_conflict_f1",
        "pred_persuasion_accuracy", "pred_persuasion_f1",
        "pred_escalation_accuracy", "pred_escalation_f1",
    ],
}

# Per-metric direction: True = higher is better (fidelity), False = lower
# is better (distance / divergence). Used so "loss" is always positive when
# an ablation hurts fidelity on that metric.
DEFAULT_METRIC_DIRECTION: dict[str, bool] = {
    "delta_q_modularity": False,   # |ΔQ| distance → lower better
    "ei_polarization_sim": True,   # match against real, larger |sim| is informative
    "ei_polarization_real": True,
    "ned": False,                  # JS-distance → lower better
    "coverage": True,
    "ks_statistic": False,         # KS-test stat → lower better (closer dist match)
    "structural_fidelity": True,
    "action_matrix_similarity": True,
    "rsa": True,
    "uniformity_sim": True,
    "uniformity_real": True,
    "uniformity_gap": False,       # |sim-real| → lower better
    "complexity_sim": True,
    "complexity_real": True,
    "discourse_marker_match": True,
    "sentiment_trajectory_similarity": True,
    "speech_act_similarity": True,
}

# Outline §3.1 falsifiable predictions: which layer each ablation should
# hurt most. Verified by AblationDissociationReport.predicted_argmax_match.
PREDICTED_ARGMAX_LAYER: dict[str, str] = {
    "cadp_minus_edna": "linguistics",
    "cadp_minus_mm": "meso",
    "cadp_minus_ap": "macro",   # E-I polarization + conflict escalation
}

THREE_DIM_ABLATIONS: list[str] = [
    "cadp_minus_edna", "cadp_minus_mm", "cadp_minus_ap",
]


def _layer_loss(
    df: pd.DataFrame,
    ablation: str,
    full: str,
    layer: str,
    layer_metrics: list[str],
    direction: dict[str, bool],
) -> float:
    """Mean signed loss of one ablation on one metric layer.

    Loss > 0 means the ablation degraded fidelity on this layer relative to
    CADP Full. Per-metric loss is standardized by the Full condition's std so
    metrics on different scales (KS-stat vs. similarity in [0,1]) contribute
    comparably.
    """
    losses: list[float] = []
    for metric in layer_metrics:
        if metric not in df.columns:
            continue
        full_vals = df[df["condition"] == full][metric].dropna().values
        abl_vals = df[df["condition"] == ablation][metric].dropna().values
        if len(full_vals) == 0 or len(abl_vals) == 0:
            continue
        full_mean = float(np.mean(full_vals))
        abl_mean = float(np.mean(abl_vals))
        # Normalizer: Full's std (guard against zero)
        norm = float(np.std(full_vals)) or 1.0
        higher_better = direction.get(metric, True)
        raw = (full_mean - abl_mean) if higher_better else (abl_mean - full_mean)
        losses.append(raw / norm)
    if not losses:
        return float("nan")
    return float(np.mean(losses))


@dataclass
class AblationDissociationReport:
    """Outline §5.8 — three-dim ↔ metric-layer dissociation verdict.

    Per-layer loss profile (rows = three ablations, cols = five layers)
    plus the 3×3 Pearson correlation matrix. High off-diagonal correlation
    means the three losses load on a single factor → the dimension split
    is *decorative* and the §3.1 "non-arbitrary decomposition" prediction
    is weakened; low correlation (with each ablation peaking on its
    predicted layer) supports it.
    """
    loss_matrix: pd.DataFrame           # ablation × layer
    correlation_matrix: pd.DataFrame    # 3×3 across ablations
    predicted_argmax_layer: dict[str, str]
    observed_argmax_layer: dict[str, str]
    predicted_argmax_match: dict[str, bool]
    mean_off_diagonal_correlation: float
    dissociation_supported: bool        # True iff mean off-diag r below threshold
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_matrix": self.loss_matrix.to_dict(),
            "correlation_matrix": self.correlation_matrix.to_dict(),
            "predicted_argmax_layer": self.predicted_argmax_layer,
            "observed_argmax_layer": self.observed_argmax_layer,
            "predicted_argmax_match": self.predicted_argmax_match,
            "mean_off_diagonal_correlation": self.mean_off_diagonal_correlation,
            "dissociation_supported": self.dissociation_supported,
            "threshold": self.threshold,
        }


def ablation_dissociation_matrix(
    df: pd.DataFrame,
    full: str = "cadp_full",
    ablations: list[str] | None = None,
    layer_metrics: dict[str, list[str]] | None = None,
    metric_direction: dict[str, bool] | None = None,
    correlation_threshold: float = 0.7,
) -> AblationDissociationReport:
    """Outline §5.8 — three-dimensional ↔ metric-layer dissociation test.

    Computes, for each of the three dimension-level ablations (minus
    Expression DNA / minus Mind Models / minus Anti-patterns), the per-layer
    fidelity loss relative to CADP Full. Then correlates the three loss
    *profiles* across layers: low off-diagonal correlation + each ablation
    peaking on its §3.1-predicted layer supports the non-arbitrary claim;
    high correlation / single-factor loading weakens it (reported honestly
    per outline §5.8 / §7.4).

    Args:
        df: Long-form results DataFrame with a ``condition`` column and one
            column per metric (as produced by MetricsAggregator.to_dataframe).
        full: Reference condition (default ``cadp_full``).
        ablations: Three dimension-level ablations. Defaults to
            :data:`THREE_DIM_ABLATIONS`. Extra ablations (shuffled,
            constraint-only) are intentionally excluded — §5.8's
            dissociation claim is specifically about the three CADP
            dimensions, not the auxiliary structural controls.
        layer_metrics: Override the metric→layer mapping. Defaults to
            :data:`DEFAULT_LAYER_METRICS`.
        metric_direction: Override per-metric direction (True = higher
            better). Defaults to :data:`DEFAULT_METRIC_DIRECTION`.
        correlation_threshold: Off-diagonal correlation above this flags
            the decomposition as decorative. Default 0.7.

    Returns:
        :class:`AblationDissociationReport`.
    """
    ablations = list(ablations) if ablations is not None else list(THREE_DIM_ABLATIONS)
    layer_metrics = layer_metrics or DEFAULT_LAYER_METRICS
    metric_direction = metric_direction or DEFAULT_METRIC_DIRECTION
    layers = list(layer_metrics.keys())

    # Build loss matrix (ablation × layer)
    rows: dict[str, dict[str, float]] = {}
    for abl in ablations:
        rows[abl] = {
            layer: _layer_loss(df, abl, full, layer, layer_metrics[layer], metric_direction)
            for layer in layers
        }
    loss_matrix = pd.DataFrame(rows).T  # ablations as rows, layers as cols

    # Drop any ablation/layer row that came back all-NaN
    loss_matrix = loss_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if loss_matrix.shape[0] < 2 or loss_matrix.shape[1] < 2:
        raise ValueError(
            "Cannot compute dissociation matrix — need at least 2 ablations "
            "and 2 metric layers with non-NaN losses. Check that df contains "
            "rows for the three cadp_minus_* conditions and metrics from "
            "DEFAULT_LAYER_METRICS."
        )

    # 3×3 correlation across the three ablations' loss profiles (Pearson
    # over the per-layer loss vector). Used to detect single-factor loading.
    correlation_matrix = loss_matrix.T.corr(method="pearson")

    # Off-diagonal mean: high → decorative decomposition
    off_diag = correlation_matrix.where(~np.eye(len(correlation_matrix), dtype=bool))
    mean_off_diag = float(off_diag.stack().mean())

    # Predicted vs observed argmax layer per ablation
    predicted = {a: PREDICTED_ARGMAX_LAYER.get(a, "?") for a in loss_matrix.index}
    observed = loss_matrix.idxmax(axis=1).to_dict()
    match = {
        a: (predicted[a] == observed[a]) for a in loss_matrix.index
        if predicted[a] != "?"
    }

    dissociation_supported = (
        mean_off_diag < correlation_threshold
        and all(match.values())
    )

    return AblationDissociationReport(
        loss_matrix=loss_matrix,
        correlation_matrix=correlation_matrix,
        predicted_argmax_layer=predicted,
        observed_argmax_layer=observed,
        predicted_argmax_match=match,
        mean_off_diagonal_correlation=mean_off_diag,
        dissociation_supported=dissociation_supported,
        threshold=correlation_threshold,
    )
