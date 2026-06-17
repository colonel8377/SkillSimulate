"""R4: Persona-Collapse Stress Test (outline §5.6.7).

Implements the longitudinal stress test specified in
``docs/r4_persona_collapse_stress_test.md``. The test runs simulations for
50+ turns and measures per-turn cluster drift to detect the persona-collapse
signature diagnosed by *The Chameleon's Limit* (arXiv:2604.24698).

Hypotheses (from protocol §2):

| # | Hypothesis | Primary metric |
|---|------------|----------------|
| H1 | Descriptive-persona agents exhibit measurable collapse within 50 turns | silhouette drift slope β₁ ≤ -Δ_collapse/50 |
| H2 | CADP agents maintain cluster separation (no monotonic collapse) | |silhouette β₁| < Δ_collapse/2/50 |
| H3 | Behavioral entropy of descriptive-persona populations shrinks | entropy(50) < entropy(5) |
| H4 | CADP populations retain baseline entropy | |entropy(50) - entropy(5)| < Δ_entropy/2 |
| H5 | Inter-agent persona cosine rises for descriptive-persona | cosine(50) > cosine(5) + Δ_cos |
| H6 | CADP-minus-Anti-patterns reproduces collapse signature | drift slope ≈ Descriptive |

Default thresholds (override via config):
    Δ_collapse = 0.10    (cumulative silhouette drop over 50 turns)
    Δ_entropy  = 0.30    (cumulative entropy drop in bits)
    Δ_cos      = 0.05    (cumulative cosine rise)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.stats import behavioral_entropy, drift_slope
from src.clustering.clusterer import (
    davies_bouldin_at_state,
    refit_labels_frozen_k,
    silhouette_at_state,
)
from src.clustering.embeddings import mean_pairwise_cosine, rolling_agent_state


# --- Defaults from protocol §3 ------------------------------------------------

STRESS_METHODS: tuple[str, ...] = (
    "descriptive",        # H1, H3, H5 positive collapse arm
    "cadp_full",          # H2, H4 — should resist collapse
    "cadp_minus_ap",      # H6 — mechanism isolation (Tier 3 removed)
)
STRESS_MODELS: tuple[str, ...] = (
    "gpt-4o", "claude-3.5-sonnet", "llama-3-70b", "qwen-2.5-72b",
)
STRESS_DATASETS: tuple[str, ...] = ("wikipedia", "reddit", "github")

DEFAULT_NUM_TURNS = 50
DEFAULT_NUM_REPEATS = 5
DEFAULT_POPULATION = 30
DEFAULT_WINDOW = 5  # rolling action window (protocol §4.3)
DEFAULT_MEASUREMENT_POINTS: tuple[int, ...] = (
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
)

# Verdict thresholds (protocol §2)
DEFAULT_DELTA_COLLAPSE = 0.10
DEFAULT_DELTA_ENTROPY = 0.30
DEFAULT_DELTA_COS = 0.05


# --- Result dataclasses ------------------------------------------------------


@dataclass
class CollapseTurnMeasurement:
    """Single per-turn measurement point."""
    turn: int
    silhouette: float
    davies_bouldin: float
    behavioral_entropy: float
    mean_pairwise_cosine: float
    cosine_ci_halfwidth: float
    n_agents_active: int


@dataclass
class CollapseRunResult:
    """Result of one simulation run under the R4 protocol."""
    method: str
    model: str
    dataset: str
    repeat: int
    K_frozen: int
    measurements: list[CollapseTurnMeasurement] = field(default_factory=list)

    # Derived slopes (set by ``finalize()``)
    silhouette_slope: float = 0.0
    entropy_slope: float = 0.0
    cosine_slope: float = 0.0
    silhouette_drop_5_to_50: float = 0.0
    entropy_drop_5_to_50: float = 0.0
    cosine_rise_5_to_50: float = 0.0

    def finalize(self) -> None:
        """Compute drift slopes and cumulative deltas from the measurement series."""
        if not self.measurements:
            return
        sil_series = np.array([m.silhouette for m in self.measurements])
        ent_series = np.array([m.behavioral_entropy for m in self.measurements])
        cos_series = np.array([m.mean_pairwise_cosine for m in self.measurements])

        self.silhouette_slope = drift_slope(sil_series)
        self.entropy_slope = drift_slope(ent_series)
        self.cosine_slope = drift_slope(cos_series)

        # Index 1 in DEFAULT_MEASUREMENT_POINTS is turn 5; index -1 is turn 50
        self.silhouette_drop_5_to_50 = float(sil_series[0] - sil_series[-1])
        self.entropy_drop_5_to_50 = float(ent_series[0] - ent_series[-1])
        self.cosine_rise_5_to_50 = float(cos_series[-1] - cos_series[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "repeat": self.repeat,
            "K_frozen": self.K_frozen,
            "silhouette_slope": self.silhouette_slope,
            "entropy_slope": self.entropy_slope,
            "cosine_slope": self.cosine_slope,
            "silhouette_drop_5_to_50": self.silhouette_drop_5_to_50,
            "entropy_drop_5_to_50": self.entropy_drop_5_to_50,
            "cosine_rise_5_to_50": self.cosine_rise_5_to_50,
            "measurements": [
                {
                    "turn": m.turn,
                    "silhouette": m.silhouette,
                    "davies_bouldin": m.davies_bouldin,
                    "behavioral_entropy": m.behavioral_entropy,
                    "mean_pairwise_cosine": m.mean_pairwise_cosine,
                    "cosine_ci_halfwidth": m.cosine_ci_halfwidth,
                    "n_agents_active": m.n_agents_active,
                }
                for m in self.measurements
            ],
        }


# --- Core analysis (pure function) ------------------------------------------


def compute_collapse_trajectory(
    sim_messages: list[dict],
    agent_ids: list[str],
    K_frozen: int,
    embedder,
    measurement_points: tuple[int, ...] = DEFAULT_MEASUREMENT_POINTS,
    window: int = DEFAULT_WINDOW,
) -> list[CollapseTurnMeasurement]:
    """Compute per-turn collapse metrics from a simulation message log.

    This is a pure function over the message log: it does NOT re-run the
    simulation. Use it either post-hoc on a completed ``SimulationResult``
    or wrap it in ``run_collapse_stress_cell`` for a fresh run.

    Args:
        sim_messages: List of message dicts from ``SimulationResult.messages``.
            Each must have ``metadata.round`` and ``text`` and ``user_id``.
        agent_ids: All agent IDs in the population (fixed across turns).
        K_frozen: Cluster count frozen from §5.2 Step 1. Re-clustering at
            each measurement point uses this exact K (protocol §4.2).
        embedder: Object exposing ``.encode(list[str]) -> np.ndarray`` (e.g.
            a SentenceTransformer or ``EmbeddingExtractor.model``).
        measurement_points: Turns at which to record a measurement.
        window: Rolling action window (protocol §4.3).

    Returns:
        One ``CollapseTurnMeasurement`` per entry in ``measurement_points``.
    """
    # Index messages by round (1-indexed in the simulation)
    by_round: dict[int, list[dict]] = {}
    for msg in sim_messages:
        md = msg.get("metadata") or {}
        rnd = md.get("round")
        if rnd is None:
            continue
        by_round.setdefault(int(rnd), []).append(msg)

    max_round = max(by_round.keys()) if by_round else 0
    max_measurement = max(measurement_points)
    if max_measurement > max_round:
        logger.warning(
            f"Requested measurement at turn {max_measurement} but simulation "
            f"only ran {max_round} turns — trajectory will be truncated"
        )

    # Cache action embeddings per message (turn-anchored)
    # We build a chronological per-agent action list as turns accrue.
    agent_actions: dict[str, list[dict]] = {aid: [] for aid in agent_ids}
    measurements: list[CollapseTurnMeasurement] = []

    sorted_turns = sorted(by_round.keys())
    for rnd in sorted_turns:
        for msg in by_round[rnd]:
            uid = msg.get("user_id")
            if uid not in agent_actions:
                agent_actions[uid] = []
            agent_actions[uid].append({
                "turn": rnd,
                "text": msg.get("text", ""),
                "action_type": (msg.get("action_type") or "").lower(),
            })

        if rnd not in measurement_points:
            continue

        # Embed each agent's actions up to this turn, then rolling-aggregate
        per_agent_state: dict[str, np.ndarray] = {}
        per_agent_actions: dict[str, list[str]] = {}

        # Batch-encode all eligible texts for efficiency
        to_encode: list[str] = []
        owner_index: list[str] = []  # which agent each row belongs to
        for aid in agent_ids:
            actions = agent_actions.get(aid, [])
            # Keep only actions with non-trivial text (per EmbeddingExtractor.min_text_length=10)
            eligible = [a for a in actions if len(a["text"].strip()) >= 10]
            per_agent_actions[aid] = [a["action_type"] for a in actions]
            if not eligible:
                continue
            for a in eligible:
                to_encode.append(a["text"])
                owner_index.append(aid)

        if to_encode:
            action_embs = embedder.encode(to_encode, show_progress_bar=False)
        else:
            action_embs = np.zeros((0, embedder.get_sentence_embedding_dimension()))

        # Group action embeddings back per agent (chronological)
        per_agent_action_embs: dict[str, list[np.ndarray]] = {aid: [] for aid in agent_ids}
        for vec, owner in zip(action_embs, owner_index):
            per_agent_action_embs[owner].append(vec)

        dim = embedder.get_sentence_embedding_dimension()
        for aid in agent_ids:
            embs = per_agent_action_embs.get(aid, [])
            if not embs:
                per_agent_state[aid] = np.zeros(dim)
            else:
                per_agent_state[aid] = rolling_agent_state(embs, window=window, embed_dim=dim)

        # Stack active agents (those with at least one action up to this turn)
        active = [aid for aid in agent_ids if per_agent_action_embs.get(aid)]
        if len(active) < 2:
            measurements.append(CollapseTurnMeasurement(
                turn=rnd,
                silhouette=0.0,
                davies_bouldin=0.0,
                behavioral_entropy=0.0,
                mean_pairwise_cosine=0.0,
                cosine_ci_halfwidth=0.0,
                n_agents_active=len(active),
            ))
            continue

        state_matrix = np.stack([per_agent_state[aid] for aid in active])
        labels = refit_labels_frozen_k(state_matrix, K_frozen)

        sil = silhouette_at_state(state_matrix, labels, K_frozen=K_frozen)
        db = davies_bouldin_at_state(state_matrix, labels)

        # Behavioral entropy across the full population's action-type distribution
        all_actions: list[str] = []
        for aid in agent_ids:
            all_actions.extend(per_agent_actions.get(aid, []))
        ent = behavioral_entropy(all_actions) if all_actions else 0.0

        mean_cos, ci_half = mean_pairwise_cosine(
            {aid: per_agent_state[aid] for aid in active}
        )

        measurements.append(CollapseTurnMeasurement(
            turn=rnd,
            silhouette=sil,
            davies_bouldin=db,
            behavioral_entropy=ent,
            mean_pairwise_cosine=mean_cos,
            cosine_ci_halfwidth=ci_half,
            n_agents_active=len(active),
        ))

    return measurements


# --- Verdict evaluation (protocol §5) ---------------------------------------


@dataclass
class HypothesisVerdict:
    hypothesis: str
    supported: bool | None  # None = inconclusive
    evidence: str


@dataclass
class CollapseStressReport:
    """Aggregated report across all cells of the R4 stress test."""
    results: list[CollapseRunResult]
    verdicts: dict[str, HypothesisVerdict] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            rows.append({
                "method": r.method,
                "model": r.model,
                "dataset": r.dataset,
                "repeat": r.repeat,
                "K_frozen": r.K_frozen,
                "silhouette_slope": r.silhouette_slope,
                "entropy_slope": r.entropy_slope,
                "cosine_slope": r.cosine_slope,
                "silhouette_drop_5_to_50": r.silhouette_drop_5_to_50,
                "entropy_drop_5_to_50": r.entropy_drop_5_to_50,
                "cosine_rise_5_to_50": r.cosine_rise_5_to_50,
            })
        return pd.DataFrame(rows)


def evaluate_hypotheses(
    results: list[CollapseRunResult],
    delta_collapse: float = DEFAULT_DELTA_COLLAPSE,
    delta_entropy: float = DEFAULT_DELTA_ENTROPY,
    delta_cos: float = DEFAULT_DELTA_COS,
) -> dict[str, HypothesisVerdict]:
    """Evaluate H1–H6 from a collection of CollapseRunResults.

    Each verdict aggregates across models/datasets/repeats by mean drift
    slope. Use ``mixed_effects_analysis`` (deferred to stats module) for
    the rigorous per-cell statistical test specified in protocol §5.1.
    """
    by_method: dict[str, list[CollapseRunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    def mean_slope(method: str, attr: str) -> float | None:
        rs = by_method.get(method, [])
        if not rs:
            return None
        return float(np.mean([getattr(r, attr) for r in rs]))

    def mean_drop(method: str, attr: str) -> float | None:
        return mean_slope(method, attr)

    verdicts: dict[str, HypothesisVerdict] = {}

    desc_sil_slope = mean_slope("descriptive", "silhouette_slope")
    desc_sil_drop = mean_drop("descriptive", "silhouette_drop_5_to_50")

    # H1: Descriptive silhouette drops by ≥ Δ_collapse over 50 turns
    if desc_sil_drop is None:
        verdicts["H1"] = HypothesisVerdict("H1", None, "No descriptive-arm runs")
    else:
        h1_supported = desc_sil_drop >= delta_collapse
        verdicts["H1"] = HypothesisVerdict(
            "H1", h1_supported,
            f"Descriptive silhouette drop 5→50: {desc_sil_drop:.4f} "
            f"(threshold {delta_collapse:.2f})"
        )

    # H2: CADP silhouette slope magnitude < Δ_collapse/2/50
    cadp_sil_slope = mean_slope("cadp_full", "silhouette_slope")
    if cadp_sil_slope is None or desc_sil_slope is None:
        verdicts["H2"] = HypothesisVerdict("H2", None, "Missing CADP or Descriptive arm")
    else:
        cadp_resists = abs(cadp_sil_slope) < abs(desc_sil_slope) / 2
        verdicts["H2"] = HypothesisVerdict(
            "H2", bool(cadp_resists),
            f"|CADP slope| {abs(cadp_sil_slope):.5f} vs |Desc slope|/2 "
            f"{abs(desc_sil_slope)/2:.5f}"
        )

    # H3: Descriptive entropy drops
    desc_ent_drop = mean_drop("descriptive", "entropy_drop_5_to_50")
    if desc_ent_drop is None:
        verdicts["H3"] = HypothesisVerdict("H3", None, "No descriptive-arm runs")
    else:
        verdicts["H3"] = HypothesisVerdict(
            "H3", desc_ent_drop > 0,
            f"Descriptive entropy drop 5→50: {desc_ent_drop:.4f} bits"
        )

    # H4: CADP entropy retained
    cadp_ent_drop = mean_drop("cadp_full", "entropy_drop_5_to_50")
    if cadp_ent_drop is None:
        verdicts["H4"] = HypothesisVerdict("H4", None, "Missing CADP arm")
    else:
        verdicts["H4"] = HypothesisVerdict(
            "H4", abs(cadp_ent_drop) < delta_entropy / 2,
            f"|CADP entropy drop| {abs(cadp_ent_drop):.4f} "
            f"vs Δ_entropy/2 {delta_entropy/2:.2f}"
        )

    # H5: Descriptive cosine rises by ≥ Δ_cos
    desc_cos_rise = mean_drop("descriptive", "cosine_rise_5_to_50")
    if desc_cos_rise is None:
        verdicts["H5"] = HypothesisVerdict("H5", None, "No descriptive-arm runs")
    else:
        verdicts["H5"] = HypothesisVerdict(
            "H5", desc_cos_rise > delta_cos,
            f"Descriptive cosine rise 5→50: {desc_cos_rise:.4f} "
            f"(threshold {delta_cos:.2f})"
        )

    # H6: CADP-minus-Anti-patterns reproduces collapse signature
    minus_ap_sil_slope = mean_slope("cadp_minus_ap", "silhouette_slope")
    if minus_ap_sil_slope is None or cadp_sil_slope is None:
        verdicts["H6"] = HypothesisVerdict("H6", None, "Missing mechanism-isolation arm")
    else:
        # H6 supported if |minus_ap slope| > |CADP slope| AND closer to descriptive
        h6 = (abs(minus_ap_sil_slope) > abs(cadp_sil_slope)) and (
            abs(minus_ap_sil_slope - (desc_sil_slope or 0))
            < abs(cadp_sil_slope - (desc_sil_slope or 0))
        )
        verdicts["H6"] = HypothesisVerdict(
            "H6", bool(h6),
            f"|minus_ap slope| {abs(minus_ap_sil_slope):.5f} vs "
            f"|CADP slope| {abs(cadp_sil_slope):.5f}"
        )

    return verdicts


# --- Orchestrator -----------------------------------------------------------
#
# The orchestrator is intentionally thin: it composes the existing Sandbox
# + PopulationBuilder to run each cell, then calls compute_collapse_trajectory
# on the resulting message log. The cell-setup logic mirrors
# src/experiment/exp1_validation.py but is reduced to the R4 3-arm factorial
# (no ablations beyond cadp_minus_ap, no scale test).


async def run_collapse_stress_cell(
    method: str,
    model: str,
    dataset: str,
    repeat: int,
    *,
    num_turns: int = DEFAULT_NUM_TURNS,
    population_size: int = DEFAULT_POPULATION,
    K_frozen: int = 4,
    alpha: float = 1.0,
    embedder=None,
    cell_runner=None,
) -> CollapseRunResult:
    """Run one R4 stress-test cell.

    Args:
        method: One of STRESS_METHODS.
        model: Model name.
        dataset: Dataset name.
        repeat: Repeat index (0-based).
        num_turns: Number of turns (protocol default 50).
        population_size: Number of agents (protocol default 30).
        K_frozen: Frozen cluster count from §5.2 Step 1.
        alpha: Enforcement α for CADP arms.
        embedder: Shared SentenceTransformer for trajectory analysis. If
            None, ``get_shared_embedder()`` is loaded lazily.
        cell_runner: Optional callable that runs a single simulation cell
            and returns ``SimulationResult``. Signature::
                cell_runner(method, model, dataset, repeat, num_turns,
                            population_size, alpha) -> SimulationResult
            If None, the function raises — callers must wire a runner
            matching their environment (the harness provides one in
            ``exp1_validation.py`` via ``_run_cell``).

    Returns:
        ``CollapseRunResult`` with all per-turn measurements and finalized
        drift slopes.
    """
    if cell_runner is None:
        raise RuntimeError(
            "run_collapse_stress_cell requires a cell_runner — wire one "
            "from your existing experiment harness (e.g. exp1_validation._run_cell)"
        )
    if embedder is None:
        from src.config.settings import get_shared_embedder
        embedder = get_shared_embedder()

    sim_result = await cell_runner(
        method=method,
        model=model,
        dataset=dataset,
        repeat=repeat,
        num_turns=num_turns,
        population_size=population_size,
        alpha=alpha,
    )

    # Recover agent IDs from agent_states (PopulationBuilder emits these)
    agent_ids = [s["agent_id"] for s in sim_result.agent_states]
    if not agent_ids:
        # Fall back to extracting from message log
        agent_ids = sorted({m["user_id"] for m in sim_result.messages})

    measurements = compute_collapse_trajectory(
        sim_messages=sim_result.messages,
        agent_ids=agent_ids,
        K_frozen=K_frozen,
        embedder=embedder,
        measurement_points=tuple(
            t for t in DEFAULT_MEASUREMENT_POINTS if t <= num_turns
        ),
    )

    result = CollapseRunResult(
        method=method, model=model, dataset=dataset, repeat=repeat,
        K_frozen=K_frozen, measurements=measurements,
    )
    result.finalize()
    return result


async def run_collapse_stress(
    *,
    methods: tuple[str, ...] = STRESS_METHODS,
    models: tuple[str, ...] = STRESS_MODELS,
    datasets: tuple[str, ...] = STRESS_DATASETS,
    num_repeats: int = DEFAULT_NUM_REPEATS,
    num_turns: int = DEFAULT_NUM_TURNS,
    population_size: int = DEFAULT_POPULATION,
    K_frozen: int = 4,
    alpha: float = 1.0,
    cell_runner=None,
    delta_collapse: float = DEFAULT_DELTA_COLLAPSE,
    delta_entropy: float = DEFAULT_DELTA_ENTROPY,
    delta_cos: float = DEFAULT_DELTA_COS,
) -> CollapseStressReport:
    """Run the full R4 stress-test grid (3 × 4 × 3 × 5 = 180 cells).

    See module docstring + docs/r4_persona_collapse_stress_test.md §3
    for the factorial design rationale. Reduced from the §5.1 grid because
    each cell yields a paired time-series, giving high power per repeat.

    Args:
        cell_runner: See ``run_collapse_stress_cell``.

    Returns:
        ``CollapseStressReport`` with per-cell results and H1–H6 verdicts.
    """
    all_results: list[CollapseRunResult] = []
    total_cells = len(methods) * len(models) * len(datasets) * num_repeats
    done = 0

    for method in methods:
        for model in models:
            for dataset in datasets:
                for repeat in range(num_repeats):
                    try:
                        result = await run_collapse_stress_cell(
                            method=method,
                            model=model,
                            dataset=dataset,
                            repeat=repeat,
                            num_turns=num_turns,
                            population_size=population_size,
                            K_frozen=K_frozen,
                            alpha=alpha,
                            cell_runner=cell_runner,
                        )
                        all_results.append(result)
                    except Exception as exc:  # noqa: BLE001 — log + continue
                        logger.exception(
                            f"R4 cell failed: {method}/{model}/{dataset}/repeat={repeat}: {exc}"
                        )
                    done += 1
                    if done % 10 == 0:
                        logger.info(f"R4 progress: {done}/{total_cells} cells done")

    verdicts = evaluate_hypotheses(
        all_results,
        delta_collapse=delta_collapse,
        delta_entropy=delta_entropy,
        delta_cos=delta_cos,
    )
    return CollapseStressReport(results=all_results, verdicts=verdicts)


def run_collapse_stress_sync(
    **kwargs: Any,
) -> CollapseStressReport:
    """Synchronous entry point — wraps ``run_collapse_stress`` in asyncio.run."""
    return asyncio.run(run_collapse_stress(**kwargs))
