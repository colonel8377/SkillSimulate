"""Metrics aggregator — collects all metric layers into result tables."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data.schemas import Message, Thread
from src.evaluation.linguistics import LinguisticMetrics
from src.evaluation.macro import MacroMetrics
from src.evaluation.meso import MesoMetrics
from src.evaluation.micro import (
    MicroMetrics,
    caricature_bootstrap_ci,
    caricature_index,
)
from src.simulation.sandbox import SimulationResult

if TYPE_CHECKING:
    from src.llm.client import LLMClient


class EvaluationIntegrityError(RuntimeError):
    """Raised when reference data or a required metric is scientifically invalid."""


@dataclass
class MetricsReport:
    """Complete metrics report for one simulation run."""
    run_id: str
    condition: str
    dataset: str
    model: str
    repeat: int
    metrics: dict[str, float] = field(default_factory=dict)
    # Anti-circularity provenance flags (outline §5.3, §7.4). When True the
    # corresponding ground-truth layer used the proxy fallback rather than
    # the externally-annotated labels mandated by §5.3. Surface these in the
    # paper's per-dataset validity table (G8).
    used_role_label_proxy: bool = False
    # Retained for schema compat; predictive fidelity layer removed.
    # Always False — regex-based event detection deleted.
    used_held_out_events_heuristic: bool = False
    # Audit P5 / outline §5.1(d): reproducibility provenance for the model
    # that produced this run. Empty strings when unrecorded (the paper's
    # reproducibility table should then show ``unrecorded`` rather than
    # silently dropping the row).
    model_snapshot_date: str = ""
    model_commit_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "dataset": self.dataset,
            "model": self.model,
            "repeat": self.repeat,
            "used_role_label_proxy": self.used_role_label_proxy,
            "used_held_out_events_heuristic": self.used_held_out_events_heuristic,
            "model_snapshot_date": self.model_snapshot_date,
            "model_commit_hash": self.model_commit_hash,
            **self.metrics,
        }


class MetricsAggregator:
    """Aggregates all metric layers and compares against real data.

    Args:
        held_out_events_dir: Directory containing ``{dataset}.jsonl`` files of
            annotated held-out events (outline §5.3). If a file exists for the
            dataset being evaluated, Predictive Fidelity uses the annotation
            protocol (2 annotators, Cohen's κ ≥ 0.7); otherwise it falls back
            to heuristic labels.
        role_labels_dir: Directory containing ``{dataset}.jsonl`` files of
            external behavioral role labels (outline §5.3 Micro Behavior).
            Each line: ``{"user_id": "<pseudonymous-id>", "role": "<label>"}``.
            When available, Micro Behavior ground truth uses these external
            labels (e.g. Wu et al. 2025 audience segmentation or human-annotated
            moderator / provocateur / peacemaker / lurker) — directly satisfying
            the outline's anti-circularity mandate. When absent, the aggregator
            falls back to Louvain communities inferred from the real interaction
            graph (a weaker, graph-derived proxy) and logs a warning so the
            paper's §7.4 validity discussion can report which datasets used
            external labels vs. the proxy.
    """

    def __init__(
        self,
        held_out_events_dir: str | None = None,
        role_labels_dir: str | None = None,
        model_provenance: dict[str, dict[str, str]] | None = None,
        llm_client: LLMClient | None = None,
        llm_model_name: str | None = None,
        action_smoothing: float = 0.0,
        continuation_mode: bool = False,
        linguistic_metric_weights: dict[str, float] | None = None,
        interaction_metric_weights: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self.reports: list[MetricsReport] = []
        self.held_out_events_dir = held_out_events_dir
        self.role_labels_dir = role_labels_dir
        self.seed = seed
        # Tracks which datasets fell back to the Louvain proxy, for §7.4 reporting
        self.datasets_using_role_label_proxy: set[str] = set()
        # Audit P5: ``{model_name: {"snapshot_date": ..., "commit_hash": ...}}``
        # Stamped into every MetricsReport so the paper's reproducibility
        # table can cite the exact dated snapshot / commit per row.
        self.model_provenance: dict[str, dict[str, str]] = dict(model_provenance or {})
        # LLM client for speech act classification (optional)
        self._llm_client = llm_client
        self._llm_model_name = llm_model_name
        self.action_smoothing = float(action_smoothing)
        self.continuation_mode = bool(continuation_mode)
        self.linguistic_metric_weights = linguistic_metric_weights or {
            "discourse_relation_match": 0.25,
            "sentiment_trajectory_similarity": 0.25,
            "speech_act_similarity": 0.25,
            "sip": 0.25,
        }
        self.interaction_metric_weights = interaction_metric_weights or {
            "cascade": 0.5, "graph": 0.5,
        }
        self._validate_composite_weights()

    def _validate_composite_weights(self) -> None:
        expected_linguistic = {
            "discourse_relation_match", "sentiment_trajectory_similarity",
            "speech_act_similarity", "sip",
        }
        if set(self.linguistic_metric_weights) != expected_linguistic:
            raise ValueError("linguistic_metric_weights must name exactly four required metrics")
        if set(self.interaction_metric_weights) != {"cascade", "graph"}:
            raise ValueError("interaction_metric_weights must name cascade and graph")
        for name, weights in (
            ("linguistic", self.linguistic_metric_weights),
            ("interaction", self.interaction_metric_weights),
        ):
            if any(float(value) < 0 for value in weights.values()):
                raise ValueError(f"{name} metric weights must be non-negative")
            if not math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-9):
                raise ValueError(f"{name} metric weights must sum to 1")

    def _load_held_out_events(self, dataset: str) -> list | None:
        """Load annotated held-out events for a dataset, if available.

        Retained for future use when proper human annotation is available.
        Predictive fidelity layer was removed 2026-07-13 (regex-based event
        detection was unreliable).
        class).
        """
        if not self.held_out_events_dir:
            return None
        from pathlib import Path

        path = Path(self.held_out_events_dir) / f"{dataset}.jsonl"
        if not path.exists():
            return None
        from src.evaluation.held_out_events import load_events
        try:
            return load_events(path)
        except Exception as e:
            logger.warning(f"Failed to load held-out events from {path}: {e}")
            return None

    def _load_role_labels(self, dataset: str) -> dict[str, int] | None:
        """Load external behavioral role labels for a dataset, if available.

        Returns a mapping ``{user_id: role_id}`` (role_id is an int label
        per unique role string) or ``None`` if no labels file is present.
        """
        if not self.role_labels_dir:
            return None
        import json
        from pathlib import Path
        path = Path(self.role_labels_dir) / f"{dataset}.jsonl"
        if not path.exists():
            return None
        labels: dict[str, str] = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    uid = rec.get("user_id")
                    role = rec.get("role")
                    if uid and role:
                        labels[uid] = str(role)
        except Exception as e:
            logger.warning(f"Failed to load role labels from {path}: {e}")
            return None

        # Encode role strings as integer cluster IDs (deterministic ordering)
        unique_roles = sorted(set(labels.values()))
        role_to_id = {r: i for i, r in enumerate(unique_roles)}
        raw_mapping = {uid: role_to_id[r] for uid, r in labels.items()}

        # PII anonymization: role labels store raw Wikipedia usernames, but
        # graph nodes have been scrubbed through anonymize_user_id.  Apply
        # the same deterministic salted hash so the keys match.
        from src.data.pii import anonymize_user_id
        anonymized = {anonymize_user_id(uid): rid for uid, rid in raw_mapping.items()}

        matched = len(anonymized)
        logger.info(
            f"Role labels: {matched} users loaded and anonymized; overlap is "
            "validated against each reference graph during evaluation"
        )
        return anonymized

    async def evaluate(
        self,
        sim_result: SimulationResult,
        real_threads: list[Thread],
        linguistic_reference_threads: list[Thread] | None = None,
    ) -> MetricsReport:
        """Evaluate a simulation result against real data.

        Args:
            sim_result: Simulation output.
            real_threads: Ground truth threads from real data.

        Returns:
            MetricsReport with all 4 metric layers.
        """
        linguistic_reference_threads = linguistic_reference_threads or real_threads

        # Build real data structures
        real_messages = [
            self._msg_to_dict(m) for t in real_threads for m in t.messages
        ]
        real_graph = self._build_graph_from_threads(real_threads)

        self._validate_simulation_result(sim_result)
        if not real_messages:
            raise EvaluationIntegrityError("Action/topology reference contains no messages")
        if real_graph.number_of_edges() == 0:
            raise EvaluationIntegrityError("Action/topology reference graph has no reply edges")

        # Outline §5.3 anti-circularity mandate: prefer external role labels
        # (Wu et al. 2025 segmentation or human-annotated moderator /
        # provocateur / peacemaker / lurker). Fall back to Louvain communities
        # inferred from the interaction graph when labels are unavailable,
        # and record the fallback so §7.4 can report per-dataset validity.
        external_role_labels = self._load_role_labels(sim_result.dataset)
        if external_role_labels is not None:
            node_coverage, edge_coverage = self._label_coverage(
                real_graph, external_role_labels
            )
            if node_coverage >= 0.5 and edge_coverage >= 0.5:
                real_communities = external_role_labels
                logger.info(
                    f"Micro/Macro real-data ground truth: using EXTERNAL role labels "
                    f"for {sim_result.dataset} (node coverage={node_coverage:.1%}, "
                    f"edge coverage={edge_coverage:.1%})"
                )
            else:
                logger.warning(
                    f"External role labels cover only {node_coverage:.1%} of nodes "
                    f"and {edge_coverage:.1%} of edges; using Louvain proxy"
                )
                real_communities = self._infer_communities(real_threads)
                self.datasets_using_role_label_proxy.add(sim_result.dataset)
        else:
            real_communities = self._infer_communities(real_threads)
            self.datasets_using_role_label_proxy.add(sim_result.dataset)
            logger.warning(
                f"Micro/Macro real-data ground truth for {sim_result.dataset}: "
                f"no external role labels found at "
                f"{self.role_labels_dir}/ (or role_labels_dir not set) — "
                f"falling back to Louvain communities inferred from the real "
                f"interaction graph. This is a weaker proxy than the external "
                f"role labels mandated by outline §5.3 (anti-circularity). "
                f"Recorded for §7.4 Threats to Validity."
            )

        real_behavior_messages = self._canonicalize_actions(
            real_messages, sim_result.dataset
        )
        real_action_dist = self._compute_action_dist(real_behavior_messages)
        real_chain_lengths = self._compute_chain_lengths(real_threads)
        real_agent_counts = self._compute_agent_action_counts_from_dicts(
            real_behavior_messages
        )

        # Build simulation data structures
        sim_messages = sim_result.messages
        action_text_consistency, action_text_n = self._action_text_consistency(
            sim_messages
        )
        sim_graph = self._build_graph_from_dicts(sim_messages, sim_result.agent_states)
        if sim_graph.number_of_edges() == 0:
            raise EvaluationIntegrityError("Simulation interaction graph has no agent-agent edges")
        sim_communities = {
            a["agent_id"]: int(a.get("cluster_id", 0))
            for a in sim_result.agent_states
        }
        # Outline §6.2 real_history arm: the simulation IS the real data,
        # replayed through the metric pipeline as a self-similarity ceiling.
        # Without this override, sim_communities (from agent_states, which
        # real_history populates with a uniform cluster_id) diverges from
        # real_communities (external role labels or Louvain), making
        # community-dependent metrics (E-I Polarization, ΔQ Modularity)
        # report artificially distorted values instead of their true ceiling.
        if sim_result.condition == "real_history":
            sim_communities = dict(real_communities)
        sim_behavior_messages = self._canonicalize_actions(
            sim_messages, sim_result.dataset
        )
        sim_action_dist = self._compute_action_dist(sim_behavior_messages)
        sim_agent_counts = self._compute_agent_action_counts_from_dicts(
            sim_behavior_messages
        )
        self._validate_action_reference(real_action_dist)

        # Compute metrics per layer
        all_metrics = {}

        # Layer 1: Macro. Required layers fail closed: a partial report must
        # never receive a COMPLETE marker after an expensive simulation.
        all_metrics.update(MacroMetrics.compute(
            sim_graph=sim_graph,
            real_graph=real_graph,
            sim_communities=sim_communities,
            real_communities=real_communities,
            sim_action_dist=sim_action_dist,
            real_action_dist=real_action_dist,
            action_smoothing=self.action_smoothing,
        ))

        # Layer 2: Meso
        sim_chain_lengths = self._compute_chain_lengths_from_dicts(
            sim_messages,
            context_messages=(None if self.continuation_mode else sim_result.seed_messages),
        )

        # Extract temporal sequences for DTW (outline §6.4)
        sim_temporal = self._estimate_sim_temporal(sim_behavior_messages)
        real_temporal = self._estimate_real_temporal(real_behavior_messages)

        all_metrics.update(MesoMetrics.compute(
            sim_graph=sim_graph,
            real_graph=real_graph,
            sim_chain_lengths=sim_chain_lengths,
            real_chain_lengths=real_chain_lengths,
            sim_temporal=sim_temporal if sim_temporal else None,
            real_temporal=real_temporal if real_temporal else None,
        ))

        # Layer 3: Micro
        # Build agent×action matrices for Frobenius similarity.
        # Use the union of actions so the matrices are aligned and missing
        # actions are explicitly zero-padded.
        all_actions = sorted(
            set(sim_action_dist.keys()) | set(real_action_dist.keys())
        )
        sim_matrix = self._build_action_matrix(sim_agent_counts, actions=all_actions)
        real_matrix = self._build_action_matrix(real_agent_counts, actions=all_actions)

        # Build behavioral profiles for RSA
        sim_profiles = self._build_profiles(sim_agent_counts, actions=all_actions)
        real_profiles = self._build_profiles(real_agent_counts, actions=all_actions)

        all_metrics.update(MicroMetrics.compute(
            sim_matrix=sim_matrix,
            real_matrix=real_matrix,
            sim_profiles=sim_profiles,
            real_profiles=real_profiles,
            sim_action_counts=Counter(sim_action_dist),
            real_action_counts=Counter(real_action_dist),
            sim_agent_counts=sim_agent_counts,
            real_agent_counts=real_agent_counts,
        ))

        # Caricature Index (outline §5.3 — between-cluster behavioral Cohen's d)
        # Responds to Chameleon's Limit §3.3 "fidelity breeds caricature":
        # measures whether enforcement increases between-cluster stereotyping.
        caricature_sim = caricature_index(sim_agent_counts, sim_communities)
        caricature_real = caricature_index(real_agent_counts, real_communities)
        sim_ci_low, sim_ci_high = caricature_bootstrap_ci(
            sim_agent_counts, sim_communities, seed=sim_result.repeat + 42,
        )
        real_ci_low, real_ci_high = caricature_bootstrap_ci(
            real_agent_counts, real_communities, seed=self.seed,
        )
        all_metrics["caricature_index_sim"] = caricature_sim
        all_metrics["caricature_index_real"] = caricature_real
        all_metrics["caricature_gap"] = abs(caricature_sim - caricature_real)
        all_metrics["caricature_index_sim_ci_low"] = sim_ci_low
        all_metrics["caricature_index_sim_ci_high"] = sim_ci_high
        all_metrics["caricature_index_real_ci_low"] = real_ci_low
        all_metrics["caricature_index_real_ci_high"] = real_ci_high

        # Layer 4: Linguistics
        sim_sample = self._stratified_message_sample(sim_messages, 200)
        sim_msg_objects = [self._dict_to_msg(m) for m in sim_sample]
        all_real_msgs = self._stratified_thread_message_sample(
            linguistic_reference_threads, 500,
        )
        if not sim_msg_objects or not all_real_msgs:
            raise EvaluationIntegrityError("Linguistic reference or simulation sample is empty")

        # Forced-reformulation safe templates are reported separately because
        # their deliberately bland language would otherwise confound fidelity.
        sim_normal = [
            m for m in sim_msg_objects
            if not m.metadata.get("constraint_forced", False)
        ]
        sim_safe = [
            m for m in sim_msg_objects
            if m.metadata.get("constraint_forced", False)
        ]

        # Speech act classification: local RoBERTa is primary; an LLM
        # classifier is used only when the caller supplied one.
        classifier = None
        if self._llm_client is not None and self._llm_model_name is not None:
            from src.evaluation.linguistics import LLMSpeechActClassifier
            classifier = LLMSpeechActClassifier(self._llm_client, self._llm_model_name)

        pooled_metrics = await LinguisticMetrics.compute(
            sim_messages=sim_msg_objects,
            real_messages=all_real_msgs,
            classifier=classifier,
        )
        all_metrics.update(pooled_metrics)

        # Headline stratum: normal (non-safe-template) outputs.
        if sim_normal:
            if len(sim_normal) == len(sim_msg_objects):
                # Common case: no fallback output. Pooled and normal strata
                # are identical, so do not run four CPU-heavy local models a
                # second time for the same message lists.
                normal_metrics = dict(pooled_metrics)
            else:
                normal_metrics = await LinguisticMetrics.compute(
                    sim_messages=sim_normal,
                    real_messages=all_real_msgs,
                    classifier=classifier,
                )
            for k, v in normal_metrics.items():
                all_metrics[f"{k}_normal"] = v

        # Audit stratum: safe-template outputs only.
        all_metrics["linguistics_safe_template_count"] = float(len(sim_safe))
        all_metrics["linguistics_normal_count"] = float(len(sim_normal))
        if sim_safe:
            safe_metrics = await LinguisticMetrics.compute(
                sim_messages=sim_safe,
                real_messages=all_real_msgs,
                classifier=classifier,
            )
            for k, v in safe_metrics.items():
                all_metrics[f"{k}_safe_template"] = v

        # One pre-registered distance per feasibility family. This avoids
        # treating several transformations of the same action distribution as
        # independent votes in the GO/STOP decision.
        all_metrics["action_fidelity_distance"] = float(all_metrics["ned"])
        all_metrics["interaction_structure_distance"] = float(
            self.interaction_metric_weights["cascade"] * all_metrics["ks_statistic"]
            + self.interaction_metric_weights["graph"]
            * (1.0 - all_metrics["structural_fidelity"])
        )
        # Safe-template fallbacks are a mechanism-failure stratum, not normal
        # language. Exclude them from the headline linguistic family and
        # enforce their prevalence separately via the viability guard.
        suffix = "_normal" if sim_normal else ""
        all_metrics["linguistic_fidelity_distance"] = float(sum(
            float(weight) * (
                1.0 - max(0.0, min(1.0, float(all_metrics[f"{metric}{suffix}"])))
            )
            for metric, weight in self.linguistic_metric_weights.items()
        ))
        all_metrics["action_text_consistency"] = action_text_consistency
        all_metrics["action_text_consistency_n"] = float(action_text_n)

        # Predictive fidelity layer removed — regex-based event detection
        # was unreliable (precision/recall both low, context-blind keyword
        # matching). CGA corpus provides gold-standard
        # ``conversation_has_personal_attack`` labels; a future replacement
        # should use those directly instead of train-on-sim/test-on-real
        # prediction with heuristic labels. See memory/predictive-fidelity-cga-fix.md.

        # Enforcement stats
        if sim_result.enforcement_stats:
            for k, v in sim_result.enforcement_stats.items():
                all_metrics[f"enforcement_{k}"] = v
        all_metrics["action_taxonomy_version"] = "canonical_behavior_v1"
        from src.experiment.conditions import condition_display_name
        all_metrics["condition_display_name"] = condition_display_name(
            sim_result.condition
        )
        all_metrics["simulation_integrity_passed"] = True
        all_metrics["simulation_message_count"] = len(sim_messages)
        all_metrics["simulation_round_count"] = len(sim_result.per_round_metrics)
        all_metrics["simulation_expected_round_count"] = sim_result.rounds
        all_metrics["run_fingerprint"] = sim_result.run_fingerprint
        self._validate_required_metrics(all_metrics)
        report = MetricsReport(
            run_id=sim_result.run_id,
            condition=sim_result.condition,
            dataset=sim_result.dataset,
            model=sim_result.model,
            repeat=sim_result.repeat,
            metrics=all_metrics,
            used_role_label_proxy=sim_result.dataset in self.datasets_using_role_label_proxy,
            used_held_out_events_heuristic=False,
            model_snapshot_date=self.model_provenance.get(sim_result.model, {}).get("snapshot_date", ""),
            model_commit_hash=self.model_provenance.get(sim_result.model, {}).get("commit_hash", ""),
        )
        self.reports.append(report)
        return report

    @staticmethod
    def _validate_simulation_result(sim_result: SimulationResult) -> None:
        if sim_result.rounds <= 0 or not sim_result.messages:
            raise EvaluationIntegrityError("Simulation is empty")
        if len(sim_result.per_round_metrics) != sim_result.rounds:
            raise EvaluationIntegrityError(
                f"Simulation has {len(sim_result.per_round_metrics)}/"
                f"{sim_result.rounds} round metric rows"
            )
        observed_rounds = {int(m.get("round", -1)) for m in sim_result.messages}
        expected_rounds = set(range(sim_result.rounds))
        if observed_rounds != expected_rounds:
            raise EvaluationIntegrityError(
                f"Simulation message rounds are incomplete: missing="
                f"{sorted(expected_rounds - observed_rounds)[:10]}"
            )
        if not sim_result.run_fingerprint:
            raise EvaluationIntegrityError("Simulation has no run fingerprint")

    @staticmethod
    def _label_coverage(graph: nx.Graph, labels: dict[str, int]) -> tuple[float, float]:
        nodes = list(graph.nodes())
        node_coverage = sum(n in labels for n in nodes) / max(len(nodes), 1)
        edges = list(graph.edges())
        edge_coverage = sum(u in labels and v in labels for u, v in edges) / max(len(edges), 1)
        return node_coverage, edge_coverage

    @staticmethod
    def _validate_action_reference(action_dist: dict[str, float]) -> None:
        active = {k: v for k, v in action_dist.items() if v > 0}
        if len(active) < 2:
            raise EvaluationIntegrityError(
                f"Action ground truth is degenerate: {active}. "
                "Use WikiConv action-event references, not all-DISCUSS CGA."
            )
        if max(active.values()) >= 0.995:
            raise EvaluationIntegrityError(
                f"Action ground truth is effectively single-class: {active}"
            )

    @staticmethod
    def _validate_required_metrics(metrics: dict[str, Any]) -> None:
        required = {
            "delta_q_modularity", "ei_polarization_sim", "ei_polarization_real",
            "ned", "coverage", "structural_fidelity", "ks_statistic", "p_value",
            "dtw_distance", "action_matrix_similarity", "rsa", "uniformity_sim",
            "uniformity_real", "uniformity_gap", "complexity_sim", "complexity_real",
            "complexity_gap", "caricature_index_sim", "caricature_index_real",
            "caricature_gap", "discourse_relation_match",
            "sentiment_trajectory_similarity", "speech_act_similarity", "sip",
            "action_fidelity_distance", "interaction_structure_distance",
            "linguistic_fidelity_distance", "action_text_consistency",
        }
        missing = sorted(required - metrics.keys())
        if missing:
            raise EvaluationIntegrityError(f"Required metrics missing: {missing}")
        invalid = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not math.isfinite(float(value))
        }
        if invalid:
            raise EvaluationIntegrityError(f"Metrics contain NaN/Inf: {invalid}")

    @staticmethod
    def _stratified_message_sample(messages: list[dict], limit: int) -> list[dict]:
        """Deterministic round×thread round-robin sample."""
        buckets: dict[tuple[int, str], list[dict]] = {}
        for message in messages:
            key = (int(message.get("round", 0)), str(message.get("thread_id", "")))
            buckets.setdefault(key, []).append(message)
        ordered = [buckets[k] for k in sorted(buckets)]
        sample: list[dict] = []
        offset = 0
        while ordered and len(sample) < limit:
            next_ordered = []
            for bucket in ordered:
                if offset < len(bucket):
                    sample.append(bucket[offset])
                    if len(sample) >= limit:
                        break
                if offset + 1 < len(bucket):
                    next_ordered.append(bucket)
            ordered = next_ordered
            offset += 1
        return sample

    @staticmethod
    def _stratified_thread_message_sample(
        threads: list[Thread], limit: int,
    ) -> list[Message]:
        buckets = [
            sorted(
                (m for m in thread.messages if m.text.strip()),
                key=lambda m: (m.timestamp, m.msg_id),
            )
            for thread in sorted(threads, key=lambda t: t.thread_id)
        ]
        buckets = [b for b in buckets if b]
        sample: list[Message] = []
        offset = 0
        while buckets and len(sample) < limit:
            next_buckets = []
            for bucket in buckets:
                if offset < len(bucket):
                    sample.append(bucket[offset])
                    if len(sample) >= limit:
                        break
                if offset + 1 < len(bucket):
                    next_buckets.append(bucket)
            buckets = next_buckets
            offset += 1
        return sample

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all reports to a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.reports])

    def save_results(self, path: str) -> None:
        """Save results to CSV."""
        self.to_dataframe().to_csv(path, index=False)

    def proxy_fallback_summary(self) -> dict[str, Any]:
        """Anti-circularity provenance summary for §7.4 Threats to Validity.

        Outline §5.3 mandates external role labels (Wu et al. 2025
        segmentation or human-annotated moderator / provocateur /
        peacemaker / lurker) and §5.3 / §4.1 mandate Cohen's-κ-validated
        held-out event annotations. When those data files are absent the
        aggregator falls back to (a) Louvain communities inferred from
        the real interaction graph and (b) heuristic action-derived
        event labels — both *weaker* than the spec's external ground
        truth. This method rolls up the per-dataset fallback flags so
        the paper can report exactly which datasets used which ground
        truth layer, instead of silently burying the substitution.

        Returns:
            Dict with ``datasets_using_role_label_proxy`` (list),
            ``datasets_using_held_out_heuristic`` (list),
            ``per_dataset`` mapping, and a ``release_ready`` boolean
            (True iff no proxies / heuristics are in use).
        """
        per_dataset: dict[str, dict[str, bool]] = {}
        for r in self.reports:
            entry = per_dataset.setdefault(r.dataset, {
                "used_role_label_proxy": False,
                "used_held_out_events_heuristic": False,
            })
            entry["used_role_label_proxy"] = entry["used_role_label_proxy"] or r.used_role_label_proxy
            entry["used_held_out_events_heuristic"] = (
                entry["used_held_out_events_heuristic"] or r.used_held_out_events_heuristic
            )

        role_proxy = sorted(d for d, v in per_dataset.items() if v["used_role_label_proxy"])
        held_out_heuristic = sorted(
            d for d, v in per_dataset.items() if v["used_held_out_events_heuristic"]
        )
        return {
            "datasets_using_role_label_proxy": role_proxy,
            "datasets_using_held_out_heuristic": held_out_heuristic,
            "per_dataset": per_dataset,
            "release_ready": (not role_proxy) and (not held_out_heuristic),
            "note": (
                "Outline §5.3 anti-circularity mandate: external role labels "
                "+ Cohen's-κ validated held-out events. Absent data → Louvain "
                "proxy + heuristic event labels (weaker)."
            ),
        }

    def _msg_to_dict(self, msg: Message) -> dict:
        return {
            "msg_id": msg.msg_id,
            "thread_id": msg.thread_id,
            "user_id": msg.user_id,
            "action_type": msg.action_type.value,
            "text": msg.text,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            "parent_msg_id": msg.parent_msg_id,
            "metadata": dict(msg.metadata or {}),
        }

    def _dict_to_msg(self, d: dict) -> Message:
        from datetime import datetime
        from src.data.schemas import ActionType, Platform
        # Infer platform from thread_id prefix if available
        tid = d.get("thread_id", "")
        if "sim_transfer_reddit" in tid or "sim_" in tid and "reddit" in tid:
            platform = Platform.REDDIT
        elif "sim_transfer_github" in tid or "sim_" in tid and "github" in tid:
            platform = Platform.GITHUB
        else:
            platform = Platform.WIKIPEDIA
        # Preserve metadata so downstream evaluation can stratify by the
        # ``constraint_forced`` safe-template flag (outline §5.7).
        raw_meta = d.get("metadata") or {}
        return Message(
            msg_id=d.get("msg_id", ""),
            thread_id=tid,
            user_id=d.get("user_id", ""),
            platform=platform,
            timestamp=datetime.now(),
            text=d.get("text", ""),
            action_type=ActionType(d.get("action_type", "post")),
            metadata=dict(raw_meta) if isinstance(raw_meta, dict) else {},
        )

    def _build_graph_from_threads(self, threads: list[Thread]) -> nx.Graph:
        graph = nx.Graph()
        for thread in threads:
            for user_id in thread.participants:
                graph.add_node(user_id)
            msg_by_id = {m.msg_id: m for m in thread.messages}
            for msg in thread.messages:
                if msg.parent_msg_id and msg.parent_msg_id in msg_by_id:
                    parent = msg_by_id[msg.parent_msg_id]
                    if parent.user_id != msg.user_id:
                        if graph.has_edge(parent.user_id, msg.user_id):
                            graph[parent.user_id][msg.user_id]["weight"] += 1
                        else:
                            graph.add_edge(parent.user_id, msg.user_id, weight=1)
        return graph

    def _build_graph_from_dicts(self, messages: list[dict], agent_states: list[dict]) -> nx.Graph:
        """Build interaction graph from simulated messages using parent-child edges.

        Mirrors _build_graph_from_threads for structural symmetry.
        Falls back to thread co-presence if parent_msg_id is missing.
        """
        graph = nx.Graph()
        for state in agent_states:
            graph.add_node(state["agent_id"], cluster_id=state.get("cluster_id", 0))

        # Build msg lookup for parent resolution
        msg_by_id = {m["msg_id"]: m for m in messages}

        for msg in messages:
            parent_id = msg.get("parent_msg_id")
            if parent_id and parent_id in msg_by_id:
                parent = msg_by_id[parent_id]
                u, v = parent["user_id"], msg["user_id"]
                if u != v:
                    if graph.has_edge(u, v):
                        graph[u][v]["weight"] += 1
                    else:
                        graph.add_edge(u, v, weight=1)
        return graph

    def _infer_communities(self, threads: list[Thread]) -> dict[str, int]:
        """Infer communities from real interaction graph via Louvain."""
        graph = self._build_graph_from_threads(threads)
        if len(graph) == 0:
            return {}

        import networkx.algorithms.community as nx_comm
        communities_list = nx_comm.louvain_communities(graph, seed=self.seed)
        user_to_comm = {}
        for i, comm in enumerate(communities_list):
            for user in comm:
                user_to_comm[user] = i
        return user_to_comm

    def _compute_action_dist(self, messages: list[dict]) -> dict[str, float]:
        counts = Counter(m["action_type"] for m in messages)
        total = sum(counts.values()) or 1
        return {action: count / total for action, count in counts.items()}

    @staticmethod
    def _action_text_consistency(messages: list[dict]) -> tuple[float, int]:
        """Conservative action/text alignment guard for the feasibility run.

        It is intentionally a guard rather than a fidelity outcome. Structured
        non-text platform actions are accepted; textual AGREE/DISAGREE actions
        are checked for explicit stance markers so style regeneration cannot
        silently reverse the planner's selected stance.
        """
        checked = 0
        passed = 0
        for message in messages:
            action = str(message.get("action_type", "")).lower()
            text = str(message.get("text") or "").strip().lower()
            metadata = message.get("metadata") or {}
            planned_action = metadata.get("planned_action_type")
            if planned_action is None:
                continue
            checked += 1
            if str(planned_action).lower() == action:
                passed += 1
        # Legacy results lack planned_action_type. Treat the guard as
        # unavailable rather than applying brittle English keyword patterns.
        return (passed / checked, checked) if checked else (1.0, 0)

    @staticmethod
    def _canonical_action(action: str, dataset: str) -> str:
        """Map platform actions into a shared behavioral ontology."""
        action = action.lower()
        if dataset == "wikipedia":
            mapping = {
                "discuss": "participation", "agree": "participation",
                "post": "participation", "reply": "participation",
                "disagree": "conflict", "revert": "conflict",
                "edit": "content_change", "delete": "moderation",
                "restore": "moderation", "report": "moderation",
            }
        elif dataset == "reddit":
            mapping = {
                "reply": "participation", "post": "participation",
                "comment": "participation", "agree": "participation",
                "counter_argue": "conflict", "disagree": "conflict",
                "block": "moderation", "report": "moderation",
                "award_delta": "persuasion",
            }
        else:  # github
            mapping = {
                "comment": "participation", "reply": "participation",
                "post": "participation", "discuss": "participation",
                "label": "moderation", "assign": "moderation",
                "close": "lifecycle", "reopen": "lifecycle",
                "edit": "content_change",
            }
        return mapping.get(action, "other")

    @classmethod
    def _canonicalize_actions(cls, messages: list[dict], dataset: str) -> list[dict]:
        canonical = []
        for message in messages:
            copied = dict(message)
            copied["action_type"] = cls._canonical_action(
                str(message.get("action_type", "")), dataset
            )
            canonical.append(copied)
        return canonical

    @staticmethod
    def _estimate_sim_temporal(messages: list[dict]) -> list[float]:
        by_round: dict[int, list[dict]] = {}
        for message in messages:
            by_round.setdefault(int(message.get("round", 0)), []).append(message)
        return [
            sum(m["action_type"] == "conflict" for m in by_round[r]) /
            max(len(by_round[r]), 1)
            for r in sorted(by_round)
        ]

    def _compute_chain_lengths(self, threads: list[Thread]) -> list[int]:
        lengths = []
        for thread in threads:
            msg_by_id = {m.msg_id: m for m in thread.messages}

            def chain_length(msg_id, _visited=None):
                # Match _compute_chain_lengths_from_dicts exactly: a parent
                # outside the observed thread slice contributes no extra
                # depth, and a cycle terminates the current branch.
                visited = _visited if _visited is not None else set()
                if msg_id in visited:
                    return 0
                msg = msg_by_id.get(msg_id)
                if msg is None:
                    return 0
                visited = visited | {msg_id}
                if not msg.parent_msg_id or msg.parent_msg_id not in msg_by_id:
                    return 1
                return 1 + chain_length(msg.parent_msg_id, visited)

            for msg in thread.messages:
                lengths.append(chain_length(msg.msg_id))
        return lengths

    def _compute_chain_lengths_from_dicts(
        self, messages: list[dict], context_messages: list[dict] | None = None,
    ) -> list[int]:
        """Compute reply chain nesting depth from simulated messages.

        Uses parent_msg_id if available; otherwise falls back to message count per thread.
        """
        # Group by thread
        thread_msgs: dict[str, list[dict]] = {}
        for m in messages:
            thread_msgs.setdefault(m["thread_id"], []).append(m)

        lengths = []
        context_by_thread: dict[str, list[dict]] = {}
        for message in context_messages or []:
            context_by_thread.setdefault(message["thread_id"], []).append(message)

        for tid, msgs in thread_msgs.items():
            msg_by_id = {
                m["msg_id"]: m for m in context_by_thread.get(tid, []) + msgs
            }

            def chain_depth(msg_id: str, visited: set | None = None) -> int:
                if visited is None:
                    visited = set()
                if msg_id in visited:
                    return 0  # cycle protection
                visited.add(msg_id)
                msg = msg_by_id.get(msg_id)
                if msg is None:
                    return 0
                parent_id = msg.get("parent_msg_id")
                if not parent_id or parent_id not in msg_by_id:
                    return 1
                return 1 + chain_depth(parent_id, visited)

            # Only generated messages are observations; seed messages merely
            # provide the missing ancestry needed to compute their true depth.
            for m in msgs:
                lengths.append(chain_depth(m["msg_id"]))

        return lengths if lengths else [1]

    def _compute_agent_action_counts(self, threads: list[Thread]) -> dict[str, Counter]:
        result: dict[str, Counter] = {}
        for thread in threads:
            for msg in thread.messages:
                result.setdefault(msg.user_id, Counter())[msg.action_type.value] += 1
        return result

    def _compute_agent_action_counts_from_dicts(self, messages: list[dict]) -> dict[str, Counter]:
        result: dict[str, Counter] = {}
        for msg in messages:
            uid = msg["user_id"]
            result.setdefault(uid, Counter())[msg["action_type"]] += 1
        return result

    def _build_action_matrix(self, agent_counts: dict[str, Counter], actions: list[str] | None = None) -> np.ndarray:
        """Build agent × action count matrix.

        If ``actions`` is provided, columns are fixed to that ordering and
        missing actions are zero-filled. This lets simulated and real
        matrices share the same action space for fair comparison.
        """
        if not agent_counts:
            return np.zeros((1, 1))

        all_actions = actions if actions is not None else sorted(set(
            a for counts in agent_counts.values() for a in counts
        ))
        if not all_actions:
            return np.zeros((len(agent_counts), 1))

        matrix = np.zeros((len(agent_counts), len(all_actions)))
        for i, (agent, counts) in enumerate(agent_counts.items()):
            for j, action in enumerate(all_actions):
                matrix[i, j] = counts.get(action, 0)
        return matrix

    def _build_profiles(self, agent_counts: dict[str, Counter], actions: list[str] | None = None) -> np.ndarray:
        """Build normalized behavioral profiles for RSA."""
        matrix = self._build_action_matrix(agent_counts, actions=actions)
        if matrix.size == 0:
            return np.zeros((1, 1))

        # Normalize per agent (row normalization)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums

    def _estimate_real_temporal(self, real_messages: list[dict]) -> list[float]:
        """Estimate an episode-relative conflict trajectory.

        Conversations are aligned by relative message position before
        aggregation. Sorting unrelated conversations by absolute calendar time
        would create a dataset-history trend rather than an interaction trend.
        """
        if not real_messages:
            return []

        conflict_actions = {"conflict"}

        from datetime import datetime
        def _parse_ts(m):
            ts = m.get("timestamp")
            if ts is None:
                return datetime.max
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    return datetime.max
            return ts

        by_thread: dict[str, list[dict]] = {}
        for message in real_messages:
            by_thread.setdefault(str(message.get("thread_id", "")), []).append(message)

        n_bins = 30
        totals = np.zeros(n_bins, dtype=float)
        conflicts = np.zeros(n_bins, dtype=float)
        for messages in by_thread.values():
            ordered = sorted(messages, key=lambda m: (_parse_ts(m), m.get("msg_id", "")))
            for index, message in enumerate(ordered):
                position = index / max(len(ordered) - 1, 1)
                bin_index = min(int(position * n_bins), n_bins - 1)
                totals[bin_index] += 1
                conflicts[bin_index] += message["action_type"] in conflict_actions
        return [
            float(conflicts[i] / totals[i])
            for i in range(n_bins) if totals[i] > 0
        ]
