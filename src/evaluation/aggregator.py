"""Metrics aggregator — collects all metric layers into result tables."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data.schemas import Message, Thread
from src.evaluation.linguistics import LinguisticMetrics
from src.evaluation.macro import MacroMetrics
from src.evaluation.meso import MesoMetrics
from src.evaluation.micro import MicroMetrics
from src.evaluation.predictive import PredictiveMetrics
from src.simulation.sandbox import SimulationResult


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
    used_held_out_events_heuristic: bool = False

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "dataset": self.dataset,
            "model": self.model,
            "repeat": self.repeat,
            "used_role_label_proxy": self.used_role_label_proxy,
            "used_held_out_events_heuristic": self.used_held_out_events_heuristic,
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
    ):
        self.reports: list[MetricsReport] = []
        self.held_out_events_dir = held_out_events_dir
        self.role_labels_dir = role_labels_dir
        # Tracks which datasets fell back to the Louvain proxy, for §7.4 reporting
        self.datasets_using_role_label_proxy: set[str] = set()

    def _load_held_out_events(self, dataset: str) -> list | None:
        """Load annotated held-out events for a dataset, if available."""
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
        return {uid: role_to_id[r] for uid, r in labels.items()}

    def evaluate(
        self,
        sim_result: SimulationResult,
        real_threads: list[Thread],
    ) -> MetricsReport:
        """Evaluate a simulation result against real data.

        Args:
            sim_result: Simulation output.
            real_threads: Ground truth threads from real data.

        Returns:
            MetricsReport with all 5 metric layers.
        """
        # Build real data structures
        real_messages = [
            self._msg_to_dict(m) for t in real_threads for m in t.messages
        ]
        real_graph = self._build_graph_from_threads(real_threads)

        # Outline §5.3 anti-circularity mandate: prefer external role labels
        # (Wu et al. 2025 segmentation or human-annotated moderator /
        # provocateur / peacemaker / lurker). Fall back to Louvain communities
        # inferred from the interaction graph when labels are unavailable,
        # and record the fallback so §7.4 can report per-dataset validity.
        external_role_labels = self._load_role_labels(sim_result.dataset)
        if external_role_labels is not None:
            real_communities = external_role_labels
            logger.info(
                f"Micro/Macro real-data ground truth: using EXTERNAL role labels "
                f"for {sim_result.dataset} ({len(external_role_labels)} users, "
                f"{len(set(external_role_labels.values()))} roles)"
            )
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

        real_action_dist = self._compute_action_dist(real_messages)
        real_chain_lengths = self._compute_chain_lengths(real_threads)
        real_agent_counts = self._compute_agent_action_counts(real_threads)

        # Build simulation data structures
        sim_messages = sim_result.messages
        sim_graph = self._build_graph_from_dicts(sim_messages, sim_result.agent_states)
        sim_communities = {
            a["agent_id"]: int(a.get("cluster_id", 0))
            for a in sim_result.agent_states
        }
        sim_action_dist = self._compute_action_dist(sim_messages)
        sim_agent_counts = self._compute_agent_action_counts_from_dicts(sim_messages)

        # Compute metrics per layer
        all_metrics = {}

        # Layer 1: Macro
        try:
            all_metrics.update(MacroMetrics.compute(
                sim_graph=sim_graph,
                real_graph=real_graph,
                sim_communities=sim_communities,
                real_communities=real_communities,
                sim_action_dist=sim_action_dist,
                real_action_dist=real_action_dist,
            ))
        except Exception as e:
            logger.warning(f"MacroMetrics computation failed: {e}")

        # Layer 2: Meso
        try:
            sim_chain_lengths = self._compute_chain_lengths_from_dicts(sim_messages)

            # Extract temporal sequences for DTW (outline §6.4)
            sim_temporal = [
                rm.get("polarization_proxy", 0.0)
                for rm in sim_result.per_round_metrics
            ]
            real_temporal = self._estimate_real_temporal(real_messages)

            all_metrics.update(MesoMetrics.compute(
                sim_graph=sim_graph,
                real_graph=real_graph,
                sim_chain_lengths=sim_chain_lengths,
                real_chain_lengths=real_chain_lengths,
                sim_temporal=sim_temporal if sim_temporal else None,
                real_temporal=real_temporal if real_temporal else None,
            ))
        except Exception as e:
            logger.warning(f"MesoMetrics computation failed: {e}")

        # Layer 3: Micro
        try:
            # Build agent×action matrices for Frobenius similarity
            sim_matrix = self._build_action_matrix(sim_agent_counts)
            real_matrix = self._build_action_matrix(real_agent_counts)

            # Build behavioral profiles for RSA
            sim_profiles = self._build_profiles(sim_agent_counts)
            real_profiles = self._build_profiles(real_agent_counts)

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
        except Exception as e:
            logger.warning(f"MicroMetrics computation failed: {e}")

        # Layer 4: Linguistics
        try:
            import random as _rand
            sim_msg_objects = [self._dict_to_msg(m) for m in sim_messages[:200]]
            # Sample real messages representatively across all threads
            all_real_msgs = [m for t in real_threads for m in t.messages if m.text.strip()]
            if len(all_real_msgs) > 500:
                all_real_msgs = _rand.Random(42).sample(all_real_msgs, 500)
            all_metrics.update(LinguisticMetrics.compute(
                sim_messages=sim_msg_objects,
                real_messages=all_real_msgs,
            ))
        except Exception as e:
            logger.warning(f"LinguisticMetrics computation failed: {e}")

        # Layer 5: Predictive
        held_out_events: list | None = None
        try:
            held_out_events = self._load_held_out_events(sim_result.dataset)
            pred_result = PredictiveMetrics.compute(
                sim_messages=sim_messages,
                real_messages=real_messages,
                held_out_events=held_out_events,
            )
            # Flatten nested prediction task dicts with prefixes
            for task_name, task_metrics in pred_result.items():
                if isinstance(task_metrics, dict):
                    for metric_name, value in task_metrics.items():
                        all_metrics[f"pred_{task_name}_{metric_name}"] = value
                else:
                    all_metrics[f"pred_{task_name}"] = task_metrics
        except Exception as e:
            logger.warning(f"PredictiveMetrics computation failed: {e}")

        # Enforcement stats
        if sim_result.enforcement_stats:
            for k, v in sim_result.enforcement_stats.items():
                all_metrics[f"enforcement_{k}"] = v

        report = MetricsReport(
            run_id=sim_result.run_id,
            condition=sim_result.condition,
            dataset=sim_result.dataset,
            model=sim_result.model,
            repeat=sim_result.repeat,
            metrics=all_metrics,
            used_role_label_proxy=sim_result.dataset in self.datasets_using_role_label_proxy,
            used_held_out_events_heuristic=(held_out_events is None),
        )
        self.reports.append(report)
        return report

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
        return Message(
            msg_id=d.get("msg_id", ""),
            thread_id=tid,
            user_id=d.get("user_id", ""),
            platform=platform,
            timestamp=datetime.now(),
            text=d.get("text", ""),
            action_type=ActionType(d.get("action_type", "post")),
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
        communities_list = nx_comm.louvain_communities(graph, seed=42)
        user_to_comm = {}
        for i, comm in enumerate(communities_list):
            for user in comm:
                user_to_comm[user] = i
        return user_to_comm

    def _compute_action_dist(self, messages: list[dict]) -> dict[str, float]:
        counts = Counter(m["action_type"] for m in messages)
        total = sum(counts.values()) or 1
        return {action: count / total for action, count in counts.items()}

    def _compute_chain_lengths(self, threads: list[Thread]) -> list[int]:
        lengths = []
        for thread in threads:
            msg_by_id = {m.msg_id: m for m in thread.messages}

            def chain_length(msg_id: str) -> int:
                msg = msg_by_id.get(msg_id)
                if msg is None or msg.parent_msg_id is None:
                    return 1
                return 1 + chain_length(msg.parent_msg_id)

            for msg in thread.messages:
                lengths.append(chain_length(msg.msg_id))
        return lengths

    def _compute_chain_lengths_from_dicts(self, messages: list[dict]) -> list[int]:
        """Compute reply chain nesting depth from simulated messages.

        Uses parent_msg_id if available; otherwise falls back to message count per thread.
        """
        # Group by thread
        thread_msgs: dict[str, list[dict]] = {}
        for m in messages:
            thread_msgs.setdefault(m["thread_id"], []).append(m)

        lengths = []
        for tid, msgs in thread_msgs.items():
            msg_by_id = {m["msg_id"]: m for m in msgs}

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

    def _build_action_matrix(self, agent_counts: dict[str, Counter]) -> np.ndarray:
        """Build agent × action count matrix."""
        if not agent_counts:
            return np.zeros((1, 1))

        all_actions = sorted(set(
            a for counts in agent_counts.values() for a in counts
        ))
        if not all_actions:
            return np.zeros((len(agent_counts), 1))

        matrix = np.zeros((len(agent_counts), len(all_actions)))
        for i, (agent, counts) in enumerate(agent_counts.items()):
            for j, action in enumerate(all_actions):
                matrix[i, j] = counts.get(action, 0)
        return matrix

    def _build_profiles(self, agent_counts: dict[str, Counter]) -> np.ndarray:
        """Build normalized behavioral profiles for RSA."""
        matrix = self._build_action_matrix(agent_counts)
        if matrix.size == 0:
            return np.zeros((1, 1))

        # Normalize per agent (row normalization)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums

    def _estimate_real_temporal(self, real_messages: list[dict]) -> list[float]:
        """Estimate polarization proxy trajectory from real data.

        Sorts messages by timestamp, partitions into temporal bins, and
        computes the conflict-action ratio per bin, mirroring the simulation's
        per-round ``polarization_proxy`` metric.
        """
        if not real_messages:
            return []

        conflict_actions = {"disagree", "revert", "counter_argue", "report", "close", "reopen", "block"}

        # Sort by timestamp if available
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

        sorted_msgs = sorted(real_messages, key=_parse_ts)

        n_bins = min(len(sorted_msgs), 30)
        bin_size = max(len(sorted_msgs) // n_bins, 1)

        trajectory = []
        for i in range(0, len(sorted_msgs), bin_size):
            batch = sorted_msgs[i : i + bin_size]
            total = len(batch)
            conflict = sum(1 for m in batch if m["action_type"] in conflict_actions)
            trajectory.append(conflict / total if total > 0 else 0.0)

        return trajectory
