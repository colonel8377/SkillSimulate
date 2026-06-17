"""Predictive fidelity metrics — held-out event prediction (outline §5.3).

Uses simulation data to predict real held-out interaction outcomes:
- Who will conflict
- Whether persuasion succeeds
- Whether conflict escalates

Ground truth follows the held-out event annotation protocol (§5.3):
2 annotators independently code controversial events, Cohen's κ ≥ 0.7,
consensus labels become ground truth. When no annotated events are
provided (or a specific event type yields no consensus labels), a
heuristic fallback derives labels from action types. The result's
``annotation_protocol`` field records this per task (conflict /
persuasion / escalation) so downstream readers can see exactly which
tasks had κ-validated GT vs heuristic GT — ``overall`` is "annotated"
iff all three tasks used consensus labels, "mixed" if some fell back,
"heuristic" if no held-out events were supplied at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from src.evaluation.held_out_events import (
    EVENT_CONFLICT,
    EVENT_ESCALATION,
    EVENT_PERSUASION,
    HeldOutEvent,
    consensus_ground_truth,
    resolve_events,
)
# Note: outline §4.1 mentions "consensus formation" as part of the held-out
# event annotation universe, but §5.3 lists only conflict / persuasion /
# escalation as predictive-fidelity tasks. EVENT_CONSENSUS remains defined
# in held_out_events.py for the annotation protocol; it is intentionally
# not a predictive task here.


_CONFLICT_ACTIONS = {"disagree", "revert", "counter_argue", "report"}
_PERSUASION_ACTIONS = {"award_delta"}
# B2 fix: ``close``/``reopen`` are GitHub issue-lifecycle actions (outline
# §4.3), not escalation signals — including them inflated the GitHub
# dataset's escalation labels. Kept in sync with sandbox conflict_actions.
_ESCALATION_ACTIONS = {
    "disagree", "revert", "counter_argue", "report", "block",
}


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------

def extract_features(messages: list[dict], thread_id: str, user_id: str) -> np.ndarray:
    """Extract behavioral features for a user in a thread."""
    thread_msgs = [m for m in messages if m["thread_id"] == thread_id]
    user_msgs = [m for m in thread_msgs if m["user_id"] == user_id]

    if not user_msgs:
        return np.zeros(8)

    actions = [m["action_type"] for m in user_msgs]
    total = len(user_msgs)

    conflict_count = sum(1 for a in actions if a in _CONFLICT_ACTIONS)
    agree_count = sum(1 for a in actions if a in _PERSUASION_ACTIONS or a == "agree")
    avg_text_len = float(np.mean([len(m.get("text", "")) for m in user_msgs]))
    n_actions = len(set(actions))
    first_action = actions[0] if actions else ""

    return np.array([
        total,
        conflict_count / total if total > 0 else 0,
        agree_count / total if total > 0 else 0,
        avg_text_len,
        n_actions,
        len(thread_msgs),
        len(set(m["user_id"] for m in thread_msgs)),
        1.0 if first_action in _CONFLICT_ACTIONS else 0.0,
    ])


def _max_chain_depth(messages: list[dict]) -> int:
    parent_map = {m.get("msg_id", ""): m.get("parent_msg_id") for m in messages}

    def depth(msg_id, cache=None):
        if cache is None:
            cache = {}
        if msg_id in cache:
            return cache[msg_id]
        parent = parent_map.get(msg_id)
        if not parent or parent not in parent_map:
            cache[msg_id] = 0
            return 0
        d = depth(parent, cache) + 1
        cache[msg_id] = d
        return d

    return max((depth(mid) for mid in parent_map), default=0)


# ----------------------------------------------------------------------
# Generic train-on-sim / test-on-ground-truth evaluator
# ----------------------------------------------------------------------

def _train_and_evaluate(
    sim_features: np.ndarray,
    sim_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float | bool | str]:
    """Train on sim features, evaluate on test features.

    M3 transparency fix: when either side has only a single class label,
    the classifier cannot be trained meaningfully and the function falls
    back to a fixed-score sentinel. The previous implementation returned
    ``accuracy=0.5`` silently, which downstream consumers (and the
    composite fidelity score) treated as a legitimate measurement. We
    now flag the result with ``degraded=True`` and a ``degradation_reason``
    string so downstream readers can see exactly which tasks hit the
    single-class ceiling and report coverage honestly in the paper.
    """
    single_class_sentinel = {
        "accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }
    if len(sim_features) == 0:
        return {**single_class_sentinel, "degraded": True, "degradation_reason": "empty_sim_features"}
    if len(set(sim_labels)) < 2:
        return {**single_class_sentinel, "degraded": True, "degradation_reason": "single_class_sim_labels"}
    if len(test_features) == 0:
        return {**single_class_sentinel, "degraded": True, "degradation_reason": "empty_test_features"}
    if len(set(test_labels)) < 2:
        return {**single_class_sentinel, "degraded": True, "degradation_reason": "single_class_test_labels"}

    min_dim = min(sim_features.shape[1], test_features.shape[1])
    sim_features = sim_features[:, :min_dim]
    test_features = test_features[:, :min_dim]

    scaler = StandardScaler()
    sim_scaled = scaler.fit_transform(sim_features)
    test_scaled = scaler.transform(test_features)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(sim_scaled, sim_labels)
    preds = clf.predict(test_scaled)

    accuracy = float(accuracy_score(test_labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "degraded": False,
        "degradation_reason": "",
    }


# ----------------------------------------------------------------------
# Per-task sim feature builders + prediction
# ----------------------------------------------------------------------

def _build_user_conflict_dataset(messages: list[dict], thread_ids):
    feats, labels = [], []
    for tid in thread_ids:
        thread_msgs = [m for m in messages if m["thread_id"] == tid]
        participants = {m["user_id"] for m in thread_msgs}
        for uid in participants:
            feats.append(extract_features(messages, tid, uid))
            has_conflict = any(
                m["action_type"] in _CONFLICT_ACTIONS
                for m in thread_msgs if m["user_id"] == uid
            )
            labels.append(1 if has_conflict else 0)
    return feats, labels


def predict_conflict(
    sim_messages: list[dict],
    ground_truth: dict[str, int],
    real_messages: list[dict],
) -> dict[str, float]:
    """Train on simulation to predict per-user conflict on held-out threads.

    Args:
        sim_messages: Simulated messages (training).
        ground_truth: {thread_id: consensus_label} from held-out annotation.
        real_messages: Real messages (for feature extraction on test set).
    """
    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _build_user_conflict_dataset(sim_messages, sim_thread_ids)

    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = [m for m in real_messages if m["thread_id"] == tid]
        participants = {m["user_id"] for m in thread_msgs}
        for uid in participants:
            test_feats.append(extract_features(real_messages, tid, uid))
            test_labels.append(label)

    return _train_and_evaluate(
        np.array(sim_feats), np.array(sim_labels),
        np.array(test_feats), np.array(test_labels),
    )


def predict_persuasion(
    sim_messages: list[dict],
    ground_truth: dict[str, int],
    real_messages: list[dict],
) -> dict[str, float]:
    """Train on simulation to predict persuasion success on held-out threads."""
    def _thread_feats(messages, thread_ids):
        feats, labels = [], []
        for tid in thread_ids:
            thread_msgs = [m for m in messages if m["thread_id"] == tid]
            if not thread_msgs:
                continue
            actions = [m["action_type"] for m in thread_msgs]
            counts = Counter(actions)
            total = len(thread_msgs)
            participants = {m["user_id"] for m in thread_msgs}
            has_persuasion = any(a in _PERSUASION_ACTIONS for a in actions)

            feats.append(np.array([
                total,
                len(participants),
                _max_chain_depth(thread_msgs),
                len(set(actions)),
                float(np.mean([len(m.get("text", "")) for m in thread_msgs])),
                counts.get("counter_argue", 0),
                sum(counts.get(a, 0) for a in _CONFLICT_ACTIONS) / total if total else 0,
            ]))
            labels.append(1 if has_persuasion else 0)
        return feats, labels

    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _thread_feats(sim_messages, sim_thread_ids)

    # Test features from real messages, labels from ground truth
    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = [m for m in real_messages if m["thread_id"] == tid]
        if not thread_msgs:
            continue
        actions = [m["action_type"] for m in thread_msgs]
        counts = Counter(actions)
        total = len(thread_msgs)
        participants = {m["user_id"] for m in thread_msgs}
        test_feats.append(np.array([
            total,
            len(participants),
            _max_chain_depth(thread_msgs),
            len(set(actions)),
            float(np.mean([len(m.get("text", "")) for m in thread_msgs])),
            counts.get("counter_argue", 0),
            sum(counts.get(a, 0) for a in _CONFLICT_ACTIONS) / total if total else 0,
        ]))
        test_labels.append(label)

    return _train_and_evaluate(
        np.array(sim_feats), np.array(sim_labels),
        np.array(test_feats), np.array(test_labels),
    )


def predict_escalation(
    sim_messages: list[dict],
    ground_truth: dict[str, int],
    real_messages: list[dict],
    conflict_threshold: float = 0.3,
) -> dict[str, float]:
    """Train on simulation to predict conflict escalation on held-out threads."""
    def _escalation_feats(messages, thread_ids):
        feats, labels = [], []
        for tid in thread_ids:
            thread_msgs = [m for m in messages if m["thread_id"] == tid]
            if not thread_msgs:
                continue
            total = len(thread_msgs)
            counts = Counter(m["action_type"] for m in thread_msgs)
            conflict_count = sum(counts.get(a, 0) for a in _ESCALATION_ACTIONS)
            conflict_ratio = conflict_count / total if total else 0
            participants = {m["user_id"] for m in thread_msgs}
            conflict_participants = {
                m["user_id"] for m in thread_msgs
                if m["action_type"] in _ESCALATION_ACTIONS
            }
            is_escalation = (
                conflict_ratio >= conflict_threshold
                and len(conflict_participants) >= 2
            )
            feats.append(np.array([
                total, len(participants), conflict_count, conflict_ratio,
                len(conflict_participants),
                counts.get("revert", 0), counts.get("report", 0),
                total / max(len(participants), 1),
            ]))
            labels.append(1 if is_escalation else 0)
        return feats, labels

    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _escalation_feats(sim_messages, sim_thread_ids)

    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = [m for m in real_messages if m["thread_id"] == tid]
        if not thread_msgs:
            continue
        total = len(thread_msgs)
        counts = Counter(m["action_type"] for m in thread_msgs)
        conflict_count = sum(counts.get(a, 0) for a in _ESCALATION_ACTIONS)
        conflict_ratio = conflict_count / total if total else 0
        participants = {m["user_id"] for m in thread_msgs}
        conflict_participants = {
            m["user_id"] for m in thread_msgs
            if m["action_type"] in _ESCALATION_ACTIONS
        }
        test_feats.append(np.array([
            total, len(participants), conflict_count, conflict_ratio,
            len(conflict_participants),
            counts.get("revert", 0), counts.get("report", 0),
            total / max(len(participants), 1),
        ]))
        test_labels.append(label)

    return _train_and_evaluate(
        np.array(sim_feats), np.array(sim_labels),
        np.array(test_feats), np.array(test_labels),
    )


# ----------------------------------------------------------------------
# Heuristic fallback (no annotations available)
# ----------------------------------------------------------------------

def _heuristic_ground_truth(
    real_messages: list[dict],
    holdout_ratio: float,
    event_type: str,
) -> dict[str, int]:
    """Derive ground-truth labels from action types when no annotations exist.

    Flags the result as heuristic — Cohen's κ is unavailable, so this does
    NOT satisfy the outline §5.3 protocol.
    """
    all_threads = sorted({m["thread_id"] for m in real_messages})
    n = max(int(len(all_threads) * holdout_ratio), 1)
    holdout = all_threads[:n]

    gt: dict[str, int] = {}
    actions_map = {
        EVENT_CONFLICT: _CONFLICT_ACTIONS,
        EVENT_PERSUASION: _PERSUASION_ACTIONS,
        EVENT_ESCALATION: _ESCALATION_ACTIONS,
    }
    target_actions = actions_map.get(event_type, _CONFLICT_ACTIONS)

    for tid in holdout:
        thread_msgs = [m for m in real_messages if m["thread_id"] == tid]
        if event_type == EVENT_ESCALATION:
            total = len(thread_msgs) or 1
            conflict = sum(
                1 for m in thread_msgs if m["action_type"] in _ESCALATION_ACTIONS
            )
            conflict_participants = {
                m["user_id"] for m in thread_msgs
                if m["action_type"] in _ESCALATION_ACTIONS
            }
            gt[tid] = 1 if (conflict / total >= 0.3 and len(conflict_participants) >= 2) else 0
        else:
            gt[tid] = 1 if any(
                m["action_type"] in target_actions for m in thread_msgs
            ) else 0
    return gt


# ----------------------------------------------------------------------
# Top-level compute
# ----------------------------------------------------------------------

class PredictiveMetrics:
    """Container for predictive fidelity metrics."""

    @staticmethod
    def compute(
        sim_messages: list[dict],
        real_messages: list[dict],
        held_out_events: list[HeldOutEvent] | None = None,
        holdout_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """Compute predictive fidelity across three prediction tasks.

        Ground-truth protocol (outline §5.3):
          - If ``held_out_events`` is provided, resolve consensus labels and
            report per-event-type Cohen's κ. Event types with κ < 0.7 are
            flagged unreliable.
          - If absent (or a specific event type yields no consensus labels),
            derive heuristic labels from action types. The returned
            ``annotation_protocol`` records per-task provenance so silent
            per-task fallbacks are visible.

        Three tasks:
          1. Conflict prediction (who will engage in conflict)
          2. Persuasion prediction (whether persuasion succeeds)
          3. Escalation prediction (whether conflict escalates)

        Args:
            sim_messages: Simulated messages (training data).
            real_messages: Real messages (test feature source).
            held_out_events: Annotated held-out events with 2 annotators.
            holdout_ratio: Used only by the heuristic fallback.

        Returns:
            Dict with per-task metrics, agreement reports, composite
            predictive fidelity, and the annotation protocol used.
        """
        result: dict[str, Any] = {}
        # Per-task protocol tracking (outline §5.3 transparency). Each of the
        # three predictive tasks may independently fall back to heuristic GT
        # when consensus labels are unavailable; reporting a single flat
        # ``annotation_protocol`` string silently hides per-task fallbacks
        # and lets downstream readers assume κ ≥ 0.7 holds for all three.
        # ``overall`` is "annotated" iff held_out_events was supplied AND all
        # three tasks used consensus labels; "mixed" if some fell back;
        # "heuristic" if no held_out_events at all.
        per_task_protocol: dict[str, str] = {
            EVENT_CONFLICT: "heuristic",
            EVENT_PERSUASION: "heuristic",
            EVENT_ESCALATION: "heuristic",
        }
        overall_protocol = "heuristic"
        agreement_reports: dict[str, Any] = {}

        if held_out_events is not None:
            agreement_reports = resolve_events(held_out_events)
            gt_conflict = consensus_ground_truth(held_out_events, EVENT_CONFLICT)
            gt_persuasion = consensus_ground_truth(held_out_events, EVENT_PERSUASION)
            gt_escalation = consensus_ground_truth(held_out_events, EVENT_ESCALATION)

            # Fall back to heuristic for any event type with no consensus labels.
            # Track which tasks fell back so consumers can see κ coverage honestly.
            def _resolve_or_fallback(gt, event_type: str, name: str):
                if gt:
                    per_task_protocol[event_type] = "annotated"
                    return gt
                logger.warning(
                    f"No consensus {name} labels; using heuristic fallback "
                    f"(κ NOT reported for this task)"
                )
                per_task_protocol[event_type] = "heuristic"
                return _heuristic_ground_truth(real_messages, holdout_ratio, event_type)

            gt_conflict = _resolve_or_fallback(gt_conflict, EVENT_CONFLICT, "conflict")
            gt_persuasion = _resolve_or_fallback(gt_persuasion, EVENT_PERSUASION, "persuasion")
            gt_escalation = _resolve_or_fallback(gt_escalation, EVENT_ESCALATION, "escalation")

            distinct = set(per_task_protocol.values())
            overall_protocol = "annotated" if distinct == {"annotated"} else "mixed"
        else:
            logger.warning(
                "No held_out_events provided; Predictive Fidelity uses heuristic "
                "labels (Cohen's κ unavailable, outline §5.3 protocol NOT met)."
            )
            gt_conflict = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_CONFLICT)
            gt_persuasion = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_PERSUASION)
            gt_escalation = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_ESCALATION)

        conflict_result = predict_conflict(sim_messages, gt_conflict, real_messages)
        persuasion_result = predict_persuasion(sim_messages, gt_persuasion, real_messages)
        escalation_result = predict_escalation(sim_messages, gt_escalation, real_messages)

        # M3 transparency: surface which of the three tasks hit the
        # single-class / empty-feature ceiling so consumers can see
        # whether the reported accuracy/f1 numbers are real measurements
        # or fixed sentinels. Downstream readers (and reviewers) can no
        # longer mistake a 0.5 sentinel for a measured chance-rate.
        per_task_results = {
            EVENT_CONFLICT: conflict_result,
            EVENT_PERSUASION: persuasion_result,
            EVENT_ESCALATION: escalation_result,
        }
        degraded_tasks: dict[str, str] = {}
        n_usable = 0
        for et, r in per_task_results.items():
            if r.get("degraded", False):
                degraded_tasks[et] = r.get("degradation_reason", "unknown")
            else:
                n_usable += 1

        # Composite fidelity uses only non-degraded tasks; falls back to
        # 0.0 when all three degraded so the composite does not silently
        # read as "0.5 chance-rate success".
        usable_f1 = [r["f1"] for r in per_task_results.values() if not r.get("degraded", False)]
        usable_acc = [r["accuracy"] for r in per_task_results.values() if not r.get("degraded", False)]
        if usable_f1:
            composite_f1 = float(np.mean(usable_f1))
            composite_accuracy = float(np.mean(usable_acc))
            composite = (composite_accuracy + composite_f1) / 2
        else:
            composite_f1 = 0.0
            composite_accuracy = 0.0
            composite = 0.0

        result.update({
            "conflict_prediction": conflict_result,
            "persuasion_prediction": persuasion_result,
            "escalation_prediction": escalation_result,
            "predictive_fidelity": composite,
            "predictive_fidelity_n_usable_tasks": n_usable,
            "predictive_fidelity_degraded_tasks": degraded_tasks,
            "annotation_protocol": {
                # ``overall`` collapses to a single label for quick filtering;
                # ``per_task`` is the authoritative, transparent record showing
                # which of conflict/persuasion/escalation actually had κ-validated
                # consensus labels vs heuristic fallback.
                "overall": overall_protocol,
                "per_task": per_task_protocol,
            },
            "inter_annotator_agreement": {
                et: r.to_dict() for et, r in agreement_reports.items()
            },
        })
        return result
