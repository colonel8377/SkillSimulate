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


import re as _re

_CONFLICT_ACTIONS = {"disagree", "revert", "counter_argue", "report"}
_PERSUASION_ACTIONS = {"award_delta"}
# B2 fix: ``close``/``reopen`` are GitHub issue-lifecycle actions (outline
# §4.3), not escalation signals — including them inflated the GitHub
# dataset's escalation labels. Kept in sync with sandbox conflict_actions.
_ESCALATION_ACTIONS = {
    "disagree", "revert", "counter_argue", "report", "block",
}

# ----------------------------------------------------------------------
# Text-content-based event detection (R3 fix for single_class_sim_labels)
#
# When agents only produce platform-default actions (e.g. ``discuss`` on
# Wikipedia, ``reply`` on Reddit), action-type-based labelling collapses
# to a single class (all-0 for conflict/persuasion).  Text-content
# heuristics detect the *semantic* signal regardless of action type,
# producing multi-class training labels from simulation data.
# ----------------------------------------------------------------------

_CONFLICT_PATTERNS = _re.compile(
    r"\b(?:i disagree|that.s (?:wrong|incorrect|not right|misleading|"
    r"inaccurate|untrue)|"
    r"no evidence|unsourced|original research|"
    r"i object|strongly disagree|oppose|"
    r"not (?:true|accurate|correct|supported)|"
    r"this is (?:wrong|false|misleading)|"
    r"revert(?:ing|ed)?|vandalism|"
    r"undid|rolled back|removed your)\b",
    _re.IGNORECASE,
)
_PERSUASION_PATTERNS = _re.compile(
    r"\b(?:i (?:agree|concede|see your point|stand corrected|"
    r"was wrong|changed my mind|accept)|"
    r"you.re (?:right|correct)|"
    r"good point|fair point|well said|"
    r"that makes sense|makes sense|"
    r"updated (?:the|my)|added (?:a |the )?citation|"
    r"i.ll (?:change|update|fix|add)|"
    r"noted|acknowledged|"
    r"thanks? for (?:the|your|clarif|point|feedback)|"
    r"convinced|persuaded|"
    r"delta|award)\b",
    _re.IGNORECASE,
)
_ESCALATION_PATTERNS = _re.compile(
    r"\b(?:you always|you never|this is (?:absurd|ridiculous|unacceptable)|"
    r"stop (?:reverting|editing|removing)|"
    r"i (?:will |am going to )?(?:report|block|ban)|"
    r"vandal|edit.war|"
    r"personal attack|ad hominem|"
    r"get lost|go away|"
    r"i.m (?:done|leaving)|"
    r"last warning|final warning)\b",
    _re.IGNORECASE,
)


def _text_has_conflict(text: str) -> bool:
    """Check if message text contains disagreement/opposition markers."""
    return bool(_CONFLICT_PATTERNS.search(text or ""))


def _text_has_persuasion(text: str) -> bool:
    """Check if message text contains concession/agreement markers."""
    return bool(_PERSUASION_PATTERNS.search(text or ""))


def _text_has_escalation(text: str) -> bool:
    """Check if message text contains escalation/intensification markers."""
    return bool(_ESCALATION_PATTERNS.search(text or ""))


def _user_has_conflict_signal(
    messages: list[dict],
    user_id: str,
    thread_msgs: list[dict] | None = None,
) -> bool:
    """Check if a user exhibits conflict signals via action type OR text.

    R3 fix: combines action-type matching (legacy) with text-content
    keyword detection so that sim data with only ``discuss`` actions
    can still produce positive conflict labels when the text content
    contains disagreement markers.
    """
    user_msgs = [m for m in messages if m["user_id"] == user_id]
    if not user_msgs:
        return False
    # Action-type signal
    if any(m["action_type"] in _CONFLICT_ACTIONS for m in user_msgs):
        return True
    # Text-content signal
    return any(_text_has_conflict(m.get("text", "")) for m in user_msgs)


def _thread_has_persuasion_signal(thread_msgs: list[dict]) -> bool:
    """Check if a thread exhibits persuasion via action type OR text."""
    if any(m["action_type"] in _PERSUASION_ACTIONS for m in thread_msgs):
        return True
    return any(_text_has_persuasion(m.get("text", "")) for m in thread_msgs)


def _thread_is_escalating(
    thread_msgs: list[dict],
    conflict_threshold: float = 0.3,
) -> bool:
    """Check if a thread is escalating via action types OR text.

    Combines action-type conflict ratio with text-content escalation
    markers, requiring at least 2 participants with conflict signals.
    """
    total = len(thread_msgs) or 1
    # Action-type based conflict count
    action_conflict = sum(
        1 for m in thread_msgs if m["action_type"] in _ESCALATION_ACTIONS
    )
    # Text-based conflict count (union with action-based)
    text_conflict = sum(
        1 for m in thread_msgs
        if _text_has_conflict(m.get("text", ""))
        or m["action_type"] in _ESCALATION_ACTIONS
    )
    conflict_count = max(action_conflict, text_conflict)
    conflict_ratio = conflict_count / total

    conflict_participants = {
        m["user_id"] for m in thread_msgs
        if m["action_type"] in _ESCALATION_ACTIONS
        or _text_has_conflict(m.get("text", ""))
    }

    if conflict_ratio >= conflict_threshold and len(conflict_participants) >= 2:
        return True
    # Also check for explicit escalation markers
    escalation_users = {
        m["user_id"] for m in thread_msgs
        if _text_has_escalation(m.get("text", ""))
    }
    return len(escalation_users) >= 2


# ----------------------------------------------------------------------
# Structural contention scoring (R3 fallback for polite-agent simulations)
#
# When agents produce only polite/collaborative text (e.g. CADP agents
# with strong Expression DNA), keyword-based detection may not fire.
# Structural signals provide a continuous contention metric that can
# be split at the median to guarantee multi-class labels.
# ----------------------------------------------------------------------

_QUESTION_RE = _re.compile(r"\?")
_HEDGE_RE = _re.compile(
    r"\b(?:i think|perhaps|maybe|possibly|not sure|unsure|"
    r"i.m not certain|i believe|it seems|might|i suppose|"
    r"could you|would you)\b",
    _re.IGNORECASE,
)


def _user_contention_score(
    thread_msgs: list[dict], user_id: str
) -> float:
    """Compute a continuous contention score for a user in a thread.

    Signals:
    - Reply density (how often this user replies to others)
    - Question frequency (questions indicate challenge / inquiry)
    - Hedging frequency (hedges indicate disagreement-in-progress)
    - Relative message length (longer replies = more engaged)
    """
    user_msgs = [m for m in thread_msgs if m["user_id"] == user_id]
    if not user_msgs:
        return 0.0

    # Reply density: fraction of user's messages that are replies
    n_replies = sum(1 for m in user_msgs if m.get("parent_msg_id"))
    reply_density = n_replies / len(user_msgs)

    # Question frequency
    n_questions = sum(
        1 for m in user_msgs if _QUESTION_RE.search(m.get("text", ""))
    )
    question_freq = n_questions / len(user_msgs)

    # Hedging frequency
    n_hedges = sum(
        1 for m in user_msgs if _HEDGE_RE.search(m.get("text", ""))
    )
    hedge_freq = n_hedges / len(user_msgs)

    # Relative message length (avg chars / thread avg chars)
    avg_user_len = float(np.mean([len(m.get("text", "")) for m in user_msgs]))
    avg_thread_len = float(np.mean([len(m.get("text", "")) for m in thread_msgs]))
    rel_length = avg_user_len / max(avg_thread_len, 1)

    return reply_density + question_freq + hedge_freq + rel_length


def _thread_contention_score(thread_msgs: list[dict]) -> float:
    """Compute a continuous contention score for a thread.

    Signals:
    - Number of distinct participants
    - Cross-reply density (replies to different users)
    - Average message length (longer = more engaged)
    - Action diversity (more action types = more contention)
    """
    if not thread_msgs:
        return 0.0

    participants = {m["user_id"] for m in thread_msgs}
    n_participants = len(participants)

    # Cross-reply density
    parent_map = {m["msg_id"]: m for m in thread_msgs}
    cross_replies = 0
    for m in thread_msgs:
        pid = m.get("parent_msg_id")
        if pid and pid in parent_map:
            parent = parent_map[pid]
            if parent["user_id"] != m["user_id"]:
                cross_replies += 1
    cross_reply_ratio = cross_replies / len(thread_msgs)

    avg_len = float(np.mean([len(m.get("text", "")) for m in thread_msgs]))
    action_diversity = len(set(m["action_type"] for m in thread_msgs))

    return (
        n_participants
        + cross_reply_ratio * 5
        + avg_len / 100
        + action_diversity
    )


def _ensure_multiclass_labels(
    labels: list[int],
    scores: list[float],
) -> list[int]:
    """Ensure labels have both classes via median-split fallback.

    When keyword-based labelling produces single-class labels (all-0),
    falls back to a median split on continuous scores to guarantee
    multi-class training data.  This is transparent — the caller can
    check whether the fallback was used.
    """
    if len(set(labels)) >= 2:
        return labels  # already multi-class
    if not scores or len(set(scores)) < 2:
        return labels  # cannot split without score variance

    median_score = float(np.median(scores))
    # Assign label=1 to above-median, 0 to at-or-below-median
    return [1 if s > median_score else 0 for s in scores]


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------

def _build_thread_index(messages: list[dict]) -> dict[str, list[dict]]:
    """Build {thread_id: [messages]} index for O(1) thread lookup."""
    index: dict[str, list[dict]] = {}
    for m in messages:
        index.setdefault(m["thread_id"], []).append(m)
    return index


def extract_features(messages: list[dict], thread_id: str, user_id: str,
                     thread_index: dict[str, list[dict]] | None = None) -> np.ndarray:
    """Extract behavioral features for a user in a thread."""
    if thread_index is not None:
        thread_msgs = thread_index.get(thread_id, [])
    else:
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
    """Build per-user conflict dataset with text-content labelling (R3 fix).

    Labels combine action-type signals AND text-content keyword detection
    so that simulation data with only default actions (e.g. ``discuss``)
    can still produce meaningful multi-class training labels.

    When keyword-based labelling yields single-class labels, a structural
    contention score fallback (median split) guarantees multi-class output.
    """
    feats, labels, scores = [], [], []
    for tid in thread_ids:
        thread_msgs = [m for m in messages if m["thread_id"] == tid]
        participants = {m["user_id"] for m in thread_msgs}
        for uid in participants:
            feats.append(extract_features(messages, tid, uid))
            labels.append(
                1 if _user_has_conflict_signal(thread_msgs, uid) else 0
            )
            scores.append(_user_contention_score(thread_msgs, uid))
    # R3 fallback: ensure multi-class via median split
    labels = _ensure_multiclass_labels(labels, scores)
    return feats, labels


def predict_conflict(
    sim_messages: list[dict],
    ground_truth: dict[str, int],
    real_messages: list[dict],
    real_thread_index: dict[str, list[dict]] | None = None,
) -> dict[str, float]:
    """Train on simulation to predict per-user conflict on held-out threads.

    Args:
        sim_messages: Simulated messages (training).
        ground_truth: {thread_id: consensus_label} from held-out annotation.
        real_messages: Real messages (for feature extraction on test set).
        real_thread_index: Pre-built {thread_id: [msgs]} index for O(1) lookup.
    """
    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _build_user_conflict_dataset(sim_messages, sim_thread_ids)

    if real_thread_index is None:
        real_thread_index = _build_thread_index(real_messages)

    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = real_thread_index.get(tid, [])
        participants = {m["user_id"] for m in thread_msgs}
        for uid in participants:
            test_feats.append(extract_features(real_messages, tid, uid, real_thread_index))
            test_labels.append(label)

    return _train_and_evaluate(
        np.array(sim_feats), np.array(sim_labels),
        np.array(test_feats), np.array(test_labels),
    )


def predict_persuasion(
    sim_messages: list[dict],
    ground_truth: dict[str, int],
    real_messages: list[dict],
    real_thread_index: dict[str, list[dict]] | None = None,
) -> dict[str, float]:
    """Train on simulation to predict persuasion success on held-out threads."""
    def _thread_feats(messages, thread_ids):
        feats, labels, scores = [], [], []
        idx = _build_thread_index(messages)
        for tid in thread_ids:
            thread_msgs = idx.get(tid, [])
            if not thread_msgs:
                continue
            actions = [m["action_type"] for m in thread_msgs]
            counts = Counter(actions)
            total = len(thread_msgs)
            participants = {m["user_id"] for m in thread_msgs}
            # R3 fix: text-content persuasion detection alongside action types
            has_persuasion = _thread_has_persuasion_signal(thread_msgs)

            # Add text-persuasion feature to the feature vector
            n_persuasion_text = sum(
                1 for m in thread_msgs
                if _text_has_persuasion(m.get("text", ""))
            )
            feats.append(np.array([
                total,
                len(participants),
                _max_chain_depth(thread_msgs),
                len(set(actions)),
                float(np.mean([len(m.get("text", "")) for m in thread_msgs])),
                counts.get("counter_argue", 0),
                sum(counts.get(a, 0) for a in _CONFLICT_ACTIONS) / total if total else 0,
                n_persuasion_text,
            ]))
            labels.append(1 if has_persuasion else 0)
            scores.append(_thread_contention_score(thread_msgs))
        # R3 fallback: ensure multi-class via median split
        labels = _ensure_multiclass_labels(labels, scores)
        return feats, labels

    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _thread_feats(sim_messages, sim_thread_ids)

    if real_thread_index is None:
        real_thread_index = _build_thread_index(real_messages)

    # R3 fix: test features must match sim feature dimensionality
    # (added n_persuasion_text feature)
    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = real_thread_index.get(tid, [])
        if not thread_msgs:
            continue
        actions = [m["action_type"] for m in thread_msgs]
        counts = Counter(actions)
        total = len(thread_msgs)
        participants = {m["user_id"] for m in thread_msgs}
        n_persuasion_text = sum(
            1 for m in thread_msgs
            if _text_has_persuasion(m.get("text", ""))
        )
        test_feats.append(np.array([
            total,
            len(participants),
            _max_chain_depth(thread_msgs),
            len(set(actions)),
            float(np.mean([len(m.get("text", "")) for m in thread_msgs])),
            counts.get("counter_argue", 0),
            sum(counts.get(a, 0) for a in _CONFLICT_ACTIONS) / total if total else 0,
            n_persuasion_text,
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
    real_thread_index: dict[str, list[dict]] | None = None,
) -> dict[str, float]:
    """Train on simulation to predict conflict escalation on held-out threads."""
    def _escalation_feats(messages, thread_ids):
        feats, labels, scores = [], [], []
        idx = _build_thread_index(messages)
        for tid in thread_ids:
            thread_msgs = idx.get(tid, [])
            if not thread_msgs:
                continue
            total = len(thread_msgs)
            counts = Counter(m["action_type"] for m in thread_msgs)
            # R3 fix: combine action-type and text-content conflict
            conflict_count = sum(
                1 for m in thread_msgs
                if m["action_type"] in _ESCALATION_ACTIONS
                or _text_has_conflict(m.get("text", ""))
            )
            conflict_ratio = conflict_count / total if total else 0
            participants = {m["user_id"] for m in thread_msgs}
            conflict_participants = {
                m["user_id"] for m in thread_msgs
                if m["action_type"] in _ESCALATION_ACTIONS
                or _text_has_conflict(m.get("text", ""))
            }
            n_escalation_text = sum(
                1 for m in thread_msgs
                if _text_has_escalation(m.get("text", ""))
            )
            is_escalation = _thread_is_escalating(
                thread_msgs, conflict_threshold=conflict_threshold
            )
            feats.append(np.array([
                total, len(participants), conflict_count, conflict_ratio,
                len(conflict_participants),
                counts.get("revert", 0), counts.get("report", 0),
                total / max(len(participants), 1),
                n_escalation_text,
            ]))
            labels.append(1 if is_escalation else 0)
            scores.append(_thread_contention_score(thread_msgs))
        # R3 fallback: ensure multi-class via median split
        labels = _ensure_multiclass_labels(labels, scores)
        return feats, labels

    sim_thread_ids = list({m["thread_id"] for m in sim_messages})
    sim_feats, sim_labels = _escalation_feats(sim_messages, sim_thread_ids)

    if real_thread_index is None:
        real_thread_index = _build_thread_index(real_messages)

    # R3 fix: test features must match sim feature dimensionality
    # (added n_escalation_text feature)
    test_feats, test_labels = [], []
    for tid, label in ground_truth.items():
        thread_msgs = real_thread_index.get(tid, [])
        if not thread_msgs:
            continue
        total = len(thread_msgs)
        counts = Counter(m["action_type"] for m in thread_msgs)
        conflict_count = sum(
            1 for m in thread_msgs
            if m["action_type"] in _ESCALATION_ACTIONS
            or _text_has_conflict(m.get("text", ""))
        )
        conflict_ratio = conflict_count / total if total else 0
        participants = {m["user_id"] for m in thread_msgs}
        conflict_participants = {
            m["user_id"] for m in thread_msgs
            if m["action_type"] in _ESCALATION_ACTIONS
            or _text_has_conflict(m.get("text", ""))
        }
        n_escalation_text = sum(
            1 for m in thread_msgs
            if _text_has_escalation(m.get("text", ""))
        )
        test_feats.append(np.array([
            total, len(participants), conflict_count, conflict_ratio,
            len(conflict_participants),
            counts.get("revert", 0), counts.get("report", 0),
            total / max(len(participants), 1),
            n_escalation_text,
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
    thread_index: dict[str, list[dict]] | None = None,
) -> dict[str, int]:
    """Derive ground-truth labels from action types when no annotations exist.

    Flags the result as heuristic — Cohen's κ is unavailable, so this does
    NOT satisfy the outline §5.3 protocol.
    """
    if thread_index is None:
        thread_index = _build_thread_index(real_messages)
    all_threads = sorted(thread_index.keys())
    n = max(int(len(all_threads) * holdout_ratio), 1)
    holdout = all_threads[:n]

    labels_list: list[int] = []
    scores_list: list[float] = []
    tids_ordered: list[str] = []

    for tid in holdout:
        thread_msgs = thread_index.get(tid, [])
        if event_type == EVENT_ESCALATION:
            label = 1 if _thread_is_escalating(
                thread_msgs, conflict_threshold=0.3
            ) else 0
        elif event_type == EVENT_PERSUASION:
            label = 1 if _thread_has_persuasion_signal(thread_msgs) else 0
        else:
            # Conflict: at least one user in the thread has conflict signal
            has_conflict = any(
                _user_has_conflict_signal(thread_msgs, uid)
                for uid in {m["user_id"] for m in thread_msgs}
            )
            label = 1 if has_conflict else 0
        labels_list.append(label)
        scores_list.append(_thread_contention_score(thread_msgs))
        tids_ordered.append(tid)

    # R3 fallback: ensure multi-class labels via median split
    labels_list = _ensure_multiclass_labels(labels_list, scores_list)
    return {tid: lbl for tid, lbl in zip(tids_ordered, labels_list)}


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
        # Build thread index once for O(1) lookup across all operations
        real_thread_index = _build_thread_index(real_messages)
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
                return _heuristic_ground_truth(real_messages, holdout_ratio, event_type, real_thread_index)

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
            gt_conflict = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_CONFLICT, real_thread_index)
            gt_persuasion = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_PERSUASION, real_thread_index)
            gt_escalation = _heuristic_ground_truth(real_messages, holdout_ratio, EVENT_ESCALATION, real_thread_index)

        conflict_result = predict_conflict(sim_messages, gt_conflict, real_messages, real_thread_index)
        persuasion_result = predict_persuasion(sim_messages, gt_persuasion, real_messages, real_thread_index)
        escalation_result = predict_escalation(sim_messages, gt_escalation, real_messages, real_thread_index=real_thread_index)

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
