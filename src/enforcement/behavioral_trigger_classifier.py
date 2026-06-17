"""Behavioral trigger classifier for Tier 3 Category C (outline §4.4.1).

The outline specifies Category C as "lightweight logistic regression over a
behavioral feature vector (stance shift rate, conflict engagement ratio, etc.)".
The legacy implementation used action-sequence substring matching, which is
both brittle (``"agree"`` matches ``"disagree"``) and not what the paper
claims. This module provides the spec'd classifier and integrates with
``Tier3AntiPatternBlock._check_behavioral`` via a graceful fallback: when
no trained model is available, the legacy substring path is preserved
(back-compat for untrained deployments / unit tests).

Feature vector (5 dims, lightweight by design — outline §4.4.1 "lightweight"):

  1. ``conflict_engagement_ratio`` — fraction of recent actions that are
     conflict-type (DISAGREE, REVERT, COUNTER_ARGUE, REPORT, BLOCK).
  2. ``stance_shift_rate`` — rate of DISAGREE→AGREE / AGREE→DISAGREE
     transitions (deliberately stance-specific, fixing the overcounting
     flagged in code review m1 on the legacy features.py implementation).
  3. ``delta_award_rate`` — fraction of AWARD_DELTA actions (Reddit-specific
     persuasion-success signal; zero on Wikipedia / GitHub).
  4. ``action_entropy`` — Shannon entropy of the recent action-type
     distribution (low entropy → repetitive / stuck behavior).
  5. ``recent_length`` — number of recent actions available (context window
     saturation proxy).

Training data: the classifier is trained during §5.3.5 Trigger Calibration
on the 60% train split of per-dataset annotated interactions. The label is
``1`` if the annotation marked the interaction as a Category-C anti-pattern
violation, else ``0``. Saved via ``joblib`` to
``data/trigger_calibration/{dataset}.behavioral_clf.joblib`` and loaded
lazily by ``Tier3AntiPatternBlock``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from loguru import logger
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Conflict / aggressive action types (kept in sync with src/data/schemas.py ActionType)
_CONFLICT_ACTIONS: frozenset[str] = frozenset({
    "disagree", "revert", "counter_argue", "report", "block",
})

# Stance-related action types (for stance-shift-rate feature)
_STANCE_POSITIVE: frozenset[str] = frozenset({"agree", "award_delta"})
_STANCE_NEGATIVE: frozenset[str] = frozenset({"disagree", "revert", "counter_argue"})

DEFAULT_WINDOW = 10
DEFAULT_FEATURE_DIM = 5


def extract_behavioral_features(
    action_history: Sequence[str],
    window: int = DEFAULT_WINDOW,
) -> np.ndarray:
    """Compute the 5-dim behavioral feature vector from recent action history.

    Args:
        action_history: Chronological list of action-type strings (lowercase).
        window: Number of recent actions to consider.

    Returns:
        Shape-(5,) float array. Zero-vector if action_history is empty.
    """
    if not action_history:
        return np.zeros(DEFAULT_FEATURE_DIM, dtype=float)

    recent = [str(a).lower() for a in action_history[-window:]]
    n = max(len(recent), 1)

    # Feature 1: conflict engagement ratio
    conflict_count = sum(1 for a in recent if a in _CONFLICT_ACTIONS)
    conflict_ratio = conflict_count / n

    # Feature 2: stance shift rate (DISAGREE↔AGREE transitions only —
    # fixes the legacy overcounting flagged in code review m1)
    stance_shifts = 0
    for prev, curr in zip(recent, recent[1:]):
        if (prev in _STANCE_POSITIVE and curr in _STANCE_NEGATIVE) or \
           (prev in _STANCE_NEGATIVE and curr in _STANCE_POSITIVE):
            stance_shifts += 1
    stance_shift_rate = stance_shifts / max(n - 1, 1)

    # Feature 3: delta award rate (Reddit persuasion-success signal)
    delta_count = sum(1 for a in recent if a == "award_delta")
    delta_rate = delta_count / n

    # Feature 4: action entropy (bits). Low → repetitive.
    counts = Counter(recent)
    probs = np.array(list(counts.values()), dtype=float) / n
    probs = probs[probs > 0]
    action_entropy = float(scipy_entropy(probs, base=2)) if probs.size else 0.0

    # Feature 5: window saturation (proxy for how much context we have)
    recent_length = len(recent) / float(window)

    return np.array([
        conflict_ratio,
        stance_shift_rate,
        delta_rate,
        action_entropy,
        recent_length,
    ], dtype=float)


class BehavioralTriggerClassifier:
    """Lightweight logistic regression over the behavioral feature vector.

    Per outline §4.4.1: trained on labeled (action_history → violation) pairs
    from the §5.3.5 trigger calibration data. Predicts P(violation | recent
    action history). Tier 3 compares this probability to a per-anti-pattern
    threshold (default 0.5, tunable per trigger).
    """

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.scaler = StandardScaler()
        self.trained: bool = False
        # Per-feature training-set means for fallback when scaler not yet fit
        self._feature_means = np.zeros(DEFAULT_FEATURE_DIM)

    def featurize(self, action_history: Sequence[str]) -> np.ndarray:
        return extract_behavioral_features(action_history, window=self.window)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BehavioralTriggerClassifier":
        """Fit the classifier on labeled behavioral feature vectors.

        Args:
            X: Shape (n_samples, DEFAULT_FEATURE_DIM) feature matrix.
            y: Shape (n_samples,) binary labels {0, 1}.

        Returns:
            self (for chaining).
        """
        if X.shape[0] < 2:
            raise ValueError(f"Need ≥2 samples to train, got {X.shape[0]}")
        self._feature_means = X.mean(axis=0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        return self

    def fit_from_histories(
        self,
        action_histories: Sequence[Sequence[str]],
        labels: Sequence[int],
    ) -> "BehavioralTriggerClassifier":
        """Convenience: fit from raw action histories + labels."""
        X = np.stack([self.featurize(h) for h in action_histories])
        y = np.asarray(labels, dtype=int)
        return self.fit(X, y)

    def predict_proba(self, action_history: Sequence[str]) -> float:
        """P(violation | recent action history). 0.0 if untrained."""
        if not self.trained:
            return 0.0
        feats = self.featurize(action_history).reshape(1, -1)
        # Defensive: if scaler/model disagree on dim, fall back to 0.0
        try:
            scaled = self.scaler.transform(feats)
            proba = self.model.predict_proba(scaled)
            # Column 1 = P(class=1) if both classes present during training
            if proba.shape[1] < 2:
                return 0.0
            return float(proba[0, 1])
        except Exception as exc:  # noqa: BLE001 — never crash enforcement
            logger.warning(f"BehavioralTriggerClassifier.predict_proba failed: {exc}")
            return 0.0

    def save(self, path: str | Path) -> None:
        """Persist via joblib. Includes trained flag + feature means."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "trained": self.trained,
            "feature_means": self._feature_means,
            "window": self.window,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "BehavioralTriggerClassifier":
        """Load a saved classifier. Returns an untrained instance if load fails."""
        path = Path(path)
        try:
            data = joblib.load(path)
            clf = cls(window=data.get("window", DEFAULT_WINDOW))
            clf.model = data["model"]
            clf.scaler = data["scaler"]
            clf.trained = bool(data.get("trained", False))
            clf._feature_means = np.asarray(
                data.get("feature_means", np.zeros(DEFAULT_FEATURE_DIM))
            )
            return clf
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to load BehavioralTriggerClassifier from {path}: {exc} "
                f"— Tier 3 Category C will fall back to legacy substring matching"
            )
            return cls()  # untrained — caller's fallback path handles it
