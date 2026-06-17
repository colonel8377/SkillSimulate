"""Base enforcement interface with α hardness parameter.

α ∈ [0, 1]:
  α=0: advisory only (degrades to Descriptive Persona baseline)
  α=0.5: soft filter (distance/confidence-proportional probabilistic enforcement)
  α=1.0: hard block (always enforced)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnforcementResult:
    """Result of an enforcement check."""
    passed: bool                        # True if text/action passes the check
    tier: str                           # which tier triggered ("tier1", "tier2", "tier3", "none")
    reason: str = ""                    # why it was blocked (if applicable)
    modified_messages: list[dict] | None = None  # modified LLM messages (for tier2 injection)
    original_text: str = ""             # original generated text
    regenerated_text: str = ""          # regenerated text after enforcement
    injection_text: str = ""            # text injected into prompt (for planner context)


class EnforcementStrategy(ABC):
    """Abstract base for enforcement strategies.

    Reproducibility (R1): every probabilistic gate (``_should_enforce``,
    ``_rejection_probability``, ``_block_probability``) draws from a
    per-instance ``random.Random`` rather than the global module RNG. The
    caller threads a deterministic seed from the experiment cell through
    the harness to each tier; tiers derive independent sub-RNGs so their
    streams do not correlate. When ``rng`` is None, behaviour falls back
    to a fresh unseeded RNG (legacy, for unit tests).
    """

    def __init__(self, alpha: float = 1.0, rng: random.Random | None = None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        # R1: per-instance seeded RNG so identical cell seeds reproduce
        # identical stochastic gating across runs.
        self._rng = rng if rng is not None else random.Random()

    def _should_enforce(self) -> bool:
        """Decide whether to enforce based on α (uniform probability gate).

        Used for Tier 2 (context injection) where enforcement is binary.
        """
        if self.alpha >= 1.0:
            return True
        if self.alpha <= 0.0:
            return False
        return self._rng.random() < self.alpha

    def _rejection_probability(self, distance_ratio: float) -> bool:
        """Distance-proportional rejection for Tier 1 (outline §4.4.3).

        When α < 1.0, P(reject) scales with how far the output deviates:
            P(reject) = α × min(distance_ratio, 1.0)

        where distance_ratio = max_z / sigma_threshold (how many σ beyond boundary).

        At α=1.0 this becomes deterministic: always reject if beyond boundary.
        At α=0.5, a 2σ violation → P(reject)=1.0; a 1σ violation → P(reject)=0.5.

        Args:
            distance_ratio: How far beyond the boundary (0 = at boundary, 1+ = far beyond).

        Returns:
            True if this instance should be rejected.
        """
        if self.alpha >= 1.0:
            return True  # hard block
        if self.alpha <= 0.0:
            return False
        p = self.alpha * min(distance_ratio, 1.0)
        return self._rng.random() < p

    def _block_probability(self, confidence: float) -> bool:
        """Confidence-proportional blocking for Tier 3 (outline §4.4.3).

        When α < 1.0, P(block) scales with trigger confidence:
            P(block) = α × confidence

        At α=1.0 this becomes deterministic: always block if triggered.
        At α=0.5, a 0.9-confidence match → P(block)=0.45.

        Args:
            confidence: Trigger confidence in [0, 1].

        Returns:
            True if this instance should be blocked.
        """
        if self.alpha >= 1.0:
            return True  # hard block
        if self.alpha <= 0.0:
            return False
        p = self.alpha * confidence
        return self._rng.random() < p

    @abstractmethod
    async def check_pre_generation(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Check/modify messages BEFORE generation (Tier 2, 3).

        Args:
            messages: Current LLM message list.
            context: Enforcement context (skill, thread state, etc).

        Returns:
            EnforcementResult with potentially modified messages.
        """

    @abstractmethod
    async def check_post_generation(
        self,
        text: str,
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Check generated text AFTER generation (Tier 1).

        Args:
            text: Generated text.
            context: Enforcement context.

        Returns:
            EnforcementResult with pass/fail and potential regeneration.
        """
