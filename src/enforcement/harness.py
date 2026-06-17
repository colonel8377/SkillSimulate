"""Composite enforcement harness combining all three tiers.

Execution order: Tier 3 (pre) → Tier 2 (pre) → generation → Tier 1 (post)
α parameter controls enforcement strength across all tiers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.enforcement.base import EnforcementResult, EnforcementStrategy
from src.enforcement.tier1_filter import Tier1ExpressionFilter
from src.enforcement.tier2_injection import Tier2MindModelInjection
from src.enforcement.tier3_block import Tier3AntiPatternBlock
from src.llm.client import LLMClient
from src.skill.schema import SkillFile


@dataclass
class EnforcementLog:
    """Log of all enforcement actions for a single generation."""
    tier3_pre: EnforcementResult | None = None
    tier2_pre: EnforcementResult | None = None
    tier3_post: EnforcementResult | None = None
    tier1_post: EnforcementResult | None = None
    total_violations: int = 0
    regenerated: bool = False
    # True when the post-gen Forced Reformulation Protocol (outline §4.4.2)
    # was invoked — i.e. a Tier 3 post-gen violation triggered replanning.
    # This is the "hard-block-with-regeneration" event the paper reports as
    # the structural RLHF-override signal.
    tier3_hard_block_triggered: bool = False
    # True when the safe-template fallback (§4.4.2 step 4) was used after
    # max_reformulation_retries was exhausted.
    safe_template_fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "tier3_pre": {"tier": self.tier3_pre.tier, "passed": self.tier3_pre.passed, "reason": self.tier3_pre.reason} if self.tier3_pre else None,
            "tier2_pre": {"tier": self.tier2_pre.tier, "passed": self.tier2_pre.passed} if self.tier2_pre else None,
            "tier3_post": {"tier": self.tier3_post.tier, "passed": self.tier3_post.passed, "reason": self.tier3_post.reason} if self.tier3_post else None,
            "tier1_post": {"tier": self.tier1_post.tier, "passed": self.tier1_post.passed, "reason": self.tier1_post.reason} if self.tier1_post else None,
            "total_violations": self.total_violations,
            "regenerated": self.regenerated,
            "tier3_hard_block_triggered": self.tier3_hard_block_triggered,
            "safe_template_fallback_used": self.safe_template_fallback_used,
        }


class EnforcementHarness:
    """Composite enforcement combining Tier 1, 2, 3."""

    def __init__(
        self,
        alpha: float = 1.0,
        skill: SkillFile | None = None,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        enable_tier1: bool = True,
        enable_tier2: bool = True,
        enable_tier3: bool = True,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.alpha_tier1 = alpha_tier1 if alpha_tier1 is not None else alpha
        self.alpha_tier2 = alpha_tier2 if alpha_tier2 is not None else alpha
        self.alpha_tier3 = alpha_tier3 if alpha_tier3 is not None else alpha
        self.skill = skill
        self.llm = llm_client
        self.model_name = model_name

        # R1 reproducibility: derive three independent sub-RNGs from the
        # caller-supplied seed so each tier's stochastic gate is decoupled
        # yet deterministic. When ``seed`` is None we fall back to fresh
        # unseeded RNGs (legacy behaviour, e.g. unit tests).
        if seed is not None:
            master = random.Random(seed)
            t1_seed = master.randrange(2**31)
            t2_seed = master.randrange(2**31)
            t3_seed = master.randrange(2**31)
            rng_t1 = random.Random(t1_seed)
            rng_t2 = random.Random(t2_seed)
            rng_t3 = random.Random(t3_seed)
        else:
            rng_t1 = rng_t2 = rng_t3 = None

        # Initialize tiers (disabled tiers use alpha=0)
        self.tier1 = Tier1ExpressionFilter(
            alpha=self.alpha_tier1 if enable_tier1 else 0.0,
            llm_client=llm_client,
            model_name=model_name,
            rng=rng_t1,
        )
        self.tier2 = Tier2MindModelInjection(
            alpha=self.alpha_tier2 if enable_tier2 else 0.0,
            rng=rng_t2,
        )
        self.tier3 = Tier3AntiPatternBlock(
            alpha=self.alpha_tier3 if enable_tier3 else 0.0,
            rng=rng_t3,
        )

        self.enable_tier1 = enable_tier1
        self.enable_tier2 = enable_tier2
        self.enable_tier3 = enable_tier3

    def _build_context(self) -> dict[str, Any]:
        """Build enforcement context from skill file."""
        ctx = {}
        if self.skill and self.skill.capability:
            ctx["expression_dna"] = self.skill.capability.expression_dna
            ctx["mind_models"] = self.skill.capability.mind_models
        if self.skill and self.skill.constraint:
            ctx["anti_patterns"] = self.skill.constraint.anti_patterns
        return ctx

    async def enforce_generation(
        self,
        messages: list[dict[str, str]],
        draft_action: str = "",
    ) -> tuple[list[dict[str, str]], EnforcementLog, str]:
        """Run pre-generation enforcement (Tier 3 → Tier 2).

        Args:
            messages: LLM messages to be sent for generation.
            draft_action: Optional draft action for anti-pattern checking.

        Returns:
            Tuple of (modified_messages, enforcement_log, enforcement_context).
            enforcement_context is the injected text to merge into the planner prompt.
        """
        ctx = self._build_context()
        ctx["draft_action"] = draft_action
        log = EnforcementLog()
        enforcement_context = ""

        # Tier 3: Anti-pattern block (pre-generation, advisory injection stage)
        # Per outline §4.4: pre-gen Tier 3 propagates a reformulation
        # instruction into the planner context and records the violation
        # for metrics. The actual hard-block-with-regeneration runs at
        # post-gen via enforce_output → Forced Reformulation Protocol (§4.4.2).
        if self.enable_tier3:
            result = await self.tier3.check_pre_generation(messages, ctx)
            log.tier3_pre = result
            if result.modified_messages:
                messages = result.modified_messages
            if result.injection_text:
                enforcement_context += result.injection_text + "\n\n"
            if not result.passed:
                log.total_violations += 1

        # Tier 2: Mind model injection (pre-generation)
        if self.enable_tier2:
            result = await self.tier2.check_pre_generation(messages, ctx)
            log.tier2_pre = result
            if result.modified_messages:
                messages = result.modified_messages
            if result.injection_text:
                enforcement_context += result.injection_text + "\n\n"

        return messages, log, enforcement_context

    async def enforce_output(
        self,
        text: str,
        original_messages: list[dict[str, str]],
        safe_template: str | None = None,
    ) -> tuple[str, EnforcementLog, str | None]:
        """Run post-generation enforcement (Tier 1) with regeneration.

        Args:
            text: Generated text.
            original_messages: Original messages for regeneration prompt.
            safe_template: Optional safe-template string passed through to
                Tier 1's regeneration fallback (outline §4.4.2 step 4).
                When provided, Tier 1 uses it after ``max_retries`` are
                exhausted instead of silently accepting the last
                off-boundary attempt (G6). When ``None``, Tier 1 preserves
                the legacy "accept best attempt" behaviour.

        Returns:
            Tuple of (final_text, enforcement_log, replan_feedback).
            replan_feedback is non-None when Tier-3 post-gen detects violations
            and the caller should re-plan.
        """
        ctx = self._build_context()
        log = EnforcementLog()
        replan_feedback: str | None = None

        # Tier 3 post-gen safety check (outline §4.4.2 — Forced Reformulation
        # Protocol entry point). A `passed=False` here signals the agent loop
        # to drive constrained regeneration up to N_retry, then fall back to
        # the safe template. This is the structural RLHF-override event.
        if self.enable_tier3:
            result3 = await self.tier3.check_post_generation(text, ctx)
            log.tier3_post = result3
            if not result3.passed:
                log.total_violations += 1
                log.tier3_hard_block_triggered = True
                replan_feedback = result3.reason

        # Tier 1: Expression DNA filter (post-generation)
        if self.enable_tier1:
            final_text, result1 = await self.tier1.enforce_and_regenerate(
                text, original_messages, ctx, safe_template=safe_template
            )
            log.tier1_post = result1
            if not result1.passed:
                log.total_violations += 1
                log.regenerated = True
                if safe_template is not None and final_text == safe_template:
                    log.safe_template_fallback_used = True
            text = final_text

        return text, log, replan_feedback
