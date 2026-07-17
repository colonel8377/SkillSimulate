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
from src.enforcement.tier3_llm_judge import Tier3LLMJudge
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
    tier3_llm_post: EnforcementResult | None = None
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
            "tier3_llm_post": {"tier": self.tier3_llm_post.tier, "passed": self.tier3_llm_post.passed, "reason": self.tier3_llm_post.reason} if self.tier3_llm_post else None,
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
        tier3_llm_judge: Tier3LLMJudge | None = None,
    ):
        self.alpha = alpha
        self.alpha_tier1 = alpha_tier1 if alpha_tier1 is not None else alpha
        self.alpha_tier2 = alpha_tier2 if alpha_tier2 is not None else alpha
        self.alpha_tier3 = alpha_tier3 if alpha_tier3 is not None else alpha
        self.skill = skill
        self.llm = llm_client
        self.model_name = model_name
        self.enable_tier3 = enable_tier3
        if enable_tier3 and tier3_llm_judge is None:
            raise ValueError(
                "enable_tier3=True requires tier3_llm_judge — the rule-based "
                "Tier-3 path was removed in the 2026-07-17 enforcement simplification."
            )
        self.tier3_llm_judge = tier3_llm_judge
        self.enable_tier3_llm_judge = tier3_llm_judge is not None

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
        # Tier 3's LLM judge is not an EnforcementStrategy, so its α-gate
        # lives here in the harness (see enforce_output).
        self._rng_t3 = rng_t3

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

        self.enable_tier1 = enable_tier1
        self.enable_tier2 = enable_tier2

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
        """Run pre-generation enforcement (Tier 2 only).

        Tier 3 is post-generation only after the 2026-07-17 simplification:
        the rule-based pre-gen advisory path was dead code (0/74 manipulation
        audit hits in v3 runs) and is removed. The LLM judge runs post-gen.

        Args:
            messages: LLM messages to be sent for generation.
            draft_action: Optional draft action (unused, kept for API compat).

        Returns:
            Tuple of (modified_messages, enforcement_log, enforcement_context).
            enforcement_context is the injected text to merge into the planner prompt.
        """
        ctx = self._build_context()
        ctx["draft_action"] = draft_action
        log = EnforcementLog()
        enforcement_context = ""

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
        context_messages: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
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
            context_messages: Recent thread messages for the LLM judge context.
            meta: Optional annotation metadata (agent_id / round / action_type /
                thread_id / attempt) forwarded to the Tier-3 judge log so its
                JSONL records are self-contained for later human labeling.

        Returns:
            Tuple of (final_text, enforcement_log, replan_feedback).
            replan_feedback is non-None when Tier-3 post-gen detects violations
            and the caller should re-plan.
        """
        ctx = self._build_context()
        ctx["recent_messages"] = context_messages or []
        log = EnforcementLog()
        replan_feedback: str | None = None

        # Non-text actions legitimately have empty text (the planner prompt
        # allows "empty for non-text actions": REVERT, REPORT, AWARD_DELTA,
        # LABEL, CLOSE, ...). There is no text to filter, so skip BOTH
        # Tier-3 (anti-pattern text match) and Tier-1 (Expression-DNA
        # embedding filter). An empty-input embedding sits ~6σ off the
        # expression centroid (empirically measured on bge-large), which
        # would otherwise falsely trigger regeneration + a safe-template
        # fallback — wasting calls AND inflating the constraint-forced /
        # safe-template count that §5.7 stratifies. Tier-2 (Mind Models)
        # already ran in pre-generation, so the agent remains conditioned.
        if not text or not text.strip():
            return text, log, replan_feedback

        # Tier 3: LLM judge post-generation. Single call per message; the
        # rule-based anti-pattern block was removed in the 2026-07-17
        # enforcement simplification (zero hits in v3 runs).
        if self.enable_tier3 and self.tier3_llm_judge is not None:
            ctx_messages = ctx.get("recent_messages", [])
            result_llm = await self.tier3_llm_judge.judge(
                text, self.skill, ctx_messages, meta=meta
            )
            log.tier3_llm_post = result_llm
            if not result_llm.passed:
                # α-gate the block: the judge detects, the harness decides.
                # At alpha_tier3=1.0 every detected violation blocks; below
                # 1.0 a detected violation is enforced with probability α
                # (deterministic per cell via the seeded tier-3 sub-RNG).
                if self.alpha_tier3 >= 1.0:
                    enforce_block = True
                else:
                    gate_rng = self._rng_t3 or random
                    enforce_block = gate_rng.random() < self.alpha_tier3
                if enforce_block:
                    log.total_violations += 1
                    log.tier3_hard_block_triggered = True
                    replan_feedback = f"LLM judge: {result_llm.reason}"

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
