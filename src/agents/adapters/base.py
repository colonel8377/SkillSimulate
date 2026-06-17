"""Base skill adapter and default CADP implementation.

A SkillAdapter translates a SkillFile into:
- system prompt fragments (role description, constraints)
- pre-generation callbacks (Tier-2/3 injection text)
- post-generation callbacks (Tier-1 filter config)
- enforcement harness configuration (which tiers enabled)

This decouples the skill representation from the agent implementation,
enabling future integration with LangChain, AutoGen, AG2, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.enforcement.harness import EnforcementHarness
from src.llm.client import LLMClient
from src.skill.schema import SkillFile


class SkillAdapter(ABC):
    """Abstract interface for translating SkillFile into agent-ready artifacts."""

    def __init__(self, skill: SkillFile):
        self.skill = skill

    @abstractmethod
    def build_system_prompt(
        self,
        show_expression_dna: bool = True,
        show_mind_models: bool = True,
    ) -> str:
        """Build the system prompt / role description from the skill's capability track."""

    @abstractmethod
    def build_constraint_text(self, show_anti_patterns: bool = True) -> str:
        """Build constraint text from the skill's constraint track."""

    @abstractmethod
    def build_enforcement_harness(
        self,
        alpha: float = 1.0,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        enable_tier1: bool = True,
        enable_tier2: bool = True,
        enable_tier3: bool = True,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        seed: int | None = None,
    ) -> EnforcementHarness:
        """Build the enforcement harness configured for this skill."""

    @abstractmethod
    def get_tier_config(self) -> dict[str, bool]:
        """Return which tiers should be enabled for this skill."""


class BaseCADPAdapter(SkillAdapter):
    """Default CADP adapter — extracts prompt text and harness from SkillFile.

    This is the reference implementation. The output is identical to what
    CADPAgent previously generated inline.
    """

    def build_system_prompt(
        self,
        show_expression_dna: bool = True,
        show_mind_models: bool = True,
    ) -> str:
        """Build role description from skill's capability track."""
        desc = "You are a participant in an online community discussion.\n\n"

        if self.skill.capability:
            if show_expression_dna:
                edna = self.skill.capability.expression_dna
                desc += "Your communication style:\n"
                desc += f"- Average sentence length: {edna.avg_sentence_length:.1f} words\n"
                desc += f"- Formal/casual: {edna.style_formal_casual:.2f}\n"
                desc += f"- Cautious/assertive: {edna.style_cautious_assertive:.2f}\n"
                if edna.high_freq_words:
                    desc += f"- Frequently uses: {', '.join(edna.high_freq_words[:10])}\n"
                desc += "\n"

            if show_mind_models and self.skill.capability.mind_models:
                desc += "Your reasoning frameworks:\n"
                for mm in self.skill.capability.mind_models:
                    desc += f"- {mm.name}: {mm.description}\n"
                    desc += f"  Apply when: {mm.application}\n"

        return desc

    def build_constraint_text(self, show_anti_patterns: bool = True) -> str:
        """Build constraint text from skill's constraint track."""
        if not show_anti_patterns:
            return ""
        if not self.skill.constraint or not self.skill.constraint.anti_patterns:
            return ""

        lines = ["You MUST avoid the following behaviors:"]
        for ap in self.skill.constraint.anti_patterns:
            lines.append(f"- {ap.description}")
            if ap.trigger_keywords:
                lines.append(f"  Avoid keywords: {', '.join(ap.trigger_keywords)}")

        return "\n".join(lines)

    def build_enforcement_harness(
        self,
        alpha: float = 1.0,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        enable_tier1: bool = True,
        enable_tier2: bool = True,
        enable_tier3: bool = True,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        seed: int | None = None,
    ) -> EnforcementHarness:
        """Build enforcement harness for this skill."""
        return EnforcementHarness(
            alpha=alpha,
            skill=self.skill,
            llm_client=llm_client,
            model_name=model_name,
            enable_tier1=enable_tier1,
            enable_tier2=enable_tier2,
            enable_tier3=enable_tier3,
            alpha_tier1=alpha_tier1,
            alpha_tier2=alpha_tier2,
            alpha_tier3=alpha_tier3,
            seed=seed,
        )

    def get_tier_config(self) -> dict[str, bool]:
        """Return default tier configuration for a full CADP skill."""
        return {"tier1": True, "tier2": True, "tier3": True}


class ColleagueSkillAdapter(BaseCADPAdapter):
    """Single-track adapter mirroring COLLEAGUE.SKILL (Zhou et al. 2026).

    Only the Capability Track (Expression DNA + Mind Models) is exposed;
    the Constraint Track is ignored and no enforcement harness is built.
    This isolates the incremental contribution of CADP's dual-track design
    and three-tier enforcement.
    """

    def build_constraint_text(self, show_anti_patterns: bool = True) -> str:
        """COLLEAGUE.SKILL has no hard constraints."""
        return ""

    def build_enforcement_harness(
        self,
        alpha: float = 1.0,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        enable_tier1: bool = True,
        enable_tier2: bool = True,
        enable_tier3: bool = True,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        seed: int | None = None,
    ) -> EnforcementHarness:
        """Return an enforcement harness with all tiers disabled."""
        return EnforcementHarness(
            alpha=alpha,
            skill=self.skill,
            llm_client=llm_client,
            model_name=model_name,
            enable_tier1=False,
            enable_tier2=False,
            enable_tier3=False,
            alpha_tier1=alpha_tier1,
            alpha_tier2=alpha_tier2,
            alpha_tier3=alpha_tier3,
            seed=seed,
        )

    def get_tier_config(self) -> dict[str, bool]:
        """No tiers are active for COLLEAGUE.SKILL."""
        return {"tier1": False, "tier2": False, "tier3": False}
