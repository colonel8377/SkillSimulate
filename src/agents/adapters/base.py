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
        **kwargs,
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
                for line in self._qualitative_expression_dna(edna):
                    desc += f"{line}\n"
                desc += "\n"

            if show_mind_models and self.skill.capability.mind_models:
                desc += "Your reasoning frameworks:\n"
                for mm in self.skill.capability.mind_models:
                    desc += f"- {mm.name}: {mm.description}\n"
                    desc += f"  Apply when: {mm.application}\n"

        return desc

    @staticmethod
    def _qualitative_expression_dna(edna) -> list[str]:
        """Render Expression DNA scalars as qualitative descriptors an LLM can
        operationalize, plus a concrete style exemplar (outline §4.3 capability
        track). LLMs inhabit qualitative voice tags far better than raw 0–1
        scalars; the numeric value is kept in parentheses for traceability and
        for the §5.8 dissociation analysis. The *hard* style guarantee remains
        the Tier-1 post-generation embedding filter (§4.4) — this only improves
        conditioning. Not an anti-circularity risk: outline §5.3 evaluates
        Linguistics on an orthogonal feature space (discourse markers /
        sentiment trajectory / speech acts).
        """
        def _pole(val: float, low: str, high: str) -> str:
            if val <= 0.33:
                return low
            if val >= 0.67:
                return high
            return f"a blend of {low} and {high}"

        lines: list[str] = []

        sents = edna.avg_sentence_length
        if sents:
            if sents <= 12:
                length_word = "short, punchy"
            elif sents >= 22:
                length_word = "long, developed"
            else:
                length_word = "medium-length"
            lines.append(f"- Sentence length: {length_word} (avg {sents:.0f} words)")

        lines.append(
            f"- Register: {_pole(edna.style_formal_casual, 'formal', 'casual')} "
            f"(formal↔casual {edna.style_formal_casual:.2f})"
        )
        lines.append(
            f"- Stance: {_pole(edna.style_cautious_assertive, 'cautious', 'assertive')} "
            f"(cautious↔assertive {edna.style_cautious_assertive:.2f})"
        )

        if edna.high_freq_words:
            lines.append(f"- Characteristic vocabulary: {', '.join(edna.high_freq_words[:10])}")

        # Concrete exemplar synthesised from the style poles — gives the model
        # a target voice to inhabit during generation. Phrased without a leading
        # article so both single-pole ("formal") and blend ("a blend of …")
        # descriptors read cleanly.
        fc = _pole(edna.style_formal_casual, "formal", "casual")
        ca = _pole(edna.style_cautious_assertive, "cautious", "assertive")
        lines.append(
            f"- Example voice: aim for {fc}, {ca} phrasing, using the vocabulary above."
        )
        return lines

    def build_constraint_text(self, show_anti_patterns: bool = True) -> str:
        """Build constraint text from skill's constraint track."""
        if not show_anti_patterns:
            return ""
        if not self.skill.constraint or not self.skill.constraint.anti_patterns:
            return ""

        lines = ["You MUST avoid the following behaviors:"]
        for ap in self.skill.constraint.anti_patterns:
            lines.append(f"- {ap.description}")
            if ap.reason:
                lines.append(f"  Reason: {ap.reason}")

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
        **kwargs,
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
            tier3_llm_judge=kwargs.get("tier3_llm_judge"),
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
        **kwargs,
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
