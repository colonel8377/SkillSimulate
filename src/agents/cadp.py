"""Full CADP agent — three-tier enforcement with dual-track skill."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.adapters import get_adapter
from src.skill.schema import SkillFile


class CADPAgent(BaseAgent):
    """Condition 7: Full CADP with three-tier enforcement (outline §5.2)."""

    def __init__(
        self,
        *args,
        skill: SkillFile,
        alpha: float = 1.0,
        show_expression_dna: bool = True,
        show_mind_models: bool = True,
        show_anti_patterns: bool = True,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        backend: str = "base",
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill = skill
        self.show_expression_dna = show_expression_dna
        self.show_mind_models = show_mind_models
        self.show_anti_patterns = show_anti_patterns
        self._alpha = alpha
        self._alpha_tier1 = alpha_tier1
        self._alpha_tier2 = alpha_tier2
        self._alpha_tier3 = alpha_tier3
        # R1: per-agent deterministic seed threaded into the enforcement
        # harness so identical cell seeds reproduce identical α-gating
        # sequences. ``None`` falls back to unseeded RNGs (legacy/unit tests).
        self._seed = seed

        self.adapter = get_adapter(backend, skill)

        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=True,
            alpha_tier1=alpha_tier1,
            alpha_tier2=alpha_tier2,
            alpha_tier3=alpha_tier3,
            seed=seed,
        )

    def get_role_description(self) -> str:
        """Build role description via adapter."""
        return self.adapter.build_system_prompt(
            show_expression_dna=self.show_expression_dna,
            show_mind_models=self.show_mind_models,
        )

    def get_constraints_text(self) -> str:
        """Build constraint text via adapter."""
        return self.adapter.build_constraint_text(
            show_anti_patterns=self.show_anti_patterns,
        )
