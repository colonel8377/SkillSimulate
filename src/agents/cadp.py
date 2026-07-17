"""Full CADP agent — three-tier enforcement with dual-track skill."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.adapters import get_adapter
from src.enforcement.tier3_llm_judge import Tier3LLMJudge
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
        tier1_max_retries: int = 1,
        tier3_llm_judge_enabled: bool = False,
        tier3_llm_judge_model: str = "classification",
        tier3_llm_judge_audit_only: bool = False,
        tier3_llm_judge_output_dir: str = "outputs/results/tier3_llm_judgments",
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

        tier3_llm_judge = None
        if tier3_llm_judge_enabled and self.llm is not None:
            tier3_llm_judge = Tier3LLMJudge(
                llm_client=self.llm,
                model_name=tier3_llm_judge_model,
                output_dir=tier3_llm_judge_output_dir,
                audit_only=tier3_llm_judge_audit_only,
            )
            tier3_llm_judge.set_run_id(self.agent_id)

        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=tier3_llm_judge is not None,
            alpha_tier1=alpha_tier1,
            alpha_tier2=alpha_tier2,
            alpha_tier3=alpha_tier3,
            seed=seed,
            tier3_llm_judge=tier3_llm_judge,
        )
        # Keep Tier-1 and Tier-3 retry budgets explicit in experiment
        # provenance. Tier 3 is owned by BaseAgent; Tier 1 by the harness.
        self.enforcement_harness.tier1.max_retries = int(tier1_max_retries)

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

    def get_reflection_directive(self) -> str | None:
        """Outline §4.6: reflection must reinforce (not just summarize) the
        agent's Mind Models. CADP is the only condition carrying verified Mind
        Models, so this is a CADP-specific long-horizon advantage — it keeps a
        consistent reasoning frame across the 50-round horizon where plain
        descriptive personas drift (the Chameleon's-Limit failure mode).

        Returns ``None`` when Mind Models are hidden (``cadp_minus_mm``,
        ``cadp_constraint_only`` set ``show_mind_models = False``) or absent, so
        those ablations fall back to plain belief consolidation. Computed at
        call time (not ``__init__``) so it respects the ablations' post-init
        flag changes without each ablation needing to refresh it.
        """
        if not self.show_mind_models:
            return None
        cap = self.skill.capability
        if not cap or not cap.mind_models:
            return None
        names = ", ".join(mm.name for mm in cap.mind_models)
        return (
            "Reinforce your reasoning frameworks (Mind Models): "
            f"{names}. When consolidating your stances, explicitly apply these "
            "frameworks to the recent interactions — note which framework you "
            "used to form or shift each stance, and whether any interaction "
            "updated your view on a framework."
        )
