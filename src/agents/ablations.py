"""CADP ablation agents — shuffled, minus Expression DNA / Mind Models / Anti-patterns."""

from __future__ import annotations

import random

from src.agents.cadp import CADPAgent
from src.skill.schema import SkillFile


class CADPShuffledAgent(CADPAgent):
    """Baseline 5: CADP with shuffled cluster assignments (permutation test).

    Same skill structure but assigned randomly — not matched to the
    agent's actual cluster. Tests whether cluster-specificity matters.
    The agent prompt is identical to full CADP (no revealing notes).
    """

    def __init__(
        self,
        *,
        all_skills: dict[str, SkillFile],
        target_cluster_id: str,
        rng: random.Random | None = None,
        **kwargs,
    ):
        shuffled = self.create_shuffled_skill(all_skills, target_cluster_id, rng)
        super().__init__(skill=shuffled, **kwargs)

    @staticmethod
    def create_shuffled_skill(
        skills: dict[str, SkillFile],
        target_cluster_id: str,
        rng: random.Random | None = None,
    ) -> SkillFile:
        """Create a shuffled skill by randomly assigning a different cluster's skill.

        Args:
            skills: All compiled skill files.
            target_cluster_id: The cluster the agent actually belongs to.
            rng: Optional random generator.

        Returns:
            A skill file from a DIFFERENT cluster.
        """
        rng = rng or random.Random()
        other_ids = [k for k in skills if k != target_cluster_id]
        if not other_ids:
            return skills[target_cluster_id]

        shuffled_id = rng.choice(other_ids)
        return skills[shuffled_id]


class CADPMinusExpressionDNAAgent(CADPAgent):
    """Baseline 6: CADP without Expression DNA (Tier 1 disabled).

    Tests the contribution of language pattern enforcement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reconfigure harness with tier1 disabled. Re-pass the per-agent
        # seed so the remaining tiers' α-gating stays reproducible (R1).
        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=self._alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=False,
            enable_tier2=True,
            enable_tier3=True,
            alpha_tier1=self._alpha_tier1,
            alpha_tier2=self._alpha_tier2,
            alpha_tier3=self._alpha_tier3,
            seed=self._seed,
        )
        # Hide Expression DNA from role description
        self.show_expression_dna = False


class CADPMinusMindModelsAgent(CADPAgent):
    """Baseline 7: CADP without Mind Models (Tier 2 disabled).

    Tests the contribution of cognitive framework injection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=self._alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=True,
            enable_tier2=False,
            enable_tier3=True,
            alpha_tier1=self._alpha_tier1,
            alpha_tier2=self._alpha_tier2,
            alpha_tier3=self._alpha_tier3,
            seed=self._seed,
        )
        # Hide Mind Models from role description
        self.show_mind_models = False


class CADPMinusAntiPatternsAgent(CADPAgent):
    """Baseline 8: CADP without Anti-patterns (Tier 3 LLM judge disabled).

    Tests whether anti-patterns are critical for overcoming RLHF
    compromise tendencies. Post-2026-07-17 simplification: Tier 3 is the
    LLM judge only, so "no anti-patterns" = no judge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=self._alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=False,
            alpha_tier1=self._alpha_tier1,
            alpha_tier2=self._alpha_tier2,
            alpha_tier3=self._alpha_tier3,
            seed=self._seed,
        )
        # Hide anti-patterns from constraints text
        self.show_anti_patterns = False


class CADPConstraintOnlyAgent(CADPAgent):
    """Ablation: CADP with only the Constraint Track (Anti-patterns).

    No Expression DNA, no Mind Models. Only Tier-3 LLM judge enforcement.
    Tests whether the Constraint Track alone is sufficient, or whether the
    dual-track design (Capability + Constraint) is necessary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enforcement_harness = self.adapter.build_enforcement_harness(
            alpha=self._alpha,
            llm_client=self.llm,
            model_name=self.model_name,
            enable_tier1=False,
            enable_tier2=False,
            enable_tier3=True,
            alpha_tier1=self._alpha_tier1,
            alpha_tier2=self._alpha_tier2,
            alpha_tier3=self._alpha_tier3,
            seed=self._seed,
            tier3_llm_judge=self.enforcement_harness.tier3_llm_judge,
        )
        # Hide Capability Track from role description
        self.show_expression_dna = False
        self.show_mind_models = False
