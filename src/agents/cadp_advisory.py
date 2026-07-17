"""Content-matched Nuwa-skill advisory control for Exp1 feasibility."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.adapters import get_adapter
from src.skill.schema import SkillFile


class CADPAdvisoryAgent(BaseAgent):
    """Render the same static skill fields as CADP Full, without Tier 1/2/3."""

    def __init__(self, *args, skill: SkillFile, backend: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        self.skill = skill
        self.adapter = get_adapter(backend, skill)
        self.enforcement_harness = None

    def get_role_description(self) -> str:
        return self.adapter.build_system_prompt(
            show_expression_dna=True,
            show_mind_models=True,
        )

    def get_constraints_text(self) -> str:
        # Deliberately byte-identical to CADPAgent's static constraint block.
        # The treatment contrast is runtime retrieval/filtering/reflection,
        # not extra prompt wording in either arm.
        return self.adapter.build_constraint_text(show_anti_patterns=True)

    # BaseAgent.get_reflection_directive() intentionally remains None.


__all__ = ["CADPAdvisoryAgent"]
