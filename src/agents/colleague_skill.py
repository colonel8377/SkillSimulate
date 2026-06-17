"""COLLEAGUE.SKILL baseline agent — single-track, no enforcement.

Mirrors Zhou et al. 2026: compiles a .skill file but only exposes the
Capability Track (Expression DNA + Mind Models). No Constraint Track, no
three-tier enforcement. Used in outline §5.2 Condition 5 to isolate the
incremental contribution of CADP's dual-track design.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.adapters import get_adapter
from src.skill.schema import SkillFile


class ColleagueSkillAgent(BaseAgent):
    """Baseline 5: COLLEAGUE.SKILL (single-track, no enforcement)."""

    def __init__(
        self,
        *args,
        skill: SkillFile,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill = skill
        # COLLEAGUE.SKILL uses the single-track adapter: capability only,
        # no enforcement harness.
        self.adapter = get_adapter("colleague_skill", skill)
        self.enforcement_harness = None

    def get_role_description(self) -> str:
        """Build role description from capability track only."""
        return self.adapter.build_system_prompt(
            show_expression_dna=True,
            show_mind_models=True,
        )

    def get_constraints_text(self) -> str:
        """COLLEAGUE.SKILL does not enforce constraints."""
        return ""
