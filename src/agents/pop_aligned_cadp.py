"""Pop-Aligned + CADP stacked condition.

Outline §5.2 Condition 12: combines Population-Aligned Persona Generation
(arXiv:2509.10127) for attribute distribution matching with CADP's
behavioral-rule distillation and three-tier enforcement. Tests whether
demographic/attitudinal attributes and behavioral rules are complementary.
"""

from __future__ import annotations

from src.agents.cadp import CADPAgent
from src.agents.pop_aligned import compute_cluster_attributes, sample_individual_attributes
from src.skill.schema import SkillFile


class PopAlignedCADPAgent(CADPAgent):
    """Baseline 12: Pop-Aligned attributes + CADP behavioral rules.

    The system prompt includes both the population-aligned sampled
    attributes and the CADP capability-track role description. The full
    three-tier enforcement harness is retained.
    """

    def __init__(
        self,
        *args,
        skill: SkillFile,
        cluster_attributes: dict | None = None,
        sampled_attributes: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, skill=skill, **kwargs)
        self.cluster_attributes = cluster_attributes or {}
        self.sampled_attributes = sampled_attributes or {}

    def get_role_description(self) -> str:
        """Merge Pop-Aligned attribute block with CADP role description."""
        desc = "You are a participant in an online community discussion.\n\n"

        if self.sampled_attributes:
            desc += "Your sampled demographic/attitudinal profile:\n"
            for attr, value in self.sampled_attributes.items():
                desc += f"  - {attr}: {value}\n"
            desc += "\n"

        desc += super().get_role_description()

        if self.cluster_attributes:
            desc += "\nYour community's population characteristics:\n"
            for attr, stats in self.cluster_attributes.items():
                if isinstance(stats, dict):
                    desc += (
                        f"  - {attr}: mean={stats.get('mean', 'N/A'):.2f}, "
                        f"std={stats.get('std', 'N/A'):.2f}\n"
                    )
                else:
                    desc += f"  - {attr}: {stats}\n"

        return desc


__all__ = ["PopAlignedCADPAgent", "compute_cluster_attributes", "sample_individual_attributes"]
