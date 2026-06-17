"""Segmentation Persona agent — Li & Cheng 2026 audience segmentation approach.

Uses demographic + psychographic tags to create segment-level personas.
"""

from __future__ import annotations

from src.agents.base import BaseAgent


class SegmentationPersonaAgent(BaseAgent):
    """Baseline 3: Segmentation persona (Li & Cheng 2026).

    Instead of individual persona, uses segment-level demographic
    and psychographic distribution as persona context.
    """

    def __init__(
        self,
        *args,
        segment_name: str = "",
        segment_demographics: str = "",
        segment_psychographics: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.segment_name = segment_name
        self.segment_demographics = segment_demographics
        self.segment_psychographics = segment_psychographics

    def get_role_description(self) -> str:
        desc = "You are a participant in an online community discussion.\n\n"
        desc += f"You belong to the segment: {self.segment_name}\n\n"
        if self.segment_demographics:
            desc += f"Demographic profile:\n{self.segment_demographics}\n\n"
        if self.segment_psychographics:
            desc += f"Psychographic profile:\n{self.segment_psychographics}\n\n"
        desc += "Behave consistently with your segment's characteristics."
        return desc

    def get_constraints_text(self) -> str:
        return "Stay consistent with your segment profile."
