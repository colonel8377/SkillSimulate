"""Descriptive Persona agent — standard system prompt with demographic/identity tags."""

from __future__ import annotations

from src.agents.base import BaseAgent


class DescriptivePersonaAgent(BaseAgent):
    """Baseline 2: Standard descriptive persona via system prompt."""

    def __init__(
        self,
        *args,
        persona_description: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.persona_description = persona_description

    def get_role_description(self) -> str:
        return f"You are a participant in an online community discussion.\n\nYour profile:\n{self.persona_description}"

    def get_constraints_text(self) -> str:
        return "Stay in character based on your profile."
