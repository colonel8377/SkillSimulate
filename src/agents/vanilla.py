"""Vanilla LLM agent — no persona."""

from __future__ import annotations

from src.agents.base import BaseAgent


class VanillaAgent(BaseAgent):
    """Baseline 1: Raw LLM with no persona."""

    def get_role_description(self) -> str:
        return "You are a participant in an online community discussion."

    def get_constraints_text(self) -> str:
        return ""
