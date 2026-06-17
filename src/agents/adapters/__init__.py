"""Skill adapters — translate SkillFile into backend-specific configurations."""

from src.agents.adapters.base import SkillAdapter, BaseCADPAdapter, ColleagueSkillAdapter
from src.agents.adapters.factory import get_adapter

__all__ = ["SkillAdapter", "BaseCADPAdapter", "ColleagueSkillAdapter", "get_adapter"]
