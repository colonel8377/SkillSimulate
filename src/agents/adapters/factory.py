"""Factory for selecting skill adapters by backend name."""

from __future__ import annotations

from src.agents.adapters.base import BaseCADPAdapter, ColleagueSkillAdapter, SkillAdapter
from src.skill.schema import SkillFile

_REGISTRY: dict[str, type[SkillAdapter]] = {
    "base": BaseCADPAdapter,
    "colleague_skill": ColleagueSkillAdapter,
}


def register_adapter(name: str, adapter_cls: type[SkillAdapter]) -> None:
    """Register a new adapter implementation."""
    _REGISTRY[name] = adapter_cls


def get_adapter(backend: str, skill: SkillFile) -> SkillAdapter:
    """Get an adapter instance for the given backend.

    Args:
        backend: Backend name (e.g. "base", "langchain", "autogen").
        skill: SkillFile to adapt.

    Returns:
        SkillAdapter instance.

    Raises:
        ValueError: If the backend is not registered.
    """
    adapter_cls = _REGISTRY.get(backend)
    if adapter_cls is None:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {available}."
        )
    return adapter_cls(skill)
