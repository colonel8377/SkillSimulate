"""Reserved agent-builder registry for third-party extensibility (§axis B).

This module is the *interface* placeholder for a future migration of
:meth:`src.simulation.population.PopulationBuilder._create_agent` from a
hard-coded ``if/elif`` dispatch chain into an open-world registry pattern,
mirroring the already-open skill-adapter registry in
:mod:`src.agents.adapters.factory`.

Design intent
-------------
The current 13 in-tree conditions are still constructed by the
``_create_agent`` chain in ``population.py`` because each condition
requires bespoke pre-construction logic (cluster feature aggregation,
descriptive-persona building, segmentation profile, population-aligned
attribute sampling, ablation-specific skill mutations, …). Refactoring
all 13 builders into closures right now would risk silently changing
the behaviour of the headline 13-condition Exp 1 grid, so that work is
deferred.

What this module provides today
-------------------------------
* A typed builder protocol (:data:`AgentBuilder`) describing the
  signature any third-party builder must implement.
* An :class:`AgentBuildContext` dataclass capturing the construction-time
  inputs that a builder needs (mirroring the kwargs threaded through
  ``_create_agent``).
* :func:`register_agent_builder` / :func:`unregister_agent_builder` /
  :func:`get_agent_builder` / :func:`is_registered` for plugin
  registration.
* Best-effort dispatch via :func:`build_registered_agent`. ``population.py``
  calls this *before* its in-tree ``if/elif`` chain so third-party
  conditions take precedence and the in-tree path stays the fallback.

What this module deliberately does NOT do
-----------------------------------------
* It does NOT register the 13 in-tree conditions (vanilla, descriptive,
  segmentation, pop_aligned, colleague_skill, clustering_only, cadp_full,
  cadp_shuffled, cadp_minus_edna/mm/ap, cadp_constraint_only,
  pop_aligned_cadp). They remain in ``population.py`` until a separate
  refactor migrates them, validated by re-running the full Exp 1 grid.
* It does NOT bypass condition validation. ``ConditionName`` in
  :mod:`src.experiment.conditions` still gates which condition strings
  are permitted; third parties wishing to add a new condition must
  register both an entry there and a builder here.
* It does NOT touch the LLM-client or skill-adapter axes — those are
  already open via :class:`src.llm.client.LLMClient` and
  :func:`src.agents.adapters.factory.register_adapter`.

Usage example (third-party extension)
-------------------------------------
::

    from src.agents.registry import register_agent_builder, AgentBuildContext
    from src.agents.base import BaseAgent

    class MyLangChainAgent(BaseAgent):
        ...

    def _build_my_agent(ctx: AgentBuildContext) -> BaseAgent:
        return MyLangChainAgent(
            agent_id=ctx.agent_id,
            llm_client=ctx.llm_client,
            model_name=ctx.model_name,
            cluster_id=ctx.cluster_id,
            skill=ctx.skills.get(ctx.cluster_id),
        )

    register_agent_builder("my_langchain", _build_my_agent)

After registration the new condition is invocable via the standard
``Population.build_population(condition="my_langchain", ...)`` path and
through the YAML config — assuming a matching ``ConditionName`` entry
has been added.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover — import only for type checking
    from src.agents.base import BaseAgent
    from src.clustering.clusterer import ClusterResult


@dataclass
class AgentBuildContext:
    """All construction-time inputs needed to instantiate an agent.

    This mirrors the kwargs already threaded through
    ``PopulationBuilder._create_agent``. Third-party builders receive a
    populated instance and return a constructed :class:`BaseAgent`.
    """

    # Identity ---------------------------------------------------------
    agent_id: str
    cluster_id: str
    condition: str

    # LLM + backend ----------------------------------------------------
    llm_client: Any
    model_name: str
    backend: str = "base"

    # α (per-tier — outline §4.4.3) -----------------------------------
    alpha: float = 1.0
    alpha_tier1: float | None = None
    alpha_tier2: float | None = None
    alpha_tier3: float | None = None

    # Clustering / population context ---------------------------------
    cluster_result: "ClusterResult | None" = None
    skills: dict[str, Any] = field(default_factory=dict)

    # Stochastic / seeding --------------------------------------------
    rng: random.Random | None = None
    agent_seed: int | None = None

    # Escape hatch — third-party builders can read but not require ----
    extra: dict[str, Any] = field(default_factory=dict)


# An ``AgentBuilder`` is any callable taking an :class:`AgentBuildContext`
# and returning a constructed :class:`BaseAgent`.
AgentBuilder = Callable[[AgentBuildContext], "BaseAgent"]


# Module-level registry. Intentionally module-private; mutate only via
# the ``register_agent_builder`` / ``unregister_agent_builder`` helpers
# so we have a single place to add validation later (e.g. duplicate
# detection, signature checks, optional setuptools entry-point loading).
_AGENT_REGISTRY: dict[str, AgentBuilder] = {}


def register_agent_builder(
    name: str,
    builder: AgentBuilder,
    *,
    overwrite: bool = False,
) -> None:
    """Register a builder for an experimental condition.

    Args:
        name: Condition name (must match the string used in the YAML
            ``conditions:`` list and in ``ConditionName``).
        builder: Callable taking :class:`AgentBuildContext` and returning
            a :class:`BaseAgent` subclass instance.
        overwrite: If False (default), raise on duplicate registration.
            If True, silently replace — useful for hot-reload during
            development.

    Raises:
        ValueError: If ``name`` is empty, ``builder`` is not callable,
            or the name is already registered and ``overwrite=False``.
    """
    if not name:
        raise ValueError("Condition name must be non-empty.")
    if not callable(builder):
        raise ValueError(
            f"Builder for condition '{name}' must be callable, got {type(builder).__name__}."
        )
    if name in _AGENT_REGISTRY and not overwrite:
        raise ValueError(
            f"Condition '{name}' is already registered. "
            f"Pass overwrite=True to replace."
        )
    _AGENT_REGISTRY[name] = builder


def unregister_agent_builder(name: str) -> None:
    """Remove a previously registered builder. No-op if absent."""
    _AGENT_REGISTRY.pop(name, None)


def get_agent_builder(name: str) -> AgentBuilder | None:
    """Return the builder for ``name`` or None if not registered."""
    return _AGENT_REGISTRY.get(name)


def is_registered(name: str) -> bool:
    """Return True iff a third-party builder is registered for ``name``."""
    return name in _AGENT_REGISTRY


def registered_conditions() -> list[str]:
    """Return the list of condition names with a registered builder."""
    return sorted(_AGENT_REGISTRY)


def build_registered_agent(ctx: AgentBuildContext) -> "BaseAgent | None":
    """Dispatch to a registered builder, or return None if absent.

    ``population.py`` calls this *before* its in-tree ``if/elif`` chain.
    A None return signals "not a third-party condition — fall through to
    the in-tree path", preserving the existing behaviour for the 13
    headline conditions exactly.
    """
    builder = _AGENT_REGISTRY.get(ctx.condition)
    if builder is None:
        return None
    return builder(ctx)
