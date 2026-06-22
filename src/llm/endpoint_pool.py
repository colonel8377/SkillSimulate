"""Multi-endpoint pool with health tracking and automatic failover.

Each model in ``models.yaml`` can list multiple endpoints (base_url +
api_key + model-name override). :class:`EndpointPool` maintains a
round-robin cursor over alive endpoints, tracks consecutive failures per
endpoint, and marks endpoints dead when they cross a configurable
threshold.

When all endpoints are dead, callers should raise
:class:`AllEndpointsExhausted`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI


@dataclass
class EndpointConfig:
    """User-facing endpoint definition from models.yaml."""
    base_url: str = ""
    api_key: str = ""
    model: str = ""  # per-endpoint model-name override (empty = use parent)


@dataclass
class EndpointState:
    """Runtime state for a single endpoint."""
    config: EndpointConfig
    client: AsyncOpenAI
    model: str          # resolved model name (endpoint override or parent)
    consecutive_failures: int = 0
    dead: bool = False
    # Opaque identifier for logging (index in the pool).
    idx: int = 0


class EndpointPool:
    """Round-robin pool of endpoints with health-based eviction.

    Usage::

        pool = EndpointPool(model_name="deepseek-v4-flash", endpoints=[...],
                            failure_threshold=10)
        # ... on each call:
        ep = await pool.pick_alive()           # raises AllEndpointsExhausted if none
        try:
            content = await do_call(ep)
            pool.record_success(ep)
        except Exception:
            pool.record_failure(ep, exc)
            raise
    """

    def __init__(
        self,
        model_name: str,
        endpoints: list[EndpointState],
        failure_threshold: int = 10,
    ):
        self.model_name = model_name
        self._endpoints = endpoints
        self.failure_threshold = failure_threshold
        self._cursor = 0
        self._rounds = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alive(self) -> list[EndpointState]:
        return [ep for ep in self._endpoints if not ep.dead]

    @property
    def alive_count(self) -> int:
        return sum(1 for ep in self._endpoints if not ep.dead)

    @property
    def dead_count(self) -> int:
        return sum(1 for ep in self._endpoints if ep.dead)

    @property
    def all_dead(self) -> bool:
        return self.alive_count == 0

    @property
    def total_endpoints(self) -> int:
        return len(self._endpoints)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def pick_alive(self) -> EndpointState:
        """Return the next alive endpoint (round-robin).

        Raises:
            AllEndpointsExhausted: when no alive endpoint remains.
        """
        from src.llm.exceptions import AllEndpointsExhausted

        if self.all_dead:
            raise AllEndpointsExhausted(self.model_name, self.total_endpoints)

        alive = self.alive
        # Advance cursor and wrap.  Use a simple round-robin so load is
        # spread evenly across alive endpoints; when one endpoint is
        # performing worse than others its failure counter will catch up
        # and evict it.
        self._cursor = (self._cursor + 1) % len(alive)
        self._rounds += 1
        return alive[self._cursor]

    def record_success(self, endpoint: EndpointState) -> None:
        """Reset the consecutive-failure counter for *endpoint*.

        Also resurrects a previously-dead endpoint (manual reset scenario).
        """
        if endpoint.consecutive_failures > 0:
            logger.info(
                f"EndpointPool[{self.model_name}]#{endpoint.idx}: "
                f"reset failure counter (was {endpoint.consecutive_failures})"
            )
        endpoint.consecutive_failures = 0
        if endpoint.dead:
            logger.warning(
                f"EndpointPool[{self.model_name}]#{endpoint.idx}: "
                f"resurrected (previously dead)"
            )
            endpoint.dead = False

    def record_failure(self, endpoint: EndpointState, error: str = "") -> None:
        """Increment failure counter; mark dead if threshold crossed."""
        endpoint.consecutive_failures += 1
        failures = endpoint.consecutive_failures
        logger.warning(
            f"EndpointPool[{self.model_name}]#{endpoint.idx}: "
            f"failure {failures}/{self.failure_threshold}: {error}"
        )
        if failures >= self.failure_threshold and not endpoint.dead:
            endpoint.dead = True
            logger.error(
                f"EndpointPool[{self.model_name}]#{endpoint.idx}: "
                f"MARKED DEAD after {failures} consecutive failures "
                f"(base_url={endpoint.config.base_url})"
            )

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Reset all endpoints to alive with zero failures."""
        for ep in self._endpoints:
            ep.consecutive_failures = 0
            ep.dead = False
        logger.info(
            f"EndpointPool[{self.model_name}]: reset all {len(self._endpoints)} endpoints"
        )

    def status(self) -> dict[str, Any]:
        """Return a human-readable status dict for logging/debugging."""
        return {
            "model": self.model_name,
            "total": self.total_endpoints,
            "alive": self.alive_count,
            "dead": self.dead_count,
            "endpoints": [
                {
                    "idx": ep.idx,
                    "base_url": ep.config.base_url,
                    "model": ep.model,
                    "consecutive_failures": ep.consecutive_failures,
                    "dead": ep.dead,
                }
                for ep in self._endpoints
            ],
        }
