"""Multi-endpoint pool with round-robin, failure tracking, dead-endpoint
detection, and cooldown-based revival.

Each model can be backed by multiple endpoints (proxies / API keys). The pool
round-robins across alive endpoints; after ``failure_threshold`` consecutive
failures an endpoint is marked dead and skipped.  Dead endpoints are
automatically revived after ``cooldown_seconds`` so that transient network
outages do not permanently exhaust the pool during long experiments.
When all endpoints are dead AND none have expired their cooldown,
:class:`AllEndpointsExhausted` is raised.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from loguru import logger

from src.llm.exceptions import AllEndpointsExhausted


@dataclass
class EndpointConfig:
    """Static configuration for one endpoint."""
    base_url: str
    api_key: str
    model: str = ""  # per-endpoint model-id override


@dataclass
class EndpointState:
    """Runtime state for one endpoint."""
    config: EndpointConfig
    client: object  # AsyncOpenAI
    model: str
    idx: int
    alive: bool = True
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    # Monotonic timestamp when the endpoint was marked dead.
    # 0.0 means never marked dead (or has been revived).
    dead_since: float = 0.0


class EndpointPool:
    """Round-robin pool of endpoints with automatic dead-endpoint fencing
    and cooldown-based revival.

    Dead endpoints are revived after ``cooldown_seconds`` (default 30 min).
    This prevents permanent pool exhaustion during long experiment runs
    where transient network issues may mark endpoints dead temporarily.
    """

    def __init__(
        self,
        model_name: str,
        endpoints: list[EndpointState],
        failure_threshold: int = 15,
        cooldown_seconds: float = 1800.0,
    ):
        self.model_name = model_name
        self._endpoints: list[EndpointState] = list(endpoints)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._cursor: int = 0
        self._lock = asyncio.Lock()

    @property
    def total_endpoints(self) -> int:
        return len(self._endpoints)

    @property
    def alive_count(self) -> int:
        return sum(1 for ep in self._endpoints if ep.alive)

    @property
    def all_dead(self) -> bool:
        return self.alive_count == 0 and self.total_endpoints > 0

    def _revive_cooled_endpoints(self) -> None:
        """Check dead endpoints and revive those whose cooldown has expired.

        Must be called while holding ``self._lock``.
        """
        now = time.monotonic()
        for ep in self._endpoints:
            if not ep.alive and ep.dead_since > 0:
                elapsed = now - ep.dead_since
                if elapsed >= self.cooldown_seconds:
                    ep.alive = True
                    ep.consecutive_failures = 0
                    ep.dead_since = 0.0
                    logger.info(
                        f"EndpointPool[{self.model_name}]#{ep.idx} REVIVED "
                        f"after {elapsed:.0f}s cooldown "
                        f"(total_successes={ep.total_successes}, "
                        f"total_failures={ep.total_failures})"
                    )

    async def pick_alive(self) -> EndpointState:
        """Round-robin to the next alive endpoint.

        When all endpoints are dead, attempts to revive any whose cooldown
        has expired before raising :class:`AllEndpointsExhausted`.

        Raises:
            AllEndpointsExhausted: when all endpoints are dead and none
                have expired their cooldown.
        """
        async with self._lock:
            # Attempt revival before checking all_dead
            if self.all_dead:
                self._revive_cooled_endpoints()
            if self.all_dead:
                raise AllEndpointsExhausted(
                    self.model_name, self.total_endpoints
                )
            for _ in range(self.total_endpoints):
                self._cursor = (self._cursor + 1) % self.total_endpoints
                ep = self._endpoints[self._cursor]
                if ep.alive:
                    return ep
            # One more revival attempt in case cursor skipped a just-revived ep
            self._revive_cooled_endpoints()
            for _ in range(self.total_endpoints):
                self._cursor = (self._cursor + 1) % self.total_endpoints
                ep = self._endpoints[self._cursor]
                if ep.alive:
                    return ep
            raise AllEndpointsExhausted(self.model_name, self.total_endpoints)

    def record_success(self, ep_state: EndpointState) -> None:
        """Reset the consecutive-failure counter on success."""
        ep_state.consecutive_failures = 0
        ep_state.total_successes += 1

    def record_failure(self, ep_state: EndpointState, error: str = "") -> None:
        """Increment failure counters; mark dead at threshold.

        When an endpoint hits the consecutive-failure threshold it is
        fenced (``alive=False``) and its ``dead_since`` timestamp is
        recorded.  :meth:`pick_alive` will auto-revive it after the
        cooldown period expires.
        """
        ep_state.consecutive_failures += 1
        ep_state.total_failures += 1
        if ep_state.consecutive_failures >= self.failure_threshold:
            ep_state.alive = False
            ep_state.dead_since = time.monotonic()
            logger.error(
                f"EndpointPool[{self.model_name}]#{ep_state.idx} marked DEAD "
                f"after {ep_state.consecutive_failures} consecutive failures "
                f"(threshold={self.failure_threshold}, "
                f"cooldown={self.cooldown_seconds:.0f}s). "
                f"Last error: {error[:200]}"
            )
