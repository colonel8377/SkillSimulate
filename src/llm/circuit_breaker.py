"""Global circuit breaker for sustained LLM API failures.

Trip condition: ``consecutive_failures >= threshold``. Once tripped,
:meth:`CircuitBreaker.check` raises :class:`CircuitBreakerOpen` and the
runner halts the whole ``run_all()`` loop, writes a ``_FAILED.json``
marker, and exits non-zero so a human can investigate.

The breaker is *process-global* by design — it tracks failures across
cells, not within a single cell, because the failure modes we want to
catch (bad credentials, exhausted quota, dead endpoint) affect every cell
equally and there is no value in letting subsequent cells burn the same
retry budget.

Concurrency note: counter mutations are simple integer increments that
the CPython GIL makes atomic at the bytecode level, so we do NOT take
an ``asyncio.Lock``. This also avoids the well-known pitfall of
``asyncio.Lock`` instances binding to a specific event loop (creating
the lock in one ``asyncio.run()`` and using it in another raises
``RuntimeError: ... is bound to a different event loop``). The sandbox
runs LLM calls concurrently under a semaphore, but each call only
increments one integer — the lost-update window is negligible and, even
if it occurred, would only delay the trip by one call.
"""

from __future__ import annotations

from typing import Optional

from src.llm.exceptions import CircuitBreakerOpen


class CircuitBreaker:
    """Counts consecutive failures across the whole run_all() loop.

    Usage:

        breaker = CircuitBreaker(threshold=3)
        # ... on every successful call:
        await breaker.record_success()
        # ... on every terminal failure (after retries exhausted):
        await breaker.record_failure(last_error=str(exc))

    The breaker is *not* time-decaying — once tripped it stays tripped
    until :meth:`reset` is called explicitly (typically on the next manual
    run after human intervention). This matches the user's "halt the whole
    run" requirement: a sustained failure is a sustained failure regardless
    of when it occurred in the run.
    """

    def __init__(self, threshold: int = 3):
        if threshold < 1:
            raise ValueError("CircuitBreaker threshold must be >= 1")
        self.threshold = threshold
        self._consecutive_failures = 0
        self._last_error: str = ""
        self._tripped = False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def last_error(self) -> str:
        return self._last_error

    async def record_success(self) -> None:
        """Reset the consecutive-failure counter on any successful call."""
        self._consecutive_failures = 0

    async def record_failure(self, last_error: str = "") -> None:
        """Increment the failure counter; trip if threshold reached.

        Does *not* raise — the caller (LLMClient.chat_completion) has
        already raised the underlying API exception. The runner checks
        :attr:`tripped` (or catches the explicit :meth:`check` raise)
        after each cell to decide whether to halt.
        """
        self._consecutive_failures += 1
        if last_error:
            self._last_error = last_error
        if self._consecutive_failures >= self.threshold:
            self._tripped = True

    async def check(self) -> None:
        """Raise :class:`CircuitBreakerOpen` if the breaker has tripped.

        Convenience for callers that want the trip to surface as an
        exception rather than a flag check.
        """
        if self._tripped:
            raise CircuitBreakerOpen(
                consecutive_failures=self._consecutive_failures,
                threshold=self.threshold,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """Clear the tripped state (manual intervention entry point).

        Called by the runner when a ``_FAILED.json`` cell is re-attempted
        after the user has fixed the underlying cause.
        """
        self._consecutive_failures = 0
        self._tripped = False
        self._last_error = ""


# Process-global singleton. The runner grabs this via :func:`get_breaker`
# so test code can substitute a fresh breaker without touching module state.
_GLOBAL_BREAKER: Optional[CircuitBreaker] = None


def get_breaker(threshold: int = 20) -> CircuitBreaker:
    """Return the process-global CircuitBreaker, creating it on first call.

    Default threshold=20: each tenacity-exhausted failure counts as one
    strike. 20 strikes = ~100 raw API attempts (20 × 5 tenacity retries)
    have failed — strong evidence the endpoint is genuinely down, not
    just experiencing transient blips (e.g. HKUST proxy empty-choices).
    """
    global _GLOBAL_BREAKER
    if _GLOBAL_BREAKER is None:
        _GLOBAL_BREAKER = CircuitBreaker(threshold=threshold)
    return _GLOBAL_BREAKER


def reset_global_breaker() -> None:
    """Reset the global breaker — used by tests and by the runner on resume."""
    global _GLOBAL_BREAKER
    if _GLOBAL_BREAKER is not None:
        _GLOBAL_BREAKER.reset()


__all__ = [
    "CircuitBreaker",
    "get_breaker",
    "reset_global_breaker",
]
