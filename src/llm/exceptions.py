"""Exceptions raised by the LLM client layer.

These are deliberately separate from the openai SDK's exception hierarchy so
that callers can catch *our* failure modes (budget exceeded, breaker open)
without also catching transient API errors that tenacity is already
retrying.
"""

from __future__ import annotations


class LLMClientError(Exception):
    """Base class for LLM-client errors."""


class TransientResponseError(LLMClientError):
    """Raised when the API returns a structurally valid but unusable
    response (e.g. ``choices=[]`` or empty content) — the kind of failure
    that the OpenAI Python SDK does not classify as an error but that we
    know from experience (DeepSeek / HKUST proxies) is transient.

    Marked retryable so the ``@retry`` decorator on
    :meth:`LLMClient.chat_completion` will back off and try again.
    """


class PromptBudgetExceeded(LLMClientError):
    """Raised when the assembled prompt would exceed the model's
    ``max_input_tokens`` budget (i.e. ``max_total_tokens`` minus
    ``max_thinking_tokens`` and ``max_output_tokens``).

    This is a *recoverable* error from the agent's perspective — the caller
    is expected to truncate memory / thread history and retry. It is included
    in the retryable exception set for the ``@retry`` decorator only when the
    caller has indicated it can compact; otherwise it propagates immediately.
    """

    def __init__(self, model: str, requested: int, budget: int):
        self.model = model
        self.requested = requested
        self.budget = budget
        super().__init__(
            f"Prompt budget exceeded for model '{model}': "
            f"estimated {requested} input tokens > budget {budget}. "
            f"Truncate memory / thread context and retry."
        )


class CircuitBreakerOpen(LLMClientError):
    """Raised when the global circuit breaker has tripped.

    Trip condition: ``consecutive_failures >= threshold`` across the whole
    ``run_all()`` loop. The caller (``runner.run_all``) catches this to
    write a ``_FAILED.json`` marker, log the failure, and exit non-zero so
    a human can investigate before more API budget is spent.
    """

    def __init__(self, consecutive_failures: int, threshold: int, last_error: str):
        self.consecutive_failures = consecutive_failures
        self.threshold = threshold
        self.last_error = last_error
        super().__init__(
            f"Circuit breaker open: {consecutive_failures} consecutive failures "
            f"(threshold={threshold}). Last error: {last_error}"
        )


class AllEndpointsExhausted(LLMClientError):
    """Raised when all endpoints for a model have been marked dead.

    Occurs when every endpoint in an :class:`EndpointPool` has exceeded
    the consecutive-failure threshold. The runner catches this to write a
    ``_FAILED.json`` marker and halt — there is no point trying other
    cells when no endpoint is functional.
    """

    def __init__(self, model_name: str, dead_count: int):
        self.model_name = model_name
        self.dead_count = dead_count
        super().__init__(
            f"All {dead_count} endpoints exhausted for model '{model_name}'. "
            f"Manual intervention required — check keys, quotas, and endpoint health."
        )
