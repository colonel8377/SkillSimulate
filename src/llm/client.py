"""Async OpenAI-compatible LLM client supporting multiple model endpoints.

Mock mode (development only)
============================
Set ``CADP_MOCK_LLM=1`` in the environment to route every ``chat_completion``
through a deterministic, no-network mock that emits valid JSON for every
contract the CADP pipeline exercises (planner.plan / replan, skill-compiler
quality check, mind-model extraction + verification, anti-pattern detection).

The mock is gated *inside* ``chat_completion`` so callers do not need to be
aware of it. It exists to let the dev config (``configs/dev.yaml``) exercise
the full data → cluster → skill compile → sim → metrics → save loop without
spending API budget. Outputs are clearly NOT paper-grade and the
``MockLLMRouter`` log warns once on first use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.circuit_breaker import get_breaker
from src.llm.cost_tracker import CostTracker
from src.llm.endpoint_pool import EndpointConfig, EndpointPool, EndpointState
from src.llm.exceptions import (
    AllEndpointsExhausted,
    CircuitBreakerOpen,
    PromptBudgetExceeded,
    TransientResponseError,
)
from src.llm.json_utils import extract_json
from src.llm.mock_router import MockLLMRouter
from src.llm.token_counter import estimate_messages_tokens


# Only these exceptions are worth retrying — they are transient by definition.
# Auth / permission / bad-request / budget-exceeded errors will fail
# identically on every retry, so propagating them immediately surfaces
# config bugs instead of burning retry budget.
_RETRYABLE = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    TransientResponseError,
)


@dataclass
class ModelEndpoint:
    name: str
    model: str          # API model identifier (sent to provider — may change)
    base_url: str = ""
    api_key: str = ""
    # api_type: which API format to call.
    #   "auto"        — detect from base_url suffix at init time
    #   "chat"        — /chat/completions  (OpenAI-compatible, default)
    #   "completions" — /completions       (legacy text-completion endpoint)
    # The config key ``name`` is the stable internal reference; ``model``
    # and ``api_type`` are provider-specific and may change when switching
    # URLs.
    api_type: str = "auto"
    # Some OpenAI-compatible proxies (e.g. HKUST gateway) do not support
    # ``response_format={"type": "json_object"}``.  When False,
    # ``chat_completion_json`` skips that kwarg and relies on
    # ``extract_json`` to parse the response.
    supports_json_mode: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    # P5 (audit): reproducibility provenance — outline §5.1(d). For hosted
    # APIs, ``snapshot_date`` is the dated snapshot ID the vendor pins
    # behaviour to (e.g. ``2024-08-06`` for ``gpt-4o-2024-08-06``); for
    # OSS models, ``commit_hash`` is the model-weight git commit / HF
    # revision. Either may be empty when unspecified, in which case the
    # paper's reproducibility table will surface ``unrecorded`` for that
    # row instead of silently omitting it.
    snapshot_date: str = ""
    commit_hash: str = ""
    # --- Token budget caps (Issue 1) ----------------------------------
    # ``max_tokens`` above remains the *output* cap (passed to the API as
    # the OpenAI ``max_tokens`` kwarg). For reasoning models whose total
    # token budget (input + output + thinking) is capped — e.g.
    # DeepSeek-V4-Flash at 65535 — the three fields below let the
    # pre-flight guard in :meth:`LLMClient.chat_completion` refuse to
    # send a prompt that would overflow the input budget. For non-
    # reasoning models, leave ``max_total_tokens=0`` and the guard is
    # skipped (back-compat with pre-existing behaviour).
    max_total_tokens: int = 0      # 0 = no shared cap (legacy behaviour)
    max_thinking_tokens: int = 0   # 0 = no thinking budget reserved
    # ``max_output_tokens`` is an alias for ``max_tokens`` — both fields
    # are kept in sync via __post_init__ so callers can use whichever
    # name reads better at the call site.
    max_output_tokens: int = 0     # 0 = mirror max_tokens in __post_init__
    # --- Multi-endpoint support ----------------------------------------
    # When non-empty, the client creates an EndpointPool and round-robins
    # across these endpoints instead of using the parent base_url/api_key.
    # Each endpoint may override the model name (e.g. when the same
    # DeepSeek-V4-Flash is called "deepseek-v4-flash-260425" on one proxy
    # and "DeepSeek-V4-Flash" on another).
    endpoints: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep max_tokens and max_output_tokens in sync. Whoever was set
        # explicitly wins; if both were set inconsistently, max_tokens
        # wins because it is the historical field name.
        if self.max_output_tokens and not self.max_tokens:
            self.max_tokens = self.max_output_tokens
        elif self.max_tokens and not self.max_output_tokens:
            self.max_output_tokens = self.max_tokens
        elif self.max_tokens and self.max_output_tokens and \
                self.max_tokens != self.max_output_tokens:
            from loguru import logger
            logger.warning(
                f"ModelEndpoint '{self.name}': max_tokens={self.max_tokens} "
                f"!= max_output_tokens={self.max_output_tokens}; "
                f"using max_tokens={self.max_tokens}."
            )
            self.max_output_tokens = self.max_tokens

    @property
    def max_input_tokens(self) -> int:
        """Input-token budget = total − thinking − output.

        Returns 0 when ``max_total_tokens`` is 0 (legacy mode, guard
        skipped). When thinking + output ≥ total (misconfigured),
        returns 0 — the pre-flight guard will then refuse every prompt,
        surfacing the config bug loudly.
        """
        if self.max_total_tokens <= 0:
            return 0
        return max(
            0,
            self.max_total_tokens - self.max_thinking_tokens - self.max_output_tokens,
        )


# Suffixes that the OpenAI Python SDK appends automatically.  If the
# user pastes a full endpoint URL into ``base_url`` (e.g.
# ``https://host/v1/chat/completions``), the SDK would produce a broken
# double-path.  ``_normalize_base_url`` strips these so the SDK always
# receives a clean root URL.
_API_SUFFIXES = (
    "/chat/completions",      # try endpoint-only first (preserves /v1)
    "/completions",
    "/v1/chat/completions",   # then full path (strips /v1 too)
    "/v1/completions",
)


def _normalize_base_url(url: str) -> tuple[str, str]:
    """Strip endpoint suffixes and detect API type from *url*.

    Returns:
        (clean_base_url, detected_api_type) where *detected_api_type*
        is ``"chat"`` or ``"completions"``.  When no known suffix is
        found the URL is returned as-is with ``"chat"`` (the modern
        default).
    """
    url = url.rstrip("/")
    for suffix in _API_SUFFIXES:
        if url.endswith(suffix):
            clean = url[: -len(suffix)]
            api_type = "completions" if "chat" not in suffix else "chat"
            return clean, api_type
    return url, "chat"


class LLMClient:
    """Async LLM client that routes to OpenAI-compatible endpoints."""

    def __init__(self, models_config_path: str = "configs/models.yaml"):
        # Load .env before reading os.getenv for endpoint vars.
        # _load_models() calls _load_endpoints_from_env() which needs them.
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)

        self.models = self._load_models(models_config_path)
        self.clients: dict[str, AsyncOpenAI] = {}
        self.endpoint_pools: dict[str, EndpointPool] = {}
        self.cost_tracker = CostTracker()
        self._mock_enabled = os.getenv("CADP_MOCK_LLM", "0") == "1"
        if self._mock_enabled:
            self._mock = MockLLMRouter()
            logger.warning(
                "CADP_MOCK_LLM=1 — LLMClient is using a deterministic mock. "
                "DO NOT use outputs as paper data."
            )
        else:
            self._mock = None
        # Process-global circuit breaker. Threshold is configurable via env
        # so operators can tighten it for fragile endpoints. The breaker is
        # reset on each new runner.run_all() invocation.
        self._breaker = get_breaker(
            threshold=int(os.getenv("CADP_BREAKER_THRESHOLD", "20"))
        )
        self._init_clients()

    def _load_models(self, path: str) -> dict[str, ModelEndpoint]:
        with open(path) as f:
            config = yaml.safe_load(f)
        endpoints = {}
        for entry in config.get("models", []):
            # Endpoints are loaded from env, not YAML.
            # YAML may contain a comment-placeholder ``endpoints:`` key
            # (parsed as None); we ignore it and always read from env.
            endpoints[entry["name"]] = ModelEndpoint(
                name=entry["name"],
                model=entry["model"],
                base_url=entry.get("base_url", ""),
                api_key=entry.get("api_key", ""),
                api_type=entry.get("api_type", "auto"),
                supports_json_mode=entry.get("supports_json_mode", True),
                temperature=entry.get("temperature", 0.7),
                max_tokens=entry.get("max_tokens", 2048),
                cost_per_1k_input=entry.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=entry.get("cost_per_1k_output", 0.0),
                snapshot_date=str(entry.get("snapshot_date", "")),
                commit_hash=str(entry.get("commit_hash", "")),
                max_total_tokens=int(entry.get("max_total_tokens", 0)),
                max_thinking_tokens=int(entry.get("max_thinking_tokens", 0)),
                max_output_tokens=int(entry.get("max_output_tokens", entry.get("max_tokens", 0))),
                endpoints=self._load_endpoints_from_env(entry["name"]),
            )
        return endpoints

    @staticmethod
    def _load_endpoints_from_env(model_name: str) -> list[dict[str, str]]:
        """Load endpoint list from environment variable.

        Reads ``CADP_ENDPOINTS_<NAME>`` where ``<NAME>`` is *model_name*
        uppercased with hyphens replaced by underscores.

        Example for ``deepseek-v4-flash``::

            CADP_ENDPOINTS_DEEPSEEK_V4_FLASH='[
              {"base_url": "https://api.deepseek.com", "api_key": "sk-xxx", "model": "deepseek-chat"},
              {"base_url": "https://proxy.example.com/v1", "api_key": "sk-yyy", "model": "deepseek-v4-flash"}
            ]'

        When the env var is unset or empty, returns an empty list (legacy
        single-endpoint path).
        """
        env_key = f"CADP_ENDPOINTS_{model_name.upper().replace('-', '_')}"
        raw = os.getenv(env_key, "")
        if not raw.strip():
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                f"Failed to parse {env_key} as JSON: {exc}. "
                f"Skipping multi-endpoint for model '{model_name}'."
            )
            return []
        if not isinstance(parsed, list):
            logger.error(
                f"{env_key} is not a JSON array. "
                f"Skipping multi-endpoint for model '{model_name}'."
            )
            return []
        # Validate each entry is a dict
        result: list[dict[str, str]] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.error(
                    f"{env_key}[{i}] is not a JSON object; skipping entry."
                )
                continue
            result.append({
                "base_url": str(item.get("base_url", "")),
                "api_key": str(item.get("api_key", "")),
                "model": str(item.get("model", "")),
            })
        if result:
            logger.info(
                f"Loaded {len(result)} endpoint(s) from {env_key} "
                f"for model '{model_name}'"
            )
        return result

    def _init_clients(self) -> None:
        # Skip real client init when mock mode is on — saves the caller
        # from having to provide an API key for dev-loop validation.
        if self._mock_enabled:
            return
        from src.config.settings import settings
        default_key = settings.openai_api_key
        default_url = settings.openai_base_url
        for name, ep in self.models.items():
            # --- Multi-endpoint path ---
            if ep.endpoints:
                pool_states: list[EndpointState] = []
                for i, ep_cfg in enumerate(ep.endpoints):
                    api_key = ep_cfg.get("api_key", "") or default_key
                    raw_url = ep_cfg.get("base_url", "") or default_url
                    clean_url, detected_type = _normalize_base_url(raw_url)
                    ep_model = ep_cfg.get("model", "") or ep.model
                    client = AsyncOpenAI(
                        api_key=api_key, base_url=clean_url,
                        timeout=float(os.getenv("CADP_API_TIMEOUT", "120")),
                    )
                    pool_states.append(EndpointState(
                        config=EndpointConfig(
                            base_url=clean_url,
                            api_key=api_key,
                            model=ep_model,
                        ),
                        client=client,
                        model=ep_model,
                        idx=i,
                    ))
                failure_threshold = int(os.getenv(
                    "CADP_ENDPOINT_FAILURE_THRESHOLD", "15"
                ))
                cooldown_seconds = float(os.getenv(
                    "CADP_ENDPOINT_COOLDOWN_SECONDS", "1800"
                ))
                pool = EndpointPool(
                    model_name=name,
                    endpoints=pool_states,
                    failure_threshold=failure_threshold,
                    cooldown_seconds=cooldown_seconds,
                )
                self.endpoint_pools[name] = pool
                logger.info(
                    f"Model '{name}': {len(pool_states)} endpoint(s) pooled "
                    f"(failure_threshold={failure_threshold})"
                )
                for st in pool_states:
                    logger.info(
                        f"  endpoint[{st.idx}]: model='{st.model}' "
                        f"base_url='{st.config.base_url}'"
                    )
                continue  # skip legacy single-client init for this model
            # --- Legacy single-client path ---
            api_key = ep.api_key or default_key
            raw_url = ep.base_url or default_url
            if not api_key.strip():
                logger.warning(
                    f"Model '{name}': no api_key configured (neither "
                    f"models.yaml nor CADP_OPENAI_API_KEY is set). "
                    f"Skipping legacy client — calls to this model will fail."
                )
                continue
            # Strip endpoint suffixes so the SDK gets a clean root URL.
            clean_url, detected_type = _normalize_base_url(raw_url)
            # Resolve api_type: explicit config > auto-detect from URL.
            if ep.api_type == "auto":
                ep.api_type = detected_type
            if ep.api_type != "chat":
                logger.info(
                    f"Model '{name}' resolved api_type='{ep.api_type}' "
                    f"(base_url='{raw_url}' → '{clean_url}')"
                )
            self.clients[name] = AsyncOpenAI(
                api_key=api_key, base_url=clean_url,
                timeout=float(os.getenv("CADP_API_TIMEOUT", "120")),
            )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat completion (public API).

        Wraps :meth:`_chat_completion_with_retry` with circuit-breaker
        accounting. The breaker is checked before every call and any
        exception that survives the inner retry loop is recorded as a
        failure — this is what makes sustained issues trip the run-all
        halt. Successful calls reset the consecutive-failure counter.

        When the model has an :class:`EndpointPool`, this method round-robins
        across alive endpoints. An endpoint that exhausts its 5 retries gets
        a failure recorded on the pool; after 10 consecutive failures the
        endpoint is marked dead and the next alive endpoint is tried. When
        all endpoints are dead, :class:`AllEndpointsExhausted` is raised.

        Args:
            messages: OpenAI-format messages [{"role": ..., "content": ...}].
            model_name: Key from models.yaml (e.g. "gpt-4o").
            temperature: Override model default if set.
            max_tokens: Override model default if set.

        Returns:
            Generated text content.

        Raises:
            CircuitBreakerOpen: breaker has tripped; halts run_all().
            AllEndpointsExhausted: all endpoints dead for this model.
            PromptBudgetExceeded: prompt exceeds max_input_tokens.
            AuthenticationError / BadRequestError / etc.: non-retryable,
                propagated after being recorded on the breaker.
            RateLimitError / APITimeoutError / etc.: retried up to 5
                times; if all retries fail, recorded on the breaker and
                re-raised.
        """
        await self._breaker.check()
        pool = self.endpoint_pools.get(model_name)

        # --- Multi-endpoint path ---
        if pool is not None:
            last_error: str = ""
            while True:
                try:
                    ep_state = await pool.pick_alive()
                except AllEndpointsExhausted:
                    await self._breaker.record_failure(
                        last_error=f"AllEndpointsExhausted for {model_name}"
                    )
                    raise

                try:
                    content = await self._chat_completion_with_retry(
                        messages, model_name, temperature, max_tokens,
                        endpoint_state=ep_state,
                        **kwargs,
                    )
                    await self._breaker.record_success()
                    pool.record_success(ep_state)
                    return content
                except AllEndpointsExhausted:
                    raise  # re-raised from pick_alive inside retry
                except CircuitBreakerOpen:
                    raise
                except Exception as exc:
                    pool.record_failure(ep_state, error=str(exc))
                    last_error = str(exc)
                    if pool.all_dead:
                        logger.error(
                            f"All endpoints dead for model '{model_name}' "
                            f"after exhausting last endpoint "
                            f"#{ep_state.idx}: {exc!r}"
                        )
                        await self._breaker.record_failure(last_error=last_error)
                        raise AllEndpointsExhausted(model_name, pool.total_endpoints)
                    logger.warning(
                        f"EndpointPool[{model_name}]#{ep_state.idx} failed "
                        f"({pool.alive_count} alive remaining). "
                        f"Retrying with next endpoint."
                    )
                    # loop to next alive endpoint
                    continue

        # --- Legacy single-client path ---
        try:
            content = await self._chat_completion_with_retry(
                messages, model_name, temperature, max_tokens, **kwargs,
            )
            await self._breaker.record_success()
            return content
        except CircuitBreakerOpen:
            raise  # already accounted for
        except Exception as exc:
            # Any exception that survived the retry loop counts as a
            # terminal failure for breaker purposes. This includes both
            # non-retryable exceptions (auth, bad request) on attempt 1
            # AND retryable exceptions (rate limit, timeout) that burnt
            # all 5 attempts.
            await self._breaker.record_failure(last_error=str(exc))
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _chat_completion_with_retry(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        temperature: float | None,
        max_tokens: int | None,
        endpoint_state: EndpointState | None = None,
        **kwargs: Any,
    ) -> str:
        """Inner retryable completion — do not call directly.

        Tenacity wraps this method; the public :meth:`chat_completion`
        handles breaker accounting around it.

        When *endpoint_state* is provided, the endpoint's client and
        model name are used instead of the parent model entry's.
        """
        if self._mock_enabled:
            return self._mock.route(messages, model_name)

        ep = self.models[model_name]
        # When routed through an endpoint pool, use the endpoint's
        # client and per-endpoint model name.
        if endpoint_state is not None:
            client = endpoint_state.client
            effective_model = endpoint_state.model
        else:
            client = self.clients[model_name]
            effective_model = ep.model
        temp = temperature if temperature is not None else ep.temperature
        tokens = max_tokens if max_tokens is not None else ep.max_tokens

        # --- Pre-flight token budget guard (Issue 1) ------------------
        # Refuse to send prompts that would overflow the input budget.
        # Uses a conservative tiktoken estimate (1.15× multiplier); when
        # max_total_tokens is 0 the guard is skipped (legacy behaviour).
        if ep.max_input_tokens > 0:
            estimated = estimate_messages_tokens(messages, model=ep.model)
            if estimated > ep.max_input_tokens:
                raise PromptBudgetExceeded(
                    model=ep.model,
                    requested=estimated,
                    budget=ep.max_input_tokens,
                )

        # --- thinking-budget passthrough (Issue 1) --------------------
        # DeepSeek-V4 reasoning models use a separate ``reasoning_content``
        # field for chain-of-thought output. The thinking budget is controlled
        # by the endpoint's ``max_thinking_tokens`` setting, which is used
        # by the pre-flight token budget guard (above) but NOT passed as an
        # API kwarg — the DeepSeek OpenAI-compatible API does not accept
        # ``thinking_budget`` or ``reasoning_effort`` kwargs. Thinking is
        # automatic and the endpoint manages the token split internally.

        # ---- route by api_type ----
        if ep.api_type == "completions":
            response = await self._call_completions(
                client, ep, messages, temp, tokens,
                effective_model=effective_model,
                **kwargs,
            )
        else:
            response = await client.chat.completions.create(
                model=effective_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                **kwargs,
            )

        # Track cost
        usage = response.usage
        if usage:
            self.cost_tracker.record(
                model_name=model_name,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost_per_1k_input=ep.cost_per_1k_input,
                cost_per_1k_output=ep.cost_per_1k_output,
            )

        # Defensive guard: some proxies (e.g. HKUST gateway) occasionally
        # return a response with ``choices=None`` on transient errors.
        # Raising TransientResponseError here triggers the @retry backoff.
        if not response.choices:
            raise TransientResponseError(
                f"API returned empty choices for model '{effective_model}' "
                f"(response.id={getattr(response, 'id', '?')})"
            )

        content = response.choices[0].message.content

        # Defensive guard: reasoning models (DeepSeek-V4, QwQ, etc.) may
        # leak their chain-of-thought <think>...</think> tags into the ``content``
        # field instead of the separate ``reasoning_content`` field,
        # depending on the proxy implementation.  Strip any <think> blocks
        # from the returned content so downstream agents receive clean text.
        if content and "<think>" in content:
            import re as _re
            content = _re.sub(
                r"<think>[\s\S]*?</think>", "", content
            ).strip()

        # Defensive guard: DeepSeek / HKUST proxies occasionally return
        # a valid response structure but with empty or None content.
        # Raising TransientResponseError here triggers the @retry backoff.
        if not content or not content.strip():
            raise TransientResponseError(
                f"API returned empty content for model '{effective_model}' "
                f"(response.id={getattr(response, 'id', '?')})"
            )

        return content

    # ------------------------------------------------------------------
    # Legacy /completions adapter
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt string.

        Used when the endpoint only supports ``/completions`` (text-in /
        text-out) rather than ``/chat/completions``.
        """
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    async def _call_completions(
        self,
        client: AsyncOpenAI,
        ep: ModelEndpoint,
        messages: list[dict[str, str]],
        temp: float,
        tokens: int,
        effective_model: str = "",
        **kwargs: Any,
    ) -> Any:
        """Call the legacy ``/completions`` endpoint.

        Converts chat messages to a flat prompt, calls
        ``client.completions.create``, and wraps the response so the
        rest of the pipeline (cost tracking, content extraction) works
        unchanged.
        """
        prompt = self._messages_to_prompt(messages)
        # Strip chat-only kwargs that /completions does not accept.
        kwargs.pop("response_format", None)
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
        kwargs.pop("thinking_budget", None)
        kwargs.pop("reasoning_effort", None)
        model_name = effective_model or ep.model
        return await client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            **kwargs,
        )

    async def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        default: Any = None,
    ) -> Any:
        """Generate a completion and parse it as JSON.

        Requests ``response_format={"type": "json_object"}`` (when not in
        mock mode) so the endpoint returns syntactically valid JSON, then
        routes the content through :func:`extract_json` as a robustness
        backstop for endpoints that ignore ``response_format`` or for the
        mock path. Callers that previously did their own
        ``json.loads`` + regex fallback should use this instead.

        Args:
            default: Value returned if no JSON can be parsed.

        Returns:
            Parsed Python object, or *default*.
        """
        kwargs: dict[str, Any] = {}
        if not self._mock_enabled:
            ep = self.models.get(model_name)
            # response_format is only supported by /chat/completions AND
            # when the endpoint actually honours it (not all proxies do).
            if (
                ep is not None
                and ep.api_type != "completions"
                and ep.supports_json_mode
            ):
                kwargs["response_format"] = {"type": "json_object"}
        response = await self.chat_completion(
            messages,
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return extract_json(response, default=default)

    async def batch_completions(
        self,
        batch: list[tuple[list[dict[str, str]], str]],
        max_concurrency: int = 4,
    ) -> list[str]:
        """Run multiple completions concurrently.

        Args:
            batch: List of (messages, model_name) tuples.
            max_concurrency: Max concurrent requests.

        Returns:
            List of generated texts, same order as input.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(idx: int) -> str:
            msgs, model = batch[idx]
            async with semaphore:
                return await self.chat_completion(msgs, model)

        tasks = [_guarded(i) for i in range(len(batch))]
        return await asyncio.gather(*tasks)

    def get_cost_summary(self) -> dict[str, dict[str, float]]:
        """Return per-model cost breakdown."""
        return self.cost_tracker.summary()

    def get_provenance(self, model_name: str) -> dict[str, str]:
        """Return reproducibility provenance for a registered model.

        Audit P5 / outline §5.1(d): every result row should carry the
        dated snapshot or commit hash that produced it. Returns
        ``{"snapshot_date": ..., "commit_hash": ..., "model_id": ...}``;
        missing fields are empty strings (never raised) so callers can
        always stamp the row.
        """
        ep = self.models.get(model_name)
        if ep is None:
            return {"snapshot_date": "", "commit_hash": "", "model_id": model_name}
        return {
            "snapshot_date": ep.snapshot_date or "",
            "commit_hash": ep.commit_hash or "",
            "model_id": ep.model,
        }

    def get_all_provenance(self) -> dict[str, dict[str, str]]:
        """Provenance for every registered model (audit P5)."""
        return {name: self.get_provenance(name) for name in self.models}
