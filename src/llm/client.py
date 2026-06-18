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
from dataclasses import dataclass
from typing import Any

import yaml
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.cost_tracker import CostTracker
from src.llm.json_utils import extract_json
from src.llm.mock_router import MockLLMRouter


@dataclass
class ModelEndpoint:
    name: str
    model: str          # API model identifier
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class LLMClient:
    """Async LLM client that routes to OpenAI-compatible endpoints."""

    def __init__(self, models_config_path: str = "configs/models.yaml"):
        self.models = self._load_models(models_config_path)
        self.clients: dict[str, AsyncOpenAI] = {}
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
        self._init_clients()

    def _load_models(self, path: str) -> dict[str, ModelEndpoint]:
        with open(path) as f:
            config = yaml.safe_load(f)
        endpoints = {}
        for entry in config.get("models", []):
            endpoints[entry["name"]] = ModelEndpoint(
                name=entry["name"],
                model=entry["model"],
                base_url=entry.get("base_url", ""),
                api_key=entry.get("api_key", ""),
                temperature=entry.get("temperature", 0.7),
                max_tokens=entry.get("max_tokens", 2048),
                cost_per_1k_input=entry.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=entry.get("cost_per_1k_output", 0.0),
            )
        return endpoints

    def _init_clients(self) -> None:
        # Skip real client init when mock mode is on — saves the caller
        # from having to provide an API key for dev-loop validation.
        if self._mock_enabled:
            return
        from src.config.settings import settings
        default_key = settings.openai_api_key
        default_url = settings.openai_base_url
        for name, ep in self.models.items():
            api_key = ep.api_key or default_key
            base_url = ep.base_url or default_url
            self.clients[name] = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: OpenAI-format messages [{"role": ..., "content": ...}].
            model_name: Key from models.yaml (e.g. "gpt-4o").
            temperature: Override model default if set.
            max_tokens: Override model default if set.

        Returns:
            Generated text content.
        """
        if self._mock_enabled:
            return self._mock.route(messages, model_name)

        ep = self.models[model_name]
        client = self.clients[model_name]
        temp = temperature if temperature is not None else ep.temperature
        tokens = max_tokens if max_tokens is not None else ep.max_tokens

        response = await client.chat.completions.create(
            model=ep.model,
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

        return response.choices[0].message.content or ""

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
