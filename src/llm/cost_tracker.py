"""Token and cost tracking per model."""

from __future__ import annotations

from collections import defaultdict


class CostTracker:
    """Tracks token usage and estimated cost per model."""

    def __init__(self) -> None:
        self._input_tokens: dict[str, int] = defaultdict(int)
        self._output_tokens: dict[str, int] = defaultdict(int)
        self._cost: dict[str, float] = defaultdict(float)

    def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ) -> None:
        self._input_tokens[model_name] += input_tokens
        self._output_tokens[model_name] += output_tokens
        cost = (input_tokens / 1000 * cost_per_1k_input
                + output_tokens / 1000 * cost_per_1k_output)
        self._cost[model_name] += cost

    def summary(self) -> dict[str, dict[str, float]]:
        models = set(self._input_tokens) | set(self._output_tokens)
        result = {}
        for m in models:
            result[m] = {
                "input_tokens": self._input_tokens[m],
                "output_tokens": self._output_tokens[m],
                "total_tokens": self._input_tokens[m] + self._output_tokens[m],
                "cost_usd": round(self._cost[m], 4),
            }
        result["_total"] = {
            "input_tokens": sum(self._input_tokens.values()),
            "output_tokens": sum(self._output_tokens.values()),
            "total_tokens": sum(self._input_tokens.values()) + sum(self._output_tokens.values()),
            "cost_usd": round(sum(self._cost.values()), 4),
        }
        return result
