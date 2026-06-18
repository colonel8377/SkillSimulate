"""Tier 2: Pre-generation Mind Models injection (retrieval-augmented).

Dynamically selects the 3-5 most relevant reasoning templates from the
cluster's Mind Models based on the current dialogue state (stance
direction, conflict intensity, topic domain), then injects them into the
LLM prompt before generation (outline §4.4).
"""

from __future__ import annotations

import random
from typing import Any

from src.enforcement.base import EnforcementResult, EnforcementStrategy
from src.enforcement.mind_model_retriever import MindModelRetriever
from src.skill.schema import MindModel
from src.config.embedder import run_embed_in_executor


INJECTION_TEMPLATE = """You are participating in an online discussion. Follow these reasoning guidelines strictly:

{mind_model_text}

Use these frameworks when deciding your stance and crafting your response."""


class Tier2MindModelInjection(EnforcementStrategy):
    """Pre-generation context injection for Mind Models.

    Uses retrieval-augmented rule conditioning: per turn, the retriever
    scores each Mind Model against the inferred dialogue state and only
    the top-k most relevant templates are injected. This contrasts with
    Descriptive Persona's static identity-tag system prompt.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        top_k: int = 5,
        rng: random.Random | None = None,
    ):
        super().__init__(alpha, rng=rng)
        self.retriever = MindModelRetriever(top_k=top_k)
        self.top_k = top_k

    async def check_pre_generation(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Inject the most relevant mind models into messages before generation.

        Args:
            messages: Current LLM message list.
            context: Must contain "mind_models" (list[MindModel]).

        Returns:
            EnforcementResult with modified messages.
        """
        if not self._should_enforce():
            return EnforcementResult(passed=True, tier="none", modified_messages=messages)

        mind_models: list[MindModel] | None = context.get("mind_models")
        if not mind_models:
            return EnforcementResult(passed=True, tier="none", modified_messages=messages)

        # Retrieval-augmented selection: only the top-k relevant models
        state = self.retriever.infer_dialogue_state(messages)
        selected = await run_embed_in_executor(
            self.retriever.retrieve, mind_models, state
        )

        injection_text = self._format_mind_models(selected)
        rule_message = INJECTION_TEMPLATE.format(mind_model_text=injection_text)

        modified = list(messages)
        system_idx = None
        for i, msg in enumerate(modified):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx is not None:
            modified[system_idx] = {
                "role": "system",
                "content": modified[system_idx]["content"] + "\n\n" + rule_message,
            }
        else:
            modified.insert(0, {"role": "system", "content": rule_message})

        return EnforcementResult(
            passed=True,
            tier="tier2",
            modified_messages=modified,
            injection_text=rule_message,
        )

    async def check_post_generation(
        self,
        text: str,
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Tier 2 is pre-generation only — pass through."""
        return EnforcementResult(passed=True, tier="none", original_text=text)

    def _format_mind_models(self, models: list[MindModel]) -> str:
        """Format the selected mind models into injection text."""
        lines = []
        for i, mm in enumerate(models, 1):
            lines.append(f"### Reasoning Framework {i}: {mm.name}")
            lines.append(f"- Principle: {mm.description}")
            lines.append(f"- Apply when: {mm.application}")
            lines.append(f"- Limitation: {mm.limitation}")
            if mm.evidence:
                lines.append(f"- Evidence: {'; '.join(mm.evidence[:2])}")
            lines.append("")
        return "\n".join(lines)
