"""Periodic reflection — belief consolidation and Mind Models reinforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schemas import Message
from src.llm.token_counter import truncate_to_token_budget


REFLECTION_PROMPT = """You are reflecting on your recent interactions in an online community.

Your recent messages:
{recent_messages}

Community context:
{context}

Summarize your key beliefs and stances after these interactions.
Focus on:
1. What positions have you taken?
2. What arguments have been most effective?
3. Have any of your stances shifted?

Respond in 2-3 sentences. Do not mention that you are reflecting."""


@dataclass
class ReflectionState:
    """Agent's consolidated beliefs after reflection."""
    summary: str = ""
    key_positions: list[str] = field(default_factory=list)
    round: int = 0


class ReflectionModule:
    """Manages periodic belief consolidation."""

    def __init__(self, llm_client, model_name: str = "gpt-4o", interval: int = 10, max_memory_tokens: int = 0):
        self.llm = llm_client
        self.model_name = model_name
        self.interval = interval
        self.max_memory_tokens = max_memory_tokens
        self.state = ReflectionState()

    def should_reflect(self, current_round: int) -> bool:
        """Check if it's time to reflect."""
        return current_round > 0 and current_round % self.interval == 0

    async def reflect(
        self,
        recent_messages: list[Message],
        context: str = "",
        current_round: int = 0,
    ) -> ReflectionState:
        """Run reflection on recent interactions.

        Args:
            recent_messages: Messages since last reflection.
            context: Community context for reflection.
            current_round: Current simulation round.

        Returns:
            Updated ReflectionState.
        """
        if not recent_messages:
            return self.state

        msgs_text = "\n".join(
            f"[Round {m.metadata.get('round', '?')}]: "
            f"{truncate_to_token_budget(m.text, max(80, self.max_memory_tokens // 20)) if self.max_memory_tokens else m.text[:150]}"
            for m in recent_messages[-10:]
        )

        prompt = REFLECTION_PROMPT.format(
            recent_messages=msgs_text,
            context=context or "(no additional context)",
        )

        messages = [
            {"role": "system", "content": "You are reflecting on your online interactions."},
            {"role": "user", "content": prompt},
        ]

        summary = await self.llm.chat_completion(messages, self.model_name, temperature=0.5)

        self.state = ReflectionState(
            summary=summary,
            key_positions=self._extract_positions(summary),
            round=current_round,
        )

        return self.state

    def _extract_positions(self, summary: str) -> list[str]:
        """Extract key positions from reflection summary."""
        sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 10]
        return sentences[:3]
