"""Importance-weighted retrieval memory for agents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.data.schemas import Message


@dataclass
class MemoryItem:
    """Single memory item."""
    message: Message
    importance: float = 1.0  # higher = more important
    round: int = 0


class AgentMemory:
    """Manages agent's conversation history with importance-weighted retrieval."""

    def __init__(self, max_context_items: int = 20):
        self.max_context_items = max_context_items
        self._items: list[MemoryItem] = []

    def add(self, message: Message, round: int = 0, importance: float = 1.0) -> None:
        """Add a message to memory."""
        self._items.append(MemoryItem(
            message=message,
            importance=importance,
            round=round,
        ))

    def add_all(self, messages: list[Message], round: int = 0) -> None:
        """Add multiple messages with auto-computed importance."""
        for msg in messages:
            importance = self._compute_importance(msg, round)
            self.add(msg, round, importance)

    def retrieve(self, thread_id: str | None = None, current_round: int = 0) -> list[Message]:
        """Retrieve top-k most relevant memories.

        Args:
            thread_id: If set, prioritize messages from this thread.
            current_round: Current round for recency weighting.

        Returns:
            List of messages, most important first.
        """
        scored = []
        for item in self._items:
            score = item.importance

            # Recency boost: more recent = higher score
            recency = 1.0 / (1.0 + (current_round - item.round) * 0.1)
            score *= recency

            # Thread relevance boost
            if thread_id and item.message.thread_id == thread_id:
                score *= 2.0

            scored.append((score, item))

        # Sort by score descending and take top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item.message for _, item in scored[:self.max_context_items]]

    def get_thread_history(self, thread_id: str) -> list[Message]:
        """Get all messages from a specific thread."""
        return [
            item.message for item in self._items
            if item.message.thread_id == thread_id
        ]

    def _compute_importance(self, msg: Message, round: int) -> float:
        """Compute importance score for a message."""
        score = 1.0

        # Longer messages are more important
        if len(msg.text) > 200:
            score += 0.5

        # Messages involving conflict are more important
        from src.data.schemas import ActionType
        if msg.action_type in {ActionType.DISAGREE, ActionType.REVERT, ActionType.COUNTER_ARGUE}:
            score += 1.0

        # Award delta events are important
        if msg.action_type == ActionType.AWARD_DELTA:
            score += 2.0

        return score

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
