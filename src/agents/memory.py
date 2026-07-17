"""Importance-weighted retrieval memory for agents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.data.schemas import Message
from src.llm.token_counter import estimate_tokens


@dataclass
class MemoryItem:
    """Single memory item.

    ``kind`` distinguishes raw observed/sent messages ("event") from
    rolling-summary compaction outputs ("summary"). Summary items are
    produced by :class:`src.agents.reflection.RollingSummaryCompactor`
    and behave like ordinary items for retrieval purposes — they are
    typically high-importance because they condense many turns.
    """
    message: Message
    importance: float = 1.0  # higher = more important
    round: int = 0
    kind: str = "event"  # "event" | "summary"


class AgentMemory:
    """Manages agent's conversation history with importance-weighted retrieval.

    Two retrieval modes:
      * **Item-count** (legacy, default): ``max_context_items`` highest-ranked items.
      * **Token-budget**: as many ranked items as fit within ``max_context_tokens``.

    The two are not mutually exclusive — :meth:`retrieve` checks the token
    budget first (when set) and falls back to the item-count cap.
    """

    def __init__(
        self,
        max_context_items: int = 20,
        max_context_tokens: int = 0,
    ):
        self.max_context_items = max_context_items
        self.max_context_tokens = max_context_tokens
        self._items: list[MemoryItem] = []

    def add(self, message: Message, round: int = 0, importance: float = 1.0, kind: str = "event") -> None:
        """Add a message to memory."""
        self._items.append(MemoryItem(
            message=message,
            importance=importance,
            round=round,
            kind=kind,
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
        scored = self._score(thread_id, current_round)
        # Token-budget path: accumulate in ranked order until budget hit.
        # When max_context_tokens is 0 we fall through to item-count mode.
        if self.max_context_tokens > 0:
            selected: list[Message] = []
            running = 0
            for _, item in scored:
                # Per-item token cost (with 1.15× safety multiplier baked
                # into estimate_tokens). Reserve 8 tokens for the
                # formatter framing ("[user] (action): " etc.).
                cost = estimate_tokens(item.message.text) + 8
                if running + cost > self.max_context_tokens and selected:
                    break
                selected.append(item.message)
                running += cost
                # Hard ceiling: never exceed item-count cap even when
                # budget allows. Prevents pathological cases where many
                # tiny messages blow up the planner prompt structurally.
                if len(selected) >= self.max_context_items:
                    break
            return selected
        # Legacy item-count path
        return [item.message for _, item in scored[:self.max_context_items]]

    def _score(self, thread_id: str | None, current_round: int) -> list[tuple[float, MemoryItem]]:
        """Return all items as (score, item) tuples, sorted desc."""
        scored = []
        for item in self._items:
            score = item.importance

            # Recency boost: more recent = higher score
            recency = 1.0 / (1.0 + (current_round - item.round) * 0.1)
            score *= recency

            # Thread relevance boost
            if thread_id and item.message.thread_id == thread_id:
                score *= 2.0

            # Summary items get a small boost so compaction outputs are
            # preferred over individual events of equal score — this
            # keeps long-horizon signal in context.
            if item.kind == "summary":
                score *= 1.5

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def get_thread_history(self, thread_id: str) -> list[Message]:
        """Get all messages from a specific thread."""
        return [
            item.message for item in self._items
            if item.message.thread_id == thread_id
        ]

    def get_items_of_kind(self, kind: str) -> list[MemoryItem]:
        """Return all stored items of the given kind (e.g. 'summary')."""
        return [item for item in self._items if item.kind == kind]

    def pop_oldest_events(self, n: int) -> list[MemoryItem]:
        """Remove and return the *n* oldest ``kind='event'`` items.

        Used by :class:`src.agents.reflection.RollingSummaryCompactor` to
        free space after a summary has been produced. Returns the popped
        items in chronological order (oldest first). If fewer than *n*
        event items exist, all of them are returned.
        """
        event_items = [it for it in self._items if it.kind == "event"]
        event_items.sort(key=lambda it: it.round)
        to_remove = set(id(it) for it in event_items[:n])
        popped = [it for it in event_items if id(it) in to_remove]
        self._items = [it for it in self._items if id(it) not in to_remove]
        return popped

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

    def export_state(self) -> list[dict]:
        """Return a JSON-serializable lossless memory snapshot."""
        return [
            {
                "message": {
                    "msg_id": item.message.msg_id,
                    "thread_id": item.message.thread_id,
                    "user_id": item.message.user_id,
                    "platform": item.message.platform.value,
                    "timestamp": item.message.timestamp.isoformat(),
                    "text": item.message.text,
                    "action_type": item.message.action_type.value,
                    "parent_msg_id": item.message.parent_msg_id,
                    "metadata": dict(item.message.metadata),
                },
                "importance": item.importance,
                "round": item.round,
                "kind": item.kind,
            }
            for item in self._items
        ]

    def restore_state(self, items: list[dict]) -> None:
        """Replace memory contents from :meth:`export_state` output."""
        from datetime import datetime
        from src.data.schemas import ActionType, Platform

        restored: list[MemoryItem] = []
        for item in items:
            raw = item["message"]
            restored.append(MemoryItem(
                message=Message(
                    msg_id=raw["msg_id"],
                    thread_id=raw["thread_id"],
                    user_id=raw["user_id"],
                    platform=Platform(raw["platform"]),
                    timestamp=datetime.fromisoformat(raw["timestamp"]),
                    text=raw.get("text", ""),
                    action_type=ActionType(raw["action_type"]),
                    parent_msg_id=raw.get("parent_msg_id"),
                    metadata=dict(raw.get("metadata") or {}),
                ),
                importance=float(item.get("importance", 1.0)),
                round=int(item.get("round", 0)),
                kind=item.get("kind", "event"),
            ))
        self._items = restored

    def __len__(self) -> int:
        return len(self._items)
