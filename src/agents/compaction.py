"""Rolling-summary memory compaction (Issue 1, R4 path).

When ``ExperimentConfig.memory_strategy == "rolling_summary"`` the agent
periodically summarizes its oldest raw memory items into a single
:class:`src.agents.memory.MemoryItem` with ``kind="summary"``. This keeps
long-horizon signal available to the planner without growing the prompt
unboundedly — required for the 50-turn R4 persona-collapse stress test
where the per-turn sliding window would otherwise forget every turn ≤ N-5.

Compaction contract:

  * Triggered every ``compaction_interval`` turns by :meth:`BaseAgent.take_turn`.
  * Picks the ``N`` oldest ``kind="event"`` items still in memory (N defaults
    to ``compaction_interval`` so one summary covers one compaction window).
  * Produces a single LLM-generated summary text and stores it as a new
    MemoryItem with ``kind="summary"``, importance=2.0 (boosted in
    retrieval), round=current_round.
  * Pops the N summarized raw items from memory via
    :meth:`AgentMemory.pop_oldest_events`.
  * Keeps the most recent ``keep_recent`` event items raw (sliding window).

Failure handling
----------------
The summary call is itself an LLM completion, so it can fail the same way
any other call fails. On failure we **do not pop the raw items** — they
stay in memory and we retry compaction next cycle. The agent continues
operating; the only cost is that the prompt stays at its uncompacted size
for one more interval. This is the explicit contract requested by the
user: "Never silently destroy raw history."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.data.schemas import Message

if TYPE_CHECKING:
    from src.agents.memory import AgentMemory
    from src.llm.client import LLMClient


SUMMARY_PROMPT = """You are consolidating the memory of a simulated online-community participant.

Below are this participant's oldest observed messages, in chronological order. Summarize them into a compact paragraph that preserves:

1. The participant's stance / position on the topic.
2. Any commitments or arguments they made.
3. The tone and style of their engagement.

Observed messages:
{messages}

Respond with ONE concise paragraph (3-5 sentences). Do not list individual messages — synthesize. Do not mention that you are summarizing."""


@dataclass
class CompactionResult:
    """Outcome of a single compaction pass."""
    summarized_count: int
    summary_text: str
    success: bool
    error: str = ""


class RollingSummaryCompactor:
    """Periodically compacts the oldest raw memory items into a summary item.

    The compactor is stateless — it reads the current state of the
    AgentMemory and modifies it in place. The decision of *when* to
    compact is delegated to the caller (:meth:`BaseAgent.maybe_compact`)
    so the trigger policy (turn count, token-pressure, etc.) can evolve
    independently.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        model_name: str,
        compaction_interval: int = 5,
        keep_recent: int = 10,
        max_messages_per_summary: int | None = None,
    ):
        self.llm = llm_client
        self.model_name = model_name
        self.compaction_interval = compaction_interval
        self.keep_recent = keep_recent
        # How many raw items to bundle into each summary. Default =
        # compaction_interval so one summary covers one compaction window.
        self.max_messages_per_summary = max_messages_per_summary or compaction_interval

    def should_compact(self, current_turn: int) -> bool:
        """Return True when ``current_turn`` is a compaction boundary.

        Uses turn count, not message count, so the compaction cadence is
        deterministic across agents regardless of how talkative each one
        is.
        """
        if self.compaction_interval <= 0:
            return False
        return current_turn > 0 and current_turn % self.compaction_interval == 0

    async def compact(self, memory: "AgentMemory", current_round: int) -> CompactionResult:
        """Run one compaction pass on *memory*.

        Returns a :class:`CompactionResult` describing what happened.
        Never raises — failures are caught and reported via the result
        so the caller (BaseAgent.take_turn) can continue uninterrupted.
        """
        # Identify the N oldest raw events (excluding the protected
        # recent-window and any pre-existing summary items).
        event_items = [it for it in memory._items if it.kind == "event"]
        event_items.sort(key=lambda it: it.round)
        # Protect the keep_recent most-recent items from compaction.
        if len(event_items) <= self.keep_recent:
            return CompactionResult(
                summarized_count=0,
                summary_text="",
                success=True,
                error="not-enough-history",
            )
        n_to_summarize = min(
            self.max_messages_per_summary,
            len(event_items) - self.keep_recent,
        )
        if n_to_summarize <= 0:
            return CompactionResult(
                summarized_count=0,
                summary_text="",
                success=True,
                error="nothing-to-compact",
            )
        to_summarize = event_items[:n_to_summarize]

        # Build the summarization prompt.
        msgs_text = "\n".join(
            f"[Round {it.round}] [{it.message.action_type.value}]: {it.message.text}"
            for it in to_summarize
        )
        prompt = SUMMARY_PROMPT.format(messages=msgs_text)
        messages = [
            {"role": "system", "content": "You are summarizing a simulated agent's recent memory."},
            {"role": "user", "content": prompt},
        ]

        try:
            summary_text = await self.llm.chat_completion(
                messages, self.model_name, temperature=0.3,
            )
        except Exception as exc:
            # DO NOT pop raw items on failure — leave memory untouched
            # and let the next compaction cycle retry. The agent's prompt
            # stays at its uncompacted size for one more interval.
            logger.warning(
                f"RollingSummaryCompactor: summary LLM call failed ({exc!r}); "
                f"keeping {len(to_summarize)} raw items in memory and retrying next cycle."
            )
            return CompactionResult(
                summarized_count=0,
                summary_text="",
                success=False,
                error=str(exc),
            )

        summary_text = (summary_text or "").strip()
        if not summary_text:
            logger.warning(
                "RollingSummaryCompactor: summary LLM returned empty text; "
                "keeping raw items and retrying next cycle."
            )
            return CompactionResult(
                summarized_count=0,
                summary_text="",
                success=False,
                error="empty-summary",
            )

        # Success — pop the raw items we just summarized and replace
        # with a single summary MemoryItem.
        memory.pop_oldest_events(n_to_summarize)
        summary_msg = Message(
            msg_id=f"summary_r{current_round}_{id(memory):x}",
            thread_id="",  # summaries span multiple threads
            user_id="memory",
            platform=to_summarize[0].message.platform,
            timestamp=to_summarize[0].message.timestamp,
            text=summary_text,
            action_type=to_summarize[0].message.action_type,
            parent_msg_id=None,
            metadata={
                "round": current_round,
                "kind": "summary",
                "summarized_count": n_to_summarize,
            },
        )
        memory.add(summary_msg, round=current_round, importance=2.0, kind="summary")

        logger.info(
            f"RollingSummaryCompactor: compacted {n_to_summarize} raw items "
            f"into 1 summary at round {current_round} "
            f"(memory size now {len(memory)})."
        )
        return CompactionResult(
            summarized_count=n_to_summarize,
            summary_text=summary_text,
            success=True,
        )


__all__ = ["RollingSummaryCompactor", "CompactionResult", "SUMMARY_PROMPT"]
