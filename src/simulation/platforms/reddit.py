"""Reddit r/changemyview platform — threaded reply + delta mechanism."""

from __future__ import annotations

from datetime import datetime

from src.data.schemas import ActionType, Message, Thread
from src.simulation.platforms.base import PlatformTopology


class RedditTopology(PlatformTopology):
    """Threaded reply + delta: reply / award delta / counter-argue / block."""

    @property
    def valid_actions(self) -> list[ActionType]:
        return [ActionType.REPLY, ActionType.AWARD_DELTA, ActionType.COUNTER_ARGUE, ActionType.BLOCK]

    def get_valid_actions(self, thread: Thread, agent_id: str) -> list[ActionType]:
        actions = [ActionType.REPLY]  # always available

        # Build message lookup once so the has_counter_to_me check below
        # is O(n) total instead of O(n²) from the prior nested
        # any(any(next(...))) scan (code review M4). On a 1500-message
        # thread this cuts the per-call cost from seconds to milliseconds.
        msg_by_id = {m.msg_id: m for m in thread.messages}

        # Can counter-argue if there are messages from others
        has_others = any(m.user_id != agent_id for m in thread.messages)
        if has_others:
            actions.append(ActionType.COUNTER_ARGUE)

        # Can award delta if someone argued against your position.
        # AWARD_DELTA targets a COUNTER_ARGUE whose own parent was
        # authored by this agent (someone rebutted this agent's claim).
        has_counter_to_me = any(
            m.action_type == ActionType.COUNTER_ARGUE
            and m.parent_msg_id is not None
            and msg_by_id.get(m.parent_msg_id) is not None
            and msg_by_id[m.parent_msg_id].user_id == agent_id
            for m in thread.messages
        )
        if has_counter_to_me:
            actions.append(ActionType.AWARD_DELTA)

        # Can block after conflict
        has_conflict = any(
            m.action_type == ActionType.COUNTER_ARGUE
            and m.user_id != agent_id
            for m in thread.messages
        )
        if has_conflict:
            actions.append(ActionType.BLOCK)

        return actions

    def apply_action(self, action: ActionType, text: str, agent_id: str, thread: Thread) -> Message:
        # Delegate parent assignment to select_reply_target so direct
        # callers get the same threaded-reply topology as the take_turn
        # path (outline §6.3), instead of the legacy flat-list behaviour.
        parent_id = self.select_reply_target(action, agent_id, thread)

        return Message(
            msg_id=f"{agent_id}_{action.value}_{datetime.now().isoformat()}",
            thread_id=thread.thread_id,
            user_id=agent_id,
            platform=thread.platform,
            timestamp=datetime.now(),
            text=text,
            action_type=action,
            parent_msg_id=parent_id,
        )

    def select_reply_target(
        self,
        action: ActionType,
        agent_id: str,
        thread: Thread,
        hint_target_msg_id: str | None = None,
    ) -> str | None:
        """Reddit CMV threaded-reply topology (outline §6.3).

        Reddit's r/changemyview is a *tree*, not a list — replies attach
        to specific prior comments. The platform-faithful target rules:

        - ``AWARD_DELTA``: target a ``COUNTER_ARGUE`` message whose own
          parent was authored by this agent (i.e. someone successfully
          argued against this agent's position). Mirrors
          ``get_valid_actions``'s gating heuristic.
        - ``COUNTER_ARGUE``: target the most recent non-BLOCK message by
          another user (the claim being countered).
        - ``REPLY``: target the most recent message by another user
          (standard threaded reply).
        - ``BLOCK``: target the most recent message (the one being
          blocked); top-level ``None`` if the thread is empty.

        Falls back to ``super()`` when no candidate matches, preserving
        the legacy "last message" behaviour rather than emitting no edge.
        """
        msg_by_id = {m.msg_id: m for m in thread.messages}

        if action == ActionType.AWARD_DELTA:
            # Look for a COUNTER_ARGUE by another user whose parent is mine
            for m in reversed(thread.messages):
                if m.action_type != ActionType.COUNTER_ARGUE:
                    continue
                if m.user_id == agent_id:
                    continue
                parent = msg_by_id.get(m.parent_msg_id or "")
                if parent is not None and parent.user_id == agent_id:
                    return m.msg_id
            # No eligible counter → still need a parent for the delta
            return super().select_reply_target(
                action, agent_id, thread, hint_target_msg_id
            )

        if action == ActionType.COUNTER_ARGUE:
            # Most recent non-BLOCK message by another user
            for m in reversed(thread.messages):
                if m.user_id == agent_id:
                    continue
                if m.action_type == ActionType.BLOCK:
                    continue
                return m.msg_id
            return super().select_reply_target(
                action, agent_id, thread, hint_target_msg_id
            )

        # REPLY / BLOCK — standard threaded reply
        return super().select_reply_target(
            action, agent_id, thread, hint_target_msg_id
        )
