"""GitHub Issues platform — issue lifecycle."""

from __future__ import annotations

from datetime import datetime

from src.data.schemas import ActionType, Message, Thread
from src.simulation.platforms.base import PlatformTopology


class GitHubTopology(PlatformTopology):
    """Issue lifecycle: comment / label / close / reopen / assign."""

    @property
    def valid_actions(self) -> list[ActionType]:
        return [ActionType.COMMENT, ActionType.LABEL, ActionType.CLOSE, ActionType.REOPEN, ActionType.ASSIGN]

    def get_valid_actions(self, thread: Thread, agent_id: str) -> list[ActionType]:
        actions = [ActionType.COMMENT]  # always available

        # Determine issue state
        is_closed = bool(thread.messages and any(
            m.action_type == ActionType.CLOSE
            and not any(
                later.action_type == ActionType.REOPEN
                and later.timestamp > m.timestamp
                for later in thread.messages
            )
            for m in thread.messages
        ))

        if not is_closed:
            actions.extend([ActionType.LABEL, ActionType.ASSIGN, ActionType.CLOSE])
        else:
            actions.append(ActionType.REOPEN)

        return list(set(actions))

    def apply_action(self, action: ActionType, text: str, agent_id: str, thread: Thread) -> Message:
        # Delegate parent assignment to select_reply_target so direct
        # callers (replay / tests / loaders) get the same platform-faithful
        # topology as the agent take_turn path, instead of the legacy
        # "reply to last message" flattening (outline §6.3).
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

    # Lifecycle events target the issue root, not a comment.
    # Outline §6.3: GitHub's topology is an issue → comment tree, not a
    # flat list. LABEL/CLOSE/REOPEN/ASSIGN mutate the issue itself; only
    # COMMENT forms a reply edge in the interaction graph.
    _LIFECYCLE_ACTIONS: frozenset[ActionType] = frozenset({
        ActionType.LABEL, ActionType.CLOSE, ActionType.REOPEN, ActionType.ASSIGN,
    })

    def select_reply_target(
        self,
        action: ActionType,
        agent_id: str,
        thread: Thread,
        hint_target_msg_id: str | None = None,
    ) -> str | None:
        if action in self._LIFECYCLE_ACTIONS:
            return None  # issue-level event, no comment-parent edge
        # COMMENT → most recent COMMENT by another user. If none exists,
        # attach to issue root (None) rather than falling back to a
        # lifecycle event (CLOSE/LABEL/...), which would create a spurious
        # interaction-graph edge (G11).
        target = self._last_message_by(
            thread, agent_id, action_filter={ActionType.COMMENT}
        )
        if target is not None:
            return target.msg_id
        return None
