"""Wikipedia platform — edit tree topology."""

from __future__ import annotations

from datetime import datetime

from src.data.schemas import ActionType, Message, Thread
from src.simulation.platforms.base import PlatformTopology


class WikipediaTopology(PlatformTopology):
    """Edit tree structure: edit → revert → re-edit.

    Action space reflects real Wikipedia talk-page behaviour:
      - DISCUSS: general talk-page comment (always available)
      - AGREE: express agreement with a prior position
      - DISAGREE: challenge or oppose a prior claim
      - EDIT: modify article content
      - REVERT: undo another editor's change
      - REPORT: flag content for admin attention
    """

    @property
    def valid_actions(self) -> list[ActionType]:
        return [
            ActionType.DISCUSS, ActionType.AGREE, ActionType.DISAGREE,
            ActionType.EDIT, ActionType.REVERT, ActionType.REPORT,
        ]

    def get_valid_actions(self, thread: Thread, agent_id: str) -> list[ActionType]:
        actions = [ActionType.DISCUSS]  # always available

        # AGREE / DISAGREE available when there are other messages to respond to
        has_others = any(m.user_id != agent_id for m in thread.messages)
        if has_others:
            actions.append(ActionType.AGREE)
            actions.append(ActionType.DISAGREE)

        # Can edit if there's content to edit
        if thread.messages:
            actions.append(ActionType.EDIT)

        # Can revert if there are edits by others
        has_edits_by_others = any(
            m.action_type == ActionType.EDIT and m.user_id != agent_id
            for m in thread.messages
        )
        if has_edits_by_others:
            actions.append(ActionType.REVERT)

        # Can report if there are conflicts
        has_conflicts = len(thread.participants) >= 2
        if has_conflicts:
            actions.append(ActionType.REPORT)

        return actions

    def apply_action(self, action: ActionType, text: str, agent_id: str, thread: Thread) -> Message:
        # Delegate parent assignment to select_reply_target so direct
        # callers get the same edit-tree topology as the take_turn path
        # (outline §6.3), instead of the legacy flat-list behaviour.
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
        """Wikipedia edit-tree topology (outline §6.3).

        - ``REVERT``: targets the most recent ``EDIT`` by another user.
        - ``DISCUSS`` / ``AGREE`` / ``DISAGREE`` / ``REPORT``: target the
          most recent message by another user (talk-page reply chain).
        - ``EDIT``: an edit modifies the article, not a message — return
          ``None`` so the topology reflects the article-vs-talk split.
        """
        if action == ActionType.EDIT:
            return None  # edit modifies the article, not a message
        if action == ActionType.REVERT:
            target = self._last_message_by(
                thread, agent_id, action_filter={ActionType.EDIT}
            )
            if target is not None:
                return target.msg_id
            return None  # nothing to revert
        # DISCUSS / AGREE / DISAGREE / REPORT — talk-page reply chain
        return super().select_reply_target(
            action, agent_id, thread, hint_target_msg_id
        )
