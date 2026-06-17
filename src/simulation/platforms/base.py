"""Abstract platform topology — action space + interaction graph."""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from src.data.schemas import ActionType, Message, Thread


class PlatformTopology(ABC):
    """Defines platform-specific interaction rules and action spaces."""

    @property
    @abstractmethod
    def valid_actions(self) -> list[ActionType]:
        """All valid action types for this platform."""

    @abstractmethod
    def get_valid_actions(self, thread: Thread, agent_id: str) -> list[ActionType]:
        """Get available actions for an agent in a thread context."""

    @abstractmethod
    def apply_action(self, action: ActionType, text: str, agent_id: str, thread: Thread) -> Message:
        """Apply an action to a thread, returning the resulting message."""

    # ------------------------------------------------------------------
    # Reply-target selection (outline §6.3 platform topology fidelity)
    # ------------------------------------------------------------------

    def select_reply_target(
        self,
        action: ActionType,
        agent_id: str,
        thread: Thread,
        hint_target_msg_id: str | None = None,
    ) -> str | None:
        """Pick the ``parent_msg_id`` for the message produced by ``action``.

        Outline §6.3 promises platform-faithful interaction topology, but
        the prior ``take_turn`` implementation hard-coded
        ``parent_msg_id = thread.messages[-1].msg_id`` for every action on
        every platform — flattening Reddit's reply tree, Wikipedia's edit
        chain, and GitHub's issue lifecycle into a single chronological
        list. The macro/meso interaction-graph metrics depend on these
        edges, so the flattening silently degenerates the graphs the
        paper reports on.

        Subclasses override this to encode platform-specific rules:
        Reddit COUNTER_ARGUE targets a counter-able message by another
        user; Wikipedia REVERT targets an EDIT by another user; GitHub
        lifecycle events (LABEL/CLOSE/REOPEN/ASSIGN) target the issue
        root (``None``) rather than a comment.

        Args:
            action: The action being applied this turn.
            agent_id: The acting agent's id (to exclude self-targets).
            thread: The current thread.
            hint_target_msg_id: Optional planner-emitted target hint.

        Returns:
            ``parent_msg_id`` for the new message, or ``None`` if the
            action is a top-level / lifecycle event (per-platform rules).
        """
        # Default: honour a planner-emitted hint when it points at an
        # existing message in this thread; else fall back to the most
        # recent message from another user; else the last message; else
        # None (top-level).
        if hint_target_msg_id:
            for m in thread.messages:
                if m.msg_id == hint_target_msg_id:
                    return hint_target_msg_id
        for m in reversed(thread.messages):
            if m.user_id != agent_id:
                return m.msg_id
        if thread.messages:
            return thread.messages[-1].msg_id
        return None

    def _last_message_by(
        self,
        thread: Thread,
        agent_id: str,
        exclude_self: bool = True,
        action_filter: set[ActionType] | None = None,
    ) -> Message | None:
        """Helper: most recent message matching the criteria, or None."""
        for m in reversed(thread.messages):
            if exclude_self and m.user_id == agent_id:
                continue
            if action_filter is not None and m.action_type not in action_filter:
                continue
            return m
        return None

    def get_interaction_graph(self, threads: list[Thread]) -> nx.Graph:
        """Build interaction graph from threads.

        Nodes = users, edges = interaction (reply chain, conflict, etc).
        """
        graph = nx.Graph()
        for thread in threads:
            # Add nodes
            for user_id in thread.participants:
                if not graph.has_node(user_id):
                    graph.add_node(user_id)

            # Add edges based on message interactions
            msg_by_id = {m.msg_id: m for m in thread.messages}
            for msg in thread.messages:
                if msg.parent_msg_id and msg.parent_msg_id in msg_by_id:
                    parent = msg_by_id[msg.parent_msg_id]
                    if parent.user_id != msg.user_id:
                        if graph.has_edge(parent.user_id, msg.user_id):
                            graph[parent.user_id][msg.user_id]["weight"] += 1
                        else:
                            graph.add_edge(parent.user_id, msg.user_id, weight=1)

        return graph
