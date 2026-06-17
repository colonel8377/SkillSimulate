"""Behavioral signal extraction from user conversation history.

Extracts per-user behavioral features as described in CADP Step 1:
- Reply depth (avg nesting level)
- Edit frequency
- Stance shift rate
- Conflict engagement ratio
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.schemas import ActionType, Message, Thread


@dataclass
class UserFeatures:
    """Behavioral feature vector for a single user."""
    user_id: str
    reply_depth: float          # avg nesting level of messages
    edit_frequency: float       # edits per message
    stance_shift_rate: float    # how often user changes stance
    conflict_engagement_ratio: float  # fraction of messages in contested threads
    message_count: int
    thread_count: int

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.reply_depth,
            self.edit_frequency,
            self.stance_shift_rate,
            self.conflict_engagement_ratio,
        ], dtype=np.float64)


class FeatureExtractor:
    """Extracts behavioral signals from user message history."""

    def __init__(self, contested_threshold: int = 3):
        """Args:
            contested_threshold: min number of unique participants
                for a thread to be considered "contested".
        """
        self.contested_threshold = contested_threshold

    def extract_all(
        self,
        threads: list[Thread],
    ) -> dict[str, UserFeatures]:
        """Extract features for all users across all threads.

        Returns:
            Mapping user_id → UserFeatures.
        """
        # Build per-thread structures
        contested_threads = self._identify_contested_threads(threads)
        thread_depths = {t.thread_id: self._compute_depths(t) for t in threads}

        # Group messages by user
        user_messages: dict[str, list[tuple[Message, Thread]]] = defaultdict(list)
        for thread in threads:
            for msg in thread.messages:
                user_messages[msg.user_id].append((msg, thread))

        features = {}
        for user_id, msg_thread_pairs in user_messages.items():
            features[user_id] = self._extract_single(user_id, msg_thread_pairs, thread_depths, contested_threads)

        return features

    def _extract_single(
        self,
        user_id: str,
        msg_thread_pairs: list[tuple[Message, Thread]],
        thread_depths: dict[str, dict[str, int]],
        contested_threads: set[str],
    ) -> UserFeatures:
        messages = [m for m, _ in msg_thread_pairs]
        threads_set = {t.thread_id for _, t in msg_thread_pairs}

        # Reply depth
        depths = []
        for msg, thread in msg_thread_pairs:
            depth = thread_depths.get(thread.thread_id, {}).get(msg.msg_id, 0)
            depths.append(depth)
        reply_depth = float(np.mean(depths)) if depths else 0.0

        # Edit frequency
        edit_count = sum(1 for m in messages if m.action_type == ActionType.EDIT)
        total_messages = len(messages)
        edit_frequency = edit_count / total_messages if total_messages > 0 else 0.0

        # Stance shift rate (M1 fix): restrict to stance-bearing action
        # flips (AGREE ↔ DISAGREE / COUNTER_ARGUE) and aggregate per
        # thread to avoid conflating stance shifts with cross-thread
        # context switches. See ``_compute_stance_shift_rate`` for the
        # policy and limitations.
        stance_shift_rate = self._compute_stance_shift_rate(msg_thread_pairs)

        # Conflict engagement ratio
        contested_msgs = sum(
            1 for m, t in msg_thread_pairs if t.thread_id in contested_threads
        )
        conflict_ratio = contested_msgs / total_messages if total_messages > 0 else 0.0

        return UserFeatures(
            user_id=user_id,
            reply_depth=reply_depth,
            edit_frequency=edit_frequency,
            stance_shift_rate=stance_shift_rate,
            conflict_engagement_ratio=conflict_ratio,
            message_count=total_messages,
            thread_count=len(threads_set),
        )

    def _identify_contested_threads(self, threads: list[Thread]) -> set[str]:
        """Threads with enough unique participants to be contested."""
        return {
            t.thread_id for t in threads
            if len(t.participants) >= self.contested_threshold
        }

    def _compute_depths(self, thread: Thread) -> dict[str, int]:
        """Compute nesting depth for each message in a thread."""
        depths: dict[str, int] = {}
        msg_by_id = {m.msg_id: m for m in thread.messages}

        def get_depth(msg_id: str, cache: dict[str, int] | None = None) -> int:
            if cache is None:
                cache = depths
            if msg_id in cache:
                return cache[msg_id]
            msg = msg_by_id.get(msg_id)
            if msg is None or msg.parent_msg_id is None:
                cache[msg_id] = 0
                return 0
            d = get_depth(msg.parent_msg_id, cache) + 1
            cache[msg_id] = d
            return d

        for msg in thread.messages:
            get_depth(msg.msg_id)
        return depths

    def _compute_stance_shift_rate(
        self,
        msg_thread_pairs: list[tuple[Message, Thread]],
    ) -> float:
        """Estimate how often a user flips stance, scoped per-thread.

        Methodology caveat (M1 fix): the previous implementation sorted
        ALL of a user's messages globally by timestamp and counted any
        action-type change as a stance shift. This conflated two
        unrelated signals:

        1. Stance flips within a thread (DISAGREE → AGREE) — the desired
           signal.
        2. Cross-thread context switches (e.g. a Reddit user posting
           ``comment`` in r/A then ``comment`` in r/B) — noise. Sorting
           across threads mixed messages from different conversations,
           inflating the rate.
        3. Benign action variation within a thread (COMMENT → EDIT) —
           noise. The docstring claimed only DISAGREE ↔ AGREE flips were
           counted, but the implementation counted *any* action-type
           change.

        The fix restricts the signal to stance-bearing action types
        (AGREE / DISAGREE / COUNTER_ARGUE) and computes flips within
        each thread independently, then averages across threads where
        the user posted ≥2 stance-bearing messages. This is still a
        heuristic (true stance requires stance-detection modelling) and
        should be disclosed in §7 Limitations, but it is a much cleaner
        proxy than the prior global action-type-diff.

        Args:
            msg_thread_pairs: (message, thread) pairs for one user.

        Returns:
            Mean per-thread stance-flip rate in [0, 1]. 0.0 if the user
            has no thread with ≥2 stance-bearing messages.
        """
        stance_types = {
            ActionType.AGREE,
            ActionType.DISAGREE,
            ActionType.COUNTER_ARGUE,
        }

        # Group stance-bearing messages by thread (chronological order is
        # preserved within each thread because the caller iterates the
        # source dataset in message order).
        per_thread: dict[str, list[Message]] = {}
        for msg, thread in msg_thread_pairs:
            if msg.action_type in stance_types:
                per_thread.setdefault(thread.thread_id, []).append(msg)

        if not per_thread:
            return 0.0

        rates: list[float] = []
        for thread_msgs in per_thread.values():
            if len(thread_msgs) < 2:
                continue
            # Defensive chronological sort on timestamp; stable when
            # timestamps tie (preserves dataset insertion order).
            ordered = sorted(thread_msgs, key=lambda m: m.timestamp)
            actions = [m.action_type for m in ordered]
            shifts = sum(
                1 for i in range(1, len(actions))
                if actions[i] != actions[i - 1]
            )
            rates.append(shifts / (len(actions) - 1))

        if not rates:
            return 0.0
        return float(np.mean(rates))
