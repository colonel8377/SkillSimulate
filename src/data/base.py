"""Abstract dataset loader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas import Message, Platform, Thread


class DatasetLoader(ABC):
    """Base class for platform-specific data loaders."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load(self) -> list[Thread]:
        """Load all threads from raw data."""

    @abstractmethod
    def get_platform(self) -> Platform:
        """Return the platform identifier."""

    def get_user_history(self, threads: list[Thread], user_id: str) -> list[Message]:
        """Extract all messages from a specific user across threads."""
        history = []
        for thread in threads:
            for msg in thread.messages:
                if msg.user_id == user_id:
                    history.append(msg)
        return history

    def get_all_users(self, threads: list[Thread]) -> set[str]:
        """Return set of all user IDs across threads."""
        users = set()
        for thread in threads:
            users.update(thread.user_ids)
        return users
