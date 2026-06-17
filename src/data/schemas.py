"""Unified data models for all platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Platform(str, Enum):
    WIKIPEDIA = "wikipedia"
    REDDIT = "reddit"
    GITHUB = "github"


class ActionType(str, Enum):
    # Wikipedia
    EDIT = "edit"
    REVERT = "revert"
    DISCUSS = "discuss"
    REPORT = "report"
    # Reddit
    REPLY = "reply"
    AWARD_DELTA = "award_delta"
    COUNTER_ARGUE = "counter_argue"
    BLOCK = "block"
    # GitHub
    COMMENT = "comment"
    LABEL = "label"
    CLOSE = "close"
    REOPEN = "reopen"
    ASSIGN = "assign"
    # Generic
    POST = "post"
    AGREE = "agree"
    DISAGREE = "disagree"

    @classmethod
    def for_platform(cls, platform: Platform) -> list["ActionType"]:
        wiki = {cls.EDIT, cls.REVERT, cls.DISCUSS, cls.REPORT}
        reddit = {cls.REPLY, cls.AWARD_DELTA, cls.COUNTER_ARGUE, cls.BLOCK}
        github = {cls.COMMENT, cls.LABEL, cls.CLOSE, cls.REOPEN, cls.ASSIGN}
        mapping = {
            Platform.WIKIPEDIA: wiki,
            Platform.REDDIT: reddit,
            Platform.GITHUB: github,
        }
        return list(mapping[platform])


@dataclass
class User:
    user_id: str
    platform: Platform
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    msg_id: str
    thread_id: str
    user_id: str
    platform: Platform
    timestamp: datetime
    text: str
    action_type: ActionType = ActionType.POST
    parent_msg_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Thread:
    thread_id: str
    platform: Platform
    topic: str
    messages: list[Message] = field(default_factory=list)
    participants: set[str] = field(default_factory=set)

    @property
    def user_ids(self) -> set[str]:
        return {m.user_id for m in self.messages}

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self.participants.add(msg.user_id)
