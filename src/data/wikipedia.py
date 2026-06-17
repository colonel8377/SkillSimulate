"""Wikipedia Talk Pages data loader.

Supports three common Wikipedia talk page corpus formats:
1. **Flat JSONL** (custom dump): thread_id, comment_id, user_id, text, timestamp, action_type
2. **WikiConv** (Khanna et al. 2012/WikiConv dataset): conversation_id, id, speaker, text, timestamp, reply_to
3. **ConvoKit export**: nested _source fields, conversation_id, id, speaker, text, meta.*

Expected input: JSONL files in the data directory.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.data.base import DatasetLoader
from src.data.pii import scrub_threads
from src.data.schemas import ActionType, Message, Platform, Thread


# Patterns that indicate a revert action in edit comments
_REVERT_PATTERNS = (
    "revert", "reverted", "undid", "rollback", "rv ", "rvv",
    "undo", "restoring previous",
)
_REPORT_PATTERNS = (
    "report", "reported", "ani", "vandalism", "block request",
)


class WikipediaLoader(DatasetLoader):

    def get_platform(self) -> Platform:
        return Platform.WIKIPEDIA

    def load(self) -> list[Thread]:
        data_dir = Path(self.data_path)
        threads_map: dict[str, Thread] = {}

        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    msg = self._parse_record(record)
                    if msg is None:
                        continue
                    if msg.thread_id not in threads_map:
                        topic = (
                            record.get("page_title")
                            or record.get("topic")
                            or record.get("meta", {}).get("page_title", "")
                            if isinstance(record.get("meta"), dict)
                            else record.get("page_title", "")
                        )
                        threads_map[msg.thread_id] = Thread(
                            thread_id=msg.thread_id,
                            platform=self.get_platform(),
                            topic=str(topic) if topic else msg.thread_id,
                        )
                    threads_map[msg.thread_id].add_message(msg)

        return scrub_threads(list(threads_map.values()))

    def _parse_record(self, record: dict) -> Message | None:
        """Parse a raw JSON record into a Message.

        Handles three schema variants:
        - Flat: thread_id, comment_id, user_id, text, timestamp
        - WikiConv: conversation_id, id, speaker, text, timestamp, reply_to
        - ConvoKit: _source.conversation_id, _source.id, _source.speaker, etc.
        """
        # Unwrap ConvoKit-style nested _source
        src = record.get("_source", record)

        # Thread ID
        thread_id = (
            src.get("thread_id")
            or src.get("conversation_id")
            or src.get("page_title")
        )
        if not thread_id:
            return None

        # Message ID
        msg_id = (
            src.get("comment_id")
            or src.get("id")
            or src.get("rev_id")
            or src.get("message_id")
        )
        if not msg_id:
            return None

        # User ID
        user_id = (
            src.get("user_id")
            or src.get("author")
            or src.get("speaker")
            or src.get("user")
        )
        if not user_id:
            return None

        # Text content
        text = (
            src.get("text")
            or src.get("comment")
            or src.get("content")
            or src.get("body", "")
        )

        # Parent message ID
        parent_id = (
            src.get("parent_comment_id")
            or src.get("parent_id")
            or src.get("reply_to")
            or src.get("meta", {}).get("reply_to")
            if isinstance(src.get("meta"), dict)
            else src.get("reply_to")
        )

        # Timestamp
        raw_ts = (
            src.get("timestamp")
            or src.get("created_at")
            or src.get("utc_timestamp")
            or src.get("meta", {}).get("timestamp", "")
            if isinstance(src.get("meta"), dict)
            else src.get("timestamp", "")
        )
        try:
            if isinstance(raw_ts, (int, float)):
                ts = datetime.fromtimestamp(raw_ts)
            else:
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = datetime.now()

        # Action type: explicit, metadata, or inferred from text
        action = self._infer_action(src, text)

        # Collect metadata (exclude consumed fields)
        consumed_keys = {
            "thread_id", "conversation_id", "page_title", "comment_id", "id",
            "rev_id", "message_id", "user_id", "author", "speaker", "user",
            "text", "comment", "content", "body", "parent_comment_id",
            "parent_id", "reply_to", "timestamp", "created_at", "utc_timestamp",
            "action_type", "_source",
        }
        metadata = {k: v for k, v in record.items() if k not in consumed_keys}
        # Preserve meta dict if present
        meta = src.get("meta")
        if isinstance(meta, dict):
            metadata.setdefault("meta", meta)

        return Message(
            msg_id=str(msg_id),
            thread_id=str(thread_id),
            user_id=str(user_id),
            platform=self.get_platform(),
            timestamp=ts,
            text=text,
            action_type=action,
            parent_msg_id=str(parent_id) if parent_id else None,
            metadata=metadata,
        )

    def _infer_action(self, src: dict, text: str) -> ActionType:
        """Infer action type from explicit field, metadata, or text content."""
        # 1. Explicit action_type field
        explicit = src.get("action_type", "")
        action_map = {
            "edit": ActionType.EDIT,
            "revert": ActionType.REVERT,
 "discuss": ActionType.DISCUSS,
            "report": ActionType.REPORT,
            "comment": ActionType.DISCUSS,
            "post": ActionType.DISCUSS,
        }
        action = action_map.get(str(explicit).lower())
        if action:
            return action

        # 2. ConvoKit-style meta fields
        meta = src.get("meta", {})
        if isinstance(meta, dict):
            if meta.get("is_revert") or meta.get("action") == "revert":
                return ActionType.REVERT
            if meta.get("is_report") or meta.get("action") == "report":
                return ActionType.REPORT
            if meta.get("action") == "edit":
                return ActionType.EDIT

        # 3. Infer from comment text patterns (Wikipedia edit summary conventions)
        text_lower = text.lower().strip()
        if any(pat in text_lower for pat in _REVERT_PATTERNS):
            return ActionType.REVERT
        if any(pat in text_lower for pat in _REPORT_PATTERNS):
            return ActionType.REPORT
        if text_lower.startswith("edit") or src.get("rev_id"):
            return ActionType.EDIT

        return ActionType.DISCUSS
