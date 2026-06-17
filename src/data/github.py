"""GitHub Issues data loader.

Expected input: JSON/JSONL dump of GitHub issues and comments.
Records should contain: repo, issue_number, event_type (issue/comment/label/close/etc),
author, body, created_at, labels.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.data.base import DatasetLoader
from src.data.pii import scrub_threads
from src.data.schemas import ActionType, Message, Platform, Thread


class GitHubLoader(DatasetLoader):

    def get_platform(self) -> Platform:
        return Platform.GITHUB

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
                        repo = record.get("repo", "unknown")
                        threads_map[msg.thread_id] = Thread(
                            thread_id=msg.thread_id,
                            platform=self.get_platform(),
                            topic=record.get("title", f"{repo}#{record.get('issue_number', '')}"),
                        )
                    threads_map[msg.thread_id].add_message(msg)

        return scrub_threads(list(threads_map.values()))

    def _parse_record(self, record: dict) -> Message | None:
        repo = record.get("repo", "")
        issue_number = record.get("issue_number")
        event_type = record.get("event_type", "comment").lower()
        author = record.get("author") or record.get("user")
        body = record.get("body", "")
        msg_id = record.get("event_id") or record.get("id")

        if not all([repo, issue_number, author]):
            return None

        thread_id = f"{repo}#{issue_number}"
        if not msg_id:
            msg_id = f"{thread_id}_{event_type}_{record.get('created_at', '')}"

        raw_ts = record.get("created_at", "")
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            ts = datetime.now()

        action_map = {
            "issue_opened": ActionType.POST,
            "issue_closed": ActionType.CLOSE,
            "issue_reopened": ActionType.REOPEN,
            "comment": ActionType.COMMENT,
            "labeled": ActionType.LABEL,
            "assigned": ActionType.ASSIGN,
        }
        action = action_map.get(event_type, ActionType.COMMENT)

        return Message(
            msg_id=str(msg_id),
            thread_id=thread_id,
            user_id=str(author),
            platform=self.get_platform(),
            timestamp=ts,
            text=body,
            action_type=action,
            parent_msg_id=record.get("parent_comment_id"),
            metadata={
                "repo": repo,
                "issue_number": issue_number,
                "labels": record.get("labels", []),
            },
        )
