"""Reddit r/changemyview data loader.

Expected input: JSON/JSONL dump of CMV submissions + comments.
Submissions should contain: submission_id, author, title, body, created_utc.
Comments should contain: submission_id, comment_id, author, body,
parent_comment_id, created_utc, delta_awarded (bool).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.data.base import DatasetLoader
from src.data.pii import scrub_threads
from src.data.schemas import ActionType, Message, Platform, Thread


class RedditLoader(DatasetLoader):

    def get_platform(self) -> Platform:
        return Platform.REDDIT

    def load(self) -> list[Thread]:
        data_dir = Path(self.data_path)
        threads_map: dict[str, Thread] = {}

        # Load submissions
        for submissions_file in sorted(data_dir.glob("*submissions*.jsonl")):
            with open(submissions_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    sub_id = record.get("submission_id") or record.get("id")
                    if not sub_id:
                        continue
                    author = record.get("author", "[deleted]")
                    body = record.get("selftext") or record.get("body", "")
                    title = record.get("title", "")

                    raw_ts = record.get("created_utc", 0)
                    ts = datetime.fromtimestamp(raw_ts) if isinstance(raw_ts, (int, float)) else datetime.now()

                    msg = Message(
                        msg_id=f"{sub_id}_0",
                        thread_id=str(sub_id),
                        user_id=str(author),
                        platform=self.get_platform(),
                        timestamp=ts,
                        text=f"{title}\n\n{body}".strip(),
                        action_type=ActionType.POST,
                    )
                    if str(sub_id) not in threads_map:
                        threads_map[str(sub_id)] = Thread(
                            thread_id=str(sub_id),
                            platform=self.get_platform(),
                            topic=title,
                        )
                    threads_map[str(sub_id)].add_message(msg)

        # Load comments
        for comments_file in sorted(data_dir.glob("*comments*.jsonl")):
            with open(comments_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    msg = self._parse_comment(record)
                    if msg is None:
                        continue
                    if msg.thread_id not in threads_map:
                        threads_map[msg.thread_id] = Thread(
                            thread_id=msg.thread_id,
                            platform=self.get_platform(),
                            topic=record.get("submission_title", ""),
                        )
                    threads_map[msg.thread_id].add_message(msg)

        return scrub_threads(list(threads_map.values()))

    def _parse_comment(self, record: dict) -> Message | None:
        sub_id = record.get("submission_id") or record.get("link_id")
        comment_id = record.get("comment_id") or record.get("id")
        author = record.get("author") or record.get("user_id")
        body = record.get("body") or record.get("text", "")
        if not all([sub_id, comment_id, author]):
            return None

        # Normalize IDs
        sub_id = str(sub_id).replace("t3_", "")
        comment_id = str(comment_id).replace("t1_", "")
        parent_id = record.get("parent_comment_id") or record.get("parent_id")
        if parent_id:
            parent_id = str(parent_id).replace("t1_", "").replace("t3_", "")

        raw_ts = record.get("created_utc", 0)
        ts = datetime.fromtimestamp(raw_ts) if isinstance(raw_ts, (int, float)) else datetime.now()

        # Determine action type
        if record.get("delta_awarded") or record.get("is_delta"):
            action = ActionType.AWARD_DELTA
        elif record.get("is_blocking") or record.get("blocked"):
            action = ActionType.BLOCK
        elif self._is_counter_argument(body, record):
            action = ActionType.COUNTER_ARGUE
        else:
            action = ActionType.REPLY

        return Message(
            msg_id=comment_id,
            thread_id=sub_id,
            user_id=str(author),
            platform=self.get_platform(),
            timestamp=ts,
            text=body,
            action_type=action,
            parent_msg_id=parent_id,
            metadata={"score": record.get("score", 0)},
        )

    @staticmethod
    def _is_counter_argument(body: str, record: dict) -> bool:
        """Heuristic: detect counter-argument comments in r/changemyview.

        Counter-arguments disagree without awarding a delta, typically containing
        disagreement markers or replying directly to the OP's thesis.
        """
        if not body or len(body) < 20:
            return False

        body_lower = body.lower().strip()
        disagreement_markers = [
            "i disagree", "i don't think", "i'd argue", "actually,",
            "but ", "however ", "that's not", "you're wrong",
            "not necessarily", "on the contrary", "the problem with",
        ]
        marker_count = sum(1 for m in disagreement_markers if m in body_lower)

        # Top-level reply to submission (parent is the post itself)
        is_top_level = (
            record.get("parent_comment_id") is None
            or record.get("parent_id", "").startswith("t3_")
        )

        return marker_count >= 2 or (is_top_level and marker_count >= 1)
