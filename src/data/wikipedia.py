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

from loguru import logger

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

    def __init__(self, data_path: str, limit: int | None = None):
        super().__init__(data_path)
        # Max base utterances to ingest across the corpus (None = all). The full
        # 2001-2014 WikiConv is ~80M+ utterances — far too large to materialise.
        # A few-million sample is statistically ample for archetype clustering.
        self.limit = limit

    def get_platform(self) -> Platform:
        return Platform.WIKIPEDIA

    def load(self) -> list[Thread]:
        data_dir = Path(self.data_path)

        # Prefer ConvoKit WikiConv corpora (directories holding utterances.jsonl
        # + index.json, typically one per year under data_dir). These carry real
        # action history (modification/deletion/restoration) + toxicity, so they
        # are used in preference to any flat *.jsonl dump in the same tree.
        convokit_dirs = self._find_convokit_corpora(data_dir)
        if convokit_dirs:
            threads = self._load_convokit(convokit_dirs, data_dir)
        else:
            threads = self._load_flat_jsonl(data_dir)

        return scrub_threads(threads)

    # ------------------------------------------------------------------
    # ConvoKit WikiConv ingestion
    # ------------------------------------------------------------------
    def _find_convokit_corpora(self, root: Path) -> list[Path]:
        """Find ConvoKit corpus dirs (utterances.jsonl + index.json) under root.

        Excludes the CGA label corpus (``cga``/``*awry*``) — that is loaded
        separately as a conflict-label side-table, not as primary data.
        """
        found: list[Path] = []
        for idx in sorted(root.rglob("index.json")):
            d = idx.parent
            if not (d / "utterances.jsonl").exists():
                continue
            name = d.name.lower()
            if "awry" in name or name == "cga":
                continue
            found.append(d)
        return found

    def _load_convokit(self, corpus_dirs: list[Path], root: Path) -> list[Thread]:
        """Load WikiConv ConvoKit corpora, expanding action events into messages.

        Each utterance is a base DISCUSS message authored by its speaker. Its
        ``meta.modification/deletion/restoration`` lists are *separate actions*
        performed (often by a different user — a moderator) on that comment, so
        each event is emitted as its own Message (EDIT / DELETE / RESTORE)
        attributed to the actor that performed it. This is the real
        action-policy signal the behaviour axis clusters on.
        """
        cga = self._load_cga_labels(root)
        threads_map: dict[str, Thread] = {}

        _event_action = {
            "modification": ActionType.EDIT,
            "deletion": ActionType.DELETE,
            "restoration": ActionType.RESTORE,
        }

        n_utts = 0
        # When limited, sample evenly across year-corpora rather than
        # front-loading the earliest (small, low-conflict) years.
        per_dir = (
            max(1, self.limit // len(corpus_dirs))
            if (self.limit is not None and corpus_dirs) else None
        )
        for cdir in corpus_dirs:
            if self.limit is not None and n_utts >= self.limit:
                break
            conv_meta = self._load_conversations(cdir)
            dir_count = 0
            with open(cdir / "utterances.jsonl") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if self.limit is not None and n_utts >= self.limit:
                        break
                    if per_dir is not None and dir_count >= per_dir:
                        break
                    n_utts += 1
                    dir_count += 1
                    utt = json.loads(line)
                    msgs = self._convokit_utterance_to_messages(
                        utt, conv_meta, cga, _event_action
                    )
                    for msg in msgs:
                        thread = threads_map.get(msg.thread_id)
                        if thread is None:
                            topic = conv_meta.get(msg.thread_id, {}).get(
                                "page_title", msg.thread_id
                            )
                            thread = Thread(
                                thread_id=msg.thread_id,
                                platform=self.get_platform(),
                                topic=str(topic) if topic else msg.thread_id,
                            )
                            threads_map[msg.thread_id] = thread
                        thread.add_message(msg)

        logger.info(
            f"Loaded {n_utts} utterances → {len(threads_map)} threads"
            + (f" (limit={self.limit}, ~{per_dir}/year)" if self.limit else "")
        )
        return list(threads_map.values())

    @staticmethod
    def _load_conversations(cdir: Path) -> dict[str, dict]:
        """Read conversations.json → {conv_id: {page_id, page_title, ...}}."""
        path = cdir / "conversations.json"
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        return {cid: (v.get("meta", {}) if isinstance(v, dict) else {}) for cid, v in data.items()}

    def _load_cga_labels(self, root: Path) -> dict[str, dict]:
        """Build conflict-label side-table from the CGA corpus if present.

        Returns {utterance_id: {"comment_has_personal_attack": bool,
        "conversation_has_personal_attack": bool}}. Empty if CGA absent or its
        conversations do not overlap the loaded WikiConv years.
        """
        cga_dirs = [
            d.parent for d in root.rglob("index.json")
            if (d.parent / "utterances.jsonl").exists()
            and ("awry" in d.parent.name.lower() or d.parent.name.lower() == "cga")
        ]
        labels: dict[str, dict] = {}
        for cdir in cga_dirs:
            # conversation-level gold labels
            conv = self._load_conversations(cdir)
            conv_attack = {
                cid: m.get("conversation_has_personal_attack")
                for cid, m in conv.items()
            }
            with open(cdir / "utterances.jsonl") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    u = json.loads(line)
                    m = u.get("meta", {}) if isinstance(u.get("meta"), dict) else {}
                    labels[str(u.get("id"))] = {
                        "comment_has_personal_attack": m.get("comment_has_personal_attack"),
                        "conversation_has_personal_attack": conv_attack.get(
                            u.get("conversation_id")
                        ),
                    }
        if labels:
            from loguru import logger
            logger.info(f"Loaded {len(labels)} CGA conflict labels for join")
        return labels

    def _convokit_utterance_to_messages(
        self,
        utt: dict,
        conv_meta: dict[str, dict],
        cga: dict[str, dict],
        event_action: dict[str, ActionType],
    ) -> list[Message]:
        """Expand one ConvoKit utterance into base + action-event messages."""
        utt_id = str(utt.get("id"))
        conv_id = str(utt.get("conversation_id") or utt.get("root") or utt_id)
        speaker = utt.get("speaker")
        if isinstance(speaker, dict):
            speaker = speaker.get("id")
        if not speaker:
            return []
        meta = utt.get("meta", {}) if isinstance(utt.get("meta"), dict) else {}
        reply_to = utt.get("reply-to") or utt.get("reply_to")

        base_meta = {
            "toxicity": meta.get("toxicity"),
            "severe_toxicity": meta.get("sever_toxicity"),
            "is_section_header": meta.get("is_section_header"),
            "indentation": meta.get("indentation"),
            "page_title": conv_meta.get(conv_id, {}).get("page_title"),
            "page_type": conv_meta.get(conv_id, {}).get("page_type"),
        }
        if utt_id in cga:
            base_meta.update(cga[utt_id])

        messages = [
            Message(
                msg_id=utt_id,
                thread_id=conv_id,
                user_id=str(speaker),
                platform=self.get_platform(),
                timestamp=self._parse_ts(utt.get("timestamp")),
                text=str(utt.get("text") or ""),
                action_type=ActionType.DISCUSS,
                parent_msg_id=str(reply_to) if reply_to else None,
                metadata=base_meta,
            )
        ]

        # Expand modification / deletion / restoration events into their own
        # messages, attributed to the user that performed the action.
        for field_name, action in event_action.items():
            for i, ev in enumerate(meta.get(field_name) or []):
                if not isinstance(ev, dict):
                    continue
                actor = ev.get("speaker")
                if isinstance(actor, dict):
                    actor = actor.get("id")
                if not actor:
                    continue
                ev_meta = ev.get("meta_dict", {}) if isinstance(ev.get("meta_dict"), dict) else {}
                messages.append(
                    Message(
                        msg_id=f"{utt_id}::{field_name}::{i}",
                        thread_id=conv_id,
                        user_id=str(actor),
                        platform=self.get_platform(),
                        timestamp=self._parse_ts(ev.get("timestamp")),
                        text=str(ev.get("text") or ""),
                        action_type=action,
                        parent_msg_id=utt_id,  # the comment acted upon
                        metadata={
                            "event": field_name,
                            "target_utt": utt_id,
                            "toxicity": ev_meta.get("toxicity"),
                        },
                    )
                )
        return messages

    @staticmethod
    def _parse_ts(raw) -> datetime:
        """Parse ConvoKit timestamps (epoch float/int or ISO string)."""
        if raw is None:
            return datetime.now()
        try:
            if isinstance(raw, (int, float)):
                return datetime.fromtimestamp(float(raw))
            s = str(raw).strip()
            # epoch like "1.189190940E09" or "1056871838.0"
            try:
                return datetime.fromtimestamp(float(s))
            except (ValueError, OverflowError):
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, TypeError, OverflowError):
            return datetime.now()

    def _load_flat_jsonl(self, data_dir: Path) -> list[Thread]:
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

        return list(threads_map.values())

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
        # NOTE: the meta-reply_to fallback must be parenthesised so the
        # ``if isinstance(…)`` guard applies ONLY to the meta lookup,
        # not to the entire ``or`` chain (Python ternary precedence).
        parent_id = (
            src.get("parent_comment_id")
            or src.get("parent_id")
            or src.get("reply_to")
            or (
                src.get("meta", {}).get("reply_to")
                if isinstance(src.get("meta"), dict)
                else None
            )
        )

        # Timestamp
        raw_ts = (
            src.get("timestamp")
            or src.get("created_at")
            or src.get("utc_timestamp")
            or (
                src.get("meta", {}).get("timestamp", "")
                if isinstance(src.get("meta"), dict)
                else None
            )
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
