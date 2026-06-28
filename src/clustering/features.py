"""Behavioral signal extraction from user conversation history.

Behaviour = a user's *action policy*, computed structurally from the real
WikiConv stream (no regex, no per-message labelling). WikiConv moderation
actions are rare, so the role axis is driven by the rich, discriminative
per-user signals that ARE present, spanning several facets:

- Conversational:   reply rate, indentation depth, verbosity, activity volume.
- Social / network: interlocutor breadth (out-degree), attention received
                    (in-degree), reciprocity.
- Topical / time:   topical breadth (distinct pages), tenure (activity span).
- Conflict / affect:own toxicity + severe toxicity, replying into toxicity,
                    toxicity received (being targeted), conflict engagement.
- Inquisitiveness:  question rate.
- Policy grounding: WP:/Wikipedia: citation rate.
- Moderation tail:  fraction of deletes, fraction of own comments deleted.

Every feature is a statistic over labels + reply graph + timestamps +
per-message toxicity that the loader already attaches; none invokes a model
or regex-on-meaning.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.schemas import ActionType, Message, Thread


TOXIC_THRESHOLD = 0.6

# WikiConv page_type → namespace bucket (where the user operates).
_NS_INTERPERSONAL = {"user_talk"}
_NS_CONTENT = {"talk", "article", "file_talk", "category_talk"}
_NS_PROJECT = {"wikipedia_talk", "template_talk", "help_talk"}

_WORD_RE = re.compile(r"[a-z]+")


def _indent_value(raw) -> float:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 0.0
    return float(v) if v > 0 else 0.0


@dataclass
class UserFeatures:
    """Action-policy feature vector for a single user."""
    user_id: str
    # Conversational
    reply_rate: float = 0.0
    mean_indentation: float = 0.0
    verbosity: float = 0.0                 # log1p(mean text length)
    activity: float = 0.0                  # log1p(message count)
    # Social / network
    interlocutor_breadth: float = 0.0      # distinct reply-targets per message (out-degree)
    attention_received: float = 0.0        # log1p(replies received) (in-degree)
    reciprocity: float = 0.0               # |out∩in| / |out∪in|
    # Topical / temporal
    topical_breadth: float = 0.0           # log1p(distinct pages touched)
    tenure: float = 0.0                    # log1p(activity span in days)
    # Conflict / affect
    own_toxicity_mean: float = 0.0
    own_severe_toxicity_mean: float = 0.0
    reply_to_toxic_rate: float = 0.0
    toxicity_received_mean: float = 0.0    # hostility directed AT this user
    conflict_engagement_ratio: float = 0.0
    # Inquisitiveness
    question_rate: float = 0.0
    # Policy grounding
    wp_citation_rate: float = 0.0
    # Namespace focus (where the user operates)
    frac_interpersonal: float = 0.0       # user_talk
    frac_content: float = 0.0             # article / talk
    frac_project: float = 0.0             # wikipedia_talk / template_talk / help_talk
    # Linguistic / affect
    exclaim_rate: float = 0.0             # share of messages with "!"
    lexical_ttr: float = 0.0              # type-token ratio (vocabulary richness)
    # Conflict extremes + temporal rhythm (validated additions)
    tox_max: float = 0.0                  # peak toxicity (worst single message)
    sev_max: float = 0.0                  # peak severe toxicity
    burstiness_cv: float = 0.0           # CV of inter-message time gaps
    activity_density: float = 0.0         # log1p(messages / (tenure_days + 1))
    # Bookkeeping (not part of the clustering vector)
    message_count: int = 0
    thread_count: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in _VECTOR_FIELDS], dtype=np.float64)


_VECTOR_FIELDS = (
    "reply_rate", "mean_indentation", "verbosity", "activity",
    "interlocutor_breadth", "attention_received", "reciprocity",
    "topical_breadth", "tenure",
    "own_toxicity_mean", "own_severe_toxicity_mean", "reply_to_toxic_rate",
    "toxicity_received_mean", "conflict_engagement_ratio",
    "question_rate", "wp_citation_rate",
    "frac_interpersonal", "frac_content", "frac_project",
    "exclaim_rate", "lexical_ttr",
    # Conflict extremes + temporal rhythm (validated: ADDS SIGNAL, |r|<0.6 vs existing)
    "tox_max", "sev_max", "burstiness_cv",
    # Activity rhythm. Dropped via data-driven audit (2003-2004 sample):
    #   - frac_revert/frac_report: substring heuristics, no native WikiConv action
    #   - frac_discuss/frac_edit/frac_delete/frac_restore/frac_moderation/
    #     frac_moderated/action_entropy: moderation actions <1% coverage (DEAD)
    #   - out_degree_log/in_degree_log/attention_per_msg: |r|>0.9 with kept cols
    #   - initiation_rate: exactly 1 - reply_rate (double-weights that axis)
    "activity_density",
)
VECTOR_FIELD_NAMES = _VECTOR_FIELDS


class FeatureExtractor:
    """Extracts the multi-facet action-policy behaviour vector per user."""

    def __init__(self, contested_threshold: int = 3, toxic_threshold: float = TOXIC_THRESHOLD):
        self.contested_threshold = contested_threshold
        self.toxic_threshold = toxic_threshold

    def extract_all(self, threads: list[Thread]) -> dict[str, UserFeatures]:
        # ---- Global pre-pass: cross-message indices + reply graph ----
        msg_author: dict[str, str] = {}
        msg_tox: dict[str, float] = {}
        contested_threads: set[str] = set()

        for thread in threads:
            if len(thread.participants) >= self.contested_threshold:
                contested_threads.add(thread.thread_id)
            for m in thread.messages:
                msg_author[m.msg_id] = m.user_id
                tv = m.metadata.get("toxicity")
                if isinstance(tv, (int, float)):
                    msg_tox[m.msg_id] = float(tv)

        # reply graph: out-targets / in-repliers / replies-received / toxicity-received
        out_targets: dict[str, set] = defaultdict(set)
        in_repliers: dict[str, set] = defaultdict(set)
        replies_received: Counter = Counter()
        tox_received: dict[str, list] = defaultdict(list)
        # moderation targets: base comments that were edited/deleted/restored by others
        moderated_targets: set[str] = set()
        for thread in threads:
            for m in thread.messages:
                if not m.parent_msg_id:
                    continue
                pa = msg_author.get(str(m.parent_msg_id))
                if not pa or pa == m.user_id:
                    continue
                out_targets[m.user_id].add(pa)
                in_repliers[pa].add(m.user_id)
                replies_received[pa] += 1
                if m.msg_id in msg_tox:
                    tox_received[pa].append(msg_tox[m.msg_id])
                # event messages (EDIT/DELETE/RESTORE) target the original comment
                if m.action_type in {ActionType.EDIT, ActionType.DELETE, ActionType.RESTORE}:
                    moderated_targets.add(str(m.parent_msg_id))

        user_msgs: dict[str, list[tuple[Message, Thread]]] = defaultdict(list)
        for thread in threads:
            for m in thread.messages:
                user_msgs[m.user_id].append((m, thread))

        return {
            uid: self._extract_single(
                uid, pairs, contested_threads, msg_tox, msg_author,
                out_targets, in_repliers, replies_received, tox_received,
                moderated_targets,
            )
            for uid, pairs in user_msgs.items()
        }

    def _extract_single(
        self, user_id, pairs, contested_threads, msg_tox, msg_author,
        out_targets, in_repliers, replies_received, tox_received,
        moderated_targets: set[str],
    ) -> UserFeatures:
        msgs = [m for m, _ in pairs]
        total = len(msgs)
        if total == 0:
            return UserFeatures(user_id=user_id)
        comments = [m for m in msgs if m.action_type == ActionType.DISCUSS]

        # Conversational
        reply_rate = sum(1 for m in msgs if m.parent_msg_id) / total
        mean_indentation = float(np.mean([_indent_value(m.metadata.get("indentation")) for m in msgs]))
        verbosity = math.log1p(float(np.mean([len(m.text) for m in msgs])))
        activity = math.log1p(total)

        # Social / network
        out_set, in_set = out_targets.get(user_id, set()), in_repliers.get(user_id, set())
        interlocutor_breadth = len(out_set) / total
        attention_received = math.log1p(replies_received.get(user_id, 0))
        union = out_set | in_set
        reciprocity = len(out_set & in_set) / len(union) if union else 0.0

        # Topical / temporal
        topical_breadth = math.log1p(len({t.topic for _, t in pairs}))
        times = [m.timestamp for m in msgs if m.timestamp is not None]
        span_days = (max(times) - min(times)).total_seconds() / 86400 if len(times) >= 2 else 0.0
        tenure = math.log1p(max(0.0, span_days))

        # Temporal rhythm: burstiness = CV of inter-message time gaps.
        sorted_times = sorted(times)
        if len(sorted_times) >= 3:
            gaps = np.diff([t.timestamp() for t in sorted_times])
            burstiness_cv = float(gaps.std() / (gaps.mean() + 1e-9)) if gaps.size else 0.0
        else:
            burstiness_cv = 0.0
        activity_density = (
            math.log1p(total / (span_days + 1.0)) if span_days > 0.0 else activity
        )

        # Conflict / affect
        own_tox = [msg_tox[m.msg_id] for m in msgs if m.msg_id in msg_tox]
        own_toxicity_mean = float(np.mean(own_tox)) if own_tox else 0.0
        tox_max = float(np.max(own_tox)) if own_tox else 0.0
        sev = [m.metadata.get("severe_toxicity") for m in msgs]
        sev = [s for s in sev if isinstance(s, (int, float))]
        own_severe_toxicity_mean = float(np.mean(sev)) if sev else 0.0
        sev_max = float(np.max(sev)) if sev else 0.0
        dwp = [m for m in comments if m.parent_msg_id]
        reply_to_toxic_rate = (
            sum(1 for m in dwp if msg_tox.get(str(m.parent_msg_id), 0.0) >= self.toxic_threshold)
            / len(dwp) if dwp else 0.0
        )
        tr = tox_received.get(user_id, [])
        toxicity_received_mean = float(np.mean(tr)) if tr else 0.0
        contested = sum(1 for _, t in pairs if t.thread_id in contested_threads)
        conflict_engagement_ratio = contested / total

        # Inquisitiveness
        question_rate = sum(1 for m in msgs if m.text.rstrip().endswith("?")) / total

        # Policy grounding
        wp_citation_rate = sum(1 for m in msgs if "WP:" in m.text or "Wikipedia:" in m.text) / total

        # Namespace focus (where they operate)
        ns = Counter()
        for m in msgs:
            pt = m.metadata.get("page_type")
            if pt in _NS_INTERPERSONAL:
                ns["interpersonal"] += 1
            elif pt in _NS_CONTENT:
                ns["content"] += 1
            elif pt in _NS_PROJECT:
                ns["project"] += 1
        frac_interpersonal = ns["interpersonal"] / total
        frac_content = ns["content"] / total
        frac_project = ns["project"] / total

        # Linguistic / affect
        exclaim_rate = sum(1 for m in msgs if "!" in m.text) / total
        toks = []
        for m in msgs[:50]:
            toks.extend(_WORD_RE.findall(m.text.lower()))
        lexical_ttr = len(set(toks)) / len(toks) if toks else 0.0

        return UserFeatures(
            user_id=user_id, reply_rate=reply_rate, mean_indentation=mean_indentation,
            verbosity=verbosity, activity=activity, interlocutor_breadth=interlocutor_breadth,
            attention_received=attention_received, reciprocity=reciprocity,
            topical_breadth=topical_breadth, tenure=tenure,
            own_toxicity_mean=own_toxicity_mean, own_severe_toxicity_mean=own_severe_toxicity_mean,
            reply_to_toxic_rate=reply_to_toxic_rate, toxicity_received_mean=toxicity_received_mean,
            conflict_engagement_ratio=conflict_engagement_ratio, question_rate=question_rate,
            wp_citation_rate=wp_citation_rate,
            frac_interpersonal=frac_interpersonal, frac_content=frac_content,
            frac_project=frac_project, exclaim_rate=exclaim_rate, lexical_ttr=lexical_ttr,
            # Conflict extremes + temporal rhythm (validated additions)
            tox_max=tox_max, sev_max=sev_max, burstiness_cv=burstiness_cv,
            activity_density=activity_density,
            message_count=total, thread_count=len({t.thread_id for _, t in pairs}),
        )
