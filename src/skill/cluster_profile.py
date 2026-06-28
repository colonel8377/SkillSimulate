"""Archetype profiling: freeze each leaf's typical utterances + behavioral tags.

For every leaf cluster (role × style) this builds:
- representative members  (top-M users nearest the leaf's language centroid),
- typical utterances       (their messages, semantically de-duplicated, coverage-
                            selected, capped, each with light context),
- behavioral tags          (deterministic: z-score the leaf's mean feature vector
                            against all leaves, emit labels from a fixed lexicon).

The frozen typical utterances are the distillation input (Step 1) AND a
reproducible, inspectable record of the archetype's voice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from src.clustering.clusterer import ClusterResult
from src.clustering.features import VECTOR_FIELD_NAMES
from src.data.schemas import Message, Thread


# (feature, direction, label) — fired when the leaf's z-score on `feature`
# exceeds +Z_THRESHOLD (direction "high") or below -Z_THRESHOLD ("low").
TAG_LEXICON: list[tuple[str, str, str]] = [
    ("own_toxicity_mean", "high", "hostile"),
    ("own_toxicity_mean", "low", "civil"),
    ("own_severe_toxicity_mean", "high", "highly-hostile"),
    ("toxicity_received_mean", "high", "frequently-targeted"),
    ("reply_to_toxic_rate", "high", "conflict-engaging"),
    ("conflict_engagement_ratio", "high", "high-conflict-exposure"),
    ("attention_received", "high", "central/popular"),
    ("reciprocity", "high", "mutual-engager"),
    ("interlocutor_breadth", "high", "broad-interaction"),
    ("topical_breadth", "high", "generalist"),
    ("topical_breadth", "low", "specialist"),
    ("tenure", "high", "veteran"),
    ("tenure", "low", "newcomer"),
    ("mean_indentation", "high", "deep-threading"),
    ("reply_rate", "high", "responder"),
    ("reply_rate", "low", "thread-initiator"),
    ("question_rate", "high", "inquisitive"),
    ("verbosity", "high", "verbose"),
    ("verbosity", "low", "terse"),
    ("activity", "high", "high-activity"),
    ("wp_citation_rate", "high", "policy-oriented"),
    ("frac_interpersonal", "high", "interpersonal-space"),
    ("frac_content", "high", "article-focused"),
    ("frac_project", "high", "project/policy-space"),
    ("exclaim_rate", "high", "emphatic"),
    ("lexical_ttr", "high", "rich-vocabulary"),
]
Z_THRESHOLD = 1.0


@dataclass
class TypicalUtterance:
    member: str
    action: str
    text: str
    parent_context: str
    topic: str


@dataclass
class LeafProfile:
    leaf_id: int
    members: list[str]
    typical_utterances: list[TypicalUtterance] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    size: int = 0


class ArchetypeProfiler:
    def __init__(
        self,
        top_m_members: int = 8,
        dedup_tau: float = 0.9,
        max_utterances: int = 40,
        max_chars: int = 12000,
        context_chars: int = 200,
    ):
        self.top_m = top_m_members
        self.dedup_tau = dedup_tau
        self.max_utterances = max_utterances
        self.max_chars = max_chars
        self.context_chars = context_chars
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    def build(
        self, threads: list[Thread], cluster_result: ClusterResult
    ) -> dict[int, LeafProfile]:
        # message lookup for parent context + per-user messages by leaf
        msg_by_id: dict[str, Message] = {}
        topic_by_thread: dict[str, str] = {}
        for t in threads:
            topic_by_thread[t.thread_id] = t.topic
            for m in t.messages:
                msg_by_id[m.msg_id] = m

        leaf_ids = [lid for lid in cluster_result.get_cluster_ids() if lid >= 0]

        # leaf mean feature vectors for tagging
        leaf_feat = {}
        for lid in leaf_ids:
            members = cluster_result.get_cluster_members(lid)
            vecs = [
                cluster_result.user_features[u].to_vector()
                for u in members if u in cluster_result.user_features
            ]
            if vecs:
                leaf_feat[lid] = np.mean(vecs, axis=0)
        tag_map = self._compute_tags(leaf_feat)

        profiles: dict[int, LeafProfile] = {}
        for lid in leaf_ids:
            members = cluster_result.get_cluster_members(lid)
            reps = self._representative_members(members, threads, lid, cluster_result)
            utts = self._typical_utterances(reps, threads, lid, cluster_result, msg_by_id, topic_by_thread)
            profiles[lid] = LeafProfile(
                leaf_id=lid, members=reps, typical_utterances=utts,
                tags=tag_map.get(lid, []), size=len(members),
            )
        return profiles

    # ------------------------------------------------------------------
    def _compute_tags(self, leaf_feat: dict[int, np.ndarray]) -> dict[int, list[str]]:
        if len(leaf_feat) < 2:
            return {lid: [] for lid in leaf_feat}
        ids = list(leaf_feat)
        M = np.stack([leaf_feat[i] for i in ids])
        mu, sd = M.mean(0), M.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Z = (M - mu) / sd
        field_idx = {f: i for i, f in enumerate(VECTOR_FIELD_NAMES)}
        out: dict[int, list[str]] = {}
        for row, lid in enumerate(ids):
            tags = []
            for feat, direction, label in TAG_LEXICON:
                z = Z[row, field_idx[feat]]
                if (direction == "high" and z >= Z_THRESHOLD) or (
                    direction == "low" and z <= -Z_THRESHOLD
                ):
                    tags.append(label)
            out[lid] = tags
        return out

    def _member_embeddings(self, members, threads, leaf_id, cr) -> dict[str, np.ndarray]:
        """Mean text embedding per member (over their messages in this corpus)."""
        member_set = set(members)
        texts: dict[str, list[str]] = {m: [] for m in members}
        for t in threads:
            for msg in t.messages:
                if msg.user_id in member_set and msg.text.strip():
                    texts[msg.user_id].append(msg.text[:400])
        out = {}
        for m, ts in texts.items():
            if ts:
                emb = self.embedder.encode(ts[:30], show_progress_bar=False)
                out[m] = np.asarray(emb).mean(axis=0)
        return out

    def _representative_members(self, members, threads, leaf_id, cr) -> list[str]:
        if len(members) <= self.top_m:
            return list(members)
        emb = self._member_embeddings(members, threads, leaf_id, cr)
        if not emb:
            return list(members)[: self.top_m]
        centroid = np.mean(list(emb.values()), axis=0)
        ranked = sorted(emb, key=lambda u: float(np.linalg.norm(emb[u] - centroid)))
        return ranked[: self.top_m]

    def _typical_utterances(self, reps, threads, leaf_id, cr, msg_by_id, topic_by_thread) -> list[TypicalUtterance]:
        rep_set = set(reps)
        # collect candidate messages authored by representative members
        cands: list[Message] = []
        for t in threads:
            for m in t.messages:
                if m.user_id in rep_set and m.text.strip() and not m.text.startswith("::"):
                    cands.append(m)
        if not cands:
            return []
        embs = np.asarray(self.embedder.encode([m.text[:400] for m in cands], show_progress_bar=False))
        centroid = embs.mean(axis=0)
        order = np.argsort([float(np.linalg.norm(embs[i] - centroid)) for i in range(len(cands))])

        # greedy semantic dedup (keep the one nearer the centroid)
        kept_idx: list[int] = []
        kept_embs: list[np.ndarray] = []
        for i in order:
            e = embs[i] / (np.linalg.norm(embs[i]) + 1e-9)
            if any(float(np.dot(e, k)) >= self.dedup_tau for k in kept_embs):
                continue
            kept_idx.append(i)
            kept_embs.append(e)
            if len(kept_idx) >= self.max_utterances:
                break

        out: list[TypicalUtterance] = []
        total = 0
        for i in kept_idx:
            m = cands[i]
            parent = msg_by_id.get(m.parent_msg_id) if m.parent_msg_id else None
            ctx = (parent.text[: self.context_chars] if parent else "")
            txt = m.text[:1000]
            total += len(txt)
            if total > self.max_chars:
                break
            out.append(TypicalUtterance(
                member=m.user_id, action=m.action_type.value, text=txt,
                parent_context=ctx, topic=topic_by_thread.get(m.thread_id, ""),
            ))
        return out

    def save(self, profiles: dict[int, LeafProfile], out_dir: str | Path, platform: str) -> None:
        out_dir = Path(out_dir)
        for lid, prof in profiles.items():
            d = out_dir / platform / f"cluster_{lid}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "typical.jsonl", "w") as f:
                for u in prof.typical_utterances:
                    f.write(json.dumps(u.__dict__, ensure_ascii=False) + "\n")
            with open(d / "profile.json", "w") as f:
                json.dump(
                    {"leaf_id": lid, "size": prof.size, "tags": prof.tags,
                     "members": prof.members, "n_typical": len(prof.typical_utterances)},
                    f, ensure_ascii=False, indent=2,
                )
        logger.info(f"Saved {len(profiles)} leaf profiles to {out_dir / platform}")
