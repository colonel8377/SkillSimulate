"""Archetype profiling: freeze each leaf's typical utterances + behavioral tags.

For every leaf cluster (role × style) this builds:
- representative members  (top-M users closest to this leaf's language centroid
                            AND farthest from its nearest-neighbour leaf's —
                            contrastive, not just "most typical", so material
                            doesn't converge on whatever's generic across leaves),
- typical utterances       (their messages, ranked by the same contrastive score,
                            semantically de-duplicated, capped, each with context),
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
    # Conflict / affect (single hostile-tail axis after tox_max/sev_max merge).
    ("hostility_score", "high", "hostile"),
    ("hostility_score", "low", "civil"),
    # Topical / temporal
    ("topical_breadth", "high", "generalist"),
    ("topical_breadth", "low", "specialist"),
    ("tenure", "high", "veteran"),
    ("tenure", "low", "newcomer"),
    # Conversational
    ("mean_indentation", "high", "deep-threading"),
    ("reply_rate", "high", "responder"),
    ("reply_rate", "low", "thread-initiator"),
    ("question_rate", "high", "inquisitive"),
    ("verbosity", "high", "verbose"),
    ("verbosity", "low", "terse"),
    ("activity", "high", "high-activity"),
    ("activity_density", "high", "intense-burst"),
    ("burstiness_cv", "high", "bursty-poster"),
    # Namespace focus: positive = content, negative = interpersonal.
    ("namespace_focus", "high", "article-focused"),
    ("namespace_focus", "low", "interpersonal-space"),
    # Linguistic / affect
    ("exclaim_rate", "high", "emphatic"),
    ("lexical_ttr", "high", "rich-vocabulary"),
    # NOTE: tags formerly driven by sparse/dropped features (frequently-targeted,
    # conflict-engaging, high-conflict-exposure, central/popular, mutual-engager,
    # broad-interaction, policy-oriented, project/policy-space) were removed when
    # those features left VECTOR_FIELD_NAMES. _compute_tags skips any lexicon
    # entry whose feature is absent, so this list stays crash-safe if features
    # change again.
]
Z_THRESHOLD = 1.0


@dataclass
class TypicalUtterance:
    member: str
    action: str
    text: str
    parent_context: str
    topic: str
    # Provenance for train/test separation (outline §5.1): the source
    # thread (conversation_id) and message id, so the simulation pool
    # can exclude any thread/message that informed skill distillation.
    # Older typical.jsonl files lack these keys; consumers must
    # ``.get("thread_id", "")`` for back-compat.
    thread_id: str = ""
    msg_id: str = ""


@dataclass
class LeafProfile:
    leaf_id: int
    members: list[str]
    typical_utterances: list[TypicalUtterance] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    size: int = 0
    n_candidates: int = 0  # pre-dedup candidate utterance count (distillation coverage)


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

        # Per-leaf language centroids, computed up front so selection can be
        # contrastive: prefer material close to THIS leaf's own voice and far
        # from its nearest-neighbour leaf's, instead of "closest to my own
        # centroid" alone — which tends to surface the most generic, most
        # cross-leaf-similar examples (the leaf-overlap finding in
        # quality_report.py's nearest_other_leaf_cosine diagnostic).
        leaf_members = {lid: cluster_result.get_cluster_members(lid) for lid in leaf_ids}
        leaf_lang_emb = {
            lid: self._member_embeddings(leaf_members[lid], threads, lid, cluster_result)
            for lid in leaf_ids
        }
        leaf_lang_centroid = {
            lid: np.mean(list(e.values()), axis=0) for lid, e in leaf_lang_emb.items() if e
        }
        nearest_other = self._nearest_other_leaf(leaf_lang_centroid)

        profiles: dict[int, LeafProfile] = {}
        for lid in leaf_ids:
            members = leaf_members[lid]
            emb = leaf_lang_emb.get(lid, {})
            own_c = leaf_lang_centroid.get(lid)
            other_c = leaf_lang_centroid.get(nearest_other.get(lid))
            reps = self._representative_members(members, emb, own_c, other_c)
            utts, n_candidates = self._typical_utterances(
                reps, threads, lid, cluster_result, msg_by_id, topic_by_thread, own_c, other_c
            )
            profiles[lid] = LeafProfile(
                leaf_id=lid, members=reps, typical_utterances=utts,
                tags=tag_map.get(lid, []), size=len(members), n_candidates=n_candidates,
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
                if feat not in field_idx:
                    continue  # feature dropped from VECTOR_FIELD_NAMES — skip
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

    def _nearest_other_leaf(self, centroids: dict[int, np.ndarray]) -> dict[int, int]:
        """Each leaf's nearest-neighbour leaf by language-centroid L2 distance."""
        ids = list(centroids)
        nearest: dict[int, int] = {}
        for lid in ids:
            best_id, best_d = None, None
            for oid in ids:
                if oid == lid:
                    continue
                d = float(np.linalg.norm(centroids[lid] - centroids[oid]))
                if best_d is None or d < best_d:
                    best_d, best_id = d, oid
            if best_id is not None:
                nearest[lid] = best_id
        return nearest

    def _contrastive_scores(
        self, embs: dict[str, np.ndarray] | list[np.ndarray], own_centroid, other_centroid
    ):
        """own-distance minus other-distance — small/negative means close to this
        leaf's voice and far from its nearest neighbour's; falls back to plain
        own-distance when there's no neighbour centroid to contrast against."""
        items = embs.items() if isinstance(embs, dict) else enumerate(embs)
        scores = {}
        for key, v in items:
            own_d = float(np.linalg.norm(v - own_centroid))
            if other_centroid is not None:
                scores[key] = own_d - float(np.linalg.norm(v - other_centroid))
            else:
                scores[key] = own_d
        return scores

    def _representative_members(self, members, emb, own_centroid, other_centroid) -> list[str]:
        if len(members) <= self.top_m:
            return list(members)
        if not emb or own_centroid is None:
            return list(members)[: self.top_m]
        scores = self._contrastive_scores(emb, own_centroid, other_centroid)
        ranked = sorted(scores, key=scores.get)
        return ranked[: self.top_m]

    def _typical_utterances(
        self, reps, threads, leaf_id, cr, msg_by_id, topic_by_thread, own_centroid=None, other_centroid=None
    ) -> tuple[list[TypicalUtterance], int]:
        rep_set = set(reps)
        # collect candidate messages authored by representative members
        cands: list[Message] = []
        for t in threads:
            for m in t.messages:
                if m.user_id in rep_set and m.text.strip() and not m.text.startswith("::"):
                    cands.append(m)
        if not cands:
            return [], 0
        embs = np.asarray(self.embedder.encode([m.text[:400] for m in cands], show_progress_bar=False))
        ref_centroid = own_centroid if own_centroid is not None else embs.mean(axis=0)
        scores = self._contrastive_scores(list(embs), ref_centroid, other_centroid)
        order = np.argsort([scores[i] for i in range(len(cands))])

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
                thread_id=m.thread_id, msg_id=m.msg_id,
            ))
        return out, len(cands)

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
