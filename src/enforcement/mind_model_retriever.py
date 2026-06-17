"""Retrieval-augmented Mind Model selection for Tier 2 (outline §4.4).

Instead of injecting ALL mind models statically, Tier 2 dynamically
retrieves the 3-5 most relevant reasoning templates based on the current
dialogue state:
  - stance direction (supportive / oppositional / neutral)
  - conflict intensity (low / medium / high)
  - topic domain (extracted keywords)

Relevance = Sentence-BERT cosine similarity between a dialogue-state query
and each mind model's (name + description + application) document, plus a
keyword-overlap bonus for stance/conflict alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.skill.schema import MindModel


# Keyword lexicons for dialogue-state inference
_SUPPORTIVE_WORDS = {
    "agree", "agreeing", "agreed", "support", "supporting", "true", "correct",
    "right", "exactly", "absolutely", "indeed", "concur", "endorse", "delta",
    "convinced", "persuaded", "fair point", "good point",
}
_OPPOSITIONAL_WORDS = {
    "disagree", "disagreeing", "disagree", "wrong", "incorrect", "false",
    "no", "not", "however", "but", "counter", "rebut", "refute", "oppose",
    "object", "challenge", "dispute", "contradict", "flawed", "mistake",
}
_CONFLICT_WORDS = {
    "attack", "insult", "hostile", "angry", "frustrated", "ridiculous",
    "stupid", "ignorant", "troll", "ban", "report", "revert", "vandal",
    "threaten", "abuse", "harass", "offensive",
}
_CONSENSUS_WORDS = {
    "consensus", "compromise", "middle ground", "common ground", "agree to",
    "settle", "mediate", "reconcile", "bridge", "accommodate",
}

# How many recent messages to scan for dialogue-state inference
_CONTEXT_WINDOW = 8


@dataclass
class DialogueState:
    """Inferred current dialogue state."""

    stance: str          # "supportive" | "oppositional" | "neutral"
    conflict_intensity: float  # [0.0, 1.0]
    topic: str           # extracted topic string
    is_consensus_seeking: bool

    @property
    def label(self) -> str:
        return (
            f"stance={self.stance}, "
            f"conflict_intensity={self.conflict_intensity:.2f}, "
            f"topic={self.topic[:40]}"
        )


class MindModelRetriever:
    """Dynamic top-k retrieval of Mind Models given dialogue state."""

    def __init__(self, top_k: int = 5):
        self.top_k = max(1, top_k)
        self._embedder = None
        # Cache: mind model doc string -> embedding (doc is stable per model)
        self._doc_cache: dict[str, np.ndarray] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Dialogue-state inference
    # ------------------------------------------------------------------

    def infer_dialogue_state(self, messages: list[dict[str, str]]) -> DialogueState:
        """Infer stance / conflict intensity / topic from recent messages."""
        recent = [m for m in messages if m.get("content", "").strip()][-_CONTEXT_WINDOW:]
        text = " ".join(m.get("content", "") for m in recent).lower()
        tokens = set(text.split())

        n_support = sum(1 for w in _SUPPORTIVE_WORDS if w in text)
        n_oppose = sum(1 for w in _OPPOSITIONAL_WORDS if w in text)

        if n_support > n_oppose and n_support > 0:
            stance = "supportive"
        elif n_oppose > n_support and n_oppose > 0:
            stance = "oppositional"
        else:
            stance = "neutral"

        # Conflict intensity: density of conflict + oppositional markers
        n_conflict = sum(1 for w in _CONFLICT_WORDS if w in text)
        token_count = max(len(text.split()), 1)
        conflict_intensity = min(1.0, (n_conflict * 3 + n_oppose) / (token_count * 0.1 + 1))

        is_consensus_seeking = any(w in text for w in _CONSENSUS_WORDS)

        topic = self._extract_topic(messages)

        return DialogueState(
            stance=stance,
            conflict_intensity=conflict_intensity,
            topic=topic,
            is_consensus_seeking=is_consensus_seeking,
        )

    @staticmethod
    def _extract_topic(messages: list[dict[str, str]]) -> str:
        """Extract topic from a 'Topic:' marker in the messages, else recent user text."""
        for m in messages:
            content = m.get("content", "")
            lower = content.lower()
            idx = lower.find("topic:")
            if idx >= 0:
                # Topic runs until the next newline or "memory:" marker
                tail = content[idx + len("topic:"):]
                end = tail.lower().find("memory:")
                if end >= 0:
                    tail = tail[:end]
                return tail.split("\n", 1)[0].strip()[:200]
        # Fallback: last user message
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content", "").strip():
                return m["content"].strip()[:200]
        return ""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        mind_models: list[MindModel],
        state: DialogueState,
    ) -> list[MindModel]:
        """Return the top-k most relevant mind models for the current state."""
        if not mind_models:
            return []
        if len(mind_models) <= self.top_k:
            return list(mind_models)

        query = self._build_query(state)
        query_emb = self._embed(query)

        scored: list[tuple[float, MindModel]] = []
        for mm in mind_models:
            doc = self._mm_doc(mm)
            doc_emb = self._doc_embedding(mm, doc)
            sim = float(np.dot(query_emb, doc_emb))
            bonus = self._keyword_bonus(mm, state)
            scored.append((sim + bonus, mm))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mm for _, mm in scored[: self.top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode(text, show_progress_bar=False)
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _doc_embedding(self, mm: MindModel, doc: str) -> np.ndarray:
        cached = self._doc_cache.get(mm.name)
        if cached is not None:
            return cached
        emb = self._embed(doc)
        self._doc_cache[mm.name] = emb
        return emb

    @staticmethod
    def _mm_doc(mm: MindModel) -> str:
        """Document representation of a mind model for matching."""
        parts = [mm.name, mm.description, mm.application]
        return " ".join(p for p in parts if p).lower()

    @staticmethod
    def _build_query(state: DialogueState) -> str:
        """Build a textual query representing the dialogue state."""
        intensity = (
            "high" if state.conflict_intensity > 0.5
            else "medium" if state.conflict_intensity > 0.2
            else "low"
        )
        goal = (
            "consensus seeking" if state.is_consensus_seeking
            else "conflict engagement" if state.conflict_intensity > 0.4
            else "discussion"
        )
        return (
            f"reasoning for {goal} stance {state.stance} "
            f"conflict intensity {intensity} topic {state.topic}".lower()
        )

    @staticmethod
    def _keyword_bonus(mm: MindModel, state: DialogueState) -> float:
        """Heuristic bonus for explicit stance/conflict/consensus alignment."""
        doc = MindModelRetriever._mm_doc(mm)
        bonus = 0.0

        if state.stance == "oppositional" and any(
            w in doc for w in ("argument", "counter", "rebuttal", "oppos", "critique")
        ):
            bonus += 0.15
        if state.stance == "supportive" and any(
            w in doc for w in ("support", "endorse", "agree", "consensus")
        ):
            bonus += 0.15
        if state.conflict_intensity > 0.4 and any(
            w in doc for w in ("conflict", "escalat", "de-escalat", "dispute", "attack")
        ):
            bonus += 0.15
        if state.is_consensus_seeking and any(
            w in doc for w in ("consensus", "compromise", "mediate", "bridge")
        ):
            bonus += 0.15

        return bonus
