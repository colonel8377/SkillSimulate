"""Retrieval-augmented Mind Model selection for Tier 2 (outline §4.4).

Instead of injecting ALL mind models statically, Tier 2 dynamically
retrieves the 3-5 most relevant reasoning templates based on the current
dialogue state:
  - stance direction (supportive / oppositional / neutral)
  - conflict intensity (low / medium / high)
  - topic domain (extracted keywords)

Relevance = Sentence-BERT cosine similarity between a dialogue-state query
and each mind model's (name + description + application) document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.skill.schema import MindModel

if TYPE_CHECKING:
    from src.llm.client import LLMClient


# DELETED 2026-07-13: _SUPPORTIVE_WORDS, _OPPOSITIONAL_WORDS,
# _CONFLICT_WORDS, _CONSENSUS_WORDS keyword lexicons.
# Context-blind word-lists cannot reliably infer dialogue state.
# LLM classifier is the primary path; fallback returns neutral.

# How many recent messages to scan for dialogue-state inference
_CONTEXT_WINDOW = 8



# ---------------------------------------------------------------------------
# LLM-based dialogue state classification (replaces context-blind word-lists)
# ---------------------------------------------------------------------------
_DIALOGUE_STATE_PROMPT = """Analyze the following recent dialogue messages and classify the dialogue state.

Respond with a JSON object with these fields:
- "stance": one of "supportive", "oppositional", "neutral"
- "conflict_intensity": a float between 0.0 and 1.0
- "is_consensus_seeking": true or false

Recent messages:
{messages_text}"""


class LLMDialogueStateClassifier:
    """LLM-based dialogue state classification with neutral fallback."""

    def __init__(self, llm_client: LLMClient, model_name: str):
        self._client = llm_client
        self._model = model_name

    async def classify(self, messages: list[dict[str, str]]) -> DialogueState:
        """Classify dialogue state from recent messages via LLM.

        Falls back to neutral DialogueState on LLM failure.
        """
        recent = [m for m in messages if m.get("content", "").strip()][-_CONTEXT_WINDOW:]
        if not recent:
            return DialogueState(
                stance="neutral",
                conflict_intensity=0.0,
                topic="",
                is_consensus_seeking=False,
            )

        messages_text = "\n".join(
            f"- {m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent
        )
        prompt = _DIALOGUE_STATE_PROMPT.format(messages_text=messages_text)

        try:
            result = await self._client.chat_completion_json(
                messages=[{"role": "user", "content": prompt}],
                model_name=self._model,
                temperature=0.0,
                max_tokens=256,
                default=None,
            )
            if isinstance(result, dict):
                stance = result.get("stance", "neutral")
                if stance not in ("supportive", "oppositional", "neutral"):
                    stance = "neutral"
                conflict = result.get("conflict_intensity", 0.0)
                if not isinstance(conflict, (int, float)):
                    conflict = 0.0
                conflict = max(0.0, min(1.0, float(conflict)))
                consensus = result.get("is_consensus_seeking", False)
                if not isinstance(consensus, bool):
                    consensus = False
                # topic extraction stays regex-based (structural)
                topic = _extract_topic_fallback(messages)
                return DialogueState(
                    stance=stance,
                    conflict_intensity=conflict,
                    topic=topic,
                    is_consensus_seeking=consensus,
                )
        except Exception:
            pass

        # Fallback to word-list heuristic
        return _dialogue_state_fallback(messages)


def _extract_topic_fallback(messages: list[dict[str, str]]) -> str:
    """Extract topic from messages — delegates to MindModelRetriever._extract_topic."""
    return MindModelRetriever._extract_topic(messages)


_dialogue_state_fallback_warned: bool = False


def _dialogue_state_fallback(messages: list[dict[str, str]]) -> DialogueState:
    """Neutral default when LLM classifier unavailable.

    Returns neutral stance, zero conflict, no consensus seeking.
    Does not attempt context-blind keyword inference.
    """
    global _dialogue_state_fallback_warned
    if not _dialogue_state_fallback_warned:
        from loguru import logger
        logger.warning(
            "Dialogue state fallback used — LLM classifier unavailable. "
            "Returning neutral state."
        )
        _dialogue_state_fallback_warned = True
    topic = _extract_topic_fallback(messages)
    return DialogueState(
        stance="neutral",
        conflict_intensity=0.0,
        topic=topic,
        is_consensus_seeking=False,
    )

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

    def __init__(
        self,
        top_k: int = 5,
        llm_client: LLMClient | None = None,
        model_name: str | None = None,
    ):
        self.top_k = max(1, top_k)
        self._embedder = None
        # Cache: mind model doc string -> embedding (doc is stable per model)
        self._doc_cache: dict[str, np.ndarray] = {}
        # LLM-based dialogue state classifier (optional)
        self._llm_classifier: LLMDialogueStateClassifier | None = None
        if llm_client is not None and model_name is not None:
            self._llm_classifier = LLMDialogueStateClassifier(llm_client, model_name)

    @property
    def embedder(self):
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Dialogue-state inference
    # ------------------------------------------------------------------

    async def infer_dialogue_state(self, messages: list[dict[str, str]]) -> DialogueState:
        """Infer stance / conflict intensity / topic from recent messages.

        Uses LLM classifier if available, else returns neutral state.
        """
        if self._llm_classifier is not None:
            return await self._llm_classifier.classify(messages)
        return _dialogue_state_fallback(messages)

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
            scored.append((sim, mm))

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
