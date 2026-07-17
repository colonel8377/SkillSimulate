"""Direct SBERT retrieval of Mind Models for Tier 2 (outline §4.4).

2026-07-17 simplification: the 5-stage LLM-state-classifier → query-string →
SBERT pipeline collapsed to direct SBERT retrieval using the recent dialogue
as the query. The intermediate ``DialogueState`` labels (stance / conflict)
were only consumed by ``_build_query`` to produce a query string — SBERT can
match dialogue text to Mind Model docs directly.
"""

from __future__ import annotations

import numpy as np

from src.skill.schema import MindModel


# How many recent messages to scan for the retrieval query
_CONTEXT_WINDOW = 8


class MindModelRetriever:
    """Dynamic top-k retrieval of Mind Models given recent dialogue."""

    def __init__(self, top_k: int = 5):
        self.top_k = max(1, top_k)
        self._embedder = None
        # Cache: mind model name -> embedding (doc is stable per model)
        self._doc_cache: dict[str, np.ndarray] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    def retrieve_for_messages(
        self,
        mind_models: list[MindModel],
        messages: list[dict[str, str]],
    ) -> list[MindModel]:
        """Return the top-k most relevant mind models for the recent dialogue.

        Synchronous by design: the body is blocking SBERT encoding, so the
        caller (Tier 2) ships it to the shared embed executor via
        ``run_embed_in_executor``. Declaring it ``async`` would make the
        executor return an un-awaitable coroutine object instead of the
        model list.

        Falls through to the full list when there are fewer models than top_k
        or when the recent dialogue is empty.
        """
        if not mind_models:
            return []
        if len(mind_models) <= self.top_k:
            return list(mind_models)

        recent = [
            m.get("content", "")
            for m in messages[-_CONTEXT_WINDOW:]
            if m.get("content", "").strip()
        ]
        if not recent:
            return list(mind_models)[: self.top_k]

        query = " \n ".join(recent)
        query_emb = self._embed(query)
        scored: list[tuple[float, MindModel]] = []
        for mm in mind_models:
            doc_emb = self._doc_embedding(mm, self._mm_doc(mm))
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
