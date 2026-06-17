"""Language embeddings for clustering.

Uses Sentence-BERT to embed user messages and aggregate via mean pooling.
Supports domain-adapted embeddings via config.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.data.schemas import Message


class EmbeddingExtractor:
    """Extracts aggregated language embeddings per user."""

    def __init__(
        self,
        model_name: str | None = None,
        min_text_length: int = 10,
    ):
        self.model_name = model_name or settings.sentence_transformer_model
        self.min_text_length = min_text_length
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from src.config.settings import get_shared_embedder
            self._model = get_shared_embedder()
        return self._model

    def embed_messages(self, messages: list[Message]) -> np.ndarray:
        """Embed a list of messages and return their vectors."""
        texts = [m.text for m in messages if len(m.text.strip()) >= self.min_text_length]
        if not texts:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(texts, show_progress_bar=False)

    def embed_user(self, messages: list[Message]) -> np.ndarray:
        """Aggregate embedding for a single user via mean pooling."""
        vectors = self.embed_messages(messages)
        if vectors.ndim == 1:
            return vectors
        return np.mean(vectors, axis=0)

    def embed_all_users(
        self,
        user_messages: dict[str, list[Message]],
    ) -> dict[str, np.ndarray]:
        """Compute aggregated embedding for each user.

        Args:
            user_messages: Mapping user_id → list of their messages.

        Returns:
            Mapping user_id → embedding vector.
        """
        # Batch encode all texts at once for efficiency
        all_texts = []
        user_text_ranges: dict[str, tuple[int, int]] = {}
        idx = 0
        for user_id, msgs in user_messages.items():
            texts = [m.text for m in msgs if len(m.text.strip()) >= self.min_text_length]
            start = idx
            for t in texts:
                all_texts.append(t)
                idx += 1
            user_text_ranges[user_id] = (start, idx)

        if not all_texts:
            dim = self.model.get_sentence_embedding_dimension()
            return {uid: np.zeros(dim) for uid in user_messages}

        all_embeddings = self.model.encode(all_texts, show_progress_bar=True)

        result = {}
        for user_id, (start, end) in user_text_ranges.items():
            if start == end:
                dim = self.model.get_sentence_embedding_dimension()
                result[user_id] = np.zeros(dim)
            else:
                result[user_id] = np.mean(all_embeddings[start:end], axis=0)

        return result


def rolling_agent_state(
    action_embeddings: list[np.ndarray] | np.ndarray,
    window: int = 5,
    embed_dim: int | None = None,
) -> np.ndarray:
    """Rolling-mean aggregation of an agent's last-N action embeddings.

    Used by the R4 persona-collapse stress test (outline §5.6.7 /
    docs/r4_persona_collapse_stress_test.md §4.3) to define per-agent
    state at each measurement turn: state(turn t) = mean of the agent's
    last `window` action embeddings up to turn t.

    The rolling window smooths turn-to-turn noise while preserving the
    drift signal that distinguishes collapsing vs. stable personas.

    Args:
        action_embeddings: Either a list of per-action embedding vectors
            (chronological order) or a (n_actions, embed_dim) array.
        window: Number of recent actions to average. The protocol default
            is 5 (per docs/r4_persona_collapse_stress_test.md §4.3).
        embed_dim: Required fallback dimension when action_embeddings is
            empty (returns a zero vector of this dim).

    Returns:
        (embed_dim,) aggregated state vector.

    Raises:
        ValueError: if action_embeddings is empty and embed_dim is None.
    """
    if isinstance(action_embeddings, list):
        if not action_embeddings:
            if embed_dim is None:
                raise ValueError(
                    "action_embeddings is empty — must pass embed_dim to "
                    "produce a zero-vector fallback"
                )
            return np.zeros(embed_dim)
        embs = np.stack(action_embeddings)
    else:
        embs = np.asarray(action_embeddings)
        if embs.size == 0:
            if embed_dim is None:
                raise ValueError(
                    "action_embeddings is empty — must pass embed_dim to "
                    "produce a zero-vector fallback"
                )
            return np.zeros(embed_dim)

    recent = embs[-window:]
    return np.mean(recent, axis=0)


def mean_pairwise_cosine(agent_states: dict[str, np.ndarray]) -> tuple[float, float]:
    """Mean pairwise cosine similarity across agents (R4 §4.2 H5 metric).

    Detects persona convergence: as agents collapse to a modal behavior
    pattern, their state embeddings cluster together and mean pairwise
    cosine rises. CADP's hard constraints are hypothesized to resist this.

    Args:
        agent_states: Mapping agent_id → state embedding.

    Returns:
        (mean_cosine, ci_95_halfwidth). Both 0.0 if <2 agents.
    """
    if len(agent_states) < 2:
        return 0.0, 0.0

    vectors = np.stack(list(agent_states.values()))
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms

    # Upper-triangle pairwise cosines (i < j)
    n = normed.shape[0]
    iu = np.triu_indices(n, k=1)
    cosines = (normed @ normed.T)[iu]

    mean = float(np.mean(cosines))
    # 95% CI halfwidth via normal approximation (large-n)
    if cosines.size > 1:
        std = float(np.std(cosines, ddof=1))
        ci_half = 1.96 * std / np.sqrt(cosines.size)
    else:
        ci_half = 0.0
    return mean, ci_half
