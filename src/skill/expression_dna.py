"""Expression DNA extraction — quantitative language pattern analysis.

Adapts nuwa-skill extraction-framework.md §2:
- Sentence-pattern fingerprint (statistical measurement)
- Style tags (7 axes)
- Taboo words and high-frequency phrases
- Vocabulary richness
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict

import numpy as np

from src.data.schemas import Message
from src.skill.schema import ExpressionDNA


# Lexicons for style measurement
CERTAIN_WORDS = {
    "certainly", "obviously", "clearly", "definitely", "absolutely",
    "undoubtedly", "surely", "must", "always", "never", "fact",
}
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seem",
    "appear", "likely", "probably", "somewhat", "arguably",
}
TRANSITION_WORDS = {
    "but", "however", "nevertheless", "nonetheless", "yet",
    "although", "though", "whereas", "while", "on the other hand",
    "conversely", "in contrast", "despite", "in spite of",
}
FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
ACADEMIC_WORDS = {
    "furthermore", "moreover", "consequently", "nevertheless",
    "hypothesis", "empirical", "methodology", "theoretical",
    "analysis", "systematic", "comprehensive", "significant",
}
ANALOGY_MARKERS = {
    "like", "as if", "similar to", "analogous", "comparable",
    "in the same way", "much like", "akin to",
}


class ExpressionDNAExtractor:
    """Extracts ExpressionDNA from a cluster's message corpus."""

    def extract(
        self,
        messages: list[Message],
        embeddings: np.ndarray | None = None,
        other_cluster_word_freq: Counter | None = None,
    ) -> ExpressionDNA:
        """Extract Expression DNA from messages.

        Args:
            messages: All messages from the cluster.
            embeddings: Optional pre-computed message embeddings for centroid.
            other_cluster_word_freq: Word frequency from other clusters,
                used to detect taboo words (common elsewhere but rare here).

        Returns:
            ExpressionDNA with quantified patterns.
        """
        texts = [m.text for m in messages if m.text.strip()]
        if not texts:
            return ExpressionDNA()

        # Sentence-pattern fingerprint
        fingerprint = self._compute_fingerprint(texts)

        # Style tags
        style = self._compute_style_tags(texts)

        # Vocabulary
        vocab = self._compute_vocabulary(texts, other_cluster_word_freq)

        # Embedding centroid + cosine-distance threshold (95th percentile of
        # 1 − cos(u, centroid) on the held-out half of the cluster utterances).
        emb_centroid = None
        emb_cosine_threshold = None
        if embeddings is not None and len(embeddings) > 0:
            embs = np.asarray(embeddings, dtype=float)
            centroid = embs.mean(axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            holdout = embs[len(embs) // 2:]
            holdout_norm = holdout / (
                np.linalg.norm(holdout, axis=1, keepdims=True) + 1e-10
            )
            dists = 1.0 - np.dot(holdout_norm, centroid_norm)
            emb_centroid = centroid.tolist()
            emb_cosine_threshold = float(np.percentile(dists, 95))

        return ExpressionDNA(
            **fingerprint,
            **style,
            **vocab,
            embedding_centroid=emb_centroid,
            embedding_cosine_threshold=emb_cosine_threshold,
        )

    def _compute_fingerprint(self, texts: list[str]) -> dict:
        all_sentences = []
        total_chars = 0
        for text in texts:
            sentences = self._split_sentences(text)
            all_sentences.extend(sentences)
            total_chars += len(text)

        total_sentences = len(all_sentences)
        total_words = sum(len(s.split()) for s in all_sentences)

        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0.0
        question_count = sum(1 for s in all_sentences if s.strip().endswith("?"))
        question_ratio = question_count / total_sentences if total_sentences > 0 else 0.0

        # Analogy density per 1000 chars
        analogy_count = 0
        for text in texts:
            lower = text.lower()
            for marker in ANALOGY_MARKERS:
                analogy_count += lower.count(marker)
        analogy_density = analogy_count / (total_chars / 1000) if total_chars > 0 else 0.0

        # First-person rate
        first_person_count = 0
        total_tokens = 0
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            total_tokens += len(tokens)
            first_person_count += sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS)
        first_person_rate = first_person_count / total_tokens if total_tokens > 0 else 0.0

        # Certainty ratio
        certain_count = 0
        hedge_count = 0
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            certain_count += sum(1 for t in tokens if t in CERTAIN_WORDS)
            hedge_count += sum(1 for t in tokens if t in HEDGE_WORDS)
        certainty_total = certain_count + hedge_count
        certainty_ratio = certain_count / certainty_total if certainty_total > 0 else 0.5

        # Transition frequency per 1000 chars
        transition_count = 0
        for text in texts:
            lower = text.lower()
            for marker in TRANSITION_WORDS:
                transition_count += len(re.findall(r"\b" + re.escape(marker) + r"\b", lower))
        transition_frequency = transition_count / (total_chars / 1000) if total_chars > 0 else 0.0

        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "question_ratio": round(question_ratio, 4),
            "analogy_density": round(analogy_density, 4),
            "first_person_rate": round(first_person_rate, 4),
            "certainty_ratio": round(certainty_ratio, 4),
            "transition_frequency": round(transition_frequency, 4),
        }

    def _compute_style_tags(self, texts: list[str]) -> dict:
        """Compute 7-axis style tags (each 0-1)."""
        all_tokens = []
        for text in texts:
            all_tokens.extend(re.findall(r"\b\w+\b", text.lower()))
        total_tokens = len(all_tokens) if all_tokens else 1

        academic_count = sum(1 for t in all_tokens if t in ACADEMIC_WORDS)
        academic_ratio = academic_count / total_tokens

        avg_sent_len = float(np.mean([
            len(s.split()) for text in texts
            for s in self._split_sentences(text)
        ])) if texts else 0.0

        certain_count = sum(1 for t in all_tokens if t in CERTAIN_WORDS)
        hedge_count = sum(1 for t in all_tokens if t in HEDGE_WORDS)
        assertive_ratio = certain_count / (certain_count + hedge_count + 1)

        # Normalize to 0-1 scales
        return {
            "style_formal_casual": round(min(max(1.0 - academic_ratio * 10, 0), 1), 3),
            "style_abstract_concrete": round(min(avg_sent_len / 30, 1), 3),
            "style_cautious_assertive": round(assertive_ratio, 3),
            "style_academic_plain": round(min(academic_ratio * 10, 1), 3),
            "style_long_short": round(min(1.0 - avg_sent_len / 40, 1), 3) if avg_sent_len <= 40 else 0.0,
            "style_preamble_conclusion_first": self._compute_conclusion_first(texts),
            "style_data_narrative": self._compute_data_ratio(all_tokens),
        }

    def _compute_conclusion_first(self, texts: list[str]) -> float:
        """Score how conclusion-first the writing style is (0=preamble-first, 1=conclusion-first).

        Measures whether first sentences tend to be shorter and more direct than later sentences.
        Short first sentences → conclusion-first (high score).
        Long first sentences → preamble-first (low score).
        """
        first_lens = []
        rest_lens = []
        for text in texts:
            sentences = self._split_sentences(text)
            if len(sentences) >= 2:
                first_lens.append(len(sentences[0].split()))
                rest_lens.append(sum(len(s.split()) for s in sentences[1:]) / len(sentences[1:]))

        if not first_lens:
            return 0.5

        avg_first = sum(first_lens) / len(first_lens)
        avg_rest = sum(rest_lens) / len(rest_lens)

        if avg_rest <= 0:
            return 0.5

        ratio = avg_first / avg_rest
        # ratio < 1 means first sentences are shorter (conclusion-first)
        # ratio > 1 means first sentences are longer (preamble-first)
        # Map ratio [0, 2] → score [1, 0]
        score = round(max(0.0, min(1.0, 1.0 - (ratio - 0.5) / 1.5)), 3)
        return score

    def _compute_data_ratio(self, tokens: list[str]) -> float:
        """Score how data-driven vs narrative the language is (0=narrative, 1=data-driven).

        Counts data words: numbers, percentages, and units.
        """
        if not tokens:
            return 0.5

        data_pattern = re.compile(r"^\d+([.,]\d+)?%?$|^\d+\w*$")
        data_count = sum(1 for t in tokens if data_pattern.match(t))
        ratio = data_count / len(tokens)
        # Normalize: typical data-heavy text has ~5-15% data words
        return round(min(ratio / 0.15, 1.0), 3)

    def _compute_vocabulary(self, texts: list[str], other_cluster_word_freq: Counter | None = None) -> dict:
        """Compute vocabulary richness, frequent words, and taboo words.

        Taboo words are words common in other clusters but rare in this cluster.
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(re.findall(r"\b\w+\b", text.lower()))

        if not all_tokens:
            return {"high_freq_words": [], "taboo_words": [], "vocab_richness": 0.0}

        # Vocabulary richness (type-token ratio)
        unique = set(all_tokens)
        ttr = len(unique) / len(all_tokens)

        # High-frequency content words (excluding stop words)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "this", "that", "these", "those", "it",
            "its", "they", "them", "their", "there", "here", "as",
            "if", "so", "not", "no", "yes",
        }
        content_words = [t for t in all_tokens if t not in stop_words and len(t) > 2]
        word_freq = Counter(content_words)
        high_freq = [w for w, _ in word_freq.most_common(30)]

        # Taboo words: common in other clusters (>1% of other tokens) but rare here (<0.1%)
        taboo_words = []
        if other_cluster_word_freq:
            other_total = sum(other_cluster_word_freq.values()) or 1
            this_total = sum(word_freq.values()) or 1
            for word, other_count in other_cluster_word_freq.most_common(200):
                if word in stop_words:
                    continue
                other_ratio = other_count / other_total
                this_ratio = word_freq.get(word, 0) / this_total
                if other_ratio > 0.01 and this_ratio < 0.001:
                    taboo_words.append(word)
            taboo_words = taboo_words[:20]

        return {
            "high_freq_words": high_freq,
            "taboo_words": taboo_words,
            "vocab_richness": round(ttr, 4),
        }

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]
