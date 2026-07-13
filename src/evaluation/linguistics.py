"""Linguistic evaluation metrics.

Uses feature spaces orthogonal to Expression DNA extraction to avoid
feature leakage (outline §5.3 warning). Three independent metrics:
- Discourse marker distribution match (KL-divergence)
- Sentiment trajectory shape similarity
- Speech act ratio similarity
Plus SIP (Semantic Information Preservation) via Sentence-BERT cosine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.data.schemas import Message

if TYPE_CHECKING:
    from src.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Discourse relation classification via PDTB2 model
# Replaces sparse DISCOURSE_MARKERS lexicon (<0.3% coverage → ≈0.97 always).
# Uses murathankurfali/bert-large-uncased-pdtb2-explicit-four-way
# 4 classes: Comparison, Contingency, Expansion, Temporal
# ---------------------------------------------------------------------------
_PDTB_CATEGORIES = ("Comparison", "Contingency", "Expansion", "Temporal")

_pdtb_pipeline = None


def _get_pdtb_pipeline():
    """Lazy-load the PDTB2 discourse relation classifier."""
    global _pdtb_pipeline
    if _pdtb_pipeline is not None:
        return _pdtb_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _pdtb_pipeline = hf_pipeline(
            "text-classification",
            model="murathankurfali/bert-large-uncased-pdtb2-explicit-four-way",
            device=device,
        )
        logger.info(
            f"PDTB2 discourse relation classifier loaded (device={device})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to load PDTB2 discourse model: {e}. "
            f"Discourse relation metrics will be unavailable."
        )
        _pdtb_pipeline = False  # sentinel: tried and failed
    return _pdtb_pipeline


def discourse_relation_distribution(messages: list[Message]) -> dict[str, float]:
    """Compute proportion of each PDTB2 discourse relation category.

    Uses BERT-large classifier instead of sparse lexicon.
    Returns dict with keys: Comparison, Contingency, Expansion, Temporal.
    Values are fraction of total messages.
    """
    pipe = _get_pdtb_pipeline()
    if pipe is False or pipe is None:
        return {cat: 0.25 for cat in _PDTB_CATEGORIES}

    if not messages:
        return {cat: 0.0 for cat in _PDTB_CATEGORIES}

    texts = [m.text for m in messages if m.text.strip()]
    if not texts:
        return {cat: 0.0 for cat in _PDTB_CATEGORIES}

    try:
        results = pipe(texts, batch_size=64, truncation=True, max_length=512)
    except Exception as e:
        logger.warning(f"PDTB2 batch classification failed: {e}")
        return {cat: 0.25 for cat in _PDTB_CATEGORIES}

    counts = {cat: 0 for cat in _PDTB_CATEGORIES}
    for r in results:
        label = r["label"]
        if label in counts:
            counts[label] += 1
        else:
            # Map lowercase or partial matches
            for cat in _PDTB_CATEGORIES:
                if cat.lower() == label.lower():
                    counts[cat] += 1
                    break

    total = sum(counts.values()) or 1
    return {cat: count / total for cat, count in counts.items()}


# Keep DISCOURSE_MARKERS for orthogonality verification (G9) only.
# Not used for the main discourse_marker_match metric anymore.
DISCOURSE_MARKERS = {
    "fillers": {
        "well", "okay", "right", "basically", "literally",
        "honestly", "seriously", "frankly", "look", "see",
    },
    "stallers": {
        "kind", "sort", "suppose", "guess", "meanwhile",
        "incidentally", "anyway", "anyhow", "regardless",
    },
    "evidentials": {
        "apparently", "evidently", "supposedly", "allegedly",
        "reportedly", "rumored", "claimed", "stated",
    },
    "hedges_interactional": {
        "stuff", "things", "whatever", "somewhere", "somehow",
        "someone", "somewhat", "hereandthere",
    },
}

# ---------------------------------------------------------------------------
# Speech act cues — DELETED: regex patterns removed 2026-07-13.
# Context-blind regex cannot reliably infer speech act category.
# LLM classifier is the primary path; fallback returns "assertive".
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# LLM-based speech act classification (replaces context-blind regex)
# ---------------------------------------------------------------------------
_SPEECH_ACT_CATEGORIES = ("assertive", "directive", "commissive", "expressive")

# Mapping from quantor-project/speech-act-classification 8-class labels
# to our 4-class taxonomy:
#   <com> = commissive, <dec> = declarative→assertive, <dir> = directive,
#   <exp> = expressive, <icu> = indirect request→directive,
#   <rep> = representative→assertive, <soc> = social→expressive,
#   <xpa> = expressive-assertive→expressive
_QUANTOR_TO_FOUR: dict[str, str] = {
    "<com>": "commissive",
    "<dec>": "assertive",
    "<dir>": "directive",
    "<exp>": "expressive",
    "<icu>": "directive",
    "<rep>": "assertive",
    "<soc>": "expressive",
    "<xpa>": "expressive",
}

_SPEECH_ACT_PROMPT = """Classify each message into exactly one speech act category.

Categories:
- assertive: statements of fact, belief, or description (e.g. "The data shows X", "I think that")
- directive: requests, commands, questions (e.g. "Please review", "Can you check?", "Fix the bug")
- commissive: commitments to future action (e.g. "I will fix it", "We plan to release tomorrow")
- expressive: emotional reactions, greetings, thanks (e.g. "Great job!", "Thanks!", "Ugh")

Messages:
{messages_json}

Respond with a JSON array of category strings, one per message, in order.
Example: ["assertive", "directive", "commissive", "expressive"]"""


class LLMSpeechActClassifier:
    """Optional LLM-based speech act classification (upgrade over local model).

    Use when higher accuracy justifies API cost. The default path uses
    a local RoBERTa model (quantor-project/speech-act-classification)
    which is fast, free, and sufficient for most evaluation needs.
    """

    def __init__(self, llm_client: LLMClient, model_name: str, batch_size: int = 20):
        self._client = llm_client
        self._model = model_name
        self._batch_size = batch_size
        self._cache: dict[str, str] = {}

    async def classify_batch(self, messages: list[Message]) -> list[str]:
        """Classify messages into speech act categories.

        Returns list of category strings in same order as input.
        Uses LLM batch calls (batch_size msgs/call), falls back to
        local RoBERTa classifier on failure.
        """
        results: list[str | None] = [None] * len(messages)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, msg in enumerate(messages):
            key = hashlib.sha256(msg.text.encode()).hexdigest()
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(msg.text)

        # Batch-classify uncached messages
        for start in range(0, len(uncached_texts), self._batch_size):
            batch_slice = slice(start, start + self._batch_size)
            batch_indices = uncached_indices[batch_slice]
            batch_texts = uncached_texts[batch_slice]
            categories = await self._llm_classify(batch_texts)

            for idx, text, cat in zip(batch_indices, batch_texts, categories):
                results[idx] = cat
                key = hashlib.sha256(text.encode()).hexdigest()
                self._cache[key] = cat

        # Fill any remaining None with local RoBERTa fallback
        for i in range(len(results)):
            if results[i] is None:
                results[i] = _speech_act_local(messages[i].text)

        return results  # type: ignore[return-value]

    async def _llm_classify(self, texts: list[str]) -> list[str]:
        """Call LLM to classify a batch of texts."""
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = _SPEECH_ACT_PROMPT.format(messages_json=numbered)

        try:
            result = await self._client.chat_completion_json(
                messages=[{"role": "user", "content": prompt}],
                model_name=self._model,
                temperature=0.0,
                max_tokens=1024,
                default=None,
            )
            if isinstance(result, list) and len(result) == len(texts):
                valid = [r for r in result if r in _SPEECH_ACT_CATEGORIES]
                if len(valid) == len(texts):
                    return valid
        except Exception:
            pass

        # Fallback: local RoBERTa classifier
        return [_speech_act_local(t) for t in texts]


# ---------------------------------------------------------------------------
# Local RoBERTa speech act classifier (fallback when LLM unavailable)
# Uses quantor-project/speech-act-classification (8-class RoBERTa-base)
# mapped to our 4-class taxonomy.
# ---------------------------------------------------------------------------

_local_speech_act_pipeline = None


def _get_local_speech_act_pipeline():
    """Lazy-load the local speech act classification pipeline."""
    global _local_speech_act_pipeline
    if _local_speech_act_pipeline is not None:
        return _local_speech_act_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _local_speech_act_pipeline = hf_pipeline(
            "text-classification",
            model="quantor-project/speech-act-classification",
            device=device,
        )
        logger.info(
            f"Local speech act classifier loaded "
            f"(quantor-project/speech-act-classification, device={device})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to load local speech act model: {e}. "
            f"Will return 'assertive' as neutral default."
        )
        _local_speech_act_pipeline = False  # sentinel: tried and failed
    return _local_speech_act_pipeline


def _speech_act_local(text: str) -> str:
    """Local RoBERTa speech act classification.

    Falls back to "assertive" only if model load fails entirely.
    """
    pipe = _get_local_speech_act_pipeline()
    if pipe is False or pipe is None:
        return "assertive"
    try:
        result = pipe(text, truncation=True, max_length=512)
        label = result[0]["label"]
        return _QUANTOR_TO_FOUR.get(label, "assertive")
    except Exception:
        return "assertive"


def _speech_act_fallback(text: str) -> str:
    """Deprecated: use _speech_act_local instead."""
    return _speech_act_local(text)

def discourse_marker_distribution(messages: list[Message]) -> dict[str, float]:
    """Compute proportion of each PDTB2 discourse relation category.

    Delegates to discourse_relation_distribution (PDTB2 classifier).
    Kept for backward compatibility with orthogonality verifier (G9).
    """
    return discourse_relation_distribution(messages)


def discourse_relation_match(
    sim_messages: list[Message],
    real_messages: list[Message],
) -> float:
    """Jensen-Shannon similarity of discourse *relation* composition.

    Uses PDTB2 BERT-large classifier instead of sparse lexicon.
    Both sets' per-category relation rates are normalised into proper
    probability distributions, then JS-divergence is computed.

    Returns score in [0, 1]. 1 = identical relation composition; 0.0 only
    when exactly one side has no classified relations at all.
    """
    sim_dist = discourse_relation_distribution(sim_messages)
    real_dist = discourse_relation_distribution(real_messages)
    cats = list(_PDTB_CATEGORIES)

    p = np.array([sim_dist.get(c, 0.0) for c in cats], dtype=float)
    q = np.array([real_dist.get(c, 0.0) for c in cats], dtype=float)
    p_sum = float(p.sum())
    q_sum = float(q.sum())

    # Both zero → compositions trivially identical.
    if p_sum <= 0.0 and q_sum <= 0.0:
        return 1.0
    # One zero, the other not → no compositional overlap.
    if p_sum <= 0.0 or q_sum <= 0.0:
        return 0.0

    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    js = max(js, 0.0)  # guard tiny-negative float error
    return float(1.0 / (1.0 + js))


# Backward-compatible alias
discourse_marker_match = discourse_relation_match


# ---------------------------------------------------------------------------
# RoBERTa-based sentiment classification (replaces lexicon)
# Uses cardiffnlp/twitter-roberta-base-sentiment-latest for contextual
# scoring with negation awareness. Falls back to VADER if model load fails.
# ---------------------------------------------------------------------------

class RoBERTaSentimentClassifier:
    """Batch sentiment scoring via cardiffnlp RoBERTa, with VADER fallback."""

    def __init__(self, device: int | None = None):
        self._pipeline = None
        self._vader = None
        self._device = device
        self._cache: dict[str, float] = {}

    def _ensure_pipeline(self):
        """Lazy-load the RoBERTa pipeline on GPU."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device = self._device
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
            )
            logger.info(
                f"RoBERTaSentimentClassifier loaded on device={device}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load RoBERTa sentiment model: {e}. "
                f"Falling back to VADER."
            )
            self._pipeline = None
            self._ensure_vader()

    def _ensure_vader(self):
        """Lazy-load VADER as fallback."""
        if self._vader is not None:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment fallback loaded")
        except ImportError:
            logger.warning("VADER not available; sentiment will return 0.0")
            self._vader = None

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a batch of texts, returning values in [-1, 1].

        Uses RoBERTa pipeline with GPU batching if available,
        else VADER, else returns 0.0 for all.
        """
        self._ensure_pipeline()
        results: list[float] = [0.0] * len(texts)

        # Check cache
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results

        if self._pipeline is not None:
            try:
                batch_results = self._pipeline(
                    uncached_texts, batch_size=64, truncation=True, max_length=512
                )
                for idx, text, r in zip(uncached_indices, uncached_texts, batch_results):
                    label = r["label"]
                    prob = r["score"]
                    # Map 3-class to continuous [-1, +1]:
                    # positive → +prob, negative → -prob, neutral → 0
                    if label == "positive":
                        score = prob
                    elif label == "negative":
                        score = -prob
                    else:  # neutral
                        score = 0.0
                    results[idx] = score
                    key = hashlib.sha256(text.encode()).hexdigest()
                    self._cache[key] = score
            except Exception as e:
                logger.warning(
                    f"RoBERTa batch failed ({type(e).__name__}), "
                    f"falling back to VADER for {len(uncached_texts)} texts"
                )
                self._pipeline = None  # don't retry neural; go straight to VADER
                self._ensure_vader()
                for idx, text in zip(uncached_indices, uncached_texts):
                    scores = self._vader.polarity_scores(text)
                    results[idx] = scores["compound"]
                    key = hashlib.sha256(text.encode()).hexdigest()
                    self._cache[key] = scores["compound"]
        elif self._vader is not None:
            for idx, text in zip(uncached_indices, uncached_texts):
                scores = self._vader.polarity_scores(text)
                results[idx] = scores["compound"]
                key = hashlib.sha256(text.encode()).hexdigest()
                self._cache[key] = scores["compound"]
        else:
            # No classifier available — return 0.0 (neutral)
            for idx, text in zip(uncached_indices, uncached_texts):
                key = hashlib.sha256(text.encode()).hexdigest()
                self._cache[key] = 0.0

        return results

    def score(self, text: str) -> float:
        """Score a single text in [-1, 1]."""
        return self.score_batch([text])[0]


# Singleton classifier — loaded once, reused across calls
_sentiment_classifier: RoBERTaSentimentClassifier | None = None


def get_sentiment_classifier(device: int | None = None) -> RoBERTaSentimentClassifier:
    """Get or create the global sentiment classifier."""
    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = RoBERTaSentimentClassifier(device=device)
    return _sentiment_classifier


def _per_message_sentiment(text: str) -> float:
    """Context-aware sentiment score per message.

    Returns value in [-1, 1]. Uses RoBERTa (GPU) by default,
    falls back to VADER if model load fails.
    """
    if not text.strip():
        return 0.0
    classifier = get_sentiment_classifier()
    return classifier.score(text)


def sentiment_trajectory_shape(messages: list[Message]) -> dict[str, float]:
    """Compute sentiment trajectory descriptors.

    Returns dict with: variance, trend_slope, oscillation_freq, range.
    """
    if len(messages) < 2:
        return {"variance": 0.0, "trend_slope": 0.0, "oscillation_freq": 0.0, "range": 0.0}

    # Batch score for efficiency (GPU batching)
    texts = [m.text for m in messages]
    classifier = get_sentiment_classifier()
    scores = np.array(classifier.score_batch(texts))

    variance = float(np.var(scores))

    # Linear regression slope
    x = np.arange(len(scores), dtype=float)
    trend_slope = float(np.polyfit(x, scores, 1)[0]) if len(scores) > 1 else 0.0

    # Oscillation frequency: fraction of direction changes
    if len(scores) > 2:
        diffs = np.diff(scores)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        oscillation_freq = float(sign_changes / (len(scores) - 2))
    else:
        oscillation_freq = 0.0

    score_range = float(np.max(scores) - np.min(scores))

    return {
        "variance": variance,
        "trend_slope": trend_slope,
        "oscillation_freq": oscillation_freq,
        "range": score_range,
    }


def sentiment_trajectory_similarity(
    sim_messages: list[Message],
    real_messages: list[Message],
) -> float:
    """Cosine similarity between sentiment trajectory descriptor vectors.

    Returns score in [0, 1].
    """
    sim_shape = sentiment_trajectory_shape(sim_messages)
    real_shape = sentiment_trajectory_shape(real_messages)

    keys = list(sim_shape.keys())
    sim_vec = np.array([sim_shape[k] for k in keys])
    real_vec = np.array([real_shape[k] for k in keys])

    norm = np.linalg.norm(sim_vec) * np.linalg.norm(real_vec)
    if norm < 1e-10:
        return 0.0

    cosine = float(np.dot(sim_vec, real_vec) / norm)
    # Clamp to [0, 1] since cosine can be negative
    return max(0.0, (cosine + 1.0) / 2.0)


async def speech_act_ratio(
    messages: list[Message],
    classifier: LLMSpeechActClassifier | None = None,
) -> dict[str, float]:
    """Classify messages into speech act categories.

    Categories: assertive, directive, commissive, expressive.
    Returns proportion of each category.
    Primary: local RoBERTa model (fast, free, no API).
    Optional: LLM classifier for higher quality (slower, costs API calls).
    """
    if not messages:
        return {"assertive": 0.25, "directive": 0.25, "commissive": 0.25, "expressive": 0.25}

    counts = {"assertive": 0, "directive": 0, "commissive": 0, "expressive": 0}

    if classifier is not None:
        # LLM path: higher quality, slower, costs API calls
        categories = await classifier.classify_batch(messages)
        for cat in categories:
            counts[cat] += 1
    else:
        # Primary path: local RoBERTa model (fast, free)
        # Batch classify for efficiency
        pipe = _get_local_speech_act_pipeline()
        if pipe is not False and pipe is not None:
            texts = [m.text for m in messages]
            try:
                results = pipe(texts, batch_size=64, truncation=True, max_length=512)
                for r in results:
                    cat = _QUANTOR_TO_FOUR.get(r["label"], "assertive")
                    counts[cat] += 1
            except Exception:
                for msg in messages:
                    cat = _speech_act_local(msg.text)
                    counts[cat] += 1
        else:
            for msg in messages:
                cat = _speech_act_local(msg.text)
                counts[cat] += 1

    total = sum(counts.values())
    return {cat: count / total for cat, count in counts.items()}


async def speech_act_similarity(
    sim_messages: list[Message],
    real_messages: list[Message],
    classifier: LLMSpeechActClassifier | None = None,
) -> float:
    """Similarity of speech act distributions (1 - normalized L1 distance).

    Returns score in [0, 1]. 1 = identical distributions.
    """
    sim_acts = await speech_act_ratio(sim_messages, classifier=classifier)
    real_acts = await speech_act_ratio(real_messages, classifier=classifier)

    cats = set(sim_acts) | set(real_acts)
    l1 = sum(abs(sim_acts.get(c, 0.0) - real_acts.get(c, 0.0)) for c in cats)

    # Normalize: L1 distance ranges [0, 2]
    return float(1.0 - l1 / 2.0)


def _stratified_subsample(
    texts: list[str],
    budget: int,
    seed: int = 42,
) -> list[str]:
    """Length-stratified subsample of ``texts`` to at most ``budget`` items.

    M2 fix: the previous SIP implementation took ``texts[:100]`` (head
    truncation), which biased the score toward whatever messages happened
    to come first in the source list — typically short opening posts on
    long Reddit/Wikipedia threads, systematically favouring short-output
    conditions (e.g. vanilla baselines whose agents produce terse replies).

    Stratified-by-length sampling draws evenly across message-length
    quantiles so the subsample preserves the long-tail length distribution
    of the full corpus. When ``len(texts) <= budget`` the input is
    returned unchanged (in original order).
    """
    if len(texts) <= budget:
        return list(texts)

    rng = np.random.default_rng(seed)
    lengths = np.array([len(t) for t in texts])
    # 5 length strata (quintiles by char length); fall back to fewer if
    # there isn't enough data to populate all bins.
    n_strata = min(5, budget, len(texts))
    try:
        bins = np.quantile(lengths, np.linspace(0, 1, n_strata + 1))
    except ValueError:
        bins = np.array([lengths.min(), lengths.max() + 1])
    # np.digitize assigns each text to a bin (1..n_strata); subtract 1.
    bin_idx = np.digitize(lengths, bins[1:-1], right=False)

    per_bin = max(1, budget // n_strata)
    selected_idx: list[int] = []
    for b in range(n_strata):
        members = np.where(bin_idx == b)[0]
        if members.size == 0:
            continue
        take = min(per_bin, members.size)
        picked = rng.choice(members, size=take, replace=False)
        selected_idx.extend(int(i) for i in picked)

    # If bins were uneven and we undershot, fill the remainder uniformly.
    if len(selected_idx) < budget:
        remaining = np.setdiff1d(np.arange(len(texts)), np.array(selected_idx))
        extra = min(budget - len(selected_idx), remaining.size)
        if extra > 0:
            picked = rng.choice(remaining, size=extra, replace=False)
            selected_idx.extend(int(i) for i in picked)

    # Preserve original chronological order so downstream temporal
    # assumptions (if any) still hold.
    selected_idx = sorted(selected_idx)
    return [texts[i] for i in selected_idx]


def semantic_information_preservation(
    sim_messages: list[Message],
    real_messages: list[Message],
    budget: int = 100,
    seed: int = 42,
) -> float:
    """Compute SIP via Sentence-BERT cosine similarity.

    Measures semantic similarity between simulated and real message distributions.

    Anti-circularity note (outline §3.1 / §5.4): this metric deliberately uses
    a SEPARATE encoder (``get_sip_embedder`` → ``paraphrase-mpnet-base-v2`` by
    default) from the one CADP actively constrains text toward via Tier 1 /
    Tier 3 / clustering / Expression DNA distillation (``get_shared_embedder``
    → ``all-MiniLM-L6-v2``). Using the shared encoder here would make the
    minus-Expression-DNA dissociation claim partly circular, since CADP would
    be steering text toward the very space SIP scores against.

    Sampling note (M2 fix): when the corpus exceeds ``budget`` messages
    per side, a length-stratified subsample is drawn (see
    :func:`_stratified_subsample`). The previous ``texts[:100]`` head
    truncation systematically biased SIP toward short-output conditions
    on long Reddit/Wikipedia threads.

    Returns:
        SIP score in [0, 1]. 1 = semantically identical, 0 = unrelated.
    """
    if not sim_messages or not real_messages:
        return 0.0

    from src.config.settings import get_sip_embedder

    model = get_sip_embedder()

    sim_texts_full = [m.text for m in sim_messages if m.text.strip()]
    real_texts_full = [m.text for m in real_messages if m.text.strip()]

    if not sim_texts_full or not real_texts_full:
        return 0.0

    sim_texts = _stratified_subsample(sim_texts_full, budget=budget, seed=seed)
    real_texts = _stratified_subsample(real_texts_full, budget=budget, seed=seed)

    # Single batched encode for both sides (GPU efficiency)
    all_texts = sim_texts + real_texts
    all_embeddings = model.encode(all_texts, show_progress_bar=False)
    sim_embeddings = all_embeddings[: len(sim_texts)]
    real_embeddings = all_embeddings[len(sim_texts) :]

    # Compute mean cosine similarity between distributions
    sim_centroid = sim_embeddings.mean(axis=0)
    real_centroid = real_embeddings.mean(axis=0)

    cosine = np.dot(sim_centroid, real_centroid) / (
        np.linalg.norm(sim_centroid) * np.linalg.norm(real_centroid) + 1e-10
    )
    return float(cosine)


class LinguisticMetrics:
    """Container for all linguistic metrics.

    Uses feature spaces orthogonal to Expression DNA extraction
    (outline §5.3 anti-leakage requirement).
    """

    @staticmethod
    async def compute(
        sim_messages: list[Message],
        real_messages: list[Message],
        classifier: LLMSpeechActClassifier | None = None,
    ) -> dict[str, float]:
        result = {
            "discourse_relation_match": discourse_relation_match(sim_messages, real_messages),
            "sentiment_trajectory_similarity": sentiment_trajectory_similarity(sim_messages, real_messages),
            "speech_act_similarity": await speech_act_similarity(sim_messages, real_messages, classifier=classifier),
        }
        try:
            result["sip"] = semantic_information_preservation(sim_messages, real_messages)
        except Exception:
            result["sip"] = 0.0
        return result


# ---------------------------------------------------------------------------
# Orthogonality verification (outline §5.3 anti-leakage note, G9)
# ---------------------------------------------------------------------------

def verify_orthogonality_to_expression_dna(
    messages: list[Message],
) -> dict[str, float]:
    """Empirically verify Linguistics eval features are orthogonal to
    Expression DNA distillation features (outline §5.3 leakage warning).

    The anti-leakage argument is currently by-construction (lexicons are
    disjoint by inspection). This function turns it into an empirical
    number the paper can cite: for each Linguistics eval feature, compute
    the absolute Pearson correlation with each Expression DNA lexicon
    hit-rate on the same message corpus, and report the max absolute
    correlation across all (Linguistics feature × Expression DNA feature)
    pairs. A low max correlation (rule of thumb: < 0.3) supports the
    orthogonality claim; a high one flags a leakage risk.

    Linguistics features (per-message, normalised to [0,1]):
      - filler_rate, staller_rate, evidential_rate, hedge_interactional_rate
        (discourse-marker categories)
      - sentiment_score (lexicon pos-minus-neg / token count, shifted to [0,1])
      - speech_act_indicator — 4 booleans one-hot aggregated into a single
        assertive-rate proxy

    Expression DNA features (per-message, normalised to [0,1]):
      - certain_rate, hedge_rate, transition_rate, first_person_rate,
        academic_rate, analogy_rate

    Returns:
        Dict with ``max_abs_correlation``, ``per_pair_correlations`` (a
        flat dict ``"ling_feat__edna_feat": corr``), and ``leakage_risk``
        (bool: True iff max_abs_correlation >= 0.3).
    """
    from src.skill.expression_dna import (
        ACADEMIC_WORDS,
        ANALOGY_MARKERS,
        CERTAIN_WORDS,
        FIRST_PERSON_PRONOUNS,
        HEDGE_WORDS,
        TRANSITION_WORDS,
    )

    if not messages:
        return {
            "max_abs_correlation": 0.0,
            "per_pair_correlations": {},
            "leakage_risk": False,
            "n_messages": 0,
        }

    # Collect per-message feature vectors
    ling_rows: list[dict[str, float]] = []
    edna_rows: list[dict[str, float]] = []
    for msg in messages:
        tokens = re.findall(r"\b\w+\b", msg.text.lower())
        if not tokens:
            continue
        n = len(tokens)
        joined = msg.text.lower()

        # Linguistics features
        ling_rows.append({
            "filler_rate": sum(1 for t in tokens if t in DISCOURSE_MARKERS["fillers"]) / n,
            "staller_rate": sum(1 for t in tokens if t in DISCOURSE_MARKERS["stallers"]) / n,
            "evidential_rate": sum(1 for t in tokens if t in DISCOURSE_MARKERS["evidentials"]) / n,
            "hedge_interactional_rate": sum(1 for t in tokens if t in DISCOURSE_MARKERS["hedges_interactional"]) / n,
            "sentiment_score": (_per_message_sentiment(msg.text) + 1.0) / 2.0,
            "speech_act_assertive_rate": _speech_act_assertive_proxy(msg.text),
        })

        # Expression DNA features (mirror ExpressionDNAExtractor lexicons)
        analogy_hits = sum(joined.count(m) for m in ANALOGY_MARKERS)
        edna_rows.append({
            "certain_rate": sum(1 for t in tokens if t in CERTAIN_WORDS) / n,
            "hedge_rate": sum(1 for t in tokens if t in HEDGE_WORDS) / n,
            "transition_rate": sum(
                len(re.findall(r"\b" + re.escape(m) + r"\b", joined))
                for m in TRANSITION_WORDS
            ) / n,
            "first_person_rate": sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS) / n,
            "academic_rate": sum(1 for t in tokens if t in ACADEMIC_WORDS) / n,
            "analogy_rate": analogy_hits / (n + 1e-9),
        })

    if len(ling_rows) < 3:
        return {
            "max_abs_correlation": 0.0,
            "per_pair_correlations": {},
            "leakage_risk": False,
            "n_messages": len(ling_rows),
        }

    ling_arr = np.array(
        [[row[k] for k in row] for row in ling_rows], dtype=float
    )  # (n_msg, n_ling_feats)
    edna_arr = np.array(
        [[row[k] for k in row] for row in edna_rows], dtype=float
    )  # (n_msg, n_edna_feats)

    ling_keys = list(ling_rows[0].keys())
    edna_keys = list(edna_rows[0].keys())

    # Pearson correlation matrix (ling_feats × edna_feats)
    correlations: dict[str, float] = {}
    max_abs = 0.0
    for i, lk in enumerate(ling_keys):
        x = ling_arr[:, i]
        x_std = x.std()
        if x_std < 1e-10:
            # Degenerate (constant) — skip; treat as no correlation evidence
            for ek in edna_keys:
                correlations[f"{lk}__{ek}"] = 0.0
            continue
        for j, ek in enumerate(edna_keys):
            y = edna_arr[:, j]
            y_std = y.std()
            if y_std < 1e-10:
                correlations[f"{lk}__{ek}"] = 0.0
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            correlations[f"{lk}__{ek}"] = corr
            if abs(corr) > max_abs:
                max_abs = abs(corr)

    return {
        "max_abs_correlation": max_abs,
        "per_pair_correlations": correlations,
        "leakage_risk": bool(max_abs >= 0.3),
        "n_messages": len(ling_rows),
    }


def _speech_act_assertive_proxy(text: str) -> float:
    """Single-value proxy for speech-act distribution used in orthogonality check.

    Returns the assertive bias of the message: punctuation-only heuristic
    (structural, not regex-based). 0 if the message ends with ? or !
    (likely directive/expressive), 1 otherwise (assertive default).
    """
    text_s = text.strip()
    if not text_s:
        return 1.0
    if text_s.endswith("!") or text_s.endswith("?"):
        return 0.0
    return 1.0
