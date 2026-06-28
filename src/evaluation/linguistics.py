"""Linguistic evaluation metrics.

Uses feature spaces orthogonal to Expression DNA extraction to avoid
feature leakage (outline §5.3 warning). Three independent metrics:
- Discourse marker distribution match (KL-divergence)
- Sentiment trajectory shape similarity
- Speech act ratio similarity
Plus SIP (Semantic Information Preservation) via Sentence-BERT cosine.
"""

from __future__ import annotations

import re

import numpy as np

from src.data.schemas import Message


# ---------------------------------------------------------------------------
# Discourse markers — orthogonal to Expression DNA lexicons
# (Expression DNA uses: CERTAIN_WORDS, HEDGE_WORDS, TRANSITION_WORDS,
#  FIRST_PERSON_PRONOUNS, ACADEMIC_WORDS, ANALOGY_MARKERS)
# ---------------------------------------------------------------------------
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
# Sentiment lexicon — simple positive/negative word lists
# (No overlap with Expression DNA lexicons by construction)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "love", "happy", "glad", "pleased", "support", "agree", "correct",
    "true", "beneficial", "positive", "useful", "effective", "helpful",
    "valuable", "important", "success", "win", "better", "best",
    "improve", "benefit", "advantage", "appreciate", "commend",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "wrong", "false", "incorrect",
    "harmful", "negative", "useless", "ineffective", "unhelpful",
    "damaging", "problem", "issue", "fail", "worse", "worst", "broken",
    "flawed", "bias", "unfair", "hate", "angry", "sad", "disappointed",
    "frustrated", "concern", "worry", "threat", "risk", "danger",
}

# ---------------------------------------------------------------------------
# Speech act cues — structural surface-form patterns
# ---------------------------------------------------------------------------
_COMMISSIVE_PATTERNS = re.compile(
    r"\b(?:i\s+(?:will|won't|ll|shall|promise|agree|plan|intend|commit|volunteer)|let's|we\s+(?:will|shall))\b",
    re.IGNORECASE,
)
_BASE_VERBS_LEAD = re.compile(
    r"^(?:do|don't|stop|start|begin|let|give|take|make|go|come|try|consider|"
    r"look|check|see|think|wait|remember|note|keep|put|use|ensure|verify|"
    r"fix|add|remove|change|update|create|delete|open|close|read|write)\b",
    re.IGNORECASE,
)


def discourse_marker_distribution(messages: list[Message]) -> dict[str, float]:
    """Compute proportion of each discourse marker category.

    Returns dict with keys: fillers, stallers, evidentials, hedges_interactional.
    Values are fraction of total tokens.
    """
    all_tokens = []
    for msg in messages:
        tokens = re.findall(r"\b\w+\b", msg.text.lower())
        all_tokens.extend(tokens)

    if not all_tokens:
        return {cat: 0.0 for cat in DISCOURSE_MARKERS}

    total = len(all_tokens)
    dist = {}
    for cat, words in DISCOURSE_MARKERS.items():
        count = sum(1 for t in all_tokens if t in words)
        dist[cat] = count / total

    return dist


def discourse_marker_match(
    sim_messages: list[Message],
    real_messages: list[Message],
) -> float:
    """Jensen-Shannon similarity of discourse-marker *composition*.

    Both sets' per-category marker rates are normalised over the
    discourse-marker categories into proper probability distributions, then
    JS-divergence = 0.5·KL(P‖M) + 0.5·KL(Q‖M) (M = (P+Q)/2) is computed by
    SUMMING the per-category contributions. The previous implementation
    averaged half-contributions over un-normalised rates — neither standard
    JS nor scale-comparable — which compressed absolute scores toward 1.

    Returns score in [0, 1]. 1 = identical marker-type composition; 0.0 only
    when exactly one side has no discourse markers at all.
    """
    sim_dist = discourse_marker_distribution(sim_messages)
    real_dist = discourse_marker_distribution(real_messages)
    cats = list(DISCOURSE_MARKERS.keys())

    p = np.array([sim_dist.get(c, 0.0) for c in cats], dtype=float)
    q = np.array([real_dist.get(c, 0.0) for c in cats], dtype=float)
    p_sum = float(p.sum())
    q_sum = float(q.sum())

    # Both marker-free → compositions trivially identical.
    if p_sum <= 0.0 and q_sum <= 0.0:
        return 1.0
    # One marker-free, the other not → no compositional overlap.
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


def _per_message_sentiment(text: str) -> float:
    """Simple lexicon-based sentiment score per message.

    Returns value in [-1, 1].
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    return (pos - neg) / len(tokens)


def sentiment_trajectory_shape(messages: list[Message]) -> dict[str, float]:
    """Compute sentiment trajectory descriptors.

    Returns dict with: variance, trend_slope, oscillation_freq, range.
    """
    if len(messages) < 2:
        return {"variance": 0.0, "trend_slope": 0.0, "oscillation_freq": 0.0, "range": 0.0}

    scores = np.array([_per_message_sentiment(m.text) for m in messages])

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


def speech_act_ratio(messages: list[Message]) -> dict[str, float]:
    """Classify messages into speech act categories.

    Categories: assertive, directive, commissive, expressive.
    Returns proportion of each category.
    """
    if not messages:
        return {"assertive": 0.25, "directive": 0.25, "commissive": 0.25, "expressive": 0.25}

    counts = {"assertive": 0, "directive": 0, "commissive": 0, "expressive": 0}

    for msg in messages:
        text = msg.text.strip()
        if not text:
            counts["assertive"] += 1
            continue

        # Expressive: exclamations
        if text.endswith("!") or text.endswith("!!"):
            counts["expressive"] += 1
            continue

        # Directive: questions or imperatives
        if text.endswith("?"):
            counts["directive"] += 1
            continue

        if _BASE_VERBS_LEAD.match(text):
            counts["directive"] += 1
            continue

        # Commissive: first-person commitment
        if _COMMISSIVE_PATTERNS.search(text):
            counts["commissive"] += 1
            continue

        # Default: assertive
        counts["assertive"] += 1

    total = sum(counts.values())
    return {cat: count / total for cat, count in counts.items()}


def speech_act_similarity(
    sim_messages: list[Message],
    real_messages: list[Message],
) -> float:
    """Similarity of speech act distributions (1 - normalized L1 distance).

    Returns score in [0, 1]. 1 = identical distributions.
    """
    sim_acts = speech_act_ratio(sim_messages)
    real_acts = speech_act_ratio(real_messages)

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
    def compute(
        sim_messages: list[Message],
        real_messages: list[Message],
    ) -> dict[str, float]:
        result = {
            "discourse_marker_match": discourse_marker_match(sim_messages, real_messages),
            "sentiment_trajectory_similarity": sentiment_trajectory_similarity(sim_messages, real_messages),
            "speech_act_similarity": speech_act_similarity(sim_messages, real_messages),
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

    Returns the assertive bias of the message: 0 if the message triggers a
    directive / commissive / expressive cue, 1 if it falls through to the
    assertive default. This mirrors ``speech_act_ratio``'s classification
    priority without expanding to a 4-vector (sufficient for a correlation
    sanity check).
    """
    text_s = text.strip()
    if not text_s:
        return 1.0
    if text_s.endswith("!") or text_s.endswith("!!"):
        return 0.0
    if text_s.endswith("?"):
        return 0.0
    if _BASE_VERBS_LEAD.match(text_s):
        return 0.0
    if _COMMISSIVE_PATTERNS.search(text_s):
        return 0.0
    return 1.0
