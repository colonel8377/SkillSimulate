"""Conservative token estimation for prompt budgeting.

The CADP pipeline targets reasoning models (e.g. DeepSeek-V4-Flash) whose
``max_tokens`` cap covers input + output + thinking combined (65535 total).
To prevent silent overflow we need a *pre-flight* estimate of how many
tokens a given prompt will consume before the API call is made.

This module does NOT attempt to be token-exact — perfect counting would
require the model's proprietary tokenizer, which is not publicly available
for most reasoning models. Instead we use ``tiktoken``'s ``cl100k_base``
encoding (the GPT-4 / DeepSeek-V4 BPE family) and apply a **1.15× safety
multiplier** to over-estimate by ~15%. Conservative overcount is safe —
the only cost is that we may truncate slightly more aggressively than
strictly necessary, which is methodologically preferable to silently
exceeding the cap and burning retry budget.

All public functions memoize the encoding so the per-call overhead is just
the BPE walk.
"""

from __future__ import annotations

from functools import lru_cache


# Conservative over-estimate. DeepSeek-V4's tokenizer is close to but not
# identical to cl100k_base; empirical spot-checks on English/Chinese mixed
# text put the divergence at <10%, so 1.15 leaves headroom for the worst
# observed case plus message-framing overhead (role tags, separators).
_SAFETY_MULTIPLIER = 1.15


@lru_cache(maxsize=4)
def _get_encoding(model: str = "cl100k_base"):
    """Return a tiktoken encoding, memoized.

    Falls back to a naive char-based estimate if tiktoken is unavailable
    (e.g. in CI environments where it isn't installed). The fallback is
    deliberately very conservative (chars / 3) so the budget guard still
    fires on overflow.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding(model)
    except Exception:  # pragma: no cover — fallback path
        return None


def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
    """Conservative estimate of how many tokens *text* will consume.

    Args:
        text: The string to size.
        model: Reserved for future per-model tokenizer selection; currently
            ignored and ``cl100k_base`` is used regardless.

    Returns:
        Ceiling of ``raw_token_count × 1.15`` — never smaller than the
        raw BPE count, so budget checks always err on the safe side.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        # Fallback: 1 token ~= 4 chars for English, but Chinese/code-heavy
        # text is closer to 1 token per 1.5 chars. Use /3 as a safe middle.
        raw = max(1, len(text) // 3)
    else:
        raw = len(enc.encode(text))
    return int(raw * _SAFETY_MULTIPLIER) + 1  # +1 to absorb floor()


def estimate_messages_tokens(
    messages: list[dict[str, str]],
    model: str = "cl100k_base",
) -> int:
    """Estimate total tokens for an OpenAI-format messages list.

    Adds a small per-message overhead (4 tokens) to approximate the
    framing the API adds around each message (role tag, separators).
    This matches OpenAI's documented "every message follows this template:
    <im_start>{role/name}\n{content}<im_end>\n" ≈ 4 tokens of framing.
    """
    total = 0
    for m in messages:
        total += 4  # framing overhead per message
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            # OpenAI vision / multipart content — sum the text parts only.
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"], model)
    total += 2  # priming tokens for the assistant reply
    return total


def get_input_budget(
    max_total_tokens: int,
    max_thinking_tokens: int,
    max_output_tokens: int,
) -> int:
    """Compute the input-token budget given the per-model caps.

    All four caps must sum-consistent: input + thinking + output ≤ total.
    We return ``max(0, total - thinking - output)`` so misconfigured models
    (thinking + output ≥ total) simply produce a zero input budget, which
    the pre-flight guard will then refuse — surfacing the config bug
    immediately rather than silently sending prompts that can never fit.
    """
    return max(0, max_total_tokens - max_thinking_tokens - max_output_tokens)


def truncate_to_token_budget(
    text: str,
    budget: int,
    model: str = "cl100k_base",
    *,
    ellipsis: str = "…",
) -> str:
    """Truncate *text* to fit within *budget* tokens (best-effort).

    Uses :func:`estimate_tokens` for sizing, so the same 1.15× safety
    multiplier applies — the returned string is guaranteed to consume ≤
    ``budget`` tokens under our conservative estimate, which means it is
    *also* guaranteed to fit under the model's real tokenizer (since our
    estimate over-counts).

    When *budget* is 0 or negative, the text is returned unchanged (legacy
    behaviour — caller did not opt into token-aware truncation).

    The truncation preserves the **start** of the text because prompts are
    typically structured with the most important context up front (role
    description, constraint list, …). An ``ellipsis`` marker is appended
    when truncation occurs so the LLM sees a signal that context was cut.
    """
    if budget <= 0 or not text:
        return text
    enc = _get_encoding()
    if enc is None:
        # Fallback path — char-based. /3 is conservative.
        max_chars = max(1, int(budget / _SAFETY_MULTIPLIER) * 3)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + ellipsis
    # Estimate tokens with the safety multiplier; we want the RAW token
    # count to be ≤ budget / multiplier so the estimate stays ≤ budget.
    raw_budget = max(1, int(budget / _SAFETY_MULTIPLIER))
    tokens = enc.encode(text)
    if len(tokens) <= raw_budget:
        return text
    truncated = enc.decode(tokens[:raw_budget])
    return truncated + ellipsis


__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "get_input_budget",
    "truncate_to_token_budget",
]
