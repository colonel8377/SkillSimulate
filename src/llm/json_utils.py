"""Robust JSON extraction from LLM responses.

LLM outputs — especially from reasoning models (DeepSeek-V4, o1-style) —
frequently break ``json.loads`` even when the content is semantically valid:

  * Markdown code fences (```` ```json ... ``` ````)
  * Prose before/after the JSON payload
  * Trailing commas inside arrays/objects
  * Unescaped control characters (literal newlines / tabs inside strings)
  * Truncated output (finish_reason=length) leaving an incomplete payload

This module provides ``extract_json``, a progressive parser that strips
the most common contaminants and, when strict ``json.loads`` fails, falls
back to ``json.loads`` on a cleaned copy. All callers in the CADP pipeline
(mind-model extraction, anti-pattern detection, planner, quality checks)
should route JSON responses through this function instead of calling
``json.loads`` directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?", re.MULTILINE)
_TRAILING_FENCE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_contaminants(text: str) -> str:
    """Remove markdown fences and trailing commas from a JSON-like string."""
    text = _FENCE_RE.sub("", text)
    text = _TRAILING_FENCE_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _find_json_span(text: str) -> str | None:
    """Locate the outermost JSON array or object in *text* by bracket matching.

    Returns the matched substring, or ``None`` if no balanced structure is
    found. Bracket matching respects string literals (so ``]`` inside a
    string does not prematurely close an array).
    """
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json(text: str, default: Any = None) -> Any:
    """Parse a JSON value from an LLM response, tolerating common contaminants.

    Tries, in order:
      1. ``json.loads`` on the raw text.
      2. Bracket/brace-matched span extraction + ``json.loads``.
      3. Same span after stripping markdown fences + trailing commas.

    Args:
        text: Raw LLM response text.
        default: Value to return if all parse attempts fail.

    Returns:
        Parsed Python object, or *default* on failure.
    """
    if not text or not text.strip():
        return default

    # Attempt 1: direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: locate the JSON span by balanced bracket matching.
    span = _find_json_span(text)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            # Attempt 3: clean contaminants from the matched span.
            cleaned = _strip_contaminants(span)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # Attempt 4: clean the full text and retry bracket matching.
    cleaned_full = _strip_contaminants(text)
    span2 = _find_json_span(cleaned_full)
    if span2 is not None:
        try:
            return json.loads(span2)
        except json.JSONDecodeError:
            pass

    return default
