"""Tier 3: Anti-pattern enforcement — two-stage per outline §4.4.

Pre-generation stage (this class's ``check_pre_generation``) is **advisory**:
it cannot block output that has not yet been produced, so it injects a
reformulation instruction into the system prompt and signals
``passed=False`` so the harness records the pre-gen violation. The
hard-block-with-regeneration runs at post-generation via
``check_post_generation`` → harness ``enforce_output`` → agent-side
Forced Reformulation Protocol loop in ``src/agents/base.py``
(outline §4.4.2: block → diagnosis injection → constrained regeneration
× N_retry=3 → safe-template fallback).

Trigger categories (outline §4.4.1):
  - Category A (lexical): regex patterns + keyword matching
  - Category B (semantic): Sentence-BERT embedding cosine similarity ≥ θ_sem
  - Category C (behavioral): action pattern / classifier matching
Any category triggering flags a violation.
"""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

from src.enforcement.base import EnforcementResult, EnforcementStrategy
from src.skill.schema import AntiPattern
from src.config.embedder import run_embed_in_executor


REFORMULATION_INSTRUCTION = """IMPORTANT: Your planned response would violate a behavioral constraint.

The following patterns are prohibited for your role:
{violations}

Please reformulate your response to AVOID these patterns entirely.
Do not mention these constraints in your output."""


class Tier3AntiPatternBlock(EnforcementStrategy):
    """Two-stage Anti-pattern enforcement (outline §4.4).

    Pre-gen = advisory reformulation injection; post-gen = violation report
    that drives the agent-side Forced Reformulation Protocol loop in
    ``src/agents/base.py`` (true hard block with N_retry + safe-template
    fallback lives there, not inside this class).
    """

    def __init__(self, alpha: float = 1.0, rng: random.Random | None = None):
        super().__init__(alpha, rng=rng)
        self._embedder = None
        # Category C behavioral classifier (outline §4.4.1). None = legacy
        # substring-matching fallback. Attach via attach_behavioral_classifier
        # after §5.3.5 trigger calibration has trained the model.
        self._behavioral_classifier = None
        self.behavioral_threshold: float = 0.5

    @property
    def embedder(self):
        """Lazy-load SentenceTransformer for Category B semantic triggers."""
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    async def check_pre_generation(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Check if messages contain anti-pattern triggers before generation.

        Args:
            messages: Current LLM message list (includes the user prompt / context).
            context: Must contain "anti_patterns" (list[AntiPattern])
                     and optionally "draft_action" (str) for action-level checks.

        Returns:
            EnforcementResult with potentially modified messages.
        """
        if not self._should_enforce():
            return EnforcementResult(passed=True, tier="none", modified_messages=messages)

        anti_patterns: list[AntiPattern] | None = context.get("anti_patterns")
        if not anti_patterns:
            return EnforcementResult(passed=True, tier="none", modified_messages=messages)

        # Check the last user message and any draft action
        check_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                check_text = msg.get("content", "")
                break

        draft_action = context.get("draft_action", "")
        check_text = f"{check_text} {draft_action}".strip()

        # Detect violations (with optional action history for Category C)
        action_history = context.get("action_history")
        violations = await run_embed_in_executor(
            self._detect_violations, check_text, anti_patterns, action_history
        )

        if not violations:
            return EnforcementResult(passed=True, tier="tier3", modified_messages=messages)

        # Build reformulation instruction
        violation_text = "\n".join(
            f"- {v['description']} (triggered by: {v['triggered_by']})"
            for v in violations
        )
        reform_msg = REFORMULATION_INSTRUCTION.format(violations=violation_text)

        # Append as system constraint
        modified = list(messages)
        system_idx = None
        for i, msg in enumerate(modified):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx is not None:
            modified[system_idx] = {
                "role": "system",
                "content": modified[system_idx]["content"] + "\n\n" + reform_msg,
            }
        else:
            modified.insert(0, {"role": "system", "content": reform_msg})

        # Pre-gen advisory injection (outline §4.4 — Tier 3 pre-gen stage):
        # we cannot "block" output that has not been generated yet, so we
        # propagate the reformulation instruction into the planner prompt and
        # signal `passed=False` so the harness can record the pre-gen violation
        # for metrics. The hard-block-with-regeneration runs at post-gen
        # (see harness.enforce_output → agent Forced Reformulation loop,
        # outline §4.4.2).
        return EnforcementResult(
            passed=False,
            tier="tier3",
            reason=f"Pre-gen anti-pattern violation: {len(violations)} patterns triggered",
            modified_messages=modified,
            injection_text=reform_msg,
        )

    async def check_post_generation(
        self,
        text: str,
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Also check generated text for anti-patterns (post-gen safety net)."""
        if not self._should_enforce():
            return EnforcementResult(passed=True, tier="none", original_text=text)

        anti_patterns: list[AntiPattern] | None = context.get("anti_patterns")
        if not anti_patterns:
            return EnforcementResult(passed=True, tier="none", original_text=text)

        action_history = context.get("action_history")
        violations = await run_embed_in_executor(
            self._detect_violations, text, anti_patterns, action_history
        )

        if not violations:
            return EnforcementResult(passed=True, tier="tier3", original_text=text)

        violation_descs = "; ".join(
            f"{v['description']} (triggered by: {v['triggered_by']})" for v in violations
        )
        return EnforcementResult(
            passed=False,
            tier="tier3",
            reason=f"Post-gen anti-pattern violation: {violation_descs}",
            original_text=text,
        )

    def _detect_violations(
        self,
        text: str,
        anti_patterns: list[AntiPattern],
        action_history: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Detect which anti-patterns are triggered by the text.

        Applies α-gated confidence-proportional blocking (outline §4.4.3):
        when α < 1.0, semantic triggers use cosine similarity as confidence,
        and lexical/behavioral triggers use confidence=1.0.

        Args:
            text: Text to check (planned or generated).
            anti_patterns: List of anti-pattern definitions.
            action_history: Chronological list of action types taken by the agent
                (for Category C behavioral triggers). Optional.

        Returns:
            List of violation dicts with 'description' and 'triggered_by'.
        """
        violations = []
        text_lower = text.lower()

        for ap in anti_patterns:
            triggered = False
            triggered_by = ""
            confidence = 1.0  # default for lexical/behavioral matches

            # Category A: Lexical (regex + keyword)
            for pattern in ap.trigger_regex:
                # Patterns containing "[A-Z]" are deliberately CASE-SENSITIVE
                # (e.g. ALL-CAPS / shouting detection) and must run against the
                # ORIGINAL text. Applying them to lowercased text with
                # re.IGNORECASE makes "[A-Z]" match lowercase letters too,
                # false-firing on virtually every message (this was the dominant
                # driver of the inflated safe-template rate).
                if "[A-Z]" in pattern:
                    hit = re.search(pattern, text)
                else:
                    hit = re.search(pattern, text_lower, re.IGNORECASE)
                if hit:
                    triggered = True
                    triggered_by = f"lexical/regex: {pattern}"
                    break

            if not triggered:
                for kw in ap.trigger_keywords:
                    # Word-boundary match: a bare substring check would flag
                    # "edit" inside "edited"/"edition"/"credit", "report"
                    # inside "reported", "agree" inside "disagree", etc.
                    # (outline §4.4.1 Category A targets the lexical item,
                    # not an arbitrary infix).
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                        triggered = True
                        triggered_by = f"lexical/keyword: {kw}"
                        break

            # Category B: Semantic (Sentence-BERT cosine similarity)
            if not triggered and ap.trigger_semantic_phrases:
                sem_result = self._check_semantic(
                    text, ap.trigger_semantic_phrases, ap.trigger_semantic_threshold
                )
                if sem_result is not None:
                    triggered = True
                    triggered_by = sem_result["description"]
                    confidence = sem_result["confidence"]

            # Category C: Behavioral (action pattern matching)
            if not triggered and ap.trigger_action_patterns and action_history:
                beh_result = self._check_behavioral(
                    action_history,
                    ap.trigger_action_patterns,
                    per_pattern_threshold=ap.trigger_behavioral_threshold,
                )
                if beh_result is not None:
                    triggered = True
                    triggered_by = beh_result

            # Apply α-gated confidence-proportional blocking
            if triggered:
                if not self._block_probability(confidence):
                    continue  # α-gate passed — do not flag this violation

                violations.append({
                    "description": ap.description,
                    "triggered_by": triggered_by,
                })

        return violations

    def _check_semantic(
        self,
        text: str,
        reference_phrases: list[str],
        threshold: float,
    ) -> dict | None:
        """Category B: Check semantic similarity against reference phrases.

        Uses Sentence-BERT cosine similarity. If text is too similar to any
        reference phrase (cosine ≥ threshold), the anti-pattern is triggered.

        Returns:
            Dict with 'description' and 'confidence' if triggered, None otherwise.
        """
        import numpy as np

        try:
            all_texts = [text] + list(reference_phrases)
            all_embs = self.embedder.encode(all_texts, show_progress_bar=False)
            text_emb = all_embs[0]
            ref_embs = all_embs[1:]
        except Exception as e:
            logger.warning(f"Tier 3 semantic check skipped — embedder unavailable: {e}")
            return None

        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
        ref_norms = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-10)
        cosines = ref_norms @ text_norm

        max_idx = int(np.argmax(cosines))
        max_cos = float(cosines[max_idx])

        if max_cos >= threshold:
            return {
                "description": (
                    f"semantic/cosine={max_cos:.3f}≥{threshold}: "
                    f'matches "{reference_phrases[max_idx][:50]}"'
                ),
                "confidence": max_cos,
            }
        return None

    def _check_behavioral(
        self,
        action_history: list[str],
        action_patterns: list[str],
        per_pattern_threshold: float | None = None,
    ) -> str | None:
        """Category C: Behavioral trigger detection (outline §4.4.1).

        Two paths, in priority order:

        1. **Spec-compliant path (preferred)** — if a trained
           ``BehavioralTriggerClassifier`` is attached to this Tier 3
           instance (via :meth:`attach_behavioral_classifier`), predict
           P(violation | recent action history) and trigger when the
           probability ≥ the decision threshold. The threshold resolution
           order is (a) ``per_pattern_threshold`` carried by the
           ``AntiPattern`` (``trigger_behavioral_threshold``), (b) the
           global ``self.behavioral_threshold``. This implements outline
           §4.4.1's per-anti-pattern calibrated thresholds (G7).

        2. **Legacy fallback path** — substring / sliding-window matching
           against ``action_patterns`` using ``->`` notation. Preserved
           for back-compat with untrained deployments and for unit tests
           that exercise the pattern-matching interface directly. The
           legacy path is deliberately conservative: ``"agree"`` does NOT
           match ``"disagree"`` (the previous substring bug, code review
           m1 / C2 — fixed here).

        Args:
            action_history: Chronological list of recent action-type strings.
            action_patterns: ``->``-joined action sequences to match
                (legacy path only).
            per_pattern_threshold: Optional per-anti-pattern threshold
                override. When ``None`` the global
                ``self.behavioral_threshold`` is used.

        Returns:
            Trigger description string if triggered, None otherwise.
        """
        effective_threshold = (
            per_pattern_threshold
            if per_pattern_threshold is not None
            else self.behavioral_threshold
        )

        # --- Spec-compliant classifier path ---
        if self._behavioral_classifier is not None and self._behavioral_classifier.trained:
            proba = self._behavioral_classifier.predict_proba(action_history)
            if proba >= effective_threshold:
                return (
                    f"behavioral/classifier: P={proba:.3f} ≥ "
                    f"{effective_threshold:.2f}"
                )
            # Classifier says no violation — but per outline §4.4.1
            # ("Any category triggering → violation"), we still run the
            # legacy pattern path as a complementary signal so explicit
            # user-defined ``->`` patterns remain enforceable.

        # --- Legacy pattern-matching path (fallback / complementary) ---
        recent = action_history[-10:]
        # NOTE: do NOT use a naive substring test on the joined string,
        # because ``"agree"`` would match ``"disagree"`` (code review m1).
        # Always go through the sliding-window equality check below.
        for pattern in action_patterns:
            pat = pattern.strip().lower()
            pat_actions = [a.strip() for a in pat.split("->")]
            if not pat_actions or len(pat_actions) > len(recent):
                continue
            for i in range(len(recent) - len(pat_actions) + 1):
                window = [a.lower() for a in recent[i:i + len(pat_actions)]]
                if window == pat_actions:
                    return f"behavioral/pattern: {pat}"

        return None

    def attach_behavioral_classifier(
        self,
        classifier,
        threshold: float = 0.5,
    ) -> None:
        """Attach a trained ``BehavioralTriggerClassifier`` (outline §4.4.1).

        Once attached, ``_check_behavioral`` uses the classifier as the
        primary Category C signal. The legacy substring path runs as a
        complementary check (so explicit ``->`` patterns remain enforceable).

        Args:
            classifier: A ``BehavioralTriggerClassifier`` instance (trained
                or untrained; untrained classifiers are silently ignored,
                preserving the legacy fallback).
            threshold: Trigger threshold on P(violation). Default 0.5.
                Lower → more sensitive (higher recall, lower precision);
                higher → more conservative.
        """
        self._behavioral_classifier = classifier
        self.behavioral_threshold = float(threshold)
