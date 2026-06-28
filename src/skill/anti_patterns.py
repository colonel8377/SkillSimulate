"""Anti-pattern detection from cluster corpus.

Identifies behaviors that this cluster consistently avoids but other clusters exhibit.
These become hard constraints in the Constraint Track.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from src.data.schemas import ActionType, Message, Thread
from src.llm.client import LLMClient
from src.skill.schema import AntiPattern


DETECTION_PROMPT = """You are analyzing what a specific type of online community participant consistently AVOIDS doing.

Below are conversation threads from this group. Analyze patterns of behavior that are ABSENT or RARE:
- Topics they never engage with
- Actions they never take
- Language styles they never use
- Response patterns they avoid

For each anti-pattern identified, provide ALL of the following fields so the
behaviour can be enforced as a hard constraint (outline §4.4.1 — three
trigger categories A / B / C, any one of which firing is sufficient):

1. description: What behavior is prohibited/avoided (one sentence).
2. trigger_conditions: List of situations where someone might do this
   (but this group wouldn't) — natural-language context cues.
3. trigger_keywords: Category A (lexical) — specific words / short phrases
   that, if present in the text, indicate this pattern. Plain strings
   (not regex); leave empty list if no reliable lexical cue exists.
4. trigger_regex: Category A (lexical) — Python regex patterns (e.g.
   r"\\\\b(you're|you are)\\\\s+(stupid|idiot)\\\\b") that match the
   prohibited surface form. Empty list if a keyword list suffices.
5. trigger_semantic_phrases: Category B (semantic) — reference phrases
   whose Sentence-BERT embedding has cosine similarity >= 0.85 with the
   agent text indicate the pattern (used when the surface form varies but
   the meaning is stable). 1-5 phrases. Empty list if N/A.
6. trigger_action_patterns: Category C (behavioral) — sequence patterns
   using "->" notation over recent action types
   (e.g. "disagree->revert->report"). Empty list if N/A.
7. reason: Why this group avoids this behavior.

Also consider what OTHER groups commonly do that this group doesn't:
{comparison_behaviors}

Threads from this group:
{threads_text}

Output a JSON object with a single key "patterns" whose value is an array
of objects, each with all seven keys:
description, trigger_conditions (list), trigger_keywords (list),
trigger_regex (list), trigger_semantic_phrases (list),
trigger_action_patterns (list), reason (str).
Output ONLY this JSON object, no other text."""


class AntiPatternDetector:
    """Detects anti-patterns from cluster corpus via behavioral gap analysis."""

    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        self.llm = llm_client
        self.model_name = model_name

    async def detect(
        self,
        cluster_threads: list[Thread],
        other_cluster_action_dist: dict[str, Counter] | None = None,
    ) -> list[AntiPattern]:
        """Detect anti-patterns for this cluster.

        Args:
            cluster_threads: Representative threads from this cluster.
            other_cluster_action_dist: Action type distributions from other clusters
                for cross-cluster behavioral gap analysis.

        Returns:
            List of AntiPatterns with trigger conditions.
        """
        # Step 1: Quantitative behavioral gap analysis
        gap_patterns = self._quantitative_gap_analysis(cluster_threads, other_cluster_action_dist)

        # Step 2: LLM-based qualitative detection
        llm_patterns = await self._llm_detect(cluster_threads, other_cluster_action_dist)

        # Merge and deduplicate
        all_patterns = gap_patterns + llm_patterns
        unique = self._deduplicate(all_patterns)

        # Fallback: every cluster should have at least one basic anti-pattern
        # so the constraint track is not empty and Tier-3 has something to enforce.
        if not unique:
            logger.warning(
                "AntiPattern detection produced zero patterns; injecting a generic "
                "civility anti-pattern as fallback."
            )
            unique = [self._generic_civility_pattern()]

        return unique

    def _generic_civility_pattern(self) -> AntiPattern:
        """Generic fallback anti-pattern: no personal attacks or harassment."""
        return AntiPattern(
            description="Avoid personal attacks, insults, or harassment toward other users.",
            trigger_conditions=[
                "When responding to a user you disagree with",
                "When frustrated by another user's actions",
            ],
            trigger_keywords=[
                "idiot", "stupid", "moron", "loser", "jerk", "asshole",
                "damn you", "shut up", "fuck", "shit", "bitch",
            ],
            trigger_regex=[
                r"\b(you\s+(are|re)\s+)?(stupid|idiot|moron|loser)\b",
                r"\b(shut\s+up|get\s+lost)\b",
            ],
            trigger_semantic_phrases=[
                "you are an idiot",
                "personal insult",
                "hostile attack",
            ],
            trigger_action_patterns=[],
            reason="Basic civility baseline that applies to every community participant.",
        )

    def _quantitative_gap_analysis(
        self,
        cluster_threads: list[Thread],
        other_dist: dict[str, Counter] | None,
    ) -> list[AntiPattern]:
        """Find actions present in other clusters but absent in this one."""
        if not other_dist:
            return []

        # Compute this cluster's action distribution
        this_actions = Counter()
        for thread in cluster_threads:
            for msg in thread.messages:
                this_actions[msg.action_type.value] += 1

        this_total = sum(this_actions.values()) or 1

        patterns = []
        for other_id, other_actions in other_dist.items():
            other_total = sum(other_actions.values()) or 1
            for action, count in other_actions.items():
                other_ratio = count / other_total
                this_ratio = this_actions.get(action, 0) / this_total

                # If other cluster does this >10% but this cluster does it <2%
                if other_ratio > 0.10 and this_ratio < 0.02:
                    # Outline §4.4.1: quantitative gap analysis yields a
                    # Category C (behavioral) trigger — fire when the
                    # avoided action shows up in the recent action history.
                    action_lower = action.lower()
                    patterns.append(AntiPattern(
                        description=f"Avoids {action} behavior (common in cluster {other_id} at {other_ratio:.1%})",
                        trigger_conditions=[f"When a {action} action would be expected"],
                        trigger_keywords=[action_lower],
                        trigger_action_patterns=[action_lower],
                        reason=f"This cluster engages in {action} at only {this_ratio:.1%} vs {other_ratio:.1%} in cluster {other_id}",
                    ))

        return patterns

    async def _llm_detect(
        self,
        cluster_threads: list[Thread],
        other_dist: dict[str, Counter] | None,
    ) -> list[AntiPattern]:
        """Use LLM to qualitatively detect avoided behaviors."""
        threads_text = self._format_threads(cluster_threads)

        comparison = ""
        if other_dist:
            lines = []
            for cid, actions in other_dist.items():
                top = actions.most_common(5)
                lines.append(f"Cluster {cid}: {', '.join(f'{a}({c})' for a, c in top)}")
            comparison = "\n".join(lines)

        prompt = DETECTION_PROMPT.format(
            threads_text=threads_text,
            comparison_behaviors=comparison or "(no comparison data)",
        )

        messages = [
            {"role": "system", "content": "You are a behavioral pattern analyst."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat_completion_json(
            messages, self.model_name, temperature=0.3, default={"patterns": []}
        )

        # _llm_detect returns the array wrapped in {"patterns": [...]}
        items = self._parse_pattern_response(response)
        if not items:
            logger.warning(
                f"AntiPattern _llm_detect returned no patterns. "
                f"Response type={type(response).__name__}, "
                f"preview={str(response)[:500]!r}"
            )

        return [
            AntiPattern(
                description=item.get("description", ""),
                trigger_conditions=item.get("trigger_conditions", []),
                trigger_keywords=item.get("trigger_keywords", []) or [],
                trigger_regex=item.get("trigger_regex", []) or [],
                trigger_semantic_phrases=item.get("trigger_semantic_phrases", []) or [],
                trigger_action_patterns=item.get("trigger_action_patterns", []) or [],
                reason=item.get("reason", ""),
            )
            for item in items
        ]

    @staticmethod
    def _parse_pattern_response(response: Any) -> list[dict]:
        """Extract pattern list from various possible LLM return shapes."""
        if isinstance(response, list):
            return [p for p in response if isinstance(p, dict) and p.get("description")]

        if isinstance(response, dict):
            for key in ("patterns", "anti_patterns", "results"):
                arr = response.get(key)
                if isinstance(arr, list):
                    return [p for p in arr if isinstance(p, dict) and p.get("description")]

            if response.get("description"):
                return [response]

        return []

    def _format_threads(self, threads: list[Thread], max_chars: int = 20000) -> str:
        """Format threads for LLM input.

        max_chars raised from 6000 to 20000 — reasoning models with 984K
        input budget can accommodate more context for richer anti-pattern
        detection.
        """
        lines = []
        total = 0
        for i, thread in enumerate(threads):
            lines.append(f"\n--- Thread {i+1}: {thread.topic} ---")
            for msg in thread.messages[:20]:
                line = f"[{msg.user_id}] ({msg.action_type.value}): {msg.text[:300]}"
                total += len(line)
                if total > max_chars:
                    return "\n".join(lines)
                lines.append(line)
        return "\n".join(lines)

    def _deduplicate(self, patterns: list[AntiPattern]) -> list[AntiPattern]:
        """Remove duplicate anti-patterns by overlap across all trigger fields."""
        seen: set[str] = set()
        unique = []
        for pattern in patterns:
            # Build a signature across all Category A/B/C trigger fields so
            # two detections that share any trigger surface form collapse.
            signature_tokens: set[str] = set()
            signature_tokens.update(kw.lower() for kw in pattern.trigger_keywords)
            signature_tokens.update(pattern.trigger_regex)
            signature_tokens.update(phrase.lower() for phrase in pattern.trigger_semantic_phrases)
            signature_tokens.update(pat.lower() for pat in pattern.trigger_action_patterns)
            if signature_tokens & seen:
                continue
            seen.update(signature_tokens)
            unique.append(pattern)
        return unique
