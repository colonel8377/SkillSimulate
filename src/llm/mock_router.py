"""Deterministic mock LLM router for dev-loop validation.

Activated when ``CADP_MOCK_LLM=1``. Produces valid JSON for every contract
the CADP pipeline exercises:

  * planner.plan / planner.replan  → ActionPlan JSON
  * SkillCompiler._quality_check   → quality-check JSON
  * MindModelExtractor._extract_candidates → JSON array of mind models
  * MindModelExtractor._verify     → verification JSON
  * AntiPatternDetector._llm_detect → JSON array of anti-patterns

Routing keys off the ``system`` message (each call site uses a distinct
system persona), which is stable across refactors of user-prompt wording.

Outputs are deterministic given the input bytes — a SHA-256 seed drives
all "random" choices, so re-running the dev config reproduces the same
trace bit-for-bit.

This is a DEVELOPMENT FIXTURE. Outputs are NOT paper-grade. Real runs
must clear ``CADP_MOCK_LLM`` and provide a real ``CADP_OPENAI_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


class MockLLMRouter:
    """Routes mock completions by inspecting the system persona."""

    DEFAULT_ACTIONS = (
        "comment", "agree", "disagree", "counter_argue",
        "edit", "revert", "post", "report", "award_delta",
    )

    def route(
        self,
        messages: list[dict[str, str]],
        model_name: str,
    ) -> str:
        """Return deterministic mock JSON for the given call."""
        system = ""
        user = ""
        for m in messages:
            if m.get("role") == "system":
                system += m.get("content", "") + "\n"
            elif m.get("role") == "user":
                user += m.get("content", "") + "\n"

        seed_hex = hashlib.sha256(
            (system + "\n" + user).encode("utf-8")
        ).hexdigest()
        # Deterministic pseudo-random floats in [0, 1) from the seed.
        def _rand(offset: int) -> float:
            chunk = seed_hex[offset * 8 : offset * 8 + 8]
            return int(chunk, 16) / 0xFFFFFFFF

        persona = system.strip().lower()

        if "quality assurance reviewer" in persona:
            return json.dumps({
                "mind_models_accurate": True,
                "anti_patterns_valid": True,
                "expression_dna_consistent": True,
                "overall_pass": True,
                "issues": [],
            })

        if "rigorous research verifier" in persona:
            # Pass two of three verifications deterministically so the
            # ``validation_threshold=2`` filter keeps the candidate.
            return json.dumps({
                "cross_domain_verified": True,
                "predictive_verified": True,
                "exclusive_verified": False,
                "reasoning": "Mock verification — pattern recurs in 2+ threads.",
            })

        if "expert behavioral analyst" in persona:
            # Mind-model candidate extraction (JSON array).
            return json.dumps([
                {
                    "name": f"mock_mind_model_{i}",
                    "description": f"Deterministic mock reasoning pattern #{i}",
                    "evidence": ["mock evidence fragment A", "mock evidence fragment B"],
                    "application": "Applied when mock trigger conditions hold.",
                    "limitation": "Mock limitation — does not generalise off-distribution.",
                }
                for i in range(3)
            ])

        if "behavioral pattern analyst" in persona:
            # Anti-pattern detection (JSON array). All seven trigger fields
            # are populated so downstream enforcement has Category A/B/C
            # triggers to bite on.
            return json.dumps([
                {
                    "description": "Mock avoided behaviour: ad-hominem escalation",
                    "trigger_conditions": ["When the user disagrees personally"],
                    "trigger_keywords": ["idiot", "stupid"],
                    "trigger_regex": [r"\\b(you are|you're)\\s+(stupid|idiot)\\b"],
                    "trigger_semantic_phrases": ["personal attack instead of argument"],
                    "trigger_action_patterns": ["disagree->report"],
                    "reason": "This cluster defuses rather than escalates.",
                }
            ])

        if "revising your response" in persona:
            # planner.replan — echo the requested action_type when given.
            action_type = self._extract_replan_action(user) or "comment"
            return json.dumps({
                "action_type": action_type,
                "text": f"[mock-replan] revised response (seed={seed_hex[:8]})",
                "reasoning": "Mock replan to satisfy constraint.",
            })

        if "planning your next action" in persona:
            # planner.plan — pick a deterministic action from those listed.
            action_type = self._pick_plan_action(user, _rand)
            target = self._pick_target_msg_id(user, _rand)
            return json.dumps({
                "action_type": action_type,
                "text": f"[mock-plan] generated message (seed={seed_hex[:8]})",
                "target_msg_id": target,
                "reasoning": "Mock plan decision.",
            })

        # Fallback: return innocuous JSON so any ``json.loads`` upstream
        # succeeds rather than crashing the pipeline. Log enough to diagnose.
        return json.dumps({
            "mock": True,
            "note": "No mock contract matched; returning empty object.",
            "system_persona_head": persona[:80],
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_plan_action(self, user_text: str, rand) -> str:
        """Pick an action_type deterministically from the prompt.

        The planning prompt always lists ``Available actions: a, b, c``.
        Prefer stance-bearing / lifecycle actions when available so the
        sandbox's action distribution is non-degenerate.
        """
        match = re.search(r"Available actions:\s*([^\n]+)", user_text)
        pool: list[str] = []
        if match:
            pool = [a.strip() for a in match.group(1).split(",") if a.strip()]
        if not pool:
            pool = list(self.DEFAULT_ACTIONS)
        # Preference order — picks first available stance-bearing action.
        preference = (
            "counter_argue", "disagree", "agree", "award_delta",
            "revert", "edit", "report", "comment", "post",
        )
        for pref in preference:
            for a in pool:
                if a == pref:
                    return a
        return pool[int(rand(0) * len(pool))]

    def _pick_target_msg_id(self, user_text: str, rand) -> str:
        """Pick a deterministic target_msg_id from <msg id=...> tags."""
        ids = re.findall(r"<msg id=([^>]+)>", user_text)
        if not ids:
            return ""
        # Pick a stable index in [0, len(ids)).
        idx_hash = hashlib.sha256(b"target" + user_text.encode()).hexdigest()
        idx = int(idx_hash[:8], 16) % len(ids)
        return ids[idx]

    def _extract_replan_action(self, user_text: str) -> str:
        """Extract the action_type the replan prompt asked for."""
        match = re.search(r'"action_type":\s*"([^"]+)"', user_text)
        return match.group(1) if match else ""
