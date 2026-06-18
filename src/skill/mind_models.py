"""Mind Models extraction via LLM with triple verification.

Adapts nuwa-skill extraction-framework.md §1:
- Verification 1: Cross-thread recurrence (appears in 2+ threads)
- Verification 2: Predictive power (can predict stance on new issues)
- Verification 3: Exclusivity (not something all clusters do)
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from src.data.schemas import Message, Thread
from src.llm.client import LLMClient
from src.skill.schema import MindModel


EXTRACTION_PROMPT = """You are analyzing conversation patterns from a specific type of online community participant.

Below are representative conversation threads from this group. Analyze their reasoning patterns, stance frameworks, and evaluation criteria.

For each distinct reasoning pattern you identify, extract:
1. name: A concise name for the mental model
2. description: One-line description
3. evidence: List of specific quotes/behaviors from the threads that demonstrate this pattern
4. application: When this reasoning model gets applied
5. limitation: When this model fails or doesn't apply

Identify 3-7 mind models. Focus on patterns that are:
- Recurring across multiple threads (not one-off)
- Predictive (can forecast stance on new issues)
- Distinctive (not something everyone does)

Threads:
{threads_text}

Output a JSON object with a single key "models" whose value is an array of
objects, each with keys: name, description, evidence (list), application, limitation.
Output ONLY this JSON object, no other text.
"""

VERIFICATION_PROMPT = """You are verifying whether a reasoning pattern is distinctive to a specific group.

Given the following mind model candidate:
- Name: {name}
- Description: {description}

And these comparison patterns from OTHER groups:
{comparison_text}

Answer these questions as JSON:
{{
  "cross_domain_verified": true/false,  // Does evidence come from 2+ different threads/topics?
  "predictive_verified": true/false,     // Can this model predict stance on a NEW unseen issue?
  "exclusive_verified": true/false,      // Is this pattern NOT something all groups do?
  "reasoning": "brief explanation"
}}

Output ONLY the JSON object."""


class MindModelExtractor:
    """Extracts and triple-verifies mind models from cluster corpus."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-4o",
        validation_threshold: int = 2,
    ):
        self.llm = llm_client
        self.model_name = model_name
        self.validation_threshold = validation_threshold

    async def extract(
        self,
        cluster_threads: list[Thread],
        other_cluster_summaries: list[str] | None = None,
    ) -> list[MindModel]:
        """Extract verified mind models from cluster threads.

        Args:
            cluster_threads: Representative threads from this cluster.
            other_cluster_summaries: Descriptions of other clusters for exclusivity check.

        Returns:
            List of triple-verified MindModels.
        """
        # Step 1: Extract candidate models
        candidates = await self._extract_candidates(cluster_threads)

        # Step 2: Verify each candidate
        verified = []
        for candidate in candidates:
            verification = await self._verify(candidate, other_cluster_summaries or [])
            mm = MindModel(
                name=candidate["name"],
                description=candidate["description"],
                evidence=candidate.get("evidence", []),
                application=candidate.get("application", ""),
                limitation=candidate.get("limitation", ""),
                cross_domain_verified=verification.get("cross_domain_verified", False),
                predictive_verified=verification.get("predictive_verified", False),
                exclusive_verified=verification.get("exclusive_verified", False),
            )
            verified.append(mm)

        # Filter: only keep models that pass at least validation_threshold of 3 verifications
        verified = [mm for mm in verified if sum([
            mm.cross_domain_verified, mm.predictive_verified, mm.exclusive_verified
        ]) >= self.validation_threshold]

        return verified[:7]  # cap at 7

    async def _extract_candidates(self, threads: list[Thread]) -> list[dict]:
        """Use LLM to extract candidate mind models from threads."""
        threads_text = self._format_threads(threads)

        prompt = EXTRACTION_PROMPT.format(threads_text=threads_text)
        messages = [
            {"role": "system", "content": "You are an expert behavioral analyst."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat_completion_json(
            messages, self.model_name, temperature=0.3, default={"models": []}
        )

        # extract_candidates returns the array wrapped in {"models": [...]}
        data = response
        if isinstance(data, dict):
            return data.get("models", [])
        if isinstance(data, list):
            return data
        return []

    async def _verify(self, candidate: dict, comparisons: list[str]) -> dict:
        """Run triple verification on a candidate model."""
        comparison_text = "\n".join(f"- {c}" for c in comparisons) if comparisons else "(no comparison data available)"

        prompt = VERIFICATION_PROMPT.format(
            name=candidate.get("name", ""),
            description=candidate.get("description", ""),
            comparison_text=comparison_text,
        )

        messages = [
            {"role": "system", "content": "You are a rigorous research verifier."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat_completion_json(
            messages,
            self.model_name,
            temperature=0.2,
            default={
                "cross_domain_verified": False,
                "predictive_verified": False,
                "exclusive_verified": False,
            },
        )
        if isinstance(response, dict):
            return response
        return {
            "cross_domain_verified": False,
            "predictive_verified": False,
            "exclusive_verified": False,
        }

    def _format_threads(self, threads: list[Thread], max_chars: int = 8000) -> str:
        """Format threads into readable text for LLM input."""
        lines = []
        total_chars = 0
        for i, thread in enumerate(threads):
            lines.append(f"\n--- Thread {i+1}: {thread.topic} ---")
            for msg in thread.messages[:15]:  # cap messages per thread
                line = f"[{msg.user_id}] ({msg.action_type.value}): {msg.text[:200]}"
                total_chars += len(line)
                if total_chars > max_chars:
                    lines.append("... (truncated)")
                    return "\n".join(lines)
                lines.append(line)
        return "\n".join(lines)
