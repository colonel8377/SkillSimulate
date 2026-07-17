"""LLM-as-judge Tier 3 anti-pattern enforcement.

A simple judge agent that reads the agent's generated message, the skill's
anti-patterns, and a short conversation context, then asks an LLM whether any
anti-pattern is violated. The judge is intentionally kept lightweight: one
LLM call per message, one JSON verdict per anti-pattern.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.enforcement.base import EnforcementResult
from src.skill.schema import AntiPattern, SkillFile


JUDGE_PROMPT = """You are an impartial judge evaluating whether a simulated message violates behavioral anti-patterns for a Wikipedia editor archetype.

Archetype: {archetype_label}
Archetype description: {archetype_description}

Anti-patterns (a message may violate zero or more of these):
{anti_patterns}

Recent conversation context (most recent first):
{context}

Message to evaluate:
"{message}"

Evaluate each anti-pattern independently. For each anti-pattern, output:
- "violated": true if the message clearly exhibits the prohibited behavior, false otherwise
- "confidence": a number from 0.0 to 1.0
- "reason": a brief explanation, empty if not violated

Output strictly as JSON with one key per anti-pattern (use the pattern index as the key):
{{
  "0": {{"violated": false, "confidence": 0.0, "reason": ""}},
  "1": {{"violated": false, "confidence": 0.0, "reason": ""}},
  ...
}}

Do not output any text other than the JSON object."""


class Tier3LLMJudge:
    """LLM judge for anti-pattern violations."""

    def __init__(
        self,
        llm_client,
        model_name: str = "classification",
        output_dir: str = "outputs/results/tier3_llm_judgments",
        audit_only: bool = False,
    ):
        self.llm = llm_client
        self.model_name = model_name
        self.audit_only = audit_only
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path: Path | None = None
        self._run_context: dict[str, Any] = {}

    def set_run_id(self, run_id: str) -> None:
        """Start a new judgment log for the given run/cell."""
        self._log_path = self.output_dir / f"{run_id}.jsonl"

    def set_run_context(self, context: dict[str, Any]) -> None:
        """Attach cell-level provenance (run_id/condition/dataset/model/repeat)
        to every subsequent log record, so JSONL rows stay self-describing
        for later human annotation."""
        self._run_context = dict(context)

    async def judge(
        self,
        text: str,
        skill: SkillFile,
        context_messages: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> EnforcementResult:
        """Run LLM judge on a generated message.

        Returns an EnforcementResult compatible with the post-generation harness.
        If audit_only is True, the result always reports passed=True to the
        harness, but the raw judgment is still logged for human review.
        """
        anti_patterns = []
        if skill.constraint:
            anti_patterns = skill.constraint.anti_patterns

        if not anti_patterns or not text or not text.strip():
            return EnforcementResult(passed=True, tier="tier3_llm_judge", reason="no anti-patterns or empty text")

        prompt = self._build_prompt(text, skill, anti_patterns, context_messages)
        try:
            parsed = await self.llm.chat_completion_json(
                messages=[{"role": "user", "content": prompt}],
                model_name=self.model_name,
                temperature=0.0,
                max_tokens=2048,
                default=None,
            )
        except Exception as exc:
            logger.error(f"Tier3 LLM judge failed for {skill.cluster_id}: {exc}")
            return EnforcementResult(
                passed=False, tier="tier3_llm_judge",
                reason=f"judge call failed (conservative block): {exc}",
                original_text=text,
            )

        if parsed is None or not isinstance(parsed, dict):
            logger.error(
                f"Tier3 LLM judge returned non-dict for {skill.cluster_id}: "
                f"{str(parsed)[:200]}"
            )
            return EnforcementResult(
                passed=False, tier="tier3_llm_judge",
                reason="judge returned non-JSON object (conservative block)",
                original_text=text,
            )

        verdict = self._parse_verdict(parsed, len(anti_patterns))
        violated = [v for v in verdict if v.get("violated")]

        self._log(text, skill, anti_patterns, verdict, json.dumps(parsed, ensure_ascii=False), context_messages, meta)

        if self.audit_only or not violated:
            return EnforcementResult(passed=True, tier="tier3_llm_judge", reason="no violation or audit-only")

        reason = "; ".join(
            f"{anti_patterns[v['idx']].description[:60]} ({v.get('reason', '')})"
            for v in violated
        )
        return EnforcementResult(
            passed=False,
            tier="tier3_llm_judge",
            reason=f"LLM judge violations: {reason}",
            original_text=text,
        )

    def _build_prompt(
        self,
        text: str,
        skill: SkillFile,
        anti_patterns: list[AntiPattern],
        context_messages: list[dict[str, Any]],
    ) -> str:
        archetype_label = skill.archetype_label or f"cluster {skill.cluster_id}"
        archetype_description = ""
        if skill.capability and skill.capability.expression_dna and skill.capability.expression_dna.expression_rules:
            archetype_description = " ".join(skill.capability.expression_dna.expression_rules[:3])

        ap_text = []
        for i, ap in enumerate(anti_patterns):
            evidence = "; ".join(ap.evidence[:2]) if ap.evidence else ""
            correct = ap.correct_alternative or ""
            ap_text.append(
                f"[{i}] {ap.description}\n"
                f"    Why prohibited: {ap.reason}\n"
                f"    When it activates: {'; '.join(ap.trigger_conditions)}\n"
                f"    Evidence from real data: {evidence}\n"
                f"    Correct alternative: {correct}"
            )

        context_text = []
        for m in context_messages[-8:]:
            user = m.get("user_id", m.get("speaker", "?"))
            content = str(m.get("text", m.get("content", "")))[:200]
            context_text.append(f"[{user}]: {content}")

        return JUDGE_PROMPT.format(
            archetype_label=archetype_label,
            archetype_description=archetype_description,
            anti_patterns="\n\n".join(ap_text),
            context="\n".join(context_text) or "(no context)",
            message=text,
        )

    def _parse_verdict(self, parsed: dict, n_patterns: int) -> list[dict[str, Any]]:
        """Parse already-decoded JSON dict into per-pattern verdicts.

        ``chat_completion_json`` returns a pre-parsed dict; this method
        normalises it into the expected list-of-dicts form, tolerating
        missing keys, non-dict values, and list-form responses.
        """
        verdicts = []
        for i in range(n_patterns):
            key = str(i)
            entry = parsed.get(key, {}) if isinstance(parsed, dict) else {}
            if not isinstance(entry, dict):
                entry = {}
            verdicts.append({
                "idx": i,
                "violated": bool(entry.get("violated", False)),
                "confidence": float(entry.get("confidence", 0.0)),
                "reason": str(entry.get("reason", "")),
            })
        return verdicts

    def _log(
        self,
        text: str,
        skill: SkillFile,
        anti_patterns: list[AntiPattern],
        verdicts: list[dict[str, Any]],
        raw: str,
        context_messages: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if self._log_path is None:
            return
        record = {
            "timestamp": time.time(),
            # Cell-level provenance (set per simulation run via
            # set_run_context) + per-turn annotation metadata (set per
            # judge call via meta). Both are required so a human labeler
            # can locate and re-judge any logged message without the
            # simulation checkpoint.
            **self._run_context,
            **(meta or {}),
            "cluster_id": skill.cluster_id,
            "archetype_label": skill.archetype_label,
            "message": text,
            "context_messages": context_messages or [],
            "verdicts": verdicts,
            "raw_response": raw,
            "anti_patterns": [
                {"description": ap.description, "reason": ap.reason}
                for ap in anti_patterns
            ],
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
