"""Tier 1: post-generation Expression DNA filter.

2026-07-17 simplification: cosine-distance-from-centroid with empirical
95th-percentile threshold. Replaces the Bonferroni-corrected per-dim-max
z-score filter, which (at d=1024) pushed the effective threshold to ~4.13σ
and never fired in v3 runs. Cosine distance answers the intended question
"does this vector belong to the cluster?" rather than "is any single dim
extreme?".
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from loguru import logger

from src.enforcement.base import EnforcementResult, EnforcementStrategy
from src.llm.client import LLMClient
from src.skill.schema import ExpressionDNA
from src.config.embedder import run_embed_in_executor


class Tier1ExpressionFilter(EnforcementStrategy):
    """Post-generation cosine-distance filter for Expression DNA compliance."""

    def __init__(
        self,
        alpha: float = 1.0,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        max_retries: int = 1,
        rng: random.Random | None = None,
    ):
        super().__init__(alpha, rng=rng)
        self.llm = llm_client
        self.model_name = model_name
        # Outline §4.4.2 specifies N_retry=3 for the constrained-regeneration
        # budget. Tier 1 mirrors this so both post-gen tiers use the same
        # regeneration budget (G6).
        self.max_retries = max_retries
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from src.config.settings import get_shared_embedder
            self._embedder = get_shared_embedder()
        return self._embedder

    async def check_pre_generation(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Tier 1 is post-generation only — pass through."""
        return EnforcementResult(passed=True, tier="none", modified_messages=messages)

    async def check_post_generation(
        self,
        text: str,
        context: dict[str, Any],
    ) -> EnforcementResult:
        """Check if generated text is within the cluster's cosine boundary.

        Args:
            text: Generated text.
            context: Must contain "expression_dna" with embedding_centroid and
                embedding_cosine_threshold.

        Returns:
            EnforcementResult indicating pass/fail.
        """
        if not self._should_enforce():
            return EnforcementResult(passed=True, tier="none", original_text=text)

        edna: ExpressionDNA | None = context.get("expression_dna")
        if edna is None:
            return EnforcementResult(passed=True, tier="none", original_text=text)

        if edna.embedding_centroid is None:
            raise ValueError(
                "Expression DNA embedding centroid is missing; "
                "Tier 1 embedding filter cannot operate."
            )

        embedder_dim = None
        try:
            embedder_dim = self.embedder.get_sentence_embedding_dimension()
        except AttributeError:
            # Test stubs may not implement this method; skip dimension guard.
            pass

        edna_dim = len(edna.embedding_centroid)
        if embedder_dim is not None and edna_dim != embedder_dim:
            logger.warning(
                f"Expression DNA dimension mismatch: skill has {edna_dim} dims, "
                f"embedder has {embedder_dim} dims. Disabling Tier 1 for this skill. "
                f"Recompile skills with the current embedder model."
            )
            return EnforcementResult(
                passed=True,
                tier="tier1",
                original_text=text,
                reason=(
                    f"Dimension mismatch: skill={edna_dim}, embedder={embedder_dim}"
                ),
            )

        text_embedding = await run_embed_in_executor(
            self.embedder.encode, text, show_progress_bar=False
        )

        centroid = np.asarray(edna.embedding_centroid, dtype=float)
        text_emb = np.asarray(text_embedding, dtype=float)
        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        cosine = float(np.dot(text_norm, centroid_norm))
        distance = 1.0 - cosine

        threshold = float(edna.embedding_cosine_threshold)
        if not (0.0 < threshold < 1.0):
            # Skill not yet recompiled with cosine threshold — fall back to a
            # permissive default so the filter is not silently a no-op.
            logger.warning(
                "ExpressionDNA.embedding_cosine_threshold missing or invalid; "
                "using fallback 0.4. Recompile the skill to calibrate the threshold."
            )
            threshold = 0.4

        if distance <= threshold:
            return EnforcementResult(passed=True, tier="tier1", original_text=text)

        # Beyond boundary — apply distance-proportional rejection (outline §4.4.3)
        distance_ratio = distance / threshold
        if not self._rejection_probability(distance_ratio):
            # α-gated: probabilistically let this pass
            return EnforcementResult(
                passed=True,
                tier="tier1",
                original_text=text,
                reason=(
                    f"Expression DNA beyond boundary (cos_dist={distance:.3f}, "
                    f"threshold={threshold:.3f}) but α-gate passed"
                ),
            )

        return EnforcementResult(
            passed=False,
            tier="tier1",
            reason=(
                f"Expression DNA violation: cosine distance {distance:.3f} > "
                f"threshold {threshold:.3f}"
            ),
            original_text=text,
        )

    async def enforce_and_regenerate(
        self,
        text: str,
        original_messages: list[dict[str, str]],
        context: dict[str, Any],
        safe_template: str | None = None,
    ) -> tuple[str, EnforcementResult]:
        """Check and regenerate if necessary.

        Outline §4.4.2 specifies a safe-template fallback after N_retry
        exhausted. When regeneration budget is spent and the text is still
        outside the boundary, fall back to a neutral safe template
        (caller-supplied) rather than silently accepting the off-boundary
        text. When no ``safe_template`` is provided, fall back to the prior
        behaviour (accept the last attempt).
        """
        for attempt in range(self.max_retries + 1):
            result = await self.check_post_generation(text, context)
            if result.passed:
                return text, result

            if attempt < self.max_retries and self.llm is not None:
                regen_messages = list(original_messages)
                regen_messages.append({
                    "role": "assistant",
                    "content": text,
                })
                regen_messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response doesn't match the expected communication style. "
                        "Rewrite only the response text. Preserve the selected action, reply target, "
                        "stance direction, and factual meaning stated in the context. Make it more "
                        "consistent with the following style guidelines: "
                        + self._format_style_hints(context.get("expression_dna"))
                    ),
                })
                text = await self.llm.chat_completion(regen_messages, self.model_name)
            else:
                if safe_template is not None:
                    return safe_template, EnforcementResult(
                        passed=False,
                        tier="tier1",
                        reason=(
                            f"Expression DNA violation persisted after "
                            f"{self.max_retries} retries — safe-template "
                            f"fallback used"
                        ),
                        original_text=safe_template,
                    )
                return text, result

        return text, EnforcementResult(passed=True, tier="tier1", original_text=text)

    def _format_style_hints(self, edna) -> str:
        if edna is None:
            return ""
        hints = [
            f"avg_sentence_length: {edna.avg_sentence_length}",
            f"formal/casual: {edna.style_formal_casual}",
            f"cautious/assertive: {edna.style_cautious_assertive}",
        ]
        return "; ".join(hints)
