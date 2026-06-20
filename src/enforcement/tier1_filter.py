"""Tier 1: Post-generation Expression DNA filter.

Checks generated text embedding against cluster's expression distribution.
Rejects + regenerates if outside 2σ boundary.
"""

from __future__ import annotations

import random

import numpy as np
from loguru import logger
from typing import Any

from src.enforcement.base import EnforcementResult, EnforcementStrategy
from src.llm.client import LLMClient
from src.skill.schema import ExpressionDNA
from src.config.embedder import run_embed_in_executor


def _bonferroni_threshold(per_dim_sigma: float, dim: int) -> float:
    """Bonferroni-corrected per-dim-max z threshold for dim dimensions.

    B1 reproducibility fix: the original code rejected when the per-dim
    max z-score exceeded ``sigma_threshold``. For high-dim embeddings
    (production: d=1024 for ``bge-large-en-v1.5``; debug: d=384 for
    ``all-MiniLM-L6-v2``), the expected max of |z| over d independent
    standard normals is ≈ sqrt(2 ln d) (d=1024: ≈ 3.79; d=384: ≈ 3.45)
    — well above the legacy 2.0 threshold — so the filter would
    systematically over-flag and inflate CADP-Full's safe-template
    fallback rate (g5 over-alignment distortion).

    Fix: treat "outside 2σ" as a multiple-comparison problem. Holding
    the family-wise error rate at the same tail probability as a single
    2σ two-sided test (P(|z|>2) ≈ 0.0455), the per-dim significance
    required to flag *any* of d dimensions is alpha/d. The corrected
    per-dim threshold is the (1 - alpha/d / 2) standard-normal quantile:
    d=1024 → ≈ 4.13σ; d=384 → ≈ 3.84σ. Both are well above their
    respective E[max|z|], preventing systematic over-rejection.

    Args:
        per_dim_sigma: User-facing σ boundary (e.g. 2.0).
        dim: Embedding dimensionality.

    Returns:
        Effective per-dim-max z-score threshold.
    """
    if dim <= 1:
        return float(per_dim_sigma)
    try:
        from scipy.stats import norm
    except Exception:
        # Fallback (no scipy): Gumbel approximation E[max|z|] ≈ sqrt(2 ln d)
        # scaled by per_dim_sigma / 2 (so d=2 gives ~2σ, d→∞ grows slowly).
        import math
        return float(per_dim_sigma) * max(1.0, math.sqrt(2.0 * math.log(dim)) / 1.177)
    # Family-wise two-sided tail probability at ``per_dim_sigma``
    alpha_family = 2.0 * (1.0 - norm.cdf(per_dim_sigma))
    alpha_per_dim = alpha_family / dim
    # Guard against numerical underflow at very large d
    alpha_per_dim = max(alpha_per_dim, 1e-12)
    return float(norm.ppf(1.0 - alpha_per_dim / 2.0))


class Tier1ExpressionFilter(EnforcementStrategy):
    """Post-generation embedding filter for Expression DNA compliance.

    B1 fix: the per-dim-max z-score is compared against a dimension-aware
    Bonferroni-corrected threshold (see :func:`_bonferroni_threshold`),
    not the raw ``sigma_threshold``. This preserves the 1-D 2σ family-wise
    error rate as embedding dimensionality grows.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        llm_client: LLMClient | None = None,
        model_name: str = "gpt-4o",
        max_retries: int = 3,
        sigma_threshold: float = 2.0,
        dimension_aware: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(alpha, rng=rng)
        self.llm = llm_client
        self.model_name = model_name
        # Outline §4.4.2 specifies N_retry=3 for the constrained-regeneration
        # budget. Tier 1 mirrors this so both post-gen tiers use the same
        # regeneration budget (G6).
        self.max_retries = max_retries
        self.sigma_threshold = sigma_threshold
        # B1: when True, apply Bonferroni correction to the per-dim-max
        # statistic so high-dim embeddings do not systematically over-flag.
        self.dimension_aware = dimension_aware
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
        """Check if generated text is within Expression DNA boundary.

        Args:
            text: Generated text.
            context: Must contain "expression_dna" with embedding_centroid and embedding_std.

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

        centroid = np.array(edna.embedding_centroid)
        std = np.array(edna.embedding_std) if edna.embedding_std else np.ones_like(centroid)

        # Compute z-score distance (cosine distance from centroid, normalized)
        diff = text_embedding - centroid
        # Avoid division by zero
        std_safe = np.where(std > 1e-8, std, 1.0)
        z_scores = np.abs(diff) / std_safe
        max_z = float(np.max(z_scores))

        # B1: dimension-aware threshold. The user-facing ``sigma_threshold``
        # (e.g. 2σ) is interpreted family-wise: the per-dim-max z-score is
        # compared against a Bonferroni-corrected threshold so the
        # probability of flagging any in-distribution embedding stays at
        # the 1-D σ level rather than growing with d. ``effective_threshold``
        # degrades gracefully to ``sigma_threshold`` when dimension_aware
        # is False (legacy behaviour).
        dim = int(centroid.shape[0])
        effective_threshold = (
            _bonferroni_threshold(self.sigma_threshold, dim)
            if self.dimension_aware
            else float(self.sigma_threshold)
        )

        if max_z <= effective_threshold:
            return EnforcementResult(
                passed=True,
                tier="tier1",
                original_text=text,
            )

        # Beyond boundary — apply distance-proportional rejection (outline §4.4.3)
        distance_ratio = max_z / effective_threshold
        if not self._rejection_probability(distance_ratio):
            # α-gated: probabilistically let this pass
            return EnforcementResult(
                passed=True,
                tier="tier1",
                original_text=text,
                reason=(
                    f"Expression DNA beyond boundary (z={max_z:.2f}, "
                    f"threshold={effective_threshold:.2f}) but α-gate passed"
                ),
            )

        # Failed — need regeneration
        return EnforcementResult(
            passed=False,
            tier="tier1",
            reason=(
                f"Expression DNA violation: max z-score {max_z:.2f} > "
                f"{effective_threshold:.2f} (dim-aware threshold for d={dim})"
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
        exhausted. Tier 1 mirrors this: when regeneration budget is spent
        and the text is still outside the 2σ boundary, fall back to a
        neutral safe template (caller-supplied) rather than silently
        accepting the off-boundary text. When no ``safe_template`` is
        provided, fall back to the prior behaviour (accept the last
        attempt) so legacy callers and unit tests are unaffected.

        Returns:
            Tuple of (final_text, enforcement_result). ``result.passed``
            is False when the safe-template fallback was used so the
            harness can count it as a violation in metrics.
        """
        for attempt in range(self.max_retries + 1):
            result = await self.check_post_generation(text, context)
            if result.passed:
                return text, result

            if attempt < self.max_retries and self.llm is not None:
                # Add regeneration hint
                regen_messages = list(original_messages)
                regen_messages.append({
                    "role": "assistant",
                    "content": text,
                })
                regen_messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response doesn't match the expected communication style. "
                        "Please rewrite it to be more consistent with the following style guidelines: "
                        + self._format_style_hints(context.get("expression_dna"))
                    ),
                })
                text = await self.llm.chat_completion(regen_messages, self.model_name)
            else:
                # Out of retries — apply safe-template fallback when the
                # caller supplied one; otherwise preserve legacy
                # "accept the best attempt" semantics (G6).
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
