"""Condition factory — maps condition names to agent configurations."""

from __future__ import annotations

from src.config.schemas import ConditionName

ALL_CONDITIONS = [c.value for c in ConditionName]


def validate_conditions(conditions: list[str]) -> list[str]:
    """Validate condition names."""
    valid = set(ALL_CONDITIONS)
    for c in conditions:
        if c not in valid:
            raise ValueError(f"Unknown condition: {c}. Valid: {ALL_CONDITIONS}")
    return conditions


def is_cadp_condition(condition: str) -> bool:
    """Check if a condition uses CADP-derived skills.

    Includes all CADP variants plus COLLEAGUE.SKILL and Pop-Aligned+CADP,
    both of which require compiled .skill files. ``real_history`` (Exp2
    §6.2 replay arm) is NOT a CADP condition — it does not compile skills.
    """
    return condition.startswith("cadp") or condition in {
        "colleague_skill",
        "pop_aligned_cadp",
    }


def distiller_suffix(condition: str) -> str | None:
    """Return the manual-distiller suffix ("colleague" | "nuwa") for
    ``cadp_full_colleague`` / ``cadp_full_nuwa``, or ``None`` for every
    other condition (including the retired bare ``cadp_full``).

    Per reframe v1 (2026-07-08), nuwa-skill is the primary base structure
    for CADP's three-dim distillation (EDNA / MM / AP). The mechanism
    ablation (``cadp_minus_ap``) and permutation test (``cadp_shuffled``)
    are nuwa-only main conditions, so they bind to the nuwa distiller
    by default. ``cadp_full_colleague`` is the methodology-comparison
    arm and binds to the colleague distiller.

    Used by ``Experiment1Runner._get_or_compile_skills`` to select
    ``skill_cluster_{cid}_{platform}_{distiller}.yaml`` instead of the
    bare (pipeline-A-compiled) filename, and to fail fast rather than
    silently falling back to pipeline A when the distilled file is
    missing (plan: "打通蒸馏产出与 CADP 实验", 2026-07-08).
    """
    if condition == "cadp_full_colleague":
        return "colleague"
    if condition in {"cadp_full_nuwa", "cadp_minus_ap", "cadp_shuffled"}:
        return "nuwa"
    return None


def is_replay_only_condition(condition: str) -> bool:
    """Conditions that replay observed traces instead of running a sandbox.

    Currently only ``real_history`` (Exp2 §6.2 fourth arm). These
    conditions skip agent construction + simulation and feed the observed
    real threads directly into the metric pipeline as both the sim input
    and the ground truth, yielding a self-similarity ceiling.
    """
    return condition == "real_history"


def needs_skills(condition: str) -> bool:
    """Alias for readability: does this condition need compiled skills?"""
    return is_cadp_condition(condition)
