"""Agent population builder — allocates agents across clusters."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.agents.base import BaseAgent
from src.clustering.clusterer import ClusterResult


@dataclass
class PopulationConfig:
    """Configuration for population building."""
    size: int = 30
    cluster_proportions: dict[int, float] | None = None  # cluster_id → proportion


class PopulationBuilder:
    """Builds agent population with proportional cluster allocation."""

    def __init__(
        self,
        llm_client,
        model_name: str = "gpt-4o",
        skills: dict[str, Any] | None = None,
        skills_dir: str | None = None,
        platform: str | None = None,
        alpha: float = 1.0,
        alpha_tier1: float | None = None,
        alpha_tier2: float | None = None,
        alpha_tier3: float | None = None,
        backend: str = "base",
        memory_strategy: str = "sliding",
        compaction_interval: int = 5,
        compaction_keep_recent: int = 10,
        max_display_items: int = 5,
        per_msg_token_ratio: int = 10,
        per_msg_token_floor: int = 60,
        max_thread_messages: int = 5,
        reflection_interval: int = 10,
        population_allocation: str = "proportional",
        max_reformulation_retries: int = 1,
        tier1_max_retries: int = 1,
        tier3_llm_judge_enabled: bool = False,
        tier3_llm_judge_model: str = "classification",
        tier3_llm_judge_audit_only: bool = False,
        tier3_llm_judge_output_dir: str = "outputs/results/tier3_llm_judgments",
    ):
        """Initialize population builder.

        Args:
            llm_client: LLM client for agent generation.
            model_name: Model name for agents.
            skills: Pre-loaded skills dict (cluster_id → SkillFile).
            skills_dir: Directory to load .skill files from (alternative to skills dict).
            platform: Platform filter for skill loading from directory.
            alpha: Constraint hardness parameter (global default).
            alpha_tier1/2/3: Per-tier alpha overrides (None = use alpha).
            backend: Skill adapter backend ("base", "langchain", etc.).
            memory_strategy: "sliding" (default, exp1/exp2) or "rolling_summary" (R4).
            compaction_interval: When memory_strategy="rolling_summary", run
                compaction every N turns.
            compaction_keep_recent: When memory_strategy="rolling_summary",
                keep this many most-recent raw items unsummarized.
            max_display_items: memory items shown to planner per turn.
            per_msg_token_ratio: per-msg budget = max_memory_tokens // ratio.
            per_msg_token_floor: per-msg token floor.
            max_thread_messages: recent thread messages in planner prompt.
            reflection_interval: periodic reflection every N rounds.
            tier3_llm_judge_enabled: Enable LLM-as-judge Tier 3.
            tier3_llm_judge_model: Model name for LLM judge.
            tier3_llm_judge_audit_only: If True, judge logs but does not block.
            tier3_llm_judge_output_dir: Directory to save judgment logs.
        """
        self.llm = llm_client
        self.model_name = model_name
        self.alpha = alpha
        self.alpha_tier1 = alpha_tier1
        self.alpha_tier2 = alpha_tier2
        self.alpha_tier3 = alpha_tier3
        self.backend = backend
        self.memory_strategy = memory_strategy
        self.compaction_interval = compaction_interval
        self.compaction_keep_recent = compaction_keep_recent
        self.max_display_items = max_display_items
        self.per_msg_token_ratio = per_msg_token_ratio
        self.per_msg_token_floor = per_msg_token_floor
        self.max_thread_messages = max_thread_messages
        self.reflection_interval = reflection_interval
        if population_allocation not in {"proportional", "balanced"}:
            raise ValueError(
                "population_allocation must be 'proportional' or 'balanced', "
                f"got {population_allocation!r}"
            )
        self.population_allocation = population_allocation
        self.max_reformulation_retries = max_reformulation_retries
        self.tier1_max_retries = tier1_max_retries
        self.tier3_llm_judge_enabled = tier3_llm_judge_enabled
        self.tier3_llm_judge_model = tier3_llm_judge_model
        self.tier3_llm_judge_audit_only = tier3_llm_judge_audit_only
        self.tier3_llm_judge_output_dir = tier3_llm_judge_output_dir
        # Issue 1: derive the per-agent memory token budget from the
        # model endpoint's input-token cap. We reserve ~30% of the
        # input budget for system prompt + thread context + planner
        # template + reflection; the remaining 70% is the agent-memory
        # slice. When the endpoint has no cap configured (max_total_tokens=0)
        # max_memory_tokens stays 0 and legacy char-slicing applies.
        ep = getattr(llm_client, "models", {}).get(model_name)
        input_budget = getattr(ep, "max_input_tokens", 0) if ep else 0
        # Reserve ~3K tokens for system/skill prompt (CADP dual-track can
        # be sizeable) and split the rest 70/30 between memory and thread.
        if input_budget > 0:
            reserved = min(3000, input_budget // 4)
            self._max_memory_tokens = max(512, int((input_budget - reserved) * 0.7))
        else:
            self._max_memory_tokens = 0

        # Load skills: either from dict or from directory
        if skills is not None:
            self.skills = skills
        elif skills_dir is not None:
            from src.skill.compiler import SkillCompiler
            self.skills = SkillCompiler.load_all_skills(skills_dir, platform)
        else:
            self.skills = {}

    def build_population(
        self,
        cluster_result: ClusterResult,
        size: int = 30,
        condition: str = "vanilla",
        cluster_proportions: dict[int, float] | None = None,
        seed: int = 42,
    ) -> list[BaseAgent]:
        """Build agent population for a specific condition.

        Args:
            cluster_result: Clustering result with cluster assignments.
            size: Number of agents.
            condition: Which experimental condition (determines agent type).
            cluster_proportions: Override proportional allocation.

        Returns:
            List of agents.
        """
        rng = random.Random(seed)

        # Compute cluster allocation
        cluster_ids = cluster_result.get_cluster_ids()
        if cluster_proportions:
            allocations = self._allocate_by_proportions(size, cluster_proportions)
        elif self.population_allocation == "balanced":
            allocations = self._allocate_balanced(size, cluster_ids)
        else:
            allocations = self._allocate_proportional(size, cluster_result, cluster_ids)

        # Build agents
        agents = []
        agent_counter = 0
        for cluster_id, n_agents in allocations.items():
            for _ in range(n_agents):
                # R1: derive a deterministic per-agent seed from the
                # population seed + agent index. The EnforcementHarness
                # then derives independent sub-RNGs per tier, so two runs
                # of the same cell produce identical α-gating sequences.
                # Falls back to None when the population seed is None
                # (legacy behaviour for unit tests).
                agent_seed = (seed + agent_counter * 1_000_003) if seed is not None else None
                agent = self._create_agent(
                    agent_id=f"agent_{agent_counter}",
                    cluster_id=str(cluster_id),
                    condition=condition,
                    cluster_result=cluster_result,
                    rng=rng,
                    agent_seed=agent_seed,
                )
                # Set per-agent engagement ratio from cluster activity distribution
                agent.engagement_ratio = self._sample_engagement_ratio(
                    cluster_id, cluster_result, rng
                )
                agents.append(agent)
                agent_counter += 1

        rng.shuffle(agents)
        return agents[:size]

    @staticmethod
    def _allocate_balanced(size: int, cluster_ids: list[int]) -> dict[int, int]:
        """Allocate agents evenly across locked skills for feasibility tests."""
        if not cluster_ids:
            return {}
        if size < len(cluster_ids):
            raise ValueError(
                f"Balanced population needs at least one agent per cluster: "
                f"size={size}, clusters={len(cluster_ids)}"
            )
        base, remainder = divmod(size, len(cluster_ids))
        return {
            cid: base + int(index < remainder)
            for index, cid in enumerate(sorted(cluster_ids))
        }

    def _allocate_proportional(
        self,
        size: int,
        cluster_result: ClusterResult,
        cluster_ids: list[int],
    ) -> dict[int, int]:
        """Allocate agents proportionally to cluster sizes."""
        cluster_sizes = {
            cid: len(cluster_result.get_cluster_members(cid))
            for cid in cluster_ids
        }
        total = sum(cluster_sizes.values()) or 1

        allocations = {}
        remaining = size
        for i, cid in enumerate(cluster_ids):
            if i == len(cluster_ids) - 1:
                allocations[cid] = remaining
            else:
                n = max(1, round(size * cluster_sizes[cid] / total))
                allocations[cid] = min(n, remaining)
                remaining -= allocations[cid]

        return allocations

    def _allocate_by_proportions(
        self,
        size: int,
        proportions: dict[int, float],
    ) -> dict[int, int]:
        """Allocate by explicit proportions."""
        allocations = {}
        remaining = size
        cluster_ids = list(proportions.keys())
        for i, cid in enumerate(cluster_ids):
            if i == len(cluster_ids) - 1:
                allocations[cid] = remaining
            else:
                n = max(1, round(size * proportions.get(cid, 1.0 / len(proportions))))
                allocations[cid] = min(n, remaining)
                remaining -= allocations[cid]
        return allocations

    def _create_agent(
        self,
        agent_id: str,
        cluster_id: str,
        condition: str,
        cluster_result: ClusterResult,
        rng: random.Random,
        agent_seed: int | None = None,
    ) -> BaseAgent:
        """Create agent based on condition.

        For CADP conditions, skill MUST be available — no silent fallback.
        """
        from loguru import logger as _logger

        from src.agents.vanilla import VanillaAgent
        from src.agents.descriptive import DescriptivePersonaAgent
        from src.agents.rich_narrative import RichNarrativeAgent
        from src.agents.segmentation import SegmentationPersonaAgent
        from src.agents.pop_aligned import (
            PopAlignedPersonaAgent,
            compute_cluster_attributes,
            sample_individual_attributes,
        )
        from src.agents.cadp import CADPAgent
        from src.agents.cadp_advisory import CADPAdvisoryAgent
        from src.agents.ablations import (
            CADPShuffledAgent,
            CADPMinusExpressionDNAAgent,
            CADPMinusMindModelsAgent,
            CADPMinusAntiPatternsAgent,
            CADPConstraintOnlyAgent,
        )
        from src.agents.colleague_skill import ColleagueSkillAgent
        from src.agents.clustering_only import ClusteringOnlyAgent
        from src.agents.length_matched import LengthMatchedControlAgent
        from src.agents.pop_aligned_cadp import PopAlignedCADPAgent

        def _cluster_features(cluster_id: str):
            members = cluster_result.get_cluster_members(int(cluster_id))
            return [
                cluster_result.user_features[uid]
                for uid in members
                if uid in cluster_result.user_features
            ]

        def _pop_attributes(cluster_id: str, rng: random.Random):
            features = _cluster_features(cluster_id)
            stats = compute_cluster_attributes(features)
            sampled = sample_individual_attributes(stats, rng=rng)
            return stats, sampled

        def _descriptive_persona(cluster_id: str):
            return self._build_descriptive_persona(cluster_id, _cluster_features(cluster_id))

        common_kwargs = dict(
            agent_id=agent_id,
            llm_client=self.llm,
            model_name=self.model_name,
            cluster_id=cluster_id,
            # Issue 1: thread the per-model input-token budget through to
            # every agent so Planner / AgentMemory / ReflectionModule can
            # truncate in token-aware fashion. 0 = legacy char-slicing.
            max_memory_tokens=self._max_memory_tokens,
            # Issue 1: memory strategy (sliding vs rolling_summary) and
            # the compactor's knobs. Default "sliding" is a no-op for
            # exp1/exp2 cells; "rolling_summary" is set by the R4 collapse
            # stress test config.
            memory_strategy=self.memory_strategy,
            compaction_interval=self.compaction_interval,
            compaction_keep_recent=self.compaction_keep_recent,
            max_display_items=self.max_display_items,
            per_msg_token_ratio=self.per_msg_token_ratio,
            per_msg_token_floor=self.per_msg_token_floor,
            max_thread_messages=self.max_thread_messages,
            reflection_interval=self.reflection_interval,
            max_reformulation_retries=self.max_reformulation_retries,
        )

        # Per-tier alpha + backend kwargs for CADP agents
        alpha_kwargs = dict(
            alpha=self.alpha,
            alpha_tier1=self.alpha_tier1,
            alpha_tier2=self.alpha_tier2,
            alpha_tier3=self.alpha_tier3,
            backend=self.backend,
            seed=agent_seed,
            tier1_max_retries=self.tier1_max_retries,
            tier3_llm_judge_enabled=self.tier3_llm_judge_enabled,
            tier3_llm_judge_model=self.tier3_llm_judge_model,
            tier3_llm_judge_audit_only=self.tier3_llm_judge_audit_only,
            tier3_llm_judge_output_dir=self.tier3_llm_judge_output_dir,
        )

        if condition == "vanilla":
            return VanillaAgent(**common_kwargs)

        elif condition == "descriptive":
            members = cluster_result.get_cluster_members(int(cluster_id))
            member_features = [
                cluster_result.user_features[uid]
                for uid in members
                if uid in cluster_result.user_features
            ]
            persona = self._build_descriptive_persona(cluster_id, member_features)
            return DescriptivePersonaAgent(
                persona_description=persona,
                **common_kwargs,
            )

        elif condition == "segmentation":
            members = cluster_result.get_cluster_members(int(cluster_id))
            member_features = [
                cluster_result.user_features[uid]
                for uid in members
                if uid in cluster_result.user_features
            ]
            # B3 fix: thread the total clustered population through so the
            # segmentation profile can report the cluster's true share of
            # the population (the previous ratio always evaluated to 1.0).
            total_population = sum(
                len(cluster_result.get_cluster_members(int(cid)))
                for cid in cluster_result.get_cluster_ids()
            )
            seg_name, seg_demos, seg_psycho = self._build_segmentation_profile(
                cluster_id, member_features, total_population=total_population,
            )
            return SegmentationPersonaAgent(
                segment_name=seg_name,
                segment_demographics=seg_demos,
                segment_psychographics=seg_psycho,
                **common_kwargs,
            )

        elif condition == "pop_aligned":
            # Internal Cluster-Stat Aligned Persona: samples the project's
            # cluster attributes; not a reproduction of arXiv:2509.10127.
            cluster_stats, sampled = _pop_attributes(cluster_id, rng)
            return PopAlignedPersonaAgent(
                cluster_attributes=cluster_stats,
                sampled_attributes=sampled,
                **common_kwargs,
            )

        elif condition == "rich_narrative":
            # Lever-1 ceiling / kill condition (reframe v1, 2026-07-08):
            # maximalist narrative persona from aggregate cluster stats,
            # rendered as multi-paragraph
            # narrative with concrete episodes and example moves. NO
            # compiled .skill rules, NO filter-retry. This tests overall
            # package viability; it does not isolate one causal variable.
            members = cluster_result.get_cluster_members(int(cluster_id))
            member_features = [
                cluster_result.user_features[uid]
                for uid in members
                if uid in cluster_result.user_features
            ]
            persona = self._build_rich_narrative_persona(cluster_id, member_features)
            return RichNarrativeAgent(
                persona_description=persona,
                **common_kwargs,
            )

        elif condition == "colleague_skill":
            # COLLEAGUE.SKILL (Zhou et al. 2026): single-track .skill, no enforcement
            skill = self._get_skill_or_raise(cluster_id, condition)
            return ColleagueSkillAgent(skill=skill, **common_kwargs)

        elif condition == "clustering_only":
            # Clustering-Only Descriptive Persona: same clusters as CADP,
            # but only a descriptive persona (no behavioral rules)
            persona = _descriptive_persona(cluster_id)
            return ClusteringOnlyAgent(
                persona_description=persona,
                **common_kwargs,
            )

        elif condition == "cadp_full":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition in ("cadp_full_colleague", "cadp_full_nuwa"):
            # Same CADPAgent runtime as cadp_full — the only difference is
            # which skill dict was loaded upstream (manually distilled via
            # colleague-skill / nuwa-skill Markdown, converted by
            # scripts/convert_distilled_skills.py, vs pipeline A's LLM
            # re-extraction). See _get_skill_or_raise / conditions.py
            # distiller_suffix for the fail-fast loading behavior.
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition == "cadp_advisory_nuwa":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPAdvisoryAgent(
                skill=skill,
                backend=self.backend,
                **common_kwargs,
            )

        elif condition == "cadp_shuffled":
            if not self.skills:
                raise ValueError(
                    f"Cannot create CADPShuffledAgent: no skills loaded. "
                    f"Need at least 2 clusters for shuffle test."
                )
            return CADPShuffledAgent(
                all_skills=self.skills,
                target_cluster_id=cluster_id,
                rng=rng,
                **alpha_kwargs,
                **common_kwargs,
            )

        elif condition == "cadp_minus_edna":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPMinusExpressionDNAAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition == "cadp_minus_mm":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPMinusMindModelsAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition == "cadp_minus_ap":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPMinusAntiPatternsAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition == "cadp_constraint_only":
            skill = self._get_skill_or_raise(cluster_id, condition)
            return CADPConstraintOnlyAgent(skill=skill, **alpha_kwargs, **common_kwargs)

        elif condition == "pop_aligned_cadp":
            # Pop-Aligned + CADP stacked: attribute distribution + behavioral rules
            skill = self._get_skill_or_raise(cluster_id, condition)
            cluster_stats, sampled = _pop_attributes(cluster_id, rng)
            return PopAlignedCADPAgent(
                skill=skill,
                cluster_attributes=cluster_stats,
                sampled_attributes=sampled,
                **alpha_kwargs,
                **common_kwargs,
            )

        elif condition == "length_matched_control":
            # Length-matched control (ARS review 2026-06-19, DA-E1):
            # token-budget-matched description built from a RANDOM OTHER
            # cluster's stats via the same _build_descriptive_persona
            # template. Isolates "matched tokens + matched form" from
            # "matched behavioral content".
            all_cluster_ids = list(cluster_result.get_cluster_ids())
            other_cluster_ids = [
                cid for cid in all_cluster_ids if str(cid) != str(cluster_id)
            ]
            if other_cluster_ids:
                target_cluster = str(rng.choice(other_cluster_ids))
            else:
                # Single-cluster edge case: degenerates to ClusteringOnly
                # semantics. The condition name still distinguishes it for
                # downstream analysis; logged via MetricsReport.condition.
                target_cluster = cluster_id
            members = cluster_result.get_cluster_members(int(target_cluster))
            member_features = [
                cluster_result.user_features[uid]
                for uid in members
                if uid in cluster_result.user_features
            ]
            persona = self._build_descriptive_persona(target_cluster, member_features)
            return LengthMatchedControlAgent(
                persona_description=persona,
                **common_kwargs,
            )

        else:
            raise ValueError(f"Unknown condition: {condition}")

    def _get_skill_or_raise(self, cluster_id: str, condition: str):
        """Get skill for cluster, raising if not found (no silent fallback)."""
        skill = self.skills.get(cluster_id)
        if skill is None:
            available = list(self.skills.keys())
            raise ValueError(
                f"No skill found for cluster '{cluster_id}' (condition={condition}). "
                f"Available skills: {available}. "
                f"Ensure skills are compiled or loaded before building population."
            )
        return skill

    @staticmethod
    def _cluster_stats(member_features: list) -> dict[str, float]:
        """Compute descriptive statistics for a cluster's behavioral features."""
        import numpy as np

        if not member_features:
            return {}

        attr_accessors = {
            "reply_depth": "mean_indentation",
            "verbosity": "verbosity",
            "question_rate": "question_rate",
            "wp_citation_rate": "wp_citation_rate",
            "conflict_engagement_ratio": "conflict_engagement_ratio",
        }
        stats = {}
        for name, field_name in attr_accessors.items():
            col = np.array([getattr(f, field_name) for f in member_features], dtype=float)
            stats[name] = float(np.mean(col))
        msg_counts = [f.message_count for f in member_features]
        thread_counts = [f.thread_count for f in member_features]
        stats["avg_messages"] = float(np.mean(msg_counts))
        stats["avg_threads"] = float(np.mean(thread_counts))
        stats["n_members"] = len(member_features)
        return stats

    @staticmethod
    def _sample_engagement_ratio(
        cluster_id: str,
        cluster_result: ClusterResult,
        rng: random.Random,
    ) -> float:
        """Sample per-agent engagement ratio from cluster's real activity distribution.

        Samples an empirical within-cluster activity quantile and maps it to a
        bounded, nonlinear participation probability. Quantile mapping avoids
        the previous global min/max collapse where a few extreme power users
        forced every sampled agent to the 0.1 floor.
        """
        import numpy as np

        members = cluster_result.get_cluster_members(int(cluster_id))
        msg_counts = [
            cluster_result.user_features[uid].message_count
            for uid in members
            if uid in cluster_result.user_features
        ]

        if not msg_counts:
            return 0.5

        msg_arr = np.sort(np.asarray(msg_counts, dtype=float))
        sampled_count = float(msg_arr[rng.randrange(len(msg_arr))])
        left = int(np.searchsorted(msg_arr, sampled_count, side="left"))
        right = int(np.searchsorted(msg_arr, sampled_count, side="right"))
        # Randomized tie rank prevents the many minimum-activity users from
        # receiving one identical engagement probability.
        rank = rng.uniform(left, max(left + 1, right))
        quantile = (rank + 0.5) / len(msg_arr)
        # A convex map preserves a long tail: many agents sit near zero while
        # high empirical quantiles engage up to half of the available threads.
        ratio = 0.05 + 0.45 * quantile * quantile
        return float(max(0.05, min(0.50, ratio)))

    def _build_descriptive_persona(
        self,
        cluster_id: str,
        member_features: list,
    ) -> str:
        """Build a descriptive persona from real cluster behavioral statistics.

        Transmits identity/behavioral tags (not behavioral rules), matching
        the definition of Descriptive Persona in outline §1.3.
        """
        stats = self._cluster_stats(member_features)
        if not stats:
            return f"A community member belonging to group {cluster_id}."

        def _level(val, low, high):
            if val >= high:
                return "high"
            if val <= low:
                return "low"
            return "moderate"

        activity = _level(
            stats["avg_messages"],
            low=2, high=10,
        )
        conflict = _level(
            stats["conflict_engagement_ratio"],
            low=0.15, high=0.4,
        )
        inquisitiveness = _level(
            stats["question_rate"],
            low=0.05, high=0.2,
        )

        lines = [
            f"Age group: adult internet user",
            f"Activity level: {activity} (avg {stats['avg_messages']:.0f} messages, "
            f"{stats['avg_threads']:.0f} threads participated)",
            f"Conflict involvement: {conflict} "
            f"(engages in {stats['conflict_engagement_ratio']:.1%} of contested discussions)",
            f"Inquisitiveness: {inquisitiveness} "
            f"(question rate: {stats['question_rate']:.2f})",
            f"Typical reply depth: {stats['reply_depth']:.1f}",
            f"Message verbosity (log characters): {stats['verbosity']:.2f}",
            f"Wikipedia policy citation rate: {stats['wp_citation_rate']:.2f}",
            f"Community group: {stats['n_members']} users with similar behavioral patterns",
        ]
        return "\n".join(lines)

    def _build_rich_narrative_persona(
        self,
        cluster_id: str,
        member_features: list,
    ) -> str:
        """Build lever-1 ceiling narrative persona from real cluster stats.

        Same input as ``_build_descriptive_persona`` (cluster behavioral
        statistics) but rendered as multi-paragraph narrative with
        concrete episodes and example moves, NOT terse stat bullets.
        Token budget (~350-400 words) matched to a CADP skill file so
        length is not the confound — only form differs (description vs
        rules).

        Used by the ``rich_narrative`` kill condition (reframe v1,
        2026-07-08). If CADP loses to this on Predictive Fidelity, the
        headline thesis fails.
        """
        stats = self._cluster_stats(member_features)
        if not stats:
            return (
                f"You are a long-time participant in this online community, "
                f"part of a cohort of regulars who have drifted together over "
                f"months of shared discussion. You know the regulars, you "
                f"remember past disputes, and you have a feel for how things "
                f"usually unfold here. Group {cluster_id}."
            )

        def _level(val, low, high):
            if val >= high:
                return "high"
            if val <= low:
                return "low"
            return "moderate"

        activity = _level(stats["avg_messages"], low=2, high=10)
        conflict = _level(stats["conflict_engagement_ratio"], low=0.15, high=0.4)
        inquisitiveness = _level(stats["question_rate"], low=0.05, high=0.2)

        # Activity narrative
        if activity == "high":
            activity_passage = (
                f"You are one of the more active regulars in your cohort — on "
                f"average you post about {stats['avg_messages']:.0f} messages "
                f"across {stats['avg_threads']:.0f} different threads. You "
                f"show up often, you read most of what's been said before "
                f"jumping in, and people recognize your handle."
            )
        elif activity == "low":
            activity_passage = (
                f"You are a quiet but consistent presence — you only post "
                f"about {stats['avg_messages']:.0f} messages and tend to "
                f"stick to {stats['avg_threads']:.0f} thread at a time. You "
                f"read far more than you write, and when you do speak up "
                f"it's usually because something specific caught your eye."
            )
        else:
            activity_passage = (
                f"You participate at a moderate clip — about "
                f"{stats['avg_messages']:.0f} messages across "
                f"{stats['avg_threads']:.0f} threads. You're not the loudest "
                f"voice, but you're not a lurker either; you contribute when "
                f"you feel you have something to add."
            )

        # Conflict narrative
        conflict_ratio = stats["conflict_engagement_ratio"]
        if conflict == "high":
            conflict_passage = (
                f"You don't shy away from disputes. Roughly "
                f"{conflict_ratio:.0%} of the threads you join turn "
                f"contentious, and that doesn't bother you — you'll push "
                f"back when a claim looks weak, call out inconsistencies, "
                f"and press for sources. A typical move for you is to quote "
                f"the line you disagree with and explain, point by point, "
                f"why it doesn't hold up."
            )
        elif conflict == "low":
            conflict_passage = (
                f"You steer around open conflict. Only about "
                f"{conflict_ratio:.0%} of the threads you join turn "
                f"contentious, and even there you tend to defuse rather than "
                f"escalate — you'll acknowledge the other side, reframe the "
                f"sticking point, or quietly walk away rather than trade "
                f"blows."
            )
        else:
            conflict_passage = (
                f"You engage with conflict selectively. About "
                f"{conflict_ratio:.0%} of the threads you join have some "
                f"heat, and you'll wade in when the issue matters to you, "
                f"but you don't go looking for fights. When you do push "
                f"back, it's usually paired with a concrete fix or a "
                f"question, not a bare objection."
            )

        # Inquisitiveness narrative (an observed linguistic rate, not an
        # inferred claim about private opinion change).
        question_rate = stats["question_rate"]
        if inquisitiveness == "high":
            stance_passage = (
                f"You ask questions frequently (about {question_rate:.0%} of "
                f"your messages contain one). You probe definitions, request "
                f"sources, and use questions to expose the exact point of disagreement."
            )
        elif inquisitiveness == "low":
            stance_passage = (
                f"You rarely phrase contributions as questions (about "
                f"{question_rate:.0%} of messages). You usually state the "
                f"relevant fact, policy, or proposed change directly."
            )
        else:
            stance_passage = (
                f"You mix direct claims with focused questions; about "
                f"{question_rate:.0%} of your messages contain a question, "
                f"usually to clarify evidence or the next concrete step."
            )

        # Depth + editing
        depth_passage = (
            f"You tend to reply about {stats['reply_depth']:.1f} levels deep "
            f"in a thread before you feel a point is settled. Your typical "
            f"verbosity is {stats['verbosity']:.2f} on the log-character "
            f"scale, and about {stats['wp_citation_rate']:.0%} of your "
            f"messages explicitly cite Wikipedia policy."
        )

        # Typical opening move — derived from conflict × flexibility combo.
        # Stays descriptive ("you tend to..."), never prescriptive, so it
        # remains lever-1 narrative rather than drifting into lever-2 rules.
        if conflict == "high" and inquisitiveness != "low":
            opening_passage = (
                f"Your typical opening move is to identify one concrete "
                f"problem with the previous post — an unsourced claim, an "
                f"inconsistency, a slant — and put a question or proposed "
                f"fix next to it. You don't post drive-by objections; if "
                f"you disagree, you say why and what would change your mind."
            )
        elif conflict == "low":
            opening_passage = (
                f"Your typical opening move is to build on the previous "
                f"post — agree where you can, add a clarifying detail or a "
                f"supporting source, then gently flag the one place you "
                f"see differently. You'd rather extend a thread than end "
                f"one."
            )
        else:
            opening_passage = (
                f"Your typical opening move depends on the thread — if it "
                f"looks solid you'll add to it, but if something's off "
                f"you'll name it plainly and suggest a fix rather than "
                f"wave it through."
            )

        # Interaction style — synthesises activity × conflict into a single
        # "how you show up" sketch. Concrete utterance snippets are first-
        # person paraphrases, not verbatim quotes, to stay descriptive.
        if activity == "high" and conflict == "high":
            interaction_passage = (
                f"You come across as a hands-on regular: you answer "
                f"questions, push threads forward, and don't let weak "
                f"claims slide. Other readers can tell from how you write "
                f"that you've been around — you reference prior discussions, "
                f"you don't re-litigate settled points, and when you do "
                f"disagree you lay out the reasoning instead of just "
                f"dismissing the other side."
            )
        elif activity == "low":
            interaction_passage = (
                f"You come across as the careful type who only speaks when "
                f"they have something to add — most of your replies carry "
                f"weight precisely because you don't fill space. People "
                f"learn to read your silences as 'nothing to add yet'."
            )
        else:
            interaction_passage = (
                f"You come across as measured — you contribute steadily "
                f"without dominating, and you'd rather ask one good "
                f"question than post three quick takes. You assume good "
                f"faith until shown otherwise, but you don't pretend a bad "
                f"argument is a good one."
            )

        cohort_passage = (
            f"You're part of a cohort of about {stats['n_members']} users "
            f"who behave similarly — you'll recognize their handles and "
            f"they'll recognize yours. Assume good faith by default, but "
            f"privilege substance over enthusiasm."
        )

        return "\n\n".join([
            activity_passage,
            conflict_passage,
            stance_passage,
            depth_passage,
            opening_passage,
            interaction_passage,
            cohort_passage,
        ])

    def _build_segmentation_profile(
        self,
        cluster_id: str,
        member_features: list,
        total_population: int | None = None,
    ) -> tuple[str, str, str]:
        """Build segmentation-level demographic + psychographic profile.

        Following Li & Cheng 2026 audience segmentation approach (outline §2.2):
        segment-level descriptive attributes, not behavioral rules.

        Args:
            cluster_id: Cluster identifier.
            member_features: Feature rows for the cluster's members.
            total_population: Total clustered population across all clusters.
                When provided, the demographics block reports the cluster's
                true share of the population (B3 fix — the previous formula
                always evaluated to 1.0 because it divided the cluster's
                own member count by itself).
        """
        stats = self._cluster_stats(member_features)
        if not stats:
            return (
                f"Segment_{cluster_id}",
                f"Users with similar behavior patterns",
                f"Shared behavioral characteristics",
            )

        # Determine segment archetype
        conflict_val = stats["conflict_engagement_ratio"]
        activity_val = stats["avg_messages"]
        question_val = stats["question_rate"]

        if conflict_val > 0.35 and question_val < 0.05:
            archetype = "Confrontational Defenders"
        elif conflict_val < 0.15 and activity_val > 8:
            archetype = "Active Contributors"
        elif question_val > 0.2:
            archetype = "Inquisitive Discussants"
        elif activity_val < 3:
            archetype = "Occasional Observers"
        else:
            archetype = "Engaged Participants"

        # B3 fix: compute the cluster's actual share of the total clustered
        # population instead of the legacy self-ratio (always 1.0).
        n_members = stats["n_members"]
        if total_population and total_population > 0:
            population_share = n_members / total_population
        else:
            population_share = 1.0
        demographics = (
            f"Segment size: {n_members} users "
            f"({population_share:.0%} of clustered population)\n"
            f"Average activity: {activity_val:.0f} messages across "
            f"{stats['avg_threads']:.0f} threads\n"
            f"Average reply depth: {stats['reply_depth']:.1f}\n"
            f"Message verbosity (log characters): {stats['verbosity']:.2f}\n"
            f"Policy citation rate: {stats['wp_citation_rate']:.1%}"
        )

        psychographics = (
            f"Conflict engagement: {conflict_val:.1%} of messages in contested discussions\n"
            f"Inquisitiveness: {question_val:.1%} question rate\n"
            f"Behavioral archetype: {archetype}\n"
            f"Interaction style: "
            f"{'confrontational' if conflict_val > 0.35 else 'collaborative' if conflict_val < 0.15 else 'mixed'}"
        )

        return f"Segment_{cluster_id}_{archetype}", demographics, psychographics
