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
        )

        # Per-tier alpha + backend kwargs for CADP agents
        alpha_kwargs = dict(
            alpha=self.alpha,
            alpha_tier1=self.alpha_tier1,
            alpha_tier2=self.alpha_tier2,
            alpha_tier3=self.alpha_tier3,
            backend=self.backend,
            seed=agent_seed,
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
            # Population-Aligned Persona (arXiv:2509.10127):
            # Match attribute distribution, not behavioral rules
            cluster_stats, sampled = _pop_attributes(cluster_id, rng)
            return PopAlignedPersonaAgent(
                cluster_attributes=cluster_stats,
                sampled_attributes=sampled,
                **common_kwargs,
            )

        elif condition == "rich_narrative":
            # Lever-1 ceiling / kill condition (reframe v1, 2026-07-08):
            # maximalist narrative persona from the SAME cluster stats as
            # the descriptive condition, rendered as multi-paragraph
            # narrative with concrete episodes and example moves. NO
            # compiled .skill rules, NO filter-retry. Same data source
            # as cadp_full_nuwa → isolates "rules vs description" as
            # the sole variable in the kill comparison.
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

        vectors = np.array([f.to_vector() for f in member_features])
        attr_names = [
            "reply_depth", "edit_frequency",
            "stance_shift_rate", "conflict_engagement_ratio",
        ]
        stats = {}
        for i, name in enumerate(attr_names):
            col = vectors[:, i]
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

        Maps user message_count to [0.1, 1.0] engagement ratio, producing
        heterogeneous participation (long-tail) rather than uniform 0.5.
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

        msg_arr = np.array(msg_counts, dtype=float)
        msg_min = float(np.min(msg_arr))
        msg_max = float(np.max(msg_arr))

        # Sample from log-normal-ish distribution matching real data
        log_counts = np.log1p(msg_arr)
        log_mean = float(np.mean(log_counts))
        log_std = float(np.std(log_counts))

        sampled_log = rng.gauss(log_mean, max(log_std, 0.1))
        sampled_count = np.expm1(sampled_log)

        # Normalize to [0.1, 1.0]
        if msg_max > msg_min:
            ratio = (sampled_count - msg_min) / (msg_max - msg_min)
        else:
            ratio = 0.5

        return float(max(0.1, min(1.0, ratio)))

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
        flexibility = _level(
            stats["stance_shift_rate"],
            low=0.15, high=0.4,
        )

        lines = [
            f"Age group: adult internet user",
            f"Activity level: {activity} (avg {stats['avg_messages']:.0f} messages, "
            f"{stats['avg_threads']:.0f} threads participated)",
            f"Conflict involvement: {conflict} "
            f"(engages in {stats['conflict_engagement_ratio']:.1%} of contested discussions)",
            f"Opinion flexibility: {flexibility} "
            f"(stance shift rate: {stats['stance_shift_rate']:.2f})",
            f"Typical reply depth: {stats['reply_depth']:.1f}",
            f"Editing tendency: {stats['edit_frequency']:.2f} edits per message",
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
        flexibility = _level(stats["stance_shift_rate"], low=0.15, high=0.4)

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

        # Stance-flexibility narrative
        stance = stats["stance_shift_rate"]
        if flexibility == "high":
            stance_passage = (
                f"You change your mind in public. In about {stance:.0%} of "
                f"disputes you've been part of, you've shifted position at "
                f"least once after hearing counter-arguments, and you're "
                f"willing to say so — 'fair point, I was wrong about that' "
                f"is something you'll actually write."
            )
        elif flexibility == "low":
            stance_passage = (
                f"You hold your positions. Your stance shift rate is only "
                f"{stance:.0%}; once you've laid out a view you tend to "
                f"defend it, and you rarely concede in-thread even if you "
                f"privately revise later."
            )
        else:
            stance_passage = (
                f"You're persuadable but not wishy-washy — you revise your "
                f"position in about {stance:.0%} of disputes, usually when "
                f"someone brings a source or a point you hadn't considered."
            )

        # Depth + editing
        depth_passage = (
            f"You tend to reply about {stats['reply_depth']:.1f} levels deep "
            f"in a thread before you feel a point is settled, and you edit "
            f"your own messages about {stats['edit_frequency']:.2f} times "
            f"each — small fixes for wording, sources, or formatting, not "
            f"full rewrites."
        )

        # Typical opening move — derived from conflict × flexibility combo.
        # Stays descriptive ("you tend to..."), never prescriptive, so it
        # remains lever-1 narrative rather than drifting into lever-2 rules.
        if conflict == "high" and flexibility != "low":
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
        flexibility_val = stats["stance_shift_rate"]

        if conflict_val > 0.35 and flexibility_val < 0.2:
            archetype = "Confrontational Defenders"
        elif conflict_val < 0.15 and activity_val > 8:
            archetype = "Active Contributors"
        elif flexibility_val > 0.35:
            archetype = "Flexible Mediators"
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
            f"Editing frequency: {stats['edit_frequency']:.2f} per message"
        )

        psychographics = (
            f"Conflict engagement: {conflict_val:.1%} of messages in contested discussions\n"
            f"Stance flexibility: {flexibility_val:.2f} stance shift rate\n"
            f"Behavioral archetype: {archetype}\n"
            f"Tendency to defend positions vs. adapt: "
            f"{'defends' if flexibility_val < 0.2 else 'adapts' if flexibility_val > 0.35 else 'balanced'}\n"
            f"Interaction style: "
            f"{'confrontational' if conflict_val > 0.35 else 'collaborative' if conflict_val < 0.15 else 'mixed'}"
        )

        return f"Segment_{cluster_id}_{archetype}", demographics, psychographics
