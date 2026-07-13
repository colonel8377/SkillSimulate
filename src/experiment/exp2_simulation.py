"""Experiment 2 runner — social simulation with temporal trajectory analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from loguru import logger

from src.clustering.clusterer import BehavioralClusterer
from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.evaluation.aggregator import MetricsAggregator
from src.experiment.conditions import is_cadp_condition, is_replay_only_condition
from src.experiment.runner import ExperimentCell, ExperimentRunner
from src.simulation.population import PopulationBuilder
from src.simulation.sandbox import SimulationResult, SimulationSandbox
from src.skill.compiler import SkillCompiler


class Experiment2Runner(ExperimentRunner):
    """Runs Experiment 2: real community dynamics simulation."""

    def __init__(self, config: ExperimentConfig, models_config: str = "configs/models.yaml"):
        super().__init__(config, models_config)
        self.metrics_agg = MetricsAggregator(
            held_out_events_dir=str(settings.held_out_events_dir),
            role_labels_dir=str(settings.role_labels_dir),
            model_provenance=self.llm.get_all_provenance(),
        )
        self._skill_cache: dict[str, dict[str, Any]] = {}
        self._data_cache: dict[str, list] = {}
        self._cluster_cache: dict[str, Any] = {}
        self.scale_test = config.scale_test
        self.scale_sizes = config.scale_test_sizes

    async def run_cell(self, cell: ExperimentCell) -> dict:
        """Run one experiment cell with trajectory tracking."""
        threads = self._load_data(cell.dataset)

        # Select controversial scenarios
        scenario_threads = self._select_controversial(threads)

        # Outline §6.2 fourth arm: real_history replays observed traces
        # through the metric pipeline as both sim and ground truth. This
        # gives a self-similarity ceiling against which the three
        # simulation arms (cadp_full, pop_aligned, cadp_minus_ap) can be
        # compared. No sandbox run, no agent construction, no skills.
        # (G5 — previously real data served only as ground truth, the
        # arm itself was unscheduled.)
        if is_replay_only_condition(cell.condition):
            return await self._run_real_history_cell(cell, scenario_threads)

        # Cluster (cached per dataset — same threads produce same clusters)
        cluster_result = self._get_or_compute_clusters(cell.dataset, scenario_threads)
        thread_cluster_map = self._assign_threads_to_clusters(scenario_threads, cluster_result)

        # Compile skills
        skills = {}
        if is_cadp_condition(cell.condition):
            skills = await self._get_or_compile_skills(cell.dataset, scenario_threads, cluster_result, thread_cluster_map)

        # Run at each scale
        results = {}
        scales = self.scale_sizes if self.scale_test else [self.config.population_size]

        for pop_size in scales:
            pop_builder = PopulationBuilder(
                llm_client=self.llm,
                model_name=cell.model,
                skills=skills,
                alpha=self.config.alpha,
                alpha_tier1=self.config.alpha_tier1,
                alpha_tier2=self.config.alpha_tier2,
                alpha_tier3=self.config.alpha_tier3,
                backend=self.config.backend,
                memory_strategy=self.config.memory_strategy,
                compaction_interval=self.config.compaction_interval,
                compaction_keep_recent=self.config.compaction_keep_recent,
            )
            agents = pop_builder.build_population(
                cluster_result=cluster_result,
                size=pop_size,
                condition=cell.condition,
                seed=self.config.seed + cell.repeat,
            )

            sim_threads = self._prepare_sim_threads(scenario_threads, cell)

            topology = self.get_platform_topology(cell.dataset)
            sandbox = SimulationSandbox(
                platform=topology,
                checkpoint_dir=str(self.checkpoint_dir),
                max_concurrency=self.config.max_concurrency,
            )

            run_id = f"{cell.cell_id}_n{pop_size}"
            result = await sandbox.run(
                agents=agents,
                threads=sim_threads,
                num_rounds=self.config.num_rounds,
                run_id=run_id,
                condition=cell.condition,
                dataset=cell.dataset,
                model=cell.model,
                repeat=cell.repeat,
                checkpoint_every=self.config.checkpoint_every,
                seed=self.config.seed + cell.repeat,
            )

            report = await self.metrics_agg.evaluate(result, scenario_threads)

            # Extract temporal trajectory analysis (outline §6.4)
            temporal = self._extract_temporal_analysis(result)

            results[f"n{pop_size}"] = {
                **report.to_dict(),
                "temporal_analysis": temporal,
            }

        return {
            "cell_id": cell.cell_id,
            "condition": cell.condition,
            "dataset": cell.dataset,
            "model": cell.model,
            "repeat": cell.repeat,
            "scale_results": results,
        }

    async def _run_real_history_cell(
        self,
        cell: ExperimentCell,
        scenario_threads: list,
    ) -> dict:
        """Replay observed real traces as the real_history arm (outline §6.2).

        Builds a ``SimulationResult`` whose ``messages`` are exactly the
        observed real messages (converted to the dict shape the metric
        aggregator expects) and whose ``agent_states`` mirror the real
        participants. The interaction graph is therefore identical to
        the real one, so structural / linguistic / action-distribution
        metrics land at their self-similarity ceiling — the upper bound
        any simulation arm can achieve.

        Caveat: community-dependent Macro metrics (e.g. E-I Polarization)
        are NOT exactly at ceiling here because ``agent_states`` carries
        a uniform ``cluster_id=0`` while the real side uses external role
        labels or Louvain communities. The arm still serves its purpose
        as the §6.2 reference baseline; treat near-ceiling (rather than
        exact ceiling) on community-dependent metrics as expected.
        """
        from src.data.schemas import ActionType

        replay_messages: list[dict] = []
        agent_states: list[dict] = []
        seen_agents: set[str] = set()

        for thread in scenario_threads:
            for msg in thread.messages:
                replay_messages.append({
                    "msg_id": msg.msg_id,
                    "thread_id": msg.thread_id,
                    "user_id": msg.user_id,
                    "platform": msg.platform.value,
                    "text": msg.text,
                    "action_type": msg.action_type.value,
                    "parent_msg_id": msg.parent_msg_id,
                    "round": 0,
                })
                if msg.user_id not in seen_agents:
                    seen_agents.add(msg.user_id)
                    agent_states.append({
                        "agent_id": msg.user_id,
                        "cluster_id": 0,
                    })

        replay_result = SimulationResult(
            run_id=f"{cell.cell_id}_real_history",
            condition=cell.condition,
            dataset=cell.dataset,
            model=cell.model,
            repeat=cell.repeat,
            rounds=1,
            messages=replay_messages,
            agent_states=agent_states,
            interaction_graph=None,
            enforcement_stats={"replay_only": True},
            per_round_metrics=[],
        )

        report = await self.metrics_agg.evaluate(replay_result, scenario_threads)
        temporal = self._extract_temporal_analysis(replay_result)

        return {
            "cell_id": cell.cell_id,
            "condition": cell.condition,
            "dataset": cell.dataset,
            "model": cell.model,
            "repeat": cell.repeat,
            "scale_results": {
                "n_real": {
                    **report.to_dict(),
                    "temporal_analysis": temporal,
                    "note": (
                        "real_history replay arm (outline §6.2). Metrics "
                        "are self-similarity ceiling (real vs real)."
                    ),
                },
            },
        }

    def _select_controversial(self, threads: list) -> list:
        """Select controversial scenarios from real data."""
        # Sort by participation and message count
        scored = sorted(
            threads,
            key=lambda t: (len(t.participants), len(t.messages)),
            reverse=True,
        )
        # Take top 20%
        top_k = max(len(scored) // 5, 5)
        return scored[:top_k]

    def _make_clusterer(self) -> BehavioralClusterer:
        """Build a BehavioralClusterer from the experiment config."""
        return BehavioralClusterer(
            method=self.config.cluster_method,
            n_clusters=self.config.num_clusters,
            role_min_cluster_size=self.config.role_min_cluster_size,
            role_min_samples=self.config.role_min_samples,
            style_min_cluster_size=self.config.style_min_cluster_size,
            style_min_samples=self.config.style_min_samples,
            target_min_leaves=self.config.target_min_leaves,
            target_max_leaves=self.config.target_max_leaves,
            scaler=self.config.scaler,
            impute_orphans=self.config.impute_orphans,
            cluster_selection_method=self.config.cluster_selection_method,
            min_style_silhouette=self.config.min_style_silhouette,
            style_umap_dim=self.config.style_umap_dim,
            random_state=self.config.seed,
        )

    def _get_or_compute_clusters(self, dataset: str, threads: list):
        """Get cached clustering result or compute new one."""
        if dataset not in self._cluster_cache:
            clusterer = self._make_clusterer()
            self._cluster_cache[dataset] = clusterer.fit(threads)
            logger.info(
                f"Clustering {dataset}: "
                f"{self._cluster_cache[dataset].n_clusters} clusters, "
                f"silhouette={self._cluster_cache[dataset].silhouette_score:.3f}"
            )
        return self._cluster_cache[dataset]

    def _load_data(self, dataset: str):
        if dataset not in self._data_cache:
            loader = self.get_dataset_loader(dataset)
            threads = loader.load()
            if self.config.max_threads is not None and len(threads) > self.config.max_threads:
                import numpy as np
                rng = np.random.RandomState(self.config.seed)
                idx = rng.choice(len(threads), size=self.config.max_threads, replace=False)
                threads = [threads[i] for i in idx]
                logger.info(
                    f"Capped {dataset} to {self.config.max_threads} threads (from full corpus)"
                )
            self._data_cache[dataset] = threads
        return self._data_cache[dataset]

    async def _get_or_compile_skills(self, dataset, threads, cluster_result, thread_cluster_map):
        if dataset not in self._skill_cache:
            from src.config.settings import settings as proj_settings

            cluster_ids = [str(cid) for cid in cluster_result.get_cluster_ids()]
            if SkillCompiler.skills_exist(proj_settings.skills_dir, dataset, cluster_ids):
                logger.info(f"Loading existing skills for {dataset} from disk")
                skills = SkillCompiler.load_all_skills(proj_settings.skills_dir, platform=dataset)
            else:
                logger.info(f"Compiling new skills for {dataset}")
                compiler = SkillCompiler(
                    llm_client=self.llm,
                    model_name=self.config.compile_model or self.config.models[0],
                    top_n_threads=self.config.top_n_threads,
                )
                skills = await compiler.compile_all(
                    platform=dataset,
                    cluster_result=cluster_result,
                    all_threads=threads,
                    thread_cluster_map=thread_cluster_map,
                )
                for cid, skill in skills.items():
                    compiler.save_skill(skill)
            self._skill_cache[dataset] = skills
        return self._skill_cache[dataset]

    def _assign_threads_to_clusters(self, threads, cluster_result) -> dict:
        thread_map = {}
        for thread in threads:
            votes = Counter()
            for uid in thread.participants:
                cid = cluster_result.labels.get(uid)
                if cid is not None:
                    votes[cid] += 1
            thread_map[thread.thread_id] = votes.most_common(1)[0][0] if votes else 0
        return thread_map

    def _prepare_sim_threads(self, threads, cell):
        from src.data.schemas import Thread as ThreadModel
        sim_threads = []
        for thread in threads[:5]:
            sim_threads.append(ThreadModel(
                thread_id=f"sim_{cell.cell_id}_{thread.thread_id}",
                platform=thread.platform,
                topic=thread.topic,
            ))
        return sim_threads

    def _extract_temporal_analysis(self, result) -> dict:
        """Extract temporal trajectory metrics from simulation (outline §6.4).

        Returns:
            Dict with:
            - trajectory: per-round polarization/conflict curves
            - key_events: first occurrence rounds for conflict, persuasion, escalation
            - escalation_episodes: rounds where conflict ratio exceeded 2 consecutive rounds
        """
        per_round = result.per_round_metrics
        messages = result.messages

        # B2 fix: ``close``/``reopen`` are normal GitHub issue-lifecycle
        # actions (outline §4.3), not conflict signals — kept in sync
        # with sandbox._compute_round_metrics' conflict_actions set.
        conflict_actions = {"disagree", "revert", "counter_argue", "report", "block"}
        persuasion_actions = {"award_delta"}

        # Polarization trajectory
        polarization_curve = [r.get("polarization_proxy", 0.0) for r in per_round]
        conflict_curve = [r.get("conflict_ratio", 0.0) for r in per_round]

        # Key event timing: first round each event type occurs
        first_conflict = None
        first_persuasion = None
        first_escalation = None

        for r in per_round:
            round_num = r["round"]
            if first_conflict is None and r.get("conflict_count", 0) > 0:
                first_conflict = round_num
            # Persuasion: check messages for delta awards
            if first_persuasion is None:
                round_msgs = [m for m in messages if m.get("round") == round_num]
                if any(m["action_type"] in persuasion_actions for m in round_msgs):
                    first_persuasion = round_num

        # Escalation: 2+ consecutive rounds with conflict_ratio > 0.3
        consecutive_high = 0
        for i, r in enumerate(per_round):
            if r.get("conflict_ratio", 0.0) > 0.3:
                consecutive_high += 1
                if consecutive_high >= 2 and first_escalation is None:
                    first_escalation = r["round"]
            else:
                consecutive_high = 0

        # Escalation episodes (sustained conflict periods)
        escalation_episodes = []
        ep_start = None
        for i, r in enumerate(per_round):
            if r.get("conflict_ratio", 0.0) > 0.3:
                if ep_start is None:
                    ep_start = r["round"]
            else:
                if ep_start is not None:
                    escalation_episodes.append({"start": ep_start, "end": r["round"] - 1})
                    ep_start = None
        if ep_start is not None:
            escalation_episodes.append({"start": ep_start, "end": per_round[-1]["round"]})

        return {
            "trajectory": {
                "polarization": polarization_curve,
                "conflict_ratio": conflict_curve,
            },
            "key_events": {
                "first_conflict": first_conflict,
                "first_persuasion": first_persuasion,
                "first_escalation": first_escalation,
            },
            "escalation_episodes": escalation_episodes,
        }
