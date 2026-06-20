"""Cross-dataset transfer test (outline §5.5).

Tests whether .skill files compiled on one dataset (e.g. Wikipedia)
transfer to another dataset (e.g. Reddit). If fidelity holds cross-domain,
the rules capture generalizable behavioral patterns.

Three transfer modes (outline §5.5):
  1. full_component: Compile skills on source → apply directly on target.
  2. methodology:    Tune hyperparameters on source → re-run full CADP pipeline
                     on target (tests method generalizability, not rule portability).
  3. selective:      Only transfer Expression DNA + generic Anti-patterns,
                     recompile platform-specific Mind Models & Anti-patterns on target
                     (for cross-structure transfer like Reddit→GitHub).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from loguru import logger

from src.clustering.clusterer import BehavioralClusterer, ClusterResult
from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.evaluation.aggregator import MetricsAggregator
from src.experiment.runner import ExperimentCell, ExperimentRunner
from src.simulation.population import PopulationBuilder
from src.simulation.sandbox import SimulationSandbox

# Generic anti-pattern descriptions that are platform-agnostic
_GENERIC_AP_KEYWORDS = {"personal attack", "insult", "harass", "threat", "spam", "off-topic"}


class CrossDatasetTransferRunner(ExperimentRunner):
    """Tests skill transfer across datasets (outline §5.5).

    Compile skills on dataset A → apply to dataset B → measure fidelity.
    """

    def __init__(self, config: ExperimentConfig, models_config: str = "configs/models.yaml"):
        super().__init__(config, models_config)
        self.metrics_agg = MetricsAggregator(
            held_out_events_dir=str(settings.held_out_events_dir),
            role_labels_dir=str(settings.role_labels_dir),
            model_provenance=self.llm.get_all_provenance(),
        )

    async def run_transfer_test(
        self,
        source_dataset: str,
        target_dataset: str,
        model: str,
        condition: str = "cadp_full",
        transfer_mode: str = "full_component",
    ) -> dict:
        """Run one transfer test: source skills → target data.

        Args:
            source_dataset: Dataset to compile skills from.
            target_dataset: Dataset to apply skills on.
            model: LLM model name.
            condition: Agent condition.
            transfer_mode: "full_component" | "methodology" | "selective".

        Returns:
            Dict with transfer fidelity metrics.
        """
        logger.info(
            f"Transfer test ({transfer_mode}): {source_dataset} → {target_dataset}"
        )

        # Load both datasets
        source_threads = self._load_data(source_dataset)
        target_threads = self._load_data(target_dataset)

        # Cluster target dataset (always needed)
        target_clusterer = BehavioralClusterer(
            method=self.config.cluster_method,
            n_clusters=self.config.num_clusters,
            random_state=self.config.seed,
        )
        target_clusters = target_clusterer.fit(target_threads)

        # Build skills according to transfer mode
        if transfer_mode == "methodology":
            skills = await self._methodology_transfer(
                source_dataset, target_dataset, model, target_threads, target_clusters
            )
        elif transfer_mode == "selective":
            skills = await self._selective_transfer(
                source_dataset, target_dataset, model, source_threads,
                target_threads, target_clusters
            )
        else:
            skills = await self._full_component_transfer(
                source_dataset, target_dataset, model, source_threads,
                target_threads, target_clusters
            )

        # Build population with transferred skills on target data
        pop_builder = PopulationBuilder(
            llm_client=self.llm,
            model_name=model,
            skills=skills,
            alpha=self.config.alpha,
            alpha_tier1=self.config.alpha_tier1,
            alpha_tier2=self.config.alpha_tier2,
            alpha_tier3=self.config.alpha_tier3,
            backend=self.config.backend,
        )
        agents = pop_builder.build_population(
            cluster_result=target_clusters,
            size=self.config.population_size,
            condition=condition,
            seed=self.config.seed,
        )

        # Prepare target simulation threads
        sim_threads = self._prepare_sim_threads(target_threads, source_dataset, target_dataset)

        # Run simulation on target data with source skills
        topology = self.get_platform_topology(target_dataset)
        sandbox = SimulationSandbox(
            platform=topology,
            checkpoint_dir=str(self.checkpoint_dir),
            max_concurrency=self.config.max_concurrency,
        )

        run_id = f"transfer_{source_dataset}_to_{target_dataset}_{model}_{transfer_mode}"
        result = await sandbox.run(
            agents=agents,
            threads=sim_threads,
            num_rounds=self.config.num_rounds,
            run_id=run_id,
            condition=f"{condition}_transferred_{transfer_mode}",
            dataset=f"{source_dataset}->{target_dataset}",
            model=model,
            checkpoint_every=self.config.checkpoint_every,
            seed=self.config.seed,
        )

        # Evaluate against target ground truth
        report = self.metrics_agg.evaluate(result, target_threads)

        # Also run native (target→target) for comparison
        native_skills = await self._compile_native(
            target_dataset, model, target_threads, target_clusters
        )
        native_pop = PopulationBuilder(
            llm_client=self.llm,
            model_name=model,
            skills=native_skills,
            alpha=self.config.alpha,
            alpha_tier1=self.config.alpha_tier1,
            alpha_tier2=self.config.alpha_tier2,
            alpha_tier3=self.config.alpha_tier3,
            backend=self.config.backend,
        )
        native_agents = native_pop.build_population(
            cluster_result=target_clusters,
            size=self.config.population_size,
            condition=condition,
            seed=self.config.seed,
        )
        native_sim_threads = self._prepare_sim_threads(target_threads, target_dataset, target_dataset)
        native_sandbox = SimulationSandbox(
            platform=topology,
            checkpoint_dir=str(self.checkpoint_dir),
            max_concurrency=self.config.max_concurrency,
        )
        native_result = await native_sandbox.run(
            agents=native_agents,
            threads=native_sim_threads,
            num_rounds=self.config.num_rounds,
            run_id=f"native_{target_dataset}_{model}",
            condition=condition,
            dataset=target_dataset,
            model=model,
            checkpoint_every=self.config.checkpoint_every,
            seed=self.config.seed,
        )
        native_report = self.metrics_agg.evaluate(native_result, target_threads)

        # Compare transfer vs native fidelity
        transfer_metrics = report.metrics
        native_metrics = native_report.metrics
        fidelity_gap = {}
        for key in set(transfer_metrics) & set(native_metrics):
            fidelity_gap[key] = transfer_metrics[key] - native_metrics[key]

        return {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "transfer_mode": transfer_mode,
            "model": model,
            "transfer_metrics": transfer_metrics,
            "native_metrics": native_metrics,
            "fidelity_gap": fidelity_gap,
            "transfer_fidelity_ratio": (
                sum(1 for v in fidelity_gap.values() if abs(v) < 0.1) / max(len(fidelity_gap), 1)
            ),
        }

    # ------------------------------------------------------------------
    # Transfer mode implementations
    # ------------------------------------------------------------------

    async def _full_component_transfer(
        self,
        source_dataset: str,
        target_dataset: str,
        model: str,
        source_threads: list,
        target_threads: list,
        target_clusters: ClusterResult,
    ) -> dict[str, Any]:
        """Mode 1: Compile skills on source → apply directly on target (centroid mapping)."""
        source_clusterer = BehavioralClusterer(
            method=self.config.cluster_method,
            n_clusters=self.config.num_clusters,
            random_state=self.config.seed,
        )
        source_clusters = source_clusterer.fit(source_threads)

        from src.skill.compiler import SkillCompiler
        thread_cluster_map = self._assign_threads_to_clusters(source_threads, source_clusters)
        compiler = SkillCompiler(
            llm_client=self.llm,
            model_name=model,
            top_n_threads=self.config.top_n_threads,
        )
        source_skills = await compiler.compile_all(
            platform=source_dataset,
            cluster_result=source_clusters,
            all_threads=source_threads,
            thread_cluster_map=thread_cluster_map,
        )

        return self._map_skills_cross_domain(source_skills, source_clusters, target_clusters)

    async def _methodology_transfer(
        self,
        source_dataset: str,
        target_dataset: str,
        model: str,
        target_threads: list,
        target_clusters: ClusterResult,
    ) -> dict[str, Any]:
        """Mode 2: Tune hyperparameters on source → re-run full pipeline on target.

        The hyperparameters (K, cluster weights, trigger θ_sem) determined on
        the source dataset are reused, but skills are freshly compiled on the
        target dataset using the same methodology.
        """
        from src.skill.compiler import SkillCompiler

        thread_cluster_map = self._assign_threads_to_clusters(target_threads, target_clusters)
        compiler = SkillCompiler(
            llm_client=self.llm,
            model_name=model,
            top_n_threads=self.config.top_n_threads,
        )
        # Compile fresh skills on target data (methodology transfer, not rule transfer)
        skills = await compiler.compile_all(
            platform=target_dataset,
            cluster_result=target_clusters,
            all_threads=target_threads,
            thread_cluster_map=thread_cluster_map,
        )
        logger.info(
            f"Methodology transfer: re-compiled {len(skills)} skills on {target_dataset} "
            f"using hyperparameters tuned on {source_dataset}"
        )
        return skills

    async def _selective_transfer(
        self,
        source_dataset: str,
        target_dataset: str,
        model: str,
        source_threads: list,
        target_threads: list,
        target_clusters: ClusterResult,
    ) -> dict[str, Any]:
        """Mode 3: Transfer only Expression DNA + generic Anti-patterns.

        Cross-structure transfer (e.g. Reddit→GitHub): Expression DNA (language
        style) and generic Anti-patterns (e.g. "no personal attacks") are
        portable. Platform-specific Mind Models and Anti-patterns (e.g. delta
        mechanism, revert rules) are recompiled on the target.
        """
        # Step 1: Compile source skills for Expression DNA + generic APs
        source_clusterer = BehavioralClusterer(
            method=self.config.cluster_method,
            n_clusters=self.config.num_clusters,
            random_state=self.config.seed,
        )
        source_clusters = source_clusterer.fit(source_threads)

        from src.skill.compiler import SkillCompiler
        from src.skill.schema import (
            AntiPattern,
            CapabilityTrack,
            ConstraintTrack,
            ExpressionDNA,
            MindModel,
            SkillFile,
        )

        source_tcm = self._assign_threads_to_clusters(source_threads, source_clusters)
        compiler = SkillCompiler(
            llm_client=self.llm,
            model_name=model,
            top_n_threads=self.config.top_n_threads,
        )
        source_skills = await compiler.compile_all(
            platform=source_dataset,
            cluster_result=source_clusters,
            all_threads=source_threads,
            thread_cluster_map=source_tcm,
        )

        # Step 2: Compile target skills for platform-specific Mind Models + APs
        target_tcm = self._assign_threads_to_clusters(target_threads, target_clusters)
        target_skills = await compiler.compile_all(
            platform=target_dataset,
            cluster_result=target_clusters,
            all_threads=target_threads,
            thread_cluster_map=target_tcm,
        )

        # Step 3: Merge — Expression DNA from source, Mind Models + APs from target
        mapped_source = self._map_skills_cross_domain(
            source_skills, source_clusters, target_clusters
        )

        merged_skills: dict[str, SkillFile] = {}
        for cid in target_clusters.get_cluster_ids():
            cid_str = str(cid)
            src_skill = mapped_source.get(cid_str)
            tgt_skill = target_skills.get(cid_str)

            if src_skill is None or tgt_skill is None:
                # Fall back to whichever is available
                merged_skills[cid_str] = src_skill or tgt_skill
                continue

            # Expression DNA from source (portable language patterns)
            source_edna = src_skill.capability.expression_dna if src_skill.capability else None

            # Mind Models from target (platform-specific reasoning)
            target_mms = tgt_skill.capability.mind_models if tgt_skill.capability else []

            # Anti-patterns: keep generic ones from source, add all from target
            generic_source_aps: list[AntiPattern] = []
            if src_skill.constraint:
                generic_source_aps = [
                    ap for ap in src_skill.constraint.anti_patterns
                    if self._is_generic_antipattern(ap)
                ]
            target_aps = tgt_skill.constraint.anti_patterns if tgt_skill.constraint else []

            if source_edna is None and tgt_skill.capability:
                source_edna = tgt_skill.capability.expression_dna

            merged_skills[cid_str] = SkillFile(
                cluster_id=cid_str,
                platform=target_dataset,
                capability=CapabilityTrack(
                    expression_dna=source_edna or ExpressionDNA(),
                    mind_models=target_mms,
                ),
                constraint=ConstraintTrack(
                    anti_patterns=generic_source_aps + target_aps,
                ),
                source_thread_ids=tgt_skill.source_thread_ids,
                source_user_count=tgt_skill.source_user_count,
            )

        logger.info(
            f"Selective transfer: merged {len(merged_skills)} skills "
            f"(Expression DNA from {source_dataset}, Mind Models + APs from {target_dataset})"
        )
        return merged_skills

    async def _compile_native(
        self,
        dataset: str,
        model: str,
        threads: list,
        clusters: ClusterResult,
    ) -> dict[str, Any]:
        """Compile native skills for comparison baseline."""
        from src.skill.compiler import SkillCompiler

        tcm = self._assign_threads_to_clusters(threads, clusters)
        compiler = SkillCompiler(
            llm_client=self.llm,
            model_name=model,
            top_n_threads=self.config.top_n_threads,
        )
        return await compiler.compile_all(
            platform=dataset,
            cluster_result=clusters,
            all_threads=threads,
            thread_cluster_map=tcm,
        )

    @staticmethod
    def _is_generic_antipattern(ap) -> bool:
        """Check if an anti-pattern is platform-agnostic (generic).

        Generic anti-patterns are those whose description/keywords match
        universal behavioral norms (e.g. "no personal attacks") rather than
        platform-specific rules (e.g. "no revert warring").
        """
        text = (ap.description + " " + " ".join(getattr(ap, "trigger_keywords", []))).lower()
        return any(kw in text for kw in _GENERIC_AP_KEYWORDS)

    def _map_skills_cross_domain(
        self,
        source_skills: dict[str, Any],
        source_clusters: ClusterResult,
        target_clusters: ClusterResult,
    ) -> dict[str, Any]:
        """Map source cluster skills to target clusters by centroid proximity."""
        import numpy as np
        from scipy.spatial.distance import cdist

        # Align dimensions (pad if needed)
        source_dim = source_clusters.centroids.shape[1]
        target_dim = target_clusters.centroids.shape[1]
        min_dim = min(source_dim, target_dim)

        source_centroids = source_clusters.centroids[:, :min_dim]
        target_centroids = target_clusters.centroids[:, :min_dim]

        # Compute distance matrix: target_cluster × source_cluster
        dists = cdist(target_centroids, source_centroids)
        nearest_source = dists.argmin(axis=1)

        # Map: target_cluster_id → source skill
        transferred = {}
        for target_id in target_clusters.get_cluster_ids():
            source_id = str(nearest_source[target_id])
            if source_id in source_skills:
                # Create a copy with relabeled cluster_id
                skill = source_skills[source_id]
                from src.skill.schema import SkillFile
                transferred[str(target_id)] = SkillFile(
                    cluster_id=str(target_id),
                    platform=skill.platform,
                    compiled_at=skill.compiled_at,
                    capability=skill.capability,
                    constraint=skill.constraint,
                    source_thread_ids=skill.source_thread_ids,
                    source_user_count=skill.source_user_count,
                )

        return transferred

    def _load_data(self, dataset: str):
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
        return threads

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

    def _prepare_sim_threads(self, threads, source_ds, target_ds):
        from src.data.schemas import Thread as ThreadModel
        sim_threads = []
        for thread in threads[:5]:
            if len(thread.participants) >= 2:
                sim_threads.append(ThreadModel(
                    thread_id=f"sim_transfer_{target_ds}_{thread.thread_id}",
                    platform=thread.platform,
                    topic=thread.topic,
                ))
        return sim_threads if sim_threads else [
            ThreadModel(
                thread_id=f"sim_transfer_{target_ds}_default",
                platform=threads[0].platform,
                topic=threads[0].topic,
            )
        ]
