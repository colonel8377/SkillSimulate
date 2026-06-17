"""Experiment 1 runner — method validation grid."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from loguru import logger

from src.clustering.clusterer import BehavioralClusterer
from src.clustering.validation import ClusterStabilityValidator
from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.evaluation.aggregator import MetricsAggregator
from src.experiment.conditions import is_cadp_condition
from src.experiment.runner import ExperimentCell, ExperimentRunner
from src.simulation.population import PopulationBuilder
from src.simulation.sandbox import SimulationSandbox
from src.skill.compiler import SkillCompiler
from src.utils.io import save_json


class Experiment1Runner(ExperimentRunner):
    """Runs Experiment 1: method validation across conditions/datasets/models."""

    def __init__(self, config: ExperimentConfig, models_config: str = "configs/models.yaml"):
        super().__init__(config, models_config)
        self.metrics_agg = MetricsAggregator(
            held_out_events_dir=str(settings.held_out_events_dir),
            role_labels_dir=str(settings.role_labels_dir),
        )
        # Cache compiled skills per dataset
        self._skill_cache: dict[str, dict[str, Any]] = {}
        self._data_cache: dict[str, list] = {}
        self._cluster_cache: dict[str, Any] = {}  # dataset → ClusterResult
        self._stability_cache: dict[str, dict] = {}  # dataset → stability report

    async def run_cell(self, cell: ExperimentCell) -> dict:
        """Run one experiment cell: cluster → compile skill → simulate → evaluate."""
        # Load data
        threads = self._load_data(cell.dataset)

        # Cluster (cached per dataset — shared across conditions)
        cluster_result = self._get_or_compute_clusters(cell.dataset, threads)

        # Validate cluster stability (once per dataset, outline §5.4)
        stability = self._validate_cluster_stability(cell.dataset, threads)

        # Build thread→cluster map
        thread_cluster_map = self._assign_threads_to_clusters(threads, cluster_result)

        # Compile skills if CADP condition (shared per dataset)
        skills = {}
        if is_cadp_condition(cell.condition):
            skills = await self._get_or_compile_skills(cell.dataset, threads, cluster_result, thread_cluster_map)

        # Build population
        pop_builder = PopulationBuilder(
            llm_client=self.llm,
            model_name=cell.model,
            skills=skills,
            alpha=self.config.alpha,
            alpha_tier1=self.config.alpha_tier1,
            alpha_tier2=self.config.alpha_tier2,
            alpha_tier3=self.config.alpha_tier3,
            backend=self.config.backend,
        )
        agents = pop_builder.build_population(
            cluster_result=cluster_result,
            size=self.config.population_size,
            condition=cell.condition,
            seed=self.config.seed + cell.repeat,
        )

        # Prepare simulation threads (sample topics from real data)
        sim_threads = self._prepare_sim_threads(threads, cell)

        # Run simulation
        topology = self.get_platform_topology(cell.dataset)
        sandbox = SimulationSandbox(
            platform=topology,
            checkpoint_dir=str(self.checkpoint_dir),
            max_concurrency=self.config.max_concurrency,
        )

        result = await sandbox.run(
            agents=agents,
            threads=sim_threads,
            num_rounds=self.config.num_rounds,
            run_id=cell.cell_id,
            condition=cell.condition,
            dataset=cell.dataset,
            model=cell.model,
            repeat=cell.repeat,
            checkpoint_every=self.config.checkpoint_every,
            seed=self.config.seed + cell.repeat,
        )

        # Evaluate
        report = self.metrics_agg.evaluate(result, threads)

        return report.to_dict()

    def _load_data(self, dataset: str):
        """Load and cache dataset."""
        if dataset not in self._data_cache:
            loader = self.get_dataset_loader(dataset)
            threads = loader.load()
            if self.config.max_threads is not None and len(threads) > self.config.max_threads:
                rng = np.random.RandomState(self.config.seed)
                idx = rng.choice(len(threads), size=self.config.max_threads, replace=False)
                threads = [threads[i] for i in idx]
                logger.info(
                    f"Capped {dataset} to {self.config.max_threads} threads (from full corpus)"
                )
            self._data_cache[dataset] = threads
            logger.info(f"Loaded {len(threads)} threads from {dataset}")
        return self._data_cache[dataset]

    def _get_or_compute_clusters(self, dataset: str, threads: list):
        """Get cached clustering result or compute new one (outline §4.2)."""
        if dataset not in self._cluster_cache:
            clusterer = BehavioralClusterer(
                method=self.config.cluster_method,
                n_clusters=self.config.num_clusters,
                random_state=self.config.seed,
            )
            cluster_result = clusterer.fit(threads)
            self._cluster_cache[dataset] = cluster_result
            logger.info(
                f"Clustering {dataset}: {cluster_result.n_clusters} clusters, "
                f"silhouette={cluster_result.silhouette_score:.3f}, "
                f"db={cluster_result.davies_bouldin_score:.3f}"
            )
        return self._cluster_cache[dataset]

    def _validate_cluster_stability(self, dataset: str, threads: list) -> dict:
        """Validate cluster stability via ARI bootstrap (outline §5.4).

        ARI variance < 0.2 indicates stable clustering.

        For large corpora we validate on a random subsample of threads:
        the bootstrap loop calls ``clusterer.fit()`` (which re-embeds every
        message) ``n_iterations`` times, so passing the full 100K+ thread
        corpus would re-embed millions of messages dozens of times. A
        2,000-thread sample is statistically sufficient for ARI variance
        estimation and keeps each fit under a few seconds.
        """
        if dataset not in self._stability_cache:
            import numpy as np

            stability_cap = 2000
            if len(threads) > stability_cap:
                rng = np.random.RandomState(self.config.seed)
                idx = rng.choice(len(threads), size=stability_cap, replace=False)
                eval_threads = [threads[i] for i in idx]
                logger.info(
                    f"Stability validation for {dataset}: subsampling "
                    f"{len(threads)} → {stability_cap} threads "
                    f"(bootstrap re-embeds each iteration)"
                )
            else:
                eval_threads = threads

            clusterer = BehavioralClusterer(
                method=self.config.cluster_method,
                n_clusters=self.config.num_clusters,
                random_state=self.config.seed,
            )
            validator = ClusterStabilityValidator(
                n_iterations=min(50, len(eval_threads)),  # scale down for large datasets
                random_state=self.config.seed,
            )
            report = validator.validate(eval_threads, clusterer)
            self._stability_cache[dataset] = report
            logger.info(
                f"Cluster stability for {dataset}: "
                f"ARI mean={report['ari_mean']:.3f}, "
                f"variance={report['ari_variance']:.4f}, "
                f"stable={report['is_stable']}"
            )
            if not report["is_stable"]:
                logger.warning(
                    f"Cluster instability detected for {dataset}! "
                    f"ARI variance {report['ari_variance']:.4f} >= 0.2 threshold"
                )
        return self._stability_cache[dataset]

    async def _get_or_compile_skills(
        self,
        dataset: str,
        threads: list,
        cluster_result,
        thread_cluster_map: dict,
    ) -> dict:
        """Get or compile skills for a dataset.

        First checks disk for existing .skill files (avoid recompilation).
        Falls back to compilation if not found.
        """
        if dataset not in self._skill_cache:
            from src.config.settings import settings as proj_settings
            from src.skill.compiler import SkillCompiler

            # Check if skill files already exist on disk
            cluster_ids = [str(cid) for cid in cluster_result.get_cluster_ids()]
            if SkillCompiler.skills_exist(
                proj_settings.skills_dir, dataset, cluster_ids
            ):
                logger.info(f"Loading existing skills for {dataset} from disk")
                skills = SkillCompiler.load_all_skills(
                    proj_settings.skills_dir, platform=dataset
                )
            else:
                logger.info(f"Compiling new skills for {dataset}")
                compiler = SkillCompiler(
                    llm_client=self.llm,
                    model_name=self.config.models[0],
                    top_n_threads=self.config.top_n_threads,
                )
                skills = await compiler.compile_all(
                    platform=dataset,
                    cluster_result=cluster_result,
                    all_threads=threads,
                    thread_cluster_map=thread_cluster_map,
                )
                # Save skills to disk for reuse
                for cid, skill in skills.items():
                    compiler.save_skill(skill)

            self._skill_cache[dataset] = skills
            logger.info(
                f"Skills ready for {dataset}: {len(skills)} clusters "
                f"({list(skills.keys())})"
            )
        return self._skill_cache[dataset]

    def _assign_threads_to_clusters(self, threads: list, cluster_result) -> dict[str, int]:
        """Assign each thread to a cluster based on its participants' majority cluster."""
        thread_map = {}
        for thread in threads:
            cluster_votes = Counter()
            for user_id in thread.participants:
                cid = cluster_result.labels.get(user_id)
                if cid is not None:
                    cluster_votes[cid] += 1
            if cluster_votes:
                thread_map[thread.thread_id] = cluster_votes.most_common(1)[0][0]
            else:
                thread_map[thread.thread_id] = 0
        return thread_map

    def _prepare_sim_threads(self, threads: list, cell: ExperimentCell) -> list:
        """Prepare thread stubs for simulation (empty messages, same topics)."""
        from src.data.schemas import Thread as ThreadModel

        # Sample diverse topics
        seen_topics = set()
        sim_threads = []
        for thread in threads:
            if thread.topic not in seen_topics and len(thread.participants) >= 2:
                seen_topics.add(thread.topic)
                sim_threads.append(ThreadModel(
                    thread_id=f"sim_{cell.cell_id}_{thread.thread_id}",
                    platform=thread.platform,
                    topic=thread.topic,
                ))
            if len(sim_threads) >= 5:  # limit topics per cell
                break

        # Ensure at least one thread
        if not sim_threads and threads:
            sim_threads.append(ThreadModel(
                thread_id=f"sim_{cell.cell_id}_default",
                platform=threads[0].platform,
                topic=threads[0].topic,
            ))

        return sim_threads

    def save_all_metrics(self) -> None:
        """Save aggregated metrics."""
        df = self.metrics_agg.to_dataframe()
        csv_path = self.results_dir / "exp1_all_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
