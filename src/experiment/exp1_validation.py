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
            model_provenance=self.llm.get_all_provenance(),
        )
        # Cache compiled skills per (dataset, distiller) — distiller is
        # None for pipeline-A/bare-file conditions, "colleague"/"nuwa" for
        # the manual-distiller comparison conditions.
        self._skill_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
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
            skills = await self._get_or_compile_skills(
                cell.dataset, threads, cluster_result, thread_cluster_map,
                condition=cell.condition,
            )

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
            memory_strategy=self.config.memory_strategy,
            compaction_interval=self.config.compaction_interval,
            compaction_keep_recent=self.config.compaction_keep_recent,
            max_display_items=self.config.max_context_items,
            per_msg_token_ratio=self.config.per_msg_token_ratio,
            per_msg_token_floor=self.config.per_msg_token_floor,
            max_thread_messages=self.config.max_thread_messages,
            reflection_interval=self.config.reflection_interval,
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
        """Load and cache dataset.

        Reframe v1 (2026-07-08): cap utterance ingestion upstream by
        passing ``cfg.max_threads`` as the loader's ``limit=`` kwarg.
        Previously the loader read the full 18-year WikiConv corpus and
        the downstream cap in this method only fired AFTER OOM. The
        loader's ``limit`` is per-corpus-balanced, so the upstream cap
        is both faster and statistically equivalent to the old
        post-hoc random subsample.

        Graph-density fix: single-message / single-user threads carry
        no interaction signal (no edges in the interaction graph), so
        they make Macro Topology metrics uninformative.  After loading,
        filter to threads with >= 2 messages AND >= 2 distinct
        participants.  To compensate for the ~95% reduction, the
        utterance cap is inflated by ``thread_filter_oversample``
        (default 10×) so that enough multi-user threads survive.
        """
        if dataset not in self._data_cache:
            # Inflate the utterance cap so filtering still yields
            # enough threads.  The loader's limit is utterance-level;
            # ~95% of utterances land in single-msg threads, so we
            # need ~10× oversampling to hit the target thread count
            # after filtering.
            oversample = getattr(self.config, "thread_filter_oversample", 10)
            effective_limit = (
                self.config.max_threads * oversample
                if self.config.max_threads is not None
                else None
            )
            loader = self.get_dataset_loader(dataset)
            if hasattr(loader, "limit"):
                loader.limit = effective_limit
            threads = loader.load()

            # Filter to threads with actual interaction structure
            before = len(threads)
            threads = [
                t for t in threads
                if len(t.messages) >= 2 and len(t.participants) >= 2
            ]
            if before > 0 and len(threads) < before:
                logger.info(
                    f"Thread filter: {len(threads)}/{before} threads kept "
                    f"(>=2 msgs, >=2 participants)"
                )

            # Cap to max_threads after filtering
            if self.config.max_threads is not None and len(threads) > self.config.max_threads:
                rng = np.random.RandomState(self.config.seed)
                idx = rng.choice(len(threads), size=self.config.max_threads, replace=False)
                threads = [threads[i] for i in idx]
                logger.info(
                    f"Capped {dataset} to {self.config.max_threads} threads (from filtered set)"
                )
            self._data_cache[dataset] = threads
            logger.info(f"Loaded {len(threads)} threads from {dataset}")
        return self._data_cache[dataset]

    def _make_clusterer(self) -> BehavioralClusterer:
        """Build a BehavioralClusterer from the experiment config."""
        from src.clustering.clusterer import BehavioralClusterer
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
        """Get cached clustering result.

        Reframe v1 (2026-07-08): prefer the locked precomputed pickle over
        live ``clusterer.fit()``. Skill files were distilled against the
        locked cluster IDs (0,2,3,4,6,7 after merging 1→0, 5→4); live
        re-fitting produces different IDs → CADP conditions silently load
        wrong skills. Live fit remains a fallback for datasets without a
        canonical pickle.
        """
        if dataset not in self._cluster_cache:
            pickle_path = self.config.clustering_pickle_path
            if pickle_path:
                cluster_result = self._load_locked_clusters(pickle_path)
                self._cluster_cache[dataset] = cluster_result
                logger.info(
                    f"Loaded locked clustering for {dataset}: "
                    f"{cluster_result.n_clusters} clusters, "
                    f"silhouette={cluster_result.silhouette_score:.3f}, "
                    f"db={cluster_result.davies_bouldin_score:.3f} "
                    f"(pickle={pickle_path})"
                )
            else:
                clusterer = self._make_clusterer()
                cluster_result = clusterer.fit(threads)
                self._cluster_cache[dataset] = cluster_result
                logger.info(
                    f"Clustering {dataset} (live fit): "
                    f"{cluster_result.n_clusters} clusters, "
                    f"silhouette={cluster_result.silhouette_score:.3f}, "
                    f"db={cluster_result.davies_bouldin_score:.3f}"
                )
        return self._cluster_cache[dataset]

    def _load_locked_clusters(self, pickle_path: str):
        """Load locked ClusterResult pickle and apply merge_map if present.

        The pickle holds the canonical K=8 source clustering. The merge
        map (JSON alongside skill_corpus_k8_quantile) collapses leaves
        {1→0, 5→4} to the 6 final skill archetypes that the distilled
        skill files were named after. Returns a ClusterResult whose
        ``labels`` and ``user_features`` already reflect the merged IDs.
        """
        import pickle
        from pathlib import Path

        from src.clustering.clusterer import ClusterResult

        path = Path(pickle_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"clustering_pickle_path set but file not found: {path}. "
                f"Either set the path to outputs/stream_cache/"
                f"clustering_k8_final_quantile.pkl or clear the field "
                f"to fall back to live clusterer.fit()."
            )

        with open(path, "rb") as f:
            cluster_result = pickle.load(f)

        # Apply merge map if provided (collapses K=8 → 6 skill archetypes).
        merge_map_path = self.config.cluster_merge_map_path
        if merge_map_path and Path(merge_map_path).is_file():
            import json
            with open(merge_map_path) as f:
                cm = json.load(f)
            merge_map = cm.get("merge_map", {})  # {"1": 0, "5": 4}
            if merge_map:
                merged_labels = {
                    uid: int(merge_map.get(str(lbl), lbl))
                    for uid, lbl in cluster_result.labels.items()
                }
                cluster_result.labels = merged_labels
                logger.info(
                    f"Applied cluster merge_map {merge_map} → "
                    f"{len(set(merged_labels.values()))} final skill archetypes"
                )

        return cluster_result

    def _load_locked_quality(self) -> dict:
        """Load precomputed cluster-quality numbers for §5.4 reporting.

        Looks for ``quality_report.json`` alongside the merge map (same
        directory). Returns a dict shaped like the live ARI report so
        downstream code can consume either transparently. ``source``
        field lets paper §5.4 cite where the numbers came from.
        """
        import json
        from pathlib import Path

        merge_path = Path(self.config.cluster_merge_map_path or "")
        candidates = [
            merge_path.parent / "quality_report.json" if merge_path else None,
        ]
        for c in candidates:
            if c and c.is_file():
                with open(c) as f:
                    data = json.load(f)
                return {
                    "source": f"locked:{c}",
                    "silhouette": data.get("silhouette"),
                    "davies_bouldin": data.get("davies_bouldin"),
                    "n_final_skills": data.get("n_final_skills"),
                    "n_users": data.get("n_users"),
                    "concerns": data.get("concerns", []),
                    # Marker so analysis layer knows this isn't a live ARI:
                    "locked_pickle": True,
                    "ari_mean": None,
                    "ari_variance": None,
                }
        logger.warning(
            "clustering_pickle_path set but no quality_report.json found "
            f"alongside {merge_path}; §5.4 stability table will be empty"
        )
        return {"source": "missing", "locked_pickle": True}

    def _validate_cluster_stability(self, dataset: str, threads: list) -> dict:
        """Validate cluster stability (outline §5.4).

        Reframe v1 (2026-07-08): when ``clustering_pickle_path`` is set,
        the clustering is LOCKED — ARI bootstrap on re-fit is meaningless
        (we no longer re-fit). Return the precomputed quality numbers
        from ``quality_report.json`` (alongside the merge map) instead.
        Falls through to live ARI bootstrap only when no pickle is set.
        """
        if dataset not in self._stability_cache:
            if self.config.clustering_pickle_path:
                self._stability_cache[dataset] = self._load_locked_quality()
                logger.info(
                    f"Stability for {dataset}: using locked-pickle quality "
                    f"report (no live ARI bootstrap — clustering is canonical)"
                )
                return self._stability_cache[dataset]

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

            clusterer = self._make_clusterer()
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
        condition: str = "cadp_full",
    ) -> dict:
        """Get or compile skills for a dataset.

        First checks disk for existing .skill files (avoid recompilation).
        Falls back to compilation if not found — EXCEPT for the manual-
        distiller conditions (``cadp_full_colleague`` / ``cadp_full_nuwa``),
        which fail fast instead. Those conditions exist specifically to
        evaluate the manually-distilled colleague-skill / nuwa-skill
        artifacts (converted via ``scripts/convert_distilled_skills.py`);
        silently falling back to pipeline A's LLM re-extraction would
        defeat the purpose of the comparison and mask a missing-file bug
        as a successful run (plan: "打通蒸馏产出与 CADP 实验", 2026-07-08).
        """
        from src.experiment.conditions import distiller_suffix

        distiller = distiller_suffix(condition)
        cache_key = (dataset, distiller)
        if cache_key not in self._skill_cache:
            from src.config.settings import settings as proj_settings
            from src.skill.compiler import SkillCompiler

            cluster_ids = [str(cid) for cid in cluster_result.get_cluster_ids()]

            if distiller is not None:
                if not SkillCompiler.skills_exist(
                    proj_settings.skills_dir, dataset, cluster_ids, distiller=distiller
                ):
                    raise FileNotFoundError(
                        f"Condition '{condition}' requires pre-converted '{distiller}' "
                        f"skill files for dataset '{dataset}' (clusters {cluster_ids}) "
                        f"under {proj_settings.skills_dir}, e.g. "
                        f"skill_cluster_{cluster_ids[0]}_{dataset}_{distiller}.yaml. "
                        f"Run scripts/convert_distilled_skills.py first — this condition "
                        f"does not fall back to pipeline A recompilation."
                    )
                logger.info(f"Loading existing '{distiller}' skills for {dataset} from disk")
                skills = SkillCompiler.load_all_skills(
                    proj_settings.skills_dir, platform=dataset, distiller=distiller
                )
            elif SkillCompiler.skills_exist(
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
                    model_name=self.config.compile_model or self.config.models[0],
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

            self._skill_cache[cache_key] = skills
            logger.info(
                f"Skills ready for {dataset} (distiller={distiller}): "
                f"{len(skills)} clusters ({list(skills.keys())})"
            )
        return self._skill_cache[cache_key]

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

    def _load_cga_seed_threads(self) -> list:
        """Load high-conflict CGA threads as seed material for simulation."""
        from src.data.schemas import Thread as ThreadModel, Message, Platform, ActionType

        if hasattr(self, '_cga_seed_cache') and self._cga_seed_cache is not None:
            return self._cga_seed_cache

        cga_dir = settings.data_dir / "raw" / "wikiconv_en" / "cga" / "conversations-gone-awry-corpus"
        conv_path = cga_dir / "conversations.json"
        utt_path = cga_dir / "utterances.jsonl"

        if not conv_path.exists() or not utt_path.exists():
            logger.warning(f"CGA corpus not found at {cga_dir}, falling back to empty stubs")
            self._cga_seed_cache = None
            return []

        import json
        from datetime import datetime

        # 1. Load attack conversations
        with open(conv_path) as f:
            convs = json.load(f)
        attack_conv_ids = {
            cid for cid, meta in convs.items()
            if meta.get("conversation_has_personal_attack")
        }
        logger.info(f"CGA: {len(attack_conv_ids)} attack conversations out of {len(convs)}")

        # 2. Load utterances, group by conversation
        conv_utterances: dict[str, list[dict]] = {}
        with open(utt_path) as f:
            for line in f:
                utt = json.loads(line)
                cid = utt.get("conversation_id", "")
                if cid in attack_conv_ids:
                    conv_utterances.setdefault(cid, []).append(utt)

        # 3. Build Thread objects, filter by max toxicity
        min_tox = self.config.seed_min_toxicity
        seed_threads: list[ThreadModel] = []
        for cid, utts in conv_utterances.items():
            # Sort utterances by timestamp/id for order
            utts.sort(key=lambda u: u.get("id", ""))
            max_tox = max(
                (u.get("meta", {}).get("toxicity", 0) for u in utts),
                default=0,
            )
            if max_tox < min_tox:
                continue
            page_title = convs[cid].get("page_title", cid)
            messages = []
            participants = set()
            for u in utts:
                if u.get("meta", {}).get("is_section_header"):
                    continue
                text = u.get("text", "").strip()
                if not text:
                    continue
                speaker = u.get("speaker", "Unknown")
                participants.add(speaker)
                ts_raw = u.get("meta", {}).get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now()
                except (ValueError, TypeError):
                    ts = datetime.now()
                messages.append(Message(
                    msg_id=u["id"],
                    thread_id=cid,
                    user_id=speaker,
                    platform=Platform.WIKIPEDIA,
                    timestamp=ts,
                    text=text,
                    action_type=ActionType.DISCUSS,
                    metadata=u.get("meta", {}),
                ))
            if len(messages) < 2:
                continue
            seed_threads.append(ThreadModel(
                thread_id=cid,
                platform=Platform.WIKIPEDIA,
                topic=page_title,
                messages=messages,
                participants=participants,
            ))

        logger.info(
            f"CGA seed threads: {len(seed_threads)} (toxicity >= {min_tox})"
        )
        self._cga_seed_cache = seed_threads
        return seed_threads

    def _prepare_sim_threads(self, threads: list, cell: ExperimentCell) -> list:
        """Prepare sim threads seeded with real CGA conflict conversations."""
        from src.data.schemas import Thread as ThreadModel
        import copy

        seed_threads = self._load_cga_seed_threads()

        if seed_threads:
            # Sample diverse topics from CGA seeds
            seen_topics = set()
            sim_threads = []
            rng = np.random.default_rng(self.config.seed + hash(cell.cell_id) % 10000)
            indices = rng.permutation(len(seed_threads))
            for idx in indices:
                seed = seed_threads[idx]
                if seed.topic in seen_topics:
                    continue
                seen_topics.add(seed.topic)
                # Clone with sim_ prefix; keep original messages as seed context
                sim = copy.deepcopy(seed)
                sim.thread_id = f"sim_{cell.cell_id}_{seed.thread_id}"
                sim_threads.append(sim)
                if len(sim_threads) >= self.config.max_sim_threads:
                    break

            if sim_threads:
                return sim_threads
            # Fall through to stub logic if no diverse topics found

        # Fallback: original empty-stub logic (CGA unavailable)
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
            if len(sim_threads) >= self.config.max_sim_threads:
                break

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
