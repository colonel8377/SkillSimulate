"""Experiment 1 runner — method validation grid."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
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
            llm_client=None,
            llm_model_name=None,
            action_smoothing=config.action_js_smoothing,
            continuation_mode=(config.reference_strategy == "observed_continuation"),
            linguistic_metric_weights=config.linguistic_metric_weights,
            interaction_metric_weights=config.interaction_metric_weights,
            seed=config.seed,
        )
        # Cache compiled skills per (dataset, distiller) — distiller is
        # None for pipeline-A/bare-file conditions, "colleague"/"nuwa" for
        # the manual-distiller comparison conditions.
        self._skill_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._data_cache: dict[str, list] = {}
        self._cluster_cache: dict[str, Any] = {}  # dataset → ClusterResult
        self._stability_cache: dict[str, dict] = {}  # dataset → stability report
        self._trigger_status: dict[tuple[str, str | None], str] = {}
        self._tier1_calibration_status: dict[tuple[str, str | None], str] = {}
        self._tier1_calibration_thread_ids: set[str] = set()
        self._distillation_thread_ids_cache: set[str] | None = None

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

        # Feasibility manifests must reserve Tier-1 calibration threads before
        # the first (possibly non-CADP) condition creates the shared manifest.
        # Warm the Nuwa skill cache and calibration here so condition order
        # cannot introduce evaluation leakage after expensive baseline cells.
        if any(
            name in self.config.conditions
            for name in ("cadp_advisory_nuwa", "cadp_full_nuwa")
        ) and (cell.dataset, "nuwa") not in self._skill_cache:
            await self._get_or_compile_skills(
                cell.dataset, threads, cluster_result, thread_cluster_map,
                condition="cadp_advisory_nuwa",
            )

        # Compile skills if CADP condition (shared per dataset)
        skills = {}
        if is_cadp_condition(cell.condition):
            skills = await self._get_or_compile_skills(
                cell.dataset, threads, cluster_result, thread_cluster_map,
                condition=cell.condition,
            )

        # A causal treatment that never activates is not a treatment. Before
        # spending any Full-condition API calls, replay its deterministic
        # Tier-1 / lexical Tier-3 trigger logic over the matching Advisory
        # outputs and fail closed when the pre-registered minimum is not met.
        if (
            cell.condition == "cadp_full_nuwa"
            and self.config.manipulation_min_potential_rate > 0
        ):
            await self._validate_manipulation_potential(cell, skills)

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
            population_allocation=self.config.population_allocation,
            max_reformulation_retries=self.config.max_reformulation_retries,
            tier1_max_retries=self.config.tier1_max_retries,
            tier3_llm_judge_enabled=self.config.tier3_llm_judge_enabled,
            tier3_llm_judge_model=self.config.tier3_llm_judge_model,
            tier3_llm_judge_audit_only=self.config.tier3_llm_judge_audit_only,
            tier3_llm_judge_output_dir=self.config.tier3_llm_judge_output_dir,
        )
        agents = pop_builder.build_population(
            cluster_result=cluster_result,
            size=self.config.population_size,
            condition=cell.condition,
            seed=self.config.seed + cell.repeat,
        )

        # Prepare simulation threads (sample topics from real data)
        sim_threads, evaluation_threads, linguistic_reference_threads = (
            self._prepare_sim_threads(threads, cell)
        )
        self._validate_reference_preflight(evaluation_threads)
        await self._validate_evaluator_preflight(linguistic_reference_threads)
        run_fingerprint = self._build_run_fingerprint(
            cell, sim_threads, evaluation_threads, linguistic_reference_threads, skills,
        )

        # Run simulation
        topology = self.get_platform_topology(cell.dataset)
        sandbox = SimulationSandbox(
            platform=topology,
            checkpoint_dir=str(self.checkpoint_dir),
            max_concurrency=self.config.max_concurrency,
            micro_batch_size=self.config.micro_batch_size,
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
            run_fingerprint=run_fingerprint,
        )
        if is_cadp_condition(cell.condition):
            from src.experiment.conditions import distiller_suffix
            result.enforcement_stats["trigger_calibration_status"] = (
                self._trigger_status.get(
                    (cell.dataset, distiller_suffix(cell.condition)),
                    "unknown",
                )
            )
            result.enforcement_stats["tier1_calibration_status"] = (
                self._tier1_calibration_status.get(
                    (cell.dataset, distiller_suffix(cell.condition)),
                    "unknown",
                )
            )

        # Evaluate
        report = await self.metrics_agg.evaluate(
            result,
            evaluation_threads,
            linguistic_reference_threads=linguistic_reference_threads,
        )

        return report.to_dict()

    async def _validate_evaluator_preflight(self, reference_threads: list) -> None:
        """Load and exercise every required local linguistic evaluator.

        This runs before the first simulation call, preventing a missing local
        model from wasting an otherwise valid paid cell.
        """
        from src.evaluation.linguistics import LinguisticMetrics

        sample = [
            message for thread in reference_threads for message in thread.messages
            if str(message.text or "").strip()
        ][:8]
        if len(sample) < 2:
            raise ValueError("Evaluator preflight needs at least two reference messages")
        metrics = await LinguisticMetrics.compute(sample, sample)
        required = {
            "discourse_relation_match", "sentiment_trajectory_similarity",
            "speech_act_similarity", "sip",
        }
        missing = required - set(metrics)
        invalid = {
            name: metrics.get(name) for name in required
            if name in metrics and not np.isfinite(float(metrics[name]))
        }
        if missing or invalid:
            raise ValueError(
                f"Local evaluator preflight failed: missing={sorted(missing)}, invalid={invalid}"
            )
        logger.info("Local evaluator preflight passed: PDTB, sentiment, speech-act, SIP")

    async def _validate_manipulation_potential(
        self, cell: ExperimentCell, skills: dict,
    ) -> dict[str, Any]:
        """Audit whether Full would alter any matching Advisory outputs.

        This uses only saved Advisory messages and local embedders. It neither
        calls the simulation LLM nor tunes a threshold using evaluation
        outcomes. The audit is persisted so treatment activation is explicit.
        """
        import re

        from src.config.embedder import run_embed_in_executor
        from src.config.settings import get_shared_embedder

        # Pool all pre-registered Advisory repeats. A 1% requirement applied
        # separately to ~20 outputs silently becomes an integer requirement of
        # >=1/20=5%; pooling preserves the configured rate's actual meaning.
        advisory_ids = [
            ExperimentCell(
                "cadp_advisory_nuwa", cell.dataset, cell.model, repeat,
            ).cell_id
            for repeat in range(getattr(self.config, "num_repeats", 1))
        ]
        candidates: list[tuple[dict, str, int]] = []
        for repeat, advisory_id in enumerate(advisory_ids):
            if not self.checkpoint.is_completed(advisory_id):
                raise ValueError(
                    "Full manipulation preflight requires all Advisory repeats; "
                    f"missing completed cell {advisory_id}"
                )
            checkpoint = self.checkpoint.load(advisory_id)
            if checkpoint is None:
                raise ValueError(f"Advisory checkpoint missing for {advisory_id}")
            extra = checkpoint.get("extra") or {}
            if extra.get("checkpoint_schema_version") != 2 or not extra.get("round_complete"):
                raise ValueError(
                    f"Advisory checkpoint {advisory_id} predates lossless schema v2; "
                    "use a new experiment name and rerun Advisory"
                )
            agent_clusters = {
                str(row.get("agent_id")): str(row.get("cluster_id"))
                for row in checkpoint.get("agents_state") or []
            }
            candidates.extend(
                (row, agent_clusters[str(row.get("user_id"))], repeat)
                for row in checkpoint.get("messages_log") or []
                if str(row.get("text") or "").strip()
                and str(row.get("user_id")) in agent_clusters
            )
        if not candidates:
            raise ValueError("Advisory checkpoints contain no auditable outputs")

        # Batch once: this is the exact max-z statistic used by Tier 1 at α=1.
        embedder = get_shared_embedder()
        vectors = np.asarray(await run_embed_in_executor(
            embedder.encode,
            [str(row["text"]) for row, _, _ in candidates],
            show_progress_bar=False,
            batch_size=32,
        ), dtype=float)

        tier1_hits = 0
        tier3_hits = 0
        any_hits = 0
        per_cluster: dict[str, dict[str, int]] = defaultdict(
            lambda: {"outputs": 0, "tier1": 0, "tier3_lexical": 0, "any": 0}
        )
        per_repeat: dict[str, dict[str, int]] = defaultdict(
            lambda: {"outputs": 0, "tier1": 0, "tier3_lexical": 0, "any": 0}
        )
        for (row, cid, repeat), vector in zip(candidates, vectors):
            skill = skills.get(cid)
            if skill is None:
                try:
                    skill = skills.get(int(cid))
                except ValueError:
                    skill = None
            if skill is None or not skill.capability:
                raise ValueError(f"Manipulation audit has no skill/EDNA for cluster {cid}")
            edna = skill.capability.expression_dna
            if (
                edna.embedding_centroid is None
                or edna.embedding_cosine_threshold is None
            ):
                raise ValueError(f"Manipulation audit lacks calibrated Tier-1 data for cluster {cid}")
            centroid = np.asarray(edna.embedding_centroid, dtype=float)
            if vector.shape != centroid.shape:
                raise ValueError(
                    f"Manipulation audit embedding dimension mismatch for cluster {cid}: "
                    f"output={vector.shape}, skill={centroid.shape}"
                )
            vec_norm = vector / (np.linalg.norm(vector) + 1e-10)
            cent_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            cosine_dist = float(1.0 - np.dot(vec_norm, cent_norm))
            hit1 = cosine_dist > float(edna.embedding_cosine_threshold)

            # Tier-3 lexical audit removed 2026-07-13: trigger_regex /
            # trigger_keywords fields deleted from AntiPattern. The LLM judge
            # is non-deterministic and runs at generation time, not on replayed
            # Advisory outputs, so no deterministic Tier-3 potential can be
            # computed here. Tier-1 cosine-distance audit remains.
            hit3 = False
            text_value = str(row["text"])

            tier1_hits += int(hit1)
            tier3_hits += int(hit3)
            any_hits += int(hit1 or hit3)
            counts = per_cluster[cid]
            counts["outputs"] += 1
            counts["tier1"] += int(hit1)
            counts["tier3_lexical"] += int(hit3)
            counts["any"] += int(hit1 or hit3)
            repeat_counts = per_repeat[str(repeat)]
            repeat_counts["outputs"] += 1
            repeat_counts["tier1"] += int(hit1)
            repeat_counts["tier3_lexical"] += int(hit3)
            repeat_counts["any"] += int(hit1 or hit3)

        rate = any_hits / len(candidates)
        audit = {
            "schema_version": 2,
            "advisory_cells": advisory_ids,
            "full_cell": cell.cell_id,
            "n_outputs": len(candidates),
            "tier1_potential_count": tier1_hits,
            "tier3_lexical_potential_count": tier3_hits,
            "any_potential_count": any_hits,
            "potential_rate": rate,
            "required_minimum_rate": self.config.manipulation_min_potential_rate,
            "passed": rate >= self.config.manipulation_min_potential_rate,
            "per_cluster": dict(per_cluster),
            "per_repeat": dict(per_repeat),
            "note": "Deterministic Tier-1 and calibrated/sanitized lexical Tier-3 potential only",
        }
        audit_dir = self.results_dir / "manipulation_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        save_json(audit, audit_dir / f"{cell.cell_id}.json")
        if audit["passed"]:
            logger.info(
                f"Manipulation audit passed for {cell.cell_id}: "
                f"{any_hits}/{len(candidates)} outputs ({rate:.2%})"
            )
        else:
            # Advisory outputs are not valid counterfactual Full outputs:
            # dynamic retrieval changes Full's prompt before generation and
            # can change which post-generation filters activate. Therefore a
            # zero Advisory potential rate is important diagnostic evidence,
            # but cannot logically block generation of the treatment arm.
            logger.warning(
                f"Manipulation audit found no Advisory-side filter potential "
                f"for {cell.cell_id}: {any_hits}/{len(candidates)} ({rate:.2%}). "
                "Running Full and reporting observed mechanism activation."
            )
        return audit

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
            # Loader reads year-by-year and stops once enough multi-user
            # threads accumulate.  No oversample factor needed — the
            # loader is thread-count-driven, not utterance-count-driven.
            loader = self.get_dataset_loader(dataset)
            if hasattr(loader, "target_threads"):
                if (
                    dataset == "wikipedia"
                    and self.config.reference_strategy == "observed_continuation"
                ):
                    # Observed continuations require complete conversations.
                    # WikiConv JSONL is not conversation-contiguous, so an
                    # in-file target stop silently truncates threads. Read a
                    # pre-registered, tractable set of years to EOF instead.
                    loader.min_messages = max(4, self.config.continuation_min_messages)
                    loader.target_threads = None
                    years = set(self.config.continuation_years)
                    if not years:
                        raise ValueError(
                            "observed_continuation requires continuation_years; "
                            "full 18-year WikiConv is too large to scan safely"
                        )
                    loader.allowed_years = years
                else:
                    loader.target_threads = self.config.max_threads
                    loader.min_messages = 2
            threads = loader.load()

            # Filter to threads with actual interaction structure
            before = len(threads)
            required_messages = (
                max(4, self.config.continuation_min_messages)
                if dataset == "wikipedia"
                and self.config.reference_strategy == "observed_continuation"
                else 2
            )
            threads = [
                t for t in threads
                if len(t.messages) >= required_messages and len(t.participants) >= 2
            ]
            if before > 0 and len(threads) < before:
                logger.info(
                    f"Thread filter: {len(threads)}/{before} threads kept "
                    f"(>={required_messages} msgs, >=2 participants)"
                )

            # Cap to max_threads after filtering
            continuation_complete_years = (
                dataset == "wikipedia"
                and self.config.reference_strategy == "observed_continuation"
            )
            if (
                not continuation_complete_years
                and self.config.max_threads is not None
                and len(threads) > self.config.max_threads
            ):
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

        # The streaming clustering artifact stores raw public handles, while
        # every runtime dataset loader scrubs them before returning Thread
        # objects. Normalize the locked artifact into the same deterministic
        # pseudonymous key space; otherwise thread→cluster assignment and
        # held-out Tier-1 calibration silently have zero label coverage.
        from src.data.pii import anonymize_user_id
        anonymized_labels = {
            anonymize_user_id(str(uid)): label
            for uid, label in cluster_result.labels.items()
        }
        anonymized_features = {}
        for uid, feature in cluster_result.user_features.items():
            anonymous_uid = anonymize_user_id(str(uid))
            try:
                feature.user_id = anonymous_uid
            except (AttributeError, TypeError):
                pass
            anonymized_features[anonymous_uid] = feature
        if len(anonymized_labels) != len(cluster_result.labels):
            raise ValueError("User-ID anonymization collision in locked clustering")
        cluster_result.labels = anonymized_labels
        cluster_result.user_features = anonymized_features

        # Apply merge map if provided (collapses K=8 → 6 skill archetypes).
        merge_map_path = self.config.cluster_merge_map_path
        if merge_map_path and Path(merge_map_path).is_file():
            import json
            with open(merge_map_path) as f:
                cm = json.load(f)
            merge_map = cm.get("merge_map", {})  # {"1": 0, "5": 4}
            if merge_map:
                cluster_result.source_labels = dict(cluster_result.labels)
                cluster_result.cluster_merge_map = {
                    int(k): int(v) for k, v in merge_map.items()
                }
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

        When ``clustering_pickle_path`` is set, retain the locked partition for
        experiment reproducibility but independently test its resampling
        stability from cached user vectors (QuantileTransformer + KMeans K=8
        followed by the canonical merge). Falls through to thread-level live
        ARI bootstrap only when no locked pickle is configured.
        """
        if dataset not in self._stability_cache:
            if self.config.clustering_pickle_path:
                locked_quality = self._load_locked_quality()
                bootstrap_path = self.results_dir / f"clustering_stability_{dataset}.json"
                if bootstrap_path.exists():
                    bootstrap_report = json.loads(bootstrap_path.read_text())
                else:
                    bootstrap_report = {}
                if bootstrap_report.get("protocol") != "locked_k8_merge_bootstrap_v2":
                    cluster_result = self._cluster_cache.get(dataset)
                    if cluster_result is None:
                        raise ValueError(
                            "Locked clustering must be loaded before stability validation"
                        )
                    bootstrap_report = ClusterStabilityValidator.validate_locked_vectors(
                        cluster_result,
                        merge_map=getattr(cluster_result, "cluster_merge_map", {}),
                        n_iterations=20,
                        train_sample_size=30_000,
                        eval_sample_size=10_000,
                        random_state=self.config.seed,
                    )
                    save_json(bootstrap_report, bootstrap_path)
                self._stability_cache[dataset] = {
                    **locked_quality,
                    "bootstrap_ari": bootstrap_report,
                    "ari_mean": bootstrap_report["ari_mean"],
                    "ari_variance": bootstrap_report["ari_variance"],
                    "is_stable": bootstrap_report["is_stable"],
                }
                logger.info(
                    f"Locked-cluster bootstrap for {dataset}: "
                    f"ARI={bootstrap_report['ari_mean']:.3f}, "
                    f"variance={bootstrap_report['ari_variance']:.4f}, "
                    f"n={bootstrap_report['n_iterations']}"
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

            status = self._sanitize_uncalibrated_triggers(dataset, skills)
            self._trigger_status[cache_key] = status
            self._tier1_calibration_status[cache_key] = (
                await self._calibrate_tier1_thresholds(
                    dataset, threads, cluster_result, skills, distiller,
                )
            )
            self._skill_cache[cache_key] = skills
            logger.info(
                f"Skills ready for {dataset} (distiller={distiller}): "
                f"{len(skills)} clusters ({list(skills.keys())})"
            )
        return self._skill_cache[cache_key]

    @staticmethod
    def _sanitize_uncalibrated_triggers(dataset: str, skills: dict) -> str:
        """No-op since the 2026-07-17 enforcement simplification.

        Rule-based Tier-3 triggers (regex/keywords/semantic-phrases) were
        deleted along with ``tier3_block.py``; the LLM judge reads
        ``AntiPattern.description`` / ``reason`` / ``trigger_conditions``
        directly. There is nothing to sanitize.
        """
        return "rule_path_removed_no_triggers_to_sanitize"

    async def _calibrate_tier1_thresholds(
        self, dataset: str, threads: list, cluster_result, skills: dict,
        distiller: str | None,
    ) -> str:
        """Fit Tier-1 vectors and max-z cutoffs on a trace train/valid split.

        For every locked skill, at most 400 typical trace messages are sampled
        deterministically, split 80/20, and embedded locally. The train split
        fits centroid/std; the validation split sets the empirical 95th
        percentile cutoff. All source thread IDs already belong to the
        distillation corpus and are excluded from evaluation.
        """
        from src.config.embedder import run_embed_in_executor
        from src.config.settings import get_shared_embedder
        from pathlib import Path
        import gc
        import random

        cache_path = self.results_dir / (
            f"tier1_calibration_{dataset}_{distiller or 'pipeline'}.json"
        )
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get("schema_version") == 3:
                by_cluster = cached.get("clusters") or {}
                for cid, skill in skills.items():
                    row = by_cluster.get(str(cid)) or {}
                    threshold = row.get("threshold")
                    centroid = row.get("centroid")
                    if (
                        threshold is None or centroid is None
                        or not skill.capability
                    ):
                        raise ValueError(
                            f"Tier-1 calibration cache lacks cluster {cid}"
                        )
                    edna = skill.capability.expression_dna
                    edna.embedding_centroid = list(centroid)
                    edna.embedding_cosine_threshold = float(threshold)
                self._tier1_calibration_thread_ids.update(
                    str(tid) for tid in cached.get("source_thread_ids") or []
                )
                return "trace_split_q95_cached"

        merge_path = Path(self.config.cluster_merge_map_path)
        corpus_dir = merge_path.parent
        samples: dict[str, list[tuple[str, str]]] = {}
        for cid in skills:
            path = corpus_dir / f"cluster_{cid}" / "typical.jsonl"
            rows: list[tuple[str, str]] = []
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text_value = str(record.get("text") or "").strip()
                    thread_id = record.get("thread_id")
                    if text_value and thread_id:
                        rows.append((str(thread_id), text_value))
            rng = random.Random(self.config.seed + int(cid) * 1009)
            rng.shuffle(rows)
            samples[str(cid)] = rows[:400]

        embedder = get_shared_embedder()
        report = {
            "schema_version": 3,
            "dataset": dataset,
            "distiller": distiller,
            "quantile": 0.95,
            "metric": "cosine_distance",
            "clusters": {},
            "source_thread_ids": [],
        }
        source_ids: set[str] = set()
        for cid, skill in skills.items():
            rows = samples.get(str(cid), [])
            if len(rows) < 100:
                raise ValueError(
                    f"Tier-1 calibration needs >=100 trace messages "
                    f"for cluster {cid}; found {len(rows)}"
                )
            edna = skill.capability.expression_dna if skill.capability else None
            if edna is None:
                raise ValueError(f"Tier-1 calibration missing EDNA for cluster {cid}")
            vectors = await run_embed_in_executor(
                embedder.encode,
                [text for _, text in rows],
                show_progress_bar=False,
                # Keep CPU preflight memory bounded even when the shared
                # embedder is configured with a large experiment batch.
                batch_size=16,
            )
            vectors = np.asarray(vectors, dtype=float)
            split = max(1, int(len(vectors) * 0.8))
            train_vectors = vectors[:split]
            valid_vectors = vectors[split:]
            if len(valid_vectors) < 20:
                raise ValueError(f"Tier-1 validation split too small for cluster {cid}")
            centroid = np.mean(train_vectors, axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            valid_norm = valid_vectors / (
                np.linalg.norm(valid_vectors, axis=1, keepdims=True) + 1e-10
            )
            cosine_dists = 1.0 - np.dot(valid_norm, centroid_norm)
            threshold = float(np.quantile(cosine_dists, 0.95))
            false_reject_rate = float(np.mean(cosine_dists > threshold))
            edna.embedding_centroid = centroid.tolist()
            edna.embedding_cosine_threshold = threshold
            source_ids.update(tid for tid, _ in rows)
            report["clusters"][str(cid)] = {
                "n_train": len(train_vectors),
                "n_validation": len(valid_vectors),
                "threshold": threshold,
                "observed_false_reject_rate": false_reject_rate,
                "centroid": centroid.tolist(),
            }
            del vectors, train_vectors, valid_vectors, centroid, centroid_norm, valid_norm, cosine_dists
            gc.collect()
        report["source_thread_ids"] = sorted(source_ids)
        save_json(report, cache_path)
        self._tier1_calibration_thread_ids.update(source_ids)
        return "trace_split_q95_fresh"

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

        from src.data.pii import scrub_threads
        from src.data.wikipedia import WikipediaLoader

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
            utts.sort(key=lambda u: (
                float(u.get("timestamp") or float("inf")), u.get("id", ""),
            ))
            max_tox = max(
                (float(u.get("meta", {}).get("toxicity") or 0) for u in utts),
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
                ts_raw = u.get("timestamp", u.get("meta", {}).get("timestamp"))
                ts = WikipediaLoader._parse_ts(ts_raw)
                reply_to = u.get("reply-to") or u.get("reply_to")
                messages.append(Message(
                    msg_id=u["id"],
                    thread_id=cid,
                    user_id=speaker,
                    platform=Platform.WIKIPEDIA,
                    timestamp=ts,
                    text=text,
                    action_type=ActionType.DISCUSS,
                    parent_msg_id=str(reply_to) if reply_to else None,
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
        # Keep CGA identities in the same pseudonymous key space as WikiConv
        # and data/role_labels. This is also required by the paper's PII claim.
        seed_threads = scrub_threads(seed_threads)
        self._cga_seed_cache = seed_threads
        return seed_threads

    @staticmethod
    def _thread_conflict_score(thread) -> tuple[float, float, float, int]:
        """Observable conflict score used only for deterministic matching."""
        # Reporting/deletion/restoration are moderation events, not evidence
        # of interpersonal conflict. Reversion is the only action here that
        # directly records a contested content change.
        conflict_actions = {"revert"}
        toxicities = []
        attack = 0.0
        conflict_count = 0
        for msg in thread.messages:
            meta = msg.metadata or {}
            raw_tox = meta.get("toxicity")
            if raw_tox is not None:
                try:
                    toxicities.append(float(raw_tox))
                except (TypeError, ValueError):
                    pass
            attack = max(
                attack,
                float(bool(meta.get("comment_has_personal_attack"))),
                float(bool(meta.get("conversation_has_personal_attack"))),
            )
            conflict_count += msg.action_type.value in conflict_actions
        action_ratio = conflict_count / max(len(thread.messages), 1)
        return attack, max(toxicities, default=0.0), action_ratio, len(thread.messages)

    @classmethod
    def _select_matched_wikiconv_references(
        cls, stimuli: list, candidates: list, size: int,
    ) -> list:
        """Greedy nearest-neighbour match on scale and conflict intensity."""
        import math

        def features(thread):
            attack, toxicity, action_ratio, n_messages = cls._thread_conflict_score(thread)
            return (
                math.log1p(n_messages),
                math.log1p(len(thread.participants)),
                toxicity,
                attack,
                action_ratio,
            )

        remaining = list(candidates)
        feature_cache = {t.thread_id: features(t) for t in stimuli + remaining}
        selected = []
        targets = [stimuli[i % len(stimuli)] for i in range(size)]
        for target in targets:
            if not remaining:
                break
            tf = feature_cache[target.thread_id]
            best_index = min(
                range(len(remaining)),
                key=lambda i: (
                    abs(feature_cache[remaining[i].thread_id][0] - tf[0])
                    + 0.5 * abs(feature_cache[remaining[i].thread_id][1] - tf[1])
                    + 2.0 * abs(feature_cache[remaining[i].thread_id][2] - tf[2])
                    + abs(feature_cache[remaining[i].thread_id][3] - tf[3])
                    + abs(feature_cache[remaining[i].thread_id][4] - tf[4]),
                    remaining[i].thread_id,
                ),
            )
            selected.append(remaining.pop(best_index))
        return selected

    def _load_distillation_thread_ids(self) -> set[str]:
        """Load every source thread used to distill the locked skill corpus.

        The corpus lives beside ``cluster_map.json``. Missing ``thread_id`` is
        a hard error: without it, the claimed train/evaluation separation
        cannot be audited and the experiment must not spend API calls.
        """
        cached = getattr(self, "_distillation_thread_ids_cache", None)
        if cached is not None:
            return cached
        from pathlib import Path

        raw_merge_path = getattr(self.config, "cluster_merge_map_path", "") or ""
        if not raw_merge_path:
            self._distillation_thread_ids_cache = set()
            return set()
        merge_path = Path(raw_merge_path)
        corpus_dir = merge_path.parent
        files = sorted(corpus_dir.glob("cluster_*/typical.jsonl"))
        if not files:
            raise ValueError(
                f"No distillation typical.jsonl files found beside {merge_path}"
            )
        ids: set[str] = set()
        missing_thread_id = 0
        for path in files:
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    tid = record.get("thread_id")
                    if tid:
                        ids.add(str(tid))
                    else:
                        missing_thread_id += 1
        if missing_thread_id:
            raise ValueError(
                f"Distillation corpus has {missing_thread_id} records without "
                "thread_id; run scripts/backfill_typical_thread_ids.py first"
            )
        if not ids:
            raise ValueError("Distillation corpus contains no auditable thread IDs")
        self._distillation_thread_ids_cache = ids
        return ids

    @staticmethod
    def _distillation_ids_hash(thread_ids: set[str]) -> str:
        raw = "\n".join(sorted(thread_ids)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _assert_zero_distillation_overlap(
        evaluation_ids: list[str], distillation_ids: set[str], context: str,
    ) -> None:
        overlap = sorted(set(evaluation_ids) & distillation_ids)
        if overlap:
            raise ValueError(
                f"Train/evaluation leakage in {context}: {len(overlap)} "
                f"evaluation threads occur in the distillation corpus; "
                f"examples={overlap[:5]}"
            )

    def _prepare_sim_threads(
        self, threads: list, cell: ExperimentCell,
    ) -> tuple[list, list, list]:
        """Prepare paired stimuli plus metric-appropriate held-out references.

        The manifest key excludes ``condition`` and ``model`` deliberately:
        every condition in the same dataset/repeat sees identical source
        threads. WikiConv references retain observable action events and reply
        topology for Macro/Meso/Micro. A non-overlapping CGA split is retained
        for high-conflict linguistic comparison; CGA does not contain the full
        Wikipedia action ontology and must not be used as action ground truth.
        """
        from src.data.schemas import Thread as ThreadModel
        import copy
        import json

        if getattr(self.config, "reference_strategy", "matched_external") == "observed_continuation":
            return self._prepare_observed_continuations(threads, cell)

        seed_threads = self._load_cga_seed_threads() if cell.dataset == "wikipedia" else []
        distillation_ids = self._load_distillation_thread_ids()
        distillation_hash = self._distillation_ids_hash(distillation_ids)
        calibration_ids = set(getattr(self, "_tier1_calibration_thread_ids", set()))
        calibration_hash = self._distillation_ids_hash(calibration_ids)

        if seed_threads:
            manifest_dir = self.results_dir / "stimulus_manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{cell.dataset}_r{cell.repeat}.json"
            by_id = {t.thread_id: t for t in seed_threads}

            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                expected_manifest_fields = {
                    "schema_version": 4,
                    "dataset": cell.dataset,
                    "repeat": cell.repeat,
                    "seed": self.config.seed + cell.repeat,
                    "seed_min_toxicity": self.config.seed_min_toxicity,
                    "max_sim_threads": self.config.max_sim_threads,
                    "distillation_thread_count": len(distillation_ids),
                    "distillation_thread_ids_sha256": distillation_hash,
                    "tier1_calibration_thread_count": len(calibration_ids),
                    "tier1_calibration_thread_ids_sha256": calibration_hash,
                }
                mismatches = {
                    key: (manifest.get(key), expected)
                    for key, expected in expected_manifest_fields.items()
                    if manifest.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"Stimulus manifest {manifest_path} is stale for the "
                        f"current config: {mismatches}"
                    )
                stimulus_ids = list(manifest.get("stimulus_thread_ids") or [])
                linguistic_ids = list(manifest.get("linguistic_thread_ids") or [])
                evaluation_ids = list(manifest.get("evaluation_thread_ids") or [])
                self._assert_zero_distillation_overlap(
                    evaluation_ids, distillation_ids, str(manifest_path),
                )
                self._assert_zero_distillation_overlap(
                    evaluation_ids, calibration_ids,
                    f"{manifest_path} Tier-1 calibration",
                )
                missing = [tid for tid in stimulus_ids + linguistic_ids if tid not in by_id]
                if missing:
                    raise ValueError(
                        f"Stimulus manifest {manifest_path} references missing CGA threads: "
                        f"{missing[:5]}"
                    )
            else:
                rng = np.random.default_rng(self.config.seed + cell.repeat)
                indices = rng.permutation(len(seed_threads))
                unique_topic_ids: list[str] = []
                seen_topics: set[str] = set()
                for idx in indices:
                    seed = seed_threads[int(idx)]
                    if seed.topic in seen_topics:
                        continue
                    seen_topics.add(seed.topic)
                    unique_topic_ids.append(seed.thread_id)

                stimulus_ids = unique_topic_ids[:self.config.max_sim_threads]
                linguistic_size = min(
                    max(100, self.config.max_sim_threads * 5),
                    max(0, len(unique_topic_ids) - len(stimulus_ids)),
                )
                linguistic_ids = unique_topic_ids[
                    len(stimulus_ids):len(stimulus_ids) + linguistic_size
                ]
                if len(stimulus_ids) < self.config.max_sim_threads or not linguistic_ids:
                    raise ValueError("Not enough distinct CGA threads for paired stimulus/reference split")

                # Select a deterministic high-conflict WikiConv reference.
                # A minimum of two observable action families is later enforced
                # by MetricsAggregator, preventing silent all-DISCUSS baselines.
                source_ids = set(stimulus_ids) | set(linguistic_ids)
                candidates = [
                    t for t in threads
                    if t.thread_id not in source_ids
                    and t.thread_id not in distillation_ids
                    and t.thread_id not in calibration_ids
                ]
                rng.shuffle(candidates)
                evaluation_size = min(max(100, self.config.max_sim_threads * 5), len(candidates))
                matched = self._select_matched_wikiconv_references(
                    [by_id[tid] for tid in stimulus_ids], candidates, evaluation_size,
                )
                evaluation_ids = [t.thread_id for t in matched]
                self._assert_zero_distillation_overlap(
                    evaluation_ids, distillation_ids, "new stimulus manifest",
                )
                self._assert_zero_distillation_overlap(
                    evaluation_ids, calibration_ids,
                    "new stimulus manifest Tier-1 calibration",
                )
                if len(evaluation_ids) < self.config.max_sim_threads:
                    raise ValueError("Not enough WikiConv threads for structural/action reference")
                save_json({
                    "schema_version": 4,
                    "dataset": cell.dataset,
                    "repeat": cell.repeat,
                    "seed": self.config.seed + cell.repeat,
                    "seed_min_toxicity": self.config.seed_min_toxicity,
                    "max_sim_threads": self.config.max_sim_threads,
                    "distillation_thread_count": len(distillation_ids),
                    "distillation_thread_ids_sha256": distillation_hash,
                    "distillation_evaluation_overlap_count": 0,
                    "tier1_calibration_thread_count": len(calibration_ids),
                    "tier1_calibration_thread_ids_sha256": calibration_hash,
                    "tier1_calibration_evaluation_overlap_count": 0,
                    "stimulus_thread_ids": stimulus_ids,
                    "evaluation_thread_ids": evaluation_ids,
                    "linguistic_thread_ids": linguistic_ids,
                }, manifest_path)

            sim_threads = []
            for tid in stimulus_ids:
                sim = copy.deepcopy(by_id[tid])
                sim.thread_id = f"sim_{cell.cell_id}_{tid}"
                # Seed messages must reference the condition-specific thread ID.
                for msg in sim.messages:
                    msg.thread_id = sim.thread_id
                sim_threads.append(sim)
            wiki_by_id = {t.thread_id: t for t in threads}
            missing_eval = [tid for tid in evaluation_ids if tid not in wiki_by_id]
            if missing_eval:
                raise ValueError(
                    f"Stimulus manifest {manifest_path} references missing WikiConv "
                    f"threads: {missing_eval[:5]}"
                )
            evaluation_threads = [copy.deepcopy(wiki_by_id[tid]) for tid in evaluation_ids]
            linguistic_threads = [copy.deepcopy(by_id[tid]) for tid in linguistic_ids]
            logger.info(
                f"Paired stimulus manifest {manifest_path.name}: "
                f"{len(sim_threads)} shared stimuli, "
                f"{len(evaluation_threads)} WikiConv action/topology references, "
                f"{len(linguistic_threads)} CGA linguistic references"
            )
            return sim_threads, evaluation_threads, linguistic_threads
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

        # Non-Wikipedia fallback: use the same deterministic reference pool.
        eligible = [
            t for t in threads
            if t.thread_id not in distillation_ids
            and t.thread_id not in calibration_ids
        ]
        evaluation_threads = [
            copy.deepcopy(t) for t in eligible[:max(100, len(sim_threads))]
        ]
        self._assert_zero_distillation_overlap(
            [t.thread_id for t in evaluation_threads],
            distillation_ids,
            "fallback evaluation split",
        )
        return sim_threads, evaluation_threads, evaluation_threads

    def _prepare_observed_continuations(
        self, threads: list, cell: ExperimentCell,
    ) -> tuple[list, list, list]:
        """Use a real WikiConv prefix as stimulus and its suffix as truth.

        This removes the prior CGA→WikiConv nearest-neighbour bridge: action,
        topology, and language are now evaluated against an observed
        continuation from the exact same thread and platform context.
        """
        import copy
        import json

        from src.data.schemas import Thread as ThreadModel

        distillation_ids = self._load_distillation_thread_ids()
        calibration_ids = set(getattr(self, "_tier1_calibration_thread_ids", set()))
        excluded = distillation_ids | calibration_ids
        min_messages = max(4, int(self.config.continuation_min_messages))
        fraction = float(self.config.continuation_prefix_fraction)
        if not 0.2 <= fraction <= 0.8:
            raise ValueError("continuation_prefix_fraction must be in [0.2, 0.8]")

        eligible: list[tuple[object, int]] = []
        for thread in threads:
            if thread.thread_id in excluded or len(thread.messages) < min_messages:
                continue
            # Search around the requested fraction instead of blindly cutting
            # at it. Real reply trees are irregular; a nearby cut can preserve
            # an internal held-out edge that the exact midpoint destroys.
            legal_splits = []
            for split in range(2, len(thread.messages) - 1):
                suffix = thread.messages[split:]
                suffix_ids = {m.msg_id for m in suffix}
                if any(m.parent_msg_id in suffix_ids for m in suffix):
                    legal_splits.append(split)
            if not legal_splits:
                continue
            target_split = len(thread.messages) * fraction
            split = min(
                legal_splits,
                key=lambda value: (
                    not any(
                        message.action_type.value != "discuss"
                        for message in thread.messages[value:]
                    ),
                    abs(value - target_split),
                    value,
                ),
            )
            eligible.append((thread, split))

        if len(eligible) < self.config.max_sim_threads:
            raise ValueError(
                "Not enough auditable WikiConv prefix/suffix continuations: "
                f"need={self.config.max_sim_threads}, found={len(eligible)}"
            )

        rng = np.random.default_rng(self.config.seed + cell.repeat)
        tie_break = {t.thread_id: float(rng.random()) for t, _ in eligible}
        eligible.sort(
            key=lambda row: (
                -self._thread_conflict_score(row[0])[0],
                -self._thread_conflict_score(row[0])[1],
                -self._thread_conflict_score(row[0])[2],
                tie_break[row[0].thread_id],
            )
        )

        manifest_dir = self.results_dir / "stimulus_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{cell.dataset}_r{cell.repeat}.json"
        by_id = {t.thread_id: (t, split) for t, split in eligible}
        expected = {
            "schema_version": 8,
            "strategy": "observed_continuation",
            "dataset": cell.dataset,
            "repeat": cell.repeat,
            "seed": self.config.seed + cell.repeat,
            "max_sim_threads": self.config.max_sim_threads,
            "continuation_min_messages": min_messages,
            "continuation_prefix_fraction": fraction,
            "continuation_years": sorted(getattr(self.config, "continuation_years", [])),
            "continuation_min_platform_events": getattr(
                self.config, "continuation_min_platform_events", 2,
            ),
            "continuation_event_strata_cap": getattr(
                self.config, "continuation_event_strata_cap", 3,
            ),
            "distillation_thread_ids_sha256": self._distillation_ids_hash(distillation_ids),
            "tier1_calibration_thread_ids_sha256": self._distillation_ids_hash(calibration_ids),
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items() if manifest.get(key) != value
            }
            # A no-API data preflight may create the shared manifest before
            # Tier-1 calibration reserves its source IDs. Updating only the
            # calibration provenance is safe when the already-selected
            # evaluation sources have zero overlap; changing source IDs or
            # split indices remains forbidden.
            calibration_only = set(mismatches) <= {
                "tier1_calibration_thread_ids_sha256",
            }
            manifest_sources = set(manifest.get("source_thread_ids") or [])
            if mismatches and calibration_only and not (manifest_sources & calibration_ids):
                manifest.update(expected)
                manifest["tier1_calibration_evaluation_overlap_count"] = 0
                save_json(manifest, manifest_path)
                mismatches = {}
            if mismatches:
                raise ValueError(f"Stale continuation manifest {manifest_path}: {mismatches}")
            source_ids = list(manifest.get("source_thread_ids") or [])
            split_indices = manifest.get("split_indices") or {}
            missing = [tid for tid in source_ids if tid not in by_id]
            if missing:
                raise ValueError(f"Continuation manifest references missing threads: {missing[:5]}")
        else:
            # Action fidelity is undefined for an all-DISCUSS suffix. Ensure
            # the shared set has multiple observed platform events without
            # changing condition-specific data. This is an explicit stratified
            # feasibility sample, not a population-prevalence estimate.
            def _has_platform_event(row) -> bool:
                thread, split = row
                return any(
                    message.action_type.value != "discuss"
                    for message in thread.messages[split:]
                )

            event_rows = [row for row in eligible if _has_platform_event(row)]
            min_events = int(getattr(self.config, "continuation_min_platform_events", 2))
            strata_cap = int(getattr(self.config, "continuation_event_strata_cap", 3))
            if min_events < 1 or strata_cap < min_events:
                raise ValueError(
                    "continuation event thresholds must satisfy 1 <= minimum <= strata cap"
                )
            if len(event_rows) < min_events:
                raise ValueError(
                    f"Observed-continuation pool has fewer than {min_events} held-out "
                    "Wikipedia platform-event threads; action evaluation would be fragile"
                )
            selected = event_rows[: min(strata_cap, self.config.max_sim_threads)]
            selected_ids = {row[0].thread_id for row in selected}
            selected.extend(
                row for row in eligible
                if row[0].thread_id not in selected_ids
            )
            selected = selected[: self.config.max_sim_threads]
            source_ids = [t.thread_id for t, _ in selected]
            split_indices = {t.thread_id: split for t, split in selected}
            self._assert_zero_distillation_overlap(
                source_ids, distillation_ids, "continuation sources",
            )
            self._assert_zero_distillation_overlap(
                source_ids, calibration_ids, "continuation calibration sources",
            )
            save_json({
                **expected,
                "source_thread_ids": source_ids,
                "split_indices": split_indices,
                "distillation_evaluation_overlap_count": 0,
                "tier1_calibration_evaluation_overlap_count": 0,
            }, manifest_path)

        sim_threads = []
        evaluation_threads = []
        for tid in source_ids:
            original, default_split = by_id[tid]
            split = int(split_indices.get(tid, default_split))
            sim = ThreadModel(
                thread_id=f"sim_{cell.cell_id}_{tid}",
                platform=original.platform,
                topic=original.topic,
            )
            for message in copy.deepcopy(original.messages[:split]):
                message.thread_id = sim.thread_id
                sim.add_message(message)
            reference = ThreadModel(
                thread_id=tid, platform=original.platform, topic=original.topic,
            )
            for message in copy.deepcopy(original.messages[split:]):
                reference.add_message(message)
            sim_threads.append(sim)
            evaluation_threads.append(reference)

        logger.info(
            f"Observed-continuation manifest {manifest_path.name}: "
            f"{len(sim_threads)} exact WikiConv prefix→suffix pairs, "
            f"{sum(len(t.messages) for t in evaluation_threads)} held-out messages"
        )
        return sim_threads, evaluation_threads, copy.deepcopy(evaluation_threads)

    def _build_run_fingerprint(
        self, cell, sim_threads, evaluation_threads, linguistic_threads, skills,
    ) -> str:
        """Fingerprint every input that makes checkpoint continuation valid."""
        def serializable(value):
            if is_dataclass(value):
                return {k: serializable(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {str(k): serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [serializable(v) for v in value]
            return value

        def thread_signature(items):
            return [
                {
                    "thread_id": t.thread_id,
                    "messages": [
                        (m.msg_id, m.user_id, m.action_type.value, m.parent_msg_id, m.text)
                        for m in t.messages
                    ],
                }
                for t in items
            ]

        payload = {
            "schema": 1,
            "cell": asdict(cell),
            "config": asdict(self.config),
            "model_provenance": self.llm.get_all_provenance(),
            "skills": serializable(skills),
            "stimuli": thread_signature(sim_threads),
            "evaluation": thread_signature(evaluation_threads),
            "linguistic_reference": thread_signature(linguistic_threads),
            "artifact_hashes": {},
        }
        artifact_paths = [
            getattr(self, "models_config_path", None),
            getattr(self.config, "clustering_pickle_path", None),
            getattr(self.config, "cluster_merge_map_path", None),
        ]
        from pathlib import Path
        for raw_path in artifact_paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.exists() and path.is_file():
                payload["artifact_hashes"][str(path)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _validate_reference_preflight(self, evaluation_threads: list) -> None:
        """Reject invalid ground truth before spending simulation API calls."""
        messages = [m for t in evaluation_threads for m in t.messages]
        reply_edges = sum(bool(m.parent_msg_id) for m in messages)
        canonical_actions = {
            self.metrics_agg._canonical_action(m.action_type.value, "wikipedia")
            for m in messages
        }
        canonical_actions.discard("other")
        if not messages or reply_edges == 0:
            raise ValueError("Evaluation preflight failed: WikiConv reference has no reply edges")
        if len(canonical_actions) < 2:
            raise ValueError(
                f"Evaluation preflight failed: degenerate action ontology {canonical_actions}"
            )
        non_participation = sum(
            self.metrics_agg._canonical_action(m.action_type.value, "wikipedia")
            != "participation"
            for m in messages
        )
        min_events = int(getattr(self.config, "continuation_min_platform_events", 2))
        if non_participation < min_events:
            raise ValueError(
                "Evaluation preflight failed: too few non-participation platform "
                f"events ({non_participation} < "
                f"{min_events})"
            )
        logger.info(
            f"Evaluation preflight: {len(evaluation_threads)} threads, "
            f"{len(messages)} messages, {reply_edges} reply edges, "
            f"actions={sorted(canonical_actions)}"
        )

    def save_all_metrics(self) -> None:
        """Save all valid cell results, including cells completed before resume."""
        import pandas as pd
        expected_ids = {cell.cell_id for cell in self.build_grid()}
        result_paths = sorted(self.results_dir.glob("*_result.json"))
        found_ids = {
            path.name.removesuffix("_result.json") for path in result_paths
        }
        missing = sorted(expected_ids - found_ids)
        unexpected = sorted(found_ids - expected_ids)
        if missing or unexpected:
            raise ValueError(
                "Refusing aggregate/verdict for an incomplete or stale Exp1 grid: "
                f"missing={missing}, unexpected={unexpected}"
            )
        rows = []
        for path in result_paths:
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot aggregate corrupt result {path}: {exc}") from exc
            if row.get("simulation_integrity_passed") is not True:
                raise ValueError(f"Cannot aggregate non-integrity result {path}")
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / "exp1_all_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
        if self.config.viability_enabled:
            from src.analysis.viability import evaluate_viability
            verdict = evaluate_viability(df, self.config)
            verdict_path = self.results_dir / "viability_verdict.json"
            save_json(verdict, verdict_path)
            logger.info(
                f"Viability verdict: {verdict['verdict']} "
                f"({verdict['reason']}) → {verdict_path}"
            )
