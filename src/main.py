"""Main entry point for running CADP experiments.

Usage:
  # Compile skills only (no simulation)
  python -m src.main compile-skills --config configs/dev.yaml

  # Run experiment
  python -m src.main run --config configs/dev.yaml --type exp1

  # Inspect a compiled skill
  python -m src.main show-skill --path outputs/skills/skill_cluster_0_wikipedia.yaml

  # List available skills
  python -m src.main list-skills --dir outputs/skills
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.utils.logging import setup_logging


def _build_clusterer(args) -> "BehavioralClusterer":
    """Build a BehavioralClusterer from CLI args (overrides) and optional config."""
    from src.clustering.clusterer import BehavioralClusterer

    cfg = ExperimentConfig.from_yaml(args.config) if getattr(args, "config", None) else None

    def _val(attr: str, default):
        cli = getattr(args, attr, None)
        if cli is not None:
            return cli
        if cfg is not None:
            return getattr(cfg, attr, default)
        return default

    return BehavioralClusterer(
        method=_val("cluster_method", "two_stage"),
        n_clusters=_val("num_clusters", -1),
        role_min_cluster_size=_val("role_min_cluster_size", None),
        role_min_samples=_val("role_min_samples", None),
        style_min_cluster_size=_val("style_min_cluster_size", None),
        style_min_samples=_val("style_min_samples", None),
        target_min_leaves=_val("target_min_leaves", 30),
        target_max_leaves=_val("target_max_leaves", 80),
        scaler=_val("scaler", "robust"),
        impute_orphans=bool(_val("impute_orphans", False)),
        cluster_selection_method=_val("cluster_selection_method", "eom"),
        min_style_silhouette=_val("min_style_silhouette", 0.10),
        style_umap_dim=_val("style_umap_dim", 15),
        random_state=args.seed,
    )


def _add_cluster_args(p):
    """Add clustering hyperparameter args to a subparser."""
    p.add_argument("--config", type=str, default=None,
                   help="Config YAML to load defaults from")
    p.add_argument("--cluster-method", type=str, default=None,
                   choices=["kmeans", "hdbscan", "two_stage"])
    p.add_argument("--num-clusters", type=int, default=None)
    p.add_argument("--role-mcs", type=int, default=None,
                   help="Role HDBSCAN min_cluster_size")
    p.add_argument("--role-min-samples", type=int, default=None)
    p.add_argument("--style-mcs", type=int, default=None,
                   help="Style HDBSCAN min_cluster_size")
    p.add_argument("--style-min-samples", type=int, default=None)
    p.add_argument("--target-min-leaves", type=int, default=None)
    p.add_argument("--target-max-leaves", type=int, default=None)
    p.add_argument("--scaler", type=str, default=None, choices=["standard", "robust"])
    p.add_argument("--impute-orphans", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cluster-selection-method", type=str, default=None,
                   choices=["eom", "leaf"])
    p.add_argument("--min-style-silhouette", type=float, default=None)
    p.add_argument("--style-umap-dim", type=int, default=None)


async def compile_skills(config_path: str) -> None:
    """Compile skills only — no simulation."""
    from collections import Counter

    from src.clustering.clusterer import BehavioralClusterer
    from src.data.wikipedia import WikipediaLoader
    from src.data.reddit import RedditLoader
    from src.data.github import GitHubLoader
    from src.llm.client import LLMClient
    from src.skill.compiler import SkillCompiler

    config = ExperimentConfig.from_yaml(config_path)
    setup_logging(level="INFO")

    llm = LLMClient("configs/models.yaml")
    compiler = SkillCompiler(
        llm_client=llm,
        model_name=config.compile_model or config.models[0],
        top_n_threads=config.top_n_threads,
    )

    loaders = {
        "wikipedia": WikipediaLoader,
        "reddit": RedditLoader,
        "github": GitHubLoader,
    }

    for dataset in config.datasets:
        logger.info(f"=== Compiling skills for {dataset} ===")

        loader_cls = loaders[dataset]
        loader = loader_cls(str(settings.raw_data_dir / dataset))
        threads = loader.load()

        if not threads:
            logger.warning(f"No threads loaded for {dataset}, skipping")
            continue

        clusterer = BehavioralClusterer(
            method=config.cluster_method,
            n_clusters=config.num_clusters,
            role_min_cluster_size=config.role_min_cluster_size,
            role_min_samples=config.role_min_samples,
            style_min_cluster_size=config.style_min_cluster_size,
            style_min_samples=config.style_min_samples,
            target_min_leaves=config.target_min_leaves,
            target_max_leaves=config.target_max_leaves,
            scaler=config.scaler,
            impute_orphans=config.impute_orphans,
            cluster_selection_method=config.cluster_selection_method,
            min_style_silhouette=config.min_style_silhouette,
            style_umap_dim=config.style_umap_dim,
            random_state=config.seed,
        )
        cluster_result = clusterer.fit(threads)

        # Assign threads to clusters
        thread_cluster_map = {}
        for thread in threads:
            votes = Counter()
            for uid in thread.participants:
                cid = cluster_result.labels.get(uid)
                if cid is not None:
                    votes[cid] += 1
            thread_cluster_map[thread.thread_id] = votes.most_common(1)[0][0] if votes else 0

        skills = await compiler.compile_all(
            platform=dataset,
            cluster_result=cluster_result,
            all_threads=threads,
            thread_cluster_map=thread_cluster_map,
        )

        for cid, skill in skills.items():
            compiler.save_skill(skill)

        logger.info(f"Compiled {len(skills)} skills for {dataset}")

    costs = llm.get_cost_summary()
    logger.info(f"Cost summary: {costs}")


def _resolve_loader(dataset: str, data_dir: str | None, limit: int | None = None):
    """Return an instantiated dataset loader, honouring a --data-dir override."""
    from src.data.wikipedia import WikipediaLoader
    from src.data.reddit import RedditLoader
    from src.data.github import GitHubLoader

    path = data_dir or str(settings.raw_data_dir / dataset)
    if dataset == "wikipedia":
        return WikipediaLoader(path, limit=limit)
    return {"reddit": RedditLoader, "github": GitHubLoader}[dataset](path)


def cluster_eda(dataset: str, data_dir: str | None, output: str | None,
                seed: int = 42, limit: int | None = None,
                min_messages: int = 5, workers: int = 8,
                clusterer: "BehavioralClusterer | None" = None) -> None:
    """Step 0a: run exploratory clustering statistics on a dataset."""
    setup_logging(level="INFO")
    out = output or str(settings.output_dir / "eda" / f"{dataset}_eda.json")

    # Streaming path for large ConvoKit corpora (no population sampling).
    if data_dir and (Path(data_dir) / "wikiconv-2010").exists():
        import numpy as np
        from src.clustering.streaming import run_streaming_pipeline
        from src.clustering.eda import save_report
        cr, accums = run_streaming_pipeline(
            data_dir, min_messages=min_messages, workers=workers,
            clusterer=clusterer,
        )
        from src.skill.cluster_profile import ArchetypeProfiler
        leaf_feat = {}
        for l in cr.get_cluster_ids():
            if l < 0:
                continue
            vecs = [cr.user_features[u].to_vector() for u in cr.get_cluster_members(l)
                    if u in cr.user_features]
            if vecs:
                leaf_feat[l] = np.mean(vecs, axis=0)
        tags = ArchetypeProfiler()._compute_tags(leaf_feat)
        report = {
            "n_users_clustered": len(cr.user_features),
            "n_pre_impute_orphans": getattr(cr, "pre_impute_orphans", 0),
            "n_orphans_kept": getattr(cr, "n_orphans_kept", 0),
            "clustering": {
                "n_leaves": cr.n_clusters,
                "leaf_sizes": {int(l): len(cr.get_cluster_members(l))
                               for l in cr.get_cluster_ids() if l >= 0},
                "silhouette": cr.silhouette_score,
                "davies_bouldin": cr.davies_bouldin_score,
                "leaf_tags": {int(k): v for k, v in tags.items()},
            },
        }
        save_report(report, out)
        return

    from src.clustering.eda import run_eda, save_report
    loader = _resolve_loader(dataset, data_dir, limit)
    logger.info(f"Loading {dataset} from {loader.data_path} ...")
    threads = loader.load()
    if not threads:
        logger.error("No threads loaded — check --data-dir.")
        return
    report = run_eda(threads, seed=seed, clusterer=clusterer)
    save_report(report, out)


def export_corpus(dataset: str, data_dir: str | None, output: str | None,
                  seed: int = 42, limit: int | None = None,
                  min_messages: int = 5, workers: int = 8,
                  clusterer: "BehavioralClusterer | None" = None) -> None:
    """Step 1: cluster, freeze typical utterances, export per-leaf material packs."""
    setup_logging(level="INFO")
    out = Path(output) if output else (settings.output_dir / "skill_corpus")

    if data_dir and (Path(data_dir) / "wikiconv-2010").exists():
        import numpy as np
        from src.clustering.streaming import (
            run_streaming_pipeline, collect_member_utterances, embed_sample_texts,
        )
        from src.skill.cluster_profile import ArchetypeProfiler, LeafProfile, TypicalUtterance
        from src.skill.corpus_export import export_corpus_packs

        cr, accums = run_streaming_pipeline(
            data_dir, min_messages=min_messages, workers=workers,
            clusterer=clusterer,
        )
        # language centroid per leaf → representative members
        emb = embed_sample_texts(accums)
        leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]
        rep_members: dict[int, list[str]] = {}
        all_members: set[str] = set()
        for l in leaf_ids:
            members = [u for u in cr.get_cluster_members(l) if u in emb]
            if not members:
                continue
            cent = np.mean([emb[u] for u in members], axis=0)
            ranked = sorted(members, key=lambda u: float(np.linalg.norm(emb[u] - cent)))
            top = ranked[:8]
            rep_members[l] = top
            all_members.update(top)

        member_utts = collect_member_utterances(
            __import__("src.clustering.streaming", fromlist=["find_year_dirs"]).find_year_dirs(data_dir),
            all_members, max_per_user=60, workers=workers,
        )

        # tags
        leaf_feat = {}
        for l in leaf_ids:
            vecs = [cr.user_features[u].to_vector() for u in cr.get_cluster_members(l)
                    if u in cr.user_features]
            if vecs:
                leaf_feat[l] = np.mean(vecs, axis=0)
        tags = ArchetypeProfiler()._compute_tags(leaf_feat)

        profiles: dict[int, LeafProfile] = {}
        for l in leaf_ids:
            members = cr.get_cluster_members(l)
            utts = []
            for m in rep_members.get(l, []):
                for it in member_utts.get(m, []):
                    utts.append(TypicalUtterance(
                        member=m, action=it["action"], text=it["text"],
                        parent_context=it["parent_context"], topic=it["topic"],
                    ))
            profiles[l] = LeafProfile(
                leaf_id=l, members=rep_members.get(l, []),
                typical_utterances=utts, tags=tags.get(l, []), size=len(members),
            )

        profiler = ArchetypeProfiler()
        profiler.save(profiles, out, dataset)
        export_corpus_packs([], cr, profiles, out, dataset)
        logger.info(f"Exported {len(profiles)} streaming leaf packs to {out / dataset}")
        return

    from src.clustering.clusterer import BehavioralClusterer
    from src.skill.cluster_profile import ArchetypeProfiler
    from src.skill.corpus_export import export_corpus_packs

    loader = _resolve_loader(dataset, data_dir, limit)
    logger.info(f"Loading {dataset} from {loader.data_path} ...")
    threads = loader.load()
    if not threads:
        logger.error("No threads loaded — check --data-dir.")
        return

    clusterer = clusterer or BehavioralClusterer(random_state=seed)
    cluster_result = clusterer.fit(threads)
    profiler = ArchetypeProfiler()
    profiles = profiler.build(threads, cluster_result)

    out = Path(output) if output else (settings.output_dir / "skill_corpus")
    profiler.save(profiles, out, dataset)            # typical.jsonl + profile.json
    export_corpus_packs(threads, cluster_result, profiles, out, dataset)  # for_colleague/for_nuwa


async def run_experiment(config_path: str, exp_type: str = "exp1") -> None:
    """Run an experiment from config."""
    from src.experiment.exp1_validation import Experiment1Runner
    from src.experiment.exp2_simulation import Experiment2Runner
    from src.experiment.transfer_test import CrossDatasetTransferRunner

    config = ExperimentConfig.from_yaml(config_path)
    setup_logging(level="INFO")

    logger.info(f"Starting {exp_type} experiment: {config.name}")

    if exp_type in ("exp1", "exp2"):
        logger.info(
            f"Grid: {len(config.conditions)} conditions × "
            f"{len(config.datasets)} datasets × "
            f"{len(config.models)} models × "
            f"{config.num_repeats} repeats"
        )

    if exp_type == "exp1":
        runner = Experiment1Runner(config)
        await runner.run_all()
        if hasattr(runner, "save_all_metrics"):
            runner.save_all_metrics()
        costs = runner.llm.get_cost_summary()
        logger.info(f"Final cost summary: {costs}")

    elif exp_type == "exp2":
        runner = Experiment2Runner(config)
        await runner.run_all()
        if hasattr(runner, "save_all_metrics"):
            runner.save_all_metrics()
        costs = runner.llm.get_cost_summary()
        logger.info(f"Final cost summary: {costs}")

    elif exp_type == "transfer":
        runner = CrossDatasetTransferRunner(config)
        datasets = config.datasets
        models = config.models
        transfer_mode = config.transfer_mode
        for model in models:
            for i in range(len(datasets)):
                for j in range(len(datasets)):
                    if i == j:
                        continue
                    result = await runner.run_transfer_test(
                        source_dataset=datasets[i],
                        target_dataset=datasets[j],
                        model=model,
                        transfer_mode=transfer_mode,
                    )
                    result_path = runner.results_dir / (
                        f"transfer_{transfer_mode}_{datasets[i]}_to_{datasets[j]}_{model}.json"
                    )
                    from src.utils.io import save_json
                    save_json(result, result_path)
                    logger.info(
                        f"Transfer ({transfer_mode}) {datasets[i]}→{datasets[j]} ({model}): "
                        f"fidelity_ratio={result['transfer_fidelity_ratio']:.3f}"
                    )
        costs = runner.llm.get_cost_summary()
        logger.info(f"Final cost summary: {costs}")

    elif exp_type == "trigger_calibration":
        from src.experiment.trigger_calibration import TriggerCalibrationRunner

        runner = TriggerCalibrationRunner(config)
        from src.utils.io import save_json

        for dataset in config.datasets:
            result = await runner.run(dataset)
            result_path = runner.results_dir / f"trigger_calibration_{dataset}.json"
            save_json(result, result_path)

            # Cross-dataset transfer (all pairs)
            for target in config.datasets:
                if target == dataset:
                    continue
                transfer = await runner.run_cross_dataset(dataset, target)
                transfer_path = runner.results_dir / (
                    f"trigger_transfer_{dataset}_to_{target}.json"
                )
                save_json(transfer, transfer_path)

        costs = runner.llm.get_cost_summary()
        logger.info(f"Final cost summary: {costs}")

    elif exp_type == "alpha_sensitivity":
        from src.experiment.alpha_sensitivity import AlphaSensitivityRunner

        runner = AlphaSensitivityRunner(config)
        results = await runner.run()

        from src.utils.io import save_json
        result_path = runner.results_dir / "alpha_sensitivity_all.json"
        save_json(results, result_path)
        logger.info(f"Saved α sensitivity results to {result_path}")

        costs = runner.llm.get_cost_summary()
        logger.info(f"Final cost summary: {costs}")

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


async def annotate_held_out_events(
    config_path: str,
    dataset: str,
    output: str | None = None,
    sim_dir: str | None = None,
) -> None:
    """Generate draft held-out event annotations via LLM dual annotator.

    Output is a DRAFT requiring human adjudication (outline §5.3 specifies
    human annotators). Saved to data/held_out_events/{dataset}.jsonl by
    default.

    When ``--sim-dir`` is provided, reads simulation threads from .json
    round snapshots instead of real data. Useful for generating pilot
    held-out events from prior simulation runs.
    """
    import json as _json
    from collections import defaultdict

    from src.data.wikipedia import WikipediaLoader
    from src.data.reddit import RedditLoader
    from src.data.github import GitHubLoader
    from src.evaluation.held_out_events import (
        LLMDualAnnotator,
        resolve_events,
        save_events,
    )
    from src.llm.client import LLMClient
    from src.utils.io import save_json

    config = ExperimentConfig.from_yaml(config_path)
    setup_logging(level="INFO")

    candidates = []

    if sim_dir:
        # --- Read threads from simulation .json snapshots ---
        sim_path = Path(sim_dir)
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

        snapshot_files = sorted(sim_path.glob(f"**/*{dataset}*_round_*.json"))
        if not snapshot_files:
            raise FileNotFoundError(
                f"No *{dataset}*_round_*.json snapshots found in {sim_dir}"
            )

        # Group by run_id, pick latest round per run
        run_snapshots: dict[str, list] = defaultdict(list)
        for sf in snapshot_files:
            name = sf.stem
            parts = name.rsplit("_round_", 1)
            if len(parts) == 2:
                run_id = parts[0]
                try:
                    round_num = int(parts[1])
                except ValueError:
                    continue
                run_snapshots[run_id].append((round_num, sf))

        logger.info(
            f"Found {len(run_snapshots)} simulation runs with "
            f"{len(snapshot_files)} total snapshots for {dataset}"
        )

        conflict_actions = {"disagree", "revert", "counter_argue", "report"}
        for run_id, snapshots in run_snapshots.items():
            snapshots.sort(key=lambda x: x[0], reverse=True)
            _, latest_path = snapshots[0]
            try:
                data = _json.loads(latest_path.read_text())
                messages_log = data.get("messages_log", [])
            except Exception as e:
                logger.warning(f"Failed to read {latest_path}: {e}")
                continue

            if not messages_log:
                continue

            # Group messages by thread
            threads: dict[str, list[dict]] = defaultdict(list)
            for m in messages_log:
                threads[m["thread_id"]].append(m)

            for tid, msgs in threads.items():
                participants = {m["user_id"] for m in msgs}
                if len(participants) < 2:
                    continue
                # Accept all threads (sim data may not have explicit
                # conflict action types; the LLM annotator will judge
                # based on text content)
                exchange = "\n".join(
                    f"[{m['user_id']}] ({m['action_type']}): {m['text'][:200]}"
                    for m in msgs[:15]
                )[:2000]
                candidates.append({
                    "thread_id": f"{run_id}__{tid}",
                    "topic": f"Simulation: {run_id}",
                    "exchange": exchange,
                })
                if len(candidates) >= 50:
                    break
            if len(candidates) >= 50:
                break

        logger.info(
            f"Extracted {len(candidates)} threads from simulation snapshots"
        )
    else:
        # --- Read threads from real data (original behaviour) ---
        loaders = {
            "wikipedia": WikipediaLoader,
            "reddit": RedditLoader,
            "github": GitHubLoader,
        }
        loader_cls = loaders.get(dataset)
        if loader_cls is None:
            raise ValueError(f"Unknown dataset: {dataset}")

        loader = loader_cls(str(settings.raw_data_dir / dataset))
        threads = loader.load()
        logger.info(f"Loaded {len(threads)} threads from {dataset}")

        # Sample controversial threads (>= 2 participants, has conflict actions)
        conflict_actions = {"disagree", "revert", "counter_argue", "report"}
        for t in threads:
            if len(t.participants) < 2:
                continue
            has_conflict = any(m.action_type.value in conflict_actions for m in t.messages)
            if not has_conflict:
                continue
            exchange = "\n".join(
                f"[{m.user_id}] ({m.action_type.value}): {m.text[:200]}"
                for m in t.messages[:15]
            )[:2000]
            candidates.append({
                "thread_id": t.thread_id,
                "topic": t.topic,
                "exchange": exchange,
            })
            if len(candidates) >= 50:
                break

    logger.info(f"Annotating {len(candidates)} controversial threads for {dataset}")

    llm = LLMClient("configs/models.yaml")
    annotator = LLMDualAnnotator(llm, model_name=config.models[0])
    events = await annotator.annotate_threads(candidates)

    reports = resolve_events(events)
    for et, rep in reports.items():
        logger.info(
            f"{et}: \u03ba={rep.cohen_kappa:.3f} "
            f"(agreed={rep.n_agreed}, disputed={rep.n_disputed}, "
            f"meets_threshold={rep.meets_threshold})"
        )

    out_path = output or str(settings.held_out_events_dir / f"{dataset}.jsonl")
    save_events(events, out_path)

    agreement_path = str(settings.results_dir / f"agreement_{dataset}.json")
    save_json({et: r.to_dict() for et, r in reports.items()}, agreement_path)

    costs = llm.get_cost_summary()
    logger.info(f"Cost summary: {costs}")


def show_skill(path: str) -> None:
    """Display a compiled skill file."""
    from src.skill.compiler import SkillCompiler

    skill = SkillCompiler.load_skill(path)
    print(f"\n{'='*60}")
    print(f"Cluster: {skill.cluster_id} | Platform: {skill.platform}")
    print(f"Compiled: {skill.compiled_at}")
    print(f"Source users: {skill.source_user_count}")
    print(f"Source threads: {len(skill.source_thread_ids)}")
    print(f"{'='*60}\n")

    if skill.capability:
        edna = skill.capability.expression_dna
        print("--- Expression DNA ---")
        print(f"  Avg sentence length: {edna.avg_sentence_length}")
        print(f"  Formal/casual: {edna.style_formal_casual}")
        print(f"  Cautious/assertive: {edna.style_cautious_assertive}")
        print(f"  Vocab richness: {edna.vocab_richness}")
        print(f"  High freq words: {edna.high_freq_words[:10]}")
        print()

        print(f"--- Mind Models ({len(skill.capability.mind_models)}) ---")
        for mm in skill.capability.mind_models:
            verified = []
            if mm.cross_domain_verified: verified.append("cross-domain")
            if mm.predictive_verified: verified.append("predictive")
            if mm.exclusive_verified: verified.append("exclusive")
            print(f"  [{', '.join(verified)}] {mm.name}: {mm.description}")
        print()

    if skill.constraint:
        print(f"--- Anti-patterns ({len(skill.constraint.anti_patterns)}) ---")
        for ap in skill.constraint.anti_patterns:
            print(f"  - {ap.description}")
            if ap.trigger_keywords:
                print(f"    Keywords: {ap.trigger_keywords}")
        print()


def list_skills(skills_dir: str) -> None:
    """List all available skill files."""
    from src.skill.compiler import SkillCompiler

    skills = SkillCompiler.load_all_skills(skills_dir)
    if not skills:
        print(f"No skill files found in {skills_dir}")
        return

    print(f"\nFound {len(skills)} skill files in {skills_dir}:\n")
    print(f"{'Cluster':<10} {'Platform':<15} {'Mind Models':<12} {'Anti-patterns':<14} {'Users'}")
    print("-" * 65)
    for cid, skill in sorted(skills.items()):
        n_mm = len(skill.capability.mind_models) if skill.capability else 0
        n_ap = len(skill.constraint.anti_patterns) if skill.constraint else 0
        print(f"{cid:<10} {skill.platform:<15} {n_mm:<12} {n_ap:<14} {skill.source_user_count}")


def main():
    parser = argparse.ArgumentParser(description="CADP Experiment Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compile-skills
    p_compile = subparsers.add_parser("compile-skills", help="Compile .skill files from data")
    p_compile.add_argument("--config", type=str, default="configs/dev.yaml")

    # run
    p_run = subparsers.add_parser("run", help="Run experiment")
    p_run.add_argument("--config", type=str, default="configs/dev.yaml")
    p_run.add_argument("--type", type=str, default="exp1", choices=["exp1", "exp2", "transfer", "trigger_calibration", "alpha_sensitivity"])

    # show-skill
    p_show = subparsers.add_parser("show-skill", help="Display a compiled skill file")
    p_show.add_argument("--path", type=str, required=True)

    # annotate-events
    p_anno = subparsers.add_parser(
        "annotate-events",
        help="Generate draft held-out event annotations (LLM dual annotator)",
    )
    p_anno.add_argument("--config", type=str, default="configs/dev.yaml")
    p_anno.add_argument("--dataset", type=str, required=True)
    p_anno.add_argument("--output", type=str, default=None)
    p_anno.add_argument(
        "--sim-dir", type=str, default=None,
        help="Read threads from simulation .json round snapshots instead "
             "of real data (e.g. outputs/simulations/exp1_pilot)",
    )

    # list-skills
    p_list = subparsers.add_parser("list-skills", help="List available skill files")
    p_list.add_argument("--dir", type=str, default=str(settings.skills_dir))

    # cluster-eda (Step 0a)
    p_eda = subparsers.add_parser("cluster-eda", help="Exploratory clustering stats")
    p_eda.add_argument("--dataset", type=str, default="wikipedia")
    p_eda.add_argument("--data-dir", type=str, default=None,
                       help="Override data dir (e.g. data/raw/wikiconv_en)")
    p_eda.add_argument("--output", type=str, default=None)
    p_eda.add_argument("--seed", type=int, default=42)
    p_eda.add_argument("--limit", type=int, default=0,
                       help="Max utterances to load (0 = all/full dataset)")
    p_eda.add_argument("--min-messages", type=int, default=5,
                       help="Activity floor: users with fewer msgs → out-of-cluster")
    p_eda.add_argument("--workers", type=int, default=8,
                       help="Parallel year-streaming workers")
    _add_cluster_args(p_eda)

    # export-corpus (Step 1)
    p_exp = subparsers.add_parser("export-corpus", help="Export per-leaf distillation material")
    p_exp.add_argument("--dataset", type=str, default="wikipedia")
    p_exp.add_argument("--data-dir", type=str, default=None,
                       help="Override data dir (e.g. data/raw/wikiconv_en)")
    p_exp.add_argument("--output", type=str, default=None)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--limit", type=int, default=0,
                       help="Max utterances to load (0 = all/full dataset)")
    p_exp.add_argument("--min-messages", type=int, default=5,
                       help="Activity floor: users with fewer msgs → out-of-cluster")
    p_exp.add_argument("--workers", type=int, default=8,
                       help="Parallel year-streaming workers")
    _add_cluster_args(p_exp)

    args = parser.parse_args()

    if args.command == "compile-skills":
        asyncio.run(compile_skills(args.config))
    elif args.command == "annotate-events":
        asyncio.run(annotate_held_out_events(
            args.config, args.dataset, args.output, args.sim_dir,
        ))
    elif args.command == "run":
        asyncio.run(run_experiment(args.config, args.type))
    elif args.command == "show-skill":
        show_skill(args.path)
    elif args.command == "list-skills":
        list_skills(args.dir)
    elif args.command == "cluster-eda":
        cluster_eda(args.dataset, args.data_dir, args.output, args.seed,
                    args.limit or None, args.min_messages, args.workers,
                    clusterer=_build_clusterer(args))
    elif args.command == "export-corpus":
        export_corpus(args.dataset, args.data_dir, args.output, args.seed,
                      args.limit or None, args.min_messages, args.workers,
                      clusterer=_build_clusterer(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
