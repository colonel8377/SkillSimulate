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
        model_name=config.models[0],
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


async def annotate_held_out_events(config_path: str, dataset: str, output: str | None = None) -> None:
    """Generate draft held-out event annotations via LLM dual annotator.

    Output is a DRAFT requiring human adjudication (outline §5.3 specifies
    human annotators). Saved to data/held_out_events/{dataset}.jsonl by
    default.
    """
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
    candidates = []
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
            f"{et}: κ={rep.cohen_kappa:.3f} "
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

    # list-skills
    p_list = subparsers.add_parser("list-skills", help="List available skill files")
    p_list.add_argument("--dir", type=str, default=str(settings.skills_dir))

    args = parser.parse_args()

    if args.command == "compile-skills":
        asyncio.run(compile_skills(args.config))
    elif args.command == "annotate-events":
        asyncio.run(annotate_held_out_events(args.config, args.dataset, args.output))
    elif args.command == "run":
        asyncio.run(run_experiment(args.config, args.type))
    elif args.command == "show-skill":
        show_skill(args.path)
    elif args.command == "list-skills":
        list_skills(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
