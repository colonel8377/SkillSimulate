"""Re-evaluate all completed exp1_pilot cells from simulation checkpoints.

Loads the final-round checkpoint for each cell, reconstructs a
SimulationResult, and runs MetricsAggregator.evaluate() with the
fixed macro.py (partition coverage + edgeless-graph guard).

Usage:
    python -m scripts.reevaluate_pilot
"""

from __future__ import annotations

import asyncio
import glob
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("outputs/simulations/exp1_pilot")
RESULTS_DIR = Path("outputs/results/exp1_pilot")
DATASET = "wikipedia"
NUM_ROUNDS = 15  # rounds 0..14 → final checkpoint is round 14


def _find_final_checkpoint(run_id: str) -> Path | None:
    """Find the highest-round checkpoint for a given run_id."""
    pattern = str(CHECKPOINT_DIR / f"{run_id}_round_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # Pick the highest round number
    best = None
    best_round = -1
    for f in files:
        m = re.search(r"_round_(\d+)\.json$", f)
        if m:
            r = int(m.group(1))
            if r > best_round:
                best_round = r
                best = Path(f)
    return best


def _load_checkpoint(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _reconstruct_result(checkpoint: dict) -> "SimulationResult":
    """Build a SimulationResult from checkpoint data."""
    from src.simulation.sandbox import SimulationResult

    extra = checkpoint.get("extra", {})
    enforcement_stats = extra.get("enforcement_stats", {})

    return SimulationResult(
        run_id=checkpoint["run_id"],
        condition=checkpoint["run_id"].rsplit("_", 2)[0],  # e.g. "vanilla"
        dataset=DATASET,
        model="deepseek-v4-flash",
        repeat=int(checkpoint["run_id"].rsplit("_r", 1)[-1]),
        rounds=checkpoint.get("round", NUM_ROUNDS - 1) + 1,
        messages=checkpoint.get("messages_log", []),
        agent_states=checkpoint.get("agents_state", []),
        interaction_graph=None,
        enforcement_stats=enforcement_stats,
        per_round_metrics=[],  # not persisted in checkpoints
    )


def _parse_condition_and_repeat(run_id: str) -> tuple[str, int]:
    """Extract condition and repeat from run_id like 'vanilla_wikipedia_deepseek-v4-flash_r0'."""
    # Pattern: {condition}_{dataset}_{model}_r{repeat}
    m = re.match(r"^(.+?)_wikipedia_deepseek-v4-flash_r(\d+)$", run_id)
    if m:
        return m.group(1), int(m.group(2))
    # Fallback
    return run_id, 0


async def main() -> None:
    from src.config.settings import settings
    from src.data.wikipedia import WikipediaLoader
    from src.evaluation.aggregator import MetricsAggregator
    from src.llm.client import LLMClient
    from src.utils.io import save_json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load real data
    logger.info("Loading Wikipedia dataset...")
    data_path = str(settings.raw_data_dir / DATASET)
    loader = WikipediaLoader(data_path)
    threads = loader.load()
    logger.info(f"Loaded {len(threads)} real threads")

    # Build aggregator
    llm = LLMClient()
    aggregator = MetricsAggregator(
        held_out_events_dir=str(settings.held_out_events_dir),
        role_labels_dir=str(settings.role_labels_dir),
        model_provenance=llm.get_all_provenance(),
    )

    # Find all completed cells
    complete_files = sorted(glob.glob(str(CHECKPOINT_DIR / "*_COMPLETE.json")))
    logger.info(f"Found {len(complete_files)} completed cells")

    success = 0
    failed = 0

    for comp_path in complete_files:
        with open(comp_path) as f:
            comp = json.load(f)
        run_id = comp["run_id"]

        # Find final checkpoint
        ckpt_path = _find_final_checkpoint(run_id)
        if ckpt_path is None:
            logger.warning(f"No checkpoint found for {run_id}, skipping")
            failed += 1
            continue

        logger.info(f"Re-evaluating {run_id} from {ckpt_path.name}...")

        try:
            checkpoint = _load_checkpoint(ckpt_path)
            sim_result = _reconstruct_result(checkpoint)

            # Fix condition/repeat from run_id
            condition, repeat = _parse_condition_and_repeat(run_id)
            sim_result.condition = condition
            sim_result.repeat = repeat

            # Evaluate
            report = aggregator.evaluate(sim_result, threads)
            result = report.to_dict()

            # Save
            result_path = RESULTS_DIR / f"{run_id}_result.json"
            save_json(result, result_path)
            logger.info(f"  Saved {result_path.name} ({len(result)} keys)")
            success += 1

        except Exception as e:
            logger.error(f"  FAILED {run_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"\nDone: {success} succeeded, {failed} failed out of {len(complete_files)}")

    # Also save CSV
    if aggregator.reports:
        df = aggregator.to_dataframe()
        csv_path = RESULTS_DIR / "exp1_all_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregated metrics to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
