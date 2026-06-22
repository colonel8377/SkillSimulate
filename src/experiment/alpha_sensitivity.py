"""α Sensitivity Experiment (outline §5.6.5).

Three pairwise 5×5 sweeps with the off-face tier pinned at α=1.0
(``sweep_pair`` / ``sweep_all_pairs``):
  - t1 × t2 (fix t3 = 1.0)
  - t1 × t3 (fix t2 = 1.0)  — primary, §3.2 anti-patterns/RLHF-override focus
  - t2 × t3 (fix t1 = 1.0)
Total: 3 × 25 = **75 cells per (dataset, model)**. Outline §5.6.5 explicitly
rules out a full three-dimensional grid as infeasible; the three 2D faces
incident on the (1,1,1) corner are the chosen approximation. The interior
of the cube (e.g. all tiers at 0.5 simultaneously) is an acknowledged
scope limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.config.schemas import ExperimentConfig
from src.evaluation.aggregator import MetricsAggregator
from src.experiment.exp1_validation import Experiment1Runner
from src.experiment.runner import ExperimentCell
from src.simulation.population import PopulationBuilder
from src.simulation.sandbox import SimulationSandbox
from src.utils.io import save_json


ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class AlphaCell:
    """One α configuration and its resulting metrics."""

    alpha_tier1: float  # Expression DNA
    alpha_tier2: float  # Mind Models
    alpha_tier3: float  # Anti-patterns
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"α=({self.alpha_tier1:.2f},{self.alpha_tier2:.2f},{self.alpha_tier3:.2f})"


class AlphaSensitivityRunner(Experiment1Runner):
    """Runs α sensitivity sweep reusing the Exp1 pipeline.

    Skills are compiled once per dataset (cached via parent class), then
    reused across all α configurations — only the PopulationBuilder and
    EnforcementHarness α values change between cells.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        models_config: str = "configs/models.yaml",
    ):
        super().__init__(config, models_config)

    async def _run_single_cell(
        self,
        dataset: str,
        model: str,
        alpha_tier1: float,
        alpha_tier2: float,
        alpha_tier3: float,
        repeat: int = 0,
    ) -> dict[str, float]:
        """Run one simulation cell with specific per-tier α values."""
        threads = self._load_data(dataset)
        cluster_result = self._get_or_compute_clusters(dataset, threads)
        thread_cluster_map = self._assign_threads_to_clusters(threads, cluster_result)
        skills = await self._get_or_compile_skills(dataset, threads, cluster_result, thread_cluster_map)

        cell = ExperimentCell(
            condition="cadp_full",
            dataset=dataset,
            model=model,
            repeat=repeat,
        )

        pop_builder = PopulationBuilder(
            llm_client=self.llm,
            model_name=model,
            skills=skills,
            alpha=1.0,  # global default irrelevant when per-tier overrides set
            alpha_tier1=alpha_tier1,
            alpha_tier2=alpha_tier2,
            alpha_tier3=alpha_tier3,
            backend=self.config.backend,
            memory_strategy=self.config.memory_strategy,
            compaction_interval=self.config.compaction_interval,
            compaction_keep_recent=self.config.compaction_keep_recent,
        )
        agents = pop_builder.build_population(
            cluster_result=cluster_result,
            size=self.config.population_size,
            condition="cadp_full",
            seed=self.config.seed + repeat,
        )

        sim_threads = self._prepare_sim_threads(threads, cell)
        topology = self.get_platform_topology(dataset)
        sandbox = SimulationSandbox(
            platform=topology,
            checkpoint_dir=str(self.checkpoint_dir),
            max_concurrency=self.config.max_concurrency,
        )

        run_id = f"alpha_{dataset}_{model}_t1_{alpha_tier1}_t2_{alpha_tier2}_t3_{alpha_tier3}_r{repeat}"

        result = await sandbox.run(
            agents=agents,
            threads=sim_threads,
            num_rounds=self.config.num_rounds,
            run_id=run_id,
            condition="cadp_full",
            dataset=dataset,
            model=model,
            repeat=repeat,
            checkpoint_every=self.config.checkpoint_every,
            seed=self.config.seed + repeat,
        )

        report = self.metrics_agg.evaluate(result, threads)
        return report.to_dict()

    # ------------------------------------------------------------------
    # 2D sweeps
    # ------------------------------------------------------------------

    async def sweep_pair(
        self,
        dataset: str,
        model: str,
        vary_dim: str,
        fixed_dims: dict[str, float],
    ) -> list[AlphaCell]:
        """Sweep two α dimensions, fixing the third.

        Args:
            vary_dim: Which pair to vary — "t1_t2", "t1_t3", or "t2_t3".
            fixed_dims: Dict mapping the fixed dimension name to its value.
                        E.g. {"t2": 1.0} when varying t1 and t3.
        """
        dim_map = {
            "t1": "alpha_tier1",
            "t2": "alpha_tier2",
            "t3": "alpha_tier3",
        }

        # Determine which two dims to vary
        vary_keys = [k for k in dim_map if k not in fixed_dims]
        fixed_key = list(fixed_dims)[0]
        fixed_val = fixed_dims[fixed_key]

        logger.info(
            f"Sweeping {vary_keys[0]}×{vary_keys[1]} "
            f"(fixed {fixed_key}={fixed_val}) for {dataset}/{model}"
        )

        cells: list[AlphaCell] = []
        for a1 in ALPHA_GRID:
            for a2 in ALPHA_GRID:
                params = {
                    dim_map[fixed_key]: fixed_val,
                    dim_map[vary_keys[0]]: a1,
                    dim_map[vary_keys[1]]: a2,
                }

                logger.info(f"  {params}")
                metrics = await self._run_single_cell(
                    dataset=dataset,
                    model=model,
                    **params,
                )

                cells.append(AlphaCell(
                    alpha_tier1=params["alpha_tier1"],
                    alpha_tier2=params["alpha_tier2"],
                    alpha_tier3=params["alpha_tier3"],
                    metrics=metrics,
                ))

        return cells

    async def sweep_all_pairs(
        self,
        dataset: str,
        model: str,
    ) -> dict[str, list[AlphaCell]]:
        """Run all three 2D sweeps (outline §5.6.5 key cells).

        Returns dict with keys: "t1_t2", "t1_t3", "t2_t3".
        """
        results = {}

        # Tier 1 × Tier 2, fix Tier 3 = 1.0
        results["t1_t2"] = await self.sweep_pair(
            dataset, model, "t1_t2", {"t3": 1.0}
        )

        # Tier 1 × Tier 3, fix Tier 2 = 1.0 (outline primary focus)
        results["t1_t3"] = await self.sweep_pair(
            dataset, model, "t1_t3", {"t2": 1.0}
        )

        # Tier 2 × Tier 3, fix Tier 1 = 1.0
        results["t2_t3"] = await self.sweep_pair(
            dataset, model, "t2_t3", {"t1": 1.0}
        )

        return results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def find_optimal(
        self,
        cells: list[AlphaCell],
        metric_key: str = "pred_predictive_fidelity",
    ) -> AlphaCell | None:
        """Find the α configuration maximizing the given metric."""
        def _score(cell: AlphaCell) -> float:
            if metric_key not in cell.metrics:
                raise KeyError(f"Metric {metric_key!r} not found in cell metrics")
            return cell.metrics[metric_key]

        return max(cells, key=_score) if cells else None

    def check_robustness(
        self,
        cells: list[AlphaCell],
        metric_key: str = "pred_predictive_fidelity",
        tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """Check whether the optimum is a plateau (robust) or sharp peak.

        A configuration is "robust" if the region within ``tolerance``
        of the optimum spans a wide range of α values.
        """
        if not cells:
            return {"type": "no_data", "plateau_width": 0.0}

        def _score(cell: AlphaCell) -> float:
            if metric_key not in cell.metrics:
                raise KeyError(f"Metric {metric_key!r} not found in cell metrics")
            return cell.metrics[metric_key]

        scores = [_score(c) for c in cells]
        best = max(scores)
        threshold = best * (1.0 - tolerance)

        near_optimal = [c for c, s in zip(cells, scores) if s >= threshold]
        n_near = len(near_optimal)
        n_total = len(cells)

        coverage = n_near / n_total if n_total > 0 else 0.0

        if coverage >= 0.3:
            peak_type = "plateau"
        elif coverage >= 0.15:
            peak_type = "moderate"
        else:
            peak_type = "sharp_peak"

        return {
            "type": peak_type,
            "plateau_coverage": coverage,
            "n_near_optimal": n_near,
            "n_total": n_total,
            "best_score": best,
            "threshold": threshold,
        }

    def find_optimal_per_dataset(
        self,
        sweep_results: dict[str, dict[str, list[AlphaCell]]],
        metric_key: str = "pred_predictive_fidelity",
    ) -> dict[str, dict[str, Any]]:
        """Find optimal α per dataset across all sweep pairs.

        Args:
            sweep_results: {dataset: {"t1_t2": [...], "t1_t3": [...], "t2_t3": [...]}}

        Returns:
            {dataset: {"optimal": AlphaCell, "robustness": {...}, "best_pair": str}}
        """
        results = {}
        for dataset, pairs in sweep_results.items():
            all_cells = []
            for pair_name, cells in pairs.items():
                all_cells.extend(cells)

            optimal = self.find_optimal(all_cells, metric_key)
            robustness = self.check_robustness(all_cells, metric_key)

            # Find which pair the optimal came from
            best_pair = None
            best_score = -1.0
            for pair_name, cells in pairs.items():
                opt = self.find_optimal(cells, metric_key)
                if opt:
                    score = opt.metrics.get(metric_key, 0.0)
                    if score > best_score:
                        best_score = score
                        best_pair = pair_name

            results[dataset] = {
                "optimal": optimal.label if optimal else None,
                "optimal_metrics": optimal.metrics if optimal else {},
                "robustness": robustness,
                "best_sweep_pair": best_pair,
            }

        return results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Run α sensitivity sweep across all configured datasets and models.

        Uses datasets and models from ExperimentConfig. For development,
        set config to use a single dataset/model to limit cost.
        """
        all_results = {}

        for dataset in self.config.datasets:
            for model in self.config.models:
                logger.info(f"=== α Sensitivity: {dataset} / {model} ===")

                pair_results = await self.sweep_all_pairs(dataset, model)

                all_results[f"{dataset}_{model}"] = {
                    pair: [
                        {
                            "alpha_tier1": c.alpha_tier1,
                            "alpha_tier2": c.alpha_tier2,
                            "alpha_tier3": c.alpha_tier3,
                            "metrics": c.metrics,
                        }
                        for c in cells
                    ]
                    for pair, cells in pair_results.items()
                }

                # Per-dataset/model optimal
                optimal = self.find_optimal_per_dataset(
                    {dataset: pair_results},
                )
                logger.info(f"  Optimal for {dataset}/{model}: {optimal[dataset]['optimal']}")
                logger.info(f"  Robustness: {optimal[dataset]['robustness']['type']}")

                # Save intermediate results
                result_path = self.results_dir / f"alpha_sensitivity_{dataset}_{model}.json"
                save_json(all_results[f"{dataset}_{model}"], result_path)

        return all_results
