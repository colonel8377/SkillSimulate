"""Base experiment runner with checkpoint/resume support."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.llm.client import LLMClient
from src.simulation.checkpoint import CheckpointManager
from src.utils.io import save_json
from src.utils.seed import seed_everything


@dataclass
class ExperimentCell:
    """Single experiment cell in the grid."""
    condition: str
    dataset: str
    model: str
    repeat: int

    @property
    def cell_id(self) -> str:
        return f"{self.condition}_{self.dataset}_{self.model}_r{self.repeat}"


class ExperimentRunner:
    """Base class for experiment runners."""

    def __init__(
        self,
        config: ExperimentConfig,
        models_config: str = "configs/models.yaml",
    ):
        self.config = config
        self.llm = LLMClient(models_config)
        self.checkpoint_dir = settings.simulations_dir / config.name
        self.checkpoint = CheckpointManager(self.checkpoint_dir)
        self.results_dir = settings.results_dir / config.name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def build_grid(self) -> list[ExperimentCell]:
        """Build full experiment grid."""
        cells = []
        for condition in self.config.conditions:
            for dataset in self.config.datasets:
                for model in self.config.models:
                    for repeat in range(self.config.num_repeats):
                        cells.append(ExperimentCell(
                            condition=condition,
                            dataset=dataset,
                            model=model,
                            repeat=repeat,
                        ))
        return cells

    def filter_incomplete(self, cells: list[ExperimentCell]) -> list[ExperimentCell]:
        """Filter out already-completed cells."""
        return [c for c in cells if not self.checkpoint.is_completed(c.cell_id)]

    def get_dataset_loader(self, dataset: str):
        """Get platform-specific data loader."""
        from src.data.wikipedia import WikipediaLoader
        from src.data.reddit import RedditLoader
        from src.data.github import GitHubLoader

        loaders = {
            "wikipedia": WikipediaLoader,
            "reddit": RedditLoader,
            "github": GitHubLoader,
        }
        loader_cls = loaders.get(dataset)
        if loader_cls is None:
            raise ValueError(f"Unknown dataset: {dataset}")

        data_path = settings.raw_data_dir / dataset
        return loader_cls(str(data_path))

    def get_platform_topology(self, dataset: str):
        """Get platform topology for simulation."""
        from src.simulation.platforms.wikipedia import WikipediaTopology
        from src.simulation.platforms.reddit import RedditTopology
        from src.simulation.platforms.github import GitHubTopology

        topo_map = {
            "wikipedia": WikipediaTopology,
            "reddit": RedditTopology,
            "github": GitHubTopology,
        }
        cls = topo_map.get(dataset)
        if cls is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        return cls()

    async def run_cell(self, cell: ExperimentCell) -> dict:
        """Run a single experiment cell. Override in subclasses."""
        raise NotImplementedError

    async def run_all(self) -> None:
        """Run all incomplete cells."""
        cells = self.build_grid()
        cells = self.filter_incomplete(cells)

        logger.info(f"Running {len(cells)} cells (out of {len(self.build_grid())} total)")

        for cell in tqdm(cells, desc="Experiment cells"):
            seed_everything(self.config.seed + cell.repeat)
            try:
                result = await self.run_cell(cell)
                result_path = self.results_dir / f"{cell.cell_id}_result.json"
                save_json(result, result_path)
                logger.info(f"Completed cell: {cell.cell_id}")
            except Exception as e:
                logger.error(f"Failed cell {cell.cell_id}: {e}")

        # Print cost summary
        costs = self.llm.get_cost_summary()
        logger.info(f"Cost summary: {costs}")
