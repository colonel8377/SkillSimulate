"""Base experiment runner with checkpoint/resume support."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.llm.circuit_breaker import get_breaker, reset_global_breaker
from src.llm.client import LLMClient
from src.llm.exceptions import CircuitBreakerOpen
from src.simulation.checkpoint import CheckpointManager
from src.utils.io import save_json
from src.utils.seed import seed_everything


# How many times a FAILED cell can be re-attempted across run_all()
# invocations before being marked permanently failed (recoverable=False).
# Default 3 — after 3 manual re-runs the user is expected to investigate
# the underlying cause rather than have us silently loop.
DEFAULT_RESUME_ATTEMPTS_LIMIT = int(os.getenv("CADP_RESUME_ATTEMPTS_LIMIT", "3"))


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
        """Filter out already-completed cells.

        Cells marked ``_FAILED.json`` with ``recoverable=False`` are also
        skipped — they've exhausted their ``resume_attempts_limit`` and
        need explicit user intervention (delete the marker file to
        re-enable). Cells with ``recoverable=True`` are re-attempted.
        """
        out: list[ExperimentCell] = []
        for c in cells:
            if self.checkpoint.is_completed(c.cell_id):
                continue
            failure = self.checkpoint.get_failure(c.cell_id)
            if failure is not None and not failure.get("recoverable", True):
                logger.warning(
                    f"Skipping cell {c.cell_id}: marked FAILED with "
                    f"recoverable=False (reason={failure.get('reason')}). "
                    f"Delete {self.checkpoint_dir}/{c.cell_id}_FAILED.json "
                    f"to re-enable."
                )
                continue
            out.append(c)
        return out

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
        """Run all incomplete cells.

        Circuit-breaker policy (Issue 2): when ``CircuitBreakerOpen`` is
        raised — i.e. the global breaker has tripped after
        ``CADP_BREAKER_THRESHOLD`` (default 3) consecutive LLM failures —
        we write a ``_FAILED.json`` marker on the current cell, log the
        trip, and exit non-zero (sys.exit 2) so an operator can
        investigate. The breaker is then reset so the next ``run_all()``
        invocation starts fresh (the FAILED cell will be re-attempted
        up to ``resume_attempts_limit`` times).
        """
        # Fresh breaker for this run_all() invocation.
        reset_global_breaker()
        _ = get_breaker()  # ensure initialized

        cells = self.build_grid()
        cells = self.filter_incomplete(cells)

        logger.info(f"Running {len(cells)} cells (out of {len(self.build_grid())} total)")

        try:
            for cell in tqdm(cells, desc="Experiment cells"):
                seed_everything(self.config.seed + cell.repeat)
                try:
                    result = await self.run_cell(cell)
                    result_path = self.results_dir / f"{cell.cell_id}_result.json"
                    save_json(result, result_path)
                    # Mark complete with the *real* result file path so
                    # the integrity check in is_completed passes. Clear
                    # any stale FAILED marker from a prior attempt.
                    self.checkpoint.mark_completed(cell.cell_id, result_path)
                    self.checkpoint.clear_failed(cell.cell_id)
                    logger.info(f"Completed cell: {cell.cell_id}")
                except CircuitBreakerOpen as exc:
                    # Trip — write FAILED marker and halt the whole run.
                    self._on_circuit_breaker_trip(cell, exc)
                    raise  # Re-raise so the outer handler runs once and exits.
                except Exception as e:
                    # Per-cell failure that did NOT trip the breaker
                    # (e.g. a one-off transient that exhausted its 5
                    # retries but didn't cross the breaker threshold of
                    # 3 *consecutive* failures). Log and continue.
                    logger.error(
                        f"Failed cell {cell.cell_id}: {e!r}. "
                        f"Cell will be re-attempted on next run_all()."
                    )
                    # Record a soft failure marker so the operator can
                    # see which cells are problematic, but keep it
                    # recoverable so the next run_all() re-attempts.
                    self.checkpoint.mark_failed(
                        cell.cell_id,
                        reason="cell_exception",
                        last_error=repr(e),
                        recoverable=True,
                    )
        except CircuitBreakerOpen:
            # Outer handler — breaker tripped, halt the run.
            logger.error(
                "Circuit breaker tripped — halting run_all() for human "
                "intervention. Fix the underlying cause and re-run; "
                "previously-completed cells are skipped via COMPLETE "
                "markers, and recoverable FAILED cells are re-attempted."
            )
            sys.exit(2)

        # Print cost summary
        costs = self.llm.get_cost_summary()
        logger.info(f"Cost summary: {costs}")

    def _on_circuit_breaker_trip(self, cell: ExperimentCell, exc: CircuitBreakerOpen) -> None:
        """Write a FAILED marker with the appropriate recoverable flag.

        Increments the cell's ``resume_attempts`` counter; when it
        crosses ``DEFAULT_RESUME_ATTEMPTS_LIMIT`` we mark the cell
        ``recoverable=False`` so the next ``run_all()`` skips it instead
        of looping forever on a permanently-broken cell.
        """
        prior = self.checkpoint.get_failure(cell.cell_id) or {}
        resume_attempts = int(prior.get("resume_attempts", 0)) + 1
        recoverable = resume_attempts < DEFAULT_RESUME_ATTEMPTS_LIMIT
        self.checkpoint.mark_failed(
            cell.cell_id,
            reason="circuit_breaker_open",
            last_error=str(exc),
            consecutive_failures=exc.consecutive_failures,
            recoverable=recoverable,
            resume_attempts=resume_attempts,
        )
        logger.error(
            f"Circuit breaker tripped on cell {cell.cell_id}: "
            f"{exc}. resume_attempts={resume_attempts}/{DEFAULT_RESUME_ATTEMPTS_LIMIT} "
            f"recoverable={recoverable}."
        )
