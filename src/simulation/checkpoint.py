"""Checkpoint manager for simulation state persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.io import save_json, load_json, save_pickle, load_pickle


class CheckpointManager:
    """Manages save/resume of simulation state."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        run_id: str,
        round_num: int,
        agents_state: list[dict],
        messages_log: list[dict],
        interaction_graph_data: dict | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Save simulation checkpoint.

        Args:
            run_id: Unique run identifier.
            round_num: Current round number.
            agents_state: Serializable agent states.
            messages_log: All messages so far.
            interaction_graph_data: Serialized interaction graph.
            extra: Any additional metadata.

        Returns:
            Path to checkpoint file.
        """
        checkpoint = {
            "run_id": run_id,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "agents_state": agents_state,
            "messages_log": messages_log,
            "interaction_graph_data": interaction_graph_data,
            "extra": extra or {},
        }

        path = self.checkpoint_dir / f"{run_id}_round_{round_num}.json"
        save_json(checkpoint, path)
        return path

    def load(self, run_id: str, round_num: int | None = None) -> dict | None:
        """Load checkpoint.

        Args:
            run_id: Run identifier.
            round_num: Specific round, or None for latest.

        Returns:
            Checkpoint dict, or None if not found.
        """
        if round_num is not None:
            path = self.checkpoint_dir / f"{run_id}_round_{round_num}.json"
            if path.exists():
                return load_json(path)
            return None

        # Find latest checkpoint for this run
        pattern = f"{run_id}_round_*.json"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None
        return load_json(checkpoints[-1])

    def get_latest_round(self, run_id: str) -> int | None:
        """Get the latest completed round for a run."""
        cp = self.load(run_id)
        if cp is None:
            return None
        return cp.get("round")

    def is_completed(self, run_id: str) -> bool:
        """Check if a run has a completion marker."""
        marker = self.checkpoint_dir / f"{run_id}_COMPLETE.json"
        return marker.exists()

    def mark_completed(self, run_id: str, result_path: str | Path) -> None:
        """Mark a run as complete."""
        marker = self.checkpoint_dir / f"{run_id}_COMPLETE.json"
        save_json({
            "run_id": run_id,
            "completed_at": datetime.now().isoformat(),
            "result_path": str(result_path),
        }, marker)
