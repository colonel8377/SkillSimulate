"""Checkpoint manager for simulation state persistence.

Three artefact types live in ``checkpoint_dir``:

  * ``{run_id}_round_{N}.json``  — round-level snapshot (every ``checkpoint_every``
    rounds). Compact form of the JSONL below; rewritten periodically.
  * ``{run_id}_turns.jsonl``     — append-only turn-level log. One JSON record
    per line, written after every successful or failed agent turn. This is
    the resume-granularity source of truth (Issue 2).
  * ``{run_id}_COMPLETE.json``   — cell-completion marker. ``is_completed``
    validates that the referenced ``result_path`` still parses, so a
    corrupt result file triggers re-run rather than silent skip.
  * ``{run_id}_FAILED.json``     — cell-failure marker for the circuit-breaker
    trip (Issue 2). Carries a ``recoverable`` flag: when True the cell is
    re-attempted on the next ``run_all()``; when False it stays skipped
    so a permanently-broken cell doesn't loop forever.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

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

    # ------------------------------------------------------------------
    # Turn-level checkpoint (Issue 2)
    # ------------------------------------------------------------------

    def _turns_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}_turns.jsonl"

    def append_turn(
        self,
        run_id: str,
        round_num: int,
        turn_idx: int,
        agent_id: str,
        status: str,  # "ok" | "failed"
        message: dict | None = None,
        error: str | None = None,
    ) -> Path:
        """Append a single turn record to the JSONL turn log.

        O(1) append — no rewrite. This is the resume-granularity source
        of truth: on crash, the next ``run()`` invocation replays the
        JSONL to rebuild the exact state at the last successful turn,
        losing at most the in-flight turn that was interrupted.

        Args:
            run_id: Cell identifier.
            round_num: Round number within the cell.
            turn_idx: Index of the turn within the round.
            agent_id: Agent that took the turn.
            status: "ok" if the turn produced a message, "failed" if it
                raised and was dropped by the per-turn try/except.
            message: Serialized Message dict on success; None on failure.
            error: Exception repr on failure; None on success.

        Returns:
            Path to the JSONL file.
        """
        record = {
            "run_id": run_id,
            "round": round_num,
            "turn_idx": turn_idx,
            "agent_id": agent_id,
            "status": status,
            "message": message,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        path = self._turns_path(run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False))
            f.write("\n")
        return path

    def load_turns(self, run_id: str) -> list[dict]:
        """Replay the turn-level JSONL log.

        Returns an empty list when the file does not exist (fresh run).
        Corrupt individual lines are logged and skipped — a partial JSONL
        is preferable to losing the whole resume state.
        """
        path = self._turns_path(run_id)
        if not path.exists():
            return []
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"CheckpointManager.load_turns({run_id}): skipping "
                        f"corrupt line {lineno} ({exc})."
                    )
        return records

    def truncate_turns_after(self, run_id: str, round_num: int, turn_idx: int) -> None:
        """Drop turn records strictly after (round_num, turn_idx).

        Used during resume when a partially-written turn record needs to
        be discarded (e.g. the runner crashed mid-turn and the last
        record is incomplete). Records at exactly (round_num, turn_idx)
        are kept.
        """
        path = self._turns_path(run_id)
        if not path.exists():
            return
        kept: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r_round = rec.get("round", 0)
                r_turn = rec.get("turn_idx", 0)
                if (r_round, r_turn) <= (round_num, turn_idx):
                    kept.append(line)
        with path.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line)
                f.write("\n")

    def get_last_successful_turn(self, run_id: str) -> tuple[int, int] | None:
        """Return (round, turn_idx) of the last ``status="ok"`` turn.

        Used by the sandbox to know where to resume. Returns None when
        no successful turn has been recorded yet.
        """
        records = self.load_turns(run_id)
        for rec in reversed(records):
            if rec.get("status") == "ok":
                return (rec.get("round", 0), rec.get("turn_idx", 0))
        return None

    # ------------------------------------------------------------------
    # COMPLETE / FAILED markers
    # ------------------------------------------------------------------

    def is_completed(self, run_id: str) -> bool:
        """Check if a run has a valid completion marker.

        Beyond checking marker existence, validates that the referenced
        ``result_path`` still exists and parses as JSON with at least one
        top-level key. A corrupt or missing result file → treat as
        not-completed so the cell re-runs rather than silently skipping.
        """
        marker = self.checkpoint_dir / f"{run_id}_COMPLETE.json"
        if not marker.exists():
            return False
        try:
            data = load_json(marker)
        except Exception as exc:
            logger.warning(
                f"CheckpointManager.is_completed({run_id}): marker corrupt ({exc}); "
                f"treating as not-completed."
            )
            return False
        result_path = data.get("result_path") if isinstance(data, dict) else None
        if not result_path:
            logger.warning(
                f"CheckpointManager.is_completed({run_id}): marker has no result_path; "
                f"treating as not-completed."
            )
            return False
        rp = Path(result_path)
        if not rp.exists():
            logger.warning(
                f"CheckpointManager.is_completed({run_id}): result file {rp} missing; "
                f"treating as not-completed."
            )
            return False
        try:
            result_data = load_json(rp)
        except Exception as exc:
            logger.warning(
                f"CheckpointManager.is_completed({run_id}): result file {rp} "
                f"corrupt ({exc}); treating as not-completed."
            )
            return False
        if not isinstance(result_data, dict) or not result_data:
            logger.warning(
                f"CheckpointManager.is_completed({run_id}): result file {rp} "
                f"empty/non-dict; treating as not-completed."
            )
            return False
        return True

    def mark_completed(self, run_id: str, result_path: str | Path) -> None:
        """Mark a run as complete."""
        marker = self.checkpoint_dir / f"{run_id}_COMPLETE.json"
        save_json({
            "run_id": run_id,
            "completed_at": datetime.now().isoformat(),
            "result_path": str(result_path),
        }, marker)

    # --- FAILED marker (Issue 2 — circuit-breaker trip) --------------

    def is_failed(self, run_id: str) -> bool:
        """Check if a run has a failure marker."""
        marker = self.checkpoint_dir / f"{run_id}_FAILED.json"
        return marker.exists()

    def get_failure(self, run_id: str) -> dict | None:
        """Return the failure marker dict, or None if not marked failed."""
        marker = self.checkpoint_dir / f"{run_id}_FAILED.json"
        if not marker.exists():
            return None
        try:
            return load_json(marker)
        except Exception as exc:
            logger.warning(
                f"CheckpointManager.get_failure({run_id}): marker corrupt ({exc})."
            )
            return None

    def mark_failed(
        self,
        run_id: str,
        reason: str,
        last_error: str = "",
        consecutive_failures: int = 0,
        recoverable: bool = True,
        resume_attempts: int = 0,
    ) -> Path:
        """Write (or update) the failure marker for a run.

        Args:
            run_id: Cell identifier.
            reason: Short reason code (e.g. "circuit_breaker_open").
            last_error: Repr of the last exception that triggered the trip.
            consecutive_failures: Breaker counter at trip time.
            recoverable: When True, the cell is re-attempted on the next
                ``run_all()`` invocation. When False (e.g. resume_attempts
                exceeded), the cell stays skipped.
            resume_attempts: How many times this cell has been re-attempted
                after a previous failure. Incremented by the runner on
                each re-attempt.

        Returns:
            Path to the marker file.
        """
        marker = self.checkpoint_dir / f"{run_id}_FAILED.json"
        save_json({
            "run_id": run_id,
            "reason": reason,
            "last_error": last_error,
            "consecutive_failures": consecutive_failures,
            "recoverable": recoverable,
            "resume_attempts": resume_attempts,
            "failed_at": datetime.now().isoformat(),
        }, marker)
        return marker

    def clear_failed(self, run_id: str) -> None:
        """Remove the failure marker — called when a previously-failed
        cell eventually completes successfully."""
        marker = self.checkpoint_dir / f"{run_id}_FAILED.json"
        if marker.exists():
            marker.unlink()
