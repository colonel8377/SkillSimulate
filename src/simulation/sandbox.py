"""Main simulation orchestrator."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from src.agents.base import BaseAgent
from src.data.schemas import Message, Thread
from src.enforcement.harness import EnforcementLog
from src.simulation.checkpoint import CheckpointManager
from src.simulation.platforms.base import PlatformTopology


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    run_id: str
    condition: str
    dataset: str
    model: str
    repeat: int
    rounds: int
    messages: list[dict] = field(default_factory=list)
    agent_states: list[dict] = field(default_factory=list)
    interaction_graph: dict | None = None
    enforcement_stats: dict = field(default_factory=dict)
    per_round_metrics: list[dict] = field(default_factory=list)  # trajectory data

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "dataset": self.dataset,
            "model": self.model,
            "repeat": self.repeat,
            "rounds": self.rounds,
            "messages": self.messages,
            "agent_states": self.agent_states,
            "interaction_graph": self.interaction_graph,
            "enforcement_stats": self.enforcement_stats,
            "per_round_metrics": self.per_round_metrics,
        }


class SimulationSandbox:
    """Orchestrates multi-agent simulation on a platform."""

    def __init__(
        self,
        platform: PlatformTopology,
        checkpoint_dir: str | None = None,
        max_concurrency: int = 4,
    ):
        self.platform = platform
        self.max_concurrency = max_concurrency
        self.checkpoint = CheckpointManager(checkpoint_dir) if checkpoint_dir else None

    async def run(
        self,
        agents: list[BaseAgent],
        threads: list[Thread],
        num_rounds: int,
        run_id: str,
        condition: str = "",
        dataset: str = "",
        model: str = "",
        repeat: int = 0,
        checkpoint_every: int = 5,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run the simulation.

        Args:
            agents: List of agents.
            threads: Discussion threads to simulate on.
            num_rounds: Number of interaction rounds.
            run_id: Unique run identifier.
            condition: Experimental condition name.
            dataset: Dataset name.
            model: Model name.
            repeat: Repeat index.
            checkpoint_every: Save checkpoint every N rounds.
            seed: Optional int seed for reproducible per-round thread sampling
                and engagement gating (R2 reproducibility fix). When None,
                falls back to a non-deterministic RNG (legacy behaviour).

        Returns:
            SimulationResult with all messages and states.
        """
        all_messages: list[dict] = []
        enforcement_violations = 0
        enforcement_total = 0
        per_round_metrics: list[dict] = []

        # Per-run seeded RNG — used for thread-sampling and any other
        # stochastic sandbox decision (R2 reproducibility). The RNG is
        # consumed only from the sequential per-round loop (not from
        # concurrent agent coroutines), so a single instance is safe.
        self._run_rng = random.Random(seed) if seed is not None else random.Random()

        # Inject the platform topology into every agent so that take_turn's
        # parent-message selection can honour platform-specific reply rules
        # (outline §6.3 platform topology fidelity). When an agent was
        # constructed before this Sandbox, it has no topology reference yet.
        for agent in agents:
            agent.platform_topology = self.platform

        # Check for existing checkpoint to resume. We consult BOTH the
        # round-level snapshot (carries agents_state) AND the turn-level
        # JSONL (carries per-turn messages, the resume-granularity source
        # of truth — Issue 2). When both exist, the JSONL wins for
        # ``all_messages`` because it is more recent.
        start_round = 0
        if self.checkpoint:
            latest = self.checkpoint.load(run_id)
            if latest:
                start_round = latest["round"] + 1
                all_messages = latest.get("messages_log", [])
                logger.info(f"Resuming run {run_id} from round {start_round} (round-level snapshot)")
            # Turn-level JSONL may have progress beyond the last round
            # snapshot (up to ``checkpoint_every - 1`` rounds + partial
            # current round). Prefer it when present.
            turns = self.checkpoint.load_turns(run_id)
            if turns:
                ok_turns = [t for t in turns if t.get("status") == "ok" and t.get("message")]
                if ok_turns:
                    jsonl_rounds = {t.get("round", 0) for t in ok_turns}
                    last_jsonl_round = max(jsonl_rounds)
                    # Rebuild all_messages from JSONL — it's at least as
                    # recent as the round snapshot and may be newer.
                    all_messages = [t["message"] for t in ok_turns]
                    # Resume from the round AFTER the last fully-attested
                    # round in the JSONL. The last round may have been
                    # partial (crash mid-round) — we accept the loss of
                    # its unfinished turns rather than try to reconstruct
                    # RNG-dependent thread sampling.
                    start_round = last_jsonl_round + 1
                    logger.info(
                        f"Resuming run {run_id} from round {start_round} "
                        f"(turn-level JSONL: {len(ok_turns)} ok turns across "
                        f"rounds {min(jsonl_rounds)}-{last_jsonl_round})"
                    )

        logger.info(
            f"Starting simulation: {run_id} | "
            f"{len(agents)} agents, {len(threads)} threads, "
            f"rounds {start_round}-{num_rounds - 1}"
        )

        for round_num in range(start_round, num_rounds):
            round_messages, round_violations, round_total = await self._run_round(
                agents, threads, round_num, run_id=run_id,
            )

            all_messages.extend(round_messages)
            enforcement_violations += round_violations
            enforcement_total += round_total

            logger.info(
                f"Run {run_id} round {round_num}/{num_rounds - 1}: "
                f"{len(round_messages)} messages, "
                f"{round_violations}/{round_total} enforcement violations"
            )

            # Compute per-round trajectory metrics
            round_metric = self._compute_round_metrics(round_messages, agents, round_num)
            per_round_metrics.append(round_metric)

            # Periodic checkpoint
            if self.checkpoint and (round_num + 1) % checkpoint_every == 0:
                self.checkpoint.save(
                    run_id=run_id,
                    round_num=round_num,
                    agents_state=[a.get_state_summary() for a in agents],
                    messages_log=all_messages,
                )

        # Build interaction graph
        # NOTE (D1): the Sandbox no longer emits a co-presence interaction
        # graph. The previous implementation threaded message co-presence
        # into ``interaction_graph``, but every consumer
        # (MetricsAggregator) rebuilds the graph from ``parent_msg_id``
        # reply chains, so the field was dead. Keep the slot in
        # SimulationResult for backward-compatible de/serialisation and
        # set it to None.
        result = SimulationResult(
            run_id=run_id,
            condition=condition,
            dataset=dataset,
            model=model,
            repeat=repeat,
            rounds=num_rounds,
            messages=all_messages,
            agent_states=[a.get_state_summary() for a in agents],
            interaction_graph=None,
            enforcement_stats={
                "total_checks": enforcement_total,
                "total_violations": enforcement_violations,
                "violation_rate": enforcement_violations / max(enforcement_total, 1),
            },
            per_round_metrics=per_round_metrics,
        )

        if self.checkpoint:
            # Final round-level snapshot. Note: we deliberately do NOT
            # call ``mark_completed`` here — the result file is written
            # by the *runner* (ExperimentRunner.run_all) under a different
            # directory (results_dir, not simulations_dir), and the
            # COMPLETE marker needs to reference that real file so the
            # integrity check in ``is_completed`` passes.
            self.checkpoint.save(
                run_id=run_id,
                round_num=num_rounds - 1,
                agents_state=result.agent_states,
                messages_log=all_messages,
                interaction_graph_data=result.interaction_graph,
                extra={"enforcement_stats": result.enforcement_stats},
            )

        return result

    async def _run_round(
        self,
        agents: list[BaseAgent],
        threads: list[Thread],
        round_num: int,
        agent_engagement_ratio: float = 0.5,
        run_id: str = "",
    ) -> tuple[list[dict], int, int]:
        """Run a single round of simulation.

        Args:
            agents: All agents.
            threads: Discussion threads.
            round_num: Current round number.
            agent_engagement_ratio: Default engagement ratio, overridden
                by per-agent ``agent.engagement_ratio`` if set.
            run_id: Cell identifier — used for turn-level checkpoint
                appends (Issue 2). Empty string disables checkpointing.

        Returns:
            Tuple of (messages, violations, total_checks).
        """
        # R2: use the per-run seeded RNG (set in ``run``) instead of the
        # global ``random`` module so identical seeds reproduce identical
        # per-round thread-sampling sequences. The RNG is only consumed
        # here in the sequential outer for-loop, never from inside the
        # concurrent ``agent.take_turn`` coroutines, so a shared instance
        # is safe across the semaphore-gated agent turns.
        rng = self._run_rng
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Phase 1 (sequential — preserves seeded-RNG reproducibility):
        # decide which threads each agent engages this round, in agent order.
        # The RNG is consumed ONLY here, never inside the concurrent agent
        # coroutines below, so identical seeds reproduce identical
        # thread-engagement decisions. The *designed* randomness (who engages
        # what) stays deterministic; only execution *interleaving* (a
        # non-designed variable) becomes concurrent in Phase 2.
        agent_plan = []
        turn_idx = 0
        for agent in agents:
            ratio = getattr(agent, "engagement_ratio", agent_engagement_ratio)
            n_threads_for_agent = max(1, int(len(threads) * ratio))
            if len(threads) > n_threads_for_agent:
                agent_threads = rng.sample(
                    threads, min(n_threads_for_agent, len(threads))
                )
            else:
                agent_threads = list(threads)
            agent_plan.append((agent, agent_threads, turn_idx))
            turn_idx += len(agent_threads)

        # Phase 2 (concurrent): one coroutine per agent, gathered. Each agent
        # runs its OWN threads sequentially → no two concurrent turns for the
        # same agent (avoids racing its memory / state). Different agents run
        # concurrently, gated per-turn by ``semaphore``. Cross-agent shared
        # state (thread.add_message, other.observe, list appends) is mutated
        # only via synchronous ops between awaits — atomic in the
        # single-threaded event loop — so concurrent turns are safe; only
        # their interleaving order is non-deterministic (acceptable: real
        # interaction is concurrent, and the aggregate metrics are computed
        # from the parent_msg_id graph / message corpus, which are
        # order-independent). asyncio.gather returns results in submission
        # (agent) order, so the merged ``messages`` list stays agent-major,
        # matching the legacy serial loop's ordering.
        async def _run_agent_turns(agent, agent_threads, base_turn_idx):
            local_messages = []
            local_violations = 0
            local_total = 0
            turn_idx = base_turn_idx
            for thread in agent_threads:
                async with semaphore:
                    try:
                        available = self.platform.get_valid_actions(
                            thread, agent.agent_id
                        )
                        if not available:
                            continue

                        msg, enf_log = await agent.take_turn(
                            thread, available, round_num
                        )

                        # Use the agent's message directly (no duplicate creation)
                        thread.add_message(msg)

                        # Record message with parent info for chain analysis.
                        # Also propagate ``metadata`` so downstream evaluation
                        # (outline §5.7 / §4.4.2 step 4) can stratify by the
                        # ``constraint_forced`` safe-template flag.
                        msg_dict = {
                            "msg_id": msg.msg_id,
                            "thread_id": msg.thread_id,
                            "user_id": msg.user_id,
                            "action_type": msg.action_type.value,
                            "text": msg.text,
                            "round": round_num,
                            "parent_msg_id": msg.parent_msg_id,
                            "metadata": dict(msg.metadata) if msg.metadata else {},
                        }
                        local_messages.append(msg_dict)

                        # Issue 2: append a turn record to the JSONL so
                        # the next resume picks up at this exact turn.
                        if self.checkpoint is not None and run_id:
                            self.checkpoint.append_turn(
                                run_id=run_id,
                                round_num=round_num,
                                turn_idx=turn_idx,
                                agent_id=agent.agent_id,
                                status="ok",
                                message=msg_dict,
                            )

                        # Other agents observe this message
                        for other in agents:
                            if other.agent_id != agent.agent_id:
                                other.observe(msg, round_num)

                        # Track enforcement
                        if enf_log:
                            local_total += enf_log.total_violations + 1
                            local_violations += enf_log.total_violations

                    except Exception as e:
                        import traceback
                        logger.error(
                            f"Error in round {round_num}, "
                            f"agent {agent.agent_id}: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                        # Issue 2: record the failure in the JSONL so the
                        # next resume knows this turn produced no message
                        # (avoids silent drops corrupting metrics).
                        if self.checkpoint is not None and run_id:
                            self.checkpoint.append_turn(
                                run_id=run_id,
                                round_num=round_num,
                                turn_idx=turn_idx,
                                agent_id=agent.agent_id,
                                status="failed",
                                error=repr(e),
                            )
                turn_idx += 1
            return local_messages, local_violations, local_total

        results = await asyncio.gather(
            *[_run_agent_turns(a, t, i) for a, t, i in agent_plan]
        )

        messages = []
        violations = 0
        total_checks = 0
        for local_messages, local_violations, local_total in results:
            messages.extend(local_messages)
            violations += local_violations
            total_checks += local_total

        return messages, violations, total_checks

    def _compute_round_metrics(
        self,
        round_messages: list[dict],
        agents: list[BaseAgent],
        round_num: int,
    ) -> dict:
        """Compute per-round metrics for trajectory analysis (outline §6.4).

        Tracks: polarization proxy, conflict ratio, unique participants,
        action diversity, message count.
        """
        from collections import Counter

        if not round_messages:
            return {"round": round_num, "message_count": 0}

        # Action distribution
        actions = Counter(m["action_type"] for m in round_messages)
        # B2 fix: ``close``/``reopen`` are normal GitHub issue-lifecycle
        # actions (outline §4.3), not conflict signals — including them
        # inflated the GitHub dataset's conflict_ratio / escalation
        # trajectory. ``block`` is kept: it carries an adversarial signal
        # (an editor reverting another editor's edit-war protection).
        conflict_actions = {"disagree", "revert", "counter_argue", "report", "block"}
        conflict_count = sum(actions.get(a, 0) for a in conflict_actions)
        total = len(round_messages)

        # Polarization proxy: fraction of conflict actions
        conflict_ratio = conflict_count / total if total > 0 else 0.0

        # Unique participants
        participants = set(m["user_id"] for m in round_messages)

        # Action diversity (entropy)
        import numpy as np
        from scipy.stats import entropy
        action_probs = np.array(list(actions.values())) / total
        action_entropy = float(entropy(action_probs)) if len(actions) > 1 else 0.0

        # Interaction graph density (this round only)
        thread_users: dict[str, set[str]] = {}
        for m in round_messages:
            thread_users.setdefault(m["thread_id"], set()).add(m["user_id"])
        n_edges = sum(len(u) * (len(u) - 1) // 2 for u in thread_users.values())
        max_edges = len(agents) * (len(agents) - 1) // 2
        graph_density = n_edges / max(max_edges, 1)

        return {
            "round": round_num,
            "message_count": total,
            "conflict_count": conflict_count,
            "conflict_ratio": conflict_ratio,
            "polarization_proxy": conflict_ratio,  # used for trajectory plots
            "unique_participants": len(participants),
            "action_diversity": action_entropy,
            "interaction_density": graph_density,
        }
