"""Main simulation orchestrator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from src.agents.base import BaseAgent
from src.data.schemas import ActionType, Message, Thread
from src.enforcement.harness import EnforcementLog
from src.simulation.checkpoint import CheckpointManager
from src.simulation.platforms.base import PlatformTopology


class SimulationIntegrityError(RuntimeError):
    """Raised when a run is incomplete and must not be evaluated."""


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
    seed_messages: list[dict] = field(default_factory=list)
    run_fingerprint: str = ""

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
            "seed_messages": self.seed_messages,
            "run_fingerprint": self.run_fingerprint,
        }


class SimulationSandbox:
    """Orchestrates multi-agent simulation on a platform."""

    def __init__(
        self,
        platform: PlatformTopology,
        checkpoint_dir: str | None = None,
        max_concurrency: int = 4,
        micro_batch_size: int = 0,
        min_turn_success_rate: float = 0.95,
    ):
        self.platform = platform
        self.max_concurrency = max_concurrency
        self.micro_batch_size = max(0, int(micro_batch_size))
        self.min_turn_success_rate = min_turn_success_rate
        self.checkpoint = CheckpointManager(checkpoint_dir) if checkpoint_dir else None
        # Safe defaults also make the lower-level round primitive deterministic
        # when exercised directly in preflight/tests. ``run`` replaces both.
        self._run_rng = random.Random()
        self._run_fingerprint = ""

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
        run_fingerprint: str = "",
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
        if not run_fingerprint:
            fallback_payload = {
                "run_id": run_id, "condition": condition, "dataset": dataset,
                "model": model, "repeat": repeat, "num_rounds": num_rounds,
                "agents": [(a.agent_id, a.state.cluster_id) for a in agents],
                "threads": [
                    (t.thread_id, [(m.msg_id, m.parent_msg_id) for m in t.messages])
                    for t in threads
                ],
            }
            run_fingerprint = hashlib.sha256(
                json.dumps(fallback_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
        self._run_fingerprint = run_fingerprint
        all_messages: list[dict] = []
        enforcement_violations = 0
        enforcement_total = 0
        per_round_metrics: list[dict] = []
        seed_messages = [
            {
                "msg_id": m.msg_id,
                "thread_id": m.thread_id,
                "user_id": m.user_id,
                "action_type": m.action_type.value,
                "text": m.text,
                "timestamp": m.timestamp.isoformat(),
                "parent_msg_id": m.parent_msg_id,
                "metadata": dict(m.metadata or {}),
            }
            for thread in threads for m in thread.messages
        ]

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
            # Re-scope the Tier-3 judge log to this cell: agent ids
            # (agent_0..agent_N) repeat across cells, so without the
            # run-prefixed file name and per-record run context, verdicts
            # from different repeats would append indistinguishably into
            # the same JSONL and be unusable for human annotation.
            harness = getattr(agent, "enforcement_harness", None)
            judge = getattr(harness, "tier3_llm_judge", None) if harness else None
            if judge is not None:
                judge.set_run_id(f"{run_id}_{agent.agent_id}")
                judge.set_run_context({
                    "run_id": run_id,
                    "condition": condition,
                    "dataset": dataset,
                    "model": model,
                    "repeat": repeat,
                })

        # Resume only from a fully completed, lossless round snapshot. A
        # turn-level JSONL can end in the middle of a concurrent round and
        # therefore cannot attest that the round is complete. Partial later
        # turns are discarded and that round is replayed in full.
        start_round = 0
        if self.checkpoint:
            latest = self.checkpoint.load(run_id)
            if latest:
                extra = latest.get("extra") or {}
                if extra.get("checkpoint_schema_version") != 2 or not extra.get("round_complete"):
                    raise SimulationIntegrityError(
                        f"Run {run_id} has a legacy/incomplete checkpoint at round "
                        f"{latest.get('round')}. Archive or remove this run's old "
                        "checkpoint artifacts before restarting from round 0."
                    )
                if extra.get("run_fingerprint") != run_fingerprint:
                    raise SimulationIntegrityError(
                        f"Run {run_id}: checkpoint fingerprint does not match current "
                        "config/stimulus/skill/model inputs"
                    )
                start_round = latest["round"] + 1
                all_messages = latest.get("messages_log", [])
                enforcement_violations = int(extra.get("enforcement_violations", 0))
                enforcement_total = int(extra.get("enforcement_total", 0))
                per_round_metrics = list(extra.get("per_round_metrics") or [])

                saved_agents = {a.get("agent_id"): a for a in latest.get("agents_state", [])}
                if set(saved_agents) != {a.agent_id for a in agents}:
                    raise SimulationIntegrityError(
                        f"Run {run_id}: checkpoint population does not match rebuilt population"
                    )
                for agent in agents:
                    agent.restore_runtime_state(saved_agents[agent.agent_id])

                # Rebuild the generated part of every thread. Seed messages are
                # already present in the freshly prepared thread objects.
                thread_by_id = {t.thread_id: t for t in threads}
                for raw in all_messages:
                    thread = thread_by_id.get(raw.get("thread_id"))
                    if thread is None:
                        raise SimulationIntegrityError(
                            f"Run {run_id}: checkpoint references unknown thread "
                            f"{raw.get('thread_id')}; stimulus manifest changed"
                        )
                    if any(m.msg_id == raw.get("msg_id") for m in thread.messages):
                        continue
                    thread.add_message(self._message_from_dict(raw, thread))

                rng_state = extra.get("rng_state")
                if rng_state is None:
                    raise SimulationIntegrityError(f"Run {run_id}: checkpoint has no RNG state")
                self._run_rng.setstate(self._lists_to_tuples(rng_state))
                self.checkpoint.truncate_turns_after_round(run_id, latest["round"])
                logger.info(
                    f"Resuming run {run_id} from fully checkpointed round {start_round}; "
                    "any partial later round will be replayed"
                )
            elif self.checkpoint.load_turns(run_id):
                # Crash before the first completed round snapshot: no state is
                # safe to restore, so replay round 0 and clear partial turns.
                self.checkpoint.truncate_turns_after_round(run_id, -1)
                logger.warning(
                    f"Run {run_id} had partial turn records but no completed "
                    "round snapshot; replaying from round 0"
                )

        logger.info(
            f"Starting simulation: {run_id} | "
            f"{len(agents)} agents, {len(threads)} threads, "
            f"rounds {start_round}-{num_rounds - 1}"
        )

        for round_num in range(start_round, num_rounds):
            round_messages, round_violations, round_total, planned_turns = await self._run_round(
                agents, threads, round_num, run_id=run_id,
            )

            success_rate = len(round_messages) / max(planned_turns, 1)
            if planned_turns <= 0 or success_rate < self.min_turn_success_rate:
                raise SimulationIntegrityError(
                    f"Run {run_id} round {round_num}: only {len(round_messages)}/"
                    f"{planned_turns} planned turns succeeded ({success_rate:.1%}); "
                    f"minimum is {self.min_turn_success_rate:.1%}. Refusing evaluation."
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
            round_metric = self._compute_round_metrics(
                round_messages, agents, round_num, planned_turns=planned_turns,
            )
            per_round_metrics.append(round_metric)

            # Periodic checkpoint
            if self.checkpoint and (round_num + 1) % checkpoint_every == 0:
                self.checkpoint.save(
                    run_id=run_id,
                    round_num=round_num,
                    agents_state=[a.get_runtime_state() for a in agents],
                    messages_log=all_messages,
                    extra=self._checkpoint_extra(
                        per_round_metrics, enforcement_violations, enforcement_total,
                    ),
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
                "safe_template_count": sum(
                    bool((m.get("metadata") or {}).get("constraint_forced", False))
                    for m in all_messages
                ),
                "safe_template_rate": sum(
                    bool((m.get("metadata") or {}).get("constraint_forced", False))
                    for m in all_messages
                ) / max(len(all_messages), 1),
                "successful_turns": sum(
                    int(row.get("message_count", 0)) for row in per_round_metrics
                ),
                "planned_turns": sum(
                    int(row.get("planned_turns", 0)) for row in per_round_metrics
                ),
                "turn_success_rate": (
                    sum(int(row.get("message_count", 0)) for row in per_round_metrics)
                    / max(sum(int(row.get("planned_turns", 0)) for row in per_round_metrics), 1)
                ),
            },
            per_round_metrics=per_round_metrics,
            seed_messages=seed_messages,
            run_fingerprint=run_fingerprint,
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
                agents_state=[a.get_runtime_state() for a in agents],
                messages_log=all_messages,
                interaction_graph_data=result.interaction_graph,
                extra=self._checkpoint_extra(
                    per_round_metrics, enforcement_violations, enforcement_total,
                    enforcement_stats=result.enforcement_stats,
                ),
            )

        return result

    async def _run_round(
        self,
        agents: list[BaseAgent],
        threads: list[Thread],
        round_num: int,
        agent_engagement_ratio: float = 0.5,
        run_id: str = "",
    ) -> tuple[list[dict], int, int, int]:
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
        # thread-engagement decisions. Generation uses a frozen round-start
        # view and commit order is deterministic.
        agent_plan = []
        turn_idx = 0
        for agent in agents:
            ratio = getattr(agent, "engagement_ratio", agent_engagement_ratio)
            # A low-activity empirical draw may legitimately skip a round.
            # The round-level integrity guard still fails if the entire
            # population plans no turns.
            n_threads_for_agent = max(0, int(len(threads) * ratio))
            if n_threads_for_agent == 0:
                agent_threads = []
                agent_plan.append((agent, agent_threads, turn_idx))
                continue
            if len(threads) > n_threads_for_agent:
                agent_threads = rng.sample(
                    threads, min(n_threads_for_agent, len(threads))
                )
            else:
                agent_threads = list(threads)
            agent_plan.append((agent, agent_threads, turn_idx))
            turn_idx += len(agent_threads)

        # Phase 2: generate concurrently against the immutable round-start
        # thread state. No coroutine mutates shared threads or another agent's
        # memory. Results are committed later in deterministic plan order.
        # This removes LLM-latency-dependent prompts and reply targets while
        # retaining API concurrency.
        async def _run_agent_turns(agent, agent_threads, base_turn_idx):
            local_results = []
            turn_idx = base_turn_idx
            for thread in agent_threads:
                async with semaphore:
                    try:
                        available = self.platform.get_valid_actions(
                            thread, agent.agent_id
                        )
                        if not available:
                            local_results.append((thread, turn_idx, None, None, "no_valid_actions"))
                            turn_idx += 1
                            continue

                        msg, enf_log = await agent.take_turn(
                            thread, available, round_num
                        )
                        local_results.append((thread, turn_idx, msg, enf_log, None))

                    except Exception as e:
                        import traceback
                        logger.error(
                            f"Error in round {round_num}, "
                            f"agent {agent.agent_id}: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                        local_results.append((thread, turn_idx, None, None, repr(e)))
                turn_idx += 1
            return agent, local_results

        # Phase 3: deterministic micro-batch commit. Later batches observe
        # messages produced by earlier batches in the same round. A value of
        # zero retains the legacy single frozen round for other experiments.
        messages: list[dict] = []
        violations = 0
        total_checks = 0
        batch_size = self.micro_batch_size or len(agent_plan) or 1
        for start in range(0, len(agent_plan), batch_size):
            batch = agent_plan[start:start + batch_size]
            results = await asyncio.gather(
                *[_run_agent_turns(a, t, i) for a, t, i in batch]
            )
            for agent, local_results in results:
                for thread, idx, msg, enf_log, error in local_results:
                    if msg is None:
                        if self.checkpoint is not None and run_id:
                            self.checkpoint.append_turn(
                                run_id=run_id, round_num=round_num, turn_idx=idx,
                                agent_id=agent.agent_id, status="failed", error=error,
                            )
                        continue

                    thread.add_message(msg)
                    msg_dict = {
                        "msg_id": msg.msg_id,
                        "thread_id": msg.thread_id,
                        "user_id": msg.user_id,
                        "action_type": msg.action_type.value,
                        "text": msg.text,
                        "timestamp": msg.timestamp.isoformat(),
                        "round": round_num,
                        "parent_msg_id": msg.parent_msg_id,
                        "metadata": dict(msg.metadata) if msg.metadata else {},
                    }
                    messages.append(msg_dict)
                    if self.checkpoint is not None and run_id:
                        self.checkpoint.append_turn(
                            run_id=run_id, round_num=round_num, turn_idx=idx,
                            agent_id=agent.agent_id, status="ok", message=msg_dict,
                        )
                    for other in agents:
                        if other.agent_id != agent.agent_id:
                            other.observe(msg, round_num)
                    if enf_log:
                        total_checks += enf_log.total_violations + 1
                        violations += enf_log.total_violations

        planned_turns = sum(len(agent_threads) for _, agent_threads, _ in agent_plan)
        return messages, violations, total_checks, planned_turns

    @staticmethod
    def _lists_to_tuples(value):
        if isinstance(value, list):
            return tuple(SimulationSandbox._lists_to_tuples(v) for v in value)
        return value

    @staticmethod
    def _message_from_dict(raw: dict, thread: Thread) -> Message:
        from datetime import datetime
        timestamp = raw.get("timestamp")
        try:
            parsed_ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        except (TypeError, ValueError):
            parsed_ts = datetime.now()
        return Message(
            msg_id=raw["msg_id"],
            thread_id=raw["thread_id"],
            user_id=raw["user_id"],
            platform=thread.platform,
            timestamp=parsed_ts,
            text=raw.get("text", ""),
            action_type=ActionType(raw["action_type"]),
            parent_msg_id=raw.get("parent_msg_id"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def _checkpoint_extra(
        self,
        per_round_metrics: list[dict],
        enforcement_violations: int,
        enforcement_total: int,
        **extra,
    ) -> dict:
        return {
            "checkpoint_schema_version": 2,
            "round_complete": True,
            "per_round_metrics": per_round_metrics,
            "enforcement_violations": enforcement_violations,
            "enforcement_total": enforcement_total,
            "rng_state": self._run_rng.getstate(),
            "run_fingerprint": self._run_fingerprint,
            **extra,
        }

    def _compute_round_metrics(
        self,
        round_messages: list[dict],
        agents: list[BaseAgent],
        round_num: int,
        planned_turns: int | None = None,
    ) -> dict:
        """Compute per-round metrics for trajectory analysis (outline §6.4).

        Tracks: polarization proxy, conflict ratio, unique participants,
        action diversity, message count.
        """
        from collections import Counter

        if not round_messages:
            return {
                "round": round_num,
                "message_count": 0,
                "planned_turns": int(planned_turns or 0),
                "turn_success_rate": 0.0,
            }

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
            "planned_turns": int(planned_turns or total),
            "turn_success_rate": total / max(int(planned_turns or total), 1),
            "conflict_count": conflict_count,
            "conflict_ratio": conflict_ratio,
            "polarization_proxy": conflict_ratio,  # used for trajectory plots
            "unique_participants": len(participants),
            "action_diversity": action_entropy,
            "interaction_density": graph_density,
        }
