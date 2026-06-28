"""Abstract agent base class — perceive → plan → act → reflect cycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from src.agents.memory import AgentMemory
from src.agents.planning import ActionPlan, Planner
from src.agents.reflection import ReflectionModule
from src.data.schemas import ActionType, Message, Thread
from src.enforcement.harness import EnforcementHarness, EnforcementLog
from src.llm.client import LLMClient
from src.llm.token_counter import truncate_to_token_budget

@dataclass
class AgentState:
    """Runtime state of an agent."""
    agent_id: str
    cluster_id: str = ""
    current_round: int = 0
    messages_sent: int = 0
    actions_taken: dict[str, int] = field(default_factory=dict)
    enforcement_logs: list[dict] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base for all agent types."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        model_name: str = "gpt-4o",
        cluster_id: str = "",
        max_context_items: int = 20,
        reflection_interval: int = 10,
        max_reformulation_retries: int = 1,
        max_memory_tokens: int = 0,
        memory_strategy: str = "sliding",
        compaction_interval: int = 5,
        compaction_keep_recent: int = 10,
        max_display_items: int = 5,
        per_msg_token_ratio: int = 10,
        per_msg_token_floor: int = 60,
        max_thread_messages: int = 5,
    ):
        self.agent_id = agent_id
        self.llm = llm_client
        self.model_name = model_name
        # When max_memory_tokens > 0, AgentMemory.retrieve() accumulates
        # ranked items until the token budget is hit (Issue 1). When 0,
        # legacy item-count behaviour applies (back-compat for unit tests
        # and for endpoints with no total-token cap configured).
        self.max_memory_tokens = max_memory_tokens
        self.memory_strategy = memory_strategy
        self.max_display_items = max_display_items
        self.per_msg_token_ratio = per_msg_token_ratio
        self.per_msg_token_floor = per_msg_token_floor
        self.max_thread_messages = max_thread_messages
        self.memory = AgentMemory(
            max_context_items=max_context_items,
            max_context_tokens=max_memory_tokens,
        )
        self.planner = Planner(llm_client, model_name, max_memory_tokens=max_memory_tokens,
                               per_msg_token_ratio=per_msg_token_ratio,
                               per_msg_token_floor=per_msg_token_floor,
                               max_thread_messages=max_thread_messages)
        self.reflection = ReflectionModule(llm_client, model_name, reflection_interval, max_memory_tokens=max_memory_tokens)
        self.state = AgentState(agent_id=agent_id, cluster_id=cluster_id)
        self.enforcement_harness: EnforcementHarness | None = None
        self.max_reformulation_retries = max_reformulation_retries
        self.engagement_ratio: float = 0.5
        # Rolling-summary compactor for R4 path (Issue 1). Lazily created
        # only when memory_strategy == "rolling_summary"; stays None for
        # the default sliding-window path so there is zero overhead on
        # exp1/exp2 cells.
        self._compactor = None
        if memory_strategy == "rolling_summary":
            from src.agents.compaction import RollingSummaryCompactor
            self._compactor = RollingSummaryCompactor(
                llm_client=llm_client,
                model_name=model_name,
                compaction_interval=compaction_interval,
                keep_recent=compaction_keep_recent,
            )
        # Injected by SimulationSandbox at run time (outline §6.3 platform
        # topology fidelity). When None, take_turn falls back to the legacy
        # "reply to last message" behaviour.
        self.platform_topology: "PlatformTopology | None" = None

    @abstractmethod
    def get_role_description(self) -> str:
        """Return this agent's role/persona description for the LLM."""

    @abstractmethod
    def get_constraints_text(self) -> str:
        """Return constraint instructions for the planner."""

    def get_reflection_directive(self) -> str | None:
        """Optional outline §4.6 reinforcement directive for periodic reflection.

        Base behaviour returns ``None`` — plain belief consolidation. CADP
        overrides this to drive reflection *through* its verified Mind Models
        (a CADP-specific long-horizon advantage). Returning ``None`` here keeps
        every non-CADP condition and every Mind-Models-disabled ablation
        unchanged.
        """
        return None

    async def take_turn(
        self,
        thread: Thread,
        available_actions: list[ActionType],
        current_round: int,
    ) -> tuple[Message, EnforcementLog | None]:
        """Execute one turn: perceive → plan → enforce → act.

        Args:
            thread: Current discussion thread.
            available_actions: Platform-allowed actions.
            current_round: Current simulation round.

        Returns:
            Tuple of (Message to send, enforcement log).
        """
        self.state.current_round = current_round

        # Perceive: retrieve relevant memory
        memory_msgs = self.memory.retrieve(thread.thread_id, current_round)
        # Per-message truncation: when max_memory_tokens > 0 we cap each
        # message to max_memory_tokens // per_msg_token_ratio so a single
        # verbose turn cannot crowd out other retrieved items.
        # When 0 we fall back to the legacy [:200] char cap (back-compat).
        per_msg_budget = max(self.per_msg_token_floor, self.max_memory_tokens // self.per_msg_token_ratio) if self.max_memory_tokens else 0
        memory_context = "\n".join(
            f"[{m.user_id}] ({m.action_type.value}): "
            f"{truncate_to_token_budget(m.text, per_msg_budget) if per_msg_budget else m.text[:200]}"
            for m in memory_msgs[:self.max_display_items]
        )

        # Build reflection context (consolidated beliefs from prior rounds)
        reflection_context = self._build_reflection_context()

        # Enforce (if harness is configured)
        enforcement_log: EnforcementLog | None = None
        final_text = ""
        # Tracks whether the Tier-3 safe-template fallback (§4.4.2 step 4)
        # was used this turn, so the emitted Message can be flagged
        # "constraint-forced" for evaluation counting.
        constraint_forced = False

        if self.enforcement_harness is not None:
            # Step 1: Pre-generation enforcement BEFORE planning
            # Build initial messages for enforcement context extraction
            gen_messages = [
                {"role": "system", "content": self.get_role_description()},
                {"role": "user", "content": f"Topic: {thread.topic}\nMemory: {truncate_to_token_budget(memory_context, self.max_memory_tokens) if self.max_memory_tokens else memory_context[:500]}"},
            ]

            gen_messages, pre_log, enforcement_context = await self.enforcement_harness.enforce_generation(
                gen_messages, draft_action=""
            )

            # Step 2: Merge enforcement context into planner constraints
            merged_constraints = self.get_constraints_text()
            if enforcement_context:
                if merged_constraints:
                    merged_constraints += "\n\n" + enforcement_context
                else:
                    merged_constraints = enforcement_context

            # Step 3: Plan with enforcement-aware constraints
            plan = await self.planner.plan(
                thread=thread,
                available_actions=available_actions,
                memory_context=memory_context,
                role_description=self.get_role_description(),
                constraints=merged_constraints,
                reflection_context=reflection_context,
            )
            final_text = plan.text
            if not final_text.strip():
                # Empty text is legitimate for non-text actions — the planner
                # prompt explicitly allows "empty for non-text actions"
                # (REVERT, REPORT, AWARD_DELTA, LABEL, CLOSE, ...). Proceed
                # with empty text so the action is recorded rather than
                # dropping the turn. (Previously this raised, silently
                # dropping non-text-action turns and under-counting them in
                # the action distribution.)
                logger.debug(
                    f"Agent {self.agent_id}: empty planner text for "
                    f"{plan.action_type.value} (allowed for non-text actions)"
                )
                final_text = ""

            # Step 4: Post-generation enforcement
            # Pass a per-turn safe template so Tier 1 has a §4.4.2-compliant
            # fallback if Expression-DNA regeneration exhausts N_retry=3 (G6).
            # We construct it lazily from the (soon-to-be-finalised) action
            # type — at this point plan.action_type is the planner's choice.
            tier1_safe_template = self._safe_template_response(thread, plan.action_type)
            final_text, post_log, replan_feedback = await self.enforcement_harness.enforce_output(
                final_text, gen_messages, safe_template=tier1_safe_template
            )

            # Step 5: Forced Reformulation Protocol (outline §4.4.2)
            # Re-plan up to max_reformulation_retries times until Tier-3 passes.
            # Each replan is a NEW post-generation event, so the composite
            # post-gen enforcement (Tier 3 + Tier 1) re-runs on it — not Tier 3
            # alone. Otherwise a replan that satisfies Tier 3 but drifts
            # stylistically outside the agent's Expression DNA 2σ boundary
            # would be silently accepted.
            if replan_feedback is not None:
                violations_persisted = True
                logger.debug(
                    f"Agent {self.agent_id}: Tier-3 post-gen violation → reformulation "
                    f"(max {self.max_reformulation_retries} attempts). initial reason: {replan_feedback}"
                )

                for attempt in range(self.max_reformulation_retries):
                    replan = await self.planner.replan(
                        thread=thread,
                        available_actions=available_actions,
                        memory_context=memory_context,
                        role_description=self.get_role_description(),
                        constraints=merged_constraints,
                        prev_text=final_text,
                        violation_feedback=replan_feedback,
                        prev_action_type=plan.action_type,
                    )
                    final_text = replan.text or final_text
                    plan = replan
                    post_log.regenerated = True

                    # Re-run BOTH post-gen tiers on the replanned text.
                    # enforce_output returns new_feedback = None iff Tier 3
                    # passes; Tier 1 may still regenerate style on the
                    # Tier-3-clean text, which is its designed behaviour.
                    rechecked_text, loop_log, new_feedback = await self.enforcement_harness.enforce_output(
                        final_text, gen_messages,
                        safe_template=self._safe_template_response(thread, plan.action_type),
                    )
                    if loop_log.total_violations > 0:
                        post_log.total_violations += loop_log.total_violations
                    # M4 fix: propagate inner-loop Tier-1 safe-template /
                    # regeneration signals into the outer ``post_log`` so
                    # §5.6.7 (R4 persona-collapse H3) can distinguish
                    # "natural pass" vs "Tier-1 inner-loop safe-template
                    # fallback" vs "outer Forced-Reformulation fallback".
                    # Without this accumulation, inner Tier-1 fallback
                    # events fired during the reformulation loop were
                    # silently dropped from metrics.
                    if loop_log.regenerated:
                        post_log.regenerated = True
                    if loop_log.safe_template_fallback_used:
                        post_log.safe_template_fallback_used = True
                        constraint_forced = True
                    final_text = rechecked_text

                    if new_feedback is None:
                        logger.debug(
                            f"Agent {self.agent_id}: reformulation attempt {attempt+1} "
                            f"CLEARED the violation"
                        )
                        violations_persisted = False
                        break
                    # Tier 3 still failed — feed its latest reason into the next replan.
                    logger.debug(
                        f"Agent {self.agent_id}: reformulation attempt {attempt+1} "
                        f"still violating: {new_feedback}"
                    )
                    replan_feedback = new_feedback

                if violations_persisted:
                    # Fallback (outline §4.4.2 step 4): safe-template response.
                    # Log the persistently-violating text + reason BEFORE the
                    # safe-template overwrites it — diagnostic for §5.3.5 trigger
                    # calibration (which anti-patterns / trigger types dominate).
                    logger.warning(
                        f"Agent {self.agent_id}: Tier-3 violation persisted after "
                        f"{self.max_reformulation_retries} reformulation attempts → safe-template fallback\n"
                        f"  violating text: {final_text[:160]!r}\n"
                        f"  reason: {replan_feedback}"
                    )
                    final_text = self._safe_template_response(thread, plan.action_type)
                    post_log.safe_template_fallback_used = True
                    constraint_forced = True

            enforcement_log = EnforcementLog(
                tier3_pre=pre_log.tier3_pre,
                tier2_pre=pre_log.tier2_pre,
                tier3_post=post_log.tier3_post,
                tier1_post=post_log.tier1_post,
                total_violations=pre_log.total_violations + post_log.total_violations,
                regenerated=post_log.regenerated,
                tier3_hard_block_triggered=post_log.tier3_hard_block_triggered,
                safe_template_fallback_used=post_log.safe_template_fallback_used,
            )
        else:
            # No enforcement harness — plain planning
            plan = await self.planner.plan(
                thread=thread,
                available_actions=available_actions,
                memory_context=memory_context,
                role_description=self.get_role_description(),
                constraints=self.get_constraints_text(),
                reflection_context=reflection_context,
            )
            final_text = plan.text
            if not final_text.strip():
                # Empty text is legitimate for non-text actions — the planner
                # prompt explicitly allows "empty for non-text actions"
                # (REVERT, REPORT, AWARD_DELTA, LABEL, CLOSE, ...). Proceed
                # with empty text so the action is recorded rather than
                # dropping the turn. (Previously this raised, silently
                # dropping non-text-action turns and under-counting them in
                # the action distribution.)
                logger.debug(
                    f"Agent {self.agent_id}: empty planner text for "
                    f"{plan.action_type.value} (allowed for non-text actions)"
                )
                final_text = ""

        # Construct message — pick parent via platform topology (outline §6.3)
        # Falls back to legacy "reply to last message" when no topology is
        # attached (e.g. unit tests, R4 collapse stress test).
        if self.platform_topology is not None:
            parent_msg_id = self.platform_topology.select_reply_target(
                plan.action_type,
                self.agent_id,
                thread,
                hint_target_msg_id=plan.target_msg_id,
            )
        else:
            parent_msg_id = thread.messages[-1].msg_id if thread.messages else None

        msg = Message(
            msg_id=f"{self.agent_id}_r{current_round}_{self.state.messages_sent}",
            thread_id=thread.thread_id,
            user_id=self.agent_id,
            platform=thread.platform,
            timestamp=datetime.now(),
            text=final_text,
            action_type=plan.action_type,
            parent_msg_id=parent_msg_id,
            metadata={
                "round": current_round,
                "reasoning": plan.reasoning,
                # outline §4.4.2 step 4: mark safe-template outputs so they
                # are counted separately in evaluation.
                "constraint_forced": constraint_forced,
            },
        )

        # Update state
        self.state.messages_sent += 1
        action_key = plan.action_type.value
        self.state.actions_taken[action_key] = self.state.actions_taken.get(action_key, 0) + 1
        if enforcement_log:
            self.state.enforcement_logs.append(enforcement_log.to_dict())

        # Store in memory
        self.memory.add(msg, current_round, importance=self._compute_importance(plan))

        # Periodic reflection
        if self.reflection.should_reflect(current_round):
            recent = self.memory.retrieve(thread_id=None, current_round=current_round)
            await self.reflection.reflect(
                recent,
                context=self.get_role_description(),
                current_round=current_round,
                reinforcement_directive=self.get_reflection_directive(),
            )

        # Rolling-summary compaction (R4 path only — Issue 1). Runs every
        # compaction_interval turns, summarizing the oldest raw memory
        # items into a single ``kind="summary"`` MemoryItem. This keeps
        # long-horizon signal available to the planner for the 50-turn
        # persona-collapse stress test. Failures are non-fatal: the
        # compactor logs a WARNING and leaves raw memory untouched.
        if self._compactor is not None and self._compactor.should_compact(current_round):
            await self._compactor.compact(self.memory, current_round)

        return msg, enforcement_log

    def observe(self, msg: Message, round: int) -> None:
        """Observe a message from another agent."""
        self.memory.add(msg, round)

    def _compute_importance(self, plan: ActionPlan) -> float:
        """Compute importance of the agent's own action."""
        if plan.action_type in {ActionType.DISAGREE, ActionType.REVERT, ActionType.COUNTER_ARGUE}:
            return 2.0
        if plan.action_type == ActionType.AWARD_DELTA:
            return 3.0
        return 1.0

    def _build_reflection_context(self) -> str:
        """Build reflection context string for the planner.

        Returns consolidated beliefs and positions from periodic reflection.
        Empty string if no reflection has occurred yet.
        """
        state = self.reflection.state
        if not state.summary:
            return ""
        parts = ["Your consolidated beliefs from prior interactions:"]
        parts.append(state.summary)
        if state.key_positions:
            parts.append("Your key positions:")
            for pos in state.key_positions:
                parts.append(f"  - {pos}")
        return "\n".join(parts)

    def _safe_template_response(self, thread: Thread, action_type: ActionType) -> str:
        """Generate a neutral but engaged safe-template response.

        Used when Tier-3 anti-pattern violations persist after re-planning.
        The response avoids personal attacks and profanity while still
        expressing a position, so disagreement language is preserved for
        downstream evaluation.
        """
        return (
            f"I see the point about {thread.topic}, but I disagree with that "
            "conclusion. Could you clarify the reasoning behind it?"
        )

    def get_state_summary(self) -> dict:
        """Return summary of agent state for logging."""
        return {
            "agent_id": self.agent_id,
            "cluster_id": self.state.cluster_id,
            "messages_sent": self.state.messages_sent,
            "actions_taken": dict(self.state.actions_taken),
            "total_enforcement_violations": sum(
                log.get("total_violations", 0) for log in self.state.enforcement_logs
            ),
        }
