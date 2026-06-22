"""Constraint-aware action planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.schemas import ActionType, Message, Thread


@dataclass
class ActionPlan:
    """Planned action for the current turn."""
    action_type: ActionType
    text: str
    target_user_id: str | None = None
    target_msg_id: str | None = None
    reasoning: str = ""


PLANNING_PROMPT = """You are participating in an online discussion.

Platform: {platform}
Topic: {topic}
Your role: {role_description}

Available actions: {available_actions}

Current thread state:
{thread_state}

Recent messages you've seen:
{recent_messages}

{reflection_context}

{constraints}

Decide your next action. If your action is a reply, counter-argument, or
directed response, identify the specific message you are responding to by
copying its ``msg_id`` into ``target_msg_id``. Leave ``target_msg_id`` empty
for top-level / non-reply actions.

Respond as JSON:
{{
  "action_type": "one of the available actions",
  "text": "your message text (empty for non-text actions)",
  "target_msg_id": "msg_id of the message you are replying to, or empty",
  "reasoning": "brief explanation of why"
}}

Output ONLY the JSON."""


class Planner:
    """Plans actions considering constraints and platform rules."""

    def __init__(self, llm_client, model_name: str = "gpt-4o"):
        self.llm = llm_client
        self.model_name = model_name

    async def plan(
        self,
        thread: Thread,
        available_actions: list[ActionType],
        memory_context: str,
        role_description: str = "",
        constraints: str = "",
        reflection_context: str = "",
    ) -> ActionPlan:
        """Decide next action.

        Args:
            thread: Current thread context.
            available_actions: Platform-allowed actions.
            memory_context: Retrieved memory as formatted text.
            role_description: Agent's persona/role.
            constraints: Additional constraints text.
            reflection_context: Consolidated beliefs from periodic reflection.

        Returns:
            ActionPlan for this turn.
        """
        actions_str = ", ".join(a.value for a in available_actions)

        # Format recent thread messages — include msg_id so the LLM can
        # emit target_msg_id for directed replies (outline §6.3 topology)
        recent = thread.messages[-5:]
        thread_text = "\n".join(
            f"<msg id={m.msg_id}> [{m.user_id}] ({m.action_type.value}): {m.text[:200]}"
            for m in recent
        )

        prompt = PLANNING_PROMPT.format(
            platform=thread.platform.value,
            topic=thread.topic,
            role_description=role_description or "A community participant",
            available_actions=actions_str,
            thread_state=thread_text,
            recent_messages=memory_context[:1000],
            reflection_context=reflection_context or "",
            constraints=constraints or "",
        )

        messages = [
            {"role": "system", "content": "You are a social simulation agent planning your next action."},
            {"role": "user", "content": prompt},
        ]

        # Retry up to 2 times if JSON parsing fails (HKUST proxy without
        # response_format support occasionally returns non-JSON text).
        data: dict = {}
        for attempt in range(3):
            response = await self.llm.chat_completion_json(
                messages, self.model_name, temperature=0.7, default=None
            )
            if isinstance(response, dict) and response:
                data = response
                break
            # JSON parsing failed — retry with stronger instruction
            if attempt < 2:
                messages = messages + [
                    {"role": "assistant", "content": str(response) if response else ""},
                    {"role": "user", "content": "Your response was not valid JSON. Please respond with ONLY a valid JSON object containing: action_type, text, target_msg_id, reasoning."},
                ]

        # Fallback: if all retries failed, use defaults
        if not data:
            data = {}

        action_str = data.get("action_type", available_actions[0].value)
        action_type = next(
            (a for a in available_actions if a.value == action_str),
            available_actions[0],
        )

        # Validate target_msg_id — must reference an existing message
        target_msg_id = data.get("target_msg_id") or None
        if target_msg_id:
            valid_ids = {m.msg_id for m in thread.messages}
            if target_msg_id not in valid_ids:
                target_msg_id = None  # drop invalid hints; topology re-routes

        return ActionPlan(
            action_type=action_type,
            text=data.get("text", ""),
            target_msg_id=target_msg_id,
            reasoning=data.get("reasoning", ""),
        )

    REPLAN_PROMPT = """You are re-planning your response because your previous output violated a behavioral constraint.

Platform: {platform}
Topic: {topic}
Your role: {role_description}

Your previous response was:
"{prev_text}"

VIOLATION FEEDBACK:
{violation_feedback}

{constraints}

Rewrite your response to AVOID the violations above while staying on topic.
Respond as JSON:
{{
  "action_type": "{action_type}",
  "text": "your revised message text",
  "reasoning": "brief explanation of what you changed"
}}

Output ONLY the JSON."""

    async def replan(
        self,
        thread: Thread,
        available_actions: list[ActionType],
        memory_context: str,
        role_description: str = "",
        constraints: str = "",
        prev_text: str = "",
        violation_feedback: str = "",
        prev_action_type: ActionType | None = None,
    ) -> ActionPlan:
        """Re-plan after a Tier-3 post-gen violation.

        Args:
            thread: Current thread context.
            available_actions: Platform-allowed actions.
            memory_context: Retrieved memory as formatted text.
            role_description: Agent's persona/role.
            constraints: Additional constraints text.
            prev_text: The previous (violating) text.
            violation_feedback: Description of violations detected.
            prev_action_type: The action type from the previous plan.

        Returns:
            Revised ActionPlan.
        """
        actions_str = ", ".join(a.value for a in available_actions)

        recent = thread.messages[-5:]
        thread_text = "\n".join(
            f"<msg id={m.msg_id}> [{m.user_id}] ({m.action_type.value}): {m.text[:200]}"
            for m in recent
        )

        prompt = self.REPLAN_PROMPT.format(
            platform=thread.platform.value,
            topic=thread.topic,
            role_description=role_description or "A community participant",
            prev_text=prev_text[:500],
            violation_feedback=violation_feedback,
            constraints=constraints or "",
            action_type=(prev_action_type or available_actions[0]).value,
        )

        messages = [
            {"role": "system", "content": "You are a social simulation agent revising your response to comply with behavioral constraints."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat_completion_json(
            messages, self.model_name, temperature=0.5, default={}
        )

        data = response if isinstance(response, dict) else {}

        action_str = data.get("action_type", (prev_action_type or available_actions[0]).value)
        action_type = next(
            (a for a in available_actions if a.value == action_str),
            prev_action_type or available_actions[0],
        )

        # Validate target_msg_id (same logic as plan())
        target_msg_id = data.get("target_msg_id") or None
        if target_msg_id:
            valid_ids = {m.msg_id for m in thread.messages}
            if target_msg_id not in valid_ids:
                target_msg_id = None

        return ActionPlan(
            action_type=action_type,
            text=data.get("text", ""),
            target_msg_id=target_msg_id,
            reasoning=data.get("reasoning", ""),
        )
