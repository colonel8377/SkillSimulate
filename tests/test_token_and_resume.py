"""Regression tests for Issue 1 (token budget) and Issue 2 (resume + circuit breaker).

These tests exercise the new infrastructure directly without going through
the full experiment pipeline (which requires real data + real API keys).
They are deliberately fast — all run in <5s on CPU.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import tempfile
from pathlib import Path

import pytest

from src.agents.compaction import RollingSummaryCompactor
from src.agents.memory import AgentMemory
from src.data.schemas import ActionType, Message, Platform
from src.llm.circuit_breaker import CircuitBreaker, get_breaker, reset_global_breaker
from src.llm.client import LLMClient, ModelEndpoint
from src.llm.exceptions import (
    CircuitBreakerOpen,
    PromptBudgetExceeded,
    TransientResponseError,
)
from src.llm.token_counter import (
    estimate_messages_tokens,
    estimate_tokens,
    get_input_budget,
    truncate_to_token_budget,
)
from src.simulation.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Issue 1: Token accounting
# ---------------------------------------------------------------------------

class TestTokenCounter:
    def test_estimate_tokens_nonempty(self):
        n = estimate_tokens("hello world")
        assert n > 0
        # 1.15× multiplier means the estimate is strictly greater than
        # the raw BPE count (which is 2 for "hello world").
        assert n >= 2

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_messages_tokens_includes_framing(self):
        msgs = [{"role": "user", "content": "hello"}]
        # 4 framing tokens per message + 2 priming tokens + content tokens
        n = estimate_messages_tokens(msgs)
        assert n > 6

    def test_get_input_budget_arithmetic(self):
        # 65535 total - 32768 thinking - 8192 output = 24575 input
        assert get_input_budget(65535, 32768, 8192) == 24575

    def test_get_input_budget_clamps_at_zero(self):
        # Misconfigured: thinking + output > total → input budget 0
        # (surfaced as a config bug, not silently negative)
        assert get_input_budget(1000, 800, 800) == 0

    def test_truncate_to_token_budget_preserves_short_text(self):
        s = "short text"
        assert truncate_to_token_budget(s, 100) == s

    def test_truncate_to_token_budget_truncates_long_text(self):
        s = "word " * 1000
        result = truncate_to_token_budget(s, 50)
        assert result.endswith("…")
        # The truncated result should fit within the budget under our
        # conservative estimator.
        assert estimate_tokens(result.rstrip("…")) <= 50

    def test_truncate_to_token_budget_zero_budget_returns_unchanged(self):
        s = "anything"
        assert truncate_to_token_budget(s, 0) == s


class TestModelEndpointBudget:
    def test_max_input_tokens_derived(self):
        ep = ModelEndpoint(
            name="test", model="m",
            max_tokens=8192,
            max_total_tokens=65535,
            max_thinking_tokens=32768,
            max_output_tokens=8192,
        )
        assert ep.max_input_tokens == 65535 - 32768 - 8192

    def test_max_input_tokens_zero_when_no_total(self):
        ep = ModelEndpoint(name="test", model="m", max_tokens=2048)
        # Legacy mode — no shared cap, guard skipped
        assert ep.max_input_tokens == 0

    def test_post_init_syncs_max_tokens_and_output(self):
        ep = ModelEndpoint(name="t", model="m", max_tokens=4096)
        assert ep.max_output_tokens == 4096


# ---------------------------------------------------------------------------
# Issue 1: Memory
# ---------------------------------------------------------------------------

class TestAgentMemory:
    def _make_msg(self, text: str, round_num: int = 0) -> Message:
        return Message(
            msg_id=f"m{round_num}_{hash(text) & 0xFFFF:x}",
            thread_id="t1",
            user_id="u1",
            platform=Platform.WIKIPEDIA,
            timestamp=datetime.datetime.now(),
            text=text,
            action_type=ActionType.REPLY,
        )

    def test_token_budget_retrieval_accumulates_until_budget(self):
        mem = AgentMemory(max_context_items=100, max_context_tokens=200)
        for i in range(20):
            mem.add(self._make_msg(f"message content number {i}", i), round=i)
        retrieved = mem.retrieve(current_round=20)
        # Budget should cap the result well below the full 20 items
        assert len(retrieved) < 20
        # Total tokens of retrieved items should fit in the budget
        total = sum(estimate_tokens(m.text) + 8 for m in retrieved)
        assert total <= 200 + 50  # small slack for last-item boundary

    def test_item_count_path_still_works(self):
        mem = AgentMemory(max_context_items=5, max_context_tokens=0)
        for i in range(20):
            mem.add(self._make_msg(f"msg {i}", i), round=i)
        assert len(mem.retrieve(current_round=20)) == 5

    def test_pop_oldest_events_returns_oldest_n(self):
        mem = AgentMemory()
        for i in range(10):
            mem.add(self._make_msg(f"msg {i}", i), round=i, kind="event")
        popped = mem.pop_oldest_events(3)
        assert len(popped) == 3
        # Oldest three rounds
        assert {p.round for p in popped} == {0, 1, 2}
        assert len(mem) == 7

    def test_summary_items_get_boost(self):
        mem = AgentMemory(max_context_items=100, max_context_tokens=0)
        # 10 low-importance raw items + 1 summary item
        for i in range(10):
            mem.add(self._make_msg(f"raw {i}", i), round=i, importance=1.0, kind="event")
        summary_msg = self._make_msg("summary of older turns", 0)
        summary_msg.msg_id = "summary_1"
        mem.add(summary_msg, round=0, importance=2.0, kind="summary")
        retrieved = mem.retrieve(current_round=10)
        # Summary should appear in top results due to the 1.5× boost
        assert any(m.msg_id == "summary_1" for m in retrieved)

    def test_agent_runtime_state_roundtrip_is_lossless(self):
        from src.agents.vanilla import VanillaAgent
        from src.agents.reflection import ReflectionState

        source = VanillaAgent(agent_id="a", llm_client=object(), cluster_id="2")
        source.state.current_round = 7
        source.state.messages_sent = 3
        source.state.actions_taken = {"discuss": 3}
        source.engagement_ratio = 0.25
        source.memory.add(self._make_msg("remember this", 4), round=4, importance=2.0)
        source.reflection.state = ReflectionState("belief", ["position"], 5)

        raw = source.get_runtime_state()
        restored = VanillaAgent(agent_id="a", llm_client=object(), cluster_id="2")
        restored.restore_runtime_state(raw)

        assert restored.state.current_round == 7
        assert restored.state.messages_sent == 3
        assert restored.state.actions_taken == {"discuss": 3}
        assert restored.engagement_ratio == 0.25
        assert restored.memory.retrieve(current_round=7)[0].text == "remember this"
        assert restored.reflection.state.summary == "belief"


# ---------------------------------------------------------------------------
# Issue 2: Checkpoint — turn JSONL + FAILED marker + integrity check
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    def test_append_turn_and_load(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.append_turn("cellA", 0, 0, "a1", "ok", {"msg_id": "m1"})
        cm.append_turn("cellA", 0, 1, "a2", "failed", error="TimeoutError")
        cm.append_turn("cellA", 1, 0, "a1", "ok", {"msg_id": "m2"})
        turns = cm.load_turns("cellA")
        assert len(turns) == 3
        assert turns[1]["status"] == "failed"
        assert turns[1]["error"] == "TimeoutError"

    def test_get_last_successful_turn(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.append_turn("c", 0, 0, "a", "ok", {"msg_id": "m1"})
        cm.append_turn("c", 0, 1, "a", "ok", {"msg_id": "m2"})
        cm.append_turn("c", 1, 0, "a", "failed", error="x")
        cm.append_turn("c", 1, 1, "a", "ok", {"msg_id": "m3"})
        last = cm.get_last_successful_turn("c")
        assert last == (1, 1)

    def test_get_last_successful_turn_returns_none_when_empty(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert cm.get_last_successful_turn("c") is None

    def test_latest_round_uses_numeric_not_lexicographic_order(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.save("c", 9, [], [])
        cm.save("c", 49, [], [])
        assert cm.load("c")["round"] == 49

    def test_corrupt_latest_checkpoint_falls_back(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.save("c", 4, [], [])
        (tmp_path / "c_round_9.json").write_text("{partial")
        assert cm.load("c")["round"] == 4

    def test_partial_turns_after_full_round_are_discarded(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.append_turn("c", 4, 0, "a", "ok", {"msg_id": "m4"})
        cm.append_turn("c", 5, 0, "a", "ok", {"msg_id": "partial"})
        cm.truncate_turns_after_round("c", 4)
        turns = cm.load_turns("c")
        assert [t["round"] for t in turns] == [4]

    def test_load_turns_tolerates_corrupt_lines(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        path = cm._turns_path("c")
        # Write three lines, the middle one corrupt.
        lines = [
            json.dumps({"round": 0, "status": "ok", "message": {"x": 1}}),
            "CORRUPT_NOT_JSON",
            json.dumps({"round": 1, "status": "ok"}),
        ]
        with path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        turns = cm.load_turns("c")
        # Corrupt line dropped, others kept
        assert len(turns) == 2

    def test_is_completed_rejects_missing_result_file(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        cm.mark_completed("c", "/nonexistent/result.json")
        assert not cm.is_completed("c")

    def test_is_completed_accepts_valid_result_file(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps({
            "final_metrics": {},
            "simulation_integrity_passed": True,
            "simulation_message_count": 10,
            "simulation_round_count": 2,
            "simulation_expected_round_count": 2,
            "run_fingerprint": "fp",
        }))
        cm.mark_completed("c", str(result_path), run_fingerprint="fp")
        assert cm.is_completed("c")

    def test_is_completed_rejects_corrupt_result_file(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        result_path = tmp_path / "result.json"
        result_path.write_text("not valid json {{{")
        cm.mark_completed("c", str(result_path))
        assert not cm.is_completed("c")

    def test_failed_marker_roundtrip(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        assert not cm.is_failed("c")
        cm.mark_failed("c", reason="circuit_breaker_open", last_error="boom",
                       consecutive_failures=3, recoverable=True, resume_attempts=1)
        assert cm.is_failed("c")
        f = cm.get_failure("c")
        assert f["reason"] == "circuit_breaker_open"
        assert f["recoverable"] is True
        assert f["resume_attempts"] == 1
        cm.clear_failed("c")
        assert not cm.is_failed("c")


# ---------------------------------------------------------------------------
# Issue 2: Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_trip_at_threshold(self):
        async def run():
            b = CircuitBreaker(threshold=3)
            await b.record_failure("e1")
            await b.record_failure("e2")
            assert not b.tripped
            await b.record_failure("e3")
            assert b.tripped
            with pytest.raises(CircuitBreakerOpen):
                await b.check()
        asyncio.run(run())

    def test_record_success_resets_counter(self):
        async def run():
            b = CircuitBreaker(threshold=3)
            await b.record_failure("e1")
            await b.record_failure("e2")
            await b.record_success()
            assert b.consecutive_failures == 0
            # Two more failures should NOT trip (counter was reset)
            await b.record_failure("e3")
            await b.record_failure("e4")
            assert not b.tripped
        asyncio.run(run())

    def test_reset_clears_state(self):
        b = CircuitBreaker(threshold=2)
        async def run():
            await b.record_failure("e1")
            await b.record_failure("e2")
            assert b.tripped
        asyncio.run(run())
        b.reset()
        assert not b.tripped
        assert b.consecutive_failures == 0

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            CircuitBreaker(threshold=0)


# ---------------------------------------------------------------------------
# Issue 1: Pre-flight token guard fires through LLMClient
# ---------------------------------------------------------------------------

class TestPromptBudgetGuard:
    def test_guard_raises_when_prompt_exceeds_budget(self, monkeypatch):
        # Force non-mock mode so the guard runs (mock mode skips it)
        monkeypatch.setenv("CADP_MOCK_LLM", "0")
        client = LLMClient()
        # Find an endpoint with max_total_tokens configured
        ep_name = next(
            (n for n, ep in client.models.items() if ep.max_total_tokens > 0),
            None,
        )
        if ep_name is None:
            pytest.skip("no endpoint in configs/models.yaml has max_total_tokens set")
        ep = client.models[ep_name]
        # Build a prompt large enough to exceed max_input_tokens
        big = "x" * (ep.max_input_tokens * 8)  # ~8 chars per token, 8× overkill
        with pytest.raises(PromptBudgetExceeded) as exc_info:
            asyncio.run(client.chat_completion(
                [{"role": "user", "content": big}],
                ep_name,
            ))
        assert exc_info.value.model == ep.model
        assert exc_info.value.requested > ep.max_input_tokens


# ---------------------------------------------------------------------------
# Issue 1: Rolling summary compactor (R4 path)
# ---------------------------------------------------------------------------

class TestRollingSummaryCompactor:
    def _make_msg(self, text: str, round_num: int = 0) -> Message:
        return Message(
            msg_id=f"m{round_num}_{hash(text) & 0xFFFF:x}",
            thread_id="t1",
            user_id="u1",
            platform=Platform.WIKIPEDIA,
            timestamp=datetime.datetime.now(),
            text=text,
            action_type=ActionType.REPLY,
        )

    def test_should_compact_on_interval_boundary(self):
        cmp = RollingSummaryCompactor.__new__(RollingSummaryCompactor)
        cmp.compaction_interval = 5
        assert cmp.should_compact(0) is False  # boundary but round 0
        assert cmp.should_compact(5) is True
        assert cmp.should_compact(10) is True
        assert cmp.should_compact(7) is False

    def test_should_compact_disabled_when_interval_zero(self):
        cmp = RollingSummaryCompactor.__new__(RollingSummaryCompactor)
        cmp.compaction_interval = 0
        assert cmp.should_compact(5) is False

    def test_compact_in_mock_mode_succeeds(self, monkeypatch):
        monkeypatch.setenv("CADP_MOCK_LLM", "1")
        client = LLMClient()

        async def run():
            mem = AgentMemory()
            # Add 30 raw event items
            for i in range(30):
                mem.add(self._make_msg(f"message number {i}", i), round=i, kind="event")
            assert len(mem) == 30
            cmp = RollingSummaryCompactor(
                client, "deepseek-v4-flash",
                compaction_interval=5, keep_recent=10,
            )
            # Default max_messages_per_summary = compaction_interval = 5,
            # so one compaction pass summarizes 5 items (not all 20
            # available). Multiple passes are required to compact more.
            result = await cmp.compact(mem, current_round=30)
            assert result.success
            assert result.summarized_count == 5
            # 30 raw - 5 summarized + 1 summary = 26 items
            assert len(mem) == 26
            summaries = mem.get_items_of_kind("summary")
            assert len(summaries) == 1
        asyncio.run(run())

    def test_compact_skips_when_insufficient_history(self, monkeypatch):
        monkeypatch.setenv("CADP_MOCK_LLM", "1")
        client = LLMClient()

        async def run():
            mem = AgentMemory()
            for i in range(5):  # less than keep_recent=10
                mem.add(self._make_msg(f"msg {i}", i), round=i, kind="event")
            cmp = RollingSummaryCompactor(
                client, "deepseek-v4-flash",
                compaction_interval=5, keep_recent=10,
            )
            result = await cmp.compact(mem, current_round=5)
            assert result.success
            assert result.summarized_count == 0
            assert "not-enough-history" in result.error
            assert len(mem) == 5  # unchanged
        asyncio.run(run())
