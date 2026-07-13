"""Tests for core modules."""

import numpy as np
import pytest
from datetime import datetime

from src.data.schemas import ActionType, Message, Platform, Thread


def test_action_type_for_platform():
    wiki_actions = ActionType.for_platform(Platform.WIKIPEDIA)
    assert ActionType.EDIT in wiki_actions
    assert ActionType.REVERT in wiki_actions

    reddit_actions = ActionType.for_platform(Platform.REDDIT)
    assert ActionType.REPLY in reddit_actions
    assert ActionType.AWARD_DELTA in reddit_actions

    github_actions = ActionType.for_platform(Platform.GITHUB)
    assert ActionType.COMMENT in github_actions
    assert ActionType.CLOSE in github_actions


def test_thread_operations():
    thread = Thread(
        thread_id="test_1",
        platform=Platform.WIKIPEDIA,
        topic="Test topic",
    )
    msg = Message(
        msg_id="m1",
        thread_id="test_1",
        user_id="user1",
        platform=Platform.WIKIPEDIA,
        timestamp=datetime.now(),
        text="Hello",
        action_type=ActionType.DISCUSS,
    )
    thread.add_message(msg)
    assert len(thread.messages) == 1
    assert "user1" in thread.participants


def test_user_features():
    from src.clustering.features import FeatureExtractor, UserFeatures

    thread = Thread(
        thread_id="t1",
        platform=Platform.REDDIT,
        topic="Test",
    )
    for i in range(5):
        msg = Message(
            msg_id=f"m{i}",
            thread_id="t1",
            user_id="user1",
            platform=Platform.REDDIT,
            timestamp=datetime.now(),
            text=f"Message {i}",
            action_type=ActionType.REPLY if i % 2 == 0 else ActionType.COUNTER_ARGUE,
        )
        thread.add_message(msg)

    extractor = FeatureExtractor()
    features = extractor.extract_all([thread])
    assert "user1" in features
    assert features["user1"].message_count == 5
    vec = features["user1"].to_vector()
    assert len(vec) == 13


def test_expression_dna():
    from src.skill.expression_dna import ExpressionDNAExtractor
    from src.skill.schema import ExpressionDNA

    messages = [
        Message(
            msg_id=f"m{i}",
            thread_id="t1",
            user_id="u1",
            platform=Platform.WIKIPEDIA,
            timestamp=datetime.now(),
            text=f"This is clearly the right approach. However, we must consider alternatives. I think this matters.",
            action_type=ActionType.DISCUSS,
        )
        for i in range(10)
    ]

    extractor = ExpressionDNAExtractor()
    edna = extractor.extract(messages)
    assert isinstance(edna, ExpressionDNA)
    assert edna.avg_sentence_length > 0
    assert 0 <= edna.certainty_ratio <= 1


def test_skill_file_serialization():
    from src.skill.schema import (
        AntiPattern, CapabilityTrack, ConstraintTrack,
        ExpressionDNA, MindModel, SkillFile,
    )

    skill = SkillFile(
        cluster_id="0",
        platform="wikipedia",
        capability=CapabilityTrack(
            expression_dna=ExpressionDNA(avg_sentence_length=15.0),
            mind_models=[
                MindModel(
                    name="Test Model",
                    description="A test model",
                    evidence=["evidence1"],
                    application="When testing",
                    limitation="Only in tests",
                ),
            ],
        ),
        constraint=ConstraintTrack(
            anti_patterns=[
                AntiPattern(
                    description="Avoid personal attacks",
                    trigger_conditions=["When disagreeing"],
                    trigger_keywords=["stupid", "idiot"],
                ),
            ],
        ),
    )

    yaml_str = skill.to_yaml()
    assert "cluster_id: '0'" in yaml_str
    assert "Test Model" in yaml_str

    restored = SkillFile.from_yaml(yaml_str)
    assert restored.cluster_id == "0"
    assert len(restored.capability.mind_models) == 1
    assert restored.capability.mind_models[0].name == "Test Model"


def test_cost_tracker():
    from src.llm.cost_tracker import CostTracker

    tracker = CostTracker()
    tracker.record("gpt-4o", 1000, 500, 2.5, 10.0)
    tracker.record("gpt-4o", 2000, 1000, 2.5, 10.0)

    summary = tracker.summary()
    assert summary["gpt-4o"]["input_tokens"] == 3000
    assert summary["gpt-4o"]["output_tokens"] == 1500
    assert summary["gpt-4o"]["cost_usd"] == round(3000/1000*2.5 + 1500/1000*10.0, 4)


def test_agent_memory():
    from src.agents.memory import AgentMemory

    memory = AgentMemory(max_context_items=3)
    for i in range(5):
        msg = Message(
            msg_id=f"m{i}",
            thread_id="t1",
            user_id=f"user{i}",
            platform=Platform.WIKIPEDIA,
            timestamp=datetime.now(),
            text=f"Message {i}",
        )
        memory.add(msg, round=i)

    retrieved = memory.retrieve(thread_id="t1")
    assert len(retrieved) <= 3


# ---------------------------------------------------------------------------
# G1–G15 regression tests (added after outline.md audit)
# ---------------------------------------------------------------------------

def test_g12_pii_ipv6_and_bare_handles():
    """G12: PII scrubber now catches IPv6 + bare-handle usernames."""
    from src.data.pii import strip_pii_from_text

    # IPv6 (full + compressed forms)
    assert "[IP]" in strip_pii_from_text("Logged in from 2001:db8::1 just now")
    assert "[IP]" in strip_pii_from_text("Edit from fe80::1ff:fe23:4567:890a")

    # Bare handle in prose (capitalized word + trailing digits, no sigil)
    scrubbed = strip_pii_from_text("As Colonel83 said earlier, this is wrong.")
    assert "Colonel83" not in scrubbed
    assert "[MENTION]" in scrubbed

    # Legacy paths still work
    assert "[IP]" in strip_pii_from_text("from 192.168.1.1")
    assert "[MENTION]" in strip_pii_from_text("ping @alice please")


def test_g10_apply_action_delegates_to_topology():
    """G10: apply_action must use select_reply_target, not last-message flattening."""
    from datetime import datetime
    from src.simulation.platforms.wikipedia import WikipediaTopology

    topo = WikipediaTopology()
    # Build a thread with a DISCUSS message, then apply REVERT
    thread = Thread(thread_id="t1", platform=Platform.WIKIPEDIA, topic="x")
    edit_msg = Message(
        msg_id="edit1", thread_id="t1", user_id="other",
        platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
        text="edit", action_type=ActionType.EDIT,
    )
    thread.add_message(edit_msg)
    out = topo.apply_action(ActionType.REVERT, "reverting", agent_id="me", thread=thread)
    # REVERT must target the EDIT, not just last_msg (which is also the EDIT here,
    # but more importantly EDIT applied to a different agent must give parent=None)
    assert out.parent_msg_id == "edit1"

    # EDIT must produce parent_msg_id=None (article, not a message)
    thread2 = Thread(thread_id="t2", platform=Platform.WIKIPEDIA, topic="x")
    thread2.add_message(Message(
        msg_id="d1", thread_id="t2", user_id="u",
        platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
        text="hi", action_type=ActionType.DISCUSS,
    ))
    edit_out = topo.apply_action(ActionType.EDIT, "new edit", agent_id="me", thread=thread2)
    assert edit_out.parent_msg_id is None


def test_g11_github_comment_fallback_no_lifecycle_edge():
    """G11: GitHub COMMENT with no prior COMMENT targets issue root (None),
    not a CLOSE/LABEL event (which would create a spurious edge)."""
    from datetime import datetime
    from src.simulation.platforms.github import GitHubTopology

    topo = GitHubTopology()
    thread = Thread(thread_id="i1", platform=Platform.GITHUB, topic="bug")
    thread.add_message(Message(
        msg_id="close1", thread_id="i1", user_id="maintainer",
        platform=Platform.GITHUB, timestamp=datetime.now(),
        text="closing", action_type=ActionType.CLOSE,
    ))
    out = topo.apply_action(ActionType.COMMENT, "first comment", agent_id="newcomer", thread=thread)
    assert out.parent_msg_id is None  # issue root, not close1


def test_g6_tier1_safe_template_fallback():
    """G6: Tier 1 uses safe-template fallback after max_retries=3 exhausted."""
    import asyncio
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    from src.skill.schema import ExpressionDNA

    edna = ExpressionDNA(
        embedding_centroid=[0.0] * 384,
        embedding_std=[1.0] * 384,
    )

    # Stub embedder so every text is "out of bounds" deterministically
    class StubEmbedder:
        def encode(self, texts, show_progress_bar=False):
            import numpy as np
            single = isinstance(texts, str)
            arr = np.full((1 if single else len(texts), 384), 100.0)
            return arr[0] if single else arr

    # No LLM client → first retry attempt exhausts immediately
    tier1 = Tier1ExpressionFilter(alpha=1.0, llm_client=None, max_retries=3)
    tier1._embedder = StubEmbedder()

    result = asyncio.get_event_loop().run_until_complete(
        tier1.enforce_and_regenerate(
            text="some text",
            original_messages=[{"role": "user", "content": "hi"}],
            context={"expression_dna": edna},
            safe_template="SAFE_RESPONSE",
        )
    )
    final_text, enforcement = result
    assert final_text == "SAFE_RESPONSE"
    assert enforcement.passed is False


def test_g7_per_antipattern_behavioral_threshold():
    """G7: AntiPattern carries trigger_behavioral_threshold; tier3 honours it."""
    from src.skill.schema import AntiPattern
    from src.enforcement.tier3_block import Tier3AntiPatternBlock

    ap_high = AntiPattern(
        description="avoid report",
        trigger_conditions=[],
        trigger_action_patterns=["report"],
        trigger_behavioral_threshold=0.99,  # very high — should not fire at default 0.5
    )
    ap_default = AntiPattern(
        description="avoid close",
        trigger_conditions=[],
        trigger_action_patterns=["close"],
        # trigger_behavioral_threshold=None → uses global 0.5
    )

    # Attach a stub classifier returning proba=0.6
    class StubClassifier:
        trained = True
        def predict_proba(self, history):
            return 0.6

    tier3 = Tier3AntiPatternBlock(alpha=1.0)
    tier3.behavioral_threshold = 0.5
    tier3.attach_behavioral_classifier(StubClassifier(), threshold=0.5)

    # ap_high threshold 0.99 → 0.6 < 0.99 → classifier path says no; but
    # action_patterns=["report"] runs legacy path which fires on history match.
    # Use history that does NOT contain "report" so legacy path also no.
    r_high = tier3._check_behavioral(
        ["edit"], ap_high.trigger_action_patterns,
        per_pattern_threshold=ap_high.trigger_behavioral_threshold,
    )
    r_default = tier3._check_behavioral(
        ["close"], ap_default.trigger_action_patterns,
        per_pattern_threshold=ap_default.trigger_behavioral_threshold,
    )
    # ap_default: classifier says 0.6 >= 0.5 → fires
    assert r_default is not None and "0.6" in r_default
    # ap_high: classifier says 0.6 < 0.99 → no; legacy action match against
    # ["edit"] for pattern ["report"] → no. Should return None.
    assert r_high is None


def test_g9_linguistics_orthogonality_verifier():
    """G9: verify_orthogonality_to_expression_dna returns a structured report."""
    from src.evaluation.linguistics import verify_orthogonality_to_expression_dna

    messages = [
        Message(
            msg_id=f"m{i}", thread_id="t1", user_id="u",
            platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
            text=f"well perhaps {w} maybe think about it",
            action_type=ActionType.DISCUSS,
        )
        for i, w in enumerate(["good", "bad", "interesting", "flawed", "useful", "wrong"])
    ]
    report = verify_orthogonality_to_expression_dna(messages)
    assert "max_abs_correlation" in report
    assert "leakage_risk" in report
    assert "per_pair_correlations" in report
    assert report["n_messages"] == len(messages)
    assert isinstance(report["leakage_risk"], bool)


def test_g2_antipattern_llm_prompt_emits_trigger_fields():
    """G2: AntiPatternDetector LLM-detection prompt asks for A/B/C trigger fields
    and the parser wires them into the AntiPattern dataclass."""
    import inspect
    from src.skill.anti_patterns import DETECTION_PROMPT, AntiPatternDetector

    # Prompt must request all seven fields
    for field in [
        "trigger_regex",
        "trigger_semantic_phrases",
        "trigger_action_patterns",
        "trigger_keywords",
    ]:
        assert field in DETECTION_PROMPT, f"prompt missing {field}"

    # AntiPattern dataclass must expose Category A/B/C fields
    from src.skill.schema import AntiPattern
    fields = {f.name for f in AntiPattern.__dataclass_fields__.values()}
    assert {"trigger_regex", "trigger_semantic_phrases", "trigger_action_patterns"} <= fields


def test_g1_dual_pass_negative_case_selection():
    """G1: SkillCompiler._select_negative_cases excludes positive threads
    and falls back gracefully on small clusters."""
    from src.skill.compiler import SkillCompiler

    # We don't need an LLM client for this pure-Python helper
    compiler = SkillCompiler.__new__(SkillCompiler)
    compiler.top_n_threads = 5

    threads = []
    for i in range(10):
        t = Thread(thread_id=f"t{i}", platform=Platform.WIKIPEDIA, topic=f"x{i}")
        # Half are short (negative), half long (positive)
        n_msgs = 1 if i < 5 else 6
        for j in range(n_msgs):
            t.add_message(Message(
                msg_id=f"t{i}m{j}", thread_id=f"t{i}", user_id=f"u{j%2}",
                platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
                text="x", action_type=ActionType.DISCUSS,
            ))
        threads.append(t)

    positive = threads[5:]  # the long ones
    negative = compiler._select_negative_cases(threads, positive)
    positive_ids = {t.thread_id for t in positive}
    # No positive thread leaks into the negative set
    assert all(t.thread_id not in positive_ids for t in negative)
    # Negative set is non-empty
    assert len(negative) > 0


def test_g5_real_history_arm_wired():
    """G5: REAL_HISTORY enum + is_replay_only_condition + config wired."""
    from src.config.schemas import ConditionName
    from src.experiment.conditions import is_replay_only_condition, is_cadp_condition
    import yaml

    assert ConditionName.REAL_HISTORY.value == "real_history"
    assert is_replay_only_condition("real_history") is True
    assert is_replay_only_condition("cadp_full") is False
    assert is_cadp_condition("real_history") is False  # no skills compiled

    with open("configs/exp2_full.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "real_history" in cfg["conditions"]
    assert len(cfg["conditions"]) == 4  # outline §6.2 four-condition design


def test_g8_proxy_fallback_summary():
    """G8: aggregator.proxy_fallback_summary reports per-dataset provenance."""
    from src.evaluation.aggregator import MetricsAggregator, MetricsReport

    agg = MetricsAggregator()
    agg.reports = [
        MetricsReport("r1", "cadp_full", "wikipedia", "gpt-4o", 0,
                      used_role_label_proxy=True, used_held_out_events_heuristic=True),
        MetricsReport("r2", "cadp_full", "reddit", "gpt-4o", 0,
                      used_role_label_proxy=False, used_held_out_events_heuristic=False),
    ]
    summary = agg.proxy_fallback_summary()
    assert "wikipedia" in summary["datasets_using_role_label_proxy"]
    assert "reddit" not in summary["datasets_using_role_label_proxy"]
    assert summary["release_ready"] is False


async def test_g8_evaluate_survives_held_out_events_load_failure():
    """Regression: if _load_held_out_events throws, evaluate() must still
    construct a MetricsReport (held_out_events is referenced at report-build
    time, so it must be pre-initialized to None outside the try)."""
    import asyncio
    from datetime import datetime
    from src.evaluation.aggregator import MetricsAggregator
    from src.simulation.sandbox import SimulationResult
    from src.data.schemas import Thread, Message, ActionType, Platform

    # Build minimal real threads + sim result
    real_threads = []
    for i in range(3):
        t = Thread(thread_id=f"rt{i}", platform=Platform.WIKIPEDIA, topic=f"r{i}")
        for j in range(3):
            t.add_message(Message(
                msg_id=f"rt{i}m{j}", thread_id=f"rt{i}", user_id=f"ru{j%2}",
                platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
                text=f"real msg {j}", action_type=ActionType.DISCUSS,
            ))
        real_threads.append(t)

    sim_result = SimulationResult(
        run_id="test", condition="cadp_full", dataset="wikipedia",
        model="gpt-4o", repeat=0, rounds=1,
        messages=[{
            "msg_id": "s1", "thread_id": "rt0", "user_id": "a1",
            "action_type": "discuss", "text": "sim msg",
            "parent_msg_id": None,
        }],
        agent_states=[{"agent_id": "a1", "cluster_id": 0}],
    )

    agg = MetricsAggregator()

    # Force _load_held_out_events to throw
    def boom(_dataset):
        raise RuntimeError("simulated load failure")
    agg._load_held_out_events = boom

    # Must not raise
    report = await agg.evaluate(sim_result, real_threads)
    # Predictive layer removed; heuristic flag always False now
    assert report.used_held_out_events_heuristic is False
    assert report.used_role_label_proxy is True  # no role_labels dir configured


def test_g4_alpha_sweep_75_cells():
    """G4: three pairwise sweeps × 25 = 75 cells per (dataset, model)."""
    import inspect
    from src.experiment.alpha_sensitivity import AlphaSensitivityRunner, ALPHA_GRID

    assert ALPHA_GRID == [0.0, 0.25, 0.5, 0.75, 1.0]
    src = inspect.getsource(AlphaSensitivityRunner.sweep_all_pairs)
    # Three faces, each pinned at α=1.0 on the off-face tier.
    # The call site uses positional dict literal, e.g. sweep_pair(..., {"t3": 1.0})
    compact = src.replace(" ", "")
    assert '{"t3":1.0}' in compact
    assert '{"t2":1.0}' in compact
    assert '{"t1":1.0}' in compact
    # Module docstring must not advertise the full 5^3 = 125 cube.
    module_src = inspect.getsource(__import__("src.experiment.alpha_sensitivity", fromlist=["x"]))
    assert "5^3" not in module_src.replace(" ", "")
    assert "=125" not in module_src.replace(" ", "")


# ----------------------------------------------------------------------
# Audit P7 — Tier-1 Bonferroni regression test
# ----------------------------------------------------------------------

def test_tier1_bonferroni_threshold_d384():
    """Pin the dimension-aware Bonferroni threshold at d=384.

    Outline §4.4 + tier1_filter docstring derive: alpha_per_dim = 2(1-Φ(2))/d,
    threshold = Φ⁻¹(1 - alpha_per_dim/2). For per_dim_sigma=2.0 and d=384
    (the embedding dim of ``all-MiniLM-L6-v2``) this evaluates to ≈3.85,
    safely above E[max|z|]≈sqrt(2 ln 384)≈3.45 so the filter does not
    systematically over-flag.
    """
    from scipy.stats import norm
    from src.enforcement.tier1_filter import _bonferroni_threshold

    expected = float(norm.ppf(1.0 - (2.0 * (1.0 - norm.cdf(2.0)) / 384) / 2.0))
    actual = _bonferroni_threshold(per_dim_sigma=2.0, dim=384)
    assert abs(actual - expected) < 1e-9
    # Anchor on the analytic value to catch silent regressions (e.g. anyone
    # reverting to the legacy 2.0 threshold or to the sqrt(2 ln d) Gumbel
    # approximation ≈3.45 used by the no-scipy fallback path).
    assert 3.80 < actual < 3.90, f"expected ≈3.85 at d=384, got {actual}"
    # And distinctly above both the legacy 2σ and the Gumbel approximation,
    # so a refactor that mistakenly uses either of those would fail here.
    assert actual > 3.50, "threshold collapsed below sqrt(2 ln 384)≈3.45"


def test_tier1_bonferroni_threshold_low_dim():
    """At d=1 the Bonferroni correction is a no-op: threshold == per_dim_sigma."""
    from src.enforcement.tier1_filter import _bonferroni_threshold
    assert abs(_bonferroni_threshold(2.0, 1) - 2.0) < 1e-9


def test_tier1_bonferroni_monotone_in_d():
    """Threshold strictly increases with embedding dimensionality."""
    from src.enforcement.tier1_filter import _bonferroni_threshold
    thresholds = [_bonferroni_threshold(2.0, d) for d in (1, 32, 128, 384, 1024)]
    assert all(b > a for a, b in zip(thresholds, thresholds[1:])), thresholds


def test_tier1_filter_dimension_aware_default():
    """Tier1ExpressionFilter must default to dimension_aware=True.

    Falling back to the legacy 2σ threshold silently would systematically
    over-flag at d=384 — the exact regression P7 is meant to prevent.
    """
    import inspect
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    sig = inspect.signature(Tier1ExpressionFilter.__init__)
    assert sig.parameters["dimension_aware"].default is True

