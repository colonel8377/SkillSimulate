"""Tests for core modules."""

import numpy as np
import pytest
from datetime import datetime

from src.data.schemas import ActionType, Message, Platform, Thread


def test_action_type_for_platform():
    wiki_actions = ActionType.for_platform(Platform.WIKIPEDIA)
    assert ActionType.EDIT in wiki_actions
    assert ActionType.REVERT in wiki_actions
    assert ActionType.AGREE not in wiki_actions
    assert ActionType.DISAGREE not in wiki_actions

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


def test_pop_aligned_uses_named_features_not_vector_positions():
    from src.agents.pop_aligned import compute_cluster_attributes
    from src.clustering.features import UserFeatures

    feature = UserFeatures(
        user_id="u",
        reply_rate=0.99,
        mean_indentation=2.5,
        verbosity=9.0,
        activity=8.0,
        question_rate=0.3,
        conflict_engagement_ratio=0.4,
        message_count=10,
        thread_count=3,
    )
    stats = compute_cluster_attributes([feature])
    assert stats["reply_depth"]["mean"] == 2.5
    assert stats["verbosity"]["mean"] == 9.0
    assert stats["question_rate"]["mean"] == 0.3
    assert stats["conflict_engagement_ratio"]["mean"] == 0.4


def test_canonical_action_taxonomy_aligns_wikipedia_actions():
    from src.evaluation.aggregator import MetricsAggregator

    assert MetricsAggregator._canonical_action("agree", "wikipedia") == "participation"
    assert MetricsAggregator._canonical_action("discuss", "wikipedia") == "participation"
    assert MetricsAggregator._canonical_action("disagree", "wikipedia") == "conflict"
    assert MetricsAggregator._canonical_action("revert", "wikipedia") == "conflict"
    assert MetricsAggregator._canonical_action("report", "wikipedia") == "moderation"


def test_rsa_unavailable_is_flagged_without_dropping_metric_schema():
    from src.evaluation.micro import MicroMetrics

    result = MicroMetrics.compute(
        sim_profiles=np.eye(3), real_profiles=np.eye(3),
    )
    assert result["rsa"] == 0.0
    assert result["rsa_available"] is False


def test_linguistic_sample_is_round_stratified():
    from src.evaluation.aggregator import MetricsAggregator

    messages = [
        {"msg_id": f"m{r}_{i}", "thread_id": f"t{i % 2}", "round": r}
        for r in range(10) for i in range(20)
    ]
    sample = MetricsAggregator._stratified_message_sample(messages, 20)
    assert {m["round"] for m in sample} == set(range(10))


def test_real_and_sim_chain_depths_match_on_identical_messages():
    from src.evaluation.aggregator import MetricsAggregator

    thread = Thread("t", Platform.WIKIPEDIA, "topic")
    thread.add_message(Message(
        msg_id="root", thread_id="t", user_id="u1",
        platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="root",
        action_type=ActionType.DISCUSS,
    ))
    thread.add_message(Message(
        msg_id="child", thread_id="t", user_id="u2",
        platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="child",
        action_type=ActionType.DISCUSS, parent_msg_id="root",
    ))
    # A sliced corpus can retain a message whose parent is outside the slice.
    thread.add_message(Message(
        msg_id="orphan", thread_id="t", user_id="u3",
        platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="orphan",
        action_type=ActionType.DISCUSS, parent_msg_id="not-in-slice",
    ))
    agg = MetricsAggregator()
    raw = [agg._msg_to_dict(m) for m in thread.messages]
    assert agg._compute_chain_lengths([thread]) == [1, 2, 1]
    assert agg._compute_chain_lengths_from_dicts(raw) == [1, 2, 1]


def test_exp1_evaluation_is_local_and_model_independent():
    from src.config.schemas import ExperimentConfig
    from src.experiment.exp1_validation import Experiment1Runner

    runner = Experiment1Runner(ExperimentConfig.from_yaml("configs/exp1_v2.yaml"))
    assert runner.metrics_agg._llm_client is None
    assert runner.metrics_agg._llm_model_name is None


def test_exp1_refuses_verdict_for_incomplete_grid(tmp_path):
    from types import SimpleNamespace
    from src.experiment.exp1_validation import Experiment1Runner

    runner = Experiment1Runner.__new__(Experiment1Runner)
    runner.results_dir = tmp_path
    runner.config = SimpleNamespace(
        conditions=["cadp_advisory_nuwa", "cadp_full_nuwa"],
        datasets=["wikipedia"], models=["m"], num_repeats=1,
    )
    with pytest.raises(ValueError, match="incomplete or stale Exp1 grid"):
        runner.save_all_metrics()


def test_degenerate_action_reference_fails_closed():
    from src.evaluation.aggregator import EvaluationIntegrityError, MetricsAggregator

    with pytest.raises(EvaluationIntegrityError):
        MetricsAggregator._validate_action_reference({"participation": 1.0})


def test_exp1_conditions_share_stimulus_manifest(tmp_path):
    from types import SimpleNamespace
    from src.experiment.exp1_validation import Experiment1Runner
    from src.experiment.runner import ExperimentCell

    seeds = []
    for i in range(120):
        thread = Thread(
            thread_id=f"cga{i:03d}", platform=Platform.WIKIPEDIA,
            topic=f"topic-{i}",
        )
        thread.add_message(Message(
            msg_id=f"m{i}", thread_id=thread.thread_id, user_id="real",
            platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="seed",
            action_type=ActionType.DISCUSS,
        ))
        seeds.append(thread)

    wiki_refs = []
    for i in range(120):
        thread = Thread(
            thread_id=f"wiki{i:03d}", platform=Platform.WIKIPEDIA,
            topic=f"wiki-topic-{i}",
        )
        thread.add_message(Message(
            msg_id=f"w{i}a", thread_id=thread.thread_id, user_id="u1",
            platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="discussion",
            action_type=ActionType.DISCUSS,
        ))
        thread.add_message(Message(
            msg_id=f"w{i}b", thread_id=thread.thread_id, user_id="u2",
            platform=Platform.WIKIPEDIA, timestamp=datetime.now(), text="changed",
            action_type=ActionType.EDIT, parent_msg_id=f"w{i}a",
        ))
        wiki_refs.append(thread)

    runner = Experiment1Runner.__new__(Experiment1Runner)
    runner.results_dir = tmp_path
    runner.config = SimpleNamespace(
        seed=42, seed_min_toxicity=0.6, max_sim_threads=3,
    )
    runner._load_cga_seed_threads = lambda: seeds

    c1 = ExperimentCell("vanilla", "wikipedia", "m", 0)
    c2 = ExperimentCell("cadp_full_nuwa", "wikipedia", "m", 0)
    sim1, ref1, ling1 = runner._prepare_sim_threads(wiki_refs, c1)
    sim2, ref2, ling2 = runner._prepare_sim_threads(wiki_refs, c2)

    source1 = [t.thread_id.rsplit("_", 1)[-1] for t in sim1]
    source2 = [t.thread_id.rsplit("_", 1)[-1] for t in sim2]
    assert source1 == source2
    assert [t.thread_id for t in ref1] == [t.thread_id for t in ref2]
    assert [t.thread_id for t in ling1] == [t.thread_id for t in ling2]
    assert set(source1).isdisjoint({t.thread_id for t in ling1})
    assert all(t.thread_id.startswith("wiki") for t in ref1)


def test_uncalibrated_trigger_sanitizer_is_noop():
    """Rule-based triggers deleted 2026-07-17; sanitizer is now a no-op."""
    from src.experiment.exp1_validation import Experiment1Runner
    status = Experiment1Runner._sanitize_uncalibrated_triggers("wikipedia", {})
    assert status == "rule_path_removed_no_triggers_to_sanitize"


def test_advisory_and_full_have_identical_static_skill_text():
    from types import SimpleNamespace
    from src.agents.cadp import CADPAgent
    from src.agents.cadp_advisory import CADPAdvisoryAgent

    adapter = SimpleNamespace(
        build_system_prompt=lambda **kwargs: f"system:{kwargs}",
        build_constraint_text=lambda **kwargs: f"constraints:{kwargs}",
    )
    full = CADPAgent.__new__(CADPAgent)
    full.adapter = adapter
    full.show_expression_dna = True
    full.show_mind_models = True
    full.show_anti_patterns = True
    advisory = CADPAdvisoryAgent.__new__(CADPAdvisoryAgent)
    advisory.adapter = adapter
    advisory.enforcement_harness = None

    assert advisory.get_role_description() == full.get_role_description()
    assert advisory.get_constraints_text() == full.get_constraints_text()
    assert advisory.enforcement_harness is None
    assert advisory.get_reflection_directive() is None


def test_locked_cluster_ids_are_anonymized_into_runtime_keyspace(tmp_path):
    import pickle
    from types import SimpleNamespace
    from src.clustering.clusterer import ClusterResult
    from src.clustering.features import UserFeatures
    from src.data.pii import anonymize_user_id
    from src.experiment.exp1_validation import Experiment1Runner

    raw_uid = "VisiblePublicHandle"
    cr = ClusterResult(
        labels={raw_uid: 3}, n_clusters=1, centroids=np.zeros((1, 1)),
        silhouette_score=0.0, davies_bouldin_score=0.0,
        behavioral_weight=1.0, language_weight=0.0,
        user_features={raw_uid: UserFeatures(user_id=raw_uid)},
    )
    path = tmp_path / "locked.pkl"
    with path.open("wb") as handle:
        pickle.dump(cr, handle)
    runner = Experiment1Runner.__new__(Experiment1Runner)
    runner.config = SimpleNamespace(cluster_merge_map_path=None)

    loaded = runner._load_locked_clusters(str(path))
    anonymous_uid = anonymize_user_id(raw_uid)
    assert loaded.labels == {anonymous_uid: 3}
    assert anonymous_uid in loaded.user_features
    assert loaded.user_features[anonymous_uid].user_id == anonymous_uid


@pytest.mark.asyncio
async def test_empty_simulation_fails_integrity_instead_of_evaluating(tmp_path):
    from src.simulation.sandbox import SimulationIntegrityError, SimulationSandbox
    from src.simulation.platforms.wikipedia import WikipediaTopology

    sandbox = SimulationSandbox(
        WikipediaTopology(), checkpoint_dir=tmp_path, min_turn_success_rate=0.95,
    )
    thread = Thread("t", Platform.WIKIPEDIA, "topic")
    with pytest.raises(SimulationIntegrityError):
        await sandbox.run(
            agents=[], threads=[thread], num_rounds=1, run_id="empty",
            condition="vanilla", dataset="wikipedia", model="mock",
        )


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
                    reason="Hostility violates wiki norms",
                    evidence=["User called another editor 'stupid'"],
                    correct_alternative="Address the edit, not the editor",
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
    """G6: Tier 1 uses safe-template fallback after max_retries exhausted."""
    import asyncio
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    from src.skill.schema import ExpressionDNA

    # Centroid at origin, threshold tiny → any non-zero text embedding is
    # well outside the cosine boundary and triggers regeneration.
    edna = ExpressionDNA(
        embedding_centroid=[0.0] * 384,
        embedding_cosine_threshold=0.001,
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

    result = asyncio.run(
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


def test_tier3_llm_judge_blocks_hostility():
    """Tier-3 LLM judge: a `violated=true` verdict blocks the message.

    Replaces the deleted rule-based test_g7_per_antipattern_behavioral_threshold
    after the 2026-07-17 enforcement simplification collapsed Tier 3 to a
    single LLM judge call.
    """
    import asyncio
    from src.enforcement.tier3_llm_judge import Tier3LLMJudge
    from src.skill.schema import (
        AntiPattern, CapabilityTrack, ConstraintTrack,
        ExpressionDNA, MindModel, SkillFile,
    )

    ap = AntiPattern(
        description="Personal attacks",
        trigger_conditions=["message insults another editor"],
        reason="Hostility is prohibited",
    )
    skill = SkillFile(
        cluster_id="c0",
        platform="wikipedia",
        capability=CapabilityTrack(
            expression_dna=ExpressionDNA(),
            mind_models=[],
        ),
        constraint=ConstraintTrack(anti_patterns=[ap]),
        archetype_label="test",
    )

    class StubLLM:
        async def chat_completion(self, *_, **__):
            return '{"0": {"violated": true, "confidence": 0.9, "reason": "insult"}}'

    judge = Tier3LLMJudge(llm_client=StubLLM(), audit_only=False)

    async def _run():
        from src.enforcement.harness import EnforcementHarness
        harness = EnforcementHarness(
            enable_tier1=False,
            enable_tier2=False,
            enable_tier3=True,
            skill=skill,
            tier3_llm_judge=judge,
        )
        text, log, replan = await harness.enforce_output(
            text="you are a troll and a fool",
            original_messages=[],
        )
        assert log.tier3_llm_post is not None
        assert log.tier3_llm_post.passed is False
        assert log.tier3_hard_block_triggered is True
        assert replan is not None and "LLM judge" in replan

    asyncio.run(_run())


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


def test_g2_antipattern_schema_is_trimmed():
    """G2 (2026-07-17 refactor): AntiPattern exposes only the 5 fields the
    LLM judge consumes; rule-based trigger fields are deleted."""
    from src.skill.schema import AntiPattern
    fields = {f.name for f in AntiPattern.__dataclass_fields__.values()}
    expected = {"description", "trigger_conditions", "reason",
                "evidence", "correct_alternative"}
    assert fields == expected, f"extra/missing fields: {fields ^ expected}"
    # Rule-based fields must be gone
    assert not ({"trigger_regex", "trigger_keywords", "trigger_semantic_phrases",
                 "trigger_action_patterns", "trigger_semantic_threshold",
                 "trigger_behavioral_threshold"} & fields)


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


async def test_g8_evaluate_survives_held_out_events_load_failure(monkeypatch):
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
    for i in range(10):
        t = Thread(thread_id=f"rt{i}", platform=Platform.WIKIPEDIA, topic=f"r{i}")
        for j in range(2):
            t.add_message(Message(
                msg_id=f"rt{i}m{j}", thread_id=f"rt{i}", user_id=f"ru{i}_{j}",
                platform=Platform.WIKIPEDIA, timestamp=datetime.now(),
                text=f"real msg {j}",
                action_type=ActionType.DISCUSS if j == 0 else ActionType.EDIT,
                parent_msg_id=f"rt{i}m0" if j else None,
            ))
        real_threads.append(t)

    sim_result = SimulationResult(
        run_id="test", condition="cadp_full", dataset="wikipedia",
        model="gpt-4o", repeat=0, rounds=1,
        messages=[
            {
                "msg_id": f"s{i}", "thread_id": "sim_t", "user_id": f"a{i}",
                "action_type": "discuss" if i % 2 == 0 else "edit",
                "text": f"sim msg {i}", "round": 0,
                "parent_msg_id": f"s{i-1}" if i else None, "metadata": {},
            }
            for i in range(10)
        ],
        agent_states=[{"agent_id": f"a{i}", "cluster_id": i % 2} for i in range(10)],
        per_round_metrics=[{"round": 0}], run_fingerprint="fp",
    )

    agg = MetricsAggregator()

    # Force _load_held_out_events to throw
    def boom(_dataset):
        raise RuntimeError("simulated load failure")
    agg._load_held_out_events = boom

    async def fake_linguistics(*_args, **_kwargs):
        return {
            "discourse_relation_match": 1.0,
            "sentiment_trajectory_similarity": 1.0,
            "speech_act_similarity": 1.0,
            "sip": 1.0,
        }
    monkeypatch.setattr(
        "src.evaluation.aggregator.LinguisticMetrics.compute", fake_linguistics,
    )

    # Must not raise
    report = await agg.evaluate(sim_result, real_threads)
    # Predictive layer removed; heuristic flag always False now
    assert report.used_held_out_events_heuristic is False
    assert report.used_role_label_proxy is True  # no role_labels dir configured
    assert "complexity_gap" in report.metrics
    assert "caricature_index_real" in report.metrics
    assert "caricature_gap" in report.metrics


def test_wikipedia_free_text_does_not_create_platform_action():
    from src.data.wikipedia import WikipediaLoader

    loader = WikipediaLoader("unused")
    assert loader._infer_action({}, "I reported that the edit was reverted") == ActionType.DISCUSS
    assert loader._infer_action({"rev_id": 123}, "ordinary summary") == ActionType.EDIT


def test_wikipedia_topology_uses_platform_actions_and_conflict_metadata():
    from src.simulation.platforms.wikipedia import WikipediaTopology

    thread = Thread("t", Platform.WIKIPEDIA, "topic")
    thread.add_message(Message(
        "m", "t", "other", Platform.WIKIPEDIA, datetime.now(), "attack",
        ActionType.DISCUSS, metadata={"toxicity": "0.8"},
    ))
    actions = WikipediaTopology().get_valid_actions(thread, "agent")
    assert ActionType.REPORT in actions
    assert ActionType.AGREE not in actions
    assert ActionType.DISAGREE not in actions


def test_action_js_smoothing_is_small_probability_epsilon():
    from src.evaluation.macro import normalized_entropy_distance

    p = {"participation": 1.0, "conflict": 0.0}
    assert normalized_entropy_distance(p, p, smoothing=0.001) == pytest.approx(0.0)
    assert 0 < normalized_entropy_distance(
        {"participation": 1.0}, {"conflict": 1.0}, smoothing=0.001,
    ) < 1
    with pytest.raises(ValueError):
        normalized_entropy_distance(p, p, smoothing=-0.1)


def test_sentiment_zero_trajectory_self_similarity(monkeypatch):
    import src.evaluation.linguistics as ling

    zeros = {"variance": 0.0, "trend_slope": 0.0, "oscillation_freq": 0.0, "range": 0.0}
    monkeypatch.setattr(ling, "sentiment_trajectory_shape", lambda _messages: zeros)
    assert ling.sentiment_trajectory_similarity([], []) == 1.0


@pytest.mark.asyncio
async def test_speech_act_model_failure_fails_closed(monkeypatch):
    import src.evaluation.linguistics as ling

    monkeypatch.setattr(ling, "_get_local_speech_act_pipeline", lambda: False)
    message = Message("m", "t", "u", Platform.WIKIPEDIA, datetime.now(), "text")
    with pytest.raises(ling.LinguisticModelUnavailable):
        await ling.speech_act_ratio([message])


def test_pdtb_model_failure_fails_closed(monkeypatch):
    import src.evaluation.linguistics as ling

    monkeypatch.setattr(ling, "_get_pdtb_pipeline", lambda: False)
    message = Message("m", "t", "u", Platform.WIKIPEDIA, datetime.now(), "text")
    with pytest.raises(ling.LinguisticModelUnavailable):
        ling.discourse_relation_distribution([message])


def test_sentiment_model_failure_fails_closed(monkeypatch):
    import src.evaluation.linguistics as ling

    classifier = ling.RoBERTaSentimentClassifier()
    monkeypatch.setattr(classifier, "_ensure_pipeline", lambda: None)
    classifier._pipeline = False
    with pytest.raises(ling.LinguisticModelUnavailable):
        classifier.score_batch(["text"])


def test_composite_metric_weights_must_be_explicit_and_normalized():
    from src.evaluation.aggregator import MetricsAggregator

    MetricsAggregator(
        linguistic_metric_weights={
            "discourse_relation_match": 0.25,
            "sentiment_trajectory_similarity": 0.25,
            "speech_act_similarity": 0.25,
            "sip": 0.25,
        },
        interaction_metric_weights={"cascade": 0.5, "graph": 0.5},
    )
    with pytest.raises(ValueError, match="sum to 1"):
        MetricsAggregator(
            linguistic_metric_weights={
                "discourse_relation_match": 1.0,
                "sentiment_trajectory_similarity": 1.0,
                "speech_act_similarity": 1.0,
                "sip": 1.0,
            },
        )


@pytest.mark.asyncio
async def test_micro_batch_exposes_earlier_same_round_messages():
    from types import SimpleNamespace
    from src.simulation.platforms.wikipedia import WikipediaTopology
    from src.simulation.sandbox import SimulationSandbox

    class Agent:
        engagement_ratio = 1.0

        def __init__(self, agent_id):
            self.agent_id = agent_id

        async def take_turn(self, thread, _actions, round_num):
            observed = len(thread.messages)
            return Message(
                f"{self.agent_id}-{round_num}", thread.thread_id, self.agent_id,
                thread.platform, datetime.now(), str(observed), ActionType.DISCUSS,
                parent_msg_id=thread.messages[-1].msg_id,
            ), None

        def observe(self, *_args):
            pass

    thread = Thread("t", Platform.WIKIPEDIA, "topic")
    thread.add_message(Message(
        "seed", "t", "real", Platform.WIKIPEDIA, datetime.now(), "seed",
        ActionType.DISCUSS,
    ))
    sandbox = SimulationSandbox(WikipediaTopology(), micro_batch_size=1)
    messages, _, _, planned = await sandbox._run_round(
        [Agent("a"), Agent("b")], [thread], 0,
    )
    assert planned == 2
    assert [row["text"] for row in messages] == ["1", "2"]


@pytest.mark.asyncio
async def test_manipulation_audit_reports_inactive_without_blocking_treatment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.experiment.exp1_validation import Experiment1Runner
    from src.experiment.runner import ExperimentCell
    from src.skill.schema import CapabilityTrack, ConstraintTrack, ExpressionDNA, SkillFile

    class Checkpoints:
        def is_completed(self, _run_id):
            return True

        def load(self, _run_id):
            return {
                "agents_state": [{"agent_id": "a", "cluster_id": 0}],
                "messages_log": [{"user_id": "a", "text": "outlier"}],
                "extra": {"checkpoint_schema_version": 2, "round_complete": True},
            }

    class Embedder:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("executor wrapper should be mocked")

    async def fake_embed(_fn, _texts, **_kwargs):
        return np.tile(np.array([[2.0, 0.0]]), (len(_texts), 1))

    monkeypatch.setattr("src.config.embedder.run_embed_in_executor", fake_embed)
    monkeypatch.setattr("src.config.settings.get_shared_embedder", lambda: Embedder())
    runner = Experiment1Runner.__new__(Experiment1Runner)
    runner.config = SimpleNamespace(manipulation_min_potential_rate=0.5, num_repeats=3)
    runner.checkpoint = Checkpoints()
    runner.results_dir = tmp_path
    skill = SkillFile(
        cluster_id="0", platform="wikipedia",
        capability=CapabilityTrack(ExpressionDNA(
            embedding_centroid=[0.0, 1.0],
            embedding_cosine_threshold=0.1,
        ), []),
        constraint=ConstraintTrack([]),
    )
    audit = await runner._validate_manipulation_potential(
        ExperimentCell("cadp_full_nuwa", "wikipedia", "flash", 0), {0: skill},
    )
    assert audit["passed"] is True
    assert audit["potential_rate"] == 1.0
    assert (tmp_path / "manipulation_audits" / "cadp_full_nuwa_wikipedia_flash_r0.json").exists()

    async def in_distribution(_fn, _texts, **_kwargs):
        return np.tile(np.array([[0.1, 1.0]]), (len(_texts), 1))
    monkeypatch.setattr("src.config.embedder.run_embed_in_executor", in_distribution)
    inactive = await runner._validate_manipulation_potential(
        ExperimentCell("cadp_full_nuwa", "wikipedia", "flash", 1), {0: skill},
    )
    assert inactive["passed"] is False
    assert inactive["potential_rate"] == 0.0


def test_observed_continuation_uses_same_thread_prefix_and_suffix(tmp_path):
    from types import SimpleNamespace
    from src.experiment.exp1_validation import Experiment1Runner
    from src.experiment.runner import ExperimentCell

    threads = []
    for i in range(4):
        thread = Thread(f"source-{i}", Platform.WIKIPEDIA, f"topic-{i}")
        for j in range(6):
            thread.add_message(Message(
                f"m{i}-{j}", thread.thread_id, f"u{j % 2}", Platform.WIKIPEDIA,
                datetime.now(), f"message {j}",
                ActionType.EDIT if j == 4 else ActionType.DISCUSS,
                parent_msg_id=f"m{i}-{j-1}" if j else None,
                metadata={"toxicity": 0.7 if i == 0 else 0.0},
            ))
        threads.append(thread)

    runner = Experiment1Runner.__new__(Experiment1Runner)
    runner.results_dir = tmp_path
    runner.config = SimpleNamespace(
        continuation_min_messages=6, continuation_prefix_fraction=0.5,
        max_sim_threads=2, seed=42,
    )
    runner._tier1_calibration_thread_ids = set()
    runner._load_distillation_thread_ids = lambda: set()
    sim, refs, ling = runner._prepare_observed_continuations(
        threads, ExperimentCell("cadp_advisory_nuwa", "wikipedia", "flash", 0),
    )
    assert len(sim) == len(refs) == len(ling) == 2
    originals = {thread.thread_id: thread for thread in threads}
    for seed, reference in zip(sim, refs):
        source_id = reference.thread_id
        original_ids = [m.msg_id for m in originals[source_id].messages]
        assert [m.msg_id for m in seed.messages] == original_ids[:3]
        assert [m.msg_id for m in reference.messages] == original_ids[3:]
        suffix_ids = {m.msg_id for m in reference.messages}
        assert any(m.parent_msg_id in suffix_ids for m in reference.messages)
    sim_full, refs_full, _ = runner._prepare_observed_continuations(
        threads, ExperimentCell("cadp_full_nuwa", "wikipedia", "flash", 0),
    )
    assert [t.thread_id for t in refs_full] == [t.thread_id for t in refs]
    assert [
        [m.msg_id for m in thread.messages] for thread in sim_full
    ] == [
        [m.msg_id for m in thread.messages] for thread in sim
    ]


def test_wikipedia_loader_balances_target_across_years(tmp_path, monkeypatch):
    import json
    from src.data.wikipedia import WikipediaLoader

    dirs = []
    for year in (2001, 2002):
        directory = tmp_path / f"wikiconv-{year}"
        directory.mkdir()
        (directory / "conversations.json").write_text(json.dumps({
            f"c{year}": {"meta": {"page_title": str(year)}}
        }))
        rows = [
            {"id": f"{year}-{i}", "conversation_id": f"c{year}",
             "speaker": f"u{i}", "text": "comment", "meta": {},
             "reply-to": f"{year}-0" if i else None}
            for i in range(2)
        ]
        (directory / "utterances.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        dirs.append(directory)

    loader = WikipediaLoader(str(tmp_path), target_threads=2, min_messages=2)
    monkeypatch.setattr(loader, "_load_cga_labels", lambda _root: {})
    loaded = loader._load_convokit(dirs, tmp_path)
    assert {thread.thread_id for thread in loaded} == {"c2001", "c2002"}


@pytest.mark.asyncio
async def test_planner_separates_platform_action_from_stance():
    from src.agents.planning import Planner

    class LLM:
        async def chat_completion_json(self, *_args, **_kwargs):
            return {
                "action_type": "discuss", "stance": "oppositional",
                "text": "I disagree with the premise", "target_msg_id": "m",
                "reasoning": "challenge",
            }

    thread = Thread("t", Platform.WIKIPEDIA, "topic")
    thread.add_message(Message(
        "m", "t", "other", Platform.WIKIPEDIA, datetime.now(), "claim",
        ActionType.DISCUSS,
    ))
    plan = await Planner(LLM()).plan(
        thread, [ActionType.DISCUSS, ActionType.EDIT], "",
    )
    assert plan.action_type == ActionType.DISCUSS
    assert plan.stance == "oppositional"


def test_viability_pre_registered_go_rule():
    import pandas as pd
    from types import SimpleNamespace
    from src.analysis.viability import evaluate_viability

    rows = []
    for repeat in range(3):
        common = {
            "dataset": "wikipedia", "model": "m", "repeat": repeat,
            "simulation_message_count": 100,
            "enforcement_safe_template_rate": 0.0,
        }
        rows.append({
            **common, "condition": "cadp_advisory_nuwa",
            "action_fidelity_distance": 0.5,
            "interaction_structure_distance": 0.6,
            "linguistic_fidelity_distance": 0.4,
            "action_text_consistency": 0.95,
        })
        rows.append({
            **common, "condition": "cadp_full_nuwa",
            "action_fidelity_distance": 0.4,
            "interaction_structure_distance": 0.5,
            "linguistic_fidelity_distance": 0.35,
            "action_text_consistency": 0.95,
        })
    config = SimpleNamespace(
        viability_treatment="cadp_full_nuwa",
        viability_control="cadp_advisory_nuwa",
        viability_primary_metrics=[
            "action_fidelity_distance", "interaction_structure_distance",
            "linguistic_fidelity_distance",
        ],
        viability_min_pairs=3,
        viability_min_metric_wins=2,
        viability_min_repeat_win_fraction=2 / 3,
        viability_min_relative_improvement=0.05,
        viability_max_family_regression=0.10,
        viability_min_message_ratio=0.95,
        viability_max_safe_template_rate=0.10,
        viability_min_action_text_consistency=0.90,
    )
    report = evaluate_viability(pd.DataFrame(rows), config)
    assert report["verdict"] == "GO"
    assert report["metric_wins"] == 3
    assert all(
        row["required_repeat_wins"] == 2
        for row in report["metric_results"].values()
    )


@pytest.mark.parametrize("failure", ["effect", "regression", "action_text"])
def test_viability_stop_guards(failure):
    import pandas as pd
    from types import SimpleNamespace
    from src.analysis.viability import evaluate_viability

    metrics = [
        "action_fidelity_distance", "interaction_structure_distance",
        "linguistic_fidelity_distance",
    ]
    rows = []
    for repeat in range(3):
        common = {
            "dataset": "wikipedia", "model": "m", "repeat": repeat,
            "simulation_message_count": 100,
            "enforcement_safe_template_rate": 0.0,
            "action_text_consistency": 0.95,
        }
        rows.append({
            **common, "condition": "cadp_advisory_nuwa",
            **{metric: 1.0 for metric in metrics},
        })
        treatment_metrics = {metric: 0.9 for metric in metrics}
        if failure == "effect":
            treatment_metrics = {metric: 0.99 for metric in metrics}
        elif failure == "regression":
            treatment_metrics["linguistic_fidelity_distance"] = 1.2
        treatment = {
            **common, "condition": "cadp_full_nuwa", **treatment_metrics,
        }
        if failure == "action_text":
            treatment["action_text_consistency"] = 0.80
        rows.append(treatment)
    config = SimpleNamespace(
        viability_treatment="cadp_full_nuwa",
        viability_control="cadp_advisory_nuwa",
        viability_primary_metrics=metrics,
        viability_min_pairs=3,
        viability_min_metric_wins=2,
        viability_min_repeat_win_fraction=2 / 3,
        viability_min_relative_improvement=0.05,
        viability_max_family_regression=0.10,
        viability_min_message_ratio=0.95,
        viability_max_safe_template_rate=0.10,
        viability_min_action_text_consistency=0.90,
    )
    report = evaluate_viability(pd.DataFrame(rows), config)
    assert report["verdict"] == "STOP"


def test_balanced_population_and_empirical_engagement_do_not_collapse():
    import random
    from src.clustering.clusterer import ClusterResult
    from src.clustering.features import UserFeatures
    from src.simulation.population import PopulationBuilder

    features = {}
    labels = {}
    for cid in range(6):
        for i in range(40):
            uid = f"{cid}-{i}"
            features[uid] = UserFeatures(user_id=uid, message_count=5 + i * i)
            labels[uid] = cid
    cr = ClusterResult(
        labels=labels, n_clusters=6, centroids=np.zeros((6, 1)),
        silhouette_score=0.0, davies_bouldin_score=0.0,
        behavioral_weight=1.0, language_weight=0.0,
        user_features=features,
    )
    allocations = PopulationBuilder._allocate_balanced(18, cr.get_cluster_ids())
    assert allocations == {cid: 3 for cid in range(6)}
    rng = random.Random(42)
    ratios = [
        PopulationBuilder._sample_engagement_ratio("0", cr, rng)
        for _ in range(18)
    ]
    assert len(set(round(value, 4) for value in ratios)) > 3
    assert min(ratios) >= 0.05
    assert max(ratios) <= 0.50


def test_distillation_overlap_fails_closed():
    from src.experiment.exp1_validation import Experiment1Runner
    with pytest.raises(ValueError, match="Train/evaluation leakage"):
        Experiment1Runner._assert_zero_distillation_overlap(
            ["held-out", "leaked"], {"leaked"}, "unit-test",
        )


def test_locked_vector_bootstrap_reports_source_and_final_ari():
    from src.clustering.clusterer import ClusterResult
    from src.clustering.features import UserFeatures
    from src.clustering.validation import ClusterStabilityValidator

    features = {}
    labels = {}
    for i in range(160):
        cid = i % 2
        features[str(i)] = UserFeatures(
            user_id=str(i), reply_rate=float(cid) + (i % 7) * 0.001,
            verbosity=float(cid) + (i % 5) * 0.001,
        )
        labels[str(i)] = cid
    cr = ClusterResult(
        labels=labels, n_clusters=2, centroids=np.zeros((2, 2)),
        silhouette_score=0.0, davies_bouldin_score=0.0,
        behavioral_weight=1.0, language_weight=0.0,
        user_features=features,
    )
    report = ClusterStabilityValidator.validate_locked_vectors(
        cr, n_iterations=3, train_sample_size=120,
        eval_sample_size=100, random_state=7,
    )
    assert report["protocol"] == "locked_k8_merge_bootstrap_v2"
    assert len(report["source_k8_ari_scores"]) == 3
    assert len(report["ari_scores"]) == 3


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
# Tier-1 cosine-distance filter regression tests
# (2026-07-17 refactor: replaced Bonferroni z-score filter)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tier1_cosine_rejects_far_text(monkeypatch):
    """Text whose embedding is far from the centroid (cosine distance > threshold)
    is rejected."""
    import src.enforcement.tier1_filter as tier1_module
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    from src.skill.schema import ExpressionDNA

    class StubEmbedder:
        def encode(self, *_args, **_kwargs):
            # Orthogonal to the centroid → cosine distance = 1.0.
            return np.array([0.0, 1.0])

        def get_sentence_embedding_dimension(self):
            return 2

    filter_ = Tier1ExpressionFilter(alpha=1.0)
    filter_._embedder = StubEmbedder()

    async def immediate_embed(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(tier1_module, "run_embed_in_executor", immediate_embed)

    edna = ExpressionDNA(
        embedding_centroid=[1.0, 0.0],
        embedding_cosine_threshold=0.1,  # tight boundary
    )
    result = await filter_.check_post_generation("sample", {"expression_dna": edna})
    assert result.passed is False
    assert "cosine distance" in result.reason


@pytest.mark.asyncio
async def test_tier1_cosine_passes_in_cluster_text(monkeypatch):
    """Text whose embedding is near the centroid passes."""
    import src.enforcement.tier1_filter as tier1_module
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    from src.skill.schema import ExpressionDNA

    class StubEmbedder:
        def encode(self, *_args, **_kwargs):
            return np.array([1.0, 0.0])

        def get_sentence_embedding_dimension(self):
            return 2

    filter_ = Tier1ExpressionFilter(alpha=1.0)
    filter_._embedder = StubEmbedder()

    async def immediate_embed(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(tier1_module, "run_embed_in_executor", immediate_embed)

    edna = ExpressionDNA(
        embedding_centroid=[1.0, 0.0],
        embedding_cosine_threshold=0.5,
    )
    result = await filter_.check_post_generation("sample", {"expression_dna": edna})
    assert result.passed is True


@pytest.mark.asyncio
async def test_tier1_safe_template_fallback(monkeypatch):
    """Tier 1 falls back to safe_template when retries exhausted."""
    from src.enforcement.tier1_filter import Tier1ExpressionFilter
    from src.skill.schema import ExpressionDNA

    edna = ExpressionDNA(
        embedding_centroid=[1.0, 0.0],
        embedding_cosine_threshold=0.001,
    )

    class StubEmbedder:
        def encode(self, *_args, **_kwargs):
            return np.array([0.0, 1.0])

    tier1 = Tier1ExpressionFilter(alpha=1.0, llm_client=None, max_retries=1)
    tier1._embedder = StubEmbedder()

    async def immediate_embed(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(
        "src.enforcement.tier1_filter.run_embed_in_executor", immediate_embed
    )

    final_text, result = await tier1.enforce_and_regenerate(
        text="some text",
        original_messages=[{"role": "user", "content": "hi"}],
        context={"expression_dna": edna},
        safe_template="SAFE_RESPONSE",
    )
    assert final_text == "SAFE_RESPONSE"
    assert result.passed is False


# ----------------------------------------------------------------------
# Tier-2 direct SBERT retrieval regression test
# (2026-07-17 refactor: dialogue-state classifier removed)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tier2_retrieval_returns_relevant():
    """Retriever returns the mind model whose doc matches the dialogue."""
    from src.enforcement.mind_model_retriever import MindModelRetriever
    from src.skill.schema import MindModel

    class StubEmbedder:
        def encode(self, texts, show_progress_bar=False):
            # Bag-of-words style 4-dim embedding based on keyword presence.
            single = isinstance(texts, str)
            inputs = [texts] if single else texts
            vecs = []
            for t in inputs:
                t_lower = t.lower()
                vecs.append(np.array([
                    float("vandal" in t_lower),
                    float("source" in t_lower),
                    float("welcome" in t_lower),
                    float("policy" in t_lower),
                ], dtype=float))
            return vecs[0] if single else vecs

    retriever = MindModelRetriever(top_k=1)
    retriever._embedder = StubEmbedder()

    models = [
        MindModel(name="patrol", description="revert vandalism", application="", limitation="", evidence=[]),
        MindModel(name="source", description="verify sources citations", application="", limitation="", evidence=[]),
        MindModel(name="welcome", description="welcome newcomers warmly", application="", limitation="", evidence=[]),
    ]
    messages = [{"role": "user", "content": "please check the source of this claim"}]
    selected = retriever.retrieve_for_messages(models, messages)
    assert len(selected) == 1
    assert selected[0].name == "source"


# ---------------------------------------------------------------------------
# Smoke test: CADPAgent construction + judge_meta attribute path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cadp_agent_cluster_id_accessible_for_judge_meta():
    """Self.state.cluster_id is where population builder stores the id.

    Regression: earlier judge-meta code used self.cluster_id, which is not
    set on BaseAgent (cluster_id lives in self.state). This smoke verifies
    the attribute path used by the two enforce_output call-sites in
    BaseAgent.take_turn.
    """
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock

    from src.agents.cadp import CADPAgent

    skill_path = Path("outputs/skills/skill_cluster_0_wikipedia_nuwa.yaml")
    if not skill_path.exists():
        pytest.skip("Skill corpus not found — run skill compilation first")
    from src.skill.compiler import SkillCompiler
    skill = SkillCompiler.load_skill(skill_path)

    mock_llm = MagicMock()
    mock_llm.models = {}
    mock_llm.model_name = "mock"

    agent = CADPAgent(
        agent_id="smoke_0",
        llm_client=mock_llm,
        model_name="mock",
        cluster_id="7",
        skill=skill,
        alpha=1.0,
        backend="base",
        seed=42,
        tier3_llm_judge_enabled=False,
    )

    assert agent.state.cluster_id == "7"
    assert agent.enforcement_harness is not None


# ---------------------------------------------------------------------------
# Smoke test: Tier2 retrieval is sync (executor-ready)
# ---------------------------------------------------------------------------

def test_tier2_retrieve_for_messages_is_sync():
    """MindModelRetriever.retrieve_for_messages must be a plain function.

    The caller (Tier2MindModelInjection.check_pre_generation) ships it
    to ``run_embed_in_executor``. If the method is ``async def``, the
    executor returns an un-awaitable coroutine, and iterating it in
    ``_format_mind_models`` raises ``TypeError: 'coroutine' object is
    not iterable``.
    """
    import inspect
    from src.enforcement.mind_model_retriever import MindModelRetriever
    assert not inspect.iscoroutinefunction(MindModelRetriever.retrieve_for_messages), (
        "retrieve_for_messages must be synchronous — "
        "Tier2 ships it to an executor thread"
    )

