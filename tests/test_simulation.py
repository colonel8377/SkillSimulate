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
    assert len(vec) == 4


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
