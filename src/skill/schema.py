"""CADP dual-track .skill file schema.

Capability Track: Expression DNA + Mind Models (what agent CAN do)
Constraint Track: Anti-patterns (what agent CANNOT do)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExpressionDNA:
    """Quantified language patterns for a cluster.

    Adapted from nuwa-skill extraction-framework.md §2.
    """
    # Sentence-pattern fingerprint
    avg_sentence_length: float = 0.0
    question_ratio: float = 0.0
    analogy_density: float = 0.0       # analogies per 1000 chars
    first_person_rate: float = 0.0
    certainty_ratio: float = 0.0       # "certain" vs "maybe" language
    transition_frequency: float = 0.0  # "but", "however" per 1000 chars

    # Style tags (each on a 0-1 scale between two poles)
    style_formal_casual: float = 0.5       # 0=formal, 1=casual
    style_abstract_concrete: float = 0.5   # 0=abstract, 1=concrete
    style_cautious_assertive: float = 0.5  # 0=cautious, 1=assertive
    style_academic_plain: float = 0.5      # 0=academic, 1=plain
    style_long_short: float = 0.5          # 0=long sentences, 1=short
    style_preamble_conclusion_first: float = 0.5  # 0=preamble, 1=conclusion first
    style_data_narrative: float = 0.5      # 0=data-driven, 1=narrative-driven

    # Vocabulary
    high_freq_words: list[str] = field(default_factory=list)
    taboo_words: list[str] = field(default_factory=list)
    vocab_richness: float = 0.0  # type-token ratio

    # Embedding centroid for Tier-1 filter
    embedding_centroid: list[float] | None = None
    embedding_std: list[float] | None = None


@dataclass
class MindModel:
    """A triple-verified reasoning template.

    Adapted from nuwa-skill extraction-framework.md §1.
    """
    name: str
    description: str                    # one-line description
    evidence: list[str]                 # supporting evidence from 2+ threads
    application: str                    # when to apply this lens
    limitation: str                     # when this model fails

    # Triple verification status
    cross_domain_verified: bool = False     # appears in 2+ different threads
    predictive_verified: bool = False        # can predict stance on new issues
    exclusive_verified: bool = False         # not something all clusters do


@dataclass
class AntiPattern:
    """A prohibited behavior pattern with trigger conditions."""
    description: str                    # what is prohibited
    trigger_conditions: list[str]       # when this pattern would activate
    trigger_regex: list[str] = field(default_factory=list)      # Category A: regex patterns
    trigger_keywords: list[str] = field(default_factory=list)   # Category A: keyword triggers
    # Category B: semantic-level triggers (Sentence-BERT cosine similarity)
    trigger_semantic_phrases: list[str] = field(default_factory=list)  # reference phrases
    trigger_semantic_threshold: float = 0.85  # θ_sem (calibrated on held-out data)
    # Category C: behavioral-level triggers (action pattern matching)
    trigger_action_patterns: list[str] = field(default_factory=list)  # e.g. "disagree->revert->report"
    # Per-pattern Category-C classifier threshold (outline §4.4.1). When the
    # behavioral trigger classifier is attached, this overrides the global
    # ``Tier3AntiPatternBlock.behavioral_threshold`` so each anti-pattern
    # can carry its own calibrated decision boundary. ``None`` = fall back
    # to the global default (G7).
    trigger_behavioral_threshold: float | None = None
    reason: str = ""                    # why this is prohibited


@dataclass
class CapabilityTrack:
    """What the agent CAN do."""
    expression_dna: ExpressionDNA
    mind_models: list[MindModel]


@dataclass
class ConstraintTrack:
    """What the agent CANNOT do."""
    anti_patterns: list[AntiPattern]


@dataclass
class SkillFile:
    """Complete CADP dual-track skill file."""
    # Metadata
    cluster_id: str
    platform: str
    compiled_at: str = ""

    # Dual-track content
    capability: CapabilityTrack | None = None
    constraint: ConstraintTrack | None = None

    # Provenance
    source_thread_ids: list[str] = field(default_factory=list)
    source_user_count: int = 0

    def __post_init__(self):
        if not self.compiled_at:
            self.compiled_at = datetime.now().isoformat()

    def to_yaml(self) -> str:
        """Serialize to YAML format for storage."""
        import yaml

        def _serialize(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            elif isinstance(obj, list):
                return [_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj

        data = _serialize(self)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SkillFile":
        """Deserialize from YAML."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SkillFile":
        """Reconstruct from dict, handling nested dataclasses."""
        edna_data = data["capability"]["expression_dna"]
        expression_dna = ExpressionDNA(**edna_data)

        mm_list = data["capability"]["mind_models"]
        mind_models = [MindModel(**mm) for mm in mm_list]

        ap_list = data.get("constraint", {}).get("anti_patterns", [])
        anti_patterns = [AntiPattern(**ap) for ap in ap_list]

        capability = CapabilityTrack(
            expression_dna=expression_dna,
            mind_models=mind_models,
        )
        constraint = ConstraintTrack(anti_patterns=anti_patterns)

        return cls(
            cluster_id=data["cluster_id"],
            platform=data["platform"],
            compiled_at=data["compiled_at"],
            capability=capability,
            constraint=constraint,
            source_thread_ids=data["source_thread_ids"],
            source_user_count=data["source_user_count"],
        )
