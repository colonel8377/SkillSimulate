"""Experiment configuration schemas."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DatasetName(str, Enum):
    WIKIPEDIA = "wikipedia"
    REDDIT = "reddit"
    GITHUB = "github"


class ConditionName(str, Enum):
    VANILLA = "vanilla"
    DESCRIPTIVE = "descriptive"
    SEGMENTATION = "segmentation"
    POP_ALIGNED = "pop_aligned"
    COLLEAGUE_SKILL = "colleague_skill"
    CLUSTERING_ONLY = "clustering_only"
    CADP_FULL = "cadp_full"
    CADP_SHUFFLED = "cadp_shuffled"
    CADP_MINUS_EDNA = "cadp_minus_edna"
    CADP_MINUS_MM = "cadp_minus_mm"
    CADP_MINUS_AP = "cadp_minus_ap"
    CADP_CONSTRAINT_ONLY = "cadp_constraint_only"
    POP_ALIGNED_CADP = "pop_aligned_cadp"
    # Length-matched control (ARS review 2026-06-19, DA-E1 anti-circularity
    # defense): token-budget-matched behavioral description built from a
    # RANDOM OTHER cluster's stats. Same template and token count as
    # Descriptive Persona; broken semantic correspondence. Tests whether
    # CADP's fidelity gain comes from rule-level content vs simply "more
    # behavioral context tokens". NOT a CADP variant — does not compile
    # or mount .skill rules.
    LENGTH_MATCHED_CONTROL = "length_matched_control"
    # Exp2 §6.2 fourth arm: replay the observed real traces through the
    # metric pipeline as both sim and ground truth, yielding a self-
    # similarity ceiling / "perfect fidelity" reference for the other
    # three Exp2 conditions. Not a CADP variant — does not compile skills.
    REAL_HISTORY = "real_history"


@dataclass
class ModelConfig:
    name: str           # display name, e.g. "gpt-4o"
    endpoint: str       # model id for API call
    base_url: str = ""  # override base URL if needed
    api_key_env: str = "CADP_OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    cost_per_1k_input: float = 0.0   # USD
    cost_per_1k_output: float = 0.0  # USD


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    num_repeats: int = 1
    num_rounds: int = 30
    population_size: int = 30
    alpha: float = 1.0           # constraint hardness (global default)
    alpha_tier1: float | None = None  # per-tier override (None = use alpha)
    alpha_tier2: float | None = None
    alpha_tier3: float | None = None
    num_clusters: int = 4        # K for clustering
    cluster_method: str = "kmeans"  # "kmeans" or "hdbscan"
    top_n_threads: int = 20      # representative threads for skill compilation
    max_threads: int | None = None  # cap on threads loaded per dataset (None = no cap)
    checkpoint_every: int = 5    # save state every N rounds
    max_concurrency: int = 4     # parallel LLM calls
    seed: int = 42
    backend: str = "base"
    # Exp 2 scale test
    scale_test: bool = False
    scale_test_sizes: list[int] = field(default_factory=lambda: [30, 100])
    # Transfer test mode (outline §5.5)
    transfer_mode: str = "full_component"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
