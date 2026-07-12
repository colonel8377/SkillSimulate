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
    # Lever-1 ceiling / kill-condition (reframe v1, 2026-07-08): the
    # richest possible persona description built from the same cluster
    # stats — long narrative form, all available descriptors — but NO
    # compiled .skill rules, NO filter-retry. Tests whether distilled
    # behavioral skills beat the strongest lever-1 description. If CADP
    # loses to this on Predictive Fidelity, the headline thesis fails.
    RICH_NARRATIVE = "rich_narrative"
    COLLEAGUE_SKILL = "colleague_skill"
    CLUSTERING_ONLY = "clustering_only"
    CADP_FULL = "cadp_full"
    # Distiller-comparison arms (plan: "打通蒸馏产出与 CADP 实验", 2026-07-08):
    # same CADPAgent runtime as cadp_full, but skills come from manually
    # distilled colleague-skill / nuwa-skill Markdown (converted via
    # scripts/convert_distilled_skills.py) instead of pipeline A's LLM
    # re-extraction. cadp_full itself is retired (no bare skill file is
    # written for either distiller) — see is_cadp_condition /
    # _get_or_compile_skills for the fail-fast behavior this implies.
    CADP_FULL_COLLEAGUE = "cadp_full_colleague"
    CADP_FULL_NUWA = "cadp_full_nuwa"
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
    compile_model: str = ""  # model for skill compilation (empty = use models[0])
    conditions: list[str] = field(default_factory=list)
    num_repeats: int = 1
    num_rounds: int = 30
    population_size: int = 30
    alpha: float = 1.0           # constraint hardness (global default)
    alpha_tier1: float | None = None  # per-tier override (None = use alpha)
    alpha_tier2: float | None = None
    alpha_tier3: float | None = None
    num_clusters: int = 4        # K for clustering
    cluster_method: str = "kmeans"  # "kmeans" or "hdbscan" or "two_stage"
    # Locked-clustering artifact loading (reframe v1, 2026-07-08). When set
    # and file exists, the runner loads the canonical precomputed
    # ClusterResult pickle instead of re-fitting live. Cluster IDs then
    # match the manually-distilled skill files (otherwise CADP conditions
    # silently load wrong skills). Pickle is the locked K=8 source
    # clustering; merge_map (in cluster_merge_map_path JSON) collapses
    # leaves {1→0, 5→4} to the 6 final skill archetypes.
    clustering_pickle_path: str = ""
    cluster_merge_map_path: str = ""
    # Two-stage HDBSCAN tuning (NEW)
    role_min_cluster_size: int | None = None
    role_min_samples: int | None = None
    style_min_cluster_size: int | None = None
    style_min_samples: int | None = None
    target_min_leaves: int = 30
    target_max_leaves: int = 80
    scaler: str = "robust"       # "standard" | "robust"
    impute_orphans: bool = False
    cluster_selection_method: str = "eom"  # "eom" | "leaf"
    min_style_silhouette: float = 0.10
    style_umap_dim: int = 15
    top_n_threads: int = 20      # representative threads for skill compilation
    max_threads: int | None = None  # cap on threads loaded per dataset (None = no cap)
    thread_filter_oversample: int = 100  # inflate utterance cap before filtering single-msg threads
    max_sim_threads: int = 5     # cap on threads used per simulation cell
    checkpoint_every: int = 5    # save state every N rounds
    max_concurrency: int = 4     # parallel LLM calls
    max_context_items: int = 5       # memory items shown per turn (to planner)
    per_msg_token_ratio: int = 10    # per-msg budget = max_memory_tokens // ratio
    per_msg_token_floor: int = 60    # per-msg token floor (legacy fallback)
    max_thread_messages: int = 5     # recent thread messages shown in prompt
    reflection_interval: int = 10    # periodic reflection every N rounds
    seed_min_toxicity: float = 0.6  # min max-toxicity for CGA seed threads
    seed: int = 42
    backend: str = "base"
    # Exp 2 scale test
    scale_test: bool = False
    scale_test_sizes: list[int] = field(default_factory=lambda: [30, 100])
    # Transfer test mode (outline §5.5)
    transfer_mode: str = "full_component"
    # --- Memory strategy (Issue 1: token budget) ----------------------
    # "sliding"        — token-aware top-K retrieval (default; exp1/exp2).
    #                   Bounded by max_input_tokens; oldest/lowest-importance
    #                   items dropped when budget exceeded.
    # "rolling_summary" — every `compaction_interval` turns, summarize the
    #                   oldest N raw messages into a single MemoryItem and
    #                   keep recent M turns raw. Used by the R4 collapse
    #                   stress test where long-horizon signal matters.
    memory_strategy: str = "sliding"
    compaction_interval: int = 5  # only used when memory_strategy="rolling_summary"
    compaction_keep_recent: int = 10  # M raw items kept after compaction

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml
        import dataclasses
        with open(path) as f:
            data = yaml.safe_load(f)
        # Filter to known fields so adding a new dataclass field doesn't
        # require every config file to be updated simultaneously. Unknown
        # keys would otherwise raise TypeError from cls(**data). Note:
        # this *does* silently drop typo'd keys — but every field has a
        # sensible default, so a typo at worst means the default applies.
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        dropped = set(data) - known
        if dropped:
            from loguru import logger
            logger.warning(
                f"ExperimentConfig.from_yaml ignored unknown keys: {sorted(dropped)}"
            )
        return cls(**filtered)
