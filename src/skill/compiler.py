"""CADP Skill Compiler — assembles dual-track .skill files.

Pipeline: Expression DNA → Mind Models → Anti-patterns → Assemble SkillFile
Includes quality self-check (adapted from nuwa-skill §6).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from src.clustering.clusterer import ClusterResult
from src.data.schemas import Message, Thread
from src.llm.client import LLMClient
from src.skill.anti_patterns import AntiPatternDetector
from src.skill.expression_dna import ExpressionDNAExtractor
from src.skill.mind_models import MindModelExtractor
from src.skill.schema import (
    AntiPattern,
    CapabilityTrack,
    ConstraintTrack,
    ExpressionDNA,
    MindModel,
    SkillFile,
)
from src.config.settings import settings


QUALITY_CHECK_PROMPT = """You are verifying the quality of a behavioral skill profile.

Given the following skill file and reference threads, check:

1. Do the mind models accurately reflect the behaviors in the threads?
2. Are the anti-patterns genuinely absent from the threads?
3. Is the expression DNA consistent with the language used?

Skill file:
{skill_yaml}

Reference threads (sample):
{threads_text}

Respond with JSON:
{{
  "mind_models_accurate": true/false,
  "anti_patterns_valid": true/false,
  "expression_dna_consistent": true/false,
  "overall_pass": true/false,
  "issues": ["list of any issues found"]
}}

Output ONLY the JSON."""


class SkillCompiler:
    """Compiles CADP dual-track .skill files from cluster corpus."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-4o",
        top_n_threads: int = 20,
    ):
        self.llm = llm_client
        self.model_name = model_name
        self.top_n_threads = top_n_threads
        self.dna_extractor = ExpressionDNAExtractor()
        self.mind_extractor = MindModelExtractor(llm_client, model_name)
        self.anti_detector = AntiPatternDetector(llm_client, model_name)

    async def compile_cluster(
        self,
        cluster_id: str,
        platform: str,
        cluster_threads: list[Thread],
        all_threads: list[Thread],
        cluster_result: ClusterResult,
        cluster_embeddings: np.ndarray | None = None,
    ) -> SkillFile:
        """Compile a single cluster's .skill file.

        Args:
            cluster_id: Cluster identifier.
            platform: Platform name.
            cluster_threads: All threads from this cluster.
            all_threads: All threads (for cross-cluster anti-pattern analysis).
            cluster_result: Full clustering result.
            cluster_embeddings: Pre-computed embeddings for messages in this cluster.

        Returns:
            Compiled SkillFile.
        """
        # Select representative threads
        rep_threads = self._select_representative(cluster_threads, cluster_result, cluster_id)

        # Get all messages from this cluster
        cluster_messages = [msg for t in cluster_threads for msg in t.messages]

        logger.info(f"Compiling skill for cluster {cluster_id}: "
                    f"{len(cluster_messages)} messages, {len(rep_threads)} rep threads")

        # Outline §4.3 mandates a dual-pass compilation:
        #   Pass 1 (positive cases) → Capability Track rules.
        #     - Expression DNA: statistical extraction over the cluster's
        #       full message distribution (the cluster's central tendency
        #       IS the positive exemplar by construction).
        #     - Mind Models: LLM extraction over the top-N representative
        #       (high-participation, multi-turn) threads — the canonical
        #       reasoning patterns this cluster exhibits.
        #   Pass 2 (negative cases) → Constraint Track (Anti-patterns).
        #     Source: low-fidelity threads (short / single-turn / low
        #     participation) PLUS cross-cluster gap evidence. These are
        #     the cases the cluster avoids / fails to produce, which is
        #     exactly what anti-patterns should be distilled from.
        # Prior implementation ran all three extractors over the same
        # ``rep_threads`` in parallel — collapsing the dual-pass structure
        # into a single pass and blurring the positive/negative source
        # distinction (G1).

        # ---- Pass 1: positive cases → capability ----
        # Step 1: Expression DNA (with cross-cluster taboo detection).
        # Uses the full cluster_messages so the embedding centroid /
        # statistical fingerprint is stable (cluster_embeddings aligns
        # with cluster_messages, not a positive-only subset).
        other_freq = self._compute_other_cluster_freq(all_threads, cluster_result, cluster_id)
        # Tier 1 (post-gen Expression DNA filter) hard-requires an embedding
        # centroid + per-dim std. When the caller did not supply pre-computed
        # embeddings, compute them now from cluster_messages using the shared
        # SentenceTransformer (same model Tier 1 uses at enforcement time —
        # outline §4.4.3 mandates the embedding space be identical between
        # distillation and enforcement so the 2σ boundary is meaningful).
        if cluster_embeddings is None and cluster_messages:
            cluster_embeddings = self._embed_messages(cluster_messages)
        expression_dna = self.dna_extractor.extract(
            cluster_messages, cluster_embeddings, other_cluster_word_freq=other_freq,
        )

        # Step 2: Mind Models (async LLM call) — from positive exemplars
        positive_threads = rep_threads
        other_summaries = self._build_other_cluster_summaries(cluster_id, cluster_result, all_threads)
        mind_models = await self.mind_extractor.extract(positive_threads, other_summaries)

        # ---- Pass 2: negative cases → anti-patterns ----
        negative_threads = self._select_negative_cases(
            cluster_threads, positive_threads,
        )
        other_dists = self._build_other_action_dists(cluster_id, all_threads, cluster_result)
        # Feed negative cases to the detector as the primary corpus; the
        # detector also consumes other-cluster action distributions for
        # cross-cluster gap analysis. When the cluster has too few
        # negative cases (small clusters), fall back to representative
        # threads so anti-pattern detection still runs.
        anti_source = negative_threads if negative_threads else positive_threads
        anti_patterns = await self.anti_detector.detect(anti_source, other_dists)

        # Step 4: Assemble
        skill = SkillFile(
            cluster_id=cluster_id,
            platform=platform,
            capability=CapabilityTrack(
                expression_dna=expression_dna,
                mind_models=mind_models,
            ),
            constraint=ConstraintTrack(anti_patterns=anti_patterns),
            source_thread_ids=[t.thread_id for t in rep_threads],
            source_user_count=len(cluster_result.get_cluster_members(int(cluster_id))),
        )

        # Step 5: Quality check
        quality = await self._quality_check(skill, rep_threads)
        if not quality.get("overall_pass", True):
            issues = quality.get("issues", [])
            logger.warning(f"Quality check issues for cluster {cluster_id}: {issues}")

        return skill

    async def compile_all(
        self,
        platform: str,
        cluster_result: ClusterResult,
        all_threads: list[Thread],
        thread_cluster_map: dict[str, int],
    ) -> dict[str, SkillFile]:
        """Compile .skill files for all clusters.

        Args:
            platform: Platform name.
            cluster_result: Clustering result.
            all_threads: All threads.
            thread_cluster_map: Mapping thread_id → cluster_id.

        Returns:
            Dict cluster_id → SkillFile.
        """
        skills = {}
        for cluster_id in cluster_result.get_cluster_ids():
            cluster_threads = [
                t for t in all_threads
                if thread_cluster_map.get(t.thread_id) == cluster_id
            ]

            if not cluster_threads:
                raise ValueError(
                    f"No threads mapped to cluster {cluster_id}; "
                    "skill compilation requires representative threads."
                )

            skill = await self.compile_cluster(
                cluster_id=str(cluster_id),
                platform=platform,
                cluster_threads=cluster_threads,
                all_threads=all_threads,
                cluster_result=cluster_result,
            )
            skills[str(cluster_id)] = skill

        return skills

    def save_skill(self, skill: SkillFile, output_dir: str | Path | None = None) -> Path:
        """Save skill file to disk."""
        out_dir = Path(output_dir) if output_dir else settings.skills_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"skill_cluster_{skill.cluster_id}_{skill.platform}.yaml"
        path = out_dir / filename
        with open(path, "w") as f:
            f.write(skill.to_yaml())
        logger.info(f"Saved skill file: {path}")
        return path

    @staticmethod
    def load_skill(path: str | Path) -> SkillFile:
        """Load a single .skill file from disk.

        Args:
            path: Path to .skill YAML file.

        Returns:
            Deserialized SkillFile object.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")
        with open(path) as f:
            yaml_str = f.read()
        skill = SkillFile.from_yaml(yaml_str)
        logger.info(f"Loaded skill file: {path} (cluster={skill.cluster_id})")
        return skill

    @staticmethod
    def load_all_skills(
        skills_dir: str | Path,
        platform: str | None = None,
    ) -> dict[str, SkillFile]:
        """Load all .skill files from a directory.

        Args:
            skills_dir: Directory containing .skill YAML files.
            platform: If set, only load files matching this platform.

        Returns:
            Dict mapping cluster_id → SkillFile.
        """
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            logger.warning(f"Skills directory not found: {skills_dir}")
            return {}

        pattern = f"*_{platform}.yaml" if platform else "*.yaml"
        skill_files = sorted(skills_dir.glob(pattern))

        skills = {}
        for sf in skill_files:
            try:
                skill = SkillCompiler.load_skill(sf)
                skills[skill.cluster_id] = skill
            except Exception as e:
                logger.error(f"Failed to load skill {sf}: {e}")

        logger.info(f"Loaded {len(skills)} skill files from {skills_dir}")
        return skills

    @staticmethod
    def skills_exist(
        skills_dir: str | Path,
        platform: str,
        cluster_ids: list[str],
    ) -> bool:
        """Check if skill files exist for all clusters of a platform.

        Args:
            skills_dir: Skills output directory.
            platform: Platform name (e.g. "wikipedia").
            cluster_ids: List of cluster IDs to check.

        Returns:
            True if all expected skill files exist.
        """
        skills_dir = Path(skills_dir)
        for cid in cluster_ids:
            path = skills_dir / f"skill_cluster_{cid}_{platform}.yaml"
            if not path.exists():
                return False
        return True

    def _embed_messages(self, messages: list[Message]) -> np.ndarray:
        """Embed cluster messages with the shared SentenceTransformer.

        Used to populate ExpressionDNA.embedding_centroid / embedding_std
        when the caller did not supply pre-computed embeddings. Empty /
        whitespace-only texts are encoded as the model sees them — the
        shared embedder handles empty strings without raising.
        """
        from src.config.settings import get_shared_embedder

        embedder = get_shared_embedder()
        texts = [m.text if m.text else " " for m in messages]
        if not texts:
            return np.zeros((0, embedder.get_sentence_embedding_dimension()))
        return np.asarray(embedder.encode(texts, show_progress_bar=False))

    def _compute_other_cluster_freq(
        self,
        all_threads: list[Thread],
        cluster_result: ClusterResult,
        cluster_id: str,
    ) -> Counter:
        """Compute word frequency distribution across all OTHER clusters.

        Used by ExpressionDNAExtractor to identify taboo words — words common
        in other clusters but rare/absent in this cluster.
        """
        import re as _re

        other_members = set()
        for other_id in cluster_result.get_cluster_ids():
            if str(other_id) == str(cluster_id):
                continue
            other_members.update(cluster_result.get_cluster_members(other_id))

        freq = Counter()
        for t in all_threads:
            for m in t.messages:
                if m.user_id in other_members:
                    tokens = _re.findall(r"\b\w+\b", m.text.lower())
                    freq.update(tok for tok in tokens if len(tok) > 2)
        return freq

    def _select_representative(
        self,
        cluster_threads: list[Thread],
        cluster_result: ClusterResult,
        cluster_id: str,
    ) -> list[Thread]:
        """Select top-N most representative threads for this cluster."""
        # Sort by number of messages and unique participants
        scored = sorted(
            cluster_threads,
            key=lambda t: (len(t.messages), len(t.participants)),
            reverse=True,
        )
        return scored[:self.top_n_threads]

    def _select_negative_cases(
        self,
        cluster_threads: list[Thread],
        positive_threads: list[Thread],
    ) -> list[Thread]:
        """Pass 2 source selection (outline §4.3 — negative case mining).

        "Negative cases" for anti-pattern distillation are threads that
        fail to exemplify this cluster's behavioural signature — short
        exchanges, single-turn drops, low-participation threads, and any
        threads not already selected as positive exemplars. These are the
        interactions where the cluster's typical constraints are absent
        or violated, which is what anti-pattern detection should be
        distilled from.

        Heuristics (deliberately conservative):
          - Exclude threads already in ``positive_threads``.
          - Keep threads with < 3 messages OR < 2 participants.
          - If too few match (small clusters), keep the lowest-scoring
            non-positive threads so Pass 2 always has *some* signal.
        """
        positive_ids = {t.thread_id for t in positive_threads}
        candidates = [t for t in cluster_threads if t.thread_id not in positive_ids]
        if not candidates:
            return []

        negative = [
            t for t in candidates
            if len(t.messages) < 3 or len(t.participants) < 2
        ]
        if len(negative) < 3:
            # Fall back: lowest-scoring non-positive threads, take up to
            # ``top_n_threads`` to give Pass 2 work to do.
            scored = sorted(
                candidates,
                key=lambda t: (len(t.messages), len(t.participants)),
            )
            negative = scored[: self.top_n_threads]
        return negative

    def _build_other_cluster_summaries(
        self,
        cluster_id: str,
        cluster_result: ClusterResult,
        all_threads: list[Thread],
    ) -> list[str]:
        """Build brief summaries of other clusters for exclusivity verification."""
        summaries = []
        for other_id in cluster_result.get_cluster_ids():
            if str(other_id) == str(cluster_id):
                continue
            members = cluster_result.get_cluster_members(other_id)
            member_threads = [t for t in all_threads if t.user_ids & set(members)]
            if member_threads:
                actions = Counter()
                for t in member_threads[:5]:
                    for m in t.messages:
                        actions[m.action_type.value] += 1
                top_actions = ", ".join(f"{a}({c})" for a, c in actions.most_common(3))
                summaries.append(f"Cluster {other_id}: {len(members)} users, common actions: {top_actions}")
        return summaries

    def _build_other_action_dists(
        self,
        cluster_id: str,
        all_threads: list[Thread],
        cluster_result: ClusterResult,
    ) -> dict[str, Counter]:
        """Build action distributions for other clusters."""
        dists = {}
        for other_id in cluster_result.get_cluster_ids():
            if str(other_id) == str(cluster_id):
                continue
            members = set(cluster_result.get_cluster_members(other_id))
            actions = Counter()
            for t in all_threads:
                if t.user_ids & members:
                    for m in t.messages:
                        if m.user_id in members:
                            actions[m.action_type.value] += 1
            dists[str(other_id)] = actions
        return dists

    async def _quality_check(self, skill: SkillFile, rep_threads: list[Thread]) -> dict:
        """Run quality self-check on compiled skill."""
        threads_text = "\n".join(
            f"[{m.user_id}]: {m.text[:100]}"
            for t in rep_threads[:3]
            for m in t.messages[:5]
        )[:3000]

        prompt = QUALITY_CHECK_PROMPT.format(
            skill_yaml=skill.to_yaml()[:3000],
            threads_text=threads_text,
        )

        messages = [
            {"role": "system", "content": "You are a quality assurance reviewer."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat_completion_json(
            messages,
            self.model_name,
            temperature=0.2,
            default={"overall_pass": True, "issues": ["Quality check parse failed"]},
        )
        if isinstance(response, dict):
            return response
        return {"overall_pass": True, "issues": ["Quality check parse failed"]}
