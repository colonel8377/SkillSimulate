"""Trigger Calibration Experiment (outline §5.3.5).

Independently evaluates anti-pattern trigger classifier performance
across three trigger categories (A lexical / B semantic / C behavioral).

Data format: JSONL at ``data/trigger_calibration/{dataset}.jsonl``, each line::

    {
        "interaction_id": "wiki_001",
        "text": "the message text",
        "action_history": ["edit", "revert", "discuss"],
        "gold_label": "violation",       // or "non-violation"
        "gold_category": "A",            // "A", "B", "C", or "none"
        "annotator_labels": ["violation", "violation", "non-violation"]
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.config.schemas import ExperimentConfig
from src.config.settings import settings
from src.enforcement.tier3_block import Tier3AntiPatternBlock
from src.experiment.runner import ExperimentRunner
from src.skill.schema import AntiPattern, SkillFile


@dataclass
class LabeledInteraction:
    """One annotated interaction for trigger calibration."""

    interaction_id: str
    text: str
    action_history: list[str] | None = None
    gold_label: str = "non-violation"  # "violation" or "non-violation"
    gold_category: str = "none"  # "A", "B", "C", or "none"
    annotator_labels: list[str] = field(default_factory=list)


@dataclass
class CategoryMetrics:
    """P/R/F1 for one trigger category."""

    category: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0


class TriggerCalibrationRunner(ExperimentRunner):
    """Runner for trigger classifier calibration (outline §5.3.5)."""

    THETA_GRID = [0.75, 0.80, 0.85, 0.90, 0.95]
    TRAIN_RATIO = 0.6
    # Outline §5.3.5 protocol floors — enforced at load so a present-but-
    # under-specified annotation file cannot silently degrade Fleiss' κ or
    # Category-C classifier training.
    MIN_INTERACTIONS = 500            # 500 interactions per dataset
    REQUIRED_NUM_ANNOTATORS = 3       # 3 independent annotators
    FLEISS_MIN_ANNOTATORS = 2         # below this Fleiss' κ is undefined

    def __init__(
        self,
        config: ExperimentConfig,
        models_config: str = "configs/models.yaml",
    ):
        super().__init__(config, models_config)
        self.data_dir = settings.project_root / "data" / "trigger_calibration"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_labeled_interactions(self, dataset: str) -> list[LabeledInteraction]:
        """Load labeled interactions from JSONL file."""
        path = self.data_dir / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Labeled interaction data not found: {path}\n"
                f"Annotation data must be collected before running calibration. "
                f"See outline §5.3.5 for annotation protocol (3 annotators, "
                f"500 interactions per dataset, Fleiss' κ ≥ 0.6)."
            )

        interactions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                interactions.append(LabeledInteraction(
                    interaction_id=obj["interaction_id"],
                    text=obj["text"],
                    action_history=obj.get("action_history"),
                    gold_label=obj.get("gold_label", "non-violation"),
                    gold_category=obj.get("gold_category", "none"),
                    annotator_labels=obj.get("annotator_labels", []),
                ))

        # --- Protocol validation (outline §5.3.5) ---
        # Previously the "500 interactions / 3 annotators" requirement lived
        # only in the FileNotFoundError message; a present-but-under-specified
        # file would silently degrade Fleiss' κ and classifier training.
        if len(interactions) < self.MIN_INTERACTIONS:
            raise ValueError(
                f"Trigger calibration data for {dataset!r} has "
                f"{len(interactions)} interactions < required "
                f"{self.MIN_INTERACTIONS} (outline §5.3.5). Fleiss' κ and "
                f"Category-C classifier training are unreliable below this. "
                f"Either complete annotation or lower MIN_INTERACTIONS for a pilot."
            )

        n_labels = [len(it.annotator_labels) for it in interactions]
        too_few = sum(1 for n in n_labels if n < self.FLEISS_MIN_ANNOTATORS)
        if too_few:
            raise ValueError(
                f"{too_few}/{len(interactions)} interactions in {dataset!r} have "
                f"<{self.FLEISS_MIN_ANNOTATORS} annotator labels — Fleiss' κ is "
                f"undefined. Re-annotate these items."
            )
        not_three = sum(1 for n in n_labels if n != self.REQUIRED_NUM_ANNOTATORS)
        if not_three:
            from collections import Counter
            dist = dict(Counter(n_labels))
            logger.warning(
                f"Trigger calibration {dataset!r}: {not_three}/{len(interactions)} "
                f"interactions do not have exactly {self.REQUIRED_NUM_ANNOTATORS} "
                f"annotator labels (distribution: {dist}). Outline §5.3.5 specifies "
                f"3 annotators; Fleiss' κ and P/R/F1 will still compute but may not "
                f"match the reported protocol."
            )

        return interactions

    def split_train_test(
        self,
        interactions: list[LabeledInteraction],
    ) -> tuple[list[LabeledInteraction], list[LabeledInteraction]]:
        """Stratified 60/40 split."""
        rng = random.Random(self.config.seed)
        shuffled = list(interactions)
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * self.TRAIN_RATIO)
        return shuffled[:n_train], shuffled[n_train:]

    # ------------------------------------------------------------------
    # Category C behavioral classifier training (outline §4.4.1 + §5.3.5)
    # ------------------------------------------------------------------

    def train_and_save_behavioral_classifier(
        self,
        train_interactions: list[LabeledInteraction],
        save_path: Path,
    ) -> "BehavioralTriggerClassifier | None":
        """Train the Category C logistic-regression classifier on the train split.

        Outline §4.4.1 specifies a "lightweight logistic regression over a
        behavioral feature vector (stance shift rate, conflict engagement
        ratio, etc.)" for Category C, and §5.3.5 mandates that it be trained
        on the 60% train split of the per-dataset annotated interactions.
        This method produces ``{dataset}.behavioral_clf.joblib``, which
        ``Tier3AntiPatternBlock.attach_behavioral_classifier`` consumes at
        runtime (lazily loaded by the agent harness).

        Label policy: ``gold_category == "C"`` OR (``gold_label ==
        "violation"`` AND ``action_history`` is non-empty) → positive (1).
        All other train rows are negatives (0). Training is silently
        skipped if fewer than 2 distinct labels are present in the train
        split — the Tier 3 block then falls back to legacy substring
        matching, which preserves back-compat with untrained deployments.

        Args:
            train_interactions: 60% train split returned by split_train_test.
            save_path: Destination ``.joblib`` path. Parent dirs are created.

        Returns:
            Trained ``BehavioralTriggerClassifier`` if training succeeded,
            else ``None`` (caller falls back to legacy Category C path).
        """
        from src.enforcement.behavioral_trigger_classifier import (
            BehavioralTriggerClassifier,
        )

        # Filter to rows with usable action history
        usable = [
            it for it in train_interactions
            if it.action_history and len(it.action_history) > 0
        ]
        if len(usable) < 2:
            logger.warning(
                f"BehavioralTriggerClassifier training skipped — only "
                f"{len(usable)} rows with action_history (need ≥2). "
                f"Tier 3 Category C will fall back to legacy substring path."
            )
            return None

        # Build labels per the spec (Category C OR violation+history)
        labels = [
            1 if (it.gold_category == "C")
            or (it.gold_label == "violation" and it.action_history)
            else 0
            for it in usable
        ]
        if len(set(labels)) < 2:
            logger.warning(
                f"BehavioralTriggerClassifier training skipped — only one "
                f"class present in train split ({set(labels)}). "
                f"Tier 3 Category C will fall back to legacy substring path."
            )
            return None

        histories = [it.action_history for it in usable]

        clf = BehavioralTriggerClassifier()
        try:
            clf.fit_from_histories(histories, labels)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"BehavioralTriggerClassifier training failed: {exc} — "
                f"Tier 3 Category C will fall back to legacy substring path."
            )
            return None

        save_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save(save_path)
        n_pos = sum(1 for x in labels if x == 1)
        logger.info(
            f"BehavioralTriggerClassifier trained on {len(usable)} rows "
            f"({n_pos} positives) → saved to {save_path}"
        )
        return clf

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _detect(
        self,
        tier3: Tier3AntiPatternBlock,
        text: str,
        anti_patterns: list[AntiPattern],
        action_history: list[str] | None,
    ) -> list[dict[str, str]]:
        """Run detection bypassing α-gating (deterministic for calibration)."""
        # Temporarily set alpha to 1.0 for deterministic evaluation
        original_alpha = tier3.alpha
        tier3.alpha = 1.0
        try:
            return tier3._detect_violations(text, anti_patterns, action_history)
        finally:
            tier3.alpha = original_alpha

    def _classify_trigger(self, triggered_by: str) -> str:
        """Infer trigger category from the triggered_by string."""
        tb = triggered_by.lower()
        if "lexical" in tb:
            return "A"
        if "semantic" in tb:
            return "B"
        if "behavioral" in tb or "action" in tb:
            return "C"
        return "A"  # default to lexical for keyword/regex matches

    def evaluate_per_category(
        self,
        test_set: list[LabeledInteraction],
        anti_patterns: list[AntiPattern],
        tier3: Tier3AntiPatternBlock,
    ) -> dict[str, CategoryMetrics]:
        """Evaluate P/R/F1 per trigger category and overall."""
        categories = ["A", "B", "C", "overall"]
        stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in categories}

        for item in test_set:
            violations = self._detect(
                tier3, item.text, anti_patterns, item.action_history
            )
            predicted_violation = len(violations) > 0
            predicted_cats = {self._classify_trigger(v["triggered_by"]) for v in violations}

            # Overall binary classification
            if item.gold_label == "violation":
                if predicted_violation:
                    stats["overall"]["tp"] += 1
                else:
                    stats["overall"]["fn"] += 1
            else:
                if predicted_violation:
                    stats["overall"]["fp"] += 1

            # Per-category evaluation
            for cat in ["A", "B", "C"]:
                gold_cat = item.gold_category == cat
                pred_cat = cat in predicted_cats
                if gold_cat and pred_cat:
                    stats[cat]["tp"] += 1
                elif (not gold_cat) and pred_cat:
                    stats[cat]["fp"] += 1
                elif gold_cat and (not pred_cat):
                    stats[cat]["fn"] += 1

        results = {}
        for cat in categories:
            s = stats[cat]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results[cat] = CategoryMetrics(
                category=cat, precision=precision, recall=recall,
                f1=f1, tp=tp, fp=fp, fn=fn,
            )

        return results

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------

    def threshold_sweep(
        self,
        test_set: list[LabeledInteraction],
        anti_patterns: list[AntiPattern],
        tier3: Tier3AntiPatternBlock,
    ) -> list[dict[str, Any]]:
        """Sweep θ_sem over {0.75, 0.80, 0.85, 0.90, 0.95} and report Category B metrics."""
        results = []
        for theta in self.THETA_GRID:
            # Create copies of anti-patterns with modified threshold
            adjusted = [
                AntiPattern(
                    description=ap.description,
                    trigger_conditions=ap.trigger_conditions,
                    trigger_regex=ap.trigger_regex,
                    trigger_keywords=ap.trigger_keywords,
                    trigger_semantic_phrases=ap.trigger_semantic_phrases,
                    trigger_semantic_threshold=theta,
                    trigger_action_patterns=ap.trigger_action_patterns,
                    reason=ap.reason,
                )
                for ap in anti_patterns
            ]
            cat_metrics = self.evaluate_per_category(test_set, adjusted, tier3)
            results.append({
                "theta": theta,
                "category_B": cat_metrics["B"],
                "overall": cat_metrics["overall"],
            })
        return results

    # ------------------------------------------------------------------
    # Cross-dataset transfer
    # ------------------------------------------------------------------

    def cross_dataset_transfer(
        self,
        source_anti_patterns: list[AntiPattern],
        target_test: list[LabeledInteraction],
        tier3: Tier3AntiPatternBlock,
    ) -> dict[str, CategoryMetrics]:
        """Evaluate source-compiled anti-patterns on target dataset."""
        return self.evaluate_per_category(target_test, source_anti_patterns, tier3)

    # ------------------------------------------------------------------
    # Fleiss' κ
    # ------------------------------------------------------------------

    def compute_fleiss_kappa(
        self,
        interactions: list[LabeledInteraction],
    ) -> float:
        """Compute Fleiss' κ from annotator labels.

        Args:
            interactions: Interactions with annotator_labels populated.

        Returns:
            Fleiss' κ value.
        """
        # Build annotator matrix: rows = items, cols = categories
        categories = sorted({lbl for item in interactions for lbl in item.annotator_labels})
        if len(categories) < 2:
            return 1.0  # perfect agreement if only one category

        cat_idx = {c: i for i, c in enumerate(categories)}
        n_items = len(interactions)
        n_raters = max(len(item.annotator_labels) for item in interactions) if interactions else 0

        if n_items == 0 or n_raters == 0:
            return 0.0

        matrix = np.zeros((n_items, len(categories)))
        for i, item in enumerate(interactions):
            for label in item.annotator_labels:
                matrix[i, cat_idx[label]] += 1

        # P_i: agreement per item
        p_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        p_bar = np.mean(p_i)

        # P_j: marginal proportions per category
        p_j = np.sum(matrix, axis=0) / (n_items * n_raters)
        p_e = np.sum(p_j ** 2)

        if p_e == 1.0:
            return 1.0

        return float((p_bar - p_e) / (1.0 - p_e))

    # ------------------------------------------------------------------
    # Anti-pattern loading
    # ------------------------------------------------------------------

    def _load_anti_patterns(self, dataset: str) -> list[AntiPattern]:
        """Load compiled anti-patterns from skill files for a dataset."""
        from src.skill.compiler import SkillCompiler

        skills_dir = settings.skills_dir
        skills = SkillCompiler.load_all_skills(str(skills_dir), platform=dataset)
        if not skills:
            raise FileNotFoundError(
                f"No compiled skill files found in {skills_dir} for platform={dataset}. "
                f"Run `python -m src.main compile-skills --config configs/dev.yaml` first."
            )

        all_aps = []
        for skill in skills.values():
            if skill.constraint:
                all_aps.extend(skill.constraint.anti_patterns)
        return all_aps

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, dataset: str) -> dict[str, Any]:
        """Run full trigger calibration for one dataset.

        Returns dict with keys: fleiss_kappa, per_category, threshold_sweep.
        """
        logger.info(f"Trigger calibration for dataset: {dataset}")

        interactions = self.load_labeled_interactions(dataset)
        logger.info(f"Loaded {len(interactions)} labeled interactions")

        # Fleiss' κ
        kappa = self.compute_fleiss_kappa(interactions)
        if kappa < 0.6:
            logger.warning(
                f"Fleiss' κ = {kappa:.3f} < 0.6 threshold. "
                f"Annotation quality insufficient — results may be unreliable."
            )
        else:
            logger.info(f"Fleiss' κ = {kappa:.3f} (≥ 0.6 ✓)")

        # Split
        train, test = self.split_train_test(interactions)
        logger.info(f"Train: {len(train)}, Test: {len(test)}")

        # Load anti-patterns
        anti_patterns = self._load_anti_patterns(dataset)
        logger.info(f"Loaded {len(anti_patterns)} anti-patterns")

        # Create Tier3 block at α=1.0 for deterministic evaluation
        tier3 = Tier3AntiPatternBlock(alpha=1.0)

        # Train Category C behavioral classifier on the 60% train split
        # (outline §4.4.1 + §5.3.5). Attach to tier3 so per-category eval
        # measures the trained classifier instead of the legacy substring
        # fallback. Persisted to {dataset}.behavioral_clf.joblib for
        # lazy runtime loading by the agent harness.
        clf_save_path = self.data_dir / f"{dataset}.behavioral_clf.joblib"
        behavioral_clf = self.train_and_save_behavioral_classifier(
            train, clf_save_path
        )
        if behavioral_clf is not None:
            tier3.attach_behavioral_classifier(behavioral_clf)

        # Per-category evaluation
        per_category = self.evaluate_per_category(test, anti_patterns, tier3)
        for cat, metrics in per_category.items():
            logger.info(
                f"  Category {cat}: P={metrics.precision:.3f} "
                f"R={metrics.recall:.3f} F1={metrics.f1:.3f}"
            )

        # Threshold sweep
        sweep = self.threshold_sweep(test, anti_patterns, tier3)
        for result in sweep:
            cat_b = result["category_B"]
            logger.info(
                f"  θ={result['theta']:.2f}: Cat-B "
                f"P={cat_b.precision:.3f} R={cat_b.recall:.3f} F1={cat_b.f1:.3f}"
            )

        # Check pass criteria (Precision ≥ 0.90, Recall ≥ 0.80)
        overall = per_category["overall"]
        passed = overall.precision >= 0.90 and overall.recall >= 0.80
        logger.info(
            f"  Overall: P={overall.precision:.3f} R={overall.recall:.3f} "
            f"→ {'PASS' if passed else 'FAIL'} "
            f"(criteria: P≥0.90, R≥0.80)"
        )

        return {
            "dataset": dataset,
            "n_interactions": len(interactions),
            "n_train": len(train),
            "n_test": len(test),
            "fleiss_kappa": kappa,
            "per_category": {
                cat: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                }
                for cat, m in per_category.items()
            },
            "threshold_sweep": [
                {
                    "theta": r["theta"],
                    "category_B": {
                        "precision": r["category_B"].precision,
                        "recall": r["category_B"].recall,
                        "f1": r["category_B"].f1,
                    },
                    "overall": {
                        "precision": r["overall"].precision,
                        "recall": r["overall"].recall,
                        "f1": r["overall"].f1,
                    },
                }
                for r in sweep
            ],
            "pass_criteria": passed,
        }

    async def run_cross_dataset(
        self,
        source: str,
        target: str,
    ) -> dict[str, Any]:
        """Run cross-dataset transfer: source anti-patterns → target test set.

        Reports P/R/F1 drop compared to native calibration.
        """
        logger.info(f"Cross-dataset transfer: {source} → {target}")

        source_aps = self._load_anti_patterns(source)
        target_interactions = self.load_labeled_interactions(target)
        _, target_test = self.split_train_test(target_interactions)

        tier3 = Tier3AntiPatternBlock(alpha=1.0)

        transfer_metrics = self.cross_dataset_transfer(source_aps, target_test, tier3)

        # Native baseline
        target_aps = self._load_anti_patterns(target)
        native_metrics = self.evaluate_per_category(target_test, target_aps, tier3)

        logger.info(f"  Transfer overall F1: {transfer_metrics['overall'].f1:.3f}")
        logger.info(f"  Native overall F1:  {native_metrics['overall'].f1:.3f}")

        return {
            "source": source,
            "target": target,
            "transfer": {
                cat: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                }
                for cat, m in transfer_metrics.items()
            },
            "native": {
                cat: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                }
                for cat, m in native_metrics.items()
            },
            "f1_drop": native_metrics["overall"].f1 - transfer_metrics["overall"].f1,
        }
