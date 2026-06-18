# Release Checklist — Digital Habitus Distillation (CADP)

Everything the pipeline and the paper's claims assume must exist at submission.
Each row is grounded in an actual code path — paths, formats, and thresholds are
quoted from `src/`, not invented. Items marked **GAP** do not exist yet and will
silently degrade a metric if absent.

> Convention: `{dataset} ∈ {wikipedia, reddit, github}`. All paths are relative
> to project root. `data_dir = project_root/data`,
> `skills_dir = project_root/outputs/skills` (`src/config/settings.py:15-22`).

---

## 0. Severity key

- **BLOCKER** — without it, a headline metric is invalid or silently faked.
- **MAJOR** — paper claims a protocol (κ threshold, annotator count) that the
  artifact must satisfy.
- **WIRING** — code supports it but no default path / config key wires it in;
  must be plumbed before runs.

---

## 1. Input data artifacts (pipeline reads these)

| # | Artifact | Path | Format | Producer | Threshold / spec | Severity if missing |
|---|----------|------|--------|----------|------------------|---------------------|
| 1.1 | Raw interaction logs | `data/raw/{dataset}/*.jsonl` | Wikipedia/GitHub: any `*.jsonl`; Reddit: `*submissions*.jsonl` + `*comments*.jsonl` (`src/data/{wikipedia,reddit,github}.py`) | Platform API dump, PII-scrubbed via `src/data/pii.py` | CC-BY-SA / ToS research use | BLOCKER — no simulation possible |
| 1.2 | **External role labels** | `data/role_labels/{dataset}.jsonl` | JSONL, `{user_id: role_id}`; loaded by `aggregator._load_role_labels` (`aggregator.py:92-124`) | Manual / Wu et al. 2025 segmentation labels | ≥1 external scheme; Louvain proxy is fallback only | **GAP / BLOCKER** — dir does not exist → every dataset falls to Louvain proxy (`aggregator.py:159-170`). Outline §7.4 under-states this: proxy is currently the *universal* path, not a rare fallback. |
| 1.3 | **Held-out event annotations** | `data/held_out_events/{dataset}.jsonl` | `HeldOutEvent` records, 2 annotators; loaded by `aggregator._load_held_out_events` (`aggregator.py:77-90`); CLI `main.py annotate-held-out-events` | 2 independent annotators → consensus | **Cohen's κ ≥ 0.7** (`held_out_events.py:46`) | **GAP / BLOCKER** — absent → `predictive.py:414-421` silently degrades to `_heuristic_ground_truth` (κ unavailable). This is the "so what" headline metric (§5.3). |
| 1.4 | **Trigger-calibration annotations** | `data/trigger_calibration/{dataset}.jsonl` | JSONL, 1 line per interaction, 3 annotator cols (`trigger_calibration.py:6, 81`) | 3 independent annotators | **500 interactions/dataset, Fleiss' κ ≥ 0.6** (`trigger_calibration.py:83-88`) | **GAP / MAJOR** — absent → Category C behavioral classifier never trains → Tier 3 falls back to legacy substring matching (`behavioral_trigger_classifier.py:8-10, 196-215`). Note: only asserted in error message, not validated at load (open issue). |
| 1.5 | Trained Category-C classifier | `data/trigger_calibration/{dataset}.behavioral_clf.joblib` | joblib'd `BehavioralTriggerClassifier` (model + scaler + feature means) | Produced by `trigger_calibration.py:444` from 1.4 | Pass bar **Precision ≥ 0.90, Recall ≥ 0.80** (`trigger_calibration.py:469-475`) | Generated from 1.4; ship the artifact so Tier 3 works without re-training |

### Annotation protocol consistency (fix before annotating)

| Metric | Annotators | κ floor | Source |
|--------|-----------|---------|--------|
| Held-out events (Predictive Fidelity) | **2** | Cohen **0.7** | `held_out_events.py:46` |
| Trigger calibration (Category C) | **3** | Fleiss **0.6** | `trigger_calibration.py:83-88` |
| Human eval (§5.6) | ≥2 (generic pairwise) | mean Cohen **0.6** | `human_eval.py:355` |

→ Harmonize or justify: the same *kind* of coding (interaction-outcome labeling) uses
2 annotators in 1.3 but 3 in 1.4. Outline should state why, or align them.

---

## 2. Generated artifacts (pipeline produces; ship these)

| # | Artifact | Path | How produced | Release treatment |
|---|----------|------|--------------|-------------------|
| 2.1 | Compiled `.skill` files | `outputs/skills/{dataset}/*.{cluster}.yaml` | `main.py compile-skills` / `SkillCompiler.compile_all` | Release **aggregate/cluster-level only**; **NOT** individual-level (dual-use, §7.5). Anonymize. |
| 2.2 | Simulation runs | `outputs/simulations/` | `main.py run --type {exp1,exp2,transfer,trigger_calibration,alpha_sensitivity}` | Ship seeds + configs used |
| 2.3 | Aggregated results | `outputs/results/` | `MetricsAggregator` | Ship tables/figures source data |

---

## 3. Code & model release (P1-5 reproducibility statement, §5.1 / §7.5)

- [ ] **Pipeline source** — full repo commit pinned (including `src/skill/compiler.py` dual-track + `src/enforcement/` three-tier).
- [ ] **Anonymized aggregate `.skill`** (2.1) — not individual-level.
- [ ] **All random seeds** — config `seed: 42` default; document any per-run overrides.
- [ ] **API model snapshots** — record exact `gpt-4o` and `claude-3.5-sonnet` version + API call dates (model behavior drifts).
- [ ] **Open-source model commits** — `Llama-3-70B` and `Qwen-2.5-72B` commit hashes + inference stack (vLLM version, GPU type).
- [ ] **Annotation protocols + raw annotation data** — items 1.2 / 1.3 / 1.4 with codebooks.
- [ ] **SentenceTransformer** — `all-MiniLM-L6-v2` pinned (`settings.py:25`); used by Tier 1, Tier 2 retriever, Linguistics SIP, Category B trigger.
- [ ] **Environment** — `environment.yml` frozen; record `scikit-learn` / `sentence-transformers` versions (affect classifier + embeddings).

---

## 4. Wiring TODOs (code supports but config doesn't expose)

- [x] **WIRING — `role_labels_dir` has no default.** ✅ RESOLVED: added `role_labels_dir: Path = data_dir / "role_labels"` to `settings.py` (alongside `held_out_events_dir`) and threaded `role_labels_dir=str(settings.role_labels_dir)` through all three `MetricsAggregator` constructions (`exp1_validation.py`, `exp2_simulation.py`, `transfer_test.py`). Aggregator handles a missing file gracefully (`aggregator.py:103` → returns None → Louvain fallback), so the default is safe before data ships. Remaining: produce the actual `data/role_labels/{dataset}.jsonl` files (item 1.2).
- [x] **WIRING — trigger-calib 500/3-annotator not enforced at load.** ✅ RESOLVED: `TriggerCalibrationRunner.load_labeled_interactions` now validates at load — raises `ValueError` if `<500` interactions or any row has `<2` annotator labels (Fleiss undefined), and warns if any row deviates from exactly 3 annotators. Floors are class constants (`MIN_INTERACTIONS=500`, `REQUIRED_NUM_ANNOTATORS=3`, `FLEISS_MIN_ANNOTATORS=2`), lowerable for pilots. Tested: count-raise, sub-Fleiss raise, and 4-annotator warn all behave correctly.

---

## 5. Pre-submission validation gates (run before claiming any number)

1. **Artifact-present vs absent delta.** For each of 1.2 / 1.3 / 1.4, run a smoke config with and without the file; confirm the metric changes and the log flag fires (`datasets_using_role_label_proxy`, `annotation_protocol="heuristic"`, legacy substring path). No metric should be *silently* unaffected.
2. **κ gates met before reporting.** Confirm held-out κ ≥ 0.7, trigger-calib Fleiss κ ≥ 0.6, human-eval mean κ ≥ 0.6 on the *actual* shipped annotations — not just the protocol.
3. **Anti-pattern trigger pass bar.** Category A/B/C P/R/F1 ≥ {0.90 / 0.80} (`trigger_calibration.py:469-475`); if below, report as limitation per outline §5.3.5.
4. **Constraint-forced counting.** Confirm evaluation reads `metadata.constraint_forced` (set in `agents/base.py`) and reports safe-template fallback rate separately (§4.4.2 step 4).
5. **Reproducibility dry-run.** From a clean checkout + shipped artifacts, regenerate one Table 1 cell end-to-end with pinned seeds/commits.

---

## 6. Outline ↔ code cross-references (so the checklist stays in sync)

| Outline claim | Code anchor |
|---------------|-------------|
| §5.3 Micro external-role-label primary GT | `aggregator.py:92-124, 151-161` |
| §5.3 Predictive Fidelity 2-annotator κ≥0.7 | `held_out_events.py:46`, `predictive.py:397-402` |
| §5.3.5 Trigger calib 500×3-annotator Fleiss≥0.6 | `trigger_calibration.py:83-88, 340-380` |
| §4.4.1 Category C logistic regression | `behavioral_trigger_classifier.py:115-182` |
| §4.4.2 Forced Reformulation N_retry=3 + fallback | `agents/base.py:139-186`, `constraint_forced` metadata |
| §7.5 no automated bias audit (unaudited fidelity) | (no code — by design, future work) |
