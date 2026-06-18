# R4: Persona Collapse Stress-Test Protocol

**Status**: Design draft for §5.6.7 of `outline.md`
**Motivation anchor**: *The Chameleon's Limit* (arXiv:2604.24698), Gao et al. (2024, NAACL Findings)
**Owns**: `src/experiment/` (new module), `src/clustering/clusterer.py` (metric extraction), `src/simulation/` (turn loop)
**Date**: 2026-06-16

---

## 1. Goal

Empirically demonstrate that CADP's three-tier enforcement structurally prevents the persona-collapse phenomenon diagnosed by *The Chameleon's Limit* (arXiv:2604.24698) and corroborated by Gao et al. (2024, NAACL Findings — LLM agents drift to consensus). The test isolates the *longitudinal* behavioral-divergence dimension that the main Exp 1 (§5) does not stress, because §5 reports only end-state metrics over 30–50 turns.

This test does **not** claim CADP "solves" persona collapse in general — it claims that, under the specified stress conditions, CADP's behavioral-rule enforcement delays / suppresses collapse relative to descriptive-persona baselines, with quantified drift trajectories.

---

## 2. Hypotheses

| # | Hypothesis | Direction | Primary metric |
|---|------------|-----------|----------------|
| H1 | Descriptive-persona agents exhibit measurable collapse within 50 turns | silhouette(turn 50) < silhouette(turn 5) by ≥ Δ_collapse | silhouette score, Δ_collapse ≥ 0.10 |
| H2 | CADP agents maintain cluster separation (no monotonic collapse) | |silhouette(turn 50) − silhouette(turn 5)| < Δ_collapse/2 | silhouette score |
| H3 | Behavioral entropy of descriptive-persona populations shrinks over turns (homogenization) | entropy(turn 50) < entropy(turn 5) | Shannon entropy over action-type distribution |
| H4 | CADP populations retain baseline entropy | |entropy(turn 50) − entropy(turn 5)| < Δ_entropy/2 | same |
| H5 | Inter-agent persona embedding cosine similarity rises for descriptive-persona (convergence) | mean cosine(turn 50) > mean cosine(turn 5) + Δ_cos | mean pairwise Sentence-BERT cosine |
| H6 | Tier 3 (Anti-patterns) removal reproduces collapse signature | CADP-minus-Anti-patterns collapses faster than CADP Full | silhouette drift slope |

H6 is the *causal-mechanism* test: if removing Anti-patterns reproduces the collapse, then the Anti-pattern tier is the load-bearing component — directly answering *why* CADP resists collapse.

---

## 3. Conditions (2 × 4 × 3 factorial, reduced)

| Axis | Levels | Notes |
|------|--------|-------|
| Persona method | (a) Descriptive Persona (§5.2 Cond 2), (b) CADP Full (Cond 7), (c) CADP minus Anti-patterns (Cond 11) | 3 levels — (c) is the mechanism-isolation arm |
| Model | GPT-4o, Claude 3.5 Sonnet, Llama-3-70B, Qwen-2.5-72B | 4 levels — matches §5.1 model matrix |
| Dataset | Wikipedia Talk, Reddit r/changemyview, GitHub Issues | 3 levels — matches §5.1 |

**Total cells**: 3 × 4 × 3 = 36 cells. Repeats: 5 per cell → **180 simulation runs**.

This is a *subset* of the §5.1 factorial (144 cells × 10 repeats). Reduction is justified: the longitudinal metric (drift slope) requires fewer repeats than end-state power analysis because each run yields a *time series* of N_turns measurements (effectively paired).

**Power**: pilot on Wikipedia + GPT-4o. If effect size on silhouette drift slope d ≥ 0.6, 5 repeats × 50 turns = 250 paired measurements per cell gives >95% power for the paired t-test on drift slopes (α = 0.05, Bonferroni-adjusted for 6 hypotheses → α' = 0.0083).

---

## 4. Protocol

### 4.1 Interaction setup
- **Population**: N = 30 agents (K = 3–5 clusters × 6–10 agents/cluster — matches §6.2)
- **Turns**: 50 per simulation (vs §5.1's 30–50; the upper bound is chosen to push past the typical collapse-onset observed in pilot data, where descriptive-persona populations show first-collapse signal around turn 20–25)
- **Topic seeding**: controversy-sampled议题 from the same distribution as §6.2 (NOT new sampling — comparability requires same议题 pool)
- **Action space**: platform-specific (matches §6.3)
- **Memory**: full per-turn memory + reflection + planning (matches §4.6 — no ablation of agent architecture, only of persona method)

### 4.2 Measurement cadence
After **every turn t ∈ {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}** (11 measurement points), record:

| Measurement | Source | Computed by |
|-------------|--------|-------------|
| Cluster silhouette (per-agent action embedding → re-cluster) | `src/clustering/clusterer.py` | re-run K-Means on action embeddings up to turn t |
| Davies-Bouldin index | same | same |
| Behavioral entropy (Shannon over action-type distribution) | action log | `src/analysis/stats.py` |
| Pairwise Sentence-BERT cosine similarity (mean + 95% CI) | per-agent aggregate utterance embedding | `src/clustering/embeddings.py` |
| Action-type distribution histogram | action log | `src/analysis/stats.py` |

**Critical**: re-clustering at each measurement point uses the **same K** as initial clustering (frozen from §5.2 Step 1). If K is allowed to re-optimize per turn, drift measurement is confounded.

### 4.3 Per-turn action embedding
- Each agent's action at turn t → Sentence-BERT embedding
- Aggregate per-agent state at turn t = mean of its last-5-action embeddings (rolling window)
- This rolling aggregation smooths turn-to-turn noise while preserving drift signal

### 4.4 Stopping rule
Run all 50 turns regardless of early signal. Early stopping would bias the drift-slope estimator (regression-to-mean).

---

## 5. Metrics & Analysis

### 5.1 Primary: silhouette drift slope
For each simulation run, fit OLS: `silhouette(t) = β₀ + β₁·t + ε` over the 11 measurement points.
- **β₁ (slope)** is the per-turn drift.
- Compare slopes across (Descriptive, CADP Full, CADP-minus-Anti) via mixed-effects model:
  `β₁ ~ method + (1 | model) + (1 | dataset)`
- **H1 supported** if Descriptive β₁ < −Δ_collapse/50 (i.e., cumulative drop ≥ 0.10 over 50 turns).
- **H2 supported** if |CADP β₁| < |Descriptive β₁|/2 and the mixed-effects coefficient for `method=CADP` is significantly positive (α' = 0.0083).

### 5.2 Secondary: behavioral entropy trajectory
- Plot entropy(t) for each method, overlaid with 95% CI bands
- **H3/H4 tested** via paired permutation test on entropy(50) − entropy(5)

### 5.3 Mechanism: persona cosine convergence
- Plot mean pairwise cosine similarity(t)
- **H5** supported if Descriptive cosine(50) − cosine(5) > Δ_cos (= 0.05 in pilot)
- **H6** (mechanism) supported if CADP-minus-Anti-patterns cosine drift > CADP Full cosine drift, with effect size matching Descriptive

### 5.4 Reporting
- **Figure R4-1**: silhouette(t) trajectories, 3 methods × 4 models, small multiples
- **Figure R4-2**: entropy(t) trajectories, same layout
- **Figure R4-3**: mean pairwise cosine(t), same layout
- **Table R4-1**: drift slopes β₁ per cell, with mixed-effects model summary
- **Table R4-2**: H1–H6 verdict matrix (Supported / Refuted / Inconclusive)

---

## 6. Validity threats (specific to this test)

| Threat | Mitigation |
|--------|-----------|
| Re-clustering instability confounds drift | Freeze K from §5.2 Step 1; report ARI between turn-5 and turn-50 cluster assignments as sanity check (expect ARI > 0.7 for stable methods) |
| Topic depletion in long runs (agents exhaust controversy) | Sample议题 pool sized 2× the expected consumption; report remaining-议题 count at turn 50 |
| Memory growth compresses diversity (architecture artifact, not persona artifact) | Run a Vanilla-LLM arm as additional control; if Vanilla also collapses → collapse is architecture-driven, not persona-driven (negative result for H1, refine claim) |
| Sentence-BERT embedding saturation (cosine → 1 trivially as utterances get longer) | Use fixed-length utterance truncation; report cosine on first-512-token window only |
| 50 turns insufficient to manifest collapse | Pilot first on Wikipedia + GPT-4o; if no collapse by turn 50, extend to 100 turns for that cell only and flag in writeup |

---

## 7. Implementation hooks

| File | Required change |
|------|-----------------|
| `src/experiment/` (new) | `collapse_stress.py` — orchestrates the 180-run loop, calls existing runner with a longitudinal metrics callback |
| `src/clustering/clusterer.py` | expose `silhouette_at_state(agent_actions, K_frozen)` and `davies_bouldin_at_state(...)` — currently only computes end-state |
| `src/clustering/embeddings.py` | verify Sentence-BERT rolling-window aggregation helper exists; add `rolling_agent_state(actions, window=5)` |
| `src/analysis/stats.py` | add `behavioral_entropy(action_log)` and `drift_slope(time_series)` |
| `src/simulation/` | add per-turn metrics emission hook in the turn loop (currently emits end-state only) |
| `src/evaluation/aggregator.py` | add R4 result schema |

**Estimated cost**: API models (GPT-4o + Claude 3.5) at 50 turns × 30 agents × 2 methods × 2 datasets × 5 repeats ≈ 30k agent-turns. At ~$0.005/turn (GPT-4o) ≈ **$150 per API model**; total API ≈ $300. Open-source: ~50 GPU-hours on 4× A100.

---

## 8. Writeup recipe (for §5.6.7 + appendix)

1. **One-paragraph summary** in §5.6.7: states H1, H2, H6 verdicts with figure cross-refs.
2. **Figure R4-1** in main text (silhouette drift) — most visually persuasive single figure for the persona-collapse claim.
3. **Full protocol + Tables R4-1/R4-2** in appendix.
4. **Discussion tie-in** to §7.1 (Alignment Tax): if H6 holds, frame as "Anti-patterns tier is the structural override against RLHF-induced homogenization, demonstrated longitudinally in §5.6.7."

---

## 9. Failure paths

- **H1 refuted (no collapse in Descriptive arm)**: This is *itself* an interesting result — it would mean 50 turns is not long enough to manifest the Chameleon's Limit signature. Action: extend to 100 turns, or report as a boundary on the Chameleon's Limit claim (contributes to §7.4 validity discussion).
- **H2 refuted (CADP also collapses)**: CADP's hard constraints are not sufficient at the longitudinal scale. Action: investigate which tier's removal accelerates collapse most (H6 sub-analysis); report as scope limitation, not as fatal to the main claim (main claim is end-state fidelity, §5).
- **H6 refuted (CADP-minus-Anti-patterns resists collapse as well as Full)**: Tier 3 is not the load-bearing component. Action: re-attribute collapse-resistance to Tier 1 (Expression DNA embedding filter) — re-run with CADP-minus-Expression-DNA to localize. Honest reporting required; do not retro-fit the narrative.

---

## 10. AI disclosure

Protocol drafted by AI-assisted research pipeline (ARS deep-research, lit-review mode). Hypotheses, factorial design, and validity-threat list reflect the cited literature (Chameleon's Limit arXiv:2604.24698; Gao et al. 2024 NAACL Findings; Chicco et al. 2025 PeerJ CS for silhouette/DBI methodology). Power analysis and cost estimates are derived, not looked up — verify against actual API pricing before budgeting.
