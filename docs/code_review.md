# Code Review: `src/` vs `outline.md`

**Date**: 2026-06-16
**Scope**: All of `src/` (~11.8k LOC, 60+ files) reviewed against the paper outline.
**Method**: Three parallel reviews (method pipeline; simulation+experiments+agents; evaluation+data+analysis), with the two most critical claims re-verified by direct code reading.
**Verdict**: Architecture is sound and broadly follows the outline, but there are **4 critical spec divergences** that would not survive ICWSM/WWW reviewer scrutiny if submitted as-is, plus several major gaps.

---

## Severity legend

- **CRITICAL** — spec violation that undermines a load-bearing claim, or a correctness bug that invalidates results.
- **MAJOR** — significant gap from the outline; must be addressed before submission.
- **MINOR** — cleanup, naming, or robustness issue; doesn't affect validity.

---

## CRITICAL findings

### C1. Tier 3 is NOT a hard block — undermines the central "structural override" claim
> **STATUS (2026-06-16 re-verification): RESOLVED.** The Forced Reformulation Protocol is implemented in `src/agents/base.py:139-186`: post-gen Tier-3 violation → `replan` up to `max_reformulation_retries=3` → re-check → safe-template fallback (`base.py:168-175`, sets `safe_template_fallback_used` + `constraint_forced` metadata flag). The original finding below conflated the *by-design* advisory pre-gen stage (outline §4.4 two-stage: pre-gen advisory + post-gen hard block) with the hard-block stage. Kept for history.
**Where**: `src/enforcement/tier3_block.py:110-116`
**Outline**: §4.4 — Tier 3 = "pre-generation hard block (trigger match → forced reformulation)". §3.2 and §7.1 lean on Tier 3 as the structural mechanism that overrides RLHF-induced homogenization.
**Code**: The pre-generation check returns `passed=True` even when violations are detected — it only appends a reformulation instruction to the system message:
```python
return EnforcementResult(
    passed=True,  # Messages modified, not blocked
    ...
    modified_messages=modified,
    injection_text=reform_msg,
)
```
The comment "Messages modified, not blocked" is an explicit admission. The post-gen check (`tier3_block.py:138-143`) does return `passed=False`, but it is consumed as `replan_feedback` (advisory) by the harness, not as a true generation-stopping block.
**Impact**: The paper's core differentiator vs COLLEAGUE.SKILL ("three-tier hard enforcement" with Tier 3 as the RLHF override) is not actually implemented as a hard constraint. A reviewer asking "what happens if the LLM still produces the anti-pattern after reformulation?" will find: nothing — the output is accepted.
**Required**: Either (a) implement a true pre-generation block (refuse to emit, force constrained regeneration with N_retry and fallback), matching outline §4.4.2 "Forced Reformulation Protocol"; or (b) rewrite the outline/§3.2/§7.1 to honestly describe Tier 3 as α-gated soft reformulation and drop the "hard block" framing.

### C2. Category C behavioral trigger is divergent — uses substring matching, not logistic regression
> **STATUS (2026-06-17 re-verification): RESOLVED.** Two layers of fix now in place:
> 1. Trained Category-C classifier (`src/enforcement/behavioral_trigger_classifier.py`, logistic regression over a 5-dim behavioral feature vector per outline §4.4.1) is the primary path, with the legacy substring path demoted to a complementary fallback (`tier3_block.py:304-315`).
> 2. The "agree" matching "disagree" substring bug noted below is fixed: the legacy fallback at `tier3_block.py:317-330` uses **sliding-window list equality** (`window == pat_actions`) over action-type sequences rather than naive substring containment, with an inline comment documenting the prior bug. Trained classifier artifact is shipped as `data/trigger_calibration/{dataset}.behavioral_clf.joblib` (release checklist item 1.5).
> Original finding retained below for history.
**Where**: `src/enforcement/tier3_block.py:198-206, 259-291`
**Outline**: §4.4.1 — Category C = "lightweight logistic regression over behavioral feature vector (stance shift rate, conflict engagement ratio, etc.)".
**Code**: `_check_behavioral()` performs action-sequence string matching — it concatenates recent action types into a string and does substring/regex matching against `trigger_action_patterns`. No logistic regression, no behavioral feature vector.
**Additional bug** (sub-finding): `tier3_block.py:280` does `if pat in recent_str.lower()`, a naive substring match. Pattern `"agree"` would match `"disagree"`. The sliding-window check (lines 284-289) is correct but the initial check is buggy.
**Impact**: §5.3.5 Trigger Calibration reports P/R/F1 "per category (A/B/C)" — Category C's numbers will not reflect the spec'd mechanism. Reviewers comparing to the outline will flag the divergence.
**Required**: Either implement the logistic-regression classifier over the behavioral feature vector, or revise §4.4.1 to describe the actual substring-matching mechanism honestly (and remove the "lightweight logistic regression" claim).

### C3. Micro Behavior ground truth uses Louvain communities, not external role labels — anti-circularity defense is incomplete
**Where**: `src/evaluation/aggregator.py:298-310`
**Outline**: §5.3 Micro Behavior — "Ground truth: 独立于 CADP 聚类的真实用户行为分布——使用外部标注的行为类型分类（如 Wu et al. 2025 的 audience segmentation 标签或人工标注的用户角色：moderator / provocateur / peacemaker / lurker）... 避免循环依赖".
**Code**: `_infer_communities()` derives real-data "ground truth" via `nx_comm.louvain_communities(graph, seed=42)` on the interaction graph. No code path loads Wu et al. 2025 audience-segmentation labels or human-annotated role labels anywhere in `src/evaluation/` or `src/data/`.
**Mitigating**: The Micro layer metrics (Frobenius, RSA, Uniformity, Complexity) compare agent-by-action matrices directly without explicit role labels, so this is *partially* circularity-safe — it does not use CADP's own clusters. But it also does not satisfy the outline's mandate for external role labels, and the Macro E-I index uses Louvain communities for real data vs CADP clusters for simulation, which is comparing two different community structures.
**Impact**: §5.3's anti-circularity defense (a key reviewer concern, given outline §4.1 explicitly flags circularity as the top validity threat) is not actually implemented as specified. A reviewer asking "how do you know the simulated moderator-like agents actually behave like real moderators?" has no answer in the current code.
**Required**: Either (a) integrate an external role-label source (Wu et al. 2025 segmentation, or human annotation with κ ≥ 0.7); or (b) revise §5.3 to describe Louvain communities as the comparison structure and acknowledge the weaker validation.

### C4. R4 Persona-Collapse Stress Test (§5.6.7) is entirely missing from code
**Where**: nowhere in `src/`. Protocol exists only at `docs/r4_persona_collapse_stress_test.md`.
**Outline**: §5.6.7 — 50+ turns, per-turn silhouette/DBI/behavioral-entropy/persona-embedding-drift, vanilla vs CADP Full vs CADP-minus-Anti-patterns.
**Code**: No `collapse_stress.py`, no per-turn silhouette/DBI emission. The sandbox emits `per_round_metrics` (`sandbox.py:128-129`) but these cover only `conflict_ratio` / `polarization_proxy` / `action_diversity` — not silhouette, DBI, or persona embedding cosine drift. The protocol doc's "Implementation hooks" table lists the modules that need to be created; none exist.
**Impact**: The strongest single piece of evidence for the "CADP resists persona collapse" claim (the answer to *Chameleon's Limit*) cannot be produced by the current code. This is the most important experiment for positioning vs the named competitors.
**Required**: Implement `src/experiment/collapse_stress.py` + extend `clusterer.py`, `embeddings.py`, `stats.py`, `sandbox.py` per the protocol doc's §7.

---

## MAJOR findings

### M1. α is scalar in default configs (functionally OK but structurally divergent)
**Where**: `configs/exp1_full.yaml:22`, `configs/exp2_full.yaml:11` — `alpha: 1.0`. `src/enforcement/base.py:32`, `harness.py:44-60`.
**Outline**: §4.4.3 — α is a per-dimension vector `(α_expr, α_mind, α_anti)`.
**Reality**: Three independent scalar parameters (`alpha_tier1/2/3`) exist in the schema (`config/schemas.py:52-55`) and `alpha_sensitivity.py` does sweep the full 3D grid. So the *capability* for per-dimension α exists. But: (1) there is no `Alpha` vector type; (2) the shipped experiment configs use only the scalar default; (3) the main exp1/exp2 runs therefore do not exercise per-dimension α at all — only the sensitivity sweep does.
**Required**: Either add `alpha_expr/alpha_mind/alpha_anti` to exp1/exp2 configs, or revise §4.4.3 to clarify that per-dimension α is explored only in §5.6.5 (and exp1/exp2 use uniform α=1.0).

### M2. 13 conditions in code vs 12 in outline
**Where**: `configs/exp1_full.yaml:17`, `src/config/schemas.py:28`, `src/agents/ablations.py:126`.
**Code**: 13 conditions including `cadp_constraint_only`.
**Outline**: §4.7/§5.2 — 12 conditions.
**Status of `cadp_constraint_only`**: It is NOT a duplicate of `cadp_minus_ap`. It is the complementary ablation (Tier1+Tier2 OFF, Tier3 ON — tests whether the Constraint Track alone suffices). This is a legitimate and useful condition.
**Impact**: Cell count = 13 × 3 × 4 × 5 = 780 (not 720). Cost estimates in outline §5.1 (≈ $3k–7k API) may be understated. More importantly, the outline's condition list and ablation logic chain (§5.2) do not mention this condition — reviewers will ask what it is and why it exists.
**Required**: Either (a) add `cadp_constraint_only` to outline §4.7/§5.2 with explicit rationale (recommended — it strengthens the dual-track necessity argument); or (b) drop it from the config.

### M3. Exp2 missing "real history" condition
**Where**: `configs/exp2_full.yaml:7`, `src/experiment/exp2_simulation.py`.
**Outline**: §6.2 — "3 conditions: real history vs CADP Full vs Pop-Aligned".
**Code**: Config lists only `cadp_full` and `pop_aligned`. "Real history" is used as ground truth (correctly) but is not a runnable condition. The outline's "3 conditions" is ambiguous — "real history" may be intended as ground-truth reference, not a runnable arm. But the wording suggests a 3-arm comparison.
**Required**: Clarify in §6.2 whether "real history" is a comparison arm or the ground-truth reference. If the former, add it as a (trivially pass-through) condition.

### M4. Fleiss' κ for 3-annotator trigger calibration is missing
**Where**: not in `src/evaluation/` or `src/experiment/trigger_calibration.py`.
**Outline**: §5.3.5 — "3 名标注者独立标注... Fleiss' κ ≥ 0.6".
**Code**: Only Cohen's κ (2-rater) is implemented (`held_out_events.py:92-125`, `human_eval.py:236-283`). Fleiss' κ (3+ raters) is not.
**Required**: Add Fleiss' κ computation to `trigger_calibration.py` (or import a library).

### M5. PII removal missing in data loaders
**Where**: `src/data/wikipedia.py`, `reddit.py`, `github.py` — none perform PII removal.
**Outline**: §7.5 line 444 — "用户名、IP 引用、个人链接等 PII 在预处理阶段移除".
**Code**: Raw `user_id`/`author`/`speaker` fields preserved through ingestion. PII anonymization exists only in `human_eval.py:209-223` for blind-review export.
**Impact**: This is an *ethics* claim in §7.5. If the paper says PII is removed in preprocessing but the code does not do this, that is a discrepancy a reviewer reproducing the work would discover.
**Required**: Either implement PII stripping in the data loaders (recommended), or revise §7.5 to describe the actual handling (e.g., PII retained internally for graph construction but stripped from published artifacts).

### M6. K-selection via silhouette/DBI is dead code under default config
**Where**: `src/clustering/clusterer.py:165-167`.
**Code**: When `n_clusters > 0` (default = 4), the silhouette/DB K-search (lines 170-185) is bypassed entirely.
**Outline**: §4.2 — "K 选择标准：silhouette score + Davies-Bouldin index + 领域可解释性".
**Impact**: The paper claims K is selected via internal validity indices, but the shipped pipeline uses a fixed K=4. The K-search logic exists but is unreachable from the default entry point.
**Required**: Either run with `n_clusters=None` (or a K range) in the experiment configs so the selection logic actually executes, or revise §4.2 to state K is fixed at 4 based on pilot silhouette analysis.

### M7. Platform topologies are flat (last-message-reply), not true trees
**Where**: `src/simulation/platforms/wikipedia.py:42`, `reddit.py:50`, `github.py:40`.
**Code**: All three platforms set `parent_msg_id = thread.messages[-1].msg_id` — every action replies to the most recent message, producing a flat chain, not an edit tree (Wikipedia) or threaded reply tree (Reddit).
**Outline**: §4.6 / §6.3 — Wikipedia edit tree, Reddit threaded reply + delta, GitHub issue lifecycle.
**Impact**: The "platform topology fidelity" claim in §6.3 is structurally not enforced. Reddit delta mechanism (which depends on the parent being the counter-argue target) may still work because `apply_action` re-derives the target, but the *topology* of the interaction graph that Macro/Meso metrics depend on is degenerate.
**Required**: Either implement true parent-targeted reply (agents choose a parent message), or revise §4.6/§6.3 to describe the actual flat-chain topology.

### M8. Temporal polarization proxy is conflict-action ratio, not E-I index
**Where**: `src/evaluation/aggregator.py:417`.
**Code**: Per-round "polarization" trajectory = conflict-action ratio per bin.
**Outline**: §6.4 — "极化指数按轮次画演化曲线" + Figure 5 ("polarization index time evolution").
**Impact**: Figure 5's y-axis label will say "polarization" but the underlying quantity is not the E-I polarization index used in the Macro layer. Internal inconsistency between Macro (true E-I) and trajectory (proxy).
**Required**: Either compute true per-turn E-I polarization, or rename the trajectory metric and update Figure 5's label.

---

## MINOR findings

### m1. Stance shift rate overcounts
`src/clustering/features.py:155-163` — counts ANY change in `action_type` between consecutive messages as a stance shift, including unrelated transitions (EDIT→POST). Outline §4.2 specifies "stance shift rate" (DISAGREE↔AGREE). Either rename to "action-type change rate" or implement stance-specific detection.

### m2. Cosine via dot product assumes pre-normalized vectors
`src/enforcement/mind_model_retriever.py:162` — `sim = float(np.dot(query_emb, doc_emb))`. Both vectors are L2-normalized in `_embed()` (line 177), so this is currently correct, but fragile if a future caller returns non-normalized vectors. Use `cosine_similarity` explicitly or assert normalization.

### m3. Bare `pass` exception handlers swallow errors
- `src/skill/compiler.py:381` — exception handler with bare `pass`.
- `src/evaluation/held_out_events.py:318` — exception handler with bare `pass`.
Both silently swallow errors. At minimum log the exception.

### m4. `LLMDualAnnotator` uses same model for both annotators
`src/evaluation/held_out_events.py:273-359` — both annotators use the same `model_name` with different prompt personas. This is a weak proxy for independent human annotation. The docstring acknowledges human adjudication is required, but if the pipeline uses the LLM annotations without human override, the κ ≥ 0.7 check is testing LLM-self-agreement, not human-LLM agreement.

### m5. NED implemented as Jensen-Shannon divergence
`src/evaluation/macro.py:71-96` — `normalized_entropy_distance()` uses JS divergence. The name "NED" suggests a different formula. Either cite the specific definition being used (JS is a defensible choice) or rename.

### m6. Stats are scattered, not centralized
`src/analysis/stats.py` has only Wilcoxon / bootstrap / Cliff's δ / BH correction. KS-test, DTW, KL-divergence, Frobenius, ARI, silhouette, DBI, Shannon entropy are all implemented inline in their respective evaluation modules. Not a bug, but a maintainability issue.

### m7. `cadp_shuffled` is per-agent random, not a global permutation
`src/agents/ablations.py:31-52` — each agent independently draws a different cluster's skill. Two agents in the same cluster may receive skills from different clusters. The outline's "permutation test" semantics (§5.2 Cond 8) is ambiguous about whether this should be a global permutation of the skill set. Clarify the intended statistical semantics.

### m8. Redundant `min(..., 1)` wrapper
`src/skill/expression_dna.py:180` — `style_long_short = min(1.0 - avg_sent_len / 40, 1)`. The expression is always ≤ 1.0 for non-negative lengths; the wrapper is dead.

---

## What is correctly implemented (for balance)

- **Five-layer metric architecture** — all five layers (Macro/Meso/Micro/Linguistics/Predictive) exist with the right metric families.
- **Linguistics feature orthogonality** — explicitly disjoint lexicons (`linguistics.py:25-42, 48-61`) with anti-leakage docstring. This is correctly handled.
- **Annotator κ checks** — Cohen's κ computed and threshold-checked for held-out events (κ ≥ 0.7) and human eval (κ ≥ 0.6).
- **Dual-track compilation** — Capability Track + Constraint Track properly separated (`skill/schema.py:93-162`).
- **Pass 1 / Pass 2 compilation** — positive case mining + negative case mining with cross-cluster gap analysis (`skill/compiler.py:105-115`).
- **Cluster stability validation** — bootstrap ARI with variance check (`clustering/validation.py`).
- **Cross-dataset transfer test** — all three modes (full-component / methodology / selective) implemented (`experiment/transfer_test.py`).
- **Agent runtime** — importance-weighted memory retrieval, periodic reflection (interval=10), per-turn planning with replan protocol.
- **Delta mechanism on Reddit** — correctly gated on counter-argue targeting (`platforms/reddit.py:27-36`).
- **Trigger calibration θ sweep** — full {0.75…0.95} sweep with per-category P/R/F1 and Precision ≥ 0.90 / Recall ≥ 0.80 target checks.
- **LLM client** — all 4 models (GPT-4o, Claude 3.5, Llama-3-70B, Qwen-2.5-72B) supported via OpenAI-compatible endpoints; cost tracking + tenacity retry wired.

---

## Recommended fix priority (for ICWSM/WWW submission)

1. **C1 (Tier 3 hard block)** — central to the paper's claim. Highest priority.
2. **C4 (R4 stress test)** — strongest positioning evidence vs *Chameleon's Limit*. High priority.
3. **C3 (Micro ground truth)** — anti-circularity is a known reviewer flash point. High priority.
4. **C2 (Category C classifier)** — divergence is documented; either implement or honestly describe. Medium priority.
5. **M5 (PII removal)** — ethics claim; cheap to fix. Medium priority.
6. **M2 (13 vs 12 conditions)** — update outline or config; one-line decision. Low priority but should not be left ambiguous.
7. **M4 (Fleiss' κ)** — one-function addition. Low priority.
8. **M7 (platform topology)** — affects Macro/Meso metrics materially. Medium priority; needs design decision.
9. Everything else — minor.

---

## AI disclosure

Review performed by an AI-assisted pipeline (three parallel ARS review agents + direct verification of the two most critical claims). All file:line citations are verifiable. No code was modified during this review.
