# Exp1 Wikipedia Smoke Test — Fix Plan

## Context

The 2026-06-24 `exp1_wikipedia_smoke` run completed end-to-end but the metrics are weak and do not clearly differentiate CADP from baselines. This plan lists the root causes and the concrete code/config changes needed to fix them.

---

## Root causes (from diagnosis)

1. **Simulation scale too small**
   - `src/experiment/exp1_validation.py::_prepare_sim_threads()` hard-codes a 5-thread cap.
   - Result: ~1,500 simulated messages vs. 391k real Wikipedia messages.

2. **Action-type collapse**
   - `src/agents/planning.py::PLANNING_PROMPT` lists actions but gives no behavioral guidance.
   - Result: agents pick `discuss` >99% of the time; real Wikipedia has ~5% `revert` and ~6% `report`.

3. **Saturated/broken metrics**
   - `src/evaluation/meso.py::structural_fidelity` normalizes feature vectors to unit length, producing ~0.934 for every condition.
   - `src/evaluation/micro.py::action_matrix_similarity` truncates a 30-agent sim matrix and a ~38k-user real matrix to `min_rows`, yielding negative values.

4. **Constraints scrub conflict language**
   - `src/agents/base.py::_safe_template_response()` produces bland fallback text.
   - `cadp_constraint_only` conflict F1 collapses to 0.03.

5. **Skill extraction under-delivers**
   - Compiled skills have empty `mind_models` and only generic anti-patterns (personal attacks / profanity).
   - Files: `src/skill/mind_models.py`, `src/skill/anti_patterns.py`, `src/skill/compiler.py`.

6. **Predictive fidelity uses heuristic ground truth**
   - No held-out κ-validated annotations; conflict F1 clusters at 0.483 because the classifier degenerates to all-positive predictions.

---

## Fix plan

### Phase 0 — Fix the metrics first (free, no API calls)

| File | Change | Expected outcome |
|---|---|---|
| `src/evaluation/meso.py` | Rewrite `structural_fidelity()` to compare raw or log-scaled graph features instead of unit-normalizing them. | Metrics stop saturating at 0.934 and start discriminating conditions. |
| `src/evaluation/micro.py` | Rewrite `action_matrix_similarity()` to align dimensions meaningfully (e.g., sample real users down to sim size, or compare action-distribution vectors instead of full matrices). | Values move into [0, 1] and reflect behavioral similarity. |

### Phase 1 — Make the simulation large enough to matter

| File | Change | Expected outcome |
|---|---|---|
| `src/config/schemas.py` | Add `max_sim_threads: int` to `ExperimentConfig`. | Configurable thread budget. |
| `src/experiment/exp1_validation.py` | Replace hard-coded `5` in `_prepare_sim_threads()` with `self.config.max_sim_threads`. | Wikipedia cells use more topics (suggest 50–100 for pilot, scale up later). |
| `configs/exp1_wikipedia_smoke.yaml` | Add `max_sim_threads: 50` (or higher). | More realistic thread diversity. |

### Phase 2 — Make agents choose conflict actions

Option A — prompt guidance (cheapest):

| File | Change |
|---|---|
| `src/agents/planning.py` | Extend `PLANNING_PROMPT` with platform-specific action-selection guidance, e.g., "If you strongly disagree with another user's edit, you may REVERT it. If you see a serious policy violation, you may REPORT it." |

Option B — stochastic action sampler (more robust):

| File | Change |
|---|---|
| `src/agents/planning.py` | After the LLM picks an action, add a sampler that re-rolls the action according to the real per-cluster action distribution with temperature, while still allowing the LLM to choose text. |

Expected outcome: coverage rises from 0.25–0.5 toward 0.75–1.0; DTW, conflict F1, and E-I polarization become meaningful.

### Phase 3 — Calibrate constraints so conflict is preserved

| File | Change | Expected outcome |
|---|---|---|
| `src/agents/base.py` | Replace `_safe_template_response()` with a fallback that still expresses disagreement neutrally (e.g., "I disagree with that point because …") rather than a fully bland statement. | Conflict language survives Tier-3 fallback. |
| `src/skill/anti_patterns.py` | Keep anti-patterns focused on personal attacks, profanity, and clear harassment; do NOT treat generic disagreement as a violation. | `cadp_constraint_only` conflict F1 recovers. |

### Phase 4 — Improve skill compiler output

| File | Action |
|---|---|
| `src/skill/mind_models.py` | Debug why `mind_models` are empty; inspect LLM prompt/response and add validation/retry. |
| `src/skill/anti_patterns.py` | Debug why anti-patterns are generic; improve negative-case selection in `src/skill/compiler.py`. |
| `outputs/skills/skill_cluster_*.yaml` | After recompiling, verify each cluster has ≥1 mind model and ≥1 specific anti-pattern. |

Note: recompiling skills costs API calls (deepseek-v4-pro). Defer this until Phase 0–2 are validated.

### Phase 5 — Switch to real held-out annotations

| Step | Command / action |
|---|---|
| Generate draft annotations | `python -m src.main annotate-events --config configs/exp1_wikipedia_smoke.yaml --dataset wikipedia` |
| Human adjudication | Review `data/held_out_events/wikipedia.jsonl` and resolve disagreements. |
| Re-run evaluation | Predictive fidelity will then use κ-validated labels instead of heuristic labels. |

Note: this is expensive (LLM dual annotator + human time). Defer until the simulation itself produces realistic conflict behavior.

---

## Validation strategy (cost-controlled)

Do not full-rerun after every fix. Use a cheap pilot config:

```yaml
# configs/exp1_wikipedia_pilot.yaml
name: exp1_wikipedia_pilot
datasets: [wikipedia]
models: [deepseek-v4-flash]
conditions:
  - vanilla
  - cadp_full
  - cadp_minus_edna
  - cadp_constraint_only
  - pop_aligned_cadp
num_repeats: 1
num_rounds: 20
population_size: 15
max_sim_threads: 50
reflection_interval: 20   # or disable reflection entirely for the pilot
max_threads: 2000
```

Estimated cost (file rates): ~$300; actual DeepSeek rates: likely ~$40–$50.

Only scale back to the full 13-condition / 50-round grid after the pilot shows:
- coverage ≥ 0.75
- structural_fidelity varies across conditions
- action_matrix_similarity is non-negative and varies
- conflict F1 is no longer clustered at 0.483

---

## Prioritized TODO (mapped to GitHub-style tasks)

- [ ] #4 Fix structural fidelity metric
- [ ] #4 Fix action matrix similarity metric
- [ ] #2 Make `max_sim_threads` configurable and raise default
- [ ] #3 Improve planner action-selection guidance / sampler
- [ ] #7 Calibrate safe-template fallback and anti-patterns
- [ ] #5 Debug skill compiler (mind models + anti-patterns)
- [ ] #6 Generate and adjudicate Wikipedia held-out annotations
- [ ] Create `configs/exp1_wikipedia_pilot.yaml` for cheap validation

---

## Files touched by this plan

- `src/evaluation/meso.py`
- `src/evaluation/micro.py`
- `src/config/schemas.py`
- `src/experiment/exp1_validation.py`
- `src/agents/planning.py`
- `src/agents/base.py`
- `src/skill/mind_models.py`
- `src/skill/anti_patterns.py`
- `src/skill/compiler.py`
- `configs/exp1_wikipedia_smoke.yaml`
- `configs/exp1_wikipedia_pilot.yaml` (new)

---

## Cost summary

| Stage | Conditions | Rounds | Population | Threads | Reflection | Est. cost (file rates) | Est. cost (actual DeepSeek) |
|---|---|---|---|---|---|---|---|
| Full current | 13 | 50 | 30 | 5 | yes | ~$4,800 | ~$685 |
| Proposed pilot | 5 | 20 | 15 | 50 | optional | ~$300 | ~$40–$50 |
| Full after fixes | 13 | 50 | 30 | 100+ | yes | ~$5,000–$6,000 | ~$700–$900 |

Use the pilot to validate fixes before scaling up.
