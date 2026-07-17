# Exp1 v3 Feasibility Pilot: Detailed Explanation

## 1. What problem are we trying to solve?

Large language models (LLMs) are good at role-playing, but they tend to collapse toward a bland, agreeable default when asked to simulate a group of people talking. When you put several LLM agents together and tell them “you are Wikipedia editors,” they often produce conversations that are too polite, too uniform, or too conflict-avoidant compared to real Wikipedia editors.

Our project tries to make the simulation more realistic by giving each agent a **behavioral skill** distilled from real Wikipedia editors, and then comparing two ways of using that skill:

- Just telling the agent the skill once (Advisory).
- Dynamically retrieving the right part of the skill during the conversation and enforcing rules after the agent writes something (Full).

The experiment is narrow: we only test this on **one model** (DeepSeek-V4-Flash) and **one platform** (English Wikipedia talk pages). The goal is feasibility: does this package of techniques look promising enough to scale up?

---

## 2. What is a “behavioral skill”?

A behavioral skill is a structured rulebook distilled from a real community. We start by clustering hundreds of thousands of real Wikipedia editors based on what they actually do in discussions. Each cluster represents a recognizable kind of editor. For this pilot we use six clusters:

- Substantive discussant
- Niche terse specialist
- Confrontational editor
- Veteran generalist
- Community patroller
- Expert fact-checker

For each cluster, we build a **Nuwa skill** with three parts:

- **Expression DNA** — typical language style (sentence length, formality, vocabulary).
- **Mind Models** — reasoning templates: how this kind of editor thinks about disputes, evidence, and policy.
- **Anti-patterns** — behaviors the archetype should avoid (e.g., personal attacks, unsupported reverts).

These skills are generated from real Wikipedia data, but the experiment uses them as identical inputs in both conditions.

---

## 3. Experimental design

### 3.0 How the “tiers” and “filters” are actually implemented

The `cadp_full_nuwa` condition runs three layers of enforcement. They are all prompt-time or output-time interventions: the system never changes model weights.

#### Tier 1 — Expression DNA style filter

- **What it checks:** Does the generated message look like it came from the cluster’s typical language distribution?
- **How it works:**
  - During skill compilation, we embed up to 400 representative utterances per cluster using a sentence embedding model (BGE).
  - We compute a centroid vector and standard deviation for each cluster.
  - We calibrate a z-score threshold at the 95th percentile on a held-out validation split of those representative utterances.
- **At runtime:**
  - The agent’s message is embedded.
  - The maximum absolute z-score (distance from centroid in standard-deviation units) is computed.
  - If `max_z > threshold`, the message is rejected and the agent must rewrite it.
- **In this run:** It almost never triggered.

#### Tier 2 — Mind-Model retriever

- **What it does:** Before the agent writes a message, select the most relevant reasoning templates (Mind Models) from the skill and inject them into the context.
- **How it works:**
  - A small LLM classifier reads the last ~8 messages and estimates the current dialogue state: topic, stance direction, and conflict intensity.
  - We embed the skill’s Mind Models and the dialogue state with Sentence-BERT.
  - We retrieve the top-5 most similar Mind Models.
  - Those templates are added to the agent’s prompt for this turn.
- **In this run:** This is the mechanism that actually produced the action-fidelity improvement. It is not a filter, but a dynamic prompt-conditioning layer.

#### Tier 3 — Anti-pattern trigger filter

- **What it checks:** Does the message violate any anti-pattern (forbidden behavior) from the skill?
- **How it works:**
  - Each anti-pattern in the skill has `trigger_keywords` (exact words) and/or `trigger_regex` (regular expressions).
  - In the current feasibility stage, only a conservative set of universal hostility keywords is active. Archetype-specific semantic triggers and behavioral triggers are disabled because they have not yet been calibrated on human annotations.
  - If a keyword or regex matches, the message is rejected and the agent is told which anti-pattern was violated.
- **Retry policy:**
  - The agent rewrites up to `max_reformulation_retries` times (set to 2 in this config).
  - If all retries still trigger, the system replaces the message with a bland safe template and marks it as `constraint_forced`.
- **In this run:** It never triggered. The 0/74 manipulation audit and 0% safe-template rate confirm this.

### 3.1 Conditions

We have only two conditions in this pilot.

#### `cadp_advisory_nuwa` — the control

- The agent receives the full Nuwa skill as a block of text in its system prompt.
- The agent sees the same rules at the start of every round.
- No runtime enforcement: the agent is simply advised to follow the rules.
- No reflection (the reflection interval is set to 10, which never fires in 4 rounds).

#### `cadp_full_nuwa` — the treatment

- The agent receives the **same** Nuwa skill content.
- Before each turn, a **retriever** selects the most relevant Mind Models based on the current conversation state (topic, conflict intensity, stance direction).
- After the agent writes a message, a **Tier-1 style filter** checks whether the message matches the cluster’s Expression DNA.
- A **Tier-3 trigger filter** checks whether the message violates anti-patterns.
- If either filter rejects the message, the agent is asked to rewrite it (up to 2 retries).
- If all retries fail, the system falls back to a safe template.

The key point: the **content** is identical. Only the **execution mechanism** differs.

### 3.2 Population

- 12 agents per run.
- Balanced allocation: 6 skills × 2 agents each.
- Each agent is assigned a fixed archetype and carries the corresponding skill.
- Agents also have memory, planning, and engagement mechanisms.

### 3.3 Stimuli: observed continuation

We do not ask the agents to start conversations from scratch. Instead, we use real 2017–2018 Wikipedia talk-page threads.

For each thread:
1. Take the first half as a prefix (the conversation so far).
2. Give that prefix to the agents.
3. Let the agents continue the thread for 4 rounds.
4. Compare the simulated continuation to the real second half of the thread.

This is called an **observed-continuation** design. It is a strong counterfactual because the ground truth is the exact same thread, just later in time.

We use 2017–2018 because those years have enough platform events (edits, reverts, reports) to make action evaluation meaningful, while still being a tractable subset of the full corpus.

### 3.4 Repeats and randomness

- 3 repeats per condition.
- Each repeat uses the same random seed offset (`seed + repeat`).
- This makes the runs deterministic and reproducible.

---

## 4. The evaluation metrics

We compare the simulated thread to the real held-out suffix on three layers.

All three metrics are formulated as **distances**, so lower values mean the simulation is closer to reality.

### 4.1 Action fidelity

- **What it measures:** The distribution of actions used by the agents versus the real editors.
- **Actions:** Discuss, Edit, Revert, Report.
- **Metric:** Normalized Entropy Distance (NED), which is based on the Jensen-Shannon divergence between the simulated and real action distributions.
- **Plain English:** Are the agents doing the right mix of things? Are they reverting, reporting, and editing at realistic rates, or are they just talking?

### 4.2 Interaction structure

- **What it measures:** The shape of the conversation, not just the actions.
- **Components:**
  - **Cascade length:** How many replies branch off from a single comment.
  - **Structural fidelity:** How similar the reply graph is to the real one.
- **Metric:** A weighted average of (1) the KS statistic comparing cascade-length distributions and (2) the inverse of structural fidelity.
- **Plain English:** Does the back-and-forth pattern look like a real Wikipedia discussion, or is it unnaturally flat or linear?

### 4.3 Linguistic fidelity

- **What it measures:** The language style of the messages.
- **Components:**
  - Discourse relation distribution (e.g., contrast, elaboration, temporal).
  - Sentiment trajectory (how sentiment changes across the thread).
  - Speech act distribution (assertive, directive, expressive, commissive).
  - SIP (semantic information preservation) using sentence embeddings.
- **Metric:** A weighted average of `1 - similarity` for each component, computed only on non-safe-template messages.
- **Plain English:** Do the agents sound like real Wikipedia editors? Do they argue, clarify, and escalate in the right way?

### 4.4 Quality guards

These checks make sure the treatment did not break the simulation in obvious ways:

- **Message ratio:** Did Full produce at least 95% as many messages as Advisory? Prevents the system from suppressing participation.
- **Safe-template rate:** Did the system fall back to bland template responses more than 10% of the time? Prevents over-filtering.
- **Action-text consistency:** Are the action labels attached to messages consistent with the message text? Prevents corrupted outputs.
- **Family regression:** No metric family can get meaningfully worse in Full than in Advisory (threshold: 10%).

---

## 5. The GO/STOP decision rule

The experiment is a **paired comparison**. For each repeat, we compare the same condition pair (Full vs Advisory) on the same dataset and model.

A metric family counts as a **win** for Full if:
1. Full’s mean distance is lower than Advisory’s mean distance by at least 5% in at least 2 of the 3 repeats.

The overall verdict is:
- **GO** if at least 2 of 3 metric families win and all quality guards pass.
- **CONDITIONAL_GO** if exactly 2 families win but only on a borderline basis, or if guards are borderline.
- **STOP** otherwise.

This is a **feasibility gate**, not a formal significance test. With only 3 repeats, the result is directional evidence for scaling up, not proof of general effectiveness.

---

## 6. Results

The pilot finished with a **GO** verdict.

### 6.1 Metric results

| Metric family | Advisory mean | Full mean | Relative improvement | Repeat wins | Verdict |
|---|---|---|---|---|---|
| Action fidelity | 0.211 | 0.114 | **+47.9%** | 3/3 | ✅ Win |
| Interaction structure | 0.085 | 0.077 | +9.5% | 2/3 | ✅ Win |
| Linguistic fidelity | 0.198 | 0.201 | -1.6% | 1/3 | ❌ Not a win |

### 6.2 Quality guard results

| Guard | Result | Required | Pass? |
|---|---|---|---|
| Message ratio | 1.00 | ≥ 0.95 | ✅ |
| Safe-template rate | 0.00 | ≤ 0.10 | ✅ |
| Action-text consistency | 1.00 | ≥ 0.90 | ✅ |
| Family regression | None > 10% | ≤ 0.10 | ✅ |

### 6.3 Final verdict

```json
{
  "verdict": "GO",
  "reason": "primary_metric_and_quality_gates_passed",
  "metric_wins": 2,
  "required_metric_wins": 2
}
```

---

## 7. What does this mean?

### 7.1 The positive signal

The strongest result is **action fidelity**: Full produced a much more realistic mix of actions than Advisory. This held in all 3 repeats. The dynamic retrieval appears to help agents choose actions that are more like real Wikipedia editors — more reverts, more reports, more substantive edits, rather than just endless discussion.

The interaction structure also improved, though only marginally and only in 2 of 3 repeats. This suggests the conversation shape is also slightly better, but the evidence is weaker.

### 7.2 The neutral signal

Linguistic fidelity was essentially unchanged. The agents did not start talking more like real Wikipedia editors in terms of discourse, sentiment, or speech acts. This is not necessarily bad: the retrieval is injecting reasoning templates, not style templates, so you would expect action choices to shift more than language style.

### 7.3 The surprising signal

The safe-template rate was **0.0**. This means the post-generation filters (Tier-1 and Tier-3) almost never rejected or rewrote an agent message. The manipulation audit also found 0 of 74 Advisory messages that would have triggered Full’s filters.

This tells us something important: **the gain did not come from “filter-retry” catching bad outputs**. It came from **Tier-2 retrieval** changing the agent’s prompt before generation. In other words, the agent made better choices because it was reminded of the right rule at the right moment, not because its bad choices were blocked after the fact.

### 7.4 Headline takeaway

For this small, single-model, single-platform pilot:

> **Retrieval-augmented rule execution improves action realism in Wikipedia-talk simulations, relative to advisory-only prompting.**

The “filter-retry” component did not activate much in this run, so the result should be described as evidence for **retrieval-driven execution**, not necessarily for post-hoc filtering.

---

## 8. Limitations

- **Small sample:** Only 3 repeats and 12 agents per run. The result is directional, not confirmatory.
- **Single model:** Only DeepSeek-V4-Flash was tested.
- **Single platform:** Only English Wikipedia.
- **Short horizon:** 4 rounds is enough for a feasibility gate but not enough to study long-term dynamics like polarization or conflict escalation.
- **Tier-3 is conservative:** The anti-pattern triggers were intentionally limited to universal hostility keywords. Full archetype-specific trigger calibration is future work.
- **Observed continuation is strong but narrow:** The ground truth is the same thread, but this only tests whether the simulation can continue a specific real conversation, not whether it can generate realistic new conversations from scratch.

---

## 9. What would come next?

A GO verdict means we should scale up, not stop. The next steps would include:

- More repeats (e.g., 10 or 20) to estimate effect size reliably.
- Add baseline conditions like Descriptive Persona and Rich Cluster Narrative to contextualize the result.
- Add the distiller comparison: does the gain depend on using Nuwa’s 5-layer structure, or would another structure work?
- Calibrate archetype-specific Tier-3 triggers so the filter-retry mechanism actually fires.
- Test on Reddit and GitHub to check cross-platform transfer.
- Run longer simulations (e.g., 30 rounds) to study long-term dynamics.
