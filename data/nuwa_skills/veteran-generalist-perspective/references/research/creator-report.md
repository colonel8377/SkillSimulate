# Skill Review: veteran-generalist-perspective

**Reviewer role:** skill-creator agent (user-perspective review)
**File reviewed:** `/home/zf/.claude/skills/veteran-generalist-perspective/SKILL.md`
**Date:** 2026-07-07

---

## Executive Summary

The skill is a vivid, well-structured behavioral archetype for simulating a long-tenured, high-activity Wikipedian. The corpus evidence, mental models, and decision heuristics are strong. However, from a user's perspective, three gaps make it harder to invoke safely and consistently:

1. **No trigger keywords** — the user must guess when this skill is appropriate.
2. **No frequency constraints on hostility/profanity** — the skill describes escalation but does not bound it.
3. **No explicit deactivation / non-use guidance** — it is unclear when the skill should stay off.

The recommended changes below add concrete, minimal text to fix these issues without diluting the archetype.

---

## Detailed Evaluation

### 1. Activation triggers

**Status:** Present but broad.

The "When to activate this skill" section lists five scenarios (simulating talk-page disputes, stress-testing NPOV, role-playing high-activity editing, generating warnings/reverts/AfD messages, modeling reactions to bias/vandalism). These are useful, but they are task descriptions, not triggers a user can easily match to their prompt. A user typing "write a Wikipedia talk-page reply" may not realize this skill exists.

### 2. Role-playing rules

**Status:** Strong, with one critical omission.

Tone, length, interaction style, priorities, and anti-patterns are clearly defined. The archetype's inner tensions are well captured. The missing piece is a control mechanism for hostility: the skill allows "outright bluntness" and cites profane examples, but it never says how often to escalate or what to avoid in generated output.

### 3. Frequency constraints

**Status:** Absent.

There are no constraints on how often the archetype should be hostile, sarcastic, or profane, nor on how often to cite policies. In practice this can lead to over-escalated or gratuitously abrasive outputs, especially since the corpus examples include insults and profanity.

### 4. Missing info

- **Trigger keywords / phrases** — not listed anywhere.
- **Escalation ceiling** — no upper bound on hostility or profanity.
- **Output format examples** — no sample signature block or post structure.
- **When NOT to activate** — no negative triggers to prevent inappropriate use (e.g., actual Wikipedia editing help, civil mediation, policy tutoring).

### 5. Trigger keywords

**Status:** Not provided.

The skill relies entirely on semantic matching against "When to activate this skill" descriptions. Adding an explicit list of trigger keywords would improve discoverability and make invocation deterministic.

---

## Recommended Text Changes

### Change 1: Add explicit trigger keywords

**Rationale:** Makes the skill discoverable and gives users concrete phrases to invoke it.

**Location:** Insert immediately after the "When to activate this skill" section (before "Role-playing rules").

**Current text to replace:**

```markdown
- Modeling how a veteran generalist reacts when encountering bias, vandalism, forum-style chatter, or poor sources.

## Role-playing rules
```

**Replacement text:**

```markdown
- Modeling how a veteran generalist reacts when encountering bias, vandalism, forum-style chatter, or poor sources.

### Trigger keywords / phrases

Use this skill when the user prompt includes any of the following:

- "pretend you're a Wikipedian" / "act like a Wikipedian"
- "simulate a talk-page discussion / AfD / merge proposal"
- "stress-test this for NPOV / sourcing / policy compliance"
- "write a warning / revert / deep-thread reply"
- "veteran generalist perspective"

## Role-playing rules
```

---

### Change 2: Add frequency constraints on hostility and profanity

**Rationale:** Prevents over-escalation and keeps generated content usable without removing the archetype's edge.

**Location:** Insert as a new subsection at the end of "Role-playing rules".

**Current text to replace:**

```markdown
- **Avoid**: Generic platitudes, pretending subject-matter expertise the archetype does not claim, ignoring wiki-process, or staying soft when the corpus shows escalation.

## Response workflow
```

**Replacement text:**

```markdown
- **Avoid**: Generic platitudes, pretending subject-matter expertise the archetype does not claim, ignoring wiki-process, or staying soft when the corpus shows escalation.

### Escalation and profanity constraints

- **First contact is never hostile.** Open with bureaucratic politeness or dry policy-speak.
- **Escalate only when** the other party in the simulation ignores process, repeats the problem, or the user explicitly asks for a hostile reply.
- **Profanity** may appear at most once per substantive post and only as mild exasperation (e.g., "bullshit," "damn"). Never generate slurs, identity-based insults, or threats of real-world harm.
- **Hostility ceiling:** stop at sarcasm, blunt correction, or a block warning. Do not produce extended abusive rants.

## Response workflow
```

---

### Change 3: Add explicit deactivation / non-use guidance

**Rationale:** Clarifies boundaries so the skill is not invoked for tasks where role-playing a hostile veteran would be inappropriate.

**Location:** Insert as a new section after the updated trigger keywords subsection and before "Role-playing rules".

**Current text to replace:**

```markdown
### Trigger keywords / phrases

Use this skill when the user prompt includes any of the following:

- "pretend you're a Wikipedian" / "act like a Wikipedian"
- "simulate a talk-page discussion / AfD / merge proposal"
- "stress-test this for NPOV / sourcing / policy compliance"
- "write a warning / revert / deep-thread reply"
- "veteran generalist perspective"

## Role-playing rules
```

**Replacement text:**

```markdown
### Trigger keywords / phrases

Use this skill when the user prompt includes any of the following:

- "pretend you're a Wikipedian" / "act like a Wikipedian"
- "simulate a talk-page discussion / AfD / merge proposal"
- "stress-test this for NPOV / sourcing / policy compliance"
- "write a warning / revert / deep-thread reply"
- "veteran generalist perspective"

### When NOT to activate

Do not use this skill if the user is asking for:

- A neutral explanation of Wikipedia policy (answer as a helpful assistant, not in character).
- Help with real Wikipedia editing, drafting, or dispute resolution (avoid hostility).
- A balanced, civil mediation or facilitation role.
- Content that violates safety policies (hate speech, harassment, doxxing, etc.).

## Role-playing rules
```

---

## Priority Ranking

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 1 | Add trigger keywords | High discoverability | Low |
| 2 | Add escalation/profanity constraints | High safety/consistency | Low |
| 3 | Add deactivation guidance | Medium safety | Low |

All three changes are additive and do not alter existing mental models, values, or corpus evidence.
