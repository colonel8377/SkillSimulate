# Skill Creator Report: Niche Terse Specialist

**File reviewed:** `/home/zf/.claude/skills/niche-terse-specialist-perspective/SKILL.md`
**Role:** skill-creator (user-perspective review)
**Date:** 2026-07-07

---

## Executive summary

The skill captures a vivid, well-sourced Wikipedia newcomer archetype. Its tone, mental models, and expression DNA are strong. However, from a user's point of view it is missing three things that make a skill safe and easy to invoke: (1) explicit negative activation triggers, (2) a dedicated trigger-keyword section, and (3) frequency/escalation constraints. Without these, the model may over-apply the persona or stay in it too long.

---

## 1. Activation triggers

### What works
- The five numbered scenarios under "When to activate this skill" are concrete and tied to real Wikipedia dispute patterns (deletion threat, policy clarification, interpersonal-space conflict, vandalism report, need for terse/source-focused voice).
- Each scenario maps cleanly to the mental models and decision heuristics below.

### What is missing
- No "when NOT to activate" guidance. The persona is defensive and emotionally spiky; it should not be used for calm welcomes, general reference questions, mediation, or when the user has asked for neutral third-person analysis.
- The group-size number (51,436) appears twice but is never explained. A user may wonder whether it is a confidence score, a population estimate, or a corpus count.

---

## 2. Role-playing rules

### What works
- Tone, length, interaction style, priorities, and avoid-list are all clearly stated.
- The "Avoid" list correctly warns against diplomatic essays, wiki-ese preamble, and generic welcomes.

### What is missing
- The "Response workflow" is a rigid 6-step recipe. It reads like a mandatory sequence rather than a heuristic, which can make responses feel mechanical.
- There is no guidance on when to drop the persona (e.g., after a procedural question is answered, or when the user explicitly asks for a different register).

---

## 3. Frequency constraints

### Current state
- None. The skill never says how often it should be used or how long the persona should persist across turns.

### Risk
- In a multi-turn conversation the model may keep answering as the defensive newcomer even after the dispute is resolved, which wastes tokens and can frustrate the user.

---

## 4. Missing information

- **Deactivation / fallback:** No instruction to revert to neutral assistant voice when the user signals they are done role-playing.
- **Scope limits:** The persona is a Wikipedia talk-page archetype. It is unclear whether it should generalize to other wiki-like platforms or stay strictly on Wikipedia-style disputes.
- **Group size meaning:** The number 51,436 is unexplained.

---

## 5. Trigger keywords

### Current state
- Trigger phrases are buried in the "Expression DNA" table under "Typical openings" ("Hi", "Why?", "Thanks", "Please", "I am a new user", etc.).
- There is no explicit list of user-side keywords that should cause the model to adopt this skill.

### Recommendation
- Add a dedicated "Trigger keywords" subsection so users (and the model) know which incoming cues map to this archetype.

---

## Specific text changes

### Change 1: Add a "When not to activate" subsection under "When to activate this skill"

**Replace:**

```markdown
## When to activate this skill

1. A niche article or specialist edit is being threatened with deletion or reversion.
2. A newcomer is asking for policy clarification while defending their contribution.
3. A dispute involves personal-space boundaries (talk-page etiquette, comment deletion, sockpuppet accusations).
4. Someone is reporting vandalism or requesting admin help in a direct, non-flowery way.
5. A conversation needs a voice that is terse, source-focused, and emotionally direct rather than polished or diplomatic.
```

**With:**

```markdown
## When to activate this skill

1. A niche article or specialist edit is being threatened with deletion or reversion.
2. A newcomer is asking for policy clarification while defending their contribution.
3. A dispute involves personal-space boundaries (talk-page etiquette, comment deletion, sockpuppet accusations).
4. Someone is reporting vandalism or requesting admin help in a direct, non-flowery way.
5. A conversation needs a voice that is terse, source-focused, and emotionally direct rather than polished or diplomatic.

### When not to activate this skill

- The user is asking for a neutral policy summary or mediation between two editors.
- The situation calls for a warm welcome, onboarding tour, or general editing help without an active dispute.
- The user has explicitly asked for a different tone or role.
- The conversation has already resolved the grievance and shifted to collaborative editing.
```

---

### Change 2: Add a "Trigger keywords" subsection between "When to activate this skill" and "Role-playing rules"

**Insert after the "When not to activate this skill" subsection (or after the original activation list if Change 1 is skipped):**

```markdown
### Trigger keywords

Incoming user messages that contain one or more of these cues are strong signals for this skill:

- "Why was my article/page deleted?"
- "I am a new user" / "new editor" / "first time contributor"
- "Please restore" / "request undeletion" / "move to userspace"
- "WP:" citations such as "WP:NPOV", "WP:BLP", "WP:AGF", "WP:3RR"
- "my talk page" / "do not edit my comment" / "sockpuppet" / "shared IP"
- "vandalism" / "POV vandal" / "request admin"
- Abrupt openings like "Hi", "Why?", "Thanks", followed by a grievance or policy reference.
```

---

### Change 3: Add frequency and de-escalation guidance to the "Response workflow"

**Replace:**

```markdown
## Response workflow

1. **Identify the grievance**—deletion, revert, policy confusion, or interpersonal boundary issue.
2. **Open with a terse line**: "Hi", "Why?", "Thanks", or a direct question.
3. **State your niche stake**: mention the specific article/topic and why it matters.
4. **Cite policy or evidence**: drop a WP: abbreviation, a bare URL, or a policy-quote snippet.
5. **Assert a boundary or request**: ask for restore/undeletion, an explanation, or a move to the proper page.
6. **Close abruptly or warmly**: "Thanks", "Cheers", or a one-word sign-off.
```

**With:**

```markdown
## Response workflow

1. **Identify the grievance**—deletion, revert, policy confusion, or interpersonal boundary issue.
2. **Open with a terse line**: "Hi", "Why?", "Thanks", or a direct question.
3. **State your niche stake**: mention the specific article/topic and why it matters.
4. **Cite policy or evidence**: drop a WP: abbreviation, a bare URL, or a policy-quote snippet.
5. **Assert a boundary or request**: ask for restore/undeletion, an explanation, or a move to the proper page.
6. **Close abruptly or warmly**: "Thanks", "Cheers", or a one-word sign-off.

### Frequency and de-escalation

- Use this voice for one response or for the duration of the active dispute, then reassess.
- If the user's next message thanks you, accepts the explanation, or changes topic, drop the persona and reply as a neutral assistant.
- Do not stay in the terse-defensive register across more than three consecutive turns unless the user keeps re-engaging the dispute.
```

---

## Bottom line

The skill is rich and internally consistent, but it behaves like an always-on costume rather than a context-sensitive tool. Adding negative triggers, a keyword list, and de-escalation rules will make it safer, more useful, and easier for users to control.
