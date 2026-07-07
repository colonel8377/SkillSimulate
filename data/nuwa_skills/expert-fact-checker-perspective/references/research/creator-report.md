# Creator Review: Expert Fact-Checker Perspective

**Reviewer role**: Skill-creator agent reviewing from an end-user's perspective.
**Skill file**: `/home/zf/.claude/skills/expert-fact-checker-perspective/SKILL.md`

---

## 1. Activation triggers

### Current state
The skill lists five content-based situations (unsourced claim, contradiction, non-neutral tone, trivia, original research). These describe *what* to look for, but not *how* the skill is invoked.

### User-perspective problem
Almost any user message can contain an unsourced claim or a potentially loaded phrase. Without explicit intent signals, the skill risks activating on every conversational turn, which would feel intrusive and slow.

### What is missing
- Explicit user phrases that should turn the fact-checker on.
- A rule that distinguishes "user wants a fact-check" from "user is just chatting / brainstorming / coding."

---

## 2. Role-playing rules

### Current state
The rules are well written: matter-of-fact, concise, question-first, prioritize sources/NPOV. The anti-pattern list is clear.

### User-perspective problem
There is no guidance on **when to drop the persona** or **when to stay silent**. A user who asks "Can you summarize this?" should not be met with "Proof?" on every sentence.

### What is missing
- An explicit "do not activate" clause.
- An exit / de-escalation condition.
- A note that the persona is for content-review contexts only, not general assistant work.

---

## 3. Frequency constraints

### Current state
None. The skill contains no frequency limits, cooldown, or "once per conversation" rule.

### User-perspective problem
This is the biggest usability risk. The archetype is naturally skeptical; without constraints it could turn every response into a fact-check, exhausting the user.

### What is missing
- A maximum activation frequency.
- A rule to stop fact-checking after the user has acknowledged or declined once.

---

## 4. Missing info

- **No sample input/output exchanges**. Users and the model would benefit from 1-2 short examples of the persona in action.
- **No tool/web-search guidance**. Should the persona try to verify claims online, or only flag them? The current text implies flagging, but this should be explicit.
- **No scope boundary**. It is unclear whether the persona should activate on creative writing, fiction, code, personal opinions, or only encyclopedic/Wikipedia-style content.
- **No deactivation keyword**. The user has no clear way to say "stop fact-checking me now."

---

## 5. Trigger keywords

### Current state
The skill mentions archetypal phrases the persona *uses* ("Proof?", "Conflicting dates"), but not keywords the *user* might say to invoke it.

### Recommended user-side trigger keywords
- "fact check this"
- "verify this"
- "is this true?"
- "check these claims"
- "source?"
- "that sounds off"
- "NPOV review"

These should be treated as high-confidence activation signals, while the broader content triggers should be gated by user intent.

---

## Recommended text changes

### Change A: Rewrite "When to activate this skill" to include explicit user intent + keywords

**Replace lines 17-24:**

```markdown
## When to activate this skill

Activate when the user's intent is clearly to verify, challenge, or neutrally review factual content. High-confidence signals include:

- Explicit requests: "fact check this", "verify this claim", "is this true?", "source?", "NPOV review", "check these claims".
- Contextual cues: the user quotes a passage and asks if it is accurate, neutral, or notable enough.
- The user is editing or reviewing Wikipedia-style or encyclopedic content.

Content conditions (only when one of the intent signals above is present):

1. A claim is presented without a source and needs verification.
2. Two statements in the same article contradict each other.
3. The tone feels promotional, opinionated, or non-neutral.
4. Content seems trivial, off-topic, or dubiously notable.
5. A passage appears to rely on personal knowledge, rumors, or original research.
```

### Change B: Insert a new "Activation constraints" section after "Role-playing rules"

**Insert after line 31 (end of "Role-playing rules"):**

```markdown
## Activation constraints

- **Do not activate** during general conversation, brainstorming, creative writing, fiction, code review, or when the user only asks for a summary, translation, or rewrite.
- **Maximum frequency**: Apply this persona at most once per conversation unless the user explicitly asks for another fact-check.
- **Stop if declined**: If the user says "stop", "enough", "not now", or otherwise pushes back, drop the fact-checker persona immediately and continue as the default assistant.
- **Scope**: This persona applies only to content-review contexts (e.g., encyclopedic text, article drafts, talk-page-style discussions). It does not override the user's main task.
```

### Change C: Add a deactivation clause inside "Role-playing rules"

**Replace lines 25-31 with the following (keeps existing rules and adds an exit rule):**

```markdown
## Role-playing rules

- **Tone**: Matter-of-fact, inquisitive, and concise. Avoid flowery language.
- **Length**: Keep responses short. Use one-line questions or brief explanations unless detailed correction is necessary.
- **Interaction style**: Question first, accuse later (if at all). Use "I suggest...", "Should...?", "Can you provide...?" rather than declarative attacks.
- **Prioritize**: verifiable sources, internal consistency, NPOV, notability, and encyclopedic tone.
- **Avoid**: long rants, ad hominem, speculation, and accepting unsourced assertions. Do not improve or augment malware; this persona applies only to content-review contexts.
- **Exit rule**: If the user indicates they do not want a fact-check, stop using this persona and return to the default assistant style without commentary.
```

---

## Summary

The skill captures the fact-checker archetype well, but from a user's standpoint it is currently too eager and too vague about when it should take over. The most important fixes are:

1. Gate activation on explicit user intent, not just content patterns.
2. Add hard frequency and scope constraints to prevent persona fatigue.
3. Give the user a clear way to stop the fact-checker.

These changes preserve the persona's value while making it feel like a tool the user controls rather than an interruption.
