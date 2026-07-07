# Skill Creator Review: Confrontational Editor

**Skill file**: `/home/zf/.claude/skills/confrontational-editor-perspective/SKILL.md`
**Review date**: 2026-07-07
**Reviewer role**: Skill-creator agent, evaluating from an end-user's perspective.

## Overall assessment

The skill is a well-documented behavioral archetype with clear identity, voice rules, mental models, and anti-patterns. It successfully avoids slurs, threats, and hate speech while still conveying hostility. However, from a user's point of view it is missing the practical "how do I invoke / stop / limit this?" information that makes a skill usable in a shared Claude Code environment.

## 1. Activation triggers

**Current state**: The "When to activate this skill" section lists 5 abstract scenarios (simulating, stress-testing, modeling, testing moderation, generating role-played comments). It does not tell the user what words or request patterns will actually switch Claude into this persona.

**Problem**: A user reading the skill cannot predict whether typing "be confrontational" will trigger it, or whether only the literal phrase "confrontational editor" works. This ambiguity leads to failed invocations or surprise activations.

## 2. Role-playing rules

**Current state**: Rules are concrete and safe — blunt language, harsh adjectives, policy-as-weapon, typos, no slurs/threats/hate speech, prioritize "truth" over diplomacy.

**Problem**: There is no deactivation or "drop the mask" rule. Once activated, the skill gives no instruction on how Claude should exit the persona if the user wants normal assistance mid-conversation. Also, the rule "Do not use slurs, threats of violence, hate speech, or explicit profanity" is good but could be reinforced with an explicit escalation boundary.

## 3. Frequency constraints

**Current state**: None. The skill can in principle fire on every relevant-looking prompt.

**Problem**: Without constraints, the persona can leak into ordinary feedback requests. A user who asks "Is this article neutral?" might get a hostile rant instead of a balanced assessment, because the skill interprets the prompt as "stress-test neutrality." There is no guardrail such as "only when explicitly requested" or "offer the adversarial view as an option, not a default."

## 4. Missing info

- **Trigger keywords / invocation phrases**: no explicit list.
- **Frequency / opt-in constraints**: no limits or default-off rule.
- **Deactivation protocol**: no instruction for exiting the persona.
- **Example outputs**: no sample talk-page comment to calibrate tone and length.
- **Skill compatibility / stacking**: no note on whether it should combine with other skills or override them.
- **User safety override**: no phrase the user can say to immediately stop the roleplay.

## 5. Trigger keywords

The skill currently lacks a trigger-keyword section. Candidate keywords derived from the identity card and use cases include:

- "confrontational editor"
- "hostile editor"
- "adversarial editor"
- "Wikipedia talk-page warrior"
- "bias hunter persona"
- "play a toxic Wikipedian"
- "stress-test this article for bias"
- "simulate an uncivil editor"

Without explicit keywords, the skill relies on semantic matching, which is unreliable.

## Recommended text changes

### Change 1: Add explicit trigger keywords and default-off constraint

Insert the following section immediately after "## When to activate this skill":

```markdown
## Trigger keywords

Activate only when the user explicitly uses one of these phrases or clear equivalents:

- "confrontational editor"
- "hostile editor"
- "adversarial editor"
- "Wikipedia talk-page warrior"
- "bias hunter persona"
- "play a toxic Wikipedian"
- "simulate an uncivil editor"
- "stress-test this article for bias"

**Default behavior**: If the user asks for ordinary feedback, neutrality review, or copy-editing, do not use this persona. Provide balanced, constructive feedback first; offer the confrontational perspective as an optional adversarial stress-test only if the user accepts or requests it.
```

### Change 2: Add deactivation and escalation boundaries to role-playing rules

Append the following two bullet points to "## Role-playing rules":

```markdown
- **Stop immediately** if the user says any of: "stop the roleplay", "drop the persona", "normal mode", "stop being confrontational", or similar. Revert to standard helpful Claude without the hostile voice.
- Stay within the existing red lines (no slurs, threats, hate speech, explicit profanity). If the user asks you to cross those lines, refuse and remain in the bounded confrontational style.
```

### Change 3: Add a brief example to calibrate tone

Insert the following section immediately after "## Expression DNA":

```markdown
## Example opening

> This article is a mess. Who wrote this crap? The lead reads like a press release, half the "sources" are blogs, and the obvious NPOV problems have been sitting here for years. If this is supposed to be encyclopedic, someone needs to rewrite it from scratch or delete the whole thing. Stop pretending polite edits will fix biased garbage.

This example illustrates the expected bluntness, policy references, and limited hostility while staying within the no-slur/no-threat boundary.
```

## Summary

The skill is strong on character definition but weak on user control. Adding explicit trigger keywords, a default-off constraint, a deactivation phrase, and one example would make it safer and more usable without weakening the archetype.
