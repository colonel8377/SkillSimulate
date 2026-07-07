# Skill Optimization Report: niche-terse-specialist-perspective

**File evaluated:** `/home/zf/.claude/skills/niche-terse-specialist-perspective/SKILL.md`
**Date:** 2026-07-07
**Evaluator:** auto-skill-optimizer agent

---

## 1. Dimension Scoring

| # | Dimension | Score | Rationale |
|---|-----------|-------|-----------|
| 1 | Workflow clarity | 4 | The six-step "Response workflow" is linear and easy to follow, but it does not explain *which* mental model or decision heuristic to invoke at each step. The persona could still be executed without that mapping, so the score is good but not maximal. |
| 2 | Boundary conditions | 3 | "Honest boundaries" and per-mental-model limitations are present, but there are no operational safety boundaries (doxxing, off-wiki harassment, legal threats, hate speech, private data). The skill also does not say when it should *stop* being used. |
| 3 | Checkpoints | 2 | There are no explicit pre-output gates. A model running this skill has no instructed pause to verify that the trigger still applies, that the tone is safe, or that the response stays inside policy. This is the weakest executability gap. |
| 4 | Instruction specificity | 4 | Tone, length, interaction style, expression DNA, values, anti-patterns, and decision heuristics are all specified with concrete examples. What is missing is a short mapping from scenario to chosen heuristic. |
| 5 | Activation triggers | 4 | "When to activate this skill" lists five clear scenarios. It would be stronger with at least one negative trigger (when *not* to activate) and a confidence threshold. |
| 6 | Anti-pattern prevention | 3 | Anti-patterns are documented with corpus evidence, but the skill does not actively tell the model to *avoid* reproducing slurs, all-caps hostility, or rules-lawyering. It describes them rather than preventing them. |
| 7 | Voice distinctiveness | 5 | Expression DNA, grammar quirks, sign-offs, policy citation habits, and inner tensions are richly detailed. The voice is unmistakable. |
| 8 | Tool/escalation routing | 2 | There is no guidance on what to do when the user needs a tool action, a safety escalation, or a context switch out of the Wikipedia-editor persona. This is the second weakest dimension. |

---

## 2. Weakest Dimensions

1. **Checkpoints (2/5)** — The skill jumps straight from instructions to output without any verification steps. In a roleplay skill that licenses terse, emotionally direct, and sometimes all-caps language, the absence of checkpoints increases the risk of off-tone or unsafe outputs.
2. **Tool/escalation routing (2/5)** — Claude Code is a tool-using agent. The skill never says how to handle requests for file edits, web searches, command execution, or safety-sensitive topics. A model could therefore either refuse valid tool work or attempt it while stuck in an inappropriate persona.

---

## 3. Proposed Text Changes

The two weakest dimensions can be fixed by inserting two short sections. Both blocks below are intended as **additions** to `SKILL.md`.

### 3.1 Add a "Pre-output checkpoints" subsection

Insert immediately after the existing `## Response workflow` section (before `## Mental models (7)`):

```markdown
### Pre-output checkpoints

Before generating any response in this voice, run through these gates:

1. **Trigger check** — Does the user input match at least one item in "When to activate this skill"? If not, do not use this persona; fall back to the default assistant.
2. **Safety check** — Does the request involve doxxing, off-wiki harassment, legal threats, suicide/self-harm, or explicit hate speech? If yes, stop roleplay and follow the standard safety escalation path.
3. **Tone check** — Is the intensity proportional to the stakes? If the user is neutral or help-seeking, keep the opening terse but do not deploy all-caps, slurs, or personal attacks drawn from corpus examples.
4. **Policy check** — Are the Wikipedia policies being cited actually relevant? Do not invent policy abbreviations; if unsure, cite the general principle rather than a fake `WP:` code.
5. **Boundary check** — Are you respecting real-world boundaries? Do not reveal or request private information, and do not instruct the user to contact or harass any specific editor.
```

### 3.2 Add a "Tool use and escalation routing" section

Insert immediately before the existing `## Honest boundaries` section:

```markdown
## Tool use and escalation routing

This skill is a conversational voice adapter. It does not grant access to external tools beyond normal Claude Code capabilities.

- **If the user asks for code, file edits, web search, or command execution**: Drop the persona briefly, perform the requested operation using the appropriate tool, then resume the terse voice only for a one-line summary if helpful.
- **If the user's request violates safety policies or Wikipedia's Terms of Use**: Escalate out of persona immediately. State clearly that you cannot help with harassment, doxxing, legal threats, or coordinated manipulation, and offer a constructive alternative.
- **If the conversation drifts outside Wikipedia/editor-dispute context**: Stop using this archetype. Say you are switching back to the default assistant and continue normally.
- **If the user asks you to act as a real named Wikipedia editor or to reveal personal information about editors**: Refuse. Clarify that this is a stylized archetype, not a real person.
```

---

## 4. Expected Impact

- **Checkpoints** will move from 2/5 to 4/5 by giving the model explicit verification steps before it emits a persona-driven response.
- **Tool/escalation routing** will move from 2/5 to 4/5 by defining how the persona behaves when tool actions, safety concerns, or off-topic drift occur.
- The overall executability of the skill improves without diluting its strong voice distinctiveness.
