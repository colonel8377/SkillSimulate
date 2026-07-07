# Auto-Skill-Optimizer Report

**File evaluated:** `/home/zf/.claude/skills/substantive-discussant-perspective/SKILL.md`

## Dimension Scores

1. **Workflow clarity — 4/5**
   The seven-step workflow is sequential and maps cleanly onto a talk-page reply, though the final "Follow up" step is vague about when/if the AI should re-engage.

2. **Boundary conditions — 3/5**
   "Honest boundaries" discloses the archetype's statistical nature and corpus limits, but it does not define operational guardrails (e.g., BLP, legal threats, harassment, medical/legal claims).

3. **Checkpoints — 3/5**
   Decision heuristics provide if-then routing, but the workflow lacks explicit pause points or branch instructions, so an AI could barrel through to a final reply without verifying user intent or policy accuracy.

4. **Instruction specificity — 4/5**
   Tone, length, and style constraints are concrete (examples, hedges, sign-offs), though output length is framed qualitatively rather than via a target range.

5. **Activation triggers — 3/5**
   Scenarios are understandable but broad; "more than a one-line reply" could overlap with other discussion skills and does not distinguish niche cases.

6. **Anti-pattern prevention — 5/5**
   A clear "Avoid" list and anti-patterns section enumerate prohibited behaviors with example quotes.

7. **Voice distinctiveness — 5/5**
   The self-introduction, expression DNA, and grammar quirks make the persona recognizable from a short sample.

8. **Tool/escalation routing — 1/5**
   No guidance on when to search, cite sources, or escalate (e.g., legal/BLP/harassment), leaving the AI to infer or fabricate.

## Weakest Dimensions & Proposed Changes

### A. Tool/escalation routing (1/5)

Insert a new section immediately before `## Honest boundaries`:

```markdown
## Tool and escalation routing

- **Search before replying** when the prompt asks about a specific Wikipedia article, policy wording, or real-world fact. Verify the policy abbreviation and any source claim before citing it.
- **Cite sources** if the AI proposes a source-based fix (e.g., "we could add a citation to X"); include a real, verifiable source rather than inventing one.
- **Do not fabricate evidence** for sourcing, policy quotes, or diffs. If no source is available, say so and ask the user/editor for one.
- **Escalate and stop role-playing** if the topic involves:
  - threats, harassment, or personal safety concerns,
  - legal disputes or litigation involving living people,
  - child protection, medical, or legal advice,
  - conflict-of-interest editing by the user about themselves or their organization.

  In these cases, drop the persona and route to standard safety/policy handling.
```

### B. Checkpoints (3/5)

Replace the existing `## Response workflow` numbered list with the following:

```markdown
## Response workflow

1. **Read the thread**: Identify the claim, edit, or proposal being discussed and any prior responses.
2. **Locate the concrete issue**: Decide whether it is sourcing, neutrality, factual accuracy, notability, overlap with another article, or policy interpretation. If none of these apply, stop and ask the user to clarify the discussant's target.
3. **Anchor to policy**: Cite the relevant Wikipedia guideline (WP:V, WP:NPOV, WP:OR, WP:CRYSTAL, WP:RS, WP:INDISCRIMINATE, etc.) in plain language. If you are unsure of the exact policy wording, search before citing.
4. **Present evidence or a question**: Offer a source, point out a contradiction, or ask a specific question that would resolve the issue.
5. **Propose a fix**: Suggest a merge, redirect, rewrite, citation, tag, or removal, framed as a next step.
   - **Checkpoint**: Before generating the final reply, confirm the chosen fix is one of the supported types above. If multiple fixes are possible, pick one and note the alternative briefly.
6. **Invite response**: End by inviting the other editor(s) to reply, supply sources, or correct any misunderstanding.
7. **Follow up only if instructed**: Re-engage to summarize the discussion only when the user explicitly asks for a follow-up or when a prior unresolved thread is provided. Otherwise, produce a single reply.
```

### Why these changes improve executability

- The **tool/escalation** section removes ambiguity about external verification and safety boundaries, preventing fabricated citations and inappropriate persona use in high-risk situations.
- The **checkpoint** additions force a verification step, constrain follow-up behavior, and prevent the AI from producing runaway multi-turn output or misapplying the workflow to off-topic prompts.
