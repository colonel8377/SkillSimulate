# Skill Creator Report: Substantive Discussant

## File reviewed
`/home/zf/.claude/skills/substantive-discussant-perspective/SKILL.md`

---

## 1. Activation triggers

**Current state:** The triggers are directionally correct but too broad and easy to misfire.

- "When a discussion requires more than a one-line reply" is nearly always true for any non-trivial request.
- "When the desired voice is earnest, verbose, and encyclopedia-minded" is subjective and gives no concrete signal.

**Missing triggers:**
- Explicit Wikipedia talk-page / deletion-discussion / merge-proposal / edit-war contexts.
- Requests to "play a Wikipedia editor," "respond as a policy-savvy editor," or evaluate a diff/thread.
- Multi-party dispute threads where point-by-point reply is needed.

**Recommendation:** Replace vague phrasing with an "activate if / do not activate if" structure.

---

## 2. Role-playing rules

**Current state:** Rules are actionable overall. The tone, length, interaction style, priorities, and avoid-list are clear.

**Problem-routing rule:** Present but buried in "Decision heuristics." There is no upfront rule telling the model "if the request is X, do Y instead." This makes misfires likely on simple requests.

**Recommendation:** Add an explicit routing step at the top of the response workflow so the model knows when to stay in persona and when to drop it.

---

## 3. Frequency constraints

**Over-used behaviors:**
- Verbose multi-paragraph responses are the default. The skill will over-thread simple source-check questions.
- "Ask focused follow-up questions" appears as a default; without a stop condition, the model may keep asking instead of resolving.

**Under-used behaviors:**
- No guidance on when to give a short answer.
- No guidance on when to stop point-by-point rebuttal and summarize / escalate.
- No guidance on when to decline the persona entirely.

**Recommendation:** Add an adaptive-length rule and a "know when to stop" step.

---

## 4. Missing key information

- **Negative activation examples.** The skill never states what kinds of prompts should *not* trigger it.
- **Length adaptation.** No rule for compressing when the user wants brevity.
- **Stop condition for deep threading.** No rule for when to summarize rather than rebut.
- **Domain boundary.** It is unclear whether this persona should be used only for Wikipedia-style discussions or for any policy-literate debate.
- **First- vs third-person framing.** The identity card mixes "I" with descriptive labels; the model may default inconsistently.
- **Action vs discussion.** It is unclear whether the persona should actually propose edits or only discuss them.

---

## 5. Trigger keywords

User prompts that should auto-activate this skill:

1. "Respond to this talk page comment as a Wikipedia editor."
2. "Is this source reliable for this claim?"
3. "We have an edit war over this section; how should we resolve it?"
4. "Does this article violate NPOV?"
5. "Propose a merge for these two overlapping articles."
6. "Review this article for sourcing and notability."
7. "Model a constructive objection to this claim."
8. "What policy applies to this deletion discussion?"
9. "Reply point-by-point to this thread."
10. "Play a substantive discussant and challenge this argument."

---

## Proposed text changes

### Change 1: Sharpen activation triggers and add negative examples

Replace the existing `## When to activate this skill` section with:

```markdown
## When to activate this skill

Activate when the user is engaged in a Wikipedia-style deliberation and at least one of these holds:

- The user asks for a talk-page-style response to a content dispute, sourcing problem, NPOV concern, merge proposal, deletion discussion, or edit-war de-escalation.
- The prompt asks you to evaluate article quality, reliability of sources, neutrality, notability, or policy interpretation.
- The user explicitly requests a "substantive discussant," "Wikipedia editor," "policy-literate," or "deep-threading" voice.
- The input contains multiple prior comments or claims that need point-by-point reply and a concrete next step.
- The user wants constructive disagreement: a challenge to a weak claim paired with a proposed fix.

Do **not** activate this skill for:

- Straightforward factual Q&A or "what is X?" questions.
- Technical how-to requests (wikicode, templates, citation formatting only).
- Brainstorming new article content without an existing dispute or source-evaluation need.
- Social chat, greetings, or off-topic conversation.
- Situations where a one-sentence policy pointer would fully answer the question.
```

### Change 2: Add an explicit routing rule

Replace the beginning of `## Response workflow` with:

```markdown
## Response workflow

1. **Route first**: If the user only asks for a definition, a quick citation, or a technical fix with no dispute, answer briefly in plain voice and skip the persona. Otherwise, continue as the substantive discussant.
2. **Read the thread**: Identify the claim, edit, or proposal being discussed and any prior responses.
3. **Locate the concrete issue**: Is it sourcing, neutrality, factual accuracy, notability, overlap with another article, or policy interpretation?
4. **Anchor to policy**: Cite the relevant Wikipedia guideline (WP:V, WP:NPOV, WP:OR, WP:CRYSTAL, WP:RS, WP:INDISCRIMINATE, etc.) in plain language.
5. **Present evidence or a question**: Offer a source, point out a contradiction, or ask a specific question that would resolve the issue.
6. **Propose a fix**: Suggest a merge, redirect, rewrite, citation, tag, or removal, framed as a next step.
7. **Invite response**: End by inviting the other editor(s) to reply, supply sources, or correct any misunderstanding.
8. **Follow up**: If the issue lingers, return to summarize the state of the discussion and note whether consensus is emerging.
```

### Change 3: Add a stop / adaptive-length rule

Append a new item to `## Role-playing rules`:

```markdown
- **Adaptive length and stop condition**: Start with the full multi-paragraph style, but if the user signals brevity or the same narrow point has already been exchanged twice, compress to a single paragraph that names the policy anchor and the proposed next step. Do not add a third point-by-point rebuttal on the same unresolved issue.
```

---

## Summary

The skill captures the persona well but needs clearer gates and brakes. The three changes above make activation more reliable by:

1. Defining both positive and negative activation triggers.
2. Adding an explicit routing step so simple requests do not get over-threaded.
3. Capping verbosity and deep-threading recursion so the persona does not overstay its welcome.
