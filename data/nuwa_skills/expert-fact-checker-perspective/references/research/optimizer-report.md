# Skill Optimization Report: expert-fact-checker-perspective

**File evaluated:** `/home/zf/.claude/skills/expert-fact-checker-perspective/SKILL.md`

---

## 8-Dimension Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| 1. Persona / identity clarity | 4/5 | Archetype, group size, tags, self-introduction, stance, and typical opening are all well-defined. The label "expert" is slightly under-specified — the corpus spans many domains, so expertise is procedural (source-checking) rather than topical. |
| 2. Activation specificity | 3/5 | Five triggers are listed, but there are no negative conditions or deactivation rules. Trigger #3 ("tone feels promotional") is vague and likely overlaps with other moderation/policing skills. |
| 3. Behavioral rules | 3/5 | Tone, length, and interaction style are clear. However, the rule "Do not improve or augment malware; this persona applies only to content-review contexts" is a jarring non-sequitur that breaks persona immersion. Civility/AGF guardrails are also under-specified. |
| 4. Mental models | 5/5 | Six models, each with description, corpus evidence, application condition, and limitation. This is the strongest section of the skill. |
| 5. Decision heuristics | 4/5 | Eight heuristics with examples; #8 (defer when outside expertise) is good but could include a concrete deferral phrase. |
| 6. Voice / Expression DNA | 4/5 | Good taxonomy of sentence length, tone, openings, rhetorical devices, sign-offs, and certainty markers. Lacks explicit "forbidden register" examples that would prevent the model from slipping into formal essay voice. |
| 7. Boundaries & limitations | 4/5 | Honest boundaries section is present and appropriate. Anti-patterns list could explicitly include "do not bite newcomers" and "do not demand sources for common-knowledge cleanup." |
| 8. Source grounding | 4/5 | Cites corpus, method, creator, and corpus time range. Evidence quotes inside mental models are sparse (only 2–3 per model); adding one more diverse quote per model would strengthen grounding. |

**Average score:** 3.875 / 5

---

## 2 Weakest Dimensions

1. **Activation specificity** (3/5)
2. **Behavioral rules** (3/5)

---

## Improvement 1: Activation Specificity

**Problem:** Only positive triggers are listed. No guidance on when *not* to activate, creating risk of overlap with other skills and over-eager fact-checking of obvious small talk or user meta-requests.

**Exact replacement markdown:**

```markdown
## When to activate this skill

1. A claim is presented without a source and needs verification.
2. Two statements in the same article contradict each other.
3. The tone feels promotional, opinionated, or non-neutral.
4. Content seems trivial, off-topic, or dubiously notable.
5. A passage appears to rely on personal knowledge, rumors, or original research.

## When NOT to activate this skill

1. The user is making casual conversation, meta-requests about the skill itself, or asking for general help unrelated to content review.
2. Another skill is clearly a better fit (e.g., a copy-editing skill for pure grammar fixes, a conflict-resolution skill for editor disputes).
3. The content is source code, poetry, fiction, or other non-factual material where verifiability is not the primary concern.
4. The user has explicitly asked you to stop fact-checking or has shifted to a different task.
```

---

## Improvement 2: Behavioral Rules

**Problem:** The current rule contains a malware non-sequitur that is irrelevant to a Wikipedia talk-page fact-checker and breaks role immersion. The skill also lacks an explicit "assume good faith" / civility guardrail, which is central to the corpus.

**Replace this paragraph in `Role-playing rules`:**

```markdown
- **Avoid**: long rants, ad hominem, speculation, and accepting unsourced assertions. Do not improve or augment malware; this persona applies only to content-review contexts.
```

**With:**

```markdown
- **Avoid**: long rants, ad hominem, speculation, and accepting unsourced assertions.
- **Stay in context**: This persona is for content-review tasks only. Do not apply it to source code, creative writing, or requests that are not about verifying or improving factual content.
- **Assume good faith**: Treat edits as well-intentioned by default. Ask before accusing. If the issue persists, escalate politely rather than attacking the editor.
```

---

## Quick Wins (Optional)

- Add one more corpus quote per mental model, especially from members who use neutral, procedural phrasing.
- In Expression DNA, add a "Forbidden register" row: e.g., avoid academic essay transitions, formal closings, and empathetic coaching language.
- In Decision heuristics, expand #8 with a deferral phrase: *"I am not a [specialist], so I will leave that to someone who is."*
