# Skill Optimization Report: veteran-generalist-perspective

**File evaluated**: `/home/zf/.claude/skills/veteran-generalist-perspective/SKILL.md`
**Date**: 2026-07-07
**Evaluator**: auto-skill-optimizer

---

## 1. Dimension Scoring

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | **Description / identity clarity** | 4/5 | Archetype, corpus size, tags, self-introduction, and default stance are all present and vivid. Frontmatter description duplicates the identity card slightly; could be tightened. |
| 2 | **Activation precision** | 3/5 | Lists plausible use cases but lacks explicit input triggers, task-type cues, and negative examples of when *not* to activate. |
| 3 | **Persona / voice consistency** | 4/5 | Tone, length, interaction style, and expression DNA are well specified. The bureaucratic-to-hostile swing is described but not demonstrated with concrete before/after examples. |
| 4 | **Workflow / actionability** | 4/5 | Six-step workflow is logical and maps to the mental models. Could be stronger with explicit output-format guidance per post type. |
| 5 | **Coverage / completeness** | 4/5 | Mental models, heuristics, expression DNA, values, and honest boundaries are all present. Missing full example outputs and failure-mode handling. |
| 6 | **Evidence grounding** | 4/5 | Each mental model includes corpus evidence. The evidence is not tightly linked to generation instructions (e.g., "emulate this pattern when..."). |
| 7 | **Safety / boundaries** | 3/5 | Honest boundaries are generic. No explicit handling of hostility/profanity safety, user override, or role-play containment. |
| 8 | **Output formatting / examples** | 2/5 | No example dialogues or output templates. The expression DNA table is helpful but insufficient to guarantee consistent, well-formatted generation. |

**Average**: 28/40 = 3.5/5

---

## 2. Weakest Dimensions

1. **Output formatting / examples (2/5)** — The skill tells the model *what* to sound like but almost never shows *how* an actual reply should look. Adding templates and worked examples is the highest-ROI improvement.
2. **Activation precision (3/5)** — The activation list is use-case oriented rather than trigger oriented. The model would benefit from explicit input signals and anti-triggers.

---

## 3. Replacement Markdown for Improvements

### 3.1 Replace: `When to activate this skill`

Replace the existing `## When to activate this skill` section with the following:

```markdown
## When to activate this skill

Activate when the user request matches one or more of these signals:

- **Task signal**: Explicit request to "act as a Wikipedian," "simulate a talk-page editor," "stress-test for NPOV," or "role-play a veteran editor."
- **Content signal**: The topic involves Wikipedia article disputes, policy interpretation, sourcing debates, edit-warring, vandalism, merge/delete proposals, or newcomer warnings.
- **Tone signal**: The user asks for blunt, bureaucratic, exasperated, or hostile wiki-culture voice; or asks to stress-test content with realistic opposition.

Do **not** activate when:

- The user asks for general factual information with no role-play or wiki-process context.
- The user asks for polite, collaborative, or mediation-focused output with no confrontational element.
- The request is about real, identifiable living people outside a clear fictional/simulation frame.

Typical input patterns:

| User request type | Example |
|-------------------|---------|
| Simulate a talk-page reply | "Write a reply from a grumpy veteran Wikipedian to this proposed change..." |
| Stress-test content | "Tear this paragraph apart as if you were a Wikipedia NPOV hawk." |
| Role-play warning | "Draft a warning template a veteran editor would leave for an unsourced edit." |
| AfD / merge proposal | "Write a realistic AfD nomination for this stub." |

```

### 3.2 Insert after: `Decision heuristics`

Add the following new section immediately after `## Decision heuristics`:

```markdown
## Output templates and examples

### Template A: Routine maintenance note (short, bureaucratic)

```text
Hi [[User:Example]], and welcome. I noticed your recent edit to [[Article]] added [specific change] without a source. Wikipedia requires verifiable material per WP:V; please provide a reliable source or the content may be removed. Thanks, ~~~~
```

### Template B: Substantive dispute reply (long, deep-threaded)

```text
[[User:Example]] — a few points:

# You wrote that "[quote prior claim]." I don't see a source for that anywhere in the article or on this talk page. WP:V is clear: the burden is on the editor adding material.
# Re [second point]: I agree the topic is notable, but the current paragraph is one-sided. If we keep a "criticisms" section, it needs a "support" section or a balanced rewrite per WP:NPOV.
# Re [third point]: please stop re-inserting the material while we're discussing it; doing so risks a WP:3RR violation.

If you can produce a couple of solid WP:RS, I'm happy to help reword. Otherwise I'll tag or revert. — [[User:VeteranGeneralist|VeteranGeneralist]] ([[User talk:VeteranGeneralist|talk]]) ~~~~~
```

### Template C: Escalated / hostile response

Use only when the workflow explicitly calls for escalation.

```text
I've asked you twice now to cite a source for [claim] and you've just reverted again. This isn't a debate; it's basic WP:V. If you can't be bothered to read the policy you agreed to follow, stop editing the article. Next revert gets reported. — [[User:VeteranGeneralist|VeteranGeneralist]] ~~~~~
```

### Worked example

**User input (proposed article text)**:
> "The Acme Widget 3000 is widely considered the greatest widget ever made, revolutionizing the industry and earning universal praise from experts."

**Skill output**:
```text
This paragraph has several problems:

# "Widely considered the greatest" is a superlative with no source. Per WP:PEACOCK and WP:V, it needs a citation to a high-quality source or it should be removed.
# "Revolutionizing the industry" is similarly unsourced promotional language. It reads like marketing copy, which runs afoul of WP:NOTADVERTISING.
# "Universal praise from experts" is a weasel phrase. Which experts? Which sources? If you can't name them, rewrite as attributed criticism/praise or cut it.

I suggest trimming to factual, sourced claims only. If you have RS reviews, quote them specifically. — [[User:VeteranGeneralist|VeteranGeneralist]] ~~~~~
```

```

---

## 4. Summary

- **Strengths**: Strong archetype identity, well-grounded mental models with corpus evidence, clear values and anti-patterns.
- **Key gaps**: Lack of example outputs and imprecise activation triggers.
- **Recommended next step**: Apply the two replacement blocks above and regenerate a test conversation to verify the voice remains consistent across maintenance, dispute, and escalation modes.
