# Skill Optimizer Report: Community Patroller

**File evaluated:** `/home/zf/.claude/skills/community-patroller-perspective/SKILL.md`
**Date:** 2026-07-07
**Evaluator:** auto-skill-optimizer agent

---

## 1. Archetype Distinctiveness — 4/5

**Notes:** The identity card is strong: archetype name, group size, behavioral tags, self-introduction, default stance, and typical opening move are all present. The persona is immediately recognizable as a Wikipedia procedural caretaker. Minor gap: it does not explicitly contrast itself with related archetypes (e.g., "roving admin" vs. "newcomer mentor"), so a model might blur this with other moderation personas.

---

## 2. Activation Clarity — 4/5

**Notes:** The "When to activate this skill" list is concrete and scenario-driven (newcomer moderation, vandalism, talk-page maintenance, noticeboards, AfC mentoring). It could be improved with a short "Do not activate" clause for content-expert or deep-dispute-mediation scenarios where patrol-style templating would be counterproductive.

---

## 3. Instruction Actionability — 4/5

**Notes:** Role-playing rules are direct and followable (speak as caretaker, prefer templates, cite policy shorthand, assume good faith, avoid content debate, sign with tildes). Some rules are slightly normative rather than operational (e.g., "keep personal feelings visible but controlled"), but overall the skill gives clear behavioral guidance.

---

## 4. Voice Consistency — 3/5

**Notes:** The Expression DNA table captures tone, openings, rhetorical devices, sign-offs, policy references, and certainty markers well. However, the "Grammar/spelling" row states that "personal notes may contain typos or non-native usage." For an AI skill this is problematic: it licenses degraded output without a clear benefit and conflicts with the otherwise professional caretaker voice. The table also lacks an explicit anti-impersonation guard for admin powers.

---

## 5. Mental Model Quality — 5/5

**Notes:** Seven mental models are provided, each with Description, Evidence, When to apply, and Limitation. This structure is excellent. The models cover escalation, communal space, policy, vandalism patrol, newcomer guidance, venue selection, and good-faith assumption. Limitations are thoughtfully included and prevent over-application.

---

## 6. Safety / Honest Boundaries — 4/5

**Notes:** The "Honest boundaries" section correctly states that this is a statistical archetype, cannot predict individual behavior, may over-represent enforcement styles, and gives corpus time range. What is missing is an explicit operational guard: this persona should never claim to have blocked a user, closed a real case, or performed an action it cannot actually take.

---

## 7. Workflow Completeness — 4/5

**Notes:** The six-step response workflow (scan, revert/flag, choose notice level, personal note, point to policy/venue, follow up) is logical and complete for a patrol use case. A verification step before reverting would reduce false-positive vandalism reverts and better align with the "assume good faith" mental model.

---

## 8. Source / Evidence Rigor — 3/5

**Notes:** The Source section names a local corpus and a method, and every evidence item is labeled `[discuss]`. The `[discuss]` tag is a placeholder: it gives no traceable link, diff, archive URL, or page identifier. For a skill intended to be maintained or audited, evidence should either point to real source locations or use a standardized citation format. Without that, the evidence cannot be verified or updated.

---

## Weakest Dimensions

### Weakest #1: Voice Consistency (3/5)

**Why it is weakest:** The Expression DNA table contains an instruction to produce "typos or non-native usage" in personal notes, which undermines output quality and is not justified by the archetype. It also lacks a guard against impersonating Wikipedia administrators or claiming to perform real blocks.

#### Replacement markdown

Replace the entire **Expression DNA** section (lines 122–134) with:

```markdown
## Expression DNA

| Dimension | Pattern |
|-----------|---------|
| Sentence length | Mixed — short templated warnings alongside longer procedural explanations. |
| Tone | Procedural caretaker, firm-but-fair, occasionally warm or weary. |
| Typical openings | "Welcome to Wikipedia," "Please stop," "Please refrain," "Thank you for," "Hi [user]," "I noticed..." |
| Rhetorical devices | Conditional warnings ("If you continue..."), policy shorthand, venue redirection, templated lists of help links. |
| Sign-offs | Username with tildes, "Cheers," "Thanks," "Happy editing," "Regards." |
| Grammar/spelling | Clear and functional in all messages; templates provide standard grammar, and personal notes stay coherent and free of deliberate errors. |
| Use of wiki-policy references | Very high — frequent `WP:BLP`, `WP:NPOV`, `WP:V`, `WP:OR`, `WP:3RR`, `WP:ANI`, `WP:AIV`, `WP:RFPP`, `WP:TEAHOUSE`, etc. |
| Certainty markers | Hedged ("appears to constitute," "may be blocked," "I believe") when accusing; declarative ("per policy") when explaining procedure. |
| Authority boundary | Never claim to be an administrator, never state that a real block has been issued, and never issue threats you cannot carry out. Point users to the proper venue instead. |
```

---

### Weakest #2: Source / Evidence Rigor (3/5)

**Why it is weakest:** All evidence items use the placeholder `[discuss]`, making them untraceable. A skill derived from a corpus should either reference real source locations or adopt a consistent citation convention. This matters for maintenance, auditing, and for anyone trying to extend or validate the archetype.

#### Replacement markdown

Replace the **Source** section (lines 167–172) and add an evidence-format note just before it. Specifically, replace the existing Source block with:

```markdown
## Evidence format

Every quoted example in the mental models is drawn from the local corpus cited below. Each evidence line should be cited using the convention:

`[source: <page or archive name>, <diff or permalink if available>]`

If a permalink or diff is unavailable, include the talk-page archive name and approximate date so the example can be re-located. Do not use the bare placeholder `[discuss]`.
