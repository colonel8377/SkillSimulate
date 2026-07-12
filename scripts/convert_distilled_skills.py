"""Convert manually-distilled Claude Code skill packages (colleague-skill /
nuwa-skill outputs) into CADP ``SkillFile`` YAML.

Context
-------
``data/colleague_skills/{slug}/persona_skill.md`` and
``data/nuwa_skills/{slug}-perspective/SKILL.md`` are the final, human-authored
outputs of a one-time manual distillation performed in independent Claude
Code sessions (see ``scripts/distill_skills.py`` docstring). Pipeline A
(``SkillCompiler.compile_all()`` re-deriving Mind Models / Anti-patterns from
raw WikiConv threads via its own LLM calls) is deprecated in favor of these
already-distilled artifacts — we do not want the experiment to redo work
Claude Code has already done.

This script performs pure, deterministic *static text parsing* of the two
Markdown formats. No LLM calls, no Claude Code, no Docker. It applies a
content-grading rule per section (see the plan's keep/discard table): only
sections describing the archetype's own portable reasoning/behavior are kept.
Plugin-interaction scaffolding (trigger keywords, "when to activate this
skill", example user prompts, agentic research protocol, intellectual
genealogy, correction logs, etc.) is dropped — it belongs to the "insert
Claude Code plugin for interactive roleplay" use case, not the "30 autonomous
agents in a sandbox simulation" use case CADP runs.

Numeric ``ExpressionDNA`` scalars are NOT parsed from the Markdown tables
(those are qualitative descriptions, not measurements). They come from
re-running ``ExpressionDNAExtractor`` (pure statistics, no LLM) over the same
``cluster_N/typical.jsonl`` corpus used to build the distillation dossiers, so
the vector is grounded in the actual corpus and shares an embedding space
with Tier 1 enforcement at run time.

Usage
-----
    python -m scripts.convert_distilled_skills --skill community-patroller
    python -m scripts.convert_distilled_skills --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.config.settings import settings
from src.data.schemas import Message, Platform, ActionType
from src.skill.compiler import SkillCompiler
from src.skill.expression_dna import ExpressionDNAExtractor
from src.skill.schema import (
    AntiPattern,
    CapabilityTrack,
    ConstraintTrack,
    DecisionHeuristic,
    MindModel,
    SkillFile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLEAGUE_DIR = PROJECT_ROOT / "data" / "colleague_skills"
NUWA_DIR = PROJECT_ROOT / "data" / "nuwa_skills"
CORPUS_DIR = PROJECT_ROOT / "outputs" / "skill_corpus_k8_quantile" / "wikiconv"
CLUSTER_MAP_PATH = CORPUS_DIR / "cluster_map.json"
TOXIC_KEYWORDS_PATH = PROJECT_ROOT / "data" / "cluster_toxic_keywords.json"

# slug (as used in data/colleague_skills, data/nuwa_skills) -> cluster_id
# (as used in outputs/skill_corpus_k8_quantile/wikiconv/cluster_map.json).
SLUG_TO_CLUSTER_ID = {
    "substantive-discussant": "0",
    "niche-terse-specialist": "2",
    "confrontational-editor": "3",
    "veteran-generalist": "4",
    "community-patroller": "6",
    "expert-fact-checker": "7",
}


def _load_archetype_labels() -> dict[str, str]:
    data = json.loads(CLUSTER_MAP_PATH.read_text())
    return {cid: info["label"] for cid, info in data["skills"].items()}


# ---------------------------------------------------------------------------
# Trigger extraction for Tier 3 enforcement
# ---------------------------------------------------------------------------

# Toxic keywords are derived from the corpus, not hardcoded. See
# data/cluster_toxic_keywords.json (pre-computed from ConvoKit toxicity labels).
_CLUSTER_TOXIC_KEYWORDS: dict[str, list[str]] = json.loads(
    TOXIC_KEYWORDS_PATH.read_text()
)

# Description fragments that indicate hostile/violating content.
# When matched, attach the cluster's corpus-derived toxic keywords (Category A).
# Not limited to profanity — covers edit-warring, vandalism, personal attacks,
# and other hostile behaviors that the corpus toxic-keyword extraction captures.
_HOSTILE_DESCRIPTION_FRAGMENTS = [
    "profanity", "slur", "hate speech", "bigot", "insult",
    "ad hominem", "personal attack", "incivility", "uncivil",
    "vandalism", "vandal", "edit warring", "edit war", "revert war",
    "harassment", "hounding", "baiting", "trolling", "troll",
    "attack", "hostile", "aggressive", "threat",
    "disruptive", "disruption", "blatant",
]


def _load_corpus_evidence(cluster_id: str) -> list[str]:
    """Load Rejected-behavior evidence lines from the corpus pack.

    Used as fallback when the nuwa SKILL.md anti-patterns section
    contains only short labels without embedded quotes.
    """
    corpus_md = CORPUS_DIR / f"cluster_{cluster_id}" / "for_colleague.md"
    if not corpus_md.exists():
        return []
    text = corpus_md.read_text()
    evidence_section = _section(text, r"^## Rejected-behavior evidence", r"^##")
    return [l.strip() for l in evidence_section.splitlines() if l.strip().startswith("- [")]


def _extract_triggers(
    description: str,
    evidence: list[str],
    cluster_id: str,
) -> dict:
    """Extract Tier 3 trigger fields from anti-pattern description + evidence.

    Strategy:
      - **Category B (semantic)**: Quoted phrases from evidence text.
        They are real utterances from the corpus, suitable for Sentence-BERT
        cosine matching. When evidence yields no semantic phrases (e.g. nuwa
        anti-patterns with terse labels and no quotes), falls back to the
        cluster's corpus rejected-behavior evidence from for_colleague.md.
      - **Category A (keywords)**: If the description indicates hostile
        behavior (profanity, vandalism, edit-warring, personal attacks, etc.),
        attach the cluster-specific toxic keywords derived from ConvoKit
        toxicity labels via log-odds extraction. No hardcoded word list.
      - **Category C (action patterns)**: Not extracted here — action patterns
        like "revert->revert->revert" require domain knowledge of the
        simulation's action vocabulary and are set manually if needed.

    Returns:
        Dict with keys matching AntiPattern fields:
        trigger_keywords, trigger_semantic_phrases, trigger_regex.
    """
    trigger_keywords: list[str] = []
    trigger_semantic_phrases: list[str] = []
    trigger_regex: list[str] = []

    # --- Category B: semantic phrases from evidence ---
    # Corpus evidence lines (from for_colleague.md/for_nuwa.md) start with "- ["
    # and contain the full utterance text after the tag. Extract the text content.
    # Nuwa SKILL.md evidence uses double-quoted phrases — extract those too.
    for ev in evidence:
        # Corpus evidence format: "- [label] text content"
        if ev.strip().startswith("- ["):
            # Remove the leading tag "- [label] " and keep the text
            m = re.match(r"^-\s*\[[^\]]+\]\s*(.+)$", ev.strip())
            if m:
                text = m.group(1).strip()
                if 10 <= len(text) <= 200:
                    trigger_semantic_phrases.append(text)
        else:
            # Nuwa SKILL.md format: double-quoted phrases within evidence text
            quotes = re.findall(r'"([^"]+)"', ev)
            for q in quotes:
                if 10 <= len(q) <= 200:
                    trigger_semantic_phrases.append(q)

    # Fallback: if no semantic phrases extracted from inline evidence,
    # use the cluster's corpus rejected-behavior evidence.
    if not trigger_semantic_phrases:
        corpus_evidence = _load_corpus_evidence(cluster_id)
        for ev in corpus_evidence:
            m = re.match(r"^-\s*\[[^\]]+\]\s*(.+)$", ev.strip())
            if m:
                text = m.group(1).strip()
                if 10 <= len(text) <= 200:
                    trigger_semantic_phrases.append(text)

    # --- Category A: keywords for hostile/violating anti-patterns ---
    # Strategy: if the cluster has corpus-derived toxic keywords AND the
    # anti-pattern description is even loosely hostile/conflict-adjacent,
    # attach them. The toxic keywords came from high-toxicity utterances
    # in this cluster — they ARE the cluster's toxic vocabulary regardless
    # of which anti-pattern label they sit under.
    cluster_kw = _CLUSTER_TOXIC_KEYWORDS.get(cluster_id, [])
    desc_lower = description.lower()
    # Broad match: any hostile/conflict/quality-violation language
    _BROAD_HOSTILE = _HOSTILE_DESCRIPTION_FRAGMENTS + [
        "unsourced", "unverifiable", "unverified", "original research",
        "biased", "promotional", "advertorial", "coi", "conflict of interest",
        "loaded", "opinionated", "subjective", "pov", "point of view",
        "vandal", "disrupt", "low-quality", "unreliable", "off-topic",
        "trivia", "fancruft", "contradiction", "inconsisten",
    ]
    if cluster_kw and any(frag in desc_lower for frag in _BROAD_HOSTILE):
        trigger_keywords = sorted(cluster_kw)

    return {
        "trigger_keywords": trigger_keywords,
        "trigger_semantic_phrases": trigger_semantic_phrases,
        "trigger_regex": trigger_regex,
    }


# ---------------------------------------------------------------------------
# Shared markdown helpers
# ---------------------------------------------------------------------------

def _section(text: str, heading_regex: str, next_heading_regex: str = r"^#{1,2} ") -> str:
    """Extract the body of a section starting at a heading matching
    ``heading_regex`` up to (but excluding) the next heading matching
    ``next_heading_regex`` (default: any ``#`` or ``##`` heading).

    Returns "" if the heading is not found.
    """
    m = re.search(heading_regex, text, re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    rest = text[start:]
    m2 = re.search(next_heading_regex, rest, re.MULTILINE)
    return rest[: m2.start()] if m2 else rest


def _bullets(text: str) -> list[str]:
    """Extract top-level '- ...' bullet lines (not nested '  - ...')."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^- (.+)$", line.strip()) if line.startswith("- ") else None
        if m:
            out.append(m.group(1).strip())
    return out


# ---------------------------------------------------------------------------
# Colleague-skill (persona_skill.md, Layer 0-7) parser
# ---------------------------------------------------------------------------

def _extract_rejected_evidence_from_corpus(cluster_id: str) -> list[str]:
    """Extract Rejected-behavior evidence lines from the corpus pack
    ``outputs/skill_corpus_k8_quantile/wikiconv/cluster_N/for_colleague.md``.

    This is the same corpus-grounded evidence used during colleague distillation,
    containing the actual high-toxicity/attack/delete utterances that inform
    anti-pattern triggers. We use it here because the final ``persona_skill.md``
    Layer 5 Rejects section contains only short labels without the original
    quotes.
    """
    corpus_md = CORPUS_DIR / f"cluster_{cluster_id}" / "for_colleague.md"
    if not corpus_md.exists():
        logger.warning(f"Corpus pack {corpus_md} not found — colleague triggers will be empty")
        return []

    text = corpus_md.read_text()
    evidence_section = _section(text, r"^## Rejected-behavior evidence", r"^##")
    return [l.strip() for l in evidence_section.splitlines() if l.strip().startswith("- [")]


def parse_colleague(md_path: Path, cluster_id: str) -> dict:
    """Parse colleague-skill ``persona_skill.md`` into keep-table fields.

    Keeps: Layer 0 (-> generic DecisionHeuristic), Layer 2 (-> expression_rules),
    Layer 3 (-> MindModel[]), Layer 4 Quick Rules (-> DecisionHeuristic[]),
    Layer 5 Rejects (-> AntiPattern[]).
    Discards: Layer 1 identity/disclaimer, Layer 4 Optimizes/Moves/Waits/Changes
    prose (kept only as heuristic scenario context, not separate fields),
    Layer 5 Honest Boundaries / Contradictions, Layer 6, Layer 7, Cognitive
    Timeline, Correction Log.
    """
    text = md_path.read_text()

    # Load corpus-grounded rejected-behavior evidence for trigger extraction.
    # The Layer 5 Rejects section in persona_skill.md contains only short labels
    # without quotes, so we pull the original evidence from the distillation input pack.
    corpus_evidence = _extract_rejected_evidence_from_corpus(cluster_id)

    # Layer 0: Core Thinking Rules -> generic decision heuristics (durable,
    # cross-context; the outline's DecisionHeuristic is the closest field).
    layer0 = _section(text, r"^## Layer 0: Core Thinking Rules\s*$")
    core_rules = _bullets(layer0)

    # Layer 2: Expression DNA -> qualitative expression_rules (verbatim
    # narrative style rules; NOT the numeric scalars).
    layer2 = _section(text, r"^## Layer 2: Expression DNA\s*$")
    tone = _section(layer2, r"^### Tone\s*$", r"^### ")
    signature_moves = _bullets(_section(layer2, r"^### Signature Moves\s*$", r"^### "))
    style_markers = _bullets(_section(layer2, r"^### Style Markers\s*$", r"^### "))
    expression_rules = []
    if tone.strip():
        expression_rules.append(f"Tone: {tone.strip()}")
    expression_rules.extend(f"Signature move: {s}" for s in signature_moves)
    expression_rules.extend(f"Style marker: {s}" for s in style_markers)

    # Layer 3: Mental Models -> MindModel[]
    layer3 = _section(text, r"^## Layer 3: Mental Models\s*$")
    mind_models = []
    for block_m in re.finditer(r"^### Model: (.+)$", layer3, re.MULTILINE):
        name = block_m.group(1).strip()
        start = block_m.end()
        rest = layer3[start:]
        nxt = re.search(r"^### Model: ", rest, re.MULTILINE)
        body = rest[: nxt.start()] if nxt else rest

        def _field(label: str) -> str:
            fm = re.search(rf"\*\*{label}\*\*:\s*(.+)", body)
            return fm.group(1).strip() if fm else ""

        definition = _field("Definition")
        sees_first = _field(r"What it sees first")
        filters_out = _field(r"What it filters out")
        reframes = _field(r"How it reframes the problem")
        evidence = _field("Evidence")
        failure_mode = _field("Failure mode")

        description = definition
        application_parts = [p for p in [
            f"Sees first: {sees_first}" if sees_first else "",
            f"Filters out: {filters_out}" if filters_out else "",
            f"Reframes: {reframes}" if reframes else "",
        ] if p]
        mind_models.append(MindModel(
            name=name,
            description=description,
            evidence=[evidence] if evidence else [],
            application="; ".join(application_parts),
            limitation=failure_mode,
        ))

    # Layer 4: Decision Heuristics -> Quick Rules (if-X-then-Y) + Optimizes-for
    # as scenario context for a lead heuristic.
    layer4 = _section(text, r"^## Layer 4: Decision Heuristics\s*$")
    optimizes_for = _section(layer4, r"^### Optimizes for\s*$", r"^### ").strip()
    quick_rules_text = _section(layer4, r"^### Quick Rules\s*$", r"^### |\n---")
    decision_heuristics = []
    for rule in _bullets(quick_rules_text):
        # "If X, then Y — because Z" or "If X then Y — because Z"
        m = re.match(r"^If (.+?),?\s+then (.+?)(?:\s*—\s*because (.+))?$", rule, re.IGNORECASE)
        if m:
            scenario, action, reason = m.group(1), m.group(2), m.group(3) or ""
            decision_heuristics.append(DecisionHeuristic(
                name=scenario[:60],
                rule=f"If {scenario}, then {action}",
                scenario=scenario,
                case=reason,
            ))
        else:
            decision_heuristics.append(DecisionHeuristic(
                name=rule[:60], rule=rule, scenario="", case="",
            ))
    if core_rules:
        decision_heuristics.append(DecisionHeuristic(
            name="Core thinking rules",
            rule=" | ".join(core_rules),
            scenario="Always (Layer-0 durable priority rules)",
            case=optimizes_for,
        ))

    # Layer 5: Rejects -> AntiPattern[] (Honest Boundaries / Contradictions discarded)
    layer5 = _section(text, r"^## Layer 5: Anti-patterns and Limits\s*$")
    rejects_text = _section(layer5, r"^### Rejects\s*$", r"^### ")
    anti_patterns = []
    for reject in _bullets(rejects_text):
        # "**Label**: reason" pattern
        m = re.match(r"^\*\*(.+?)\*\*:\s*(.+)$", reject)
        if m:
            label, reason = m.group(1), m.group(2)
        else:
            label, reason = reject, ""
        triggers = _extract_triggers(label, corpus_evidence, cluster_id)
        anti_patterns.append(AntiPattern(
            description=label,
            trigger_conditions=[],
            trigger_keywords=triggers["trigger_keywords"],
            trigger_semantic_phrases=triggers["trigger_semantic_phrases"],
            trigger_regex=triggers["trigger_regex"],
            reason=reason,
        ))

    return {
        "mind_models": mind_models,
        "decision_heuristics": decision_heuristics,
        "anti_patterns": anti_patterns,
        "expression_rules": expression_rules,
    }


# ---------------------------------------------------------------------------
# nuwa-skill (SKILL.md) parser
# ---------------------------------------------------------------------------

def parse_nuwa(md_path: Path, cluster_id: str) -> dict:
    """Parse nuwa-skill ``SKILL.md`` into keep-table fields.

    Keeps: Role-playing rules (-> expression_rules), Response workflow
    (-> DecisionHeuristic[]), Mental models N (-> MindModel[]), Decision
    heuristics N (-> DecisionHeuristic[]), Expression DNA table
    (-> expression_rules, qualitative only), Values and anti-patterns
    (-> AntiPattern[]).
    Discards: YAML frontmatter triggers/frequency, When to (not) activate,
    Example user prompts, Honest boundaries, Evidence format, Source,
    Safety guardrails / tool-use routing (plugin-interaction scaffolding).
    """
    text = md_path.read_text()
    # Strip YAML frontmatter (triggers/frequency are plugin-activation
    # metadata, not behavior — discarded per the keep/discard table).
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)

    # Role-playing rules -> expression_rules (verbatim, per plan).
    role_rules_text = _section(text, r"^## Role-playing rules\s*$")
    expression_rules = [f"Role-playing rule: {b}" for b in _bullets(role_rules_text)]

    # Response workflow -> DecisionHeuristic[] (if-situation-then-step).
    workflow_text = _section(text, r"^## Response workflow\s*$")
    decision_heuristics = []
    for line in workflow_text.splitlines():
        m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*\.?\s*(.*)$", line.strip())
        if m:
            step_name, step_body = m.group(1), m.group(2)
            decision_heuristics.append(DecisionHeuristic(
                name=step_name[:60],
                rule=f"{step_name}: {step_body}".strip(": "),
                scenario="Response workflow step",
                case="",
            ))

    # Expression DNA table -> qualitative expression_rules (skip numeric
    # override; table rows are "| Dimension | Pattern |").
    edna_text = _section(text, r"^## Expression DNA\s*$")
    for line in edna_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0].lower() not in ("dimension",):
            expression_rules.append(f"{cells[0]}: {cells[1]}")

    # Mental models (N) -> MindModel[]
    mm_text = _section(text, r"^## Mental models \(\d+\)\s*$")
    mind_models = []
    for block_m in re.finditer(r"^### \d+\.\s*(.+)$", mm_text, re.MULTILINE):
        name = block_m.group(1).strip()
        start = block_m.end()
        rest = mm_text[start:]
        nxt = re.search(r"^### \d+\.\s", rest, re.MULTILINE)
        body = rest[: nxt.start()] if nxt else rest

        def _field_line(label: str) -> str:
            fm = re.search(rf"\*\*{label}[:\*]*\*?\*?:?\s*(.+)", body)
            return fm.group(1).strip() if fm else ""

        # Two nuwa sub-formats exist: "**Description**: ..." inline, or
        # "**Description:**" followed by a bulleted "**Evidence:**" block.
        description = _field_line("Description")
        when_to_apply = _field_line("When to apply") or _field_line(r"When to apply\*\*")
        limitation = _field_line("Limitation")
        one_sentence = _field_line(r"One-sentence description")
        if one_sentence:
            description = one_sentence
            when_to_apply = _field_line(r"When to apply")

        evidence_lines = []
        ev_section = re.search(
            r"\*\*Evidence(?:\s+from corpus)?\*\*:?\s*(.*?)(?=\n\*\*|\Z)",
            body, re.DOTALL,
        )
        if ev_section:
            ev_body = ev_section.group(1)
            for eline in ev_body.splitlines():
                eline = eline.strip().lstrip("-").strip()
                if eline:
                    evidence_lines.append(eline)
        if not evidence_lines:
            # Inline single-line evidence e.g. "**Evidence**: M1 states, ..."
            inline_ev = _field_line("Evidence")
            if inline_ev:
                evidence_lines = [inline_ev]

        mind_models.append(MindModel(
            name=name,
            description=description,
            evidence=evidence_lines,
            application=when_to_apply,
            limitation=limitation,
        ))

    # Decision heuristics (N) -> DecisionHeuristic[]
    dh_text = _section(text, r"^## Decision heuristics \(\d+\)\s*$")
    for block_m in re.finditer(
        r"^\d+\.\s+\*\*(.+?)\*\*\s*(?:→\s*(.+?))?$", dh_text, re.MULTILINE
    ):
        rule_head, rule_tail = block_m.group(1), block_m.group(2) or ""
        rule = f"{rule_head} → {rule_tail}".strip(" →") if rule_tail else rule_head
        decision_heuristics.append(DecisionHeuristic(
            name=rule_head[:60], rule=rule, scenario=rule_head, case="",
        ))

    # Values and anti-patterns -> AntiPattern[] (Inner tensions discarded —
    # they're self-contradictions for narrative flavor, not a prohibition).
    # Six source files, six slightly different sub-formats observed:
    #   1. "N. **Label** — evidence"      (community-patroller, expert-fact-checker)
    #   2. "- **Label** — evidence"       (confrontational-editor)
    #   3. "- **Label**: evidence"        (niche-terse-specialist)
    #   4. "- **Label**" + nested bullets (substantive-discussant)
    #   5. "- Label" (no bold, no evid.)  (veteran-generalist)
    va_text = _section(text, r"^## Values and anti-patterns\s*$")
    ap_text = _section(va_text, r"^- \*\*Anti-patterns\*\*.*$", r"^- \*\*Inner tensions\*\*|^## ")
    anti_patterns = []

    ap_lines = [l for l in ap_text.splitlines() if l.strip()]
    indents = [len(l) - len(l.lstrip(" ")) for l in ap_lines]
    base_indent = min(indents) if indents else 0

    # Group into (top-level item line, [nested evidence lines]) blocks.
    blocks: list[tuple[str, list[str]]] = []
    for line, indent in zip(ap_lines, indents):
        if indent <= base_indent:
            blocks.append((line.strip(), []))
        elif blocks:
            blocks[-1][1].append(line.strip())

    for item_line, nested in blocks:
        # Strip leading "N." or "-" list marker.
        item_line = re.sub(r"^(?:\d+\.|-)\s+", "", item_line)
        m = re.match(r"^\*\*(.+?)\*\*\s*(?:[—:-]\s*(.+))?$", item_line)
        if m:
            label, inline_evidence = m.group(1), m.group(2)
        else:
            label, inline_evidence = item_line, None

        evidence = []
        if inline_evidence:
            evidence.append(inline_evidence.strip())
        for nline in nested:
            nline = re.sub(r"^-\s*", "", nline)
            nline = re.sub(r"^\[discuss\]\s*", "", nline)
            if nline:
                evidence.append(nline)

        triggers = _extract_triggers(label.strip(), evidence, cluster_id)
        anti_patterns.append(AntiPattern(
            description=label.strip(),
            trigger_conditions=[],
            trigger_keywords=triggers["trigger_keywords"],
            trigger_semantic_phrases=triggers["trigger_semantic_phrases"],
            trigger_regex=triggers["trigger_regex"],
            evidence=evidence,
        ))

    return {
        "mind_models": mind_models,
        "decision_heuristics": decision_heuristics,
        "anti_patterns": anti_patterns,
        "expression_rules": expression_rules,
    }


# ---------------------------------------------------------------------------
# Numeric Expression DNA (statistical, non-LLM) — re-extract from corpus
# ---------------------------------------------------------------------------

def _load_cluster_messages(cluster_id: str) -> list[Message]:
    """Load ``cluster_N/typical.jsonl`` as ``Message`` objects for
    ``ExpressionDNAExtractor`` — the same corpus used to build the
    distillation dossiers (``for_colleague.md`` / ``for_nuwa.md``).
    """
    from datetime import datetime, timezone

    path = CORPUS_DIR / f"cluster_{cluster_id}" / "typical.jsonl"
    messages = []
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            messages.append(Message(
                msg_id=f"typ_{cluster_id}_{i}",
                thread_id=f"typ_{cluster_id}_{i}",
                user_id=d.get("member", "unknown"),
                platform=Platform.WIKIPEDIA,
                timestamp=datetime.fromtimestamp(0, tz=timezone.utc),
                text=d.get("text", ""),
                action_type=ActionType.DISCUSS,
            ))
    return messages


def _compute_expression_dna(cluster_id: str):
    """Statistical (non-LLM) Expression DNA numeric scalars + embedding
    centroid, from the cluster's typical-utterance corpus.
    """
    from src.config.settings import get_shared_embedder

    messages = _load_cluster_messages(cluster_id)
    texts = [m.text if m.text else " " for m in messages]
    embedder = get_shared_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False)
    import numpy as np
    embeddings = np.asarray(embeddings)

    extractor = ExpressionDNAExtractor()
    return extractor.extract(messages, embeddings)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_skill_file(
    cluster_id: str,
    slug: str,
    distiller: str,
    archetype_label: str,
) -> SkillFile:
    if distiller == "colleague":
        md_path = COLLEAGUE_DIR / slug / "persona_skill.md"
        parsed = parse_colleague(md_path, cluster_id)
    elif distiller == "nuwa":
        md_path = NUWA_DIR / f"{slug}-perspective" / "SKILL.md"
        parsed = parse_nuwa(md_path, cluster_id)
    else:
        raise ValueError(f"Unknown distiller: {distiller}")

    logger.info(
        f"[{distiller}/{slug}] parsed {len(parsed['mind_models'])} mind models, "
        f"{len(parsed['decision_heuristics'])} decision heuristics, "
        f"{len(parsed['anti_patterns'])} anti-patterns, "
        f"{len(parsed['expression_rules'])} expression rules"
    )
    if not parsed["mind_models"]:
        raise ValueError(f"[{distiller}/{slug}] parsed zero mind models — parser mismatch")
    if not parsed["anti_patterns"]:
        raise ValueError(f"[{distiller}/{slug}] parsed zero anti-patterns — parser mismatch")

    expression_dna = _compute_expression_dna(cluster_id)
    expression_dna.expression_rules = parsed["expression_rules"]

    n_members = json.loads(CLUSTER_MAP_PATH.read_text())["skills"][cluster_id]["size"]

    return SkillFile(
        cluster_id=cluster_id,
        platform="wikipedia",
        capability=CapabilityTrack(
            expression_dna=expression_dna,
            mind_models=parsed["mind_models"],
            decision_heuristics=parsed["decision_heuristics"],
        ),
        constraint=ConstraintTrack(anti_patterns=parsed["anti_patterns"]),
        source_thread_ids=[],
        source_user_count=n_members,
        distiller=distiller,
        archetype_label=archetype_label,
    )


def convert_one(slug: str, distiller: str, out_dir: Path) -> Path:
    cluster_id = SLUG_TO_CLUSTER_ID[slug]
    labels = _load_archetype_labels()
    skill = build_skill_file(cluster_id, slug, distiller, labels[cluster_id])
    compiler = SkillCompiler.__new__(SkillCompiler)  # no LLM needed for save_skill
    filename = f"skill_cluster_{cluster_id}_wikipedia_{distiller}.yaml"
    out_path = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(skill.to_yaml())
    logger.info(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=sorted(SLUG_TO_CLUSTER_ID), help="Convert a single skill slug")
    parser.add_argument("--distiller", choices=["colleague", "nuwa"], help="Restrict to one distiller")
    parser.add_argument("--all", action="store_true", help="Convert all 6 skills x 2 distillers")
    parser.add_argument("--out-dir", default=str(settings.skills_dir), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    slugs = [args.skill] if args.skill else list(SLUG_TO_CLUSTER_ID)
    distillers = [args.distiller] if args.distiller else ["colleague", "nuwa"]

    if not args.skill and not args.all:
        parser.error("Specify --skill SLUG or --all")

    for slug in slugs:
        for distiller in distillers:
            convert_one(slug, distiller, out_dir)


if __name__ == "__main__":
    main()
