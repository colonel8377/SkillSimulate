"""Step 1: export per-leaf distillation material for the third-party skills.

For each leaf (archetype) we emit two tailored packs from its frozen typical
utterances + behavioural tags + rejected-behaviour evidence:
- ``for_colleague.md`` — chat-log + persona-description framing (colleague-skill).
- ``for_nuwa.md``      — writings/decisions dossier framing (nuwa-skill, local mode).

We do NOT distill here. The user feeds these packs to the real colleague / nuwa
skills manually; their outputs come back via the ingest step.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from loguru import logger

from src.clustering.clusterer import ClusterResult
from src.data.schemas import ActionType, Thread
from src.skill.cluster_profile import LeafProfile, TypicalUtterance


# Rejection-evidence policy (mirrored by streaming.collect_rejection_evidence).
# Tox 0.6 catches many borderline/false-positive cases; 0.8 isolates genuinely
# hostile utterances while still leaving room for the rarer moderation signals
# (deletions) that ground archetype-specific anti-patterns.
EVIDENCE_TOX_THRESHOLD = 0.8
# Per-leaf, per-type budget. Order of collection/printing: delete, tox, attack.
EVIDENCE_BUDGET = {"delete": 13, "tox": 10, "attack": 2}  # 25 total


_HEADER = (
    "> This is corpus from {m} representative members of one behavioral archetype. "
    "Extract the **cross-member recurring** shared expression style, mental models, "
    "decision habits, and anti-patterns — ignore traits that appear in only a single "
    "member.\n"
)


def _tier(utts: list[TypicalUtterance]) -> dict[str, list[TypicalUtterance]]:
    """Split utterances by source-priority tier the skills expect."""
    tiers: dict[str, list[TypicalUtterance]] = {"substantive": [], "decision": [], "casual": []}
    for u in utts:
        if u.action in (ActionType.DELETE.value, ActionType.RESTORE.value, ActionType.EDIT.value):
            tiers["decision"].append(u)
        elif len(u.text) >= 200:
            tiers["substantive"].append(u)
        else:
            tiers["casual"].append(u)
    return tiers


def _rejected_evidence(
    profile: LeafProfile, threads: list[Thread], cluster_result: ClusterResult
) -> list[str]:
    """Collect this archetype's rejection signals with per-type budgets.

    Grounds anti-patterns in actual in-corpus violations:
      - deletions the archetype's members performed on others' comments,
      - the members' own genuinely toxic utterances (tox >= 0.8),
      - personal-attack flags (CGA, when available).
    Per-type budgets prevent the abundant high-tox signal from crowding out the
    rarer deletion / attack signals (which are more specific to an archetype).
    """
    members = set(cluster_result.get_cluster_members(profile.leaf_id))
    counts: dict[str, int] = {}
    lines: list[str] = []
    budget = EVIDENCE_BUDGET
    threshold = EVIDENCE_TOX_THRESHOLD

    def _full() -> bool:
        return all(counts.get(k, 0) >= v for k, v in budget.items())

    for t in threads:
        for m in t.messages:
            if m.user_id not in members:
                continue
            tox = m.metadata.get("toxicity")
            attack = m.metadata.get("comment_has_personal_attack")
            text = m.text[:160]
            etype: str | None = None
            label = ""
            if m.action_type == ActionType.DELETE:
                etype, label = "delete", "deleted another's comment"
            elif attack:
                etype, label = "attack", "personal-attack flagged"
            elif isinstance(tox, (int, float)) and float(tox) >= threshold:
                etype, label = "tox", f"high-conflict/flagged tox={round(float(tox), 2)}"
            if etype and counts.get(etype, 0) < budget[etype]:
                lines.append(f"- [{label}] {text}")
                counts[etype] = counts.get(etype, 0) + 1
            if _full():
                break
        if _full():
            break
    return lines


def _member_blocks(utts: list[TypicalUtterance]) -> str:
    """Group utterances by member, preserving each individual voice."""
    by_member: dict[str, list[TypicalUtterance]] = defaultdict(list)
    for u in utts:
        by_member[u.member].append(u)
    out = []
    for i, (member, items) in enumerate(by_member.items(), 1):
        out.append(f"\n#### Representative member {i}")
        for u in items:
            ctx = f"  (replying to: {u.parent_context[:80]})" if u.parent_context else ""
            out.append(f"- [{u.action}] {u.text}{ctx}")
    return "\n".join(out)


def _render(profile: LeafProfile, threads, cr, *, flavour: str,
            leaf_evidence: dict[int, list[str]] | None = None) -> str:
    tiers = _tier(profile.typical_utterances)
    # Streaming path can't hold all threads in memory, so it pre-collects
    # rejection evidence via a dedicated pass (streaming.collect_rejection_evidence);
    # the non-streaming path derives it from in-memory threads here.
    if leaf_evidence is not None:
        rejected = list(leaf_evidence.get(profile.leaf_id, []))
    else:
        rejected = _rejected_evidence(profile, threads, cr)
    tag_str = ", ".join(profile.tags) if profile.tags else "(no salient tags)"

    intro = (
        "# Colleague archetype material pack\n" if flavour == "colleague"
        else "# Perspective archetype research dossier\n"
    )
    lines = [intro, _HEADER.format(m=len(profile.members)), ""]
    lines.append(f"## Identity\n- Codename: cluster {profile.leaf_id} archetype")
    lines.append(f"- Behavioral tags (auto-generated from statistics): {tag_str}")
    lines.append(f"- Group size: {profile.size} users\n")

    label = {
        "substantive": "## Long-form / substantive contributions",
        "decision": "## Decision actions (edit / delete / restore)",
        "casual": "## Short everyday remarks",
    }
    for key in ("substantive", "decision", "casual"):
        if tiers[key]:
            lines.append(label[key])
            lines.append(_member_blocks(tiers[key]))
            lines.append("")

    if rejected:
        lines.append("## Rejected-behavior evidence (for anti-pattern extraction)")
        lines.extend(rejected)
        lines.append("")
    return "\n".join(lines)


def export_corpus_packs(
    threads: list[Thread],
    cluster_result: ClusterResult,
    profiles: dict[int, LeafProfile],
    out_dir: str | Path,
    platform: str,
    leaf_evidence: dict[int, list[str]] | None = None,
) -> None:
    """Write for_colleague.md / for_nuwa.md per leaf.

    ``leaf_evidence`` (leaf_id → pre-collected rejection-evidence lines) is the
    streaming path's source for the anti-pattern grounding section — collected by
    ``streaming.collect_rejection_evidence`` since the full corpus can't be held
    as in-memory threads. When omitted (non-streaming path), evidence is derived
    from ``threads`` via ``_rejected_evidence``.
    """
    out_dir = Path(out_dir)
    for lid, profile in profiles.items():
        d = out_dir / platform / f"cluster_{lid}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "for_colleague.md").write_text(
            _render(profile, threads, cluster_result, flavour="colleague",
                    leaf_evidence=leaf_evidence),
            encoding="utf-8",
        )
        (d / "for_nuwa.md").write_text(
            _render(profile, threads, cluster_result, flavour="nuwa",
                    leaf_evidence=leaf_evidence),
            encoding="utf-8",
        )
    logger.info(f"Exported {len(profiles)} leaf material packs to {out_dir / platform}")
