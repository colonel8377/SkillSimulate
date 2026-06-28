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
    """Collect this archetype's rejection signals: own comments deleted / high-tox /
    CGA personal-attack, plus deletes the members performed on others."""
    members = set(cluster_result.get_cluster_members(profile.leaf_id))
    lines: list[str] = []
    for t in threads:
        for m in t.messages:
            if m.user_id not in members:
                continue
            tox = m.metadata.get("toxicity")
            attack = m.metadata.get("comment_has_personal_attack")
            if m.action_type == ActionType.DELETE:
                lines.append(f"- [deleted another's comment] {m.text[:160]}")
            elif attack or (isinstance(tox, (int, float)) and tox >= 0.6):
                lines.append(f"- [high-conflict/flagged tox={tox}] {m.text[:160]}")
        if len(lines) >= 25:
            break
    return lines[:25]


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


def _render(profile: LeafProfile, threads, cr, *, flavour: str) -> str:
    tiers = _tier(profile.typical_utterances)
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
) -> None:
    out_dir = Path(out_dir)
    for lid, profile in profiles.items():
        d = out_dir / platform / f"cluster_{lid}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "for_colleague.md").write_text(
            _render(profile, threads, cluster_result, flavour="colleague"), encoding="utf-8"
        )
        (d / "for_nuwa.md").write_text(
            _render(profile, threads, cluster_result, flavour="nuwa"), encoding="utf-8"
        )
    logger.info(f"Exported {len(profiles)} leaf material packs to {out_dir / platform}")
