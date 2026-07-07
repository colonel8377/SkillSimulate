#!/usr/bin/env python3
"""Helper script for K=8 WikiConv skill distillation.

This script provides file paths and operation guides for manual skill
distillation in independent Claude Code sessions (colleague-skill and nuwa-skill).

It does NOT call any LLM APIs — it only:
1. Reads corpus metadata from cluster_map.json and profile.json
2. Prints file paths for corpus files
3. Generates a markdown checklist for tracking progress

Usage
--------
Print guide for all clusters:
    python scripts/distill_skills.py

Print guide for a specific cluster:
    python scripts/distill_skills.py --cluster 0

Custom checklist output path:
    python scripts/distill_skills.py --output outputs/my_checklist.md

Output
--------
The script prints to stdout and generates a checklist markdown file at:
    outputs/distill_checklist.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make SkillSimulate src importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_DIR = REPO_ROOT / "outputs" / "skill_corpus_k8_quantile" / "wikiconv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

COLLEAGUE_REPO = Path("/mnt/sda/qm/dev/colleague-skill").resolve()
NUWA_REPO = Path("/mnt/sda/qm/dev/nuwa-skill").resolve()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cluster_map() -> dict[str, Any]:
    """Load cluster_map.json containing skill metadata."""
    path = CORPUS_DIR / "cluster_map.json"
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile(skill_id: str) -> dict[str, Any]:
    """Load profile.json for a specific cluster."""
    path = CORPUS_DIR / f"cluster_{skill_id}" / "profile.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Installation Instructions
# ---------------------------------------------------------------------------
def print_install_instructions() -> None:
    """Print installation instructions for colleague-skill and nuwa-skill."""
    print("\n" + "=" * 70)
    print("插件安装说明")
    print("=" * 70)

    print("\n【colleague-skill】")
    print("安装方式：")
    print(f"  git clone https://github.com/titanwings/colleague-skill {{~/.claude/skills/dot-skill,.claude/skills/dot-skill}}")
    print(f"  或: cd {COLLEAGUE_REPO} && python3 tools/install_hermes_skill.py --force")
    print("\n使用方式：")
    print("  在 Claude Code 中说 `/dot-skill`")

    print("\n【nuwa-skill】")
    print("安装方式：")
    print(f"  npx skills add alchaincyf/nuwa-skill")
    print(f"  或: git clone https://github.com/alchaincyf/nuwa-skill {{~/.claude/skills/nuwa-skill,.claude/skills/nuwa-skill}}")
    print("\n使用方式：")
    print("  在 Claude Code 中说 `/huashu-nuwa` 或「蒸馏XX」")
    print("  如果没有 npx/skills，可以直接安装到本地：")
    print(f"  git clone https://github.com/alchaincyf/nuwa-skill {{~/.claude/skills/nuwa-skill,.claude/skills/nuwa-skill}}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Guide Generation
# ---------------------------------------------------------------------------
def print_colleague_guide(skill_id: str, label: str, tags: list[str]) -> None:
    """Print step-by-step guide for colleague-skill distillation."""
    slug = f"wikiconv_cluster_{skill_id}"
    colleague_corpus = CORPUS_DIR / f"cluster_{skill_id}" / "for_colleague.md"

    print(f"\n  Colleague-skill:")
    print(f"    1. cd {COLLEAGUE_REPO}")
    print(f"    2. 启动 claude，说 `/dot-skill`")
    print(f"    3. 选择 celebrity, budget-friendly")
    print(f"    4. 名字填「{label}」，slug 填「{slug}」")
    print(f"    5. 语料选择 [E] 直接粘贴内容")
    print(f"    6. 粘贴以下文件内容（用 cat 复制）:")
    print(f"       {colleague_corpus}")


def print_nuwa_guide(skill_id: str, label: str, tags: list[str]) -> None:
    """Print step-by-step guide for nuwa-skill distillation."""
    slug = f"wikiconv_cluster_{skill_id}"
    nuwa_corpus = CORPUS_DIR / f"cluster_{skill_id}" / "for_nuwa.md"

    print(f"\n  Nuwa-skill:")
    print(f"    1. cd {NUWA_REPO}")
    print(f"    2. 启动 claude，说 `/huashu-nuwa` 或触发女娲")
    print(f"    3. 名字填「{label}」，slug 填「{slug}-perspective」")
    print(f"    4. 问到本地语料时，粘贴以下文件内容（用 cat 复制）:")
    print(f"       {nuwa_corpus}")


def print_guide_for_cluster(skill_id: str, label: str, tags: list[str]) -> None:
    """Print complete guide for one cluster."""
    print("\n" + "-" * 70)
    print(f"Cluster {skill_id}: {label}")
    print("-" * 70)
    print(f"  Tags: {', '.join(tags)}")
    print(f"  Slug: wikiconv_cluster_{skill_id}")

    print_colleague_guide(skill_id, label, tags)
    print_nuwa_guide(skill_id, label, tags)


# ---------------------------------------------------------------------------
# Checklist Generation
# ---------------------------------------------------------------------------
def generate_checklist(
    skills: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate markdown checklist for tracking distillation progress."""
    ordered_ids = sorted(skills.keys(), key=lambda x: int(x))

    lines = [
        "# K=8 WikiConv Skill Distillation Checklist\n",
        "> Generated by scripts/distill_skills.py\n",
        f"> Total clusters: {len(ordered_ids)}\n",
        "\n## 插件安装\n",
        "\n### colleague-skill\n",
        "```bash\n",
        "# 全局安装\n",
        "git clone https://github.com/titanwings/colleague-skill ~/.claude/skills/dot-skill\n",
        "\n",
        "# 或项目本地安装\n",
        "git clone https://github.com/titanwings/colleague-skill .claude/skills/dot-skill\n",
        "```\n",
        "\n### nuwa-skill\n",
        "```bash\n",
        "# 方式1: npx 安装（推荐）\n",
        "npx skills add alchaincyf/nuwa-skill\n",
        "\n",
        "# 方式2: 直接 clone\n",
        "git clone https://github.com/alchaincyf/nuwa-skill ~/.claude/skills/nuwa-skill\n",
        "\n",
        "# 或项目本地\n",
        "git clone https://github.com/alchaincyf/nuwa-skill .claude/skills/nuwa-skill\n",
        "```\n",
        "\n## 使用说明\n",
        "- Colleague-skill: 在 Claude Code 中说 `/dot-skill`\n",
        "- Nuwa-skill: 在 Claude Code 中说 `/huashu-nuwa` 或「蒸馏XX」\n",
        "- 语料文件用 `cat` 命令复制后粘贴到对应 skill 的输入位置\n",
        "\n## Clusters\n",
    ]

    for skill_id in ordered_ids:
        skill_info = skills[skill_id]
        label = skill_info.get("label", f"Cluster {skill_id}")
        source_leaves = skill_info.get("source_leaves", [])
        size = skill_info.get("size", 0)

        # Load tags from profile.json
        profile = load_profile(skill_id)
        tags = profile.get("tags", [])

        colleague_corpus = CORPUS_DIR / f"cluster_{skill_id}" / "for_colleague.md"
        nuwa_corpus = CORPUS_DIR / f"cluster_{skill_id}" / "for_nuwa.md"

        lines.append(f"\n### Cluster {skill_id}: {label}\n")
        lines.append(f"- **Tags**: {', '.join(tags)}\n")
        lines.append(f"- **Source clusters**: {source_leaves}\n")
        lines.append(f"- **Size**: {size:,} users\n")
        lines.append(f"- **Slug**: `wikiconv_cluster_{skill_id}`\n")
        lines.append(f"\n**语料文件:**\n")
        lines.append(f"- Colleague: `{colleague_corpus}`\n")
        lines.append(f"- Nuwa: `{nuwa_corpus}`\n")
        lines.append("\n**Progress:**\n")
        lines.append(f"- [ ] Colleague-skill distilled\n")
        lines.append(f"- [ ] Nuwa-skill distilled\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nChecklist written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        help="Only print guide for a specific cluster ID (e.g. '0', '2')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "outputs" / "distill_checklist.md"),
        help="Path to write the checklist markdown file (default: outputs/distill_checklist.md)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip printing installation instructions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load cluster metadata
    cluster_map = load_cluster_map()
    skills = cluster_map.get("skills", {})

    if not skills:
        print("Error: No skills found in cluster_map.json", file=sys.stderr)
        return 1

    # Print header
    print("\n" + "=" * 70)
    print("K=8 WikiConv Skill Distillation Helper")
    print("=" * 70)
    print(f"Total clusters: {len(skills)}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Colleague repo: {COLLEAGUE_REPO}")
    print(f"Nuwa repo: {NUWA_REPO}")

    if not args.no_install:
        print_install_instructions()

    # Process specific cluster or all clusters
    if args.cluster:
        if args.cluster not in skills:
            print(f"Error: Cluster '{args.cluster}' not found", file=sys.stderr)
            print(f"Available: {', '.join(sorted(skills.keys(), key=lambda x: int(x)))}")
            return 1

        skill_info = skills[args.cluster]
        profile = load_profile(args.cluster)
        tags = profile.get("tags", skill_info.get("tags", []))

        print_guide_for_cluster(
            args.cluster,
            skill_info.get("label", f"Cluster {args.cluster}"),
            tags,
        )
    else:
        # Print guide for all clusters
        ordered_ids = sorted(skills.keys(), key=lambda x: int(x))

        for skill_id in ordered_ids:
            skill_info = skills[skill_id]
            profile = load_profile(skill_id)
            tags = profile.get("tags", skill_info.get("tags", []))

            print_guide_for_cluster(
                skill_id,
                skill_info.get("label", f"Cluster {skill_id}"),
                tags,
            )

        # Generate checklist
        checklist_path = Path(args.output)
        generate_checklist(skills, checklist_path)

    print("\n" + "=" * 70)
    print("Done! Follow the guides above in independent Claude Code sessions.")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
