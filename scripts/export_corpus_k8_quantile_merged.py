"""Export merged skill corpus from the locked K=8 Quantile clustering.

Reads ``outputs/skill_corpus_k8_quantile/wikiconv/cluster_map.json`` to merge
selected leaves (currently 1->0 and 5->4) and samples more utterances per skill
than the first-pass export, so downstream colleague/nuwa distillation has a
richer corpus.

Output: overwrites ``outputs/skill_corpus_k8_quantile/wikiconv/`` with the
merged 6-skill packs.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.clustering.streaming import (
    find_year_dirs,
    collect_member_utterances,
    collect_rejection_evidence,
)
from src.skill.cluster_profile import ArchetypeProfiler, LeafProfile, TypicalUtterance
from src.skill.corpus_export import export_corpus_packs, EVIDENCE_BUDGET

DATA_DIR = "data/raw/wikiconv_en"
CLUSTER_PATH = "outputs/stream_cache/clustering_k8_final_quantile.pkl"
MAP_PATH = Path("outputs/skill_corpus_k8_quantile/wikiconv/cluster_map.json")
OUT_DIR = Path("outputs/skill_corpus_k8_quantile")
WORKERS = 8

# More representative members + more utterances per member than the first pass.
REP_PER_LEAF = 30          # per source leaf
MAX_PER_USER = 300         # utterances collected per representative member


def load_merge_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    # normalize skill keys to int
    plan["skills"] = {int(k): v for k, v in plan["skills"].items()}
    plan["merge_map"] = {int(k): int(v) for k, v in plan.get("merge_map", {}).items()}
    return plan


def _evidence_type(line: str) -> str | None:
    if "deleted another's comment" in line:
        return "delete"
    if "personal-attack flagged" in line:
        return "attack"
    if "high-conflict/flagged tox=" in line:
        return "tox"
    return None


def _evidence_fill(lines: list[str]) -> dict[str, dict[str, int]]:
    counts = {t: 0 for t in EVIDENCE_BUDGET}
    for line in lines:
        t = _evidence_type(line)
        if t:
            counts[t] += 1
    return {t: {"filled": counts[t], "budget": EVIDENCE_BUDGET[t]} for t in EVIDENCE_BUDGET}


def _mean_centered_centroids(lang_centroids: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if not lang_centroids:
        return {}
    global_mean = np.mean(list(lang_centroids.values()), axis=0)
    return {lid: c - global_mean for lid, c in lang_centroids.items()}


def _nearest_other_cosine(leaf_id: int, lang_centroids: dict[int, np.ndarray]) -> float | None:
    if len(lang_centroids) < 2 or leaf_id not in lang_centroids:
        return None
    c = lang_centroids[leaf_id]
    c = c / (np.linalg.norm(c) + 1e-9)
    best = None
    for other_id, oc in lang_centroids.items():
        if other_id == leaf_id:
            continue
        oc = oc / (np.linalg.norm(oc) + 1e-9)
        cos = float(np.dot(c, oc))
        if best is None or cos > best:
            best = cos
    return best


def build_merged_quality_report(
    cr,
    skills: dict[int, dict],
    profiles: dict[int, LeafProfile],
    merged_evidence: dict[int, list[str]],
) -> dict:
    """Quality report computed on the merged skill set, not raw leaves."""

    # merged language centroids (size-weighted mean of source leaf centroids)
    merged_centroids: dict[int, np.ndarray] = {}
    for sid, meta in skills.items():
        srcs = [int(x) for x in meta["source_leaves"]]
        weights = [len(cr.get_cluster_members(lid)) for lid in srcs]
        total = sum(weights)
        if total == 0:
            continue
        cent = np.zeros_like(cr.leaf_language_centroids[srcs[0]])
        for lid, w in zip(srcs, weights):
            cent += w * cr.leaf_language_centroids[lid]
        merged_centroids[sid] = cent / total

    centered = _mean_centered_centroids(merged_centroids)

    leaves: dict[int, dict] = {}
    concerns: list[dict] = []
    for sid, meta in skills.items():
        prof = profiles[sid]
        evidence = _evidence_fill(merged_evidence.get(sid, []))
        nearest_cos = _nearest_other_cosine(sid, centered)
        n_typical = len(prof.typical_utterances)
        n_candidates = prof.n_candidates
        dedup_ratio = round(n_typical / n_candidates, 3) if n_candidates else 0.0

        # action silhouette: weighted mean of source leaf silhouettes
        srcs = [int(x) for x in meta["source_leaves"]]
        weights = [len(cr.get_cluster_members(lid)) for lid in srcs]
        sils = [cr.leaf_silhouette.get(lid) for lid in srcs
                if cr.leaf_silhouette.get(lid) is not None]
        merged_sil = float(np.average(
            sils,
            weights=[w for lid, w in zip(srcs, weights)
                     if cr.leaf_silhouette.get(lid) is not None]
        )) if sils else None

        leaves[sid] = {
            "size": prof.size,
            "merged_action_silhouette": merged_sil,
            "n_representative_members": len(prof.members),
            "n_typical_utterances": n_typical,
            "n_candidates": n_candidates,
            "dedup_ratio": dedup_ratio,
            "evidence_fill": evidence,
            "n_tags": len(prof.tags),
            "tags": prof.tags,
            "nearest_other_skill_cosine": nearest_cos,
        }

        issues: list[str] = []
        if merged_sil is not None and merged_sil < 0.05:
            issues.append(f"low merged action silhouette ({merged_sil:.3f} < 0.05)")
        if n_candidates < 10:
            issues.append(f"too few candidate utterances ({n_candidates} < 10)")
        if len(prof.tags) == 0:
            issues.append("no behavioral tags — skill may be generic/indistinguishable")
        total_filled = sum(v["filled"] for v in evidence.values())
        total_budget = sum(v["budget"] for v in evidence.values())
        if total_budget and total_filled / total_budget < 0.3:
            issues.append(
                f"low rejection-evidence fill ({total_filled}/{total_budget}) — weak anti-pattern grounding"
            )
        if nearest_cos is not None and nearest_cos > 0.70:
            issues.append(
                f"language centroid close to another skill (cosine {nearest_cos:.3f} > 0.70)"
            )
        if issues:
            concerns.append({"skill_id": sid, "issues": issues})

    mean_cos = (
        float(np.mean([v["nearest_other_skill_cosine"] for v in leaves.values()
                       if v["nearest_other_skill_cosine"] is not None]))
        if leaves else None
    )

    return {
        "n_final_skills": len(skills),
        "n_users": len(cr.labels),
        "silhouette": cr.silhouette_score,
        "davies_bouldin": cr.davies_bouldin_score,
        "mean_skill_nearest_cosine": mean_cos,
        "skills": leaves,
        "concerns": concerns,
    }


def main():
    cr = pickle.load(open(CLUSTER_PATH, "rb"))
    print(f"Loaded clustering: K={cr.n_clusters}, n_users={len(cr.labels):,}")

    plan = load_merge_plan(MAP_PATH)
    merge_map = plan["merge_map"]
    skills = plan["skills"]
    print(f"Merge plan loaded: {len(skills)} final skills, merge_map={merge_map}")

    year_dirs = find_year_dirs(DATA_DIR)
    leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]

    # Cached BGE embeddings for centroid-closest representative selection.
    emb_path = Path("outputs/stream_cache/embeddings.pkl")
    embeddings = pickle.load(open(emb_path, "rb")) if emb_path.exists() else {}

    # ------------------------------------------------------------------
    # 1. Select representative members per *source leaf*.
    # ------------------------------------------------------------------
    leaf_rep_members: dict[int, list[str]] = {}
    all_members: set[str] = set()

    for lid in leaf_ids:
        members = [u for u in cr.get_cluster_members(lid) if u in embeddings]
        if not members:
            members = cr.get_cluster_members(lid)[:REP_PER_LEAF]
            leaf_rep_members[lid] = members
            all_members.update(members)
            continue
        cent = cr.leaf_language_centroids[lid]
        ranked = sorted(members, key=lambda u: float(np.linalg.norm(embeddings[u] - cent)))
        top = ranked[:REP_PER_LEAF]
        leaf_rep_members[lid] = top
        all_members.update(top)

    # ------------------------------------------------------------------
    # 2. Collect utterances and rejection evidence for all reps.
    # ------------------------------------------------------------------
    member_utts = collect_member_utterances(
        year_dirs, all_members, max_per_user=MAX_PER_USER, workers=WORKERS,
    )

    user_to_leaf = {u: int(lid) for lid in leaf_ids for u in cr.get_cluster_members(lid)}
    leaf_evidence = collect_rejection_evidence(year_dirs, user_to_leaf, workers=WORKERS)

    # ------------------------------------------------------------------
    # 3. Build merged LeafProfiles.
    # ------------------------------------------------------------------
    leaf_feat = {}
    for lid in leaf_ids:
        vecs = [cr.user_features[u].to_vector() for u in cr.get_cluster_members(lid)
                if u in cr.user_features]
        if vecs:
            leaf_feat[lid] = np.mean(vecs, axis=0)

    # Recompute tags per source leaf so we can merge them deterministically.
    leaf_tags = ArchetypeProfiler()._compute_tags(leaf_feat)

    profiles: dict[int, LeafProfile] = {}
    for sid, meta in skills.items():
        src_leaves = [int(x) for x in meta["source_leaves"]]

        # merge rep members, preserving order and deduplicating
        merged_reps: list[str] = []
        seen = set()
        for lid in src_leaves:
            for m in leaf_rep_members.get(lid, []):
                if m not in seen:
                    merged_reps.append(m)
                    seen.add(m)

        # merge typical utterances from all reps
        utts = []
        for m in merged_reps:
            for it in member_utts.get(m, []):
                utts.append(TypicalUtterance(
                    member=m, action=it["action"], text=it["text"],
                    parent_context=it["parent_context"], topic=it["topic"],
                ))

        n_candidates = sum(len(member_utts.get(m, [])) for m in merged_reps)

        # merge tags and size
        merged_tags = sorted({tag for lid in src_leaves for tag in leaf_tags.get(lid, [])})
        merged_size = sum(len(cr.get_cluster_members(lid)) for lid in src_leaves)

        profiles[sid] = LeafProfile(
            leaf_id=sid,
            members=merged_reps,
            typical_utterances=utts,
            tags=merged_tags,
            size=merged_size,
            n_candidates=n_candidates,
        )

    # ------------------------------------------------------------------
    # 4. Export merged packs (overwrite previous per-leaf output).
    # ------------------------------------------------------------------
    platform_dir = OUT_DIR / "wikiconv"
    # remove obsolete cluster_* dirs that are not in final skills
    for d in platform_dir.glob("cluster_*"):
        try:
            idx = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        if idx not in skills:
            shutil.rmtree(d)
            print(f"Removed obsolete {d}")

    # merge rejection evidence for final skills
    merged_evidence: dict[int, list[str]] = {}
    for sid, meta in skills.items():
        lines = []
        seen = set()
        for lid in meta["source_leaves"]:
            for line in leaf_evidence.get(int(lid), []):
                if line not in seen:
                    lines.append(line)
                    seen.add(line)
        merged_evidence[sid] = lines

    profiler = ArchetypeProfiler()
    platform = "wikiconv"
    profiler.save(profiles, OUT_DIR, platform)
    export_corpus_packs([], cr, profiles, OUT_DIR, platform, leaf_evidence=merged_evidence)

    total_utts = sum(len(p.typical_utterances) for p in profiles.values())
    print(
        f"Exported {len(profiles)} merged skill packs to {platform_dir} "
        f"(total typical utterances: {total_utts:,})"
    )

    # ------------------------------------------------------------------
    # 5. Quality report on merged skills.
    # ------------------------------------------------------------------
    quality = build_merged_quality_report(cr, skills, profiles, merged_evidence)
    quality["merge_map"] = {str(k): v for k, v in merge_map.items()}
    with open(platform_dir / "quality_report.json", "w") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)
    print(
        f"Quality report → {platform_dir / 'quality_report.json'} "
        f"({len(quality['concerns'])}/{quality['n_final_skills']} skills flagged)"
    )


if __name__ == "__main__":
    main()
