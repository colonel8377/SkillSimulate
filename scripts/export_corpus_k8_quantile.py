"""Export skill corpus directly from a pre-generated clustering pickle.

This avoids re-running the clustering pipeline; it loads
``outputs/stream_cache/clustering_k8_final_quantile.pkl`` and produces the same
per-leaf material packs that ``src.main export-corpus`` would emit.

Output: outputs/skill_corpus_k8_quantile/wikiconv/
"""

from __future__ import annotations

import sys, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.streaming import (
    find_year_dirs,
    collect_member_utterances,
    collect_rejection_evidence,
)
from src.skill.cluster_profile import ArchetypeProfiler, LeafProfile, TypicalUtterance
from src.skill.corpus_export import export_corpus_packs
from src.skill.quality_report import build_quality_report

DATA_DIR = "data/raw/wikiconv_en"
CLUSTER_PATH = "outputs/stream_cache/clustering_k8_final_quantile.pkl"
OUT_DIR = Path("outputs/skill_corpus_k8_quantile")
WORKERS = 8


def main():
    cr = pickle.load(open(CLUSTER_PATH, "rb"))
    print(f"Loaded clustering: K={cr.n_clusters}, n_users={len(cr.labels):,}")

    year_dirs = find_year_dirs(DATA_DIR)
    leaf_ids = [l for l in cr.get_cluster_ids() if l >= 0]

    # Representative members: top-8 closest to leaf language centroid.
    # Re-use cached BGE embeddings from the clustering pickle if present,
    # otherwise fall back to leaf_language_centroids only.
    emb_path = Path("outputs/stream_cache/embeddings.pkl")
    if emb_path.exists():
        embeddings = pickle.load(open(emb_path, "rb"))
    else:
        embeddings = {}

    rep_members: dict[int, list[str]] = {}
    all_members: set[str] = set()
    import numpy as np

    for lid in leaf_ids:
        members = [u for u in cr.get_cluster_members(lid) if u in embeddings]
        if not members:
            # no embeddings — take any members
            members = cr.get_cluster_members(lid)[:8]
            rep_members[lid] = members
            all_members.update(members)
            continue
        cent = cr.leaf_language_centroids[lid]
        ranked = sorted(members, key=lambda u: float(np.linalg.norm(embeddings[u] - cent)))
        top = ranked[:8]
        rep_members[lid] = top
        all_members.update(top)

    member_utts = collect_member_utterances(
        year_dirs, all_members, max_per_user=60, workers=WORKERS,
    )

    user_to_leaf = {u: int(lid) for lid in leaf_ids for u in cr.get_cluster_members(lid)}
    leaf_evidence = collect_rejection_evidence(year_dirs, user_to_leaf, workers=WORKERS)

    # behavioral tags
    leaf_feat = {}
    for lid in leaf_ids:
        vecs = [cr.user_features[u].to_vector() for u in cr.get_cluster_members(lid)
                if u in cr.user_features]
        if vecs:
            leaf_feat[lid] = np.mean(vecs, axis=0)
    tags = ArchetypeProfiler()._compute_tags(leaf_feat)

    profiles: dict[int, LeafProfile] = {}
    for lid in leaf_ids:
        members = cr.get_cluster_members(lid)
        utts = []
        for m in rep_members.get(lid, []):
            for it in member_utts.get(m, []):
                utts.append(TypicalUtterance(
                    member=m, action=it["action"], text=it["text"],
                    parent_context=it["parent_context"], topic=it["topic"],
                ))
        n_candidates = sum(len(member_utts.get(m, [])) for m in rep_members.get(lid, []))
        profiles[lid] = LeafProfile(
            leaf_id=lid, members=rep_members.get(lid, []),
            typical_utterances=utts, tags=tags.get(lid, []), size=len(members),
            n_candidates=n_candidates,
        )

    profiler = ArchetypeProfiler()
    platform = "wikiconv"
    profiler.save(profiles, OUT_DIR, platform)
    export_corpus_packs([], cr, profiles, OUT_DIR, platform, leaf_evidence=leaf_evidence)

    n_ev = sum(1 for v in leaf_evidence.values() if v)
    print(
        f"Exported {len(profiles)} leaf packs to {OUT_DIR / platform} "
        f"(rejection-evidence populated for {n_ev}/{len(profiles)} leaves)"
    )

    quality = build_quality_report(cr, profiles, leaf_evidence, user_embeddings=embeddings)
    with open(OUT_DIR / platform / "quality_report.json", "w") as f:
        import json
        json.dump(quality, f, ensure_ascii=False, indent=2)
    print(
        f"Quality report → {OUT_DIR / platform / 'quality_report.json'} "
        f"({len(quality['concerns'])}/{quality['n_leaves']} leaves flagged)"
    )


if __name__ == "__main__":
    main()
