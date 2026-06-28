"""Q5: which pooling preserves per-user STYLE best?

No style ground truth exists, so we use a label-free standard protocol:

  split-half self-consistency retrieval —
    For each user with >= 8 sampled texts, split their texts into two halves,
    pool each half independently into vecA, vecB. A pooling that captures stable
    individual style makes vecA's nearest neighbour (over all users' vecB) be the
    SAME user. Report rank-1 accuracy + MRR. Higher = style better preserved.

Secondary (label-free):
  - mean pairwise cosine between user vectors (lower = users more separable)
  - effective dimensionality (participation ratio; higher = richer representation)

Pooling variants compared: mean (current), max, mean+std, mean+max.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.streaming import (  # noqa: E402
    stream_counts, _json_loads, _speaker,
)

YEARS = [Path("data/raw/wikiconv_en/wikiconv-2003"),
         Path("data/raw/wikiconv_en/wikiconv-2004")]
MIN_MSGS = 50
MAX_TEXTS = 16          # cap per user (bounds GPU cost)
MIN_TEXTS_SPLIT = 8     # need >=8 to split 4/4
TEXT_CHARS = 512
CACHE = "outputs/stream_cache"


def collect_texts(active):
    """Up to MAX_TEXTS texts per active user (first-K, deterministic)."""
    texts = defaultdict(list)
    for cdir in YEARS:
        with open(cdir / "utterances.jsonl", "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                r = _json_loads(line)
                sp = _speaker(r.get("speaker"))
                if not sp or sp not in active:
                    continue
                t = (r.get("text") or "").strip()
                if t and len(texts[sp]) < MAX_TEXTS:
                    texts[sp].append(t[:TEXT_CHARS])
    return texts


def pool(embs, kind):
    """embs: (k, d) per-text embeddings -> pooled vector."""
    if kind == "mean":
        return embs.mean(axis=0)
    if kind == "max":
        return embs.max(axis=0)
    if kind == "mean+std":
        return np.concatenate([embs.mean(axis=0), embs.std(axis=0)])
    if kind == "mean+max":
        return np.concatenate([embs.mean(axis=0), embs.max(axis=0)])
    raise ValueError(kind)


def l2norm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def split_half_retrieval(per_user_embs, kind):
    """Build vecA/vecB per user via two halves; rank-1 acc + MRR."""
    users, A, B = [], [], []
    for u, embs in per_user_embs.items():
        if len(embs) < MIN_TEXTS_SPLIT:
            continue
        half = len(embs) // 2
        A.append(pool(embs[:half], kind))
        B.append(pool(embs[half:], kind))
        users.append(u)
    A = l2norm(np.array(A)); B = l2norm(np.array(B))
    S = A @ B.T                       # cosine sim (normalized)
    n = S.shape[0]
    nn = S.argmax(axis=1)
    rank1 = float((nn == np.arange(n)).mean())
    # MRR
    order = np.argsort(-S, axis=1)
    ranks = np.where(order == np.arange(n)[:, None])[1] + 1
    mrr = float((1.0 / ranks).mean())
    return n, rank1, mrr


def separability(per_user_embs, kind):
    """Full-user pooled matrix: mean pairwise cosine + effective dim."""
    V = l2norm(np.array([pool(e, kind) for e in per_user_embs.values()]))
    n = V.shape[0]
    # mean pairwise cosine (upper triangle); subsample if large
    rng = np.random.default_rng(42)
    idx = rng.choice(n, min(n, 1500), replace=False)
    Vs = V[idx]
    C = Vs @ Vs.T
    iu = np.triu_indices(Vs.shape[0], k=1)
    mean_cos = float(C[iu].mean())
    # effective dimensionality (participation ratio of covariance eigenvalues)
    Vc = V - V.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov(Vc, rowvar=False))
    ev = ev[ev > 0]
    eff_dim = float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else 0.0
    return mean_cos, eff_dim


def main():
    counts, _ = stream_counts(YEARS, workers=2, cache_dir=CACHE)
    active = {u for u, c in counts.items() if c >= MIN_MSGS}
    print(f"active (>= {MIN_MSGS}): {len(active)} users")

    texts = collect_texts(active)
    n_split = sum(1 for v in texts.values() if len(v) >= MIN_TEXTS_SPLIT)
    print(f"users with texts: {len(texts)}, with >= {MIN_TEXTS_SPLIT} texts: {n_split}")

    # encode every text once, individually
    from src.config.settings import get_shared_embedder
    embedder = get_shared_embedder()
    flat, ranges = [], {}
    for u, ts in texts.items():
        s = len(flat); flat.extend(ts); ranges[u] = (s, len(flat))
    print(f"encoding {len(flat)} texts ...")
    E = np.asarray(embedder.encode(flat, show_progress_bar=True, batch_size=256))
    per_user = {u: E[s:e] for u, (s, e) in ranges.items()}

    print(f"\n{'='*72}")
    print(f"{'pooling':<12}{'dim':>6}{'users':>7}{'rank1':>9}{'MRR':>8}{'meanCos':>10}{'effDim':>9}")
    print(f"{'(split-half: higher rank1/MRR = style better preserved)':<72}")
    print('='*72)
    base_dim = E.shape[1]
    for kind in ("mean", "max", "mean+std", "mean+max"):
        dim = base_dim * (2 if "+" in kind else 1)
        n, rank1, mrr = split_half_retrieval(per_user, kind)
        mean_cos, eff_dim = separability(per_user, kind)
        print(f"{kind:<12}{dim:>6}{n:>7}{rank1:>9.3f}{mrr:>8.3f}{mean_cos:>10.3f}{eff_dim:>9.1f}")
    print('='*72)
    print("rank1 = P(same user's other half is nearest neighbour); higher better")
    print("meanCos lower = users more spread/separable; effDim higher = richer")


if __name__ == "__main__":
    main()
