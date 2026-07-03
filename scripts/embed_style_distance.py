"""
Re-embed all active users with StyleDistance/styledistance.

StyleDistance (NAACL 2025) is trained on SynthSTEL: 40 style features with
synthetic parallel examples (same content, different style). This makes it
explicitly content-independent — critical for splitting leaf_3 whose users
all discuss similar Wikipedia article topics but may have different registers.

Output: outputs/stream_cache/embeddings_styledistance.pkl
  dict[user_id -> np.ndarray(shape=(1536,), float32)]
  1536 = 768 (mean) + 768 (std) pooling over per-text embeddings
"""

import glob, pickle, sys, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clustering.streaming import stratified_sample_texts, UserAccum

MODEL_ID   = "StyleDistance/styledistance"
BATCH_SIZE = 1024
MIN_MSGS   = 5
OUT_PATH   = Path("outputs/stream_cache/embeddings_styledistance.pkl")


def main():
    print(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device="cuda")
    dim = model.get_embedding_dimension()
    print(f"  dim={dim}, pooled output dim={dim*2}")

    # Load all v6 accums
    pkls = sorted(glob.glob("outputs/stream_cache/detail_v6_*.pkl"))
    print(f"Loading {len(pkls)} year caches…")
    accums: dict[str, UserAccum] = {}
    for p in pkls:
        year_accums = pickle.load(open(p, "rb"))
        for uid, a in year_accums.items():
            if a.n >= MIN_MSGS:
                if uid not in accums:
                    accums[uid] = a
                else:
                    # merge: take the one with more messages
                    if a.n > accums[uid].n:
                        accums[uid] = a
    print(f"  active users (n>={MIN_MSGS}): {len(accums)}")

    # Flatten all users' texts into one encode queue
    users = list(accums.keys())
    flat_texts: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    for uid in users:
        texts = [t for t in stratified_sample_texts(accums[uid]) if t.strip()] or [" "]
        start = len(flat_texts)
        flat_texts.extend(texts)
        ranges[uid] = (start, len(flat_texts))

    print(f"  total texts to encode: {len(flat_texts):,}")
    print(f"  avg texts/user: {len(flat_texts)/len(users):.1f}")

    # Encode in batches
    t0 = time.time()
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(flat_texts), BATCH_SIZE):
        batch = flat_texts[i : i + BATCH_SIZE]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs.astype(np.float32))
        if (i // BATCH_SIZE) % 50 == 0:
            done = min(i + BATCH_SIZE, len(flat_texts))
            elapsed = time.time() - t0
            eta = elapsed / done * (len(flat_texts) - done)
            print(f"  {done:>8,}/{len(flat_texts):,}  {done/elapsed:.0f} texts/s  ETA {eta/60:.1f}min")

    allv = np.vstack(all_vecs)  # (total_texts, dim)
    elapsed = time.time() - t0
    print(f"Encoding done in {elapsed/60:.1f} min")

    # Pool per user: mean + std → (dim*2,)
    out: dict[str, np.ndarray] = {}
    for uid, (s, e) in ranges.items():
        block = allv[s:e]
        out[uid] = np.concatenate([block.mean(axis=0), block.std(axis=0)])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT_PATH, "wb"), protocol=4)
    print(f"Saved {len(out)} user embeddings → {OUT_PATH}")
    print(f"  embed shape: {next(iter(out.values())).shape}")


if __name__ == "__main__":
    main()
