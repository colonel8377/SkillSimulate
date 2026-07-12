"""Pre-compute per-cluster toxic keywords from ConvoKit toxicity labels.

Fully corpus-driven: no hardcoded lexicon. For each cluster, we compare
word frequency in high-toxicity utterances vs. low-toxicity utterances,
then select words that are disproportionately associated with toxic
content in that cluster. This captures cluster-specific hostile
language (e.g. "crap", "stupid", "moron" for edit-war clusters;
"fuck", "nigger" for attack clusters) without a predetermined word list.

Output: data/cluster_toxic_keywords.json

Performance: processes year-corpora in parallel via multiprocessing.
~50M utterances across 18 years; each year processed independently.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLUSTER_PICKLE_PATH = Path("outputs/stream_cache/clustering_k8_final_quantile.pkl")
RAW_DIR = Path("data/raw/wikiconv_en")
OUTPUT_PATH = Path("data/cluster_toxic_keywords.json")

# Stopwords / generic function words to exclude from toxic-keyword
# extraction. These are never distinctive of toxic content regardless
# of frequency.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "if", "then", "than", "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
    "when", "where", "how", "why", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "any", "only", "own", "same",
    "so", "too", "very", "just", "also", "about", "up", "out", "into",
    "over", "after", "before", "between", "under", "again", "there",
    "here", "once", "during", "while", "as", "until", "because",
    "through", "above", "below", "against", "further",
    # Wikipedia-specific generic words
    "article", "page", "wikipedia", "edit", "section", "talk", "wiki",
    "please", "thanks", "think", "know", "like", "get", "make", "go",
    "see", "way", "want", "need", "use", "good", "well", "look",
    "really", "right", "thing", "things", "something", "much", "many",
}


def _clean_word(w: str) -> str | None:
    """Filter out noise tokens from word extraction.

    Rejects: repeated chars (lololol, aaaaa), mixed garbage,
    URL fragments, usernames with digits. Keeps real English words
    and common profanity/slurs.
    """
    if len(w) < 3 or len(w) > 12:
        return None
    # Reject words with 3+ consecutive repeated chars
    if re.search(r"(.)\1{2,}", w):
        return None
    # Reject words that are mostly one char (abababab)
    if len(set(w)) <= 2:
        return None
    # Reject words with too many consecutive consonants (keyboard smash)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", w):
        return None
    return w


def build_member_to_cluster(pickle_path: Path, cluster_ids: list[int]) -> dict[str, int]:
    with open(pickle_path, "rb") as f:
        result = pickle.load(f)
    mapping: dict[str, int] = {}
    for cid in cluster_ids:
        for member in result.get_cluster_members(cid):
            mapping[member] = cid
    return mapping


def _process_year(args: tuple) -> dict:
    """Process one year directory. Returns {cluster_id: {toxic: Counter, nontoxic: Counter}}."""
    year_dir, member_to_cluster, tox_threshold, low_tox_threshold = args
    utt_path = year_dir / "utterances.jsonl"
    if not utt_path.exists():
        return {}

    toxic_counts: dict[int, Counter] = defaultdict(Counter)
    nontoxic_counts: dict[int, Counter] = defaultdict(Counter)
    n_processed = 0
    with open(utt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            speaker = d.get("speaker", "")
            cid = member_to_cluster.get(speaker)
            if cid is None:
                continue

            n_processed += 1
            tox = d.get("meta", {}).get("toxicity", 0)

            if tox >= tox_threshold:
                text = d.get("text", "")
                words = [_clean_word(w) for w in re.findall(r"[a-z]+", text.lower())
                         if w not in _STOPWORDS and 3 <= len(w) <= 12]
                words = [w for w in words if w]  # remove None from _clean_word
                toxic_counts[cid].update(words)
            elif tox <= low_tox_threshold:
                text = d.get("text", "")
                words = [_clean_word(w) for w in re.findall(r"[a-z]+", text.lower())
                         if w not in _STOPWORDS and 3 <= len(w) <= 12]
                words = [w for w in words if w]
                nontoxic_counts[cid].update(words)

    result = {}
    all_cids = set(toxic_counts.keys()) | set(nontoxic_counts.keys())
    for cid in all_cids:
        result[cid] = {
            "toxic": dict(toxic_counts.get(cid, Counter())),
            "nontoxic": dict(nontoxic_counts.get(cid, Counter())),
        }
    return {"year": year_dir.name, "n_processed": n_processed, "data": result}


def extract(
    raw_dir: Path,
    member_to_cluster: dict[str, int],
    tox_threshold: float = 0.7,
    low_tox_threshold: float = 0.2,
    min_occurrences: int = 5,
    top_k: int = 50,
    min_log_odds: float = 0.5,
    workers: int | None = None,
):
    """Extract toxic keywords per cluster using log-odds ratio.

    Processes year-corpora in parallel via multiprocessing.
    """
    # Collect year directories
    year_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("wikiconv-")
        and (d / "utterances.jsonl").exists()
    ])
    if not year_dirs:
        print(f"No WikiConv year directories found in {raw_dir}")
        return {}

    print(f"Processing {len(year_dirs)} year corpora with {workers or cpu_count()} workers")

    # Build per-year args (member_to_cluster is read-only, safe to share)
    args_list = [
        (yd, member_to_cluster, tox_threshold, low_tox_threshold)
        for yd in year_dirs
    ]

    # Merge results across years
    merged_toxic: dict[int, Counter] = defaultdict(Counter)
    merged_nontoxic: dict[int, Counter] = defaultdict(Counter)

    n_workers = min(workers or cpu_count(), len(year_dirs))
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_process_year, args_list):
            if not result:
                continue
            year_name = result["year"]
            n_proc = result["n_processed"]
            if n_proc > 0:
                print(f"  {year_name}: {n_proc:,} utterances from cluster members")
            for cid, counts in result["data"].items():
                merged_toxic[cid].update(counts["toxic"])
                merged_nontoxic[cid].update(counts["nontoxic"])

    # Score and select keywords per cluster
    result: dict[str, list[str]] = {}
    for cid in sorted(merged_toxic.keys()):
        scored: list[tuple[str, float, int]] = []
        tox_total = sum(merged_toxic[cid].values()) or 1
        ntox_total = sum(merged_nontoxic[cid].values()) or 1
        for word, count in merged_toxic[cid].items():
            if count < min_occurrences:
                continue
            p_tox = (count + 1) / (tox_total + 1)
            ntox_count = merged_nontoxic[cid].get(word, 0)
            p_ntox = (ntox_count + 1) / (ntox_total + 1)
            log_odds = math.log(p_tox / p_ntox)
            if log_odds >= min_log_odds:
                scored.append((word, log_odds, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        result[str(cid)] = [w for w, _, _ in scored[:top_k]]

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tox-threshold", type=float, default=0.7)
    parser.add_argument("--low-tox-threshold", type=float, default=0.2)
    parser.add_argument("--min-occurrences", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-log-odds", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cluster_ids = [0, 2, 3, 4, 6, 7]
    mapping = build_member_to_cluster(CLUSTER_PICKLE_PATH, cluster_ids)
    print(f"Loaded {len(mapping)} cluster member mappings")
    keywords = extract(
        RAW_DIR, mapping,
        tox_threshold=args.tox_threshold,
        low_tox_threshold=args.low_tox_threshold,
        min_occurrences=args.min_occurrences,
        top_k=args.top_k,
        min_log_odds=args.min_log_odds,
        workers=args.workers,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(keywords, indent=2))
    print(f"\nSaved {out_path}")
    for cid, words in keywords.items():
        print(f"  cluster {cid}: {len(words)} keywords")
        if words:
            print(f"    top 10: {words[:10]}")


if __name__ == "__main__":
    main()
