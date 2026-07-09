"""Pre-compute per-cluster toxic keywords from ConvoKit toxicity labels.

The distillation pipeline uses these keywords for Tier 3 anti-pattern
enforcement. They are derived from the actual corpus, not hardcoded.

Output: data/cluster_toxic_keywords.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLUSTER_PICKLE_PATH = Path("outputs/stream_cache/clustering_k8_final_quantile.pkl")
RAW_DIR = Path("data/raw/wikiconv_en")
OUTPUT_PATH = Path("data/cluster_toxic_keywords.json")

# Unambiguous profanity / slur words. Used as a FILTER over the corpus:
# only words that both appear in this set AND occur in high-toxicity
# utterances from a cluster are emitted for that cluster.
_TOXIC_LEXICON = {
    "fuck", "fucking", "fuckin", "fucker", "fucked",
    "shit", "shitty",
    "dick", "dickhead",
    "cock",
    "cunt",
    "bitch",
    "bastard",
    "ass", "asshole",
    "fag", "fags", "faggot",
    "slut", "whore",
    "retard", "retarded",
    "nigger", "niggers", "nigga",
    "kike", "kikes",
    "wetback", "wetbacks",
    "spic", "spics",
    "chink", "chinks",
    "dyke",
    "tranny",
}


def build_member_to_cluster(pickle_path: Path, cluster_ids: list[int]) -> dict[str, int]:
    with open(pickle_path, "rb") as f:
        result = pickle.load(f)
    mapping: dict[str, int] = {}
    for cid in cluster_ids:
        for member in result.get_cluster_members(cid):
            mapping[member] = cid
    return mapping


def extract(raw_dir: Path, member_to_cluster: dict[str, int], tox_threshold: float, min_occurrences: int):
    counts: defaultdict[int, Counter] = defaultdict(Counter)

    for year_dir in sorted(raw_dir.iterdir()):
        utt_path = year_dir / "utterances.jsonl"
        if not utt_path.exists():
            continue
        with open(utt_path) as f:
            for line in f:
                d = json.loads(line)
                tox = d.get("meta", {}).get("toxicity", 0)
                if tox < tox_threshold:
                    continue
                speaker = d.get("speaker", "")
                cid = member_to_cluster.get(speaker)
                if cid is None:
                    continue
                text = d.get("text", "")
                words = set(re.findall(r"[a-z]+", text.lower()))
                for w in words:
                    if w in _TOXIC_LEXICON:
                        counts[cid][w] += 1

    return {
        str(cid): sorted([w for w, c in counts[cid].most_common() if c >= min_occurrences])
        for cid in sorted(counts.keys())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tox-threshold", type=float, default=0.7)
    parser.add_argument("--min-occurrences", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cluster_ids = [0, 2, 3, 4, 6, 7]
    mapping = build_member_to_cluster(CLUSTER_PICKLE_PATH, cluster_ids)
    keywords = extract(RAW_DIR, mapping, args.tox_threshold, args.min_occurrences)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(keywords, indent=2))
    print(f"Saved {out_path}")
    for cid, words in keywords.items():
        print(f"  cluster {cid}: {len(words)} keywords")


if __name__ == "__main__":
    main()
