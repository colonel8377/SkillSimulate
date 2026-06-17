"""Download Cornell ConvoKit WikiConv corpus and convert to loader schema.

Emits ``data/raw/wikipedia/wiki_wikiconv.jsonl`` with one record per
utterance, in the flat schema ``WikipediaLoader`` (see
``src/data/wikipedia.py``) ingests directly:

    thread_id, page_title, comment_id, user_id, author, text,
    timestamp, parent_comment_id

``action_type`` is intentionally left unset — the loader infers it from
text patterns (revert / report / discuss / edit). WikiConv does not carry
explicit edit-action metadata anyway.

The default download is ``wikiconv-en-2004`` (the smallest English
slice). Override with ``--corpus wikiconv-en-2008`` etc. for larger
samples; see ConvoKit docs for available year slices.

Prerequisites:
    pip install convokit          # also pulls nltk, spacy, etc.
    python -m spacy download en_core_web_sm    # optional but recommended

Usage::

    python scripts/download_wikipedia.py
    python scripts/download_wikipedia.py --corpus wikiconv-en-2004 \\
        --max-threads 500 --force

Outline §5 prerequisite — replace the synthetic dev fixture with this
before any paper-scale run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "wikipedia"
DEFAULT_CORPUS = "wikiconv-en-2004"


def _to_iso(ts) -> str:
    """Coerce ConvoKit timestamp (int epoch / iso str / datetime) to iso."""
    if ts is None:
        return datetime.now().isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now().isoformat()


def _extract_records(corpus, max_threads: int | None) -> list[dict]:
    """Walk a ConvoKit corpus and emit flat JSONL records."""
    records: list[dict] = []
    seen_threads = 0

    for convo in corpus.iter_conversations():
        if max_threads is not None and seen_threads >= max_threads:
            break
        seen_threads += 1

        thread_id = convo.id
        page_title = ""
        meta = getattr(convo, "meta", {}) or {}
        page_title = (
            meta.get("page_title")
            or meta.get("title")
            or meta.get("pageTitle", "")
            or thread_id
        )

        for utt in convo.iter_utterances():
            speaker_id = utt.speaker.id if utt.speaker is not None else ""
            if not speaker_id:
                continue
            text = utt.text or ""
            records.append({
                "thread_id": str(thread_id),
                "page_title": str(page_title),
                "comment_id": str(utt.id),
                "user_id": str(speaker_id),
                "author": str(speaker_id),
                "text": text,
                "timestamp": _to_iso(utt.timestamp),
                "parent_comment_id": (
                    str(utt.reply_to) if utt.reply_to else None
                ),
            })
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help=f"ConvoKit corpus name (default {DEFAULT_CORPUS}).")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="Cap number of conversations extracted (default all).")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "wiki_wikiconv.jsonl",
                        help="Output JSONL path.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file.")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"[skip] {args.out} exists — use --force to overwrite")
        return

    try:
        import convokit
    except ImportError as e:
        raise SystemExit(
            "convokit not installed. Install with: pip install convokit\n"
            "Then optionally: python -m spacy download en_core_web_sm"
        ) from e

    print(f"[download] fetching ConvoKit corpus {args.corpus!r} …")
    corpus = convokit.Corpus(filename=convokit.download(args.corpus))
    print(f"[loaded] {corpus}")

    records = _extract_records(corpus, args.max_threads)
    if not records:
        raise SystemExit(
            "No records extracted — corpus appears empty. "
            "Check the corpus name and ConvoKit download logs."
        )

    write_jsonl(args.out, records)
    n_threads = len({r["thread_id"] for r in records})
    print(f"[wrote] {args.out} — {len(records)} records, {n_threads} threads")


if __name__ == "__main__":
    main()
