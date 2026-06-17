"""Download Cornell ConvoKit "Winning Arguments" (r/changemyview) corpus.

Emits two files into ``data/raw/reddit/`` matching ``RedditLoader``
(see ``src/data/reddit.py``):

  * ``reddit_submissions.jsonl``  — one record per CMV submission
  * ``reddit_comments.jsonl``     — one record per comment, with
    ``delta_awarded`` and ``parent_comment_id`` populated

The RedditLoader keys off ``*submissions*.jsonl`` and ``*comments*.jsonl``
globs, so existing synthetic fixtures can co-exist; use ``--force`` to
overwrite or pick a non-default ``--out-prefix`` to switch files.

ConvoKit "winning-args" corpus (Tan, Niculae, Danescu-Niculescu-Mizil
2016) covers r/changemyview 2013-01-01 → 2015-05-07 and ships per-comment
delta metadata, which is exactly what the CADP persuasion metric needs.

Prerequisite:
    pip install convokit

Usage::

    python scripts/download_reddit.py
    python scripts/download_reddit.py --max-submissions 200 --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "reddit"
DEFAULT_CORPUS = "winning-args"


def _to_epoch(ts) -> int:
    """Coerce ConvoKit timestamp to POSIX seconds (RedditLoader expects int)."""
    if ts is None:
        return int(datetime.now().timestamp())
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, datetime):
        return int(ts.timestamp())
    try:
        return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return int(datetime.now().timestamp())


def _meta_get(meta: dict, *keys, default=None):
    """Fetch first present key from a ConvoKit meta dict."""
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


def _is_delta_awarded(utt) -> bool:
    """Detect delta awarded from per-utterance metadata.

    ConvoKit winning-args stores delta info under several alternative
    keys depending on corpus version. We accept any of them and fall
    back to scanning the body for the ``Δ`` / ``!delta`` markers CMV
    uses to award a delta.
    """
    meta = getattr(utt, "meta", {}) or {}
    flag = _meta_get(
        meta,
        "delta_awarded",
        "delta_label",
        "is_delta",
        "delta",
        default=None,
    )
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return bool(flag)
    if isinstance(flag, str):
        return flag.strip().lower() in {"true", "1", "yes", "delta"}

    body = (utt.text or "").strip().lower()
    if body.startswith("δ") or body.startswith("!delta"):
        return True
    if "delta awarded" in body or "awarded a delta" in body:
        return True
    return False


def _split_corpus(corpus, max_submissions: int | None):
    """Walk the winning-args corpus and split into submissions + comments."""
    submissions: list[dict] = []
    comments: list[dict] = []
    seen = 0

    for convo in corpus.iter_conversations():
        if max_submissions is not None and seen >= max_submissions:
            break
        seen += 1

        sub_id = str(convo.id)
        convo_meta = getattr(convo, "meta", {}) or {}
        title = str(_meta_get(convo_meta, "title", "submission_title", default=sub_id))
        selftext = str(_meta_get(convo_meta, "selftext", "body", "submission_body", default=""))
        op_author = str(_meta_get(convo_meta, "author", "op", "submission_author", default="[deleted]"))
        created = _meta_get(convo_meta, "created_utc", "created", "timestamp", default=None)

        submissions.append({
            "submission_id": sub_id,
            "id": sub_id,
            "author": op_author,
            "title": title,
            "selftext": selftext,
            "created_utc": _to_epoch(created),
        })

        # Top-level (depth 0) comments reply to the submission t3_<id>.
        for utt in convo.iter_utterances():
            speaker = utt.speaker.id if utt.speaker is not None else "[deleted]"
            if not speaker:
                continue
            parent = utt.reply_to
            # ConvoKit stores parents as utterance IDs; the loader expects
            # either ``t3_<submission_id>`` (top-level) or a comment id.
            if not parent or parent == sub_id or parent == convo.root.id:
                parent_id = f"t3_{sub_id}"
            else:
                parent_id = str(parent)

            comments.append({
                "submission_id": sub_id,
                "comment_id": str(utt.id),
                "author": str(speaker),
                "body": utt.text or "",
                "parent_comment_id": parent_id,
                "created_utc": _to_epoch(utt.timestamp),
                "delta_awarded": _is_delta_awarded(utt),
                "submission_title": title,
            })

    return submissions, comments


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help=f"ConvoKit corpus name (default {DEFAULT_CORPUS!r}).")
    parser.add_argument("--max-submissions", type=int, default=None,
                        help="Cap number of CMV submissions extracted (default all).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory (default data/raw/reddit).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing JSONL files.")
    args = parser.parse_args()

    subs_path = args.out_dir / "reddit_submissions.jsonl"
    comms_path = args.out_dir / "reddit_comments.jsonl"
    if (subs_path.exists() or comms_path.exists()) and not args.force:
        print(f"[skip] {subs_path} / {comms_path} exist — use --force to overwrite")
        return

    try:
        import convokit
    except ImportError as e:
        raise SystemExit(
            "convokit not installed. Install with: pip install convokit"
        ) from e

    print(f"[download] fetching ConvoKit corpus {args.corpus!r} …")
    corpus = convokit.Corpus(filename=convokit.download(args.corpus))
    print(f"[loaded] {corpus}")

    submissions, comments = _split_corpus(corpus, args.max_submissions)
    if not submissions:
        raise SystemExit("No submissions extracted — corpus appears empty.")

    write_jsonl(subs_path, submissions)
    write_jsonl(comms_path, comments)
    print(
        f"[wrote] {subs_path} — {len(submissions)} submissions; "
        f"{comms_path} — {len(comments)} comments "
        f"({sum(1 for c in comments if c['delta_awarded'])} with delta)"
    )


if __name__ == "__main__":
    main()
