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
DATA_DIR = PROJECT_ROOT / "data" / "external" / "convokit"
DEFAULT_CORPUS = "winning-args-corpus"


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

    The canonical key for ``winning-args-corpus`` is ``success``
    (1 = received a delta, 0 = matched non-delta reply, None = other).
    """
    meta = getattr(utt, "meta", {}) or {}
    flag = _meta_get(
        meta,
        "success",
        "delta_awarded",
        "delta_label",
        "is_delta",
        "delta",
        default=None,
    )
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return flag == 1
    if isinstance(flag, str):
        return flag.strip().lower() in {"true", "1", "yes", "delta"}

    body = (utt.text or "").strip().lower()
    if body.startswith("δ") or body.startswith("!delta"):
        return True
    if "delta awarded" in body or "awarded a delta" in body:
        return True
    return False


def _split_corpus(corpus, max_submissions: int | None):
    """Walk the winning-args corpus and split into submissions + comments.

    In ``winning-args-corpus`` each conversation is one CMV thread. The
    root utterance (id ``t3_<id>``, ``reply_to == None``) carries the
    submission body in ``.text``; downstream utterances are comments.
    ConvoKit does not ship the submission title, so we fall back to the
    thread id. Submission timestamp is taken from the first non-None
    utterance timestamp in the thread (the root is often None).
    """
    submissions: list[dict] = []
    comments: list[dict] = []
    seen = 0

    for convo in corpus.iter_conversations():
        if max_submissions is not None and seen >= max_submissions:
            break
        seen += 1

        sub_id_full = str(convo.id)  # e.g. ``t3_2ro9ux``
        sub_id_bare = sub_id_full.replace("t3_", "")
        convo_meta = getattr(convo, "meta", {}) or {}

        # First pass: find the root utterance (the submission) and the
        # earliest non-None timestamp in the thread.
        root_utt = None
        first_ts = None
        utts = list(convo.iter_utterances())
        for utt in utts:
            if root_utt is None and getattr(utt, "reply_to", None) is None:
                root_utt = utt
            ts = getattr(utt, "timestamp", None)
            if ts is not None and first_ts is None:
                first_ts = ts

        # Submission body lives in the root utterance's text field.
        if root_utt is not None:
            selftext = root_utt.text or ""
            op_speaker = root_utt.speaker
            op_author = str(op_speaker.id) if op_speaker is not None and op_speaker.id else "[deleted]"
            sub_ts = root_utt.timestamp if root_utt.timestamp is not None else first_ts
        else:
            selftext = str(_meta_get(convo_meta, "selftext", "body", default=""))
            op_author = str(_meta_get(convo_meta, "author", "op", default="[deleted]"))
            sub_ts = first_ts

        # ConvoKit winning-args ships no submission title in convo.meta.
        # Real CMV titles (e.g. "CMV: man-made things are natural") would
        # require joining with the Reddit pushshift dump. Fall back to the
        # selftext prefix so the downstream ``topic`` field carries real
        # semantic content for clustering rather than a bare submission id.
        title = str(_meta_get(convo_meta, "title", "submission_title", default=""))
        if not title or title == sub_id_bare:
            stripped = (selftext or "").strip().replace("\n", " ")
            title = stripped[:120] if stripped else sub_id_bare

        submissions.append({
            "submission_id": sub_id_bare,
            "id": sub_id_full,
            "author": op_author,
            "title": title,
            "selftext": selftext,
            "created_utc": _to_epoch(sub_ts),
        })

        root_utt_id = str(root_utt.id) if root_utt is not None else sub_id_full

        # Emit one comment per non-root utterance.
        for utt in utts:
            if root_utt is not None and str(utt.id) == root_utt_id:
                continue
            speaker = utt.speaker.id if utt.speaker is not None else "[deleted]"
            if not speaker:
                continue
            parent = utt.reply_to
            # Top-level comments reply to the root submission; ConvoKit
            # stores parents as utterance IDs. Normalise to ``t3_<bare>``
            # for top-level, otherwise keep the comment id.
            if not parent or parent == sub_id_full or parent == sub_id_bare or parent == root_utt_id:
                parent_id = f"t3_{sub_id_bare}"
            else:
                parent_id = str(parent)

            comments.append({
                "submission_id": sub_id_bare,
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
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory to cache downloaded ConvoKit corpora (default data/external/convokit).")
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

    print(f"[download] fetching ConvoKit corpus {args.corpus!r} to {args.data_dir} …")
    dataset_path = convokit.download(args.corpus, data_dir=str(args.data_dir))
    corpus = convokit.Corpus(filename=dataset_path)
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
