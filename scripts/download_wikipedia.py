"""Download Cornell ConvoKit Wikipedia Talk Pages corpus and convert to loader schema.

Emits ``data/raw/wikipedia/wiki_wikiconv.jsonl`` with one record per
utterance, in the flat schema ``WikipediaLoader`` (see
``src/data/wikipedia.py``) ingests directly:

    thread_id, page_title, comment_id, user_id, author, text,
    timestamp, parent_comment_id

``action_type`` is intentionally left unset — the loader infers it from
text patterns (revert / report / discuss / edit). WikiConv does not carry
explicit edit-action metadata anyway.

Default corpus is ``wiki-corpus`` (stable, medium-size English Wikipedia
talk page corpus). If the server has the multi-year slices, you can
override with ``--corpus wikiconv-en`` or ``--corpus wikiconv-de`` etc.

The script downloads ConvoKit corpora to ``data/external/convokit/`` on
the main disk (not ``~/.convokit``). Set ``--data-dir`` to relocate.

Prerequisites:
    pip install convokit          # also pulls nltk, spacy, etc.
    python -m spacy download en_core_web_sm    # optional but recommended

Usage::

    python scripts/download_wikipedia.py
    python scripts/download_wikipedia.py --corpus wiki-corpus \\
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
DATA_DIR = PROJECT_ROOT / "data" / "external" / "convokit"
DEFAULT_CORPUS = "wiki-corpus"


def _to_iso(ts) -> str:
    """Coerce ConvoKit timestamp to ISO-8601.

    ConvoKit's wiki-corpus stores utterance timestamps as scientific-notation
    strings (e.g. ``'1.189190940E09'`` = epoch seconds). Reddit and other
    corpora use int epochs or ISO strings. We handle all three before
    falling back to ``datetime.now()``.
    """
    if ts is None:
        return datetime.now().isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).isoformat()
    # String path — covers ISO-8601, plain int, and scientific notation.
    s = str(ts).strip()
    try:
        return datetime.fromtimestamp(float(s)).isoformat()
    except (ValueError, OverflowError, OSError):
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return datetime.now().isoformat()


def _extract_records(corpus, max_threads: int | None) -> list[dict]:
    """Walk a ConvoKit corpus and emit flat JSONL records.

    ``wiki-corpus`` ships empty ``conversations.json`` entries — no
    page_id, no page_title. The convo.id is a synthetic thread id, not a
    Wikipedia page id. As a result there is no canonical page title to
    surface; we fall back to the root utterance's text (truncated) so the
    downstream ``topic`` field still carries semantic content for
    clustering instead of a bare numeric id. The full WikiConv release
    (with page metadata) would be needed to recover real titles.
    """
    records: list[dict] = []
    seen_threads = 0

    for convo in corpus.iter_conversations():
        if max_threads is not None and seen_threads >= max_threads:
            break
        seen_threads += 1

        thread_id = convo.id
        meta = getattr(convo, "meta", {}) or {}
        page_title = (
            meta.get("page_title")
            or meta.get("title")
            or meta.get("pageTitle", "")
            or ""
        )

        utts = list(convo.iter_utterances())
        # If the corpus exposes no page title, derive a topic stub from
        # the root utterance (the message that opened the thread).
        if not page_title:
            root_utt = next((u for u in utts if u.reply_to is None), None)
            if root_utt is not None and root_utt.text:
                page_title = (root_utt.text or "").strip().replace("\n", " ")[:120]
            if not page_title:
                page_title = str(thread_id)

        for utt in utts:
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
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory to cache downloaded ConvoKit corpora (default data/external/convokit).")
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

    print(f"[download] fetching ConvoKit corpus {args.corpus!r} to {args.data_dir} …")
    dataset_path = convokit.download(args.corpus, data_dir=str(args.data_dir))
    corpus = convokit.Corpus(filename=dataset_path)
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
