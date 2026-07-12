"""Backfill ``thread_id`` + ``msg_id`` into typical.jsonl files.

Background
----------
``src/skill/cluster_profile.py::TypicalUtterance`` originally stored only
``member / action / text / parent_context / topic`` (with ``topic`` = page_title).
Train/test separation for the simulation pool (outline §5.1) needs the source
``conversation_id`` so sim threads can exclude any thread that informed skill
distillation. The schema now carries ``thread_id`` + ``msg_id``; this script
retrofits the 6 existing typical.jsonl files without re-running the full
clustering pipeline.

Method
------
Each typical.jsonl row's ``text`` is the first 1000 chars of a real
``Message.text`` (set in ``_typical_utterances``), and ``member`` is the
``Message.user_id`` (= ConvoKit ``speaker``).

Three-pass to avoid the ~110GB / ~180M-line full corpus scan:

Pass 1 (fast): scan ``outputs/stream_cache/detail_v4_wikiconv-<year>.pkl``
to learn which years each representative member was active in.

Pass 2 (fast): use ``ripgrep`` (``rg -f``) to pre-filter utterances.jsonl
lines containing any representative member name. Only years where members
are active are scanned. ripgrep handles the 6-11GB/year files in ~30s each.

Pass 3: parse only the pre-filtered lines, build per-member text index,
then stamp ``thread_id`` + ``msg_id`` into typical.jsonl rows.

Usage
-----
    python -m scripts.backfill_typical_thread_ids
    python -m scripts.backfill_typ_ids --corpus-root data/raw/wikiconv_en \\\\
        --corpus-dir outputs/skill_corpus_k8_quantile/wikiconv \\\\
        --detail-cache-dir outputs/stream_cache \\\\
        --prefix-len 200
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "data" / "raw" / "wikiconv_en"
DEFAULT_CORPUS_DIR = (
    PROJECT_ROOT / "outputs" / "skill_corpus_k8_quantile" / "wikiconv"
)
DEFAULT_DETAIL_CACHE_DIR = PROJECT_ROOT / "outputs" / "stream_cache"
DEFAULT_PREFIX_LEN = 200


def _normalize(text: str) -> str:
    """Collapse whitespace + lowercase — robust to indentation differences."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _load_typical_members(corpus_dir: Path) -> dict[str, set[str]]:
    """Return {cluster_id_str: set(representative members)} for all clusters."""
    members: dict[str, set[str]] = defaultdict(set)
    for cluster_dir in sorted(corpus_dir.glob("cluster_*")):
        cid = cluster_dir.name.replace("cluster_", "")
        path = cluster_dir / "typical.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                m = row.get("member")
                if m:
                    members[cid].add(m)
    return members


def _years_per_member(
    detail_cache_dir: Path,
    members_of_interest: set[str],
) -> dict[str, set[str]]:
    """Pass 1: for each rep member, which years they appear in (per detail cache)."""
    member_years: dict[str, set[str]] = {m: set() for m in members_of_interest}
    caches = sorted(detail_cache_dir.glob("detail_v4_wikiconv-*.pkl"))
    if not caches:
        raise FileNotFoundError(
            f"No detail_v4_wikiconv-*.pkl under {detail_cache_dir}"
        )
    for cache in caches:
        year = cache.stem.rsplit("-", 1)[-1]
        try:
            d = pickle.load(open(cache, "rb"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  failed to load {cache.name}: {e}")
            continue
        n_in_year = 0
        for m in members_of_interest:
            if m in d:
                member_years[m].add(year)
                n_in_year += 1
        logger.info(f"  {cache.name}: {n_in_year} rep members present")
    return member_years


def _build_member_text_index(
    corpus_root: Path,
    members_of_interest: set[str],
    member_years: dict[str, set[str]],
    prefix_len: int,
) -> dict[str, dict[str, tuple[str, str]]]:
    """Pass 2+3: ripgrep pre-filter, then parse matching lines to build
    ``{member: {normalized_text_prefix: (utt_id, conv_id)}}``."""
    index: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    year_dirs = sorted(
        d for d in corpus_root.glob("wikiconv-*")
        if (d / "utterances.jsonl").exists()
    )
    if not year_dirs:
        raise FileNotFoundError(
            f"No wikiconv-* dirs with utterances.jsonl under {corpus_root}"
        )

    # Invert: which members to look for in each year
    members_by_year: dict[str, set[str]] = defaultdict(set)
    for m, years in member_years.items():
        for y in years:
            members_by_year[y].add(m)

    # Write member names to temp file for rg -f
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tf:
        for m in sorted(members_of_interest):
            tf.write(m + "\n")
        member_file = tf.name

    logger.info(
        f"Scanning utterances.jsonl via ripgrep for {len(members_of_interest)} "
        f"representative members"
    )
    try:
        for d in year_dirs:
            year = d.name.rsplit("-", 1)[-1]
            target_members = members_by_year.get(year)
            if not target_members:
                logger.info(f"  {d.name}: no rep members active — skip")
                continue
            path = d / "utterances.jsonl"
            # ripgrep pre-filter: lines containing any member name
            try:
                proc = subprocess.run(
                    ["rg", "-f", member_file, "--no-filename", str(path)],
                    capture_output=True, text=True, timeout=300,
                )
            except FileNotFoundError:
                logger.warning("ripgrep (rg) not found, falling back to full scan")
                proc = None
            except subprocess.TimeoutExpired:
                logger.warning(f"  {d.name}: rg timed out, falling back to full scan")
                proc = None

            n_match = 0
            if proc and proc.returncode == 0:
                # Parse only pre-filtered lines
                for line in proc.stdout.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    speaker = rec.get("speaker")
                    if isinstance(speaker, dict):
                        speaker = speaker.get("id")
                    if not speaker or speaker not in target_members:
                        continue
                    text = rec.get("text") or ""
                    key = _normalize(text[:prefix_len])
                    if not key:
                        continue
                    utt_id = str(rec.get("id"))
                    conv_id = str(
                        rec.get("conversation_id") or rec.get("root") or utt_id
                    )
                    index[speaker].setdefault(key, (utt_id, conv_id))
                    n_match += 1
                logger.info(
                    f"  {d.name}: rg pre-filter + parse, indexed {n_match} utterances"
                )
            else:
                # Fallback: full line-by-line scan (slow)
                n_lines = 0
                with open(path) as f:
                    for line in f:
                        n_lines += 1
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        speaker = rec.get("speaker")
                        if isinstance(speaker, dict):
                            speaker = speaker.get("id")
                        if not speaker or speaker not in target_members:
                            continue
                        text = rec.get("text") or ""
                        key = _normalize(text[:prefix_len])
                        if not key:
                            continue
                        utt_id = str(rec.get("id"))
                        conv_id = str(
                            rec.get("conversation_id")
                            or rec.get("root")
                            or utt_id
                        )
                        index[speaker].setdefault(key, (utt_id, conv_id))
                        n_match += 1
                logger.info(
                    f"  {d.name}: full scan {n_lines} lines, indexed {n_match}"
                )
    finally:
        Path(member_file).unlink(missing_ok=True)

    return index


def backfill(
    corpus_dir: Path,
    corpus_root: Path,
    detail_cache_dir: Path,
    prefix_len: int,
    dry_run: bool,
) -> None:
    members_per_cluster = _load_typical_members(corpus_dir)
    all_members = set()
    for m in members_per_cluster.values():
        all_members |= m
    logger.info(
        f"Loaded {len(all_members)} representative members across "
        f"{len(members_per_cluster)} clusters"
    )

    member_years = _years_per_member(detail_cache_dir, all_members)
    unmatched = [m for m, ys in member_years.items() if not ys]
    if unmatched:
        logger.warning(
            f"{len(unmatched)} rep members not in any detail cache: {unmatched[:5]}"
        )

    index = _build_member_text_index(
        corpus_root, all_members, member_years, prefix_len
    )

    total_filled = 0
    total_rows = 0
    for cid, members in members_per_cluster.items():
        path = corpus_dir / f"cluster_{cid}" / "typical.jsonl"
        rows_out: list[str] = []
        n_filled = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                member = row.get("member")
                key = _normalize((row.get("text") or "")[:prefix_len])
                if member and key and member in index and key in index[member]:
                    utt_id, conv_id = index[member][key]
                    row["msg_id"] = utt_id
                    row["thread_id"] = conv_id
                    n_filled += 1
                    total_filled += 1
                else:
                    row.setdefault("msg_id", "")
                    row.setdefault("thread_id", "")
                rows_out.append(json.dumps(row, ensure_ascii=False))
        if dry_run:
            logger.info(
                f"[dry-run] cluster_{cid}: would fill {n_filled}/{len(rows_out)} rows"
            )
            continue
        with open(path, "w") as f:
            for r in rows_out:
                f.write(r + "\n")
        logger.info(
            f"cluster_{cid}: filled {n_filled}/{len(rows_out)} rows "
            f"({100*n_filled/max(len(rows_out),1):.1f}%)"
        )

    logger.info(
        f"Done. {total_filled}/{total_rows} rows now carry thread_id "
        f"({100*total_filled/max(total_rows,1):.1f}% overall)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Root holding wikiconv-<year>/utterances.jsonl (ConvoKit format)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Output dir holding cluster_*/typical.jsonl",
    )
    parser.add_argument(
        "--detail-cache-dir",
        type=Path,
        default=DEFAULT_DETAIL_CACHE_DIR,
        help="Dir holding detail_v4_wikiconv-<year>.pkl (Pass 1 presence check)",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=DEFAULT_PREFIX_LEN,
        help="Normalized text prefix length used for matching (default 200)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backfill(
        corpus_dir=args.corpus_dir,
        corpus_root=args.corpus_root,
        detail_cache_dir=args.detail_cache_dir,
        prefix_len=args.prefix_len,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
