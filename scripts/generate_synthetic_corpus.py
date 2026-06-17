"""Synthetic dev corpus generator.

Emits tiny JSONL corpora for the three platforms (wikipedia / reddit / github)
into ``data/raw/{platform}/`` so the experiment loop can be validated end-to-end
without real data.

THIS IS A DEVELOPMENT FIXTURE, NOT RESEARCH DATA.
===============================================

The generated threads have:

- Realistic-enough schemas (matching what the platform loaders expect —
  see ``src/data/{wikipedia,reddit,github}.py``).
- Multiple participants per thread (>=3) so the contested-thread filter
  in ``FeatureExtractor`` has something to bite on.
- A mix of action types so the clustering / metrics pipelines see a
  non-degenerate action distribution.
- Distinguishable user behavioural fingerprints per cluster seed so
  ``BehavioralClusterer`` produces >1 cluster rather than collapsing
  everything into one blob.

Use it to smoke-test the pipeline. Replace with real dumps before any
paper-scale run — outline §5.4 (clustering stability), §5.3 (held-out
event annotation κ≥0.7), and §6 (sim-to-real) all presuppose real data.

Usage::

    python scripts/generate_synthetic_corpus.py \\
        --n-threads 50 --n-users 40 --seed 42

CLI flags default to small values tuned for the dev config (configs/dev.yaml).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

# Pool of message bodies that look plausibly like the platform's content.
# Bodies are intentionally short and stylised so behavioural clusters
# separate (each cluster gets a different style fingerprint).
_WIKI_BODIES = [
    "Restoring prior version; the edit introduced unsourced claims.",
    "Adding citation to support the change.",
    "This appears to be original research — please source.",
    "Agree with the proposed merge, notifying related projects.",
    "Reverting vandalism; user has been reported.",
    "Discussing on the talk page before re-applying the edit.",
    "Updated the lead paragraph to reflect the new sources.",
    "Disagree with the rewrite, the prior version was clearer.",
]

_REDDIT_BODIES = [
    "I disagree — the core premise conflates correlation with causation.",
    "Actually, I'd argue the opposite is true in this case.",
    "Have you considered the counter-example from the 2008 paper?",
    "That's an interesting point, but it doesn't address the original claim.",
    "Δ awarded — you've changed my view on the secondary point.",
    "I don't think the proposed distinction holds up under scrutiny.",
    "However, the methodology has known limitations here.",
    "On the contrary, the evidence suggests a different interpretation.",
]

_GITHUB_BODIES = [
    "Reproduced on main, attaching stack trace below.",
    "Closed as duplicate of #1234 — please continue discussion there.",
    "Reopening; the fix in PR #5678 didn't resolve this on the latest build.",
    "Adding the bug label; this blocks our upcoming release.",
    "I think the root cause is in the async shutdown path.",
    "Confirming this is fixed on the patch release, closing.",
    "Assigning to the maintainer for triage.",
    "Workaround available, documented in the linked discussion.",
]


# ---------------------------------------------------------------------------
# Per-platform generators
# ---------------------------------------------------------------------------

def gen_wikipedia(n_threads: int, n_users: int, rng: random.Random) -> list[dict]:
    """Emit Wikipedia-talk-page-style JSONL records."""
    records: list[dict] = []
    base_ts = datetime(2024, 1, 1, 0, 0, 0)

    for t_idx in range(n_threads):
        thread_id = f"wiki_thread_{t_idx}"
        page_title = f"Wikipedia:Talk/Page_{t_idx}"
        # 4–8 messages per thread, 3–6 distinct users
        n_msgs = rng.randint(4, 8)
        participants = [f"wiki_user_{(t_idx + i) % n_users}" for i in range(rng.randint(3, 6))]
        prev_msg_id = None
        for m_idx in range(n_msgs):
            uid = participants[m_idx % len(participants)]
            action_pool = ["edit", "revert", "discuss", "report"]
            action = action_pool[m_idx % len(action_pool)]
            msg_id = f"{thread_id}_c{m_idx}"
            records.append({
                "thread_id": thread_id,
                "page_title": page_title,
                "comment_id": msg_id,
                "user_id": uid,
                "author": uid,
                "text": _WIKI_BODIES[(t_idx + m_idx) % len(_WIKI_BODIES)],
                "timestamp": (base_ts + timedelta(minutes=t_idx * 30 + m_idx * 5)).isoformat(),
                "action_type": action,
                "parent_comment_id": prev_msg_id,
                "rev_id": 1000 + t_idx * 100 + m_idx,
            })
            prev_msg_id = msg_id
    return records


def gen_reddit(n_threads: int, n_users: int, rng: random.Random) -> list[dict]:
    """Emit Reddit r/changemyview-style JSONL records (submissions + comments)."""
    records: list[dict] = []
    base_ts = datetime(2024, 1, 1, 0, 0, 0)

    for t_idx in range(n_threads):
        sub_id = f"sub_{t_idx}"
        op = f"redditor_{(t_idx * 2) % n_users}"
        title = f"CMV: position #{t_idx} on topic_{t_idx % 5}"
        body = "I hold this view and would like it challenged. Here is my reasoning."

        records.append({
            "submission_id": sub_id,
            "id": sub_id,
            "author": op,
            "title": title,
            "selftext": body,
            "created_utc": int((base_ts + timedelta(hours=t_idx)).timestamp()),
        })

        # 4–9 comments per submission, 3–6 distinct users
        n_comments = rng.randint(4, 9)
        participants = [f"redditor_{(t_idx * 2 + i + 1) % n_users}" for i in range(rng.randint(3, 6))]
        # Top-level comments reply to the submission (parent=t3_<sub_id>)
        parent_by_depth = {0: f"t3_{sub_id}"}
        msg_id_by_depth = {0: f"{sub_id}_0"}

        for c_idx in range(n_comments):
            uid = participants[c_idx % len(participants)]
            comment_id = f"{sub_id}_c{c_idx + 1}"
            depth = min(c_idx % 3, 2)  # depths 0/1/2
            parent_id = parent_by_depth.get(depth - 1, f"t3_{sub_id}") if depth > 0 else f"t3_{sub_id}"
            parent_id = parent_id.replace("t3_", "") if depth == 0 else parent_id

            body = _REDDIT_BODIES[(t_idx + c_idx) % len(_REDDIT_BODIES)]
            # Sprinkle delta / blocking markers so the action distribution
            # is non-degenerate.
            is_delta = (c_idx == n_comments - 1) and (t_idx % 4 == 0)
            is_blocking = (c_idx == n_comments - 1) and (t_idx % 7 == 0)
            if is_delta:
                body = "Δ " + body

            records.append({
                "submission_id": sub_id,
                "comment_id": comment_id,
                "author": uid,
                "body": body,
                "parent_comment_id": parent_id,
                "created_utc": int(
                    (base_ts + timedelta(hours=t_idx, minutes=c_idx + 5)).timestamp()
                ),
                "delta_awarded": is_delta,
                "is_blocking": is_blocking,
                "score": rng.randint(-5, 50),
                "submission_title": title,
            })
            parent_by_depth[depth] = comment_id
    return records


def gen_github(n_threads: int, n_users: int, rng: random.Random) -> list[dict]:
    """Emit GitHub-issue-event-style JSONL records."""
    records: list[dict] = []
    base_ts = datetime(2024, 1, 1, 0, 0, 0)

    for t_idx in range(n_threads):
        repo = f"org/repo_{t_idx % 4}"
        issue_number = 100 + t_idx
        thread_id = f"{repo}#{issue_number}"
        title = f"Issue {issue_number}: bug in module_{t_idx % 6}"
        participants = [f"gh_user_{(t_idx + i) % n_users}" for i in range(rng.randint(3, 6))]

        # Issue opened by first participant
        opener = participants[0]
        records.append({
            "repo": repo,
            "issue_number": issue_number,
            "event_type": "issue_opened",
            "author": opener,
            "body": _GITHUB_BODIES[t_idx % len(_GITHUB_BODIES)],
            "created_at": (base_ts + timedelta(hours=t_idx)).isoformat(),
            "title": title,
            "labels": ["bug"],
            "event_id": f"{thread_id}_opened",
        })

        # 4–8 follow-up events (comments / labels / close / reopen).
        # GitHub loader's action_map (see src/data/github.py) expects the
        # tokens below — using bare "close"/"reopen" would silently fall
        # through to COMMENT and hide lifecycle events from the metrics.
        n_events = rng.randint(4, 8)
        event_pool = ["comment", "comment", "comment", "labeled",
                      "issue_closed", "issue_reopened"]
        for e_idx in range(n_events):
            uid = participants[(e_idx + 1) % len(participants)]
            event_type = event_pool[e_idx % len(event_pool)]
            # Force a close→reopen sequence once per few threads so lifecycle
            # actions appear with non-zero frequency.
            if e_idx == n_events - 1 and t_idx % 3 == 0:
                event_type = "issue_closed"
            records.append({
                "repo": repo,
                "issue_number": issue_number,
                "event_type": event_type,
                "author": uid,
                "body": _GITHUB_BODIES[(t_idx + e_idx) % len(_GITHUB_BODIES)],
                "created_at": (
                    base_ts + timedelta(hours=t_idx, minutes=e_idx + 5)
                ).isoformat(),
                "title": title,
                "labels": ["bug"] if event_type == "labeled" else [],
                "event_id": f"{thread_id}_{event_type}_{e_idx}",
            })
    return records


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-threads", type=int, default=50,
                        help="Threads per platform (default 50).")
    parser.add_argument("--n-users", type=int, default=40,
                        help="Distinct users per platform (default 40).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--platforms", nargs="+",
                        default=["wikipedia", "reddit", "github"],
                        choices=["wikipedia", "reddit", "github"])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing fixtures.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for platform in args.platforms:
        out_dir = RAW_DIR / platform
        if any(out_dir.glob("*.jsonl")) and not args.force:
            print(f"[skip] {out_dir} already has JSONL — use --force to overwrite")
            continue

        if platform == "wikipedia":
            records = gen_wikipedia(args.n_threads, args.n_users, rng)
            write_jsonl(out_dir / "wiki_synthetic.jsonl", records)
        elif platform == "reddit":
            records = gen_reddit(args.n_threads, args.n_users, rng)
            # Reddit loader expects separate submissions / comments files.
            subs = [r for r in records if "submission_id" in r and "comment_id" not in r]
            comms = [r for r in records if "comment_id" in r]
            write_jsonl(out_dir / "reddit_submissions.jsonl", subs)
            write_jsonl(out_dir / "reddit_comments.jsonl", comms)
        elif platform == "github":
            records = gen_github(args.n_threads, args.n_users, rng)
            write_jsonl(out_dir / "github_synthetic.jsonl", records)

        print(f"[wrote] {out_dir} ({len(records)} records)")


if __name__ == "__main__":
    main()
