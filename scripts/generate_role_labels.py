"""Generate external role labels for Reddit and GitHub datasets.

Produces data/role_labels/{reddit,github}.jsonl from raw data,
using behavior-based heuristics (outline §5.3 anti-circularity mandate).

Reddit r/changemyview roles:
  - original_poster: created submissions (OP)
  - persuader: received deltas (successfully changed views)
  - active_deliberator: frequent counter-arguments without receiving deltas
  - regular: moderate participation
  - lurker: very few interactions

GitHub Issues roles:
  - maintainer: performs label/close/reopen/assign actions
  - contributor: frequent commenter across issues
  - reporter: primarily opens issues
  - casual: few actions

Usage:
  python scripts/generate_role_labels.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "role_labels"


def generate_reddit_labels() -> None:
    """Generate role labels for Reddit r/changemyview."""
    comments_path = RAW_DIR / "reddit" / "reddit_comments.jsonl"
    submissions_path = RAW_DIR / "reddit" / "reddit_submissions.jsonl"

    # Collect OP authors from submissions
    op_authors: set[str] = set()
    if submissions_path.exists():
        with open(submissions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                author = rec.get("author", "")
                if author and author != "[deleted]":
                    op_authors.add(author)

    # Compute per-user stats from comments
    user_stats: dict[str, dict] = defaultdict(lambda: {
        "total_comments": 0,
        "deltas_received": 0,
        "counter_arguments": 0,
        "threads": set(),
        "replies_to_op": 0,
    })

    # Identify OPs per submission for reply-to-OP detection
    submission_ops: dict[str, str] = {}
    if submissions_path.exists():
        with open(submissions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sub_id = str(rec.get("submission_id") or rec.get("id", ""))
                author = rec.get("author", "")
                if sub_id and author:
                    submission_ops[sub_id.replace("t3_", "")] = author

    with open(comments_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            author = rec.get("author", "")
            if not author or author in ("[deleted]", "AutoModerator", "deltabot", "DeltaBot"):
                continue

            sub_id = str(rec.get("submission_id", "")).replace("t3_", "")
            stats = user_stats[author]
            stats["total_comments"] += 1
            stats["threads"].add(sub_id)

            if rec.get("delta_awarded") or rec.get("is_delta"):
                stats["deltas_received"] += 1

            # Track replies to OP
            parent_id = rec.get("parent_comment_id") or rec.get("parent_id", "")
            parent_id = str(parent_id).replace("t1_", "").replace("t3_", "")
            if sub_id in submission_ops:
                op = submission_ops[sub_id]
                # Direct reply to OP's post (top-level) or reply to OP's comment
                if parent_id == sub_id or parent_id == sub_id.replace("t3_", ""):
                    if author != op:
                        stats["replies_to_op"] += 1

    # Assign roles
    labels: list[dict[str, str]] = []
    all_authors = set(user_stats.keys()) | op_authors

    # Compute thresholds for role separation
    # In r/changemyview: delta_awarded on a comment = OP awarded delta to commenter
    # Most active OPs also receive deltas when they comment on other threads
    comment_counts = [user_stats[a]["total_comments"] for a in user_stats]
    median_comments = sorted(comment_counts)[len(comment_counts) // 2] if comment_counts else 1

    for author in sorted(all_authors):
        stats = user_stats.get(author, {
            "total_comments": 0,
            "deltas_received": 0,
            "counter_arguments": 0,
            "threads": set(),
            "replies_to_op": 0,
        })
        n_comments = stats["total_comments"]
        n_deltas = stats["deltas_received"]
        n_threads = len(stats["threads"]) if isinstance(stats["threads"], set) else 0

        # Role assignment logic (priority order):
        # 1. original_poster: primarily OP identity, very few comments elsewhere
        # 2. persuader: received >= 1 delta (successfully changed someone's view)
        # 3. active_deliberator: frequent commenter across threads, no deltas
        # 4. regular: moderate participation
        # 5. lurker: very few interactions
        if author in op_authors and n_comments <= 2:
            role = "original_poster"
        elif n_deltas >= 1:
            role = "persuader"
        elif n_comments >= 10 and n_threads >= 3:
            role = "active_deliberator"
        elif n_comments >= 3:
            role = "regular"
        else:
            role = "lurker"

        labels.append({"user_id": author, "role": role})

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "reddit.jsonl"
    with open(out_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    # Summary
    role_counts = Counter(l["role"] for l in labels)
    print(f"Reddit: {len(labels)} users labeled → {out_path}")
    for role, count in role_counts.most_common():
        print(f"  {role}: {count}")


def generate_github_labels() -> None:
    """Generate role labels for GitHub Issues."""
    data_dir = RAW_DIR / "github"

    # Compute per-user stats
    user_stats: dict[str, dict] = defaultdict(lambda: {
        "total_events": 0,
        "issues_opened": 0,
        "comments": 0,
        "admin_actions": 0,  # label + close + reopen + assign
        "issues_participated": set(),
    })

    admin_event_types = {"labeled", "issue_closed", "issue_reopened", "assigned"}

    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                author = rec.get("author") or rec.get("user", "")
                if not author:
                    continue

                event_type = rec.get("event_type", "comment").lower()
                repo = rec.get("repo", "")
                issue_number = rec.get("issue_number", "")
                issue_key = f"{repo}#{issue_number}"

                stats = user_stats[author]
                stats["total_events"] += 1
                stats["issues_participated"].add(issue_key)

                if event_type == "issue_opened":
                    stats["issues_opened"] += 1
                elif event_type == "comment":
                    stats["comments"] += 1
                elif event_type in admin_event_types:
                    stats["admin_actions"] += 1

    # Assign roles using adaptive thresholds.
    # Synthetic data tends to distribute actions uniformly, so we use
    # ratios rather than absolute counts for role separation.
    #
    # For small corpora (< 100 users) the action ratios cluster tightly
    # and a single dominant-action approach (admin > open > comment) may
    # collapse all users into one role.  We use quantile-based splits
    # on the dominant ratio to guarantee 3–4 distinct roles (needed for
    # Micro Behavior evaluation ground-truth diversity, outline §5.3).
    labels: list[dict[str, str]] = []

    # --- Step 1: compute per-user dominant-action ratio ---
    user_ratios: dict[str, dict] = {}
    for author in sorted(user_stats.keys()):
        s = user_stats[author]
        n = max(s["total_events"], 1)
        user_ratios[author] = {
            "admin_r": s["admin_actions"] / n,
            "open_r": s["issues_opened"] / n,
            "comment_r": s["comments"] / n,
            "n_issues": len(s["issues_participated"]),
            "n_comments": s["comments"],
            "n_events": s["total_events"],
        }

    # --- Step 2: determine adaptive thresholds ---
    n_users = len(user_ratios)
    if n_users <= 100:
        # Use quartile-based thresholds for small corpora.
        admin_vals = sorted(v["admin_r"] for v in user_ratios.values())
        open_vals = sorted(v["open_r"] for v in user_ratios.values())
        comment_vals = sorted(v["comment_r"] for v in user_ratios.values())
        q3 = lambda vals: vals[len(vals) * 3 // 4] if vals else 0
        median = lambda vals: vals[len(vals) // 2] if vals else 0
        # Maintainer: top-quartile admin ratio
        admin_thresh = max(q3(admin_vals), 0.25)
        # Reporter: above-median open ratio (and NOT top-quartile admin)
        open_thresh = max(median(open_vals), 0.10)
        # Contributor: above-median comment ratio OR high activity
        comment_thresh = max(median(comment_vals), 0.40)
    else:
        admin_thresh = 0.4
        open_thresh = 0.3
        comment_thresh = 0.5

    # --- Step 3: assign roles by priority ---
    for author in sorted(user_ratios.keys()):
        r = user_ratios[author]
        # 1. maintainer: top-quartile admin ratio
        if r["admin_r"] >= admin_thresh:
            role = "maintainer"
        # 2. reporter: above-median open ratio (not also a maintainer)
        elif r["open_r"] >= open_thresh:
            role = "reporter"
        # 3. contributor: above-median comment ratio + multi-issue
        elif r["comment_r"] >= comment_thresh and r["n_issues"] >= 2:
            role = "contributor"
        # 3b. contributor: high activity across issues
        elif r["n_comments"] >= 3 and r["n_issues"] >= 2:
            role = "contributor"
        # 4. casual: everything else
        else:
            role = "casual"
        labels.append({"user_id": author, "role": role})

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "github.jsonl"
    with open(out_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    # Summary
    role_counts = Counter(l["role"] for l in labels)
    print(f"GitHub: {len(labels)} users labeled → {out_path}")
    for role, count in role_counts.most_common():
        print(f"  {role}: {count}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating external role labels (outline §5.3)")
    print("=" * 60)
    generate_reddit_labels()
    print()
    generate_github_labels()
    print()
    print("Done. Files ready for MetricsAggregator anti-circularity check.")
