"""Fetch GitHub issue lifecycles via the REST API and emit loader JSONL.

Emits ``data/raw/github/github_<repo_owner>-<repo_name>.jsonl`` per repo,
in the schema ``GitHubLoader`` (see ``src/data/github.py``) ingests:

    repo, issue_number, event_type, author, body, created_at,
    title, labels, event_id

Event types map 1-to-1 onto the loader's ``action_map`` so issue
lifecycle events (open/close/reopen/label) surface in the action
distribution rather than silently falling through to COMMENT:

    issue_opened, issue_reopened, issue_closed, labeled, comment

Auth / rate limits
------------------
GitHub's unauthenticated REST limit is 60 req/hour — far below what any
real paper-scale scrape needs. Set ``CADP_GITHUB_TOKEN`` (or the
canonical ``GITHUB_TOKEN`` / ``GH_TOKEN``) in the environment to a
personal access token with ``public_repo`` scope; this raises the limit
to 5000 req/hour. Tokens are read but never logged.

GHTorrent vs this script
------------------------
GHTorrent's MongoDB dumps are the canonical academic source for GitHub
mining, but they are ~1 TB and require joining ``issues`` +
``issue_events`` tables to reconstruct lifecycle. For CADP distillation
on a bounded repo set, direct API scraping is cheaper, always
up-to-date, and produces schema-aligned JSONL out of the box. Outline
§6.2 lists the repo set per dataset.

Usage::

    # default repo set
    export CADP_GITHUB_TOKEN=ghp_xxx
    python scripts/download_github.py

    # custom repo list + cap
    python scripts/download_github.py \\
        --repos pandas-dev/pandas numpy/numpy scikit-learn/scikit-learn \\
        --max-issues-per-repo 200 --force
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "github"
API_BASE = "https://api.github.com"

# Curated default set — well-known OSS projects with active issue
# lifecycles (close/reopen cycles, label churn, multi-participant
# discussions). Outline §6.2 details selection criteria.
DEFAULT_REPOS = [
    "pandas-dev/pandas",
    "numpy/numpy",
    "scikit-learn/scikit-learn",
    "python/cpython",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict[str, str]:
    token = (
        os.getenv("CADP_GITHUB_TOKEN")
        or os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or ""
    )
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(session: requests.Session, url: str, params: dict | None = None) -> tuple[list | dict, dict]:
    """GET with rate-limit-aware retry. Returns (json, response_headers)."""
    while True:
        resp = session.get(url, params=params, timeout=30)
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if resp.status_code == 403 and remaining == "0":
            wait = max(int(reset) - int(time.time()), 1) if reset else 60
            print(f"[ratelimit] sleeping {wait}s until reset …")
            time.sleep(min(wait, 3600))
            continue
        resp.raise_for_status()
        return resp.json(), resp.headers


def _paginate(session: requests.Session, url: str, params: dict | None, max_items: int | None) -> list:
    out: list = []
    page = 1
    base_params = dict(params or {})
    while True:
        page_params = {**base_params, "page": page, "per_page": 100}
        items, headers = _get_json(session, url, page_params)
        if not items:
            break
        out.extend(items)
        if max_items is not None and len(out) >= max_items:
            return out[:max_items]
        link = headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        page += 1
    return out


# ---------------------------------------------------------------------------
# Per-issue event stream construction
# ---------------------------------------------------------------------------

def _events_for_issue(session: requests.Session, repo: str, issue: dict, max_events: int | None) -> list[dict]:
    """Build loader-schema records for one issue."""
    owner_repo = repo
    number = issue["number"]
    title = issue.get("title", f"{repo}#{number}")
    labels_now = [lab.get("name", "") for lab in issue.get("labels", []) if lab.get("name")]
    out: list[dict] = []

    # Seed: issue_opened event from the issue itself.
    out.append({
        "repo": owner_repo,
        "issue_number": number,
        "event_type": "issue_opened",
        "author": (issue.get("user") or {}).get("login", ""),
        "body": issue.get("body") or "",
        "created_at": issue.get("created_at", ""),
        "title": title,
        "labels": labels_now,
        "event_id": f"{repo}#{number}_opened",
    })

    # Comments
    comments_url = issue.get("comments_url") or f"{API_BASE}/repos/{owner_repo}/issues/{number}/comments"
    for c in _paginate(session, comments_url, None, max_events):
        out.append({
            "repo": owner_repo,
            "issue_number": number,
            "event_type": "comment",
            "author": (c.get("user") or {}).get("login", ""),
            "body": c.get("body") or "",
            "created_at": c.get("created_at", ""),
            "title": title,
            "labels": [],
            "event_id": str(c.get("id")) or f"{repo}#{number}_comment_{c.get('id')}",
        })

    # Lifecycle events (labeled / closed / reopened / assigned / etc.)
    events_url = f"{API_BASE}/repos/{owner_repo}/issues/{number}/events"
    for ev in _paginate(session, events_url, None, max_events):
        etype = ev.get("event", "")
        # Loader's action_map keys onto these exact tokens. Anything else
        # (assigned, referenced, mentioned, …) carries little behavioural
        # signal for CADP distillation, so we skip them rather than let
        # them silently become COMMENTs.
        if etype == "labeled":
            label_name = (ev.get("label") or {}).get("name", "")
            out.append({
                "repo": owner_repo,
                "issue_number": number,
                "event_type": "labeled",
                "author": (ev.get("actor") or {}).get("login", ""),
                "body": "",
                "created_at": ev.get("created_at", ""),
                "title": title,
                "labels": [label_name] if label_name else [],
                "event_id": str(ev.get("id")),
            })
        elif etype == "closed":
            out.append({
                "repo": owner_repo,
                "issue_number": number,
                "event_type": "issue_closed",
                "author": (ev.get("actor") or {}).get("login", ""),
                "body": "",
                "created_at": ev.get("created_at", ""),
                "title": title,
                "labels": [],
                "event_id": str(ev.get("id")),
            })
        elif etype == "reopened":
            out.append({
                "repo": owner_repo,
                "issue_number": number,
                "event_type": "issue_reopened",
                "author": (ev.get("actor") or {}).get("login", ""),
                "body": "",
                "created_at": ev.get("created_at", ""),
                "title": title,
                "labels": [],
                "event_id": str(ev.get("id")),
            })

    return out


def _safe_repo_slug(repo: str) -> str:
    return repo.replace("/", "-").replace("\\", "-")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fetch_repo(session: requests.Session, repo: str, max_issues: int | None, max_events: int | None) -> list[dict]:
    """Fetch all issues (excluding PRs) for one repo and emit loader records."""
    print(f"[fetch] {repo}: listing issues …")
    url = f"{API_BASE}/repos/{repo}/issues"
    # state=all so close/reopen cycles surface. Sort ascending by created_at
    # so threads come out in chronological order.
    issues = _paginate(session, url, {"state": "all", "sort": "created", "direction": "asc"}, max_issues)
    # Filter out pull requests — the issues endpoint also returns them.
    issues = [i for i in issues if "pull_request" not in i]
    print(f"[fetch] {repo}: {len(issues)} issues (PRs filtered out)")

    records: list[dict] = []
    for i, issue in enumerate(issues, 1):
        try:
            records.extend(_events_for_issue(session, repo, issue, max_events))
        except requests.HTTPError as e:
            print(f"[warn] {repo}#{issue.get('number')}: {e} — skipping")
        if i % 25 == 0:
            print(f"[progress] {repo}: {i}/{len(issues)} issues processed")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS,
                        help="GitHub repos as owner/name. Defaults to a curated OSS set.")
    parser.add_argument("--max-issues-per-repo", type=int, default=None,
                        help="Cap issues per repo (default all).")
    parser.add_argument("--max-events-per-issue", type=int, default=None,
                        help="Cap events/comments per issue (default all).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory (default data/raw/github).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite per-repo output files.")
    args = parser.parse_args()

    headers = _auth_headers()
    if "Authorization" not in headers:
        print(
            "[warn] no GitHub token in env (CADP_GITHUB_TOKEN / GITHUB_TOKEN / GH_TOKEN) — "
            "unauthenticated rate limit is 60 req/hour. Scraping will be slow."
        )

    session = requests.Session()
    session.headers.update(headers)

    for repo in args.repos:
        slug = _safe_repo_slug(repo)
        out_path = args.out_dir / f"github_{slug}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists — use --force to overwrite")
            continue

        try:
            records = fetch_repo(session, repo, args.max_issues_per_repo, args.max_events_per_issue)
        except requests.HTTPError as e:
            print(f"[error] {repo}: {e} — skipping")
            continue

        if not records:
            print(f"[warn] {repo}: no records extracted")
            continue

        write_jsonl(out_path, records)
        n_issues = len({r["issue_number"] for r in records})
        print(f"[wrote] {out_path} — {len(records)} events across {n_issues} issues")


if __name__ == "__main__":
    main()
