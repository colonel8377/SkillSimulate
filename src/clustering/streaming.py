"""Streaming, per-year-parallel feature extraction over the full WikiConv corpus.

The full 2001-2018 WikiConv is ~80M+ utterances — materialising them all as
in-memory Message/Thread objects is infeasible (~100GB+). Instead we stream the
raw ConvoKit JSONL twice, never holding more than a bounded conversation buffer:

- Pass A: count messages per user (tiny memory) → filter ultra-low-activity users.
- Pass B: stream again; accumulate the 21-dim behavioural detail + a bounded
  reply graph + a small per-user text sample, ONLY for active users. The reply
  graph is resolved locally inside a conversation buffer (utterances are ~92%
  contiguous by conversation), so parent-author lookups need no global index.

Reply graph + contested-thread status are computed per conversation at buffer
flush, so they stay exact within a contiguous conversation segment. Cross-year
/ cross-segment reply edges are a negligible loss (short conversations).

Each year corpus is processed by an independent worker (multiprocessing); the
GPU embedding runs in the main process (models are not fork-safe). The
population is never sampled — every user above the activity floor is clustered.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from src.clustering.features import (
    _NS_CONTENT, _NS_INTERPERSONAL, _NS_PROJECT, _WORD_RE, TOXIC_THRESHOLD,
)
from src.data.schemas import ActionType

try:  # faster JSON when available
    import orjson
    def _json_loads(line: bytes | str):
        return orjson.loads(line)
except ImportError:  # pragma: no cover
    import json
    def _json_loads(line: str):
        return json.loads(line)

# Per-user sample text cap (for the language embedding). Estimating one user's
# style from K texts is statistically stable; this is NOT population sampling.
# Texts are kept in two toxicity-stratified pools so the style axis sees a user's
# conflict register, which a plain first-K sample almost always misses.
SAMPLE_TEXTS_PER_USER = 15
SAMPLE_TEXT_CHARS = 512
# Capped per-user timestamp sample for the burstiness (inter-message gap CV)
# feature. A few hundred gaps give a stable CV; this bounds memory on heavy users.
SAMPLE_TS_CAP = 256
# Caps on reply-graph neighbour sets (breadth beyond this is saturating).
MAX_NEIGHBOURS = 200
# Conversation buffer eviction threshold (utterances) per worker.
CONV_BUFFER_CAP = 200_000

# Bump whenever the UserAccum schema OR the accum->feature derivation changes,
# to invalidate old detail caches and the feature-keyed clustering cache.
_DETAIL_CACHE_VERSION = 6


@dataclass
class UserAccum:
    """Mutable per-user accumulator (picklable for multiprocessing)."""
    # Schema version; old caches without this field will default to 0.
    version: int = 4
    n: int = 0
    # action-type counters; n serves as the discuss/comment count.
    n_discuss: int = 0
    n_edit: int = 0
    n_delete: int = 0
    n_restore: int = 0
    n_modified_received: int = 0
    n_deleted_received: int = 0
    n_restored_received: int = 0
    reply_n: int = 0
    reply_to_toxic_n: int = 0
    indent_sum: float = 0.0
    text_len_sum: int = 0
    tox_sum: float = 0.0
    tox_n: int = 0
    tox_max: float = 0.0
    sev_sum: float = 0.0
    sev_n: int = 0
    sev_max: float = 0.0
    exclaim_n: int = 0
    question_n: int = 0
    wp_n: int = 0
    contested_n: int = 0
    ns_interpersonal_n: int = 0
    ns_content_n: int = 0
    ns_project_n: int = 0
    ts_min: float = 0.0
    ts_max: float = 0.0
    ts_samples: list = field(default_factory=list)  # capped timestamps for burstiness CV
    page_ids: set = field(default_factory=set)   # distinct pages/conversations
    out_targets: set = field(default_factory=set)   # users this one replies to
    in_repliers: set = field(default_factory=set)   # users who reply to this one
    replies_received: int = 0
    tox_recv_sum: float = 0.0
    tox_recv_n: int = 0
    # Toxicity-stratified text pools (Q4): high-tox texts are reserved so the
    # style embedding sees a user's conflict register, not just their first-K.
    hi_tox_texts: list = field(default_factory=list)
    lo_tox_texts: list = field(default_factory=list)

    def __post_init__(self):
        # Ensure loaded old pickles migrate to current version.
        if self.version != _DETAIL_CACHE_VERSION:
            self.version = _DETAIL_CACHE_VERSION


def stratified_sample_texts(a: "UserAccum", k: int = SAMPLE_TEXTS_PER_USER) -> list:
    """Assemble k texts reserving up to k//2 high-toxicity, rest first-K low-tox.

    Backfills from the other pool when one is short, so the count stays at k
    whenever the user has >= k texts total. Deterministic (no RNG).
    """
    n_hi = k // 2
    hi = a.hi_tox_texts[:n_hi]
    lo = a.lo_tox_texts[:k - len(hi)]
    if len(hi) + len(lo) < k:                 # not enough low-tox: take more high-tox
        hi = a.hi_tox_texts[:k - len(lo)]
    return (hi + lo) or [" "]


def _speaker(sp):
    return sp.get("id") if isinstance(sp, dict) else sp


def _tox(mmeta, key):
    v = mmeta.get(key)
    return v if isinstance(v, (int, float)) else None


def _accumulate(accum: UserAccum, text: str, indent, tox, sev, ts, page_id,
                page_type, contested: bool):
    """Update an active user's accumulator for one of their own messages."""
    accum.n += 1
    accum.n_discuss += 1
    if text:
        accum.text_len_sum += len(text)
        if "!" in text:
            accum.exclaim_n += 1
        if text.rstrip().endswith("?"):
            accum.question_n += 1
        if "WP:" in text or "Wikipedia:" in text:
            accum.wp_n += 1
        # Toxicity-stratified text pools: reserve high-tox texts (conflict
        # register) separately from ordinary ones so the style embedding isn't
        # dominated by a heavy user's mundane first-K messages.
        clip = text[:SAMPLE_TEXT_CHARS]
        if tox is not None and tox >= TOXIC_THRESHOLD:
            if len(accum.hi_tox_texts) < SAMPLE_TEXTS_PER_USER:
                accum.hi_tox_texts.append(clip)
        elif len(accum.lo_tox_texts) < SAMPLE_TEXTS_PER_USER:
            accum.lo_tox_texts.append(clip)
    try:
        accum.indent_sum += max(0, int(indent))
    except (TypeError, ValueError):
        pass
    if tox is not None:
        accum.tox_sum += tox
        accum.tox_n += 1
        if tox > accum.tox_max:
            accum.tox_max = tox
    if sev is not None:
        accum.sev_sum += sev
        accum.sev_n += 1
        if sev > accum.sev_max:
            accum.sev_max = sev
    if ts:
        accum.ts_min = ts if not accum.ts_min or ts < accum.ts_min else accum.ts_min
        accum.ts_max = ts if ts > accum.ts_max else accum.ts_max
        if len(accum.ts_samples) < SAMPLE_TS_CAP:
            accum.ts_samples.append(ts)
    if page_id and len(accum.page_ids) < 2000:
        accum.page_ids.add(page_id)
    if page_type in _NS_INTERPERSONAL:
        accum.ns_interpersonal_n += 1
    elif page_type in _NS_CONTENT:
        accum.ns_content_n += 1
    elif page_type in _NS_PROJECT:
        accum.ns_project_n += 1
    if contested:
        accum.contested_n += 1


def _accumulate_action(accum: UserAccum, action_type: str, ts, page_id,
                       contested: bool):
    """Count a moderation action performed by the actor (no conversational text)."""
    if action_type == "edit":
        accum.n_edit += 1
    elif action_type == "delete":
        accum.n_delete += 1
    elif action_type == "restore":
        accum.n_restore += 1
    if ts:
        accum.ts_min = ts if not accum.ts_min or ts < accum.ts_min else accum.ts_min
        accum.ts_max = ts if ts > accum.ts_max else accum.ts_max
    if page_id and len(accum.page_ids) < 2000:
        accum.page_ids.add(page_id)
    if contested:
        accum.contested_n += 1


def _add_neighbour(s: set, item):
    if len(s) < MAX_NEIGHBOURS:
        s.add(item)


# ---------------------------------------------------------------------------
# Pass A: cheap per-user message counts (cached per year for resumability)
# ---------------------------------------------------------------------------

def _count_year(args) -> tuple:
    """Count messages per user for one year corpus. Writes a year cache."""
    cdir, cache_dir = args
    cdir = Path(cdir)
    cache_path = Path(cache_dir) / f"count_{cdir.name}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            counts, n = pickle.load(f)
        return counts, n, cache_path
    counts: Counter = Counter()
    n = 0
    with open(cdir / "utterances.jsonl", "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = _json_loads(line)
            sp = _speaker(r.get("speaker"))
            if sp:
                counts[sp] += 1
            n += 1
    with open(cache_path, "wb") as f:
        pickle.dump((counts, n), f)
    return counts, n, cache_path


def stream_counts(year_dirs: list[Path], workers: int,
                  cache_dir: str | Path = "outputs/stream_cache") -> tuple[Counter, int]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c, k, _ in ex.map(_count_year, [(d, cache_dir) for d in year_dirs]):
            total.update(c)
            n += k
    logger.info(f"Pass A: {n} utterances, {len(total)} users")
    return total, n


# ---------------------------------------------------------------------------
# Pass B: detailed accumulation for active users (conversation-buffered)
# ---------------------------------------------------------------------------

def _detail_year(args) -> dict:
    """Accumulate detail for active users in one year, conversation-buffered."""
    cdir, active, contested_threshold, toxic_threshold, cache_dir = args
    cdir = Path(cdir)
    cache_path = Path(cache_dir) / f"detail_v{_DETAIL_CACHE_VERSION}_{cdir.name}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    conv_meta = _load_conversations(cdir)

    accums: dict[str, UserAccum] = {}
    from collections import OrderedDict
    conv_buffer: "OrderedDict" = OrderedDict()  # conv_id -> [utt dicts], LRU-ordered
    buffered_n = 0                               # running total of buffered utterances
    n = 0

    def _flush(conv_id):
        nonlocal buffered_n
        utts = conv_buffer.pop(conv_id, None)
        if utts is None:
            return
        buffered_n -= len(utts)
        meta = conv_meta.get(conv_id, {})
        page_id = meta.get("page_id") or conv_id
        page_type = meta.get("page_type")
        # local msg_id -> author + toxicity for this conversation
        local_author: dict = {}
        local_tox: dict = {}
        participants: set = set()
        for u in utts:
            sp = _speaker(u.get("speaker"))
            if sp:
                local_author[u.get("id")] = sp
                participants.add(sp)
            tv = _tox(u.get("meta", {}), "toxicity")
            if tv is not None:
                local_tox[u.get("id")] = tv
        contested = len(participants) >= contested_threshold
        for u in utts:
            sp = _speaker(u.get("speaker"))
            if not sp or sp not in active:
                continue
            a = accums.setdefault(sp, UserAccum())
            m = u.get("meta", {})
            text = u.get("text") or ""
            ts = u.get("timestamp")
            try:
                ts = float(ts) if ts is not None else 0.0
            except (TypeError, ValueError):
                ts = 0.0
            _accumulate(
                a, text, m.get("indentation"),
                _tox(m, "toxicity"), _tox(m, "sever_toxicity"),
                ts, page_id, page_type, contested,
            )
            # reply graph: this user replies to parent author
            rt = u.get("reply-to") or u.get("reply_to")
            if rt:
                a.reply_n += 1
                pa = local_author.get(rt)
                if pa and pa != sp:
                    _add_neighbour(a.out_targets, pa)
                    tgt_a = accums.setdefault(pa, UserAccum())
                    _add_neighbour(tgt_a.in_repliers, sp)
                    tgt_a.replies_received += 1
                    if u.get("id") in local_tox:
                        tv = local_tox[u.get("id")]
                        tgt_a.tox_recv_sum += tv
                        tgt_a.tox_recv_n += 1
                    # reply-to-toxic: parent's toxicity
                    if local_tox.get(rt, 0.0) >= toxic_threshold:
                        a.reply_to_toxic_n += 1
            # Process moderation events performed on / to this comment
            for field_name, action in (("modification", "edit"), ("deletion", "delete"), ("restoration", "restore")):
                for ev in m.get(field_name) or []:
                    actor = _speaker(ev.get("speaker"))
                    if actor and actor in active:
                        ev_a = accums.setdefault(actor, UserAccum())
                        ev_ts = ev.get("timestamp")
                        try:
                            ev_ts = float(ev_ts) if ev_ts is not None else 0.0
                        except (TypeError, ValueError):
                            ev_ts = 0.0
                        _accumulate_action(ev_a, action, ev_ts, page_id, contested)
                    if sp and sp in active:
                        if action == "edit":
                            a.n_modified_received += 1
                        elif action == "delete":
                            a.n_deleted_received += 1
                        elif action == "restore":
                            a.n_restored_received += 1

    with open(cdir / "utterances.jsonl", "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = _json_loads(line)
            n += 1
            conv_id = r.get("conversation_id") or r.get("root")
            # LRU: touch existing conversation (move to end = most-recent)
            if conv_id in conv_buffer:
                conv_buffer.move_to_end(conv_id)
                conv_buffer[conv_id].append(r)
            else:
                conv_buffer[conv_id] = [r]
            buffered_n += 1
            # evict oldest (least-recent) conversations when buffer exceeds cap
            while buffered_n > CONV_BUFFER_CAP:
                old = next(iter(conv_buffer))
                _flush(old)
    while conv_buffer:
        old = next(iter(conv_buffer))
        _flush(old)
    with open(cache_path, "wb") as f:
        pickle.dump(accums, f)
    logger.info(f"Pass B {cdir.name}: {n} utts, {len(accums)} active users touched")
    return accums


def stream_features(
    year_dirs: list[Path],
    active: set[str],
    contested_threshold: int = 3,
    toxic_threshold: float = 0.6,
    workers: int = 8,
    cache_dir: str | Path = "outputs/stream_cache",
) -> dict[str, UserAccum]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    merged: dict[str, UserAccum] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(
            _detail_year,
            [(d, active, contested_threshold, toxic_threshold, cache_dir)
             for d in year_dirs],
        )
        for accums in results:
            for u, a in accums.items():
                if u in merged:
                    _merge_accum(merged[u], a)
                else:
                    merged[u] = a
    logger.info(f"Pass B merged: {len(merged)} active users")
    return merged


def _merge_accum(dst: UserAccum, src: UserAccum) -> None:
    dst.n += src.n
    # action counters
    dst.n_discuss += src.n_discuss
    dst.n_edit += src.n_edit
    dst.n_delete += src.n_delete
    dst.n_restore += src.n_restore
    dst.n_modified_received += src.n_modified_received
    dst.n_deleted_received += src.n_deleted_received
    dst.n_restored_received += src.n_restored_received
    dst.reply_n += src.reply_n
    dst.reply_to_toxic_n += src.reply_to_toxic_n
    dst.indent_sum += src.indent_sum
    dst.text_len_sum += src.text_len_sum
    dst.tox_sum += src.tox_sum
    dst.tox_n += src.tox_n
    dst.tox_max = max(dst.tox_max, src.tox_max)
    dst.sev_sum += src.sev_sum
    dst.sev_n += src.sev_n
    dst.sev_max = max(dst.sev_max, src.sev_max)
    dst.exclaim_n += src.exclaim_n
    dst.question_n += src.question_n
    dst.wp_n += src.wp_n
    dst.contested_n += src.contested_n
    dst.ns_interpersonal_n += src.ns_interpersonal_n
    dst.ns_content_n += src.ns_content_n
    dst.ns_project_n += src.ns_project_n
    if src.ts_min and (not dst.ts_min or src.ts_min < dst.ts_min):
        dst.ts_min = src.ts_min
    if src.ts_max > dst.ts_max:
        dst.ts_max = src.ts_max
    if len(dst.ts_samples) < SAMPLE_TS_CAP:
        dst.ts_samples.extend(src.ts_samples[:SAMPLE_TS_CAP - len(dst.ts_samples)])
    dst.page_ids |= src.page_ids
    dst.out_targets |= src.out_targets
    dst.in_repliers |= src.in_repliers
    dst.replies_received += src.replies_received
    dst.tox_recv_sum += src.tox_recv_sum
    dst.tox_recv_n += src.tox_recv_n
    for pool, spool in ((dst.hi_tox_texts, src.hi_tox_texts),
                        (dst.lo_tox_texts, src.lo_tox_texts)):
        for t in spool:
            if len(pool) >= SAMPLE_TEXTS_PER_USER:
                break
            pool.append(t)


# ---------------------------------------------------------------------------
# Accumulator -> UserFeatures + language embedding inputs
# ---------------------------------------------------------------------------

def accum_to_features(uid: str, a: UserAccum) -> "UserFeatures":
    from src.clustering.features import UserFeatures

    n = a.n or 1
    out_set, in_set = a.out_targets, a.in_repliers
    union = out_set | in_set
    times_span_days = (a.ts_max - a.ts_min) / 86400.0 if (a.ts_max and a.ts_min) else 0.0
    # lexical ttr over the sampled texts (both toxicity pools)
    toks = []
    for t in a.hi_tox_texts + a.lo_tox_texts:
        toks.extend(_WORD_RE.findall(t.lower()))

    activity = math.log1p(n)
    activity_density = (
        math.log1p(n / (times_span_days + 1.0))
        if times_span_days > 0.0 else activity
    )
    # Burstiness: CV of inter-message time gaps (rhythm signature).
    if len(a.ts_samples) >= 3:
        gaps = np.diff(np.sort(np.asarray(a.ts_samples, dtype=np.float64)))
        burstiness_cv = float(gaps.std() / (gaps.mean() + 1e-9)) if gaps.size else 0.0
    else:
        burstiness_cv = 0.0

    return UserFeatures(
        user_id=uid,
        reply_rate=a.reply_n / n,
        mean_indentation=min(a.indent_sum / n, 20.0),
        verbosity=math.log1p(a.text_len_sum / n),
        activity=activity,
        interlocutor_breadth=len(a.out_targets) / n,
        attention_received=math.log1p(a.replies_received),
        reciprocity=(len(out_set & in_set) / len(union)) if union else 0.0,
        topical_breadth=math.log1p(len(a.page_ids)),
        tenure=math.log1p(max(0.0, times_span_days)),
        own_toxicity_mean=(a.tox_sum / a.tox_n) if a.tox_n else 0.0,
        own_severe_toxicity_mean=(a.sev_sum / a.sev_n) if a.sev_n else 0.0,
        reply_to_toxic_rate=(a.reply_to_toxic_n / a.reply_n) if a.reply_n else 0.0,
        toxicity_received_mean=(a.tox_recv_sum / a.tox_recv_n) if a.tox_recv_n else 0.0,
        conflict_engagement_ratio=a.contested_n / n,
        question_rate=a.question_n / n,
        wp_citation_rate=a.wp_n / n,
        frac_interpersonal=a.ns_interpersonal_n / n,
        frac_content=a.ns_content_n / n,
        frac_project=a.ns_project_n / n,
        exclaim_rate=a.exclaim_n / n,
        lexical_ttr=(len(set(toks)) / len(toks)) if toks else 0.0,
        # Conflict extremes + temporal rhythm
        tox_max=a.tox_max, sev_max=a.sev_max, burstiness_cv=burstiness_cv,
        activity_density=activity_density,
        hostility_score=max(a.tox_max, a.sev_max),
        namespace_focus=(a.ns_content_n / n) - (a.ns_interpersonal_n / n),
        message_count=a.n,
        thread_count=0,
    )


# ---------------------------------------------------------------------------
# helpers (late import to keep workers light)
# ---------------------------------------------------------------------------

def _load_conversations(cdir: Path) -> dict:
    import json
    path = cdir / "conversations.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {cid: (v.get("meta", {}) if isinstance(v, dict) else {}) for cid, v in data.items()}


def find_year_dirs(data_dir: str | Path) -> list[Path]:
    """Return ConvoKit year-corpus dirs under data_dir (exclude CGA)."""
    root = Path(data_dir)
    dirs = []
    for idx in root.rglob("index.json"):
        d = idx.parent
        if not (d / "utterances.jsonl").exists():
            continue
        name = d.name.lower()
        if "awry" in name or name == "cga":
            continue
        dirs.append(d)
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Pass C: targeted utterance collection for representative members
# ---------------------------------------------------------------------------

def _collect_year(args) -> dict:
    """Collect utterances (text + light context) for specific member users."""
    cdir, member_ids, max_per_user = args
    cdir = Path(cdir)
    conv_meta = _load_conversations(cdir)
    out: dict[str, list] = {}
    # one pass; keep parent text via a small rolling window of recent msgs
    recent: dict = {}   # msg_id -> (text, author)  (bounded below)
    n = 0
    with open(cdir / "utterances.jsonl", "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = _json_loads(line)
            n += 1
            sp = _speaker(r.get("speaker"))
            mid = r.get("id")
            text = (r.get("text") or "")[:1000]
            recent[mid] = (text, sp)
            if len(recent) > 50_000:           # bound the parent-lookup window
                recent.pop(next(iter(recent)))
            if sp in member_ids and text.strip():
                rt = r.get("reply-to") or r.get("reply_to")
                parent_text = ""
                if rt and rt in recent:
                    parent_text = (recent[rt][0] or "")[:200]
                lst = out.setdefault(sp, [])
                if len(lst) < max_per_user:
                    conv_id = r.get("conversation_id") or r.get("root")
                    lst.append({
                        "text": text,
                        "action": "discuss",
                        "parent_context": parent_text,
                        "topic": conv_meta.get(conv_id, {}).get("page_title", ""),
                    })
    return out


def collect_member_utterances(
    year_dirs: list[Path], member_ids: set[str], max_per_user: int = 60,
    workers: int = 8,
) -> dict[str, list]:
    """Stream all years, collecting utterances only for the given members."""
    merged: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for partial in ex.map(_collect_year,
                              [(d, member_ids, max_per_user) for d in year_dirs]):
            for u, items in partial.items():
                merged.setdefault(u, []).extend(items)
    return merged


# ---------------------------------------------------------------------------
# Pass C': per-leaf rejection-evidence collection (anti-pattern grounding)
# ---------------------------------------------------------------------------

# Evidence semantics are owned by corpus_export; keep them in sync.
from src.skill.corpus_export import EVIDENCE_BUDGET, EVIDENCE_TOX_THRESHOLD


def _evidence_line(label: str, text: str) -> str:
    return f"- [{label}] {(text or '')[:160]}"


def _collect_evidence_year(args) -> dict:
    """Collect per-leaf rejection-evidence lines from one year.

    Mirrors ``corpus_export._rejected_evidence`` but over the raw ConvoKit stream
    (no in-memory threads). Per-type budgets keep the abundant high-tox signal
    from crowding out rarer deletion / attack signals. One bounded pass per year.
    """
    cdir, user_to_leaf, budget = args
    cdir = Path(cdir)
    out: dict[int, dict[str, list[str]]] = {}
    counts: dict[int, dict[str, int]] = {}
    n = 0
    with open(cdir / "utterances.jsonl", "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = _json_loads(line)
            n += 1
            m = r.get("meta", {}) or {}
            text = r.get("text") or ""

            def _room(leaf: int | None, etype: str) -> bool:
                if leaf is None or leaf < 0:
                    return False
                return counts.get(leaf, {}).get(etype, 0) < budget.get(etype, 0)

            # (a) speaker's own genuinely-toxic / personal-attack utterance
            sp = _speaker(r.get("speaker"))
            leaf_sp = user_to_leaf.get(sp)
            if text.strip():
                attack = m.get("comment_has_personal_attack")
                tox = _tox(m, "toxicity")
                if attack and _room(leaf_sp, "attack"):
                    out.setdefault(leaf_sp, {}).setdefault("attack", []).append(
                        _evidence_line("personal-attack flagged", text)
                    )
                    counts.setdefault(leaf_sp, {})["attack"] = counts.get(leaf_sp, {}).get("attack", 0) + 1
                elif isinstance(tox, (int, float)) and float(tox) >= EVIDENCE_TOX_THRESHOLD and _room(leaf_sp, "tox"):
                    label = f"high-conflict/flagged tox={round(float(tox), 2)}"
                    out.setdefault(leaf_sp, {}).setdefault("tox", []).append(_evidence_line(label, text))
                    counts.setdefault(leaf_sp, {})["tox"] = counts.get(leaf_sp, {}).get("tox", 0) + 1

            # (b) deletion events this user performed on another's comment.
            # The event lives in the target comment's meta; its text IS the
            # moderated comment's text, so no cross-utterance lookup is needed.
            for ev in m.get("deletion") or []:
                actor = _speaker(ev.get("speaker"))
                leaf_ac = user_to_leaf.get(actor)
                if _room(leaf_ac, "delete"):
                    out.setdefault(leaf_ac, {}).setdefault("delete", []).append(
                        _evidence_line("deleted another's comment", text)
                    )
                    counts.setdefault(leaf_ac, {})["delete"] = counts.get(leaf_ac, {}).get("delete", 0) + 1
    return out


def collect_rejection_evidence(
    year_dirs: list[Path],
    user_to_leaf: dict[str, int],
    budget: dict[str, int] | None = None,
    workers: int = 8,
) -> dict[int, list[str]]:
    """Stream all years → per-leaf rejection-evidence lines with per-type budgets.

    Grounds anti-patterns in actual in-corpus violations (deletions / high
    toxicity / personal attacks) for the streaming export path, which cannot
    materialise full threads. Output format matches
    ``corpus_export._rejected_evidence``. ``user_to_leaf`` maps only clustered
    users (leaf ≥ 0); inactive (-1) users are excluded.
    """
    if budget is None:
        budget = EVIDENCE_BUDGET
    merged: dict[int, dict[str, list[str]]] = {}
    seen: dict[int, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for partial in ex.map(
            _collect_evidence_year,
            [(d, user_to_leaf, budget) for d in year_dirs],
        ):
            for leaf, types in partial.items():
                leaf = int(leaf)
                for etype, lines in types.items():
                    cap = budget.get(etype, 0)
                    if seen.get(leaf, {}).get(etype, 0) >= cap:
                        continue
                    room = cap - seen.get(leaf, {}).get(etype, 0)
                    take = lines[:room]
                    if take:
                        merged.setdefault(leaf, {}).setdefault(etype, []).extend(take)
                        seen.setdefault(leaf, {})[etype] = seen.get(leaf, {}).get(etype, 0) + len(take)
    # flatten in type order (delete, tox, attack) for readability
    result: dict[int, list[str]] = {}
    for leaf in sorted(merged):
        parts: list[str] = []
        for etype in ("delete", "tox", "attack"):
            parts.extend(merged[leaf].get(etype, []))
        if parts:
            result[leaf] = parts
    return result


def embed_sample_texts(
    accums: dict[str, UserAccum], batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Per-user style embedding (GPU, main proc): mean+std pooling of per-text vecs.

    Each sampled text is encoded INDIVIDUALLY (truncation-free, matching the
    non-streaming EmbeddingExtractor), then pooled per user as the concatenation
    of the per-dimension MEAN and STD across that user's texts. A Q5 experiment
    (split-half self-consistency on the 2003-2004 sample) showed mean+std beats
    plain mean (rank1 0.316 vs 0.299) and max-pool (0.218): the std half encodes
    how much a user varies their register — a real, stable style signal that mean
    alone discards. Texts come from the toxicity-stratified pools so a user's
    conflict register is represented, not just their first-K mundane messages.
    """
    from src.config.settings import get_shared_embedder

    embedder = get_shared_embedder()
    users = list(accums.keys())
    # Flatten every user's stratified texts into one encode queue; remember slices.
    flat_texts: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    for u in users:
        texts = [t for t in stratified_sample_texts(accums[u]) if t.strip()] or [" "]
        start = len(flat_texts)
        flat_texts.extend(texts)
        ranges[u] = (start, len(flat_texts))

    chunks: list[np.ndarray] = []
    for i in range(0, len(flat_texts), batch_size):
        chunks.append(np.asarray(
            embedder.encode(flat_texts[i:i + batch_size], show_progress_bar=True)
        ))
    allv = np.vstack(chunks) if chunks else np.zeros((0, 0))

    out: dict[str, np.ndarray] = {}
    for u, (s, e) in ranges.items():
        block = allv[s:e]
        out[u] = np.concatenate([block.mean(axis=0), block.std(axis=0)])
    return out


def run_streaming_pipeline(
    data_dir: str | Path,
    min_messages: int = 5,
    workers: int = 8,
    contested_threshold: int = 3,
    toxic_threshold: float = 0.6,
    cache_dir: str | Path = "outputs/stream_cache",
    clusterer: "BehavioralClusterer | None" = None,
) -> tuple:
    """Full-dataset streaming clustering. No population sampling.

    Returns (cluster_result, accums). Users below ``min_messages`` are filtered
    to out-of-cluster (label -1); every other user is clustered. Year-level
    caches in ``cache_dir`` make the run resumable — rerun skips finished years.
    """
    from src.clustering.clusterer import BehavioralClusterer
    from src.clustering.features import UserFeatures

    year_dirs = find_year_dirs(data_dir)
    if not year_dirs:
        raise FileNotFoundError(f"No ConvoKit year corpora under {data_dir}")
    logger.info(f"Streaming {len(year_dirs)} year corpora from {data_dir}")

    counts, n_utts = stream_counts(year_dirs, workers=workers, cache_dir=cache_dir)
    active = {u for u, c in counts.items() if c >= min_messages}
    logger.info(
        f"Activity filter (>= {min_messages} msgs): {len(active)}/{len(counts)} users kept"
    )

    accums = stream_features(
        year_dirs, active, contested_threshold, toxic_threshold,
        workers=workers, cache_dir=cache_dir,
    )
    # stream_features also creates zero-activity accumulators for reply TARGETS
    # (parent authors) who themselves fall below the activity floor. Their
    # reply-graph signal is already folded into the active users that point at
    # them; their own vector is all-zeros. Keep only users above the floor so
    # these "ghosts" don't flood the clustering with identical zero rows.
    accums = {u: a for u, a in accums.items() if u in active and a.n > 0}
    logger.info(f"After dropping zero-activity reply-target ghosts: {len(accums)} users")
    user_features: dict[str, UserFeatures] = {
        u: accum_to_features(u, a) for u, a in accums.items()
    }

    # Embeddings are the expensive GPU step — cache them so a crash in the
    # later clustering phase doesn't force a re-embed.
    emb_cache = Path(cache_dir) / "embeddings.pkl"
    import pickle
    user_embeddings = {}
    if emb_cache.exists():
        with open(emb_cache, "rb") as f:
            cached = pickle.load(f)
        # keep only users still present (drop any that changed)
        user_embeddings = {u: v for u, v in cached.items() if u in accums}
        logger.info(f"Loaded cached embeddings for {len(user_embeddings)} users")
    # Re-embed any active user missing from the cache. A stale cache from a
    # smaller run (e.g. a smoke test) would otherwise leave most users
    # un-embedded, silently dropping them from clustering.
    missing = {u: a for u, a in accums.items() if u not in user_embeddings}
    if missing:
        logger.info(f"Embedding {len(missing)} users missing from cache ...")
        user_embeddings.update(embed_sample_texts(missing))
        with open(emb_cache, "wb") as f:
            pickle.dump(user_embeddings, f)
        logger.info(f"Cached embeddings for {len(user_embeddings)} users → {emb_cache}")

    logger.info(f"Clustering {len(user_features)} users (behaviour-only roles)...")
    if clusterer is None:
        clusterer = BehavioralClusterer()
    # Cache the clustering result keyed by every input that affects the output,
    # so the (KMeans/HDBSCAN) clustering is one-time and param changes auto-
    # recompute. cluster_algo tags the algorithm shape: dropping the language
    # sub-clustering stage (behaviour-only) bumps it and invalidates older caches
    # even when role_k / features are unchanged.
    import hashlib, pickle as _pk
    from src.clustering.features import VECTOR_FIELD_NAMES
    key = {
        "cluster_algo": "behavior_only_v1",
        "feature_version": _DETAIL_CACHE_VERSION,
        # the actual clustered vector composition, so any feature add/drop
        # auto-invalidates the clustering cache (independent of detail caches).
        "vector_fields": list(VECTOR_FIELD_NAMES),
        "n_active": len(user_features),
        "sample_users": sorted(user_features)[:8],
        "seed": clusterer.random_state,
        "role_method": clusterer.role_method,
        "role_k": clusterer.role_k,
        "role_mcs": clusterer.role_min_cluster_size,
        "role_min_samples": clusterer.role_min_samples,
        "target_min_leaves": clusterer.target_min_leaves,
        "target_max_leaves": clusterer.target_max_leaves,
        "scaler": clusterer.scaler,
        "impute_orphans": clusterer.impute_orphans,
        "cluster_selection_method": clusterer.cluster_selection_method,
    }
    digest = hashlib.md5(_pk.dumps(key, protocol=4)).hexdigest()[:10]
    cr_cache = Path(cache_dir) / f"clustering_{digest}.pkl"
    if cr_cache.exists():
        with open(cr_cache, "rb") as f:
            cr = _pk.load(f)
        logger.info(f"Loaded cached clustering result ({cr.n_clusters} leaves) → {cr_cache}")
    else:
        cr = clusterer.fit_from_vectors(user_features, user_embeddings)
        with open(cr_cache, "wb") as f:
            # drop per-user embeddings (kept in embeddings.pkl) to keep cache small
            cr_for_cache = cr
            _pk.dump(cr_for_cache, f, protocol=4)
        logger.info(f"Cached clustering result → {cr_cache}")

    # route filtered (inactive) users to out-of-cluster
    n_activity_filtered = 0
    for u in counts:
        if u not in accums:
            cr.labels.setdefault(u, -1)
            n_activity_filtered += 1
    n_orphans = getattr(cr, "pre_impute_orphans", 0)
    n_orphans_kept = getattr(cr, "n_orphans_kept", n_orphans) if not clusterer.impute_orphans else 0
    logger.info(
        f"Pipeline done: {cr.n_clusters} leaves; "
        f"pre-impute HDBSCAN orphans={n_orphans} "
        f"(kept={n_orphans_kept}, imputed={n_orphans - n_orphans_kept}); "
        f"activity-filtered (background)={n_activity_filtered}"
    )
    return cr, accums, user_embeddings
