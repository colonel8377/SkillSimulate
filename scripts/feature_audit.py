"""Data-driven feature audit on a small early-year sample (2003+2004).

Decides, with evidence rather than intuition:
  Q3  which existing vector columns are DEAD (near-constant) or REDUNDANT (|r|>0.9)
  Q2  whether toxicity-derived / structural candidate features add real signal
  Q4  whether toxicity-stratified sampling captures conflict text that first-K misses

Existing features go through the REAL streaming pipeline (faithful). Candidate
features + per-text toxicity are collected in a second lightweight pass.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.streaming import (  # noqa: E402
    stream_counts, stream_features, accum_to_features, _json_loads, _speaker, _tox,
)
from src.clustering.features import VECTOR_FIELD_NAMES  # noqa: E402

YEARS = [Path("data/raw/wikiconv_en/wikiconv-2003"),
         Path("data/raw/wikiconv_en/wikiconv-2004")]
MIN_MSGS = 50
CACHE = "outputs/stream_cache"


def existing_feature_matrix(active):
    accums = stream_features(YEARS, active, workers=2, cache_dir=CACHE)
    # drop zero-activity reply-target ghosts (same filter as the real pipeline)
    accums = {u: a for u, a in accums.items() if u in active and a.n > 0}
    uids = sorted(accums)
    feats = {u: accum_to_features(u, accums[u]) for u in uids}
    M = np.stack([feats[u].to_vector() for u in uids])
    return uids, M, accums


def candidate_pass(active):
    """Second pass: per-user raw distributions UserAccum doesn't keep."""
    tox = defaultdict(list); sev = defaultdict(list); indent = defaultdict(list)
    ts = defaultdict(list); tlen = defaultdict(list)
    samples = defaultdict(list)   # (text, toxicity) for Q4, capped
    for cdir in YEARS:
        with open(cdir / "utterances.jsonl", "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                r = _json_loads(line)
                sp = _speaker(r.get("speaker"))
                if not sp or sp not in active:
                    continue
                m = r.get("meta", {})
                tv = _tox(m, "toxicity"); sv = _tox(m, "sever_toxicity")
                if tv is not None:
                    tox[sp].append(tv)
                if sv is not None:
                    sev[sp].append(sv)
                try:
                    indent[sp].append(max(0, int(m.get("indentation"))))
                except (TypeError, ValueError):
                    pass
                t = r.get("timestamp")
                try:
                    ts[sp].append(float(t))
                except (TypeError, ValueError):
                    pass
                txt = r.get("text") or ""
                tlen[sp].append(len(txt))
                if txt.strip() and len(samples[sp]) < 400:
                    samples[sp].append((txt[:512], tv if tv is not None else 0.0))
    return tox, sev, indent, ts, tlen, samples


def cand_features(uids, tox, sev, indent, ts, tlen):
    """Build candidate feature matrix aligned to uids."""
    names = ["tox_mean", "tox_std", "tox_p90", "tox_max", "tox_high_frac",
             "sev_std", "sev_max", "indent_std", "indent_max",
             "burstiness_cv", "active_days", "len_std"]
    rows = []
    for u in uids:
        tv = np.array(tox.get(u, [])) if tox.get(u) else np.array([0.0])
        sv = np.array(sev.get(u, [])) if sev.get(u) else np.array([0.0])
        iv = np.array(indent.get(u, [])) if indent.get(u) else np.array([0.0])
        tt = np.sort(np.array(ts.get(u, [])) ) if ts.get(u) else np.array([0.0])
        lv = np.array(tlen.get(u, [])) if tlen.get(u) else np.array([0.0])
        gaps = np.diff(tt) if tt.size >= 2 else np.array([0.0])
        cv = float(gaps.std() / (gaps.mean() + 1e-9)) if gaps.size else 0.0
        span_days = float((tt[-1] - tt[0]) / 86400.0) if tt.size >= 2 else 0.0
        rows.append([
            float(tv.mean()), float(tv.std()), float(np.percentile(tv, 90)),
            float(tv.max()), float((tv >= 0.6).mean()),
            float(sv.std()), float(sv.max()),
            float(iv.std()), float(iv.max()),
            cv, np.log1p(span_days), float(lv.std()),
        ])
    return names, np.array(rows)


def report_variance(names, M, tag):
    print(f"\n{'='*70}\n{tag}: per-column spread ({M.shape[0]} users, {M.shape[1]} cols)\n{'='*70}")
    print(f"{'col':<24}{'std':>9}{'nonzero%':>10}{'p50':>9}{'p90':>9}  flag")
    dead = []
    for i, nm in enumerate(names):
        c = M[:, i]
        std = c.std(); nz = (c != 0).mean()
        flag = ""
        if std < 1e-4 or nz < 0.01:
            flag = "DEAD"; dead.append(nm)
        elif nz < 0.05:
            flag = "near-dead"
        print(f"{nm:<24}{std:>9.4f}{nz:>9.1%}{np.percentile(c,50):>9.3f}{np.percentile(c,90):>9.3f}  {flag}")
    return dead


def report_correlation(names, M, threshold=0.9):
    # standardize, guard zero-variance
    std = M.std(axis=0); std[std == 0] = 1.0
    Z = (M - M.mean(axis=0)) / std
    C = np.corrcoef(Z, rowvar=False)
    print(f"\n{'='*70}\nHIGH CORRELATION pairs |r| > {threshold}\n{'='*70}")
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            r = C[i, j]
            if np.isfinite(r) and abs(r) > threshold:
                pairs.append((abs(r), names[i], names[j], r))
    for ar, a, b, r in sorted(pairs, reverse=True):
        print(f"  r={r:+.3f}   {a:<22} <-> {b}")
    if not pairs:
        print("  (none)")
    return pairs


def report_candidate_vs_existing(cand_names, Mc, ex_names, Me):
    """Does each candidate carry signal NOT already in existing features?"""
    print(f"\n{'='*70}\nQ2: candidate features — spread + max|r| vs ANY existing col\n{'='*70}")
    sd = Me.std(axis=0); sd[sd == 0] = 1.0
    Ze = (Me - Me.mean(axis=0)) / sd
    sdc = Mc.std(axis=0); sdc_safe = sdc.copy(); sdc_safe[sdc_safe == 0] = 1.0
    Zc = (Mc - Mc.mean(axis=0)) / sdc_safe
    print(f"{'candidate':<20}{'std':>9}{'max|r|exist':>14}  {'closest existing':<22}{'verdict'}")
    for i, nm in enumerate(cand_names):
        if sdc[i] < 1e-6:
            print(f"{nm:<20}{sdc[i]:>9.4f}{'--':>14}  {'(constant)':<22}DROP")
            continue
        cors = [abs(np.corrcoef(Zc[:, i], Ze[:, j])[0, 1]) for j in range(Ze.shape[1])]
        k = int(np.nanargmax(cors)); mx = cors[k]
        verdict = "redundant" if mx > 0.85 else ("ADDS SIGNAL" if mx < 0.6 else "partial")
        print(f"{nm:<20}{sdc[i]:>9.4f}{mx:>14.3f}  {ex_names[k]:<22}{verdict}")


def report_stratified_sampling(samples, k=15):
    """Q4: first-K vs toxicity-stratified reservoir — toxic-text capture."""
    print(f"\n{'='*70}\nQ4: sampling — toxic-text capture (first-{k} vs stratified, k={k})\n{'='*70}")
    fk_caps, st_caps, true_rates = [], [], []
    for u, items in samples.items():
        if len(items) < k:
            continue
        toxs = np.array([t for _, t in items])
        true_hi = (toxs >= 0.6).mean()
        if true_hi == 0:
            continue   # only audit users who DO have conflict text
        # first-K
        fk = toxs[:k]
        # stratified: take all high-tox up to k/2, fill rest from low-tox (deterministic order)
        hi_idx = [i for i, t in enumerate(toxs) if t >= 0.6]
        lo_idx = [i for i, t in enumerate(toxs) if t < 0.6]
        take_hi = hi_idx[:k // 2]
        take_lo = lo_idx[:k - len(take_hi)]
        st = toxs[take_hi + take_lo]
        fk_caps.append((fk >= 0.6).mean())
        st_caps.append((st >= 0.6).mean())
        true_rates.append(true_hi)
    if not fk_caps:
        print("  (no users with conflict text and >=k samples)")
        return
    print(f"  users with conflict text: {len(fk_caps)}")
    print(f"  mean TRUE high-tox rate     : {np.mean(true_rates):.3%}")
    print(f"  first-K  captured high-tox  : {np.mean(fk_caps):.3%}")
    print(f"  stratified captured high-tox: {np.mean(st_caps):.3%}")
    print(f"  → stratified surfaces {np.mean(st_caps)/max(np.mean(fk_caps),1e-9):.1f}x more conflict text")


def main():
    print(f"Loading counts for {[y.name for y in YEARS]} ...")
    counts, _ = stream_counts(YEARS, workers=2, cache_dir=CACHE)
    active = {u for u, c in counts.items() if c >= MIN_MSGS}
    print(f"active (>= {MIN_MSGS} msgs): {len(active)} users")

    uids, M, accums = existing_feature_matrix(active)
    print(f"existing feature matrix: {M.shape}")

    dead = report_variance(list(VECTOR_FIELD_NAMES), M, "Q3 EXISTING")
    pairs = report_correlation(list(VECTOR_FIELD_NAMES), M, threshold=0.9)

    print("\nLoading candidate raw distributions (2nd pass) ...")
    tox, sev, indent, ts, tlen, samples = candidate_pass(active)
    cand_names, Mc = cand_features(uids, tox, sev, indent, ts, tlen)
    report_variance(cand_names, Mc, "Q2 CANDIDATES")
    report_candidate_vs_existing(cand_names, Mc, list(VECTOR_FIELD_NAMES), M)
    report_stratified_sampling(samples, k=15)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"DEAD existing cols ({len(dead)}): {dead}")
    print(f"REDUNDANT pairs ({len(pairs)}): {[(a,b) for _,a,b,_ in pairs]}")


if __name__ == "__main__":
    main()
