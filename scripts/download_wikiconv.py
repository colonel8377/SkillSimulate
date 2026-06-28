#!/usr/bin/env python3
"""Download the full English WikiConv (2001-2018) + CGA into one directory.

All years land in ONE dir: data/raw/wikiconv_en/wikiconv-<year>/ , plus cga/.
~25.7 GB zipped for all years; unzipped is larger. Idempotent: skips a year
whose utterances.jsonl already exists. Safe to re-run / resume.

Usage:
    python scripts/download_wikiconv.py                # all years 2001-2018 + CGA
    python scripts/download_wikiconv.py 2005 2006 2007 # only these years
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

BASE = "https://zissou.infosci.cornell.edu/convokit/datasets/"
WIKICONV = BASE + "wikiconv-corpus/corpus-zipped/english/wikiconv-{year}/full.corpus.zip"
CGA = BASE + "conversations-gone-awry-corpus/conversations-gone-awry-corpus.zip"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "wikiconv_en")
OUT_DIR = os.path.abspath(OUT_DIR)


def _fetch(url: str, zip_path: str, dest_dir: str, done_marker: str) -> None:
    if os.path.exists(done_marker):
        print(f"  skip (exists): {os.path.relpath(dest_dir, OUT_DIR)}")
        return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  unzipping ({size_mb:.0f} MB) -> {os.path.relpath(dest_dir, OUT_DIR)}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    os.remove(zip_path)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    years = sys.argv[1:] or [str(y) for y in range(2001, 2019)]

    for year in years:
        dest = os.path.join(OUT_DIR, f"wikiconv-{year}")
        marker = os.path.join(dest, "utterances.jsonl")
        try:
            _fetch(WIKICONV.format(year=year), os.path.join(OUT_DIR, f"_{year}.zip"), dest, marker)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {year}: {type(e).__name__}: {e}")

    # CGA gold conflict labels
    cga_dest = os.path.join(OUT_DIR, "cga")
    _fetch(CGA, os.path.join(OUT_DIR, "_cga.zip"), cga_dest,
           os.path.join(cga_dest, "utterances.jsonl"))

    print(f"\nDone. Data in: {OUT_DIR}")


if __name__ == "__main__":
    main()
