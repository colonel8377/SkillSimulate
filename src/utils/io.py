"""I/O utilities for save/load operations."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(data: Any, path: str | Path) -> None:
    """Atomically save JSON in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as f:
            tmp_name = f.name
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: str | Path) -> Any:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)


def save_pickle(data: Any, path: str | Path) -> None:
    """Save data as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str | Path) -> Any:
    """Load data from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path)
