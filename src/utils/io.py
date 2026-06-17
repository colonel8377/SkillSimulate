"""I/O utilities for save/load operations."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


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
