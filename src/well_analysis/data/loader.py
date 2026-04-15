"""Data loading and basic validation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _resolve_data_dir() -> Path:
    env = os.environ.get("WELL_ANALYSIS_DATA_DIR")
    if env:
        return Path(env).expanduser() / "raw"
    return Path(__file__).resolve().parents[4] / "well-analysis-data" / "raw"


DATA_DIR = _resolve_data_dir()


def load_test_data(path: Path | None = None) -> pd.DataFrame:
    """Load and parse Test.csv.

    Returns a DataFrame with:
        - Timestamp  : datetime64[ns, UTC] — sorted ascending
        - Acceleration: float64 (m/s², includes gravity offset)
        - Load       : float64 (N)
    """
    if path is None:
        path = DATA_DIR / "Test.csv"

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    expected_cols = {"Timestamp", "Acceleration", "Load"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df
