"""Data loading and basic validation."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[4] / "data" / "raw"


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
