"""Sampling frequency analysis."""

import numpy as np
import pandas as pd


def compute_intervals(timestamps: pd.Series) -> np.ndarray:
    """Return successive time differences in seconds."""
    return timestamps.diff().dt.total_seconds().dropna().values


def check_even_sampling(
    timestamps: pd.Series, tol: float = 1e-3
) -> tuple[bool, float]:
    """Check whether timestamps are evenly sampled.

    Efficient O(n) method: compute median interval, then check that the
    maximum deviation from the median is within ``tol`` (relative).

    Parameters
    ----------
    timestamps:
        Sorted datetime Series.
    tol:
        Relative tolerance. Default 0.1 % of the median interval.

    Returns
    -------
    is_even : bool
    sampling_frequency : float
        1 / median_interval in Hz.
    """
    intervals = compute_intervals(timestamps)
    median_dt = float(np.median(intervals))
    max_deviation = float(np.max(np.abs(intervals - median_dt)))
    is_even = max_deviation < tol * median_dt
    fs = 1.0 / median_dt
    return is_even, fs
