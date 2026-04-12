"""Tests for sampling analysis."""

import numpy as np
import pandas as pd
import pytest

from well_analysis.signal.sampling import check_even_sampling, compute_intervals


def _make_timestamps(n: int, dt: float, jitter: float = 0.0) -> pd.Series:
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    offsets = np.arange(n) * dt + np.random.default_rng(0).uniform(-jitter, jitter, n)
    return pd.Series([t0 + pd.Timedelta(seconds=s) for s in offsets])


def test_intervals_shape():
    ts = _make_timestamps(100, 0.1)
    ivs = compute_intervals(ts)
    assert ivs.shape == (99,)


def test_even_sampling_perfect():
    ts = _make_timestamps(1000, 0.1)
    is_even, fs = check_even_sampling(ts)
    assert is_even
    assert abs(fs - 10.0) < 0.01


def test_even_sampling_with_small_jitter():
    ts = _make_timestamps(1000, 0.1, jitter=1e-5)
    is_even, fs = check_even_sampling(ts)
    assert is_even


def test_uneven_sampling_detected():
    ts = _make_timestamps(1000, 0.1, jitter=0.05)
    is_even, _ = check_even_sampling(ts)
    assert not is_even
