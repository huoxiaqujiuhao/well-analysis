"""Tests for well state detection."""

import numpy as np
import pandas as pd

from well_analysis.detection.well_state import detect_well_state, classify_controller_mode


FS = 25.0
PUMP_FREQ = 0.0667


def _make_signal(pattern: list[tuple[str, int]]) -> np.ndarray:
    """Build acceleration signal with on/off blocks.

    pattern: list of ('on'|'off', n_samples)
    """
    segments = []
    for state, n in pattern:
        t = np.arange(n) / FS
        if state == "on":
            seg = -9.81 + 2.0 * np.sin(2 * np.pi * PUMP_FREQ * t)
        else:
            seg = np.full(n, -9.81)
        segments.append(seg)
    return np.concatenate(segments)


def test_detect_all_running():
    accel = _make_signal([("on", 5000)])
    state = detect_well_state(accel, fs=FS)
    # Allow short edge transients
    assert state[500:-500].mean() > 0.9


def test_detect_all_stopped():
    accel = _make_signal([("off", 5000)])
    state = detect_well_state(accel, fs=FS)
    assert state[500:-500].mean() < 0.1


def test_detect_mixed():
    accel = _make_signal([("on", 3000), ("off", 3000), ("on", 3000)])
    state = detect_well_state(accel, fs=FS)
    assert state[500:2500].mean() > 0.8
    assert state[3500:5500].mean() < 0.2
    assert state[6500:8500].mean() > 0.8
