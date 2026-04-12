"""Well running-state detection and controller mode classification.

When running: polished rod oscillates periodically → high AC variance.
When stopped: acceleration ≈ constant (gravity + sensor offset) → low variance.

Efficient method: rolling RMS of the AC component (median-removed signal),
computed via a uniform moving average of squared values — O(n) with scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    """O(n) rolling RMS via uniform_filter1d on squared signal."""
    return np.sqrt(np.maximum(0.0, uniform_filter1d(x**2, size=window)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_well_state(
    accel: np.ndarray,
    fs: float,
    window_seconds: float = 30.0,
    threshold: float | None = None,
) -> np.ndarray:
    """Classify each sample as running (True) or stopped (False).

    Parameters
    ----------
    accel:
        Raw acceleration array.
    fs:
        Sampling frequency (Hz).
    window_seconds:
        Smoothing window for RMS — should be several pump periods.
        At 0.05 Hz one period is 20 s, so 30 s is a safe default.
    threshold:
        RMS threshold separating running from stopped.
        If None, estimated automatically via Otsu-like two-class split
        on the RMS histogram.

    Returns
    -------
    is_running : np.ndarray[bool]
    """
    window = max(1, int(window_seconds * fs))
    ac = accel - np.median(accel)
    rms = _rolling_rms(ac, window)

    if threshold is None:
        # Simple two-class threshold: midpoint between the two histogram modes
        counts, edges = np.histogram(rms, bins=200)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Find valley between stopped (low RMS) and running (high RMS) peaks
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(counts, distance=10)
        if len(peaks) >= 2:
            p1, p2 = peaks[0], peaks[-1]
            threshold = float(centers[p1:p2][np.argmin(counts[p1:p2])])
        else:
            threshold = float(np.percentile(rms, 50))

    return rms > threshold


def classify_controller_mode(
    timestamps: pd.Series,
    is_running: np.ndarray,
    shutdown_hours: float = 12.0,
) -> pd.DataFrame:
    """Classify each run/stop interval by controller mode.

    Modes
    -----
    - ``continuous``  : well ran without interruption for the whole segment.
    - ``timer``       : alternating on/off cycles (regular scheduling).
    - ``shutdown``    : off for > ``shutdown_hours`` (maintenance / workover).

    Returns
    -------
    pd.DataFrame with columns [start, end, state, mode, duration_h]
    """
    # Run-length encoding
    state = is_running.astype(int)
    change = np.diff(state, prepend=state[0] - 1)
    starts = np.where(change != 0)[0]
    ends = np.append(starts[1:], len(state))

    records = []
    for s, e in zip(starts, ends):
        ts_start = timestamps.iloc[s]
        ts_end = timestamps.iloc[e - 1]
        duration_h = (ts_end - ts_start).total_seconds() / 3600
        running = bool(state[s])
        records.append(
            dict(start=ts_start, end=ts_end, running=running, duration_h=duration_h)
        )

    segments = pd.DataFrame(records)

    # Classify mode
    def _mode(row: pd.Series) -> str:
        if not row["running"] and row["duration_h"] > shutdown_hours:
            return "shutdown"
        return "running" if row["running"] else "off"

    segments["mode"] = segments.apply(_mode, axis=1)

    # Upgrade isolated "running" segments surrounded by "off" segments to "timer"
    # (detect regular on/off alternation)
    modes = segments["mode"].values
    for i in range(1, len(modes) - 1):
        if modes[i] == "running" and modes[i - 1] == "off" and modes[i + 1] == "off":
            modes[i] = "timer_on"
            modes[i - 1] = "timer_off"
    segments["mode"] = modes

    return segments
