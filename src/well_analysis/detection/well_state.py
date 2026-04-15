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


def _otsu_threshold(x: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Return (between-class variance, threshold) by 1-D Otsu."""
    counts, edges = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_T = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s2b = (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega))
    s2b = np.nan_to_num(s2b)
    k = int(np.argmax(s2b))
    return float(s2b[k]), float(centers[k])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_well_state(
    accel: np.ndarray,
    fs: float,
    window_seconds: float = 30.0,
    threshold: float | None = None,
    chunk_hours: float | None = None,
    min_otsu_ratio: float = 0.3,
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
        If None, estimated automatically via Otsu (global or per-chunk).
    chunk_hours:
        If given, compute a separate Otsu threshold within each chunk of
        this duration (e.g. 24 h). A chunk falls back to the global
        threshold when its bimodality is weak — see ``min_otsu_ratio``.
    min_otsu_ratio:
        For chunked mode: a chunk's local Otsu between-class variance must
        be at least ``min_otsu_ratio`` × the global between-class variance,
        otherwise the chunk is judged unimodal (e.g. all-stopped during a
        shutdown day) and the global threshold is used instead.

    Returns
    -------
    is_running : np.ndarray[bool]
    """
    window = max(1, int(window_seconds * fs))
    ac = accel - np.median(accel)
    rms = _rolling_rms(ac, window)

    s2b_global, thr_global = _otsu_threshold(rms)
    if threshold is not None:
        thr_global = float(threshold)

    if chunk_hours is None:
        return rms > thr_global

    is_running = np.zeros_like(rms, dtype=bool)
    chunk_n = max(1, int(chunk_hours * 3600 * fs))
    for start in range(0, len(rms), chunk_n):
        end = min(start + chunk_n, len(rms))
        chunk = rms[start:end]
        if chunk.std() < 1e-9:
            thr = thr_global
        else:
            s2b_local, thr_local = _otsu_threshold(chunk, bins=128)
            ratio = s2b_local / max(s2b_global, 1e-12)
            thr = thr_local if ratio >= min_otsu_ratio else thr_global
        is_running[start:end] = chunk > thr
    return is_running


def cluster_segments_by_duration(
    segments: pd.DataFrame, log_gap_threshold: float = 0.4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster segments by (running, log10(duration_s)) using 1-D single-link
    hierarchical clustering: split where consecutive sorted log-durations are
    farther than ``log_gap_threshold`` apart.

    A threshold of 0.4 ⇔ a factor of 10**0.4 ≈ 2.5 in linear duration. The data
    typically shows tight clusters (≤ 0.05 spread) separated by gaps > 0.5,
    so the result is insensitive to the exact threshold in [0.3, 0.7].

    Returns
    -------
    (segments_with_cluster, summary)
        ``segments_with_cluster`` is the input frame with two extra columns:
            - ``cluster`` (int)
            - ``cluster_label`` (str): one of timer_on / timer_off / shutdown
              / noise_run / noise_off / other_run / other_off
        ``summary`` describes each cluster (n, median_dur_s, label).
    """
    out = segments.copy().reset_index(drop=True)
    cluster_ids = np.full(len(out), -1, dtype=int)
    next_id = 0
    summary_rows = []

    for run_val in [True, False]:
        idx = np.where(out["running"].values == run_val)[0]
        if len(idx) == 0:
            continue
        durs_s = out.loc[idx, "duration_h"].values * 3600
        log_d = np.log10(durs_s)
        order = np.argsort(log_d)
        gaps = np.diff(log_d[order])
        sorted_ids = np.zeros(len(idx), dtype=int)
        for b in np.where(gaps > log_gap_threshold)[0]:
            sorted_ids[b + 1 :] += 1
        local_ids = np.empty_like(sorted_ids)
        local_ids[order] = sorted_ids
        for c in np.unique(local_ids):
            mask = local_ids == c
            cluster_ids[idx[mask]] = next_id + c
            summary_rows.append(
                dict(
                    cluster=next_id + c,
                    running=run_val,
                    n=int(mask.sum()),
                    median_dur_s=float(np.median(durs_s[mask])),
                    min_dur_s=float(durs_s[mask].min()),
                    max_dur_s=float(durs_s[mask].max()),
                )
            )
        next_id += int(local_ids.max()) + 1

    summary = pd.DataFrame(summary_rows)

    def _label(row: pd.Series) -> str:
        dur = row["median_dur_s"]
        running = row["running"]
        if dur < 60:
            return "noise_run" if running else "noise_off"
        if running:
            return "timer_on" if row["n"] >= 10 else "other_run"
        if dur > 12 * 3600:
            return "shutdown"
        return "timer_off" if row["n"] >= 10 else "other_off"

    summary["label"] = summary.apply(_label, axis=1)
    label_map = dict(zip(summary["cluster"], summary["label"]))
    out["cluster"] = cluster_ids
    out["cluster_label"] = out["cluster"].map(label_map)
    return out, summary


def validate_timer_regularity(seg_clustered: pd.DataFrame) -> dict:
    """Quantify timer cycle regularity using the cluster labels.

    A *valid timer cycle* is a triple (timer_on → timer_off → timer_on) of
    consecutive segments. The cycle length is the time between the two
    timer_on starts. CV is computed over those cycle lengths only — no
    outlier filtering, no thresholds: invalid sequences (e.g. spanning a
    shutdown) simply do not contribute.

    Returns
    -------
    dict with keys:
        n_segments_timer_on, n_valid_cycles, mean_cycle_min, std_cycle_min,
        cv, is_true_timer (bool: cv < 0.05).
    """
    if "cluster_label" not in seg_clustered.columns:
        raise ValueError(
            "Input must come from cluster_segments_by_duration() (no cluster_label)."
        )
    seg = seg_clustered.sort_values("start").reset_index(drop=True)
    labels = seg["cluster_label"].values
    starts = seg["start"].values

    cycles = []
    for i in range(len(seg) - 2):
        if (
            labels[i] == "timer_on"
            and labels[i + 1] == "timer_off"
            and labels[i + 2] == "timer_on"
        ):
            cycles.append(
                (pd.Timestamp(starts[i + 2]) - pd.Timestamp(starts[i])).total_seconds()
                / 60
            )

    n_timer_on = int((labels == "timer_on").sum())
    if len(cycles) < 2:
        return {
            "n_segments_timer_on": n_timer_on,
            "n_valid_cycles": len(cycles),
            "is_true_timer": False,
        }
    cycles = np.asarray(cycles)
    cv = float(cycles.std() / cycles.mean())
    return {
        "n_segments_timer_on": n_timer_on,
        "n_valid_cycles": len(cycles),
        "mean_cycle_min": float(cycles.mean()),
        "std_cycle_min": float(cycles.std()),
        "cv": cv,
        "is_true_timer": cv < 0.05,
    }


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
