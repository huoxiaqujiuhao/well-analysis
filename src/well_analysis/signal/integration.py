"""Double integration of acceleration to reconstruct position.

Key challenge: double integration accumulates low-frequency drift
(integration of numerical noise). Strategy:
  1. Remove gravity offset (sensor at rest ≈ -g + offset).
  2. High-pass filter before each integration step to suppress drift.
  3. Use scipy.integrate.cumulative_trapezoid for accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, signal


def highpass_filter(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter."""
    sos = signal.butter(4, cutoff, btype="high", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def estimate_gravity_offset(accel: np.ndarray, is_running: np.ndarray) -> float:
    """Estimate gravity + sensor offset from stopped intervals."""
    stopped = accel[~is_running]
    if stopped.size == 0:
        return float(np.median(accel))
    return float(np.median(stopped))


def integrate_acceleration(
    accel: np.ndarray,
    fs: float,
    gravity_offset: float | None = None,
    hp_cutoff: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct velocity and position from acceleration.

    Parameters
    ----------
    accel:
        Raw acceleration array (m/s²).
    fs:
        Sampling frequency (Hz).
    gravity_offset:
        Value subtracted to remove gravity + sensor offset.
        If None, uses the signal median (assumes mostly stopped or symmetric).
    hp_cutoff:
        High-pass cutoff frequency (Hz) applied after each integration.
        Must be well below the pump frequency (~0.05–0.1 Hz) and above DC.
        Default 0.04 Hz gives strong drift rejection for pumps ≥ 0.08 Hz;
        use 0.02 if the pump frequency approaches 0.05 Hz.

    Returns
    -------
    velocity : np.ndarray (m/s)
    position : np.ndarray (m)
    """
    if gravity_offset is None:
        gravity_offset = float(np.median(accel))

    a_net = accel - gravity_offset

    # Filter before first integration to remove residual DC
    a_filt = highpass_filter(a_net, fs, cutoff=hp_cutoff)

    # Integrate: acceleration -> velocity
    velocity = integrate.cumulative_trapezoid(a_filt, dx=1.0 / fs, initial=0.0)
    velocity = highpass_filter(velocity, fs, cutoff=hp_cutoff)

    # Integrate: velocity -> position
    position = integrate.cumulative_trapezoid(velocity, dx=1.0 / fs, initial=0.0)
    position = highpass_filter(position, fs, cutoff=hp_cutoff)

    return velocity, position


def transient_mask(
    is_running: np.ndarray,
    fs: float,
    hp_cutoff: float = 0.04,
    n_periods: float = 1.5,
) -> np.ndarray:
    """Mark samples near running-segment boundaries as unreliable.

    At every stop↔run transition, the high-pass output briefly carries
    contamination from the neighbouring segment's DC/drift. The width of
    the unreliable region scales as 1/hp_cutoff.

    Returns
    -------
    np.ndarray[bool] — True where the sample is TRUSTED.
    """
    n_samples = int(n_periods / hp_cutoff * fs)
    mask = np.ones(len(is_running), dtype=bool)
    diffs = np.diff(is_running.astype(np.int8), prepend=0, append=0)
    starts = np.where(diffs > 0)[0]
    ends = np.where(diffs < 0)[0]
    for s in starts:
        mask[s : min(s + n_samples, len(mask))] = False
    for e in ends:
        mask[max(0, e - n_samples) : e] = False
    return mask
