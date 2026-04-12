"""Double integration of acceleration to reconstruct position.

Key challenge: double integration accumulates low-frequency drift
(integration of numerical noise). Strategy:
  1. Remove gravity offset (sensor at rest ≈ -g + offset).
  2. High-pass filter before each integration step to suppress drift.
  3. Use scipy.integrate.cumulative_trapezoid for accuracy.
"""

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
    hp_cutoff: float = 0.02,
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
        Should be well below the pump frequency (~0.05–0.1 Hz) and above
        DC (e.g. 0.01–0.02 Hz works well).

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
