"""Tests for double integration and drift suppression."""

import numpy as np
import pytest

from well_analysis.signal.integration import integrate_acceleration, highpass_filter


FS = 25.0  # Hz
PUMP_FREQ = 0.0667  # Hz (~1 stroke per 15 s)


def _synthetic_accel(n: int = 5000, fs: float = FS) -> np.ndarray:
    """Sinusoidal acceleration mimicking a running pump jack."""
    t = np.arange(n) / fs
    gravity_offset = -9.81
    amplitude = 2.0  # m/s²
    return gravity_offset + amplitude * np.sin(2 * np.pi * PUMP_FREQ * t)


def test_highpass_removes_dc():
    x = np.ones(1000) * 5.0 + np.sin(2 * np.pi * 0.1 * np.arange(1000) / FS)
    filtered = highpass_filter(x, fs=FS, cutoff=0.02)
    assert abs(filtered.mean()) < 0.1


def test_integrate_returns_correct_shape():
    accel = _synthetic_accel()
    vel, pos = integrate_acceleration(accel, fs=FS)
    assert vel.shape == accel.shape
    assert pos.shape == accel.shape


def test_position_is_periodic():
    """Reconstructed position should oscillate at the pump frequency."""
    accel = _synthetic_accel(n=10000)
    _, pos = integrate_acceleration(accel, fs=FS, gravity_offset=-9.81)
    # After transient, position should have non-zero amplitude
    assert pos[1000:].std() > 0.01
