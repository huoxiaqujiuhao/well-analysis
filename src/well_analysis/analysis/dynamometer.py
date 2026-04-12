"""Dynamometer card extraction and characterisation.

A dynamometer card is the closed loop traced by (position, load) over one
pump stroke.  Extracting individual cycles requires:
  1. Identifying the pump frequency from the acceleration spectrum.
  2. Segmenting the position signal into individual strokes.
  3. Re-sampling each stroke to a common grid for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Pump frequency
# ---------------------------------------------------------------------------


def estimate_pump_frequency(accel: np.ndarray, fs: float) -> float:
    """Estimate pumping frequency via Welch PSD.

    Looks for the dominant peak in [0.03, 0.15] Hz (covers 0.05–0.1 Hz spec).
    """
    nperseg = min(len(accel), int(fs * 300))  # 5-minute segments
    freqs, psd = signal.welch(accel, fs=fs, nperseg=nperseg)
    mask = (freqs >= 0.03) & (freqs <= 0.15)
    peak_idx = np.argmax(psd[mask])
    return float(freqs[mask][peak_idx])


# ---------------------------------------------------------------------------
# Cycle segmentation
# ---------------------------------------------------------------------------


def segment_cycles(
    position: np.ndarray,
    fs: float,
    pump_freq: float,
    min_fraction: float = 0.7,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for individual pump cycles.

    Uses zero-crossings of the position signal around its median to find
    cycle boundaries (one crossing per upstroke start).

    Parameters
    ----------
    min_fraction:
        Discard cycles shorter than ``min_fraction`` of the expected period.
    """
    period_samples = int(fs / pump_freq)
    median_pos = np.median(position)
    centered = position - median_pos

    # Upward zero-crossings → start of upstroke
    crossings = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]

    min_len = int(min_fraction * period_samples)
    cycles = []
    for i in range(len(crossings) - 1):
        s, e = crossings[i], crossings[i + 1]
        if (e - s) >= min_len:
            cycles.append((int(s), int(e)))
    return cycles


# ---------------------------------------------------------------------------
# Card extraction
# ---------------------------------------------------------------------------


def extract_dynamometer_cards(
    position: np.ndarray,
    load: np.ndarray,
    fs: float,
    pump_freq: float,
    n_points: int = 200,
) -> list[dict]:
    """Extract one dynamometer card per pump cycle.

    Each card is re-sampled to ``n_points`` for uniform comparison.

    Returns
    -------
    list of dicts with keys: ``pos``, ``load``, ``start``, ``end``.
    """
    cycles = segment_cycles(position, fs, pump_freq)
    cards = []
    for s, e in cycles:
        pos_c = position[s:e]
        load_c = load[s:e]
        t_orig = np.linspace(0, 1, len(pos_c))
        t_new = np.linspace(0, 1, n_points)
        cards.append(
            dict(
                pos=np.interp(t_new, t_orig, pos_c),
                load=np.interp(t_new, t_orig, load_c),
                start=s,
                end=e,
            )
        )
    return cards


# ---------------------------------------------------------------------------
# Stroke amplitude
# ---------------------------------------------------------------------------


def estimate_stroke_amplitude(position: np.ndarray, q: float = 1.0) -> float:
    """Peak-to-peak stroke amplitude (metres), robust to outliers.

    Uses the ``q``–``(100-q)`` percentile range.
    """
    return float(np.percentile(position, 100 - q) - np.percentile(position, q))
