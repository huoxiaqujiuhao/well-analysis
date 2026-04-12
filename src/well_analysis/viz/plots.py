"""Reusable plotting helpers.

All functions return the Axes object so callers can further customise.
Use ``plt.show()`` or ``fig.savefig()`` externally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes

PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def plot_time_series(
    timestamps: pd.Series,
    values: np.ndarray | pd.Series,
    ylabel: str = "",
    title: str = "",
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))
    kwargs.setdefault("lw", 0.6)
    ax.plot(timestamps, values, **kwargs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    return ax


# ---------------------------------------------------------------------------
# Well state
# ---------------------------------------------------------------------------


def plot_well_state(
    timestamps: pd.Series,
    is_running: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Colour band: green = running, red = stopped."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 1.5))
    ax.fill_between(
        timestamps,
        is_running.astype(float),
        step="post",
        color="steelblue",
        alpha=0.7,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Off", "On"])
    ax.set_ylim(-0.1, 1.3)
    ax.set_title("Well State")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    return ax


# ---------------------------------------------------------------------------
# Dynamometer card
# ---------------------------------------------------------------------------


def plot_dynamometer_card(
    position: np.ndarray,
    load: np.ndarray,
    title: str = "Dynamometer Card",
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    kwargs.setdefault("lw", 0.8)
    ax.plot(position, load, **kwargs)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Load (N)")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Cluster scatter
# ---------------------------------------------------------------------------


def plot_cluster_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """2-D PCA scatter coloured by cluster label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for k in np.unique(labels):
        mask = labels == k
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=10,
            label=f"Cluster {k}",
            alpha=0.6,
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Operating Conditions (PCA projection)")
    ax.legend(markerscale=2)
    return ax
