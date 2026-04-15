"""Operating condition identification via unsupervised clustering.

Each dynamometer card is summarised into a feature vector.  KMeans gives
fixed-k interpretable centroids; HDBSCAN finds variable-k clusters and an
explicit noise label.  Downstream helpers turn a cluster-label time series
into episodes, a transition matrix, and a 1st-order Markov predictor for
Q10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * cross.sum()
    if abs(A) < 1e-9:
        return float(np.mean(x)), float(np.mean(y))
    Cx = ((x + x1) * cross).sum() / (6 * A)
    Cy = ((y + y1) * cross).sum() / (6 * A)
    return float(Cx), float(Cy)


FEATURE_NAMES = [
    "area",
    "p10_load",
    "p90_load",
    "asym_tilt",
    "upstroke_heaviness",
    "load_median",
]


def extract_card_features(cards: list[dict]) -> np.ndarray:
    """Feature matrix (n_cards, 6) for clustering.

    Features (dropped from earlier ad-hoc version for redundancy):
      - area — net work per stroke (shoelace)
      - p10_load / p90_load — lower / upper plateau
      - asym_tilt — polygon-centroid horizontal offset / stroke
      - upstroke_heaviness — (mean load upstroke − mean load downstroke)
        / (sum), split at position apex.  Temporal asymmetry complement
        to the geometric asym_tilt.
      - load_median — baseline load; catches system-wide shifts that
        preserve card shape (nb04 Family B signature).

    ``stroke`` and ``load_range`` are intentionally excluded: stroke is
    near-constant inside the trusted subset (StandardScaler would inflate
    its tiny variance), and load_range is colinear with p90 − p10.
    """
    rows = []
    for c in cards:
        p = c["pos"]
        l = c["load"]
        # Shoelace area
        area = 0.5 * abs(np.dot(p, np.roll(l, -1)) - np.dot(l, np.roll(p, -1)))

        p10 = float(np.percentile(l, 10))
        p90 = float(np.percentile(l, 90))
        lmed = float(np.percentile(l, 50))

        stroke = float(np.percentile(p, 99) - np.percentile(p, 1))
        Cx, _ = _polygon_centroid(p, l)
        p_mid = 0.5 * (p.max() + p.min())
        asym = (Cx - p_mid) / stroke if stroke > 0 else 0.0

        apex = int(np.argmax(p))
        if apex < 2 or apex > len(p) - 3:
            up_heavy = 0.0
        else:
            up_mean = float(l[: apex + 1].mean())
            dn_mean = float(l[apex:].mean())
            denom = abs(up_mean) + abs(dn_mean) + 1e-9
            up_heavy = (up_mean - dn_mean) / denom

        rows.append([area, p10, p90, asym, up_heavy, lmed])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Clustering algorithms
# ---------------------------------------------------------------------------


def cluster_operating_conditions(
    features: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """KMeans on standardised features."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km, scaler


def cluster_with_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> tuple[np.ndarray, HDBSCAN, StandardScaler]:
    """HDBSCAN on standardised features.  Noise points receive label -1."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    h = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = h.fit_predict(X)
    return labels, h, scaler


def reduce_for_viz(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """PCA → 2D using an already-fit scaler."""
    X = scaler.transform(features)
    return PCA(n_components=2, random_state=42).fit_transform(X)


# ---------------------------------------------------------------------------
# Q10: episodes, transitions, Markov
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    regime: int
    start: pd.Timestamp
    end: pd.Timestamp
    length: int   # number of cards in the episode


def episodes_from_labels(
    labels: np.ndarray,
    timestamps: pd.Series,
    min_len: int = 5,
) -> pd.DataFrame:
    """Collapse a card-level label sequence into episode rows.

    An episode is a maximal run of identical labels of length ≥ ``min_len``.
    Runs shorter than ``min_len`` are still returned but flagged via
    ``length``; callers typically filter.
    """
    if len(labels) == 0:
        return pd.DataFrame(columns=["regime", "start", "end", "length"])

    labels = np.asarray(labels)
    ts = pd.Series(timestamps).reset_index(drop=True)
    change = np.flatnonzero(np.diff(labels, prepend=labels[0] - 1) != 0)
    bounds = np.r_[change, len(labels)]
    rows = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1] - 1
        rows.append(
            dict(
                regime=int(labels[s]),
                start=ts.iloc[s],
                end=ts.iloc[e],
                length=int(e - s + 1),
            )
        )
    ep = pd.DataFrame(rows)
    return ep[ep["length"] >= min_len].reset_index(drop=True)


def transition_matrix(episodes: pd.DataFrame) -> pd.DataFrame:
    """Empirical P(next | prev) at the episode level.  Rows sum to 1."""
    if len(episodes) < 2:
        return pd.DataFrame()
    regs = sorted(episodes["regime"].unique())
    n = len(regs)
    idx = {r: i for i, r in enumerate(regs)}
    C = np.zeros((n, n), dtype=float)
    prev = episodes["regime"].values[:-1]
    nxt = episodes["regime"].values[1:]
    for a, b in zip(prev, nxt):
        C[idx[a], idx[b]] += 1
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = C / row_sums
    return pd.DataFrame(P, index=regs, columns=regs)


def fit_markov_1st_order(
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Alias for `transition_matrix` — kept for naming clarity in Q10."""
    return transition_matrix(episodes)


def predict_next(transmat: pd.DataFrame, last_regime: int) -> int:
    """Greedy argmax next-regime prediction."""
    if last_regime not in transmat.index:
        # Fall back to marginal most-frequent row destination
        return int(transmat.sum(axis=0).idxmax())
    row = transmat.loc[last_regime]
    return int(row.idxmax())
