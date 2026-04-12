"""Operating condition identification via unsupervised clustering.

Each dynamometer card is summarised into a feature vector.
KMeans (or HDBSCAN for variable cluster count) then groups cards
into operating regimes: full pump, partial fillage, gas interference,
valve failure, etc.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_card_features(cards: list[dict]) -> np.ndarray:
    """Compute a feature vector for each dynamometer card.

    Features (all physics-motivated):
      - max_load, min_load, load_range
      - mean_load
      - card_area (proxy for net work done per stroke — via shoelace formula)
      - upstroke_area, downstroke_area (asymmetry)
      - pos_range (stroke amplitude)
      - load_std

    Returns
    -------
    X : np.ndarray of shape (n_cards, n_features)
    """
    rows = []
    for c in cards:
        pos = c["pos"]
        load = c["load"]
        n = len(pos)
        half = n // 2

        # Shoelace area (signed → take abs)
        area = 0.5 * abs(
            np.dot(pos, np.roll(load, -1)) - np.dot(load, np.roll(pos, -1))
        )

        rows.append(
            [
                load.max(),
                load.min(),
                load.max() - load.min(),
                load.mean(),
                load.std(),
                area,
                load[:half].mean(),   # upstroke mean load
                load[half:].mean(),   # downstroke mean load
                pos.max() - pos.min(),
            ]
        )
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_operating_conditions(
    features: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """KMeans clustering of dynamometer card features.

    Parameters
    ----------
    features:
        Output of ``extract_card_features``.
    n_clusters:
        Number of operating regimes to identify.

    Returns
    -------
    labels : np.ndarray[int]
    kmeans : fitted KMeans
    scaler : fitted StandardScaler (needed to transform new data)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)
    return labels, kmeans, scaler


def reduce_for_viz(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Project features to 2D via PCA for scatter-plot visualisation."""
    X = scaler.transform(features)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)
