"""Q8 helpers for anomaly-oriented dynamometer-card analysis.

These helpers keep notebooks short while preserving the main analysis logic:
  1. Extract cards and attach interpretable shape metrics.
  2. Build a trusted subset for robust anomaly scoring.
  3. Score all cards, then promote one representative candidate per round.
  4. Add time-structure metadata (within-round worsening, cross-round repeats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import IsolationForest

from .clustering import extract_card_features
from .dynamometer import extract_dynamometer_cards


def card_area(pos: np.ndarray, load: np.ndarray) -> float:
    """Polygon area via the shoelace formula."""
    return float(
        0.5
        * abs(np.sum(pos * np.roll(load, -1) - np.roll(pos, -1) * load))
    )


def polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Centroid of a closed polygonal loop."""
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-9:
        return float(np.mean(x)), float(np.mean(y))
    cx = ((x + x1) * cross).sum() / (6 * area)
    cy = ((y + y1) * cross).sum() / (6 * area)
    return float(cx), float(cy)


def card_metrics(card: dict) -> dict:
    """Interpretable per-card metrics used in Q8 and Q9."""
    pos = card["pos"]
    load = card["load"]
    stroke = float(np.percentile(pos, 99) - np.percentile(pos, 1))
    cx, _ = polygon_centroid(pos, load)
    p_mid = 0.5 * (pos.max() + pos.min())
    return {
        "area": card_area(pos, load),
        "stroke": stroke,
        "p10_load": float(np.percentile(load, 10)),
        "p50_load": float(np.percentile(load, 50)),
        "p90_load": float(np.percentile(load, 90)),
        "gap_frac": float(abs(pos[0] - pos[-1]) / stroke) if stroke > 0 else 0.0,
        "asym_tilt": (cx - p_mid) / stroke if stroke > 0 else 0.0,
    }


def extract_cards_with_metrics(
    timer_on: pd.DataFrame,
    timestamps_ns: np.ndarray,
    position: np.ndarray,
    load: np.ndarray,
    fs: float,
    pump_freq: float,
    valid_mask: np.ndarray | None = None,
    use_card_timestamp: bool = False,
) -> list[dict]:
    """Extract every card from every timer_on segment and attach metrics."""
    all_cards: list[dict] = []
    min_seg_len = int(fs / pump_freq) * 3
    for _, row in timer_on.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        i0 = np.searchsorted(timestamps_ns, np.datetime64(start.tz_convert(None)))
        i1 = np.searchsorted(timestamps_ns, np.datetime64(end.tz_convert(None)))
        if i1 - i0 < min_seg_len:
            continue
        cards = extract_dynamometer_cards(
            position[i0:i1],
            load[i0:i1],
            fs=fs,
            pump_freq=pump_freq,
            valid_mask=None if valid_mask is None else valid_mask[i0:i1],
        )
        for card in cards:
            card.update(card_metrics(card))
            if use_card_timestamp:
                card["t"] = pd.Timestamp(row["start"]) + pd.Timedelta(
                    seconds=float(card["start"]) / fs
                )
            else:
                card["t"] = row["start"]
            all_cards.append(card)
    return all_cards


def build_card_metric_frame(all_cards: list[dict]) -> pd.DataFrame:
    """Tabular metric view of the extracted cards."""
    cols = (
        "t",
        "area",
        "stroke",
        "p10_load",
        "p50_load",
        "p90_load",
        "gap_frac",
        "asym_tilt",
    )
    return pd.DataFrame([{k: c[k] for k in cols} for c in all_cards])


def select_segment_median_area_cards(
    all_cards: list[dict],
    dm: pd.DataFrame,
    segment_starts: pd.Series,
) -> list[tuple[pd.Timestamp, dict]]:
    """One representative median-area card per timer_on segment."""
    segs_ts = pd.Series([c["t"] for c in all_cards])
    daily_cards: list[tuple[pd.Timestamp, dict]] = []
    for t_seg in segment_starts:
        idx = np.where(segs_ts == t_seg)[0]
        if len(idx) == 0:
            continue
        areas = dm["area"].iloc[idx].to_numpy()
        mid = int(np.argsort(areas)[len(areas) // 2])
        daily_cards.append((t_seg, all_cards[idx[mid]]))
    return daily_cards


def build_trusted_subset(
    dm: pd.DataFrame,
    stroke_tol: float = 0.10,
    gap_quantile: float = 0.95,
) -> tuple[pd.Series, float, np.ndarray, np.ndarray]:
    """Return stroke deviation, gap threshold, trusted mask, trusted indices."""
    stroke_dev = (dm["stroke"] - dm["stroke"].median()).abs() / dm["stroke"].median()
    gap_thresh = float(dm["gap_frac"].quantile(gap_quantile))
    trusted_mask = ((stroke_dev < stroke_tol) & (dm["gap_frac"] < gap_thresh)).to_numpy()
    trusted_idx = np.where(trusted_mask)[0]
    return stroke_dev, gap_thresh, trusted_mask, trusted_idx


def fit_global_anomaly_scores(
    trusted_cards: list[dict],
    all_cards: list[dict],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit Isolation Forest on trusted cards and cross-check with MAD distance."""
    x_trust = extract_card_features(trusted_cards)
    x_all = extract_card_features(all_cards)
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
    )
    iso.fit(x_trust)
    if_all = -iso.score_samples(x_all)

    med_t = np.median(x_trust, axis=0)
    mad_t = np.median(np.abs(x_trust - med_t), axis=0) * 1.4826 + 1e-9
    z_all = (x_all - med_t) / mad_t
    mad_all = np.linalg.norm(z_all, axis=1)
    rho, _ = spearmanr(if_all, mad_all)
    return x_trust, x_all, if_all, z_all, float(rho)


def build_era_references(
    all_cards: list[dict],
    trusted_mask: np.ndarray,
    if_all: np.ndarray,
    era_split: pd.Timestamp,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], pd.DatetimeIndex]:
    """Build era labels plus a pointwise-median reference per era."""
    card_t = pd.to_datetime([c["t"] for c in all_cards])
    era_id = np.where(card_t < era_split, 0, 1)
    era_ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for era in [0, 1]:
        in_era = (era_id == era) & trusted_mask
        p50_scr = np.percentile(if_all[in_era], 50)
        core = np.where(in_era & (if_all <= p50_scr))[0]
        pos_stk = np.stack(
            [all_cards[i]["pos"] - np.median(all_cards[i]["pos"]) for i in core]
        )
        load_stk = np.stack(
            [all_cards[i]["load"] - np.median(all_cards[i]["load"]) for i in core]
        )
        era_ref[era] = (np.median(pos_stk, axis=0), np.median(load_stk, axis=0))
    return era_id, era_ref, card_t


def build_round_candidate_frame(
    all_cards: list[dict],
    era_id: np.ndarray,
    if_all: np.ndarray,
    gap_frac: np.ndarray,
    trusted_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-card frame and one argmax candidate per round."""
    round_t_ns = np.array([pd.Timestamp(c["t"]).value for c in all_cards], dtype="int64")
    df_all = pd.DataFrame(
        {
            "all_idx": np.arange(len(all_cards)),
            "round_t": round_t_ns,
            "era": era_id,
            "if": if_all,
            "gap_frac": gap_frac,
            "trusted": trusted_mask,
            "c_start": [c["start"] for c in all_cards],
        }
    )
    candidates = df_all.loc[df_all.groupby("round_t")["if"].idxmax()].reset_index(
        drop=True
    )
    return df_all, candidates


def select_gallery_candidates(
    candidates: pd.DataFrame,
    if_all: np.ndarray,
    threshold_q: float = 0.99,
) -> tuple[pd.DataFrame, float]:
    """Global quantile cut on per-round argmax candidates."""
    threshold = float(np.quantile(if_all, threshold_q))
    gallery = (
        candidates[candidates["if"] >= threshold]
        .sort_values("if", ascending=False)
        .reset_index(drop=True)
    )
    return gallery, threshold


def compute_round_taus(df_all: pd.DataFrame, min_cards: int = 10) -> dict[int, float]:
    """Kendall tau of anomaly score vs within-round card order."""
    round_tau: dict[int, float] = {}
    for round_t, grp in df_all.groupby("round_t"):
        if len(grp) < min_cards:
            round_tau[round_t] = np.nan
            continue
        ordered = grp.sort_values("c_start")
        tau, _ = kendalltau(np.arange(len(ordered)), ordered["if"].to_numpy())
        round_tau[round_t] = float(tau)
    return round_tau


def cluster_candidate_shapes(
    candidates: pd.DataFrame,
    all_cards: list[dict],
    distance_quantile: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Shape clustering on per-round argmax cards using channel-normalised L2."""
    cand_pos = np.stack(
        [all_cards[i]["pos"] - np.median(all_cards[i]["pos"]) for i in candidates["all_idx"]]
    )
    cand_load = np.stack(
        [all_cards[i]["load"] - np.median(all_cards[i]["load"]) for i in candidates["all_idx"]]
    )
    pos_scale = float(cand_pos.std())
    load_scale = float(cand_load.std())
    cand_shape = np.concatenate([cand_pos / pos_scale, cand_load / load_scale], axis=1)
    dists = pdist(cand_shape)
    dist_thr = float(np.quantile(dists, distance_quantile))
    z_link = linkage(dists, method="average")
    cand_cluster = fcluster(z_link, t=dist_thr, criterion="distance")

    out = candidates.copy()
    out["shape_cluster"] = cand_cluster
    cluster_nr = out.groupby("shape_cluster")["round_t"].nunique().to_dict()
    out["n_rounds_in_cluster"] = out["shape_cluster"].map(cluster_nr)
    return out, dist_thr


def dominant_feature_info(
    z_all: np.ndarray,
    indices: pd.Series | np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Return dominant robust-z feature for each requested row index."""
    dom = []
    for idx in indices:
        z = z_all[int(idx)]
        j = int(np.argmax(np.abs(z)))
        dom.append((feature_names[j], float(z[j])))
    return dom
