"""Helpers for Q9/Q10 regime analysis.

The goal is to keep notebooks focused on method + interpretation while moving
reusable calculations into the library layer.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder

from .clustering import (
    cluster_operating_conditions,
    extract_card_features,
    predict_next,
    transition_matrix,
)


def evaluate_kmeans_grid(
    features: np.ndarray,
    ks: list[int] | range,
    interpretable_ks: list[int] | None = None,
) -> dict:
    """Return silhouette/DB scores and the chosen interpretable k."""
    ks = list(ks)
    sil, db = [], []
    for k in ks:
        lab, _, _ = cluster_operating_conditions(features, n_clusters=k)
        sil.append(float(silhouette_score(features, lab)))
        db.append(float(davies_bouldin_score(features, lab)))
    if interpretable_ks is None:
        interpretable_ks = [k for k in ks if 3 <= k <= 6]
    sil_int = [sil[ks.index(k)] for k in interpretable_ks]
    best_k = int(interpretable_ks[int(np.argmax(sil_int))])
    return {"ks": ks, "silhouette": sil, "davies_bouldin": db, "best_k": best_k}


def medoid_index(
    features: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    scaler,
) -> int:
    """Index of the point closest to the scaled cluster mean."""
    xs = scaler.transform(features)
    mask = labels == target_label
    mean = xs[mask].mean(axis=0)
    d = np.linalg.norm(xs[mask] - mean, axis=1)
    return int(np.where(mask)[0][int(np.argmin(d))])


def summarize_cluster_feature_means(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Cluster sizes plus mean feature values in original units."""
    rows = []
    for k in sorted(set(labels)):
        rows.append([k, int((labels == k).sum()), *features[labels == k].mean(axis=0)])
    return pd.DataFrame(rows, columns=["cluster", "n_cards", *feature_names])


def build_trusted_cluster_frame(
    dm: pd.DataFrame,
    trusted_idx: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    labels_km: np.ndarray,
    labels_hd: np.ndarray,
) -> pd.DataFrame:
    """Per-trusted-card frame with features + cluster labels."""
    dm_tr = dm.iloc[trusted_idx].reset_index(drop=True).copy()
    for j, name in enumerate(feature_names):
        dm_tr[name] = features[:, j]
    dm_tr["cluster_km"] = labels_km
    dm_tr["cluster_hd"] = labels_hd
    return dm_tr


def validate_nb04_references_against_q9(
    all_cards: list[dict],
    dm: pd.DataFrame,
    trusted_mask: pd.Series | np.ndarray,
    trusted_idx: np.ndarray,
    trusted_cards: list[dict],
    features: np.ndarray,
    labels_km: np.ndarray,
    scaler_km,
    era_split: pd.Timestamp,
) -> dict:
    """Recreate nb04-style era references and compare them to Q9 medoids."""
    x_all = extract_card_features(all_cards)
    iso_ref = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso_ref.fit(features)
    if_all = -iso_ref.score_samples(x_all)

    t_all = pd.to_datetime([c["t"] for c in all_cards])
    era_id = np.where(t_all < era_split, 0, 1)
    trusted_lookup = pd.Index(np.asarray(trusted_idx))

    era_ref = {}
    core_mix = {}
    all_trusted_mix = {}
    trusted_mask = np.asarray(trusted_mask)
    for e in [0, 1]:
        in_era_all = (era_id == e) & trusted_mask
        p50_scr = np.percentile(if_all[in_era_all], 50)
        core_idx_all = np.where(in_era_all & (if_all <= p50_scr))[0]
        core_cards = [all_cards[i] for i in core_idx_all]

        pos_stk = np.stack([c["pos"] - np.median(c["pos"]) for c in core_cards])
        load_stk = np.stack([c["load"] - np.median(c["load"]) for c in core_cards])
        era_ref[e] = (np.median(pos_stk, axis=0), np.median(load_stk, axis=0))

        core_pos = trusted_lookup.get_indexer(core_idx_all)
        core_pos = core_pos[core_pos >= 0]
        era_pos = np.where((pd.to_datetime(dm.iloc[trusted_idx]["t"]) < era_split) if e == 0
                           else (pd.to_datetime(dm.iloc[trusted_idx]["t"]) >= era_split))[0]

        core_lab = labels_km[core_pos]
        era_lab = labels_km[era_pos]
        core_mix[e] = (
            pd.Series(core_lab).value_counts(normalize=True).sort_index().reindex([0, 1, 2], fill_value=0.0)
        )
        all_trusted_mix[e] = (
            pd.Series(era_lab).value_counts(normalize=True).sort_index().reindex([0, 1, 2], fill_value=0.0)
        )

    medoids = {}
    for k in sorted(set(labels_km)):
        idx = medoid_index(features, labels_km, int(k), scaler_km)
        c = trusted_cards[idx]
        medoids[int(k)] = {
            "pos": c["pos"] - np.median(c["pos"]),
            "load": c["load"] - np.median(c["load"]),
            "trusted_idx": int(idx),
        }

    all_pos = np.stack([era_ref[e][0] for e in [0, 1]] + [medoids[k]["pos"] for k in [0, 1, 2]])
    all_load = np.stack([era_ref[e][1] for e in [0, 1]] + [medoids[k]["load"] for k in [0, 1, 2]])
    pos_scale = float(all_pos.std())
    load_scale = float(all_load.std())

    def shape_l2(p1, l1, p2, l2):
        v1 = np.r_[p1 / pos_scale, l1 / load_scale]
        v2 = np.r_[p2 / pos_scale, l2 / load_scale]
        return float(np.linalg.norm(v1 - v2))

    dist_tbl = pd.DataFrame(
        index=["era 0 ref", "era 1 ref"],
        columns=["cluster 0", "cluster 1", "cluster 2"],
        dtype=float,
    )
    for e in [0, 1]:
        for k in [0, 1, 2]:
            dist_tbl.loc[f"era {e} ref", f"cluster {k}"] = shape_l2(
                era_ref[e][0], era_ref[e][1], medoids[k]["pos"], medoids[k]["load"]
            )
    nearest = dist_tbl.idxmin(axis=1)

    return {
        "if_all": if_all,
        "era_id": era_id,
        "era_ref": era_ref,
        "core_mix": core_mix,
        "all_trusted_mix": all_trusted_mix,
        "medoids": medoids,
        "dist_tbl": dist_tbl,
        "nearest": nearest,
    }


def episode_statistics(
    episodes: pd.DataFrame,
    n_cards_total: int,
) -> pd.DataFrame:
    """Episode summary table used in Q10."""
    stats = (
        episodes.groupby("regime")
        .agg(
            n_episodes=("length", "size"),
            cards_total=("length", "sum"),
            median_len=("length", "median"),
            p95_len=("length", lambda s: s.quantile(0.95)),
        )
        .sort_index()
    )
    stats["dwell_frac"] = stats["cards_total"] / n_cards_total
    return stats


def transition_entropy_bits(transmat: pd.DataFrame) -> pd.Series:
    """Per-row transition entropy in bits."""
    row_ent = pd.Series([entropy(r) for r in transmat.values], index=transmat.index)
    return row_ent / np.log(2)


def regime_mix_diagnostics(
    dm_tr: pd.DataFrame,
    timer_on: pd.DataFrame,
) -> dict:
    """Timer-boundary, hour-of-day, and day-of-record mixes."""
    out = dm_tr.copy()
    t_ns = out["t"].astype("int64").to_numpy()
    sec_since_start = np.full(len(out), np.nan)
    for _, row in timer_on.iterrows():
        s_ns = pd.Timestamp(row["start"]).value
        e_ns = pd.Timestamp(row["end"]).value
        m = (t_ns >= s_ns) & (t_ns <= e_ns)
        sec_since_start[m] = (t_ns[m] - s_ns) / 1e9
    out["sec_since_timer_start"] = sec_since_start
    out["hour"] = out["t"].dt.hour
    out["day"] = out["t"].dt.floor("D")

    early_mask = out["sec_since_timer_start"] <= 300
    early_mix = out.loc[early_mask, "cluster_km"].value_counts(normalize=True).sort_index()
    late_mix = out.loc[~early_mask, "cluster_km"].value_counts(normalize=True).sort_index()
    mix = pd.concat([early_mix.rename("first 5 min"), late_mix.rename("after 5 min")], axis=1).fillna(0)

    hour_mix = pd.crosstab(out["hour"], out["cluster_km"], normalize="index")
    day_mix = pd.crosstab(out["day"], out["cluster_km"], normalize="index")
    day_mix.index = [d.strftime("%m-%d") for d in day_mix.index]

    return {"dm_tr": out, "mix": mix, "hour_mix": hour_mix, "day_mix": day_mix}


def evaluate_bonus_prediction(
    episodes: pd.DataFrame,
    split_quantile: float = 0.75,
    random_state: int = 42,
) -> dict:
    """Era-aware bonus prediction benchmark for Q10."""
    era_split = pd.Timestamp("2020-12-31 23:59:59", tz=episodes["start"].dt.tz)
    ep_era1 = episodes[episodes["start"] > era_split].reset_index(drop=True)
    split_ts = pd.Timestamp(ep_era1["start"].quantile(split_quantile)).floor("h")
    train = ep_era1[ep_era1["start"] < split_ts].reset_index(drop=True)
    test = ep_era1[ep_era1["start"] >= split_ts].reset_index(drop=True)

    regs = sorted(ep_era1["regime"].unique())
    p1 = transition_matrix(train)

    seq_train = train["regime"].astype(int).to_numpy()
    counts2 = {}
    for i in range(2, len(seq_train)):
        key = (int(seq_train[i - 2]), int(seq_train[i - 1]))
        counts2.setdefault(key, Counter())[int(seq_train[i])] += 1
    cov = sum(1 for v in counts2.values() if sum(v.values()) >= 3)

    ctx = pd.concat([train.tail(2), test]).reset_index(drop=True)
    truths, p_uni, p_marg, p_m1, p_m2 = [], [], [], [], []
    marg_pred = int(train["regime"].mode().iloc[0])
    rng = np.random.default_rng(random_state)

    for i in range(2, len(ctx)):
        prev_prev = int(ctx["regime"].iloc[i - 2])
        prev = int(ctx["regime"].iloc[i - 1])
        truth = int(ctx["regime"].iloc[i])
        if ctx["start"].iloc[i] < split_ts:
            continue

        others = [r for r in regs if r != prev]
        p_uni.append(int(rng.choice(others)))
        p_marg.append(marg_pred)
        p_m1.append(int(predict_next(p1, prev)))

        key = (prev_prev, prev)
        if key in counts2 and sum(counts2[key].values()) >= 3:
            p_m2.append(int(counts2[key].most_common(1)[0][0]))
        else:
            p_m2.append(int(predict_next(p1, prev)))
        truths.append(truth)

    truths = np.array(truths)
    results = pd.DataFrame(
        {
            "predictor": [
                "uniform (non-self)",
                "marginal",
                "1st-Markov",
                "2nd-Markov (w/ fallback)",
            ],
            "accuracy": [
                np.mean(np.array(p_uni) == truths),
                np.mean(np.array(p_marg) == truths),
                np.mean(np.array(p_m1) == truths),
                np.mean(np.array(p_m2) == truths),
            ],
            "n_test": [len(truths)] * 4,
        }
    )

    best_idx = int(results["accuracy"].idxmax())
    best_name = results["predictor"].iloc[best_idx]
    best_preds = np.array([p_uni, p_marg, p_m1, p_m2][best_idx])
    conf = (
        pd.crosstab(
            pd.Series(truths, name="truth"),
            pd.Series(best_preds, name="predicted"),
            normalize="index",
        )
        .reindex(index=regs, columns=regs, fill_value=0)
    )

    # Logistic-regression bonus baseline
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        categories=[regs, regs],
    )

    def build_xy(seq_df):
        rows, ys = [], []
        r = seq_df["regime"].astype(int).to_numpy()
        lengths = np.log1p(seq_df["length"].to_numpy(dtype=float))
        for i in range(2, len(seq_df)):
            rows.append([r[i - 2], r[i - 1], lengths[i - 2], lengths[i - 1]])
            ys.append(r[i])
        arr = np.array(rows, dtype=float)
        if len(arr) == 0:
            return np.zeros((0, 2 * len(regs) + 2)), np.array(ys)
        pair = ohe.transform(arr[:, :2].astype(int))
        durs = arr[:, 2:].reshape(-1, 2)
        return np.concatenate([pair, durs], axis=1), np.array(ys)

    ohe.fit(np.array([[r, r] for r in regs]).reshape(-1, 2))
    xtr, ytr = build_xy(train)
    ctx_feat = pd.concat([train.tail(2), test]).reset_index(drop=True)
    xte_all, yte_all = build_xy(ctx_feat)
    test_mask = (
        pd.to_datetime(ctx_feat["start"].iloc[2:]).astype("int64").to_numpy()
        >= pd.Timestamp(split_ts).value
    )
    xte, yte = xte_all[test_mask], yte_all[test_mask]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xtr, ytr)
    p_lr = clf.predict(xte)
    acc_lr = float((p_lr == yte).mean())

    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [
                    {
                        "predictor": "LogReg (prev-pair + durations)",
                        "accuracy": acc_lr,
                        "n_test": len(yte),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    return {
        "era_split": era_split,
        "ep_era1": ep_era1,
        "split_ts": split_ts,
        "train": train,
        "test": test,
        "p1": p1,
        "counts2": counts2,
        "coverage": cov,
        "truths": truths,
        "results": results,
        "best_name": best_name,
        "best_confusion": conf,
        "acc_lr": acc_lr,
    }
