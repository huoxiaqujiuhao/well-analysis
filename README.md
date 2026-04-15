# well-analysis

Sucker-rod pump-jack surface-signal analysis on a 21-day public dataset.
Ten case-study questions (Q1–Q10) answered across five Jupyter notebooks
plus a small installable Python package.

This README is a **map**, optimised for an AI reader who wants to locate
the relevant code quickly. It does not explain the physics or the
numerical results — those live inside each notebook's narrative cells.

---

## Repository layout

```
well-analysis/
├── pyproject.toml                  # editable-install package spec
├── src/well_analysis/              # reusable library code
│   ├── data/loader.py              # raw CSV → DataFrame
│   ├── signal/
│   │   ├── sampling.py             # sample-rate checks, even-sampling QC (Q1)
│   │   └── integration.py          # accel → velocity → position (Q5/Q6)
│   ├── detection/well_state.py     # running/stopped, controller mode (Q2–Q4)
│   ├── analysis/
│   │   ├── dynamometer.py          # card extraction, per-card metrics (Q7/Q8)
│   │   ├── clustering.py           # card features + KMeans/HDBSCAN/episodes (Q9/Q10)
│   │   ├── q8.py                   # legacy family-based anomaly rules (superseded)
│   │   └── q9.py                   # (thin wrapper)
│   └── viz/plots.py                # shared plotting helpers
├── tests/                          # pytest — sampling, integration, detection
└── notebooks/
    ├── 01_data_exploration.ipynb          # Q1
    ├── 02_well_state_detection.ipynb      # Q2, Q3, Q4
    ├── 03_position_reconstruction.ipynb   # Q5, Q6
    ├── 04_dynamometer_card.ipynb          # Q7, Q8
    ├── 05_operating_conditions.ipynb      # Q9, Q10
    └── _generated/                        # mp4 animations produced by nb04 §9
```

---

## Question → notebook → section map

| Q | Topic | Notebook | Section |
|---|---|---|---|
| Q1 | Even sampling / sampling frequency | `01_data_exploration.ipynb` | §2 |
| Q2 | Running vs stopped detection | `02_well_state_detection.ipynb` | §1 |
| Q3 | Controller-mode classification | `02_well_state_detection.ipynb` | §2, §7 |
| Q4 | Runtime fraction | `02_well_state_detection.ipynb` | §3 |
| Q5 | Position reconstruction feasibility & challenges | `03_position_reconstruction.ipynb` | §1, §1b |
| Q6 | Actual reconstruction over a 10-min window | `03_position_reconstruction.ipynb` | §2 |
| Q7 | Dynamometer card: extraction + physical meaning | `04_dynamometer_card.ipynb` | §2, §3 |
| Q8 | Anomaly taxonomy (fluid / mechanical / uncertain) | `04_dynamometer_card.ipynb` | §4, §7, **§8** (3-layer framework), §9 (animations) |
| Q9 | Unsupervised operating regimes | `05_operating_conditions.ipynb` | §2–§6 |
| Q10 | Time structure + next-regime prediction | `05_operating_conditions.ipynb` | §7–§11, §12 (caveats) |

---

## Where each non-trivial method lives

The following pointers are intended for an AI reader who wants to jump
directly into the code that implements a specific method.

- **Sampling QC (Q1)** — `src/well_analysis/signal/sampling.py`.
  Notebook call site: `01_data_exploration.ipynb` §2.
- **Running-state detection, controller-mode clustering, timer-interval
  clustering (Q2–Q4)** — `src/well_analysis/detection/well_state.py`.
  Notebook §7 of nb02 is where the parameter-free version lives;
  earlier sections are the naive baseline.
- **Dead-reckoning integration with drift correction (Q5/Q6)** —
  `src/well_analysis/signal/integration.py`. Notebook §2 of nb03 is
  the 10-min reconstruction.
- **Card extraction + per-card shape metrics (Q7/Q8)** —
  `src/well_analysis/analysis/dynamometer.py`. Key functions:
  `extract_dynamometer_cards`, `card_metrics`, `_polygon_centroid`.
- **Anomaly pipeline (Q8, current version)** — lives **inline in
  `04_dynamometer_card.ipynb` §8, cell 17**, not in `src/`. Three
  layers: confidence tag, per-era visual reference, Isolation
  Forest + per-round argmax + global P99 threshold + Kendall τ
  for within-round worsening + agglomerative shape clustering for
  cross-round repetition. The classification rule is in `classify()`
  inside that cell. Legacy family-rule code still exists in
  `src/well_analysis/analysis/q8.py` but is **superseded** by §8.
- **Animated evidence (Q8)** — `04_dynamometer_card.ipynb` §9, cell 20.
  Produces `notebooks/_generated/nb04_anim_round.mp4` (single-round
  within-round trajectory, fluid hypothesis) and
  `nb04_anim_timelapse.mp4` (multi-round time-lapse, mechanical
  hypothesis).
- **Card feature engineering + KMeans/HDBSCAN (Q9)** —
  `src/well_analysis/analysis/clustering.py`, functions
  `extract_card_features`, `FEATURE_NAMES`, `cluster_with_hdbscan`.
  Notebook call sites: nb05 §2–§4.
- **Episode detection + transition matrix + Markov prediction (Q10)** —
  `src/well_analysis/analysis/clustering.py`, functions
  `episodes_from_labels`, `transition_matrix`,
  `fit_markov_1st_order`, `predict_next`. The **era-aware
  multi-baseline prediction** (§11, the main Q10 result) lives inline
  in `05_operating_conditions.ipynb` cell 23, not in `src/`.

---

## Reading order for a fresh AI reader

1. Skim this README to see which notebook covers which Q.
2. Open the notebook of interest; the first markdown cell names the
   questions it answers.
3. Section headings (`## N. ...`) correspond to the table above.
4. Every quantitative claim is either (a) produced by the cell
   immediately above it or (b) cited from a prior notebook — the
   narrative cells make this explicit.
5. For methods whose implementation lives in `src/`, the notebook
   imports from `well_analysis.*`; grep for the function name to find
   the file.

---

## Known scope boundaries

- No downhole-card inversion (Gibbs wave-equation). Analysis stops at
  surface cards.
- "Fault category" labels in nb04 §8 are **hypotheses**, not validated
  diagnoses. Each notebook's narrative flags this explicitly.
- Dataset is 21 days, which includes a one-way regime shift at
  2020-12-31 / 2021-01-01. Several statistics are reported per-era
  because of this.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/
jupyter lab notebooks/
```

Raw data is not tracked in this repo; see the notebook §1 cells for
the expected path and schema.
