# Predictive Modelling of Residential Property Prices for Real Estate Pricing Decisions

**Module:** IT3091 — Machine Learning
**Programme:** B.Sc. (Hons) in Information Technology, SLIIT
**Track:** Guided Data Track (Group code 8)

## Team

| ID | Name | Primary Responsibility |
|---|---|---|
| IT24103124 | Walawwaththage H.D.N | Business problem framing, lens selection, EDA |
| IT24102584 | Samarasinghe G.K.D.T | Data preparation, feature engineering, leakage/realism audit |
| IT24102616 | Chandrasekaran I.S | Model development, quantile models, hyperparameter tuning |
| IT24103989 | Gunarathna H.M.S.S | Evaluation, interval calibration, routing threshold, recommendation |

## Business Problem

How can a real estate agency estimate the sale price of a residential property from its recorded
attributes, and know how much confidence to place in each estimate?

**Primary lens:** Price Prediction. **Secondary lens:** Property Segment Analysis (supporting role).

The project does not stop at a single predicted price. For every property it produces:

1. A **point estimate** of sale price.
2. A **prediction interval** (e.g. 80% nominal level) expressing how certain that estimate is.
3. A **price decomposition** into business-meaningful attribute groups (location, size, quality,
   age, amenities), plus a land-vs-structure summary.
4. A **routing flag** — `AUTO` (price automatically) or `ESCALATE` (send to a human appraiser) —
   derived from the interval width.

See [`2026-DS-38-InitialSubmission.pdf`](2026-DS-38-InitialSubmission.pdf) for the full proposal
this project implements.

## Dataset

[Ames Housing — House Prices: Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Only `train.csv` (1,460 rows, 81 columns, labelled) is used for all modelling and evaluation.
`test.csv` is unlabelled and, per the proposal, is used only as an optional leaderboard sanity
check — never as the reported evaluation.

**Setup:** download `train.csv`, `test.csv`, and `data_description.txt` from the Kaggle competition
page above and place them in [`data/raw/`](data/raw/) (gitignored — not committed).

```bash
# with the Kaggle CLI configured (~/.kaggle/kaggle.json):
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
cd data/raw && unzip house-prices-advanced-regression-techniques.zip && cd ../..
```

## Project Structure

```
.
├── data/
│   ├── raw/            # Original Kaggle files (gitignored)
│   └── processed/      # Cleaned / feature-engineered datasets (gitignored)
├── notebooks/           # EDA and exploratory modelling notebooks
├── src/
│   ├── data/            # Loading, cleaning, leakage/realism audit
│   ├── features/        # Feature engineering, preprocessing pipeline
│   ├── models/           # Baseline, alternative, and quantile models
│   └── evaluation/       # Metrics, interval calibration, segment analysis, routing threshold
├── reports/
│   └── figures/          # Generated plots for the final report
├── docs/                 # Data dictionary, EDA insight log, preprocessing/feature log
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validation Design

- Repeated k-fold cross-validation, stratified by price band, on `train.csv` only.
- The entire preprocessing pipeline is fit inside each training fold only (no leakage).
- Secondary temporal validation: train on earlier `YrSold`, test on later.
- Hyperparameter tuning happens within the CV structure, never on folds used for final reporting.

## Models Compared

Dummy regressor → OLS (small predictor set) → Regularized linear (Ridge/Lasso/ElasticNet) →
Random Forest → Gradient Boosting → Quantile Gradient Boosting (lower/upper, for intervals).

## Scope

**Core (committed):** point estimate, prediction interval, price decomposition.
**Stretch (only if core is complete and evaluated by end of Week 4):** comparable sales
retrieval, improvement lever ranking.

No production pricing system is built or deployed. The project ends at decision support.
