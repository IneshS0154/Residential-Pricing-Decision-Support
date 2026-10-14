# Predictive Modelling of Residential Property Prices for Real Estate Pricing Decisions

**Module:** IT3091 - Machine Learning (SLIIT) | **Group:** 2026-DS-38 | **Programme:** B.Sc. (Hons) in Information Technology
**Track:** Guided Data Track, group code 8 (Real Estate, Ames Housing)

## Team

| ID | Name | Primary responsibility |
|---|---|---|
| IT24103124 | Walawwaththage H.D.N | Business problem framing, lens selection, EDA |
| IT24102584 | Samarasinghe G.K.D.T | Data preparation, feature engineering, leakage / realism audit |
| IT24102616 | Chandrasekaran I.S | Model development, quantile models, hyperparameter tuning |
| IT24103989 | Gunarathna H.M.S.S | Evaluation, interval calibration, routing threshold, recommendation |

## Business problem

How can a real estate agency estimate the sale price of a residential property from its recorded attributes, and know how much confidence to place in each estimate?

**Primary lens:** Price Prediction. **Secondary lens:** Property Segment Analysis (supporting role: where does the model fail?).
Unit of analysis: one completed residential property sale. Task: supervised regression on `log1p(SalePrice)`.

For a property the project produces:

1. a **point estimate** of `SalePrice` (a median: `expm1` of the log-scale prediction, no bias correction);
2. an **80 % prediction interval** (split conformalized quantile regression);
3. **group-level evidence on where value comes from** (grouped permutation importance, feature-group ablation, ElasticNet coefficients). The initial submission also promised a per-property additive decomposition and a land-vs-structure summary: an exact additive decomposition exists only for the ElasticNet benchmark (adopted by the group as the explanation panel, D5) and a land-vs-structure value split is not supportable with this data (not delivered; a descriptive replacement was adopted by the group, D6). No SHAP is used (see `docs/worked_examples.md`, `docs/decision_log.md` DL-21, DL-29, DL-30);
4. an **AUTO / ESCALATE** flag from relative interval width: AUTO = candidate for light-touch review, ESCALATE = send to a human appraiser.

The system is a **research prototype / controlled-pilot candidate**, not a production pricing system. There is no independent test set, and the routing target is hypothetical. See `docs/business_interpretation.md`.

## Dataset

[Ames Housing - House Prices: Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- `Dataset/train.csv` (1,460 rows x 81 columns, labelled) is the only file used for modelling and evaluation. `Dataset/test.csv` (1,459 x 80) has no `SalePrice` and is never used for evaluation. `Dataset/data_description.txt` gives the variable definitions. `Dataset/` is read-only raw data; `data/raw/` holds a tracked copy that the code does not read.
- File sizes and MD5 hashes are printed in section 1 of `notebooks/01_eda.ipynb` (finding F01).
- **Always load through `src.data.loading.load_raw()`**, never plain `pd.read_csv`: the string "NA" means "feature absent" in 14 columns and "missing" elsewhere.
- Data dictionary: `docs/data_dictionary.md`.
- To obtain the files elsewhere: `kaggle competitions download -c house-prices-advanced-regression-techniques` and unzip into `Dataset/`.

## Notebooks (run order 01 to 10)

Each notebook has a stage document in `docs/`. Reusable logic lives in `src/`, not in the notebooks. Notebooks were renamed from `01, 02, 03, 03b ... 03h` to a clean 01-10 sequence; `docs/project_map.md` maps the stages and explains older wording inside notebooks.

| # | Notebook | What it does | Stage document |
|---|---|---|---|
| 01 | `01_eda.ipynb` | Data understanding, quality, missingness, target, correlation, outliers, leakage / realism audit | `docs/eda_insight_log.md` |
| 02 | `02_preprocessing.ipynb` | Leakage-safe preprocessing pipeline and its validation | `docs/preprocessing_log.md`, `docs/preprocessing_feature_log.md` |
| 03 | `03_feature_selection.ipynb` | Feature sets (reduced / expanded / manual), group ablation, nested selection | `docs/feature_set_decision.md` |
| 04 | `04_baseline_model_comparison.ipynb` | Seven models with fixed hyperparameters on the same folds (no tuning) | `docs/baseline_comparison.md` |
| 05 | `05_hyperparameter_tuning.ipynb` | Optuna with nested CV for ElasticNet and Gradient Boosting (tuning is kept separate) | `docs/tuning.md` |
| 06 | `06_final_evaluation.ipynb` | Out-of-fold evaluation, residuals, segments, outliers, grouped permutation importance | `docs/final_evaluation.md` |
| 07 | `07_prediction_intervals.ipynb` | Raw quantile intervals vs split CQR, coverage with width | `docs/prediction_intervals.md` |
| 08 | `08_segment_temporal_analysis.ipynb` | Segment reliability and temporal (train 2006-2008 / test 2009-2010) analysis | `docs/segment_temporal.md` |
| 09 | `09_auto_escalate_routing.ipynb` | AUTO / ESCALATE rule, threshold selection on development data only | `docs/routing.md` |
| 10 | `10_business_interpretation.ipynb` | Evidence registry, stakeholder tables, recommendation, limitations | `docs/business_interpretation.md` |

Workflow diagram: `docs/workflow.md` (picture: `docs/figures/workflow_diagram.png`). Decisions with alternatives, reasons and evidence: `docs/decision_log.md`. Group decisions (D2, D3, D5, D6 decided; D7 pending): `docs/pending_group_decisions.md`.

## Methodology in brief

- **Target and metric:** `log1p(SalePrice)`; Log-RMSE is the primary metric, with MAE / RMSE in USD, MAPE, R2, fold SD and paired Nadeau-Bengio corrected intervals. Interval coverage is always reported together with mean width.
- **Validation:** repeated 5-fold x 3 cross-validation stratified by price band, seed 42, identical folds for every model (`src/evaluation/cv.py`); nested CV for tuning (outer = these folds, inner = 3-fold); secondary temporal split (train `YrSold` <= 2008 = 947 houses, test 2009-2010 = 513).
- **No leakage:** every fitted step (imputation, encoding, scaling, skew transform, any selection) lives in one scikit-learn `Pipeline` that is cloned and fitted on training folds only. Sale-context variables (`SaleType`, `SaleCondition`, `MoSold`, `YrSold`) are not predictors. Tests assert these rules.
- **Preprocessing:** `build_preprocessor("baseline")` for every model (79 raw predictors -> 172 columns); `FEATURE_SETS["reduced"]` is the same pipeline and is retained after the feature-set comparison.
- **Models:** Dummy, OLS, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting (notebook 04); tuned ElasticNet and Gradient Boosting (notebook 05); working model tuned Gradient Boosting, statistically tied with tuned ElasticNet on Log-RMSE (low-to-moderate confidence, depends on how decision D2 is applied: the group decided to exclude the two houses from training, but the stored results were not re-run).
- **Interpretation without SHAP:** correlations and mutual information, coefficients, grouped permutation importance, feature-group ablation, residual and segment analysis. None of this is causal.

## Headline results (development evidence; read from `reports/tables/`, traced in `docs/numerical_traceability.md`)

| Quantity | Value | Source |
|---|---|---|
| Fixed-configuration Log-RMSE (best / Gradient Boosting / Dummy) | ElasticNet 0.1289 / 0.1311 / 0.3992 | `baseline_comparison.csv` |
| Tuned Gradient Boosting, nested CV | Log-RMSE 0.1260, MAE $15,535, RMSE $26,774, MAPE 8.76 %, R2 0.884 | `tuning_vs_baseline.csv` |
| 80 % interval coverage / mean width | raw quantile 65.1 % / $43.2k; CQR 80.7 % / $56.7k | `intervals_main_comparison.csv` |
| Temporal CQR coverage | 76.7 % | `intervals_temporal.csv` |
| Routing (hypothetical 7 % AUTO MAPE target, nested) | AUTO 45.0 % of houses, AUTO MAPE 6.49 %, AUTO coverage 78.5 %; ESCALATE MAPE 10.66 % | `routing_heldout_policy.csv` |

All numbers come from the same 1,460 rows that informed the design; they are not an independent test.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install jupyter nbconvert nbformat
```

Versions that produced the stored results: Python 3.14.2, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, optuna 5.0.0, lightgbm 4.7.0, matplotlib 3.11.2. `requirements.txt` is not pinned. It also lists `xgboost` and `mapie`, which no code in this repository imports (CQR is implemented directly in `src/evaluation/intervals.py`), so they do not need to be installed to run the notebooks or tests. On this Windows machine use `python` (Docstrings in `src/` say `python3`).

## Running

Run everything from the project root (modules import as `src.*`).

```bash
python -m unittest discover -s tests            # about 1-2 minutes; 155 tests
jupyter nbconvert --to notebook --execute --output-dir exec notebooks/01_eda.ipynb   # repeat for 02 ... 10 in order
```

Notebooks 03-10 read cached experiment artifacts from `artifacts/` (not committed; `.gitignore` excludes `artifacts/` and `reports/figures/`), so a fresh clone must regenerate them first, in this order (times are approximate; wall-clock seconds are machine-specific, so timing columns differ between runs while every other number reproduces exactly: `docs/submission/clean_environment_reproduction.md`):

```bash
python -m src.evaluation.run_feature_selection          # ~8 min, caches in artifacts/feature_selection/
python -m src.evaluation.baseline_comparison            # ~3.5 min
python -m src.evaluation.nested_tuning --nested --trials elasticnet=150 gbr=50    # ~17 min
python -m src.evaluation.nested_tuning --final --trials elasticnet=300 gbr=100    # ~10 min
python -m src.evaluation.nested_tuning --pilot --trials elasticnet=12 gbr=12       # ~1 min, timing only; notebook 05 reads its trial durations
python -m src.evaluation.final_evaluation               # ~7 min
python -m src.evaluation.intervals                      # ~2 min
python -m src.evaluation.temporal                       # ~3 min
python -m src.evaluation.routing_runs                   # ~2 min
```

Other generators: `python -m src.evaluation.decision_experiments` (D2 / D3 evidence), `python -m src.features.export_feature_list` (`docs/final_features.md`), `python -m src.evaluation.feature_set_decision` (feature-set table), `python -m src.evaluation.doc_traceability` (documents vs stored results), `python -m src.evaluation.workflow_diagram` (diagram), `python -m src.evaluation.worked_examples` (four-house demonstration and decomposition tables from stored outputs; no fitting). Add `--force` to the experiment runners to recompute cached results.

## Repository layout

```
Dataset/          raw Kaggle files (read-only; what load_raw() reads)
data/             raw copy, interim and processed placeholders
notebooks/        01_eda ... 10_business_interpretation
src/data/         loading, schema, validation, EDA checks
src/features/     preprocessing.py (build_preprocessor), feature_groups.py, feature_sets.py, export_feature_list.py
src/models/       baselines.py (fixed registry), candidates.py, tuning.py (Optuna search spaces)
src/evaluation/   cv.py, feature_selection.py, baseline_comparison.py, nested_tuning.py, final_evaluation.py, intervals.py,
                  segments.py, temporal.py, routing.py, routing_runs.py, evidence.py, decision_experiments.py,
                  feature_set_decision.py, doc_traceability.py, workflow_diagram.py, worked_examples.py, decomposition.py
docs/             data dictionary, logs, stage documents, decision log, workflow, plan/ (per-member plans)
reports/tables/   generated result tables (committed)    reports/figures/  generated plots (not committed)
artifacts/        cached experiment results and fitted pipelines (not committed)
tests/            leakage, pipeline, evaluation and consistency tests
```

## Documentation map

`docs/final_report.md` (final report), `docs/proposal_vs_implementation.md` (37 proposal commitments and their status), `docs/worked_examples.md` (four-house demonstration, per-property decomposition and land-vs-structure investigation), `docs/project_map.md` (stage map), `docs/workflow.md` (workflow diagram), `docs/decision_log.md` (consolidated decisions), `docs/pending_group_decisions.md` (D2, D3, D5, D6, D7), `docs/eda_insight_log.md`, `docs/preprocessing_log.md`, `docs/preprocessing_feature_log.md`, `docs/data_dictionary.md`, `docs/final_features.md`, `docs/reproducibility_notes.md`, `docs/numerical_traceability.md`, and one document per modelling stage (see the notebook table).

## Reproducibility and limits

- Seeds: folds 42, models 42, Optuna 42 + fold index. Re-executing notebooks 02-10 reproduced the saved outputs; notebook 01's gradient-boosting probes differ by up to 0.0006 Log-RMSE from the earlier saved run (cause not established; `docs/reproducibility_notes.md`).
- No independent test set; the same rows informed features, model family and thresholds. The routing target (7 % AUTO MAPE) is hypothetical because the proposal states none. AUTO interval coverage is below nominal. No cost savings or causal claims are made.
- Decisions: D2 (exclude Ids 524 / 1299 from training), D3 (keep one-hot), D5 (adopt the ElasticNet per-property decomposition) and D6 (adopt the descriptive replacement) were **decided by the group**; D7 (temporal-split reason) is **pending group confirmation**. D2 is not applied to the stored results, which keep both rows in training; its effect is documented in `docs/final_report.md` Section 6.4.
- AI-use declaration and individual / video submissions are not part of the repository content yet; they require the group's own statements.

## Scope

**Core (committed):** point estimate, prediction interval, group-level importance, AUTO / ESCALATE routing. **Out of scope:** comparable-sales retrieval and improvement-lever ranking (stretch items), any production or deployed pricing system. The project ends at decision support. The initial proposal is `2026-DS-38-InitialSubmission.pdf`.
