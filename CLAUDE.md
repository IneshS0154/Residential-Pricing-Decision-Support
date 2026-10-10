# CLAUDE.md

Do not include a Co-Authored-By line in commit messages.

## Project

IT3091 Machine Learning (SLIIT), group 2026-DS-38. Predictive modelling of residential property prices on the Ames Housing dataset (Kaggle: House Prices: Advanced Regression Techniques) for a real-estate pricing decision.

The deliverable is not a single price. For each property the project produces:
1. a point estimate of `SalePrice`
2. a prediction interval (quantile models, nominal level e.g. 80%)
3. a price decomposition into business attribute groups (additive attributions, e.g. SHAP), including a land-vs-structure summary
4. an AUTO / ESCALATE routing flag derived from interval width

The proposal PDF is the source of truth for scope. Stretch items (comparable sales retrieval, improvement lever ranking) are out of scope unless core is finished.

## Data

- `Dataset/train.csv` is the only labelled file (1,460 rows x 81 cols). `test.csv` has no `SalePrice`, so it is never used for evaluation. `Dataset/` is read-only raw data; never modify it.
- Always load the CSVs with `src.data.loading.load_raw()`, never plain `pd.read_csv`. In the raw files the string `"NA"` means two different things: a documented category "feature absent" (Alley, Bsmt*, FireplaceQu, Garage* except GarageYrBlt, PoolQC, Fence, MiscFeature) or a genuinely missing value (LotFrontage 259, MasVnrType/MasVnrArea 8, Electrical 1, GarageYrBlt for the 81 no-garage houses). Pandas defaults turn all of them, and the real veneer category `"None"` (864 rows), into NaN. `load_raw()` keeps `"NA"` as a string only in the documented-category columns and makes it NaN elsewhere.
- Category spellings in the CSVs differ from `data_description.txt` (e.g. `2fmCon`, `Twnhs`, `CmentBd`, `Wd Shng`, `Brk Cmn`, `C (all)`, `NAmes`). Use the spellings in the CSV.
- The columns `BedroomAbvGr` and `KitchenAbvGr` are called `Bedroom` and `Kitchen` in the data description.
- Target is modelled as `log1p(SalePrice)`. Interval endpoints back-transform directly; point estimates are medians unless a bias correction is applied. State which.

## Rules that must not be broken

- No leakage. Every fitted transformation (imputation, scaling, encoding, target encoding, feature selection) lives inside one sklearn `Pipeline` and is fitted on training folds only.
- Same preprocessing and same CV folds for every model.
- Hyperparameter search uses nested/inner CV, never the folds used to report results.
- CV is repeated k-fold stratified by price band. A secondary temporal split (train on earlier `YrSold`, test on later) is reported; disagreement between the two is a finding, not something to hide.
- Sale-context variables (`SaleType`, `SaleCondition`, `MoSold`, `YrSold`) are not predictors, since they are not known when a listing price is set. `YrSold` may be used as the split key and for age reference. Final handling is recorded in `docs/preprocessing_log.md`.
- Report interval coverage together with mean interval width, never one alone.
- Notebooks only when asked (`01_eda.ipynb`, `02_preprocessing.ipynb` exist). Reusable logic goes in `src/` modules, not in notebooks.
- Preprocessing: always use `build_preprocessor(strategy)` from `src/features/preprocessing.py` (strategies: baseline, neighborhood, neighborhood_indicator, knn; default selected strategy `baseline`, default `scale="robust"`, `reduce_redundancy=True` (172 output columns, no exact linear dependencies), use `scale=None` for trees). Never fit it on the full data before evaluation; get folds from `src.evaluation.cv.get_folds` and score with `cv_scores`.
- Ids 524 and 1299 are kept by the pipeline; `SUGGESTED_TRAINING_EXCLUSIONS` lists them for the modelling team (decision D2: exclusion recommended, see docs/eda_insight_log.md).
- Tests: `python3 -m unittest discover -s tests -v` from the project root.

## Layout

```
Dataset/          raw Kaggle files (read-only)
data/interim/     intermediate cleaned data
data/processed/   model-ready data
notebooks/        01_eda.ipynb, 02_preprocessing.ipynb
src/data/         loading, schema, validation
src/features/     preprocessing.py (build_preprocessor and custom transformers), feature_groups.py
src/models/       baseline, regularised linear, RF, GBM, quantile models
src/evaluation/   cv.py (shared folds, cv_scores), later: metrics, calibration, routing
docs/             data dictionary, preprocessing log, EDA insight log
reports/figures/  plots for the report
artifacts/        fitted models and metrics
tests/            leakage and pipeline tests
```

## Logs to maintain

- `docs/preprocessing_log.md`: each decision with options considered, choice and evidence.
- `docs/eda_insight_log.md`: each finding with evidence and implication.
- `docs/data_dictionary.md`: every variable with full name, type and treatment.

## Environment

Python 3.12, pandas, scikit-learn, lightgbm, xgboost, scipy, matplotlib are installed. seaborn is not installed.
