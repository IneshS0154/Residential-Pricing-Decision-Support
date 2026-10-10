# EDA Insight Log

Project: Predictive Modelling of Residential Property Prices (Ames Housing), IT3091, group 2026-DS-38.
Source: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) (section numbers below refer to it). Supporting tables: `reports/tables/`. Figures: `reports/figures/`.
Scope: `Dataset/train.csv` only (1,460 rows). `test.csv` was used only to see which gaps the pipeline must tolerate. Nothing in the EDA is fitted for reuse; the probes and ablations are exploratory.

Hand-off owners (from the proposal): **EDA** IT24103124 | **Prep** IT24102584 (preprocessing, features, leakage audit) | **Model** IT24102616 | **Eval** IT24103989 (evaluation, routing, limitations).

Status: **Confirmed** = measured in the data, no decision needed. **Decision** = needs a group decision (see the register at the end).

---

## A. Data integrity and loading

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F01 | The dataset matches the proposal: train 1,460 x 81 (with `SalePrice`), test 1,459 x 80, one row = one property sale; no duplicate feature rows; `Id` unique. | Sec. 1 (shapes, MD5 and size of all four files, duplicate check). | The unit of analysis and the proposal's table are correct. | EDA | Confirmed |
| F02 | In the raw CSVs the string `NA` means two different things: a documented category "feature absent" (14 columns: `Alley`, `Bsmt*` x5, `FireplaceQu`, `Garage*` x4, `PoolQC`, `Fence`, `MiscFeature`) or a genuinely missing value (`LotFrontage`, `MasVnrType`, `MasVnrArea`, `Electrical`, and `GarageYrBlt` for no-garage houses). Pandas' defaults turn both, plus the real category `None` (`MasVnrType`, 864 rows), into NaN. | Sec. 3; the zero-twin checks. | Always load with `src.data.loading.load_raw()`, never plain `pd.read_csv`. This corrects my first suggestion to use `keep_default_na=False` alone, which left genuine gaps looking like valid data. | Prep | Confirmed |
| F03 | Category spellings in the CSVs differ from `data_description.txt` (`NAmes`, `2fmCon`, `Twnhs`, `CmentBd`, `Wd Shng`, `Brk Cmn`, `C (all)`), and `BedroomAbvGr`/`KitchenAbvGr` are called `Bedroom`/`Kitchen` in the description. | `docs/data_dictionary.md`. | Use the CSV spellings in all mappings. A mapping built from the description would silently miss levels. | Prep | Confirmed |

## B. Target

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F04 | `SalePrice` is right-skewed (skew 1.88, kurtosis 6.5, range $34.9k-$755k, median $163k). After `log1p` skew is 0.12 and the Q-Q plot is near-linear. | Sec. 2; `01_target_distribution.png`. | Model `log1p(SalePrice)`. Interval endpoints back-transform without correction; a point estimate from the log scale is a median unless a bias correction is applied, and the report must say which. | Model, Eval | Confirmed |

## C. Missing values

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F05 | 15 of the 19 columns that contain NA are structural ("feature absent"). NA coincides exactly with a zero-size twin: no garage (81 rows), no basement (37), no fireplace (690), no pool (1,453); `GarageYrBlt` is NaN for exactly the 81 no-garage houses. `Alley`, `Fence`, `MiscFeature` follow the same convention but have no numeric twin. | Sec. 3 (four checks identical; `GarageYrBlt`); `02_missingness.png`. | Fill absence with an explicit "None" level (categoricals) or 0 (areas), and add `Has*` flags (`HasGarage`, `HasBsmt`, `HasFireplace`, `HasPool`). Mis-imputing these ~1,000 cells is a bigger risk than the real gaps. | Prep | Confirmed |
| F06 | Only four columns are genuinely missing: `LotFrontage` (259, 17.7%), `MasVnrType`/`MasVnrArea` (the same 8 rows), `Electrical` (1, Id 1380). Two further rows have a basement but one rating NA: `BsmtExposure` (Id 949) and `BsmtFinType2` (Id 333). | Sec. 3. | Impute only these, inside CV folds. | Prep | Confirmed |
| F07 | `LotFrontage` is not missing completely at random. Missing share is 52% on cul-de-sacs vs 13% on inside lots, and 30-37% on irregular lots vs 8% on regular ones. A model predicts missingness with AUC 0.84; 17 of 78 variables are significantly associated (strongest: `LotShape`, `Neighborhood`, `LotConfig`, `LotArea`). | Sec. 3 and 12; `reports/tables/lotfrontage_missingness_assoc.csv`. | Impute conditionally, not with a global median. | Prep | Confirmed |
| F08 | In a mask-and-recover test (statistics from training folds only) the best `LotFrontage` fill is neighbourhood-median ratio x sqrt(`LotArea`): RMSE 19.7 ft, vs neighbourhood median 22.3, neighbourhood + `LotConfig` median 22.3, global median 24.3. Adding `LotConfig` gives no gain over neighbourhood alone. | Sec. 12; `reports/tables/lotfrontage_imputer_comparison.csv`. | Use the ratio fill, keep the global median as the simple comparison the proposal requires, add a `LotFrontage_missing` flag. | Prep | Confirmed |
| F09 | `test.csv` has gaps that train does not: `MSZoning` (4), `Utilities` (2), `KitchenQual` (1), `Functional` (2), `SaleType` (1), `Exterior1st/2nd` (1 each), `GarageCars`/`GarageArea` (1 each), `TotalBsmtSF` and other basement areas (1), `BsmtFullBath`/`BsmtHalfBath` (2 each). | Sec. 3 (test column). | The pipeline's imputers must handle columns that never had a gap in training, or inference fails on a real listing. | Prep | Confirmed |

## D. Features: distributions, signal, redundancy

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F10 | Area variables are right-skewed (`LotArea` skew 12.2) and several are mostly zero: `PoolArea` 99.5%, `3SsnPorch` 98%, `LowQualFinSF` 98%, `MiscVal` 96%, `ScreenPorch` 92%, `BsmtFinSF2` 89%. | Sec. 4; `03_numeric_distributions.png`. | `log1p` the skewed areas in the linear branch only (trees do not need it); mostly-zero columns are drop-or-flag candidates. | Prep | Confirmed |
| F11 | Many categoricals are nearly constant: `Utilities` 99.9%, `Street` 99.6%, `Condition2` 99%, `RoofMatl` 98%, `Heating` 98%. | Sec. 6 (cardinality table). | Drop or merge rare levels; they carry no usable signal (also F14). | Prep | Confirmed |
| F12 | `MSSubClass` is a numeric code but nominal in meaning (16 codes). Test contains code 150, which train never has. | Sec. 1, 6. | Cast to category; group rare codes; encoders need `handle_unknown` set. | Prep | Confirmed |
| F13 | Price is driven by quality and size. Leaders by mutual information: `OverallQual` (0.57), `Neighborhood` (0.52), `GrLivArea` (0.46), `GarageCars`, `GarageArea`, `TotalBsmtSF`, `YearBuilt`, then the quality ratings (`BsmtQual`, `ExterQual`, `KitchenQual`). Pearson vs log price: `OverallQual` 0.82, `GrLivArea` 0.70. | Sec. 5 and 13; `reports/tables/signal_table.csv`; `12_signal_table.png`. | These are the backbone of the model and the decomposition. `Neighborhood` is the second-strongest signal and is invisible to a correlation screen, so it needs a deliberate encoding (D3). | Prep, Model | Confirmed |
| F14 | Some signal is non-linear and a correlation-only screen would discard it. Largest Pearson-vs-MI gaps: `OverallCond` (Pearson -0.04, MI 0.11), `LotArea`, `2ndFlrSF`, `BsmtUnfSF`. Spearman is well above Pearson for `YearBuilt` (0.65 vs 0.52) and `LotArea` (0.46 vs 0.26). `OverallCond` shows no monotone relation with price. | Sec. 5 and 13. | Do not select features on Pearson alone. `OverallCond` needs an interaction (with quality or age) to help. Test ordinal vs one-hot encoding for the rating scales. | Prep | Confirmed |
| F15 | The weakest signal by every measure is in `MoSold`, `YrSold`, `PoolQC`/`PoolArea`, `3SsnPorch`, `Street`, `Utilities`, `MiscFeature`, `Condition2`, `LandSlope`, `RoofMatl`. | Sec. 13 (weakest-15 table). | Dropping them loses nothing. Matches F10, F11 and the sale-context rule (F21). | Prep | Confirmed |
| F16 | Redundancy: four clusters at \|Spearman\| >= 0.7: `GrLivArea`/`TotRmsAbvGrd`, `GarageCars`/`GarageArea`, `YearBuilt`/`GarageYrBlt` (r = 0.83), `TotalBsmtSF`/`1stFlrSF`. Eight columns have infinite VIF because they are exact sums: `GrLivArea` = `1stFlrSF` + `2ndFlrSF` + `LowQualFinSF` and `TotalBsmtSF` = `BsmtFinSF1` + `BsmtFinSF2` + `BsmtUnfSF`. Other VIFs stay below 6. | Sec. 9 and 14; `reports/tables/correlation_clusters.csv`. | Linear models cannot take totals and parts together (singular design matrix): keep one set. Keep one feature per cluster (the table suggests the strongest-signal member). SHAP credit split between duplicates is arbitrary, so report the decomposition at group level and state this as a limitation. | Prep, Model, Eval | Confirmed |

## E. Outliers and data consistency

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F17 | Ids 524 and 1299 (4,676 and 5,642 sq ft; $184,750 and $160,000; both `Partial` sales in Edwards, quality 10) have the lowest or near-lowest price per sq ft in the data. Ids 692 and 1183 are equally large but priced consistently ($755k, $745k) and should stay. | Sec. 5, 8; `05_scatter_key_features.png`. | Drop only 524 and 1299 (D2). Do not winsorise the rest: it would distort the quantile intervals. | Prep | Decision |
| F18 | Evaluating on the two outliers inflates CV RMSE (0.1311 vs 0.1255 on the same folds without them); training without them slightly improves the clean-row RMSE (0.1255 to 0.1238). | Sec. 10; `reports/tables/ablation_outliers.csv`. | Dropping them helps a little and keeps the evaluation honest; report results both ways. | Prep, Eval | Decision |
| F19 | The area fields are internally consistent (0 violations for both sum rules). Small date oddities: 1 house has `YearRemodAdd` after `YrSold`; 9 garages are dated before their house (`GarageYrBlt` < `YearBuilt`). | Sec. 8. | One more reason to drop `GarageYrBlt` (redundant with `YearBuilt`, F16). Cap or clean the remodel date if used in age features. | Prep | Confirmed |

## F. Time and sale context

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F20 | Sales run 2006-2010, about 300-340 per year; 2010 has only 175 and ends in July. Median price is nearly flat ($164k in 2006 to $155k in 2010) and sales peak May-July. The proposal expected a market disruption; the visible price shift is modest. | Sec. 7; `09_time_and_season.png`. | The temporal split (train 2006-2009, test 2010) is feasible but its test fold is small and a half-year. Interpret its result with care and expect only a mild shift. | Eval | Confirmed |
| F21 | `SaleCondition` = Partial and `SaleType` = New carry a large premium (median ~$245k vs ~$160k). Part of this is confounding: Partial/New homes are newer (median build year 2006 vs 1970), higher quality (7 vs 6) and larger (1,646 vs 1,440 sq ft). In any case the four sale-context columns (`SaleType`, `SaleCondition`, `MoSold`, `YrSold`) are not known when a listing price is set. | Sec. 7, 10. | Exclude them as predictors (D1). Keep `YrSold` as the split key and for age arithmetic. | Prep | Decision |
| F22 | The realism rule is nearly free. Dropping all four sale-context columns changes CV RMSE (log) from 0.1311 to 0.1319 (+0.0008), far below the fold-to-fold SD (about 0.013); dropping `SaleType` + `SaleCondition` alone costs +0.0013, `MoSold` + `YrSold` alone +0.0001. | Sec. 10; `reports/tables/ablation_sale_context.csv`. | The exclusion rule costs almost no accuracy, so there is no accuracy argument for breaking it. | Prep, Eval | Confirmed |

## G. Where uncertainty lives (supports the interval and routing premise)

| ID | Finding | Evidence | Implication | Hand-off | Status |
|---|---|---|---|---|---|
| F23 | Error is heterogeneous, using out-of-fold predictions (no in-sample optimism). A crude linear probe (RMSE 0.20) and a gradient-boosting probe on numeric features (RMSE 0.137) both show unequal residual spread. Levene's test rejects equal variance across quality tiers (p = 0.003 linear, < 0.001 boosting) and across neighbourhoods (p < 0.001, both). Boosting residual SD by quality tier: 0.20 (1-4), 0.12 (5-8), 0.25 (9-10). By neighbourhood it ranges from 0.07 (CollgCr) to 0.25 (IDOTRR), about 3.5x. | Sec. 11 and 15; `11_uncertainty_by_segment.png`, `13_cv_uncertainty_probe.png`. | The project's core premise holds: error is predictable from segment, so interval width can separate "price automatically" from "send to an appraiser". Expect low- and top-quality homes and a few older neighbourhoods (IDOTRR, Edwards, OldTown) to be escalated most. | Model, Eval | Confirmed |
| F24 | Within expensive neighbourhoods the price spread is large (StoneBr, NoRidge, NridgHt std $95-120k); median price runs from $88k (MeadowV) to $315k (NridgHt), 3.6x. Several neighbourhoods are tiny (Blueste n = 2, NPkVill n = 9). | Sec. 6; `07_price_by_neighborhood.png`. | `Neighborhood` is a natural segment for the secondary lens. Tiny neighbourhoods make unsmoothed target encoding unstable; a target encoding must be fitted in-fold and smoothed. | Prep, Eval | Decision (D3) |
| F25 | Trap for evaluation: grouping residuals by actual price band makes cheap homes look over-predicted (mean residual -0.16) and expensive ones under-predicted (+0.12). This is regression to the mean from conditioning on the outcome. | Sec. 11 (residuals by predicted vs actual band). | Band segment errors on predicted price (known at decision time), never on actual price. | Eval | Confirmed |

---

## Decision register

| ID | Decision | Evidence | Recommendation | Owner | Status |
|---|---|---|---|---|---|
| D1 | Treat `SaleType`, `SaleCondition`, `MoSold`, `YrSold` as non-predictors? And keep `YrSold` as split key / age reference? | F21, F22: exclusion costs +0.0008 RMSE; the variables are unknowable at listing. | Exclude all four as predictors; keep `YrSold` for the temporal split and age arithmetic only. | Group; Prep implements | Decided (preprocessing brief); implemented in `build_preprocessor` |
| D2 | Exclude Ids 524 and 1299 from training? | F17, F18; `reports/tables/decision_d2_outliers.csv` (final 172-column pipeline). Excluding them from training improves RMSE on the remaining rows for both model families: Ridge 0.1186 -> 0.1144 (gain 0.00416, 4.2 SE), gradient boosting 0.1249 -> 0.1234 (gain 0.00147, 2.5 SE). Dropping two *random* other rows gains on average +0.00001 (largest of 25 trials +0.00016), so the effect is specific to these two sales. | Exclude from TRAINING (`SUGGESTED_TRAINING_EXCLUSIONS`), and report CV results both on all rows and on clean rows. The pre-registered rule (> 2 SE for both models and above every placebo) is met. | Group confirms; Prep, Model, Eval | Recommended, awaiting group confirmation |
| D3 | Encoding for `Neighborhood`: one-hot, smoothed in-fold target encoding, or both? | `reports/tables/decision_d3_neighborhood_encoding.csv`. Target encoding alone is worse than one-hot for Ridge (RMSE 0.1338 vs 0.1309, 4.6 SE worse) and better for gradient boosting (0.1269 vs 0.1290, 2.7 SE). Dropping `Neighborhood` costs 0.0063 (Ridge) and 0.0020 (boosting). | Keep one-hot as the default (pre-registered rule: target encoding fails because it is clearly worse for Ridge). Post-hoc observation, not part of the rule: one-hot plus target encoding together gave the best boosting result (0.1258, 4.4 SE better than one-hot) and no change for Ridge; retest it in the modelling stage with tuned tree models. | Group confirms; Prep, Model | Recommended, awaiting group confirmation |
| D4 | `log` or `log1p` for the target and skewed features? | The earlier EDA notebook (previous project) used `np.log`; this one uses `log1p`. The difference is negligible here (all prices well above 1). | Use `log1p` everywhere and say so in the report. | Group | Decided in practice (`log1p` used everywhere) |

## Corrections made during the EDA

- The first suggestion to load with `keep_default_na=False` alone was wrong (F02); `load_raw()` replaces it, and CLAUDE.md was updated.
- An early count said 14 structural-NA columns; the correct count is 15 of 19 (F05).
- Rough estimates for the cost of excluding sale-context columns ("about 0.001") and for removing the two outliers (an uncontrolled comparison) were replaced by like-for-like measurements on identical folds (F18, F22).

## Limitations of this EDA

- The probes and ablations use all 1,460 training rows for exploration. They are not reported results and none of their fitted objects is reused.
- The boosting probe uses numeric features only, so it measures segment spread roughly, not the final model's.
- The audit's "derived from price: No" for every column is a judgement from the data dictionary, not a tested fact.
- Mutual information was estimated on the full training set; its ranking, not its absolute values, should be used.


## Addendum: redundancy handling (applied)

F16 recommended keeping one column per correlated pair and totals instead of parts. This is now implemented in `build_preprocessor` (`reduce_redundancy=True`, 12 columns in `REDUNDANT`: `TotRmsAbvGrd`, `GarageArea`, `1stFlrSF`, `2ndFlrSF`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, the four bathroom counts and the engineered `TotalSF`). Matrix rank of the numeric inputs went from 27 of 30 to 18 of 18 (largest VIF 3.8); Ridge RMSE changed by -0.00010 (SE 0.00158), i.e. no measurable accuracy effect. A rank check showed that the first version missed the exact dependency between `TotalBathrooms` and its four parts. See notebook 02, section 11b. The default output is now **172 columns** from 57 attributes (`docs/final_features.md`).