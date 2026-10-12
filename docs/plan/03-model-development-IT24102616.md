# Plan: Model Development

**Member:** IT24102616 — Chandrasekaran I.S
**Proposal phases covered:** Phase 4 (Modelling)
**Runs:** Third, starting once the preprocessing pipeline exists and is stable.

**Depends on:** the preprocessing pipeline and engineered features (IT24102584), the CV fold
structure (defined here, but stratified by price band using insight from IT24103124's EDA).
**Blocks:** all of evaluation (IT24103989) — nothing can be evaluated before models exist.

## Task checklist (in order)

1. **Set up cross-validation folds** — repeated k-fold, stratified by price band, shared across
   every model so comparisons are apples-to-apples. Implement once, reuse everywhere.
2. **Select candidate algorithms** — confirm the set below; each is included for a stated reason,
   not to pad the comparison:

   | Model | Role |
   |---|---|
   | Dummy regressor (mean/median) | Trivial benchmark — the floor |
   | OLS on a small obvious predictor set | Naive, honest baseline |
   | Regularized linear (Ridge/Lasso/ElasticNet) | Handles wide encoded matrix, interpretable coefficients |
   | Random Forest | Captures interactions, robust to outliers/skew |
   | Gradient Boosting | Expected strongest point-accuracy performer |
   | Quantile Gradient Boosting (lower & upper) | Produces the per-property prediction intervals |

3. **Train the baseline models** (Dummy, OLS) — establish the floor every other model must beat.
4. **Train the alternative models** (Ridge/Lasso/ElasticNet, Random Forest, Gradient Boosting) —
   same pipeline, same folds as the baselines.
5. **Train the quantile models** — fit lower and upper quantile regressors at the chosen nominal
   level (e.g. 10th/90th percentile for an 80% interval). Model on the log scale — quantiles are
   preserved under the monotone log transform, so interval endpoints back-transform correctly
   without extra correction (see `CLAUDE.md`).
6. **Hyperparameter tuning** — search within the CV structure, never on the folds used for final
   reporting. Keep a tuning log (what was searched, what won, why).
7. **Produce the price decomposition** — compute additive feature attributions (e.g. SHAP) that
   sum with a base value to exactly recover each prediction. Group the raw attributions into the
   business-meaningful categories (location, size, quality/condition, age, amenities), and
   produce the land-vs-structure summary (location+lot vs. building).
   - Word this as model behavior, not causation — see `CLAUDE.md`.
8. **Investigate feature importance** where the model permits it (e.g. permutation importance,
   coefficient magnitudes for linear models).
9. **Document the modelling process** — algorithms, settings, tuning decisions, and reasoning —
   in `docs/preprocessing_feature_log.md` or a dedicated modelling log, team's choice.

## Deliverables

- Trained baseline, alternative, and quantile models in [`src/models/`](../../src/models/).
- A model comparison table (point accuracy across candidates, same folds).
- A working decomposition function that outputs grouped attributions + land-vs-structure split
  for any given property.

## Reminder from CLAUDE.md

- Point estimates from a log-scale model approximate the **median**, not the mean — say so
  explicitly wherever a back-transformed point estimate is reported.
- Nothing here gets evaluated/reported on `test.csv` — that file has no labels.
