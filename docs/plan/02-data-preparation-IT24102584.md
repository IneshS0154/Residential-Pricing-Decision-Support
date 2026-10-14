# Plan: Data Preparation

**Member:** IT24102584 — Samarasinghe G.K.D.T
**Proposal phases covered:** Phase 2 (Data Understanding — quality half), Phase 3 (Data
Preparation)
**Runs:** Second, starting once the dataset is acquired and the EDA insight log has an initial
pass (can overlap with the tail end of IT24103124's EDA rather than waiting for it to fully
finish, but needs the business problem statement to do the realism audit correctly).

**Depends on:** dataset in `data/raw/` (IT24103124), business problem / "what's knowable at
listing time" framing (IT24103124).
**Blocks:** everything in modelling (IT24102616) — the preprocessing pipeline is a hard
prerequisite for training any model.

## Task checklist (in order)

1. **Understand the dataset structure** — rows, columns, variable types, confirm one row = one
   observation (already set by IT24103124, verify it holds after seeing all 81 columns).
2. **Build the data dictionary** — document every variable: meaning, type, units, in
   [`docs/data_dictionary.md`](../data_dictionary.md). Use `data/raw/data_description.txt` as
   the source of truth.
3. **Identify variable types** — numerical / nominal / ordinal / binary, and the treatment each
   implies (ordinal → ordered encoding, high-cardinality nominal → target/frequency encoding
   candidate, etc.).
4. **Separate absent-feature markers from genuinely missing data** — e.g. `NA` in `PoolQC` means
   "no pool," not "unrecorded." Go column by column using `data_description.txt`; get this wrong
   and imputation downstream is wrong too.
5. **Handle genuinely missing values** — pick and justify an imputation approach per variable
   type, compared against at least one simpler alternative (e.g. median vs. group-median by
   neighborhood for `LotFrontage`).
6. **Encode categorical variables** — ordered encoding for genuine ordinal scales (`ExterQual`,
   `KitchenQual`, etc.), an appropriate approach for high-cardinality nominals (`Neighborhood`).
7. **Engineer features** — e.g. `TotalArea`, `TotalBath`, `PropertyAge` — each with a stated
   reason, logged as you go.
8. **Audit for leakage and realism** — this is the most consequential step in this plan:
   - Verify no variable derived from `SalePrice` enters the predictors.
   - For every candidate predictor, ask "would this be known at the time a listing price is
     set?" Sale-context variables (`SaleType`, `SaleCondition`, `MoSold`, `YrSold`) are
     excluded on this basis — see `CLAUDE.md` for the non-negotiable list.
9. **Build the preprocessing pipeline** — implement every transformation (imputation, encoding,
   scaling) as a single `sklearn` `Pipeline`/`ColumnTransformer` object in
   [`src/features/`](../../src/features/), so it can be fit inside CV training folds only. This
   is a hard constraint, not a style choice (see `CLAUDE.md` — no leakage).
10. **Maintain the preprocessing and feature log** — every decision (options considered, choice
    made, evidence) goes into
    [`docs/preprocessing_feature_log.md`](../preprocessing_feature_log.md) as you go.

## Deliverables

- `docs/data_dictionary.md` completed for all variables under consideration.
- `docs/preprocessing_feature_log.md` completed, including the leakage/realism audit table.
- A single, reusable preprocessing pipeline object in `src/features/`.

## Docs to update

- [`docs/data_dictionary.md`](../data_dictionary.md)
- [`docs/preprocessing_feature_log.md`](../preprocessing_feature_log.md)

## Note on separation of duties

Per the proposal, IT24103989 (not you) independently verifies that this pipeline is fit inside
CV folds only, with no transformation fit on the full dataset before splitting. Make the pipeline
easy to inspect for this — keep fit/transform steps explicit and out of ad-hoc notebook cells.
