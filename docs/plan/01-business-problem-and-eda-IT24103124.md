# Plan: Business Problem & Exploratory Analysis

**Member:** IT24103124 — Walawwaththage H.D.N
**Proposal phases covered:** Phase 1 (Business Framing), Phase 2 (Data Understanding — EDA half)
**Runs:** First. Nothing else in the project can start meaningfully before this, because the
realism audit, the feature set, and the model choices all depend on the business problem being
pinned down and the target/segment structure being understood.

**Blocks:** IT24102584's leakage/realism audit needs the business problem statement (what's
knowable at listing time). IT24102616's model choices and IT24103989's segment error analysis
both lean on the EDA insight log.

## Task checklist (in order)

1. **Define the business problem** — write out the pricing decision the project supports, and
   the cost of getting it wrong in each direction (unsold listing vs. seller under-payment).
2. **Justify the lens selection** — document why Price Prediction (with interval + decomposition)
   was chosen over the two alternative lenses (Valuation Feature Analysis, Property Segment
   Analysis), and why Property Segment Analysis is kept as a supporting secondary lens.
3. **Identify stakeholders** — valuation team, listing agents, appraisers — and what each needs
   from the output (a number vs. a trust signal vs. a reviewable explanation).
4. **Confirm the unit of analysis** — one row = one completed property sale, one prediction per
   property.
5. **Acquire and verify the dataset** — download `train.csv`, `test.csv`, `data_description.txt`
   from Kaggle into `data/raw/`. Verify row/column counts against the actual files (1,460 rows,
   81 columns in `train.csv`) — do not trust the proposal's numbers or a blog post, check them
   yourself in the notebook.
6. **Run exploratory data analysis** in [`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb):
   - Target distribution: confirm right-skew, evaluate the case for `log1p(SalePrice)`.
   - Relationships with `SalePrice`: location (`Neighborhood`, `MSZoning`), size (`GrLivArea`,
     `TotalBsmtSF`), quality (`OverallQual`, `KitchenQual`), age (`YearBuilt`), amenities.
   - Segment structure: which property categories (neighborhood, quality tier, price band)
     look like they'd behave differently — this directly feeds the secondary lens and later
     the segment error analysis.
7. **Maintain the EDA insight log** — every finding goes into
   [`docs/eda_insight_log.md`](../eda_insight_log.md) with its evidence (plot/stat) and its
   implication for the task, as you go, not retroactively.

## Deliverables

- Business problem, lens justification, and stakeholder analysis (written up, e.g. in
  `docs/plan/business_problem.md` or directly in the final report — team's choice).
- `data/raw/` populated and verified.
- `notebooks/01_eda.ipynb` completed.
- `docs/eda_insight_log.md` filled in.

## Docs to update

- [`docs/eda_insight_log.md`](../eda_insight_log.md)
