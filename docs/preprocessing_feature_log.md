# Preprocessing and Feature Log

Owner: IT24102584 (built) — independently verified by IT24103989 (no fit outside CV folds)

Record each preprocessing/feature decision: options considered, choice made, and evidence/reason.

## Leakage & realism audit

List every candidate predictor excluded because it would not be known at listing time
(e.g. SaleType, SaleCondition, MoSold, YrSold — sale-context variables) or because it is
derived from SalePrice.

| Variable | Known at listing time? | Included? | Reason |
|---|---|---|---|
| SaleType | No | No | Sale-context, unknown at listing |
| SaleCondition | No | No | Sale-context, unknown at listing |

## Missing value handling

| Variable | Missing marker means absence or unrecorded? | Imputation approach | Alternative considered |
|---|---|---|---|

## Encoding

| Variable | Ordinal / Nominal | Encoding used | Reason |
|---|---|---|---|

## Engineered features

| Feature | Derived from | Reason |
|---|---|---|
| TotalArea | GrLivArea + TotalBsmtSF | Combines above/below grade living space |
| TotalBath | Full/half bath counts | Single comparable bathroom metric |
| PropertyAge | YrSold - YearBuilt | Age at time of sale |
