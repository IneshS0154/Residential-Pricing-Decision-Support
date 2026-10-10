"""Load the Ames CSVs with 'NA' interpreted correctly.

In the raw files the string "NA" has two meanings:
  * a documented category meaning the feature is absent (no alley, no garage, ...)
  * a genuinely missing value (LotFrontage, MasVnrArea, MasVnrType, Electrical,
    and GarageYrBlt for houses with no garage)
Pandas' defaults turn all of them into NaN. Here "NA" is kept as a string only in
columns where it is a documented category; everywhere else it becomes NaN.
The literal string "None" (MasVnrType) is always kept as a category.
"""
from pathlib import Path

import pandas as pd

# Columns where "NA" is a documented level meaning "feature absent".
NA_IS_CATEGORY = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "Dataset" / "train.csv"
TEST_PATH = ROOT / "Dataset" / "test.csv"


def load_raw(path=TRAIN_PATH) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    na_values = {c: ["", "NA"] for c in header if c not in NA_IS_CATEGORY}
    na_values.update({c: [""] for c in NA_IS_CATEGORY})
    return pd.read_csv(path, keep_default_na=False, na_values=na_values)
