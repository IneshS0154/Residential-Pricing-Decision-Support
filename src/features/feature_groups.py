"""Business attribute groups for every column of the Ames dataset.

Single source of truth for the colour-coded reference sheet and, later, for
grouping feature importance (grouped permutation importance, grouped ablation) in the price decomposition.
"""

FEATURE_GROUPS = {
    "Identifier / Target": ["Id", "SalePrice"],
    "Location and lot": [
        "Neighborhood", "MSZoning", "LotFrontage", "LotArea", "LotShape",
        "LotConfig", "LandContour", "LandSlope", "Street", "Alley",
        "Condition1", "Condition2", "Utilities", "PavedDrive",
    ],
    "Dwelling type and structure": [
        "MSSubClass", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
        "Foundation", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea",
        "Functional", "Electrical", "Heating", "CentralAir",
    ],
    "Size": [
        "GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "TotalBsmtSF",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotRmsAbvGrd", "BedroomAbvGr",
        "KitchenAbvGr", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
        "GarageArea", "GarageCars",
    ],
    "Quality and condition": [
        "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual",
        "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
        "GarageFinish", "PoolQC", "Fence",
    ],
    "Age": ["YearBuilt", "YearRemodAdd", "GarageYrBlt"],
    "Amenities": [
        "Fireplaces", "GarageType", "WoodDeckSF", "OpenPorchSF",
        "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
        "MiscFeature", "MiscVal",
    ],
    "Sale context": ["MoSold", "YrSold", "SaleType", "SaleCondition"],
}

# Header fill colours (hex, no '#') per group.
GROUP_COLORS = {
    "Identifier / Target": "BFBFBF",
    "Location and lot": "9DC3E6",
    "Dwelling type and structure": "C5A3E0",
    "Size": "F4B183",
    "Quality and condition": "A9D18E",
    "Age": "FFD966",
    "Amenities": "F8A5C2",
    "Sale context": "FF9B9B",
}

COLUMN_TO_GROUP = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
