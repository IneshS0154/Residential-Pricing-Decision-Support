"""Leakage-safe preprocessing for the Ames housing price task.

Typical use (modelling team)::

    from src.data.loading import load_raw
    from src.features.preprocessing import build_preprocessor

    df = load_raw()
    X, y = df.drop(columns=["SalePrice", "Id"]), df["SalePrice"]
    prep = build_preprocessor("neighborhood_indicator")      # unfitted
    model = Pipeline([("prep", prep), ("model", Ridge(alpha=10))])
    # fit `model` on training folds only; never fit `prep` on the full data before evaluation

Input must come from ``src.data.loading.load_raw`` so that the string "NA" is still
present in the 14 columns where it means "feature absent".
Every statistic (medians, modes, KNN donors, encoder categories, scaler parameters,
skew-based column choices) is learned in ``fit`` from the training rows only.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder

NO_FEATURE = "NoFeature"
REFERENCE_YEAR = 2010          # valuation reference year = last year in the data; the real sale year is not assumed known

# ---------------------------------------------------------------- column policy
SALE_CONTEXT = ["SaleType", "SaleCondition", "MoSold", "YrSold"]     # unknown when a listing price is set
IDENTIFIERS_AND_TARGET = ["Id", "SalePrice"]
ALWAYS_DROP_FEATURES = ["GarageYrBlt"]                                # redundant with YearBuilt; 9 rows dated before the house (EDA F16, F19)
LOW_INFORMATION = {                                                   # EDA F10, F11
    "Utilities": "near-constant (99.9%)", "Street": "near-constant (99.6%)",
    "Condition2": "near-constant (99.0%)", "RoofMatl": "near-constant (98.2%)",
    "PoolArea": "99.5% zero", "3SsnPorch": "98.4% zero",
    "LowQualFinSF": "98.2% zero", "MiscVal": "96.4% zero",
}

# Removed by default (`reduce_redundancy=True`) so that no two numeric inputs are exact duplicates or exact sums of others.
# Keeps the stronger-signal member of each correlated pair and the totals rather than their parts (EDA F16; section 11b of notebook 02).
REDUNDANT = {
    "TotRmsAbvGrd": "r = 0.83 with GrLivArea (kept)",
    "GarageArea": "r = 0.88 with GarageCars (kept; weaker price signal)",
    "1stFlrSF": "part of GrLivArea (= 1stFlrSF + 2ndFlrSF + LowQualFinSF); r = 0.82 with TotalBsmtSF",
    "2ndFlrSF": "part of GrLivArea",
    "BsmtFinSF1": "part of TotalBsmtSF (= BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF)",
    "BsmtFinSF2": "part of TotalBsmtSF",
    "BsmtUnfSF": "part of TotalBsmtSF",
    "FullBath": "part of TotalBathrooms (= FullBath + 0.5 HalfBath + BsmtFullBath + 0.5 BsmtHalfBath)",
    "HalfBath": "part of TotalBathrooms",
    "BsmtFullBath": "part of TotalBathrooms",
    "BsmtHalfBath": "part of TotalBathrooms",
    "TotalSF": "= TotalBsmtSF + GrLivArea once LowQualFinSF is dropped (exact linear combination of kept columns)",
}

# "NA" means "feature absent" in these groups. evidence = numeric columns that can confirm absence.
STRUCTURAL_GROUPS = {
    "garage":    dict(cats=["GarageType", "GarageFinish", "GarageQual", "GarageCond"],
                      evidence=["GarageArea", "GarageCars"], numerics=["GarageArea", "GarageCars"]),
    "basement":  dict(cats=["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
                      evidence=["TotalBsmtSF"],
                      numerics=["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]),
    "fireplace": dict(cats=["FireplaceQu"], evidence=["Fireplaces"], numerics=[]),
    "pool":      dict(cats=["PoolQC"], evidence=["PoolArea"], numerics=["PoolArea"]),
    "misc":      dict(cats=["MiscFeature"], evidence=["MiscVal"], numerics=["MiscVal"]),
    "alley":     dict(cats=["Alley"], evidence=None, numerics=[]),     # no related column: rely on data_description.txt
    "fence":     dict(cats=["Fence"], evidence=None, numerics=[]),
}
STRUCTURAL_COLUMNS = [c for g in STRUCTURAL_GROUPS.values() for c in g["cats"]]

QUALITY = [NO_FEATURE, "Po", "Fa", "TA", "Gd", "Ex"]
ORDINAL_ORDERS = {
    "ExterQual": QUALITY[1:], "ExterCond": QUALITY[1:], "HeatingQC": QUALITY[1:], "KitchenQual": QUALITY[1:],
    "BsmtQual": QUALITY, "BsmtCond": QUALITY, "FireplaceQu": QUALITY, "GarageQual": QUALITY,
    "GarageCond": QUALITY, "PoolQC": QUALITY,
    "BsmtExposure": [NO_FEATURE, "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": [NO_FEATURE, "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": [NO_FEATURE, "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": [NO_FEATURE, "Unf", "RFn", "Fin"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"], "LandSlope": ["Sev", "Mod", "Gtl"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"], "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
}
BINARY_ORDERS = {"CentralAir": ["N", "Y"], "Street": ["Grvl", "Pave"]}
NOMINAL = ["MSSubClass", "MSZoning", "Alley", "LotConfig", "LandContour", "Neighborhood", "Condition1",
           "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
           "MasVnrType", "Foundation", "Heating", "Electrical", "GarageType", "Fence", "MiscFeature"]
NUMERIC = ["LotFrontage", "LotArea", "OverallQual", "OverallCond", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
           "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
           "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
           "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
           "PoolArea", "MiscVal", "TotalSF", "TotalBathrooms", "HouseAge", "RemodAge"]
FLAGS = ["IsRemodeled", "LotFrontage_Missing"]
# Two unfinished (Partial) sales, 4,700 and 5,600 sq ft, that sold for $28-40 per sq ft vs $166-175 for comparable large homes (EDA F17).
# Not removed by the pipeline; the modelling team may exclude them from TRAINING rows (decision D2).
SUGGESTED_TRAINING_EXCLUSIONS = [524, 1299]
YEAR_FEATURES = ("age", "raw")      # age: HouseAge/RemodAge (default); raw: YearBuilt/YearRemodAdd
ENGINEERED = ["TotalSF", "TotalBathrooms", "HouseAge", "RemodAge", "IsRemodeled"]


def _is_marker(s):
    """True where the cell holds the raw 'NA' string or a real NaN."""
    return s.isna() | (s.astype(object) == "NA")


# ---------------------------------------------------------------- custom transformers
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns if present (missing columns are ignored)."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([c for c in self.feature_names_in_ if c not in set(self.columns)], dtype=object)


class StructuralAbsenceImputer(BaseEstimator, TransformerMixin):
    """Resolve the string 'NA' in the 14 structural columns.

    * Absence is *confirmed* when related numeric columns are all 0 (e.g. GarageArea = GarageCars = 0),
      or when the numeric columns are themselves missing and every category in the group is 'NA',
      or (Alley, Fence) by documentation. Confirmed absence -> 'NoFeature' and missing related numerics -> 0.
    * 'NA' on a property that has the facility is *genuinely unknown*. It is replaced by the most frequent
      category among training rows that really have the facility (never by 'NoFeature').
    * MasVnrArea is 0 when MasVnrType is 'None'.
    """

    def __init__(self, groups=None):
        self.groups = groups

    def _groups(self):
        return self.groups or STRUCTURAL_GROUPS

    def _absent(self, X, spec):
        cats = [c for c in spec["cats"] if c in X.columns]
        all_marker = pd.concat([_is_marker(X[c]) for c in cats], axis=1).all(axis=1)
        ev = [c for c in (spec["evidence"] or []) if c in X.columns]
        if spec["evidence"] is None:
            return all_marker
        zero = (X[ev] == 0).all(axis=1)
        all_nan = X[ev].isna().all(axis=1)
        return zero | (all_nan & all_marker)

    def fit(self, X, y=None):
        self.existing_mode_ = {}
        for spec in self._groups().values():
            for c in spec["cats"]:
                real = X.loc[~_is_marker(X[c]), c]
                self.existing_mode_[c] = real.mode().iloc[0] if len(real) else NO_FEATURE
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for spec in self._groups().values():
            absent = self._absent(X, spec)
            for c in spec["cats"]:
                marker = _is_marker(X[c])
                X[c] = X[c].astype(object)
                X.loc[marker & absent, c] = NO_FEATURE
                X.loc[marker & ~absent, c] = self.existing_mode_[c]       # facility exists, category unknown
            for c in spec["numerics"]:
                X.loc[absent & X[c].isna(), c] = 0
        if {"MasVnrType", "MasVnrArea"} <= set(X.columns):
            X.loc[X.MasVnrArea.isna() & (X.MasVnrType == "None"), "MasVnrArea"] = 0
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    def classify(self, X):
        """Per-column counts of confirmed-absent vs unknown 'NA' cells (for reporting)."""
        rows = []
        for spec in self._groups().values():
            absent = self._absent(X, spec)
            for c in spec["cats"]:
                m = _is_marker(X[c])
                rows.append((c, int(m.sum()), int((m & absent).sum()), int((m & ~absent).sum())))
        return pd.DataFrame(rows, columns=["feature", "n_NA", "structural_confirmed", "unknown_facility_exists"]).set_index("feature")


class AddMissingIndicator(BaseEstimator, TransformerMixin):
    """Append `<col>_Missing` (1 if the original value is NaN). Must run BEFORE imputation."""

    def __init__(self, columns=("LotFrontage",), suffix="_Missing"):
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c + self.suffix] = X[c].isna().astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [c + self.suffix for c in self.columns], dtype=object)


class NeighborhoodMedianImputer(BaseEstimator, TransformerMixin):
    """Fill `column` with the median of its `group` (default Neighborhood), learned from training rows.
    A group with no observed value, or one never seen in training, falls back to the global training median."""

    def __init__(self, column="LotFrontage", group="Neighborhood"):
        self.column = column
        self.group = group

    def fit(self, X, y=None):
        self.global_median_ = float(X[self.column].median())
        self.medians_ = X.groupby(self.group)[self.column].median().dropna()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group].map(self.medians_).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class GlobalMedianImputer(BaseEstimator, TransformerMixin):
    """Single-column global median imputation (used only for the imputer comparison)."""

    def __init__(self, column="LotFrontage"):
        self.column = column

    def fit(self, X, y=None):
        self.median_ = float(X[self.column].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].fillna(self.median_)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class KNNFrontageImputer(BaseEstimator, TransformerMixin):
    """Impute `target` from the k most similar training properties (similarity on `features`).

    Area columns are log1p-transformed (heavy right skew), all columns are standardised with
    training statistics so no feature dominates the distance, and the imputed value is converted
    back to the original unit. SalePrice is never used."""

    def __init__(self, target="LotFrontage", features=("LotArea", "GrLivArea", "OverallQual"),
                 log_cols=("LotArea", "GrLivArea"), n_neighbors=5, scale=True, weights="uniform"):
        self.target = target
        self.features = features
        self.log_cols = log_cols
        self.n_neighbors = n_neighbors
        self.scale = scale
        self.weights = weights

    def _matrix(self, X):
        cols = [self.target, *self.features]
        M = X[cols].astype(float).copy()
        for c in self.log_cols:
            M[c] = np.log1p(M[c])
        return M.values

    def fit(self, X, y=None):
        M = self._matrix(X)
        self.scaler_ = StandardScaler().fit(M) if self.scale else None
        Ms = self.scaler_.transform(M) if self.scale else M
        self.knn_ = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights).fit(Ms)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        miss = X[self.target].isna()
        if not miss.any():
            return X
        M = self._matrix(X)
        Ms = self.scaler_.transform(M) if self.scale else M
        out = self.knn_.transform(Ms)[:, 0]
        if self.scale:
            out = out * self.scaler_.scale_[0] + self.scaler_.mean_[0]
        X.loc[miss, self.target] = out[miss.values]
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class PropertyFeatureBuilder(BaseEstimator, TransformerMixin):
    """Type fixing and derived features. Uses only attributes known at listing time.

    TotalSF         = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    TotalBathrooms  = FullBath + 0.5 HalfBath + BsmtFullBath + 0.5 BsmtHalfBath
    HouseAge        = reference_year - YearBuilt        (reference year, not the unknown sale year)
    RemodAge        = reference_year - YearRemodAdd
    IsRemodeled     = YearRemodAdd != YearBuilt
    MSSubClass is cast to string (nominal code). With year_features="age" (default) the two raw year columns are
    then dropped (exact linear transforms of HouseAge / RemodAge); year_features="raw" keeps YearBuilt and
    YearRemodAdd instead and creates no HouseAge / RemodAge.
    garage_age=True additionally creates GarageAge = reference_year - GarageYrBlt (NaN for houses without a garage;
    median-imputed downstream). GarageYrBlt itself is removed by the pruning step as before."""

    def __init__(self, reference_year=REFERENCE_YEAR, year_features="age", garage_age=False):
        self.reference_year = reference_year
        self.year_features = year_features
        self.garage_age = garage_age

    def fit(self, X, y=None):
        if self.year_features not in YEAR_FEATURES:
            raise ValueError(f"year_features must be one of {YEAR_FEATURES}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalBathrooms"] = X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        if self.year_features == "age":
            X["HouseAge"] = (self.reference_year - X["YearBuilt"]).clip(lower=0)
            X["RemodAge"] = (self.reference_year - X["YearRemodAdd"]).clip(lower=0)
        if self.garage_age:
            X["GarageAge"] = (self.reference_year - X["GarageYrBlt"]).clip(lower=0)
        X["MSSubClass"] = X["MSSubClass"].astype(str)
        return X.drop(columns=["YearBuilt", "YearRemodAdd"]) if self.year_features == "age" else X

    def get_feature_names_out(self, input_features=None):
        age = self.year_features == "age"
        keep = [c for c in self.feature_names_in_ if not (age and c in ("YearBuilt", "YearRemodAdd"))]
        new = ["TotalSF", "TotalBathrooms", "IsRemodeled"] + (["HouseAge", "RemodAge"] if age else [])
        return np.asarray(keep + new + (["GarageAge"] if self.garage_age else []), dtype=object)


class SkewLog1p(BaseEstimator, TransformerMixin):
    """log1p the non-negative columns whose training skew exceeds `threshold` (chosen in fit)."""

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(getattr(X, "columns", range(np.shape(X)[1])), dtype=object)
        X = np.asarray(X, dtype=float)
        sk = stats.skew(X, axis=0, nan_policy="omit")
        self.mask_ = (np.asarray(sk) > self.threshold) & (np.nanmin(X, axis=0) >= 0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[:, self.mask_] = np.log1p(X[:, self.mask_])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


# ---------------------------------------------------------------- pipeline factory
STRATEGIES = ("baseline", "neighborhood", "neighborhood_indicator", "knn")


def _scaler(kind):
    return {"standard": StandardScaler(), "robust": RobustScaler(), None: "passthrough"}[kind]


NEIGHBORHOOD_ENCODINGS = ("onehot", "target", "both", "none")


def build_preprocessor(strategy="baseline", scale="robust", drop_low_information=True,
                       knn_neighbors=5, min_frequency=10, skew_threshold=0.75, neighborhood_encoding="onehot",
                       reduce_redundancy=True, drop_columns=None, year_features="age", garage_age=False):
    """Return an UNFITTED preprocessing Pipeline (DataFrame in, DataFrame out).

    strategy: baseline | neighborhood | neighborhood_indicator | knn   (how LotFrontage is handled)
    scale:    'robust' (default, justified in notebook section 13) | 'standard' | None  (None for tree models)
    reduce_redundancy: drop the columns in REDUNDANT (default True). No accuracy effect was measured (notebook 02,
        section 11b); it removes exact linear dependencies and keeps coefficients / attributions interpretable.
    neighborhood_encoding: 'onehot' (default) | 'target' | 'both' | 'none'.
        'target' uses sklearn's TargetEncoder (smoothed, cross-fitted inside fit, so the encoding of a
        training row never uses its own price). The pipeline then needs `y` (log1p price) at fit time.
    drop_columns: explicit iterable of columns to prune (original or engineered names). When given it REPLACES the
        drop_low_information / reduce_redundancy rules; used by src.features.feature_sets (full / domain_informed / ablations).
    year_features: 'age' (default; HouseAge, RemodAge) | 'raw' (YearBuilt, YearRemodAdd).
    garage_age: add GarageAge = reference year - GarageYrBlt (default False; GarageYrBlt is otherwise dropped).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if neighborhood_encoding not in NEIGHBORHOOD_ENCODINGS:
        raise ValueError(f"neighborhood_encoding must be one of {NEIGHBORHOOD_ENCODINGS}")
    if year_features not in YEAR_FEATURES:
        raise ValueError(f"year_features must be one of {YEAR_FEATURES}")
    if drop_columns is not None:
        dropped = set(drop_columns)
    else:
        dropped = set(LOW_INFORMATION) if drop_low_information else set()
        dropped |= set(REDUNDANT) if reduce_redundancy else set()
    keep = lambda cols: [c for c in cols if c not in dropped]

    steps = [
        ("guard", ColumnDropper(IDENTIFIERS_AND_TARGET + SALE_CONTEXT)),
        ("structural", StructuralAbsenceImputer()),
    ]
    flags = [f for f in FLAGS if f != "LotFrontage_Missing"]
    if strategy == "neighborhood_indicator":
        steps.append(("indicator", AddMissingIndicator(("LotFrontage",))))
        flags.append("LotFrontage_Missing")
    if strategy in ("neighborhood", "neighborhood_indicator"):
        steps.append(("lotfrontage", NeighborhoodMedianImputer()))
    elif strategy == "knn":
        steps.append(("lotfrontage", KNNFrontageImputer(n_neighbors=knn_neighbors)))
    steps.append(("features", PropertyFeatureBuilder(year_features=year_features, garage_age=garage_age)))
    steps.append(("prune", ColumnDropper(ALWAYS_DROP_FEATURES + sorted(dropped))))

    ordinal = keep(list(ORDINAL_ORDERS))
    binary = keep(list(BINARY_ORDERS))
    nominal = [c for c in keep(NOMINAL) if not (c == "Neighborhood" and neighborhood_encoding in ("target", "none"))]
    te_cols = ["Neighborhood"] if neighborhood_encoding in ("target", "both") else []
    numeric_all = list(NUMERIC)
    if year_features == "raw":
        numeric_all = [c for c in numeric_all if c not in ("HouseAge", "RemodAge")] + ["YearBuilt", "YearRemodAdd"]
    if garage_age:
        numeric_all.append("GarageAge")
    numeric = keep(numeric_all)
    ct = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("skew", SkewLog1p(skew_threshold)),
                              ("scale", _scaler(scale))]), numeric),
        ("ordinal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OrdinalEncoder(categories=[ORDINAL_ORDERS[c] for c in ordinal],
                                                        handle_unknown="use_encoded_value", unknown_value=np.nan)),
                              ("impute2", SimpleImputer(strategy="median")),
                              ("scale", _scaler(scale))]), ordinal),
        ("binary", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("encode", OrdinalEncoder(categories=[BINARY_ORDERS[c] for c in binary],
                                                       handle_unknown="use_encoded_value", unknown_value=np.nan)),
                             ("impute2", SimpleImputer(strategy="most_frequent"))]), binary),
        ("nominal", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                                       sparse_output=False))]), nominal),
        ("flags", "passthrough", flags),
    ] + ([("neighborhood_te", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("te", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=42)),
                                        ("scale", _scaler(scale))]), te_cols)] if te_cols else []),
        remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    steps.append(("encode", ct))
    return Pipeline(steps)
