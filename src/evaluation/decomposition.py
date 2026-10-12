"""Exact additive per-property decomposition of the stored final ElasticNet prediction into business attribute groups.

No SHAP and no Shapley computation: the ElasticNet is linear on the log1p scale, so for a house with preprocessed feature vector z

    f(z) = b + sum_j w_j z_j                       (log1p price)
    f(z) - f(zbar) = sum_g C_g,   C_g = sum_{j in g} w_j (z_j - zbar_j)

where zbar is the mean preprocessed feature vector of the 1,460 training houses (the REFERENCE: a model input, not a real house) and g runs
over business groups. The identity is exact by construction (checked in `decompose`). Dollar contributions use the logarithmic-mean
(LMDI) weight L = (P - P_ref) / (f - f_ref) with P = exp(f): contribution_USD_g = L * C_g, which sums EXACTLY to the predicted price minus the
reference price and does not depend on the order of the groups.

What this is, and is not:
  * It explains the ElasticNet (the interpretable benchmark), NOT the working Gradient Boosting model; a tree ensemble has no exact additive
    decomposition from the stored artifacts without Shapley-type attributions, which this project does not use.
  * It uses the stored final pipeline (fitted on all 1,460 houses), so its predictions for training houses are in-sample: they describe how the
    model uses features, not how accurately it predicts. Validated accuracy is the nested out-of-fold evidence of notebooks 05-06.
  * It is a statement about a fitted model's association structure, not about causes of price, and not a land-vs-structure VALUE split
    (see docs/worked_examples.md).
Only attribute values of the house enter; no label does.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loading import load_raw
from src.evaluation.final_evaluation import PERM_GROUPS

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts" / "tuning" / "final_model_elasticnet_tuning_v1.joblib"

# Same nine business groups as the grouped permutation importance (notebook 06). Attributes that appear in the 172 output columns but were not in a
# permutation group (pool quality, miscellaneous feature) are assigned to "Amenities and other"; engineered columns follow their source attributes.
EXTRA = {"PoolQC": "Amenities and other", "MiscFeature": "Amenities and other", "HouseAge": "Property age", "RemodAge": "Property age",
         "IsRemodeled": "Property age", "TotalBathrooms": "Bathrooms"}
LAND_GROUP = "Location and lot"            # the only group of land-related attributes; everything else describes the building and its use


def attribute_group_map():
    m = {attr: g for g, attrs in PERM_GROUPS.items() for attr in attrs}
    m.update(EXTRA)
    return m


def column_group(col, amap=None):
    """Business group of an output column: exact attribute name or `<attribute>_<level>` (one-hot); longest attribute match wins."""
    amap = amap or attribute_group_map()
    hits = [a for a in amap if col == a or col.startswith(a + "_")]
    if not hits:
        raise KeyError(f"output column {col!r} has no business group")
    return amap[max(hits, key=len)]


def load_bundle(path=BUNDLE):
    return joblib.load(path)


def raw_predictors():
    return load_raw().drop(columns=["SalePrice", "Id"])


def decompose(rows, bundle=None, reference_rows=None):
    """Long DataFrame: one row per (house, group) with the log-scale contribution, its multiplicative factor and its LMDI dollar contribution.

    rows: raw predictor rows (no label needed). reference_rows: raw rows defining the reference vector zbar (default: all 1,460 training houses).
    Raises AssertionError if the additive identity fails."""
    bundle = bundle or load_bundle()
    prep, model = bundle["prep"], bundle["model"]
    ref = raw_predictors() if reference_rows is None else reference_rows
    zbar = prep.transform(ref).mean(axis=0)
    z = prep.transform(rows)
    w = pd.Series(model.coef_, index=z.columns)
    amap = attribute_group_map()
    group_of = pd.Series({c: column_group(c, amap) for c in z.columns})
    contrib = (z - zbar).mul(w, axis=1)                                   # per house, per output column: w_j (z_ij - zbar_j)
    by_group = contrib.T.groupby(group_of).sum().T                         # houses x groups
    f = model.intercept_ + z.values @ w.values
    f_ref = model.intercept_ + float(zbar.values @ w.values)
    assert np.allclose(by_group.sum(axis=1).values, f - f_ref, atol=1e-9), "additive identity failed on the log scale"
    price, price_ref = np.expm1(f), float(np.expm1(f_ref))
    d = f - f_ref
    lmdi = np.where(np.abs(d) > 1e-12, ((np.exp(f) - np.exp(f_ref)) / np.where(np.abs(d) > 1e-12, d, 1.0)), np.exp(f))
    usd = by_group.mul(lmdi, axis=0)
    assert np.allclose(usd.sum(axis=1).values, price - price_ref, atol=1e-6), "additive identity failed in USD"
    out = []
    for i, idx in enumerate(rows.index):
        for g in by_group.columns:
            out.append({"row_index": idx, "group": g, "log_contribution": by_group.iloc[i][g], "multiplicative_factor": float(np.exp(by_group.iloc[i][g])),
                        "usd_contribution": usd.iloc[i][g], "prediction_usd": price[i], "reference_prediction_usd": price_ref})
    return pd.DataFrame(out)


def land_vs_building(dec):
    """Aggregate of the decomposition into the land-related group and all other (building / use) groups. NOT a land value estimate."""
    g = dec.assign(part=np.where(dec["group"] == LAND_GROUP, "Location-and-lot attributes", "Building and use attributes"))
    return g.groupby(["row_index", "part"])[["log_contribution", "usd_contribution"]].sum().reset_index()
