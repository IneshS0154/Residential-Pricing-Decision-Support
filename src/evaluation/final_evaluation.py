"""Final model evaluation, residual diagnostics, robustness and non-SHAP interpretation (notebook 06).

Run the heavy, cached part from the project root:   python -m src.evaluation.final_evaluation [--force]

Validation context (what the numbers mean)
  All predictions here are out-of-fold: every house is predicted by models that never saw it. For the tuned models they are
  the NESTED outer-fold predictions (hyperparameters chosen on outer-training rows only), so they estimate the performance of
  the "tune, then fit" PROCEDURE. They are NOT an independent final test: the same 1,460 rows informed the feature-set and
  model-family choices, and no untouched labelled data exists (test.csv has no labels).

Conventions
  residual = actual - predicted  (positive = underprediction, negative = overprediction), in USD and on the log1p scale
  predictions are medians: expm1 of the log1p prediction, no smearing correction
  APE = |actual - predicted| / actual
  segments are banded on PREDICTED price (the project rule, EDA F25); banding on actual price is reported only to show
  the regression-to-the-mean artefact it creates
  repeated CV gives each house 3 out-of-fold predictions (one per repeat); per-house diagnostics use repeat 0 (a genuine
  5-fold partition, one prediction per house) and say so; headline metrics are means of fold-level metrics over the 15 folds
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation import feature_selection as F
from src.evaluation import nested_tuning as NT
from src.evaluation.baseline_comparison import OUT as BASE_OUT
from src.evaluation.run_feature_selection import load
from src.features import preprocessing as P
from src.features.feature_sets import FEATURE_SETS
from src.models import tuning as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "final_evaluation"
EVAL_ID = "final_eval_v1"

# Pre-specified (fixed in code before any segment result was computed): rounded 25th and 90th percentiles of the
# SalePrice distribution (USD 129,975 and 278,000). They depend only on the label distribution, never on a model.
SEGMENT_EDGES = (130_000, 280_000)
SEGMENTS = ("low", "mid", "luxury")
FLAGGED_IDS = tuple(P.SUGGESTED_TRAINING_EXCLUSIONS)          # 524, 1299 (EDA F17, decision D2)
PRIMARY_REPEAT = 0

MODELS = {                                   # model id -> (label, source)
    "gbr_tuned": ("Gradient Boosting (tuned, nested)", "nested"),
    "enet_tuned": ("ElasticNet (tuned, nested)", "nested"),
    "lasso_fixed": ("Lasso (fixed baseline, not tuned)", "baseline"),
}

PERM_GROUPS = {
    "Garage": ["GarageCars", "GarageArea", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt"],
    "Living space": ["GrLivArea", "1stFlrSF", "2ndFlrSF", "TotRmsAbvGrd", "LowQualFinSF", "BedroomAbvGr", "KitchenAbvGr"],
    "Basement": ["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtQual", "BsmtCond", "BsmtExposure",
                 "BsmtFinType1", "BsmtFinType2"],
    "Bathrooms": ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
    "Property age": ["YearBuilt", "YearRemodAdd"],
    "Quality and condition": ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "KitchenQual", "HeatingQC"],
    "Location and lot": ["Neighborhood", "MSZoning", "Condition1", "LotArea", "LotFrontage", "LotShape", "LotConfig",
                         "LandContour", "LandSlope"],
    "Dwelling type and exterior": ["MSSubClass", "BldgType", "HouseStyle", "RoofStyle", "Foundation", "Exterior1st",
                                   "Exterior2nd", "MasVnrType", "MasVnrArea"],
    "Amenities and other": ["Fireplaces", "FireplaceQu", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch",
                            "Fence", "CentralAir", "Electrical", "Functional", "PavedDrive", "Heating", "Alley"],
}


# ---------------------------------------------------------------- core definitions
def segment_of(price, edges=SEGMENT_EDGES):
    """'low' (< edges[0]), 'mid' ([edges[0], edges[1])), 'luxury' (>= edges[1]) for USD prices."""
    p = np.asarray(price, dtype=float)
    return np.where(p < edges[0], "low", np.where(p < edges[1], "mid", "luxury"))


def add_errors(d):
    """Residual, error and segment columns from `y` (USD) and `pred_log`."""
    d = d.copy()
    d["y_log"] = np.log1p(d["y"])
    d["pred"] = np.expm1(d["pred_log"])
    d["residual_usd"] = d["y"] - d["pred"]                       # positive = underprediction
    d["residual_log"] = d["y_log"] - d["pred_log"]
    d["abs_err_usd"] = d["residual_usd"].abs()
    d["sq_err_usd"] = d["residual_usd"] ** 2
    d["ape"] = d["abs_err_usd"] / d["y"]
    d["segment_actual"] = segment_of(d["y"])
    d["segment_pred"] = segment_of(d["pred"])
    return d


def load_oof():
    """Long out-of-fold prediction table for the three models (15 folds, 3 predictions per house)."""
    df, X, y, folds = load()
    ids = df["Id"].values
    nested = pd.read_csv(NT.OUT / "nested_predictions.csv.gz")
    base = pd.read_csv(BASE_OUT / "expA_predictions.csv.gz")
    parts = [nested[nested.family == "gbr"].assign(model="gbr_tuned"),
             nested[nested.family == "elasticnet"].assign(model="enet_tuned"),
             base[base.model == "lasso"].assign(model="lasso_fixed")]
    d = pd.concat([p[["model", "fold", "row", "y", "pred_log"]] for p in parts], ignore_index=True)
    d["repeat"] = d["fold"] // 5
    d["Id"] = ids[d["row"].values]
    d["dataset_version"] = "train.csv, reduced feature set"
    return add_errors(d).sort_values(["model", "fold", "row"]).reset_index(drop=True)


def add_matched_segment(oof, ref_model="gbr_tuned"):
    """Add `segment_matched`: the segment (banded on the REFERENCE model's predicted price, same repeat) of each house, so
    every model is scored on exactly the same houses within each segment."""
    ref = oof[oof.model == ref_model][["repeat", "row", "segment_pred"]].rename(columns={"segment_pred": "segment_matched"})
    return oof.merge(ref, on=["repeat", "row"], how="left")


def fold_metrics(oof, exclude_ids=()):
    """Per model and fold: Log-RMSE, MAE, RMSE (USD), MAPE (%), R2 (USD). `exclude_ids` drops houses from the VALIDATION
    rows only (evaluation-only exclusion; predictions are unchanged)."""
    d = oof[~oof["Id"].isin(exclude_ids)] if len(exclude_ids) else oof
    rows = []
    for (m, k), g in d.groupby(["model", "fold"]):
        y, p = g["y"].values, g["pred"].values
        rows.append(dict(model=m, fold=k, repeat=k // 5, n=len(g), rmse_log=np.sqrt(np.mean(g["residual_log"] ** 2)), mae=np.mean(np.abs(y - p)),
                         rmse_usd=np.sqrt(np.mean((y - p) ** 2)), mape=100 * np.mean(np.abs(y - p) / y),
                         r2_usd=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))
    return pd.DataFrame(rows)


def segment_stats(g):
    return pd.Series(dict(n=len(g), rmse_log=np.sqrt(np.mean(g["residual_log"] ** 2)), mae=g["abs_err_usd"].mean(),
                          rmse_usd=np.sqrt(g["sq_err_usd"].mean()), mape=100 * g["ape"].mean(),
                          bias_usd=g["residual_usd"].mean(), bias_log=g["residual_log"].mean(),
                          median_ae=g["abs_err_usd"].median()))


def segment_table(oof, model, by="segment_pred"):
    """Segment metrics computed within each of the 3 repeats (each repeat predicts every house once), then averaged.
    Returns mean over repeats; `n` is the number of distinct houses per repeat (not 3x)."""
    d = oof[oof.model == model]
    per = d.groupby(["repeat", by]).apply(segment_stats, include_groups=False).reset_index()
    out = per.groupby(by).mean(numeric_only=True).drop(columns="repeat")
    out["rmse_log_range_over_repeats"] = per.groupby(by).rmse_log.apply(lambda s: s.max() - s.min())
    return out.reindex([s for s in SEGMENTS if s in out.index])


def bootstrap_segment_ci(oof, model, by="segment_pred", stat="mae", n_boot=2000, seed=0):
    """95% bootstrap interval over HOUSES (resampled within the segment) using the repeat-0 prediction of each house.
    Reflects sampling of houses, not model-fitting variability."""
    d = oof[(oof.model == model) & (oof.repeat == PRIMARY_REPEAT)]
    rng = np.random.default_rng(seed)
    res = {}
    for s in SEGMENTS:
        g = d[d[by] == s]
        if len(g) == 0:
            continue
        vals = np.empty(n_boot)
        for b in range(n_boot):
            gg = g.iloc[rng.integers(0, len(g), len(g))]
            vals[b] = {"mae": gg.abs_err_usd.mean(), "rmse_log": np.sqrt(np.mean(gg.residual_log ** 2)),
                       "bias_usd": gg.residual_usd.mean(), "mape": 100 * gg.ape.mean()}[stat]
        res[s] = (len(g), np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return pd.DataFrame(res, index=["n_houses", "ci_lo", "ci_hi"]).T


def error_quantiles(abs_err):
    a = np.asarray(abs_err, float)
    return pd.Series({"mean": a.mean(), "median": np.median(a), "p75": np.percentile(a, 75), "p90": np.percentile(a, 90),
                      "p95": np.percentile(a, 95), "max": a.max()})


def concentration(abs_err, top_frac=0.10):
    """Share of the total absolute error contributed by the worst `top_frac` of predictions."""
    a = np.sort(np.asarray(abs_err, float))[::-1]
    k = max(1, int(round(top_frac * len(a))))
    return float(a[:k].sum() / a.sum())


# ---------------------------------------------------------------- refits: permutation importance + training-exclusion
class RawPredictor:
    """Fitted preprocessing + model that maps RAW columns to a log1p prediction (engineered features are recomputed from
    whatever raw values it is given, so permuting raw columns perturbs them consistently)."""

    def __init__(self, prep, model):
        self.prep, self.model = prep, model

    def predict(self, X):
        return self.model.predict(self.prep.transform(X).values)


def _fit(family, params, X, ly, tr):
    prep = FEATURE_SETS[NT.FEATURE_SET].build("baseline", scale=T.SCALE[family]).fit(X.iloc[tr], ly[tr])
    model = T.make_estimator(family, params).fit(prep.transform(X.iloc[tr]).values, ly[tr])
    return RawPredictor(prep, model)


def _params_for(outer_row, family):
    out = {}
    for p, (lo, hi, kind) in T.SPACES[family].items():
        v = outer_row[f"best_{p}"]
        out[p] = int(round(v)) if kind.startswith("int") else float(v)
    return out


def refit_fold(family, k, tr, va, params, X, y, ids):
    """Refit the fold's tuned model on the outer training rows (reproduces the nested prediction), score permutation
    importance on the held-out rows, and refit once more with the flagged houses removed from TRAINING rows only."""
    ly = np.log1p(y)
    m = _fit(family, params, X, ly, tr)
    pred = m.predict(X.iloc[va])
    perm = F.grouped_permutation_importance(m, X.iloc[va], ly[va],
                                            {g: [c for c in cols if c in X.columns] for g, cols in PERM_GROUPS.items()},
                                            n_repeats=5, seed=T.SEED + k)
    tr2 = tr[~np.isin(ids[tr], FLAGGED_IDS)]
    pred2 = _fit(family, params, X, ly, tr2).predict(X.iloc[va])
    return dict(family=family, fold=k, row=va, pred_log=pred, pred_log_trainexcl=pred2, perm=perm, n_removed=len(tr) - len(tr2))


def run_refits(n_jobs=6):
    df, X, y, folds = load()
    ids = df["Id"].values
    outer = pd.read_csv(NT.OUT / "nested_outer_folds.csv", float_precision="round_trip")   # exact floats: GBR is ULP-sensitive
    res, t0 = [], time.perf_counter()
    for fam in T.FAMILIES:
        o = outer[outer.family == fam].set_index("fold")
        res += Parallel(n_jobs=n_jobs)(delayed(refit_fold)(fam, k, *folds[k], _params_for(o.loc[k], fam), X, y, ids) for k in range(len(folds)))
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.concat([pd.DataFrame(dict(family=r["family"], fold=r["fold"], row=r["row"], pred_log_refit=r["pred_log"],
                                        pred_log_trainexcl=r["pred_log_trainexcl"])) for r in res])
    pred.to_csv(OUT / "refit_predictions.csv.gz", index=False)
    check_refit_reproduces(pred)
    perm = pd.concat([r["perm"].rename("delta_logrmse").reset_index().rename(columns={"index": "group"}).assign(family=r["family"], fold=r["fold"]) for r in res])
    perm.to_csv(OUT / "grouped_permutation_importance.csv", index=False)
    return time.perf_counter() - t0


def check_refit_reproduces(refit, tol=1e-8):
    """The refits behind permutation importance must be the SAME models that produced the nested out-of-fold predictions."""
    nested = pd.read_csv(NT.OUT / "nested_predictions.csv.gz", float_precision="round_trip")
    m = nested.merge(refit, on=["family", "fold", "row"])
    worst = float((m["pred_log"] - m["pred_log_refit"]).abs().max())
    if len(m) != len(nested) or worst > tol:
        raise RuntimeError(f"refit does not reproduce the nested predictions (max abs diff {worst:.3e}, matched {len(m)}/{len(nested)})")
    return worst


# ---------------------------------------------------------------- linear-model interpretation (no SHAP)
def coefficient_table(bundle, X):
    """Coefficients of the final ElasticNet pipeline in model units, per standard deviation, and in ORIGINAL predictor units
    where that conversion is mathematically valid.

    Model units: the model is linear in the preprocessed columns z (robust-scaled; some numeric columns log1p'd first), target
    log1p(price). For a numeric column with fitted centre c and scale s and optional log1p transform g:
        z = (g(x) - c) / s     =>   d(log price) / d g(x) = coef / s       (log-price change per unit of g(x))
    * no log1p (g = identity): coef/s is the log-price change per original unit; 100*coef/s is approximately % price per unit
      (exact multiplicative effect exp(coef/s) - 1). It is NOT a dollar change.
    * log1p applied: coef/s is the change in log price per unit change in log1p(x), i.e. approximately an elasticity
      (% price change per 1% change in x) for large x; it is not per original unit.
    * ordinal codes: per one-level step. One-hot indicators: effect of the category indicator under the penalty; there is
      no single reference level, so exp(coef)-1 is an approximate multiplicative association, not a comparison with a baseline.
    Per-SD (standardised) coefficients = coef * std(z column) on the training data."""
    prep, model = bundle["prep"], bundle["model"]
    Z = prep.transform(X)
    ct = prep.named_steps["encode"]
    info = {}
    for name, trans, cols in ct.transformers_:
        if name in ("numeric", "ordinal"):
            scaler = trans.named_steps["scale"]
            mask = trans.named_steps["skew"].mask_ if name == "numeric" else np.zeros(len(cols), bool)
            for j, c in enumerate(cols):
                info[c] = dict(kind=name, center=float(scaler.center_[j]), scale=float(scaler.scale_[j]), log1p=bool(mask[j]))
        elif name in ("binary", "flags"):
            for c in cols:
                info[c] = dict(kind=name, center=0.0, scale=1.0, log1p=False)
    rows = []
    sd = Z.std()
    for c, b in zip(Z.columns, model.coef_):
        i = info.get(c)
        if i is None:                                    # one-hot / other expanded columns
            kind, unit_coef, note = "one-hot indicator", b, "approx. multiplicative association: exp(coef)-1, no single reference level"
        else:
            kind = i["kind"]
            unit_coef = b / i["scale"] if i["scale"] else np.nan
            note = ("per unit of log1p(x): ~ elasticity" if i["log1p"] else ("per ordinal level" if kind == "ordinal" else "per original unit"))
        rows.append(dict(feature=c, kind=kind, coef_model_units=b, coef_per_sd=b * sd[c], coef_original_units=unit_coef,
                         approx_pct_effect=(np.nan if (i or {}).get("log1p", False) else 100 * np.expm1(unit_coef)), log1p_input=(i or {}).get("log1p", False), unit_note=note))
    t = pd.DataFrame(rows)
    return t.assign(abs_per_sd=t.coef_per_sd.abs()).sort_values("abs_per_sd", ascending=False).drop(columns="abs_per_sd").reset_index(drop=True), info, Z


def verify_numeric_conversion(bundle, X, info, Z, n_rows=200):
    """Check z = (g(x)-c)/s reproduces the fitted numeric columns on rows without missing raw values (max abs difference)."""
    from sklearn.pipeline import Pipeline
    pre = Pipeline(bundle["prep"].steps[:-1]).transform(X)
    worst = 0.0
    for c, i in info.items():
        if i["kind"] != "numeric" or c not in pre.columns:
            continue
        x = pre[c].astype(float).iloc[:n_rows]
        ok = x.notna()
        g = np.log1p(x[ok]) if i["log1p"] else x[ok]
        worst = max(worst, float(np.abs((g - i["center"]) / i["scale"] - Z.loc[x.index[ok], c]).max()))
    return worst


def complexity(bundle_en, bundle_gbr, X, n_repeat=20):
    """Parameter-count indicators and prediction latency of the final (all-data) fitted pipelines on the 292-row validation size."""
    out = {}
    coef = bundle_en["model"].coef_
    out["enet_tuned"] = dict(n_parameters_total=int(len(coef) + 1), n_nonzero=int((coef != 0).sum()))
    est = bundle_gbr["model"]
    nodes = int(sum(t[0].tree_.node_count for t in est.estimators_))
    out["gbr_tuned"] = dict(n_parameters_total=nodes, n_nonzero=int(est.n_estimators), n_trees=int(est.n_estimators))
    sample = X.iloc[:292]
    for k, b in (("enet_tuned", bundle_en), ("gbr_tuned", bundle_gbr)):
        ts = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            b["model"].predict(b["prep"].transform(sample).values)
            ts.append(time.perf_counter() - t0)
        out[k]["predict_292_rows_s"] = float(np.median(ts))
    return out


def run_all(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and (OUT / "grouped_permutation_importance.csv").exists() and (OUT / "refit_predictions.csv.gz").exists():
        return
    secs = run_refits()
    oof = load_oof()
    oof.to_csv(OUT / "oof_predictions.csv.gz", index=False)
    meta = dict(eval_id=EVAL_ID, refit_seconds=round(secs, 1), segment_edges_usd=SEGMENT_EDGES, flagged_ids=FLAGGED_IDS,
                primary_repeat=PRIMARY_REPEAT, seed=T.SEED, models=MODELS, permutation=dict(n_repeats=5, groups=PERM_GROUPS))
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
