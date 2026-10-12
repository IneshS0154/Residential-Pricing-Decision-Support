"""Prediction intervals: quantile gradient boosting and split conformalized quantile regression (CQR).
Run the cached experiments from the project root:   python -m src.evaluation.intervals [--force]

Target and scale
  All interval models are fitted on log1p(SalePrice). Conformity scores, the conformal correction and the endpoints are on the
  LOG1P scale; USD endpoints are expm1 of the log endpoints (monotone, so coverage is identical on both scales).

Protocol (recreated inside every outer partition; no label of an evaluation house is ever used for fitting or calibration)
  outer fold (shared 15 folds of `get_folds`)  ->  evaluation rows (validation fold)
  outer training rows are split ONCE per fold, stratified by price band, into
      proper-training (75%)  : fits the preprocessing and BOTH quantile models (alpha 0.10 and 0.90)
      calibration (25%)      : used only to compute conformity scores s_i = max(L_i - y_i, y_i - U_i)
  the same evaluation rows are scored by both methods and both methods share the same base quantile models:
      Method A  quantile GB    [L(x), U(x)]
      Method B  split CQR      [L(x) - q_hat, U(x) + q_hat],  q_hat = k-th smallest score, k = ceil((n_cal + 1)(1 - alpha))
  If k > n_cal the quantile is +inf (documented, never an invalid index).
  Quantile crossing (L > U): endpoints are rearranged (sorted) BEFORE scoring and the number of crossings is reported.
  Scores are NOT clipped at zero (the standard CQR procedure; negative scores let the interval shrink).

What is and is not guaranteed
  Split CQR gives MARGINAL coverage >= 1 - alpha under exchangeability of calibration and evaluation houses, with an upper bound
  about 1 - alpha + 1/(n_cal + 1) when scores are continuous. It does not guarantee coverage within any neighbourhood, price band or
  single property, and does not apply under distribution shift (e.g. later sale years) without further assumptions.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from fractions import Fraction
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.evaluation import nested_tuning as NT
from src.evaluation.cv import price_bands
from src.evaluation.run_feature_selection import load
from src.features.feature_sets import FEATURE_SETS
from src.models import tuning as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "intervals"
EXPERIMENT_ID = "intervals_v1"
SEED = 42
ALPHA = 0.20                     # 80% nominal marginal coverage
Q_LO, Q_HI = 0.10, 0.90
CAL_FRAC = 0.25
# Fixed, NOT tuned for quantile loss and not tuned on any evaluation house: the configuration used in notebook 03's ablations.
QUANTILE_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=5)
# Pre-specified SENSITIVITY configuration (declared before any evaluation coverage was inspected): a smoother base model, to show how
# much the raw-quantile coverage depends on the base configuration. It is reported separately and never used to choose the primary one.
QUANTILE_PARAMS_REGULARISED = dict(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, min_samples_leaf=20)
FLAGGED_IDS = (524, 1299)        # EDA F17 / decision D2; kept in all primary results
TEMPORAL_LAST_TRAIN_YEAR = 2008  # the split defined in notebook 03 (train YrSold <= 2008, test 2009-2010)


# ---------------------------------------------------------------- conformal core
def conformal_k(n, alpha=ALPHA):
    """Order-statistic index k = ceil((n + 1)(1 - alpha)), computed in exact rational arithmetic (float ceil((n+1)*0.8)
    is wrong whenever the product lands on an integer, e.g. n = 9)."""
    a = Fraction(str(alpha))
    x = (n + 1) * (1 - a)
    return int(-(-x.numerator // x.denominator))


def conformal_quantile(scores, alpha=ALPHA):
    """Finite-sample conformal quantile of calibration scores: the k-th smallest, or +inf when k > n (explicit, not an
    invalid index). Returns (q_hat, k)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = conformal_k(n, alpha)
    if n == 0 or k > n:
        return np.inf, k
    return float(s[k - 1]), k


def conformity_scores(lower, upper, y):
    """s_i = max(L_i - y_i, y_i - U_i): negative if y_i is inside the interval, positive by the distance outside."""
    return np.maximum(np.asarray(lower) - np.asarray(y), np.asarray(y) - np.asarray(upper))


def rearrange(lower, upper):
    """Quantile-crossing policy: sort the two endpoints. Returns (lower, upper, n_crossed)."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    return np.minimum(lo, hi), np.maximum(lo, hi), int((lo > hi).sum())


def cqr_interval(lower, upper, q_hat):
    """[L - q_hat, U + q_hat] on the scale the scores were computed on."""
    return np.asarray(lower) - q_hat, np.asarray(upper) + q_hat


# ---------------------------------------------------------------- base quantile models
def make_quantile_model(alpha, seed=SEED, params=None):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=seed, **(params or QUANTILE_PARAMS))


def fit_split_cqr(X_train, y_log_train, y_raw_train, X_eval, seed, cal_frac=CAL_FRAC, alpha=ALPHA, params=None):
    """Fit both quantile models on a proper-training part, calibrate on a disjoint calibration part, and return intervals
    for X_eval. All inputs are TRAINING-side data except X_eval (features only; no evaluation label is passed in)."""
    idx = np.arange(len(X_train))
    proper, cal = train_test_split(idx, test_size=cal_frac, stratify=price_bands(y_raw_train), random_state=seed)
    prep = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X_train.iloc[proper], y_log_train[proper])
    Zp, Zc, Ze = (prep.transform(X_train.iloc[proper]).values, prep.transform(X_train.iloc[cal]).values, prep.transform(X_eval).values)
    lo_m = make_quantile_model(Q_LO, params=params).fit(Zp, y_log_train[proper])
    hi_m = make_quantile_model(Q_HI, params=params).fit(Zp, y_log_train[proper])
    Lc, Uc, cross_cal = rearrange(lo_m.predict(Zc), hi_m.predict(Zc))
    scores = conformity_scores(Lc, Uc, y_log_train[cal])
    q_hat, k = conformal_quantile(scores, alpha)
    Le, Ue, cross_eval = rearrange(lo_m.predict(Ze), hi_m.predict(Ze))
    Lq, Uq = cqr_interval(Le, Ue, q_hat)
    info = dict(q_hat=q_hat, k=k, n_cal=len(cal), n_proper=len(proper), cross_cal=cross_cal, cross_eval=cross_eval,
                raw_cal_coverage=float(np.mean((y_log_train[cal] >= Lc) & (y_log_train[cal] <= Uc))),
                proper_idx=proper, cal_idx=cal)
    return dict(L_raw=Le, U_raw=Ue, L_cqr=Lq, U_cqr=Uq), info


def run_fold(k, tr, va, X, y, params=None):
    ly = np.log1p(y)
    iv, info = fit_split_cqr(X.iloc[tr], ly[tr], y[tr], X.iloc[va], seed=SEED + k, params=params)
    d = pd.DataFrame(dict(fold=k, repeat=k // 5, row=va, y=y[va], y_log=ly[va], **iv))
    for c in ("q_hat", "n_cal", "n_proper", "cross_cal", "cross_eval", "raw_cal_coverage"):
        d[c] = info[c]
    return d


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    """Wilson 95% interval for a binomial proportion (sampling uncertainty of houses only)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def add_usd(d):
    d = d.copy()
    for m in ("raw", "cqr"):
        d[f"lo_{m}"], d[f"hi_{m}"] = np.expm1(d[f"L_{m}"]), np.expm1(d[f"U_{m}"])
        d[f"width_{m}"] = d[f"hi_{m}"] - d[f"lo_{m}"]
        d[f"cover_{m}"] = (d["y"] >= d[f"lo_{m}"]) & (d["y"] <= d[f"hi_{m}"])
        d[f"relwidth_{m}"] = d[f"width_{m}"] / d["point"]
        a2 = 2 / ALPHA
        d[f"winkler_{m}"] = d[f"width_{m}"] + a2 * np.maximum(d[f"lo_{m}"] - d["y"], 0) + a2 * np.maximum(d["y"] - d[f"hi_{m}"], 0)
        d[f"miss_{m}"] = np.where(d["y"] < d[f"lo_{m}"], "below", np.where(d["y"] > d[f"hi_{m}"], "above", "inside"))
    return d


def interval_metrics(d, method):
    """Coverage, widths and interval score for one method on the rows of `d` (pass the rows of ONE repeat)."""
    cov = d[f"cover_{method}"]
    lo, hi = wilson(int(cov.sum()), len(d))
    return pd.Series(dict(n=len(d), coverage=cov.mean(), coverage_gap_pp=100 * (cov.mean() - (1 - ALPHA)), cov_ci_lo=lo, cov_ci_hi=hi,
                          mean_width_usd=d[f"width_{method}"].mean(), median_width_usd=d[f"width_{method}"].median(),
                          median_rel_width_pct=100 * d[f"relwidth_{method}"].median(), mean_winkler_usd=d[f"winkler_{method}"].mean(),
                          mean_log_width=(d[f"U_{method}"] - d[f"L_{method}"]).mean()))


def metrics_over_repeats(d, method, by=None):
    """Metrics computed within each CV repeat (each repeat evaluates every house exactly once) and averaged over repeats.
    The Wilson interval is that of repeat 0 (house sampling only)."""
    rows = []
    for r, g in d.groupby("repeat"):
        if by is None:
            m = interval_metrics(g, method); m["repeat"] = r; rows.append(m.to_frame().T)
        else:
            for key, gg in g.groupby(by, observed=True):
                m = interval_metrics(gg, method); m["repeat"] = r; m[by] = key; rows.append(m.to_frame().T)
    t = pd.concat(rows, ignore_index=True)
    num = [c for c in t.columns if c not in ("repeat", by)]
    t[num] = t[num].astype(float)
    keys = [by] if by else []
    mean = t.groupby(keys)[num].mean() if by else t[num].mean().to_frame().T
    first = t[t.repeat == 0].set_index(by)[["cov_ci_lo", "cov_ci_hi"]] if by else t[t.repeat == 0][["cov_ci_lo", "cov_ci_hi"]]
    mean[["cov_ci_lo", "cov_ci_hi"]] = first.values if not by else first.reindex(mean.index).values
    mean["coverage_sd_over_repeats"] = (t.groupby(keys)["coverage"].std() if by else pd.Series([t.coverage.std()])).values
    return mean


# ---------------------------------------------------------------- experiments
def point_predictions():
    """Point prediction = tuned Gradient Boosting nested outer-fold prediction (05), same folds: identical for both methods."""
    p = pd.read_csv(NT.OUT / "nested_predictions.csv.gz", float_precision="round_trip")
    p = p[p.family == "gbr"][["fold", "row", "pred_log"]].rename(columns={"pred_log": "point_log"})
    p["point"] = np.expm1(p["point_log"])
    return p


def run_cv(n_jobs=6, params=None):
    df, X, y, folds = load()
    res = Parallel(n_jobs=n_jobs)(delayed(run_fold)(k, tr, va, X, y, params) for k, (tr, va) in enumerate(folds))
    d = pd.concat(res, ignore_index=True).merge(point_predictions(), on=["fold", "row"], how="left")
    assert d["point"].notna().all()
    d["Id"] = df["Id"].values[d["row"].values]
    return add_usd(d)


def run_temporal(n_seeds=20, params=None):
    """The notebook-03 temporal split (train YrSold <= 2008, test 2009-2010). Calibration is a random part of the TRAINING
    years, so the calibration data are older than the test houses: coverage here is a robustness analysis under shift, not a
    conformal guarantee. Repeated over `n_seeds` random proper/calibration splits to show split-to-split variability."""
    df, X, y, _ = load()
    ly = np.log1p(y)
    tr = np.where(df.YrSold.values <= TEMPORAL_LAST_TRAIN_YEAR)[0]
    te = np.where(df.YrSold.values > TEMPORAL_LAST_TRAIN_YEAR)[0]
    # point model for the relative-width denominator: tuned final GBR parameters fitted on the training years only
    point_params = json.loads((NT.OUT / f"final_params_{NT.EXPERIMENT_ID}.json").read_text())["gbr"]["params"]     # point model only
    prep = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X.iloc[tr], ly[tr])
    point = T.make_estimator("gbr", point_params).fit(prep.transform(X.iloc[tr]).values, ly[tr]).predict(prep.transform(X.iloc[te]).values)

    def one(seed):
        iv, info = fit_split_cqr(X.iloc[tr], ly[tr], y[tr], X.iloc[te], seed=seed, params=params)
        return pd.DataFrame(dict(seed=seed, row=te, y=y[te], y_log=ly[te], **iv, point=np.expm1(point), q_hat=info["q_hat"], n_cal=info["n_cal"]))
    d = pd.concat(Parallel(n_jobs=6)(delayed(one)(s) for s in range(SEED, SEED + n_seeds)), ignore_index=True)
    d["Id"] = df["Id"].values[d["row"].values]
    d["YrSold"] = df["YrSold"].values[d["row"].values]
    return add_usd(d)


def run_all(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and (OUT / "cv_intervals.csv.gz").exists() and (OUT / "temporal_intervals.csv.gz").exists():
        return
    t0 = time.time()
    cv = run_cv()
    cv.to_csv(OUT / "cv_intervals.csv.gz", index=False)
    tm = run_temporal()
    tm.to_csv(OUT / "temporal_intervals.csv.gz", index=False)
    run_cv(params=QUANTILE_PARAMS_REGULARISED).to_csv(OUT / "cv_intervals_regularised.csv.gz", index=False)
    run_temporal(params=QUANTILE_PARAMS_REGULARISED).to_csv(OUT / "temporal_intervals_regularised.csv.gz", index=False)
    meta = dict(experiment_id=EXPERIMENT_ID, timestamp=datetime.now().isoformat(timespec="seconds"), runtime_seconds=round(time.time() - t0, 1),
                alpha=ALPHA, quantiles=(Q_LO, Q_HI), cal_fraction=CAL_FRAC, seed=SEED, quantile_params=QUANTILE_PARAMS, quantile_params_regularised_sensitivity=QUANTILE_PARAMS_REGULARISED,
                feature_set="reduced", target="log1p(SalePrice); conformity scores and corrections on the log1p scale",
                flagged_ids=FLAGGED_IDS, temporal_split=f"train YrSold <= {TEMPORAL_LAST_TRAIN_YEAR}, test later",
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
