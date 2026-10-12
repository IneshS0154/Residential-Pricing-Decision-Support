"""Prediction generation for the routing study.  From the project root:   python -m src.evaluation.routing_runs [--force]

Data roles (nothing is shared between roles within a fold)
  model fitting            proper-training part of the (inner) training rows
  interval calibration     calibration part of the same (inner) training rows (split CQR of src.evaluation.intervals)
  threshold selection      DEVELOPMENT data = inner out-of-fold predictions on the OUTER-TRAINING houses of each outer fold
                           (5-fold inside the outer training rows; every dev prediction comes from models that saw neither that
                           house nor any outer-validation house)
  final routing evaluation the outer validation fold: its point predictions and CQR intervals are the existing 05 / 07
                           out-of-fold artifacts (models fitted on the outer-training rows only); the validation labels were never
                           used to select its threshold

Point model: tuned Gradient Boosting; the fold's tuned parameters from the 05 nested run (chosen on outer-training rows only).
Interval method: split CQR (80% nominal) with the fixed base configuration of 07. Both interval versions (raw quantile and CQR) are
stored so the routing study can also be run on the raw quantile intervals as a sensitivity.

Temporal: train YrSold <= 2008 / test 2009-2010 (the notebook-03 split). The development data for the temporal threshold are inner
out-of-fold predictions on the 947 training houses only; the point model uses the historical-only tuned parameters of 08.
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation import final_evaluation as E
from src.evaluation import intervals as V
from src.evaluation import nested_tuning as NT
from src.evaluation import temporal as TP
from src.evaluation.cv import get_folds
from src.evaluation.run_feature_selection import load
from src.features.feature_sets import FEATURE_SETS
from src.models import tuning as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "routing"
EXPERIMENT_ID = "routing_v1"
SEED = 42
INNER_SPLITS = 5
TEMPORAL_SEEDS = range(42, 52)          # random calibration splits for the temporal CQR (the first one defines the frozen policy)


def inner_oof(X, y, dev_idx, point_params, seed):
    """Out-of-fold point predictions and CQR / raw-quantile intervals for the houses `dev_idx`, using only `dev_idx` houses."""
    ly = np.log1p(y)
    rows = []
    for j, (a, b) in enumerate(get_folds(y[dev_idx], n_splits=INNER_SPLITS, n_repeats=1, seed=seed)):
        tri, vai = dev_idx[a], dev_idx[b]
        prep = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X.iloc[tri], ly[tri])
        pt = T.make_estimator("gbr", point_params).fit(prep.transform(X.iloc[tri]).values, ly[tri]).predict(prep.transform(X.iloc[vai]).values)
        iv, info = V.fit_split_cqr(X.iloc[tri], ly[tri], y[tri], X.iloc[vai], seed=seed + j)
        rows.append(pd.DataFrame(dict(inner_fold=j, row=vai, y=y[vai], point_log=pt, **iv, q_hat=info["q_hat"])))
    return pd.concat(rows, ignore_index=True)


def _dev_fold(k, tr, X, y, params):
    return inner_oof(X, y, tr, params, seed=SEED + 100 + k).assign(outer_fold=k)


def outer_eval_frame():
    """Outer-validation predictions and intervals (05 / 07 artifacts) with the routing columns; ONE row per house per CV repeat."""
    df, X, y, folds = load()
    iv = pd.read_csv(V.OUT / "cv_intervals.csv.gz", float_precision="round_trip")
    d = iv[["fold", "repeat", "row", "Id", "y", "point", "L_raw", "U_raw", "L_cqr", "U_cqr", "lo_raw", "hi_raw", "lo_cqr", "hi_cqr"]].copy()
    assert d["point"].gt(0).all()
    return d


def run_all(force=False, n_jobs=6):
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and (OUT / "dev_inner_oof.csv.gz").exists() and (OUT / "temporal_dev_inner.csv.gz").exists():
        return
    t0 = time.time()
    df, X, y, folds = load()
    outer = pd.read_csv(NT.OUT / "nested_outer_folds.csv", float_precision="round_trip")
    og = outer[outer.family == "gbr"].set_index("fold")
    res = Parallel(n_jobs=n_jobs)(delayed(_dev_fold)(k, tr, X, y, E._params_for(og.loc[k], "gbr")) for k, (tr, va) in enumerate(folds))
    dev = pd.concat(res, ignore_index=True)
    dev["point"] = np.expm1(dev["point_log"])
    for m in ("raw", "cqr"):
        dev[f"lo_{m}"], dev[f"hi_{m}"] = np.expm1(dev[f"L_{m}"]), np.expm1(dev[f"U_{m}"])
    dev.to_csv(OUT / "dev_inner_oof.csv.gz", index=False)
    outer_eval_frame().to_csv(OUT / "outer_eval.csv.gz", index=False)

    # temporal: everything from the training years only
    tr, te = TP.split_indices(df)
    hist = json.loads((TP.OUT / "historical_tuning.json").read_text())["gbr"]["params"]
    hist = {k: (int(round(v)) if k in ("n_estimators", "max_depth", "min_samples_leaf") else float(v)) for k, v in hist.items()}
    tdev = inner_oof(X, y, tr, hist, seed=SEED + 500)
    tdev["point"] = np.expm1(tdev["point_log"])
    for m in ("raw", "cqr"):
        tdev[f"lo_{m}"], tdev[f"hi_{m}"] = np.expm1(tdev[f"L_{m}"]), np.expm1(tdev[f"U_{m}"])
    tdev.to_csv(OUT / "temporal_dev_inner.csv.gz", index=False)
    ly = np.log1p(y)
    prep = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X.iloc[tr], ly[tr])
    point_te = T.make_estimator("gbr", hist).fit(prep.transform(X.iloc[tr]).values, ly[tr]).predict(prep.transform(X.iloc[te]).values)

    def one(seed):
        iv, info = V.fit_split_cqr(X.iloc[tr], ly[tr], y[tr], X.iloc[te], seed=seed)
        return pd.DataFrame(dict(seed=seed, row=te, Id=df.Id.values[te], YrSold=df.YrSold.values[te], y=y[te], point=np.expm1(point_te), q_hat=info["q_hat"],
                                 lo_raw=np.expm1(iv["L_raw"]), hi_raw=np.expm1(iv["U_raw"]), lo_cqr=np.expm1(iv["L_cqr"]), hi_cqr=np.expm1(iv["U_cqr"])))
    pd.concat(Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in TEMPORAL_SEEDS), ignore_index=True).to_csv(OUT / "temporal_eval.csv.gz", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps(dict(
        experiment_id=EXPERIMENT_ID, timestamp=datetime.now().isoformat(timespec="seconds"), runtime_seconds=round(time.time() - t0, 1), seed=SEED,
        inner_splits=INNER_SPLITS, point_model="tuned GBR (05 fold-specific parameters; temporal: 08 historical-only parameters)",
        interval_method="split CQR, 80% nominal, 07 base configuration", interval_calibration_fraction=V.CAL_FRACTION if hasattr(V, "CAL_FRACTION") else V.CAL_FRAC,
        quantile_params=V.QUANTILE_PARAMS, temporal_split="train YrSold <= 2008, test 2009-2010", temporal_calibration_seeds=list(TEMPORAL_SEEDS)), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
