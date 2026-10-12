"""Export the deployable bundle used by the dashboard.   python -m src.serving.export_model [--keep-d2] [--version v1]

Bundle contents (one joblib file, `artifacts/serving/bundle_<version>.joblib`):
  point   : tuned Gradient Boosting pipeline on log1p(SalePrice), fitted on all training rows kept after the D2 setting
  linear  : tuned ElasticNet pipeline (same rows) used only for the exact group decomposition
  lo, hi  : quantile GBs (0.10 / 0.90) fitted on a 75% proper-training part; `q_hat` from the disjoint 25% calibration part (split CQR)
  tau     : routing threshold on (U - L) / point (USD); HYPOTHETICAL objective, see docs/routing.md
  meta    : D2 setting, row counts, seeds, library versions, tuned parameters, schema

Hyperparameters are read from `artifacts/tuning/final_params_tuning_v1.json` (round-trip floats; GBR is ULP-sensitive). They were tuned with
Ids 524/1299 in the data; they are not re-tuned here. The exported model is fitted on all kept rows, so it has no honest hold-out score:
accuracy statements must come from the nested-CV evidence registry (src/evaluation/evidence.py), never from this bundle.
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.data.loading import load_raw
from src.evaluation import intervals as I
from src.evaluation.cv import price_bands
from src.features.feature_sets import FEATURE_SETS
from src.features.preprocessing import SUGGESTED_TRAINING_EXCLUSIONS
from src.models import tuning as T

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "serving"
PARAMS = ROOT / "artifacts" / "tuning" / "final_params_tuning_v1.json"
TAU = 0.22            # deployable tau from docs/routing.md (selected with D2 houses in the data; hypothetical 7% AUTO-MAPE objective)
SEED = 42


def load_training(exclude_d2=True):
    df = load_raw()
    if exclude_d2:
        df = df[~df["Id"].isin(SUGGESTED_TRAINING_EXCLUSIONS)]
    df = df.reset_index(drop=True)
    return df.drop(columns=["SalePrice", "Id"]), df["SalePrice"].values.astype(float)


def schema(X):
    cat = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "string"}
    num = {c: [float(X[c].min()), float(X[c].max())] for c in X.columns if c not in cat}
    return dict(columns=list(X.columns), categorical=cat, numeric_range=num)


def build_bundle(exclude_d2=True, version="v1"):
    X, y = load_training(exclude_d2)
    ly = np.log1p(y)
    params = json.loads(PARAMS.read_text(encoding="utf-8"), parse_float=float)

    def fit_family(family):
        prep = FEATURE_SETS["reduced"].build("baseline", scale=T.SCALE[family]).fit(X, ly)
        model = T.make_estimator(family, params[family]["params"]).fit(prep.transform(X).values, ly)
        return dict(prep=prep, model=model, family=family, params=params[family]["params"])

    point, linear = fit_family("gbr"), fit_family("elasticnet")

    idx = np.arange(len(X))
    proper, cal = train_test_split(idx, test_size=I.CAL_FRAC, stratify=price_bands(y), random_state=SEED)
    qprep = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X.iloc[proper], ly[proper])
    Zp, Zc = qprep.transform(X.iloc[proper]).values, qprep.transform(X.iloc[cal]).values
    lo = I.make_quantile_model(I.Q_LO).fit(Zp, ly[proper])
    hi = I.make_quantile_model(I.Q_HI).fit(Zp, ly[proper])
    Lc, Uc, _ = I.rearrange(lo.predict(Zc), hi.predict(Zc))
    q_hat, k = I.conformal_quantile(I.conformity_scores(Lc, Uc, ly[cal]), I.ALPHA)

    return dict(
        point=point, linear=linear, quantile=dict(prep=qprep, lo=lo, hi=hi, q_hat=float(q_hat), k=int(k), n_cal=len(cal)),
        tau=TAU, reference_rows=X,
        meta=dict(version=version, created=datetime.now().isoformat(timespec="seconds"), exclude_d2=exclude_d2,
                  excluded_ids=SUGGESTED_TRAINING_EXCLUSIONS if exclude_d2 else [], n_train=len(X), seed=SEED,
                  nominal_coverage=1 - I.ALPHA, point_is="median (expm1 of log1p prediction)", tau_note="hypothetical objective; see docs/routing.md",
                  quantile_params=I.QUANTILE_PARAMS, versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                                                   pandas=pd.__version__, numpy=np.__version__),
                  schema=schema(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-d2", action="store_true", help="keep Ids 524/1299 in training (default: exclude, user decision pending group approval)")
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    b = build_bundle(exclude_d2=not a.keep_d2, version=a.version)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bundle_{a.version}.joblib"
    joblib.dump(b, path)
    print(f"wrote {path}  n_train={b['meta']['n_train']}  q_hat={b['quantile']['q_hat']:.4f}  tau={b['tau']}  exclude_d2={b['meta']['exclude_d2']}")


if __name__ == "__main__":
    main()
