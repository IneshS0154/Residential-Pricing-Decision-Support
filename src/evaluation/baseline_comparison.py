"""Baseline regression comparison with fixed hyperparameters.  From the project root:

    python -m src.evaluation.baseline_comparison [--force]

Experiment A  common feature set (`reduced`, the project default that notebook 03 found not worse than the others)
              x 7 primary models (+ 2 supplementary, previously-informed configurations reported separately)
Experiment B  feature-set sensitivity: 6 non-dummy primary models x {full, reduced, domain_informed}

Same folds everywhere: `get_folds(y, 5 splits, 3 repeats, seed 42)` (repeated K-fold stratified by price band, the
project rule; the stratum is only a split-assignment device, nothing is fitted on it). No hyperparameter search.
Results are cached under artifacts/baseline_comparison/ together with run metadata.
"""
import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.data.loading import TRAIN_PATH
from src.evaluation import feature_selection as F
from src.evaluation.run_feature_selection import load
from src.features.feature_sets import FEATURE_SETS
from src.models.baselines import PRIMARY, REGISTRY, build_baseline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "baseline_comparison"
EXPERIMENT_ID = "baseline_v1"
NON_DUMMY = tuple(n for n in PRIMARY if n != "dummy")


def _metadata(runtime_s):
    _, X, y, folds = load()
    fold_bytes = np.concatenate([np.r_[tr, va] for tr, va in folds]).astype(np.int64).tobytes()
    return dict(
        experiment_id=EXPERIMENT_ID, timestamp=datetime.now().isoformat(timespec="seconds"),
        runtime_seconds=round(runtime_s, 1),
        dataset_sha256=hashlib.sha256(Path(TRAIN_PATH).read_bytes()).hexdigest(), n_rows=len(X),
        n_columns_after_guard=X.shape[1],
        cv=dict(splitter="get_folds: RepeatedStratifiedKFold on 5 price bands", n_splits=5, n_repeats=3, seed=42,
                n_folds=len(folds), fold_hash=hashlib.sha256(fold_bytes).hexdigest()),
        target="log1p(SalePrice); USD predictions = expm1 (median, no smearing correction)",
        feature_sets={k: dict(drop_columns=list(v.drop_columns), year_features=v.year_features, garage_age=v.garage_age)
                      for k, v in FEATURE_SETS.items()},
        models={n: dict(label=b.label, config_class=b.config_class, scale=b.scale, primary=b.primary,
                        params=b.params()) for n, b in REGISTRY.items()},
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                      pandas=pd.__version__),
        machine=platform.processor() or platform.machine())


def run_experiment(models, feature_sets, X, y, folds, n_jobs=6):
    rows, preds = [], []
    for fs in feature_sets:
        for m in models:
            met, pr = F.run_cv(build_baseline(m, fs), X, y, folds, n_jobs=n_jobs)
            tag = fs if m != "dummy" else "none"
            rows.append(met.assign(model=m, feature_set=tag))
            preds.append(pr.assign(model=m, feature_set=tag))
            print(m, fs, round(met.rmse_log.mean(), 4), flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(preds, ignore_index=True)


def run_all(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "expA_folds.csv").exists() and (OUT / "expB_folds.csv").exists() and not force:
        return
    t0 = time.time()
    _, X, y, folds = load()
    assert (y > 0).all() and np.isfinite(y).all()
    supp = tuple(n for n in REGISTRY if not REGISTRY[n].primary)
    a, pa = run_experiment(PRIMARY + supp, ["reduced"], X, y, folds)
    a.to_csv(OUT / "expA_folds.csv", index=False)
    pa.to_csv(OUT / "expA_predictions.csv.gz", index=False)
    b, _ = run_experiment(NON_DUMMY, ["full", "domain_informed"], X, y, folds)
    b = pd.concat([b, a[a.model.isin(NON_DUMMY)]], ignore_index=True)       # reduced = experiment A rows (not refitted)
    b.to_csv(OUT / "expB_folds.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps(_metadata(time.time() - t0), indent=2), encoding="utf-8")


# ---------------------------------------------------------------- summaries
def comparison_table(folds_df, predictions=None):
    """One row per model: mean and SD over folds of every metric, min/max Log-RMSE, mean fit time, and (when
    `predictions` is given) pooled Log-RMSE over all out-of-fold predictions. Pooled RMSE is not the mean of the
    fold RMSEs: it weights every prediction equally and is dominated by the hardest folds."""
    g = folds_df.groupby("model", sort=False)
    t = pd.DataFrame({
        "Log-RMSE Mean": g.rmse_log.mean(), "Log-RMSE SD": g.rmse_log.std(),
        "Log-RMSE Min": g.rmse_log.min(), "Log-RMSE Max": g.rmse_log.max(),
        "MAE USD": g.mae.mean(), "MAE USD SD": g.mae.std(), "RMSE USD": g.rmse_usd.mean(), "RMSE USD SD": g.rmse_usd.std(),
        "MAPE %": g.mape.mean(), "MAPE % SD": g.mape.std(), "R2 USD": g.r2_usd.mean(), "R2 USD SD": g.r2_usd.std(),
        "R2 log-scale": g.r2_log.mean(), "Fit Time s": g.fit_time.mean()})
    if predictions is not None:
        pr = predictions.assign(e=lambda d: d.pred_log - np.log1p(d.y))
        t["Pooled Log-RMSE"] = pr.groupby("model", sort=False).e.apply(lambda e: float(np.sqrt(np.mean(e ** 2))))
    return t.sort_values("Log-RMSE Mean")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
