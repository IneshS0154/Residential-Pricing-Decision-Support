"""Chronological generalisation experiment (notebook 08).  From the project root:

    python -m src.evaluation.temporal [--force]

Split (the one defined in notebook 03 and reused unchanged): train = houses with YrSold <= 2008 (947 houses),
test = YrSold in {2009, 2010} (513 houses). YrSold is the split key and evaluation metadata only; it is never a predictor
(the `guard` step of the preprocessing drops it).

Everything learned is learned from the historical training houses only:
  preprocessing   fitted on the 947 training houses (imputation, medians, skew masks, scalers, encoder categories)
  hyperparameters  `hist_tuned`: Optuna TPE with 3-fold inner CV INSIDE the 947 training houses (nothing from 2009-2010 is seen)
  feature set     `reduced` is a fixed, previously decided configuration (it was chosen in notebook 03 using all years; disclosed)
Configurations compared on the same split
  fixed             library/fixed baselines of notebook 04 (gbr defaults; ElasticNet alpha 1e-3, l1_ratio 0.5): no tuning at all
  hist_tuned        hyperparameters selected from historical training houses only (the clean temporal result)
  all_years_tuned   the 05 final parameters, selected using ALL years: EXPLORATORY, optimistic for the test years
Matched-training-size sensitivity: random splits with 947 training and 513 test houses (30 seeds) using the `fixed` configurations
(no tuning, so no hyperparameter information from the test houses), to separate the effect of training-set size and design from
that of the chronological shift.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from src.evaluation import feature_selection as F
from src.evaluation import nested_tuning as NT
from src.evaluation.run_feature_selection import load
from src.features.feature_sets import FEATURE_SETS
from src.models import tuning as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "segments"
SEED = 42
LAST_TRAIN_YEAR = 2008
FAMILIES = ("gbr", "elasticnet")
TUNING_BUDGET = {"elasticnet": 150, "gbr": 50}          # same budgets as the nested tuning of 05
N_RANDOM_SPLITS = 30
FLAGGED_IDS = (524, 1299)           # EDA F17 / D2 (both sold before 2009, so neither is in the temporal test set)


def fixed_estimator(family):
    if family == "gbr":
        return GradientBoostingRegressor(random_state=SEED)
    return ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=T.MAX_ITER, random_state=SEED)


def estimator_for(family, config, params=None):
    return fixed_estimator(family) if config == "fixed" else T.make_estimator(family, params)


def split_indices(df):
    tr, te = F.temporal_split(df, LAST_TRAIN_YEAR)
    assert df.YrSold.values[tr].max() <= LAST_TRAIN_YEAR < df.YrSold.values[te].min()      # no future sale in training
    return tr, te


def metrics_dict(pred_log, y):
    pred = np.expm1(pred_log)
    res = y - pred
    return dict(n=len(y), log_rmse=float(np.sqrt(np.mean((np.log1p(y) - pred_log) ** 2))), mae=float(np.mean(np.abs(res))),
                rmse_usd=float(np.sqrt(np.mean(res ** 2))), mape=float(100 * np.mean(np.abs(res) / y)),
                r2_usd=float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)), signed_usd=float(res.mean()),
                signed_log=float(np.mean(np.log1p(y) - pred_log)), median_ae=float(np.median(np.abs(res))))


def fit_predict(family, config, params, X, ly, tr, te):
    """Fit preprocessing and model on `tr` only; predict `te`. Returns log1p predictions."""
    prep = FEATURE_SETS["reduced"].build("baseline", scale=T.SCALE[family]).fit(X.iloc[tr], ly[tr])
    model = estimator_for(family, config, params).fit(prep.transform(X.iloc[tr]).values, ly[tr])
    return model.predict(prep.transform(X.iloc[te]).values)


def tune_historical(family, X, y, tr, n_trials):
    """Optuna study using ONLY the historical training houses (3-fold inner CV inside them)."""
    ly = np.log1p(y)
    cache = NT._inner_cache(family, X.iloc[tr].reset_index(drop=True), ly[tr], y[tr], seed=SEED)
    study = NT._study(family, cache, n_trials, seed=SEED)
    assert T.validate_params(family, dict(study.best_params))
    return dict(study.best_params), float(study.best_value), len(study.trials), NT.trials_frame(study, family=family)


def _random_split(seed, family, X, y, ids, n_train):
    perm = np.random.RandomState(seed).permutation(len(y))
    tr, te = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    p = fit_predict(family, "fixed", None, X, np.log1p(y), tr, te)
    keep = ~np.isin(ids[te], FLAGGED_IDS)                                   # same split scored without the two flagged houses
    flagged_in_test = int((~keep).sum())
    return dict(seed=seed, family=family, flagged_in_test=flagged_in_test, **metrics_dict(p, y[te]),
                **{f"{k}_noflag": v for k, v in metrics_dict(p[keep], y[te][keep]).items()})


def run_all(force=False, n_jobs=6):
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and (OUT / "temporal_metrics.csv").exists() and (OUT / "random_matched_splits.csv").exists():
        return
    t0 = time.time()
    df, X, y, _ = load()
    ly = np.log1p(y)
    tr, te = split_indices(df)
    yr_te = df.YrSold.values[te]

    hist, trials = {}, []
    for fam in FAMILIES:
        p, best, n, tf = tune_historical(fam, X, y, tr, TUNING_BUDGET[fam])
        hist[fam] = dict(params=p, inner_best_logrmse=best, n_trials=n, training_houses=len(tr))
        trials.append(tf)
    (OUT / "historical_tuning.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    pd.concat(trials).to_csv(OUT / "historical_tuning_trials.csv", index=False)
    allyears = json.loads((NT.OUT / f"final_params_{NT.EXPERIMENT_ID}.json").read_text())
    cfgs = {"fixed": lambda f: None, "hist_tuned": lambda f: hist[f]["params"], "all_years_tuned": lambda f: allyears[{"gbr": "gbr", "elasticnet": "elasticnet"}[f]]["params"]}

    rows, preds = [], []
    for fam in FAMILIES:
        for cfg, getp in cfgs.items():
            p = fit_predict(fam, "fixed" if cfg == "fixed" else "tuned", getp(fam), X, ly, tr, te)
            for period, mask in [("2009-2010", np.ones(len(te), bool)), ("2009", yr_te == 2009), ("2010", yr_te == 2010)]:
                rows.append(dict(family=fam, config=cfg, period=period, n_train=len(tr), **metrics_dict(p[mask], y[te][mask])))
            preds.append(pd.DataFrame(dict(family=fam, config=cfg, row=te, YrSold=yr_te, y=y[te], pred_log=p)))
    pd.DataFrame(rows).to_csv(OUT / "temporal_metrics.csv", index=False)
    pd.concat(preds).to_csv(OUT / "temporal_predictions.csv.gz", index=False)

    rnd = Parallel(n_jobs=n_jobs)(delayed(_random_split)(s, fam, X, y, df.Id.values, len(tr)) for fam in FAMILIES for s in range(SEED, SEED + N_RANDOM_SPLITS))
    pd.DataFrame(rnd).to_csv(OUT / "random_matched_splits.csv", index=False)

    meta = dict(timestamp=datetime.now().isoformat(timespec="seconds"), runtime_seconds=round(time.time() - t0, 1), seed=SEED,
                split=f"train YrSold <= {LAST_TRAIN_YEAR} (n={len(tr)}), test later (n={len(te)})", feature_set="reduced",
                tuning_budget=TUNING_BUDGET, n_random_splits=N_RANDOM_SPLITS,
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
