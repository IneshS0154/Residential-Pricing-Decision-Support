"""Leakage-safe Optuna tuning with nested cross-validation.  From the project root:

    python -m src.evaluation.nested_tuning --pilot                       # 1 outer fold, few trials: timing only
    python -m src.evaluation.nested_tuning --nested --trials elasticnet=60 gbr=40
    python -m src.evaluation.nested_tuning --final  --trials elasticnet=120 gbr=80

Design
  outer loop   the 15 shared folds of `get_folds` (5-fold x 3 repeats, seed 42, price-band stratified): the same folds
               as the fixed-hyperparameter baselines, so tuned and untuned procedures are compared fold by fold.
  inner loop   3-fold CV INSIDE each outer training set (own seed per outer fold), objective = mean inner Log-RMSE.
  per outer    one TPE study (seed 42 + fold index, 10 random start-up trials, no pruning) sees only the outer training
  fold         rows; the best parameters refit the pipeline on all outer training rows; the outer validation fold is
               scored once and never influences any trial.
  preprocessing is fitted on the inner training rows only (cached per inner fold: it does not depend on the model
               hyperparameters, so reusing it across trials is exact and leakage-free).
  final study  after the procedure is evaluated, one study on all 1,460 labelled rows (inner 3-fold) gives the final
               parameters. Those are NOT themselves evaluated by the nested CV, which evaluates the procedure.
Studies of the final stage are persisted in SQLite; outer-fold trial histories are saved as CSV.
"""
import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from src.evaluation import feature_selection as F
from src.evaluation.cv import get_folds
from src.evaluation.run_feature_selection import load
from src.features.feature_sets import FEATURE_SETS
from src.models import tuning as T

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "tuning"
EXPERIMENT_ID = "tuning_v1"
FEATURE_SET = "reduced"            # same common feature set as the baseline comparison
INNER_SPLITS = 3
N_STARTUP = 10


# ---------------------------------------------------------------- one study
def _inner_cache(family, X, y_log, y_raw, seed):
    """Per inner fold: preprocessing fitted on the inner TRAINING rows only, applied to the inner validation rows."""
    cache = []
    for itr, iva in get_folds(y_raw, n_splits=INNER_SPLITS, n_repeats=1, seed=seed):
        prep = FEATURE_SETS[FEATURE_SET].build("baseline", scale=T.SCALE[family]).fit(X.iloc[itr], y_log[itr])
        cache.append((prep.transform(X.iloc[itr]).values, y_log[itr], prep.transform(X.iloc[iva]).values, y_log[iva]))
    return cache


def _objective(family, cache):
    def objective(trial):
        try:
            params = T.suggest(trial, family)
            scores = []
            with T.ConvergenceRecorder() as rec:
                for Ztr, ytr, Zva, yva in cache:
                    m = T.make_estimator(family, params).fit(Ztr, ytr)
                    p = m.predict(Zva)
                    if not np.isfinite(p).all():
                        raise ValueError("non-finite predictions")
                    scores.append(float(np.sqrt(np.mean((p - yva) ** 2))))
            trial.set_user_attr("n_convergence_warnings", rec.count)
            trial.set_user_attr("inner_fold_scores", scores)
            return float(np.mean(scores))
        except Exception as e:                       # recorded on the trial, then Optuna marks it FAIL (never silent)
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            raise
    return objective


def _study(family, cache, n_trials, seed, storage=None, name=None):
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage, study_name=name)
    study.optimize(_objective(family, cache), n_trials=n_trials, catch=(Exception,), n_jobs=1)
    return study


def trials_frame(study, **tags):
    d = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration", "user_attrs"))
    d["duration_s"] = d["duration"].dt.total_seconds()
    d["best_so_far"] = d["value"].where(d["state"] == "COMPLETE").cummin()
    d = d.drop(columns=["duration"])
    for k, v in tags.items():
        d[k] = v
    return d


def _importance(study):
    try:
        return optuna.importance.get_param_importances(study)
    except Exception:
        return {}


# ---------------------------------------------------------------- nested CV
def tune_outer_fold(family, k, tr, va, n_trials, X, y):
    """Tune on outer training rows only, refit, score the outer validation fold once."""
    t0 = time.perf_counter()
    ly = np.log1p(y)
    cache = _inner_cache(family, X.iloc[tr].reset_index(drop=True), ly[tr], y[tr], seed=T.SEED + k)
    study = _study(family, cache, n_trials, seed=T.SEED + k)
    tune_time = time.perf_counter() - t0
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not done:
        raise RuntimeError(f"all {n_trials} trials failed for {family}, outer fold {k}: {study.trials[0].user_attrs}")
    best = dict(study.best_params)
    assert T.validate_params(family, best)
    t1 = time.perf_counter()
    prep = FEATURE_SETS[FEATURE_SET].build("baseline", scale=T.SCALE[family]).fit(X.iloc[tr], ly[tr])
    with T.ConvergenceRecorder() as rec:
        model = T.make_estimator(family, best).fit(prep.transform(X.iloc[tr]).values, ly[tr])
    p = model.predict(prep.transform(X.iloc[va]).values)
    refit_time = time.perf_counter() - t1
    rmse_log, mae, mape = F._metrics(p, y[va])
    rmse_usd, r2_usd, r2_log = F._extra_metrics(p, y[va])
    outer = dict(family=family, fold=k, repeat=k // 5, rmse_log=rmse_log, mae=mae, mape=mape, rmse_usd=rmse_usd,
                 r2_usd=r2_usd, r2_log=r2_log, inner_best=study.best_value, n_trials=n_trials,
                 n_complete=len(done), n_failed=sum(t.state.name == "FAIL" for t in study.trials),
                 n_convergence_warnings=int(sum(t.user_attrs.get("n_convergence_warnings", 0) for t in done) + rec.count),
                 tune_time=tune_time, refit_time=refit_time, all_finite=bool(np.isfinite(p).all()),
                 n_train=len(tr), n_val=len(va), **{f"best_{a}": b for a, b in best.items()})
    pred = pd.DataFrame(dict(family=family, fold=k, row=va, y=y[va], pred_log=p))
    return outer, trials_frame(study, family=family, outer_fold=k), _importance(study), pred


def run_nested(budget, n_jobs=6, folds_subset=None, tag="nested"):
    _, X, y, folds = load()
    ks = list(range(len(folds))) if folds_subset is None else list(folds_subset)
    OUT.mkdir(parents=True, exist_ok=True)
    outers, trials, imps, preds = [], [], [], []
    t0 = time.perf_counter()
    for family, n_trials in budget.items():
        t_fam = time.perf_counter()
        res = Parallel(n_jobs=n_jobs)(delayed(tune_outer_fold)(family, k, *folds[k], n_trials, X, y) for k in ks)
        wall = time.perf_counter() - t_fam
        for o, tr_, im, pr in res:
            outers.append({**o, "family_wall_s": wall})
            trials.append(tr_)
            imps.append(pd.DataFrame([dict(family=family, outer_fold=o["fold"], param=a, importance=b) for a, b in im.items()]))
            preds.append(pr)
        print(family, "outer folds:", len(res), "wall s:", round(wall), "mean outer Log-RMSE:",
              round(np.mean([r[0]["rmse_log"] for r in res]), 4), flush=True)
    pd.DataFrame(outers).to_csv(OUT / f"{tag}_outer_folds.csv", index=False)
    pd.concat(trials, ignore_index=True).to_csv(OUT / f"{tag}_trials.csv", index=False)
    pd.concat(imps, ignore_index=True).to_csv(OUT / f"{tag}_param_importance.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / f"{tag}_predictions.csv.gz", index=False)
    return time.perf_counter() - t0


# ---------------------------------------------------------------- final study on all development data
def run_final(budget, force=False):
    _, X, y, _ = load()
    ly = np.log1p(y)
    storage = f"sqlite:///{(OUT / 'final_studies.db').as_posix()}"
    OUT.mkdir(parents=True, exist_ok=True)
    final, frames = {}, []
    for family, n_trials in budget.items():
        name = f"{EXPERIMENT_ID}_{family}_final"
        if force:
            try:
                optuna.delete_study(study_name=name, storage=storage)
            except KeyError:
                pass
        t0 = time.perf_counter()
        cache = _inner_cache(family, X, ly, y, seed=T.SEED)
        study = _study(family, cache, n_trials, seed=T.SEED, storage=storage, name=name)
        elapsed = time.perf_counter() - t0
        best = dict(study.best_params)
        assert T.validate_params(family, best)
        prep = FEATURE_SETS[FEATURE_SET].build("baseline", scale=T.SCALE[family]).fit(X, ly)
        model = T.make_estimator(family, best).fit(prep.transform(X).values, ly)
        joblib.dump(dict(prep=prep, model=model, params=best, family=family, experiment_id=EXPERIMENT_ID),
                    OUT / f"final_model_{family}_{EXPERIMENT_ID}.joblib")
        final[family] = dict(params=best, inner_cv_best_logrmse=study.best_value, n_trials=n_trials, seconds=round(elapsed, 1),
                             note="inner-CV optimum on ALL labelled rows: optimistic, not a generalisation estimate")
        frames.append(trials_frame(study, family=family))
        _importance_df = _importance(study)
        final[family]["param_importance"] = _importance_df
    (OUT / f"final_params_{EXPERIMENT_ID}.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(OUT / "final_trials.csv", index=False)
    return final


def metadata(extra):
    _, X, y, folds = load()
    fold_bytes = np.concatenate([np.r_[tr, va] for tr, va in folds]).astype(np.int64).tobytes()
    return dict(experiment_id=EXPERIMENT_ID, timestamp=datetime.now().isoformat(timespec="seconds"),
                feature_set=FEATURE_SET, inner_splits=INNER_SPLITS, tpe_startup_trials=N_STARTUP, seed=T.SEED,
                sampler="TPESampler(seed=42+outer_fold, n_startup_trials=10), no pruning",
                search_spaces={f: {k: list(v) for k, v in s.items()} for f, s in T.SPACES.items()},
                max_iter_convergence_control=T.MAX_ITER, fold_hash=hashlib.sha256(fold_bytes).hexdigest(),
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, optuna=optuna.__version__,
                              numpy=np.__version__, pandas=pd.__version__), **extra)


def _parse_budget(items):
    return {k: int(v) for k, v in (s.split("=") for s in items)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trials", nargs="+", default=["elasticnet=60", "gbr=40"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    budget = _parse_budget(a.trials)
    meta_path = OUT / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if a.pilot:
        secs = run_nested(budget, n_jobs=1, folds_subset=[0], tag="pilot")
        meta["pilot"] = dict(budget=budget, folds=[0], seconds=round(secs, 1))
    if a.nested:
        secs = run_nested(budget, n_jobs=a.jobs)
        meta["nested"] = dict(budget=budget, outer_folds=15, n_jobs=a.jobs, seconds=round(secs, 1))
    if a.final:
        t0 = time.perf_counter()
        run_final(budget, force=a.force)
        meta["final"] = dict(budget=budget, seconds=round(time.perf_counter() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**meta, **metadata({})}, indent=2, default=str), encoding="utf-8")
