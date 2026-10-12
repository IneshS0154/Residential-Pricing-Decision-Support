"""Run (and cache) every feature-selection experiment.  From the project root:

    python -m src.evaluation.run_feature_selection [--force]

Stages (each cached as CSV under artifacts/feature_selection/, so notebook 03 re-reads them instead of refitting):
  ablation     single-group ablations x {ridge (alpha 10), gbr}        exploratory
  comparison   3 feature sets x 6 candidate models, outer folds          exploratory
  nested       FeatureSetSelector (inner-CV choice of set) x {ridge, gbr}, outer folds   selection-bias-aware
  temporal     3 feature sets x 6 models, train YrSold <= 2008, test later
  lasso        coefficient stability over the outer folds (full and reduced sets)
  permutation  grouped permutation importance on held-out folds (full set)
Same folds everywhere: get_folds(y, 5 splits, 3 repeats, seed 42).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from src.data.loading import load_raw
from src.evaluation import feature_selection as F
from src.evaluation.cv import get_folds
from src.features.feature_sets import FEATURE_SETS, GROUPS, all_group_variants
from src.models.candidates import MODELS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "feature_selection"
SEED = 42
ABLATION_MODELS = ("ridge", "gbr")

PERM_GROUPS = {
    "GarageCars": ["GarageCars"], "GarageArea": ["GarageArea"], "Garage (both)": ["GarageCars", "GarageArea"],
    "GrLivArea": ["GrLivArea"], "TotRmsAbvGrd": ["TotRmsAbvGrd"], "Living (both)": ["GrLivArea", "TotRmsAbvGrd"],
    "TotalBsmtSF": ["TotalBsmtSF"], "1stFlrSF": ["1stFlrSF"], "Floor area (TotalBsmtSF+1stFlrSF)": ["TotalBsmtSF", "1stFlrSF"],
    "BsmtFinSF1": ["BsmtFinSF1"], "BsmtUnfSF": ["BsmtUnfSF"],
    "Basement (total+parts)": ["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"],
    "FullBath": ["FullBath"], "HalfBath": ["HalfBath"], "BsmtFullBath": ["BsmtFullBath"],
    "Bathrooms (all four)": ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
    "YearBuilt": ["YearBuilt"], "YearRemodAdd": ["YearRemodAdd"], "GarageYrBlt": ["GarageYrBlt"],
    "Age (all three years)": ["YearBuilt", "YearRemodAdd", "GarageYrBlt"],
    "OverallQual": ["OverallQual"], "Neighborhood": ["Neighborhood"], "LotArea": ["LotArea"],
}


def load():
    df = load_raw()
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"].values.astype(float)
    return df, X, y, get_folds(y, n_splits=5, n_repeats=3, seed=SEED)


def cached(name, fn, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    res = fn()
    res.to_csv(path, index=False)
    return res


# ---------------------------------------------------------------- stages
def stage_ablation(force=False):
    def go():
        _, X, y, folds = load()
        cells, seen, rows, preds = all_group_variants(), {}, [], []
        for g, v, fs in cells:
            key = (fs.drop_columns, fs.year_features, fs.garage_age)
            for m in ABLATION_MODELS:
                if (key, m) not in seen:
                    seen[(key, m)] = F.run_cv(F.make_pipeline(fs, m), X, y, folds)
                met, pr = seen[(key, m)]
                rows.append(met.assign(group=g, variant=v, model=m))
                preds.append(pr.assign(group=g, variant=v, model=m))
        pd.concat(preds).to_csv(OUT / "ablation_predictions.csv.gz", index=False)
        return pd.concat(rows, ignore_index=True)
    return cached("ablation_folds", go, force)


def stage_comparison(force=False):
    def go():
        _, X, y, folds = load()
        rows = []
        for name in FEATURE_SETS:
            for m in MODELS:
                met, _ = F.run_cv(F.make_pipeline(name, m, fixed=False), X, y, folds)
                rows.append(met.assign(feature_set=name, model=m))
        return pd.concat(rows, ignore_index=True)
    return cached("comparison_folds", go, force)


def stage_nested(force=False):
    def go():
        _, X, y, folds = load()
        rows = []
        for m in ABLATION_MODELS:
            sel = F.FeatureSetSelector(m, fixed=False)
            met, _ = F.run_cv(sel, X, y, folds)
            rows.append(met.assign(model=m))
        return pd.concat(rows, ignore_index=True)
    return cached("nested_folds", go, force)


def stage_temporal(force=False):
    def go():
        df, X, y, _ = load()
        tr, te = F.temporal_split(df)
        rows = []
        for name in FEATURE_SETS:
            for m in MODELS:
                e = F.make_pipeline(name, m, fixed=False).fit(X.iloc[tr], np.log1p(y[tr]))
                r, mae, mape = F._metrics(e.predict(X.iloc[te]), y[te])
                rows.append(dict(feature_set=name, model=m, n_train=len(tr), n_test=len(te), rmse_log=r, mae=mae,
                                 mape=mape, n_features=e[-1].n_features_in_))
        return pd.DataFrame(rows)
    return cached("temporal", go, force)


def stage_lasso(force=False):
    def go():
        _, X, y, folds = load()
        rows = []
        for name in ("full", "reduced", "domain_informed"):
            coefs, alphas = F.lasso_stability(name, X, y, folds)
            nz = (coefs.abs() > 1e-10)
            rows.append(pd.DataFrame(dict(feature_set=name, feature=coefs.columns, mean_coef=coefs.mean().values,
                                          sd_coef=coefs.std().values, nonzero_share=nz.mean().values,
                                          mean_alpha=alphas.mean())))
        return pd.concat(rows, ignore_index=True)
    return cached("lasso_stability", go, force)


def stage_permutation(force=False):
    def go():
        _, X, y, folds = load()
        ly = np.log1p(y)

        def one(k, tr, va, model):
            fit = F.make_pipeline("full", model).fit(X.iloc[tr], ly[tr])
            s = F.grouped_permutation_importance(fit, X.iloc[va], ly[va], PERM_GROUPS, n_repeats=3, seed=k)
            return s.rename(k)
        rows = []
        for m in ABLATION_MODELS:
            res = Parallel(n_jobs=6)(delayed(one)(k, tr, va, m) for k, (tr, va) in enumerate(folds))
            d = pd.concat(res, axis=1)
            rows.append(pd.DataFrame(dict(model=m, group=d.index, mean_delta=d.mean(axis=1).values,
                                          sd_delta=d.std(axis=1).values, share_folds_positive=(d > 0).mean(axis=1).values)))
        return pd.concat(rows, ignore_index=True)
    return cached("permutation_importance", go, force)


def screening_diagnostics(fold=0):
    """VIF / rank / mutual information of the three sets on the TRAINING rows of one outer fold (not cached: fast)."""
    _, X, y, folds = load()
    tr, _ = folds[fold]
    ly = np.log1p(y)
    vif_rows, mi = [], {}
    for name, fs in FEATURE_SETS.items():
        prep = fs.build("baseline", scale="robust").fit(X.iloc[tr], ly[tr])
        Z = prep.transform(X.iloc[tr])
        from src.features import preprocessing as P
        num = Z[[c for c in Z.columns if c in P.NUMERIC + ["YearBuilt", "YearRemodAdd", "GarageAge"]]]
        v, rank, ncol = F.vif_table(num)
        vif_rows.append(dict(feature_set=name, n_output_cols=Z.shape[1], n_numeric=ncol, numeric_rank=rank,
                             n_inf_vif=int(np.isinf(v).sum()), n_vif_gt10_finite=int(((v > 10) & np.isfinite(v)).sum()),
                             max_finite_vif=float(v[np.isfinite(v)].max())))
        mi[name] = F.mutual_information(Z, ly[tr])
    return pd.DataFrame(vif_rows), pd.DataFrame(mi)


def run_all(force=False):
    for s in (stage_ablation, stage_comparison, stage_nested, stage_temporal, stage_lasso, stage_permutation):
        print("stage", s.__name__, flush=True)
        s(force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run_all(ap.parse_args().force)
