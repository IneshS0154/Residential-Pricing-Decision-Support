"""Leakage-safe evaluation tools for feature-set selection.

Every estimator handled here is a full Pipeline (raw load_raw() columns in, prediction of log1p(price) out), cloned
and fitted on the training rows of each fold only, on folds from `src.evaluation.cv.get_folds`.

Three kinds of numbers must not be mixed up (see notebooks/03_feature_selection.ipynb):
  exploratory ablation scores    `run_cv` on the shared outer folds, used to LOOK at feature groups
  selection scores               inner CV inside `FeatureSetSelector.fit`, using training rows of one outer fold only
  generalisation estimates       outer-fold scores of `FeatureSetSelector` (nested CV) and the temporal split
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline

from src.evaluation.cv import get_folds
from src.features.feature_sets import get_feature_set
from src.models.candidates import make_model, scale_for


# ---------------------------------------------------------------- estimators
def make_pipeline(fs, model, fixed=True, strategy="baseline"):
    """Preprocessing for feature set `fs` (FeatureSet or name) + model, scaled appropriately for the model family."""
    fs = get_feature_set(fs) if isinstance(fs, str) else fs
    return Pipeline([("prep", fs.build(strategy, scale=scale_for(model))), ("model", make_model(model, fixed))])


def _metrics(pred_log, y_true):
    pred = np.expm1(pred_log)
    return (np.sqrt(np.mean((pred_log - np.log1p(y_true)) ** 2)),
            np.mean(np.abs(pred - y_true)),
            100 * np.mean(np.abs(pred - y_true) / y_true))


def _n_iter(m):
    """Iterations used by a solver that reports them (coordinate descent, boosting); NaN otherwise."""
    it = getattr(m[-1], "n_iter_", None) if isinstance(m, Pipeline) else None
    return np.nan if it is None else float(np.max(it))


def _extra_metrics(pred_log, y_true):
    """RMSE (USD), R2 on the USD scale and R2 on the log1p scale (reported separately, never mixed)."""
    pred, yl = np.expm1(pred_log), np.log1p(y_true)
    r2 = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)
    return np.sqrt(np.mean((pred - y_true) ** 2)), r2(y_true, pred), r2(yl, pred_log)


def _one_fold(est, X, y, tr, va, k):
    t0 = time.perf_counter()
    m = clone(est).fit(X.iloc[tr], np.log1p(y[tr]))
    fit_time = time.perf_counter() - t0
    p = m.predict(X.iloc[va])
    rmse, mae, mape = _metrics(p, y[va])
    rmse_usd, r2_usd, r2_log = _extra_metrics(p, y[va])
    nf = getattr(m[-1], "n_features_in_", np.nan) if isinstance(m, Pipeline) else np.nan
    extra = getattr(m, "selected_", None)
    return (dict(fold=k, repeat=k // 5, rmse_log=rmse, mae=mae, mape=mape, n_features=nf, selected=extra,
                 rmse_usd=rmse_usd, r2_usd=r2_usd, r2_log=r2_log, fit_time=fit_time, all_finite=bool(np.isfinite(p).all()),
                 n_iter=_n_iter(m)),
            pd.DataFrame(dict(fold=k, row=va, y=y[va], pred_log=p)))


def run_cv(estimator, X, y, folds, n_jobs=6):
    """Fold-level metrics and out-of-fold predictions of `estimator` on `folds` (point predictions are medians:
    expm1 of the log prediction, no bias correction).
    Returns (metrics DataFrame with rmse_log / mae (USD) / mape (%) / n_features per fold, predictions DataFrame)."""
    y = np.asarray(y, dtype=float)
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(estimator, X, y, tr, va, k) for k, (tr, va) in enumerate(folds))
    return pd.DataFrame([o[0] for o in out]), pd.concat([o[1] for o in out], ignore_index=True)


def summarise(metrics):
    return pd.Series(dict(n_features=metrics.n_features.mean(), rmse_log=metrics.rmse_log.mean(),
                          rmse_sd=metrics.rmse_log.std(), mae_usd=metrics.mae.mean(), mape_pct=metrics.mape.mean()))


# ---------------------------------------------------------------- paired comparison with dependent folds
def paired_vs_reference(metrics, ref, k_splits=5, level=0.95):
    """Paired fold differences (candidate - reference; negative = candidate better) in rmse_log.

    The J = n_splits * n_repeats fold scores are NOT independent (training sets overlap, validation rows repeat across
    repeats), so the naive standard error sd/sqrt(J) is too small. The Nadeau-Bengio correction multiplies the variance
    by (1/J + n_test/n_train) instead of 1/J; for 5-fold CV n_test/n_train = 1/(k-1). The interval uses a t
    distribution with J-1 degrees of freedom. It is still approximate and is a screening tool, not a proof.

    `metrics` is {name: fold metrics DataFrame}, all from the same folds."""
    r = metrics[ref].rmse_log.values
    J = len(r)
    corr = 1.0 / J + 1.0 / (k_splits - 1)
    t = stats.t.ppf(0.5 + level / 2, J - 1)
    rows = []
    for name, m in metrics.items():
        d = m.rmse_log.values - r
        sd = d.std(ddof=1)
        se_nb = np.sqrt(corr) * sd
        by_rep = pd.Series(d).groupby(np.arange(J) // k_splits).mean()
        rows.append(dict(variant=name, n_features=m["n_features"].mean() if "n_features" in m else np.nan, rmse_log=m.rmse_log.mean(),
                         rmse_sd=m.rmse_log.std(), mae_usd=m.mae.mean(), mape_pct=m.mape.mean(),
                         diff=d.mean(), se_naive=sd / np.sqrt(J), se_nb=se_nb,
                         ci_lo=d.mean() - t * se_nb, ci_hi=d.mean() + t * se_nb,
                         folds_better=int((d < 0).sum()), folds_worse=int((d > 0).sum()),
                         repeats_better=int((by_rep < 0).sum()), mae_diff=(m.mae.values - metrics[ref].mae.values).mean()))
    out = pd.DataFrame(rows).set_index("variant")
    out["verdict"] = [verdict(r_, ref == i) for i, r_ in out.iterrows()]
    return out


def verdict(row, is_ref=False, practical=0.001):
    """Pre-set reading rule: a change counts as evidence only if the corrected interval excludes 0; a mean change
    smaller than `practical` (0.001 log-RMSE, about 0.1% in price) is treated as a tie either way."""
    if is_ref:
        return "reference"
    if row.ci_hi < 0 and row["diff"] < -practical:
        return "better"
    if row.ci_lo > 0 and row["diff"] > practical:
        return "worse"
    return "tie"


# ---------------------------------------------------------------- nested selection among feature sets
class FeatureSetSelector(BaseEstimator, RegressorMixin):
    """Chooses one of several feature sets by INNER CV on the training data it is given, then refits on all of it.

    Within one outer fold this sees training rows only. Selection rule (one-standard-error): take the candidate with
    the lowest inner mean log-RMSE, then prefer the candidate with the fewest output columns among those whose inner
    mean is within one standard error of it. Ties therefore go to the simpler set."""

    def __init__(self, model, candidates=("full", "reduced", "domain_informed"), fixed=True, inner_splits=5, seed=42):
        self.model = model
        self.candidates = candidates
        self.fixed = fixed
        self.inner_splits = inner_splits
        self.seed = seed

    def fit(self, X, y):
        folds = get_folds(y, n_splits=self.inner_splits, n_repeats=1, seed=self.seed)
        scores = {}
        for c in self.candidates:
            est = make_pipeline(c, self.model, self.fixed)
            r, nf = [], []
            for tr, va in folds:
                m = clone(est).fit(X.iloc[tr], np.asarray(y)[tr])
                r.append(np.sqrt(np.mean((m.predict(X.iloc[va]) - np.asarray(y)[va]) ** 2)))
                nf.append(m[-1].n_features_in_)
            scores[c] = (np.mean(r), np.std(r, ddof=1) / np.sqrt(len(r)), np.mean(nf))
        best = min(scores, key=lambda c: scores[c][0])
        thr = scores[best][0] + scores[best][1]
        ok = [c for c in self.candidates if scores[c][0] <= thr]
        self.selected_ = min(ok, key=lambda c: scores[c][2])
        self.inner_scores_ = scores
        self.estimator_ = make_pipeline(self.selected_, self.model, self.fixed).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def __sklearn_tags__(self):
        t = super().__sklearn_tags__()
        t.estimator_type = "regressor"
        return t


# ---------------------------------------------------------------- temporal split
def temporal_split(df, last_train_year=2008):
    """Indices (train: YrSold <= last_train_year, test: later). YrSold is used as the split key only."""
    yr = df["YrSold"].values
    return np.where(yr <= last_train_year)[0], np.where(yr > last_train_year)[0]


# ---------------------------------------------------------------- permutation importance (held-out folds)
def grouped_permutation_importance(fitted, Xva, yva_log, groups, n_repeats=3, seed=0):
    """Increase in held-out log-RMSE when the columns of each group are permuted TOGETHER (same row shuffle, so the
    columns of a correlated group keep their mutual relationships but lose their link to the target).
    `groups` is {label: [raw column names]}. Uses raw input columns, so engineered features move with their sources."""
    rng = np.random.default_rng(seed)
    base = np.sqrt(np.mean((fitted.predict(Xva) - yva_log) ** 2))
    rows = []
    for label, cols in groups.items():
        d = []
        for _ in range(n_repeats):
            Xp = Xva.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xva[cols].iloc[perm].values
            d.append(np.sqrt(np.mean((fitted.predict(Xp) - yva_log) ** 2)) - base)
        rows.append((label, np.mean(d)))
    return pd.Series(dict(rows))


# ---------------------------------------------------------------- Lasso coefficient stability
def lasso_stability(fs, X, y, folds, n_jobs=6):
    """Per outer fold: fit preprocessing + LassoCV (alpha chosen by an inner CV on that training fold) and return the
    coefficient of every output column. Rows = folds, columns = output features."""
    def one(tr):
        p = make_pipeline(fs, "lasso").fit(X.iloc[tr], np.log1p(np.asarray(y, float)[tr]))
        names = p[0].transform(X.iloc[tr[:2]]).columns
        return pd.Series(p[-1].coef_, index=names), p[-1].alpha_
    out = Parallel(n_jobs=n_jobs)(delayed(one)(tr) for tr, _ in folds)
    return pd.DataFrame([o[0] for o in out]), np.array([o[1] for o in out])


# ---------------------------------------------------------------- training-only screening diagnostics
def vif_table(Z):
    """VIF of each column of a numeric design matrix (computed on TRAINING rows by the caller).
    VIF_j = 1 / (1 - R^2_j) from regressing column j on the others; exact linear dependence gives inf. Constant
    columns are skipped and reported by the caller via `matrix rank`."""
    A = Z.loc[:, Z.std() > 0].astype(float)
    A = (A - A.mean()) / A.std()
    M = A.values
    out = {}
    for j, c in enumerate(A.columns):
        others = np.delete(M, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, M[:, j], rcond=None)
        resid = M[:, j] - others @ beta
        r2 = 1 - resid.var() / M[:, j].var()
        out[c] = np.inf if r2 > 1 - 1e-9 else 1 / (1 - r2)
    return pd.Series(out), int(np.linalg.matrix_rank(M)), M.shape[1]


def mutual_information(Z, y_log, seed=42, max_unique_discrete=15):
    """Mutual information (nats) between each output column and log price, on the rows given (training only).
    Columns with few distinct values (one-hot, ordinal codes, counts, flags) are treated as discrete."""
    discrete = (Z.nunique() <= max_unique_discrete).values
    return pd.Series(mutual_info_regression(Z.values, y_log, discrete_features=discrete, random_state=seed),
                     index=Z.columns)
