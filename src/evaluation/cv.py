"""Shared cross-validation folds so every model and preprocessing strategy is compared on identical splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def price_bands(y, n_bands=5):
    """Quantile price bands used to stratify folds (low / mid / high priced homes in every fold)."""
    return pd.qcut(np.asarray(y), n_bands, labels=False, duplicates="drop")


def get_folds(y, n_splits=5, n_repeats=3, seed=42, n_bands=5):
    """List of (train_idx, val_idx), repeated k-fold stratified by price band."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), price_bands(y, n_bands)))


def cv_scores(estimator, X, y, folds, ids=None, drop_train_ids=(), eval_exclude_ids=()):
    """Per-fold scores for a model that predicts log1p(price).

    Returns a DataFrame with `rmse_log` (RMSE on log1p(SalePrice)) and `mae` (mean absolute error in
    currency, computed on expm1 of the prediction). The estimator (including all preprocessing) is
    cloned and fitted on the training part of every fold only.
    `drop_train_ids` removes rows from training folds only; `eval_exclude_ids` removes rows from validation folds.
    """
    from sklearn.base import clone
    y = np.asarray(y, dtype=float)
    ly = np.log1p(y)
    ids = None if ids is None else np.asarray(ids)
    rows = []
    for tr, va in folds:
        if len(drop_train_ids):
            tr = tr[~np.isin(ids[tr], list(drop_train_ids))]
        if len(eval_exclude_ids):
            va = va[~np.isin(ids[va], list(eval_exclude_ids))]
        model = clone(estimator).fit(X.iloc[tr], ly[tr])
        pred = model.predict(X.iloc[va])
        rows.append((np.sqrt(np.mean((pred - ly[va]) ** 2)), np.mean(np.abs(np.expm1(pred) - y[va]))))
    return pd.DataFrame(rows, columns=["rmse_log", "mae"])


def summarize(scores):
    """Mean and standard deviation of RMSE and MAE over folds."""
    return pd.Series({"RMSE_mean": scores.rmse_log.mean(), "RMSE_sd": scores.rmse_log.std(),
                      "MAE_mean": scores.mae.mean(), "MAE_sd": scores.mae.std()})


def paired_difference(scores_a, scores_b, metric="rmse_log"):
    """Mean of (a - b) over the same folds and its standard error (negative = a is better).
    Folds from repeated CV overlap, so the standard error is approximate."""
    d = scores_a[metric].values - scores_b[metric].values
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
