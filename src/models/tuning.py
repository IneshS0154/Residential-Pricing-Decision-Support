"""Optuna search spaces for the shortlisted model families (notebook 05 / run: src.evaluation.nested_tuning).

Shortlist (from notebook 04, fixed-hyperparameter baselines): ElasticNet and Gradient Boosting. Ridge and Lasso are not
tuned separately: ElasticNet with `l1_ratio` in [0.1, 0.9] spans the regularised linear family (the pure-L2 and pure-L1
limits are the boundaries of the l1_ratio range, not included). Random Forest was not shortlisted (see docs/baseline_comparison.md).

Search-space principles
  * regularisation strengths are searched on a logarithmic scale (their effect is multiplicative);
  * ranges bracket the values seen in earlier notebooks (alphas 6e-4 .. 7.5e-4 from notebook 03; Ridge alpha 1 .. 10) with
    several decades of margin on each side, so a boundary optimum signals a genuinely too-narrow range;
  * every sampled combination is valid by construction (no conditional or dependent parameters);
  * max_iter is a convergence control, not a predictive hyperparameter, and is fixed.
"""
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

SEED = 42
MAX_ITER = 100_000

FAMILIES = ("elasticnet", "gbr")
SCALE = {"elasticnet": "robust", "gbr": None}          # linear model scaled, trees unscaled (same as the baselines)

# (low, high, scale) per parameter; used both for sampling and for boundary diagnostics
SPACES = {
    "elasticnet": {"alpha": (1e-5, 1e1, "log"), "l1_ratio": (0.1, 0.9, "float")},
    "gbr": {"n_estimators": (50, 500, "int(step 25)"), "learning_rate": (0.01, 0.2, "log"), "max_depth": (2, 5, "int"),
            "subsample": (0.6, 1.0, "float"), "min_samples_leaf": (1, 15, "int")},
}


def suggest(trial, family):
    """Sample one parameter set for `family` from SPACES."""
    p = {}
    for name, (lo, hi, kind) in SPACES[family].items():
        if kind == "log":
            p[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "float":
            p[name] = trial.suggest_float(name, lo, hi)
        elif kind == "int":
            p[name] = trial.suggest_int(name, lo, hi)
        else:
            p[name] = trial.suggest_int(name, lo, hi, step=25)
    return p


def make_estimator(family, params, seed=SEED):
    if family == "elasticnet":
        return ElasticNet(max_iter=MAX_ITER, random_state=seed, **params)
    if family == "gbr":
        return GradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"family must be one of {FAMILIES}")


def validate_params(family, params):
    """True if every parameter lies inside its documented range (used by tests and after tuning)."""
    for name, (lo, hi, _) in SPACES[family].items():
        if name not in params or not (lo <= params[name] <= hi):
            return False
    return set(params) == set(SPACES[family])


def at_boundary(family, name, value, tol=0.02):
    """True if `value` lies within `tol` of the range edge (fraction of the range; of log10-range for log spaces)."""
    lo, hi, kind = SPACES[family][name]
    if kind == "log":
        lo, hi, value = np.log10(lo), np.log10(hi), np.log10(value)
    span = hi - lo
    return bool(value <= lo + tol * span or value >= hi - tol * span)


class ConvergenceRecorder:
    """Context manager counting sklearn ConvergenceWarnings raised while it is active."""

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)

    @property
    def count(self):
        return sum("onverge" in str(w.category.__name__) + str(w.message) for w in self._log)
