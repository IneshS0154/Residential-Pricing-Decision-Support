"""Candidate point-model factories used for feature-set comparison.

Every factory returns an UNFITTED regressor for log1p(SalePrice). Linear models are meant to follow a scaled
preprocessor (scale='robust'), tree ensembles an unscaled one (scale=None). Hyperparameters that depend on the data
(alpha, l1_ratio) are chosen by an inner CV inside `fit`, so they only ever see the training rows of the outer fold.
"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge, RidgeCV
import numpy as np

SEED = 42
ALPHAS = np.logspace(-3, 2, 30)

LINEAR = ("ols", "ridge", "lasso", "elasticnet")
TREES = ("rf", "gbr")
MODELS = LINEAR + TREES


def scale_for(model):
    """Preprocessor scaling appropriate for the model family."""
    return "robust" if model in LINEAR else None


def make_model(name, fixed=False):
    """fixed=True gives the fixed-hyperparameter versions used in ablations (Ridge alpha=10 as in notebook 02)."""
    if name == "ols":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=10) if fixed else RidgeCV(alphas=np.logspace(-1, 2.5, 30))
    if name == "lasso":
        return LassoCV(alphas=np.logspace(-4, -1, 25), cv=5, max_iter=20000, random_state=SEED)
    if name == "elasticnet":
        return ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-4, -1, 25), cv=5, max_iter=20000,
                            random_state=SEED)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33, n_jobs=1,
                                     random_state=SEED)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                         min_samples_leaf=5, random_state=SEED)
    raise ValueError(f"model must be one of {MODELS}")
