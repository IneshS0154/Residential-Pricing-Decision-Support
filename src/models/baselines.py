"""Central registry of the FIXED-hyperparameter baseline models (notebook 04).

No search and no CV-tuned estimator (no RidgeCV / LassoCV / ElasticNetCV / *SearchCV) appears here, so the baseline
comparison stays independent of later tuning. Each entry records how its configuration was obtained:

  library default      sklearn defaults (plus a seed): nothing chosen by us
  fixed practical      a documented departure from the default for numerical or computational reasons
  previously informed  magnitude or value taken from earlier project experiments; labelled so it is never read as a
                       neutral baseline

Entries with primary=False are supplementary and never enter the primary ranking.
Scaling: linear models get a robust-scaled preprocessor, tree ensembles an unscaled one, the dummy no preprocessor.
"""
from dataclasses import dataclass

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.features.feature_sets import get_feature_set

SEED = 42
MAX_ITER = 100_000


@dataclass(frozen=True)
class Baseline:
    name: str
    label: str
    family: str                 # reference | linear | tree
    config_class: str           # library default | fixed practical | previously informed
    scale: object               # 'robust' | None
    factory: object             # () -> unfitted estimator
    rationale: str
    primary: bool = True

    def params(self):
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in self.factory().get_params().items()}


REGISTRY = {b.name: b for b in [
    Baseline("dummy", "Dummy (mean)", "reference", "library default", None,
             lambda: DummyRegressor(strategy="mean"),
             "predicts the training mean of log1p(price); no preprocessing; a model that cannot beat it has learned nothing"),
    Baseline("ols", "OLS", "linear", "library default", "robust", lambda: LinearRegression(),
             "interpretable linear reference; lstsq gives the minimum-norm solution if columns are collinear"),
    Baseline("ridge", "Ridge", "linear", "library default", "robust", lambda: Ridge(alpha=1.0),
             "sklearn default alpha=1.0 on robust-scaled inputs; not tuned"),
    Baseline("lasso", "Lasso", "linear", "previously informed", "robust",
             lambda: Lasso(alpha=1e-3, max_iter=MAX_ITER, random_state=SEED),
             "default alpha=1.0 shrinks every coefficient to zero on this target (sd 0.40 on the log scale), so it would "
             "equal the Dummy; alpha=1e-3 has the order of magnitude of the inner-CV alphas seen in notebook 03 "
             "(6e-4 to 7.5e-4)"),
    Baseline("elasticnet", "ElasticNet", "linear", "previously informed", "robust",
             lambda: ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=MAX_ITER, random_state=SEED),
             "same alpha scale as Lasso; l1_ratio 0.5 is the symmetric mixture, not searched"),
    Baseline("rf", "Random Forest", "tree", "fixed practical", None,
             lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
             "sklearn defaults (fully grown trees, all features per split) with 300 trees for stable averaging; "
             "n_jobs=1 because parallelism is across folds"),
    Baseline("gbr", "Gradient Boosting", "tree", "library default", None,
             lambda: GradientBoostingRegressor(random_state=SEED),
             "sklearn defaults: 100 trees, depth 3, learning rate 0.1, no subsampling"),
    # ---- supplementary: reported separately, never part of the primary ranking
    Baseline("gbr_nb03", "GBR (nb03 config)", "tree", "previously informed", None,
             lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                               min_samples_leaf=5, random_state=SEED),
             "configuration used in the notebook 03 ablations; chosen a priori there but in a context where results were "
             "visible, so not a neutral baseline", primary=False),
    Baseline("ridge_a10", "Ridge (alpha 10)", "linear", "previously informed", "robust", lambda: Ridge(alpha=10.0),
             "alpha used as a fixed measuring instrument in notebooks 02-03", primary=False),
]}
PRIMARY = tuple(n for n, b in REGISTRY.items() if b.primary)


def build_baseline(name, feature_set="reduced", strategy="baseline"):
    """Unfitted Pipeline for one baseline on one feature set (raw load_raw() columns in, log1p price out)."""
    b = REGISTRY[name]
    model = b.factory()
    if name == "dummy":
        return Pipeline([("model", model)])
    return Pipeline([("prep", get_feature_set(feature_set).build(strategy, scale=b.scale)), ("model", model)])
