"""Baseline registry, metric correctness and fold-sharing checks.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src.data.loading import load_raw
from src.evaluation import feature_selection as F
from src.evaluation.baseline_comparison import OUT
from src.evaluation.cv import get_folds
from src.models.baselines import PRIMARY, REGISTRY, build_baseline


class RegistryTests(unittest.TestCase):
    def test_seven_primary_models_including_dummy(self):
        self.assertEqual(set(PRIMARY), {"dummy", "ols", "ridge", "lasso", "elasticnet", "rf", "gbr"})

    def test_no_tuning_estimators_or_searches(self):
        for n in REGISTRY:
            cls = type(REGISTRY[n].factory()).__name__
            self.assertFalse("CV" in cls or "Search" in cls, (n, cls))

    def test_seeds_fixed_and_configs_labelled(self):
        for n, b in REGISTRY.items():
            p = b.params()
            if n in ("lasso", "elasticnet", "rf", "gbr", "gbr_nb03"):       # the stochastic estimators
                self.assertEqual(p["random_state"], 42, n)
            self.assertIn(b.config_class, {"library default", "fixed practical", "previously informed"})
        self.assertFalse(any(REGISTRY[n].primary for n in ("gbr_nb03", "ridge_a10")))

    def test_dummy_has_no_preprocessing_and_linear_models_are_scaled(self):
        self.assertEqual([s for s, _ in build_baseline("dummy").steps], ["model"])
        self.assertEqual(REGISTRY["ols"].scale, "robust")
        self.assertIsNone(REGISTRY["rf"].scale)


class MetricTests(unittest.TestCase):
    def test_metrics_on_known_values(self):
        y = np.array([100.0, 200.0, 400.0])
        pred_log = np.log1p(np.array([110.0, 180.0, 400.0]))
        rmse_log, mae, mape = F._metrics(pred_log, y)
        self.assertAlmostEqual(mae, 10.0)
        self.assertAlmostEqual(mape, 100 * np.mean([0.1, 0.1, 0.0]))
        self.assertAlmostEqual(rmse_log, np.sqrt(np.mean((pred_log - np.log1p(y)) ** 2)))
        rmse_usd, r2_usd, r2_log = F._extra_metrics(pred_log, y)
        self.assertAlmostEqual(rmse_usd, np.sqrt((100 + 400 + 0) / 3))
        self.assertAlmostEqual(r2_usd, 1 - 500 / np.sum((y - y.mean()) ** 2))
        self.assertLess(r2_log, 1.0)

    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 400.0])
        rmse, r2, _ = F._extra_metrics(np.log1p(y), y)
        self.assertAlmostEqual(rmse, 0.0, places=6)
        self.assertAlmostEqual(r2, 1.0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_raw()
        cls.X = df.drop(columns=["SalePrice", "Id"]).iloc[:500].reset_index(drop=True)
        cls.y = df["SalePrice"].values[:500].astype(float)

    def test_dummy_predicts_training_mean_and_ignores_validation_target(self):
        tr, va = np.arange(300), np.arange(300, 500)
        m = build_baseline("dummy").fit(self.X.iloc[tr], np.log1p(self.y[tr]))
        np.testing.assert_allclose(m.predict(self.X.iloc[va]), np.log1p(self.y[tr]).mean())

    def test_all_primary_models_run_and_are_finite_on_shared_folds(self):
        folds = get_folds(self.y, n_splits=3, n_repeats=1)
        for n in PRIMARY:
            met, pr = F.run_cv(build_baseline(n), self.X, self.y, folds, n_jobs=2)
            self.assertTrue(met.all_finite.all(), n)
            self.assertTrue(np.isfinite(pr.pred_log).all(), n)
            self.assertEqual(sorted(met.fold), [0, 1, 2], n)

    def test_cached_results_use_identical_folds_for_every_model(self):
        path = OUT / "expA_folds.csv"
        if not path.exists():
            self.skipTest("run python -m src.evaluation.baseline_comparison first")
        a = pd.read_csv(path)
        self.assertEqual(a.groupby("model").fold.apply(lambda s: tuple(sorted(s))).nunique(), 1)
        pr = pd.read_csv(OUT / "expA_predictions.csv.gz")
        keys = pr.groupby("model").apply(lambda d: tuple(zip(d.fold, d.row)))
        self.assertEqual(keys.nunique(), 1)                                  # same validation rows in the same folds
        self.assertTrue(np.isfinite(pr.pred_log).all())


if __name__ == "__main__":
    unittest.main()
