"""Segment definitions, inference classes, bootstrap and temporal-leakage checks.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src.data.loading import load_raw
from src.evaluation import segments as S
from src.evaluation import temporal as TP
from src.evaluation import final_evaluation as E


class DefinitionTests(unittest.TestCase):
    def test_bins_are_the_documented_ones(self):
        self.assertEqual(S.PRICE_EDGES, (130_000, 280_000))
        self.assertEqual(S.QUALITY_BINS[0], [0, 4, 8, 10])                # EDA F23 tiers 1-4 / 5-8 / 9-10
        df = load_raw().iloc[:50]
        d = S.add_segments(pd.DataFrame(dict(row=np.arange(50), Id=df.Id.values)), df)
        self.assertTrue(set(d.quality_tier.dropna().unique()) <= set(S.QUALITY_BINS[1]))
        self.assertTrue((d.house_age == 2010 - df.YearBuilt.values).all())  # fixed reference year, not the sale year

    def test_segment_stats_match_hand_computation(self):
        d = E.add_errors(pd.DataFrame(dict(model="m", fold=0, row=[0, 1], y=[100_000.0, 300_000.0], pred_log=np.log1p([110_000.0, 270_000.0]))))
        s = S.seg_stats(d)
        self.assertAlmostEqual(s.mae, 20_000, places=3)
        self.assertAlmostEqual(s.signed_usd, (-10_000 + 30_000) / 2, places=3)       # positive = underprediction
        self.assertAlmostEqual(s.mape, 100 * (0.10 + 0.10) / 2, places=6)
        self.assertAlmostEqual(s.rmse_usd, np.sqrt((1e8 + 9e8) / 2), places=3)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 600
        y = rng.uniform(100_000, 300_000, n)
        grp = np.where(np.arange(n) < 100, "bad", "good")
        noise = np.where(grp == "bad", 0.30, 0.08) * rng.standard_normal(n)
        d = pd.DataFrame(dict(model="m", fold=0, row=np.arange(n), y=y, pred_log=np.log1p(y) - noise, grp=grp))
        self.d = E.add_errors(d)

    def test_clearly_worse_segment_is_supported_and_practical(self):
        t = S.seg_vs_rest(self.d, "grp", n_boot=500)
        self.assertEqual(t.loc["bad", "class"], "supported + practical")
        self.assertLess(t.loc["good", "logrmse_diff_vs_rest"], 0)

    def test_equal_segments_are_descriptive_only(self):
        d = self.d.copy(); d["grp"] = np.where(np.arange(len(d)) % 2 == 0, "a", "b")
        t = S.seg_vs_rest(d, "grp", n_boot=500)
        self.assertTrue((t["class"] == "descriptive only").all())

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self):
        a = S.bootstrap_ci(self.d, n_boot=300); b = S.bootstrap_ci(self.d, n_boot=300)
        self.assertEqual(a, b)
        est = np.mean(np.abs(self.d.residual_usd))
        self.assertLess(a["mae"][0], est); self.assertGreater(a["mae"][1], est)


class TemporalLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw()
        cls.X = cls.df.drop(columns=["SalePrice", "Id"])
        cls.y = cls.df["SalePrice"].values.astype(float)
        cls.tr, cls.te = TP.split_indices(cls.df)

    def test_split_matches_notebook_03_definition(self):
        self.assertEqual((len(self.tr), len(self.te)), (947, 513))
        self.assertLessEqual(self.df.YrSold.values[self.tr].max(), 2008)
        self.assertGreaterEqual(self.df.YrSold.values[self.te].min(), 2009)
        self.assertFalse(set(self.tr) & set(self.te))

    def test_yrsold_is_not_a_model_input(self):
        prep = TP.FEATURE_SETS["reduced"].build("baseline", scale=None).fit(self.X.iloc[self.tr], np.log1p(self.y[self.tr]))
        cols = list(prep.transform(self.X.iloc[self.te[:5]]).columns)
        self.assertFalse([c for c in cols if c.startswith(("YrSold", "MoSold", "SaleType", "SaleCondition"))])

    def test_historical_tuning_cannot_see_test_year_labels(self):
        y2 = self.y.copy(); y2[self.te] = y2[self.te] * 9 + 1e6               # corrupt only the 2009-2010 labels
        a = TP.tune_historical("elasticnet", self.X, self.y, self.tr, 6)
        b = TP.tune_historical("elasticnet", self.X, y2, self.tr, 6)
        self.assertEqual(a[0], b[0]); self.assertEqual(a[1], b[1])

    def test_temporal_prediction_does_not_depend_on_test_labels(self):
        ly = np.log1p(self.y); ly2 = ly.copy(); ly2[self.te] += 3.0
        p1 = TP.fit_predict("elasticnet", "fixed", None, self.X, ly, self.tr, self.te)
        p2 = TP.fit_predict("elasticnet", "fixed", None, self.X, ly2, self.tr, self.te)
        np.testing.assert_allclose(p1, p2)


class CachedTests(unittest.TestCase):
    def setUp(self):
        if not (TP.OUT / "temporal_metrics.csv").exists():
            self.skipTest("run python -m src.evaluation.temporal first")

    def test_metrics_files_are_complete_and_finite(self):
        m = pd.read_csv(TP.OUT / "temporal_metrics.csv")
        self.assertEqual(set(m.config), {"fixed", "hist_tuned", "all_years_tuned"})
        self.assertTrue(np.isfinite(m[["log_rmse", "mae", "rmse_usd", "mape"]].values).all())
        self.assertTrue((m.n_train == 947).all())
        r = pd.read_csv(TP.OUT / "random_matched_splits.csv")
        self.assertTrue((r.groupby("family").size() == TP.N_RANDOM_SPLITS).all())


if __name__ == "__main__":
    unittest.main()
