"""Final-evaluation definitions: residual sign, APE, segments, quantiles, flagged ids, out-of-fold structure.
Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src.evaluation import final_evaluation as E
from src.features import preprocessing as P


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.d = pd.DataFrame(dict(model="m", fold=0, row=[0, 1, 2], y=[100_000.0, 200_000.0, 400_000.0],
                                   pred_log=np.log1p([110_000.0, 150_000.0, 400_000.0])))

    def test_residual_is_actual_minus_predicted(self):
        r = E.add_errors(self.d)
        np.testing.assert_allclose(r.residual_usd, [-10_000, 50_000, 0], atol=1e-6)      # negative = overprediction
        self.assertTrue((r.residual_log.iloc[:2].values * np.array([-1, 1]) > 0).all())
        np.testing.assert_allclose(r.abs_err_usd, [10_000, 50_000, 0], atol=1e-6)
        np.testing.assert_allclose(r.sq_err_usd, r.residual_usd ** 2)

    def test_ape_and_back_transform(self):
        r = E.add_errors(self.d)
        np.testing.assert_allclose(r.pred, [110_000, 150_000, 400_000], rtol=1e-9)
        np.testing.assert_allclose(r.ape, [0.10, 0.25, 0.0], atol=1e-9)

    def test_segments_use_fixed_edges(self):
        self.assertEqual(E.SEGMENT_EDGES, (130_000, 280_000))
        self.assertEqual(list(E.segment_of([129_999, 130_000, 279_999, 280_000])), ["low", "mid", "mid", "luxury"])
        r = E.add_errors(self.d)
        self.assertEqual(list(r.segment_pred), ["low", "mid", "luxury"])
        self.assertEqual(list(r.segment_actual), ["low", "mid", "luxury"])

    def test_quantiles_and_concentration(self):
        q = E.error_quantiles(np.arange(1, 101))
        self.assertEqual((q["max"], q["median"]), (100, 50.5))
        self.assertAlmostEqual(E.concentration(np.ones(100)), 0.10)
        self.assertGreater(E.concentration(np.r_[np.ones(99), 1000.0], 0.01), 0.9)

    def test_flagged_ids_come_from_the_documented_decision(self):
        self.assertEqual(E.FLAGGED_IDS, (524, 1299))
        self.assertEqual(tuple(P.SUGGESTED_TRAINING_EXCLUSIONS), E.FLAGGED_IDS)

    def test_evaluation_only_exclusion_changes_metrics_not_predictions(self):
        d = pd.concat([self.d.assign(Id=[1, 2, 524])]).assign(model="m")
        r = E.add_errors(d)
        full, ex = E.fold_metrics(r), E.fold_metrics(r, (524,))
        self.assertEqual(int(ex.n.iloc[0]), 2)
        self.assertAlmostEqual(full.mae.iloc[0], 20_000, places=3)       # (10k + 50k + 0) / 3
        self.assertAlmostEqual(ex.mae.iloc[0], 30_000, places=3)         # (10k + 50k) / 2: only the scored rows change
        pd.testing.assert_series_equal(r.pred_log, E.add_errors(d).pred_log)

    def test_bootstrap_is_deterministic(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(80_000, 400_000, 300)
        d = E.add_errors(pd.DataFrame(dict(model="m", fold=0, row=np.arange(300), y=y, pred_log=np.log1p(y * rng.normal(1, .1, 300)), repeat=0)))
        a = E.bootstrap_segment_ci(d, "m", n_boot=100, seed=3)
        b = E.bootstrap_segment_ci(d, "m", n_boot=100, seed=3)
        pd.testing.assert_frame_equal(a, b)


class OutOfFoldStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oof = E.load_oof()
        except FileNotFoundError:
            cls.oof = None

    def setUp(self):
        if self.oof is None:
            self.skipTest("run the tuning and baseline stages first")

    def test_every_house_is_predicted_once_per_repeat_by_every_model(self):
        for m, g in self.oof.groupby("model"):
            self.assertEqual(g.groupby(["repeat", "row"]).size().unique().tolist(), [1], m)
            self.assertEqual(g.row.nunique(), 1460, m)
            self.assertEqual(len(g), 3 * 1460, m)

    def test_models_share_identical_validation_rows_in_every_fold(self):
        keys = self.oof.groupby("model").apply(lambda d: tuple(zip(d.fold, d.row)), include_groups=False)
        self.assertEqual(keys.nunique(), 1)

    def test_values_are_finite_and_positive_prices(self):
        self.assertTrue(np.isfinite(self.oof[["pred", "residual_usd", "ape", "pred_log"]].values).all())
        self.assertTrue((self.oof.y > 0).all())

    def test_segment_table_counts_houses_not_repeated_predictions(self):
        t = E.segment_table(self.oof, "gbr_tuned")
        self.assertAlmostEqual(t.n.sum(), 1460, places=6)        # per-repeat house counts, not 3 x 1460


if __name__ == "__main__":
    unittest.main()
