"""Conformal quantile, CQR score/interval maths, split roles and leakage checks.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from src.data.loading import load_raw
from src.evaluation import intervals as I


class ConformalQuantileTests(unittest.TestCase):
    def test_k_is_exact_ceiling_of_n_plus_1_times_1_minus_alpha(self):
        self.assertEqual(I.conformal_k(9), 8)            # (9+1)*0.8 = 8 exactly; naive float ceil gives 9
        self.assertEqual(I.conformal_k(4), 4)
        self.assertEqual(I.conformal_k(99), 80)
        self.assertEqual(I.conformal_k(100), 81)         # 101*0.8 = 80.8
        self.assertEqual(I.conformal_k(292), 235)        # the calibration size used in the experiments
        self.assertEqual(I.conformal_k(19, 0.10), 18)    # 20*0.9 = 18 exactly

    def test_kth_smallest_score_is_returned(self):
        s = np.arange(1, 10)[::-1].astype(float)          # unsorted 9..1
        q, k = I.conformal_quantile(s, 0.2)
        self.assertEqual((q, k), (8.0, 8))

    def test_infinite_quantile_when_k_exceeds_n(self):
        q, k = I.conformal_quantile([1.0, 2.0, 3.0], 0.2)   # k = ceil(4*0.8) = 4 > 3
        self.assertTrue(np.isinf(q) and q > 0)
        self.assertEqual(k, 4)
        self.assertTrue(np.isinf(I.conformal_quantile([], 0.2)[0]))
        q, k = I.conformal_quantile([1, 2, 3, 4], 0.2)       # k = 4 = n: the maximum, finite
        self.assertEqual((q, k), (4.0, 4))

    def test_scores_are_not_clipped_and_signed(self):
        s = I.conformity_scores([1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [2.0, 0.5, 4.0])
        np.testing.assert_allclose(s, [-1.0, 0.5, 1.0])      # inside: -min(y-L, U-y) <0; below: L-y; above: y-U
        self.assertTrue((s[0] < 0))

    def test_cqr_interval_applies_correction_on_both_sides(self):
        lo, hi = I.cqr_interval([1.0, 2.0], [3.0, 4.0], 0.5)
        np.testing.assert_allclose(lo, [0.5, 1.5]); np.testing.assert_allclose(hi, [3.5, 4.5])
        lo, hi = I.cqr_interval([1.0], [3.0], -0.25)         # negative correction shrinks
        self.assertAlmostEqual(hi[0] - lo[0], 1.5)

    def test_marginal_coverage_of_split_cqr_on_exchangeable_data(self):
        rng = np.random.default_rng(0)
        n_cal, trials, cov = 49, 4000, []
        for _ in range(trials):                              # deliberately miscalibrated base interval
            y = rng.normal(0, 1, n_cal + 1)
            L, U = np.full(n_cal + 1, -0.3), np.full(n_cal + 1, 0.3)
            q, _ = I.conformal_quantile(I.conformity_scores(L[:n_cal], U[:n_cal], y[:n_cal]), 0.2)
            lo, hi = I.cqr_interval(L[-1:], U[-1:], q)
            cov.append(float(lo[0] <= y[-1] <= hi[0]))
        # for continuous scores coverage is k/(n+1) = 40/50 = 0.80 exactly in expectation
        self.assertAlmostEqual(np.mean(cov), 0.80, delta=0.02)

    def test_crossing_policy_sorts_and_counts(self):
        lo, hi, n = I.rearrange([3.0, 1.0, 2.0], [1.0, 2.0, 2.0])
        self.assertEqual(n, 1)
        self.assertTrue((lo <= hi).all())
        np.testing.assert_allclose(lo, [1.0, 1.0, 2.0]); np.testing.assert_allclose(hi, [3.0, 2.0, 2.0])


class MetricTests(unittest.TestCase):
    def test_usd_conversion_coverage_width_and_winkler(self):
        d = pd.DataFrame(dict(y=[100.0, 200.0, 300.0], point=[100.0, 150.0, 200.0],
                              L_raw=np.log1p([90.0, 210.0, 250.0]), U_raw=np.log1p([110.0, 260.0, 280.0])))
        d["L_cqr"], d["U_cqr"] = d["L_raw"], d["U_raw"]
        r = I.add_usd(d)
        np.testing.assert_allclose(r.lo_raw, [90, 210, 250], atol=1e-9)
        self.assertEqual(list(r.cover_raw), [True, False, False])
        self.assertEqual(list(r.miss_raw), ["inside", "below", "above"])
        np.testing.assert_allclose(r.width_raw, [20, 50, 30], atol=1e-9)
        np.testing.assert_allclose(r.relwidth_raw, [0.2, 50 / 150, 0.15], atol=1e-9)
        np.testing.assert_allclose(r.winkler_raw, [20, 50 + 10 * 10, 30 + 10 * 20], atol=1e-6)     # 2/alpha = 10
        self.assertTrue((r.width_raw >= 0).all() and (r.point > 0).all())

    def test_interval_metrics_gap_and_ci(self):
        d = pd.DataFrame(dict(y=np.arange(1, 11) * 100.0, point=np.arange(1, 11) * 100.0,
                              L_raw=np.log1p(np.arange(1, 11) * 100.0 - 10), U_raw=np.log1p(np.arange(1, 11) * 100.0 + 10)))
        d["L_cqr"], d["U_cqr"] = d["L_raw"], d["U_raw"]
        m = I.interval_metrics(I.add_usd(d), "raw")
        self.assertEqual(m.coverage, 1.0); self.assertAlmostEqual(m.coverage_gap_pp, 20.0)
        self.assertLessEqual(m.cov_ci_lo, 1.0); self.assertGreater(m.cov_ci_lo, 0.6)

    def test_wilson_interval(self):
        lo, hi = I.wilson(80, 100)
        self.assertLess(lo, 0.8); self.assertGreater(hi, 0.8); self.assertTrue(0 <= lo < hi <= 1)


class SplitRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_raw()
        cls.X = df.drop(columns=["SalePrice", "Id"]).iloc[:600].reset_index(drop=True)
        cls.y = df["SalePrice"].values[:600].astype(float)
        cls.ly = np.log1p(cls.y)
        cls.tr, cls.va = np.arange(450), np.arange(450, 600)

    def run_it(self, ly):
        return I.fit_split_cqr(self.X.iloc[self.tr], ly[self.tr], self.y[self.tr], self.X.iloc[self.va], seed=7)

    def test_proper_calibration_and_evaluation_sets_are_disjoint(self):
        iv, info = self.run_it(self.ly)
        p, c = set(info["proper_idx"]), set(info["cal_idx"])
        self.assertFalse(p & c)
        self.assertEqual(p | c, set(range(len(self.tr))))
        self.assertAlmostEqual(len(c) / len(self.tr), I.CAL_FRAC, delta=0.01)
        self.assertEqual(len(iv["L_cqr"]), len(self.va))                      # evaluation rows are never part of either set
        self.assertTrue(np.isfinite(info["q_hat"]))

    def test_intervals_valid_and_correction_symmetric(self):
        iv, info = self.run_it(self.ly)
        self.assertTrue((iv["L_raw"] <= iv["U_raw"]).all())
        np.testing.assert_allclose(iv["L_cqr"], iv["L_raw"] - info["q_hat"]); np.testing.assert_allclose(iv["U_cqr"], iv["U_raw"] + info["q_hat"])

    def test_calibration_labels_change_only_the_correction_not_the_base_models(self):
        iv0, info0 = self.run_it(self.ly)
        ly2 = self.ly.copy(); ly2[self.tr[info0["cal_idx"]]] += 1.0          # shift ONLY calibration labels
        iv1, info1 = self.run_it(ly2)
        np.testing.assert_allclose(iv0["L_raw"], iv1["L_raw"]); np.testing.assert_allclose(iv0["U_raw"], iv1["U_raw"])
        self.assertNotAlmostEqual(info0["q_hat"], info1["q_hat"], places=3)

    def test_proper_training_labels_change_the_base_models(self):
        iv0, info0 = self.run_it(self.ly)
        ly2 = self.ly.copy(); ly2[self.tr[info0["proper_idx"]]] += 1.0
        iv1, _ = self.run_it(ly2)
        self.assertGreater(np.abs(iv0["L_raw"] - iv1["L_raw"]).max(), 0.1)

    def test_evaluation_labels_cannot_affect_intervals(self):
        iv0, info0 = self.run_it(self.ly)
        ly2 = self.ly.copy(); ly2[self.va] += 5.0                              # corrupt evaluation labels
        iv1, info1 = self.run_it(ly2)
        for k in iv0: np.testing.assert_allclose(iv0[k], iv1[k])
        self.assertEqual(info0["q_hat"], info1["q_hat"])

    def test_deterministic(self):
        a, _ = self.run_it(self.ly); b, _ = self.run_it(self.ly)
        for k in a: np.testing.assert_allclose(a[k], b[k])

    def test_quantile_models_use_documented_alphas_and_loss(self):
        self.assertEqual((I.Q_LO, I.Q_HI, I.ALPHA), (0.10, 0.90, 0.20))
        m = I.make_quantile_model(I.Q_LO)
        self.assertEqual((m.loss, m.alpha), ("quantile", 0.10))


class ConfigurationPlumbingTests(unittest.TestCase):
    def test_temporal_run_passes_the_quantile_configuration_not_the_point_model_parameters(self):
        import inspect
        src = inspect.getsource(I.run_temporal)
        self.assertIn("point_params", src)                       # point-model parameters have their own name
        self.assertIn("seed=seed, params=params", src)           # the argument `params` reaches the quantile models untouched
        import re
        self.assertIsNone(re.search(r"(?<![A-Za-z_])params = json", src))      # no bare `params` reassignment from the tuned point parameters

    def test_different_configurations_give_different_intervals(self):
        df = load_raw()
        X = df.drop(columns=["SalePrice", "Id"]).iloc[:400].reset_index(drop=True)
        y = df["SalePrice"].values[:400].astype(float)
        a, _ = I.fit_split_cqr(X.iloc[:300], np.log1p(y[:300]), y[:300], X.iloc[300:], seed=3)
        b, _ = I.fit_split_cqr(X.iloc[:300], np.log1p(y[:300]), y[:300], X.iloc[300:], seed=3, params=I.QUANTILE_PARAMS_REGULARISED)
        self.assertGreater(np.abs(a["L_raw"] - b["L_raw"]).max(), 1e-3)


class CachedResultTests(unittest.TestCase):
    def setUp(self):
        if not (I.OUT / "cv_intervals.csv.gz").exists():
            self.skipTest("run python -m src.evaluation.intervals first")
        self.d = pd.read_csv(I.OUT / "cv_intervals.csv.gz")

    def test_same_evaluation_rows_for_both_methods_each_house_once_per_repeat(self):
        self.assertEqual(self.d.groupby(["repeat", "row"]).size().unique().tolist(), [1])
        self.assertEqual(self.d.row.nunique(), 1460)

    def test_valid_finite_nonnegative_widths_and_cqr_wider_when_q_positive(self):
        self.assertTrue(np.isfinite(self.d[["lo_raw", "hi_raw", "lo_cqr", "hi_cqr"]].values).all())
        self.assertTrue((self.d.width_raw >= 0).all() and (self.d.width_cqr >= 0).all())
        self.assertTrue((self.d.point > 0).all())
        pos = self.d.q_hat > 0
        self.assertTrue((self.d[pos].width_cqr >= self.d[pos].width_raw).all())

    def test_coverage_indicator_matches_definition(self):
        c = (self.d.y >= self.d.lo_cqr) & (self.d.y <= self.d.hi_cqr)
        self.assertTrue((c == self.d.cover_cqr).all())


if __name__ == "__main__":
    unittest.main()
