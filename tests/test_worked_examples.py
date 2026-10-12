"""Worked demonstration and per-property decomposition: reproducibility, label independence and exact additivity.
Run: python -m unittest tests.test_worked_examples -v"""
import unittest

import numpy as np
import pandas as pd

from src.evaluation import decomposition as DC
from src.evaluation import routing as R
from src.evaluation import worked_examples as W


class WorkedExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routed = W.routed_frame()

    def test_stored_thresholds_reproduce_the_published_primary_policy(self):
        m = R.policy_metrics(self.routed)
        pub = pd.read_csv(W.TAB / "routing_heldout_policy.csv").set_index("scenario")
        p = pub.loc[[i for i in pub.index if i.startswith("P:")][0]]
        self.assertAlmostEqual(100 * m.auto_share, p["AUTO %"], places=2)
        self.assertAlmostEqual(m.auto_mape, p["AUTO MAPE"], places=2)

    def test_selection_does_not_read_prices_or_errors(self):
        base = W.select_examples(self.routed)
        rng = np.random.default_rng(0)
        shuffled = self.routed.copy()
        for c in ("y", "point", "lo", "hi", "ape", "cover"):
            if c in shuffled:
                shuffled[c] = rng.permutation(shuffled[c].values)           # destroys any label / error information
        self.assertEqual(W.select_examples(shuffled), base)
        stripped = self.routed[["repeat", "Id", "rel", "tau", "route"]]
        self.assertEqual(W.select_examples(stripped), base)                  # works with no price column at all

    def test_routes_follow_relative_width_only(self):
        d = self.routed
        self.assertTrue(((d.rel <= d.tau) == (d.route == "AUTO")).all())

    def test_examples_cover_both_routes_and_use_stored_values(self):
        ex, dec, lb = W.build()
        self.assertEqual(len(ex), 4)
        self.assertTrue({"AUTO", "ESCALATE"} <= set(ex.route))
        self.assertTrue(np.allclose(ex.relative_width, (ex.interval_high - ex.interval_low) / ex.predicted_price))
        for _, r in ex.iterrows():
            h = self.routed[(self.routed.repeat == 0) & (self.routed.Id == r.Id)].iloc[0]
            self.assertAlmostEqual(r.predicted_price, h.point)
            self.assertEqual(r.route, h.route)


class DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = DC.load_bundle()
        cls.raw = DC.raw_predictors()
        cls.rows = cls.raw.iloc[:40]

    def test_every_output_column_has_a_business_group(self):
        cols = self.bundle["prep"].transform(self.rows.iloc[:5]).columns
        groups = {DC.column_group(c) for c in cols}
        self.assertLessEqual(groups, set(DC.PERM_GROUPS))

    def test_contributions_sum_exactly_on_both_scales(self):
        dec = DC.decompose(self.rows, self.bundle)
        s = dec.groupby("row_index")[["log_contribution", "usd_contribution"]].sum()
        first = dec.groupby("row_index")[["prediction_usd", "reference_prediction_usd"]].first()
        z = self.bundle["prep"].transform(self.rows)
        f = self.bundle["model"].predict(z)
        zbar = self.bundle["prep"].transform(self.raw).mean(axis=0)
        f_ref = self.bundle["model"].intercept_ + float(zbar.values @ self.bundle["model"].coef_)
        self.assertTrue(np.allclose(s.log_contribution.values, f - f_ref, atol=1e-9))
        self.assertTrue(np.allclose(first.prediction_usd.values, np.expm1(f)))
        self.assertTrue(np.allclose(s.usd_contribution.values, first.prediction_usd.values - first.reference_prediction_usd.values, atol=1e-6))

    def test_factors_multiply_to_the_prediction_ratio(self):
        dec = DC.decompose(self.rows.iloc[:10], self.bundle)
        prod = dec.groupby("row_index").multiplicative_factor.prod()
        first = dec.groupby("row_index")[["prediction_usd", "reference_prediction_usd"]].first()
        ratio = (first.prediction_usd + 1) / (first.reference_prediction_usd + 1)
        self.assertTrue(np.allclose(prod.values, ratio.values, atol=1e-9))

    def test_decomposition_ignores_any_label_and_other_houses(self):
        one = DC.decompose(self.rows.iloc[[3]], self.bundle).set_index("group").log_contribution
        many = DC.decompose(self.rows, self.bundle)
        again = many[many.row_index == self.rows.index[3]].set_index("group").log_contribution
        self.assertTrue(np.allclose(one.sort_index().values, again.sort_index().values))

    def test_land_vs_building_aggregate_preserves_the_total(self):
        dec = DC.decompose(self.rows.iloc[:6], self.bundle)
        lb = DC.land_vs_building(dec)
        self.assertTrue(np.allclose(lb.groupby("row_index").usd_contribution.sum().values, dec.groupby("row_index").usd_contribution.sum().values))
        self.assertEqual(set(lb.part), {"Location-and-lot attributes", "Building and use attributes"})

    def test_no_shap_in_the_new_modules(self):
        import inspect
        for mod in (DC, W):
            src = inspect.getsource(mod)
            self.assertNotRegex(src, r"^\s*(import|from)\s+shap\b", mod.__name__)


if __name__ == "__main__":
    unittest.main()
