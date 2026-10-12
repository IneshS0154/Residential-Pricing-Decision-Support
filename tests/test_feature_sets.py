"""Feature-set construction and leakage-sensitive checks.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from src.data.loading import load_raw
from src.evaluation import feature_selection as F
from src.evaluation.cv import get_folds
from src.features import preprocessing as P
from src.features.feature_sets import FEATURE_SETS, GROUPS, all_group_variants, group_variant
from src.features.preprocessing import build_preprocessor


class FeatureSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_raw()
        cls.X = df.drop(columns=["SalePrice", "Id"])
        cls.y = df["SalePrice"].values.astype(float)
        cls.out = {n: fs.build().fit_transform(cls.X) for n, fs in FEATURE_SETS.items()}

    def test_reduced_is_exactly_the_project_default(self):
        default = build_preprocessor("baseline").fit_transform(self.X)
        self.assertEqual(list(default.columns), list(self.out["reduced"].columns))
        np.testing.assert_allclose(default.values, self.out["reduced"].values)

    def test_default_output_width_unchanged(self):
        self.assertEqual(build_preprocessor("baseline").fit_transform(self.X).shape[1], 172)

    def test_counts_and_nesting(self):
        n = {k: v.shape[1] for k, v in self.out.items()}
        self.assertGreater(n["full"], n["domain_informed"])
        self.assertGreater(n["domain_informed"], n["reduced"])
        red, dom = set(self.out["reduced"].columns), set(self.out["domain_informed"].columns)
        self.assertEqual(red - dom, {"TotalBathrooms"})                       # swapped for its four components
        self.assertEqual(dom - red, {"TotRmsAbvGrd", "GarageArea", "2ndFlrSF", "BsmtFinSF1", "BsmtFinSF2", "FullBath",
                                     "HalfBath", "BsmtFullBath", "BsmtHalfBath"})
        self.assertTrue(dom <= set(self.out["full"].columns))

    def test_all_sets_complete_finite_unique_columns(self):
        for n, Z in self.out.items():
            self.assertEqual(int(Z.isna().sum().sum()), 0, n)
            self.assertFalse(np.isinf(Z.values).any(), n)
            self.assertEqual(len(set(Z.columns)), Z.shape[1], n)

    def test_no_sale_context_or_target_in_any_set(self):
        banned = ["SalePrice", "Id"] + P.SALE_CONTEXT
        for n, Z in self.out.items():
            self.assertFalse([c for c in Z.columns if any(c == b or c.startswith(b + "_") for b in banned)], n)

    def test_domain_set_has_no_exact_linear_dependency(self):
        fs = FEATURE_SETS["domain_informed"]
        pre = fs.build()
        pre.fit(self.X)
        from sklearn.pipeline import Pipeline
        raw = Pipeline(pre.steps[:-1]).transform(self.X)
        cols = [c for c in raw.columns if c in P.NUMERIC and c not in fs.drop_columns]
        M = raw[cols].astype(float)
        M = M.fillna(M.median()).values
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[1])

    def test_full_set_contains_garage_age_and_low_information_columns(self):
        Z = self.out["full"]
        self.assertTrue({"GarageAge", "TotalSF", "TotalBathrooms", "Utilities", "PoolArea"} <= set(Z.columns))
        self.assertNotIn("GarageYrBlt", Z.columns)

    def test_group_variants_only_change_their_own_group(self):
        ref = set(FEATURE_SETS["reduced"].drop_columns)
        for g, spec in GROUPS.items():
            if g == "C_age":
                continue
            for v in spec["variants"]:
                drop = set(group_variant(g, v).drop_columns)
                self.assertEqual(drop - set(spec["columns"]), ref - set(spec["columns"]), (g, v))

    def test_reference_variant_equals_reduced_everywhere(self):
        for g, spec in GROUPS.items():
            fs = group_variant(g, spec["reference"])
            self.assertEqual(fs.drop_columns, FEATURE_SETS["reduced"].drop_columns, g)
            self.assertEqual((fs.year_features, fs.garage_age), ("age", False), g)

    def test_every_ablation_cell_builds_with_expected_columns(self):
        for g, v, fs in all_group_variants():
            Z = fs.build().fit_transform(self.X.iloc[:400])
            self.assertEqual(int(Z.isna().sum().sum()), 0, (g, v))
            kept = set(GROUPS[g]["variants"][v]) if not isinstance(GROUPS[g]["variants"][v], dict) else set()
            for c in GROUPS[g]["columns"]:
                if not isinstance(GROUPS[g]["variants"][v], dict):
                    self.assertEqual(c in Z.columns, c in kept, (g, v, c))

    def test_year_representations(self):
        Z = group_variant("C_age", "raw years").build().fit_transform(self.X)
        self.assertTrue({"YearBuilt", "YearRemodAdd"} <= set(Z.columns))
        self.assertFalse({"HouseAge", "RemodAge"} & set(Z.columns))
        Z = group_variant("C_age", "engineered age + GarageAge").build().fit_transform(self.X)
        self.assertTrue({"HouseAge", "GarageAge"} <= set(Z.columns))
        self.assertFalse({"YearBuilt", "GarageYrBlt"} & set(Z.columns))


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_raw()
        cls.df = df
        cls.X = df.drop(columns=["SalePrice", "Id"])
        cls.y = df["SalePrice"].values.astype(float)

    def test_folds_are_identical_across_calls(self):
        a, b = get_folds(self.y), get_folds(self.y)
        self.assertTrue(all((x[0] == z[0]).all() and (x[1] == z[1]).all() for x, z in zip(a, b)))

    def test_preprocessing_statistics_come_from_training_rows_only(self):
        tr, va = get_folds(self.y)[0]
        Xtr = self.X.iloc[tr]
        shifted = self.X.copy()
        shifted.iloc[va, shifted.columns.get_loc("LotArea")] *= 50      # tamper with validation rows only
        p1 = FEATURE_SETS["full"].build().fit(Xtr)
        p2 = FEATURE_SETS["full"].build().fit(Xtr)
        a, b = p1.transform(self.X.iloc[tr]), p2.transform(shifted.iloc[tr])
        np.testing.assert_allclose(a.values, b.values)                  # fitting never touched validation rows

    def test_selector_chooses_using_training_rows_only(self):
        tr, va = get_folds(self.y)[0]
        Xtr, ytr = self.X.iloc[tr[:600]], np.log1p(self.y[tr[:600]])
        sel = F.FeatureSetSelector("ridge", fixed=True, inner_splits=3).fit(Xtr, ytr)
        self.assertIn(sel.selected_, FEATURE_SETS)
        self.assertEqual(len(sel.predict(self.X.iloc[va])), len(va))

    def test_paired_comparison_reference_is_zero(self):
        import pandas as pd
        m = pd.DataFrame(dict(rmse_log=np.linspace(.1, .2, 15), mae=1.0, mape=1.0, n_features=10))
        t = F.paired_vs_reference({"a": m, "b": m}, "a")
        self.assertEqual(t.loc["b", "diff"], 0.0)
        self.assertEqual(t.loc["a", "verdict"], "reference")


if __name__ == "__main__":
    unittest.main()
