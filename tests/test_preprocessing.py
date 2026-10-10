"""Run from the project root:  python3 -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.data.loading import TEST_PATH, load_raw
from src.features import preprocessing as P
from src.features.preprocessing import STRATEGIES, build_preprocessor


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw()
        cls.X = cls.df.drop(columns=["SalePrice", "Id"])
        cls.test = load_raw(TEST_PATH)

    def test_all_strategies_complete_and_finite(self):
        for s in STRATEGIES:
            Z = build_preprocessor(s).fit_transform(self.X)
            self.assertEqual(len(Z), len(self.X))
            self.assertEqual(int(Z.isna().sum().sum()), 0)
            self.assertFalse(np.isinf(Z.values).any())

    def test_no_target_or_sale_context_in_output(self):
        Z = build_preprocessor("neighborhood_indicator").fit_transform(self.X)
        banned = ["SalePrice", "Id"] + P.SALE_CONTEXT
        self.assertFalse([c for c in Z.columns if any(c == b or c.startswith(b + "_") for b in banned)])

    def test_unseen_records_transform_to_same_columns(self):
        for s in STRATEGIES:
            p = build_preprocessor(s).fit(self.X)
            self.assertEqual(list(p.transform(self.test).columns), list(p.transform(self.X).columns))
            self.assertEqual(int(p.transform(self.test).isna().sum().sum()), 0)

    def test_structural_absence_not_filled_with_mode(self):
        r = P.StructuralAbsenceImputer().fit(self.X).transform(self.X)
        no_garage = self.X.GarageArea == 0
        self.assertTrue((r.loc[no_garage, "GarageType"] == P.NO_FEATURE).all())
        self.assertTrue((r.loc[self.X.PoolArea == 0, "PoolQC"] == P.NO_FEATURE).all())

    def test_facility_exists_but_category_missing_is_not_nofeature(self):
        r = P.StructuralAbsenceImputer().fit(self.X).transform(self.X)
        self.assertNotEqual(r.loc[self.df.Id == 949, "BsmtExposure"].iloc[0], P.NO_FEATURE)

    def test_neighborhood_median_uses_training_rows_and_handles_unseen(self):
        tr = self.X[self.X.Neighborhood != "Blueste"]
        imp = P.NeighborhoodMedianImputer().fit(tr)
        probe = self.X[self.X.Neighborhood == "Blueste"].copy()
        probe["LotFrontage"] = np.nan
        self.assertTrue((imp.transform(probe).LotFrontage == imp.global_median_).all())

    def test_indicator_matches_original_missingness(self):
        Z = build_preprocessor("neighborhood_indicator").fit_transform(self.X)
        self.assertEqual(int(Z.LotFrontage_Missing.sum()), int(self.X.LotFrontage.isna().sum()))

    def test_input_not_modified(self):
        before = self.X.copy(deep=True)
        build_preprocessor("knn").fit_transform(self.X)
        pd.testing.assert_frame_equal(before, self.X)


if __name__ == "__main__":
    unittest.main()


class RedundancyTests(unittest.TestCase):
    def test_no_exact_linear_dependency_among_numeric_inputs(self):
        df = load_raw()
        X = df.drop(columns=["SalePrice", "Id"])
        p = build_preprocessor("baseline").fit(X)
        pre = Pipeline(p.steps[:-1]).transform(X)
        M = pre[[c for c in pre.columns if c in P.NUMERIC]].astype(float)
        M = M.fillna(M.median()).values
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[1])
        self.assertFalse([c for c in P.REDUNDANT if c in pre.columns])

    def test_redundancy_removal_can_be_switched_off(self):
        X = load_raw().drop(columns=["SalePrice", "Id"])
        Z = build_preprocessor("baseline", reduce_redundancy=False).fit_transform(X)
        self.assertTrue({"TotalSF", "GarageArea", "1stFlrSF"} <= set(Z.columns))
