import unittest

import numpy as np

from src.serving import predict as P
from src.serving.export_model import build_bundle, load_training


class TestServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = build_bundle(exclude_d2=True)
        cls.X, cls.y = load_training(True)

    def test_d2_excluded(self):
        self.assertEqual(len(self.X), 1458)
        self.assertEqual(self.b["meta"]["excluded_ids"], [524, 1299])

    def test_predict_shape_and_order(self):
        out = P.predict(self.X.head(20), bundle=self.b)
        self.assertTrue((out["lower"] <= out["point"]).mean() > 0.9)
        self.assertTrue((out["lower"] < out["upper"]).all())
        self.assertTrue(set(out["route"]) <= {"AUTO", "ESCALATE"})

    def test_decomposition_sums(self):
        dec, _ = P.explain(self.X.head(5), bundle=self.b)
        s = dec.groupby("row_index")["usd_contribution"].sum()
        ref = dec.groupby("row_index")[["prediction_usd", "reference_prediction_usd"]].first()
        self.assertTrue(np.allclose(s, ref["prediction_usd"] - ref["reference_prediction_usd"], atol=1e-4))

    def test_labels_not_needed(self):
        self.assertNotIn("SalePrice", self.b["meta"]["schema"]["columns"])


if __name__ == "__main__":
    unittest.main()
