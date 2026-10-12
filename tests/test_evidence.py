"""Evidence registry structure and cross-source consistency.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np

from src.evaluation import evidence as EV


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.reg = EV.build_registry()
        except FileNotFoundError as e:
            cls.reg = None
            cls.err = e

    def setUp(self):
        if self.reg is None:
            self.skipTest(f"run the earlier stages first: {self.err}")

    def test_unique_ids_and_required_columns(self):
        self.assertTrue(self.reg.claim_id.is_unique)
        for c in ("claim_id", "claim", "source_notebook", "experiment_id", "model_version", "feature_set", "evaluation_protocol", "metric", "value", "units", "sample_size", "limitations", "source_file", "status"):
            self.assertIn(c, self.reg.columns)

    def test_unsupported_claims_are_registered_without_values(self):
        u = self.reg[self.reg.status == "unsupported"]
        self.assertGreaterEqual(len(u), 6)
        self.assertTrue(u.value.isna().all())
        self.assertTrue({"UNS-COST", "UNS-PROD", "UNS-CAUSAL", "UNS-SUBGROUP"} <= set(u.claim_id))

    def test_unknown_claim_cannot_be_cited(self):
        with self.assertRaises(KeyError):
            EV.value(self.reg, "NOT-A-CLAIM")

    def test_values_come_from_artifacts_not_literals(self):
        self.assertAlmostEqual(EV.value(self.reg, "ACC-GBR-LOGRMSE"), EV.csv(EV.ART / "tuning" / "nested_outer_folds.csv").query("family == 'gbr'").rmse_log.mean())

    def test_all_consistency_checks_pass(self):
        res = EV.consistency_checks(self.reg)
        failed = [n for n, ok, d in res if not ok]
        self.assertFalse(failed, failed)

    def test_evidence_classes_are_labelled_and_not_merged(self):
        protos = " ".join(self.reg.evaluation_protocol.unique())
        for word in ("NESTED-CV", "TEMPORAL", "ROUTING-NESTED", "RETROSPECTIVE"):
            self.assertIn(word, protos)

    def test_no_untouched_test_claim_is_supported(self):
        self.assertEqual(self.reg[self.reg.claim_id == "UNS-INDEP"].status.iloc[0], "unsupported")


if __name__ == "__main__":
    unittest.main()
