"""Search-space, determinism, failure-handling and nested-CV leakage checks.  Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import optuna

from src.evaluation import nested_tuning as N
from src.evaluation.cv import get_folds
from src.evaluation.run_feature_selection import load
from src.models import tuning as T


class SpaceTests(unittest.TestCase):
    def test_every_sampled_combination_is_valid_and_buildable(self):
        for fam in T.FAMILIES:
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
            seen = []
            def obj(trial):
                p = T.suggest(trial, fam)
                seen.append(p)
                T.make_estimator(fam, p)
                return 0.0
            study.optimize(obj, n_trials=60)
            self.assertTrue(all(T.validate_params(fam, p) for p in seen), fam)

    def test_log_spaces_for_regularisation_and_learning_rate(self):
        self.assertEqual(T.SPACES["elasticnet"]["alpha"][2], "log")
        self.assertEqual(T.SPACES["gbr"]["learning_rate"][2], "log")
        self.assertEqual(T.MAX_ITER, 100_000)

    def test_boundary_detection(self):
        self.assertTrue(T.at_boundary("elasticnet", "alpha", 1.01e-5))
        self.assertFalse(T.at_boundary("elasticnet", "alpha", 1e-3))
        self.assertTrue(T.at_boundary("gbr", "max_depth", 5))
        self.assertFalse(T.at_boundary("gbr", "max_depth", 3))

    def test_study_minimises(self):
        self.assertEqual(optuna.create_study(direction="minimize").direction.name, "MINIMIZE")


class FailureHandlingTests(unittest.TestCase):
    def test_failed_trials_are_recorded_not_dropped(self):
        bad = [(np.full((10, 3), np.nan), np.zeros(10), np.zeros((5, 3)), np.zeros(5))]
        study = N._study("elasticnet", bad, n_trials=3, seed=0)
        self.assertEqual([t.state.name for t in study.trials], ["FAIL"] * 3)
        self.assertTrue(all("error" in t.user_attrs for t in study.trials))
        fr = N.trials_frame(study, family="elasticnet")
        self.assertEqual(len(fr), 3)


class NestedLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.X, cls.y, cls.folds = load()

    def test_inner_folds_never_touch_outer_validation_rows(self):
        tr, va = self.folds[0]
        for itr, iva in get_folds(self.y[tr], n_splits=N.INNER_SPLITS, n_repeats=1, seed=T.SEED):
            self.assertFalse(set(tr[itr]) & set(va))
            self.assertFalse(set(tr[iva]) & set(va))
            self.assertFalse(set(itr) & set(iva))

    def test_outer_validation_targets_cannot_influence_tuning(self):
        tr, va = self.folds[0]
        y2 = self.y.copy()
        y2[va] = y2[va] * 7.0 + 123456.0                      # corrupt ONLY the outer validation targets
        a = N.tune_outer_fold("elasticnet", 0, tr, va, 6, self.X, self.y)
        b = N.tune_outer_fold("elasticnet", 0, tr, va, 6, self.X, y2)
        self.assertEqual({k: v for k, v in a[0].items() if k.startswith("best_")},
                         {k: v for k, v in b[0].items() if k.startswith("best_")})
        self.assertEqual(a[0]["inner_best"], b[0]["inner_best"])
        self.assertEqual(list(a[1]["value"]), list(b[1]["value"]))      # every trial score identical
        self.assertNotEqual(a[0]["rmse_log"], b[0]["rmse_log"])          # but the outer score does see the targets

    def test_tuning_is_deterministic_and_outputs_are_sane(self):
        tr, va = self.folds[1]
        a = N.tune_outer_fold("elasticnet", 1, tr, va, 6, self.X, self.y)
        b = N.tune_outer_fold("elasticnet", 1, tr, va, 6, self.X, self.y)
        self.assertEqual(a[0]["best_alpha"], b[0]["best_alpha"])
        self.assertEqual(a[0]["rmse_log"], b[0]["rmse_log"])
        self.assertTrue(a[0]["all_finite"])
        self.assertEqual(a[0]["n_complete"] + a[0]["n_failed"], 6)
        self.assertTrue(T.validate_params("elasticnet", {k[5:]: v for k, v in a[0].items() if k.startswith("best_")}))


if __name__ == "__main__":
    unittest.main()
