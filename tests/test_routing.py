"""Routing signal, route assignment, threshold selection and nested-design checks.  Run: python -m unittest discover -s tests -v"""
import inspect
import unittest

import numpy as np
import pandas as pd

from src.evaluation import routing as R


def synth(n=1200, seed=0, slope=0.25):
    """Houses whose APE grows with relative width, so AUTO MAPE rises with the threshold."""
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0.1, 0.7, n)
    point = rng.uniform(100_000, 300_000, n)
    ape = slope * rel * np.abs(rng.standard_normal(n)) + 0.01
    y = point * (1 + rng.choice([-1, 1], n) * ape)
    lo, hi = point * (1 - rel / 2), point * (1 + rel / 2)
    d = pd.DataFrame(dict(y=y, point=point, lo_cqr=lo, hi_cqr=hi, fold=0, repeat=0))
    return R.prepare(d, "cqr")


class SignalTests(unittest.TestCase):
    def test_relative_width_formula(self):
        np.testing.assert_allclose(R.relative_width([90.0, 100.0], [110.0, 140.0], [100.0, 120.0]), [0.2, 40 / 120])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError): R.relative_width([1.0], [2.0], [0.0])          # point must be > 0
        with self.assertRaises(ValueError): R.relative_width([1.0], [2.0], [-5.0])
        with self.assertRaises(ValueError): R.relative_width([3.0], [2.0], [5.0])          # upper below lower
        with self.assertRaises(ValueError): R.relative_width([np.nan], [2.0], [5.0])
        with self.assertRaises(ValueError): R.relative_width([1.0], [np.inf], [5.0])

    def test_route_rule_boundary_and_exclusivity(self):
        rel = np.array([0.10, 0.25, 0.25000001, 0.9])
        r = R.assign_route(rel, 0.25)
        self.assertEqual(list(r), ["AUTO", "AUTO", "ESCALATE", "ESCALATE"])               # tau itself is AUTO (<=)
        self.assertTrue(set(r) <= {"AUTO", "ESCALATE"}); self.assertEqual(len(r), len(rel))

    def test_route_function_cannot_see_labels(self):
        self.assertEqual(list(inspect.signature(R.assign_route).parameters), ["rel_width", "tau"])
        src = inspect.getsource(R.assign_route)
        for banned in ("y", "actual", "error", "cover"):
            self.assertNotIn(f'["{banned}"]', src)

    def test_infinite_threshold_escalates_everything_and_large_threshold_automates_everything(self):
        rel = np.array([0.1, 0.5, 2.0])
        self.assertTrue((R.assign_route(rel, -np.inf) == "ESCALATE").all())
        self.assertTrue((R.assign_route(rel, np.inf) == "AUTO").all())


class MetricTests(unittest.TestCase):
    def test_group_metrics_denominators_and_values(self):
        d = pd.DataFrame(dict(y=[100.0, 200.0, 400.0], point=[110.0, 180.0, 400.0], lo_cqr=[100.0, 150.0, 300.0], hi_cqr=[120.0, 190.0, 500.0]))
        d = R.prepare(d, "cqr")
        s = R.route_summary(d, 0.3)          # rel = 0.2, 0.2222, 0.5  -> first two AUTO
        self.assertEqual((s["auto_n"], s["esc_n"]), (2, 1))
        self.assertAlmostEqual(s["auto_share"] + s["escalate_share"], 1.0)
        self.assertAlmostEqual(s["auto_mae"], 15.0); self.assertAlmostEqual(s["auto_mape"], 100 * (0.10 + 0.10) / 2)
        self.assertAlmostEqual(s["auto_coverage"], 0.5)          # 100 in [100,120] covered; 200 not in [150,190]
        self.assertAlmostEqual(s["esc_mape"], 0.0)

    def test_empty_groups_are_nan_not_errors(self):
        d = synth(200)
        s = R.route_summary(d, 0.0001)                           # nobody AUTO
        self.assertEqual(s["auto_n"], 0); self.assertTrue(np.isnan(s["auto_mape"]) and np.isnan(s["auto_coverage"]))
        s2 = R.route_summary(d, 10.0)                            # nobody ESCALATE
        self.assertEqual(s2["esc_n"], 0); self.assertTrue(np.isnan(s2["esc_mape"]))

    def test_policy_metrics_within_repeat(self):
        d = synth(400); d["repeat"] = np.repeat([0, 1], 200); d["fold"] = d["repeat"] * 5
        d["route"] = R.assign_route(d.rel.values, 0.3)
        m = R.policy_metrics(d)
        self.assertAlmostEqual(m["auto_share"] + m["escalate_share"], 1.0)
        self.assertEqual(m["n_total"], 200)

    def test_segment_routing_handles_group_with_no_auto(self):
        d = synth(300); d["repeat"] = 0; d["route"] = R.assign_route(d.rel.values, 0.3)
        d["grp"] = np.where(d.rel > 0.5, "wide", "other"); d.loc[d.grp == "wide", "route"] = "ESCALATE"
        t = R.segment_routing(d, "grp")
        self.assertEqual(t.loc["wide", "auto_share"], 0.0); self.assertTrue(np.isnan(t.loc["wide", "auto_mape"]))
        self.assertEqual(t.n.sum(), 300)


class SelectionTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_paired_across_thresholds(self):
        d = synth(500)
        a = R.bootstrap_auto(d, n_boot=100, seed=3); b = R.bootstrap_auto(d, n_boot=100, seed=3)
        np.testing.assert_array_equal(np.nan_to_num(a["mape"]), np.nan_to_num(b["mape"]))
        self.assertTrue((np.diff(a["share"], axis=1) >= -1e-12).all())      # same resamples: AUTO share non-decreasing in tau

    def test_selects_largest_feasible_threshold(self):
        d = synth(3000, slope=0.30)
        sc = R.Scenario("t", 8.0, conservative=False, min_n=100)
        tau, tab = R.select_threshold(d, sc, n_boot=100)
        feas = tab.index[tab.feasible]
        self.assertEqual(tau, feas.max())
        self.assertTrue((tab.loc[tab.index > tau, "feasible"] == False).all())

    def test_infeasible_target_gives_escalate_everything(self):
        d = synth(1000)
        tau, tab = R.select_threshold(d, R.Scenario("t", 0.01, conservative=True), n_boot=100)
        self.assertTrue(np.isneginf(tau)); self.assertFalse(tab.feasible.any())
        self.assertTrue((R.assign_route(d.rel.values, tau) == "ESCALATE").all())

    def test_minimum_group_size_blocks_tiny_auto_groups(self):
        d = synth(300)
        tau, tab = R.select_threshold(d, R.Scenario("t", 99.0, conservative=False, min_n=250), n_boot=50)
        self.assertTrue((tab.loc[tab.feasible, "auto_n"] >= 250).all())

    def test_conservative_rule_is_never_more_aggressive_than_point_rule(self):
        d = synth(1500, slope=0.3)
        t_pt, _ = R.select_threshold(d, R.Scenario("a", 7.0, conservative=False), n_boot=200)
        t_cs, _ = R.select_threshold(d, R.Scenario("b", 7.0, conservative=True), n_boot=200)
        self.assertLessEqual(t_cs, t_pt)

    def test_selection_depends_only_on_the_development_frame(self):
        dev, ev = synth(800, seed=1), synth(800, seed=2)
        t1, _ = R.select_threshold(dev, R.SCENARIOS["P"], n_boot=100)
        ev2 = ev.copy(); ev2["y"] = ev2["y"] * 5                                  # corrupt evaluation labels
        t2, _ = R.select_threshold(dev, R.SCENARIOS["P"], n_boot=100)
        self.assertEqual(t1, t2)

    def test_grid_is_fixed_and_contains_the_historical_reference_as_an_ordinary_candidate(self):
        self.assertIn(0.217, R.GRID); self.assertEqual(R.GRID, tuple(sorted(R.GRID)))
        self.assertEqual(R.PRIMARY, "P"); self.assertTrue(all(s.hypothetical for s in R.SCENARIOS.values()))


class NestedDesignTests(unittest.TestCase):
    def setUp(self):
        from src.evaluation import routing_runs as RR
        if not (RR.OUT / "dev_inner_oof.csv.gz").exists():
            self.skipTest("run python -m src.evaluation.routing_runs first")
        self.RR = RR
        self.dev = pd.read_csv(RR.OUT / "dev_inner_oof.csv.gz")

    def test_development_rows_never_include_the_fold_they_are_selected_for(self):
        from src.evaluation.run_feature_selection import load
        _, X, y, folds = load()
        for k, (tr, va) in enumerate(folds):
            rows = set(self.dev[self.dev.outer_fold == k].row)
            self.assertFalse(rows & set(va), k)                   # no evaluation house in its own threshold-selection data
            self.assertEqual(rows, set(tr), k)                    # dev data = exactly the outer-training houses, once each

    def test_each_dev_house_is_predicted_once_per_outer_fold(self):
        self.assertTrue((self.dev.groupby(["outer_fold", "row"]).size() == 1).all())

    def test_dev_intervals_valid(self):
        self.assertTrue(np.isfinite(self.dev[["lo_cqr", "hi_cqr", "point"]].values).all())
        self.assertTrue((self.dev.hi_cqr >= self.dev.lo_cqr).all() and (self.dev.point > 0).all())

    def test_outer_eval_one_route_per_house_per_repeat(self):
        ev = pd.read_csv(self.RR.OUT / "outer_eval.csv.gz"); d = R.prepare(ev, "cqr")
        d["tau"] = 0.3; d["route"] = R.assign_route(d.rel.values, d.tau.values)
        self.assertTrue((d.groupby(["repeat", "row"]).size() == 1).all()); self.assertEqual(d.row.nunique(), 1460)


if __name__ == "__main__":
    unittest.main()
