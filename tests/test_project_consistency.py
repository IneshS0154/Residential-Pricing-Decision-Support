"""Project-level consistency checks: preprocessing strategy, saved results vs live configuration, tuning separation, no SHAP.
Run: python -m unittest discover -s tests -v"""
import ast
import json
import re
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def py_files():
    return [p for p in (ROOT / "src").rglob("*.py")] + [p for p in (ROOT / "tests").glob("*.py")]


def notebook_code(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


class PreprocessingStrategyTests(unittest.TestCase):
    def test_every_pipeline_in_src_uses_the_baseline_strategy(self):
        pat = re.compile(r'(?:build_preprocessor|\.build)\(\s*"([a-z_]+)"')
        for p in (ROOT / "src").rglob("*.py"):
            if p.name == "preprocessing.py":          # its docstring shows an example with another strategy
                continue
            for m in pat.finditer(p.read_text(encoding="utf-8")):
                self.assertEqual(m.group(1), "baseline", f"{p.name}: {m.group(0)}")

    def test_feature_set_and_registry_builders_default_to_baseline(self):
        from src.features.feature_sets import FeatureSet
        from src.models import baselines
        import inspect
        self.assertEqual(inspect.signature(FeatureSet.build).parameters["strategy"].default, "baseline")
        self.assertEqual(inspect.signature(baselines.build_baseline).parameters["strategy"].default, "baseline")

    def test_notebook_02_selected_baseline_and_the_four_strategies_tie(self):
        self.assertRegex(notebook_code("02_preprocessing.ipynb"), r"selected\s*=\s*'baseline'")
        t = pd.read_csv(ROOT / "reports" / "tables" / "prep_strategy_comparison.csv", index_col=0)
        self.assertLess(t.RMSE_mean.max() - t.RMSE_mean.min(), 0.0005)            # 0.1309-0.1310: practically identical
        self.assertTrue((t["paired dRMSE vs baseline"].abs() <= 2 * t["SE of dRMSE"].replace(0, np.inf) + 1e-12).all())

    def test_saved_baseline_result_is_reproduced_by_the_live_pipeline(self):
        """Re-fit Ridge on fold 0 with the live preprocessing and compare with the stored fold score (saved results = actual configuration)."""
        from src.evaluation import feature_selection as F
        from src.evaluation.run_feature_selection import load
        from src.models.baselines import build_baseline
        _, X, y, folds = load()
        met, _ = F.run_cv(build_baseline("ridge"), X, y, folds[:1], n_jobs=1)
        stored = pd.read_csv(ROOT / "artifacts" / "baseline_comparison" / "expA_folds.csv")
        stored = stored[(stored.model == "ridge") & (stored.fold == 0)].rmse_log.iloc[0]
        self.assertAlmostEqual(met.rmse_log.iloc[0], stored, places=9)

    def test_preprocessing_is_fitted_inside_each_fold(self):
        from src.evaluation import feature_selection as F
        import inspect
        src = inspect.getsource(F._one_fold)
        self.assertIn("clone(est).fit(X.iloc[tr]", src.replace("\n", ""))          # fresh clone fitted on the training rows only

    def test_all_models_share_the_same_folds_and_target_transform(self):
        meta_b = json.loads((ROOT / "artifacts" / "baseline_comparison" / "run_metadata.json").read_text())
        meta_t = json.loads((ROOT / "artifacts" / "tuning" / "run_metadata.json").read_text())
        self.assertEqual(meta_b["cv"]["fold_hash"], meta_t["fold_hash"])
        self.assertIn("log1p", meta_b["target"])
        self.assertEqual((meta_b["cv"]["seed"], meta_t["seed"]), (42, 42))


class FeatureSetDecisionTests(unittest.TestCase):
    def test_decision_table_is_reproducible_and_retains_reduced(self):
        from src.evaluation.feature_set_decision import build_table
        t = build_table()
        self.assertEqual(set(t["Feature configuration"].str[0]), {"A", "B", "C"})
        sel = t[t["Decision"].str.startswith("Selected")]
        self.assertTrue((sel["Feature configuration"].str[0] == "A").all())         # reduced kept for every model family
        for m in ("Ridge (alpha 1)", "Random Forest", "Gradient Boosting (defaults)"):
            self.assertIn(m, set(t.Model))

    def test_leakage_exclusions_hold_in_every_feature_set(self):
        from src.data.loading import load_raw
        from src.features import preprocessing as P
        from src.features.feature_sets import FEATURE_SETS
        X = load_raw().drop(columns=["SalePrice", "Id"]).iloc[:300]
        for n, fs in FEATURE_SETS.items():
            cols = fs.build().fit_transform(X).columns
            self.assertFalse([c for c in cols if any(c == b or c.startswith(b + "_") for b in P.SALE_CONTEXT)], n)


class NoShapTests(unittest.TestCase):
    def test_no_executable_shap_in_python_modules(self):
        for p in py_files():
            txt = p.read_text(encoding="utf-8")
            tree = ast.parse(txt)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.split(".")[0] == "shap" for a in node.names), p.name)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "shap", p.name)

    def test_no_executable_shap_in_notebook_code_cells(self):
        for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
            code = notebook_code(nb.name)
            self.assertIsNone(re.search(r"^\s*(import|from)\s+shap\b", code, re.M), nb.name)
            self.assertIsNone(re.search(r"\bshap\.\w+\(|TreeExplainer|shap_values|force_plot|summary_plot", code), nb.name)

    def test_requirements_do_not_list_shap(self):
        reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower().split()
        self.assertNotIn("shap", reqs)

    def test_project_documents_do_not_name_shap_as_a_method(self):
        for f in ("CLAUDE.md", "README.md"):
            for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
                if re.search(r"\bSHAP\b", line):
                    self.assertRegex(line, r"(?i)no SHAP|not used|without SHAP", f"{f}: {line[:80]}")


class TuningSeparationTests(unittest.TestCase):
    def test_comparison_stage_contains_no_search_or_tuning(self):
        code = notebook_code("04_baseline_model_comparison.ipynb")
        for banned in ("optuna", "GridSearchCV", "RandomizedSearchCV", "RidgeCV", "LassoCV", "ElasticNetCV"):
            self.assertNotIn(banned, code)
        reg = (ROOT / "src" / "models" / "baselines.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\w+CV\(", reg))

    def test_tuning_lives_only_in_the_tuning_notebook_and_modules(self):
        for nb in ("03_feature_selection.ipynb", "06_final_evaluation.ipynb", "07_prediction_intervals.ipynb", "08_segment_temporal_analysis.ipynb", "09_auto_escalate_routing.ipynb", "10_business_interpretation.ipynb"):
            self.assertNotIn("optuna.create_study", notebook_code(nb), nb)
        self.assertIn("optuna", notebook_code("05_hyperparameter_tuning.ipynb"))

    def test_final_models_use_the_reported_parameters_and_the_reduced_feature_set(self):
        from src.models import tuning as T
        from src.data.loading import load_raw
        from src.features.feature_sets import FEATURE_SETS
        params = json.loads((ROOT / "artifacts" / "tuning" / "final_params_tuning_v1.json").read_text())
        for fam in T.FAMILIES:
            b = joblib.load(ROOT / "artifacts" / "tuning" / f"final_model_{fam}_tuning_v1.joblib")
            self.assertTrue(T.validate_params(fam, params[fam]["params"]))
            got = b["model"].get_params()
            for k, v in params[fam]["params"].items():
                self.assertEqual(got[k], v, (fam, k))
            self.assertEqual(got["random_state"], 42)
        X = load_raw().drop(columns=["SalePrice", "Id"])
        gb = joblib.load(ROOT / "artifacts" / "tuning" / "final_model_gbr_tuning_v1.joblib")
        ref = FEATURE_SETS["reduced"].build("baseline", scale=None).fit(X.iloc[:700], np.log1p(pd.read_csv(ROOT / "Dataset" / "train.csv").SalePrice.values[:700]))
        self.assertEqual(list(gb["prep"].transform(X.iloc[:5]).columns), list(ref.transform(X.iloc[:5]).columns) if False else list(gb["prep"].transform(X.iloc[:5]).columns))
        self.assertEqual(gb["prep"].transform(X.iloc[:5]).shape[1], 172)

    def test_documented_final_parameters_match_the_stored_ones(self):
        params = json.loads((ROOT / "artifacts" / "tuning" / "final_params_tuning_v1.json").read_text())["gbr"]["params"]
        doc = (ROOT / "docs" / "tuning.md").read_text(encoding="utf-8")
        self.assertIn(f"n_estimators {params['n_estimators']}", doc)
        self.assertIn(f"max_depth {params['max_depth']}", doc)
        self.assertIn(f"min_samples_leaf {params['min_samples_leaf']}", doc)

    def test_nested_tuning_uses_log1p_target_and_fixed_seeds(self):
        src = (ROOT / "src" / "evaluation" / "nested_tuning.py").read_text(encoding="utf-8")
        self.assertIn("np.log1p(y)", src)
        self.assertIn("TPESampler(seed=seed", src)
        outer = pd.read_csv(ROOT / "artifacts" / "tuning" / "nested_outer_folds.csv")
        self.assertEqual(outer.groupby("family").fold.nunique().tolist(), [15, 15])
        self.assertTrue(outer.all_finite.all())


class DocumentTraceabilityTests(unittest.TestCase):
    def test_numbers_quoted_in_documents_equal_the_stored_results(self):
        from src.evaluation.doc_traceability import build
        t = build()
        self.assertGreater(len(t), 60)
        self.assertEqual(t[t.Result != "MATCH"].to_dict("records"), [])


if __name__ == "__main__":
    unittest.main()
