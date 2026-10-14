"""Quality assurance of the final report and the submission documents: references resolve, figures match their sources, claims are consistent,
historical numbers are not presented as current, the decomposition and land-versus-structure statements are not overstated, and the
worked demonstration is genuinely out-of-fold and label-free.
Run: python -m unittest tests.test_report_quality -v"""
import re
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = (DOCS / "final_report.md").read_text(encoding="utf-8")
SUBMISSION_DOCS = sorted((DOCS / "submission").glob("*.md")) if (DOCS / "submission").exists() else []
NEGATION = re.compile(r"(?i)\b(not|never|no|none|without|cannot|unsupported|historical|not reproduced|not in the repository|not used|undelivered)\b")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


class LinksAndFiles(unittest.TestCase):
    def test_every_relative_link_and_image_in_report_and_submission_docs_exists(self):
        for f in [DOCS / "final_report.md", ROOT / "README.md"] + SUBMISSION_DOCS:
            for m in re.finditer(r"\]\(([^)#]+?)\)", f.read_text(encoding="utf-8")):
                t = m.group(1)
                if t.startswith(("http", "mailto")):
                    continue
                self.assertTrue((f.parent / t).resolve().exists(), f"{f.name}: broken link {t}")

    def test_files_named_in_the_report_exist(self):
        for name in sorted(set(re.findall(r"`reports/tables/([\w\-.]+\.csv)`", REPORT))):
            self.assertTrue((ROOT / "reports" / "tables" / name).exists(), name)
        for name in sorted(set(re.findall(r"`docs/([\w\-./]+\.md)`", REPORT))):
            self.assertTrue((DOCS / name).exists(), name)
        for name in sorted(set(re.findall(r"`(src/[\w/]+\.py)`", REPORT))):
            self.assertTrue((ROOT / name).exists(), name)

    def test_report_figures_are_identical_to_their_source_figures(self):
        pairs = {"fig02_target_distribution.png": "01_target_distribution.png", "fig03_missingness.png": "02_missingness.png",
                 "fig04_model_comparison.png": "final_fig1_model_comparison.png", "fig05_actual_vs_predicted.png": "final_fig1_actual_vs_predicted.png",
                 "fig06_group_importance.png": "final_fig2_feature_groups.png", "fig07_coefficients.png": "final_fig9a_coefficients.png",
                 "fig08_interval_segments.png": "final_fig4_interval_coverage_width.png", "fig09_segment_errors.png": "final_fig3_segment_errors.png",
                 "fig10_temporal.png": "final_fig6_temporal.png", "fig11_routing_tradeoff.png": "final_fig5_routing_tradeoff.png"}
        src_dir = ROOT / "reports" / "figures"
        for copy, src in pairs.items():
            self.assertTrue((DOCS / "figures" / "report" / copy).exists(), copy)
            self.assertIn(f"figures/report/{copy}", REPORT)
            if (src_dir / src).exists():                                  # reports/figures is not committed; compare when present
                self.assertEqual((DOCS / "figures" / "report" / copy).read_bytes(), (src_dir / src).read_bytes(), copy)


class EvidenceReferences(unittest.TestCase):
    def test_claim_ids_cited_in_the_report_exist_in_the_registry_or_are_prefixes(self):
        reg = set(pd.read_csv(ROOT / "reports" / "tables" / "evidence_registry.csv").claim_id)
        log = (DOCS / "decision_log.md").read_text(encoding="utf-8")
        for token in sorted(set(re.findall(r"\b[A-Z]{2,6}(?:-[A-Z0-9_]+)+\b", REPORT))):
            if token.startswith("DS-38"):
                continue
            if token.startswith("DL-"):
                self.assertIn(f"| {token} |", log, token)
            else:
                self.assertTrue(token in reg or any(r.startswith(token + "-") for r in reg), f"unknown claim id or prefix {token}")

    def test_registry_size_quoted_in_the_report_is_current(self):
        r = pd.read_csv(ROOT / "reports" / "tables" / "evidence_registry.csv")
        c = r.status.value_counts()
        self.assertIn(f"{len(r)} claims: {c['supported']} supported, {c['unsupported']} explicitly unsupported, {c['pending']} pending", REPORT)

    def test_all_document_numbers_still_match_the_stored_results(self):
        from src.evaluation.doc_traceability import build
        t = build()
        self.assertGreaterEqual(len(t), 89)
        self.assertEqual(t[t.Result != "MATCH"].to_dict("records"), [])


class ConsistencyOfCurrentAndHistoricalNumbers(unittest.TestCase):
    def test_current_primary_routing_metrics_are_used_in_the_report(self):
        for s in ("45.0 %", "6.49 %", "78.5 %", "10.66 %", "55.0 %"):
            self.assertIn(s, REPORT)
        h = pd.read_csv(ROOT / "reports" / "tables" / "routing_heldout_policy.csv").iloc[0]
        self.assertAlmostEqual(h["AUTO %"], 45.0, delta=0.1)
        self.assertAlmostEqual(100 * h["AUTO coverage"], 78.5, delta=0.1)

    def test_historical_brief_figures_are_never_presented_as_current_results(self):
        # 79.2 % coverage and 0.217 / 65.1 % / 7.15 % routing come from the project brief, not from any stored result.
        for f in [DOCS / "final_report.md", ROOT / "README.md"] + SUBMISSION_DOCS:
            for s in sentences(f.read_text(encoding="utf-8")):
                if re.search(r"79\.2\s?%|\b7\.15\s?%|0\.217\s?/\s?65\.1", s):          # 0.217 alone is a legitimate point of the threshold grid
                    self.assertRegex(s, NEGATION, f"{f.name}: {s[:120]}")


class OverclaimGuards(unittest.TestCase):
    def test_decomposition_is_never_presented_as_a_gradient_boosting_explanation(self):
        texts = [REPORT, (DOCS / "worked_examples.md").read_text(encoding="utf-8")] + [f.read_text(encoding="utf-8") for f in SUBMISSION_DOCS]
        for t in texts:
            for s in sentences(t):
                if re.search(r"(?i)decompos\w*[^.]{0,100}(Gradient Boosting|GBR)|(Gradient Boosting|GBR)[^.]{0,100}decompos", s):
                    self.assertRegex(s, NEGATION, s[:140])
        self.assertIn("explains the ElasticNet", REPORT)
        self.assertIn("**not** the Gradient Boosting", REPORT)

    def test_land_versus_structure_value_split_is_not_claimed_as_delivered(self):
        for f in [DOCS / "final_report.md", ROOT / "README.md", DOCS / "proposal_vs_implementation.md"] + SUBMISSION_DOCS:
            for line in f.read_text(encoding="utf-8").splitlines():
                for m in re.finditer(r"(?i)land[- ]?(versus|vs\.?)[- ]?structure", line):
                    window = line[m.start(): m.start() + 200]
                    if re.search(r"\b(IMPLEMENTED|COMPLETE)\b", window) and not re.search(r"(?i)not delivered|not supportable|undelivered", window):
                        self.fail(f"{f.name} presents land-versus-structure as delivered: {line[:140]}")
        row = [l for l in REPORT.splitlines() if l.startswith("| **Land-versus-structure summary**")]
        self.assertEqual(len(row), 1)
        self.assertIn("**NOT DELIVERED.**", row[0])

    def test_pending_decisions_are_never_marked_approved(self):
        for f in [DOCS / "pending_group_decisions.md", DOCS / "final_report.md"] + SUBMISSION_DOCS:
            for line in f.read_text(encoding="utf-8").splitlines():
                if re.search(r"\bD[2-7]\b", line) and re.search(r"(?i)\bapproved by\b[^_|]*[A-Za-z0-9]{3}", line.replace("Approved by: ______", "")):
                    self.fail(f"{f.name}: possible filled approval: {line[:140]}")
        self.assertIn("PENDING GROUP CONFIRMATION", REPORT)                   # D7 is still open
        for d in ("D2", "D3", "D5", "D6"):                                    # decided by the group, recorded as such
            self.assertTrue(any("DECIDED BY THE GROUP" in l and d in l for l in REPORT.splitlines()), d)

    def test_no_financial_savings_claim(self):
        for s in sentences(REPORT):
            if re.search(r"(?i)\b(saving|savings|cost reduction|roi|profit)\b", s):
                self.assertRegex(s, NEGATION, s[:140])


class WorkedDemonstrationIsOutOfFoldAndLabelFree(unittest.TestCase):
    def test_each_example_house_was_in_the_validation_fold_of_its_prediction(self):
        from src.data.loading import load_raw
        from src.evaluation.cv import get_folds
        from src.evaluation import worked_examples as W
        df = load_raw()
        folds = get_folds(df["SalePrice"])                                 # 15 outer folds: repeat r = folds[5r:5r+5]
        routed = W.routed_frame()
        ex = pd.read_csv(ROOT / "reports" / "tables" / "worked_examples.csv")
        for Id in ex.Id:
            h = routed[(routed.repeat == 0) & (routed.Id == Id)].iloc[0]
            tr, va = folds[int(h.fold)]
            row = int(h.row)
            self.assertIn(row, set(va), f"Id {Id} not in the validation fold {h.fold}")
            self.assertNotIn(row, set(tr), f"Id {Id} in the training rows of fold {h.fold}")
            self.assertEqual(int(df.iloc[row].Id), Id)

    def test_selection_function_signature_and_columns_contain_no_label(self):
        import inspect
        from src.evaluation import worked_examples as W
        src = inspect.getsource(W.select_examples)
        for forbidden in ('["y"]', "['y']", "ape", "cover", "point", "error"):
            self.assertNotIn(forbidden, src.split('"""')[2] if src.count('"""') >= 2 else src, forbidden)


class SubmissionDocumentsKeepBlanksBlank(unittest.TestCase):
    def test_templates_exist_and_contain_placeholders_not_invented_content(self):
        needed = ["problem_framing_canvas.md", "ai_use_declaration_template.md", "personal_learning_journey_template.md", "video_script.md",
                  "group_approval_sheet.md"]
        for n in needed:
            p = DOCS / "submission" / n
            self.assertTrue(p.exists(), n)
        for n in ("ai_use_declaration_template.md", "personal_learning_journey_template.md", "group_approval_sheet.md"):
            self.assertIn("______", (DOCS / "submission" / n).read_text(encoding="utf-8"), n)


if __name__ == "__main__":
    unittest.main()
