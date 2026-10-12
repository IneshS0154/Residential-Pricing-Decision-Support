"""Consolidated feature-set decision table built ONLY from existing experiment artifacts (no model is fitted here).

    python -m src.evaluation.feature_set_decision

Feature configurations (all exclude identifiers, the target and SaleType / SaleCondition / MoSold / YrSold):
  A  reduced          the project default: 8 near-constant and 12 redundant columns pruned (REDUNDANT, LOW_INFORMATION)
  B  full  (expanded) every pruned correlated / redundant predictor restored (TotRmsAbvGrd, GarageArea, 1stFlrSF, 2ndFlrSF, BsmtFin*, BsmtUnfSF,
                      bathroom components, TotalSF) plus the near-constant columns and GarageAge; leakage exclusions unchanged
  C  domain_informed  manually specified before any result: keeps economically distinct components, prunes only exact composites
                      (TotalSF, TotalBathrooms) and duplicates (1stFlrSF, BsmtUnfSF); supported by the group ablations and grouped
                      permutation importance of notebooks 03 / 06
Evidence: artifacts/baseline_comparison/expB_folds.csv (notebook 04, Experiment B): fixed model configurations (Ridge alpha 1, Lasso /
ElasticNet alpha 1e-3, Random Forest 300 trees, Gradient Boosting defaults), no tuning, the same 15 price-band-stratified folds for every cell.
Paired fold differences use the Nadeau-Bengio corrected interval (folds are dependent).
Decision rule (stated before the table is read): a non-reduced set replaces `reduced` for a model family only if its paired Log-RMSE
improvement is 'better' by `feature_selection.verdict` (corrected 95% interval excludes 0 and the mean gain exceeds 0.001); otherwise `reduced` is kept.
"""
from pathlib import Path

import pandas as pd

from src.evaluation import feature_selection as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "baseline_comparison" / "expB_folds.csv"
OUT = ROOT / "reports" / "tables" / "feature_set_decision.csv"
LABEL = {"reduced": "A  Reduced (default)", "full": "B  Expanded (all correlated predictors restored)", "domain_informed": "C  Manually selected (domain-informed)"}
MODELS = ("ridge", "rf", "gbr", "ols", "lasso", "elasticnet")
NAME = {"ridge": "Ridge (alpha 1)", "rf": "Random Forest", "gbr": "Gradient Boosting (defaults)", "ols": "OLS", "lasso": "Lasso", "elasticnet": "ElasticNet"}


def build_table(path=SRC):
    d = pd.read_csv(path)
    rows = []
    for m in MODELS:
        mets = {s: d[(d.model == m) & (d.feature_set == s)].sort_values("fold").reset_index(drop=True) for s in LABEL}
        t = F.paired_vs_reference(mets, "reduced")
        replace = [s for s in ("full", "domain_informed") if t.loc[s, "verdict"] == "better"]
        for s in LABEL:
            r = t.loc[s]
            if s == "reduced":
                decision, reason = ("Selected", "default; no alternative beat it by the pre-set rule" if not replace else f"default, but {replace} improved: model-specific set considered")
            else:
                sign = "better" if r["diff"] < 0 else "worse"
                decision = "Selected (model-specific)" if s in replace else "Rejected"
                reason = (f"mean change {r['diff']:+.4f} ({sign}), corrected 95% interval {r['ci_lo']:+.4f} to {r['ci_hi']:+.4f}, better in {int(r['folds_better'])}/15 folds: "
                          + ("meaningful improvement" if s in replace else "no meaningful or consistent improvement; more columns"))
            rows.append({"Feature configuration": LABEL[s], "Model": NAME[m], "n features (mean)": r["n_features"], "Mean Log-RMSE": r["rmse_log"], "SD over folds": r["rmse_sd"], "MAE USD": r["mae_usd"],
                         "Paired diff vs reduced": r["diff"], "Folds better than reduced": int(r["folds_better"]), "Decision": decision, "Reason": reason})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = build_table()
    t.round(5).to_csv(OUT, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 120)
    print(t.drop(columns="Reason").round(4).to_string(index=False))
    print("\nModel-specific replacements:", t[t.Decision == "Selected (model-specific)"][["Model", "Feature configuration"]].values.tolist() or "none: `reduced` retained for every model family")
