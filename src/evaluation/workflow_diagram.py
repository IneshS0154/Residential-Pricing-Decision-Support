"""Draws docs/figures/workflow_diagram.png: the implemented pipeline, from the business problem to routing and interpretation.

    python -m src.evaluation.workflow_diagram

No data is read and nothing is fitted; the boxes name notebooks, modules and the headline design choices documented in docs/decision_log.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "workflow_diagram.png"

C = {"frame": "#dbe9f6", "data": "#e3f1df", "prep": "#fdf0d5", "model": "#e8def8", "unc": "#fde2e2", "biz": "#d9f0f0"}
ROWS = [
    [("Business problem", "Price Prediction (primary)\nProperty Segment Analysis (secondary)\nStakeholder: valuation team", "frame"),
     ("Data", "Dataset/train.csv, 1,460 x 81\nload_raw(), data dictionary\ntest.csv unlabelled: not used", "data"),
     ("01  EDA and data quality", "25 findings (F01-F25)\nstructural vs genuine NA, skew,\nleakage / realism audit", "data"),
     ("02  Preprocessing", "build_preprocessor('baseline')\nfitted inside every fold\n79 raw -> 172 columns", "prep")],
    [("03  Feature sets", "reduced / expanded / manual\nnested selector, group ablation\nretained: reduced", "prep"),
     ("04  Baseline comparison", "Dummy, OLS, Ridge, Lasso,\nElasticNet, RF, GBR\nfixed configs, 15 shared folds", "model"),
     ("05  Hyperparameter tuning", "Optuna, nested CV (separate)\nElasticNet + Gradient Boosting\nouter = shared folds", "model"),
     ("06  Final evaluation", "residuals, segments, outliers\ngrouped permutation importance\nworking model: tuned GBR", "model")],
    [("07  Prediction intervals", "quantile GB + split CQR\n80 % nominal; coverage with width\nraw 65.1 % vs CQR 80.7 %", "unc"),
     ("08  Segments and temporal", "error by predicted price,\nquality, age, area, neighbourhood\ntrain <= 2008 / test 2009-2010", "unc"),
     ("09  AUTO / ESCALATE routing", "AUTO iff relative width <= tau\ntau from development data only\nhypothetical 7 % target", "unc"),
     ("10  Business interpretation", "evidence registry (every number\nsourced); pilot with human review\nnot production-ready", "biz")],
]


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#44546a", lw=1.2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1f2d3d")
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=7.6, color="#1f2d3d", linespacing=1.35)


def arrow(ax, p, q, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#44546a", lw=1.4, connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8.6); ax.axis("off")
    w, h, gx = 3.2, 1.75, 0.65
    xs = [0.35 + i * (w + gx) for i in range(4)]
    ys = [6.05, 3.75, 1.45]
    pos = {}
    for r, row in enumerate(ROWS):
        order = range(4) if r % 2 == 0 else range(3, -1, -1)           # snake layout so the flow reads continuously
        for c, (t, b, col) in zip(order, row):
            box(ax, xs[c], ys[r], w, h, t, b, C[col])
        pos[r] = [(xs[c], ys[r]) for c in order]
    for r in range(3):
        row_pos = pos[r]
        for a, b in zip(row_pos, row_pos[1:]):
            if r % 2 == 0:
                arrow(ax, (a[0] + w + 0.02, a[1] + h / 2), (b[0] - 0.02, b[1] + h / 2))
            else:
                arrow(ax, (a[0] - 0.02, a[1] + h / 2), (b[0] + w + 0.02, b[1] + h / 2))
    for r in range(2):                                                    # row to row
        last = pos[r][-1]
        nxt = pos[r + 1][0]
        arrow(ax, (last[0] + w / 2, last[1] - 0.02), (nxt[0] + w / 2, nxt[1] + h + 0.02))
    ax.text(7.5, 8.35, "Implemented workflow: residential price prediction with intervals and AUTO / ESCALATE routing", ha="center", fontsize=12.5, fontweight="bold", color="#1f2d3d")
    ax.text(0.35, 0.95, "Shared by every modelling stage: same 15 folds (repeated 5-fold x 3, stratified by price band, seed 42)  |  target log1p(SalePrice)  |  sale-context variables are not predictors  |  all fitted steps inside one Pipeline per fold  |  no SHAP",
            fontsize=8.4, color="#1f2d3d")
    ax.text(0.35, 0.55, "Open items (group approval pending, not applied): D2 exclude Ids 524 / 1299 from training;  D3 Neighborhood encoding.   Deviation from the initial submission: no per-property additive decomposition (grouped importance is used instead).",
            fontsize=8.4, color="#9b1c1c")
    ax.text(0.35, 0.15, "Recommendation: research prototype / controlled-pilot candidate, human-in-the-loop; no independent test set exists.", fontsize=8.4, color="#1f2d3d")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
