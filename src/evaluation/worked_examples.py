"""Worked demonstration on four real houses from the STORED, validated out-of-fold outputs (no model is fitted or retrained).

    python -m src.evaluation.worked_examples        # writes reports/tables/worked_examples.csv and worked_examples_decomposition.csv

Prediction, interval and route come from notebooks 05 / 07 / 09: tuned Gradient Boosting out-of-fold point prediction, split-CQR 80 % interval,
and the primary routing policy P with the threshold frozen for the house's outer fold (reports/tables/routing_thresholds_by_fold.csv, column P).
The route uses ONLY the relative width and tau. The actual price is shown as retrospective evidence and never enters selection or routing.

Selection rule (fixed here, uses no label): within CV repeat 0 let r = relative width / tau of the house's fold (AUTO iff r <= 1). Choose
  1. AUTO, typical        the AUTO house whose r is closest to the median r of AUTO houses
  2. AUTO, borderline     the AUTO house with the largest r (closest to the threshold from below)
  3. ESCALATE, typical    the ESCALATE house whose r is closest to the median r of ESCALATE houses
  4. ESCALATE, wide       the ESCALATE house whose r is closest to the 95th percentile of r among ESCALATE houses
Ties are broken by the smaller Id. The explanation columns come from src.evaluation.decomposition (ElasticNet; see its docstring).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import load_raw
from src.evaluation import decomposition as DC
from src.evaluation import routing as R
from src.evaluation import routing_runs as RR

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "reports" / "tables"
SHOW = ["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "LotArea"]


def frozen_taus():
    t = pd.read_csv(TAB / "routing_thresholds_by_fold.csv").set_index("outer fold")["P"]
    return t.to_dict()


def routed_frame():
    """All stored out-of-fold evaluation rows with CQR interval, relative width, frozen tau and route (primary policy P)."""
    ev = R.prepare(pd.read_csv(RR.OUT / "outer_eval.csv.gz"), "cqr")
    return R.apply_frozen_thresholds(ev, frozen_taus())


def select_examples(routed):
    """Label-free selection (see module docstring). `routed` needs: repeat, Id, rel, tau, route; no price or error column is read."""
    d = routed.loc[routed["repeat"] == 0, ["Id", "rel", "tau", "route"]].copy()
    d["r"] = d["rel"] / d["tau"]

    def nearest(g, value):
        i = (g["r"] - value).abs().sort_values(kind="stable").index
        cand = g.loc[i].assign(_d=(g.loc[i, "r"] - value).abs()).sort_values(["_d", "Id"])
        return int(cand["Id"].iloc[0])

    auto, esc = d[d.route == "AUTO"], d[d.route == "ESCALATE"]
    picks = [("AUTO, typical", nearest(auto, auto["r"].median())),
             ("AUTO, borderline", int(auto.sort_values(["r", "Id"], ascending=[False, True])["Id"].iloc[0])),
             ("ESCALATE, typical", nearest(esc, esc["r"].median())),
             ("ESCALATE, wide", nearest(esc, esc["r"].quantile(0.95)))]
    return picks


def build():
    df = load_raw()
    routed = routed_frame()
    picks = select_examples(routed)
    rows = []
    for label, Id in picks:
        h = routed[(routed.repeat == 0) & (routed.Id == Id)].iloc[0]
        other = routed[(routed.Id == Id) & (routed.repeat > 0)].sort_values("repeat")
        attrs = df[df.Id == Id].iloc[0]
        rows.append({"example": label, "Id": Id, **{k: attrs[k] for k in SHOW},
                     "predicted_price": h.point, "interval_low": h.lo, "interval_high": h.hi, "interval_width": h.hi - h.lo,
                     "relative_width": h.rel, "tau_fold": h.tau, "width_over_tau": h.rel / h.tau, "route": h.route,
                     "route_repeat1": other.route.iloc[0], "route_repeat2": other.route.iloc[1],
                     "actual_price_retrospective": h.y, "error_retrospective": h.y - h.point, "ape_retrospective": abs(h.y - h.point) / h.y,
                     "actual_inside_interval_retrospective": bool(h.lo <= h.y <= h.hi)})
    ex = pd.DataFrame(rows)
    raw = DC.raw_predictors()
    sel = raw.loc[(df.Id.isin(ex.Id)).values].copy()
    sel.index = df.loc[sel.index, "Id"].values
    dec = DC.decompose(sel)
    dec = dec.rename(columns={"row_index": "Id"})
    en = dec.groupby("Id")[["prediction_usd", "reference_prediction_usd"]].first().reset_index()
    ex = ex.merge(en.rename(columns={"prediction_usd": "elasticnet_final_fit_prediction_in_sample", "reference_prediction_usd": "reference_prediction_usd"}), on="Id")
    return ex, dec, DC.land_vs_building(dec.rename(columns={"Id": "row_index"})).rename(columns={"row_index": "Id"})


def model_agreement():
    """How far the explanation model (ElasticNet) is from the working model (Gradient Boosting) on validated out-of-fold predictions (repeat 0)."""
    o = pd.read_csv(ROOT / "artifacts" / "final_evaluation" / "oof_predictions.csv.gz")
    o = o[o.repeat == 0]
    g = o[o.model == "gbr_tuned"].set_index("Id").pred_log
    e = o[o.model == "enet_tuned"].set_index("Id").pred_log.reindex(g.index)
    d = g - e
    return {"n": int(len(g)), "pearson_log_prediction": float(np.corrcoef(g, e)[0, 1]), "rmse_log_difference": float(np.sqrt((d ** 2).mean())),
            "median_abs_pct_price_difference": float(100 * np.median(np.abs(np.expm1(g) - np.expm1(e)) / np.expm1(g)))}


if __name__ == "__main__":
    ex, dec, lb = build()
    ex.round(6).to_csv(TAB / "worked_examples.csv", index=False)
    dec.round(8).to_csv(TAB / "worked_examples_decomposition.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(ex.T.to_string())
    print(model_agreement())
