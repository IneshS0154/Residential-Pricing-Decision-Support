"""AUTO / ESCALATE routing from prediction-interval width (notebook 09).  Pure functions: nothing here fits a model.

Routing rule (the only inputs are prediction-time quantities; no label ever enters `assign_route`):
    relative interval width  w = (U - L) / point_prediction          (USD-scale endpoints, point prediction > 0)
    route = AUTO      if w <= tau
            ESCALATE  otherwise
ESCALATE errors reported anywhere are the errors of the AUTOMATED prediction BEFORE human review, not post-appraisal errors.

Threshold selection (pre-specified; see Scenario):
    candidate grid          GRID (fixed in code, plus the historical reference value 0.217, which is NOT pre-selected)
    development data        one out-of-fold prediction per house, produced by models that never saw the evaluation fold
    rule                    choose the LARGEST grid threshold (largest AUTO volume) that satisfies every constraint:
                              AUTO MAPE bound <= target   (conservative: one-sided 90% bootstrap upper bound over houses;
                                                           non-conservative variant: the point estimate)
                              AUTO count >= min_n         (stability of the estimate)
                              optional: AUTO interval coverage (point estimate) >= min_cov
                            if no threshold qualifies the policy is "escalate everything" (tau = -inf)
    AUTO MAPE is not monotone in tau in finite samples; the rule never assumes it is.
All MAPE targets are HYPOTHETICAL: the proposal (2026-DS-38) states no numeric routing target.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID = tuple(sorted(set(np.round(np.arange(0.16, 0.6001, 0.02), 3)) | {0.217}))
HISTORICAL_REFERENCE_TAU = 0.217               # from the brief; protocol and interval scale unverified; evaluated as an ordinary candidate
N_BOOT = 500
SEED = 42


# ---------------------------------------------------------------- routing signal and rule
def relative_width(lo, hi, point):
    """(U - L) / point prediction. Endpoints must be finite with U >= L and the point prediction strictly positive."""
    lo, hi, point = (np.asarray(a, dtype=float) for a in (lo, hi, point))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(point).all()):
        raise ValueError("non-finite interval endpoint or point prediction")
    if (hi < lo).any():
        raise ValueError("upper bound below lower bound")
    if (point <= 0).any():
        raise ValueError("point prediction must be strictly positive")
    return (hi - lo) / point


def assign_route(rel_width, tau):
    """AUTO if relative width <= tau else ESCALATE. Takes NO label argument by design."""
    return np.where(np.asarray(rel_width, dtype=float) <= tau, "AUTO", "ESCALATE")


# ---------------------------------------------------------------- group metrics (labels used only to EVALUATE a route)
def _m(d, mask):
    g = d[mask]
    n = len(g)
    if n == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, mape=np.nan, coverage=np.nan, mean_width=np.nan, median_rel_width=np.nan, bias=np.nan)
    err = g["y"] - g["point"]
    return dict(n=n, mae=float(err.abs().mean()), rmse=float(np.sqrt((err ** 2).mean())), mape=float(100 * (err.abs() / g["y"]).mean()),
                coverage=float(g["cover"].mean()), mean_width=float((g["hi"] - g["lo"]).mean()), median_rel_width=float(g["rel"].median()),
                bias=float(err.mean()))


def route_summary(d, tau):
    """AUTO / ESCALATE / overall metrics for one threshold. `d` needs: y, point, lo, hi, rel, cover (bool)."""
    auto = d["rel"] <= tau
    a, e = _m(d, auto), _m(d, ~auto)
    out = {"tau": tau, "n_total": len(d), "auto_share": a["n"] / len(d), "escalate_share": e["n"] / len(d)}
    out.update({f"auto_{k}": v for k, v in a.items()}); out.update({f"esc_{k}": v for k, v in e.items()})
    allm = _m(d, np.ones(len(d), bool)); out.update({f"all_{k}": v for k, v in allm.items()})
    return out


def evaluate_grid(d, taus=GRID):
    return pd.DataFrame([route_summary(d, t) for t in taus]).set_index("tau")


def prepare(df, method="cqr"):
    """Standard column set from a frame with y, point (USD), lo_<method>, hi_<method>."""
    d = df.copy()
    d["lo"], d["hi"] = d[f"lo_{method}"], d[f"hi_{method}"]
    d["rel"] = relative_width(d["lo"], d["hi"], d["point"])
    d["cover"] = (d["y"] >= d["lo"]) & (d["y"] <= d["hi"])
    d["ape"] = (d["y"] - d["point"]).abs() / d["y"]
    return d


# ---------------------------------------------------------------- uncertainty on the development data
def bootstrap_auto(d, taus=GRID, n_boot=N_BOOT, seed=SEED):
    """House bootstrap of AUTO MAPE (%), AUTO MAE and AUTO share for every candidate threshold, using the SAME resamples for all
    thresholds (so differences between neighbouring thresholds are paired). Returns {stat: array (n_boot, n_tau)}; NaN when the
    resampled AUTO group is empty."""
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, (n_boot, n))
    rel, ape, ae, cov = d["rel"].values[idx], d["ape"].values[idx], (d["y"] - d["point"]).abs().values[idx], d["cover"].values[idx].astype(float)
    mape, mae, share, coverage = [], [], [], []
    for t in taus:
        m = rel <= t
        k = m.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mape.append(np.where(k > 0, 100 * (ape * m).sum(axis=1) / k, np.nan)); mae.append(np.where(k > 0, (ae * m).sum(axis=1) / k, np.nan))
            coverage.append(np.where(k > 0, (cov * m).sum(axis=1) / k, np.nan))
        share.append(k / n)
    return dict(mape=np.array(mape).T, mae=np.array(mae).T, share=np.array(share).T, coverage=np.array(coverage).T)


@dataclass(frozen=True)
class Scenario:
    name: str
    mape_target: float                 # AUTO MAPE (%) that must not be exceeded (HYPOTHETICAL business target)
    conservative: bool = True          # True: one-sided 90% bootstrap upper bound; False: point estimate
    min_n: int = 100                   # minimum AUTO houses in the development data
    min_cov: float = None              # optional: AUTO coverage (point estimate) at least this
    hypothetical: bool = True


SCENARIOS = {
    "P": Scenario("P: AUTO MAPE <= 7% (conservative)", 7.0),
    "S6": Scenario("S6: AUTO MAPE <= 6% (conservative)", 6.0),
    "S8": Scenario("S8: AUTO MAPE <= 8% (conservative)", 8.0),
    "S7pt": Scenario("S7pt: AUTO MAPE <= 7% (point estimate, non-conservative)", 7.0, conservative=False),
    "S7cov": Scenario("S7cov: P and AUTO coverage >= 80%", 7.0, min_cov=0.80),
}
PRIMARY = "P"                          # fixed in advance: the illustrative 7% target of the brief, made conservative


def select_threshold(dev, scenario, taus=GRID, n_boot=N_BOOT, seed=SEED, boot=None):
    """Largest feasible grid threshold on DEVELOPMENT data only. Returns (tau or -inf, table of per-threshold quantities)."""
    s = evaluate_grid(dev, taus)
    b = boot if boot is not None else bootstrap_auto(dev, taus, n_boot, seed)
    up = np.nanpercentile(b["mape"], 90, axis=0)
    s["auto_mape_upper90"] = up
    ref = up if scenario.conservative else s["auto_mape"].values
    feas = (s["auto_n"].values >= scenario.min_n) & np.isfinite(ref) & (ref <= scenario.mape_target)
    if scenario.min_cov is not None:
        feas &= (s["auto_coverage"].values >= scenario.min_cov)
    s["feasible"] = feas
    tau = float(np.max(np.array(taus)[feas])) if feas.any() else -np.inf
    return tau, s


# ---------------------------------------------------------------- nested policy evaluation
def apply_frozen_thresholds(eval_df, tau_by_fold):
    """Route each evaluation house with the threshold frozen for ITS fold (selected without that fold's labels)."""
    d = eval_df.copy()
    d["tau"] = d["fold"].map(tau_by_fold)
    d["route"] = assign_route(d["rel"].values, d["tau"].values)
    return d


def policy_metrics(d):
    """AUTO / ESCALATE / overall metrics for routed houses (column `route`), computed within each CV repeat and averaged."""
    rows = []
    for r, g in d.groupby("repeat"):
        a, e, al = _m(g, g.route == "AUTO"), _m(g, g.route == "ESCALATE"), _m(g, np.ones(len(g), bool))
        row = {"repeat": r, "n_total": len(g), "auto_share": a["n"] / len(g), "escalate_share": e["n"] / len(g)}
        row.update({f"auto_{k}": v for k, v in a.items()}); row.update({f"esc_{k}": v for k, v in e.items()}); row.update({f"all_{k}": v for k, v in al.items()})
        rows.append(row)
    return pd.DataFrame(rows).mean(numeric_only=True)


def bootstrap_policy(d0, n_boot=1000, seed=SEED):
    """95% house-bootstrap intervals (one prediction per house, i.e. one CV repeat) of AUTO share, AUTO MAPE, AUTO MAE, AUTO coverage."""
    rng = np.random.default_rng(seed)
    n = len(d0)
    auto = (d0["route"] == "AUTO").values; ape = d0["ape"].values; ae = (d0["y"] - d0["point"]).abs().values; cov = d0["cover"].values.astype(float)
    idx = rng.integers(0, n, (n_boot, n))
    m = auto[idx]; k = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats_ = dict(auto_share=k / n, auto_mape=np.where(k > 0, 100 * (ape[idx] * m).sum(axis=1) / k, np.nan),
                      auto_mae=np.where(k > 0, (ae[idx] * m).sum(axis=1) / k, np.nan), auto_coverage=np.where(k > 0, (cov[idx] * m).sum(axis=1) / k, np.nan))
    return {k_: tuple(np.nanpercentile(v, [2.5, 97.5])) for k_, v in stats_.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def segment_routing(d, by, min_n=0):
    """Per segment: n, AUTO share, AUTO MAE / MAPE / coverage, ESCALATE share. Metrics within each CV repeat, averaged; `n` = houses
    per repeat; Wilson intervals (repeat 0) for the AUTO share. Undefined (NaN) when a group has no AUTO houses."""
    rows = []
    for r, g in d.groupby("repeat"):
        for key, gg in g.groupby(by, observed=True):
            a = _m(gg, gg.route == "AUTO")
            rows.append({"repeat": r, by: key, "n": len(gg), "auto_share": a["n"] / len(gg), "auto_n": a["n"], "auto_mae": a["mae"], "auto_mape": a["mape"], "auto_coverage": a["coverage"]})
    t = pd.DataFrame(rows).groupby(by, observed=True).mean(numeric_only=True).drop(columns="repeat")
    d0 = d[d.repeat == 0]
    w = {k: wilson(int((g.route == "AUTO").sum()), len(g)) for k, g in d0.groupby(by, observed=True)}
    t["auto_share_lo"] = [w[k][0] for k in t.index]; t["auto_share_hi"] = [w[k][1] for k in t.index]
    t["escalate_share"] = 1 - t["auto_share"]
    return t[t.n >= min_n]
