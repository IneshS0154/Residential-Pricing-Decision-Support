"""Segment analysis of held-out predictions (notebook 08).

Evidence used (all genuinely held-out; never training predictions)
  * point predictions: nested outer-fold predictions of the tuned Gradient Boosting (the working model of 06) and, for
    comparison, the tuned ElasticNet, from the 15 shared folds (src.evaluation.final_evaluation.load_oof)
  * interval results: out-of-fold split-CQR intervals of 07 (calibration data disjoint from the base models and from the
    evaluated houses)

Conventions (same as 06): residual = actual - predicted (positive = underprediction); APE = |residual| / actual.
Each house has 3 out-of-fold predictions (one per CV repeat); headline segment metrics are computed within each repeat and
averaged, and uncertainty intervals resample HOUSES using the repeat-0 prediction of each house (repeats are not independent
houses). Intervals describe house sampling only, not model-fitting variability.

Segment definitions (fixed in advance; reused from 06/07 where they exist)
  price (USD)      low < 130,000 <= mid < 280,000 <= luxury     rounded P25 / P90 of SalePrice (06)
                   'actual' bands use the realised price (retrospective diagnostics only); 'predicted' bands use the point
                   prediction (known at valuation time, deployable)
  quality          OverallQual 1-4 / 5-8 / 9-10                 the tiers of EDA finding F23
  neighbourhood    detailed comparison needs >= MIN_N_NEIGHBORHOOD (30) houses; smaller ones stay in the supplementary table
  age              HouseAge = 2010 - YearBuilt, the pipeline's FIXED reference year (not the sale year): <=10 / 11-30 / 31-60 / >60
  living area      GrLivArea <= 1,200 / 1,201-1,800 / > 1,800 sq ft
  sale year        YrSold (evaluation metadata only; never a predictor)
Inference classes (pre-specified): a segment difference is DESCRIPTIVE if it is merely observed; PRACTICALLY IMPORTANT if its
Log-RMSE differs from the rest by >= 0.02 or its mean signed log error is >= 0.03 in magnitude; STATISTICALLY SUPPORTED if the
Bonferroni-adjusted bootstrap interval of the Log-RMSE difference (segment minus the rest) excludes 0.
"""
import numpy as np
import pandas as pd

from src.evaluation import final_evaluation as E
from src.evaluation import intervals as V
from src.evaluation.run_feature_selection import load

PRICE_EDGES = E.SEGMENT_EDGES                      # (130_000, 280_000)
MIN_N_NEIGHBORHOOD = 30
PRACTICAL_LOGRMSE_GAP = 0.02
PRACTICAL_SIGNED_LOG = 0.03
N_BOOT = 2000
SEED = 42

QUALITY_BINS = ([0, 4, 8, 10], ["low (1-4)", "medium (5-8)", "high (9-10)"])
AGE_BINS = ([-1, 10, 30, 60, 1000], ["<=10 yr", "11-30 yr", "31-60 yr", ">60 yr"])
AREA_BINS = ([0, 1200, 1800, 1e6], ["<=1,200 sq ft", "1,201-1,800", ">1,800 sq ft"])


def add_segments(d, df):
    """Attach segment columns. `df` is the raw frame (position = `row`); sale year is metadata, not a predictor."""
    f = df[["Id", "OverallQual", "Neighborhood", "YearBuilt", "GrLivArea", "YrSold"]].reset_index().rename(columns={"index": "row"})
    d = d.merge(f.drop(columns="Id"), on="row", how="left")
    d["house_age"] = 2010 - d["YearBuilt"]
    d["quality_tier"] = pd.cut(d["OverallQual"], QUALITY_BINS[0], labels=QUALITY_BINS[1])
    d["age_group"] = pd.cut(d["house_age"], AGE_BINS[0], labels=AGE_BINS[1])
    d["area_group"] = pd.cut(d["GrLivArea"], AREA_BINS[0], labels=AREA_BINS[1])
    return d


def build_eval_frame():
    """Long out-of-fold frame (gbr_tuned and enet_tuned, 3 repeats) with segment metadata and the CQR interval columns."""
    df, X, y, folds = load()
    oof = E.load_oof()
    oof = oof[oof.model.isin(["gbr_tuned", "enet_tuned"])].copy()
    oof = add_segments(oof, df)
    iv = pd.read_csv(V.OUT / "cv_intervals.csv.gz")[["fold", "row", "lo_cqr", "hi_cqr", "width_cqr", "cover_cqr", "relwidth_cqr", "cover_raw", "width_raw"]]
    iv[["cover_cqr", "cover_raw"]] = iv[["cover_cqr", "cover_raw"]].astype(float)          # 0/1, so non-GBR rows can hold NaN
    oof = oof.merge(iv, on=["fold", "row"], how="left")      # intervals belong to the tuned-GBR point predictions
    oof.loc[oof.model != "gbr_tuned", ["lo_cqr", "hi_cqr", "width_cqr", "cover_cqr", "relwidth_cqr", "cover_raw", "width_raw"]] = np.nan
    return oof


# ---------------------------------------------------------------- metrics
def seg_stats(g):
    return pd.Series(dict(n=len(g), mean_actual=g["y"].mean(), mean_pred=g["pred"].mean(), log_rmse=np.sqrt(np.mean(g["residual_log"] ** 2)),
                          mae=g["abs_err_usd"].mean(), median_ae=g["abs_err_usd"].median(), rmse_usd=np.sqrt(g["sq_err_usd"].mean()),
                          mape=100 * g["ape"].mean(), signed_usd=g["residual_usd"].mean(), signed_log=g["residual_log"].mean(),
                          sd_log_resid=g["residual_log"].std(), median_ape=100 * g["ape"].median()))


def segment_table(oof, model, by):
    """Segment metrics within each repeat (every house once), averaged over the 3 repeats. `n` = distinct houses per repeat."""
    d = oof[oof.model == model]
    per = d.groupby(["repeat", by], observed=True).apply(seg_stats, include_groups=False).reset_index()
    out = per.groupby(by, observed=True).mean(numeric_only=True).drop(columns="repeat")
    out["log_rmse_range_repeats"] = per.groupby(by, observed=True).log_rmse.apply(lambda s: s.max() - s.min())
    return out


def _stat(arr_y, arr_pred_log, name):
    """Vectorised statistic of (rows) arrays for the bootstrap: arrays are (n_boot, n)."""
    pred = np.expm1(arr_pred_log)
    res = arr_y - pred
    if name == "mae":
        return np.abs(res).mean(axis=1)
    if name == "mape":
        return 100 * (np.abs(res) / arr_y).mean(axis=1)
    if name == "log_rmse":
        return np.sqrt(((np.log1p(arr_y) - arr_pred_log) ** 2).mean(axis=1))
    if name == "signed_usd":
        return res.mean(axis=1)
    if name == "signed_log":
        return (np.log1p(arr_y) - arr_pred_log).mean(axis=1)
    raise ValueError(name)


def bootstrap_ci(g, stats=("log_rmse", "mae", "mape", "signed_usd", "signed_log"), n_boot=N_BOOT, seed=SEED):
    """95% percentile bootstrap over houses of one segment (repeat-0 prediction of each house)."""
    y, p = g["y"].values, g["pred_log"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    return {s: tuple(np.percentile(_stat(y[idx], p[idx], s), [2.5, 97.5])) for s in stats}


def seg_vs_rest(d0, by, level_adj=None, n_boot=N_BOOT, seed=SEED):
    """For each segment: Log-RMSE difference (segment - rest of the houses), stratified house bootstrap, and the inference class.
    d0 must contain ONE prediction per house (repeat 0). The interval is Bonferroni-adjusted over the segments compared."""
    groups = [k for k in d0[by].dropna().unique()]
    m = len(groups)
    lo_q, hi_q = (100 * (0.05 / m) / 2, 100 * (1 - 0.05 / m / 2))
    rng = np.random.default_rng(seed)
    rows = []
    for k in groups:
        a = d0[d0[by] == k]; b = d0[d0[by] != k]
        ia = rng.integers(0, len(a), (n_boot, len(a))); ib = rng.integers(0, len(b), (n_boot, len(b)))
        sa = _stat(a.y.values[ia], a.pred_log.values[ia], "log_rmse"); sb = _stat(b.y.values[ib], b.pred_log.values[ib], "log_rmse")
        diff = sa - sb
        point = np.sqrt(np.mean(a.residual_log ** 2)) - np.sqrt(np.mean(b.residual_log ** 2))
        lo, hi = np.percentile(diff, [lo_q, hi_q])
        slog = a.residual_log.mean()
        practical = abs(point) >= PRACTICAL_LOGRMSE_GAP or abs(slog) >= PRACTICAL_SIGNED_LOG
        supported = (lo > 0) or (hi < 0)
        rows.append({by: k, "n": len(a), "logrmse_diff_vs_rest": point, "adj_ci_lo": lo, "adj_ci_hi": hi, "signed_log": slog,
                     "practically_important": practical, "statistically_supported": supported,
                     "class": ("supported + practical" if supported and practical else "supported only" if supported else "practical only" if practical else "descriptive only")})
    return pd.DataFrame(rows).set_index(by)


def with_ci(table, d0, by):
    """Add repeat-0 house-bootstrap 95% intervals for MAE, MAPE, Log-RMSE and signed errors to a segment table."""
    rows = []
    for k in table.index:
        ci = bootstrap_ci(d0[d0[by] == k])
        rows.append({f"{s}_{b}": ci[s][i] for s in ci for i, b in enumerate(("lo", "hi"))})
    return table.join(pd.DataFrame(rows, index=table.index))


def interval_segment_table(oof, by):
    """CQR coverage, mean width and median relative width by segment (tuned-GBR rows only), averaged over repeats;
    Wilson interval from repeat 0 (house sampling only)."""
    d = oof[(oof.model == "gbr_tuned") & oof.cover_cqr.notna()].copy()
    d["cover_cqr"] = d["cover_cqr"].astype(bool)
    per = d.groupby(["repeat", by], observed=True).agg(n=("Id", "size"), coverage=("cover_cqr", "mean"), mean_width_usd=("width_cqr", "mean"),
                                                        median_rel_width_pct=("relwidth_cqr", lambda s: 100 * s.median())).reset_index()
    out = per.groupby(by, observed=True)[["n", "coverage", "mean_width_usd", "median_rel_width_pct"]].mean()
    r0 = d[d.repeat == 0].groupby(by, observed=True)["cover_cqr"].agg(["sum", "count"])
    w = [V.wilson(int(r0.loc[k, "sum"]), int(r0.loc[k, "count"])) for k in out.index]
    out["cov_ci_lo"], out["cov_ci_hi"] = [a for a, _ in w], [b for _, b in w]
    out["gap_pp"] = 100 * (out.coverage - 0.8)
    return out
