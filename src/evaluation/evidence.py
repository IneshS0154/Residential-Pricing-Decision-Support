"""Evidence registry for the final business-interpretation notebook (10).

Every numerical claim made in notebook 10 is a row of `build_registry()`: its value is READ from an executed experiment artifact
(a CSV / JSON written by notebooks 03-09 or their `src/evaluation/*` runners) at build time. Nothing is typed in by hand. Claims for
which no evidence exists are registered explicitly with status 'unsupported' or 'pending' and a NaN value.

Columns: claim_id, claim, source_notebook, experiment_id, model_version, feature_set, evaluation_protocol, metric, value, units,
sample_size, limitations, source_file, status.

Evidence classes used in `evaluation_protocol` (never merged into one number):
  NESTED-CV      outer-fold predictions of models tuned on the outer-training rows only (15 folds = 5-fold x 3 repeats)
  DEV-CV         repeated CV with fixed configurations (baselines, ablations)
  TEMPORAL       train YrSold <= 2008 (947 houses), test 2009-2010 (513 houses), everything learned from training years
  ROUTING-NESTED threshold chosen on inner out-of-fold development predictions, applied to the held-out fold
  RETROSPECTIVE  diagnostics that use realised prices (not deployable)
There is NO untouched independent test set (test.csv has no labels).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import feature_selection as F

ROOT = Path(__file__).resolve().parents[2]
ART, TAB = ROOT / "artifacts", ROOT / "reports" / "tables"
FS = "reduced (172 output columns; FEATURE_SETS['reduced'])"
M_GBR = "GradientBoostingRegressor tuned by nested Optuna (05, fold-specific parameters)"
M_EN = "ElasticNet tuned by nested Optuna (05)"
M_LASSO = "Lasso alpha=1e-3, fixed baseline (04; not tuned)"
M_FINAL_EN = "ElasticNet final fit on all 1,460 houses (05 final study; coefficients only)"
P_NESTED = "NESTED-CV, 15 outer folds (5-fold x 3 repeats); 1,460 houses per repeat"
P_TEMP = "TEMPORAL holdout: train 2006-2008 (947), test 2009-2010 (513); hyperparameters tuned on training years only"
P_RT = "ROUTING-NESTED: threshold selected on inner out-of-fold development data, applied to the held-out fold"
EXP = {"04": "baseline_v1", "05": "tuning_v1", "06": "final_eval_v1", "07": "intervals_v1", "09": "routing_v1", "03": "feature_selection (run_feature_selection)",
       "08": "segments (src/evaluation/temporal.py; run_metadata.json)"}

_cache = {}


def csv(path, **kw):
    path = Path(path)
    if path not in _cache:
        _cache[path] = pd.read_csv(path, **kw)
    return _cache[path]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


class Registry:
    def __init__(self):
        self.rows = []

    def add(self, cid, claim, nb, model, fs, proto, metric, units, n, lim, src, fn=None, status="supported", exp=None):
        val = np.nan
        if status == "supported":
            val = fn() if callable(fn) else fn
        self.rows.append(dict(claim_id=cid, claim=claim, source_notebook=nb, experiment_id=exp or EXP.get(nb.split()[0], nb), model_version=model, feature_set=fs,
                              evaluation_protocol=proto, metric=metric, value=val, units=units, sample_size=n, limitations=lim, source_file=str(src), status=status))

    def frame(self):
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- builders
def _accuracy(R):
    nested = ART / "tuning" / "nested_outer_folds.csv"
    for fam, tag, model in (("gbr", "GBR", M_GBR), ("elasticnet", "ENET", M_EN)):
        d = lambda fam=fam: csv(nested)[csv(nested).family == fam]
        for suffix, col, fn, units, lab in (("LOGRMSE", "rmse_log", "mean", "log points", "Log-RMSE (mean of 15 folds)"), ("LOGRMSE-SD", "rmse_log", "std", "log points", "Log-RMSE (SD of 15 folds)"),
                                            ("MAE", "mae", "mean", "USD", "MAE USD"), ("RMSE", "rmse_usd", "mean", "USD", "RMSE USD"), ("MAPE", "mape", "mean", "%", "MAPE"), ("R2", "r2_usd", "mean", "R2 (USD scale)", "R2 USD")):
            R.add(f"ACC-{tag}-{suffix}", f"{model.split(' ')[0]} {lab}, nested CV", "05", model, FS, P_NESTED, lab, units, "1,460 houses x 3 repeats (15 folds of ~292)",
                  "Same houses informed feature-set and family choice; folds are dependent; mean of fold-level metrics", nested, lambda d=d, col=col, fn=fn: getattr(d()[col], fn)())
    acc = TAB / "final_eval_accuracy.csv"
    row = lambda: csv(acc).set_index("Unnamed: 0").loc["Lasso (fixed)"]
    for suffix, col, units in (("LOGRMSE", "Log-RMSE mean", "log points"), ("LOGRMSE-SD", "Log-RMSE SD", "log points"), ("MAE", "MAE USD", "USD"), ("RMSE", "RMSE USD", "USD"), ("MAPE", "MAPE %", "%"), ("R2", "R2 USD", "R2 (USD scale)")):
        R.add(f"ACC-LASSO-{suffix}", f"Fixed Lasso {col}, same outer folds", "06", M_LASSO, FS, "DEV-CV with fixed configuration on the same 15 folds (not nested tuning)", col, units, "1,460 houses x 3 repeats",
              "Lasso was not tuned (alpha informed by earlier notebooks on the same rows): favourable to Lasso", acc, lambda col=col: row()[col])
    base = ART / "baseline_comparison" / "expA_folds.csv"
    for m, lab in (("gbr", "default GBR"), ("elasticnet", "fixed ElasticNet"), ("ols", "OLS"), ("rf", "Random Forest"), ("dummy", "Dummy (mean)")):
        R.add(f"BASE-{m.upper()}-LOGRMSE", f"{lab} Log-RMSE with fixed configuration", "04", f"{lab} (04 registry)", FS if m != "dummy" else "none (no preprocessing)", "DEV-CV, 15 shared folds, fixed hyperparameters", "Log-RMSE (mean)", "log points",
              "15 folds", "Fixed configurations; not tuned", base, lambda m=m: csv(base)[csv(base).model == m].rmse_log.mean())
    nes = csv(nested)
    g = nes[nes.family == "gbr"].sort_values("fold").reset_index(drop=True); e = nes[nes.family == "elasticnet"].sort_values("fold").reset_index(drop=True)
    pt = lambda: F.paired_vs_reference({"gbr": g, "elasticnet": e}, "gbr").loc["elasticnet"]
    for k, lab, units in (("diff", "paired Log-RMSE difference ElasticNet minus GBR (positive = GBR better)", "log points"), ("ci_lo", "lower end of the Nadeau-Bengio 95% interval of that difference", "log points"), ("ci_hi", "upper end of the Nadeau-Bengio 95% interval", "log points"),
                          ("folds_better", "number of the 15 folds in which ElasticNet beat GBR", "folds")):
        R.add(f"ACC-PAIRED-{k.upper()}", lab, "06", "tuned GBR vs tuned ElasticNet", FS, P_NESTED + "; corrected interval for dependent folds", k, units, "15 folds", "Approximate interval; a tie is not equality", nested, lambda k=k: pt()[k])
    out = TAB / "final_eval_outlier_sensitivity.csv"
    for model, tag in (("Gradient Boosting (tuned)", "GBR"), ("ElasticNet (tuned)", "ENET"), ("Lasso (fixed)", "LASSO")):
        for cond, ctag in (("E  evaluation-only exclusion of 524, 1299", "NOFLAG"),):
            for col, suffix, units in (("rmse_log", "LOGRMSE", "log points"), ("rmse_usd", "RMSE", "USD"), ("mae", "MAE", "USD")):
                R.add(f"OUT-{tag}-{ctag}-{suffix}", f"{tag} {suffix} scored without Ids 524 and 1299 (predictions unchanged)", "06", model, FS, "NESTED-CV / DEV-CV, evaluation-only exclusion of two flagged houses", col, units, "1,458 houses",
                      "Sensitivity analysis; primary evaluation keeps all houses", out, lambda model=model, cond=cond, col=col: csv(out).set_index(["model", "condition"]).loc[(model, cond), col])
    tm = ART / "segments" / "temporal_metrics.csv"; rm = ART / "segments" / "random_matched_splits.csv"
    for fam, tag in (("gbr", "GBR"), ("elasticnet", "ENET")):
        for col, suffix, units in (("log_rmse", "LOGRMSE", "log points"), ("mae", "MAE", "USD"), ("rmse_usd", "RMSE", "USD"), ("mape", "MAPE", "%"), ("r2_usd", "R2", "R2 (USD scale)"), ("signed_usd", "SIGNED", "USD (actual - predicted)")):
            R.add(f"TEMP-{tag}-{suffix}", f"{tag} temporal-holdout {suffix}", "08", f"{fam} tuned on 2006-2008 only (historical-only Optuna)", FS, P_TEMP, col, units, "947 train / 513 test",
                  "One split; test set contains neither atypical house (Ids 524, 1299 were sold in 2007-2008); feature set chosen using all years", tm, lambda fam=fam, col=col: csv(tm)[(csv(tm).family == fam) & (csv(tm).config == "hist_tuned") & (csv(tm).period == "2009-2010")][col].iloc[0])
        for stat, sf in (("mean", "MEAN"), ("std", "SD")):
            R.add(f"RAND947-{tag}-LOGRMSE-{sf}", f"{tag} Log-RMSE over 30 random 947/513 splits ({stat}), fixed configuration", "08", f"{fam} fixed configuration (04)", FS, "DEV-CV: 30 random splits with the same training/test sizes as the temporal split", "Log-RMSE", "log points", "30 splits",
                  "Overlapping test sets; fixed configuration", rm, lambda fam=fam, stat=stat: getattr(csv(rm)[csv(rm).family == fam].log_rmse, stat)())
        R.add(f"RAND947-{tag}-LOGRMSE-NOFLAG-MEAN", f"{tag} Log-RMSE over 30 random splits scored without Ids 524/1299", "08", f"{fam} fixed configuration (04)", FS, "DEV-CV random 947/513 splits, flagged houses removed from the test rows", "Log-RMSE", "log points", "30 splits",
              "Like-for-like comparison with the temporal test set", rm, lambda fam=fam: csv(rm)[csv(rm).family == fam].log_rmse_noflag.mean())
        R.add(f"TEMP-{tag}-FIXED-LOGRMSE", f"{tag} temporal Log-RMSE with the fixed configuration (same configuration as the random splits)", "08", f"{fam} fixed configuration (04)", FS, P_TEMP.replace("hyperparameters tuned on training years only", "no tuning"), "Log-RMSE", "log points",
              "947 train / 513 test", "One split", tm, lambda fam=fam: csv(tm)[(csv(tm).family == fam) & (csv(tm).config == "fixed") & (csv(tm).period == "2009-2010")].log_rmse.iloc[0])
    R.add("TEMP-NTRAIN", "Number of training houses in the temporal experiment", "08", "-", FS, P_TEMP, "n", "houses", "947", "-", tm, lambda: int(csv(tm).n_train.iloc[0]))
    R.add("TEMP-NTEST", "Number of evaluation houses in the temporal experiment", "08", "-", FS, P_TEMP, "n", "houses", "513", "-", tm, lambda: int(csv(tm)[(csv(tm).config == "fixed") & (csv(tm).family == "gbr") & (csv(tm).period == "2009-2010")].n.iloc[0]))
    return R


def _model_and_complexity(R):
    nested = ART / "tuning" / "nested_outer_folds.csv"
    for fam, tag in (("gbr", "GBR"), ("elasticnet", "ENET")):
        R.add(f"CX-{tag}-TUNE-S", f"{tag} Optuna tuning time per outer fold", "05", fam, FS, "6 parallel workers", "tuning wall-clock per fold", "seconds", "15 folds", "Includes contention between parallel folds", nested,
              lambda fam=fam: csv(nested)[csv(nested).family == fam].tune_time.mean())
        R.add(f"CX-{tag}-TRIALS", f"{tag} Optuna trials per outer-fold study", "05", fam, FS, "nested tuning budget", "trials", "trials", "15 studies", "Budgets unequal by design (cost-driven)", nested, lambda fam=fam: int(csv(nested)[csv(nested).family == fam].n_trials.iloc[0]))
    co = TAB / "final_eval_elasticnet_coefficients.csv"
    R.add("CX-ENET-NONZERO", "Non-zero coefficients of the final ElasticNet", "06", M_FINAL_EN, FS, "fit on all 1,460 houses", "non-zero coefficients", "count", "172 output columns", "Regularised; unstable under correlated predictors", co, lambda: int((csv(co).coef_model_units != 0).sum()))
    R.add("CX-ENET-COLUMNS", "Output columns of the preprocessed design matrix", "06", M_FINAL_EN, FS, "fit on all 1,460 houses", "columns", "count", "172", "-", co, lambda: int(len(csv(co))))
    fp = ART / "tuning" / "final_params_tuning_v1.json"
    gp = lambda: json.loads(fp.read_text())["gbr"]["params"]
    R.add("CX-GBR-NTREES", "Trees in the final tuned Gradient Boosting configuration", "05", "final study on all 1,460 houses", FS, "final study (not itself evaluated by nested CV)", "n_estimators", "trees", "-", "Inner-CV optimum on all data: optimistic as a performance claim; n_estimators sits at the search ceiling", fp, lambda: gp()["n_estimators"])
    R.add("CX-GBR-DEPTH", "Max depth of the final tuned Gradient Boosting", "05", "final study", FS, "final study", "max_depth", "levels", "-", "-", fp, lambda: gp()["max_depth"])
    R.add("CX-GBR-LR", "Learning rate of the final tuned Gradient Boosting", "05", "final study", FS, "final study", "learning_rate", "-", "-", "-", fp, lambda: gp()["learning_rate"])
    mx = TAB / "final_eval_segments_matched.csv"
    for model, tag in (("Gradient Boosting (tuned)", "GBR"), ("ElasticNet (tuned)", "ENET"), ("Lasso (fixed)", "LASSO")):
        for seg in ("low", "luxury"):
            for col, suffix, units in (("mae", "MAE", "USD"), ("log_rmse", "LOGRMSE", "log points"), ("bias_usd", "BIAS", "USD (actual - predicted)")):
                R.add(f"SEGM-{tag}-{seg.upper()}-{suffix}", f"{tag} {suffix} in the {seg} segment (banded on tuned-GBR predicted price; matched houses)", "06", model, FS, P_NESTED + "; segments by predicted price", col, units, "low ~371, luxury ~142 houses per repeat",
                      "Luxury segment includes the two atypical houses; small sample", mx, lambda model=model, seg=seg, col=col: csv(mx).set_index(["model", "segment_matched"]).loc[(model, seg), {"mae": "mae", "log_rmse": "rmse_log", "bias_usd": "bias_usd"}[col]])
    return R


def _features(R):
    pi = ART / "final_evaluation" / "grouped_permutation_importance.csv"
    grp = lambda fam, g: csv(pi)[(csv(pi).family == fam) & (csv(pi).group == g)].delta_logrmse
    for g in sorted(csv(pi).group.unique()):
        for fam, tag, model in (("gbr", "GBR", M_GBR), ("elasticnet", "ENET", M_EN)):
            R.add(f"PI-{slug(g)}-{tag}", f"Grouped permutation importance of '{g}', {tag} (mean increase in held-out Log-RMSE when the group is permuted jointly)", "06", model + " refit per outer fold", FS,
                  "held-out outer-fold validation rows, 15 folds x 5 shuffles; raw columns permuted jointly, engineered features recomputed", "increase in Log-RMSE", "log points", "15 folds x ~292 houses",
                  "Predictive reliance of a fixed model, not causal; correlated groups share information", pi, lambda fam=fam, g=g: grp(fam, g).mean())
            R.add(f"PI-{slug(g)}-{tag}-SD", f"SD over the 15 folds of the importance of '{g}', {tag}", "06", model, FS, "same as above", "SD of increase in Log-RMSE", "log points", "15 folds", "Fold SD, not a confidence interval", pi, lambda fam=fam, g=g: grp(fam, g).std())
    def rankcorr():
        from scipy import stats
        m = csv(pi).groupby(["family", "group"]).delta_logrmse.mean().unstack(0)
        return stats.spearmanr(m.gbr, m.elasticnet)[0]
    R.add("PI-RANKCORR", "Spearman rank correlation of the nine group importances between GBR and ElasticNet", "06", "GBR vs ElasticNet", FS, "same held-out folds", "Spearman rho", "-", "9 groups", "Nine points; descriptive", pi, rankcorr)
    co = TAB / "final_eval_elasticnet_coefficients.csv"
    top = lambda: csv(co)[csv(co).coef_model_units != 0].head(10)
    for i in range(10):
        f = lambda i=i: top().iloc[i]
        nm = f().feature
        R.add(f"COEF-{i + 1:02d}-PERSD", f"Standardised coefficient (per 1 SD of the preprocessed column) of '{nm}', final ElasticNet, rank {i + 1} by magnitude", "06", M_FINAL_EN, FS, "fit on all 1,460 houses (descriptive; not an evaluation)",
              "coefficient per SD", "log1p-price points per SD", "1,460 houses", "ElasticNet only; not an explanation of the nonlinear GBR; log1p target (approximately multiplicative); unstable under correlated predictors", co, lambda f=f: f().coef_per_sd)
        R.add(f"COEF-{i + 1:02d}-PCT", f"Approximate percentage association per original unit (exp(b) - 1) of '{nm}', final ElasticNet (undefined for log1p-transformed inputs, which give elasticities)", "06", M_FINAL_EN, FS, "fit on all 1,460 houses (descriptive)",
              "approx. % change in price per unit", "%", "1,460 houses", "Association, not a dollar effect or a causal effect; one-hot / ordinal / log1p caveats", co, lambda f=f: f().approx_pct_effect)
        R.add(f"COEF-{i + 1:02d}-ELASTICITY", f"Coefficient per unit of log1p(x) (approximate elasticity) of '{nm}' if it is a log1p input", "06", M_FINAL_EN, FS, "fit on all 1,460 houses (descriptive)", "coefficient per unit of log1p(x)", "log points", "1,460 houses", "Only meaningful for log1p inputs", co,
              lambda f=f: f().coef_original_units if bool(f().log1p_input) else np.nan)
    ab = ART / "feature_selection" / "ablation_folds.csv"
    for g, v, tag in (("A_garage", "neither", "GARAGE-NONE"), ("B_living", "neither", "LIVING-NONE"), ("B_living", "TotRmsAbvGrd only", "LIVING-ROOMS-ONLY"), ("F_bathrooms", "none", "BATH-NONE"), ("A_garage", "both", "GARAGE-BOTH")):
        for m, mt in (("ridge", "RIDGE"), ("gbr", "GBR")):
            def delta(g=g, v=v, m=m):
                d = csv(ab); a = d[(d.model == m) & (d.group == g)]
                ref = a[a.variant.str.contains("reference")].sort_values("fold").rmse_log.values; c = a[a.variant == v].sort_values("fold").rmse_log.values
                return float((c - ref).mean())
            R.add(f"ABL-{tag}-{mt}", f"Change in Log-RMSE when the group '{g}' is set to '{v}' (refit), {m}", "03", f"{m} fixed configuration (Ridge alpha 10 / GBR 03 config)", FS, "DEV-CV ablation on the 15 shared folds, one group changed at a time",
                  "change in Log-RMSE vs reference", "log points", "15 folds", "Fixed hyperparameters; exploratory (about 20 variants, no multiplicity correction); different from permutation importance", ab, delta)
    return R


def _intervals(R):
    ic = TAB / "intervals_main_comparison.csv"
    for meth, tag in (("CQR", "CQR"), ("Quantile GB", "RAW")):
        r = lambda meth=meth: csv(ic).set_index("Unnamed: 0").loc[meth]
        for col, suffix, units in (("Empirical coverage", "COVERAGE", "share"), ("Coverage gap (pp)", "GAP", "percentage points"), ("Mean width USD", "MEANWIDTH", "USD"), ("Median width USD", "MEDWIDTH", "USD"), ("Median relative width %", "MEDRELWIDTH", "% of point prediction"), ("Mean interval score USD", "WINKLER", "USD")):
            R.add(f"INT-{tag}-{suffix}", f"{meth} 80% interval: {col}", "07", f"{meth} intervals; base quantile GB (300 trees, depth 3, fixed, untuned)", FS,
                  "NESTED-style split CQR inside each outer fold (proper-training 75% / calibration 25% / evaluation fold); mean of 3 repeats", col, units, "1,460 houses per repeat", "Marginal coverage only; exchangeability assumed; base configuration not tuned", ic, lambda r=r, col=col: r()[col])
    isg = TAB / "segments_interval_coverage.csv"
    d = csv(isg)
    for _, row in d.iterrows():
        i = row["Unnamed: 0"]
        for col, suffix, units in (("coverage", "COVERAGE", "share"), ("mean_width_usd", "MEANWIDTH", "USD"), ("median_rel_width_pct", "MEDRELWIDTH", "% of point prediction")):
            R.add(f"INTSEG-{slug(i)}-{suffix}", f"CQR {suffix.lower()} for segment '{i}'", "08", "tuned GBR point + split CQR (07 intervals)", FS, "NESTED-style out-of-fold CQR; segment from deployable features unless labelled retrospective",
                  col, units, f"{row['n']:.0f} houses per repeat", "Segment coverage has no guarantee; small groups; Wilson interval from repeat 0 only", isg, lambda i=i, col=col: csv(isg).set_index("Unnamed: 0").loc[i, col])
    cvi = ART / "intervals" / "cv_intervals.csv.gz"
    R.add("INT-CQR-POINT-OUTSIDE", "Share of houses whose tuned-GBR point prediction lies outside the CQR interval (not forced inside)", "07", "tuned GBR point + split CQR", FS, "NESTED-style out-of-fold; all 4,380 predictions", "share outside", "share", "1,460 houses x 3 repeats",
          "Point and interval models are separate models; informative about their consistency", cvi, lambda: float(((csv(cvi).point < csv(cvi).lo_cqr) | (csv(cvi).point > csv(cvi).hi_cqr)).mean()))
    R.add("INT-RAW-POINT-OUTSIDE", "Share of houses whose point prediction lies outside the raw quantile interval", "07", "tuned GBR point + raw quantile GB intervals", FS, "NESTED-style out-of-fold", "share outside", "share", "1,460 houses x 3 repeats", "-", cvi,
          lambda: float(((csv(cvi).point < csv(cvi).lo_raw) | (csv(cvi).point > csv(cvi).hi_raw)).mean()))
    it = TAB / "intervals_temporal.csv"
    R.add("INT-TEMP-CQR-COVERAGE", "CQR coverage under the chronological split (calibrated on 2006-2008)", "07", "split CQR, 20 random calibration splits", FS, "TEMPORAL; calibration older than test houses (exchangeability violated by design)", "coverage", "share", "513 test houses",
          "Robustness indication, not a conformal guarantee", it, lambda: csv(it).set_index("method").loc["CQR", "coverage (mean of 20 splits)"])
    R.add("INT-TEMP-CQR-COVERAGE-MIN", "Lowest CQR coverage over the 20 calibration splits (temporal)", "07", "split CQR", FS, "TEMPORAL", "coverage", "share", "513", "-", it, lambda: csv(it).set_index("method").loc["CQR", "min"])
    R.add("INT-TEMP-CQR-COVERAGE-MAX", "Highest CQR coverage over the 20 calibration splits (temporal)", "07", "split CQR", FS, "TEMPORAL", "coverage", "share", "513", "-", it, lambda: csv(it).set_index("method").loc["CQR", "max"])
    return R


def _segments(R):
    sc = TAB / "segments_consolidated.csv"
    for i, row in csv(sc).iterrows():
        t, gname = row["Segment Type"], row["Group"]
        proto = ("RETROSPECTIVE (banded on realised price; diagnostic only)" if "actual" in t else "NESTED-CV out-of-fold; deployable grouping" if "Sale year" not in t else "NESTED-CV out-of-fold by sale-year metadata (not temporal validation)")
        for col, suffix, units in (("N", "N", "houses per repeat"), ("Log-RMSE", "LOGRMSE", "log points"), ("MAE USD", "MAE", "USD"), ("RMSE USD", "RMSE", "USD"), ("MAPE %", "MAPE", "%"), ("Mean Signed Error USD", "SIGNED", "USD (actual - predicted)")):
            R.add(f"SEG-{slug(t)}-{slug(gname)}-{suffix}", f"{col} for {t}: {gname}", "08", M_GBR, FS, proto, col, units, f"{row['N']:.0f} houses per repeat",
                  "Descriptive; overlapping groups; two atypical houses inflate luxury / high-quality / newest-house / Edwards figures", sc, lambda i=i, col=col: csv(sc).loc[i, col])
    ds = TAB / "segments_distribution_shift.csv"
    from src.data.loading import TRAIN_PATH
    R.add("DATA-N4000", "Number of houses with living area above 4,000 sq ft in the labelled data", "01", "-", "raw data", "descriptive count", "houses with GrLivArea > 4000", "houses", "1,460", "Very few comparable training examples", TRAIN_PATH, lambda: int((pd.read_csv(TRAIN_PATH).GrLivArea > 4000).sum()))
    R.add("SHIFT-MAXABS-SMD", "Largest absolute standardised mean difference between training years and test years over ten predictors", "08", "-", FS, "descriptive comparison of predictors, 947 vs 513 houses", "max |SMD|", "SD units", "10 features", "Descriptive; no causal claim", ds, lambda: csv(ds).SMD.abs().max())
    R.add("SHIFT-MAXKS", "Largest KS statistic between training years and test years over ten predictors", "08", "-", FS, "descriptive comparison", "max KS statistic", "-", "10 features", "-", ds, lambda: csv(ds)["KS stat"].max())
    return R


def _routing(R):
    hp = TAB / "routing_heldout_policy.csv"
    P = lambda: csv(hp).iloc[0]
    for col, suffix, units in (("AUTO %", "AUTOSHARE", "% of houses"), ("AUTO N (per repeat)", "AUTON", "houses per repeat"), ("AUTO MAE", "AUTOMAE", "USD"), ("AUTO RMSE", "AUTORMSE", "USD"), ("AUTO MAPE", "AUTOMAPE", "%"), ("AUTO coverage", "AUTOCOV", "share"),
                               ("AUTO median rel. width", "AUTOMEDREL", "relative width"), ("ESCALATE %", "ESCSHARE", "% of houses"), ("ESCALATE MAE", "ESCMAE", "USD"), ("ESCALATE MAPE", "ESCMAPE", "%"), ("ESCALATE coverage", "ESCCOV", "share"),
                               ("overall MAE", "OVERALLMAE", "USD"), ("overall MAPE", "OVERALLMAPE", "%"), ("overall coverage", "OVERALLCOV", "share"), ("median tau", "MEDIANTAU", "relative width")):
        R.add(f"RT-P-{suffix}", f"Primary routing policy (hypothetical target AUTO MAPE <= 7%, conservative), held-out: {col}", "09", "tuned GBR + split CQR; routing rule w <= tau", FS, P_RT + "; mean of 3 repeats", col, units, "1,460 houses per repeat",
              "Hypothetical target (the proposal states none); ESCALATE errors are errors of the automated prediction before human review; AUTO coverage is a subgroup, not covered by the marginal guarantee", hp, lambda col=col: P()[col])
    for sc_i, tag in ((1, "S6"), (2, "S8"), (3, "S7PT"), (4, "S7COV")):
        for col, suffix in (("AUTO %", "AUTOSHARE"), ("AUTO MAPE", "AUTOMAPE"), ("AUTO coverage", "AUTOCOV")):
            R.add(f"RT-{tag}-{suffix}", f"Routing scenario {tag} (hypothetical), held-out: {col}", "09", "tuned GBR + split CQR", FS, P_RT, col, "%" if "MAPE" in col or "%" in col else "share", "1,460 houses per repeat", "Hypothetical scenario", hp, lambda sc_i=sc_i, col=col: csv(hp).iloc[sc_i][col])
    rep = TAB / "routing_primary_by_repeat.csv"
    for stat, f in (("MIN", "min"), ("MAX", "max")):
        R.add(f"RT-P-AUTOSHARE-{stat}-REPEAT", f"{f} AUTO share over the three CV repeats (primary policy)", "09", "tuned GBR + split CQR", FS, P_RT, "AUTO share", "share", "3 repeats", "Each repeat uses its own 5 fold thresholds", rep, lambda f=f: getattr(csv(rep)["AUTO share"], f)())
    th = TAB / "routing_thresholds_by_fold.csv"
    for f, nm in (("min", "MIN"), ("median", "MEDIAN"), ("max", "MAX")):
        R.add(f"RT-P-TAUK-{nm}", f"{f} of the 15 fold-wise thresholds selected for the primary scenario", "09", "routing rule", FS, "threshold selection on development data only", "tau_k", "relative width", "15 folds", "Neighbouring thresholds are not statistically distinguishable", th, lambda f=f: getattr(csv(th)["P"], f)())
    dp = TAB / "routing_deployment_primary.csv"
    for col, suffix, units in (("tau_final", "TAUFINAL", "relative width"), ("dev_auto_share", "DEVAUTOSHARE", "share"), ("dev_auto_mape", "DEVAUTOMAPE", "%"), ("boot_tau_p10", "BOOTP10", "relative width"), ("boot_tau_p90", "BOOTP90", "relative width")):
        R.add(f"RT-FINAL-{suffix}", f"Deployable threshold by the pre-specified rule on all development data: {col}", "09", "routing rule", FS, "development data only (in-sample for this threshold; no independent evaluation possible)", col, units, "1,460 houses",
              "No untouched data remain: performance of this exact threshold is not independently measured", dp, lambda col=col: csv(dp)[col].iloc[0])
    ge = TAB / "routing_primary_group_errors.csv"
    for grp, tag in (("AUTO", "AUTO"), ("ESCALATE", "ESC")):
        for col, suffix, units in (("share_APE_over_15pct", "APE15", "share of houses"), ("p95_APE", "P95APE", "%"), ("median_APE", "MEDAPE", "%"), ("median_AE", "MEDAE", "USD")):
            R.add(f"RT-P-{tag}-{suffix}", f"Primary routing policy, held-out, {grp} group: {col}", "09", "tuned GBR + split CQR; primary frozen policy", FS, P_RT + "; repeat 0 routes of the 15 frozen fold thresholds", col, units, "1,460 houses (one prediction per house)",
                  "Errors of the automated prediction before human review; hypothetical target", ge, lambda grp=grp, col=col: csv(ge).set_index("Unnamed: 0").loc[grp, col])
    ss = TAB / "routing_signal_strength.csv"
    R.add("RT-SIGNAL-SPEARMAN-APE", "Spearman correlation between relative interval width and absolute percentage error of the point prediction (held-out)", "09", "tuned GBR + split CQR", FS, P_RT, "Spearman rho", "-", "1,460 houses", "Moderate rank association: width is an imperfect signal of error", ss, lambda: csv(ss).spearman_width_vs_ape.iloc[0])
    R.add("RT-SIGNAL-SPEARMAN-USD", "Spearman correlation between relative interval width and absolute USD error (held-out)", "09", "tuned GBR + split CQR", FS, P_RT, "Spearman rho", "-", "1,460 houses", "-", ss, lambda: csv(ss).spearman_width_vs_abs_usd_error.iloc[0])
    tp = TAB / "routing_temporal.csv"
    for col, suffix, units in (("tau_T", "TAU", "relative width"), ("AUTO % (mean)", "AUTOSHARE", "% of houses"), ("AUTO MAPE (mean)", "AUTOMAPE", "%"), ("AUTO MAE (mean)", "AUTOMAE", "USD"), ("AUTO coverage (mean)", "AUTOCOV", "share"), ("ESCALATE % (mean)", "ESCSHARE", "% of houses"), ("ESCALATE MAPE (mean)", "ESCMAPE", "%")):
        R.add(f"RT-TEMP-P-{suffix}", f"Temporal routing (threshold from 2006-2008 only), primary scenario: {col}", "09", "tuned GBR (historical-only) + split CQR", FS, "TEMPORAL + threshold selected on training-year development data; mean of 10 calibration splits", col, units,
              "513 test houses", "One split; hypothetical target", tp, lambda col=col: csv(tp).set_index("scenario").loc["P", col])
    for lab, f, key in (("price_pred", "routing_segments_price_pred.csv", "price_pred"), ("quality_tier", "routing_segments_quality_tier.csv", "quality_tier"), ("age_group", "routing_segments_age_group.csv", "age_group"),
                        ("area_group", "routing_segments_area_group.csv", "area_group"), ("Neighborhood", "routing_segments_Neighborhood.csv", "Neighborhood")):
        p = TAB / f
        for _, row in csv(p).iterrows():
            gname = row[key]
            for col, suffix, units in (("auto_share", "AUTOSHARE", "share"), ("auto_n", "AUTON", "houses per repeat"), ("auto_mape", "AUTOMAPE", "%"), ("auto_coverage", "AUTOCOV", "share")):
                R.add(f"RTSEG-{slug(lab)}-{slug(gname)}-{suffix}", f"Routing, group '{gname}' ({lab}): {col}", "09", "tuned GBR + split CQR; primary frozen policy", FS, P_RT + "; deployable grouping", col, units, f"{row['n']:.0f} houses per repeat",
                      "AUTO subsets can be tiny (see auto_n); NaN means no AUTO houses", p, lambda p=p, key=key, gname=gname, col=col: csv(p).set_index(key).loc[gname, col])
    return R


def _unsupported(R):
    for cid, claim, why in (
        ("UNS-COST", "Automatic valuation reduces appraisal cost / increases revenue (any financial benefit)", "No cost, revenue or appraiser data exist; only hypothetical cost scenarios were run, not used for selection"),
        ("UNS-PROD", "The system is production-ready / ready for unrestricted commercial deployment", "No independent test set, no monitoring, no cost or audit evidence; see the readiness assessment"),
        ("UNS-CAUSAL", "A feature or feature group causes higher or lower sale prices (causal effect)", "Coefficients, permutation importance and ablation measure predictive association only"),
        ("UNS-SUBGROUP", "80% interval coverage is guaranteed within every neighbourhood, price band or routing group", "Split CQR gives marginal coverage only; observed subgroup coverage ranges widely (Crawfor ~69%, AUTO group ~78%)"),
        ("UNS-FUTURE", "Performance after 2010 / in another city or market", "Data cover Ames, Iowa, 2006-2010; one temporal split of 513 houses"),
        ("UNS-INDEP", "An independent final test-set performance estimate", "No untouched labelled data exist (test.csv has no labels); nested CV evaluates the procedure under selection bias from shared rows"),
        ("UNS-HIST", "Historical figures quoted in the brief (about 79.2% coverage; threshold 0.217; AUTO 65.1%; AUTO MAPE 7.15%)", "Not recorded anywhere in the repository; not reproduced"),
        ("UNS-TUNEDLASSO", "A tuned Lasso model was evaluated", "Only a fixed-configuration Lasso baseline exists; tuning covered ElasticNet and Gradient Boosting")):
        R.add(cid, claim, "n/a", "-", "-", "-", "-", "-", "-", why, "no evidence", status="unsupported")
    R.add("PEND-DOMAIN-AUC", "Domain-classifier AUC between training and test years (0.55 reported in the 08 notebook output)", "08", "-", FS, "-", "AUC", "-", "1,460", "Printed in the notebook but not saved as an artifact: not traceable from a file", "notebooks/08 (stream output only)", status="pending")
    return R


def build_registry():
    R = Registry()
    for f in (_accuracy, _model_and_complexity, _features, _intervals, _segments, _routing, _unsupported):
        f(R)
    d = R.frame()
    assert d.claim_id.is_unique, d.claim_id[d.claim_id.duplicated()].tolist()
    return d


def value(reg, cid):
    """Value of one claim (raises KeyError for unknown ids, so prose cannot cite a claim that does not exist)."""
    r = reg.loc[reg.claim_id == cid]
    if r.empty:
        raise KeyError(cid)
    return r.value.iloc[0]


# ---------------------------------------------------------------- consistency checks across independent sources
def consistency_checks(reg):
    """Returns [(check, ok, detail)]. Independent recomputations and cross-notebook agreement; none depends on how the registry was typed."""
    from src.evaluation import final_evaluation as E
    out = []
    add = lambda name, ok, detail="": out.append((name, bool(ok), detail))
    sup = reg[reg.status == "supported"]
    add("every supported claim has a source file that exists", all(Path(p).exists() for p in sup.source_file.unique()), f"{sup.source_file.nunique()} distinct source files")
    add("every supported claim has a finite value except explicitly undefined group metrics", sup.value.apply(lambda v: np.isfinite(v)).sum() >= 0.98 * len(sup), f"{int((~sup.value.apply(np.isfinite)).sum())} undefined of {len(sup)}")
    add("unsupported / pending claims carry no value", reg[reg.status != "supported"].value.isna().all())
    add("required fields are filled for every claim", reg[["claim_id", "claim", "source_notebook", "experiment_id", "evaluation_protocol", "metric", "limitations", "source_file"]].notna().all().all())
    fm = E.fold_metrics(E.load_oof())
    g = fm[fm.model == "gbr_tuned"]
    add("nested GBR Log-RMSE / MAE recomputed from the stored out-of-fold predictions match 05", np.isclose(g.rmse_log.mean(), value(reg, "ACC-GBR-LOGRMSE")) and np.isclose(g.mae.mean(), value(reg, "ACC-GBR-MAE")), "06 OOF table vs 05 fold metrics")
    add("routing overall MAE / MAPE equal the nested GBR MAE / MAPE (same predictions)", np.isclose(value(reg, "RT-P-OVERALLMAE"), value(reg, "ACC-GBR-MAE"), rtol=1e-6) and np.isclose(value(reg, "RT-P-OVERALLMAPE"), value(reg, "ACC-GBR-MAPE"), rtol=1e-3))
    add("routing overall coverage equals the CQR marginal coverage of 07", np.isclose(value(reg, "RT-P-OVERALLCOV"), value(reg, "INT-CQR-COVERAGE"), atol=1e-3))
    seg = pd.read_csv(TAB / "segments_temporal_comparison.csv")
    r = seg[(seg.Model == "Gradient Boosting") & seg["Evaluation Protocol"].str.contains("historical-only")].iloc[0]
    add("temporal GBR Log-RMSE agrees between the temporal artifact and the 08 comparison table", np.isclose(r["Log-RMSE"], value(reg, "TEMP-GBR-LOGRMSE"), atol=1e-4))
    add("temporal split sizes: 947 + 513 = 1,460", value(reg, "TEMP-NTRAIN") + value(reg, "TEMP-NTEST") == 1460)
    add("AUTO + ESCALATE shares sum to 100%", np.isclose(value(reg, "RT-P-AUTOSHARE") + value(reg, "RT-P-ESCSHARE"), 100.0))
    sp = [value(reg, f"SEG-price_predicted_deployable-{s}-N") for s in ("low", "mid", "luxury")]
    add("deployable price segments sum to 1,460 houses per repeat", np.isclose(sum(sp), 1460))
    metas = {k: json.loads((ART / k / "run_metadata.json").read_text()) for k in ("baseline_comparison", "tuning", "intervals", "routing", "segments", "final_evaluation") if (ART / k / "run_metadata.json").exists()}
    add("target transformation recorded as log1p wherever it is stated", all("log1p" in str(m.get("target", "log1p")) for m in metas.values()))
    add("feature set is 'reduced' wherever it is recorded", all(m.get("feature_set", "reduced") == "reduced" for m in metas.values()))
    add("random seed is 42 wherever it is recorded", all(m.get("seed", 42) == 42 for m in metas.values()))
    add("baseline and tuning used the same 15 folds (fold hash)", metas["baseline_comparison"]["cv"]["fold_hash"] == metas["tuning"]["fold_hash"])
    add("experiment identifiers match the run metadata", metas["intervals"]["experiment_id"] == EXP["07"] and metas["routing"]["experiment_id"] == EXP["09"] and metas["tuning"]["experiment_id"] == EXP["05"] and metas["baseline_comparison"]["experiment_id"] == EXP["04"])
    srcs = [p for p in (ROOT / "src").rglob("*.py")]
    add("no SHAP import anywhere in src/", not any(re.search(r"^\s*(import|from)\s+shap", p.read_text(encoding="utf-8"), re.M) for p in srcs))
    add("GBR is not claimed superior on Log-RMSE: the paired interval contains zero", value(reg, "ACC-PAIRED-CI_LO") < 0 < value(reg, "ACC-PAIRED-CI_HI"))
    add("AUTO coverage claim is consistent: below nominal 80% in the held-out primary policy", value(reg, "RT-P-AUTOCOV") < 0.80)
    return out
