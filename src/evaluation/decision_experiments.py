"""Evidence for open decisions D2 (outliers) and D3 (Neighborhood encoding).

Run from the project root:  python3 -m src.evaluation.decision_experiments

Decision rules, fixed BEFORE the results were seen
  D2  Exclude Ids 524 and 1299 from TRAINING iff, on validation rows that exclude them, the paired RMSE gain
      (keep - drop) is > 2 standard errors for BOTH a Ridge and a gradient-boosting model, and the Ridge gain is
      larger than every gain obtained by dropping two random other rows (placebo). Otherwise keep them and report both ways.
  D3  Use smoothed in-fold target encoding of Neighborhood instead of one-hot iff it beats one-hot by > 2 standard
      errors for at least one model family and is not worse by > 1 standard error for the other. Otherwise keep one-hot.
Ridge (alpha = 10) and HistGradientBoosting (depth 4, 300 iterations, lr 0.05) are fixed measuring instruments,
not the final models. Preprocessing is the `baseline` strategy, refitted inside every training fold.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.data.loading import load_raw
from src.evaluation.cv import cv_scores, get_folds, paired_difference, summarize
from src.features.preprocessing import SUGGESTED_TRAINING_EXCLUSIONS, build_preprocessor

SEED = 42


def make_model(kind, **prep):
    if kind == "ridge":
        return Pipeline([("prep", build_preprocessor("baseline", **prep)), ("m", Ridge(alpha=10))])
    return Pipeline([("prep", build_preprocessor("baseline", scale=None, **prep)),
                     ("m", HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, random_state=SEED))])


def d2(X, y, ids, folds):
    out, rows = {}, []
    for kind in ("ridge", "gbm"):
        m = make_model(kind)
        keep = cv_scores(m, X, y, folds, ids=ids, eval_exclude_ids=SUGGESTED_TRAINING_EXCLUSIONS)
        drop = cv_scores(m, X, y, folds, ids=ids, drop_train_ids=SUGGESTED_TRAINING_EXCLUSIONS, eval_exclude_ids=SUGGESTED_TRAINING_EXCLUSIONS)
        allr = cv_scores(m, X, y, folds, ids=ids)
        gain, se = paired_difference(keep, drop)           # keep - drop > 0 means dropping helps
        out[kind] = (gain, se)
        rows.append((kind, allr.rmse_log.mean(), keep.rmse_log.mean(), drop.rmse_log.mean(), gain, se, gain / se,
                     keep.mae.mean(), drop.mae.mean()))
    t = pd.DataFrame(rows, columns=["model", "RMSE all rows", "RMSE clean rows, train keeps", "RMSE clean rows, train drops",
                                    "gain (keep-drop)", "SE", "gain/SE", "MAE keep ($)", "MAE drop ($)"]).set_index("model")
    # placebo: dropping two random OTHER rows should not help by this much
    rng = np.random.default_rng(SEED)
    pool = ids[~np.isin(ids, SUGGESTED_TRAINING_EXCLUSIONS)]
    placebo = []
    m = make_model("ridge")
    for _ in range(25):
        pair = list(rng.choice(pool, 2, replace=False))
        k = cv_scores(m, X, y, folds, ids=ids, eval_exclude_ids=pair)
        d = cv_scores(m, X, y, folds, ids=ids, drop_train_ids=pair, eval_exclude_ids=pair)
        placebo.append(paired_difference(k, d)[0])
    return t, np.array(placebo), out


def d3(X, y, ids, folds):
    ly = np.log1p(y)
    res, rows = {}, []
    for kind in ("ridge", "gbm"):
        for enc in ("onehot", "target", "both", "none"):
            res[(kind, enc)] = cv_scores(make_model(kind, neighborhood_encoding=enc), X, y, folds)
        base = res[(kind, "onehot")]
        for enc in ("onehot", "target", "both", "none"):
            s = summarize(res[(kind, enc)])
            gain, se = paired_difference(base, res[(kind, enc)])      # onehot - enc > 0 means enc is better
            rows.append((kind, enc, s.RMSE_mean, s.RMSE_sd, s.MAE_mean, gain, se))
    return pd.DataFrame(rows, columns=["model", "Neighborhood encoding", "RMSE mean", "RMSE sd", "MAE mean ($)",
                                       "gain vs one-hot", "SE"]).set_index(["model", "Neighborhood encoding"])


def main():
    df = load_raw()
    X, y, ids = df.drop(columns=["SalePrice", "Id"]), df["SalePrice"], df.Id.values
    folds = get_folds(y)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    t2, placebo, o2 = d2(X, y, ids, folds)
    print("== D2: effect of excluding Ids 524 and 1299 from training ==")
    print(t2.round(5))
    print(f"placebo (25 random pairs, Ridge): gain mean {placebo.mean():+.5f}, sd {placebo.std(ddof=1):.5f}, max {placebo.max():+.5f}")
    ok = all(g / s > 2 for g, s in o2.values()) and o2["ridge"][0] > placebo.max()
    print("D2 rule ->", "EXCLUDE from training" if ok else "KEEP (report both ways)")
    t2.to_csv("reports/tables/decision_d2_outliers.csv"); pd.Series(placebo, name="placebo_gain").to_csv("reports/tables/decision_d2_placebo.csv", index=False)
    t3 = d3(X, y, ids, folds)
    print("\n== D3: Neighborhood encoding ==")
    print(t3.round(5))
    t3.to_csv("reports/tables/decision_d3_neighborhood_encoding.csv")
    def better(kind): g, s = t3.loc[(kind, "target"), ["gain vs one-hot", "SE"]]; return g / s
    adopt = (better("ridge") > 2 or better("gbm") > 2) and better("ridge") > -1 and better("gbm") > -1
    print("D3 rule ->", "USE target encoding" if adopt else "KEEP one-hot")


if __name__ == "__main__":
    main()
