"""Reusable EDA analyses. Exploratory only: nothing here is fitted for reuse downstream.

Each function takes the frame from `src.data.loading.load_raw()` and returns tables.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder

SALE_CONTEXT = ["MoSold", "YrSold", "SaleType", "SaleCondition"]
OUTLIER_IDS = [524, 1299]
TARGET = "SalePrice"


def feature_types(df):
    """numeric vs categorical predictors (MSSubClass is a nominal code)."""
    cat = [c for c in df.columns if df[c].dtype == object] + ["MSSubClass"]
    num = [c for c in df.columns if c not in cat + ["Id", TARGET]]
    return num, cat


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    return np.sqrt(chi2 / (n * (min(t.shape) - 1))) if min(t.shape) > 1 else np.nan


# ---------------------------------------------------------------- 1. missingness
def missingness_association(df, col, alpha=0.01):
    """Is `col` missing completely at random? For every other predictor test whether
    missing and observed rows differ (Mann-Whitney U for numeric, chi-square / Cramer's V
    for categorical). Also reports a cross-validated AUC for predicting missingness."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    num, cat = feature_types(df)
    miss = df[col].isna()
    rows = []
    for c in [c for c in num + cat if c != col]:
        s = df[c]
        if s.isna().all() or (s.dropna().nunique() < 2):
            continue
        if c in num:
            a, b = s[miss].dropna(), s[~miss].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            p = stats.mannwhitneyu(a, b).pvalue
            auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))
            rows.append((c, "numeric", p, abs(auc - 0.5) * 2))      # rank-biserial effect
        else:
            t = pd.crosstab(s.fillna("NaN"), miss)
            if t.shape[0] < 2:
                continue
            p = stats.chi2_contingency(t)[1]
            rows.append((c, "categorical", p, cramers_v(s.fillna("NaN"), miss)))
    res = pd.DataFrame(rows, columns=["variable", "kind", "p_value", "effect_size"])
    res["p_bonferroni"] = (res.p_value * len(res)).clip(upper=1)
    res["significant"] = res.p_bonferroni < alpha
    res = res.sort_values("effect_size", ascending=False)
    X = df[[c for c in num + cat if c != col]].copy()
    for c in cat:
        if c in X:
            X[c] = X[c].astype(str)
    Xe = _encode(X)
    clf = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0)
    oof = cross_val_predict(clf, Xe, miss.astype(float), cv=KFold(5, shuffle=True, random_state=0))
    return res, roc_auc_score(miss, oof)


def _encode(X):
    X = X.copy()
    cats = [c for c in X.columns if X[c].dtype == object]
    if cats:
        X[cats] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cats].fillna("NaN"))
    return X


def compare_lotfrontage_imputers(df, n_splits=5, seed=0):
    """Mask known LotFrontage values and see how well each fill recovers them.
    Statistics are computed on the training part of each fold only."""
    obs = df[df.LotFrontage.notna()].reset_index(drop=True)
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    out = {k: [] for k in ["global median", "median by Neighborhood", "median by Neighborhood+LotConfig", "ratio to sqrt(LotArea) by Neighborhood"]}
    truth = []
    for tr, va in kf.split(obs):
        a, b = obs.iloc[tr], obs.iloc[va]
        g = a.LotFrontage.median()
        nb = a.groupby("Neighborhood").LotFrontage.median()
        nc = a.groupby(["Neighborhood", "LotConfig"]).LotFrontage.median()
        ratio = (a.LotFrontage / np.sqrt(a.LotArea)).groupby(a.Neighborhood).median()
        out["global median"].append(np.full(len(b), g))
        out["median by Neighborhood"].append(b.Neighborhood.map(nb).fillna(g).values)
        out["median by Neighborhood+LotConfig"].append([nc.get((n, c), nb.get(n, g)) for n, c in zip(b.Neighborhood, b.LotConfig)])
        out["ratio to sqrt(LotArea) by Neighborhood"].append((b.Neighborhood.map(ratio).fillna((a.LotFrontage / np.sqrt(a.LotArea)).median()) * np.sqrt(b.LotArea)).values)
        truth.append(b.LotFrontage.values)
    t = np.concatenate(truth)
    rows = []
    for k, v in out.items():
        p = np.concatenate([np.asarray(x, float) for x in v])
        rows.append((k, np.sqrt(np.mean((p - t) ** 2)), np.mean(np.abs(p - t))))
    return pd.DataFrame(rows, columns=["imputer", "RMSE (ft)", "MAE (ft)"]).sort_values("RMSE (ft)")


# ---------------------------------------------------------------- 2. signal table
def signal_table(df):
    num, cat = feature_types(df)
    ly = np.log1p(df[TARGET])
    rows = []
    X = df[num + cat].copy()
    Xe = _encode(X.assign(**{c: X[c].astype(str) for c in cat}))
    Xe = Xe.fillna(Xe.median())
    discrete = np.array([c in cat or df[c].nunique() <= 15 for c in Xe.columns])
    mi = pd.Series(mutual_info_regression(Xe, ly, discrete_features=discrete, random_state=0), index=Xe.columns)
    for c in num:
        s = df[c]
        ok = s.notna()
        rows.append((c, "numeric", stats.pearsonr(s[ok], ly[ok])[0], stats.spearmanr(s[ok], ly[ok])[0], np.nan, mi[c]))
    for c in cat:
        g = [ly[df[c].astype(str) == lv].values for lv in df[c].astype(str).unique()]
        g = [x for x in g if len(x) > 0]
        n = len(ly)
        ss_b = sum(len(x) * (x.mean() - ly.mean()) ** 2 for x in g)
        eta2 = ss_b / ((ly - ly.mean()) ** 2).sum()
        rows.append((c, "categorical", np.nan, np.nan, eta2, mi[c]))
    t = pd.DataFrame(rows, columns=["feature", "kind", "pearson", "spearman", "eta_squared", "mutual_info"])
    t["abs_spearman_or_sqrt_eta2"] = np.where(t.kind == "numeric", t.spearman.abs(), np.sqrt(t.eta_squared))
    t["mi_rank"] = t.mutual_info.rank(ascending=False).astype(int)
    t["assoc_rank"] = t.abs_spearman_or_sqrt_eta2.rank(ascending=False).astype(int)
    return t.sort_values("mutual_info", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 3. correlation clusters
def correlation_clusters(df, threshold=0.7):
    """Connected groups of numeric features with |Spearman| >= threshold, plus VIF."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    num, _ = feature_types(df)
    X = df[num].copy()
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    c = X.corr(method="spearman").fillna(0)
    d = 1 - c.abs().values
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    ly = np.log1p(df[TARGET])
    sig = X.apply(lambda s: abs(stats.spearmanr(s, ly)[0]))
    t = pd.DataFrame({"feature": X.columns, "cluster": labels, "abs_spearman_price": sig.values})
    keep = t.sort_values("abs_spearman_price", ascending=False).groupby("cluster").head(1).feature
    t["suggest_keep"] = t.feature.isin(keep)
    t = t[t.groupby("cluster").feature.transform("size") > 1].sort_values(["cluster", "abs_spearman_price"], ascending=[True, False])
    # VIF_j = 1 / (1 - R^2_j) from regressing each feature on all the others (inf = exact linear dependency)
    Z_ = ((X - X.mean()) / X.std()).values
    vif = {}
    for j, name in enumerate(X.columns):
        others = np.delete(Z_, j, axis=1)
        beta = np.linalg.lstsq(others, Z_[:, j], rcond=None)[0]
        r2 = 1 - np.sum((Z_[:, j] - others @ beta) ** 2) / np.sum(Z_[:, j] ** 2)
        vif[name] = np.inf if r2 > 1 - 1e-9 else 1 / (1 - r2)
    vif = pd.Series(vif)
    return t.reset_index(drop=True), vif.sort_values(ascending=False), c


# ---------------------------------------------------------------- 4. CV uncertainty probe
def oof_residuals(df, model="linear", n_splits=5, n_repeats=2, seed=0):
    """Out-of-fold residuals on log1p(price), averaged over repeats."""
    ly = np.log1p(df[TARGET]).values
    num, cat = feature_types(df)
    if model == "linear":
        X = np.c_[np.ones(len(df)), np.log(df.GrLivArea), df.OverallQual]
    else:
        X = df[num].values
    oof = np.zeros((n_repeats, len(df)))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, va) in enumerate(rkf.split(X)):
        r = i // n_splits
        if model == "linear":
            b = np.linalg.lstsq(X[tr], ly[tr], rcond=None)[0]
            oof[r, va] = X[va] @ b
        else:
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, random_state=seed).fit(X[tr], ly[tr])
            oof[r, va] = m.predict(X[va])
    pred = oof.mean(axis=0)
    return pred, ly - pred


def segment_spread(df, pred, res, min_n=25):
    d = df.assign(pred=pred, res=res)
    d["pred_band"] = pd.qcut(d.pred, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    d["qual_tier"] = pd.cut(d.OverallQual, [0, 4, 6, 8, 10], labels=["1-4", "5-6", "7-8", "9-10"])
    out = {}
    for key in ["pred_band", "qual_tier", "Neighborhood"]:
        g = d.groupby(key, observed=True).res.agg(["count", "mean", "std"])
        out[key] = g[g["count"] >= min_n].round(3)
    tests = {}
    for key in ["pred_band", "qual_tier", "Neighborhood"]:
        groups = [x.res.values for _, x in d.groupby(key, observed=True) if len(x) >= min_n]
        tests[key] = stats.levene(*groups, center="median").pvalue
    # can |residual| be predicted from features known before pricing? (width proxy)
    return out, tests, d


# ---------------------------------------------------------------- 5. leakage / realism ablation
def cv_rmse(df, drop_cols=(), drop_train_ids=(), eval_exclude_ids=(), n_splits=5, n_repeats=3, seed=0):
    """CV RMSE on log1p(price) with a HistGB. Validation rows exclude `eval_exclude_ids`;
    `drop_train_ids` are removed from training folds only. Same folds for every call."""
    num, cat = feature_types(df)
    X = df[[c for c in num + cat if c not in drop_cols]].copy()
    for c in [c for c in cat if c in X]:
        X[c] = X[c].astype(str)
    Xe = _encode(X)
    ly = np.log1p(df[TARGET]).values
    ids = df.Id.values
    scores = []
    for tr, va in RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(Xe):
        tr = tr[~np.isin(ids[tr], list(drop_train_ids))]
        va = va[~np.isin(ids[va], list(eval_exclude_ids))]
        m = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, random_state=seed).fit(Xe.iloc[tr], ly[tr])
        scores.append(np.sqrt(np.mean((m.predict(Xe.iloc[va]) - ly[va]) ** 2)))
    return float(np.mean(scores)), float(np.std(scores))


def leakage_audit(df):
    rows = []
    sale = set(SALE_CONTEXT)
    notes = {
        "SaleType": "describes the transaction; 'New' is also a property fact, but it is only known at sale",
        "SaleCondition": "transaction fact; 'Partial' = home not complete when assessed",
        "MoSold": "sale-time; not known when a listing price is set",
        "YrSold": "sale-time; keep as temporal split key and age reference only",
        "MiscVal": "dollar value of a misc feature; confirm it is not derived from the sale price",
        "YearRemodAdd": "1 row has remodel year after YrSold",
        "GarageYrBlt": "9 rows dated before YearBuilt; redundant with YearBuilt",
        "Neighborhood": "high-cardinality; target encoding must be fitted in-fold",
    }
    for c in df.columns:
        if c in ("Id", TARGET):
            continue
        rows.append((c, "No" if c in sale else "Yes", "No", notes.get(c, "")))
    return pd.DataFrame(rows, columns=["column", "known_at_listing", "derived_from_price", "note"])
