"""Streamlit dashboard.   streamlit run app/dashboard.py     (needs artifacts/serving/bundle_v1.joblib: python3 -m src.serving.export_model)

Shows, for a property: point estimate (median), 80% split-CQR interval, AUTO/ESCALATE flag and the ElasticNet group decomposition.
Every accuracy number comes from the evidence registry (nested CV / temporal / routing artifacts), never from the exported model itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import coverage_chart
from app.labels import label
from app.result_panel import result_panel
from src.evaluation import evidence as E
from src.serving import predict as P

st.set_page_config(page_title="Ames price estimator", page_icon=":material/home:", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, .stApp {font-family: "Poppins", sans-serif;}
.block-container {padding: 1.1rem 1.8rem 0 1.8rem; max-width: none;}
header[data-testid="stHeader"] {display: none;}
[data-testid="stVerticalBlock"] {gap: .55rem;}
[data-baseweb="tab-list"] {gap: .25rem;}
[data-baseweb="tab-list"] button[role="tab"] {padding: .45rem .65rem; font-size: .88rem; white-space: nowrap; flex: 0 0 auto;}
.mast {display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; padding-bottom: .5rem;}   /* no border: the tab bar's underline is the only divider */
.mast h1 {margin: 0; padding: 0; font-size: 1.45rem; font-weight: 600; letter-spacing: -.01em; line-height: 1.2; display: inline;}
.mast .sub {display: block; font-size: .82rem; opacity: .6; margin: .1rem 0 0 0; white-space: nowrap;}
.mast > div:first-child {min-width: 0;}
.mast .right {display: flex; align-items: center; gap: 1rem; font-size: .74rem;}
.status {display: inline-flex; align-items: center; gap: .45rem; font-size: .74rem; font-weight: 500; padding: .22rem .6rem; border-radius: 6px;
         border: 1px solid rgba(128,128,128,.35); white-space: nowrap;}
.status i {width: .45rem; height: .45rem; border-radius: 50%; background: #e0a030; display: inline-block;}
.meta {display: flex; flex-wrap: nowrap; white-space: nowrap; opacity: .6; font-variant-numeric: tabular-nums;}
.meta span {padding-right: .8rem; margin-right: .8rem; border-right: 1px solid rgba(128,128,128,.4); line-height: 1;}
.meta span:last-child {border-right: 0; margin-right: 0; padding-right: 0;}
.hint {font-size: .8rem; opacity: .65; margin-top: -.4rem; margin-bottom: .6rem;}
/* keep the page steady while Streamlit reruns: no dimming of stale elements */
[data-stale="true"] {opacity: 1 !important; transition: none !important;}
iframe[title*="result_panel"] {border: 0; width: 100%;}
.mast h1 {white-space: nowrap;}
@media (max-width: 1150px) {.mast .sub {display: none;} .meta span:nth-child(2) {display: none;}}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2); padding: .7rem .9rem; border-radius: 14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

QUAL_NAMES = {1: "Very poor", 2: "Poor", 3: "Fair", 4: "Below average", 5: "Average", 6: "Above average", 7: "Good", 8: "Very good", 9: "Excellent",
              10: "Very excellent"}
COND_NAMES = {k: v for k, v in QUAL_NAMES.items() if k <= 9} | {9: "Excellent"}


@st.cache_resource
def bundle():
    return P.load_bundle()


@st.cache_data
def registry():
    return E.build_registry()


@st.cache_data
def group_order():
    """Business groups ordered by mean absolute dollar contribution over a training sample (fixed, so the waterfall rows stay put)."""
    dec, _ = P.explain(X.sample(200, random_state=0), b)
    return dec.groupby("group")["usd_contribution"].apply(lambda v: v.abs().mean()).sort_values(ascending=False).index.tolist()


def usd(v):
    return f"${v:,.0f}"


def sgn(v):
    return ("+" if v >= 0 else "−") + usd(abs(v))


b = bundle()
meta, sch = b["meta"], b["meta"]["schema"]
X = b["reference_rows"]
WHOLE = {c: bool((X[c].dropna() % 1 == 0).all()) for c in sch["numeric_range"]}


# ------------------------------------------------------------------ example houses
def _py(v):
    return v.item() if hasattr(v, "item") else v


@st.cache_data
def example_rows():
    """'Typical' = median / mode of every column; the others are real training houses: the one closest to the median living area within a quality band."""
    typ = {c: (X[c].mode(dropna=True).iloc[0] if X[c].dtype == object else X[c].median()) for c in X.columns}
    out = {"Typical Ames house": {c: _py(v) for c, v in typ.items()}}
    for name, q in (("Starter home (quality 4)", 4), ("Family home (quality 7)", 7), ("Premium home (quality 9)", 9)):
        g = X[X.OverallQual == q]
        i = (g.GrLivArea - g.GrLivArea.median()).abs().idxmin()
        r = {c: _py(v) for c, v in g.loc[i].items()}
        out[name] = {c: (typ[c] if (isinstance(v, float) and np.isnan(v)) else v) for c, v in r.items()}   # blank frontage etc. -> typical value
    return {k: {c: _py(v) for c, v in r.items()} for k, r in out.items()}


FORM_FIELDS = ["Neighborhood", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
               "LotArea", "LotFrontage", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "KitchenAbvGr", "KitchenQual", "ExterQual", "CentralAir",
               "GarageCars", "GarageArea", "Fireplaces"]


def apply_example(name):
    r = example_rows()[name]
    st.session_state["base"] = dict(r)
    for c in FORM_FIELDS:
        v = r[c]
        st.session_state[f"f_{c}"] = int(round(v)) if c in WHOLE and WHOLE[c] else v


if "base" not in st.session_state:
    apply_example("Typical Ames house")


def on_example():
    apply_example(st.session_state["example"])


# ------------------------------------------------------------------ input widgets (state lives under f_<column>)
def cat(name, title, help=None):
    return st.selectbox(title, sch["categorical"][name], key=f"f_{name}", format_func=lambda c: label(name, c), help=help)


def num(name, title, step=1, help=None):
    lo, hi = sch["numeric_range"][name]
    if WHOLE[name]:
        return st.number_input(title, min_value=int(lo), max_value=int(hi * 1.5 + 1), step=int(step), key=f"f_{name}", help=help)
    return st.number_input(title, min_value=float(lo), max_value=float(hi * 1.5 + 1), step=float(step), key=f"f_{name}", help=help)


# ------------------------------------------------------------------ header
st.markdown(
    f"""<div class="mast"><div><h1>Property price estimator</h1><span class="sub">Ames, Iowa &middot; price, 80% range, review flag</span></div>
    <div class="right"><div class="meta"><span>Model {meta['version']}</span><span>Built {meta['created'][:10]}</span><span>{meta['n_train']:,} sales</span>
    <span>Outliers {'excluded' if meta['exclude_d2'] else 'kept'} &middot; group decision D2</span></div>
    <span class="status"><i></i>Research prototype</span></div></div>""",
    unsafe_allow_html=True)

tab_est, tab_batch, tab_ev = st.tabs([":material/home: Single property", ":material/upload_file: Batch (CSV)", ":material/analytics: Model evidence & limits"])

# ------------------------------------------------------------------ single property
with tab_est:
    left, right = st.columns([1, 2.1], gap="medium")

    with left:
        st.selectbox("Start from an example", list(example_rows()), key="example", on_change=on_example,
                     help="Fills the form with a typical house or a real house from the training data. Change anything afterwards.")
        t_loc, t_qual, t_size, t_extra = st.tabs(["Location", "Quality", "Size", "Extras"])
        with t_loc:
            cat("Neighborhood", "Neighborhood (area of Ames)")
            cat("BldgType", "Building type")
            cat("HouseStyle", "House style")
        with t_qual:
            st.select_slider("Overall quality of materials and finish", options=list(QUAL_NAMES), key="f_OverallQual",
                             format_func=lambda k: f"{k} · {QUAL_NAMES[k]}")
            st.markdown('<div class="hint">How good the materials and workmanship are. Most Ames houses are 5–7.</div>', unsafe_allow_html=True)
            st.select_slider("Overall condition", options=list(COND_NAMES), key="f_OverallCond", format_func=lambda k: f"{k} · {COND_NAMES[k]}")
            st.markdown('<div class="hint">How well the house has been maintained. 5 means kept up as expected.</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                num("YearBuilt", "Year built")
                cat("KitchenQual", "Kitchen quality")
            with c2:
                num("YearRemodAdd", "Year remodelled", help="Same as year built if never remodelled.")
                cat("ExterQual", "Exterior quality")
        with t_size:
            c1, c2 = st.columns(2)
            with c1:
                num("1stFlrSF", "1st floor (sq ft)", 10)
                num("TotalBsmtSF", "Basement (sq ft)", 10, help="0 if no basement.")
                num("FullBath", "Full bathrooms")
                num("BedroomAbvGr", "Bedrooms")
                num("KitchenAbvGr", "Kitchens")
            with c2:
                num("2ndFlrSF", "2nd floor (sq ft)", 10, help="0 for a single-level house.")
                num("LotArea", "Lot area (sq ft)", 100)
                num("HalfBath", "Half bathrooms", help="Toilet and sink, no shower or tub.")
                num("TotRmsAbvGrd", "Rooms above ground", help="Does not include bathrooms.")
                num("LotFrontage", "Lot frontage (ft)", help="Length of the lot along the street.")
        with t_extra:
            c1, c2 = st.columns(2)
            with c1:
                num("GarageCars", "Garage (cars)", help="0 if no garage.")
                num("Fireplaces", "Fireplaces")
            with c2:
                num("GarageArea", "Garage (sq ft)", 10)
                cat("CentralAir", "Central air conditioning")
        st.caption("Attributes not shown take the value of the selected example.")

    # ---- predict
    row = dict(st.session_state["base"])
    for c in FORM_FIELDS:
        row[c] = st.session_state[f"f_{c}"]
    row["GrLivArea"] = row["1stFlrSF"] + row["2ndFlrSF"] + float(row.get("LowQualFinSF", 0))
    one = pd.DataFrame([row])[sch["columns"]].astype(X.dtypes.to_dict())
    pred = P.predict(one, b).iloc[0]
    dec, lvb = P.explain(one, b)

    with right:
        loc = lvb.set_index("part")["usd_contribution"]
        dec = dec.set_index("group").loc[group_order()].reset_index()          # fixed row order: bars change length, rows never jump
        rw = float(pred.rel_width)
        result_panel(dict(
            point=float(pred.point), lower=float(pred.lower), upper=float(pred.upper), route=pred.route, rel_width=rw, tau=float(b["tau"]),
            reference=float(dec["reference_prediction_usd"].iloc[0]), end=float(dec["prediction_usd"].iloc[0]),
            groups=[dict(name=g, value=float(v)) for g, v in zip(dec["group"], dec["usd_contribution"])],
            evidence=[["Typical error (MAPE)", f"{E.value(registry(), 'ACC-GBR-MAPE'):.1f}%"], ["Typical error in dollars", usd(E.value(registry(), 'ACC-GBR-MAE'))],
                      ["80% range really covers", f"{E.value(registry(), 'INT-CQR-COVERAGE'):.1%}"],
                      ["Coverage on later years", f"{E.value(registry(), 'INT-TEMP-CQR-COVERAGE'):.1%}"]],
            land=float(loc.get("Location-and-lot attributes", 0)), building=float(loc.get("Building and use attributes", 0)),
            gauge_note=(f"Relative width = (upper \u2212 lower) \u00f7 estimate. Up to {b['tau']:.0%} is marked AUTO (hypothetical threshold); "
                        "AUTO ranges still cover slightly under 80% of sale prices in testing."),
            waterfall_note="Each bar is what this house's attributes in that group add or subtract, starting from a typical Ames house. "
                           "It explains the ElasticNet model, so the end point can differ slightly from the estimate above."))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.markdown("##### Score many properties at once")
    st.write("Upload a CSV with the raw Ames predictor columns (same names and spellings as `train.csv`; `\"NA\"` means the feature is absent). "
             "`Id` and `SalePrice` are ignored.")
    st.download_button("Download column template", pd.DataFrame(columns=sch["columns"]).to_csv(index=False), "template.csv", icon=":material/download:")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up, keep_default_na=False, na_values=[""])
        ids = raw["Id"] if "Id" in raw else pd.Series(range(len(raw)))
        try:
            raw = raw[sch["columns"]]
            for c, t in X.dtypes.items():
                if t != object:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            out = P.predict(raw, b)
            out.insert(0, "Id", ids.values)
            n_auto = int((out.route == "AUTO").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Properties scored", f"{len(out):,}")
            m2.metric("AUTO", f"{n_auto:,}", f"{n_auto / len(out):.0%}", delta_color="off")
            m3.metric("ESCALATE", f"{len(out) - n_auto:,}", f"{1 - n_auto / len(out):.0%}", delta_color="off")
            st.dataframe(out.style.format({"point": "${:,.0f}", "lower": "${:,.0f}", "upper": "${:,.0f}", "rel_width": "{:.1%}"}), use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", icon=":material/download:")
        except Exception as e:  # surface schema problems to the user instead of a stack trace
            st.error(f"Could not score the file: {e}")

# ------------------------------------------------------------------ evidence
with tab_ev:
    reg = registry()
    v = lambda cid: E.value(reg, cid)
    st.markdown("##### How good are these estimates?")
    st.caption("Measured on houses the model had not seen (nested cross-validation and a train-on-earlier-years test). The deployed model is fitted on all "
               "houses, so it has no hold-out score of its own. Values are read from the project's evidence registry.")
    a, c, d_, e = st.columns(4)
    a.metric("Typical error", f"{v('ACC-GBR-MAPE'):.1f}%", help="Mean absolute percentage error, Gradient Boosting, 15 outer folds")
    c.metric("Typical error in dollars", usd(v("ACC-GBR-MAE")), help="Mean absolute error")
    d_.metric("Range coverage", f"{v('INT-CQR-COVERAGE'):.1%}", help="Share of true prices inside the 80% range (nominal 80%)")
    e.metric("Average range width", usd(v("INT-CQR-MEANWIDTH")), help="Always read together with coverage")

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**Do the 80% ranges really contain 80% of prices?**")
            st.plotly_chart(coverage_chart([
                ("Uncorrected quantile model", v("INT-RAW-COVERAGE")),
                ("With conformal correction (used here)", v("INT-CQR-COVERAGE")),
                ("Conformal, tested on later years (2009–10)", v("INT-TEMP-CQR-COVERAGE"))]), use_container_width=True, config={"displayModeBar": False})
            st.caption("The correction brings coverage to the target on average. When the market moves to later years, coverage drops below target.")
    with c2:
        with st.container(border=True):
            st.markdown("**Read these limits before relying on a number**")
            st.markdown(
                "- Coverage is an **average** over houses: no guarantee for one neighbourhood, price band or property.\n"
                "- AUTO ranges cover slightly **less than 80%** in testing.\n"
                "- The AUTO threshold comes from a **hypothetical** objective; the proposal gives no numeric target.\n"
                "- Luxury, very high or low quality, old and large houses are escalated often.\n"
                "- The price breakdown describes how the ElasticNet uses attributes, **not causes** of price, and not a land value.\n"
                "- Sale-context variables (sale type, month, year) are not used as predictors.")

    with st.expander("Evidence registry rows behind these numbers"):
        ids_ = ["ACC-GBR-MAPE", "ACC-GBR-MAE", "INT-RAW-COVERAGE", "INT-CQR-COVERAGE", "INT-CQR-MEANWIDTH", "INT-TEMP-CQR-COVERAGE"]
        st.dataframe(reg[reg.claim_id.isin(ids_)][["claim_id", "metric", "value", "units", "evaluation_protocol", "limitations"]], use_container_width=True)
