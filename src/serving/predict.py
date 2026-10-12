"""Inference on the exported bundle: point, interval, route and group decomposition for raw property rows.

All outputs use only attribute values (no label). Point = median (expm1 of the log1p prediction). The interval is split CQR (marginal 80%
coverage, no per-property guarantee). Route uses src.evaluation.routing. The decomposition explains the ElasticNet, not the GBR point model.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation import intervals as I
from src.evaluation.decomposition import decompose, land_vs_building
from src.evaluation.routing import assign_route, relative_width

BUNDLE = Path(__file__).resolve().parents[2] / "artifacts" / "serving" / "bundle_v1.joblib"


@lru_cache(maxsize=2)
def load_bundle(path=BUNDLE):
    return joblib.load(path)


def validate(rows, bundle):
    sch = bundle["meta"]["schema"]
    missing = [c for c in sch["columns"] if c not in rows.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return rows[sch["columns"]]


def predict(rows, bundle=None, tau=None):
    """DataFrame of raw predictor rows -> DataFrame(point, lower, upper, rel_width, route, q_hat)."""
    b = bundle or load_bundle()
    rows = validate(rows, b)
    pz = b["point"]["prep"].transform(rows).values
    point_log = b["point"]["model"].predict(pz)
    q = b["quantile"]
    qz = q["prep"].transform(rows).values
    L, U, _ = I.rearrange(q["lo"].predict(qz), q["hi"].predict(qz))
    Lq, Uq = I.cqr_interval(L, U, q["q_hat"])
    point, lo, hi = np.expm1(point_log), np.expm1(Lq), np.expm1(Uq)
    w = relative_width(lo, hi, point)
    return pd.DataFrame(dict(point=point, lower=lo, upper=hi, rel_width=w, route=assign_route(w, b["tau"] if tau is None else tau)),
                        index=rows.index)


def explain(rows, bundle=None):
    """Long group decomposition of the ElasticNet prediction (+ land-vs-building summary)."""
    b = bundle or load_bundle()
    rows = validate(rows, b)
    dec = decompose(rows, bundle=b["linear"], reference_rows=b["reference_rows"])
    return dec, land_vs_building(dec)
