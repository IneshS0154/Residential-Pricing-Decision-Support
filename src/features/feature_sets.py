"""Configuration-driven feature sets and correlated-group ablation variants.

Three named strategies (all exclude identifiers, the target and the sale-context variables; that is the `guard`
step of `build_preprocessor`, which these configurations never touch):

  full             every eligible predictor: nothing pruned, low-information columns kept, GarageAge added
  reduced          the project default (`build_preprocessor` defaults): LOW_INFORMATION and REDUNDANT pruned
  domain_informed  keeps the economically distinct components, prunes only exact composites and near-constants

The domain_informed rule was written down BEFORE any ablation result was computed (same discipline as
src/evaluation/decision_experiments.py). Rationale, per attribute:
  * above-grade living area and basement area are valued differently per sq ft, so GrLivArea, TotalBsmtSF and
    2ndFlrSF are kept and the composite TotalSF (an equal-weight sum of the two) and the duplicate 1stFlrSF are pruned;
  * finished and unfinished basement are valued differently, so BsmtFinSF1/2 are kept with the total and BsmtUnfSF
    (implied by the others) is pruned; they never appear together with all of their sum;
  * garage capacity (GarageCars) and size (GarageArea) are physically distinct, so both are kept;
  * room count (TotRmsAbvGrd) describes layout, not size, so it is kept next to GrLivArea;
  * above-grade and basement bathrooms are valued differently, so the four counts are kept and the composite
    TotalBathrooms is pruned.
No matrix of numeric inputs built from this set contains an exact linear dependency (checked in tests).

`group_variant_drop` builds the single-group ablations: everything outside the group keeps its REFERENCE
(`reduced`) status, only the chosen group's columns change.
"""
from dataclasses import dataclass, field

from . import preprocessing as P

LOW_INFO = tuple(P.LOW_INFORMATION)
REDUCED_DROP = tuple(sorted(set(P.LOW_INFORMATION) | set(P.REDUNDANT)))
DOMAIN_DROP = tuple(sorted(set(P.LOW_INFORMATION) | {"TotalSF", "TotalBathrooms", "1stFlrSF", "BsmtUnfSF"}))


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    drop_columns: tuple
    year_features: str = "age"
    garage_age: bool = False

    def build(self, strategy="baseline", scale="robust", **kw):
        """Unfitted preprocessing Pipeline for this feature set (extra kwargs go to build_preprocessor)."""
        return P.build_preprocessor(strategy, scale=scale, drop_columns=self.drop_columns,
                                    year_features=self.year_features, garage_age=self.garage_age, **kw)


FEATURE_SETS = {
    "full": FeatureSet("full", "all eligible predictors; nothing pruned; GarageAge added", (), garage_age=True),
    "reduced": FeatureSet("reduced", "project default: near-constant and redundant columns pruned", REDUCED_DROP),
    "domain_informed": FeatureSet("domain_informed", "distinct economic components kept; exact composites pruned",
                                  DOMAIN_DROP),
}


def get_feature_set(name):
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise ValueError(f"feature set must be one of {tuple(FEATURE_SETS)}") from None


# ---------------------------------------------------------------- correlated groups (ablation design)
# Each group: the columns whose presence varies, and named variants = columns KEPT (the rest of the group is pruned).
# Columns not in the group keep their `reduced` status. Variants use engineered names where relevant.
GROUPS = {
    "A_garage": {
        "columns": ("GarageCars", "GarageArea"),
        "variants": {"GarageCars only (reference)": ("GarageCars",), "GarageArea only": ("GarageArea",),
                     "both": ("GarageCars", "GarageArea"), "neither": ()},
        "reference": "GarageCars only (reference)",
    },
    "B_living": {
        "columns": ("GrLivArea", "TotRmsAbvGrd"),
        "variants": {"GrLivArea only (reference)": ("GrLivArea",), "TotRmsAbvGrd only": ("TotRmsAbvGrd",),
                     "both": ("GrLivArea", "TotRmsAbvGrd"), "neither": ()},
        "reference": "GrLivArea only (reference)",
    },
    "C_age": {   # handled through year_features / garage_age rather than column pruning
        "columns": ("HouseAge", "RemodAge", "YearBuilt", "YearRemodAdd", "GarageAge"),
        "variants": {"engineered age (reference)": dict(year_features="age", garage_age=False),
                     "raw years": dict(year_features="raw", garage_age=False),
                     "engineered age + GarageAge": dict(year_features="age", garage_age=True),
                     "engineered age, no RemodAge": dict(year_features="age", garage_age=False, extra_drop=("RemodAge",))},
        "reference": "engineered age (reference)",
    },
    "D_floor_area": {
        "columns": ("TotalBsmtSF", "1stFlrSF", "TotalSF"),
        "variants": {"TotalBsmtSF only (reference)": ("TotalBsmtSF",), "1stFlrSF only": ("1stFlrSF",),
                     "both": ("TotalBsmtSF", "1stFlrSF"), "neither": (), "TotalSF only": ("TotalSF",)},
        "reference": "TotalBsmtSF only (reference)",
    },
    "E_basement": {
        "columns": ("TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"),
        "variants": {"total only (reference)": ("TotalBsmtSF",),
                     "components only": ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"),
                     "total + finished parts": ("TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2"),
                     "total + all parts (exact sum)": ("TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF")},
        "reference": "total only (reference)",
    },
    "F_bathrooms": {
        "columns": ("TotalBathrooms", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"),
        "variants": {"TotalBathrooms only (reference)": ("TotalBathrooms",),
                     "four components only": ("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"),
                     "total + components (exact sum)": ("TotalBathrooms", "FullBath", "HalfBath", "BsmtFullBath",
                                                        "BsmtHalfBath"),
                     "above-grade baths only": ("FullBath", "HalfBath"),
                     "none": ()},
        "reference": "TotalBathrooms only (reference)",
    },
}


def group_variant(group, variant):
    """FeatureSet for one ablation cell: reference (reduced) everywhere except the group, which follows `variant`."""
    spec = GROUPS[group]
    v = spec["variants"][variant]
    ref_drop = set(REDUCED_DROP)
    if isinstance(v, dict):                                    # group C: representation switch
        drop = ref_drop | set(v.get("extra_drop", ()))
        return FeatureSet(f"{group}:{variant}", variant, tuple(sorted(drop)),
                          year_features=v["year_features"], garage_age=v["garage_age"])
    cols = set(spec["columns"])
    drop = (ref_drop - cols) | (cols - set(v))
    return FeatureSet(f"{group}:{variant}", variant, tuple(sorted(drop)))


def all_group_variants():
    """[(group, variant, FeatureSet)] for every ablation cell; reference cells of different groups are identical."""
    return [(g, v, group_variant(g, v)) for g, spec in GROUPS.items() for v in spec["variants"]]
