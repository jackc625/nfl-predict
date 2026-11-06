"""The fixed v2 recipe: which inputs each target reads and which estimator maps them to a value.

The inputs are chosen once, by hand, and never re-picked inside a walk-forward step. Every
estimator is deterministic (lbfgs logistic regression, closed-form ridge), so a prediction is
reproducible from the same input snapshot. ASCII only.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "V2_INPUTS",
    "V2_LABEL",
    "V2_RECIPE_NAME",
    "V2_TARGETS",
    "V2_TRAIN_START",
    "build_v2_estimator",
]

V2_RECIPE_NAME: str = "recipe-2026-v2"

#: The first season a v2 model trains on.
V2_TRAIN_START: int = 2002

V2_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

#: The gold label column each target is fitted to.
V2_LABEL: dict[str, str] = {
    "wp": "home_win",
    "ats": "home_margin",
    "ou": "total_points",
}

_SIDE_INPUTS: tuple[str, ...] = (
    "elo_diff",
    "success_rate_net",
    "opp_adj_epa_net",
    "qb_value_diff",
    "mkt_rating_diff",
    "hfa_used",
)

V2_INPUTS: dict[str, tuple[str, ...]] = {
    "wp": _SIDE_INPUTS,
    "ats": _SIDE_INPUTS,
    "ou": (
        "league_points_level",
        "mkt_total_sum",
        "off_epa_sum",
        "def_epa_sum",
        "pace_sum",
        "wind_mph",
    ),
}


def build_v2_estimator(target: str) -> Pipeline:
    """A fresh, unfitted median-impute -> standardise -> linear model for *target*."""
    if target not in V2_INPUTS:
        msg = f"unknown target {target!r}; expected one of {list(V2_TARGETS)}"
        raise ValueError(msg)
    model = (
        LogisticRegression(C=1.0, max_iter=1000) if target == "wp" else Ridge(alpha=1.0)
    )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
