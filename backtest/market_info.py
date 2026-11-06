"""Does the model know anything the market line does not?

Two regressions answer it. Each fits the model's departure from the market
and reports whether that departure carries signal about the result:

- ``information_beyond_line``: OLS of (result - line) on an intercept and
  (model - line), for spreads and totals.
- ``information_beyond_probability``: logistic regression of a home win with
  the market log-odds as a fixed offset, an intercept, and a slope on
  (model log-odds - market log-odds).

The slope is the quantity of interest. The test is one-sided (H1: slope > 0).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm

_PROB_CLIP = 1e-6
_MIN_ROWS = 3
_MAX_ITER = 100
_TOL = 1e-10
_PERFECT_FIT_RTOL = 1e-20


@dataclass(frozen=True)
class InfoResult:
    """Slope on the model-minus-market term, its standard error and p-value."""

    coef: float
    se: float
    p_one_sided: float
    n: int


def _usable(*arrays: np.ndarray) -> list[np.ndarray]:
    """Validate shapes, drop rows with any NaN, and return float arrays."""
    cols = [np.asarray(a, dtype=float) for a in arrays]
    if any(c.ndim != 1 for c in cols):
        raise ValueError("inputs must be 1-D arrays")
    if len({len(c) for c in cols}) != 1:
        raise ValueError("inputs must have equal length")
    keep = ~np.any(np.isnan(np.column_stack(cols)), axis=1)
    cols = [c[keep] for c in cols]
    if len(cols[0]) < _MIN_ROWS:
        raise ValueError(f"need at least {_MIN_ROWS} usable rows, got {len(cols[0])}")
    return cols


def _result(coef: float, se: float, n: int) -> InfoResult:
    if not (np.isfinite(se) and se > 0.0):
        raise ValueError(
            f"degenerate fit: standard error {se!r} is not finite and positive, "
            "so no significance can be reported"
        )
    p = float(norm.sf(coef / se))  # = 1 - cdf, without underflow to 0
    return InfoResult(coef=float(coef), se=float(se), p_one_sided=p, n=n)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _PROB_CLIP, 1.0 - _PROB_CLIP)
    return np.log(p / (1.0 - p))


def information_beyond_line(
    outcome: np.ndarray, line: np.ndarray, model: np.ndarray
) -> InfoResult:
    """OLS of ``outcome - line`` on an intercept and ``model - line``.

    Standard errors are classical: ``sigma^2 (X'X)^-1`` with
    ``sigma^2 = RSS / (n - 2)``.
    """
    y_raw, line_c, model_c = _usable(outcome, line, model)
    y = y_raw - line_c
    x = np.column_stack([np.ones(len(y)), model_c - line_c])
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        rss = float(resid @ resid)
        if rss <= _PERFECT_FIT_RTOL * float(y @ y):
            raise ValueError("degenerate fit: residuals are numerically zero")
        sigma2 = rss / (len(y) - 2)
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(x.T @ x)))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"singular design in line regression: {exc}") from exc
    return _result(beta[1], se[1], len(y))


def information_beyond_probability(
    home_win: np.ndarray, market_prob: np.ndarray, model_prob: np.ndarray
) -> InfoResult:
    """Logistic regression with offset ``logit(market_prob)``.

    Free parameters are an intercept and the slope on
    ``logit(model_prob) - logit(market_prob)``. Fitted by Newton-Raphson;
    standard errors come from the inverse Fisher information at the solution.
    """
    y, market, model = _usable(home_win, market_prob, model_prob)
    offset = _logit(market)
    x = np.column_stack([np.ones(len(y)), _logit(model) - offset])
    beta = np.zeros(2)
    try:
        for _ in range(_MAX_ITER):
            p = expit(offset + x @ beta)
            info = x.T @ (x * (p * (1.0 - p))[:, None])
            step = np.linalg.solve(info, x.T @ (y - p))
            beta = beta + step
            if np.max(np.abs(step)) < _TOL:
                break
        else:
            raise ValueError("logistic fit did not converge")
        p = expit(offset + x @ beta)
        info = x.T @ (x * (p * (1.0 - p))[:, None])
        se = np.sqrt(np.diag(np.linalg.inv(info)))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"singular design in logistic fit: {exc}") from exc
    return _result(beta[1], se[1], len(y))
