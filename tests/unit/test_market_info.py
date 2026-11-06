"""Tests for the information-beyond-market regressions."""

import numpy as np
import pytest

from backtest.market_info import (
    information_beyond_line,
    information_beyond_probability,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_line_regression_finds_a_planted_slope():
    rng = np.random.default_rng(7)
    line = rng.normal(0, 6, 4000)
    edge = rng.normal(0, 2, 4000)
    outcome = line + 0.3 * edge + rng.normal(0, 13, 4000)
    r = information_beyond_line(outcome, line, line + edge)
    assert r.coef == pytest.approx(0.3, abs=0.12)
    assert r.p_one_sided < 0.05
    assert r.n == 4000


def test_line_regression_sees_nothing_in_noise():
    rng = np.random.default_rng(8)
    line = rng.normal(0, 6, 4000)
    outcome = line + rng.normal(0, 13, 4000)
    r = information_beyond_line(outcome, line, line + rng.normal(0, 2, 4000))
    assert r.p_one_sided > 0.05


def test_probability_regression_finds_a_planted_slope():
    # market logit ~ N(0, 1); model logit = market logit + d; true logit = market + 0.5 d
    rng = np.random.default_rng(11)
    n = 20000
    market_logit = rng.normal(0, 1, n)
    d = rng.normal(0, 1, n)
    market_prob = _sigmoid(market_logit)
    model_prob = _sigmoid(market_logit + d)
    home_win = (rng.random(n) < _sigmoid(market_logit + 0.5 * d)).astype(float)
    r = information_beyond_probability(home_win, market_prob, model_prob)
    assert r.coef == pytest.approx(0.5, abs=0.2)
    assert r.p_one_sided < 0.05
    assert r.n == n


def test_nan_rows_are_dropped_and_counted():
    rng = np.random.default_rng(3)
    line = rng.normal(0, 6, 100)
    outcome = line + rng.normal(0, 13, 100)
    model = line + rng.normal(0, 2, 100)
    model[:5] = np.nan
    assert information_beyond_line(outcome, line, model).n == 95


def test_too_few_rows_or_mismatched_lengths_raise():
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        information_beyond_line(x, x, x)
    with pytest.raises(ValueError):
        information_beyond_line(np.ones(5), np.ones(4), np.ones(5))
