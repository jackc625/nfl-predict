"""Plan 16 test stubs for the shared pure-function metric helpers.

These tests target the module Plan 16-02 will create at
``api/insights_metrics.py``. Full assertion bodies are written now; tests are
gated by ``@pytest.mark.skip(reason="unblocked by plan 16-02")`` until the
module lands. Plan 16-02's verify step removes the skip markers.

Why skip instead of xfail (per REVIEWS.md Codex MEDIUM #10): xfail silently
passes if the implementation raises the wrong exception. Skip makes the
"not yet active" state explicit. xfail is reserved for negative contracts.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# compute_wp_market_prob (WP market probability with devig)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_wp_market_prob_devig_both_sides_present() -> None:
    """Both moneylines present => devig both sides so home + away = 1."""
    from api.insights_metrics import compute_wp_market_prob

    result = compute_wp_market_prob(ml_home=-150, ml_away=130)
    # Raw American odds conversions:
    #   -150 -> 150/(150+100) = 0.60
    #   +130 -> 100/(130+100) = 100/230 ≈ 0.434782...
    # Devigged home = 0.60 / (0.60 + 100/230)
    home_raw = 150.0 / 250.0
    away_raw = 100.0 / 230.0
    expected_prob = home_raw / (home_raw + away_raw)
    assert isinstance(result, dict)
    assert result["devig_method"] == "standard"
    assert result["prob"] == pytest.approx(expected_prob, abs=1e-6)


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_wp_market_prob_missing_home_side() -> None:
    """Only away side present => return the implied (non-devigged) probability
    with ``devig_method == "implied"`` tag per Plan 16-02 contract."""
    from api.insights_metrics import compute_wp_market_prob

    result = compute_wp_market_prob(ml_home=None, ml_away=130)
    assert isinstance(result, dict)
    assert result["devig_method"] == "implied"
    # Home implied from away side only: 1 - away_raw.
    away_raw = 100.0 / 230.0
    expected = 1.0 - away_raw
    assert result["prob"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_wp_market_prob_missing_both_sides() -> None:
    """Both sides null => return None so caller can skip the row."""
    from api.insights_metrics import compute_wp_market_prob

    assert compute_wp_market_prob(ml_home=None, ml_away=None) is None


# ---------------------------------------------------------------------------
# compute_log_loss (with epsilon clipping)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_log_loss_clipping_at_zero() -> None:
    """p=0.0 with y=1 must be finite (clipped at epsilon=1e-15) and equal
    ``-log(1e-15)`` to 6 decimals."""
    from api.insights_metrics import compute_log_loss

    loss = compute_log_loss(y_true=[1], y_prob=[0.0])
    assert math.isfinite(loss)
    assert loss == pytest.approx(-math.log(1e-15), abs=1e-6)


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_log_loss_clipping_at_one() -> None:
    """Mirror: p=1.0 with y=0 must clip at 1 - 1e-15."""
    from api.insights_metrics import compute_log_loss

    loss = compute_log_loss(y_true=[0], y_prob=[1.0])
    assert math.isfinite(loss)
    assert loss == pytest.approx(-math.log(1e-15), abs=1e-6)


# ---------------------------------------------------------------------------
# compute_brier (standard Brier score)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_brier_formula() -> None:
    """Brier = mean((y_prob - y_true)^2)."""
    from api.insights_metrics import compute_brier

    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.2, 0.6, 0.4]
    expected = sum((p - y) ** 2 for p, y in zip(y_prob, y_true, strict=True)) / 4
    assert compute_brier(y_true=y_true, y_prob=y_prob) == pytest.approx(
        expected,
        abs=1e-9,
    )


# ---------------------------------------------------------------------------
# compute_accuracy (threshold at p=0.5, inclusive)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_accuracy_threshold_half() -> None:
    """Threshold is ``p >= 0.5`` (inclusive). For y=[0,1,1] and p=[0.49,0.5,0.51],
    predictions are [0,1,1] so accuracy = 3/3 = 1.0."""
    from api.insights_metrics import compute_accuracy

    assert compute_accuracy(
        y_true=[0, 1, 1], y_prob=[0.49, 0.50, 0.51]
    ) == pytest.approx(1.0)
    # Sanity: p=0.5 with y=0 should be wrong (predicted 1, actual 0).
    assert compute_accuracy(y_true=[0], y_prob=[0.5]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ATS MAE with market sign convention
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_ats_mae_sign_convention() -> None:
    """Model ATS prediction is home margin; market prediction is
    ``-market_spread`` (market_spread < 0 means home favored)."""
    from api.insights_metrics import compute_ats_mae_model_and_market

    # 3 games: actual home margin = [3, -5, 7], market_spread = [-3, 5, -7]
    # => market prediction = [+3, -5, +7] identical to actual => market MAE = 0.
    # Model prediction = [2, -4, 8] => model MAE = mean(|[1, -1, 1]|) = 1.0
    rows = [
        {"model_prob": 2.0, "actual": 3.0, "market_spread": -3.0},
        {"model_prob": -4.0, "actual": -5.0, "market_spread": 5.0},
        {"model_prob": 8.0, "actual": 7.0, "market_spread": -7.0},
    ]
    model_mae, market_mae = compute_ats_mae_model_and_market(rows)
    assert model_mae == pytest.approx(1.0, abs=1e-9)
    assert market_mae == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# OU MAE straightforward
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_compute_ou_mae() -> None:
    from api.insights_metrics import compute_ou_mae_model_and_market

    rows = [
        {"model_prob": 45.0, "actual": 47.0, "market_total": 46.0},
        {"model_prob": 50.0, "actual": 48.0, "market_total": 49.0},
    ]
    model_mae, market_mae = compute_ou_mae_model_and_market(rows)
    # Model residuals: |45-47|=2, |50-48|=2 => MAE = 2.0
    # Market residuals: |46-47|=1, |49-48|=1 => MAE = 1.0
    assert model_mae == pytest.approx(2.0, abs=1e-9)
    assert market_mae == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Bin helpers — ATS (left-closed/right-open, symmetric, with overflow bins)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_bin_ats_residuals_semantics() -> None:
    """Bins are left-closed/right-open at boundaries ``-21, -19, -17, ..., 19, 21``;
    values below -21 map to ``<=-21`` and values >= 21 map to ``>=21``; empty
    bins are retained in output (key present with count 0)."""
    from api.insights_metrics import bin_ats_residuals

    # Provide residuals that hit both overflow bins plus a couple of normal bins.
    residuals = [-30.0, -5.0, 0.0, 5.0, 30.0]
    bins = bin_ats_residuals(residuals)
    assert isinstance(bins, dict)
    # Overflow bin keys must be present.
    assert "<=-21" in bins
    assert ">=21" in bins
    assert bins["<=-21"] == 1
    assert bins[">=21"] == 1
    # Empty bins retained: one arbitrary bin that no residual landed in should
    # appear in the output with count 0 (e.g., the "[-11, -9)" region).
    # Check at least one zero-count key exists so the retained-empty-bin contract
    # is visible.
    assert 0 in bins.values() or len(bins) >= 20


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_bin_ou_residuals_fixed_bins() -> None:
    """OU bins fixed 3-point width across 35-65 with ``<=35`` and ``>=65``
    overflow bins."""
    from api.insights_metrics import bin_ou_values

    values = [30.0, 36.0, 45.0, 64.0, 70.0]
    bins = bin_ou_values(values)
    assert isinstance(bins, dict)
    assert "<=35" in bins
    assert ">=65" in bins
    assert bins["<=35"] == 1
    assert bins[">=65"] == 1


# ---------------------------------------------------------------------------
# American odds -> probability helper
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="unblocked by plan 16-02")
def test_american_odds_formula() -> None:
    """Sanity check: -150 -> 0.60, +130 -> 100/230 ≈ 0.4347826..."""
    from api.insights_metrics import american_odds_to_prob

    assert american_odds_to_prob(-150) == pytest.approx(0.60, abs=1e-4)
    assert american_odds_to_prob(130) == pytest.approx(100.0 / 230.0, abs=1e-4)


# Sanity guard: file must import and collect cleanly even before Plan 16-02.
def test_module_collects() -> None:
    assert True
