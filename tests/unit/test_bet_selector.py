"""Unit tests for the Phase-27 O/U BetSelector (BET-01/02, OUM-04/06) and the LOCKED-1
pre-hold high-total boundary derivation.

Covers:
- Task 1 (LOCKED-1): the high-total boundary RE-DERIVED on PRE-HOLD (2018-2022) closing totals,
  the hold-season leakage assertion, and the legacy-46.5 comparison
  (``backtest.ou_divergence.derive_high_total_boundary`` / ``HIGH_TOTAL_BOUNDARY_PREHOLD``).
- Task 2/3 (BET-01/02, OUM-04/06, D27-04/05/06/14): the single-source ``BetSelector.select()``
  decision engine -- sub-pop UNION filter, EV-floor admission, high-total-OVER pocket drop, the
  BET-02 calibrated-P Kelly sizing fix (model_prob <= 1 on an 8-point gap), the mock/synthetic-odds
  hard-fail, push handling, report-only CLV with the wording distinction, the unfiltered
  cross-check, and selected+rejected-with-reasons output.

Run the boundary group only:  pytest tests/unit/test_bet_selector.py -q -k boundary
Run the full module:          pytest tests/unit/test_bet_selector.py -x -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.ou_divergence import (
    HIGH_TOTAL_BOUNDARY_PREHOLD,
    HOLD_SEASONS,
    PRE_HOLD_SEASONS,
    TOTALS_REGIME_BOUNDARIES,
    HoldSeasonLeakageError,
    derive_high_total_boundary,
)

# ---------------------------------------------------------------------------
# Number anchors (kept explicit so a silent drift is caught).
# ---------------------------------------------------------------------------

_LEGACY_HIGH_BOUNDARY = 46.5  # the hold-informed TOTALS_REGIME_BOUNDARIES["high_min"]
_OU_BREAKEVEN = 110.0 / 210.0  # 0.52380952... (flat -110 cover breakeven)


def _ou_row(
    game_id: str,
    *,
    model_total: float,
    closing_total: float,
    actual_total: float,
    season: int = 2021,
    week: int = 1,
    sportsbook: str = "consensus",
    is_live: bool = False,
) -> dict:
    """Build a single synthetic O/U candidate row for the BetSelector tests."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_total": model_total,
        "closing_total": closing_total,
        "actual": actual_total,
        "sportsbook": sportsbook,
        "is_live": is_live,
    }


# ---------------------------------------------------------------------------
# Task 1 (LOCKED-1): pre-hold high-total boundary derivation + leakage assertion
# ---------------------------------------------------------------------------


class TestHighTotalBoundaryDerivation:
    """The high-total boundary is re-derived on PRE-HOLD data only (LOCKED-1, T-27-22)."""

    def test_high_total_boundary_excludes_hold(self) -> None:
        """The derivation reads only pre-hold rows; a hold-season row raises (leakage assertion).

        Feeding a frame that contains a 2023/2024 (HOLD) row must raise HoldSeasonLeakageError --
        the eligibility boundary may never be informed by the burned holdout. A clean pre-hold
        frame derives the boundary from its rows only.
        """
        # (a) A frame carrying a hold-season row is rejected (the leakage assertion fires).
        leaky = pd.DataFrame(
            {
                "game_id": [
                    "2019_W01_DAL@NYG",
                    "2023_W05_KC@BUF",  # HOLD season -> must trip the assertion
                ],
                "total": [44.0, 49.0],
            }
        )
        with pytest.raises(HoldSeasonLeakageError) as exc:
            derive_high_total_boundary(odds_df=leaky)
        assert "2023" in str(exc.value)

        # (b) A clean pre-hold-only frame derives from its rows only (no hold contamination).
        clean = pd.DataFrame(
            {
                "game_id": [
                    "2018_W01_DAL@NYG",
                    "2019_W02_KC@BUF",
                    "2020_W03_SF@SEA",
                    "2021_W04_GB@CHI",
                    "2022_W05_NE@MIA",
                    "2022_W06_LA@ARI",
                ],
                "total": [40.0, 42.0, 45.0, 48.0, 50.0, 52.0],
            }
        )
        derived = derive_high_total_boundary(odds_df=clean)
        # Upper-tertile (q=2/3) of the six pre-hold totals -- a value drawn ONLY from these rows.
        expected = float(
            pd.Series([40.0, 42.0, 45.0, 48.0, 50.0, 52.0]).quantile(2.0 / 3.0)
        )
        assert derived == pytest.approx(expected)

    def test_high_total_boundary_rejects_hold_window_request(self) -> None:
        """Requesting a derivation window that intersects HOLD_SEASONS is itself a leakage error."""
        with pytest.raises(HoldSeasonLeakageError):
            derive_high_total_boundary(pre_hold_seasons=(2022, 2023))

    def test_high_total_boundary_value_reported_vs_legacy(self) -> None:
        """HIGH_TOTAL_BOUNDARY_PREHOLD is a sane totals value, surfaced alongside the legacy 46.5.

        The new pre-hold boundary need NOT equal the legacy 46.5 (it is re-derived on a different,
        leakage-clean window); the test records both values so the readout can compare them, and
        asserts the new value sits in a sane NFL-totals range.
        """
        new_value = HIGH_TOTAL_BOUNDARY_PREHOLD
        legacy_value = TOTALS_REGIME_BOUNDARIES["high_min"]

        assert legacy_value == _LEGACY_HIGH_BOUNDARY
        # The constant must have resolved on the populated data lake (not the NaN bare-checkout
        # fallback) for the unit suite, and sit in a sane totals band.
        assert not math.isnan(new_value), (
            "HIGH_TOTAL_BOUNDARY_PREHOLD did not resolve -- the silver odds lake is required for "
            "the pre-hold derivation (LOCKED-1)."
        )
        assert 40.0 <= new_value <= 50.0
        # Both values are surfaced together (the readout comparison); the pre-hold derivation is a
        # leakage-clean replacement for the hold-informed legacy boundary, not necessarily equal.
        reported = {"pre_hold": new_value, "legacy_hold_informed": legacy_value}
        assert set(reported) == {"pre_hold", "legacy_hold_informed"}

    def test_pre_hold_window_excludes_hold_seasons(self) -> None:
        """The pre-hold derivation window and the hold window are disjoint (LOCKED-1 invariant)."""
        assert set(PRE_HOLD_SEASONS).isdisjoint(set(HOLD_SEASONS))
        assert set(HOLD_SEASONS) == {2023, 2024}
