"""Unit tests for ``api.season_metrics`` (the season-tracking pure-math module).

These exercise the LOCKED per-target hit-rate convention (CONTEXT D-05): WP
excludes tie games; ATS/OU use the authoritative ``BettingSimulator`` sign
convention WITH 0.5-pt slippage so the season page reproduces the ``/betting``
numbers (ATS ~51%, NOT the sign-flipped ~78% from ``_compute_week_summary``);
pushes/ties are excluded from every denominator; the cumulative series uses a
count-based denominator so a push/tie never advances it.

Mirrors ``tests/test_betting_metrics.py``: small hand-built fixture rows and
hand-computed oracles (no live DB), so assertions are exact and fast.

Test names include ``ats_convention`` and ``cumulative`` to match the Nyquist
``-k`` selectors from 18-VALIDATION.md.

The per-game row dict keys mirror the ``predictions`` cache table columns the
math reads: ``season, week, status, home_score, away_score, wp_prob,
ats_prediction, ou_prediction, market_spread, market_total``.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixture row builders (hand-built; oracles computed in the asserts)
# ---------------------------------------------------------------------------


def _game(
    *,
    season: int = 2024,
    week: int = 1,
    status: str = "completed",
    home_score: float | None = 24,
    away_score: float | None = 20,
    wp_prob: float | None = 0.60,
    ats_prediction: float | None = -2.0,
    ou_prediction: float | None = 45.0,
    market_spread: float | None = -3.0,
    market_total: float | None = 44.0,
) -> dict:
    """Return a single per-game ``predictions``-shaped row dict with defaults.

    Only the keys the metrics module reads are required; the rest of the cache
    schema is irrelevant to the math and omitted for brevity.
    """
    return {
        "season": season,
        "week": week,
        "status": status,
        "home_score": home_score,
        "away_score": away_score,
        "wp_prob": wp_prob,
        "ats_prediction": ats_prediction,
        "ou_prediction": ou_prediction,
        "market_spread": market_spread,
        "market_total": market_total,
    }


# ---------------------------------------------------------------------------
# ATS sign convention — the headline-number regression (Pitfall 2 / D-05)
# ---------------------------------------------------------------------------


def _naive_ats_hit(ats_prediction: float, market_spread: float, margin: float) -> bool:
    """The ANTI-PATTERN ``_compute_week_summary`` ATS test (sign-flipped).

    Reproduced here ONLY to prove the season module does NOT use it. Source of
    the bug: ``api/routes/pages.py`` reads ``ats_prediction`` as a home margin
    and tests ``(margin > -market_spread) == (ats_prediction > -market_spread)``.
    This yields a spurious ~78% ATS hit-rate.
    """
    return (margin > -market_spread) == (ats_prediction > -market_spread)


def test_ats_convention_single_disagreement_row_returns_simulator_answer() -> None:
    """A row where the naive and simulator conventions DISAGREE.

    Scenario (18-RESEARCH "Critical correctness properties" #2): home underdog
    (market_spread > 0), the model leans home (ats_prediction < market_spread =>
    home_cover side), and the away team blows out (large negative margin). The
    simulator scores this a MISS (home was picked to cover but got crushed); the
    sign-flipped naive convention wrongly scores it a HIT.

    Row: ats_prediction=-6.0, market_spread=+2.0, home 10 / away 31 (margin -21).
      Simulator: side=home_cover, slipped = 2.0 - 0.5 = 1.5; home_covers =
        (-21 > 1.5) = False; home_cover hit = False -> MISS.
      Naive: (margin > -ms)=(-21 > -2.0)=False; (pred > -ms)=(-6.0 > -2.0)=False;
        False == False -> True -> HIT.
    The module MUST return the simulator answer (a miss).
    """
    from api.season_metrics import _ats_outcome

    row = _game(
        season=2021,
        week=1,
        ats_prediction=-6.0,
        market_spread=2.0,
        home_score=10,
        away_score=31,
    )
    # Module (simulator convention) -> MISS (False).
    assert _ats_outcome(row) is False
    # The naive convention would have called it a HIT -> proves they disagree.
    assert _naive_ats_hit(-6.0, 2.0, -21) is True


def test_ats_convention_aggregate_is_about_51_not_78() -> None:
    """The aggregate season ATS hit-rate reproduces the /betting ~51.4%, NOT ~78%.

    Build a controlled fixture of decided ATS games with exactly 18 hits and 17
    misses (35 decided) => 51.43%, within +-2 points of the betting_bets-derived
    51.4%. A ~78% result (the sign-flipped bug) would fail ``ats_rate < 60``.

    Every row uses a clean home_cover/away_cover decision with no push (the
    actual margin never lands on the slipped line), so all 35 are decided.
    """
    from api.season_metrics import compute_season_kpis

    rows: list[dict] = []
    week = 1
    # 18 hits: model picks home_cover (pred < ms) and home covers the slipped line.
    #   ms=-3.0 -> slipped=-3.5; margin=+10 (> -3.5) -> home_covers -> HIT.
    for _ in range(18):
        rows.append(
            _game(
                season=2021,
                week=week,
                ats_prediction=-6.0,  # < market_spread (-3.0) => home_cover
                market_spread=-3.0,
                home_score=24,
                away_score=14,  # margin +10 > slipped -3.5 => home covers => HIT
            )
        )
        week += 1
    # 17 misses: model picks home_cover but home fails to cover the slipped line.
    #   ms=-3.0 -> slipped=-3.5; margin=-10 (< -3.5) -> home does NOT cover -> MISS.
    for _ in range(17):
        rows.append(
            _game(
                season=2021,
                week=week,
                ats_prediction=-6.0,  # < market_spread (-3.0) => home_cover
                market_spread=-3.0,
                home_score=14,
                away_score=24,  # margin -10 < slipped -3.5 => home fails => MISS
            )
        )
        week += 1

    kpis = compute_season_kpis(rows)
    ats_rate = kpis["ats_hit_rate"]

    # Honest, betting-consistent number — NOT the ~78% sign-flipped artifact.
    assert ats_rate == pytest.approx(18 / 35 * 100, abs=1e-6)
    assert abs(ats_rate - 51.4) <= 2.0  # within +-2 of the /betting 51.4%
    assert ats_rate < 60  # a ~78% result (the bug) would FAIL here
    assert kpis["ats_decided"] == 35  # all rows decided (no pushes)


def test_ats_convention_no_directional_pick_is_excluded() -> None:
    """ats_prediction == market_spread => no side => excluded from ATS denom."""
    from api.season_metrics import _ats_outcome, compute_season_kpis

    row = _game(ats_prediction=-3.0, market_spread=-3.0)
    assert _ats_outcome(row) is None

    kpis = compute_season_kpis([row])
    assert kpis["ats_decided"] == 0
    assert kpis["ats_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# Cumulative running hit-rate — count-based denominator (DASH-07 / D-07)
# ---------------------------------------------------------------------------


def test_cumulative_uses_count_based_denominator_push_does_not_advance() -> None:
    """The cumulative WP series accumulates hits/decided counts.

    A tie (push) row does NOT advance the denominator: the running series only
    has a point per decided game. Five WP games ordered by week, week 3 a tie:

      w1 wp=0.70 28-20 (margin +8)  -> predicted home, home won -> HIT
      w2 wp=0.40 17-24 (margin -7)  -> predicted away, away won -> HIT
      w3 wp=0.60 21-21 (margin  0)  -> TIE -> push (EXCLUDED, no advance)
      w4 wp=0.65 14-30 (margin -16) -> predicted home, away won -> MISS
      w5 wp=0.55 27-13 (margin +14) -> predicted home, home won -> HIT

    Decided-only running hit-rate: [100.0, 100.0, 66.667, 75.0] (4 points; the
    tie produced NO point and did not move the denominator).
    """
    from api.season_metrics import compute_cumulative_series

    rows = [
        _game(season=2024, week=1, wp_prob=0.70, home_score=28, away_score=20),
        _game(season=2024, week=2, wp_prob=0.40, home_score=17, away_score=24),
        _game(season=2024, week=3, wp_prob=0.60, home_score=21, away_score=21),  # tie
        _game(season=2024, week=4, wp_prob=0.65, home_score=14, away_score=30),
        _game(season=2024, week=5, wp_prob=0.55, home_score=27, away_score=13),
    ]
    series = compute_cumulative_series(rows)
    wp_values = series["wp"]["values"]
    wp_weeks = series["wp"]["weeks"]

    # Four decided points; the week-3 tie is absent (it never advanced the denom).
    assert wp_values == pytest.approx([100.0, 100.0, 200.0 / 3, 75.0], abs=1e-6)
    assert wp_weeks == [1, 2, 4, 5]
    # The final cumulative value equals the season KPI (3 hits / 4 decided).
    assert wp_values[-1] == pytest.approx(75.0, abs=1e-6)


def test_cumulative_empty_rows_returns_empty_series() -> None:
    """Empty input yields empty per-target series without raising."""
    from api.season_metrics import compute_cumulative_series

    series = compute_cumulative_series([])
    for target in ("wp", "ats", "ou"):
        assert series[target]["weeks"] == []
        assert series[target]["values"] == []


# ---------------------------------------------------------------------------
# WP tie exclusion (Pitfall 3 / D-05)
# ---------------------------------------------------------------------------


def test_wp_tie_excluded_from_denominator() -> None:
    """A margin==0 tie row is excluded from the WP denominator.

    Three completed games, one a tie: WP denominator == 2 (completed minus the
    tie), not 3. Both decided games are correct calls -> 100% over 2 decided.
    """
    from api.season_metrics import _wp_outcome, compute_season_kpis

    tie = _game(season=2021, week=1, wp_prob=0.60, home_score=21, away_score=21)
    assert _wp_outcome(tie) is None  # tie -> push, excluded

    rows = [
        _game(season=2021, week=1, wp_prob=0.70, home_score=28, away_score=20),  # HIT
        _game(season=2021, week=2, wp_prob=0.30, home_score=10, away_score=27),  # HIT
        tie,  # excluded
    ]
    kpis = compute_season_kpis(rows)
    assert kpis["wp_decided"] == 2  # the tie is NOT in the denominator
    assert kpis["wp_hits"] == 2
    assert kpis["wp_hit_rate"] == pytest.approx(100.0)


def test_wp_known_correct_row_is_hit() -> None:
    """A clearly-correct WP call scores hit==True."""
    from api.season_metrics import _wp_outcome

    # wp_prob 0.62 (lean home), home wins 27-20 -> hit.
    assert _wp_outcome(_game(wp_prob=0.62, home_score=27, away_score=20)) is True
    # wp_prob 0.38 (lean away), away wins 20-24 -> hit.
    assert _wp_outcome(_game(wp_prob=0.38, home_score=20, away_score=24)) is True


# ---------------------------------------------------------------------------
# ATS push and OU push (exact-line landing excluded)
# ---------------------------------------------------------------------------


def test_ats_push_on_slipped_line_excluded() -> None:
    """A row whose actual margin lands exactly on the slipped line is excluded.

    ats_prediction=-6.0 < market_spread=-3.0 => home_cover, slipped=-3.5.
    Set the margin to exactly -3.5 (home loses by 3.5 is impossible with ints,
    so use a half-point via scores 20.5/24 is not valid; instead use a market
    spread that produces an integer slipped line). Use market_spread=+2.0 =>
    home_cover slipped = 1.5 still half. Use away_cover: ats_prediction >
    market_spread. ats_prediction=+1.0 > market_spread=0.0 => away_cover,
    slipped = 0.0 + 0.5 = 0.5 (half). To land EXACTLY on the slipped line we need
    a half-point actual margin, which integer scores cannot make — so instead set
    market_spread such that slipped is an integer: market_spread=-3.5 =>
    home_cover slipped = -4.0; margin = -4 (home loses by 4) lands on -4.0.
    """
    from api.season_metrics import _ats_outcome, compute_season_kpis

    # home_cover, slipped = -3.5 - 0.5 = -4.0; margin = 20 - 24 = -4 -> push.
    row = _game(
        ats_prediction=-6.0,
        market_spread=-3.5,
        home_score=20,
        away_score=24,  # margin -4.0 == slipped -4.0 => push
    )
    assert _ats_outcome(row) is None

    kpis = compute_season_kpis([row])
    assert kpis["ats_decided"] == 0  # push excluded from the denominator


def test_ou_push_on_slipped_line_excluded() -> None:
    """A row whose actual total lands exactly on the slipped OU line is excluded.

    ou_prediction=50.0 > market_total=44.5 => over, slipped = 44.5 + 0.5 = 45.0;
    actual total = 24 + 21 = 45 -> push.
    """
    from api.season_metrics import _ou_outcome, compute_season_kpis

    row = _game(
        ou_prediction=50.0,
        market_total=44.5,
        home_score=24,
        away_score=21,  # total 45.0 == slipped 45.0 => push
    )
    assert _ou_outcome(row) is None

    kpis = compute_season_kpis([row])
    assert kpis["ou_decided"] == 0  # push excluded from the denominator


def test_ou_known_correct_over_is_hit() -> None:
    """A correct OU 'over' call scores hit==True.

    ou_prediction=50.0 > market_total=44.0 => over, slipped = 44.5; actual total
    = 30 + 20 = 50 (> 44.5) -> went over -> over hit == True.
    """
    from api.season_metrics import _ou_outcome

    row = _game(ou_prediction=50.0, market_total=44.0, home_score=30, away_score=20)
    assert _ou_outcome(row) is True


def test_ats_known_cover_is_hit() -> None:
    """A correct ATS home_cover call scores hit==True.

    ats_prediction=-6.0 < market_spread=-3.0 => home_cover, slipped=-3.5; margin
    = 24 - 14 = +10 (> -3.5) -> home covers -> home_cover hit == True.
    """
    from api.season_metrics import _ats_outcome

    row = _game(ats_prediction=-6.0, market_spread=-3.0, home_score=24, away_score=14)
    assert _ats_outcome(row) is True


# ---------------------------------------------------------------------------
# NaN-safe coercion (WR-03 idiom) — a NaN score/line must not propagate
# ---------------------------------------------------------------------------


def test_nan_score_does_not_propagate_into_series() -> None:
    """A row with a NaN score is treated as undecided, not a NaN in the output.

    The bad row must drop to the excluded path (None outcome) rather than
    poisoning the cumulative series or the KPI with NaN.
    """
    import math

    from api.season_metrics import compute_cumulative_series, compute_season_kpis

    good = _game(season=2024, week=1, wp_prob=0.70, home_score=28, away_score=20)
    nan_row = _game(
        season=2024,
        week=2,
        wp_prob=0.65,
        home_score=float("nan"),  # corrupt score
        away_score=20,
    )
    kpis = compute_season_kpis([good, nan_row])
    # Only the good row is decided; the NaN row is excluded.
    assert kpis["wp_decided"] == 1
    assert kpis["wp_hit_rate"] == pytest.approx(100.0)
    assert not math.isnan(kpis["wp_hit_rate"])

    series = compute_cumulative_series([good, nan_row])
    assert series["wp"]["values"] == pytest.approx([100.0])
    assert all(not math.isnan(v) for v in series["wp"]["values"])


def test_nan_line_does_not_propagate() -> None:
    """A NaN market line drops the row from that target's denominator."""
    from api.season_metrics import _ats_outcome, _ou_outcome

    assert _ats_outcome(_game(ats_prediction=-3.0, market_spread=float("nan"))) is None
    assert _ou_outcome(_game(ou_prediction=50.0, market_total=float("nan"))) is None


# ---------------------------------------------------------------------------
# Weekly series + rolling overlay (DASH-08 / D-07)
# ---------------------------------------------------------------------------


def test_weekly_series_per_week_hit_rate_and_rolling_overlay() -> None:
    """compute_weekly_series returns per-week hit-rate + a rolling-average overlay.

    Two WP games per week across three weeks, each week one hit + one miss =>
    50% per week. The rolling average of a flat 50% series is also 50%.
    """
    from api.season_metrics import compute_weekly_series

    rows: list[dict] = []
    for wk in (1, 2, 3):
        rows.append(
            _game(season=2024, week=wk, wp_prob=0.70, home_score=28, away_score=20)
        )  # HIT
        rows.append(
            _game(season=2024, week=wk, wp_prob=0.70, home_score=14, away_score=28)
        )  # MISS
    weekly = compute_weekly_series(rows)

    assert weekly["wp"]["weeks"] == [1, 2, 3]
    assert weekly["wp"]["values"] == pytest.approx([50.0, 50.0, 50.0])
    # Rolling overlay aligns to the same weeks and smooths a flat 50% to 50%.
    assert weekly["wp"]["rolling"] == pytest.approx([50.0, 50.0, 50.0])


def test_weekly_series_empty_rows_returns_empty() -> None:
    """Empty input yields empty per-target weekly series without raising."""
    from api.season_metrics import compute_weekly_series

    weekly = compute_weekly_series([])
    for target in ("wp", "ats", "ou"):
        assert weekly[target]["weeks"] == []
        assert weekly[target]["values"] == []


# ---------------------------------------------------------------------------
# compute_season_kpis — shape, record, completed-only, empty input
# ---------------------------------------------------------------------------


def test_compute_season_kpis_returns_expected_keys() -> None:
    """compute_season_kpis returns per-target hit-rates + a W-L record."""
    from api.season_metrics import compute_season_kpis

    kpis = compute_season_kpis([_game()])
    for key in (
        "wp_hit_rate",
        "ats_hit_rate",
        "ou_hit_rate",
        "wp_decided",
        "ats_decided",
        "ou_decided",
        "wp_hits",
        "ats_hits",
        "ou_hits",
        "record_wins",
        "record_losses",
        "record",
    ):
        assert key in kpis, f"missing key {key!r}"
    assert isinstance(kpis["record"], str)
    assert "-" in kpis["record"]  # "W-L" format


def test_compute_season_kpis_record_is_straight_up_wl() -> None:
    """The W-L record counts straight-up correct/incorrect WP calls (ties excluded)."""
    from api.season_metrics import compute_season_kpis

    rows = [
        _game(season=2021, week=1, wp_prob=0.70, home_score=28, away_score=20),  # W
        _game(season=2021, week=2, wp_prob=0.70, home_score=14, away_score=28),  # L
        _game(season=2021, week=3, wp_prob=0.55, home_score=27, away_score=13),  # W
        _game(season=2021, week=4, wp_prob=0.60, home_score=21, away_score=21),  # tie
    ]
    kpis = compute_season_kpis(rows)
    assert kpis["record_wins"] == 2
    assert kpis["record_losses"] == 1  # the tie is neither a win nor a loss
    assert kpis["record"] == "2-1"


def test_compute_season_kpis_ignores_non_completed_games() -> None:
    """Scheduled / non-completed games are excluded from every denominator."""
    from api.season_metrics import compute_season_kpis

    rows = [
        _game(
            season=2024,
            week=1,
            status="completed",
            wp_prob=0.70,
            home_score=28,
            away_score=20,
        ),  # decided HIT
        _game(
            season=2024,
            week=2,
            status="scheduled",
            wp_prob=0.65,
            home_score=None,
            away_score=None,
        ),  # not completed
    ]
    kpis = compute_season_kpis(rows)
    assert kpis["wp_decided"] == 1
    assert kpis["wp_hit_rate"] == pytest.approx(100.0)


def test_compute_season_kpis_empty_input_is_zeroed() -> None:
    """compute_season_kpis([]) returns zeroed values without raising."""
    from api.season_metrics import compute_season_kpis

    kpis = compute_season_kpis([])
    assert kpis["wp_hit_rate"] == 0.0
    assert kpis["ats_hit_rate"] == 0.0
    assert kpis["ou_hit_rate"] == 0.0
    assert kpis["wp_decided"] == 0
    assert kpis["record"] == "0-0"


def test_breakeven_win_rate_constant_is_copied_value() -> None:
    """BREAKEVEN_WIN_RATE is the -110 break-even (0.524), a copied value (UIAP-01)."""
    from api.season_metrics import BREAKEVEN_WIN_RATE, SLIPPAGE_POINTS

    assert pytest.approx(0.524) == BREAKEVEN_WIN_RATE
    assert pytest.approx(0.5) == SLIPPAGE_POINTS
