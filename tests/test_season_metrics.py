"""Unit tests for ``api.season_metrics`` (the season-tracking pure-math module).

These exercise the LOCKED per-target hit-rate convention (CONTEXT D-05): WP
excludes tie games; ATS grades on the home-margin convention (DEF-31-01,
re-measured in step 33.2-26c: model above the line = home_cover) and ATS/OU
apply 0.5-pt slippage against the bettor (NOT the ~78% ``-market_spread``
artifact the old ``_compute_week_summary`` produced);
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


def test_ats_convention_single_disagreement_row_returns_home_margin_answer() -> None:
    """A row where the home-margin rule, the naive rule and the retired rule disagree.

    Home favoured by 7 (market_spread=+7.0); the model expects the home team to win
    by only 3 (ats_prediction=+3.0), so it picks the AWAY side; the home team wins
    by 10 and covers.

      Home-margin rule: pred < line => away_cover, slipped = 7.0 - 0.5 = 6.5;
        home_covers = (10 > 6.5) = True -> away pick MISSES.
      Naive: (margin > -ms)=(10 > -7)=True; (pred > -ms)=(3 > -7)=True -> HIT.
      Retired rule (pred < line => home_cover, slipped 6.5): 10 > 6.5 -> HIT.
    The module MUST return the home-margin answer (a miss).
    """
    from api.season_metrics import _ats_outcome

    row = _game(
        season=2021,
        week=1,
        ats_prediction=3.0,
        market_spread=7.0,
        home_score=27,
        away_score=17,
    )
    assert _ats_outcome(row) is False
    # The naive convention would have called it a HIT -> proves they disagree.
    assert _naive_ats_hit(3.0, 7.0, 10) is True


def test_ats_convention_aggregate_is_about_51_not_78() -> None:
    """The aggregate season ATS hit-rate is hits over decided games, NOT ~78%.

    Build a controlled fixture of decided ATS games with exactly 18 hits and 17
    misses (35 decided) => 51.43%. A ~78% result (the old ``-market_spread``
    artifact) would fail ``ats_rate < 60``.

    Every row uses a clean home_cover/away_cover decision with no push (the
    actual margin never lands on the slipped line), so all 35 are decided.
    """
    from api.season_metrics import compute_season_kpis

    rows: list[dict] = []
    week = 1
    # 18 hits: model picks home_cover (pred > ms) and home covers the slipped line.
    #   ms=-3.0 -> slipped=-2.5; margin=+10 (> -2.5) -> home_covers -> HIT.
    for _ in range(18):
        rows.append(
            _game(
                season=2021,
                week=week,
                ats_prediction=0.0,  # > market_spread (-3.0) => home_cover
                market_spread=-3.0,
                home_score=24,
                away_score=14,  # margin +10 > slipped -2.5 => home covers => HIT
            )
        )
        week += 1
    # 17 misses: model picks home_cover but home fails to cover the slipped line.
    #   ms=-3.0 -> slipped=-2.5; margin=-10 (< -2.5) -> home does NOT cover -> MISS.
    for _ in range(17):
        rows.append(
            _game(
                season=2021,
                week=week,
                ats_prediction=0.0,  # > market_spread (-3.0) => home_cover
                market_spread=-3.0,
                home_score=14,
                away_score=24,  # margin -10 < slipped -2.5 => home fails => MISS
            )
        )
        week += 1

    kpis = compute_season_kpis(rows)
    ats_rate = kpis["ats_hit_rate"]

    # Hits over decided games — NOT the ~78% ``-market_spread`` artifact.
    assert ats_rate == pytest.approx(18 / 35 * 100, abs=1e-6)
    assert ats_rate < 60  # a ~78% result (the bug) would FAIL here
    assert kpis["ats_decided"] == 35  # all rows decided (no pushes)


def test_ats_convention_no_directional_pick_is_excluded() -> None:
    """ats_prediction == market_spread => no side => excluded from ATS denom."""
    from api.season_metrics import _ats_outcome, compute_season_kpis

    row = _game(ats_prediction=-3.0, market_spread=-3.0)
    assert _ats_outcome(row) is None

    kpis = compute_season_kpis([row])
    assert kpis["ats_decided"] == 0
    # Nothing decided is NOT MEASURED, never a measured-looking 0.0 (33.2 review C2 WR-02).
    assert kpis["ats_hit_rate"] is None


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

    Integer scores need an integer slipped line, so the line is a half point:
    ats_prediction=0.0 > market_spread=-3.5 => home_cover, slipped = -3.5 + 0.5
    = -3.0; margin = 21 - 24 = -3 lands exactly on it -> push.
    """
    from api.season_metrics import _ats_outcome, compute_season_kpis

    row = _game(
        ats_prediction=0.0,
        market_spread=-3.5,
        home_score=21,
        away_score=24,  # margin -3.0 == slipped -3.0 => push
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
    """Correct ATS calls on each side score hit==True.

    Home: ats_prediction=0.0 > market_spread=-3.0 => home_cover, slipped=-2.5;
    margin = 24 - 14 = +10 (> -2.5) -> home covers -> hit.
    Away: ats_prediction=-6.0 < market_spread=-3.0 => away_cover, slipped=-3.5;
    margin = 14 - 24 = -10 (< -3.5) -> home fails to cover -> away hit.
    """
    from api.season_metrics import _ats_outcome

    home = _game(ats_prediction=0.0, market_spread=-3.0, home_score=24, away_score=14)
    assert _ats_outcome(home) is True
    away = _game(ats_prediction=-6.0, market_spread=-3.0, home_score=14, away_score=24)
    assert _ats_outcome(away) is True


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


def test_compute_season_kpis_empty_input_is_not_measured() -> None:
    """compute_season_kpis([]) returns zero counts and NO rate or record, without raising.

    33.2 review C2 WR-02: the zeroed 0.0 rates and "0-0" record rendered on /season as
    measurements for a season with nothing graded.
    """
    from api.season_metrics import compute_season_kpis, season_has_graded_games

    kpis = compute_season_kpis([])
    assert kpis["wp_hit_rate"] is None
    assert kpis["ats_hit_rate"] is None
    assert kpis["ou_hit_rate"] is None
    assert kpis["wp_decided"] == 0
    assert kpis["record"] is None
    assert season_has_graded_games(kpis) is False


def test_season_has_graded_games_reads_the_decided_counts() -> None:
    """One decided game on any target is a season with something to show."""
    from api.season_metrics import season_has_graded_games

    assert season_has_graded_games({"wp_decided": 0, "ats_decided": 1, "ou_decided": 0})
    assert not season_has_graded_games({"wp_decided": 0, "ats_decided": 0})
    assert not season_has_graded_games({})


def test_breakeven_win_rate_constant_is_copied_value() -> None:
    """BREAKEVEN_WIN_RATE is the -110 break-even (0.524), a copied value (UIAP-01)."""
    from api.season_metrics import BREAKEVEN_WIN_RATE, SLIPPAGE_POINTS

    assert pytest.approx(0.524) == BREAKEVEN_WIN_RATE
    assert pytest.approx(0.5) == SLIPPAGE_POINTS
