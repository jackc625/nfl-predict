"""Unit tests for ``api.betting_metrics`` (the betting-page pure-math module).

These are real, non-skipped tests: the module landed in Plan 17-02 Task 1. They
exercise KPI math (push exclusion, ROI, max-drawdown), the scope filter, per-type
edge bucketing, the ROI-table row shape, and empty-input handling — all with small
hand-built fixture rows and hand-computed oracles (NOT the real 3,158-row CSV) so
assertions are exact and fast.

Test names include ``betting_kpis`` and ``betting_scope_filter`` to match the
Nyquist ``-k`` selectors from 17-VALIDATION.md.

The per-bet row dict keys mirror the ``betting_bets`` cache table / bundle:
``game_id, season, week, target, bet_side, model_value, market_value, edge,
slipped_line, odds, flat_stake, kelly_stake, outcome, payout_flat, payout_kelly``.
``outcome`` is a nullable bool: ``True`` (win), ``False`` (loss), ``None`` (push).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixture row builders (hand-built; oracles computed in the asserts)
# ---------------------------------------------------------------------------


def _bet(
    *,
    target: str = "wp",
    season: int = 2021,
    week: int = 1,
    edge: float = 0.05,
    flat_stake: float = 100.0,
    kelly_stake: float = 50.0,
    outcome: bool | None = True,
    payout_flat: float = 0.0,
    payout_kelly: float = 0.0,
) -> dict:
    """Return a single per-bet row dict with sensible defaults.

    Only the keys the metrics module reads are required; the rest of the cache
    schema is irrelevant to the math and omitted for brevity.
    """
    return {
        "target": target,
        "season": season,
        "week": week,
        "edge": edge,
        "flat_stake": flat_stake,
        "kelly_stake": kelly_stake,
        "outcome": outcome,
        "payout_flat": payout_flat,
        "payout_kelly": payout_kelly,
    }


# ---------------------------------------------------------------------------
# compute_kpis — win rate (push exclusion), ROI, net, final bankroll, drawdown
# ---------------------------------------------------------------------------


def test_betting_kpis_known_values() -> None:
    """A hand-built list with known wins/losses/pushes/stakes asserts every KPI.

    Rows (flat_stake=100 each, win pays +90.9, loss pays -100, push pays 0):
      1. WP win,  kelly_stake=50, payout_flat=+90.9, payout_kelly=+45.45
      2. WP loss, kelly_stake=40, payout_flat=-100,  payout_kelly=-40
      3. ATS win, kelly_stake=0,  payout_flat=+90.9, payout_kelly=0
      4. OU push, kelly_stake=0,  payout_flat=0,     payout_kelly=0

    Hand-computed oracles:
      total_bets        = 4 (push counts as a bet)
      wins=2, losses=1, push excluded -> win_rate = 2/3*100 = 66.666...%
      roi_flat   = (90.9 - 100 + 90.9 + 0) / 400 * 100 = 81.8/400*100 = 20.45%
      roi_kelly  = (45.45 - 40 + 0 + 0) / (50 + 40) * 100 = 5.45/90*100 = 6.0555...%
      net_profit_flat     = 81.8
      final_bankroll_flat = 10000 + 81.8 = 10081.8
      max_drawdown_flat: equity path from 10000:
        +90.9  -> 10090.9 (peak)
        -100   ->  9990.9 (dd = 100.0)
        +90.9  -> 10081.8
        +0     -> 10081.8
        -> max_drawdown = 100.0
    """
    from api.betting_metrics import STARTING_BANKROLL, compute_kpis

    rows = [
        _bet(
            target="wp",
            outcome=True,
            kelly_stake=50.0,
            payout_flat=90.9,
            payout_kelly=45.45,
        ),
        _bet(
            target="wp",
            outcome=False,
            kelly_stake=40.0,
            payout_flat=-100.0,
            payout_kelly=-40.0,
        ),
        _bet(
            target="ats",
            outcome=True,
            kelly_stake=0.0,
            payout_flat=90.9,
            payout_kelly=0.0,
        ),
        _bet(
            target="ou",
            outcome=None,
            kelly_stake=0.0,
            payout_flat=0.0,
            payout_kelly=0.0,
        ),
    ]

    kpis = compute_kpis(rows)

    assert kpis["total_bets"] == 4
    assert kpis["win_rate"] == pytest.approx(2 / 3 * 100, abs=1e-9)
    assert kpis["roi_flat"] == pytest.approx(81.8 / 400 * 100, abs=1e-9)
    assert kpis["roi_kelly"] == pytest.approx(5.45 / 90 * 100, abs=1e-9)
    assert kpis["net_profit_flat"] == pytest.approx(81.8, abs=1e-9)
    assert kpis["final_bankroll_flat"] == pytest.approx(
        STARTING_BANKROLL + 81.8, abs=1e-9
    )
    assert kpis["max_drawdown_flat"] == pytest.approx(100.0, abs=1e-9)


def test_betting_kpis_returns_exact_key_set() -> None:
    """compute_kpis returns a dict with exactly the 7 documented keys."""
    from api.betting_metrics import compute_kpis

    kpis = compute_kpis([_bet()])
    assert set(kpis) == {
        "total_bets",
        "win_rate",
        "roi_flat",
        "roi_kelly",
        "net_profit_flat",
        "final_bankroll_flat",
        "max_drawdown_flat",
    }


def test_betting_kpis_wins_only_is_100_percent() -> None:
    """A wins-only list yields win_rate 100.0."""
    from api.betting_metrics import compute_kpis

    rows = [
        _bet(outcome=True, payout_flat=90.9),
        _bet(outcome=True, payout_flat=90.9),
    ]
    assert compute_kpis(rows)["win_rate"] == pytest.approx(100.0)


def test_betting_kpis_push_exclusion_from_win_rate() -> None:
    """Pushes (outcome is None) are excluded from the win-rate denominator.

    A list of [win, loss, push] has win_rate = 1/2 = 50.0% (the push is NOT in
    the denominator), and total_bets = 3 (the push IS counted as a bet). Adding
    a second push does not change win_rate but does increment total_bets.
    """
    from api.betting_metrics import compute_kpis

    base = [
        _bet(outcome=True, payout_flat=90.9),
        _bet(outcome=False, payout_flat=-100.0),
    ]
    with_one_push = [*base, _bet(outcome=None, payout_flat=0.0)]
    with_two_push = [*with_one_push, _bet(outcome=None, payout_flat=0.0)]

    kpis_one = compute_kpis(with_one_push)
    kpis_two = compute_kpis(with_two_push)

    # Win rate computed over decided bets only (1 win / 2 decided = 50%).
    assert kpis_one["win_rate"] == pytest.approx(50.0)
    # Adding another push leaves the win rate unchanged but bumps total_bets.
    assert kpis_two["win_rate"] == pytest.approx(50.0)
    assert kpis_one["total_bets"] == 3
    assert kpis_two["total_bets"] == 4


def test_betting_kpis_pushes_only_is_zero_win_rate() -> None:
    """A pushes-only list yields win_rate 0.0 and counts the pushes as bets."""
    from api.betting_metrics import compute_kpis

    rows = [_bet(outcome=None, payout_flat=0.0), _bet(outcome=None, payout_flat=0.0)]
    kpis = compute_kpis(rows)
    assert kpis["win_rate"] == pytest.approx(0.0)
    assert kpis["total_bets"] == 2


def test_betting_kpis_does_not_coerce_outcome_truthiness() -> None:
    """Win rate uses identity checks, not bool() — a stray truthy/falsy value
    that is neither True/False/None is treated as undecided, not a win/loss.

    This guards the Pitfall-1 trap: ``bool("False") is True`` and
    ``bool(0.0) is False`` would both corrupt the count under truthiness. A row
    whose outcome is the string "False" (a corruption) must NOT be counted as a
    decided bet by an identity-based implementation, so a single genuine win
    among such rows yields win_rate 100.0 (1 win / 1 decided), not 50%.
    """
    from api.betting_metrics import compute_kpis

    rows = [
        _bet(outcome=True, payout_flat=90.9),
        # Corrupt non-bool outcomes: identity checks ignore both.
        {**_bet(), "outcome": "False"},
        {**_bet(), "outcome": 0.0},
    ]
    # Only the first row is a decided bet (a win) under identity semantics.
    assert compute_kpis(rows)["win_rate"] == pytest.approx(100.0)
    assert compute_kpis(rows)["total_bets"] == 3


# ---------------------------------------------------------------------------
# filter_scope — recommended == kelly_stake > 0
# ---------------------------------------------------------------------------


def test_betting_scope_filter_recommended_and_all() -> None:
    """recommended -> only kelly_stake>0 rows; all -> every row."""
    from api.betting_metrics import filter_scope

    rows = [
        _bet(kelly_stake=50.0),
        _bet(kelly_stake=0.0),
        _bet(kelly_stake=12.5),
        _bet(kelly_stake=0.0),
    ]
    recommended = filter_scope(rows, "recommended")
    assert len(recommended) == 2  # two rows with kelly_stake > 0
    assert all((r["kelly_stake"] or 0) > 0 for r in recommended)
    assert len(filter_scope(rows, "all")) == len(rows)


def test_betting_scope_filter_unknown_scope_returns_all() -> None:
    """An unrecognized scope falls back to all rows."""
    from api.betting_metrics import filter_scope

    rows = [_bet(kelly_stake=50.0), _bet(kelly_stake=0.0)]
    assert len(filter_scope(rows, "garbage")) == 2


def test_betting_scope_filter_handles_missing_kelly_stake() -> None:
    """A row missing kelly_stake (or None) is excluded from recommended."""
    from api.betting_metrics import filter_scope

    rows = [
        {**_bet(), "kelly_stake": None},
        {k: v for k, v in _bet().items() if k != "kelly_stake"},
        _bet(kelly_stake=5.0),
    ]
    assert len(filter_scope(rows, "recommended")) == 1


def test_betting_scope_filter_does_not_mutate_input() -> None:
    """filter_scope returns a new list and leaves the input untouched."""
    from api.betting_metrics import filter_scope

    rows = [_bet(kelly_stake=50.0), _bet(kelly_stake=0.0)]
    original_len = len(rows)
    _ = filter_scope(rows, "recommended")
    assert len(rows) == original_len


# ---------------------------------------------------------------------------
# edge_bucket_for — per-type small/medium/big
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "edge", "expected"),
    [
        # ATS (points): small <1.1, medium 1.1-2.6, big >2.6
        ("ats", 0.5, "small"),
        ("ats", 1.5, "medium"),
        ("ats", 3.0, "big"),
        # OU (points): small <1.3, medium 1.3-2.9, big >2.9
        ("ou", 1.0, "small"),
        ("ou", 2.0, "medium"),
        ("ou", 3.5, "big"),
        # WP (|edge|): small <0.03, medium 0.03-0.08, big >0.08
        ("wp", 0.01, "small"),
        ("wp", 0.05, "medium"),
        ("wp", 0.12, "big"),
    ],
)
def test_edge_bucket_for_per_type_cut_points(
    target: str, edge: float, expected: str
) -> None:
    """edge_bucket_for returns the expected label at representative cut points."""
    from api.betting_metrics import edge_bucket_for

    assert edge_bucket_for(target, edge) == expected


def test_edge_bucket_for_wp_uses_magnitude() -> None:
    """WP edge is signed; the bucket keys on |edge| so -0.05 == +0.05."""
    from api.betting_metrics import edge_bucket_for

    assert edge_bucket_for("wp", -0.05) == edge_bucket_for("wp", 0.05) == "medium"
    assert edge_bucket_for("wp", -0.12) == "big"


def test_edge_bucket_for_boundaries_are_left_closed() -> None:
    """A value exactly at a cut point falls into the upper bucket (right-open)."""
    from api.betting_metrics import edge_bucket_for

    # ATS small_max == 1.1 -> exactly 1.1 is "medium"; medium_max 2.6 -> "big".
    assert edge_bucket_for("ats", 1.1) == "medium"
    assert edge_bucket_for("ats", 2.6) == "big"


# ---------------------------------------------------------------------------
# _max_drawdown — peak-to-trough on cumulative equity
# ---------------------------------------------------------------------------


def test_max_drawdown_empty_is_zero() -> None:
    """An empty list has no drawdown."""
    from api.betting_metrics import _max_drawdown

    assert _max_drawdown([], "payout_flat") == 0.0


def test_max_drawdown_monotonic_rise_is_zero() -> None:
    """A monotonically rising equity series never draws down."""
    from api.betting_metrics import _max_drawdown

    rows = [_bet(payout_flat=100.0), _bet(payout_flat=50.0), _bet(payout_flat=25.0)]
    assert _max_drawdown(rows, "payout_flat") == 0.0


def test_max_drawdown_peak_to_trough() -> None:
    """Drawdown is the largest peak-minus-equity along the cumulative path.

    Path from 10000: +200 -> 10200 (peak), -500 -> 9700 (dd=500), +100 ->
    9800, -50 -> 9750. Largest peak-to-trough decline = 10200 - 9700 = 500.
    """
    from api.betting_metrics import _max_drawdown

    rows = [
        _bet(payout_flat=200.0),
        _bet(payout_flat=-500.0),
        _bet(payout_flat=100.0),
        _bet(payout_flat=-50.0),
    ]
    assert _max_drawdown(rows, "payout_flat") == pytest.approx(500.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_roi_table — per-slice rows with tri-state roi_favorable + *_fmt keys
# ---------------------------------------------------------------------------


def test_betting_roi_table_row_shape() -> None:
    """Every ROI-table row carries the documented keys and a tri-state flag."""
    from api.betting_metrics import compute_roi_table

    rows = [
        _bet(
            target="wp",
            season=2021,
            edge=0.05,
            outcome=True,
            payout_flat=90.9,
            payout_kelly=45.0,
        ),
        _bet(
            target="ats",
            season=2022,
            edge=1.5,
            outcome=False,
            payout_flat=-100.0,
            kelly_stake=0.0,
        ),
        _bet(
            target="ou",
            season=2022,
            edge=3.5,
            outcome=True,
            payout_flat=90.9,
            kelly_stake=10.0,
            payout_kelly=9.0,
        ),
    ]
    table = compute_roi_table(rows)

    assert isinstance(table, list)
    assert len(table) > 0
    for row in table:
        for key in (
            "roi_flat",
            "roi_kelly",
            "win_rate",
            "bet_count",
            "roi_flat_fmt",
            "roi_kelly_fmt",
            "win_rate_fmt",
            "roi_favorable",
        ):
            assert key in row, f"missing key {key!r} in {row!r}"
        assert row["roi_favorable"] in {True, False, None}
        assert isinstance(row["bet_count"], int)
        assert isinstance(row["roi_flat_fmt"], str)


def test_betting_roi_table_tri_state_favorable() -> None:
    """roi_favorable is True for a profitable slice, False for a losing one, and
    None for a slice with no decided bets (e.g. a push-only slice)."""
    from api.betting_metrics import compute_roi_table

    rows = [
        # WP: one clear win -> positive ROI -> favorable True.
        _bet(target="wp", season=2021, edge=0.05, outcome=True, payout_flat=90.9),
        # ATS: one clear loss -> negative ROI -> favorable False.
        _bet(target="ats", season=2021, edge=1.5, outcome=False, payout_flat=-100.0),
        # OU: a single push -> no decided bets -> favorable None.
        _bet(target="ou", season=2021, edge=3.5, outcome=None, payout_flat=0.0),
    ]
    table = compute_roi_table(rows)
    by_type = {r["label"]: r for r in table if r["slice_kind"] == "type"}

    assert by_type["wp"]["roi_favorable"] is True
    assert by_type["ats"]["roi_favorable"] is False
    assert by_type["ou"]["roi_favorable"] is None


def test_betting_roi_table_has_type_season_and_bucket_slices() -> None:
    """The table covers all three slice kinds: type, season, and edge bucket."""
    from api.betting_metrics import compute_roi_table

    rows = [
        _bet(target="wp", season=2021, edge=0.05, outcome=True, payout_flat=90.9),
        _bet(target="ats", season=2022, edge=3.0, outcome=False, payout_flat=-100.0),
    ]
    table = compute_roi_table(rows)
    kinds = {r["slice_kind"] for r in table}
    assert kinds == {"type", "season", "bucket"}


# ---------------------------------------------------------------------------
# Empty input — no raise, zeroed KPIs, empty table
# ---------------------------------------------------------------------------


def test_betting_kpis_empty_input_is_zeroed() -> None:
    """compute_kpis([]) returns zeroed values (final bankroll == start) without raising."""
    from api.betting_metrics import STARTING_BANKROLL, compute_kpis

    kpis = compute_kpis([])
    assert kpis["total_bets"] == 0
    assert kpis["win_rate"] == 0.0
    assert kpis["roi_flat"] == 0.0
    assert kpis["roi_kelly"] == 0.0
    assert kpis["net_profit_flat"] == 0.0
    assert kpis["final_bankroll_flat"] == pytest.approx(STARTING_BANKROLL)
    assert kpis["max_drawdown_flat"] == 0.0


def test_compute_roi_table_empty_input_returns_empty_list() -> None:
    """compute_roi_table([]) returns an empty list."""
    from api.betting_metrics import compute_roi_table

    assert compute_roi_table([]) == []
