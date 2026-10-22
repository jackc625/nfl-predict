"""The season-close readout's binding mode (Phase 38, Plan 38-09; SPEC R8, R9).

The binding verdict is computed exactly as SEASON-VERDICT-PREREGISTRATION.md froze it: each target's
raw figures come from the preview's one computation (``scripts.season_readout.target_figures``),
its p-value from ``backtest.roi_significance.roi_ci_and_p`` over its counted bets' (season, week)
blocks, the correction from ``benjamini_hochberg_adjusted`` at m = 3, and the word from
``assign_verdict_token``. Decisions read unrounded values, and no closing-line value enters.

COLD-05: every store and every git repository these tests use lives under ``tmp_path`` -- never
the real ``docs/records/``, ``ledger/``, ``data/``, ``artifacts/`` or ``logs/``. The ledger rows are
written through the real store (``append_rows`` / ``apply_updates``) and read back through
``forward_ledger.verdict.verdict_rows``, the one reader of counted rows.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.profitability_2025 import benjamini_hochberg_adjusted
from backtest.roi_significance import roi_ci_and_p
from backtest.verdict_test_2026 import (
    BH_FAMILY_SIZE,
    UNDISCHARGEABLE_NO_BETS,
    VERDICT_TARGETS,
    VERDICT_TOKENS,
)
from forward_ledger.declarations import load_verdict_scope
from forward_ledger.shadow import SHADOW_PIN
from forward_ledger.store import append_rows, apply_updates, read_entries
from forward_ledger.verdict import verdict_rows
from scripts import season_readout
from tests.unit.test_forward_ledger_store import graded, key_of, make_row

# The verdict words, by their VERDICT_TOKENS position (profitable, unprofitable, inconclusive,
# no bets, no chain) -- never retyped.
_PROFITABLE, _UNPROFITABLE, _INCONCLUSIVE, _NO_BETS, _NO_CHAIN = VERDICT_TOKENS
assert _NO_BETS == UNDISCHARGEABLE_NO_BETS

# The complete fixture's games: two in week 5, two in week 6, the Super Bowl in week 22.
WEEK5_GAMES: tuple[str, ...] = ("2026_W05_TB@DAL", "2026_W05_BUF@LA")
WEEK6_GAMES: tuple[str, ...] = ("2026_W06_SF@SEA", "2026_W06_MIA@NE")
SUPER_BOWL: str = "2026_W22_KC@PHI"
BET_GAMES: tuple[str, ...] = (*WEEK5_GAMES, *WEEK6_GAMES, SUPER_BOWL)

# A game whose only counted row is a suppressed (no-bet) WP candidate.
NO_BET_GAME: str = "2026_W06_DEN@LV"

# Every counted bet's in-force grade, per (game, target).
COMPLETE_GRADES: dict[tuple[str, str], str] = {
    ("2026_W05_TB@DAL", "wp"): "win",
    ("2026_W05_TB@DAL", "ats"): "loss",
    ("2026_W05_TB@DAL", "ou"): "win",
    ("2026_W05_BUF@LA", "wp"): "loss",
    ("2026_W05_BUF@LA", "ats"): "win",
    ("2026_W05_BUF@LA", "ou"): "push",
    ("2026_W06_SF@SEA", "wp"): "win",
    ("2026_W06_SF@SEA", "ats"): "win",
    ("2026_W06_SF@SEA", "ou"): "loss",
    ("2026_W06_MIA@NE", "wp"): "loss",
    ("2026_W06_MIA@NE", "ats"): "push",
    ("2026_W06_MIA@NE", "ou"): "win",
    (SUPER_BOWL, "wp"): "win",
    (SUPER_BOWL, "ats"): "win",
    (SUPER_BOWL, "ou"): "loss",
}

# (payout_flat, realized_units) per grade, at -110 and a 1.25-unit stake.
PAYOUTS: dict[str, tuple[float, float]] = {
    "win": (0.9090909090909091, 1.1363636363636365),
    "loss": (-1.0, -1.25),
    "push": (0.0, 0.0),
}

# When each week's bets were graded: weeks 5 and 6 before the rule's lock, week 22 after it.
GRADED_AT: dict[int, datetime] = {
    5: datetime(2026, 10, 13, 21, 0, tzinfo=UTC),
    6: datetime(2026, 10, 20, 21, 0, tzinfo=UTC),
    22: datetime(2027, 2, 15, 21, 0, tzinfo=UTC),
}

_MODEL_VALUES: dict[str, float] = {"wp": 0.6, "ats": -3.2, "ou": 44.5}
_BET_SIDES: dict[str, str] = {"wp": "home", "ats": "home", "ou": "over"}
_CLV_COLUMNS: tuple[str, ...] = ("clv", "forward_clv", "closing_line", "closing_odds")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _week_of(game_id: str) -> int:
    """The week a standard game id names."""
    return int(game_id.split("_W")[1][:2])


def ledger_row(
    game_id: str, target: str, *, status: str = "live", arm: str = "live"
) -> dict[str, Any]:
    """One ledger row for *game_id*, stamped with the frozen pin's ids (generation 0)."""
    return make_row(
        game_id=game_id,
        week=_week_of(game_id),
        target=target,
        status=status,
        arm=arm,
        bet_side=_BET_SIDES[target],
        model_value=_MODEL_VALUES[target],
        rejection_reason=None if status == "live" else "below_ev_floor",
        model_artifact_id=SHADOW_PIN.model_ids[target],
        blend_id=SHADOW_PIN.resolved().blend,
        recipe_id=SHADOW_PIN.recipe,
    )


def grade_update(grade: str, week: int) -> dict[str, Any]:
    """The grading update that moves a bet of *week* to *grade*."""
    payout, units = PAYOUTS[grade]
    return {**graded(grade, payout, units), "graded_at": GRADED_AT[week]}


def write_ledger(
    ledger_dir: Path,
    rows: Sequence[dict[str, Any]],
    grades: Mapping[tuple[str, str], str],
) -> None:
    """Append *rows* through the real store, then grade the live-arm bets *grades* names."""
    append_rows(ledger_dir, list(rows))
    updates = {
        key_of(row): grade_update(grades[(row["game_id"], row["target"])], row["week"])
        for row in rows
        if row["arm"] == "live" and (row["game_id"], row["target"]) in grades
    }
    if updates:
        apply_updates(ledger_dir, grading_updates=updates)


def complete_rows() -> list[dict[str, Any]]:
    """The complete fixture's rows: every bet, one suppressed row and one shadow twin."""
    return [
        *(
            ledger_row(game_id, target)
            for game_id in BET_GAMES
            for target in VERDICT_TARGETS
        ),
        ledger_row(NO_BET_GAME, "wp", status="suppressed"),
        ledger_row(SUPER_BOWL, "ats", arm="shadow"),
    ]


def counted_of(ledger_dir: Path) -> pd.DataFrame:
    """The counted rows of *ledger_dir*, read by the one reader of counted rows."""
    return verdict_rows(read_entries(ledger_dir), load_verdict_scope())


def complete_counted(tmp_path: Path) -> pd.DataFrame:
    """The complete fixture's counted frame (ledger under ``tmp_path/ledger``)."""
    ledger_dir = tmp_path / "ledger"
    write_ledger(ledger_dir, complete_rows(), COMPLETE_GRADES)
    return counted_of(ledger_dir)


def _per_bet(counted: pd.DataFrame, target: str) -> pd.DataFrame:
    """The test's own selection of *target*'s bets, for an independent recomputation."""
    keep = (counted["target"] == target) & (counted["status"] == "live")
    return counted.loc[keep, ["season", "week", "payout_flat", "flat_stake"]]


# ---------------------------------------------------------------------------
# Task 1: the verdict computation
# ---------------------------------------------------------------------------


def test_decisions_use_unrounded_values(tmp_path: Path) -> None:
    """SPEC AC 25 (the decision half): the floats the verdict reads are the functions' own."""
    counted = complete_counted(tmp_path)
    no_void: frozenset[str] = frozenset()

    results = season_readout.measure_targets(counted, no_void)

    assert tuple(result.figures for result in results) == season_readout.target_figures(
        counted, no_void
    )
    assert [result.figures.target for result in results] == list(VERDICT_TARGETS)
    stats = [roi_ci_and_p(_per_bet(counted, target)) for target in VERDICT_TARGETS]
    raw = [stat["p_value"] for stat in stats]
    assert all(p is not None for p in raw)
    adjusted = benjamini_hochberg_adjusted(raw, m=BH_FAMILY_SIZE)
    for result, stat, adjusted_p in zip(results, stats, adjusted, strict=True):
        assert result.figures.flat_roi == stat["point_estimate"]
        assert result.raw_p == stat["p_value"]
        assert result.adjusted_p == adjusted_p
        assert result.n_blocks == stat["n_blocks"] == 3
        assert result.p_absent_reason is None
        assert result.token in VERDICT_TOKENS


def test_the_verdict_reads_no_clv(tmp_path: Path) -> None:
    """CLV never enters: perturbing every closing-line column changes no figure and no word."""
    counted = complete_counted(tmp_path)
    perturbed = counted.copy()
    for offset, column in enumerate(_CLV_COLUMNS, start=1):
        assert column in perturbed.columns
        perturbed[column] = [1000.0 * offset + index for index in range(len(perturbed))]

    assert season_readout.measure_targets(
        perturbed, frozenset()
    ) == season_readout.measure_targets(counted, frozenset())


def test_the_degenerate_targets_keep_the_family_at_three(tmp_path: Path) -> None:
    """The zero-bet word, the no-chain word, an absent p for one block, and m = 3 throughout."""
    # Frame A: wp suppressed only, ats absent, ou's bets all in week 5 (one block, ROI > 0).
    a_dir = tmp_path / "a"
    write_ledger(
        a_dir,
        [
            ledger_row(WEEK5_GAMES[0], "wp", status="suppressed"),
            ledger_row(WEEK6_GAMES[0], "wp", status="suppressed"),
            ledger_row(WEEK5_GAMES[0], "ou"),
            ledger_row(WEEK5_GAMES[1], "ou"),
        ],
        {(WEEK5_GAMES[0], "ou"): "win", (WEEK5_GAMES[1], "ou"): "push"},
    )
    wp, ats, ou = season_readout.measure_targets(counted_of(a_dir), frozenset())

    assert (wp.figures.counted_rows, wp.figures.bets, wp.token) == (2, 0, _NO_BETS)
    assert (ats.figures.counted_rows, ats.token) == (0, _NO_CHAIN)
    one_block = roi_ci_and_p(_per_bet(counted_of(a_dir), "ou"))
    assert (ou.raw_p, ou.adjusted_p, ou.n_blocks) == (None, None, 1)
    assert ou.p_absent_reason == one_block["p_value_absent_reason"] is not None
    assert ou.figures.flat_roi is not None and ou.figures.flat_roi > 0
    assert ou.token == _INCONCLUSIVE

    # Frame B: wp bets across three blocks, ats in one block, ou suppressed only.
    b_dir = tmp_path / "b"
    b_rows = [
        *(
            ledger_row(game_id, "wp")
            for game_id in (WEEK5_GAMES[0], WEEK6_GAMES[0], SUPER_BOWL)
        ),
        ledger_row(WEEK5_GAMES[0], "ats"),
        ledger_row(WEEK5_GAMES[1], "ats"),
        ledger_row(WEEK6_GAMES[1], "ou", status="suppressed"),
    ]
    write_ledger(
        b_dir,
        b_rows,
        {
            (WEEK5_GAMES[0], "wp"): "win",
            (WEEK6_GAMES[0], "wp"): "loss",
            (SUPER_BOWL, "wp"): "win",
            (WEEK5_GAMES[0], "ats"): "win",
            (WEEK5_GAMES[1], "ats"): "loss",
        },
    )
    wp, ats, ou = season_readout.measure_targets(counted_of(b_dir), frozenset())

    assert wp.raw_p is not None and wp.n_blocks == 3
    assert wp.adjusted_p == benjamini_hochberg_adjusted([wp.raw_p], m=3)[0]
    assert wp.adjusted_p == min(1.0, 3 * wp.raw_p)
    assert (ats.raw_p, ats.adjusted_p, ats.n_blocks) == (None, None, 1)
    assert ats.figures.flat_roi is not None and ats.figures.flat_roi < 0
    assert ats.token == _UNPROFITABLE  # a non-positive return needs no p-value
    assert ou.token == _NO_BETS
