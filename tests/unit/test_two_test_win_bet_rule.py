"""A 2026 win bet needs BOTH tests, and the live path reads only the corrected rule (Plan 33.2-26).

THE RULE (D33.2-11)
-------------------
A WP bet is placed only when the model's edge over the SPREAD-DERIVED market probability, on the
side bet, clears WP's corrected threshold AND the moneyline captured at that game's lock still
leaves positive value after the book's cut. Either alone is insufficient: the conversion can
manufacture an apparent edge at every spread, and the threshold was derived on a spread-derived
probability and never calibrated on moneyline prices. So every single-test case below must place
NO bet, and the both-pass case must place one -- the non-vacuity control that stops the four
negatives passing against a selector that never bets.

A target whose corrected threshold is ``None`` places no bets at all, asserted by BEHAVIOUR: zero
live rows, each of its games recorded with a named reason, whatever its moneyline says.

THE LIVE PATH READS THE CORRECTED MODULE, CHECKED BY IDENTITY
--------------------------------------------------------------
Over the WHOLE serving graph -- ``backtest/weekly_bet_list.py``, ``backtest/bet_selector.py``,
``scripts/generate_bet_list.py`` and ``utils/edge_tier.py`` -- no module imports
``backtest.cold_start_constants``, at least one imports the corrected module, the live objects ARE
the corrected module's objects, and a counting delegate over the original sees zero reads while
the deferred edge-band import runs. Each negative has a planted-violation control.

Run this module:  uv run python -m pytest tests/unit/test_two_test_win_bet_rule.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import backtest.cold_start_constants as original
import backtest.weekly_bet_list as weekly
from backtest.weekly_bet_list import WeeklyChainFit, select_weekly_bets

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_MODULE = "backtest.cold_start_constants"
CORRECTED_MODULE = "backtest.corrected_cold_start_constants"
SERVING_MODULES: tuple[str, ...] = (
    "backtest/weekly_bet_list.py",
    "backtest/bet_selector.py",
    "scripts/generate_bet_list.py",
    "utils/edge_tier.py",
)

SEASON = 2026
WEEK = 3
# A Sunday game; its lock is 18:00 Eastern on the Saturday, and the snapshot is before it.
GAMEDAY = "2026-09-27"
SNAPSHOT = "2026-09-26T12:00:00-04:00"

# The model likes the home side at 0.65 in every case; only the market inputs move.
MODEL_PROB = 0.65
MARKET_CLEARS = 0.55  # side edge 0.10, above WP's 0.02 medium threshold
MARKET_TIGHT = 0.64  # side edge 0.01, not above it
PRICE_GOOD = (100.0, -120.0)  # home at even money: EV = 0.65 * 1.0 - 0.35 = +0.30
PRICE_BAD = (-400.0, 300.0)  # home at -400: EV = 0.65 * 0.25 - 0.35 = -0.1875


def _corrected() -> Any:
    import backtest.corrected_cold_start_constants as corrected

    return corrected


def _fits() -> dict[str, WeeklyChainFit]:
    return {
        target: WeeklyChainFit(
            target=target,
            ev_floor_t=0.0,
            frozen_sd=None if target == "wp" else 13.0,
            season_bias_by_season={SEASON: 0.0},
        )
        for target in ("wp", "ats", "ou")
    }


def _wp_row(
    game_id: str, market_prob: float | None, price: tuple[float, float]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": SEASON,
        "week": WEEK,
        "target": "wp",
        "gameday": GAMEDAY,
        "snapshot_ts": SNAPSHOT,
        "sportsbook": "consensus",
        "is_live": False,
        "model_prob": MODEL_PROB,
        "ml_home": price[0],
        "ml_away": price[1],
    }
    if market_prob is not None:
        row[weekly.SPREAD_MARKET_PROB_COLUMN] = market_prob
    return row


def _select(
    rows: list[dict[str, Any]],
    thresholds: dict[str, tuple[float, float] | None] | None = None,
) -> Any:
    schedule = pd.DataFrame(
        {
            "game_id": [row["game_id"] for row in rows],
            "season": SEASON,
            "week": WEEK,
            "gameday": GAMEDAY,
        }
    )
    kwargs: dict[str, Any] = {}
    if thresholds is not None:
        kwargs["edge_thresholds"] = thresholds
    return select_weekly_bets(pd.DataFrame(rows), schedule, _fits(), **kwargs)


def _wp_outcome(result: Any, game_id: str) -> str:
    """``"bet"`` for a live WP row, else the WP row's rejection reason."""
    if any(r["game_id"] == game_id and r["target"] == "wp" for r in result.selected):
        return "bet"
    reasons = [
        r["rejection_reason"]
        for r in result.rejected
        if r["game_id"] == game_id and r["target"] == "wp"
    ]
    assert len(reasons) == 1, reasons
    return str(reasons[0])


# ---------------------------------------------------------------------------
# 1. Both tests, each necessary
# ---------------------------------------------------------------------------


def test_both_tests_passing_places_a_bet() -> None:
    """The non-vacuity control: the selector under test DOES bet."""
    result = _select([_wp_row("g_both", MARKET_CLEARS, PRICE_GOOD)])
    assert _wp_outcome(result, "g_both") == "bet"
    (record,) = [r for r in result.selected if r["target"] == "wp"]
    assert record["spread_market_edge"] == pytest.approx(MODEL_PROB - MARKET_CLEARS)


def test_threshold_cleared_but_the_lock_price_loses_places_no_bet() -> None:
    result = _select([_wp_row("g_edge_only", MARKET_CLEARS, PRICE_BAD)])
    assert _wp_outcome(result, "g_edge_only") == "ev_below_floor"


def test_positive_price_but_the_threshold_not_cleared_places_no_bet() -> None:
    result = _select([_wp_row("g_price_only", MARKET_TIGHT, PRICE_GOOD)])
    assert _wp_outcome(result, "g_price_only") == "edge_below_threshold"


def test_neither_test_passing_places_no_bet() -> None:
    result = _select([_wp_row("g_neither", MARKET_TIGHT, PRICE_BAD)])
    assert _wp_outcome(result, "g_neither") != "bet"


def test_an_away_bet_is_judged_on_the_away_side_s_market_probability() -> None:
    """The side edge is side-correct: an away bet compares 1 - p against 1 - q."""
    row = _wp_row("g_away", 0.60, (120.0, 100.0))
    row["model_prob"] = 0.35  # the model likes the away side at 0.65
    result = _select([row])
    assert _wp_outcome(result, "g_away") == "bet"
    (record,) = [r for r in result.selected if r["target"] == "wp"]
    assert record["spread_market_edge"] == pytest.approx((1 - 0.35) - (1 - 0.60))


def test_a_win_bet_with_no_spread_derived_probability_is_suppressed_not_priced() -> (
    None
):
    """The threshold test cannot run without its yardstick; the row is not a bet.

    The column present but EMPTY for this game means the game had no spread -- a market gap.
    """
    rows = [
        _wp_row("g_priced", MARKET_CLEARS, PRICE_GOOD),
        _wp_row("g_no_market", None, PRICE_GOOD),
    ]
    result = _select(rows)
    assert _wp_outcome(result, "g_no_market") == "missing_snapshot"
    assert _wp_outcome(result, "g_priced") == "bet"


def test_no_bound_converter_is_named_as_such_not_as_a_market_gap() -> None:
    """A33.2-review IN-06: the column ABSENT everywhere means no converter was bound.

    ``_attach_spread_market_probability`` adds the column only when the live blend binds a
    converter. Labelling that ``missing_snapshot`` sent the reader to the odds feed for an
    artifact fault.
    """
    result = _select([_wp_row("g_unbound", None, PRICE_GOOD)])
    assert _wp_outcome(result, "g_unbound") == "no_bound_converter"


def test_the_live_threshold_is_the_corrected_wp_medium_value() -> None:
    pair = _corrected().EDGE_TIER_THRESHOLDS_BY_TARGET["wp"]
    assert weekly.edge_admission_thresholds() == {"wp": pair[1]}


# ---------------------------------------------------------------------------
# 2. A None threshold: no bets for that target, by behaviour
# ---------------------------------------------------------------------------


def test_a_none_threshold_places_zero_bets_for_that_target_whatever_the_price() -> None:
    thresholds = dict(_corrected().EDGE_TIER_THRESHOLDS_BY_TARGET)
    thresholds["wp"] = None
    rows = [
        _wp_row("g_a", MARKET_CLEARS, PRICE_GOOD),
        _wp_row("g_b", MARKET_CLEARS, PRICE_GOOD),
    ]
    result = _select(rows, thresholds)

    assert [r for r in result.selected if r["target"] == "wp"] == []
    for game_id in ("g_a", "g_b"):
        assert _wp_outcome(result, game_id) == "no_honest_edge_threshold"

    control = _select(rows)
    assert _wp_outcome(control, "g_a") == "bet", "the same rows DO bet with a threshold"


def test_a_none_line_target_threshold_refuses_that_target_too() -> None:
    thresholds = dict(_corrected().EDGE_TIER_THRESHOLDS_BY_TARGET)
    thresholds["ats"] = None
    assert weekly.threshold_refused_targets(thresholds) == frozenset({"ats"})
    result = _select([_wp_row("g_c", MARKET_CLEARS, PRICE_GOOD)], thresholds)
    ats = [r for r in result.rejected if r["target"] == "ats"]
    assert [r["rejection_reason"] for r in ats] == ["no_honest_edge_threshold"]
    assert _wp_outcome(result, "g_c") == "bet"


# ---------------------------------------------------------------------------
# 3. The live path reads the corrected module, by identity
# ---------------------------------------------------------------------------


def test_the_live_objects_are_the_corrected_module_s_objects() -> None:
    corrected = _corrected()
    assert weekly.CHAIN_FIT_BIAS_2026 is corrected.CHAIN_FIT_BIAS_2026
    assert weekly.CHAIN_FIT_BIAS_SEASONS is corrected.CHAIN_FIT_BIAS_SEASONS
    assert (
        weekly.EDGE_TIER_THRESHOLDS_BY_TARGET
        is corrected.EDGE_TIER_THRESHOLDS_BY_TARGET
    )
    assert weekly.CHAIN_FIT_BIAS_2026 is not original.CHAIN_FIT_BIAS_2026
    assert weekly.CHAIN_FIT_BIAS_2026 != original.CHAIN_FIT_BIAS_2026


class _CountingModule(types.ModuleType):
    """A delegate over a real module that records every attribute read."""

    def __init__(self, real: types.ModuleType) -> None:
        super().__init__(real.__name__)
        self.__dict__["_real"] = real
        self.__dict__["reads"] = []

    def __getattr__(self, name: str) -> Any:
        # The import machinery probes dunders such as ``__path__`` on every ``from`` import;
        # those are not reads of the rule, so only named constants are recorded.
        if not (name.startswith("__") and name.endswith("__")):
            self.__dict__["reads"].append(name)
        return getattr(self.__dict__["_real"], name)


def _install_delegates(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    old = _CountingModule(original)
    new = _CountingModule(_corrected())
    monkeypatch.setitem(sys.modules, ORIGINAL_MODULE, old)
    monkeypatch.setitem(sys.modules, CORRECTED_MODULE, new)
    return old, new


def test_the_deferred_edge_band_import_reads_the_corrected_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import utils.edge_tier as edge_tier_module

    old, new = _install_delegates(monkeypatch)
    monkeypatch.setattr(edge_tier_module, "_THRESHOLDS_BY_TARGET", None)
    edge_tier_module.edge_tier(1.0, "ats")

    assert old.reads == []
    assert "EDGE_TIER_THRESHOLDS_BY_TARGET" in new.reads


def test_the_delegate_detects_a_planted_read_of_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planted-violation control: the delegate is not blind."""
    old, _new = _install_delegates(monkeypatch)
    namespace: dict[str, Any] = {}
    exec(f"from {ORIGINAL_MODULE} import EDGE_TIER_THRESHOLDS_BY_TARGET", namespace)
    assert old.reads == ["EDGE_TIER_THRESHOLDS_BY_TARGET"]


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    found = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    found |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return found


def test_no_serving_module_imports_the_original_and_one_imports_the_correction() -> (
    None
):
    """Parsed Import / ImportFrom nodes, so a comment naming the original cannot trip it."""
    imported = {
        path: _imported_modules((REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in SERVING_MODULES
    }
    union = set().union(*imported.values())
    assert len(imported) == 4
    assert ORIGINAL_MODULE not in union, {
        p: sorted(m for m in mods if "cold_start" in m) for p, mods in imported.items()
    }
    assert CORRECTED_MODULE in union


def test_the_import_scan_flags_a_planted_import_of_the_original() -> None:
    planted = f"import os\nfrom {ORIGINAL_MODULE} import CHAIN_FIT_BIAS_2026\n"
    assert ORIGINAL_MODULE in _imported_modules(planted)


def _reads_phase31_record(path: Path) -> bool:
    return "profitability_2025_verdict" in Path(path).as_posix()


def test_the_live_chain_fit_path_is_the_corrected_record() -> None:
    from backtest.corrected_ev_chain_constants import CORRECTED_CHAIN_FIT_RECORD_PATH

    assert weekly.DEFAULT_CHAIN_FIT_PATH.as_posix() == CORRECTED_CHAIN_FIT_RECORD_PATH
    assert not _reads_phase31_record(weekly.DEFAULT_CHAIN_FIT_PATH)
    assert _reads_phase31_record(
        Path("outputs") / "p31" / "profitability_2025_verdict.json"
    ), "the planted Phase-31 path must be flagged by the same predicate"
