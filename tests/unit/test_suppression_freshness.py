"""Suppression inside the selector: the candidate universe, the taxonomy, and freshness.

Phase 31, plan 31-09 (SPEC R6/R7, D31-17/18/19, PROD-03). Two claims are under test here and they
are deliberately in one module, because they are one seam:

1. **The universe is complete.** Every scheduled game times every REGISTERED target produces a
   record -- live or suppressed -- so "never silently dropped" is a structural property of the
   selector rather than a reconciliation pass that can disagree with it. Every non-live record
   carries exactly one reason drawn from ``REJECTION_REASONS`` and never a free-form string.

2. **Admissibility is measured once, per game, in Eastern.** Each game's lock comes from the ONE
   rule, ``utils.game_lock`` -- the same rule the ingest uses to STAMP ``snapshot_ts`` -- so the
   value written and the value compared come from one rule. At-lock is ADMISSIBLE; one second
   after the lock is not (D33.2-01). Plan 33.2-02 reversed this module's former staleness
   direction on purpose; see the section header above ``TestAdmissibilityBoundary``.

WHY THE REGISTRY LENGTH IS READ AT RUNTIME (REVIEW-REGISTRY).

  This plan and plan 31-10 are both in wave 4, and it is 31-10 that registers ``ATSStrategy`` and
  ``WPStrategy``. A test asserting "three registered targets produce 48 records" would therefore
  fail on wave order alone, through no defect in either plan. So the identity asserted below is
  ``len(scheduled_games) * len(selector.strategies)`` with BOTH factors read from the objects
  under test, and it is proven at three cardinalities: the production default registry (whatever
  its size when this runs), a test-local registry of three Protocol-conforming strategies, and a
  fourth added to the same unmodified assertion.

WHY THE TEST STRATEGIES ARE LEGITIMATE.

  Plan 31-06 forbids a registerable PRODUCTION strategy whose methods are unimplemented -- a
  registered strategy that cannot decide is worse than a missing one. That restriction is on the
  production classes. ``_MiniStrategy`` below is test-local, fully implemented, and checked
  against the ``TargetStrategy`` Protocol by ``isinstance`` before it is used, which is exactly
  the conformance guarantee that makes a stub safe here.

Run this module:  .venv/Scripts/python.exe -m pytest tests/unit/test_suppression_freshness.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.bet_selector import REJECTION_REASONS, BetSelector
from backtest.selector_strategies import OUStrategy, TargetStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = REPO_ROOT / "backtest" / "bet_selector.py"

# ---------------------------------------------------------------------------
# Fixture constants (kept explicit so a silent drift is caught)
# ---------------------------------------------------------------------------

_SEASON = 2023
_WEEK = 1
_FROZEN_SD = 13.0
_SEASON_BIAS = {_SEASON: -1.0}
_BANKROLL = 10_000.0

# A Sunday kickoff and a snapshot on the Friday evening before it. The game locks on the Saturday
# (18:00 ET), so this snapshot is a day BEFORE the lock and admissible -- every fixture week here
# is admissible unless a test deliberately moves the snapshot past the lock.
_SUNDAY_GAMEDAY = "2023-09-10"
_FREEZE_AT_SUNDAY = "2023-09-08T18:00:00-04:00"

# The members the taxonomy carries. Enumerated rather than sampled: the taxonomy is the ONE list
# the page maps to labels, and a member appearing without a label would render a blank cell. Plan
# 31-09 took it from four to eight; plan 31-10 appended ``no_bet_side`` (D31-05) for the two
# targets that have no eligibility gate to fail and nothing priced to judge on expected value.
_EXPECTED_REASONS = (
    "not_subpop",
    "ev_below_floor",
    "real_odds_failed",
    "zero_kelly_stake",
    "stale_line",
    "missing_snapshot",
    "missing_prediction",
    "ev_not_finite",
    "no_bet_side",
    "no_honest_ev_floor",
    # Plan 33.2-26: a win bet's second test (D33.2-11) and a target with no honest threshold.
    "edge_below_threshold",
    "no_honest_edge_threshold",
)

# The commit the guard's body is pinned to. ``assert_real_odds`` is the LOCKED provenance guard,
# so its body is compared against this anchor rather than against "the tests still pass".
#
# Was: ``94f5f56`` (the Phase-31 wave-3 HEAD). RE-ANCHORED, deliberately and in its own commit, to
# ``900430f`` -- Plan 33.2-27 Task 1's GREEN commit, which widened the allowlist to the real live
# book names and removed the ``is_live`` arm (a genuine live capture tripped both). The pin keeps
# its purpose: any FURTHER edit to the guard's body still fails here until it is re-anchored on
# purpose.
_PROVENANCE_ANCHOR_COMMIT = "900430f"

# The suppression-label table lives in the phase's UI-SPEC, which sits under ``.planning/``. That
# directory is GITIGNORED by project policy, so it is present in a working tree and absent from a
# fresh clone. The assertion runs wherever the file exists and says plainly why it did not.
_UI_SPEC_PATH = (
    REPO_ROOT
    / ".planning"
    / "phases"
    / "31-ship-the-ev-bet-list-profitability-readout"
    / "31-UI-SPEC.md"
)


# ---------------------------------------------------------------------------
# Test-local strategies (Protocol-conforming, fully implemented)
# ---------------------------------------------------------------------------


class _MiniStrategy:
    """A minimal, fully implemented ``TargetStrategy`` for a test-local target.

    It makes NO claim about how any real target prices a bet; its only job is to put a second and
    third target into the registry so the universe identity can be proven at more than one
    cardinality before plan 31-10 lands the real strategies.

    ``required_prediction_fields`` is OPTIONAL by design. Declared, it names the model-output
    columns explicitly. Omitted (``prediction_fields=None``), the core falls back to the
    ``model_`` naming convention -- and the ``spread`` strategy below deliberately omits it so
    that fallback is exercised rather than assumed.
    """

    def __init__(
        self,
        target: str,
        market_fields: tuple[str, ...],
        prediction_fields: tuple[str, ...] | None,
        p_side: float = 0.60,
    ) -> None:
        self.target = target
        self.required_market_fields = market_fields
        if prediction_fields is not None:
            self.required_prediction_fields = prediction_fields
        self._p_side = p_side

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        return row.get(f"{self.target}_side", "home")

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        return None if bet_side is not None else "not_subpop"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        return "all"

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float]:
        p_side = float(row.get(f"{self.target}_p", self._p_side))
        line = float(row[self.required_market_fields[-1]])
        return p_side, line

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        return {}

    def grade(self, record: dict[str, Any]) -> bool | None:
        return None


def _winner_strategy() -> _MiniStrategy:
    return _MiniStrategy(
        "winner", ("model_win_prob", "ml_home", "ml_away"), ("model_win_prob",)
    )


def _spread_strategy() -> _MiniStrategy:
    # No declared prediction fields: the ``model_`` convention must classify ``model_spread``.
    return _MiniStrategy("spread", ("model_spread", "closing_spread"), None)


def _totals_strategy() -> _MiniStrategy:
    return _MiniStrategy("totals", ("model_total", "closing_total"), ("model_total",))


def _fourth_strategy() -> _MiniStrategy:
    return _MiniStrategy("props", ("model_prop", "closing_prop"), ("model_prop",))


def _mini_registry() -> list[_MiniStrategy]:
    """The three test-local targets, in a fixed order."""
    return [_winner_strategy(), _spread_strategy(), _totals_strategy()]


def _selector(
    strategies: list[Any] | None = None, ev_floor_t: float = 0.0
) -> BetSelector:
    """A BetSelector over ``strategies``; with none supplied it is the PRODUCTION default."""
    return BetSelector(
        frozen_sd=_FROZEN_SD,
        season_bias_by_season=_SEASON_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=_BANKROLL,
        strategies=strategies,
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _game_id(index: int) -> str:
    return f"{_SEASON}_W{_WEEK:02d}_T{index:02d}@H{index:02d}"


def _schedule(
    n_games: int = 16, gameday: str = _SUNDAY_GAMEDAY
) -> list[dict[str, Any]]:
    """A hand-built week of ``n_games`` scheduled games, each carrying its own kickoff date."""
    return [
        {
            "game_id": _game_id(i),
            "season": _SEASON,
            "week": _WEEK,
            "gameday": gameday,
        }
        for i in range(n_games)
    ]


_TARGET_MARKET_VALUES: dict[str, dict[str, Any]] = {
    "winner": {"model_win_prob": 0.61, "ml_home": -130.0, "ml_away": 110.0},
    "spread": {"model_spread": -3.5, "closing_spread": -2.5},
    "totals": {"model_total": 41.0, "closing_total": 45.0},
    "props": {"model_prop": 1.5, "closing_prop": 1.0},
    "ou": {"model_total": 41.0, "closing_total": 45.0},
}


def _candidate(
    game_id: str,
    target: str,
    *,
    snapshot_ts: str | Any = _FREEZE_AT_SUNDAY,
    sportsbook: str | None = "consensus",
    is_live: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """One candidate row for ``target`` on ``game_id``, complete unless overridden.

    An override set to ``None`` REMOVES the column, which is how a missing market value and a
    missing model value are expressed: the row exists and the column does not.
    """
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": _SEASON,
        "week": _WEEK,
        "target": target,
        "snapshot_ts": snapshot_ts,
        "is_live": is_live,
        **_TARGET_MARKET_VALUES[target],
    }
    if sportsbook is not None:
        row["sportsbook"] = sportsbook
    for key, value in overrides.items():
        if value is None:
            row.pop(key, None)
        else:
            row[key] = value
    return row


def _full_week(
    schedule: list[dict[str, Any]], targets: list[str]
) -> list[dict[str, Any]]:
    """A candidate row for every (scheduled game, target) pair."""
    return [
        _candidate(game["game_id"], target) for game in schedule for target in targets
    ]


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------


class TestRejectionTaxonomy:
    """The taxonomy is the ONE exported list of reasons, enumerated in both directions."""

    def test_taxonomy_is_exactly_the_enumerated_members(self) -> None:
        """Enumerated in both directions so a new reason cannot appear unannounced."""
        assert REJECTION_REASONS == _EXPECTED_REASONS
        assert len(set(REJECTION_REASONS)) == len(REJECTION_REASONS)

    def test_the_original_four_keep_their_positions(self) -> None:
        """The taxonomy GREW; it was not rewritten. The Phase-27 four are still first."""
        assert REJECTION_REASONS[:4] == (
            "not_subpop",
            "ev_below_floor",
            "real_odds_failed",
            "zero_kelly_stake",
        )

    def test_every_reason_has_a_ui_spec_label(self) -> None:
        """Each code appears in the UI-SPEC's fixed label table -- no reason renders blank.

        Skips only where ``.planning/`` was never checked out; that directory is gitignored by
        project policy, so its absence is a checkout fact and not a missing assertion.
        """
        if not _UI_SPEC_PATH.exists():
            pytest.skip(
                f"{_UI_SPEC_PATH} is absent; .planning/ is gitignored by project policy, so the "
                "suppression-label table is unavailable in this checkout."
            )
        spec = _UI_SPEC_PATH.read_text(encoding="utf-8")
        for reason in REJECTION_REASONS:
            assert f"`{reason}`" in spec, (
                f"rejection reason {reason!r} has no label row in the UI-SPEC table"
            )


# ---------------------------------------------------------------------------
# The candidate universe
# ---------------------------------------------------------------------------


class TestCandidateUniverse:
    """T-31-39/43c: every scheduled game times every REGISTERED target produces a record."""

    def test_universe_identity_holds_for_the_three_target_registry(self) -> None:
        """16 games x 3 registered strategies == 48 records, both factors read at runtime."""
        schedule = _schedule(16)
        selector = _selector(_mini_registry())
        result = selector.select(
            _full_week(schedule, ["winner", "spread", "totals"]),
            scheduled_games=schedule,
        )

        expected = len(schedule) * len(selector.strategies)
        assert expected == 48, "the fixture no longer expresses the 16 x 3 case"
        assert len(result.unfiltered) == expected
        assert len(result.selected) + len(result.rejected) == expected

    def test_universe_identity_follows_a_fourth_registered_target(self) -> None:
        """The SAME unmodified identity expects 64 once a fourth strategy is registered.

        This is what proves the assertion follows the registry rather than a transcribed count.
        """
        schedule = _schedule(16)
        selector = _selector([*_mini_registry(), _fourth_strategy()])
        result = selector.select(
            _full_week(schedule, ["winner", "spread", "totals", "props"]),
            scheduled_games=schedule,
        )

        expected = len(schedule) * len(selector.strategies)
        assert expected == 64
        assert len(result.unfiltered) == expected

    def test_universe_identity_holds_for_the_production_default_registry(self) -> None:
        """The same week against the PRODUCTION registry, whatever its size when this runs.

        It passes whether or not plan 31-10 has landed: the count is the registry's length times
        the schedule's, never the literal 1 the registry happens to hold today. Only the O/U rows
        are supplied, because O/U is the only target whose candidate columns this fixture can name
        today; any further registered target simply contributes suppressed skeleton records, which
        is precisely the behaviour under test.
        """
        schedule = _schedule(16)
        selector = _selector()
        rows = [_candidate(game["game_id"], "ou") for game in schedule]
        result = selector.select(rows, scheduled_games=schedule)

        assert len(result.unfiltered) == len(schedule) * len(selector.strategies)
        assert len([r for r in result.unfiltered if r["target"] == "ou"]) == len(
            schedule
        )
        assert result.selected, (
            "no O/U bet was selected; the fixture would prove little"
        )

    def test_no_scheduled_game_is_absent_from_the_universe(self) -> None:
        """Completeness stated on the games themselves, not only on a count."""
        schedule = _schedule(16)
        selector = _selector(_mini_registry())
        result = selector.select(
            _full_week(schedule, ["winner", "spread", "totals"]),
            scheduled_games=schedule,
        )

        covered = {(r["game_id"], r["target"]) for r in result.unfiltered}
        assert covered == {
            (game["game_id"], target)
            for game in schedule
            for target in selector.strategies
        }

    def test_a_candidate_outside_the_schedule_is_refused(self) -> None:
        """A candidate for an unscheduled game cannot be reported, so it raises rather than drops."""
        schedule = _schedule(2)
        rows = [
            *_full_week(schedule, ["totals"]),
            _candidate("2023_W01_XX@YY", "totals"),
        ]
        with pytest.raises(ValueError, match="not in the schedule"):
            _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

    def test_a_schedule_without_a_kickoff_date_is_refused(self) -> None:
        """No kickoff date means no per-game freeze, so the freshness fence could not fire."""
        schedule = [
            {"game_id": _game_id(0), "season": _SEASON, "week": _WEEK},
        ]
        with pytest.raises(ValueError, match="gameday"):
            _selector([_totals_strategy()]).select(
                _full_week(_schedule(1), ["totals"]), scheduled_games=schedule
            )


# ---------------------------------------------------------------------------
# Per-target suppression: market gaps and model gaps are different things
# ---------------------------------------------------------------------------


class TestPerTargetSuppression:
    """T-31-42: a market gap is never reported as a model gap."""

    def test_a_total_without_a_moneyline_yields_one_live_row_and_one_suppressed_row(
        self,
    ) -> None:
        """SPEC R6's partial-coverage case, asserted on the SAME game_id.

        The game carries a total and no moneyline. Suppression is per-TARGET, so the totals row is
        live and the winner row is suppressed -- the same game legitimately appears twice.
        """
        schedule = _schedule(1)
        game_id = schedule[0]["game_id"]
        rows = [_candidate(game_id, "totals")]  # no winner candidate at all
        result = _selector([_winner_strategy(), _totals_strategy()]).select(
            rows, scheduled_games=schedule
        )

        by_target = {r["target"]: r for r in result.unfiltered}
        assert set(by_target) == {"winner", "totals"}

        live_ids = {r["game_id"] for r in result.selected}
        assert live_ids == {game_id}
        assert result.selected[0]["target"] == "totals"

        suppressed = [r for r in result.rejected if r["target"] == "winner"]
        assert len(suppressed) == 1
        assert suppressed[0]["game_id"] == game_id
        assert suppressed[0]["rejection_reason"] == "missing_snapshot"

    def test_market_data_without_a_model_output_is_missing_prediction(self) -> None:
        """A model gap is labelled as one, and NO record in that case says missing_snapshot."""
        schedule = _schedule(3)
        rows = [
            _candidate(game["game_id"], "totals", model_total=None) for game in schedule
        ]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

        reasons = {r["rejection_reason"] for r in result.rejected}
        assert reasons == {"missing_prediction"}
        assert "missing_snapshot" not in reasons
        assert result.selected == []

    def test_a_missing_market_column_is_missing_snapshot_not_missing_prediction(
        self,
    ) -> None:
        """The mirror case: the model spoke and the market did not."""
        schedule = _schedule(3)
        rows = [
            _candidate(game["game_id"], "totals", closing_total=None)
            for game in schedule
        ]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

        assert {r["rejection_reason"] for r in result.rejected} == {"missing_snapshot"}

    def test_the_model_prefix_convention_classifies_an_undeclared_strategy(
        self,
    ) -> None:
        """The ``spread`` strategy declares no prediction fields; ``model_spread`` is still one.

        Without the convention its absence would be reported as a market gap, hiding a broken
        prediction path behind an odds-coverage label -- the exact confusion D31-19 forbids.
        """
        schedule = _schedule(2)
        spread = _spread_strategy()
        assert not hasattr(spread, "required_prediction_fields")

        rows = [
            _candidate(game["game_id"], "spread", model_spread=None)
            for game in schedule
        ]
        result = _selector([spread]).select(rows, scheduled_games=schedule)
        assert {r["rejection_reason"] for r in result.rejected} == {
            "missing_prediction"
        }

    def test_every_non_live_record_carries_a_taxonomy_reason(self) -> None:
        """Set containment over the WHOLE result: no free-form string ever reaches a reason."""
        schedule = _schedule(6)
        rows = [
            # A complete winner row, a market gap, a model gap and a post-lock line, mixed.
            _candidate(schedule[0]["game_id"], "winner"),
            _candidate(schedule[1]["game_id"], "winner", ml_home=None),
            _candidate(schedule[2]["game_id"], "winner", model_win_prob=None),
            _candidate(
                schedule[3]["game_id"],
                "winner",
                snapshot_ts="2023-09-09T19:00:00-04:00",
            ),
            _candidate(schedule[4]["game_id"], "winner", winner_side=None),
            _candidate(schedule[5]["game_id"], "winner"),
        ]
        # Remove the side on one row so ``not_subpop`` is reachable too.
        rows[4]["winner_side"] = None
        result = _selector([_winner_strategy()], ev_floor_t=0.0).select(
            rows, scheduled_games=schedule
        )

        emitted = {r["rejection_reason"] for r in result.rejected}
        assert emitted <= set(REJECTION_REASONS)
        assert emitted, "the fixture produced no rejections and would prove nothing"


# ---------------------------------------------------------------------------
# The non-finite refusal
# ---------------------------------------------------------------------------


class TestNonFiniteExpectedValue:
    """T-31-43: a non-finite EV is suppressed, never tiered."""

    def test_a_non_finite_ev_is_suppressed_as_ev_not_finite(self) -> None:
        """A NaN calibrated P(side) yields a NaN EV, which is refused rather than compared.

        A bare ``NaN < floor`` is False, so without this branch the row would fall through the EV
        floor and be BOOKED -- the reason the check exists at all.
        """
        schedule = _schedule(1)
        strategy = _MiniStrategy(
            "totals", ("model_total", "closing_total"), ("model_total",)
        )
        rows = [
            _candidate(schedule[0]["game_id"], "totals", totals_p=float("nan")),
        ]
        result = _selector([strategy]).select(rows, scheduled_games=schedule)

        assert result.selected == []
        assert len(result.rejected) == 1
        assert result.rejected[0]["rejection_reason"] == "ev_not_finite"

    def test_a_suppressed_non_finite_record_carries_no_tier(self) -> None:
        """No tier field is set on it, so a non-finite value can never reach the badge."""
        schedule = _schedule(1)
        rows = [_candidate(schedule[0]["game_id"], "totals", totals_p=float("nan"))]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

        record = result.rejected[0]
        assert "ev_tier" not in record
        assert record.get("ev_tier") is None


# ---------------------------------------------------------------------------
# The provenance frame is narrowed; the provenance CHECK is not
# ---------------------------------------------------------------------------


def _function_source(source: str, name: str) -> str:
    """Return the source segment of the top-level function ``name`` in ``source``."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    msg = f"function {name!r} not found"
    raise AssertionError(msg)


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run one git command in the repo root, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


class TestProvenanceFrameScope:
    """T-31-43b: the skeleton universe narrows the FRAME, and never the CHECK."""

    def test_a_skeleton_week_completes_and_labels_rather_than_raising(self) -> None:
        """Games carrying no market data at all are labelled, not reported as contamination.

        A missing or NaN sportsbook is outside the allowlist by construction, so passing the
        skeleton through ``assert_real_odds`` would RAISE before it could ever be labelled.
        """
        schedule = _schedule(4)
        rows = [_candidate(schedule[0]["game_id"], "totals")]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

        assert len(result.unfiltered) == 4
        skeleton_reasons = {
            r["rejection_reason"]
            for r in result.rejected
            if r["game_id"] != schedule[0]["game_id"]
        }
        assert skeleton_reasons == {"missing_snapshot"}

    def test_a_game_the_model_scored_with_no_odds_row_gets_no_bet_not_a_crash(
        self,
    ) -> None:
        """33.2 review, batch 1a follow-up: one game with NO odds row must not sink the list.

        The weekly candidates are a LEFT join of the scored games onto the odds store, so a game
        the model scored and the market never priced arrives with its model value present and
        its sportsbook, line and snapshot all NaN. A model value is not a provenance claim:
        the row is a market gap, suppressed as ``missing_snapshot`` with its reason, and every
        other game of the week is still decided.
        """
        schedule = _schedule(3)
        nan = float("nan")
        unpriced = _candidate(
            schedule[0]["game_id"],
            "totals",
            sportsbook=nan,
            closing_total=nan,
            snapshot_ts=nan,
        )
        rows = [unpriced, *_full_week(schedule[1:], ["totals"])]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

        by_game = {r["game_id"]: r for r in result.rejected}
        assert by_game[schedule[0]["game_id"]]["rejection_reason"] == "missing_snapshot"
        assert {r["game_id"] for r in result.selected} == {
            game["game_id"] for game in schedule[1:]
        }

    def test_a_disallowed_sportsbook_still_raises(self) -> None:
        """One offending row in the same skeleton week still hard-fails (OUM-06 unchanged).

        Was: ``bovada``, which Plan 33.2-27 Task 1 admitted as a real live book; an unknown name
        keeps the intent.
        """
        schedule = _schedule(4)
        rows = [_candidate(schedule[0]["game_id"], "totals", sportsbook="mock_book")]
        with pytest.raises(ValueError, match="provenance check FAILED"):
            _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)

    def test_an_is_live_row_is_no_longer_refused(self) -> None:
        """The guard's is_live arm was REMOVED by Plan 33.2-27 Task 1, and the narrowing did not.

        Was: ``test_an_is_live_row_still_raises``. A genuine live capture legitimately carries
        ``is_live``; admissibility at the lock is the lock fence's question, not provenance's.
        The intent kept: the frame narrowing does not change what the guard decides -- here, that
        a real book's row passes whether or not it is live.
        """
        schedule = _schedule(4)
        rows = [_candidate(schedule[0]["game_id"], "totals", is_live=True)]
        result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)
        assert len(result.unfiltered) == 4

    def test_assert_real_odds_body_is_unchanged_since_the_anchor_commit(self) -> None:
        """The LOCKED guard's body is byte-identical to the commit this plan started from.

        Compared against git rather than against a copied recording: a recording in this file
        would be a second statement of the guard that could itself be edited.
        """
        if _git("rev-parse", "--git-dir").returncode != 0:
            pytest.skip("not a git checkout; the anchor blob is unavailable")
        anchor = _git("show", f"{_PROVENANCE_ANCHOR_COMMIT}:backtest/bet_selector.py")
        assert anchor.returncode == 0, (
            f"could not read backtest/bet_selector.py at {_PROVENANCE_ANCHOR_COMMIT}: "
            f"{anchor.stderr.strip()}"
        )
        current = SELECTOR_PATH.read_text(encoding="utf-8")
        assert _function_source(current, "assert_real_odds") == _function_source(
            anchor.stdout, "assert_real_odds"
        ), (
            "assert_real_odds changed; it is the LOCKED Phase-27 provenance guard and the "
            "Phase-27/30 record depends on its behaviour"
        )


# ---------------------------------------------------------------------------
# The legacy path is untouched
# ---------------------------------------------------------------------------


class TestLegacyPathUnchanged:
    """Every pre-31-09 caller passes no schedule and must behave exactly as before."""

    def test_without_a_schedule_the_rows_are_the_universe(self) -> None:
        """No expansion happens: one candidate row still produces one record."""
        rows = [
            {
                "game_id": _game_id(0),
                "season": _SEASON,
                "week": _WEEK,
                "model_total": 38.0,
                "closing_total": 45.0,
                "actual": 40.0,
                "sportsbook": "consensus",
                "is_live": False,
            }
        ]
        selector = BetSelector(
            frozen_sd=_FROZEN_SD,
            season_bias_by_season=_SEASON_BIAS,
        )
        result = selector.select(rows)
        assert len(result.unfiltered) == 1
        assert result.unfiltered[0]["target"] == "ou"

    def test_the_production_default_registry_still_registers_ou(self) -> None:
        """A guard on the fixture above: the default registry is the O/U strategy."""
        selector = _selector()
        assert "ou" in selector.strategies
        assert all(isinstance(s, TargetStrategy) for s in selector.strategies.values())
        assert isinstance(
            OUStrategy(frozen_sd=_FROZEN_SD, season_bias_by_season=_SEASON_BIAS),
            TargetStrategy,
        )

    def test_every_test_local_strategy_conforms_to_the_protocol(self) -> None:
        """The stubs are safe BECAUSE they are checked, not because they look complete."""
        for strategy in [*_mini_registry(), _fourth_strategy()]:
            assert isinstance(strategy, TargetStrategy), (
                f"test strategy {strategy.target!r} does not satisfy TargetStrategy"
            )


def test_python_executable_is_available_for_subprocess_tests() -> None:
    """A guard for the timezone test below: it needs a real interpreter path."""
    assert Path(sys.executable).exists()


# ---------------------------------------------------------------------------
# Per-game admissibility (D31-18, T-31-40/41; D33.2-01, Plan 33.2-02)
# ---------------------------------------------------------------------------
#
# THE DIRECTION OF THIS FENCE WAS REVERSED DELIBERATELY. It used to be a STALENESS test on a
# market quote -- at-freeze fresh, strictly BEFORE the freeze stale. It is now the lock rule's
# ADMISSIBILITY test on information time -- at-lock admissible, one second AFTER the lock not
# admissible, and a quote well before the lock admissible because it is old information rather
# than future information (RESEARCH 2.4). Two intents died with the old rule and are deleted by
# ruling rather than rewritten: "a quote strictly before the freeze is stale", and the
# per-week-freeze counterfactual that only existed to defend the Friday rule's Thursday case.

# The kickoffs the per-game rule turns on. 2023-09-14 is a THURSDAY that locks Wednesday
# 2023-09-13; the Sunday games of that same week kick off 2023-09-17 and lock Saturday
# 2023-09-16. A single per-week instant would stamp both with one value.
_THURSDAY_GAMEDAY = "2023-09-14"
_LATER_SUNDAY_GAMEDAY = "2023-09-17"


def _lock_for(gameday: str):
    """That game's own lock, read from the SHARED rule (never re-derived here)."""
    from scripts.ingest_historical_odds import EASTERN, gameday_lock

    return gameday_lock(gameday).astimezone(EASTERN)


def _totals_verdict(
    snapshot: Any, gameday: str = _SUNDAY_GAMEDAY
) -> tuple[list[str], dict[str, Any]]:
    """Run one O/U-shaped candidate through the selector and return (reasons, record)."""
    schedule = _schedule(1, gameday=gameday)
    rows = [_candidate(schedule[0]["game_id"], "totals", snapshot_ts=snapshot)]
    result = _selector([_totals_strategy()]).select(rows, scheduled_games=schedule)
    return [r["rejection_reason"] for r in result.rejected], result.unfiltered[0]


class TestAdmissibilityBoundary:
    """D33.2-01: at-lock is ADMISSIBLE; one second after is not, measured in Eastern."""

    def test_a_snapshot_exactly_at_the_lock_is_admissible(self) -> None:
        """The boundary is asserted DIRECTLY, on the lock instant itself."""
        lock = _lock_for(_SUNDAY_GAMEDAY)
        reasons, record = _totals_verdict(lock)

        assert "stale_line" not in reasons
        assert record["snapshot_ts"] == lock
        assert record["freeze_ts"] == lock

    def test_a_snapshot_one_second_after_the_lock_is_suppressed(self) -> None:
        """One second later flips the verdict, so the assertion above sits ON the boundary."""
        one_second_late = _lock_for(_SUNDAY_GAMEDAY) + timedelta(seconds=1)
        reasons, record = _totals_verdict(one_second_late)

        assert reasons == ["stale_line"]
        assert record["snapshot_ts"] == one_second_late

    def test_a_snapshot_well_before_the_lock_is_admissible_not_stale(self) -> None:
        """Old information is not future information: the retired staleness rule is gone.

        Seventy-two hours before the lock would have been STALE under the retired fence. It
        is admissible now; picking the freshest admissible quote is Plan 33.2-13's subject,
        not a suppression.
        """
        early = _lock_for(_SUNDAY_GAMEDAY) - timedelta(hours=72)
        reasons, _record = _totals_verdict(early)
        assert "stale_line" not in reasons

    def test_the_lock_on_the_record_is_the_shared_rule_value(self) -> None:
        """The published ``freeze_ts`` is the SAME instant the ingest stamps snapshot_ts from.

        The column keeps its published name (HOST-07 owns any rename); its value is the lock.
        """
        _reasons, record = _totals_verdict(_lock_for(_SUNDAY_GAMEDAY))
        assert record["freeze_ts"] == _lock_for(_SUNDAY_GAMEDAY)


class TestPerGameLockIsLoadBearing:
    """T-31-40: each game is judged against its OWN lock, not the week's."""

    def test_a_thursday_game_at_its_own_wednesday_lock_is_admissible(self) -> None:
        thursday_lock = _lock_for(_THURSDAY_GAMEDAY)
        reasons, record = _totals_verdict(thursday_lock, gameday=_THURSDAY_GAMEDAY)

        assert "stale_line" not in reasons
        assert record["freeze_ts"] == thursday_lock

    def test_a_quote_after_the_thursday_lock_is_suppressed_for_that_game_only(
        self,
    ) -> None:
        """The same Saturday quote is post-lock for the Thursday game and pre-lock for Sunday.

        Both halves are asserted so the per-game rule is shown to be LOAD-BEARING: a single
        week-level instant would give these two games the same verdict.
        """
        # Retargeted by Plan 33.2-20 at ``utils.game_lock.is_admissible``, THE one rule.
        # Was: ``scripts.ingest_historical_odds.is_admissible_at_lock``, a second
        # admissibility entry point that never gained a production caller and was deleted
        # rather than kept as a rival answer.
        from utils.game_lock import is_admissible

        saturday_quote = _lock_for(_LATER_SUNDAY_GAMEDAY) - timedelta(hours=1)
        assert saturday_quote > _lock_for(_THURSDAY_GAMEDAY)

        thursday_reasons, _ = _totals_verdict(saturday_quote, gameday=_THURSDAY_GAMEDAY)
        sunday_reasons, _ = _totals_verdict(
            saturday_quote, gameday=_LATER_SUNDAY_GAMEDAY
        )
        assert thursday_reasons == ["stale_line"]
        assert "stale_line" not in sunday_reasons
        assert is_admissible(saturday_quote, _lock_for(_THURSDAY_GAMEDAY)) is False
        assert is_admissible(saturday_quote, _lock_for(_LATER_SUNDAY_GAMEDAY)) is True


class TestSnapshotValueShapes:
    """T-31-41: every stored shape parses to the same instant, and none is string-compared."""

    def test_every_aware_shape_at_the_lock_agrees(self) -> None:
        """The four shapes the live column is known to hold all resolve to one verdict."""
        from scripts.ingest_historical_odds import EASTERN

        at_lock = datetime(2023, 9, 9, 18, 0, tzinfo=EASTERN)
        shapes: list[Any] = [
            at_lock,  # a tz-aware datetime, what the ingest now writes
            pd.Timestamp(at_lock),  # a pandas Timestamp off a DataFrame
            "2023-09-09T18:00:00-04:00",  # the legacy per-season string shape
            "2023-09-09 22:00:00+00:00",  # the non-consensus row's space-separated UTC form
        ]
        for shape in shapes:
            reasons, record = _totals_verdict(shape)
            assert "stale_line" not in reasons, f"{shape!r} was read as inadmissible"
            assert record["snapshot_ts"] == at_lock, (
                f"{shape!r} parsed to another instant"
            )

    def test_every_shape_one_second_late_is_suppressed(self) -> None:
        """The mirror: the same four shapes one second after the lock are all suppressed.

        Without this, a parse that silently returned a constant would pass the test above.
        """
        from scripts.ingest_historical_odds import EASTERN

        late = datetime(2023, 9, 9, 18, 0, 1, tzinfo=EASTERN)
        shapes: list[Any] = [
            late,
            pd.Timestamp(late),
            "2023-09-09T18:00:01-04:00",
            "2023-09-09 22:00:01+00:00",
        ]
        for shape in shapes:
            reasons, _record = _totals_verdict(shape)
            assert reasons == ["stale_line"], f"{shape!r} was read as admissible"

    def test_a_naive_snapshot_raises_rather_than_being_anchored(self) -> None:
        """A snapshot with no timezone is refused by the strict parser, never relabelled."""
        from scripts.ingest_historical_odds import NaiveTimestampError

        with pytest.raises(NaiveTimestampError):
            _totals_verdict("2023-09-09T18:00:00")

    def test_a_complete_row_with_no_snapshot_is_stale_never_assumed_admissible(
        self,
    ) -> None:
        """A row with market data and a kickoff date but no information time is not admissible.

        Owner ruling 2026-09-22 (Option B): only a real recorded capture time counts, so a line
        with none is suppressed as ``stale_line`` -- never priced, never assumed admissible.
        """
        reasons, _record = _totals_verdict(None)
        assert reasons == ["stale_line"]


class TestAdmissibilityIsTimezoneIndependent:
    """T-31-41: the verdict is identical under a non-Eastern PROCESS timezone.

    Run in a SUBPROCESS on purpose. ``time.tzset`` does not exist on Windows and
    ``datetime.astimezone()`` reads the OS zone fixed at process start, so an in-process
    monkeypatch of the timezone is a silent no-op that passes vacuously. The child therefore
    proves its own offset actually differs from Eastern BEFORE it reports any verdict.
    """

    _CHILD = """
import json
from datetime import datetime
from zoneinfo import ZoneInfo

local = datetime.now().astimezone().utcoffset()
eastern = datetime.now(ZoneInfo("America/New_York")).utcoffset()

from backtest.bet_selector import BetSelector

GAME_ID = "2023_W01_T00@H00"
SCHEDULE = [
    {"game_id": GAME_ID, "season": 2023, "week": 1, "gameday": "2023-09-10"},
]
SELECTOR = BetSelector(
    frozen_sd=13.0,
    season_bias_by_season={2023: -1.0},
)


def verdict(snapshot):
    rows = [
        {
            "game_id": GAME_ID,
            "season": 2023,
            "week": 1,
            "target": "ou",
            "model_total": 41.0,
            "closing_total": 45.0,
            "sportsbook": "consensus",
            "is_live": False,
            "snapshot_ts": snapshot,
        }
    ]
    result = SELECTOR.select(rows, scheduled_games=SCHEDULE)
    return {
        "reasons": [r["rejection_reason"] for r in result.rejected],
        "lock": result.unfiltered[0]["freeze_ts"].isoformat(),
        "snapshot": result.unfiltered[0]["snapshot_ts"].isoformat(),
    }


print(
    json.dumps(
        {
            "tz_differs": local != eastern,
            "local_offset": str(local),
            "at_lock": verdict("2023-09-09T18:00:00-04:00"),
            "one_second_late": verdict("2023-09-09T18:00:01-04:00"),
        }
    )
)
"""

    def test_verdicts_are_identical_under_a_foreign_process_timezone(self) -> None:
        """UTC+14 is as far from Eastern as a zone gets; the verdicts do not move."""
        env = {**os.environ, "TZ": "XXX-14"}
        child = subprocess.run(
            [sys.executable, "-c", self._CHILD],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        payload = json.loads(child.stdout.strip().splitlines()[-1])

        # The control, asserted rather than assumed: if the child's own zone did not move, the
        # test proved nothing and must say so rather than pass.
        assert payload["tz_differs"], (
            "the child process's local timezone did not differ from Eastern "
            f"(offset {payload['local_offset']}), so this test would pass vacuously"
        )

        assert "stale_line" not in payload["at_lock"]["reasons"]
        assert payload["one_second_late"]["reasons"] == ["stale_line"]

        # The instants themselves match what this process computes, not merely the verdicts.
        expected_lock = _lock_for(_SUNDAY_GAMEDAY)
        assert payload["at_lock"]["lock"] == expected_lock.isoformat()
        assert payload["at_lock"]["snapshot"] == expected_lock.isoformat()


def _calls_to(tree: ast.AST, attribute: str) -> list[ast.Call]:
    """Every ``<module>.<attribute>(...)`` call in *tree* -- the ATTRIBUTE form.

    The selector reaches the lock rule as ``lock_rule.game_lock(...)`` /
    ``lock_rule.is_admissible(...)`` so the phase's identity delegate can see each call at call
    time (Plan 33.2-02). That parses as an ``ast.Attribute``, not an ``ast.Name``, so a guard
    matching ``node.func.id`` would count ZERO calls on a correct implementation.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


_PLANTED_SECOND_CALL_SITE = """
import utils.game_lock as lock_rule

def first(day):
    return lock_rule.game_lock(day), lock_rule.is_admissible(day, day)

def second(day):
    return lock_rule.game_lock(day), lock_rule.is_admissible(day, day)
"""


class TestOneLockDerivation:
    """D31-17: the selector derives a lock in exactly ONE place, and it is the shared rule."""

    def test_the_lock_rule_is_called_exactly_once(self) -> None:
        """An AST count over the attribute form, so a second derivation cannot hide."""
        tree = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
        calls = _calls_to(tree, "game_lock")
        assert len(calls) == 1, (
            f"the selector derives a lock at {len(calls)} call sites; D31-18 requires "
            "exactly one, and it must be the shared rule"
        )

    def test_the_comparison_is_the_shared_one(self) -> None:
        """The at-lock-is-admissible comparison is not restated here either."""
        tree = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
        assert len(_calls_to(tree, "is_admissible")) == 1

    def test_both_guards_flag_a_planted_second_call_site(self) -> None:
        """NON-VACUITY: the same counters see TWO sites when two exist.

        Without this, a guard that counted nothing at all would still be asserting ``== 1``
        against whatever the helper happened to return.
        """
        planted = ast.parse(_PLANTED_SECOND_CALL_SITE)
        assert len(_calls_to(planted, "game_lock")) == 2
        assert len(_calls_to(planted, "is_admissible")) == 2

    def test_the_selector_contains_no_second_lock_derivation(self) -> None:
        """No zone literal, no weekday arithmetic, no bare local-zone read.

        Each forbidden token is a way a second rule has actually been written before: a
        hard-coded zone, a hand-rolled weekday walk, or a bare ``astimezone()`` that silently
        anchors on whatever zone the process happens to run in.
        """
        source = SELECTOR_PATH.read_text(encoding="utf-8")
        for token in (
            "ZoneInfo(",
            "America/New_York",
            "timedelta(",
            "weekday()",
            "FREEZE_HOUR",
            "LOCK_HOUR",
            "tz_localize",
            "fromisoformat",
            ".astimezone()",
        ):
            assert token not in source, (
                f"backtest/bet_selector.py contains {token!r}, which is how a SECOND lock "
                "derivation gets written; the lock comes from one shared rule (D31-18)"
            )
