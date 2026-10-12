"""No closing row can enter a decision, a prediction or a gold input (Phase 34, LDGR-07, T-34-49).

The closing capture (Plan 34-14) writes post-lock, pre-kickoff lines into the SAME accumulating
odds store the decision readers read. They stay out of every decision because every reader judges
a line's admissibility on its recorded capture instant, ``created_at``, against the game's lock --
never on ``snapshot_ts``, which a closing row also carries as the lock (its documented meaning).

Each reader that decides anything is driven here over a store holding a pre-lock DECISION row and
a later post-lock CLOSING row for one game, both produced by the real transform, and must admit the
decision row:

* the market-anchor admission (``features.market_anchors.admissible_market_rows``);
* the prediction dedupe (``scripts.generate_current_week_predictions.load_market_data``);
* the weekly candidate build, driven through the REAL ``build_weekly_candidates`` over the
  stored odds (only the model scorer stands in), where the admissible decision row must outrank
  the later inadmissible closing row. Was: a hand copy of its two dedupe steps, which would have
  kept passing had production dropped ``locks=`` (34-REVIEW WR-21).

And a game whose ONLY rows are closing rows is suppressed by the selector as ``stale_line`` --
never priced (research Pitfall 8).

Every test runs on a sandboxed store under ``tmp_path``. No network.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import backtest.diagnose as diagnose_module
import data.storage as storage_mod
from backtest.bet_selector import BetSelector
from backtest.ou_divergence import dedupe_odds_by_book_preference
from backtest.selector_strategies import OUStrategy
from backtest.weekly_bet_list import CANONICAL_TARGETS, build_weekly_candidates
from features.market_anchors import admissible_market_rows
from scripts.ingest_odds import OddsDataIngester
from utils.date_utils import kickoff_wall_clock_et
from utils.game_lock import game_lock, lock_frame

GAME_ID = "2026_W06_DAL@PHI"
SEASON = 2026
WEEK = 6
KICKOFF = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)  # Sunday 1:00 PM ET
LOCK = game_lock(KICKOFF)  # Saturday 18:00 ET
DECISION_CAPTURE = datetime(2026, 10, 10, 21, 0, tzinfo=UTC)  # Saturday 17:00 ET
CLOSING_CAPTURE = KICKOFF - timedelta(minutes=30)

# The decision board has Philadelphia -3.5 (stored +3.5) and 44.5; by the close the line moved to
# -6.5 (stored +6.5) and 47.5, so a reader that took the closing row would show it.
DECISION_SPREAD, DECISION_TOTAL = 3.5, 44.5
CLOSING_SPREAD, CLOSING_TOTAL = 6.5, 47.5


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "season": SEASON,
                "week": WEEK,
                "home_team": "PHI",
                "away_team": "DAL",
                "kickoff_et": KICKOFF,
            }
        ]
    )


def _event(home_point: float, total: float) -> dict[str, Any]:
    """One Odds API v4 event at DraftKings (a fill-v1 preferred book)."""
    return {
        "id": "event0",
        "commence_time": "2026-10-11T17:00:00Z",
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Philadelphia Eagles", "price": -170},
                            {"name": "Dallas Cowboys", "price": 145},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {
                                "name": "Philadelphia Eagles",
                                "price": -110,
                                "point": home_point,
                            },
                            {
                                "name": "Dallas Cowboys",
                                "price": -110,
                                "point": -home_point,
                            },
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": total},
                            {"name": "Under", "price": -110, "point": total},
                        ],
                    },
                ],
            }
        ],
    }


def _captured_rows(
    captured_at: datetime, home_point: float, total: float, kind: str
) -> pd.DataFrame:
    """Rows produced by the REAL transform and schema validation, as the store receives them."""
    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.sportsbook_priority = []
    frame = ingester.transform_odds_data(
        [_event(home_point, total)],
        schedule=_schedule(),
        locks={GAME_ID: LOCK},
        captured_at=captured_at,
        capture_kind=kind,
    )
    validated = ingester.validate_odds_data(frame)
    assert len(validated) == 1, f"the {kind} fixture row was not produced"
    return validated


def _decision_rows() -> pd.DataFrame:
    return _captured_rows(
        DECISION_CAPTURE, -DECISION_SPREAD, DECISION_TOTAL, "decision"
    )


def _closing_rows() -> pd.DataFrame:
    return _captured_rows(CLOSING_CAPTURE, -CLOSING_SPREAD, CLOSING_TOTAL, "closing")


def _store(base: Path, *frames: pd.DataFrame) -> pd.DataFrame:
    """Write *frames* through the real accumulating writer under *base*; return the stored file."""
    for frame in frames:
        storage_mod.append_odds_captures(frame, base_path=base)
    return pd.read_parquet(base / "silver" / "odds_snapshot.parquet")


@pytest.fixture
def both_rows(tmp_path: Path) -> pd.DataFrame:
    """A stored decision row and a stored closing row for one game."""
    stored = _store(tmp_path, _decision_rows(), _closing_rows())
    assert len(stored) == 2, (
        "both captures must be in the store for these tests to mean much"
    )
    return stored


@pytest.fixture
def closing_only(tmp_path: Path) -> pd.DataFrame:
    """A store whose only row for the game is a closing row."""
    return _store(tmp_path, _closing_rows())


def _locks() -> pd.Series:
    return lock_frame(_schedule())


class TestTheFixtureIsLoadBearing:
    """Non-vacuity: the closing row is the one a reader without the lock fence would take."""

    def test_the_closing_row_wins_without_the_lock(
        self, both_rows: pd.DataFrame
    ) -> None:
        chosen = dedupe_odds_by_book_preference(both_rows, locks=None)
        assert chosen.iloc[0]["spread"] == CLOSING_SPREAD
        assert pd.Timestamp(chosen.iloc[0]["snapshot_ts"]) == pd.Timestamp(LOCK), (
            "a closing row carries snapshot_ts = lock, so snapshot_ts alone cannot tell them apart"
        )


class TestEveryDecisionReaderIgnoresClosingRows:
    """Each reader that decides anything admits the decision row and never the closing row."""

    def test_market_anchor_ignores_closing_rows(self, both_rows: pd.DataFrame) -> None:
        admitted = admissible_market_rows(both_rows, _locks())
        assert len(admitted) == 1
        assert admitted.iloc[0]["spread"] == DECISION_SPREAD
        assert admitted.iloc[0]["information_time"] == pd.Timestamp(DECISION_CAPTURE)

    def test_market_anchor_admits_nothing_from_a_closing_only_store(
        self, closing_only: pd.DataFrame
    ) -> None:
        assert admissible_market_rows(closing_only, _locks()).empty

    def test_prediction_dedupe_ignores_closing_rows(
        self, tmp_path: Path, both_rows: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.generate_current_week_predictions import load_market_data

        _write_lake(tmp_path, both_rows)
        monkeypatch.chdir(tmp_path)
        market = load_market_data([GAME_ID])
        assert len(market) == 1
        assert market.iloc[0]["spread"] == DECISION_SPREAD
        assert market.iloc[0]["total"] == DECISION_TOTAL

    def test_prediction_dedupe_publishes_no_line_from_a_closing_only_store(
        self,
        tmp_path: Path,
        closing_only: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from scripts.generate_current_week_predictions import load_market_data

        _write_lake(tmp_path, closing_only)
        monkeypatch.chdir(tmp_path)
        assert load_market_data([GAME_ID]).empty

    def test_weekly_candidate_dedupe_ignores_closing_rows(
        self,
        tmp_path: Path,
        both_rows: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        candidates = _weekly_candidates(tmp_path, monkeypatch)
        assert sorted(candidates["target"]) == sorted(CANONICAL_TARGETS)
        by_target = candidates.set_index("target")
        assert by_target.loc["ats", "closing_spread"] == DECISION_SPREAD
        assert by_target.loc["ou", "closing_total"] == DECISION_TOTAL
        for target in CANONICAL_TARGETS:
            assert by_target.loc[target, "snapshot_ts"] == pd.Timestamp(
                DECISION_CAPTURE
            ), (
                "the selector's freshness fence must be handed the decision row's capture instant"
            )


def _write_lake(base: Path, odds: pd.DataFrame) -> None:
    """The two files ``load_market_data`` reads by relative path, under ``base/data/silver``."""
    silver = base / "data" / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(silver / "odds_snapshot.parquet")
    _schedule()[["game_id", "kickoff_et"]].to_parquet(silver / "games.parquet")


# What the stand-in scorer returns per target: the one seam that would otherwise need models.
_MODEL_VALUES: dict[str, dict[str, float]] = {
    "wp": {"model_prob": 0.61},
    "ats": {"model_spread": -3.5},
    "ou": {"model_total": 41.0},
}


def _weekly_candidates(base: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """The REAL ``build_weekly_candidates`` over the odds stored under ``base/silver``.

    A one-game schedule and a one-row gold matrix per target are written beside the store, and
    only ``backtest.diagnose.score_deployed_artifacts`` stands in (as in
    ``tests/unit/test_weekly_candidates_exclusion.py``). The lock fence, the dedupe and the
    capture-instant hand-off to the selector are production's own.
    """
    silver, gold = base / "silver", base / "gold"
    _schedule().to_parquet(silver / "games.parquet", index=False)
    gold.mkdir()
    for target in CANONICAL_TARGETS:
        _schedule()[["game_id", "season", "week"]].to_parquet(
            gold / f"features_{target}.parquet", index=False
        )

    def _score(target: str, *, gold_df: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        scored = gold_df[["game_id", "season", "week"]].copy()
        for column, value in _MODEL_VALUES[target].items():
            scored[column] = value
        return scored

    monkeypatch.setattr(diagnose_module, "score_deployed_artifacts", _score)
    candidates, _spine = build_weekly_candidates(
        SEASON, WEEK, artifacts_dir=base / "artifacts", gold_dir=gold, silver_dir=silver
    )
    return candidates


def _ou_verdict(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Run the game's REAL O/U weekly candidate through the real selector."""
    candidates = _weekly_candidates(base, monkeypatch)
    row = candidates.loc[candidates["target"] == "ou"].iloc[0]
    gameday = kickoff_wall_clock_et(KICKOFF).date().isoformat()
    candidate = {
        "game_id": GAME_ID,
        "season": SEASON,
        "week": WEEK,
        "target": "ou",
        "model_total": 41.0,
        "closing_total": float(row["closing_total"]),
        "sportsbook": row["sportsbook"],
        "is_live": bool(row["is_live"]),
        "snapshot_ts": row["snapshot_ts"],
        "total_over_ju": row["total_over_ju"],
        "total_under_ju": row["total_under_ju"],
        "gameday": gameday,
    }
    selector = BetSelector(
        frozen_sd=13.0,
        season_bias_by_season={SEASON: 0.0},
        strategies=[OUStrategy(13.0, {SEASON: 0.0})],
    )
    result = selector.select(
        [candidate],
        scheduled_games=[
            {"game_id": GAME_ID, "season": SEASON, "week": WEEK, "gameday": gameday}
        ],
    )
    reasons = [record["rejection_reason"] for record in result.rejected]
    return reasons, result.selected, result.unfiltered[0]


class TestAClosingOnlyGameIsNeverPriced:
    """``keep_inadmissible=True`` keeps the closing row as a placeholder; the fence suppresses it."""

    def test_closing_only_game_is_suppressed_stale_line(
        self,
        tmp_path: Path,
        closing_only: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reasons, selected, record = _ou_verdict(tmp_path, monkeypatch)
        assert reasons == ["stale_line"]
        assert selected == []
        assert record["per_bet_ev"] is None, "a stale line is never priced"
        assert record["selected_odds"] is None
        assert record["kelly_stake"] == 0.0

    def test_a_game_with_a_decision_row_is_not_stale(
        self,
        tmp_path: Path,
        both_rows: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Control: the same selector reaches a non-stale verdict once a pre-lock line exists."""
        reasons, _selected, _record = _ou_verdict(tmp_path, monkeypatch)
        assert "stale_line" not in reasons
