"""The letdown flag and the rest-days window read a prior game only once its RESULT existed.

Plan 33.2-14 Task 1 (SPEC R5, D33.2-01). Both selections admit a prior game only when its
END instant -- kickoff plus ``features.provenance.DECLARED_GAME_DURATION``, the same duration
the Elo replay declares -- is at or before the TARGET game's own lock (18:00 ET on the ET
calendar day before its kickoff). The retired comparison used the prior game's KICKOFF.

THIS IS A SEMANTIC BOUNDARY TEST, SYNTHETIC BY DESIGN. On an ordinary NFL schedule a team's
previous game ends days before its next lock, so no real game sits on this boundary and no
assertion here claims that any historical value moves. The pair exists because the letdown
flag reads a RESULT, and the rule must be written as "when did the result exist", which only
the end instant expresses. The real-world case it guards is a RESCHEDULED prior game that
finished after the target game's lock.

EACH CASE IS A PLANTED PAIR:

1. A prior game whose END falls ONE SECOND AFTER the target's lock is NOT admitted: the value
   is byte-identical to the value computed without that game. Under the retired kickoff
   comparison the same game IS admitted (its kickoff is four hours before its end), so this
   case fails on the old semantics and passes on the new -- which is what makes it evidence.
2. The same game moved so its END is exactly AT the lock IS admitted, and the value moves.
   Without this positive control, case 1 would be satisfied by a value that never moves.

The same pair is carried for the rest-days window, which takes the identical rule.

No source scan is used for this property on purpose: a scan for the word "end" is true of
almost every module, and no scan over source text can tell a comparison against a kickoff
from a comparison against an end instant.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from features.contextual import ContextualFeaturesCalculator
from features.provenance import DECLARED_GAME_DURATION
from utils.game_lock import game_lock

SEASON = 2021
TARGET_ID = "2021_W05_NYJ@KC"
PRIOR_ID = "2021_W04_BUF@KC"
TARGET_KICKOFF = pd.Timestamp("2021-10-10 17:00", tz="UTC")  # Sunday 13:00 ET
LOCK = pd.Timestamp(game_lock(TARGET_KICKOFF))  # Saturday 18:00 ET
ONE_SECOND = timedelta(seconds=1)

#: A far-future frame-wide "now", as production used to pass: it admits everything played.
PRODUCTION_NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _prior_kickoff(end: pd.Timestamp) -> pd.Timestamp:
    """The kickoff that puts a game's END at *end*, in UTC like every silver kickoff."""
    return (end - DECLARED_GAME_DURATION).tz_convert("UTC")


# ---------------------------------------------------------------------------
# The letdown flag.
# ---------------------------------------------------------------------------


def _schedule(prior_end: pd.Timestamp | None) -> pd.DataFrame:
    """KC (1700) hosts a weak NYJ (1450). The planted prior game: KC BEAT a strong BUF
    (1650, at least one Elo step above NYJ) -- a textbook letdown spot if admitted."""
    rows = [
        {
            "game_id": TARGET_ID,
            "season": SEASON,
            "week": 5,
            "home_team": "KC",
            "away_team": "NYJ",
            "home_elo_pre": 1700.0,
            "away_elo_pre": 1450.0,
            "kickoff_et": TARGET_KICKOFF,
            "home_score": None,
            "away_score": None,
        }
    ]
    if prior_end is not None:
        rows.append(
            {
                "game_id": PRIOR_ID,
                "season": SEASON,
                "week": 4,
                "home_team": "KC",
                "away_team": "BUF",
                "home_elo_pre": 1690.0,
                "away_elo_pre": 1650.0,
                "kickoff_et": _prior_kickoff(prior_end),
                "home_score": 31.0,
                "away_score": 20.0,
            }
        )
    return pd.DataFrame(rows)


def _letdown(prior_end: pd.Timestamp | None) -> float:
    calculator = ContextualFeaturesCalculator()
    schedule = _schedule(prior_end)
    target = schedule[schedule["game_id"] == TARGET_ID]
    locks = pd.Series([LOCK], index=pd.Index([TARGET_ID], name="game_id"), name="lock")
    flags = calculator._derive_spot_flags(target, schedule, locks)
    return flags[TARGET_ID]["home_letdown_spot"]


class TestTheLetdownReadsAPriorResultOnlyOnceItExisted:
    def test_a_prior_game_ending_one_second_after_the_lock_changes_nothing(
        self,
    ) -> None:
        without = _letdown(None)
        late = _letdown(LOCK + ONE_SECOND)
        assert late == without, (
            "a prior game whose result arrived one second after the lock moved the "
            f"letdown flag ({without} -> {late}); its KICKOFF was four hours earlier, so "
            "this is the retired kickoff comparison"
        )

    def test_the_same_game_ending_exactly_at_the_lock_moves_the_flag(self) -> None:
        """POSITIVE CONTROL: at-lock is admissible (<=), and the flag is load-bearing."""
        assert _letdown(None) == 0.0
        assert _letdown(LOCK) == 1.0

    def test_the_boundary_is_the_end_not_the_kickoff(self) -> None:
        """The late game's kickoff is BEFORE the lock; only its end is after it."""
        assert _prior_kickoff(LOCK + ONE_SECOND) < LOCK < LOCK + ONE_SECOND


# ---------------------------------------------------------------------------
# The rest-days window, through the builder's public path.
# ---------------------------------------------------------------------------


def _games(prior_end: pd.Timestamp | None) -> pd.DataFrame:
    rows = [
        {
            "game_id": TARGET_ID,
            "season": SEASON,
            "week": 5,
            "home_team": "KC",
            "away_team": "NYJ",
            "kickoff_et": TARGET_KICKOFF,
            "stadium_id": "KAN00",
            "neutral_site": False,
        }
    ]
    if prior_end is not None:
        rows.append(
            {
                "game_id": PRIOR_ID,
                "season": SEASON,
                "week": 4,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": _prior_kickoff(prior_end),
                "stadium_id": "KAN00",
                "neutral_site": False,
            }
        )
    return pd.DataFrame(rows)


def _home_rest(
    prior_end: pd.Timestamp | None, monkeypatch: pytest.MonkeyPatch
) -> float:
    calculator = ContextualFeaturesCalculator()
    # The spot flags reload the silver schedule, which holds none of these synthetic games;
    # an empty schedule keeps the case independent of the local data lake.
    monkeypatch.setattr(
        calculator, "_load_full_season_schedule", lambda seasons: pd.DataFrame()
    )
    built = calculator.build_features(_games(prior_end), PRODUCTION_NOW)
    return float(built.set_index("game_id").loc[TARGET_ID, "home_rest_days"])


class TestTheRestWindowTakesTheSameRule:
    def test_a_prior_game_ending_one_second_after_the_lock_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        without = _home_rest(None, monkeypatch)
        late = _home_rest(LOCK + ONE_SECOND, monkeypatch)
        assert late == without == 7.0, (
            f"rest days moved {without} -> {late} for a prior game whose result arrived "
            "one second after the lock"
        )

    def test_the_same_game_ending_exactly_at_the_lock_moves_the_rest_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSITIVE CONTROL: the at-lock game is read (a 23-hour gap is 0 whole days)."""
        assert _home_rest(LOCK, monkeypatch) == 0.0

    def test_the_provenance_reports_the_admitted_game_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """information_times names the END of the game actually read, and nothing late."""
        calculator = ContextualFeaturesCalculator()
        monkeypatch.setattr(
            calculator, "_load_full_season_schedule", lambda seasons: pd.DataFrame()
        )
        for prior_end, expected in ((LOCK, LOCK), (LOCK + ONE_SECOND, None)):
            games = _games(prior_end)
            calculator.build_features(games, PRODUCTION_NOW)
            frame = calculator.information_times(games).set_index("game_id")
            when = frame.loc[TARGET_ID, "information_time"]
            if expected is None:
                assert frame.loc[TARGET_ID, "basis"] == "no_information"
                assert pd.isna(when)
            else:
                assert frame.loc[TARGET_ID, "basis"] == "per_row"
                assert pd.Timestamp(when) == expected


class TestASpotFlagFailureIsNeverANeutralZero:
    """33.2 review B WR-09: a broken spot-flag input must not read as "no spot"."""

    def test_a_bug_in_the_derivation_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard used to wrap the derivation too, so a renamed column became 0.0 flags."""
        calculator = ContextualFeaturesCalculator()
        monkeypatch.setattr(
            calculator, "_load_full_season_schedule", lambda seasons: pd.DataFrame()
        )

        def broken(*args: object, **kwargs: object) -> dict:
            raise KeyError("home_elo_pre")

        monkeypatch.setattr(calculator, "_derive_spot_flags", broken)
        with pytest.raises(KeyError, match="home_elo_pre"):
            calculator.build_features(_games(None), PRODUCTION_NOW)

    def test_an_unloadable_schedule_leaves_the_flags_unknown_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils.exceptions import DataIngestionError

        calculator = ContextualFeaturesCalculator()

        def unreadable(seasons: list[int]) -> pd.DataFrame:
            raise DataIngestionError("elo_game_snapshots missing")

        monkeypatch.setattr(calculator, "_load_full_season_schedule", unreadable)
        built = calculator.build_features(_games(None), PRODUCTION_NOW)
        row = built.set_index("game_id").loc[TARGET_ID]
        for column in (
            "home_look_ahead_spot",
            "away_look_ahead_spot",
            "home_letdown_spot",
            "away_letdown_spot",
        ):
            assert pd.isna(row[column]), f"{column} = {row[column]!r}, not unknown"
