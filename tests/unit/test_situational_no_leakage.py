"""Leakage proof for the situational-spots signal (SIG-03 / SIG-06 / SC3).

Following the `tests/unit/test_elo_no_leakage.py` convention. The look-ahead /
letdown / off-bye additions land in `features/contextual.py` in Plan 28-04, and
this module makes the D-18a 3-part standard LIVE for the situational group.

Leakage contract (D-16): the next-opponent identity and the pre-freeze Elo
ratings are NOT leakage (the schedule and ratings are known at the Friday
freeze); only a future game *result* is future information. The withhold-future
test therefore reveals future game RESULTS and asserts the spot flags are
unchanged.

Parts:
  1. Time-fence -- the new spots respect the `as_of_datetime` cutoff (a prior
     game's RESULT only counts once it has been played before the freeze).
  2. Withhold-future byte-unchanged -- reveal future game RESULTS, assert the
     look-ahead/letdown flags are byte-identical.
  3. off_bye derivation -- `off_bye = 1.0 if rest_days >= 13 else 0.0`.
  4. Full-schedule sourcing (review #4) -- a target-week (`--current-week`)
     build still derives the spots from the FULL season schedule, NOT the
     target-week-filtered games_df that would collapse next-/prior-opponent
     context to a degenerate default.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from features.contextual import ContextualFeaturesCalculator

# Minimal venues so the calculator instantiates without the on-disk JSON; the
# spot derivation never touches venue data, and build_features only needs the
# home venue (KC) for the rows under test.
_MOCK_VENUES = {
    "venues": [
        {
            "venue_id": "kc_stadium",
            "venue_name": "KC Stadium",
            "city": "Kansas City",
            "state": "MO",
            "country": "USA",
            "latitude": 39.049,
            "longitude": -94.484,
            "elevation_ft": 909,
            "roof_type": "outdoor",
            "surface": "Bermuda Grass",
            "capacity": 76416,
            "climate_zone": "humid_continental",
            "timezone": "America/Chicago",
            "home_teams": ["KC"],
        },
    ]
}


@pytest.fixture
def calc():
    """A ContextualFeaturesCalculator with mock venues data."""
    with patch.object(
        ContextualFeaturesCalculator, "_load_venues_data", return_value=_MOCK_VENUES
    ):
        return ContextualFeaturesCalculator()


def _make_trap_schedule(*, reveal_future: bool = True) -> pd.DataFrame:
    """A 3-game KC schedule with a known look-ahead/letdown setup, grounded in
    the real raw silver Elo scale (~1150-1840, std ~125).

    Shape:
      * W1 KC(1720) vs CAR(1480): CAR is weak (240 Elo below KC) and next week
        KC plays the much stronger BUF -> a classic look-ahead trap.
      * W2 KC(1725) vs BUF(1690): KC BEATS the strong BUF (28-24).
      * W3 KC(1710) vs DEN(1500): DEN is weak (210 below KC) the week AFTER the
        emotional BUF win -> a classic letdown.

    A BUF week-1 game (BUF 1700 vs NYJ 1690) is also included so the look-ahead
    can read BUF's FREEZE-KNOWN (entering-week-1) Elo rather than BUF's
    entering-week-2 Elo, which would embed BUF's week-1 result -- future at KC's
    week-1 freeze (WR-01). BUF is not in a spot in its own week-1 game (NYJ is
    not weak), so it contributes only the freeze-known rating lookup.

    Args:
        reveal_future: When False, the W2 and W3 RESULTS are withheld (NaN
            scores) -- the leakage-sensitive future information. The schedule
            (identity) and pre-freeze Elo are always present.

    Returns:
        A 4-row schedule DataFrame with raw Elo, kickoff, and results.
    """
    w2_home, w2_away = (28.0, 24.0) if reveal_future else (np.nan, np.nan)
    w3_home, w3_away = (31.0, 17.0) if reveal_future else (np.nan, np.nan)
    return pd.DataFrame(
        {
            "game_id": ["TRAP_W1", "BUF_W1", "TRAP_W2", "TRAP_W3"],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 3],
            "home_team": ["KC", "BUF", "KC", "KC"],
            "away_team": ["CAR", "NYJ", "BUF", "DEN"],
            "home_elo_pre": [1720.0, 1700.0, 1725.0, 1710.0],
            "away_elo_pre": [1480.0, 1690.0, 1690.0, 1500.0],
            "kickoff_et": pd.to_datetime(
                [
                    "2023-09-10T13:00:00",
                    "2023-09-10T13:00:00",
                    "2023-09-17T13:00:00",
                    "2023-09-24T13:00:00",
                ]
            ),
            "home_score": [27.0, 20.0, w2_home, w3_home],
            "away_score": [13.0, 17.0, w2_away, w3_away],
        }
    )


# Freeze cutoffs relative to the trap schedule.
_AS_OF_AFTER_W1 = datetime(2023, 9, 12, 18, 0)  # before W2 and W3
_AS_OF_AFTER_W2 = datetime(2023, 9, 20, 18, 0)  # after W2, before W3


class TestSituationalTimeFence:
    """Part 1: the new spots respect the `as_of_datetime` cutoff."""

    def test_letdown_respects_as_of_cutoff(self, calc):
        """The letdown's "beat last week" component reads a prior RESULT, so it
        must only fire once that prior game has been played before the freeze.

        For W3 (KC vs weak DEN, after beating strong BUF in W2):
          * freeze AFTER W2 -> the BUF win is known -> letdown fires.
          * freeze BEFORE W2 -> the BUF result is future -> letdown withheld.
        """
        schedule = _make_trap_schedule()
        target = schedule[schedule["week"] == 3]

        flags_after = calc._derive_spot_flags(target, schedule, _AS_OF_AFTER_W2)
        flags_before = calc._derive_spot_flags(target, schedule, _AS_OF_AFTER_W1)

        assert flags_after["TRAP_W3"]["home_letdown_spot"] == 1.0
        assert flags_before["TRAP_W3"]["home_letdown_spot"] == 0.0


class TestSituationalWithholdFuture:
    """Part 2: revealing future RESULTS must not move the flags."""

    def test_revealing_future_results_does_not_change_flags(self, calc):
        """Build the spot flags with the future (>= freeze) game RESULTS absent
        and again with them revealed; the flags must be byte-identical.

        Only the future *result* is leakage -- the next-opponent identity and
        pre-freeze Elo are known at the freeze. With the freeze set just after
        W1, every flag is independent of the withheld W2/W3 results (look-ahead
        never reads a result; the W3 letdown is gated off because W2 has not
        been played yet), so the two builds must match exactly.
        """
        target = _make_trap_schedule()  # emit flags for all 3 games

        sched_hidden = _make_trap_schedule(reveal_future=False)
        sched_revealed = _make_trap_schedule(reveal_future=True)

        flags_hidden = calc._derive_spot_flags(target, sched_hidden, _AS_OF_AFTER_W1)
        flags_revealed = calc._derive_spot_flags(
            target, sched_revealed, _AS_OF_AFTER_W1
        )

        assert flags_hidden == flags_revealed
        # And the proof is non-trivial: the W1 look-ahead trap actually fires.
        assert flags_revealed["TRAP_W1"]["home_look_ahead_spot"] == 1.0


class TestSituationalFutureEloNoLeak:
    """Part 2b (WR-01): the look-ahead must not read the next opponent's W+1
    pre-game Elo, which embeds the as-yet-unplayed week-W result."""

    def test_perturbing_future_elo_pre_does_not_change_look_ahead(self, calc):
        """Build the W1 look-ahead flag, then perturb the next opponent's (BUF)
        pre-game Elo ENTERING week 2 -- a value that incorporates BUF's week-1
        result, future relative to KC's week-1 Friday freeze. A leakage-free
        look-ahead reads BUF's freeze-known (week<=1) Elo instead, so the flag
        must be byte-unchanged.
        """
        schedule = _make_trap_schedule()
        target = schedule[schedule["week"] == 1]

        baseline = calc._derive_spot_flags(target, schedule, _AS_OF_AFTER_W1)
        assert baseline["TRAP_W1"]["home_look_ahead_spot"] == 1.0

        perturbed_sched = schedule.copy()
        w2_mask = perturbed_sched["week"] == 2
        # Drive BUF's entering-week-2 Elo down to "weak"; the OLD (leaky)
        # implementation read this directly and would flip the flag to 0.0.
        perturbed_sched.loc[w2_mask, "away_elo_pre"] = 1480.0
        perturbed = calc._derive_spot_flags(target, perturbed_sched, _AS_OF_AFTER_W1)

        assert (
            perturbed["TRAP_W1"]["home_look_ahead_spot"]
            == baseline["TRAP_W1"]["home_look_ahead_spot"]
        )


class TestSituationalOffBye:
    """Part 3: off_bye = 1.0 iff rest_days >= 13."""

    def test_off_bye_iff_rest_ge_13(self, calc):
        """Build features for a KC schedule with a 14-day gap (bye) and assert
        off_bye is 1.0 exactly when rest_days >= 13, else 0.0."""
        games = pd.DataFrame(
            [
                {
                    "game_id": "BYE_W1",
                    "season": 2024,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "MIA",
                    "venue": "KC Stadium",
                    "kickoff_et": datetime(2024, 9, 10, 13, 0),
                },
                {
                    "game_id": "BYE_W3",
                    "season": 2024,
                    "week": 3,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "venue": "KC Stadium",
                    "kickoff_et": datetime(2024, 9, 24, 13, 0),  # 14 days -> bye
                },
            ]
        )

        # Schedule reload is irrelevant for off_bye; return empty so the spot
        # derivation stays neutral and no on-disk silver is touched.
        with patch.object(
            calc, "_load_full_season_schedule", return_value=pd.DataFrame()
        ):
            result = calc.build_features(games, datetime(2024, 9, 25, 18, 0))

        result = result.set_index("game_id")

        # The genuinely-new off_bye flag matches the rest threshold exactly.
        for gid in result.index:
            expected = 1.0 if result.loc[gid, "home_rest_days"] >= 13 else 0.0
            assert result.loc[gid, "home_off_bye"] == expected

        # W3 has a 14-day rest (bye) -> off_bye; W1 has no prior game -> not.
        assert result.loc["BYE_W3", "home_rest_days"] == 14.0
        assert result.loc["BYE_W3", "home_off_bye"] == 1.0
        assert result.loc["BYE_W1", "home_off_bye"] == 0.0


class TestSituationalFullSchedule:
    """Part 4 (review #4): a target-week build sources the FULL schedule."""

    def _week1_target(self) -> pd.DataFrame:
        """The single target-week (W1) game as build_features would receive it
        after the :783 target-week filter -- no next-/prior-opponent rows."""
        return pd.DataFrame(
            [
                {
                    "game_id": "TRAP_W1",
                    "season": 2023,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "CAR",
                    "venue": "KC Stadium",
                    "kickoff_et": datetime(2023, 9, 10, 13, 0),
                }
            ]
        )

    def test_target_week_build_uses_full_season_schedule(self, calc):
        """With games_df filtered to W1 only, the look-ahead flag must still be
        derived from the FULL season schedule (which knows W2's strong BUF), not
        collapsed to a default by the :783 target-week filter."""
        target = self._week1_target()

        with patch.object(
            calc, "_load_full_season_schedule", return_value=_make_trap_schedule()
        ):
            result = calc.build_features(
                target,
                _AS_OF_AFTER_W1,
                target_season=2023,
                target_week=1,
            )

        row = result.set_index("game_id").loc["TRAP_W1"]
        assert row["home_look_ahead_spot"] == 1.0

    def test_starved_schedule_collapses_to_default(self, calc):
        """Contrast: if the schedule were starved to the target-week row only
        (the bug), the look-ahead flag collapses to 0.0 -- this is exactly what
        the full-schedule reload (review #4) prevents."""
        target = self._week1_target()
        starved = _make_trap_schedule()
        starved = starved[starved["week"] == 1]

        with patch.object(calc, "_load_full_season_schedule", return_value=starved):
            result = calc.build_features(
                target,
                _AS_OF_AFTER_W1,
                target_season=2023,
                target_week=1,
            )

        row = result.set_index("game_id").loc["TRAP_W1"]
        assert row["home_look_ahead_spot"] == 0.0
