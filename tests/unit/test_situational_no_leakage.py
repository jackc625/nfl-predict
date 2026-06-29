"""Leakage proof scaffold for the situational-spots signal (SIG-03 / SC3).

Following the `tests/unit/test_elo_no_leakage.py` convention. The look-ahead /
letdown / off-bye additions land in `features/contextual.py` in Plan 28-04, so
every test here is a RED scaffold skipped with a plan-named reason.

Leakage contract (D-16): the next-opponent identity and the pre-freeze Elo
ratings are NOT leakage (the schedule and ratings are known at the Friday freeze);
only the look-ahead game *result* is future information. The withhold-future test
therefore reveals future game RESULTS and asserts the spot flags are unchanged.

Parts (filled by Plan 28-04):
  1. Time-fence assertion -- the new spots respect the `as_of_datetime` cutoff.
  2. Withhold-future byte-unchanged -- reveal future look-ahead-game results,
     assert look-ahead/letdown flags unchanged.
  3. off_bye derivation -- `off_bye = 1.0 if rest_days >= 13 else 0.0`.
"""

import pandas as pd
import pytest


def _make_trap_schedule() -> pd.DataFrame:
    """A 3-game schedule with a known look-ahead/letdown setup, grounded in the
    real raw silver Elo scale (~1150-1840, std ~125). KC (strong) plays a weak
    opponent this week with a strong divisional opponent next week -- the classic
    look-ahead trap shape.

    Returns:
        A 3-row schedule DataFrame.
    """
    return pd.DataFrame(
        {
            "game_id": ["TRAP_W1", "TRAP_W2", "TRAP_W3"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "home_team": ["KC", "KC", "KC"],
            "away_team": ["CAR", "BUF", "DEN"],
            "home_elo_pre": [1720.0, 1725.0, 1710.0],
            "away_elo_pre": [1480.0, 1690.0, 1500.0],
            "kickoff_et": pd.to_datetime(
                ["2023-09-10T13:00:00", "2023-09-17T13:00:00", "2023-09-24T13:00:00"]
            ),
        }
    )


class TestSituationalTimeFence:
    """SKIPPED scaffold: contextual spot additions time-fence (Plan 28-04)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- look-ahead/letdown/off-bye land in "
        "features/contextual.py in Plan 28-04 (as_of_datetime cutoff respected)"
    )
    def test_situational_spots_respect_as_of_datetime(self):
        """The new spot features must respect the same `as_of_datetime` rest-fence
        cutoff as the existing contextual features. Filled by Plan 28-04."""
        raise NotImplementedError("Plan 28-04")


class TestSituationalWithholdFuture:
    """SKIPPED scaffold: withhold-future byte-unchanged proof (Plan 28-04)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- look-ahead/letdown land in Plan 28-04 "
        "(reveal the future look-ahead-game RESULT, assert the flag unchanged; "
        "next-opp identity + pre-freeze Elo are NOT leakage)"
    )
    def test_revealing_future_results_does_not_change_flags(self):
        """Building the look-ahead/letdown flags once with future game RESULTS
        present and once without must produce byte-identical flags -- only the
        future result is leakage, not the known schedule/Elo. Filled by Plan 28-04."""
        raise NotImplementedError("Plan 28-04")


class TestSituationalOffBye:
    """SKIPPED scaffold: off_bye = rest_days >= 13 (Plan 28-04)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- off_bye flag lands in Plan 28-04 "
        "(off_bye = 1.0 if rest_days >= 13 else 0.0)"
    )
    def test_off_bye_threshold(self):
        """off_bye is 1.0 exactly when rest_days >= 13, else 0.0. Filled by
        Plan 28-04."""
        raise NotImplementedError("Plan 28-04")
