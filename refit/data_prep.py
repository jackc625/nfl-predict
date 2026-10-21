"""The data preparation every re-fit attempt runs first (Phase 37 D-37; record section 7).

WHY IT EXISTS
-------------
A re-fit trains on every finished game before its cutoff. On a cadence Tuesday the gold the last
prediction day built is stale for the week that just finished: late games' labels have not landed
in gold, and their Elo snapshots are still the PROVISIONAL pre-game rows the training guard
refuses. The owner chose correctness over speed: every attempt first runs the SAME nightly build
a prediction day runs, in the same order, so no finished game lacks its label or rests on a
provisional Elo row:

1. current-season Elo, with provisional rows only for the day's slate (none on a no-games day);
2. team form;
3. weather features through the season;
4. the full-history gold build, leaving out every unplayed game that locks after the run date's
   18:00 ET lock;
5. the leakage scan of the gold that landed.

WHY IT CANNOT CHANGE A DECISION, AND WHY THE NEXT RUN ACCEPTS WHAT IT LEAVES
--------------------------------------------------------------------------
It runs only AFTER the day's ledger write (Plan 37-13 places the call, D-15), so it cannot alter
the gold a decision already used. And it is the same function calls with the same arguments a
daily run makes, so the gold it leaves is exactly the gold a daily run would have left, which the
next daily run accepts. Worst case about 10 minutes (the full build measured ~591.5 s).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import date

from backtest.refit_preregistration_2026 import REFIT_SEASON
from pipeline.daily_steps import (
    DailySlate,
    build_weather_features_through,
    slate_lock,
)
from pipeline.steps import (
    build_and_save_gold,
    persist_current_season_elo,
    step_build_team_form,
    step_validate_features,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["prepare_gold"]


def prepare_gold(run_date_et: date, *, slate: DailySlate | None) -> None:
    """Run the prediction day's nightly build for a re-fit attempt (D-37).

    Args:
        run_date_et: The ET calendar day of the daily run attempting the re-fit.
        slate: The day's slate when the attempt lands on a prediction day (a catch-up), so the
            build keeps that slate's provisional Elo rows and gold rows exactly as the day's run
            built them; None on a day with no games.
    """
    slate_ids = slate.game_ids if slate is not None else frozenset()
    weather_failures = slate.weather_failures if slate is not None else {}

    logger.info(
        "Re-fit data preparation: the nightly build",
        run_date_et=run_date_et.isoformat(),
        slate_games=len(slate_ids),
    )
    persist_current_season_elo(slate_ids)
    step_build_team_form()
    build_weather_features_through(REFIT_SEASON, slate_ids, weather_failures)
    build_and_save_gold(slate_ids, unplayed_through_lock=slate_lock(run_date_et))
    step_validate_features()
