"""A weather row may not carry a measurement its own fence could not date (Plan 33.2-20).

THE DEFECT, FOUND BY THE ARMED GATE ON THE FIRST CLEAN 2026 BUILD
------------------------------------------------------------------
The weather VALUES that reach gold are read straight off silver ``weather_features``. The
weather PROVENANCE is derived from silver ``weather`` through ``select_weather_row``, the
one fence. Two reads, two answers, and nothing made them agree.

MEASURED 2026-09-22 over all 6,771 games in silver: they disagreed for exactly ONE,
``2026_W02_DET@BUF`` -- and that disagreement is a REAL POST-LOCK VALUE, not bookkeeping.
The game kicked off 2026-09-17 20:15 ET, so its lock was 2026-09-16 18:00 ET. Its silver
weather row was captured at 2026-09-19 15:37 UTC -- TWO DAYS AFTER THE GAME WAS PLAYED --
and carries no ``forecast_issue_time`` at all. The fence correctly refused to date it and
the provenance said ``no_information``; ``weather_features`` carried ``temp_f`` 66.6,
``wind_mph`` 3.3 and ``weather_coverage`` 1.0 for the same game. Gold would have learned
that game's weather from a reading taken after the whistle.

The build REFUSED rather than writing it, by name
(``UndatedSourceError: 'weather' declares game 2026_W02_DET@BUF no_information, but its
'temp_f' is 66.6 where the declared signature requires null``), and production ``data/``
was digest-identical across the refused run: 1,128 files UNCHANGED. That is the gate
working. This module is the fix, at its cause.

THE RULE
--------
A game whose forecast the fence could not date carries NO MEASUREMENT. Where a row
contradicts its own fence the whole weather family is rewritten to the ABSENT-OBSERVATION
shape -- every measurement NULL, ``weather_coverage`` 0.0 -- the same NULL-plus-flag
treatment Plan 33.2-12 gave the 56 games played abroad and ``2019_W18_BUF@HOU``.
Applicability is PRESERVED: an outdoor game stays outdoor, because what is absent is the
observation and not the fact that weather applies.

It is deliberately NOT applied to every ``no_information`` row. A dome is
``no_information`` too and its row is already correct -- it reports no measurement because
there is none, beside flags saying weather does not apply. Rewriting those would change
1,345 rows of settled history to say something slightly different about domes, which is
not this fence's business.

Run this module:  uv run pytest tests/unit/test_weather_frame_follows_its_fence.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.storage import load_dataframe
from features.weather import (
    WEATHER_NO_INFORMATION_SIGNATURE,
    WeatherFeaturesCalculator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The game the armed gate caught, and the fact that makes it a leak rather than a gap.
THE_POST_LOCK_CAPTURE = "2026_W02_DET@BUF"


@pytest.fixture(scope="module")
def silver() -> dict[str, pd.DataFrame]:
    for name in ("games", "weather_features", "weather"):
        if not (REPO_ROOT / "data" / "silver" / f"{name}.parquet").exists():
            pytest.skip(f"silver {name} is not built on this checkout")
    return {
        "games": load_dataframe("games", layer="silver"),
        "weather_features": load_dataframe("weather_features", "silver", "parquet"),
        "weather": load_dataframe("weather", "silver", "parquet"),
    }


class TestTheDefectIsRealAndIsStillThereInSilver:
    """Non-vacuity: the fix has a live subject, and that subject is a post-lock capture."""

    def test_the_offending_silver_row_was_captured_after_the_game_was_played(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        rows = silver["weather"]
        row = rows[rows["game_id"] == THE_POST_LOCK_CAPTURE]
        if row.empty:
            pytest.skip(f"{THE_POST_LOCK_CAPTURE} has no silver weather row here")
        row = row.iloc[0]
        captured = pd.Timestamp(row["forecast_time"])
        kickoff = pd.Timestamp(row["game_time"])
        assert captured > kickoff, (
            f"the capture {captured} is not after the kickoff {kickoff}, so this module "
            "is asserting about a row that is no longer the post-lock one it names. "
            "Re-measure rather than adjusting the expectation."
        )
        assert pd.isna(row["forecast_issue_time"]), (
            "the row now carries a forecast_issue_time, so the fence may be able to date "
            "it. Re-measure: the defect this module fixes was an UNDATABLE capture."
        )

    def test_the_raw_feature_frame_still_carries_its_measurement(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        """The fix is at the BUILDER, so silver is unchanged and still contradicts."""
        frame = silver["weather_features"]
        row = frame[frame["game_id"] == THE_POST_LOCK_CAPTURE]
        if row.empty:
            pytest.skip(f"{THE_POST_LOCK_CAPTURE} has no silver weather_features row")
        assert pd.notna(row.iloc[0]["temp_f"])


class TestTheFenceIsAppliedToTheFeatureFrame:
    def test_exactly_the_contradicting_row_is_rewritten(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        calc = WeatherFeaturesCalculator()
        before = silver["weather_features"]
        after = calc.enforce_fence_on_feature_frame(before, silver["games"])
        moved = [
            str(game_id)
            for game_id, changed in zip(
                before["game_id"],
                (before.astype(str) != after.astype(str)).any(axis=1),
                strict=True,
            )
            if changed
        ]
        assert moved == [THE_POST_LOCK_CAPTURE], (
            f"the fence rewrote {moved}. It must reach exactly the rows whose measurement "
            "survived a fence they failed -- no dome, and nothing in 2002-2025."
        )

    def test_the_rewritten_row_is_the_absent_observation_shape(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        calc = WeatherFeaturesCalculator()
        after = calc.enforce_fence_on_feature_frame(
            silver["weather_features"], silver["games"]
        )
        row = after[after["game_id"] == THE_POST_LOCK_CAPTURE].iloc[0]
        for column, declared in WEATHER_NO_INFORMATION_SIGNATURE.items():
            if column not in after.columns:
                continue
            if declared is None:
                assert pd.isna(row[column]), f"{column} is {row[column]!r}, not null"
            else:
                assert row[column] == declared
        assert row["weather_coverage"] == 0.0
        assert row["weather_affects_game"] == 1.0, (
            "applicability was not preserved. What is absent is the OBSERVATION; the game "
            "is still outdoors, and 0.0 here is the value that means INDOOR."
        )
        assert pd.isna(row["wind_mph"]), (
            "the whole family must go. A NULL temperature beside a real wind speed is "
            "half an answer, and the wind came from the same post-game capture."
        )

    def test_no_2002_2025_row_moves(self, silver: dict[str, pd.DataFrame]) -> None:
        """The settled history, including 1,345 dome and absence rows, is untouched."""
        calc = WeatherFeaturesCalculator()
        before = silver["weather_features"]
        after = calc.enforce_fence_on_feature_frame(before, silver["games"])
        historical = ~before["game_id"].astype(str).str.startswith("2026_")
        assert historical.sum() > 6000
        left = before[historical].astype(str).reset_index(drop=True)
        right = after[historical].astype(str).reset_index(drop=True)
        assert left.equals(right)

    def test_an_honest_frame_is_returned_as_the_same_object(self) -> None:
        """A build with nothing to rewrite is byte-for-byte the build it was."""
        calc = WeatherFeaturesCalculator()
        empty = pd.DataFrame(columns=["game_id"])
        assert calc.enforce_fence_on_feature_frame(empty, empty) is empty


class TestTheGateNowAcceptsTheFencedFrame:
    """End to end: the frame and its provenance agree, so the gate passes."""

    def test_the_information_time_gate_accepts_the_fenced_weather_source(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        from features.provenance import (
            InformationTimeGate,
            SourceCheckState,
            build_lock_frame,
        )

        calc = WeatherFeaturesCalculator()
        games = silver["games"]
        frame = calc.enforce_fence_on_feature_frame(silver["weather_features"], games)
        scoped = games[games["game_id"].isin(frame["game_id"])]
        state = InformationTimeGate().check(
            "weather",
            frame,
            calc.information_times(
                games, feature_game_ids=frozenset(frame["game_id"].astype(str))
            ),
            build_lock_frame(scoped),
            no_information_signature=calc.no_information_signature(),
        )
        assert state is SourceCheckState.CHECKED

    def test_the_unfenced_frame_is_still_refused_by_name(
        self, silver: dict[str, pd.DataFrame]
    ) -> None:
        """The control: without the fence, the gate raises on exactly that game."""
        from features.provenance import (
            InformationTimeGate,
            UndatedSourceError,
            build_lock_frame,
        )

        calc = WeatherFeaturesCalculator()
        games = silver["games"]
        frame = silver["weather_features"]
        scoped = games[games["game_id"].isin(frame["game_id"])]
        with pytest.raises(UndatedSourceError) as exc:
            InformationTimeGate().check(
                "weather",
                frame,
                calc.information_times(
                    games, feature_game_ids=frozenset(frame["game_id"].astype(str))
                ),
                build_lock_frame(scoped),
                no_information_signature=calc.no_information_signature(),
            )
        assert exc.value.details["game_ids"] == [THE_POST_LOCK_CAPTURE]
