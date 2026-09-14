"""The O/U weather cut's row routing: whose roof, and where an unknown game goes.

WHAT THIS MODULE PROVES, IN PLAIN TERMS
---------------------------------------
The O/U divergence harness splits its graded bets two ways on weather: games the
weather could touch against games it could not, and rough conditions against calm
ones. Both splits were routing rows wrongly, and neither fault was visible while
the underlying columns were a fabricated constant.

1. WHOSE ROOF. The applicability split keyed on ``venue_outdoor``, which is a
   property of the STADIUM, not of the game. Five stadiums in this league have a
   roof that opens. All 749 of their games in this population carry one single
   venue-level value, while 621 of them were actually played with the roof CLOSED
   and 128 with it OPEN. So 621 games whose weather could not reach the field were
   being graded as though it had. The split is re-keyed onto the game's own roof
   fact, which this project already stores per game in the silver layer.

2. WHERE AN UNKNOWN GAME GOES. Severe weather was ``severity > median`` and mild
   was everything else. In pandas, every comparison against a missing value is
   False, so a game whose weather nobody observed fell into MILD. That is a bucket
   that reads as a measurement of calm days and is partly a list of days nobody
   measured. The applicability split had the identical fault one level up: its
   indoor side was the complement of its outdoor side, so an unknown game was
   filed as INDOOR. Both are repaired the same way -- build BOTH sides of a split
   positively, over observed rows only, and report how many rows were set aside.

WHY THESE TESTS RUN ON FIXTURES AND READ NO GOLD
-------------------------------------------------
The claim under test is about WHICH ROWS land in WHICH BUCKET. That is decided
entirely by the masks, before a single bet is graded. Grading is done by the
LOCKED ``BettingSimulator`` through ``_grade_bucket``, which needs the deployed
artifact and the full prediction frame; running it here would make a routing test
depend on a model file and would let a change in grading break a test about
membership. So ``_grade_bucket`` is replaced by a recorder that reports only which
game ids reached it, and the assertions are about that list.

TWO SPLITS, NOT FOUR, AND WHY
------------------------------
The plan that commissioned this work describes four splits: applicability plus
three median bands (calm/windy, dry/wet, mild/severe). The harness has TWO -- the
applicability split and one severity band. Its own governing ruling is the reason:
adding a wind band and a precipitation band now would add two new comparisons to a
correction family that was fixed in Phase 26, which is the forking-paths defect
the module's own ``DEBIAS_RESIDUAL_THRESHOLD`` note exists to prevent. Wind and
precipitation already reach the severity score that the single band splits on. So
the repair covers every split that exists, and adds none.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import backtest.ou_divergence as ou

# The five stadiums whose roof opens. One of them is enough to prove the point.
RETRACTABLE_STADIUM_ID = "HOU00"


def _recording_grade_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _grade_bucket with a recorder that reports bucket MEMBERSHIP only.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        None. The patched function returns ``{"n", "n_excluded", "game_ids"}`` so
        every assertion below is about which rows arrived, never about a graded
        number.
    """

    def recorder(
        bucket_per_game: pd.DataFrame,
        n_total: int,
        preds: Any,
        odds: Any,
    ) -> dict[str, Any]:
        return {
            "n": len(bucket_per_game),
            "n_excluded": n_total - len(bucket_per_game),
            "game_ids": tuple(bucket_per_game["game_id"]),
            "hit_rate": 0.5 if len(bucket_per_game) else None,
            "n_graded": len(bucket_per_game),
        }

    monkeypatch.setattr(ou, "_grade_bucket", recorder)


def _per_game_fixture() -> pd.DataFrame:
    """Five games: a closed and an open game at ONE retractable stadium, plus three.

    Returns:
        A per-game frame carrying both the venue-level key the cut used to split
        on and the per-game roof fact it splits on now, plus one game whose
        weather was never observed.

    The rows:
      * ``RETRACT-CLOSED`` -- retractable stadium, roof CLOSED that day.
      * ``RETRACT-OPEN``   -- the SAME stadium, roof OPEN that day.
      * ``OPEN-AIR``       -- an ordinary outdoor stadium.
      * ``FIXED-DOME``     -- a fixed dome.
      * ``NO-OBSERVATION`` -- outdoor, but nobody recorded the weather: the roof
        fact, the coverage flag and the severity score are all missing.
    """
    return pd.DataFrame(
        {
            "game_id": [
                "RETRACT-CLOSED",
                "RETRACT-OPEN",
                "OPEN-AIR",
                "FIXED-DOME",
                "NO-OBSERVATION",
            ],
            "stadium_id": [
                RETRACTABLE_STADIUM_ID,
                RETRACTABLE_STADIUM_ID,
                "PIT00",
                "DET00",
                "BUF00",
            ],
            # VENUE-level: both retractable games share one value, which is the
            # whole defect. Graded as outdoor, including the one played closed.
            "venue_outdoor": [0.0, 0.0, 1.0, 0.0, 1.0],
            # PER-GAME: the game's own roof. The two retractable games differ.
            "weather_affects_game": [0.0, 1.0, 1.0, 0.0, np.nan],
            "weather_coverage": [1.0, 1.0, 1.0, 1.0, np.nan],
            "weather_severity_score": [0.10, 0.90, 0.80, 0.05, np.nan],
            "line_clv": [0.1, -0.2, 0.3, 0.0, 0.4],
        }
    )


def _cut(per_game: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Run the weather cut over *per_game* with a recording grader.

    Args:
        per_game: A fixture frame.

    Returns:
        The cut's bucket dictionary.
    """
    return ou._weather_cut(per_game, len(per_game), preds=None, odds=None)


def _members(cut: dict[str, dict[str, Any]], bucket: str) -> tuple[str, ...]:
    """The game ids that reached *bucket*.

    Args:
        cut: A weather-cut result.
        bucket: A bucket name.

    Returns:
        The tuple of game ids, or an empty tuple when the bucket is unavailable.
    """
    return tuple(cut.get(bucket, {}).get("game_ids", ()))


class TestWhoseRoofDecidesApplicability:
    """The split asks about the GAME's roof, not about its stadium's."""

    def test_open_and_closed_at_one_retractable_stadium_land_in_different_buckets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two games share a stadium and a venue key, and must still split."""
        _recording_grade_bucket(monkeypatch)
        cut = _cut(_per_game_fixture())

        assert "RETRACT-OPEN" in _members(cut, "outdoor"), (
            "a game played with the roof OPEN is a game the weather reached"
        )
        assert "RETRACT-CLOSED" in _members(cut, "indoor"), (
            "a game played with the roof CLOSED is not an outdoor game, however "
            "its stadium is classified"
        )

    def test_the_old_venue_level_key_puts_that_same_pair_together(self) -> None:
        """The CONTROL: this proves a CHANGE, not merely a state.

        Under the venue-level key the identical pair is inseparable, because the
        stadium has one value and the two games do not.
        """
        frame = _per_game_fixture()
        pair = frame[frame["stadium_id"] == RETRACTABLE_STADIUM_ID]

        assert pair["venue_outdoor"].nunique() == 1, (
            "the venue key cannot tell the pair apart -- that is the defect"
        )
        assert pair["weather_affects_game"].nunique() == 2, (
            "the per-game roof fact can tell them apart -- that is the repair"
        )

        venue_a, venue_b, _ = ou._bucket_masks_excluding_missing(
            pair["venue_outdoor"], split="indicator"
        )
        assert bool(venue_a.sum()) != bool(venue_b.sum()), (
            "keyed on the venue, both games land on ONE side of the split"
        )

        game_a, game_b, _ = ou._bucket_masks_excluding_missing(
            pair["weather_affects_game"], split="indicator"
        )
        assert int(game_a.sum()) == 1
        assert int(game_b.sum()) == 1

    def test_the_venue_level_constant_is_declared_but_unused_by_the_cut(self) -> None:
        """The old key stays on the record with its warning, and stays out of use."""
        import inspect

        assert hasattr(ou, "_WEATHER_OUTDOOR_COL"), (
            "the venue-level name is kept so a later reader meets the warning "
            "rather than rediscovering the defect"
        )
        body = [
            line
            for line in inspect.getsource(ou._weather_cut).splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert not [line for line in body if "_WEATHER_OUTDOOR_COL" in line], (
            "the venue-level key must not decide any bucket"
        )


class TestAnUnobservedGameIsNotQuietlyFiledSomewhere:
    """A game nobody measured is set aside and counted, never assigned a side."""

    def test_an_absent_observation_is_not_filed_as_mild(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The anti-assertion for `~(severity > median)`.

        Before the repair this game IS in ``mild_weather``, because comparing a
        missing value against the median yields False and mild was the
        complement. A bucket of calm days must not contain unmeasured days.
        """
        _recording_grade_bucket(monkeypatch)
        cut = _cut(_per_game_fixture())

        assert "NO-OBSERVATION" not in _members(cut, "mild_weather")
        assert "NO-OBSERVATION" not in _members(cut, "severe_weather")

    def test_an_absent_observation_is_not_filed_as_indoor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same fault one level up, at the applicability split."""
        _recording_grade_bucket(monkeypatch)
        cut = _cut(_per_game_fixture())

        assert "NO-OBSERVATION" not in _members(cut, "indoor")
        assert "NO-OBSERVATION" not in _members(cut, "outdoor")

    def test_an_absent_observation_is_excluded_from_both_denominators(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Set aside AND counted -- an excluded count nobody sees is the same defect."""
        _recording_grade_bucket(monkeypatch)
        cut = _cut(_per_game_fixture())

        for bucket in ("outdoor", "indoor", "severe_weather", "mild_weather"):
            assert cut[bucket]["excluded_missing"] == 1, (
                f"{bucket} must report the one unmeasured game it set aside"
            )

    def test_a_row_without_coverage_is_treated_as_unobserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A roof fact recorded for a game with no weather observation is not usable."""
        _recording_grade_bucket(monkeypatch)
        frame = _per_game_fixture()
        frame.loc[frame["game_id"] == "OPEN-AIR", "weather_coverage"] = 0.0
        cut = _cut(frame)

        assert "OPEN-AIR" not in _members(cut, "outdoor")
        assert "OPEN-AIR" not in _members(cut, "indoor")
        assert cut["outdoor"]["excluded_missing"] == 2


class TestTheSplitsAccountForEveryRow:
    """Two sides plus the set-aside count equal the population. No row vanishes."""

    def test_every_split_is_exhaustive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """For each split: side + side + excluded == the rows the cut was given."""
        _recording_grade_bucket(monkeypatch)
        frame = _per_game_fixture()
        cut = _cut(frame)

        for left, right in (("outdoor", "indoor"), ("severe_weather", "mild_weather")):
            total = cut[left]["n"] + cut[right]["n"] + cut[left]["excluded_missing"]
            assert total == len(frame), (
                f"{left}/{right} does not account for every row given to the cut"
            )
            assert cut[left]["excluded_missing"] == cut[right]["excluded_missing"], (
                "both sides of one split report the same set-aside count"
            )


class TestTheCutStaysTheSameCut:
    """A correctness repair to which rows a bucket holds -- not a new comparison."""

    def test_the_cut_adds_no_bucket_and_no_p_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bucket names are unchanged and no bucket carries a significance claim."""
        _recording_grade_bucket(monkeypatch)
        cut = _cut(_per_game_fixture())

        assert set(cut) == {"outdoor", "indoor", "severe_weather", "mild_weather"}
        assert not [b for b in cut.values() if "p_value" in b], (
            "the Phase-26 correction family is fixed; this cut adds no trial"
        )

    def test_the_degeneracy_guards_still_fire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The honest-refusal path survives the repair, on both splits."""
        _recording_grade_bucket(monkeypatch)
        frame = _per_game_fixture()
        frame["weather_severity_score"] = 0.5
        frame["weather_affects_game"] = 1.0
        cut = _cut(frame)

        assert cut["severe_weather"]["unavailable"] is True
        assert "coverage_note" in cut["severe_weather"]
        assert cut["outdoor"]["unavailable"] is True
        assert "coverage_note" in cut["outdoor"]

    def test_an_absent_applicability_column_is_reported_not_guessed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A column that is not there is disclosed, never substituted for."""
        _recording_grade_bucket(monkeypatch)
        frame = _per_game_fixture().drop(columns=["weather_affects_game"])
        cut = _cut(frame)

        assert cut["outdoor"]["unavailable"] is True
        assert "weather_affects_game" in cut["outdoor"]["coverage_note"]


class TestTheSplitHelperItself:
    """The shared helper, tested directly rather than only through the cut."""

    def test_a_median_split_puts_a_missing_value_on_neither_side(self) -> None:
        """The whole point: NaN is set aside, not defaulted to the low side."""
        series = pd.Series([1.0, 2.0, 3.0, np.nan], name="severity")
        high, low, excluded = ou._bucket_masks_excluding_missing(series, split="median")

        assert excluded == 1
        assert not bool(high.iloc[3])
        assert not bool(low.iloc[3])
        assert int(high.sum()) + int(low.sum()) + excluded == len(series)

    def test_an_indicator_split_builds_both_sides_positively(self) -> None:
        """Neither side is the complement of the other, so NaN joins neither."""
        series = pd.Series([1.0, 0.0, np.nan], name="roof")
        ones, zeros, excluded = ou._bucket_masks_excluding_missing(
            series, split="indicator"
        )

        assert list(ones) == [True, False, False]
        assert list(zeros) == [False, True, False]
        assert excluded == 1

    def test_an_indicator_split_refuses_a_value_that_is_neither_zero_nor_one(
        self,
    ) -> None:
        """A named refusal beats a row that silently belongs to no side."""
        series = pd.Series([1.0, 0.0, 0.37], name="roof")

        with pytest.raises(ValueError, match="indicator"):
            ou._bucket_masks_excluding_missing(series, split="indicator")

    def test_an_all_missing_series_excludes_everything(self) -> None:
        """The degenerate case does not raise and does not invent a median."""
        series = pd.Series([np.nan, np.nan], name="severity")
        high, low, excluded = ou._bucket_masks_excluding_missing(series, split="median")

        assert excluded == 2
        assert int(high.sum()) == 0
        assert int(low.sum()) == 0
