"""The per-game freeze is a BINDING REFUSAL at selection time, not a descriptive value.

Phase 33, Plan 33-05 Task 1 (COLD-03, R6, D33-27/28).

WHAT WAS MISSING AND WHY IT MATTERS
-----------------------------------
``backtest.weekly_bet_list._is_frozen`` only protects an ALREADY-STORED row from being
overwritten during an upsert. Nothing refused to EMIT a row whose game freeze was already
past. The Friday 6 PM freeze lands roughly 26 hours AFTER that week's Thursday kickoff, so a
Thursday-night row selected on the Friday would carry a post-hoc pick wearing a pre-game
timestamp -- the repudiation failure COLD-03 exists to prevent (T-33-21).

THE FENCE IS ``>=`` AND THAT IS DELIBERATE
------------------------------------------
``_is_frozen`` has returned ``now >= freeze_dt`` since Plan 31-17 and this plan KEEPS IT.
A game whose freeze equals the run instant TO THE SECOND is REFUSED, because at that instant
the market has frozen and any pick made now is made with the frozen line in hand. The three
clock positions below -- one second after, exactly at, one second before -- are the whole
content of that claim.

WHY A STRICT PARSE WRAPPER RATHER THAN A WIDENED PARSER (D33-27, T-33-23)
------------------------------------------------------------------------
``scripts.ingest_historical_odds.normalize_snapshot_ts`` deliberately anchors a NAIVE value
in EASTERN and argues that at length in its own docstring: an unqualified wall-clock time in
this project's odds data is a market-local time. The freeze path needs the opposite -- a naive
value must RAISE, because a freeze comparison has no defensible default timezone. One function
cannot hold both behaviours without becoming a second answer wearing one name, so the strict
behaviour is a WRAPPER (``require_aware_snapshot_ts``) and the single-source property is
preserved by composition.

ALL THREE NAIVE SHAPES ARE DELIBERATELY CONSTRUCTED. Every one of the 234 stored bet-list
rows carries a tz-aware Eastern ``freeze_ts`` string, so the naive branch is UNREACHABLE from
real data -- which is exactly why it is safe to convert from a silent UTC assumption into a
raise now, and exactly why the inputs have to be built by hand.

Run this module:  uv run pytest tests/unit/test_freeze_fence_binding.py -q

Plain unit module: it reads nothing under ``data/``, ``artifacts/`` or ``outputs/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest.weekly_bet_list import (
    FreezePassedError,
    select_games_for_freeze_instant,
)
from scripts.ingest_historical_odds import (
    NaiveTimestampError,
    get_synthetic_snapshot_ts,
    normalize_snapshot_ts,
    require_aware_snapshot_ts,
)

EASTERN = ZoneInfo("America/New_York")

# Week 2's own Friday instant in the real 2026 season, MEASURED rather than assumed:
# get_synthetic_snapshot_ts("2026-09-20") is 2026-09-18T22:00:00+00:00.
_WEEK_2_SUNDAY = "2026-09-20"
_WEEK_2_MONDAY = "2026-09-21"
_WEEK_3_THURSDAY = "2026-09-24"

# One second, the granularity the three clock positions are separated by. A whole second
# rather than a microsecond here because the claim is about the OPERATOR (>= against >), and a
# second is unambiguous in every representation the row can be stored in.
_ONE_SECOND = timedelta(seconds=1)


def _schedule() -> pd.DataFrame:
    """A constructed two-week slate whose games SHARE one freeze instant.

    Week 2's Sunday and Monday games and week 3's Thursday game all take the Friday
    2026-09-18 18:00 ET freeze -- that sharing is D33-28's whole point and is asserted from
    the one freeze rule rather than hand-typed.
    """
    return pd.DataFrame(
        [
            {
                "game_id": "2026_02_CAR_ATL",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_SUNDAY,
            },
            {
                "game_id": "2026_02_LV_NYJ",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_MONDAY,
            },
            {
                "game_id": "2026_03_ATL_GB",
                "season": 2026,
                "week": 3,
                "gameday": _WEEK_3_THURSDAY,
            },
        ]
    )


def _week_2_friday_instant() -> datetime:
    return get_synthetic_snapshot_ts(_WEEK_2_SUNDAY)


# ---------------------------------------------------------------------------
# The strict parse wrapper: three naive shapes, each deliberately constructed.
# ---------------------------------------------------------------------------


class TestANaiveTimestampRaisesRatherThanBeingAssumed:
    """T-33-23: a naive value is assumed neither UTC (today's bug) nor Eastern (the parser)."""

    def test_a_naive_datetime_raises_and_the_message_names_the_value_and_its_type(
        self,
    ) -> None:
        naive = datetime(2026, 9, 17, 18, 0)
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts(naive)

        message = str(excinfo.value)
        assert "2026-09-17 18:00:00" in message
        assert "datetime" in message
        assert "timezone" in message.lower()

    def test_a_naive_pandas_timestamp_raises(self) -> None:
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts(pd.Timestamp("2026-09-17 18:00:00"))

        message = str(excinfo.value)
        assert "2026-09-17 18:00:00" in message
        assert "Timestamp" in message

    def test_a_naive_iso_string_raises(self) -> None:
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts("2026-09-17T18:00:00")

        message = str(excinfo.value)
        assert "2026-09-17T18:00:00" in message
        assert "str" in message

    def test_the_message_says_a_freeze_comparison_has_no_defensible_default(
        self,
    ) -> None:
        """The reason travels with the refusal, so the next reader does not add a default."""
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts("2026-09-17 18:00:00")
        assert "defensible" in str(excinfo.value)


class TestEveryAwareShapeTheParserDocumentsStillResolves:
    """The wrapper narrows NOTHING except naiveness: every aware shape round-trips."""

    @pytest.mark.parametrize(
        "value",
        [
            # The legacy per-season string.
            "2021-09-19T18:00:00-04:00",
            # The single non-consensus row's space-separated UTC string.
            "2025-09-29 18:44:09.707942+00:00",
            # A tz-aware datetime.
            datetime(2026, 9, 18, 18, 0, tzinfo=EASTERN),
            # A tz-aware pandas Timestamp.
            pd.Timestamp("2026-09-18T18:00:00-04:00"),
        ],
    )
    def test_an_aware_value_returns_a_utc_aware_datetime(self, value: object) -> None:
        resolved = require_aware_snapshot_ts(value)

        assert resolved.tzinfo is not None
        assert resolved.utcoffset() == timedelta(0)
        # The same INSTANT the one parse path resolves, only expressed in UTC.
        assert resolved == normalize_snapshot_ts(value)

    def test_the_wrapper_delegates_rather_than_reimplementing_the_parse(self) -> None:
        """Every aware shape agrees with ``normalize_snapshot_ts`` to the microsecond."""
        value = "2025-09-29 18:44:09.707942+00:00"
        assert require_aware_snapshot_ts(value) == normalize_snapshot_ts(value)
        assert require_aware_snapshot_ts(value).isoformat().endswith("+00:00")

    @pytest.mark.parametrize("value", [None, "", float("nan"), pd.NaT])
    def test_a_null_or_empty_value_keeps_the_parsers_own_refusal(
        self, value: object
    ) -> None:
        """CONTROL: the wrapper adds a naiveness rule and takes nothing away.

        A null is still the parser's ``ValueError`` and is NOT relabelled as naive -- a
        missing instant and an unqualified one are different failures with different fixes.
        """
        with pytest.raises(ValueError) as excinfo:
            require_aware_snapshot_ts(value)
        assert not isinstance(excinfo.value, NaiveTimestampError)


# ---------------------------------------------------------------------------
# The binding selection fence: three clock positions.
# ---------------------------------------------------------------------------


class TestSelectionRefusesAGameWhoseFreezeHasPassed:
    """R6: a past-freeze game is refused BY NAME -- never emitted, never silently dropped."""

    def test_one_second_after_the_freeze_selection_raises_naming_the_game(self) -> None:
        instant = _week_2_friday_instant()
        with pytest.raises(FreezePassedError) as excinfo:
            select_games_for_freeze_instant(
                _schedule(), instant, now=instant + _ONE_SECOND
            )

        message = str(excinfo.value)
        assert "2026_02_CAR_ATL" in message
        # BOTH instants are named, so the reader can see which side of the fence it is on.
        assert instant.isoformat() in message
        assert (instant + _ONE_SECOND).isoformat() in message

    def test_exactly_at_the_freeze_instant_selection_raises_because_the_fence_is_ge(
        self,
    ) -> None:
        """The equality edge, stated explicitly: ``>=`` refuses, ``>`` would admit.

        This is the operator ``_is_frozen`` has carried since Plan 31-17 and that this plan
        keeps unchanged. A test that only covered one-second-after would pass under either
        operator and would therefore assert nothing about the edge.
        """
        instant = _week_2_friday_instant()
        with pytest.raises(FreezePassedError):
            select_games_for_freeze_instant(_schedule(), instant, now=instant)

    def test_one_second_before_the_freeze_the_row_is_emitted(self) -> None:
        instant = _week_2_friday_instant()
        selected = select_games_for_freeze_instant(
            _schedule(), instant, now=instant - _ONE_SECOND
        )

        assert set(selected["game_id"]) == {
            "2026_02_CAR_ATL",
            "2026_02_LV_NYJ",
            "2026_03_ATL_GB",
        }

    def test_a_game_belonging_to_a_different_instant_is_simply_not_selected(
        self,
    ) -> None:
        """Scoping is not refusing. A week-1 game is OUT OF SCOPE for week 2's instant.

        It is absent because its freeze is a DIFFERENT instant, not because the fence
        refused it -- so nothing raises. Keeping those two outcomes distinct is what makes
        ``FreezePassedError`` a tripwire rather than an ordinary control-flow signal.
        """
        schedule = pd.concat(
            [
                _schedule(),
                pd.DataFrame(
                    [
                        {
                            "game_id": "2026_01_NE_SEA",
                            "season": 2026,
                            "week": 1,
                            "gameday": "2026-09-13",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        instant = _week_2_friday_instant()
        selected = select_games_for_freeze_instant(
            schedule, instant, now=instant - _ONE_SECOND
        )
        assert "2026_01_NE_SEA" not in set(selected["game_id"])

    def test_a_naive_run_clock_raises_rather_than_being_assumed_utc(self) -> None:
        """Both sides of the comparison go through the strict wrapper (R6 timezone edge)."""
        naive_now = datetime(2026, 9, 18, 17, 59)
        with pytest.raises(NaiveTimestampError):
            select_games_for_freeze_instant(
                _schedule(), _week_2_friday_instant(), now=naive_now
            )

    def test_the_per_game_freeze_is_never_re_derived_here(self) -> None:
        """The selected set is exactly the set the ONE freeze rule assigns to this instant.

        Asserted against ``get_synthetic_snapshot_ts`` per row rather than against a
        hand-written date arithmetic, so a second "the prior Friday" implementation inside
        the selection would show up as a disagreement.
        """
        schedule = _schedule()
        instant = _week_2_friday_instant()
        expected = {
            str(row.game_id)
            for row in schedule.itertuples()
            if get_synthetic_snapshot_ts(str(row.gameday)) == instant
        }
        selected = select_games_for_freeze_instant(
            schedule, instant, now=instant - _ONE_SECOND
        )
        assert set(selected["game_id"]) == expected
        assert expected, "the fixture shares no freeze instant -- the check is vacuous"


# ---------------------------------------------------------------------------
# The back-compat READ shim.
# ---------------------------------------------------------------------------
#
# WHY THESE LIVE HERE. Plan 33-05 Task 1 creates ``read_bet_list_with_schema_shim`` but names no
# test module for it -- its three named modules are all about the fence. It is filed under this
# module, the task's primary home, rather than under ``tests/unit/test_bet_list_schema.py``,
# which Task 3 owns and which cannot be extended before the owner checkpoint that locks the
# column count has been ruled on.


class TestTheBackCompatReadShim:
    """A 28-column parquet still reads after the schema goes to 29 -- and is never rewritten."""

    def _pre_bump_columns(self) -> list[str]:
        """The 28 columns a stored file carries, derived rather than copied.

        Taken as ``BET_LIST_READ_COLUMNS`` minus the one column this phase adds, so this list is
        the pre-bump order BOTH before and after ``api.cache`` carries it. Deriving it from
        ``BET_LIST_COLUMNS`` instead would silently become the 29-column order after the bump,
        and the shim would then be handed a frame that needs no shimming -- a test that passes by
        no longer exercising the thing it names.
        """
        from backtest.weekly_bet_list import BET_LIST_READ_COLUMNS, DECIDED_AT_COLUMN

        return [c for c in BET_LIST_READ_COLUMNS if c != DECIDED_AT_COLUMN]

    def _stored_28_column_parquet(self, tmp_path: Path) -> Path:
        from backtest.weekly_bet_list import BET_LIST_ARTIFACT_NAME

        columns = self._pre_bump_columns()
        assert len(columns) == 28, len(columns)
        frame = pd.DataFrame(
            [dict.fromkeys(columns)],
            columns=pd.Index(columns),
        )
        frame["game_id"] = "2025_W01_DET@KC"
        frame["provenance"] = "backtest_replay"
        path = tmp_path / BET_LIST_ARTIFACT_NAME
        frame.to_parquet(path, index=False)
        return path

    def test_a_28_column_file_reads_back_as_29_with_a_null_observation_time(
        self, tmp_path: Path
    ) -> None:
        from backtest.weekly_bet_list import (
            BET_LIST_READ_COLUMNS,
            DECIDED_AT_COLUMN,
            read_bet_list_with_schema_shim,
        )

        path = self._stored_28_column_parquet(tmp_path)
        shimmed = read_bet_list_with_schema_shim(path)

        assert list(shimmed.columns) == list(BET_LIST_READ_COLUMNS)
        assert len(shimmed.columns) == 29
        assert shimmed[DECIDED_AT_COLUMN].isna().all(), (
            "the shim invented an observation time for a row that was never observed"
        )

    def test_the_shim_writes_nothing(self, tmp_path: Path) -> None:
        """The plan's own named prohibition, asserted on the file's BYTES.

        Every stored row is already past its freeze, so stamping an observation time onto one
        now would record a time at which nobody observed anything.
        """
        import hashlib

        from backtest.weekly_bet_list import read_bet_list_with_schema_shim

        path = self._stored_28_column_parquet(tmp_path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        read_bet_list_with_schema_shim(path)
        read_bet_list_with_schema_shim(path)
        after = hashlib.sha256(path.read_bytes()).hexdigest()

        assert after == before, "the READ shim rewrote the stored ledger"
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]

    def test_a_column_other_than_the_new_one_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """Only ``decided_at_utc`` is filled. A file written by a different schema is refused."""
        from backtest.weekly_bet_list import (
            BET_LIST_ARTIFACT_NAME,
            read_bet_list_with_schema_shim,
        )

        columns = [c for c in self._pre_bump_columns() if c != "ev_tier"]
        path = tmp_path / BET_LIST_ARTIFACT_NAME
        pd.DataFrame([dict.fromkeys(columns)], columns=pd.Index(columns)).to_parquet(
            path, index=False
        )

        with pytest.raises(ValueError, match="missing column"):
            read_bet_list_with_schema_shim(path)

    def test_the_one_stored_artifact_reader_is_routed_through_the_shim(self) -> None:
        """``data.graded_weeks`` and the cache-source reader both come through this one seam."""
        import inspect

        from backtest import weekly_bet_list

        source = inspect.getsource(weekly_bet_list.read_bet_list_artifact)
        assert "read_bet_list_with_schema_shim" in source
        assert "pd.read_parquet" not in source, (
            "read_bet_list_artifact reads the parquet itself again, so the shim is bypassed"
        )
