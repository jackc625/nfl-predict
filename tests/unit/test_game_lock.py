"""THE per-game lock rule (D33.2-01), proved across every kickoff shape.

Phase 33.2, Plan 33.2-01 Task 1 (SPEC R1).

The rule: each game's information locks at 18:00 America/New_York on the ET CALENDAR
DAY BEFORE its kickoff. At-lock information is admissible (``<=``). A naive timestamp
raises. A game with no kickoff cannot have a lock and raises by name.

WHAT THE TABLE PROVES, AND WHY IT IS A TABLE
--------------------------------------------
Every row is one kickoff SHAPE the retired preceding-Friday rule got wrong or could get
wrong: a Friday-night game (the Friday rule reached back a full week), a Sunday-morning
international game (09:30 ET -- still a Sunday in ET, so the Saturday is the lock), the
COVID-era Tuesday and Wednesday games, and both daylight-saving transitions. The DST rows
assert the UTC OFFSET as well as the wall clock, because the offset in force on the LOCK
date -- not the kickoff date -- is what makes the instant right.

Two rows feed the kickoff as a UTC-AWARE instant, which is how silver ``games.kickoff_et``
actually stores it (``datetime64[us, UTC]``). Those are the rows that prove the rule reads
the ET calendar day, not the UTC day: a 20:15 ET Friday kickoff is already Saturday in UTC.

Run this module:  uv run pytest tests/unit/test_game_lock.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from utils import game_lock as gl

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REPO_ROOT = Path(__file__).resolve().parents[2]

#: (case id, kickoff instant, expected lock as an ISO string with its offset).
KICKOFF_TABLE: tuple[tuple[str, datetime, str], ...] = (
    (
        "thursday",
        datetime(2023, 9, 7, 20, 20, tzinfo=ET),
        "2023-09-06T18:00:00-04:00",
    ),
    (
        "friday_night",
        datetime(2023, 12, 15, 20, 15, tzinfo=ET),
        "2023-12-14T18:00:00-05:00",
    ),
    (
        "saturday",
        datetime(2023, 12, 16, 16, 30, tzinfo=ET),
        "2023-12-15T18:00:00-05:00",
    ),
    (
        "sunday_morning_international",
        datetime(2023, 10, 1, 9, 30, tzinfo=ET),
        "2023-09-30T18:00:00-04:00",
    ),
    (
        "sunday",
        datetime(2023, 10, 15, 13, 0, tzinfo=ET),
        "2023-10-14T18:00:00-04:00",
    ),
    (
        "monday",
        datetime(2023, 9, 11, 20, 15, tzinfo=ET),
        "2023-09-10T18:00:00-04:00",
    ),
    (
        "tuesday",
        datetime(2020, 12, 8, 20, 5, tzinfo=ET),
        "2020-12-07T18:00:00-05:00",
    ),
    (
        "wednesday",
        datetime(2020, 12, 2, 15, 40, tzinfo=ET),
        "2020-12-01T18:00:00-05:00",
    ),
    # Spring-forward is 2024-03-10 02:00 ET. A kickoff ON that day locks on the Saturday,
    # which is still EST; a kickoff the day after locks ON the transition day, in EDT.
    (
        "spring_forward_kickoff_on_transition_day",
        datetime(2024, 3, 10, 13, 0, tzinfo=ET),
        "2024-03-09T18:00:00-05:00",
    ),
    (
        "spring_forward_lock_on_transition_day",
        datetime(2024, 3, 11, 20, 15, tzinfo=ET),
        "2024-03-10T18:00:00-04:00",
    ),
    # Fall-back is 2023-11-05 02:00 ET. The Sunday kickoff locks on Saturday (still EDT);
    # the Monday night kickoff locks ON the transition day, in EST.
    (
        "fall_back_kickoff_on_transition_day",
        datetime(2023, 11, 5, 13, 0, tzinfo=ET),
        "2023-11-04T18:00:00-04:00",
    ),
    (
        "fall_back_lock_on_transition_day",
        datetime(2023, 11, 6, 20, 15, tzinfo=ET),
        "2023-11-05T18:00:00-05:00",
    ),
)


class TestTheLockTable:
    """One row per kickoff shape. Each must return its own ET day-before 18:00."""

    @pytest.mark.parametrize(
        ("kickoff", "expected"),
        [(row[1], row[2]) for row in KICKOFF_TABLE],
        ids=[row[0] for row in KICKOFF_TABLE],
    )
    def test_each_shape_locks_at_18_et_the_day_before(
        self, kickoff: datetime, expected: str
    ) -> None:
        lock = gl.game_lock(kickoff)
        assert lock.isoformat() == expected
        wall = lock.astimezone(ET)
        assert (wall.hour, wall.minute, wall.second) == (gl.LOCK_HOUR_ET, 0, 0)

    def test_the_table_covers_every_required_shape(self) -> None:
        """NON-VACUITY: the SPEC R1 acceptance names ten shapes; every one is present."""
        ids = {row[0] for row in KICKOFF_TABLE}
        for required in (
            "thursday",
            "friday_night",
            "saturday",
            "sunday_morning_international",
            "sunday",
            "monday",
            "tuesday",
            "wednesday",
        ):
            assert required in ids
        assert any(i.startswith("spring_forward") for i in ids)
        assert any(i.startswith("fall_back") for i in ids)


class TestTheEtCalendarDayNotTheUtcDay:
    """Silver stores ``kickoff_et`` as a UTC instant. The lock must still read the ET date."""

    def test_a_friday_night_kickoff_stored_in_utc_locks_on_thursday(self) -> None:
        # 2023-12-15 20:15 ET is 2023-12-16 01:15 UTC -- already SATURDAY in UTC. A rule
        # that took the UTC day before would lock on Friday 18:00 ET, after the kickoff's
        # own ET day had begun.
        kickoff_utc = pd.Timestamp("2023-12-16 01:15:00", tz="UTC")
        assert gl.game_lock(kickoff_utc).isoformat() == "2023-12-14T18:00:00-05:00"

    def test_the_first_game_of_2002_as_silver_stores_it(self) -> None:
        # 2002_W01_SF@NYG: silver reads 2002-09-06 00:30:00+00:00, i.e. Thursday 20:30 ET.
        kickoff_utc = pd.Timestamp("2002-09-06 00:30:00", tz="UTC")
        assert gl.game_lock(kickoff_utc).isoformat() == "2002-09-04T18:00:00-04:00"

    def test_it_is_not_the_retired_preceding_friday_rule(self) -> None:
        """A Friday-night game under the retired rule locked a WEEK earlier (the prior Friday)."""
        lock = gl.game_lock(datetime(2023, 12, 15, 20, 15, tzinfo=ET))
        assert lock.astimezone(ET).weekday() == 3  # Thursday, not the prior Friday
        assert lock.astimezone(ET).date() != datetime(2023, 12, 8).date()


class TestRefusals:
    """A naive instant and a missing kickoff both RAISE. Neither is defaulted."""

    def test_a_naive_kickoff_raises(self) -> None:
        with pytest.raises(ValueError, match="NO timezone"):
            gl.game_lock(datetime(2023, 12, 15, 20, 15))

    @pytest.mark.parametrize(
        "missing",
        [None, pd.NaT, np.nan, float("nan")],
        ids=["none", "nat", "np_nan", "nan"],
    )
    def test_a_missing_kickoff_raises_by_name(self, missing: object) -> None:
        with pytest.raises(gl.MissingKickoffError, match="2010_W14_NYG@MIN"):
            gl.game_lock(missing, game_id="2010_W14_NYG@MIN")

    def test_a_missing_kickoff_without_a_game_id_still_raises(self) -> None:
        with pytest.raises(gl.MissingKickoffError):
            gl.game_lock(None)

    def test_missing_kickoff_error_is_a_value_error(self) -> None:
        assert issubclass(gl.MissingKickoffError, ValueError)


class TestAdmissibility:
    """``is_admissible`` is ``information_time <= lock``: at-lock passes, lock + 1 s fails."""

    LOCK = datetime(2023, 12, 14, 18, 0, tzinfo=ET)

    def test_at_lock_is_admissible(self) -> None:
        assert gl.is_admissible(self.LOCK, self.LOCK) is True

    def test_one_second_after_the_lock_is_refused(self) -> None:
        assert gl.is_admissible(self.LOCK + timedelta(seconds=1), self.LOCK) is False

    def test_one_second_before_the_lock_is_admissible(self) -> None:
        assert gl.is_admissible(self.LOCK - timedelta(seconds=1), self.LOCK) is True

    def test_the_same_instant_in_a_different_zone_is_the_same_instant(self) -> None:
        """Convert, never relabel: 23:00 UTC IS 18:00 EST."""
        at_lock_utc = datetime(2023, 12, 14, 23, 0, tzinfo=UTC)
        assert gl.is_admissible(at_lock_utc, self.LOCK) is True
        assert gl.is_admissible(at_lock_utc + timedelta(seconds=1), self.LOCK) is False

    def test_a_naive_information_time_raises(self) -> None:
        with pytest.raises(ValueError, match="NO timezone"):
            gl.is_admissible(datetime(2023, 12, 14, 18, 0), self.LOCK)

    def test_a_naive_lock_raises(self) -> None:
        with pytest.raises(ValueError, match="NO timezone"):
            gl.is_admissible(self.LOCK, datetime(2023, 12, 14, 18, 0))


class TestLockFrame:
    """One lock per game, built once per build, refusing any game with no kickoff."""

    def _games(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["2023_W15_LAC@LV", "2023_W15_MIN@CIN"],
                "kickoff_et": pd.to_datetime(
                    ["2023-12-15 01:15:00", "2023-12-16 21:30:00"], utc=True
                ),
            }
        )

    def test_it_is_game_id_indexed_and_tz_aware(self) -> None:
        locks = gl.lock_frame(self._games())
        assert list(locks.index) == ["2023_W15_LAC@LV", "2023_W15_MIN@CIN"]
        assert locks.index.name == "game_id"
        assert locks.dt.tz is not None
        # 2023-12-15 01:15 UTC is Thursday 20:15 ET -> Wednesday 18:00 ET.
        assert locks["2023_W15_LAC@LV"].isoformat() == "2023-12-13T18:00:00-05:00"
        # 2023-12-16 21:30 UTC is Saturday 16:30 ET -> Friday 18:00 ET.
        assert locks["2023_W15_MIN@CIN"].isoformat() == "2023-12-15T18:00:00-05:00"

    def test_every_value_agrees_with_game_lock(self) -> None:
        """ONE rule: the frame is game_lock applied per game, not a second derivation."""
        games = self._games()
        locks = gl.lock_frame(games)
        for gid, kickoff in zip(games["game_id"], games["kickoff_et"], strict=True):
            assert locks[gid] == gl.game_lock(kickoff)

    def test_a_null_kickoff_raises_naming_every_offending_game(self) -> None:
        games = pd.DataFrame(
            {
                "game_id": ["A", "B", "C"],
                "kickoff_et": pd.to_datetime(
                    ["2023-12-15 01:15:00", None, None], utc=True
                ),
            }
        )
        with pytest.raises(gl.MissingKickoffError) as exc:
            gl.lock_frame(games)
        assert "B" in str(exc.value)
        assert "C" in str(exc.value)

    def test_a_duplicate_game_id_raises(self) -> None:
        games = pd.concat([self._games(), self._games().iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="2023_W15_LAC@LV"):
            gl.lock_frame(games)


class TestTheLeafImportProperty:
    """Importing ``utils.game_lock`` drags in no ``backtest``/``models``/``features``/``scripts``.

    The measurement is taken IMMEDIATELY AFTER THE IMPORT AND BEFORE ANY CALL. The strict
    naive-instant parser is reached through a LAZY import on the CALL path, so a call
    legitimately pulls ``scripts`` in; an import-plus-call assertion would go red on a
    correct implementation, and the tempting fix -- weakening the leaf assertion -- would
    throw away the module's whole import-cycle rationale.
    """

    def test_a_fresh_interpreter_import_is_a_leaf(self) -> None:
        probe = (
            "import sys\n"
            "import utils.game_lock\n"
            "snap = list(sys.modules)\n"
            "bad = sorted({m.split('.')[0] for m in snap} & "
            "{'backtest', 'models', 'features', 'scripts'})\n"
            "print(len(snap))\n"
            "print(','.join(bad))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        count_line, bad_line = result.stdout.strip().splitlines()[-2:]
        assert int(count_line) > 0, "empty sys.modules snapshot would pass vacuously"
        assert bad_line == "", f"importing utils.game_lock pulled in: {bad_line}"


class TestTheNamedConstantAndEvidence:
    def test_the_lock_hour_is_18(self) -> None:
        assert gl.LOCK_HOUR_ET == 18

    def test_the_zone_is_new_york(self) -> None:
        assert ZoneInfo("America/New_York") == gl.ET

    def test_the_evidence_tuple_carries_the_measurement(self) -> None:
        text = " ".join(gl.LOCK_RULE_EVIDENCE)
        assert len(gl.LOCK_RULE_EVIDENCE) >= 3
        assert "324" in text
        assert "D33.2-01" in text
