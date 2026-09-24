"""The `/bets` freeze instant survives storage unshifted -- ACROSS DuckDB sessions (CR-01).

WHY THIS FILE EXISTS SEPARATELY FROM ``tests/api/test_bets_page.py``: the defect it pins is
INVISIBLE to any test that writes and reads inside ONE DuckDB session. The freeze column used to
be a naive ``TIMESTAMP``, so a tz-aware UTC instant was cast through the session ``TimeZone`` on
the way in; a reader on the same connection saw the same shifted value the writer put there and
the round trip looked self-consistent. The shift only shows up when the recovered instant is
compared against the ORIGINAL -- which is what these tests do, and they close the connection
between the write and the read so the storage is genuinely re-parsed.

The stake: ``/bets`` judges each game against its own lock (``_lock_has_passed``, 33.2 review C2
CR-02), and the per-game lock table is written by the same writer as this one. A four- or five-hour
shift would move every game's lock by the server's own UTC offset, so a game whose lock had passed
would read as not locked yet -- a missing list reported as a pending one.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pytest

from api.cache import CACHE_SCHEMA, materialize_bet_week_freeze
from api.routes.pages import _as_utc, _lock_has_passed
from api.services import DataService, clear_cache
from scripts.ingest_historical_odds import gameday_lock

_SEASON = 2025
_WEEK = 2
_GAME_ID = "2025_W02_CHI@DET"
# A Sunday gameday, so its own lock is Saturday 18:00 ET -- the real rule (D33.2-01), taken from
# the one source, not a hand-authored instant. The cache column keeps its published name
# ``latest_game_freeze_ts`` (a rename is HOST-07's schema change); only the instant changed.
_GAMEDAY = "2025-09-14"

# Every zone the shift would be a different size in. America/New_York is the machine this was
# reproduced on; Asia/Tokyo is east of UTC, where the old code was too STRICT rather than too
# lenient; UTC is the one host on which the old code happened to be correct.
_SESSION_ZONES = ("America/New_York", "Asia/Tokyo", "UTC", "America/Los_Angeles")


def _write_freeze(db_path: Path, session_tz: str, freeze: pd.Timestamp) -> None:
    """Write ONE freeze row through the real writer, then CLOSE the connection."""
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(f"SET TimeZone='{session_tz}'")
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": _GAME_ID,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": freeze,
                    }
                ]
            ),
        )
    finally:
        conn.close()


def _read_freeze(db_path: Path, session_tz: str) -> datetime | None:
    """Read it back through the real getter on a FRESH connection."""
    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        conn.execute(f"SET TimeZone='{session_tz}'")
        service = DataService(conn)
        return _as_utc(service.get_bet_week_freeze(_SEASON, _WEEK))
    finally:
        conn.close()
        clear_cache()


def _read_game_lock(db_path: Path, session_tz: str) -> object:
    """Read the per-game lock back, RAW, through the real getter on a FRESH connection."""
    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        conn.execute(f"SET TimeZone='{session_tz}'")
        (game,) = DataService(conn).get_bet_game_locks(_SEASON, _WEEK)
        assert game["game_id"] == _GAME_ID
        return game["lock_ts"]
    finally:
        conn.close()
        clear_cache()


@pytest.mark.parametrize("session_tz", _SESSION_ZONES)
def test_the_freeze_instant_survives_storage_unshifted(
    tmp_path: Path, session_tz: str
) -> None:
    """materialize -> close -> reopen -> get_bet_week_freeze -> _as_utc is the IDENTITY (CR-01)."""
    expected = gameday_lock(_GAMEDAY)
    db_path = tmp_path / f"freeze_{session_tz.replace('/', '_')}.duckdb"

    _write_freeze(db_path, session_tz, pd.Timestamp(expected))
    recovered = _read_freeze(db_path, session_tz)

    assert recovered is not None, (
        "the freeze could not be read back at all; an unreadable threshold makes no claim and "
        "the stale-cache hard-block cannot fire"
    )
    assert recovered == expected, (
        f"the freeze came back as {recovered} instead of {expected} under session TimeZone "
        f"{session_tz}. The staleness threshold is off by "
        f"{(recovered - expected).total_seconds() / 3600:.1f} hours, which is exactly the "
        "defect CR-01 named: a bet list priced before the lines stopped moving is served with "
        "no refusal."
    )


@pytest.mark.parametrize("session_tz", _SESSION_ZONES)
def test_a_game_lock_is_judged_at_its_true_instant_in_every_session_zone(
    tmp_path: Path, session_tz: str
) -> None:
    """The CR-01 scenario on the per-game lock: 22:00Z is passed at 22:00Z, not at 18:00Z.

    Under a naive column the lock would come back shifted by the session offset, and a game
    one minute before its lock would read as locked (or four hours after it as not locked).
    """
    lock = datetime(2025, 9, 12, 22, 0, tzinfo=UTC)
    db_path = tmp_path / f"lock_{session_tz.replace('/', '_')}.duckdb"

    _write_freeze(db_path, session_tz, pd.Timestamp(lock))
    recovered = _read_game_lock(db_path, session_tz)

    assert _as_utc(recovered) == lock
    assert (
        _lock_has_passed(recovered, datetime(2025, 9, 12, 21, 59, tzinfo=UTC)) is False
    ), (
        f"a game one minute before its lock read as locked under session TimeZone {session_tz}"
    )
    assert _lock_has_passed(recovered, lock) is True, (
        f"a game AT its lock did not read as locked under session TimeZone {session_tz}"
    )


def test_a_naive_instant_is_refused_rather_than_read_as_utc() -> None:
    """``_as_utc`` no longer guesses a zone -- guessing is what hid the shift (CR-01)."""
    assert _as_utc(datetime(2025, 9, 12, 18, 0)) is None
    assert _as_utc("2025-09-12T18:00:00") is None
    assert _as_utc("2025-09-12T18:00:00+00:00") == datetime(
        2025, 9, 12, 18, 0, tzinfo=UTC
    )
    assert _as_utc(
        datetime(2025, 9, 12, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    ) == (datetime(2025, 9, 12, 18, 0, tzinfo=UTC))
