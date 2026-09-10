"""The `/bets` freeze instant survives storage unshifted -- ACROSS DuckDB sessions (CR-01).

WHY THIS FILE EXISTS SEPARATELY FROM ``tests/api/test_bets_page.py``: the defect it pins is
INVISIBLE to any test that writes and reads inside ONE DuckDB session. The freeze column used to
be a naive ``TIMESTAMP``, so a tz-aware UTC instant was cast through the session ``TimeZone`` on
the way in; a reader on the same connection saw the same shifted value the writer put there and
the round trip looked self-consistent. The shift only shows up when the recovered instant is
compared against the ORIGINAL -- which is what these tests do, and they close the connection
between the write and the read so the storage is genuinely re-parsed.

The stake: ``_is_bet_cache_stale`` compares a true-UTC populated-at marker against this instant.
A four- or five-hour shift moved the staleness threshold by the server's own UTC offset, so a bet
list priced BEFORE the lines stopped moving was served with no refusal on any host west of UTC.

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
from api.routes.pages import _as_utc, _is_bet_cache_stale
from api.services import DataService, clear_cache
from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

_SEASON = 2025
_WEEK = 2
# A Friday-night gameday, so its own freeze is the PRIOR Friday 6 PM ET -- the real rule, taken
# from the one source, not a hand-authored instant.
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
                [{"season": _SEASON, "week": _WEEK, "game_freeze_ts": freeze}]
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


@pytest.mark.parametrize("session_tz", _SESSION_ZONES)
def test_the_freeze_instant_survives_storage_unshifted(
    tmp_path: Path, session_tz: str
) -> None:
    """materialize -> close -> reopen -> get_bet_week_freeze -> _as_utc is the IDENTITY (CR-01)."""
    expected = get_synthetic_snapshot_ts(_GAMEDAY)
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
def test_a_cache_populated_before_the_freeze_is_refused_in_every_session_zone(
    tmp_path: Path, session_tz: str
) -> None:
    """The concrete CR-01 scenario: populated 20:00Z, freeze 22:00Z -- STALE everywhere.

    Under the naive column this evaluated as ``20:00Z < 18:00Z -> False`` on an
    America/New_York host and the stale week was served.
    """
    freeze = datetime(2025, 9, 12, 22, 0, tzinfo=UTC)
    populated_at = datetime(2025, 9, 12, 20, 0, tzinfo=UTC).isoformat()
    db_path = tmp_path / f"stale_{session_tz.replace('/', '_')}.duckdb"

    _write_freeze(db_path, session_tz, pd.Timestamp(freeze))
    recovered = _read_freeze(db_path, session_tz)

    assert _is_bet_cache_stale(populated_at, recovered) is True, (
        "a bet list populated two hours BEFORE the week's latest line freeze was not refused "
        f"under session TimeZone {session_tz}"
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
