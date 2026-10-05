"""The live odds store ACCUMULATES captures; a re-pull cannot destroy an earlier one.

THE DEFECT (Plan 33.2-27 Task 1, measured 2026-09-20)
-----------------------------------------------------
The live capture wrote silver ``odds_snapshot`` through ``data.storage.save_dataframe``,
whose append merge REMOVES every stored row whose ``game_id`` matches an incoming row
before appending. So a second pull of a game -- which a daily cadence makes routine --
deleted the first. What the market said at an earlier instant is exactly the evidence the
lock rule needs, and it was being thrown away on every re-pull.

THE KEY, AND WHY IT HAS FOUR COLUMNS
------------------------------------
The historical ingest enforces one row per ``(game_id, sportsbook, snapshot_ts)``
(``data.schemas.ODDS_KEY_COLUMNS``). Since Plan 33.2-02 a live row's ``snapshot_ts`` is its
game's LOCK, which is the same for every pull of that game, so that triple alone would still
collapse two pulls into one. The capture instant ``created_at`` is what tells two pulls
apart, so the live key is the historical triple plus ``created_at``
(``data.schemas.LIVE_ODDS_CAPTURE_KEY``). A re-write of the SAME capture (same instant) is
still idempotent.

EVERYTHING HERE IS SANDBOXED under ``tmp_path``. No network, no production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import data.storage as storage_mod
import scripts.ingest_odds as ingest_odds_module
from backtest.bet_selector import assert_real_odds
from data.schemas import OddsSchema
from data.storage import ParquetManager
from scripts.ingest_odds import OddsDataIngester

GAME_ID = "2026_W04_DAL@PHI"
LOCK = datetime(2026, 9, 26, 22, 0, tzinfo=UTC)
FIRST_CAPTURE = datetime(2026, 9, 25, 21, 30, tzinfo=UTC)
LATER_CAPTURE = datetime(2026, 9, 26, 21, 30, tzinfo=UTC)

# The nine bookmaker keys a real live Odds API response carried (bronze
# odds_raw_bronze_2025_W01.parquet and _W05.parquet, read 2026-09-24).
REAL_LIVE_BOOKS = (
    "betmgm",
    "betonlineag",
    "betrivers",
    "betus",
    "bovada",
    "draftkings",
    "fanduel",
    "lowvig",
    "mybookieag",
)


def _capture(
    *,
    created_at: datetime,
    sportsbook: str = "draftkings",
    spread: float = 3.5,
    snapshot_ts: datetime = LOCK,
    game_id: str = GAME_ID,
) -> pd.DataFrame:
    """One validated live row, in the shape the live ingest hands the store."""
    row = OddsSchema(
        game_id=game_id,
        snapshot_ts=snapshot_ts,
        sportsbook=sportsbook,
        spread=spread,
        total=44.5,
        ml_home=-150,
        ml_away=130,
        is_live=False,
        last_update=created_at,
        created_at=created_at,
    ).model_dump()
    return pd.DataFrame([row])


def _stored(base: Path) -> pd.DataFrame:
    return pd.read_parquet(base / "silver" / "odds_snapshot.parquet")


class TestCapturesAccumulate:
    """The composite-key writer keeps every distinct capture."""

    def test_two_captures_of_one_game_at_different_instants_both_persist(
        self, tmp_path: Path
    ):
        storage_mod.append_odds_captures(
            _capture(created_at=FIRST_CAPTURE, spread=3.5), base_path=tmp_path
        )
        storage_mod.append_odds_captures(
            _capture(created_at=LATER_CAPTURE, spread=2.5), base_path=tmp_path
        )

        stored = _stored(tmp_path)
        assert len(stored) == 2, f"expected both captures, found {len(stored)} row(s)"
        by_instant = stored.set_index(pd.to_datetime(stored["created_at"], utc=True))
        assert by_instant.loc[pd.Timestamp(FIRST_CAPTURE), "spread"] == 3.5
        assert by_instant.loc[pd.Timestamp(LATER_CAPTURE), "spread"] == 2.5

    def test_two_captures_with_different_snapshot_ts_both_persist(self, tmp_path: Path):
        other_lock = datetime(2026, 9, 27, 22, 0, tzinfo=UTC)
        storage_mod.append_odds_captures(
            _capture(created_at=FIRST_CAPTURE), base_path=tmp_path
        )
        storage_mod.append_odds_captures(
            _capture(created_at=FIRST_CAPTURE, snapshot_ts=other_lock),
            base_path=tmp_path,
        )
        assert len(_stored(tmp_path)) == 2

    def test_a_rerun_of_the_identical_capture_is_idempotent(self, tmp_path: Path):
        capture = _capture(created_at=FIRST_CAPTURE)
        storage_mod.append_odds_captures(capture, base_path=tmp_path)
        storage_mod.append_odds_captures(capture, base_path=tmp_path)
        assert len(_stored(tmp_path)) == 1

    def test_two_books_at_the_same_instant_both_persist(self, tmp_path: Path):
        both = pd.concat(
            [
                _capture(created_at=FIRST_CAPTURE, sportsbook="draftkings"),
                _capture(created_at=FIRST_CAPTURE, sportsbook="fanduel"),
            ],
            ignore_index=True,
        )
        storage_mod.append_odds_captures(both, base_path=tmp_path)
        assert sorted(_stored(tmp_path)["sportsbook"]) == ["draftkings", "fanduel"]

    def test_the_live_key_extends_the_historical_key_by_the_capture_instant(self):
        from data.schemas import LIVE_ODDS_CAPTURE_KEY, ODDS_KEY_COLUMNS
        from scripts.ingest_historical_odds import ODDS_KEY_COLUMNS as historical_key

        assert historical_key is ODDS_KEY_COLUMNS, "two key conventions, not one"
        assert (*ODDS_KEY_COLUMNS, "created_at") == LIVE_ODDS_CAPTURE_KEY


class TestTheDestroyedCaptureDefect:
    """Regression control: the old path destroys capture A; the new one keeps it."""

    def test_the_old_game_id_merge_destroys_the_earlier_capture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """CONTROL: proves the fixture reaches the defect, so the next test means something."""
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )
        storage_mod.save_dataframe(
            _capture(created_at=FIRST_CAPTURE), "odds_snapshot", save_to_db=False
        )
        storage_mod.save_dataframe(
            _capture(created_at=LATER_CAPTURE), "odds_snapshot", save_to_db=False
        )
        stored = _stored(tmp_path)
        assert len(stored) == 1
        assert pd.Timestamp(stored["created_at"].iloc[0]) == pd.Timestamp(LATER_CAPTURE)

    def test_capture_a_survives_capture_b(self, tmp_path: Path):
        storage_mod.append_odds_captures(
            _capture(created_at=FIRST_CAPTURE), base_path=tmp_path
        )
        storage_mod.append_odds_captures(
            _capture(created_at=LATER_CAPTURE), base_path=tmp_path
        )
        instants = set(pd.to_datetime(_stored(tmp_path)["created_at"], utc=True))
        assert pd.Timestamp(FIRST_CAPTURE) in instants

    def test_the_store_is_non_empty_after_the_writes(self, tmp_path: Path):
        """Non-vacuity: 'nothing was destroyed' cannot pass because nothing was written."""
        storage_mod.append_odds_captures(
            _capture(created_at=FIRST_CAPTURE), base_path=tmp_path
        )
        assert len(_stored(tmp_path)) > 0


# ---------------------------------------------------------------------------
# One lock for the odds store (Phase 34, LDGR-07 concurrency, D-08)
# ---------------------------------------------------------------------------


class TestTheOddsStoreLock:
    """The daily decision capture and the closing capture serialize on ONE OS lock.

    Both write ``silver/odds_snapshot.parquet`` by read-merge-replace, so two interleaved writes
    would lose one capture. ``append_odds_captures`` takes ``silver/.odds_snapshot.lock`` around
    the composite upsert with a bounded wait: a held lock is waited for, and a lock held past the
    wait is refused by name with nothing written.
    """

    @staticmethod
    def _lock_path(base: Path) -> Path:
        return base / "silver" / storage_mod.ODDS_STORE_LOCK_NAME

    def test_the_lock_and_its_wait_are_the_named_constants(self):
        assert storage_mod.ODDS_STORE_LOCK_NAME == ".odds_snapshot.lock"
        assert storage_mod.ODDS_STORE_LOCK_WAIT_SECONDS == 120.0

    def test_append_odds_captures_refuses_past_the_bounded_wait(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from utils.file_lock import FileLockHeldError, exclusive_file_lock

        monkeypatch.setattr(storage_mod, "ODDS_STORE_LOCK_WAIT_SECONDS", 0.3)
        (tmp_path / "silver").mkdir()
        with (
            exclusive_file_lock(self._lock_path(tmp_path)),
            pytest.raises(FileLockHeldError, match=r"\.odds_snapshot\.lock"),
        ):
            storage_mod.append_odds_captures(
                _capture(created_at=FIRST_CAPTURE), base_path=tmp_path
            )
        assert not (tmp_path / "silver" / "odds_snapshot.parquet").exists(), (
            "a refused write must write nothing"
        )

    def test_append_odds_captures_waits_for_the_holder_then_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import threading
        import time

        from utils.file_lock import exclusive_file_lock

        monkeypatch.setattr(storage_mod, "ODDS_STORE_LOCK_WAIT_SECONDS", 10.0)
        (tmp_path / "silver").mkdir()
        held = threading.Event()
        release = threading.Event()

        def _holder() -> None:
            with exclusive_file_lock(self._lock_path(tmp_path)):
                held.set()
                release.wait(timeout=10.0)

        holder = threading.Thread(target=_holder)
        holder.start()
        try:
            assert held.wait(timeout=5.0), "the holder never took the lock"
            threading.Timer(0.6, release.set).start()
            started = time.monotonic()
            storage_mod.append_odds_captures(
                _capture(created_at=FIRST_CAPTURE), base_path=tmp_path
            )
            waited = time.monotonic() - started
        finally:
            release.set()
            holder.join(timeout=10.0)

        assert waited >= 0.4, f"the write did not wait for the holder ({waited:.2f}s)"
        assert len(_stored(tmp_path)) == 1, "the write landed once the lock came free"

    def test_the_lock_file_is_not_a_guarded_data_suffix(self):
        """The test-write guard digests data suffixes; the lock file must not trip it."""
        from tests.data_boundary import TRACKED_SUFFIXES

        assert Path(storage_mod.ODDS_STORE_LOCK_NAME).suffix not in TRACKED_SUFFIXES


# ---------------------------------------------------------------------------
# The live ingest path
# ---------------------------------------------------------------------------


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "season": 2026,
                "week": 4,
                "home_team": "PHI",
                "away_team": "DAL",
                "kickoff_et": pd.Timestamp("2026-09-27T17:00:00Z"),
            }
        ]
    )


def _market(key: str, last_update: str | None, outcomes: list[dict]) -> dict:
    market: dict = {"key": key, "outcomes": outcomes}
    if last_update is not None:
        market["last_update"] = last_update
    return market


def _event(market_updates: tuple[str | None, str | None, str | None]) -> dict:
    h2h, spreads, totals = market_updates
    return {
        "id": "event0",
        "commence_time": "2026-09-27T17:00:00Z",
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "bookmakers": [
            {
                "key": "draftkings",
                "last_update": "2026-09-25T20:00:00Z",
                "markets": [
                    _market(
                        "h2h",
                        h2h,
                        [
                            {"name": "Philadelphia Eagles", "price": -150},
                            {"name": "Dallas Cowboys", "price": 130},
                        ],
                    ),
                    _market(
                        "spreads",
                        spreads,
                        [
                            {
                                "name": "Philadelphia Eagles",
                                "price": -110,
                                "point": -3.5,
                            },
                            {"name": "Dallas Cowboys", "price": -110, "point": 3.5},
                        ],
                    ),
                    _market(
                        "totals",
                        totals,
                        [
                            {"name": "Over", "price": -110, "point": 44.5},
                            {"name": "Under", "price": -110, "point": 44.5},
                        ],
                    ),
                ],
            }
        ],
    }


def _bare_ingester(payload: list[dict]) -> OddsDataIngester:
    class _StandIn:
        def get_nfl_odds(self, **_kwargs):
            return payload

        def close(self):
            return None

    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.api_client = _StandIn()
    ingester.sportsbook_priority = []
    return ingester


class TestTheMarketLevelLastUpdateIsRetained:
    """The per-market ``last_update`` real payloads carry is kept, not discarded."""

    def test_the_latest_market_stamp_is_carried_on_the_row(self):
        ingester = _bare_ingester([])
        ingester._unmatched = []
        ingester._kickoff_disagreements = []
        rows = ingester._process_game_odds(
            _event(
                (
                    "2026-09-25T19:00:00Z",
                    "2026-09-25T19:45:00Z",
                    "2026-09-25T19:30:00Z",
                )
            ),
            schedule=_schedule(),
            locks={GAME_ID: LOCK},
            captured_at=FIRST_CAPTURE,
        )
        assert rows[0]["market_last_update"] == datetime(
            2026, 9, 25, 19, 45, tzinfo=UTC
        )

    def test_absent_market_stamps_are_null_not_manufactured(self):
        ingester = _bare_ingester([])
        ingester._unmatched = []
        ingester._kickoff_disagreements = []
        rows = ingester._process_game_odds(
            _event((None, None, None)),
            schedule=_schedule(),
            locks={GAME_ID: LOCK},
            captured_at=FIRST_CAPTURE,
        )
        assert rows[0]["market_last_update"] is None

    def test_the_schema_declares_the_column(self):
        """Pydantic defaults to extra='ignore': an undeclared column vanishes silently."""
        dumped = OddsSchema(
            game_id=GAME_ID,
            snapshot_ts=LOCK,
            sportsbook="draftkings",
            created_at=FIRST_CAPTURE,
            market_last_update=datetime(2026, 9, 25, 19, 45, tzinfo=UTC),
        ).model_dump()
        assert dumped["market_last_update"] == datetime(2026, 9, 25, 19, 45, tzinfo=UTC)


class TestTheLiveCapturePathAccumulates:
    """The live ingest writes silver through the accumulating writer, proved by identity."""

    def test_the_live_path_uses_neither_latest_wins_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        real_append = storage_mod.append_odds_captures
        appended: list[int] = []
        saved_tables: list[str] = []
        latest_wins_calls: list[str] = []

        def _counting_append(frame, base_path=None):
            appended.append(len(frame))
            return real_append(frame, base_path=tmp_path)

        def _counting_save(frame, table_name, *args, **kwargs):
            saved_tables.append(table_name)

        def _counting_upsert(*args, **kwargs):
            latest_wins_calls.append("upsert_silver")

        monkeypatch.setattr(
            ingest_odds_module, "append_odds_captures", _counting_append
        )
        monkeypatch.setattr(ingest_odds_module, "save_dataframe", _counting_save)
        monkeypatch.setattr(storage_mod, "upsert_silver", _counting_upsert)
        # Pinned before the lock: a post-lock capture writes nothing (33.2 review CR-02).
        monkeypatch.setattr(
            ingest_odds_module, "_observe_capture_instant", lambda: FIRST_CAPTURE
        )

        ingester = _bare_ingester([_event(("2026-09-25T19:00:00Z", None, None))])
        ingester.ingest_odds(season=2026, week=4, schedule=_schedule())

        assert appended == [1], f"the accumulating writer was called {appended}"
        assert "odds_snapshot" not in saved_tables, (
            "the live capture still writes silver odds_snapshot through save_dataframe, "
            "whose game_id merge destroys the previous capture"
        )
        assert latest_wins_calls == []
        assert len(_stored(tmp_path)) == 1


# ---------------------------------------------------------------------------
# The provenance guard
# ---------------------------------------------------------------------------


class TestTheSportsbookAllowlist:
    """Real live books and ``is_live`` pass; an absent or unknown book still raises."""

    @pytest.mark.parametrize("book", REAL_LIVE_BOOKS)
    def test_a_real_live_book_passes(self, book: str):
        assert_real_odds(
            pd.DataFrame(
                {"game_id": [GAME_ID], "sportsbook": [book], "is_live": [False]}
            )
        )

    def test_the_owned_timelines_book_name_passes(self):
        assert_real_odds(
            pd.DataFrame(
                {
                    "game_id": [GAME_ID],
                    "sportsbook": ["consensus_median"],
                    "is_live": [False],
                }
            )
        )

    def test_is_live_true_passes(self):
        assert_real_odds(
            pd.DataFrame(
                {"game_id": [GAME_ID], "sportsbook": ["fanduel"], "is_live": [True]}
            )
        )

    def test_a_row_with_no_sportsbook_is_refused(self):
        with pytest.raises(ValueError, match=GAME_ID):
            assert_real_odds(
                pd.DataFrame(
                    {"game_id": [GAME_ID], "sportsbook": [None], "is_live": [False]}
                )
            )

    def test_an_unrecognised_book_is_refused(self):
        with pytest.raises(ValueError, match="mock_book"):
            assert_real_odds(
                pd.DataFrame(
                    {
                        "game_id": [GAME_ID],
                        "sportsbook": ["mock_book"],
                        "is_live": [False],
                    }
                )
            )

    def test_the_allowlist_is_a_named_constant_that_was_widened(self):
        import backtest.bet_selector as selector

        assert "consensus_median" in selector._ALLOWED_SPORTSBOOKS
        assert set(REAL_LIVE_BOOKS) <= selector._ALLOWED_SPORTSBOOKS
        assert len(selector._ALLOWED_SPORTSBOOKS) > 2
