"""Regression tests for ``api.cache._load_betting_bets`` (Phase 17 loader).

Guards the HIGH-value Pitfall-1 behavior that was previously only verified by
hand against the real ``betting_simulation.csv``: the CSV ``outcome`` column is
an OBJECT column holding real Python ``bool`` values mixed with ``float`` NaN
(it is NOT strings). The loader must normalize NaN -> ``None`` so DuckDB stores
a nullable BOOLEAN -- ``True`` win / ``False`` loss / SQL ``NULL`` push.

Every naive cast corrupts the push rows and must fail these tests:
- ``astype(bool)`` turns NaN into ``True`` (push counted as a win)
- ``bool("False")`` is ``True``
- ``.map({"True": ...})`` returns all-NaN (keys are bools, not strings)

The loader does a positional ``INSERT ... SELECT *`` into an EXISTING
``betting_bets`` table (the table is created from ``CACHE_SCHEMA``; it has no
PRIMARY KEY per WR-02 so every per-bet row is preserved), so the test creates
the table first then asserts the stored rows back.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from api.cache import CACHE_SCHEMA, _load_betting_bets

# The 15 betting_bets columns in the exact CSV header order the loader selects
# (matches the betting_bets CREATE TABLE block in CACHE_SCHEMA).
_BETTING_BETS_COLS = [
    "game_id",
    "season",
    "week",
    "target",
    "bet_side",
    "model_value",
    "market_value",
    "edge",
    "slipped_line",
    "odds",
    "flat_stake",
    "kelly_stake",
    "outcome",
    "payout_flat",
    "payout_kelly",
]


def _create_betting_bets_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create just the schema tables so betting_bets exists before the INSERT.

    The loader runs ``INSERT INTO betting_bets SELECT * FROM subset`` against an
    already-created table (in production ``populate_cache`` builds the whole
    schema first), so the table must exist or the INSERT raises.
    """
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)


def _write_betting_csv(outputs_dir: Path) -> None:
    """Write a tiny betting_simulation.csv with REAL Python outcome values.

    Four rows exercising the Pitfall-1 surface:
      - win  : outcome bool True,  kelly_stake > 0 (recommended scope)
      - loss : outcome bool False, kelly_stake > 0 (recommended scope)
      - push : outcome float NaN,  kelly_stake > 0
      - win  : outcome bool True,  kelly_stake == 0 (dropped by recommended)

    The frame is built record-wise so ``outcome`` stays an object column of
    {True, False, NaN}; after the CSV round-trip ``pd.read_csv`` re-infers that
    same object column ([True, False, nan]) -- the exact shape the real CSV has.
    """
    records = [
        {
            "game_id": "2021_W01_DAL@TB",
            "season": 2021,
            "week": 1,
            "target": "wp",
            "bet_side": "home",
            "model_value": 0.62,
            "market_value": 0.55,
            "edge": 0.07,
            "slipped_line": np.nan,  # WP rows carry NaN slipped_line (real CSV)
            "odds": -150.0,
            "flat_stake": 100.0,
            "kelly_stake": 120.0,  # recommended (kelly_stake > 0)
            "outcome": True,  # real bool win
            "payout_flat": 66.7,
            "payout_kelly": 80.0,
        },
        {
            "game_id": "2021_W02_KC@BUF",
            "season": 2021,
            "week": 2,
            "target": "ats",
            "bet_side": "away",
            "model_value": -6.5,
            "market_value": -4.0,
            "edge": 2.5,
            "slipped_line": -3.5,
            "odds": -110.0,
            "flat_stake": 100.0,
            "kelly_stake": 90.0,  # recommended (kelly_stake > 0)
            "outcome": False,  # real bool loss
            "payout_flat": -100.0,
            "payout_kelly": -90.0,
        },
        {
            "game_id": "2023_W15_DAL@PHI",
            "season": 2023,
            "week": 15,
            "target": "ats",
            "bet_side": "away",
            "model_value": 3.0,
            "market_value": 3.0,
            "edge": 1.2,
            "slipped_line": 3.0,
            "odds": -110.0,
            "flat_stake": 100.0,
            "kelly_stake": 60.0,  # recommended (kelly_stake > 0)
            "outcome": float("nan"),  # PUSH -> must store SQL NULL, never True
            "payout_flat": 0.0,
            "payout_kelly": 0.0,
        },
        {
            "game_id": "2022_W01_SF@CHI",
            "season": 2022,
            "week": 1,
            "target": "ou",
            "bet_side": "over",
            "model_value": 48.0,
            "market_value": 45.5,
            "edge": 2.5,
            "slipped_line": 45.0,
            "odds": -110.0,
            "flat_stake": 100.0,
            "kelly_stake": 0.0,  # NOT recommended (kelly_stake == 0)
            "outcome": True,  # real bool win
            "payout_flat": 90.9,
            "payout_kelly": 0.0,
        },
    ]
    df = pd.DataFrame.from_records(records, columns=_BETTING_BETS_COLS)
    # Sanity: outcome must be object dtype (real bool + NaN), not bool/float.
    assert df["outcome"].dtype == object
    df.to_csv(outputs_dir / "betting_simulation.csv", index=False)


def test_betting_bets_load_parses_push_outcome_to_null(tmp_path: Path) -> None:
    """Push (NaN) outcome stores SQL NULL; win stores True, loss stores False.

    This is the core Pitfall-1 regression guard. A loader that cast the object
    column with ``astype(bool)`` would store ``True`` for the push row -- this
    test asserts ``outcome IS NULL`` for that row precisely to catch that.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_betting_csv(outputs_dir)

    conn = duckdb.connect()
    try:
        _create_betting_bets_table(conn)

        inserted = _load_betting_bets(conn, outputs_dir)

        # Loader returns the inserted row count (all 4 per-bet rows preserved).
        assert inserted == 4
        total = conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()[0]
        assert total == 4

        # Map each game_id to its stored outcome to assert the exact tri-state.
        stored = dict(
            conn.execute("SELECT game_id, outcome FROM betting_bets").fetchall()
        )

        # Win -> True (real Python bool, not None / not a string).
        assert stored["2021_W01_DAL@TB"] is True
        # Loss -> False.
        assert stored["2021_W02_KC@BUF"] is False
        # Push (NaN in CSV) -> SQL NULL (None), NOT True. <-- the regression edge.
        assert stored["2023_W15_DAL@PHI"] is None

        # And via SQL identity predicates, the push row is the ONLY NULL, and it
        # is neither TRUE nor FALSE (an astype(bool) bug would flip it to TRUE).
        null_rows = conn.execute(
            "SELECT game_id FROM betting_bets WHERE outcome IS NULL"
        ).fetchall()
        assert null_rows == [("2023_W15_DAL@PHI",)]

        true_rows = {
            r[0]
            for r in conn.execute(
                "SELECT game_id FROM betting_bets WHERE outcome IS TRUE"
            ).fetchall()
        }
        assert true_rows == {"2021_W01_DAL@TB", "2022_W01_SF@CHI"}

        false_rows = {
            r[0]
            for r in conn.execute(
                "SELECT game_id FROM betting_bets WHERE outcome IS FALSE"
            ).fetchall()
        }
        assert false_rows == {"2021_W02_KC@BUF"}
    finally:
        conn.close()


def test_betting_bets_load_preserves_kelly_stake_zero_and_positive(
    tmp_path: Path,
) -> None:
    """Both kelly_stake>0 and kelly_stake==0 rows survive (no scope drop at load).

    The recommended-scope filter (kelly_stake > 0) is applied downstream, not by
    the loader -- every per-bet row must be present so the dashboard can compute
    both the ``all`` and ``recommended`` scopes. A plain INSERT (no PK) keeps
    them all.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_betting_csv(outputs_dir)

    conn = duckdb.connect()
    try:
        _create_betting_bets_table(conn)
        _load_betting_bets(conn, outputs_dir)

        positive = conn.execute(
            "SELECT COUNT(*) FROM betting_bets WHERE kelly_stake > 0"
        ).fetchone()[0]
        zero = conn.execute(
            "SELECT COUNT(*) FROM betting_bets WHERE kelly_stake = 0"
        ).fetchone()[0]

        # 3 recommended rows + 1 dropped-by-recommended row = the full ledger.
        assert positive == 3
        assert zero == 1
    finally:
        conn.close()


def test_betting_bets_load_missing_csv_returns_zero(tmp_path: Path) -> None:
    """A directory with no betting_simulation.csv returns 0 and does not raise."""
    empty_outputs = tmp_path / "no_csv_here"
    empty_outputs.mkdir()

    conn = duckdb.connect()
    try:
        _create_betting_bets_table(conn)

        inserted = _load_betting_bets(conn, empty_outputs)

        assert inserted == 0
        total = conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()[0]
        assert total == 0
    finally:
        conn.close()
