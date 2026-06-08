"""Smoke + column-order + edge-case suite for ``api.cache.materialize_ou_bet_list`` (Phase 27, 27-04).

BET-01's SECOND consumer (the backtest is the first): a THIN cache-population fn that materializes
the BetSelector bet-list blob into a sibling ``ou_bet_list`` DuckDB table (D27-15, UIAP-01). This
suite mirrors the ``tests/api/test_cache_betting.py`` cache-loader pattern + the boundary style.

Guards (the plan's Task-2 acceptance):
  (a) the round-tripped row count equals the input;
  (b) pushes are stored as SQL NULL (the push-NULL convention -- df[col].where(notna, None));
  (c) the fn does ZERO metric math (UIAP-01): a known precomputed input value round-trips byte-equal;
  (d) the NEW O/U columns (calibrated P(side), per-bet EV, sub-pop label, scaled stake,
      totals_regime) AND the validation_type column are present + populated
      (validation_type == "PROVISIONAL_CONTAMINATED");
  (e) the INSERT is EXPLICIT-COLUMN, not positional: shuffling the input DataFrame's column order
      still lands every value in the correct column (#9, T-27-24);
  (f) EDGE CASES (#8): an empty BetSelector frame writes zero rows without error; a frame missing a
      required odds column raises a named error.

Selectors (``-k``): cache_row_count, cache_push_null, cache_no_math, cache_ou_columns,
cache_validation_type, cache_explicit_column_insert, cache_empty_frame, cache_missing_column.

The new table is ADDITIVE (sibling to betting_bets); existing betting_bets consumers are untouched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from api.cache import OU_BET_LIST_COLUMNS, materialize_ou_bet_list

# The PROVISIONAL_CONTAMINATED structural honesty label every materialized row carries (#6).
_VALIDATION_TYPE = "PROVISIONAL_CONTAMINATED"


def _selector_frame() -> pd.DataFrame:
    """A small BetSelector-shaped selected-bets frame (the blob the cache fn persists).

    Three rows exercise the surface:
      - an UNDER win (outcome True),
      - a high-total OVER loss (outcome False),
      - a push (outcome None -> stored as SQL NULL).

    Every numeric field is PRECOMPUTED by the BetSelector (the cache fn does ZERO math, UIAP-01).
    """
    return pd.DataFrame(
        [
            {
                "game_id": "2023_W01_DAL@NYG",
                "season": 2023,
                "week": 1,
                "bet_side": "under",
                "totals_regime": "not_high",
                "subpop_label": "under",
                "model_total": 41.2,
                "closing_total": 44.0,
                "calibrated_p_side": 0.561,
                "per_bet_ev": 0.071,
                "slipped_line": 43.5,
                "kelly_stake": 88.5,
                "outcome": True,
                "clv": -2.8,
            },
            {
                "game_id": "2023_W02_KC@JAX",
                "season": 2023,
                "week": 2,
                "bet_side": "over",
                "totals_regime": "high",
                "subpop_label": "high_total",
                "model_total": 53.0,
                "closing_total": 49.5,
                "calibrated_p_side": 0.547,
                "per_bet_ev": 0.045,
                "slipped_line": 50.0,
                "kelly_stake": 62.0,
                "outcome": False,
                "clv": 3.5,
            },
            {
                "game_id": "2023_W03_BUF@WAS",
                "season": 2023,
                "week": 3,
                "bet_side": "under",
                "totals_regime": "high",
                "subpop_label": "under+high_total",
                "model_total": 45.0,
                "closing_total": 48.5,
                "calibrated_p_side": 0.572,
                "per_bet_ev": 0.082,
                "slipped_line": 48.0,
                "kelly_stake": 95.0,
                "outcome": None,  # push -> SQL NULL
                "clv": 3.5,
            },
        ]
    )


def _conn() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection (tmp DB; no file write)."""
    return duckdb.connect(":memory:")


class TestMaterializeOuBetList:
    """Smoke / column-order / edge-case guards for the thin cache-population fn."""

    def test_cache_row_count(self) -> None:
        """The round-tripped row count equals the input (no silent drop, no PK collision)."""
        conn = _conn()
        df = _selector_frame()
        n = materialize_ou_bet_list(conn, df)
        assert n == len(df)
        stored = conn.execute("SELECT COUNT(*) FROM ou_bet_list").fetchone()[0]
        assert stored == len(df)

    def test_cache_push_null(self) -> None:
        """Pushes (outcome None) are stored as SQL NULL (the push-NULL convention)."""
        conn = _conn()
        materialize_ou_bet_list(conn, _selector_frame())
        null_count = conn.execute(
            "SELECT COUNT(*) FROM ou_bet_list WHERE outcome IS NULL"
        ).fetchone()[0]
        assert null_count == 1
        # The two graded rows store real booleans (not coerced).
        true_count = conn.execute(
            "SELECT COUNT(*) FROM ou_bet_list WHERE outcome = TRUE"
        ).fetchone()[0]
        false_count = conn.execute(
            "SELECT COUNT(*) FROM ou_bet_list WHERE outcome = FALSE"
        ).fetchone()[0]
        assert true_count == 1
        assert false_count == 1

    def test_cache_no_math(self) -> None:
        """The fn does ZERO metric math (UIAP-01): a precomputed value round-trips byte-equal."""
        conn = _conn()
        df = _selector_frame()
        materialize_ou_bet_list(conn, df)
        row = conn.execute(
            "SELECT calibrated_p_side, per_bet_ev, kelly_stake, clv "
            "FROM ou_bet_list WHERE game_id = '2023_W01_DAL@NYG'"
        ).fetchone()
        assert row[0] == pytest.approx(0.561)
        assert row[1] == pytest.approx(0.071)
        assert row[2] == pytest.approx(88.5)
        assert row[3] == pytest.approx(-2.8)

    def test_cache_ou_columns(self) -> None:
        """The NEW O/U columns are all present + populated (calibrated P(side), EV, sub-pop, etc.)."""
        conn = _conn()
        materialize_ou_bet_list(conn, _selector_frame())
        cols = {
            row[0]
            for row in conn.execute("PRAGMA table_info('ou_bet_list')").fetchall()
        }
        for required in (
            "calibrated_p_side",
            "per_bet_ev",
            "subpop_label",
            "totals_regime",
            "kelly_stake",
            "validation_type",
        ):
            assert required in cols, f"ou_bet_list missing column '{required}'"
        # The sub-pop labels round-trip exactly (precomputed, not recomputed).
        labels = [
            r[0]
            for r in conn.execute(
                "SELECT subpop_label FROM ou_bet_list ORDER BY week"
            ).fetchall()
        ]
        assert labels == ["under", "high_total", "under+high_total"]

    def test_cache_validation_type(self) -> None:
        """Every row carries validation_type == PROVISIONAL_CONTAMINATED (#6, structural honesty)."""
        conn = _conn()
        df = _selector_frame()
        materialize_ou_bet_list(conn, df)
        types = [
            r[0]
            for r in conn.execute("SELECT validation_type FROM ou_bet_list").fetchall()
        ]
        assert types == [_VALIDATION_TYPE] * len(df)

    def test_cache_explicit_column_insert(self) -> None:
        """The INSERT is EXPLICIT-COLUMN, not positional: shuffled input lands in the right columns.

        Shuffle the input DataFrame's column order; a positional ``INSERT ... SELECT *`` would
        mis-align the values, but an explicit-column INSERT lands every value in the correct column
        (#9, T-27-24).
        """
        conn = _conn()
        df = _selector_frame()
        shuffled = df[list(reversed(df.columns))].copy()
        materialize_ou_bet_list(conn, shuffled)
        # The under-win row's calibrated_p_side must still be 0.561 (not a mis-aligned value).
        row = conn.execute(
            "SELECT calibrated_p_side, bet_side, totals_regime "
            "FROM ou_bet_list WHERE game_id = '2023_W01_DAL@NYG'"
        ).fetchone()
        assert row[0] == pytest.approx(0.561)
        assert row[1] == "under"
        assert row[2] == "not_high"

    def test_cache_empty_frame(self) -> None:
        """An empty BetSelector frame writes zero rows without error (#8)."""
        conn = _conn()
        empty = pd.DataFrame(columns=OU_BET_LIST_COLUMNS)
        n = materialize_ou_bet_list(conn, empty)
        assert n == 0
        stored = conn.execute("SELECT COUNT(*) FROM ou_bet_list").fetchone()[0]
        assert stored == 0

    def test_cache_missing_column(self) -> None:
        """A frame missing a required column raises a NAMED error (#8), never a silent mis-write."""
        conn = _conn()
        df = _selector_frame().drop(columns=["closing_total"])
        with pytest.raises((KeyError, ValueError), match="closing_total"):
            materialize_ou_bet_list(conn, df)


def test_betting_bets_unaffected() -> None:
    """The sibling ou_bet_list table does NOT disturb the existing betting_bets schema (additive)."""
    from api.cache import CACHE_SCHEMA

    conn = _conn()
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    # betting_bets still exists with its 15 columns (untouched by the additive sibling table).
    cols = {
        row[0] for row in conn.execute("PRAGMA table_info('betting_bets')").fetchall()
    }
    assert "kelly_stake" in cols
    assert "validation_type" not in cols  # the new column lives ONLY on ou_bet_list
    # Materializing into the sibling table leaves betting_bets empty.
    materialize_ou_bet_list(conn, _selector_frame())
    bb = conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()[0]
    assert bb == 0


# Re-export for the np import guard (the frame uses np.nan-free explicit None for pushes).
_ = np
