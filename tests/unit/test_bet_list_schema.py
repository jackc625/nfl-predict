"""Contract suite for the generic ``bet_list`` schema and its writer (Phase 31, 31-01; SPEC R4/R9).

REPLACES ``tests/unit/test_ou_monetization_cache.py``, retired in the same commit that removed
``materialize_ou_bet_list`` / ``OU_BET_LIST_COLUMNS`` / ``OU_BET_LIST_SCHEMA`` and the 0-row,
0-reader ``ou_bet_list`` table (D31-20). Every guard that module carried is re-expressed here
against the generic table, INCLUDING ``test_betting_bets_unaffected`` -- the additive-sibling claim
it proved is still a claim this phase makes.

What is pinned, one contract fact per test:

  (a) the INSERT is EXPLICIT-COLUMN, not positional -- a SHUFFLED input frame still lands every
      value in its named column (SPEC R4 ordering, T-31-02);
  (b) a missing required column raises a NAMED KeyError listing the absent fields;
  (c) an empty frame writes zero rows and returns 0 without raising, at BOTH call shapes
      (SPEC R4 empty and the R9 empty-prediction-set edge are the same assertion twice);
  (d) there is NO PRIMARY KEY -- two rows sharing (game_id, season, week, target) both persist, so
      a re-bet is never silently dropped;
  (e) ``outcome`` None / NaN round-trips as SQL NULL and never as False (the betting_bets pitfall);
  (f) PRECISION (the SPEC R9 backstop): low-bit IEEE-754 doubles round-trip through DuckDB DOUBLE
      and compare EXACTLY equal -- no tolerance, no pytest.approx;
  (g) ``classify_row_provenance`` is pinned for every case, including the refusals;
  (h) the IMMUTABLE / GRADING split is disjoint, ordered and concatenates to BET_LIST_COLUMNS, and
      a ``pending`` row carries NULL grading facts while every immutable fact is populated;
  (i) grading transitions: pending -> win / loss / push permitted, a fifth status raises, and an
      already-graded row returning to ``pending`` raises;
  (j) a PUSH and a PENDING row both carry ``outcome`` NULL and are still distinguishable by
      ``grading_status`` -- the exact ambiguity the retired writer had;
  (k) a flat return under ASYMMETRIC American prices is computable FROM THE ROW rather than
      inferred from ``outcome`` alone;
  (l) ``betting_bets`` is unaffected, and the retired ou_bet_list surface is genuinely gone.

Plain unit module: it touches nothing under ``data/``, ``artifacts/`` or ``outputs/`` -- every
database here is in-memory.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import math
import pathlib
import subprocess
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

import api.cache as cache_module
from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    BET_LIST_SCHEMA,
    CACHE_SCHEMA,
    GRADING_STATUSES,
    assert_grading_transition,
    classify_row_provenance,
    materialize_bet_list,
)

# The retired Phase-27 surface. Asserted absent by ATTRIBUTE LOOKUP, never by text search, so a
# code comment mentioning the retirement cannot trip the check.
_RETIRED_SYMBOLS = (
    "materialize_ou_bet_list",
    "OU_BET_LIST_COLUMNS",
    "OU_BET_LIST_SCHEMA",
)

# Path prefixes this module never reads: the data lake, the artifact store and the outputs dir.
_NON_SOURCE_PREFIXES = (
    "data/bronze/",
    "data/silver/",
    "data/gold/",
    "artifacts/",
    "outputs/",
)


def _conn() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection (no file write, nothing under data/ touched)."""
    return duckdb.connect(":memory:")


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return a table's column names via information_schema (by NAME, never by ordinal)."""
    return {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
    }


def _row(**overrides: Any) -> dict[str, Any]:
    """A fully-populated bet_list row, overridable field by field.

    Every value is PRECOMPUTED -- the writer performs zero metric math (UIAP-01), so these are the
    numbers a caller hands it, not numbers it derives.
    """
    row: dict[str, Any] = {
        "game_id": "2023_W01_DAL@NYG",
        "season": 2023,
        "week": 1,
        "target": "ou",
        "bet_side": "under",
        "model_value": 41.2,
        "market_value": 44.0,
        "line": 44.0,
        "slipped_line": 43.5,
        "calibrated_p_side": 0.561,
        "per_bet_ev": 0.071,
        "stake_units": 0.885,
        "ev_tier": "high",
        "status": "live",
        "rejection_reason": None,
        "eligibility_label": "under",
        "snapshot_ts": "2023-09-08T18:00:00-04:00",
        "freeze_ts": "2023-09-08T18:00:00-04:00",
        "selected_odds": -110.0,
        "flat_stake": 1.0,
        "provenance": "backtest_replay",
        "validation_type": "contaminated",
        "grading_status": "win",
        "outcome": True,
        "clv": -2.8,
        "payout_flat": 0.9090909090909091,
        "realized_units": 0.9090909090909091,
        "graded_at": pd.Timestamp("2023-09-11 03:30:00"),
    }
    row.update(overrides)
    assert set(row) == set(BET_LIST_COLUMNS), (
        "fixture row drifted from BET_LIST_COLUMNS"
    )
    return row


def _american_to_profit(odds: float, stake: float) -> float:
    """Profit on a WINNING flat bet at American *odds* -- the arithmetic the row must support."""
    return stake * (odds / 100.0) if odds > 0 else stake * (100.0 / abs(odds))


# ---------------------------------------------------------------------------
# (a) explicit-column INSERT / (b) missing column / (c) empty frame
# ---------------------------------------------------------------------------


def test_insert_is_explicit_column_not_positional() -> None:
    """A SHUFFLED input frame still lands every value in its NAMED column (SPEC R4 ordering).

    A positional ``INSERT ... SELECT *`` would mis-align every field here. This is the assertion
    that makes T-31-02 (a mis-ordered write silently mislabelling a bet) structurally impossible.
    """
    conn = _conn()
    df = pd.DataFrame([_row()])
    shuffled = df[list(reversed(df.columns))].copy()
    assert list(shuffled.columns) != BET_LIST_COLUMNS

    assert materialize_bet_list(conn, shuffled) == 1
    stored = conn.execute(
        "SELECT game_id, target, bet_side, calibrated_p_side, ev_tier, provenance, "
        "validation_type, grading_status FROM bet_list"
    ).fetchone()
    assert stored == (
        "2023_W01_DAL@NYG",
        "ou",
        "under",
        0.561,
        "high",
        "backtest_replay",
        "contaminated",
        "win",
    )


def test_missing_column_raises_named_key_error() -> None:
    """A missing required column raises a KeyError NAMING the absent fields, never a mis-write."""
    conn = _conn()
    df = pd.DataFrame([_row()]).drop(columns=["per_bet_ev", "stake_units"])
    with pytest.raises(KeyError) as excinfo:
        materialize_bet_list(conn, df)
    message = str(excinfo.value)
    assert "per_bet_ev" in message
    assert "stake_units" in message


def test_empty_frame_writes_zero_rows_at_both_call_shapes() -> None:
    """An empty frame returns 0 and raises nothing -- with columns, and with none at all.

    The SPEC R4 empty edge (a week where no candidate cleared the floor) and the R9
    empty-prediction-set edge (cache population over an empty prediction set) are the same
    assertion at two call sites, so both shapes are exercised.
    """
    conn = _conn()
    assert materialize_bet_list(conn, pd.DataFrame(columns=BET_LIST_COLUMNS)) == 0
    assert materialize_bet_list(conn, pd.DataFrame()) == 0
    assert conn.execute("SELECT COUNT(*) FROM bet_list").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# (d) no PRIMARY KEY / (e) push-NULL
# ---------------------------------------------------------------------------


def test_no_primary_key_so_a_re_bet_is_never_dropped() -> None:
    """Two rows sharing (game_id, season, week, target) BOTH persist (SPEC R4 adjacency)."""
    assert "PRIMARY KEY" not in BET_LIST_SCHEMA
    conn = _conn()
    rows = [
        _row(bet_side="under", per_bet_ev=0.071),
        _row(bet_side="under", per_bet_ev=0.052),
    ]
    assert materialize_bet_list(conn, pd.DataFrame(rows)) == 2
    stored = conn.execute(
        "SELECT COUNT(*) FROM bet_list WHERE game_id = ? AND season = ? AND week = ? "
        "AND target = ?",
        ["2023_W01_DAL@NYG", 2023, 1, "ou"],
    ).fetchone()[0]
    assert stored == 2


def test_outcome_none_and_nan_round_trip_as_sql_null() -> None:
    """``outcome`` None or NaN stores as SQL NULL and NEVER as False (the betting_bets pitfall)."""
    conn = _conn()
    rows = [
        _row(game_id="2023_W01_A@B", grading_status="push", outcome=None),
        _row(game_id="2023_W01_C@D", grading_status="pending", outcome=np.nan),
        _row(game_id="2023_W01_E@F", grading_status="loss", outcome=False),
    ]
    materialize_bet_list(conn, pd.DataFrame(rows))

    nulls = conn.execute(
        "SELECT COUNT(*) FROM bet_list WHERE outcome IS NULL"
    ).fetchone()[0]
    falses = conn.execute(
        "SELECT COUNT(*) FROM bet_list WHERE outcome = FALSE"
    ).fetchone()[0]
    assert nulls == 2, "a None / NaN outcome was coerced instead of stored as SQL NULL"
    assert falses == 1


# ---------------------------------------------------------------------------
# (f) PRECISION -- the SPEC R9 float round-trip backstop
# ---------------------------------------------------------------------------


def test_double_round_trip_is_exact_not_approximate() -> None:
    """Low-bit IEEE-754 doubles survive DuckDB DOUBLE EXACTLY -- asserted with ``==``.

    Deliberately NOT ``pytest.approx`` and not a tolerance: a tolerance would pass even if the
    served number differed from the selector's in the last bits, which is precisely the drift this
    backstop exists to exclude.
    """
    ev = (
        0.1 + 0.2
    )  # 0.30000000000000004 -- a value whose last bits are not what a human writes
    stake = 1.2345678901234567
    assert repr(ev) == "0.30000000000000004"

    conn = _conn()
    materialize_bet_list(conn, pd.DataFrame([_row(per_bet_ev=ev, stake_units=stake)]))
    stored_ev, stored_stake = conn.execute(
        "SELECT per_bet_ev, stake_units FROM bet_list"
    ).fetchone()

    assert stored_ev == ev
    assert stored_stake == stake
    assert math.isfinite(stored_ev) and math.isfinite(stored_stake)


# ---------------------------------------------------------------------------
# (g) provenance classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", [2021, 2022, 2023, 2024])
def test_replay_seasons_are_contaminated(season: int) -> None:
    """Every burned replay season is labelled contaminated, never clean."""
    assert classify_row_provenance(season, "replay") == (
        "backtest_replay",
        "contaminated",
    )


def test_replay_2025_is_the_clean_holdout_and_forward_is_realized() -> None:
    """2025 in replay is the single clean holdout; forward mode is a live realized record."""
    assert classify_row_provenance(2025, "replay") == (
        "backtest_replay",
        "clean_holdout",
    )
    assert classify_row_provenance(2026, "forward") == ("forward", "forward_realized")
    assert classify_row_provenance(2019, "forward") == ("forward", "forward_realized")


def test_provenance_refuses_rather_than_defaults() -> None:
    """An out-of-vocabulary run_mode or replay season RAISES -- a defaulted label is the mislabel."""
    with pytest.raises(ValueError, match="run_mode"):
        classify_row_provenance(2023, "backfill")
    with pytest.raises(ValueError, match="validation_type"):
        classify_row_provenance(2019, "replay")


# ---------------------------------------------------------------------------
# (h) the IMMUTABLE / GRADING split
# ---------------------------------------------------------------------------


def test_immutable_and_grading_halves_are_disjoint_and_ordered() -> None:
    """The two halves are disjoint and CONCATENATE to BET_LIST_COLUMNS in order (REVIEW-FWD-GRADE).

    Asserted by set and sequence operations, not by eye.
    """
    assert set(BET_LIST_IMMUTABLE_COLUMNS).isdisjoint(set(BET_LIST_GRADING_COLUMNS))
    assert BET_LIST_IMMUTABLE_COLUMNS + BET_LIST_GRADING_COLUMNS == BET_LIST_COLUMNS
    assert len(BET_LIST_IMMUTABLE_COLUMNS) == 22
    assert len(BET_LIST_GRADING_COLUMNS) == 6
    assert len(BET_LIST_COLUMNS) == 28


def test_pending_row_carries_null_grading_facts_and_populated_recommendation_facts() -> (
    None
):
    """A forward row written at freeze: every immutable fact set, every gradable fact still NULL.

    This is what makes a forward recommendation gradable LATER without any recommendation fact
    changing. If the whole row were immutable the self-grading loop would be silently disabled.
    """
    conn = _conn()
    materialize_bet_list(
        conn,
        pd.DataFrame(
            [
                _row(
                    grading_status="pending",
                    outcome=None,
                    clv=None,
                    payout_flat=None,
                    realized_units=None,
                    graded_at=None,
                )
            ]
        ),
    )
    stored = conn.execute(
        f"SELECT {', '.join(BET_LIST_COLUMNS)} FROM bet_list"
    ).fetchone()
    record = dict(zip(BET_LIST_COLUMNS, stored))

    for column in BET_LIST_IMMUTABLE_COLUMNS:
        if column == "rejection_reason":
            continue  # a LIVE row has no rejection reason by construction
        assert record[column] is not None, (
            f"immutable column {column} was not populated"
        )

    assert record["grading_status"] == "pending"
    for column in ("outcome", "payout_flat", "realized_units", "graded_at"):
        assert record[column] is None, f"grading column {column} should still be NULL"


# ---------------------------------------------------------------------------
# (i) grading transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", ["win", "loss", "push"])
def test_pending_may_transition_to_each_terminal_status(terminal: str) -> None:
    """``pending`` -> win / loss / push is the permitted forward transition."""
    assert_grading_transition("pending", terminal)


def test_a_fifth_grading_status_raises_rather_than_being_stored() -> None:
    """The vocabulary is CLOSED: a fifth value raises at the writer and at the transition guard."""
    assert set(GRADING_STATUSES) == {"pending", "win", "loss", "push"}

    with pytest.raises(ValueError, match="voided"):
        assert_grading_transition("pending", "voided")

    conn = _conn()
    with pytest.raises(ValueError, match="voided"):
        materialize_bet_list(conn, pd.DataFrame([_row(grading_status="voided")]))
    assert conn.execute("SELECT COUNT(*) FROM bet_list").fetchone()[0] == 0


def test_a_graded_row_may_not_return_to_pending() -> None:
    """Ungrading a settled bet is REFUSED -- a losing record cannot be quietly reopened."""
    for settled in ("win", "loss", "push"):
        with pytest.raises(ValueError, match="pending"):
            assert_grading_transition(settled, "pending")
    with pytest.raises(ValueError):
        assert_grading_transition("loss", "win")
    # Re-asserting the SAME terminal status is idempotent, so a grader re-run is safe.
    assert_grading_transition("loss", "loss")


# ---------------------------------------------------------------------------
# (j) push versus ungraded / (k) asymmetric-price return
# ---------------------------------------------------------------------------


def test_push_and_pending_share_a_null_outcome_but_differ_on_status() -> None:
    """The exact ambiguity ``materialize_ou_bet_list`` had, now resolved.

    Both rows store ``outcome`` SQL NULL. Under the retired writer that made a push and an
    ungraded forward game indistinguishable, which the tracker must be able to tell apart.
    """
    conn = _conn()
    materialize_bet_list(
        conn,
        pd.DataFrame(
            [
                _row(game_id="2023_W01_PUSH@X", grading_status="push", outcome=None),
                _row(game_id="2023_W01_PEND@Y", grading_status="pending", outcome=None),
            ]
        ),
    )
    rows = conn.execute(
        "SELECT grading_status FROM bet_list WHERE outcome IS NULL ORDER BY grading_status"
    ).fetchall()
    assert [r[0] for r in rows] == ["pending", "push"]
    assert len({r[0] for r in rows}) == 2, "two NULL outcomes collapsed to one status"


def test_flat_return_is_computable_from_the_stored_price_not_from_outcome() -> None:
    """Two WINS at the same stake but DIFFERENT American prices yield different realized units.

    ``outcome`` alone cannot express this: a win at -110 and a win at +150 return different
    amounts, so the price has to live on the row. That is why ``selected_odds`` / ``flat_stake`` /
    ``payout_flat`` are stored, exactly as the shipped ``betting_bets`` ledger already does.
    """
    stake = 1.0
    favourite_profit = _american_to_profit(-110.0, stake)
    underdog_profit = _american_to_profit(150.0, stake)
    assert favourite_profit != underdog_profit

    conn = _conn()
    materialize_bet_list(
        conn,
        pd.DataFrame(
            [
                _row(
                    game_id="2023_W01_FAV@A",
                    selected_odds=-110.0,
                    flat_stake=stake,
                    grading_status="win",
                    outcome=True,
                    payout_flat=favourite_profit,
                    realized_units=favourite_profit,
                ),
                _row(
                    game_id="2023_W01_DOG@B",
                    selected_odds=150.0,
                    flat_stake=stake,
                    grading_status="win",
                    outcome=True,
                    payout_flat=underdog_profit,
                    realized_units=underdog_profit,
                ),
            ]
        ),
    )
    stored = conn.execute(
        "SELECT selected_odds, flat_stake, grading_status, realized_units "
        "FROM bet_list ORDER BY selected_odds"
    ).fetchall()

    assert [r[2] for r in stored] == ["win", "win"]
    assert stored[0][3] != stored[1][3], (
        "two wins at different prices returned the same units"
    )
    for odds, flat_stake, _status, realized in stored:
        assert realized == _american_to_profit(odds, flat_stake)


# ---------------------------------------------------------------------------
# (l) betting_bets untouched, and the retired surface is genuinely gone
# ---------------------------------------------------------------------------


def test_betting_bets_unaffected() -> None:
    """CARRIED OVER from the retired module: bet_list does not disturb the betting_bets ledger."""
    conn = _conn()
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)

    cols = _table_columns(conn, "betting_bets")
    assert "kelly_stake" in cols
    assert "validation_type" not in cols  # that column lives ONLY on bet_list
    assert "grading_status" not in cols

    materialize_bet_list(conn, pd.DataFrame([_row()]))
    assert conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM bet_list").fetchone()[0] == 1


def test_retired_symbols_are_gone_from_api_cache() -> None:
    """The three retired symbols are absent, asserted by ATTRIBUTE LOOKUP not by text search."""
    for symbol in _RETIRED_SYMBOLS:
        assert not hasattr(cache_module, symbol), f"api.cache still exports {symbol}"


def test_no_tracked_module_imports_the_retired_symbols() -> None:
    """No tracked ``.py`` file still IMPORTS a retired symbol, asserted by AST not by grep.

    A raw text search would match prose ABOUT the retirement (including this module's own
    docstring), so the check walks ``ImportFrom`` aliases instead.
    """
    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        pytest.skip("git is unavailable; cannot enumerate tracked files")

    # Source only. The data LAKE (data/bronze, data/silver, data/gold), artifacts/ and outputs/
    # are excluded explicitly so this stays a plain unit module that reads no data artifact --
    # a no-op for tracked files today, and the exclusion is stated rather than assumed.
    tracked = [
        path
        for line in listing.stdout.splitlines()
        if line.strip()
        for path in [pathlib.Path(line)]
        if not str(path).replace("\\", "/").startswith(_NON_SOURCE_PREFIXES)
    ]
    assert tracked, (
        "git ls-files returned no Python files -- the guard would pass vacuously"
    )

    offenders: list[str] = []
    for path in tracked:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (
            SyntaxError
        ):  # pragma: no cover -- a non-importable file is not our concern here
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _RETIRED_SYMBOLS:
                        offenders.append(f"{path}:{node.lineno} imports {alias.name}")

    assert not offenders, "retired ou_bet_list symbols still imported:\n" + "\n".join(
        offenders
    )


def test_ou_bet_list_table_is_gone_from_the_schema() -> None:
    """A fresh cache built from CACHE_SCHEMA carries NO ``ou_bet_list`` table."""
    conn = _conn()
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert "ou_bet_list" not in tables
    assert {
        "bet_list",
        "available_bet_weeks",
        "bet_week_freeze",
        "bet_tracker_blocks",
    } <= tables
