"""``decided_at_utc``: the row's own observation time, and the fence that keeps it honest.

Phase 33, Plan 33-05 Task 3 (COLD-03, R7, D33-06/D33-27, T-33-21/22/24). The OWNER RULING of
2026-09-12 selected option ``decided-at-utc-only``: ONE new column, typed ``VARCHAR``, holding an
ISO-8601 string with an explicit UTC offset, placed in the IMMUTABLE half at index 22 --
immediately after ``validation_type`` and immediately before ``grading_status``, the last position
of the immutable half. ``BET_LIST_COLUMNS`` becomes 23 immutable + 6 grading = 29 entries in one
locked order.

WHAT THIS MODULE PINS, one contract fact per test:

  (1) the locked order is 29 long, 23 + 6, and ``decided_at_utc`` sits at index 22 -- inside the
      immutable half, because a row's claim about when it was decided is a RECOMMENDATION fact and
      must never transition;
  (2) all THREE DDL sites build the same 29 columns in the same order -- the two
      ``conn.execute(BET_LIST_SCHEMA)`` calls and the copy embedded in ``CACHE_SCHEMA``;
  (3) a FORWARD row decided AFTER its own freeze is REFUSED by name, and the message carries the
      game and both instants;
  (4) a FORWARD row decided EXACTLY ON its freeze is ACCEPTED -- the assertion is ``<=``. This is a
      BOUNDARY-ONLY case on a constructed row: the selection fence refuses ``now >= freeze``, so in
      production every emitted row is STRICTLY before its own freeze and this equality is
      unreachable. It is proven anyway, because a peer reviewer read the two fences as
      contradictory and the record should settle it rather than argue it;
  (5) a FORWARD row with a NULL ``decided_at_utc`` is refused by a DISTINCT named error -- a
      forward row that cannot be checked is exactly the row the check exists for;
  (6) a ``backtest_replay`` row with a NULL ``decided_at_utc`` is ACCEPTED and is never backfilled;
  (7) a week emitting ZERO rows performs the write, raises nothing, and holds the invariant
      vacuously;
  (8) NO code path stamps ``decided_at_utc`` onto a row it did not create -- asserted by AST over
      the live tree AND by a runtime guard inside the upsert, with the four controls the plan
      names (non-vacuity, the assertion, a planted violation, and a no-false-positive case).

Plain unit module: it touches nothing under ``data/``, ``artifacts/`` or ``outputs/``. Every
database is in-memory and every frame is constructed here.

Run this module:  uv run pytest tests/unit/test_decided_at_utc.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

import api.cache as cache_module
import backtest.weekly_bet_list as wbl
from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    BET_LIST_SCHEMA,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
)
from backtest.weekly_bet_list import (
    DECIDED_AT_COLUMN,
    DecidedAfterFreezeError,
    MissingDecidedAtError,
    assert_decided_at_before_freeze,
    upsert_bet_list_rows,
)
from tests.phase33_state import (
    BET_LIST_COLUMN_COUNT_AFTER,
    BET_LIST_COLUMN_COUNT_BEFORE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two production modules the retroactive-stamp prohibition is scanned over.
_SCANNED_SOURCES = (
    REPO_ROOT / "backtest" / "weekly_bet_list.py",
    REPO_ROOT / "api" / "cache.py",
)

# The ONLY functions permitted to put a non-null observation time on the column. A frame built
# inside the emission path is constructed THIS RUN from THIS RUN's records, so stamping it is the
# legitimate act; a frame loaded from the store is a record of something already decided.
_EMISSION_FUNCTIONS = frozenset({"records_to_bet_list_frame", "_row_from_record"})

# The instants every fixture row below is anchored on. A Sunday kickoff, its own preceding
# Friday 6 PM Eastern freeze, and a decision made one minute before that freeze.
_FREEZE = "2026-09-18T18:00:00-04:00"
_DECIDED_BEFORE = "2026-09-18T17:59:00-04:00"
_DECIDED_AFTER = "2026-09-18T18:00:01-04:00"


def _forward_row(**overrides: Any) -> dict[str, Any]:
    """A fully-populated FORWARD bet_list row at the locked width, overridable field by field."""
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": "2026_03_DAL_NYG",
            "season": 2026,
            "week": 3,
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
            "snapshot_ts": _FREEZE,
            "freeze_ts": _FREEZE,
            "selected_odds": -110.0,
            "flat_stake": 1.0,
            "provenance": PROVENANCE_FORWARD,
            "validation_type": "forward_realized",
            DECIDED_AT_COLUMN: _DECIDED_BEFORE,
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
            "clv": None,
            "payout_flat": None,
            "realized_units": None,
            "graded_at": None,
        }
    )
    row.update(overrides)
    assert set(row) == set(BET_LIST_COLUMNS), (
        "fixture row drifted from BET_LIST_COLUMNS"
    )
    return row


def _replay_row(**overrides: Any) -> dict[str, Any]:
    """A ``backtest_replay`` row: derived, fully regenerable, and carrying NO observation time."""
    base = _forward_row(
        game_id="2023_01_DAL_NYG",
        season=2023,
        week=1,
        provenance=PROVENANCE_BACKTEST_REPLAY,
        validation_type="contaminated",
    )
    base[DECIDED_AT_COLUMN] = None
    base.update(overrides)
    return base


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    """Rows as a frame in the locked column order (empty frames keep the columns)."""
    return pd.DataFrame(list(rows), columns=pd.Index(BET_LIST_COLUMNS))


def _ddl_columns(statement: str) -> list[str]:
    """Build ``bet_list`` from one DDL *statement* and read its column names back, in order.

    ``PRAGMA table_info`` returns ``(cid, name, type, ...)``, so the NAME is at index 1 and index
    0 is the ordinal. Reading index 0 yields ``[0, 1, 2, ...]``, which compares unequal to any
    column list and would make an order assertion fail for a reason that has nothing to do with
    the schema. Named here because the plan's own verification command carries that mistake.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(statement)
    return [row[1] for row in conn.execute("PRAGMA table_info('bet_list')").fetchall()]


def _cache_schema_bet_list_statement() -> str:
    """The ``bet_list`` CREATE embedded in ``CACHE_SCHEMA`` -- THE THIRD SITE, isolated."""
    statements = [s.strip() for s in CACHE_SCHEMA.strip().split(";") if s.strip()]
    # Matched with the opening parenthesis: Phase 34 (Plan 34-16) added `bet_list_corrections`,
    # whose CREATE also contains the bare prefix. Was: `"CREATE TABLE IF NOT EXISTS bet_list" in s`.
    matches = [s for s in statements if "CREATE TABLE IF NOT EXISTS bet_list (" in s]
    assert len(matches) == 1, (
        f"expected exactly one embedded bet_list CREATE in CACHE_SCHEMA, found {len(matches)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# (1) and (2): the locked order, and the three DDL sites that must agree on it
# ---------------------------------------------------------------------------


def test_the_locked_order_is_twenty_nine_with_decided_at_last_in_the_immutable_half() -> (
    None
):
    """OWNER RULING 2026-09-12: 23 immutable + 6 grading = 29, ``decided_at_utc`` at index 22.

    The index is asserted, not merely the membership. The ruling placed the column immediately
    after ``validation_type`` and immediately before ``grading_status`` -- the LAST position of
    the immutable half -- so that Phase 34's own bump (D33-06) appends against a known base
    rather than against a column set whose order was never written down.
    """
    # The Phase-33 witness values are history and stay pinned as recorded.
    assert BET_LIST_COLUMN_COUNT_BEFORE == 28
    assert BET_LIST_COLUMN_COUNT_AFTER == 29

    # Phase 34 (Plan 34-01 Task 2) appended 11 stamps after ``decided_at_utc`` and the FILL and
    # CLOSING classes after the grading half (D33-06: it appended against this written base).
    # Was: 23 immutable, ``len(BET_LIST_COLUMNS) == BET_LIST_COLUMN_COUNT_AFTER`` (29), and
    # ``IMMUTABLE + GRADING == BET_LIST_COLUMNS``. The Phase-33 prefix is what is still asserted.
    assert len(BET_LIST_IMMUTABLE_COLUMNS) == 34
    assert len(BET_LIST_GRADING_COLUMNS) == 6
    assert len(BET_LIST_COLUMNS) == 51

    assert BET_LIST_COLUMNS.index(DECIDED_AT_COLUMN) == 22
    assert BET_LIST_IMMUTABLE_COLUMNS[21] == "validation_type"
    assert BET_LIST_IMMUTABLE_COLUMNS[22] == DECIDED_AT_COLUMN
    assert BET_LIST_GRADING_COLUMNS[0] == "grading_status"
    assert DECIDED_AT_COLUMN not in BET_LIST_GRADING_COLUMNS
    assert (
        BET_LIST_COLUMNS[:40] == BET_LIST_IMMUTABLE_COLUMNS + BET_LIST_GRADING_COLUMNS
    )


def test_all_three_ddl_sites_build_the_same_twenty_nine_columns_in_the_same_order() -> (
    None
):
    """T-33-24: the standalone CREATE and the ``CACHE_SCHEMA`` copy cannot drift apart.

    Both are BUILT and read back through ``PRAGMA table_info`` rather than compared as text, so a
    site that formats its DDL differently but produces the same table still passes and a site that
    produces a 28-column table fails. The two ``conn.execute(BET_LIST_SCHEMA)`` call sites share
    one constant, so the constant is the thing under test for both of them.
    """
    standalone = _ddl_columns(BET_LIST_SCHEMA)
    embedded = _ddl_columns(_cache_schema_bet_list_statement())

    assert standalone == list(BET_LIST_COLUMNS)
    assert embedded == list(BET_LIST_COLUMNS)
    assert standalone == embedded
    # Was: ``== BET_LIST_COLUMN_COUNT_AFTER`` (29). Phase 34 (Plan 34-01 Task 2) widened all three
    # sites to 51 and its own suite pins that width; here the Phase-33 column keeps its index.
    assert len(standalone) == 51
    assert standalone.index(DECIDED_AT_COLUMN) == 22


def test_decided_at_is_a_varchar_not_a_timestamp() -> None:
    """The OWNER RULING typed it VARCHAR, matching its ``snapshot_ts`` / ``freeze_ts`` siblings.

    NOT ``graded_at``'s TIMESTAMP. The three instants this column is COMPARED AGAINST are stored
    as offset-carrying ISO strings, and one representation is what lets the single strict parse
    helper serve every comparison. A second representation would need a second code path, which
    is the shape D33-27 refused.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(BET_LIST_SCHEMA)
    types = {
        row[1]: row[2]
        for row in conn.execute("PRAGMA table_info('bet_list')").fetchall()
    }
    assert types[DECIDED_AT_COLUMN] == "VARCHAR"
    assert types["snapshot_ts"] == "VARCHAR"
    assert types["freeze_ts"] == "VARCHAR"
    assert types["graded_at"] == "TIMESTAMP"


# ---------------------------------------------------------------------------
# (3), (4), (5), (6): the write-time assertion, one behaviour per test
# ---------------------------------------------------------------------------


def test_a_forward_row_decided_after_its_freeze_is_refused_by_name() -> None:
    """T-33-21: a post-hoc pick wearing a pre-game timestamp is the one claim that must not ship.

    The message is asserted on CONTENT, not merely on exception type: it must name the game and
    BOTH instants, or a reader of the failure cannot tell which row was refused or by how much.
    """
    row = _forward_row(**{DECIDED_AT_COLUMN: _DECIDED_AFTER})
    with pytest.raises(DecidedAfterFreezeError) as excinfo:
        assert_decided_at_before_freeze(row)

    # BOTH instants appear, and both appear NORMALIZED TO UTC. The message renders what the one
    # strict parse helper returned, not the Eastern strings it was handed, so a reader comparing
    # two instants in a failure is never comparing across offsets.
    message = str(excinfo.value)
    assert "2026_03_DAL_NYG" in message
    assert "2026-09-18T22:00:01+00:00" in message
    assert "2026-09-18T22:00:00+00:00" in message


def test_a_forward_row_decided_exactly_on_its_freeze_is_accepted() -> None:
    """The assertion is ``<=``, and this equality is a BOUNDARY-ONLY case on a constructed row.

    DELIBERATELY SEPARATE from the strictly-greater test above, with its own message, because the
    two are the two sides of one operator and collapsing them into one parametrized test would
    let a ``<`` silently pass as a ``<=`` if only the greater case were checked.

    Unreachable in production: selection refuses ``now >= freeze``, so a row that exists at all
    was decided STRICTLY before its freeze. Proven here so the record settles the reviewer's
    reading that the two fences contradict each other -- they sit on opposite sides of the same
    boundary and are jointly satisfiable.
    """
    row = _forward_row(**{DECIDED_AT_COLUMN: _FREEZE})
    assert_decided_at_before_freeze(row)  # raises nothing


def test_a_forward_row_with_no_decided_at_is_refused_by_a_distinct_error() -> None:
    """A forward row that cannot be checked is exactly the row the check exists for.

    The error TYPE is distinct from the after-freeze one on purpose: "this row lied about when it
    was decided" and "this row makes no claim at all" are different failures with different fixes,
    and one exception type for both would collapse them at the call site.
    """
    row = _forward_row(**{DECIDED_AT_COLUMN: None})
    with pytest.raises(MissingDecidedAtError) as excinfo:
        assert_decided_at_before_freeze(row)
    assert DECIDED_AT_COLUMN in str(excinfo.value)
    assert not issubclass(MissingDecidedAtError, DecidedAfterFreezeError)
    assert not issubclass(DecidedAfterFreezeError, MissingDecidedAtError)


def test_a_replay_row_with_a_null_decided_at_is_accepted_and_never_backfilled() -> None:
    """The 234 stored ``backtest_replay`` rows take NULL, and NULL is what they keep.

    A replay row is derived and fully regenerable, so it carries no observation time and needs
    none. Filling it from ``snapshot_ts`` would stamp an instant at which nobody observed
    anything -- this plan's own named prohibition.
    """
    row = _replay_row()
    assert_decided_at_before_freeze(row)  # exempt: not a forward row

    stored = _frame(row)
    merged = upsert_bet_list_rows(
        stored,
        _frame(),
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )
    assert len(merged) == 1
    assert merged[DECIDED_AT_COLUMN].isna().all()


def test_a_replay_row_decided_after_its_freeze_is_still_exempt() -> None:
    """The assertion is SCOPED to ``provenance == forward`` and the scope is asserted, not assumed.

    Without this test the exemption would be an implementation detail nobody had pinned, and a
    later change tightening the assertion onto every row would silently make the 234 stored rows
    unwritable.
    """
    assert_decided_at_before_freeze(_replay_row(**{DECIDED_AT_COLUMN: _DECIDED_AFTER}))


# ---------------------------------------------------------------------------
# The assertion is WIRED: the upsert refuses, it does not merely offer a helper
# ---------------------------------------------------------------------------


def test_the_upsert_refuses_an_incoming_forward_row_decided_after_its_freeze() -> None:
    """The check has to be ON the write path, not merely available beside it.

    An assertion function nobody calls is a comment. This drives the refusal through
    ``upsert_bet_list_rows`` -- the one merge every writer goes through -- against a NON-EMPTY
    stored frame and against an EMPTY one, because the empty-stored branch returns early and
    would otherwise bypass the check entirely.
    """
    bad = _frame(_forward_row(**{DECIDED_AT_COLUMN: _DECIDED_AFTER}))
    now = datetime(2026, 9, 18, 17, 0, tzinfo=UTC)

    with pytest.raises(DecidedAfterFreezeError):
        upsert_bet_list_rows(_frame(), bad, now=now)

    with pytest.raises(DecidedAfterFreezeError):
        upsert_bet_list_rows(_frame(_replay_row()), bad, now=now)


def test_the_upsert_refuses_an_incoming_forward_row_with_no_decided_at() -> None:
    """A forward row arriving with no observation time is refused, at both stored shapes."""
    bad = _frame(_forward_row(**{DECIDED_AT_COLUMN: None}))
    now = datetime(2026, 9, 18, 17, 0, tzinfo=UTC)

    with pytest.raises(MissingDecidedAtError):
        upsert_bet_list_rows(_frame(), bad, now=now)
    with pytest.raises(MissingDecidedAtError):
        upsert_bet_list_rows(_frame(_replay_row()), bad, now=now)


def test_a_well_formed_forward_row_passes_the_upsert_and_keeps_its_stamp() -> None:
    """The happy path: the refusal is a tripwire, so the ordinary row must sail through unchanged."""
    good = _frame(_forward_row())
    merged = upsert_bet_list_rows(
        _frame(_replay_row()),
        good,
        now=datetime(2026, 9, 18, 17, 0, tzinfo=UTC),
    )
    assert len(merged) == 2
    stamps = set(merged[DECIDED_AT_COLUMN].dropna())
    assert stamps == {_DECIDED_BEFORE}


# ---------------------------------------------------------------------------
# (7) the zero-row week
# ---------------------------------------------------------------------------


def test_a_zero_row_week_performs_the_write_and_raises_nothing() -> None:
    """A week where no candidate cleared the floor holds the invariant VACUOUSLY.

    Both directions are exercised -- nothing stored and nothing incoming, and something stored
    with nothing incoming -- because a guard written as "check every incoming row" must not
    become "raise when there are none".
    """
    now = datetime(2026, 9, 18, 17, 0, tzinfo=UTC)

    empty = upsert_bet_list_rows(_frame(), _frame(), now=now)
    assert empty.empty
    assert list(empty.columns) == list(BET_LIST_COLUMNS)

    kept = upsert_bet_list_rows(_frame(_forward_row()), _frame(), now=now)
    assert len(kept) == 1

    conn = duckdb.connect(":memory:")
    assert cache_module.materialize_bet_list(conn, _frame()) == 0
    assert conn.execute("SELECT COUNT(*) FROM bet_list").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# The emission stamp: forward rows carry one, replay rows do not
# ---------------------------------------------------------------------------


def _selection_result(target: str = "ou", season: int = 2026) -> Any:
    """A minimal ``SelectionResult`` with one selected and one rejected record."""
    from backtest.bet_selector import SelectionResult

    base = {
        "game_id": f"{season}_03_DAL_NYG",
        "season": season,
        "week": 3,
        "target": target,
        "bet_side": "under",
        "model_total": 41.2,
        "closing_total": 44.0,
        "slipped_line": 43.5,
        "calibrated_p_side": 0.561,
        "per_bet_ev": 0.071,
        "kelly_stake": 88.5,
        "selected_odds": -110.0,
        "snapshot_ts": _FREEZE,
        "freeze_ts": _FREEZE,
        "subpop_label": "under",
        "clv": None,
    }
    rejected = dict(base)
    rejected["game_id"] = f"{season}_03_KC_LAC"
    rejected["rejection_reason"] = "ev_below_floor"
    return SelectionResult(selected=[base], rejected=[rejected])


def _fits(target: str = "ou") -> dict[str, wbl.WeeklyChainFit]:
    return {
        target: wbl.WeeklyChainFit(
            target=target,
            ev_floor_t=0.01,
            frozen_sd=13.0,
            season_bias_by_season={2023: -1.0, 2026: -1.0},
        )
    }


def test_a_forward_emission_stamps_the_run_instant_on_every_row() -> None:
    """Every row a forward run emits -- live AND suppressed -- carries that run's instant.

    A suppressed row is part of the record (D31-21), so it is decided at the same instant the
    live rows were and must say so; a stamp on only the live half would make the completeness
    query's two classes carry different evidentiary weight.
    """
    decided = datetime(2026, 9, 18, 21, 59, tzinfo=UTC)
    frame = wbl.records_to_bet_list_frame(
        _selection_result(),
        _fits(),
        run_mode=cache_module.RUN_MODE_FORWARD,
        decided_at=decided,
    )
    assert len(frame) == 2
    assert list(frame.columns) == list(BET_LIST_COLUMNS)
    assert set(frame[DECIDED_AT_COLUMN]) == {decided.isoformat()}
    assert (frame["provenance"] == PROVENANCE_FORWARD).all()

    for _index, row in frame.iterrows():
        assert_decided_at_before_freeze(row)


def test_a_replay_emission_leaves_the_column_null() -> None:
    """A replay run re-derives history, so it observes nothing and stamps nothing.

    Season 2023 rather than 2026: the replay window is pre-registered as 2021-2024 (contaminated)
    plus 2025 (clean holdout), and ``classify_row_provenance`` REFUSES a season outside it rather
    than defaulting -- so a 2026 replay is not a thing that can exist, and asking for one would
    test the provenance refusal instead of the stamp.
    """
    frame = wbl.records_to_bet_list_frame(
        _selection_result(season=2023),
        _fits(),
        run_mode=cache_module.RUN_MODE_REPLAY,
    )
    assert frame[DECIDED_AT_COLUMN].isna().all()
    assert (frame["provenance"] == PROVENANCE_BACKTEST_REPLAY).all()


def test_a_replay_emission_refuses_a_caller_supplied_observation_time() -> None:
    """Silently DROPPING the argument would be a second answer wearing the same name.

    A caller who passes ``decided_at`` under ``run_mode='replay'`` believes the stamp is being
    written. It is not, and cannot honestly be, so the mismatch is refused rather than ignored.
    """
    with pytest.raises(ValueError, match="replay"):
        wbl.records_to_bet_list_frame(
            _selection_result(season=2023),
            _fits(),
            run_mode=cache_module.RUN_MODE_REPLAY,
            decided_at=datetime(2023, 9, 8, 21, 59, tzinfo=UTC),
        )


def test_a_forward_emission_refuses_a_naive_run_instant() -> None:
    """The stamp goes through the ONE strict parse helper, so a naive clock raises (T-33-23)."""
    from scripts.ingest_historical_odds import NaiveTimestampError

    with pytest.raises(NaiveTimestampError):
        wbl.records_to_bet_list_frame(
            _selection_result(),
            _fits(),
            run_mode=cache_module.RUN_MODE_FORWARD,
            decided_at=datetime(2026, 9, 18, 21, 59),
        )


# ---------------------------------------------------------------------------
# (8) the retroactive-stamp prohibition: AST scan + runtime guard, four controls
# ---------------------------------------------------------------------------


class _DecidedAtWriteVisitor(ast.NodeVisitor):
    """Collect every ``<frame>[decided_at_utc] = <value>`` assignment and where it lives.

    An assignment whose value is a NULL literal is recorded but NOT an offence: the back-compat
    read shim fills the absent column with ``None``, which adds no observation time to anything.
    An assignment inside a named EMISSION function is likewise permitted, because that frame was
    built this run from this run's records. Everything else is a retroactive stamp.
    """

    def __init__(self) -> None:
        self.assignments: list[tuple[str, int]] = []
        self.offences: list[str] = []
        self._scope: list[str] = ["<module>"]

    def _visit_scoped(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node, node.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not self._targets_the_column(target):
                continue
            scope = self._scope[-1]
            self.assignments.append((scope, node.lineno))
            writes_null = (
                isinstance(node.value, ast.Constant) and node.value.value is None
            )
            if not writes_null and scope not in _EMISSION_FUNCTIONS:
                self.offences.append(
                    f"{scope}:{node.lineno} stamps {DECIDED_AT_COLUMN} outside the emission path"
                )
        self.generic_visit(node)

    @staticmethod
    def _targets_the_column(target: ast.expr) -> bool:
        if not isinstance(target, ast.Subscript):
            return False
        key = target.slice
        if isinstance(key, ast.Constant) and key.value == DECIDED_AT_COLUMN:
            return True
        return isinstance(key, ast.Name) and key.id == "DECIDED_AT_COLUMN"


def _scan(source: str) -> _DecidedAtWriteVisitor:
    visitor = _DecidedAtWriteVisitor()
    visitor.visit(ast.parse(source))
    return visitor


def test_control_the_scanner_finds_assignments_at_all_in_the_live_tree() -> None:
    """CONTROL 1 (non-vacuity): a scanner that finds nothing would pass every check for free."""
    found: list[tuple[str, int]] = []
    for path in _SCANNED_SOURCES:
        found += _scan(path.read_text(encoding="utf-8")).assignments
    assert found, (
        "the AST scan located no decided_at_utc assignment anywhere in the production tree, so "
        "the prohibition below would hold vacuously"
    )


def test_no_production_path_stamps_decided_at_onto_a_row_it_did_not_create() -> None:
    """CONTROL 2 (the assertion): T-33-22, over the live tree.

    The read shim's ``stored[DECIDED_AT_COLUMN] = None`` is permitted and present -- it fills an
    ABSENT column with NULL and records no time. Anything that writes a VALUE outside the emission
    path is a retroactive stamp and is flagged.
    """
    offences: list[str] = []
    for path in _SCANNED_SOURCES:
        offences += [
            f"{path.name}:{offence}"
            for offence in _scan(path.read_text(encoding="utf-8")).offences
        ]
    assert offences == [], "retroactive decided_at_utc stamps found:\n" + "\n".join(
        offences
    )


def test_control_a_planted_violation_is_caught() -> None:
    """CONTROL 3: the scan is proven load-bearing by making it fail on purpose."""
    planted = textwrap.dedent(
        """
        def backfill_stored_rows(stored):
            stored[DECIDED_AT_COLUMN] = stored["snapshot_ts"]
            return stored
        """
    )
    visitor = _scan(planted)
    assert len(visitor.offences) == 1
    assert "backfill_stored_rows" in visitor.offences[0]


def test_control_the_legitimate_shim_and_emission_writes_are_not_flagged() -> None:
    """CONTROL 4 (no false positive): the two writes that MUST remain legal stay legal.

    A guard that flagged the NULL fill would force the shim to be deleted, and a guard that
    flagged the emission stamp would forbid the column ever being populated at all -- either way
    the check would be removed rather than obeyed, which is how real guards die.
    """
    legitimate = textwrap.dedent(
        """
        def read_bet_list_with_schema_shim(path):
            stored = read(path)
            stored[DECIDED_AT_COLUMN] = None
            return stored

        def records_to_bet_list_frame(result, fits, *, run_mode, decided_at=None):
            frame = build(result)
            frame[DECIDED_AT_COLUMN] = decided_at.isoformat()
            return frame
        """
    )
    visitor = _scan(legitimate)
    assert visitor.offences == []
    assert len(visitor.assignments) == 2


def test_the_upsert_carries_a_runtime_guard_not_only_a_source_scan() -> None:
    """A source scan proves nobody WROTE the stamp; the runtime guard proves nobody DOES.

    A stored forward row already past its freeze wins whole, so its observation time must come out
    of the merge byte-identical. The guard is asserted here by DRIVING the merge, and the guard's
    own presence in the function is asserted by name so it cannot be deleted while this test still
    passes on the arithmetic alone.
    """
    stored_stamp = "2026-09-11T17:59:00-04:00"
    stored = _frame(
        _forward_row(
            game_id="2026_02_DET_BUF",
            week=2,
            snapshot_ts="2026-09-11T18:00:00-04:00",
            freeze_ts="2026-09-11T18:00:00-04:00",
            **{DECIDED_AT_COLUMN: stored_stamp},
        )
    )
    incoming = _frame(
        _forward_row(
            game_id="2026_02_DET_BUF",
            week=2,
            snapshot_ts="2026-09-11T18:00:00-04:00",
            freeze_ts="2026-09-11T18:00:00-04:00",
            **{DECIDED_AT_COLUMN: "2026-09-11T17:00:00-04:00"},
        )
    )
    # A clock AFTER the stored row's freeze: the stored row is frozen and wins whole.
    now = datetime(2026, 9, 11, 22, 0, tzinfo=UTC) + timedelta(seconds=1)
    merged = upsert_bet_list_rows(stored, incoming, now=now)

    assert len(merged) == 1
    assert merged.iloc[0][DECIDED_AT_COLUMN] == stored_stamp

    source = inspect.getsource(wbl.upsert_bet_list_rows)
    assert "_assert_stored_stamps_untouched" in source, (
        "the runtime no-retroactive-stamp guard is no longer called from the upsert"
    )


def test_the_runtime_guard_fires_when_a_stored_stamp_is_altered() -> None:
    """The guard is proven load-bearing: hand it a mutated stored half and it must raise."""
    original = _frame(_forward_row())
    mutated = original.copy()
    mutated.loc[0, DECIDED_AT_COLUMN] = "2026-09-18T17:58:00-04:00"

    with pytest.raises(RuntimeError, match=DECIDED_AT_COLUMN):
        wbl._assert_stored_stamps_untouched(original, mutated)

    # And it does NOT fire on the untouched case, or it would be a guard nobody could satisfy.
    wbl._assert_stored_stamps_untouched(original, original.copy())
