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
        count_row = conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()
        assert count_row is not None
        assert count_row[0] == 4

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

        positive_row = conn.execute(
            "SELECT COUNT(*) FROM betting_bets WHERE kelly_stake > 0"
        ).fetchone()
        zero_row = conn.execute(
            "SELECT COUNT(*) FROM betting_bets WHERE kelly_stake = 0"
        ).fetchone()
        assert positive_row is not None
        assert zero_row is not None

        # 3 recommended rows + 1 dropped-by-recommended row = the full ledger.
        assert positive_row[0] == 3
        assert zero_row[0] == 1
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
        count_row = conn.execute("SELECT COUNT(*) FROM betting_bets").fetchone()
        assert count_row is not None
        assert count_row[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The collapsed edge band (Phase 31, plan 31-17; D31-23, T-31-88/89)
# ---------------------------------------------------------------------------
#
# EXTENDED, not rewritten. Plan 31-17 collapsed two byte-equivalent tier helpers --
# ``api/cache.py::_compute_confidence`` and
# ``scripts/generate_current_week_predictions.py::compute_confidence`` -- into ONE shared source in
# ``utils/edge_tier.py`` and RENAMED the concept to an EDGE BAND. Existing page behaviour is
# UNCHANGED, which is exactly what makes the snapshot below a genuine regression test rather than a
# rewrite with a new expectation.

# The 23-point grid and the labels BOTH retired helpers produced, recorded by running them side by
# side BEFORE the collapse (2026-09-06). It spans all three targets' scales and every boundary --
# each threshold exactly, just below and just above -- because the retired comparison was STRICT
# (``>``), so a value exactly at a threshold belongs to the LOWER band and an accidental ``>=``
# would move a published label. The two helpers agreed on every one of these 23 values.
_EDGE_TIER_SNAPSHOT: tuple[tuple[float, str], ...] = (
    (-1.0, "high"),
    (-0.5, "high"),
    (-0.10, "high"),
    (-0.0500000001, "high"),
    (-0.05, "medium"),
    (-0.0499999, "medium"),
    (-0.03, "medium"),
    (-0.020000001, "medium"),
    (-0.02, "low"),
    (-0.0199999, "low"),
    (-0.001, "low"),
    (0.0, "low"),
    (0.001, "low"),
    (0.0199999, "low"),
    (0.02, "low"),
    (0.020000001, "medium"),
    (0.03, "medium"),
    (0.0499999, "medium"),
    (0.05, "medium"),
    (0.0500000001, "high"),
    (0.10, "high"),
    (0.5, "high"),
    (1.0, "high"),
)


def test_the_collapsed_edge_band_reproduces_the_pre_collapse_labels_value_by_value() -> (
    None
):
    """Every one of the 23 recorded values still bands exactly as it did (T-31-88)."""
    from utils.edge_tier import edge_tier

    for value, expected in _EDGE_TIER_SNAPSHOT:
        assert edge_tier(value, "wp") == expected, (
            f"edge {value!r} now bands as {edge_tier(value, 'wp')!r}, was {expected!r} before "
            "the collapse; a published label on / and /betting has moved"
        )


def test_the_vectorized_form_agrees_with_the_scalar_one_on_the_same_snapshot() -> None:
    """``edge_tier_series`` DISPATCHES; it must not be a second rule that can drift."""
    from utils.edge_tier import edge_tier_series

    values = [value for value, _label in _EDGE_TIER_SNAPSHOT]
    expected = [label for _value, label in _EDGE_TIER_SNAPSHOT]
    assert list(edge_tier_series(pd.Series(values), "wp")) == expected


def test_an_absent_edge_has_no_band() -> None:
    """An absent edge is UNBANDED (33.2 review C2 CR-04 = B WR-11).

    Both retired helpers answered NaN with "low" -- one via ``np.where``, one via a ``pd.notna``
    guard -- so every game with no line was stored and exported with a band claiming a small
    measured edge. The answer is now ``None``, still given inside the shared helper so the call
    sites cannot diverge on it.
    """
    from utils.edge_tier import edge_tier, edge_tier_series

    assert edge_tier(np.nan, "wp") is None
    assert edge_tier(None, "wp") is None
    assert list(edge_tier_series(pd.Series([np.nan, 0.09]), "wp")) == [None, "high"]


def test_exactly_one_function_computes_the_edge_band() -> None:
    """A source scan finds ONE definition, and the prediction script declares none of its own.

    Structural: it walks the production tree for a function whose body performs the band's own
    threshold comparison, rather than trusting that the retired twins were both deleted.
    """
    import ast
    import pathlib

    definitions: list[str] = []
    for package in ("api", "backtest", "models", "pipeline", "scripts", "utils", "web"):
        root = pathlib.Path(package)
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                literals = {
                    child.value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, float)
                }
                labels = {
                    child.value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                }
                if {0.05, 0.02} <= literals and {"high", "medium", "low"} <= labels:
                    definitions.append(f"{path.as_posix()}:{node.lineno} {node.name}")

    assert not definitions, (
        "a function reimplements the edge band from literal thresholds again; the rule lives in "
        "utils/edge_tier.py behind NAMED constants and every caller imports it:\n"
        + "\n".join(definitions)
    )

    # And the shared module itself defines EXACTLY ONE banding function. Asserted separately
    # because the scan above deliberately cannot see it (it uses no literals), so without this the
    # whole check would pass on an EMPTY tree.
    #
    # RE-IDENTIFIED for the per-target signature (Plan 33-17, D33-22). The old form looked for a
    # function naming both module-level threshold constants; ``edge_tier`` now unpacks its pair
    # from the FROZEN per-target mapping and names no such constant, so that form would find
    # nothing and assert on an empty list. The structural property that actually matters is
    # unchanged and is now stated directly: exactly one function in the module both HOLDS a
    # ``(high, medium)`` pair and COMPARES against it. ``edge_tier_series`` must therefore stay a
    # pure dispatch, which ``tests/unit/test_edge_tier_per_target.py`` asserts from the other side.
    shared = ast.parse(pathlib.Path("utils/edge_tier.py").read_text(encoding="utf-8"))
    banding = [
        node.name
        for node in ast.walk(shared)
        if isinstance(node, ast.FunctionDef)
        and {"high", "medium"}
        <= {c.id for c in ast.walk(node) if isinstance(c, ast.Name)}
        and any(isinstance(c, ast.Compare) for c in ast.walk(node))
    ]
    assert banding == ["edge_tier"], (
        f"utils/edge_tier.py must define exactly one banding function; found {banding}"
    )

    script = pathlib.Path("scripts/generate_current_week_predictions.py").read_text(
        encoding="utf-8"
    )
    assert "from utils.edge_tier import edge_tier" in script, (
        "the prediction script must IMPORT the shared band rather than declare its own"
    )
    assert "def compute_confidence" not in script, (
        "the script's duplicate band helper survives the collapse"
    )


def test_the_edge_band_and_the_expected_value_tier_are_different_functions() -> None:
    """``/bets`` owns ``ev_tier``; the existing pages own the renamed ``edge_tier`` (D31-23/24).

    Different names, different modules, different rules -- so the word "high" cannot mean two
    incompatible things across the two surfaces. This is the separation
    ``backtest/simulation.py``'s warning comment asked for, asserted rather than described.
    """
    from backtest.ev_chain_constants import assign_ev_tier
    from utils.edge_tier import edge_tier

    assert assign_ev_tier is not edge_tier
    assert assign_ev_tier.__name__ != edge_tier.__name__
    assert assign_ev_tier.__module__ == "backtest.ev_chain_constants"
    assert edge_tier.__module__ == "utils.edge_tier"


def test_no_call_site_feeds_a_per_bet_expected_value_into_the_edge_band() -> None:
    """Structural: every expression naming the band also names an ``*_edge`` column.

    ``backtest/simulation.py`` warns that a selector-produced row's edge field carries per-bet
    EXPECTED VALUE, and that banding one with this helper would misclassify it. Both call forms are
    covered -- a direct ``edge_tier_series(frame["wp_edge"])`` and an
    ``frame["wp_edge"].apply(edge_tier)`` -- because the scan reads the whole unparsed call rather
    than only its argument list.
    """
    import ast
    import pathlib

    offenders: list[str] = []
    for package in ("api", "backtest", "models", "pipeline", "scripts", "web"):
        root = pathlib.Path(package)
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Match a reference to the band itself, not the substring: a keyword such as
                # ``edge_tier_thresholds=`` (backtest/recipe_registry.py) is not a call site.
                referenced = {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                } | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
                if not referenced & {"edge_tier", "edge_tier_series"}:
                    continue
                text = ast.unparse(node)
                if "_edge" not in text:
                    offenders.append(f"{path.as_posix()}:{node.lineno} {text}")

    assert not offenders, (
        "a call site passes something other than an *_edge column into the edge band; a per-bet "
        "expected value banded on this scale would be misclassified (backtest/simulation.py's "
        "warning):\n" + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# THE THREE bet_list DDL SITES MUST AGREE, COLUMN FOR COLUMN AND IN ORDER
# ---------------------------------------------------------------------------
#
# Phase 33, Plan 33-05 Task 3 (T-33-24). `bet_list` is created from THREE places, not two:
#
#   1. `conn.execute(BET_LIST_SCHEMA)` inside `materialize_bet_list`,
#   2. `conn.execute(BET_LIST_SCHEMA)` inside the cache build,
#   3. the copy EMBEDDED in `CACHE_SCHEMA`, which the comment above `BET_LIST_SCHEMA` names as
#      "the SAME definition embedded in CACHE_SCHEMA above".
#
# Sites 1 and 2 share one constant, so the constant is what is under test for both. Site 3 is a
# separate literal and is the one that can silently drift: a column added to `BET_LIST_COLUMNS`
# and to `BET_LIST_SCHEMA` but not to `CACHE_SCHEMA` produces a real cache whose table is one
# column narrower than the explicit-column INSERT names, and the INSERT then fails in
# production rather than in this suite.
#
# Both are BUILT and read back through `PRAGMA table_info` rather than compared as text, so a
# formatting difference is not a failure and a width difference is.


def _bet_list_ddl_columns(statement: str) -> list[str]:
    """Create ``bet_list`` from one DDL *statement* and return its column names, in order.

    ``PRAGMA table_info`` returns ``(cid, name, type, ...)`` -- the NAME is index 1, and index 0
    is the ordinal. Reading index 0 gives ``[0, 1, 2, ...]``, which fails every order comparison
    for a reason unrelated to the schema.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(statement)
    return [row[1] for row in conn.execute("PRAGMA table_info('bet_list')").fetchall()]


def _embedded_bet_list_statement() -> str:
    """The single ``bet_list`` CREATE embedded in ``CACHE_SCHEMA``, isolated by name."""
    statements = [s.strip() for s in CACHE_SCHEMA.strip().split(";") if s.strip()]
    matches = [s for s in statements if "CREATE TABLE IF NOT EXISTS bet_list" in s]
    assert len(matches) == 1, (
        f"expected exactly one embedded bet_list CREATE in CACHE_SCHEMA, found {len(matches)}"
    )
    return matches[0]


def test_all_three_bet_list_ddl_sites_produce_the_locked_column_order() -> None:
    """Each site builds the table, and every built table equals ``BET_LIST_COLUMNS`` exactly."""
    from api.cache import BET_LIST_COLUMNS, BET_LIST_SCHEMA

    standalone = _bet_list_ddl_columns(BET_LIST_SCHEMA)
    embedded = _bet_list_ddl_columns(_embedded_bet_list_statement())

    assert standalone == list(BET_LIST_COLUMNS)
    assert embedded == list(BET_LIST_COLUMNS)
    assert standalone == embedded
    # Phase 34 (Plan 34-01 Task 2) widened every site from 29 to 51 columns. Was: the width was
    # pinned only through ``BET_LIST_COLUMNS`` (29). Pinned here too, so a site that drops the
    # whole Phase-34 block cannot pass by agreeing with a list that dropped it as well.
    assert len(standalone) == 51


def test_the_two_ddl_literals_build_the_same_column_types() -> None:
    """Names agreeing is not enough: a column typed differently at one site is a second schema.

    Phase 34 (Plan 34-01 Task 2) added 22 columns to BOTH literals; each is BUILT and its
    ``(name, type)`` pairs compared, so a VARCHAR at one site and a DOUBLE at the other fails.
    """
    from api.cache import BET_LIST_SCHEMA

    def _typed(statement: str) -> list[tuple[str, str]]:
        conn = duckdb.connect(":memory:")
        conn.execute(statement)
        return [
            (row[1], row[2])
            for row in conn.execute("PRAGMA table_info('bet_list')").fetchall()
        ]

    assert _typed(BET_LIST_SCHEMA) == _typed(_embedded_bet_list_statement())


def test_the_full_cache_build_yields_the_same_bet_list_table() -> None:
    """The WHOLE ``CACHE_SCHEMA`` executed statement by statement, as the real build does.

    Isolating the embedded CREATE proves the literal is right; running the whole schema proves
    nothing LATER in it alters the table -- an ALTER or a second CREATE would be invisible to the
    isolated check.
    """
    from api.cache import BET_LIST_COLUMNS

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    columns = [
        row[1] for row in conn.execute("PRAGMA table_info('bet_list')").fetchall()
    ]
    assert columns == list(BET_LIST_COLUMNS)


def test_the_explicit_column_insert_names_every_ddl_column() -> None:
    """The INSERT and the DDL are the two halves of one contract; a width gap breaks the write.

    Driven rather than inspected: a fully-populated row is written through
    ``materialize_bet_list`` and read back by NAME, so a column present in the DDL but absent from
    the INSERT's name list would surface as a NULL that the assertion catches.
    """
    from api.cache import BET_LIST_COLUMNS, materialize_bet_list

    row = dict.fromkeys(BET_LIST_COLUMNS, None)
    row.update(
        {
            "game_id": "2026_03_DAL_NYG",
            "season": 2026,
            "week": 3,
            "target": "ou",
            "bet_side": "under",
            "status": "live",
            "provenance": "forward",
            "validation_type": "forward_realized",
            "snapshot_ts": "2026-09-18T18:00:00-04:00",
            "freeze_ts": "2026-09-18T18:00:00-04:00",
            "decided_at_utc": "2026-09-18T17:59:00-04:00",
            "grading_status": "pending",
            # Phase 34 (Plan 34-01 Task 2): one column from each new block, so a Phase-34 column
            # present in the DDL but absent from the INSERT's name list reads back NULL here.
            # Was: the row stopped at ``grading_status``.
            "arm": "live",
            "fill_convention_id": "fill-v1",
            "fill_line": 44.5,
            "closing_null_reason": "capture_missed",
        }
    )

    conn = duckdb.connect(":memory:")
    assert materialize_bet_list(conn, pd.DataFrame([row])) == 1
    stored = conn.execute(
        "SELECT decided_at_utc, freeze_ts, provenance, arm, fill_convention_id, fill_line, "
        "closing_null_reason FROM bet_list"
    ).fetchone()
    # Was: the first three fields only.
    assert stored == (
        "2026-09-18T17:59:00-04:00",
        "2026-09-18T18:00:00-04:00",
        "forward",
        "live",
        "fill-v1",
        44.5,
        "capture_missed",
    )


# ---------------------------------------------------------------------------
# Phase 34 (Plan 34-16): the corrections and graded-outcome tables, and the verdict / evidence meta
# ---------------------------------------------------------------------------
#
# Two new cache tables carry the forward record's corrected outcomes (D-06) and the /bets result
# strip's per-bet marks (review finding 4). Each is spelled at THREE sites exactly like bet_list:
# the CACHE_SCHEMA literal, a standalone schema constant, and an explicit-column INSERT. They are
# BUILT and compared, and a full row is driven through the writer and read back by name.


def _embedded_statement(table: str) -> str:
    statements = [s.strip() for s in CACHE_SCHEMA.strip().split(";") if s.strip()]
    matches = [s for s in statements if f"CREATE TABLE IF NOT EXISTS {table} " in s]
    assert len(matches) == 1, (
        f"expected exactly one embedded {table} CREATE in CACHE_SCHEMA, found {len(matches)}"
    )
    return matches[0]


def _typed_columns(statement: str, table: str) -> list[tuple[str, str]]:
    conn = duckdb.connect(":memory:")
    conn.execute(statement)
    return [
        (row[1], row[2])
        for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    ]


def test_corrections_table_three_sites() -> None:
    """CACHE_SCHEMA, the standalone schema and the explicit INSERT agree on the column list."""
    from api.cache import (
        BET_LIST_CORRECTIONS_COLUMNS,
        BET_LIST_CORRECTIONS_SCHEMA,
        materialize_bet_list_corrections,
    )

    standalone = _typed_columns(BET_LIST_CORRECTIONS_SCHEMA, "bet_list_corrections")
    embedded = _typed_columns(
        _embedded_statement("bet_list_corrections"), "bet_list_corrections"
    )
    assert standalone == embedded
    assert [name for name, _type in standalone] == list(BET_LIST_CORRECTIONS_COLUMNS)

    row = {
        "game_id": "2026_06_KC_BUF",
        "season": 2026,
        "week": 6,
        "target": "ats",
        "arm": "live",
        "original_grading_status": "win",
        "corrected_grading_status": "loss",
        "corrected_outcome": False,
        "corrected_payout_flat": -1.0,
        "corrected_realized_units": -0.5,
        "realized_value": -3.0,
        "detected_at_utc": "2026-10-20T12:00:00+00:00",
        "corrected_at_utc": "2026-10-20T21:00:00+00:00",
    }
    conn = duckdb.connect(":memory:")
    # Shuffled column order: the INSERT is by name, never positional.
    shuffled = pd.DataFrame([row])[list(reversed(BET_LIST_CORRECTIONS_COLUMNS))]
    assert materialize_bet_list_corrections(conn, shuffled) == 1
    stored = conn.execute(
        f"SELECT {', '.join(BET_LIST_CORRECTIONS_COLUMNS)} FROM bet_list_corrections"
    ).fetchone()
    assert stored == tuple(row[name] for name in BET_LIST_CORRECTIONS_COLUMNS)


def test_graded_outcomes_table_three_sites(tmp_path: Path) -> None:
    """The strip table agrees at all three sites; ``None`` leaves it present and empty."""
    from api.cache import (
        BET_GRADED_OUTCOMES_COLUMNS,
        BET_GRADED_OUTCOMES_SCHEMA,
        materialize_bet_graded_outcomes,
        populate_cache,
    )

    standalone = _typed_columns(BET_GRADED_OUTCOMES_SCHEMA, "bet_graded_outcomes")
    embedded = _typed_columns(
        _embedded_statement("bet_graded_outcomes"), "bet_graded_outcomes"
    )
    assert standalone == embedded
    assert [name for name, _type in standalone] == list(BET_GRADED_OUTCOMES_COLUMNS)

    row = {
        "provenance": "forward",
        "validation_type": "pre_verdict",
        "season": 2026,
        "week": 4,
        "game_id": "2026_04_DAL_NYG",
        "target": "ou",
        "arm": "live",
        "grading_status": "win",
        "corrected": True,
    }
    conn = duckdb.connect(":memory:")
    shuffled = pd.DataFrame([row])[list(reversed(BET_GRADED_OUTCOMES_COLUMNS))]
    assert materialize_bet_graded_outcomes(conn, shuffled) == 1
    stored = conn.execute(
        f"SELECT {', '.join(BET_GRADED_OUTCOMES_COLUMNS)} FROM bet_graded_outcomes"
    ).fetchone()
    assert stored == tuple(row[name] for name in BET_GRADED_OUTCOMES_COLUMNS)

    db_path = tmp_path / "cache.duckdb"
    populate_cache(
        db_path=db_path,
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=tmp_path / "outputs",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        bet_graded_outcomes_df=None,
    )
    built = duckdb.connect(str(db_path), read_only=True)
    try:
        count = built.execute("SELECT COUNT(*) FROM bet_graded_outcomes").fetchone()
        corrections = built.execute(
            "SELECT COUNT(*) FROM bet_list_corrections"
        ).fetchone()
    finally:
        built.close()
    assert count == (0,)
    assert corrections == (0,)


def _meta(db_path: Path) -> dict[str, str | None]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return dict(conn.execute("SELECT key, value FROM cache_meta").fetchall())
    finally:
        conn.close()


def test_verdict_context_and_betting_sim_pair_stamped(tmp_path: Path) -> None:
    """The verdict context and the betting-sim evidence pair are stamped at cache build."""
    from api.cache import (
        BETTING_SIM_PROVENANCE_KEY,
        BETTING_SIM_VALIDATION_TYPE_KEY,
        FORWARD_VERDICT_DECLARED_KEY,
        FORWARD_VERDICT_SEASON_KEY,
        FORWARD_VERDICT_START_WEEK_KEY,
        REPLAY_VALIDATION_TYPE_SEASONS,
        betting_simulation_evidence_pair,
        populate_cache,
    )

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _write_betting_csv(outputs_dir)  # seasons 2021, 2022, 2023

    declared_db = tmp_path / "declared.duckdb"
    populate_cache(
        db_path=declared_db,
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=outputs_dir,
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        forward_verdict_context={"declared": True, "season": 2026, "start_week": 6},
    )
    meta = _meta(declared_db)
    assert meta[FORWARD_VERDICT_DECLARED_KEY] == "true"
    assert meta[FORWARD_VERDICT_SEASON_KEY] == "2026"
    assert meta[FORWARD_VERDICT_START_WEEK_KEY] == "6"
    assert meta[BETTING_SIM_PROVENANCE_KEY] == "backtest_replay"
    assert meta[BETTING_SIM_VALIDATION_TYPE_KEY] == "contaminated"

    undeclared_db = tmp_path / "undeclared.duckdb"
    populate_cache(
        db_path=undeclared_db,
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=tmp_path / "no_outputs",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
    )
    meta = _meta(undeclared_db)
    assert meta[FORWARD_VERDICT_DECLARED_KEY] == "false"
    assert meta[FORWARD_VERDICT_SEASON_KEY] is None
    assert meta[FORWARD_VERDICT_START_WEEK_KEY] is None
    # No simulation rows: no evidence pair is claimed.
    assert BETTING_SIM_PROVENANCE_KEY not in meta
    assert BETTING_SIM_VALIDATION_TYPE_KEY not in meta

    # The pair is derived through the replay season map, not re-typed.
    contaminated = REPLAY_VALIDATION_TYPE_SEASONS["contaminated"]
    assert betting_simulation_evidence_pair(contaminated) == (
        "backtest_replay",
        "contaminated",
    )
    assert betting_simulation_evidence_pair(
        REPLAY_VALIDATION_TYPE_SEASONS["clean_holdout"]
    ) == ("backtest_replay", "clean_holdout")
    assert betting_simulation_evidence_pair({2024, 2025}) is None
    assert betting_simulation_evidence_pair(set()) is None
