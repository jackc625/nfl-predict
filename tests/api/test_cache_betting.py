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
        assert edge_tier(value) == expected, (
            f"edge {value!r} now bands as {edge_tier(value)!r}, was {expected!r} before the "
            "collapse; a published label on / and /betting has moved"
        )


def test_the_vectorized_form_agrees_with_the_scalar_one_on_the_same_snapshot() -> None:
    """``edge_tier_series`` DISPATCHES; it must not be a second rule that can drift."""
    from utils.edge_tier import edge_tier_series

    values = [value for value, _label in _EDGE_TIER_SNAPSHOT]
    expected = [label for _value, label in _EDGE_TIER_SNAPSHOT]
    assert list(edge_tier_series(pd.Series(values))) == expected


def test_an_absent_edge_bands_low_exactly_as_both_retired_helpers_did() -> None:
    """NaN reached ``low`` via ``np.where`` in one helper and a ``pd.notna`` guard in the other.

    Answering it inside the shared helper is what stops the two call sites diverging on the case
    neither of them stated explicitly.
    """
    from utils.edge_tier import edge_tier, edge_tier_series

    assert edge_tier(np.nan) == "low"
    assert edge_tier(None) == "low"
    assert list(edge_tier_series(pd.Series([np.nan, 0.09]))) == ["low", "high"]


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

    # And the shared module itself defines EXACTLY ONE banding function -- the one reading both
    # named thresholds. Asserted separately because the scan above deliberately cannot see it (it
    # uses no literals), so without this the whole check would pass on an EMPTY tree.
    shared = ast.parse(pathlib.Path("utils/edge_tier.py").read_text(encoding="utf-8"))
    banding = [
        node.name
        for node in ast.walk(shared)
        if isinstance(node, ast.FunctionDef)
        and {"EDGE_TIER_HIGH_THRESHOLD", "EDGE_TIER_MEDIUM_THRESHOLD"}
        <= {c.id for c in ast.walk(node) if isinstance(c, ast.Name)}
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
                text = ast.unparse(node)
                if "edge_tier" not in text:
                    continue
                if "_edge" not in text:
                    offenders.append(f"{path.as_posix()}:{node.lineno} {text}")

    assert not offenders, (
        "a call site passes something other than an *_edge column into the edge band; a per-bet "
        "expected value banded on this scale would be misclassified (backtest/simulation.py's "
        "warning):\n" + "\n".join(offenders)
    )
