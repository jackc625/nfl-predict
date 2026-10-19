"""The cache's contract, writer side (Phase 36, Plan 36-01).

What this file pins, and why:

* ONE SPELLING PER WRITER CONSTANT (D-11, research PITFALLS E2). The request path reads the cache
  writer's constants and parsers, but must never import the writer (``api/cache.py`` loads numpy,
  pandas and scipy). The names moved to the standard-library leaf ``api/cache_schema.py`` and the
  writer imports them back, so ``api.cache.X is api.cache_schema.X`` for every moved name. A second
  definition would be two spellings that can drift.
* THE SERVICE NO LONGER REACHES THE WRITER. ``api.services`` and ``api.routes.pages`` import the
  leaf, never ``api.cache``, and a fresh import of ``api.services`` leaves ``api.cache`` unloaded.
* THE DATA-FORMAT VERSION (D-20). ``populate_cache`` stamps ``DATA_FORMAT_VERSION`` into
  ``cache_meta``; the reader parses it strictly, so a missing, non-integer or different value is a
  mismatch.
* THE LAYOUT PIN (D-04). ``PINNED_LAYOUT`` is every table and its columns as the writer actually
  builds them, beside ``PINNED_VERSION``. A layout change without a version bump fails here. It was
  generated once from a real build, never hand-typed. Its known limit: it sees table and column
  names only, never the inside of a stored blob.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

import api.cache
from api import cache_schema
from api.cache import CACHE_META_SCHEMA, populate_cache
from api.cache_reader import open_cache_readonly, read_data_format
from api.cache_schema import (
    DATA_FORMAT_VERSION,
    DATA_FORMAT_VERSION_KEY,
    data_format_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every name Plan 36-01 moved out of api/cache.py into api/cache_schema.py: the 23 the request path
# imports, their dependents, and the block-mates of the two closed vocabularies they belong to.
MOVED_NAMES: tuple[str, ...] = (
    "ARM_LIVE",
    "ARM_SHADOW",
    "ARMS",
    "BET_STATUS_LIVE",
    "BET_LIST_IMMUTABLE_COLUMNS",
    "BET_LIST_GRADING_COLUMNS",
    "BET_LIST_FILL_COLUMNS",
    "BET_LIST_CLOSING_COLUMNS",
    "BET_LIST_COLUMNS",
    "BET_GRADED_OUTCOMES_COLUMNS",
    "BET_LIST_CORRECTIONS_COLUMNS",
    "BET_TRACKER_BLOCK_COLUMNS",
    "BET_WEEK_RESULTS_COLUMNS",
    "BET_WEEK_CLASS_ORDER",
    "PROVENANCE_BACKTEST_REPLAY",
    "PROVENANCE_FORWARD",
    "VALIDATION_TYPE_CONTAMINATED",
    "VALIDATION_TYPE_CLEAN_HOLDOUT",
    "VALIDATION_TYPE_FORWARD_REALIZED",
    "VALIDATION_TYPE_PRE_VERDICT",
    "_REPLAY_CONTAMINATED_SEASONS",
    "_REPLAY_CLEAN_HOLDOUT_SEASON",
    "REPLAY_VALIDATION_TYPE_SEASONS",
    "BETTING_SIM_PROVENANCE_KEY",
    "BETTING_SIM_VALIDATION_TYPE_KEY",
    "FORWARD_VERDICT_DECLARED_KEY",
    "FORWARD_VERDICT_SEASON_KEY",
    "FORWARD_VERDICT_START_WEEK_KEY",
    "PREDICTIONS_TABLE_COLUMNS",
    "BET_LIST_POPULATED_AT_KEY",
    "BET_LIST_POPULATED_AT_SEPARATOR",
    "bet_list_populated_at_key",
    "CURRENT_SLATE_KEY",
    "_CURRENT_SLATE_SEPARATOR",
    "format_current_slate",
    "parse_current_slate",
    "BACKTEST_SEASON_RANGE_KEY",
    "BETTING_SEASON_RANGE_KEY",
    "NO_SEASONS",
    "format_season_range",
    "parse_season_range",
)


def _fresh_interpreter(probe: str) -> list[str]:
    """Run *probe* in a fresh interpreter at the repo root and return the JSON list it prints.

    Not ``-I``: isolated mode drops the working directory from ``sys.path``, so ``api`` would not
    import at all.
    """
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_moved_name_has_one_spelling() -> None:
    """Each moved name is ONE object, reachable from both modules (T-36-01)."""
    for name in MOVED_NAMES:
        assert getattr(api.cache, name) is getattr(cache_schema, name), (
            f"api.cache.{name} is not api.cache_schema.{name}: the name has two definitions"
        )


def test_the_leaf_and_the_service_load_no_writer() -> None:
    """The leaf imports only the standard library; the service never loads the writer.

    The leaf is checked on its own import statements, because a fresh ``import api.cache_schema``
    still runs ``api/__init__.py``, which builds the whole app until Plan 36-03 empties it. The
    service is checked by what a fresh interpreter actually loads.
    """
    leaf = REPO_ROOT / "api" / "cache_schema.py"
    roots: set[str] = set()
    for node in ast.walk(ast.parse(leaf.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the leaf must not import a sibling module"
            roots.add((node.module or "").split(".")[0])
    outside = sorted(roots - set(sys.stdlib_module_names))
    assert roots, "the leaf's imports were not read"
    assert outside == [], (
        f"api/cache_schema.py imports outside the standard library: {outside}"
    )

    for path in ("api/services.py", "api/routes/pages.py"):
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        writer_imports = [
            node.lineno
            for node in ast.walk(tree)
            if (isinstance(node, ast.ImportFrom) and node.module == "api.cache")
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "api.cache" for alias in node.names)
            )
        ]
        assert writer_imports == [], (
            f"{path} still imports api.cache at {writer_imports}"
        )

    loaded = _fresh_interpreter(
        "import json, sys, api.services; print(json.dumps(sorted(sys.modules)))"
    )
    assert "api.services" in loaded
    assert "api.cache" not in loaded, (
        "a fresh import of api.services loaded the cache writer"
    )


# ---------------------------------------------------------------------------
# The data-format version (D-20) and the layout pin (D-04)
# ---------------------------------------------------------------------------


@pytest.fixture
def built_cache(tmp_path: Path) -> Path:
    """A real cache, built by the writer over an empty inputs directory, under tmp_path only."""
    empty = tmp_path / "empty"
    empty.mkdir()
    db_path = tmp_path / "web_cache.duckdb"
    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
    )
    return db_path


def test_populate_stamps_the_data_format_version(built_cache: Path) -> None:
    """The writer stamps the integer the code expects, and the reader reads it back as a match."""
    conn = open_cache_readonly(built_cache)
    try:
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE key = ?", [DATA_FORMAT_VERSION_KEY]
        ).fetchone()
        state = read_data_format(conn)
    finally:
        conn.close()
    assert row is not None, "populate_cache stamped no data-format version"
    assert row[0] == str(DATA_FORMAT_VERSION)
    assert state.matches
    assert state.raw == str(DATA_FORMAT_VERSION)


# The layout the writer builds, at the data-format version it was pinned at. GENERATED from a real
# populate_cache build (information_schema.columns, ordinal order) and pasted; never hand-typed.
PINNED_VERSION = 1
PINNED_LAYOUT: dict[str, tuple[str, ...]] = {
    "available_bet_weeks": (
        "season",
        "week",
        "game_count",
    ),
    "backtest_metrics": (
        "season",
        "target",
        "metric_name",
        "metric_value",
    ),
    "backtest_predictions": (
        "game_id",
        "season",
        "week",
        "target",
        "model_prob",
        "actual",
        "probability_clv",
        "has_closing_odds",
    ),
    "bet_game_lock": (
        "season",
        "week",
        "game_id",
        "lock_ts",
    ),
    "bet_graded_outcomes": (
        "provenance",
        "validation_type",
        "season",
        "week",
        "game_id",
        "target",
        "arm",
        "grading_status",
        "corrected",
    ),
    "bet_list": (
        "game_id",
        "season",
        "week",
        "target",
        "bet_side",
        "model_value",
        "market_value",
        "line",
        "slipped_line",
        "calibrated_p_side",
        "per_bet_ev",
        "stake_units",
        "ev_tier",
        "status",
        "rejection_reason",
        "eligibility_label",
        "snapshot_ts",
        "freeze_ts",
        "selected_odds",
        "flat_stake",
        "provenance",
        "validation_type",
        "decided_at_utc",
        "arm",
        "model_artifact_id",
        "blend_id",
        "recipe_id",
        "fill_convention_id",
        "upstream_capture_key",
        "gold_generation_key",
        "odds_snapshot_digest",
        "decision_snapshot_digest",
        "verdict_scope",
        "regime_label",
        "grading_status",
        "outcome",
        "clv",
        "payout_flat",
        "realized_units",
        "graded_at",
        "fill_sportsbook",
        "fill_line",
        "fill_odds",
        "fill_stake_dollars",
        "fill_at_utc",
        "closing_line",
        "closing_odds",
        "closing_sportsbook",
        "closing_captured_at",
        "forward_clv",
        "closing_null_reason",
    ),
    "bet_list_corrections": (
        "game_id",
        "season",
        "week",
        "target",
        "arm",
        "original_grading_status",
        "corrected_grading_status",
        "corrected_outcome",
        "corrected_payout_flat",
        "corrected_realized_units",
        "realized_value",
        "detected_at_utc",
        "corrected_at_utc",
    ),
    "bet_tracker_blocks": (
        "provenance",
        "validation_type",
        "bets_graded",
        "wins",
        "losses",
        "pushes",
        "hit_rate",
        "flat_return_units",
    ),
    "bet_week_freeze": (
        "season",
        "week",
        "latest_game_freeze_ts",
    ),
    "bet_week_results": (
        "season",
        "week",
        "counting_class",
        "wins",
        "losses",
        "pushes",
        "bets_graded",
        "bets_ungraded",
        "actual_return_units",
        "expected_return_units",
    ),
    "betting_bets": (
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
    ),
    "cache_meta": (
        "key",
        "value",
        "updated_at",
    ),
    "chart_cache": (
        "chart_id",
        "html_div",
        "generated_at",
    ),
    "equity_curve": (
        "strategy",
        "bet_index",
        "bankroll",
    ),
    "feature_importances": (
        "game_id",
        "target",
        "feature_name",
        "importance",
    ),
    "game_context": (
        "game_id",
        "home_elo",
        "away_elo",
        "home_last5",
        "away_last5",
        "h2h_record",
        "venue_name",
        "surface",
        "roof_type",
        "weather_severity",
        "weather_severity_band",
        "wind_mph",
        "is_outdoor",
        "is_divisional",
        "is_primetime",
    ),
    "predictions": (
        "game_id",
        "season",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "status",
        "home_score",
        "away_score",
        "wp_prob",
        "wp_confidence",
        "ats_prediction",
        "ats_confidence",
        "ou_prediction",
        "ou_confidence",
        "market_spread",
        "market_total",
        "market_ml_home",
        "market_ml_away",
        "wp_edge",
        "ats_edge",
        "ou_edge",
        "blended_wp",
        "blended_ats",
        "blended_ou",
        "market_wp",
    ),
    "simulation_results": (
        "strategy",
        "metric_name",
        "metric_value",
    ),
}


def test_the_pinned_layout_matches_what_the_writer_builds(built_cache: Path) -> None:
    """A layout change without a version bump fails here (D-04, T-36-04)."""
    conn = open_cache_readonly(built_cache)
    try:
        rows = conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ).fetchall()
    finally:
        conn.close()
    built: dict[str, list[str]] = {}
    for table, column in rows:
        built.setdefault(table, []).append(column)
    layout = {table: tuple(columns) for table, columns in built.items()}

    message = (
        "the cache layout changed: raise api.cache_schema.DATA_FORMAT_VERSION and update "
        "PINNED_VERSION and PINNED_LAYOUT in this test"
    )
    assert layout == PINNED_LAYOUT, message
    assert PINNED_VERSION == DATA_FORMAT_VERSION, message


@pytest.mark.parametrize(
    ("raw", "matches", "found_label"),
    [
        (str(DATA_FORMAT_VERSION), True, str(DATA_FORMAT_VERSION)),
        (str(DATA_FORMAT_VERSION + 1), False, str(DATA_FORMAT_VERSION + 1)),
        (str(DATA_FORMAT_VERSION - 1), False, str(DATA_FORMAT_VERSION - 1)),
        (None, False, "missing"),
        ("", False, "missing"),
        ("abc", False, "non_integer"),
        ("1.0", False, "non_integer"),
    ],
)
def test_the_data_format_state_tells_a_match_from_a_mismatch(
    raw: str | None, matches: bool, found_label: str
) -> None:
    """Only the exact integer the code expects matches; anything else is a named mismatch."""
    state = data_format_state(raw)
    assert state.matches is matches
    assert state.found_label == found_label
    assert state.expected == DATA_FORMAT_VERSION
    assert state.raw == raw


def test_a_cache_with_no_meta_table_reads_as_a_mismatch(tmp_path: Path) -> None:
    """A DuckDB file with no cache_meta table is a mismatch with no raw value, never a crash."""
    db_path = tmp_path / "no_meta.duckdb"
    duckdb.connect(str(db_path)).close()
    conn = open_cache_readonly(db_path)
    try:
        state = read_data_format(conn)
    finally:
        conn.close()
    assert state.matches is False
    assert state.raw is None

    # And a cache_meta table without the key reads the same way.
    keyless = tmp_path / "keyless.duckdb"
    writer = duckdb.connect(str(keyless))
    writer.execute(CACHE_META_SCHEMA)
    writer.close()
    conn = open_cache_readonly(keyless)
    try:
        assert read_data_format(conn).found_label == "missing"
    finally:
        conn.close()
