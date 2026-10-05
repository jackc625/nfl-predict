"""Integration tests for ingestion pipeline idempotency.

These tests verify:
- Bronze snapshots are append-only (each call creates a new file, old files untouched)
- Silver upsert is idempotent (re-running with same data produces identical output)
- Silver upsert correctly adds new rows and replaces updated rows
- get_latest_bronze_file returns the most recent snapshot
- SPEC R1 (Plan 30-08): a full GOLD rebuild on unchanged inputs reproduces gold
- SPEC R9 (Plan 31-18): a second cache population on the same week reproduces the bet list's
  recommendation facts BYTE-IDENTICALLY, the durable artifact is genuinely the source those
  rows survive a rebuild from, and the grading half can still transition exactly once

The Bronze and Silver classes use tmp_path for complete isolation -- no real data files
required. ``TestAFullGoldRebuildIsReproducible`` is the one exception: it judges the two
fingerprint documents a pair of full-history rebuilds produced, and skips with a
remediation command when those gitignored run records are absent. Its own docstring says
why the comparison names the per-build clock instead of hiding it.
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from data.storage import get_latest_bronze_file, save_bronze_snapshot, upsert_silver

# -- Helpers -------------------------------------------------------------------


def _make_games_df(game_ids: list[str], home_score: int = 21) -> pd.DataFrame:
    """Create a minimal games DataFrame suitable for storage tests."""
    records = []
    for gid in game_ids:
        records.append(
            {
                "game_id": gid,
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": home_score,
                "away_score": 17,
            }
        )
    return pd.DataFrame(records)


# -- Tests ---------------------------------------------------------------------


@pytest.mark.integration
class TestBronzeAppendOnly:
    """Verify Bronze layer append-only semantics."""

    def test_save_bronze_twice_creates_two_files(self, tmp_path):
        """Calling save_bronze_snapshot twice produces two distinct files."""
        df = _make_games_df(["2024_W01_BUF@KC"])

        path1 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)
        # Brief pause so timestamp differs (filename includes seconds)
        time.sleep(1.1)
        path2 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)

        assert path1.exists(), f"First Bronze file should exist: {path1}"
        assert path2.exists(), f"Second Bronze file should exist: {path2}"
        assert path1 != path2, "Two snapshots should produce distinct file paths"

        # Both files should have the same content
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_bronze_files_never_modified_after_creation(self, tmp_path):
        """Existing Bronze files are never overwritten or modified."""
        df1 = _make_games_df(["2024_W01_BUF@KC"], home_score=21)

        path1 = save_bronze_snapshot(df1, "games", 2024, 1, base_path=tmp_path)
        mtime1 = path1.stat().st_mtime

        # Wait and create a second snapshot with different data
        time.sleep(1.1)
        df2 = _make_games_df(["2024_W01_BUF@KC"], home_score=35)
        path2 = save_bronze_snapshot(df2, "games", 2024, 1, base_path=tmp_path)

        # First file should not have been modified
        assert path1.stat().st_mtime == mtime1, (
            "First Bronze file modification time should be unchanged"
        )
        assert path1 != path2, "Second call should create a different file"

        # Verify two distinct files exist
        bronze_dir = tmp_path / "bronze"
        all_files = list(bronze_dir.glob("games_raw_bronze_2024_W01_*.parquet"))
        assert len(all_files) == 2, f"Expected 2 Bronze files, found {len(all_files)}"

    def test_get_latest_bronze_file_returns_newest(self, tmp_path):
        """get_latest_bronze_file returns the most recently created file."""
        df = _make_games_df(["2024_W01_BUF@KC"])

        save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)
        time.sleep(1.1)
        path2 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)

        latest = get_latest_bronze_file("games", 2024, 1, base_path=tmp_path)
        assert latest is not None, "Should find at least one Bronze file"
        assert latest == path2, (
            f"Expected latest file to be {path2.name}, got {latest.name}"
        )


@pytest.mark.integration
class TestSilverUpsert:
    """Verify Silver layer idempotent upsert semantics."""

    def test_upsert_silver_idempotent(self, tmp_path):
        """Upserting the same data twice produces the same number of rows."""
        df = _make_games_df(
            ["g1", "g2", "g3", "g4", "g5"],
            home_score=21,
        )

        upsert_silver(df, "games", base_path=tmp_path)
        upsert_silver(df, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 5, (
            f"Expected 5 games after idempotent upsert, got {len(result)}"
        )

    def test_upsert_silver_adds_new_games(self, tmp_path):
        """Upserting new game_ids appends them to existing data."""
        df_a = _make_games_df(["g1", "g2", "g3"])
        upsert_silver(df_a, "games", base_path=tmp_path)

        df_b = _make_games_df(["g4", "g5"])
        upsert_silver(df_b, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 5, (
            f"Expected 5 games after adding new games, got {len(result)}"
        )
        result_ids = set(result["game_id"])
        assert result_ids == {"g1", "g2", "g3", "g4", "g5"}, (
            f"Expected all 5 game_ids, got {result_ids}"
        )

    def test_upsert_silver_replaces_updated_games(self, tmp_path):
        """Upserting an existing game_id with new data replaces the old row."""
        df_a = _make_games_df(["g1"], home_score=21)
        upsert_silver(df_a, "games", base_path=tmp_path)

        df_b = _make_games_df(["g1"], home_score=28)
        upsert_silver(df_b, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 1, f"Expected 1 game after replace, got {len(result)}"
        assert result.iloc[0]["home_score"] == 28, (
            f"Expected updated home_score=28, got {result.iloc[0]['home_score']}"
        )


@pytest.mark.integration
class TestAFullGoldRebuildIsReproducible:
    """SPEC R1: re-running the rebuild on unchanged inputs reproduces gold exactly.

    THIS IS ONLY POSSIBLE BECAUSE OF THE ``replace_mode=True`` GOLD WRITE landed at rung
    3 (Plan 30-07). Under the previous append path ``save_dataframe`` merged the incoming
    frame with the existing table through ``pd.concat``, so a rebuild could neither
    narrow a schema nor be relied on to reproduce one: dropped columns returned as
    all-NaN and an ``int32`` column was silently upcast to ``float64`` by the
    concatenation. Replace mode writes the passed frame AS the table, which is what makes
    a byte comparison across two runs mean anything at all. The dependency is stated here
    so it is visible rather than inferred.

    THE ONE COLUMN THAT MUST DIFFER, AND WHY THE COMPARISON NAMES IT RATHER THAN HIDING
    IT. ``scripts/build_features.py`` stamps ``feature_timestamp = datetime.now(UTC)``
    once per build, so it takes exactly one distinct value per build and MUST move on
    every rebuild by construction. Plan 30-18 established it as its own ``build_clock``
    category in the rung judge for precisely this reason (D30-OWNER-08): counting a clock
    as a moved VALUE made rung 3's empty-changed-set criterion structurally
    unsatisfiable, and a criterion that cannot be satisfied has stopped discriminating
    between a right rebuild and a wrong one.

    The same argument applies to a literal byte comparison of two fingerprint documents,
    so this asserts the stronger and actually meaningful claim in two halves: every
    column EXCEPT the named build clock reproduces its per-season digest EXACTLY in every
    season of all three matrices, AND the set of entries that do differ is asserted to be
    EXACTLY the clock's -- so nothing else can hide behind the exclusion.

    The rebuilds themselves are operator actions: two full-history
    ``python -m scripts.build_features`` runs on unchanged inputs, with their timestamps,
    durations and gold digests recorded in the Plan 30-08 SUMMARY. This test is the
    committed judge of the two documents they produced, in the same shape as every other
    fingerprint assertion in Phase 30.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _RUNG4 = _REPO_ROOT / "outputs" / "fingerprints" / "rung4.json"
    _RERUN = _REPO_ROOT / "outputs" / "fingerprints" / "rung4_rerun.json"
    _BUILD_CLOCK = "feature_timestamp"

    def _documents(self) -> tuple[dict, dict]:
        missing = [
            str(path) for path in (self._RUNG4, self._RERUN) if not path.exists()
        ]
        if missing:
            pytest.skip(
                f"fingerprint document(s) absent: {', '.join(missing)}. These are "
                "gitignored run records (.gitignore:26). Reproduce with two consecutive "
                "`python -m scripts.build_features` runs, fingerprinting after each via "
                "`python -m scripts.fingerprint_gold --out outputs/fingerprints/<name>.json`."
            )
        return (
            json.loads(self._RUNG4.read_text(encoding="utf-8")),
            json.loads(self._RERUN.read_text(encoding="utf-8")),
        )

    def test_the_rerun_reproduces_every_non_clock_column_exactly(self):
        first, second = self._documents()

        assert set(first) == set(second)
        for matrix in sorted(first):
            before, after = first[matrix], second[matrix]
            assert after["rows"] == before["rows"], (
                f"{matrix}: a re-run on unchanged inputs changed the row count "
                f"{before['rows']} -> {after['rows']}"
            )
            assert after["width"] == before["width"], (
                f"{matrix}: a re-run on unchanged inputs changed the width "
                f"{before['width']} -> {after['width']}"
            )
            assert after["rows_per_season"] == before["rows_per_season"]

            moved = [
                column
                for column in sorted(before["columns"])
                if column != self._BUILD_CLOCK
                and before["columns"][column] != after["columns"].get(column)
            ]
            assert not moved, (
                f"{matrix}: {len(moved)} column(s) did not reproduce across two builds "
                f"on unchanged inputs -- {moved[:10]}. SPEC R1 requires a rebuild to be "
                "reproducible; a column that moves without an input moving means the "
                "build carries hidden state or a non-deterministic fit."
            )

    def test_the_only_entries_that_differ_are_the_build_clocks(self):
        """Nothing may hide behind the clock exclusion, so the diff set is pinned."""
        first, second = self._documents()

        differing = {
            (matrix, column)
            for matrix in first
            for column in first[matrix]["columns"]
            if first[matrix]["columns"][column] != second[matrix]["columns"].get(column)
        }
        assert differing == {(matrix, self._BUILD_CLOCK) for matrix in first}, (
            "the set of entries differing between two builds on unchanged inputs is "
            f"{sorted(differing)}. It must be EXACTLY the per-build clock in each "
            "matrix and nothing else."
        )

    def test_the_build_clock_did_move_so_the_comparison_is_not_vacuous(self):
        """If the clock did NOT move, the two documents describe the SAME build."""
        first, second = self._documents()

        for matrix in sorted(first):
            assert (
                first[matrix]["columns"][self._BUILD_CLOCK]
                != second[matrix]["columns"][self._BUILD_CLOCK]
            ), (
                f"{matrix}'s build clock is identical across the two documents, which "
                "means they were fingerprinted from the SAME build. A re-run comparison "
                "against itself proves nothing -- re-run the build."
            )


# -- SPEC R9: the bet list across repeated cache populations --------------------
#
# THE PROPERTY, AND WHY IT IS NOT AUTOMATIC. ``api.cache.populate_cache`` REPLACES the whole
# database: it builds a temp DuckDB and renames it over the destination. So the bet list a Friday
# run publishes does not persist because the cache kept it -- it persists because the population
# RELOADS it from the durable ``outputs/bet_list/`` artifact into the temp build. That is the
# claim ``TestTheDurableArtifactIsGenuinelyTheSource`` pins, in both directions: the rows come
# back after a rebuild, and they STOP coming back when the artifact is removed.
#
# WHY THE BYTE-IDENTITY IS OVER THE IMMUTABLE HALF ONLY (REVIEW-FWD-GRADE). A forward row is the
# record of what was recommended before kickoff, so its recommendation facts must never move. Its
# GRADING columns must move exactly once, when the result exists -- a whole-row immutability rule
# would leave every forward row ``pending`` forever, and a permanently zero-graded forward tracker
# is indistinguishable to a reader from a system that recommended nothing that won. The two halves
# are asserted TOGETHER in ``test_a_run_a_freeze_a_grading_pass_and_a_rerun_...`` because the pair
# is the actual contract.

import hashlib
from datetime import UTC, datetime

import duckdb

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_WIN,
    bet_list_populated_at_key,
    populate_cache,
)
from backtest.weekly_bet_list import (
    grade_row,
    read_bet_list_cache_sources,
    upsert_bet_list_rows,
    write_bet_list_artifact,
)

_BL_SEASON = 2023
_BL_WEEK = 1
_BL_GAME = "2023_W01_DET@KC"
# The game's own Friday-6PM-ET freeze for a Sunday 2023-09-10 kickoff.
_BL_FREEZE = datetime(2023, 9, 8, 22, 0, 0, tzinfo=UTC)
_BEFORE_FREEZE = datetime(2023, 9, 7, 12, 0, 0, tzinfo=UTC)
_AFTER_FREEZE = datetime(2023, 9, 11, 12, 0, 0, tzinfo=UTC)
# The row's OWN observation time (Phase 33, Plan 33-05 Task 3). It is ``_BEFORE_FREEZE``, the same
# instant the pre-freeze merge below is judged at, because a forward row's observation time and
# the fence it is judged against are one clock read in production. A forward row carrying no
# stamp is now refused by name at the upsert, so the fixture has to carry one.
_BL_DECIDED_AT = _BEFORE_FREEZE.isoformat()


def _bet_row(**overrides) -> dict:
    """One COMPLETE forward bet-list row. Every locked column present; nothing invented."""
    row = {
        "game_id": _BL_GAME,
        "season": _BL_SEASON,
        "week": _BL_WEEK,
        "target": "ou",
        "bet_side": "under",
        "model_value": 41.0,
        "market_value": 45.5,
        "line": 45.5,
        "slipped_line": 45.5,
        "calibrated_p_side": 0.5612345678901234,
        "per_bet_ev": 0.07123456789012345,
        "stake_units": 1.25,
        "ev_tier": "high",
        "status": "live",
        "rejection_reason": None,
        "eligibility_label": "UNDER pick",
        "snapshot_ts": "2023-09-08T18:00:00-04:00",
        "freeze_ts": _BL_FREEZE.isoformat(),
        "selected_odds": -110.0,
        "flat_stake": 1.0,
        "provenance": "forward",
        "validation_type": "forward_realized",
        "decided_at_utc": _BL_DECIDED_AT,
        "grading_status": GRADING_STATUS_PENDING,
        "outcome": None,
        "clv": 0.0234567890123,
        "payout_flat": None,
        "realized_units": None,
        "graded_at": None,
    }
    # Phase 34 (Plan 34-01 Task 2) widened the locked schema from 29 to 51. Was: the literal above
    # was the whole width. The 22 Phase-34 columns are NULL, as on every row the pre-ledger writer
    # produced; the drift check below still proves the row spans the locked width exactly.
    for column in BET_LIST_COLUMNS:
        row.setdefault(column, None)
    row.update(overrides)
    assert set(row) == set(BET_LIST_COLUMNS), (
        f"the fixture row drifted from BET_LIST_COLUMNS: {set(row) ^ set(BET_LIST_COLUMNS)}"
    )
    return row


def _silver_with_one_week(silver_dir: Path) -> Path:
    """A one-game silver schedule, so ``build_bet_week_schedule`` runs for real and hermetically."""
    silver_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_id": _BL_GAME,
                "season": _BL_SEASON,
                "week": _BL_WEEK,
                "kickoff_et": pd.Timestamp("2023-09-10T17:00:00Z"),
            }
        ]
    ).to_parquet(silver_dir / "games.parquet", index=False)
    return silver_dir


def _populate(db_path: Path, bet_dir: Path, silver_dir: Path, empty: Path) -> None:
    """Run the FULL production chain: the shared source reader, then ``populate_cache``.

    Deliberately not a hand-assembled call. The reader is the seam ``pipeline/steps.py`` and
    ``scripts/populate_cache.py`` both use, so a change that broke the artifact round trip for the
    real callers breaks it here too.
    """
    sources = read_bet_list_cache_sources(bet_dir, silver_dir)
    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
        bet_list_df=sources.bet_list,
        bet_tracker_df=sources.tracker,
        bet_schedule_df=sources.schedule,
    )


def _served_rows(db_path: Path) -> list[dict]:
    """Every bet_list row as served from the built cache, in the locked column order."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        columns = ", ".join(BET_LIST_COLUMNS)
        result = conn.execute(
            f"SELECT {columns} FROM bet_list ORDER BY season, week, game_id, target"
        )
        names = [d[0] for d in result.description]
        return [dict(zip(names, row, strict=True)) for row in result.fetchall()]
    finally:
        conn.close()


def _marker(db_path: Path, season: int, week: int) -> str | None:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE key = ?",
            [bet_list_populated_at_key(season, week)],
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _immutable_digest(rows: list[dict]) -> str:
    """A sha256 over the served IMMUTABLE half, serialised with full float precision.

    ``repr`` on a float is round-trip exact in Python 3, so this digest moves on a one-ulp change.
    A tolerance-based comparison would not, and "byte-identical" is the claim being made.
    """
    payload = "\n".join(
        "|".join(f"{column}={row[column]!r}" for column in BET_LIST_IMMUTABLE_COLUMNS)
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.mark.integration
class TestTheDurableArtifactIsGenuinelyTheSource:
    """REVIEW-CACHE: forward rows survive a rebuild because they are RELOADED, not preserved."""

    def test_a_forward_row_survives_a_full_cache_rebuild(self, tmp_path):
        """Written once to the artifact, present after a population that replaced the whole DB."""
        bet_dir = tmp_path / "bet_list"
        empty = tmp_path / "empty"
        empty.mkdir()
        silver = _silver_with_one_week(tmp_path / "silver")
        write_bet_list_artifact(pd.DataFrame([_bet_row()]), bet_dir)

        db_path = tmp_path / "web_cache.duckdb"
        _populate(db_path, bet_dir, silver, empty)

        rows = _served_rows(db_path)
        assert len(rows) == 1, f"the forward row did not survive the rebuild: {rows}"
        assert rows[0]["game_id"] == _BL_GAME
        assert _marker(db_path, _BL_SEASON, _BL_WEEK) is not None

        # And again, over the SAME artifact: a rebuild is not a one-shot survival.
        _populate(db_path, bet_dir, silver, empty)
        assert len(_served_rows(db_path)) == 1

    def test_removing_the_artifact_empties_the_rebuilt_cache(self, tmp_path):
        """The other direction, which is what makes the claim above non-vacuous.

        If the rows came from anywhere else -- a preserved table, a re-derivation -- deleting the
        artifact would leave them in place. They vanish, and the marker vanishes with them, which
        is the state the ``/bets`` hard-block refuses on rather than rendering as an empty week.
        """
        bet_dir = tmp_path / "bet_list"
        empty = tmp_path / "empty"
        empty.mkdir()
        silver = _silver_with_one_week(tmp_path / "silver")
        artifact = write_bet_list_artifact(pd.DataFrame([_bet_row()]), bet_dir)

        db_path = tmp_path / "web_cache.duckdb"
        _populate(db_path, bet_dir, silver, empty)
        assert len(_served_rows(db_path)) == 1

        artifact.unlink()
        _populate(db_path, bet_dir, silver, empty)

        assert _served_rows(db_path) == [], (
            "rows survived a rebuild with the durable artifact deleted, so they are not actually "
            "being reloaded from it and the durability claim rests on something unproven"
        )
        assert _marker(db_path, _BL_SEASON, _BL_WEEK) is None, (
            "the populated-at marker survived a population that inserted no rows; the hard-block "
            "would read a bet-list-less cache as fresh"
        )
        # The schedule-derived freeze is still there, which is exactly what lets the page refuse
        # this state rather than render it as a week in which nothing was recommended.
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            assert (
                conn.execute("SELECT COUNT(*) FROM bet_week_freeze").fetchone()[0] == 1
            )
        finally:
            conn.close()


@pytest.mark.integration
class TestASecondPopulationReproducesTheRecommendationFacts:
    """SPEC R9 idempotency, asserted with digests across two runs rather than by inspection."""

    def test_two_populations_of_the_same_week_are_byte_identical(self, tmp_path):
        bet_dir = tmp_path / "bet_list"
        empty = tmp_path / "empty"
        empty.mkdir()
        silver = _silver_with_one_week(tmp_path / "silver")
        write_bet_list_artifact(pd.DataFrame([_bet_row()]), bet_dir)

        first = tmp_path / "first.duckdb"
        second = tmp_path / "second.duckdb"
        _populate(first, bet_dir, silver, empty)
        _populate(second, bet_dir, silver, empty)

        before = _served_rows(first)
        after = _served_rows(second)
        assert _immutable_digest(before) == _immutable_digest(after)

        # Column by column as well, so a failure names the field rather than a hash.
        for column in BET_LIST_IMMUTABLE_COLUMNS:
            assert [r[column] for r in before] == [r[column] for r in after], (
                f"{column} moved between two populations of the same week"
            )

    def test_the_digest_is_not_vacuous(self, tmp_path):
        """A changed recommendation fact MUST move the digest, or the comparison proves nothing."""
        base = [_bet_row()]
        moved = [_bet_row(stake_units=1.26)]
        assert _immutable_digest(base) != _immutable_digest(moved)
        # And a change confined to the GRADING half must NOT move it -- that is the split.
        graded = [_bet_row(grading_status=GRADING_STATUS_WIN, outcome=True)]
        assert _immutable_digest(base) == _immutable_digest(graded)


@pytest.mark.integration
class TestTheFreezeCrossingKeepsTheTwoHalvesTogether:
    """REVIEW-FWD-GRADE: the recommendation facts freeze while the grading half stays writable."""

    def test_a_run_a_freeze_a_grading_pass_and_a_rerun_hold_both_halves(self, tmp_path):
        """One week, four states, one assertion set. The pair IS the contract.

        1. selected before the freeze and populated;
        2. re-selected AFTER the freeze with DIFFERENT numbers -- the stored row wins;
        3. graded -- the grading half transitions exactly once out of ``pending``;
        4. re-selected again after the freeze -- still frozen, still graded.

        Every ``BET_LIST_IMMUTABLE_COLUMNS`` value is asserted identical across all four, and the
        grading transition is asserted to happen exactly once.
        """
        bet_dir = tmp_path / "bet_list"
        empty = tmp_path / "empty"
        empty.mkdir()
        silver = _silver_with_one_week(tmp_path / "silver")
        db_path = tmp_path / "web_cache.duckdb"

        # 1. Selected before the freeze.
        stored = pd.DataFrame([_bet_row()])
        write_bet_list_artifact(stored, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        state_one = _served_rows(db_path)

        # 2. A later run AFTER the freeze proposes different recommendation facts.
        rewritten = pd.DataFrame(
            [_bet_row(stake_units=9.99, per_bet_ev=0.99, line=99.5, clv=0.99)]
        )
        merged = upsert_bet_list_rows(stored, rewritten, now=_AFTER_FREEZE)
        write_bet_list_artifact(merged, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        state_two = _served_rows(db_path)

        # 3. The grading pass settles it, one way and one time.
        graded_row = grade_row(
            merged.to_dict("records")[0], True, graded_at=_AFTER_FREEZE
        )
        graded = pd.DataFrame([graded_row], columns=pd.Index(BET_LIST_COLUMNS))
        write_bet_list_artifact(graded, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        state_three = _served_rows(db_path)

        # 4. One more post-freeze run, over the settled row.
        merged_again = upsert_bet_list_rows(graded, rewritten, now=_AFTER_FREEZE)
        write_bet_list_artifact(merged_again, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        state_four = _served_rows(db_path)

        states = [state_one, state_two, state_three, state_four]
        for label, state in zip(("one", "two", "three", "four"), states, strict=True):
            assert len(state) == 1, f"state {label} served {len(state)} rows"

        # The immutable half NEVER moved, across all four states, column by column.
        for column in BET_LIST_IMMUTABLE_COLUMNS:
            values = [state[0][column] for state in states]
            assert len(set(values)) == 1, (
                f"the immutable column {column!r} moved across the freeze: {values}. A forward row "
                "is the record of what was recommended before kickoff; a later run may not restate "
                "it."
            )

        # The grading half transitioned EXACTLY once, out of pending, and stayed there.
        statuses = [state[0]["grading_status"] for state in states]
        assert statuses == [
            GRADING_STATUS_PENDING,
            GRADING_STATUS_PENDING,
            GRADING_STATUS_WIN,
            GRADING_STATUS_WIN,
        ], f"the grading status did not transition exactly once: {statuses}"
        assert state_one[0]["graded_at"] is None
        assert state_three[0]["graded_at"] is not None
        assert state_four[0]["graded_at"] == state_three[0]["graded_at"]

        # And the grading half is the ONLY half that moved between states two and three.
        moved = [
            column
            for column in BET_LIST_COLUMNS
            if state_two[0][column] != state_three[0][column]
        ]
        assert set(moved) <= set(BET_LIST_GRADING_COLUMNS), (
            "grading moved a non-grading column: "
            f"{sorted(set(moved) - set(BET_LIST_GRADING_COLUMNS))}"
        )
        assert moved, "the grading pass changed nothing at all"

    def test_a_run_before_the_freeze_is_replaced_and_then_stops_changing(
        self, tmp_path
    ):
        """Before the freeze the latest run WINS -- late odds can still land, so replacing is right.

        The control for the test above: without it, an upsert that discarded every incoming row
        would satisfy the immutability assertions while being plainly wrong.
        """
        bet_dir = tmp_path / "bet_list"
        empty = tmp_path / "empty"
        empty.mkdir()
        silver = _silver_with_one_week(tmp_path / "silver")
        db_path = tmp_path / "web_cache.duckdb"

        stored = pd.DataFrame([_bet_row()])
        write_bet_list_artifact(stored, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        before = _served_rows(db_path)

        late = pd.DataFrame([_bet_row(line=44.5, per_bet_ev=0.09, stake_units=1.5)])
        replaced = upsert_bet_list_rows(stored, late, now=_BEFORE_FREEZE)
        write_bet_list_artifact(replaced, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        after = _served_rows(db_path)

        assert after[0]["line"] == 44.5, (
            "a PRE-freeze re-run did not replace the row; late odds are legitimate before the "
            "freeze and refusing them would publish a stale line as the recommendation"
        )
        assert _immutable_digest(before) != _immutable_digest(after)

        # And now the freeze passes: a further run leaves it alone.
        later = pd.DataFrame([_bet_row(line=1.5, per_bet_ev=0.5, stake_units=5.0)])
        frozen = upsert_bet_list_rows(replaced, later, now=_AFTER_FREEZE)
        write_bet_list_artifact(frozen, bet_dir)
        _populate(db_path, bet_dir, silver, empty)
        settled = _served_rows(db_path)

        assert _immutable_digest(after) == _immutable_digest(settled)
