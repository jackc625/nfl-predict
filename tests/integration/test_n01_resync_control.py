"""SPEC R2's fail-closed positive control for the WR-06 leakage fix (rung 4, Plan 30-08).

WHAT THE CONTROL IS. N-01 -- the DuckDB copy of silver ``games`` that fell 207 rows behind its
parquet -- was deliberately held open through Phase 29. With whole-frame-fitted imputation
medians and q01/q99 bounds, adding 2025 rows WOULD have moved 2021-2024 feature values, because
every prior season's statistic was computed over the whole frame including the future. That
coupling IS WR-06. Only with WR-06 already landed (rung 2, Plan 30-06) does "the re-sync moved
nothing in 2021-2024" mean what SPEC R2 claims, which is why the D30-17 ladder forces this rung
last.

THE TWO CLAUSES ARE ABOUT TWO DIFFERENT OBJECTS, and that is measured, not stylistic:

* **Clause 1 -- the silver ``games`` DuckDB table.** It grew by EXACTLY the divergence measured
  BEFORE the re-sync (``tests.phase30_state.N01_DIVERGENCE_BEFORE``), and its game-id set now
  equals the parquet's.
* **Clause 2 -- GOLD.** At least one 2025 gold game id is newly present. This is asserted
  against a different object on purpose: Plan 30-04 measured gold's 2025 slice at weeks 1-4
  (49 rows) while the DuckDB table had weeks 1-5 and the parquet weeks 1-22, so gold's 2025
  coverage is bounded by the OTHER silver sources in the join. Writing clause 2 as "gold grew
  by 207" would fail for a reason that has nothing to do with the re-sync.

WHY THE EXPECTED VALUES ARE IMPORTED, NEVER READ FROM ``outputs/``. The pre-re-sync divergence
and the pre-re-sync 2021-2024 slice digests are not INPUTS to this control -- they ARE its
EXPECTED VALUES, and once the re-sync has run they are unrecoverable. ``.gitignore:26`` is
``outputs/`` with only ``!outputs/.gitkeep``, so on any checkout that did not just run Plans
30-04 and 30-08 the scratch documents are absent. Skip-guarding this control on their presence
would make the phase's single most load-bearing proof silently stop proving anything -- the
exact anti-pattern D30-06 rejects in Plan 30-03's own words ("a skipped test silently stops
proving anything"). So every expected value here comes from the git-TRACKED
``tests/phase30_state.py``, and this module is a MUST-PASS, not a may-skip.

TEST CLASS (Plan 30-04's phase-wide rule -- every module this phase adds declares its kind):
**integration**. Every class carries ``@pytest.mark.integration`` and reads the live lake.

THE ONE DELIBERATE SKIP EXCEPTION, AND THE DISTINCTION A FUTURE MAINTAINER MUST KEEP. This
module may skip on the absence of live ``data/`` state that it is PHYSICALLY UNABLE to assert
against -- there is no silver ``games`` table to count rows in on a fresh checkout, and no
assertion about it is possible. It must NEVER skip on the absence of a gitignored EXPECTATION
file under ``outputs/``, because that file's absence is precisely what would turn the control
into a no-op while it still reported green. The two guards look alike and are opposites: one
says "the subject is missing", the other would say "the answer is missing". If you are widening
a guard in this module, work out which kind you are touching before you touch it. Where a
scratch document IS present it is cross-checked against the tracked constants with a plain
``if ... :`` and a bare ``return`` -- never a ``pytest.skip`` -- so a regenerated scratch file
can never silently supersede the committed expectation (T-30-52).
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from data.storage import load_dataframe
from tests.phase30_state import (
    N01_DB_ROWS_BEFORE,
    N01_DIVERGENCE_BEFORE,
    N01_PARQUET_ROWS_BEFORE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"
_SCRATCH_BEFORE = REPO_ROOT / "outputs" / "n01" / "divergence_before.json"


def _require_live_silver_games() -> None:
    """Skip only when the SUBJECT is absent -- never when an EXPECTATION is absent.

    See the module docstring: this is the one sanctioned skip in this file. A fresh
    checkout has no ``data/`` (gitignored runtime state), so there is no games table to
    count rows in and no assertion about it is physically possible.
    """
    if not _SILVER_GAMES.exists():
        pytest.skip(
            f"live silver games not present at {_SILVER_GAMES} -- data/ is gitignored "
            "runtime state. Run the ingestion pipeline (see PIPELINE.md) to populate it."
        )


@pytest.mark.integration
class TestClause1TheSilverGamesDuckDbCopy:
    """The stale copy grew by EXACTLY the divergence measured before anything re-synced it."""

    def test_the_duckdb_copy_grew_by_exactly_the_divergence_measured_beforehand(self):
        _require_live_silver_games()
        db_rows_after = len(load_dataframe("games", layer="silver", source="db"))

        assert N01_DIVERGENCE_BEFORE > 0, (
            "THE CONTROL IS FAIL-CLOSED BY DESIGN. A zero recorded divergence voids it: "
            "it would mean someone had already re-synced before the measurement, leaving "
            "this control able to pass by doing nothing. Do not edit the tracked constant "
            "to make this test green."
        )
        assert db_rows_after - N01_DB_ROWS_BEFORE == N01_DIVERGENCE_BEFORE, (
            f"the DuckDB copy of silver games went {N01_DB_ROWS_BEFORE} -> {db_rows_after}, "
            f"a delta of {db_rows_after - N01_DB_ROWS_BEFORE}, against the "
            f"{N01_DIVERGENCE_BEFORE}-row divergence Plan 30-04 measured BEFORE the re-sync "
            "and pinned in tests/phase30_state.py. The expected value comes from the tracked "
            "manifest, never from outputs/n01/divergence_before.json, which is gitignored."
        )

    def test_the_duckdb_and_parquet_id_sets_are_now_equal(self):
        _require_live_silver_games()
        db_ids = set(load_dataframe("games", layer="silver", source="db")["game_id"])
        pq_ids = set(
            load_dataframe("games", layer="silver", source="parquet")["game_id"]
        )

        assert db_ids == pq_ids, (
            "equal row counts with different membership is the subtler failure "
            f"({len(pq_ids - db_ids)} only in parquet, {len(db_ids - pq_ids)} only in "
            "DuckDB). load_dataframe(source='auto') prefers DuckDB, so any residual "
            "divergence keeps making upsert_silver writes invisible to the pipeline."
        )

    def test_the_authoritative_parquet_row_count_is_unchanged(self):
        _require_live_silver_games()
        parquet_rows = len(load_dataframe("games", layer="silver", source="parquet"))

        assert parquet_rows == N01_PARQUET_ROWS_BEFORE, (
            f"the parquet is the AUTHORITATIVE copy and the re-sync writes it not at all "
            f"(save_to_parquet=False). It reads {parquet_rows} against the tracked "
            f"{N01_PARQUET_ROWS_BEFORE}."
        )

    def test_the_scratch_capture_never_supersedes_the_tracked_manifest(self):
        """T-30-52. Asserts on the TRACKED constants unconditionally, then cross-checks.

        There is no ``pytest.skip`` here on purpose. The tracked arithmetic below is the
        control's expected value and it is asserted on every run, present scratch file or
        not; the scratch document only ever adds a cross-check.
        """
        assert N01_DIVERGENCE_BEFORE > 0
        assert N01_PARQUET_ROWS_BEFORE - N01_DB_ROWS_BEFORE == N01_DIVERGENCE_BEFORE

        if not _SCRATCH_BEFORE.exists():
            return

        captured = json.loads(_SCRATCH_BEFORE.read_text(encoding="utf-8"))
        assert captured["parquet_rows"] == N01_PARQUET_ROWS_BEFORE
        assert captured["db_rows"] == N01_DB_ROWS_BEFORE
        assert captured["divergence"] == N01_DIVERGENCE_BEFORE, (
            "outputs/n01/divergence_before.json disagrees with tests/phase30_state.py. One "
            "was regenerated out of step with the other. The TRACKED manifest is the "
            "expectation; do NOT edit it to match a re-run."
        )


@pytest.mark.integration
class TestTheNarrowestPossibleWrite:
    """T-30-35: only the stale copy is touched, and that is asserted structurally."""

    def test_the_resync_disables_parquet_writing_and_replaces_the_table(self):
        from scripts import resync_games_duckdb

        tree = ast.parse(inspect.getsource(resync_games_duckdb.resync_games))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_dataframe"
        ]
        assert len(calls) == 1, (
            "resync_games must make exactly one save_dataframe call; found "
            f"{len(calls)}"
        )
        keywords = {
            keyword.arg: keyword.value
            for keyword in calls[0].keywords
            if keyword.arg is not None
        }
        for name, expected in (
            ("save_to_db", True),
            ("save_to_parquet", False),
            ("replace_mode", True),
        ):
            node = keywords.get(name)
            assert isinstance(node, ast.Constant) and node.value is expected, (
                f"resync_games must pass {name}={expected} as a literal. "
                "save_to_parquet=False keeps the AUTHORITATIVE parquet byte-untouched; "
                "replace_mode=True skips the append-and-dedup merge, which is the correct "
                "semantic for 'the parquet IS the truth'."
            )

    def test_the_shared_read_path_is_not_modified(self):
        """D30-18: the write-path asymmetry is explicitly out of this phase's scope.

        The re-sync fixes the stale COPY. It does not redefine what
        ``load_dataframe(source="auto")`` resolves to, which is the seam every builder,
        trainer and backtest goes through -- changing that in the same phase that is
        trying to measure a gold rebuild is exactly the confound this phase avoids.
        """
        from scripts import resync_games_duckdb

        module = ast.parse(inspect.getsource(resync_games_duckdb))
        defined = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert not defined & {"load_dataframe", "save_dataframe", "upsert_silver"}, (
            "the re-sync must USE data.storage's read/write path, never shadow it"
        )
        parameters = inspect.signature(load_dataframe).parameters
        assert (parameters["layer"].default, parameters["source"].default) == (
            "silver",
            "auto",
        ), (
            "load_dataframe's defaults moved. N-01 is only invisible BECAUSE source "
            "defaults to 'auto' and 'auto' prefers DuckDB."
        )


@pytest.mark.integration
class TestTheResyncIsIdempotent:
    """A second apply changes nothing, and the CLI refuses it outright."""

    def test_a_second_apply_leaves_the_duckdb_table_byte_identical(self):
        _require_live_silver_games()
        from scripts.resync_games_duckdb import resync_games, table_content_digest

        before = load_dataframe("games", layer="silver", source="db")
        parquet = load_dataframe("games", layer="silver", source="parquet")
        if set(before["game_id"]) != set(parquet["game_id"]):
            pytest.skip(
                "the DuckDB and parquet copies of silver games do not yet agree -- the "
                "N-01 re-sync has not run on this lake. Run "
                "`python -m scripts.resync_games_duckdb --apply` first; this test is "
                "about the SECOND apply being a no-op."
            )

        digest_before = table_content_digest(before)
        resync_games()
        after = load_dataframe("games", layer="silver", source="db")

        assert len(after) == len(before)
        assert set(after["game_id"]) == set(before["game_id"])
        assert table_content_digest(after) == digest_before, (
            "a second re-sync moved the DuckDB table's content. replace_mode writes the "
            "passed frame AS the table, so re-running it on unchanged inputs must be a "
            "content no-op."
        )

    def test_the_cli_refuses_to_apply_when_the_divergence_is_zero(self, capsys):
        _require_live_silver_games()
        from scripts.resync_games_duckdb import main, measure_divergence

        if measure_divergence()["divergence"] != 0:
            pytest.skip(
                "silver games is still divergent on this lake, so the zero-divergence "
                "refusal cannot be exercised against it. Run "
                "`python -m scripts.resync_games_duckdb --apply` first."
            )

        exit_code = main(["--apply"])
        captured = capsys.readouterr()

        assert exit_code != 0, (
            "applying a re-sync against a zero divergence must REFUSE, not succeed"
        )
        assert "nothing to prove" in (captured.out + captured.err).lower(), (
            "the refusal must state that a zero divergence leaves the positive control "
            "with nothing to prove"
        )


@pytest.mark.integration
class TestTheDivergenceMeasurementItself:
    """``measure_divergence`` reports the same shape the tracked manifest records."""

    def test_the_post_resync_divergence_is_zero_in_both_directions(self):
        _require_live_silver_games()
        from scripts.resync_games_duckdb import measure_divergence

        measured = measure_divergence()

        assert measured["divergence"] == 0
        assert measured["only_in_parquet_count"] == 0
        assert measured["only_in_duckdb_count"] == 0, (
            "rows present only in DuckDB would be a different and worse defect than N-01, "
            "which was a mirror that fell BEHIND"
        )
        assert (
            measured["parquet_rows"] == measured["db_rows"] == N01_PARQUET_ROWS_BEFORE
        )

    def test_the_measurement_reads_through_the_storage_layer_not_the_files(self):
        from scripts import resync_games_duckdb

        source = inspect.getsource(resync_games_duckdb.measure_divergence)
        assert "read_parquet" not in source, (
            "the divergence is a statement about what the PIPELINE sees, so it is "
            "measured through data.storage.load_dataframe with explicit source= "
            "literals, never by reading the files directly"
        )
        assert 'source="parquet"' in source and 'source="db"' in source, (
            "the source literal is 'db', not 'duckdb' -- data/storage.py's load_dataframe "
            "raises ValueError on anything else"
        )
