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

AND THE BYTE-IDENTITY CLAUSE. Every 2021-2024 value in every column of every matrix
reproduces its pre-re-sync digest EXACTLY, with one named exclusion -- the per-build clock
``feature_timestamp``, which is a build artifact rather than a feature value and moves on every
rebuild by construction (D30-OWNER-08). The exclusion is by NAME from the single registry
``scripts.fingerprint_gold.BUILD_CLOCK_COLUMNS``, and it is paired with an assertion that the
moved set is EXACTLY that registry -- so nothing can hide behind it. See
``TestEvery2021To2024ValueIsByteIdentical``.

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

import pandas as pd
import pytest

from data.storage import load_dataframe
from scripts.fingerprint_gold import BUILD_CLOCK_COLUMNS, GOLD_MATRICES
from tests.phase30_state import (
    GOLD_2025_ROWS_BEFORE_RESYNC,
    GOLD_2025_WEEKS_BEFORE_RESYNC,
    N01_DB_ROWS_BEFORE,
    N01_DIVERGENCE_BEFORE,
    N01_PARQUET_ROWS_BEFORE,
    SLICE_DIGESTS_2021_2024,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"
_GOLD_DIR = REPO_ROOT / "data" / "gold"
_SCRATCH_BEFORE = REPO_ROOT / "outputs" / "n01" / "divergence_before.json"
_SCRATCH_SLICE_BEFORE = REPO_ROOT / "outputs" / "n01" / "slice_hash_before.json"


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


def _require_live_gold() -> None:
    """The same one sanctioned skip, for the gold matrices clause 2 reads."""
    missing = [
        matrix
        for matrix in GOLD_MATRICES
        if not (_GOLD_DIR / f"{matrix}.parquet").exists()
    ]
    if missing:
        pytest.skip(
            f"live gold absent ({', '.join(missing)}) -- data/ is gitignored runtime "
            "state. Run `python -m scripts.build_features` to populate data/gold."
        )


@pytest.mark.integration
class TestClause1TheSilverGamesDuckDbCopy:
    """The stale copy grew by EXACTLY the divergence measured before anything re-synced it."""

    def test_the_duckdb_copy_caught_up_and_has_stayed_caught_up(self):
        """RE-ANCHORED (Plan 33.2-20): the delta is no longer the re-sync's alone.

        This asserted ``db_rows_after - N01_DB_ROWS_BEFORE == N01_DIVERGENCE_BEFORE``,
        i.e. that the DuckDB copy grew by EXACTLY the 207-row divergence Plan 30-04
        measured. It measured 479, because the live 2026 capture has since appended 272
        rows to BOTH copies (207 + 272 = 479). The subtraction was only ever a proxy
        for the real claim -- the stale copy caught up and is no longer behind -- and a
        proxy that counts every later write as if it were the re-sync stops being one
        the moment anything else writes.

        So the claim is asserted DIRECTLY: the DuckDB copy is not behind the parquet,
        and the recorded catch-up is still accounted for as arithmetic. The fail-closed
        premise (a zero recorded divergence voids the control) is untouched.
        """
        _require_live_silver_games()
        db_rows_after = len(load_dataframe("games", layer="silver", source="db"))
        parquet_rows = len(load_dataframe("games", layer="silver", source="parquet"))

        assert N01_DIVERGENCE_BEFORE > 0, (
            "THE CONTROL IS FAIL-CLOSED BY DESIGN. A zero recorded divergence voids it: "
            "it would mean someone had already re-synced before the measurement, leaving "
            "this control able to pass by doing nothing. Do not edit the tracked constant "
            "to make this test green."
        )
        assert db_rows_after == parquet_rows, (
            f"the DuckDB copy holds {db_rows_after} rows against the parquet's "
            f"{parquet_rows}. N-01 was a mirror that fell behind; a copy that is behind "
            "again makes every upsert_silver write invisible to the pipeline."
        )
        appended_since = db_rows_after - N01_DB_ROWS_BEFORE - N01_DIVERGENCE_BEFORE
        assert appended_since >= 0, (
            f"the DuckDB copy holds {db_rows_after} rows, FEWER than the "
            f"{N01_DB_ROWS_BEFORE} it held before the re-sync plus the "
            f"{N01_DIVERGENCE_BEFORE} rows the re-sync brought it. A store cannot lose "
            "rows it already had, so this is a copy that went backwards."
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

    def test_the_re_sync_never_moved_the_authoritative_parquet(self):
        """RE-ANCHORED (Plan 33.2-20): the re-sync did not write it; later ingests did.

        This asserted the parquet still holds exactly its pre-re-sync row count. It
        holds more, because the live 2026 capture ingested 272 games into it -- through
        ``upsert_silver``, not through the re-sync, which writes the parquet not at all
        (``save_to_parquet=False``). The claim the assertion was making is that the
        re-sync brought the STALE copy up and left the AUTHORITATIVE one alone, and the
        durable form of that is a FLOOR: the parquet never lost a row to the re-sync.
        A parquet that shrank below the pre-re-sync count would still fail here.
        """
        _require_live_silver_games()
        parquet_rows = len(load_dataframe("games", layer="silver", source="parquet"))

        assert parquet_rows >= N01_PARQUET_ROWS_BEFORE, (
            f"the parquet is the AUTHORITATIVE copy and the re-sync writes it not at all "
            f"(save_to_parquet=False). It reads {parquet_rows} against the tracked "
            f"{N01_PARQUET_ROWS_BEFORE} it held before -- fewer rows than it started "
            "with, which no ingest produces."
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

    # COLD-05 / D33-24. This is the repository's ONE legitimate production writer, and
    # the exemption is scoped to the single path it writes. The write is a positive
    # control on a REAL lake: `resync_games()` replaces the silver `games` table in
    # `data/nfl_predictions.duckdb`, and the test's own assertion is that the TABLE's
    # content digest is unchanged -- a second apply must be a logical no-op. A DuckDB
    # `replace_mode` rewrite nevertheless moves the FILE's bytes, which is what the
    # content guard sees and what this marker declares. Neither
    # `TestEvery2021To2024ValueIsByteIdentical` test carries it: both are deliberate
    # tripwires and keep zero write exemption.
    @pytest.mark.writes_production_store(paths=["data/nfl_predictions.duckdb"])
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
        # RE-ANCHORED (Plan 33.2-20): the two copies must AGREE, which is the divergence
        # claim; the count itself is no longer the pre-re-sync one, because the live 2026
        # capture ingested 272 games into both. Asserted as a floor for the same reason
        # as the parquet node above.
        assert measured["parquet_rows"] == measured["db_rows"]
        assert measured["parquet_rows"] >= N01_PARQUET_ROWS_BEFORE

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


@pytest.mark.integration
class TestClause2GoldGainedA2025Row:
    """Clause 2 is asserted against GOLD -- a DIFFERENT object from clause 1.

    THIS IS NOT A STYLE CHOICE, IT IS A MEASUREMENT. Plan 30-04 measured, before any
    rung ran, that gold's 2025 slice is behind BOTH copies of silver ``games``:

    ======================================  ==================
    object                                  2025 weeks present
    ======================================  ==================
    ``data/gold/features_{wp,ats,ou}``      1-4 (49 rows)
    silver ``games``, DuckDB                1-5
    silver ``games``, parquet               1-22
    ======================================  ==================

    Gold's 2025 coverage is bounded by the OTHER silver sources in the join, not by
    ``games`` alone. So writing this clause as "gold grew by 207" -- the same number
    clause 1 uses -- would fail for a reason that has nothing whatever to do with the
    re-sync, and the phase would be halted by its own control misreading its own
    subject. The two clauses are about two different objects and are written as two
    separate assertions against them.
    ``test_no_assertion_here_reuses_the_silver_row_delta`` below is the mechanical
    guard that keeps it that way.
    """

    def test_at_least_one_2025_gold_row_is_newly_present(self):
        _require_live_gold()

        for matrix in GOLD_MATRICES:
            frame = pd.read_parquet(_GOLD_DIR / f"{matrix}.parquet", engine="pyarrow")
            rows_2025 = int((frame["season"] == 2025).sum())
            assert rows_2025 > GOLD_2025_ROWS_BEFORE_RESYNC, (
                f"{matrix} carries {rows_2025} season-2025 rows against the "
                f"{GOLD_2025_ROWS_BEFORE_RESYNC} pinned in tests/phase30_state.py before "
                "the re-sync. THE CONTROL CANNOT PASS BY DOING NOTHING: if the re-sync "
                "had added no row that reached gold, this reads equal and FAILS. A count "
                "below the pinned value would be worse still -- the rebuild lost rows."
            )

    def test_gold_now_carries_2025_weeks_beyond_the_pre_resync_set(self):
        _require_live_gold()

        for matrix in GOLD_MATRICES:
            frame = pd.read_parquet(_GOLD_DIR / f"{matrix}.parquet", engine="pyarrow")
            weeks = {
                int(week)
                for week in frame.loc[frame["season"] == 2025, "week"].dropna().unique()
            }
            assert weeks > set(GOLD_2025_WEEKS_BEFORE_RESYNC), (
                f"{matrix}'s season-2025 weeks are {sorted(weeks)} against the pinned "
                f"pre-re-sync {list(GOLD_2025_WEEKS_BEFORE_RESYNC)}. A STRICT superset is "
                "required: the re-synced rows are weeks 6-22, so a week that was not "
                "there before must be there now."
            )

    def test_the_scratch_id_set_confirms_the_newly_present_ids(self):
        """Tracked first, unconditionally; the scratch document only ever adds detail.

        No ``pytest.skip`` here. The tracked count is the durable expected value and is
        asserted on every run; where the gitignored run record happens to be present it
        additionally names WHICH ids arrived, which a count cannot.
        """
        _require_live_gold()

        assert GOLD_2025_ROWS_BEFORE_RESYNC > 0
        assert len(GOLD_2025_WEEKS_BEFORE_RESYNC) == GOLD_2025_WEEKS_BEFORE_RESYNC[-1]

        if not _SCRATCH_BEFORE.exists():
            return

        captured = json.loads(_SCRATCH_BEFORE.read_text(encoding="utf-8"))
        before_ids = set(captured["gold_2025_game_ids_before_resync"])
        assert len(before_ids) == GOLD_2025_ROWS_BEFORE_RESYNC, (
            "outputs/n01/divergence_before.json's recorded gold 2025 id set disagrees "
            "with tests/phase30_state.GOLD_2025_ROWS_BEFORE_RESYNC. The TRACKED manifest "
            "is the expectation; do NOT edit it to match a re-run (T-30-52)."
        )

        for matrix in GOLD_MATRICES:
            frame = pd.read_parquet(_GOLD_DIR / f"{matrix}.parquet", engine="pyarrow")
            now = set(frame.loc[frame["season"] == 2025, "game_id"].astype(str))
            newly_present = now - before_ids
            assert newly_present, (
                f"{matrix} has no season-2025 game id that was absent before the "
                "re-sync. The control cannot pass by doing nothing."
            )

    def test_no_assertion_here_reuses_the_silver_row_delta(self):
        """Mechanical guard: clause 2 must never be written as "gold grew by 207".

        The two clauses are about two different objects. If someone later "tidies" this
        class by reusing clause 1's expected delta, the arithmetic becomes a claim about
        silver applied to gold -- empirically false, and it would halt the phase for a
        reason unrelated to the re-sync.
        """
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        clause2 = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.ClassDef)
            and node.name == "TestClause2GoldGainedA2025Row"
        )
        referenced = {
            node.id for node in ast.walk(clause2) if isinstance(node, ast.Name)
        }
        assert not referenced & {
            "N01_DIVERGENCE_BEFORE",
            "N01_DB_ROWS_BEFORE",
            "N01_PARQUET_ROWS_BEFORE",
        }, (
            "clause 2 referenced one of clause 1's silver-games constants. Gold's 2025 "
            "coverage is bounded by the other silver sources in the join and will NOT "
            "grow by the silver divergence."
        )


@pytest.mark.integration
class TestEvery2021To2024ValueIsByteIdentical:
    """SPEC R2's actual control: adding 2025 rows moved NOTHING in 2021-2024.

    Exact hash equality on a game-id-joined, deterministically sorted slice, per column,
    per matrix. No tolerance, no approximate comparison, no float re-formatting on
    either side -- the AFTER digests are produced by the same
    ``scripts.resync_games_duckdb.slice_digests`` that produced the BEFORE ones, which
    itself reuses ``scripts.fingerprint_gold._column_bytes``.

    The BEFORE side is read from ``tests.phase30_state.SLICE_DIGESTS_2021_2024`` -- the
    git-TRACKED pin -- never from ``outputs/n01/slice_hash_before.json``, which is
    gitignored and absent on any checkout that did not just run this plan.

    THE ONE COLUMN EXCLUDED, WHY, AND THE GUARD THAT STOPS ANYTHING HIDING BEHIND IT.
    ``feature_timestamp`` is ``datetime.now(UTC)`` stamped once per build
    (``scripts/build_features.py``). It takes exactly one distinct value per build and
    MUST therefore move on every rebuild, in every season, including 2021-2024 -- it is a
    build artifact, not a feature VALUE, and no rebuild of any kind can leave it alone.
    Counting it as a moved value is exactly what made rung 3's empty-changed-set criterion
    structurally unsatisfiable, which the owner ruled on in D30-OWNER-08: Plan 30-18 gave
    the clock its own ``build_clock`` category in the rung judge at EVERY rung rather than
    a tolerance or an escape.

    This control follows that ruling rather than re-litigating it, and it is excluded the
    same way -- by NAME, from the single registry
    ``scripts.fingerprint_gold.BUILD_CLOCK_COLUMNS``, never re-listed here -- and paired
    with ``test_the_moved_set_is_exactly_the_build_clock``, which pins the set of moved
    columns to be EXACTLY the clock. So the exclusion cannot become a hiding place: one
    genuine data column moving alongside it fails that test. The clock's own digests stay
    in the tracked pin deliberately; recording that it moved is more honest than
    declining to hash it.

    THIS IS NOT A TOLERANCE. Every other column is compared by exact hash equality and
    any movement at all blocks the phase.

    BOTH NODES BELOW ARE NAMED DELIBERATE TRIPWIRES AND MUST STAY RED. They are two of
    the five in ``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`` -- "the five reds
    that MUST STAY RED ... a phase that turned one of these green would have erased a
    disclosure, not fixed a defect". They were NOT re-anchored by Plan 33.2-20's data-pin
    sweep, and could not have been.

    WHAT THEY DISCLOSE TODAY (measured 2026-09-22, recorded because a tripwire nobody
    reads is a tripwire nobody heeds). The control proved its claim ONCE, at Plan 30-08,
    against gold that no longer exists: Phase 33.2's nine-rung p332_ ladder replaced the
    2021-2024 values ON PURPOSE, rung by rung, each with a declared cause, a digest
    bracket and a per-column per-season attribution -- the market family removed, the
    normalization re-ordered on lock instants, unscorable cells blanked, eight
    never-populated columns dropped. So this class now reports both a changed COLUMN SET
    and moved VALUES, correctly, and that report IS the disclosure.

    WHY NOT RE-ANCHORED, in this class's own words: "do not re-capture the BEFORE
    digests". Re-capturing them would compare today's gold against today's gold, which is
    true by construction and proves nothing -- the same circularity the frozen-band and
    forensic-copy reasoning rejects elsewhere. There is no honest measurement available:
    the object the expected values describe is gone, and saying so is the point.

    WHAT CARRIES THE LIVE CLAIM INSTEAD, and it is strictly more: the p332_ attribution
    modules (``tests/integration/test_p332_rung*_attribution.py``,
    ``..._step*_attribution.py``) assert per rung WHICH columns moved, in WHICH seasons,
    against a fingerprint document taken before that rung ran -- a byte-identity claim per
    step rather than one across a phase. Those are green.
    """

    def test_every_data_column_reproduces_its_pre_resync_digest_exactly(self):
        _require_live_gold()
        from scripts.resync_games_duckdb import slice_digests

        after = slice_digests()

        for matrix, expected in SLICE_DIGESTS_2021_2024.items():
            actual = after[matrix]
            assert set(actual) == set(expected), (
                f"{matrix}'s 2021-2024 slice gained "
                f"{sorted(set(actual) - set(expected))} and lost "
                f"{sorted(set(expected) - set(actual))}. Rung 4 adds and removes no "
                "column."
            )
            moved = [
                column
                for column in sorted(expected)
                if column not in BUILD_CLOCK_COLUMNS
                and actual[column] != expected[column]
            ]
            assert not moved, (
                f"{matrix}: {len(moved)} of {len(expected)} columns moved in 2021-2024 "
                f"across the N-01 re-sync -- {moved[:10]}.\n\n"
                "THE WR-06 FIX IS INCOMPLETE AND THE PHASE IS BLOCKED (SPEC R2). Future "
                "rows are still influencing past values: a whole-frame statistic -- an "
                "imputation median, a q01/q99 bound -- is being fitted over data that "
                "now includes the newly arrived 2025 rows and applied to prior seasons. "
                "This is the exact coupling N-01 was held open through Phase 29 to test "
                "for, and rung 4's judge offers NO upstream-drift escape for it. Do not "
                "widen a tolerance, do not re-capture the BEFORE digests, and do not "
                "proceed to the gate."
            )

    def test_the_moved_set_is_exactly_the_build_clock(self):
        """Nothing may hide behind the build-clock exclusion, so the moved set is pinned.

        The companion to the exclusion above. If a genuine data column moved in 2021-2024
        alongside the clock, this fails and NAMES it -- so the exclusion can never widen
        into a tolerance by accident.
        """
        _require_live_gold()
        from scripts.resync_games_duckdb import slice_digests

        after = slice_digests()

        for matrix, expected in SLICE_DIGESTS_2021_2024.items():
            moved = {
                column
                for column in expected
                if after[matrix][column] != expected[column]
            }
            assert moved == set(BUILD_CLOCK_COLUMNS), (
                f"{matrix}: the 2021-2024 columns that moved across the re-sync are "
                f"{sorted(moved)}. They must be EXACTLY the per-build clock "
                f"{sorted(BUILD_CLOCK_COLUMNS)} and nothing else. A clock that did NOT "
                "move would mean this gold was never rebuilt; any other column moving "
                "means the WR-06 fix is incomplete and the phase is BLOCKED (SPEC R2)."
            )

    def test_the_scratch_capture_agrees_with_the_tracked_digests(self):
        """T-30-52, and NOT a skip. The tracked shape is asserted unconditionally."""
        assert set(SLICE_DIGESTS_2021_2024) == set(GOLD_MATRICES)
        assert all(
            len(columns) >= 190 for columns in SLICE_DIGESTS_2021_2024.values()
        ), (
            "the tracked slice digests must cover every gold column; a truncated pin "
            "would silently narrow the control"
        )
        assert all(
            len(digest) == 64
            for columns in SLICE_DIGESTS_2021_2024.values()
            for digest in columns.values()
        ), "every pinned value must be a full sha256 hex digest"

        if not _SCRATCH_SLICE_BEFORE.exists():
            return

        captured = json.loads(_SCRATCH_SLICE_BEFORE.read_text(encoding="utf-8"))
        assert captured == SLICE_DIGESTS_2021_2024, (
            "outputs/n01/slice_hash_before.json disagrees with "
            "tests/phase30_state.SLICE_DIGESTS_2021_2024. One was regenerated out of "
            "step with the other. The TRACKED manifest is the expectation; a scratch "
            "file re-run AFTER the re-sync would record the after values and silently "
            "make the control tautological."
        )

    def test_this_module_contains_no_approximate_comparison(self):
        """A tolerance anywhere in this file would void the control it is written for."""
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not called & {"approx", "isclose", "allclose", "assert_allclose"}, (
            "an approximate comparison appeared in the byte-identity control. The whole "
            "claim is EXACT equality: a 2021-2024 value that moved by any amount at all "
            "means a whole-frame statistic is still reaching backwards."
        )

    def test_the_anti_vacuity_guard_is_durable_and_visible(self):
        """The recorded divergence clause 1 asserts against was strictly positive.

        Stated as its own test, read from the TRACKED constant, so a future reader sees
        the anti-vacuity guard rather than having to infer it from an assertion message.
        """
        assert N01_DIVERGENCE_BEFORE > 0
        assert N01_PARQUET_ROWS_BEFORE > N01_DB_ROWS_BEFORE
        assert SLICE_DIGESTS_2021_2024, (
            "an empty digest pin would make the byte-identity control vacuous: it would "
            "iterate over nothing and pass"
        )
