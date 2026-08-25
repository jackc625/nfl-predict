"""WR-05 and WR-07: two guards that argued for a rule and did not apply it.

WR-07 -- THE ``data/`` WRITE PROHIBITION. Three tools state that nothing writes under
``data/`` outside the ONE sanctioned, fingerprinted gold rebuild.
``backtest.group_gate._reject_data_path`` was the only one that ENFORCED it, and its own
docstring explains why the check has to run BEFORE the work: "a refusal that arrived
after twelve walk-forward re-fits would be a refusal nobody could afford to trust."

The two tools that operate directly on the gold tree were the two that only asserted it.
``scripts/fingerprint_gold.py``'s module docstring said it "is strictly read-only with
respect to data/ -- the JSON output must be written somewhere else", while ``--out``
accepted any path and ``main()`` mkdir'd the parent and wrote.
``scripts/resync_games_duckdb.py``'s ``--out`` help said "never under data/" with the
same absence of enforcement. The phase's hard-boundary hash manifest reads through
``load_dataframe``, so it would not have caught such a write either.

WR-05 -- MEMBERSHIP, NOT COUNTS. ``resync_games_duckdb`` keyed its ``--apply`` refusal
and its exit code on ``divergence``, which is ``len(parquet) - len(database)``. A lake
with equal row counts and DIFFERENT membership -- a row REPLACED rather than lost -- was
refused with a message asserting something false ("someone has already re-synced this
lake"), and a re-sync that left rows only in DuckDB would have exited 0. The sibling
guard written in the same phase, ``data_qa.check_duckdb_parquet_consistency``, argues the
opposite explicitly: "MEMBERSHIP, NOT COUNTS. Equal row counts with DIFFERENT membership
is the subtler failure and is reported as one."

TEST CLASS: plain unit tests. No data/, no artifacts/, no outputs/, no live lake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.group_gate import _reject_data_path
from scripts import resync_games_duckdb as resync
from scripts.fingerprint_gold import build_parser as fingerprint_parser
from utils.paths import reject_data_path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTheDataWriteProhibitionIsEnforcedNotAsserted:
    """One implementation of the rule, reached by every tool that states it."""

    @pytest.mark.parametrize(
        "candidate",
        [
            "data/gold/x.json",
            "data/silver/y.json",
            "data/x.json",
            "data/gold/../gold/nested/z.json",
        ],
    )
    def test_a_path_under_data_is_refused(self, candidate: str) -> None:
        with pytest.raises(ValueError, match="Refusing to write"):
            reject_data_path(REPO_ROOT / candidate)

    @pytest.mark.parametrize(
        "candidate",
        ["outputs/fingerprints/before.json", "outputs/n01/divergence.json"],
    )
    def test_a_path_outside_data_is_resolved_and_returned(self, candidate: str) -> None:
        resolved = reject_data_path(REPO_ROOT / candidate)
        assert resolved.is_absolute()
        assert resolved.name.endswith(".json")

    def test_a_sibling_directory_whose_name_merely_starts_with_data_is_allowed(
        self,
    ) -> None:
        """``databases/`` is not ``data/``. The check is on the tree, not the prefix."""
        assert reject_data_path(REPO_ROOT / "databases" / "x.json")

    def test_the_refusal_says_where_to_write_instead(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            reject_data_path(
                REPO_ROOT / "data" / "x.json", suggestion="outputs/fingerprints/"
            )
        assert "outputs/fingerprints/" in str(excinfo.value), (
            "a refusal that does not say what to do instead just gets worked around"
        )

    def test_the_group_gate_guard_delegates_to_the_one_implementation(self) -> None:
        """WR-07's fix is ONE rule, not a third copy of it.

        A second copy is free to drift away from the rule everything else applies --
        the failure mode this project already names in
        ``fingerprint_gold._discrete_indicator_predicate``.
        """
        import inspect

        source = inspect.getsource(_reject_data_path)
        assert "reject_data_path(" in source
        assert "relative_to" not in source, (
            "backtest.group_gate._reject_data_path re-implements the check instead of "
            "delegating to utils.paths.reject_data_path"
        )

    def test_the_group_gate_guard_still_refuses(self) -> None:
        with pytest.raises(ValueError, match="group-gate result"):
            _reject_data_path(REPO_ROOT / "data" / "gold" / "x.json")


class TestFingerprintGoldRefusesADataOut:
    """The module docstring's claim, made true."""

    def test_the_out_flag_exists_and_takes_a_path(self) -> None:
        args = fingerprint_parser().parse_args(["--out", "outputs/x.json"])
        assert args.out == Path("outputs/x.json")

    def test_main_guards_out_before_reading_any_gold(self) -> None:
        """The refusal must precede the work, not follow it."""
        import inspect

        from scripts.fingerprint_gold import main

        source = inspect.getsource(main)
        guard_at = source.find("reject_data_path")
        work_at = min(
            (
                i
                for i in (source.find("fingerprint_gold()"), source.find("read_text"))
                if i != -1
            ),
            default=-1,
        )
        assert guard_at != -1, (
            "scripts/fingerprint_gold.main() does not guard --out. Its module docstring "
            "claims the tool is strictly read-only with respect to data/."
        )
        assert work_at == -1 or guard_at < work_at, (
            "the --out guard runs AFTER the tool has started work; a refusal that "
            "arrives once the work is done is a refusal nobody can afford to trust"
        )

    def test_the_docstring_no_longer_points_at_the_wrong_directory(self) -> None:
        import scripts.fingerprint_gold as module

        assert module.__doc__ is not None
        assert "outputs/fingerprints/" in module.__doc__


class TestResyncRefusesADataOut:
    def test_main_guards_out_before_measuring(self) -> None:
        import inspect

        source = inspect.getsource(resync.main)
        guard_at = source.find("reject_data_path")
        work_at = source.find("measure_divergence()")

        assert guard_at != -1, (
            "scripts/resync_games_duckdb.main() does not guard --out, though its help "
            "text says 'never under data/'."
        )
        assert guard_at < work_at


class TestTheResyncRefusalKeysOnMembership:
    """WR-05: the arithmetic difference is not the question the re-sync answers."""

    @staticmethod
    def _measured(*, only_parquet: int, only_duckdb: int) -> dict:
        return {
            "only_in_parquet_count": only_parquet,
            "only_in_duckdb_count": only_duckdb,
            # Deliberately zero: this is the case a count-keyed guard gets wrong.
            "divergence": 0,
        }

    def test_equal_counts_with_different_membership_is_DIVERGED(self) -> None:
        """A row REPLACED rather than lost. divergence == 0 and the lake is broken."""
        assert resync._is_diverged(self._measured(only_parquet=1, only_duckdb=1)), (
            "A lake with one row only in parquet and one only in DuckDB nets to a "
            "divergence of ZERO. Refusing it, with a message asserting 'someone has "
            "already re-synced this lake', both refuses a needed repair and says "
            "something false."
        )

    def test_rows_only_in_duckdb_is_DIVERGED(self) -> None:
        assert resync._is_diverged(self._measured(only_parquet=0, only_duckdb=3))

    def test_rows_only_in_parquet_is_DIVERGED(self) -> None:
        assert resync._is_diverged(self._measured(only_parquet=207, only_duckdb=0))

    def test_agreeing_membership_is_NOT_diverged(self) -> None:
        assert not resync._is_diverged(self._measured(only_parquet=0, only_duckdb=0))

    def test_the_refusal_and_the_exit_code_both_read_membership(self) -> None:
        """Neither branch may key on the netting arithmetic."""
        import inspect

        source = inspect.getsource(resync.main)
        assert source.count("_is_diverged") == 2, (
            "both the --apply refusal and the exit code must ask the membership "
            f"question; found {source.count('_is_diverged')} call(s)"
        )
        assert 'measured["divergence"] == 0' not in source
        assert '["divergence"] == 0' not in source

    def test_an_apply_against_agreeing_membership_is_REFUSED_with_an_honest_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Driven through main(), so the message a real operator sees is the one asserted."""
        monkeypatch.setattr(
            resync,
            "measure_divergence",
            lambda *a, **k: {
                "parquet_rows": 10,
                "db_rows": 10,
                "divergence": 0,
                "only_in_parquet_count": 0,
                "only_in_duckdb_count": 0,
                "missing_by_season": {},
            },
        )
        monkeypatch.setattr(
            resync,
            "resync_games",
            lambda: pytest.fail("resync_games ran despite an agreeing lake"),
        )

        assert resync.main(["--apply"]) == 2

        printed = capsys.readouterr().err
        assert "row-set MEMBERSHIP" in printed
        assert "already re-synced this lake" not in printed, (
            "the refusal message asserted a cause it cannot know. Equal counts with "
            "DIFFERENT membership means the lake was never re-synced at all."
        )

    def test_an_apply_against_different_membership_is_NOT_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case a count-keyed refusal got wrong: divergence 0, membership diverged."""
        diverged = {
            "parquet_rows": 10,
            "db_rows": 10,
            "divergence": 0,
            "only_in_parquet_count": 1,
            "only_in_duckdb_count": 1,
            "missing_by_season": {"2025": 1},
        }
        agreeing = {**diverged, "only_in_parquet_count": 0, "only_in_duckdb_count": 0}
        monkeypatch.setattr(resync, "measure_divergence", lambda *a, **k: diverged)
        monkeypatch.setattr(
            resync, "resync_games", lambda: {"before": diverged, "after": agreeing}
        )

        assert resync.main(["--apply"]) == 0, (
            "A lake with a row REPLACED rather than lost nets to divergence 0 and was "
            "refused. It is exactly what this script exists to repair."
        )

    def test_a_resync_that_leaves_rows_only_in_duckdb_exits_NONZERO(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The count-keyed exit code reported success on a surplus in DuckDB."""
        before = {
            "parquet_rows": 12,
            "db_rows": 10,
            "divergence": 2,
            "only_in_parquet_count": 2,
            "only_in_duckdb_count": 0,
            "missing_by_season": {"2025": 2},
        }
        after = {
            "parquet_rows": 12,
            "db_rows": 12,
            "divergence": 0,
            "only_in_parquet_count": 0,
            "only_in_duckdb_count": 3,
            "missing_by_season": {},
        }
        monkeypatch.setattr(resync, "measure_divergence", lambda *a, **k: before)
        monkeypatch.setattr(
            resync, "resync_games", lambda: {"before": before, "after": after}
        )

        assert resync.main(["--apply"]) == 1, (
            "The re-sync left three rows only in DuckDB, and the row COUNTS netted to "
            "zero. Keying the exit code on that arithmetic reports success on a lake "
            "that is still diverged."
        )

    def test_neither_branch_keys_on_the_netting_arithmetic(self) -> None:
        import inspect

        source = inspect.getsource(resync.main)
        assert source.count("_is_diverged") == 2, (
            "both the --apply refusal and the exit code must ask the membership "
            f"question; found {source.count('_is_diverged')} call(s)"
        )
        assert '["divergence"] == 0' not in source
