"""A green Elo step may no longer mean an unwritten snapshot table (COLD-02, T-33-13).

THE DEFECT THIS MODULE CLOSES
-----------------------------
``pipeline/steps.step_build_elo`` called ``builder.update_current_season()``, discarded
the return value, and persisted NOTHING. The step reported success every Friday. The
snapshot table never gained a current-season row, the gold LEFT JOIN therefore produced
NaN Elo for every current-season game, and the imputer quietly filled those NaNs before
handing them to the deployed WP model. Nothing in the run log said so, because from the
outside a step that persisted nothing is indistinguishable from one that worked.

The refusal asserts on the MESSAGE, never on a process exit status. This milestone's
standing invariant is that a criterion asserts the EFFECT: an exit code says a process
ended unhappily and nothing about which fact was violated, and a test pinned to one
would pass for any nonzero exit including an import error.

TWO CASES, AND THEY MUST NOT BE CONFLATED
-----------------------------------------
ZERO computed snapshots is a normal outcome for a season with no completed games -- the
2026 cold start is exactly that, mid-week-1 with two graded games out of 272 -- and it
must pass. A NON-ZERO computed count with nothing written is the refusal. A guard that
fired on both would make the phase's own cold start un-runnable; a guard that fired on
neither would be the defect.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    seed_sandbox_games,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_SEASON = 2026


def _sandbox_builder_class(monkeypatch, sandbox):
    """Make ``step_build_elo``'s own ``EloBuilder()`` construct inside the sandbox.

    The step constructs its builder itself, with no arguments, which is right for
    production and leaves a test no seam. Subclassing and rebinding the NAME the step
    imports keeps the step's body untouched -- the code under test is the real one.
    """
    import scripts.build_elo as build_elo_mod

    real_cls = build_elo_mod.EloBuilder

    class _SandboxedEloBuilder(real_cls):
        def __init__(self, data_root=None):
            super().__init__(data_root=sandbox)

    monkeypatch.setattr(build_elo_mod, "EloBuilder", _SandboxedEloBuilder)
    return _SandboxedEloBuilder


class TestTheStepRefusesToReportSuccessWithoutPersisting:
    """The refusal, its message, and the control that proves it is not always on."""

    def test_a_skipped_persist_raises_and_names_the_missing_rows(
        self, tmp_path, monkeypatch
    ) -> None:
        from pipeline.steps import step_build_elo
        from scripts.build_elo import EloSnapshotNotPersistedError

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        seed_sandbox_games(make_season_games(LIVE_SEASON, weeks=2))
        builder_cls = _sandbox_builder_class(monkeypatch, sandbox)

        # The writer is stubbed OUT: the run computes snapshots and skips the persist.
        monkeypatch.setattr(
            builder_cls, "save_live_append", lambda self, *a, **k: None, raising=False
        )

        with pytest.raises(EloSnapshotNotPersistedError) as excinfo:
            step_build_elo()

        message = str(excinfo.value)
        assert "8" in message, (
            "the refusal must NAME the missing row count. The sandbox season has two "
            f"weeks of four graded games, so eight snapshots were computed. Got: "
            f"{message}"
        )
        assert str(LIVE_SEASON) in message, (
            f"the refusal must name the season it computed them for. Got: {message}"
        )
        assert read_sandbox_table(sandbox, "elo_game_snapshots").empty, (
            "the stub was supposed to skip the persist, but rows were written -- so "
            "this test is not exercising the case it claims to."
        )

    def test_the_refusal_carries_its_recovery_command(self) -> None:
        """A refusal without a recovery command tells an operator nothing to do."""
        from scripts.build_elo import EloSnapshotNotPersistedError

        message = str(EloSnapshotNotPersistedError(rows=17, season=2026))
        assert "17" in message
        assert "2026" in message
        assert "python -m" in message, (
            f"the refusal must name the command that persists them. Got: {message}"
        )

    def test_a_completed_persist_does_not_raise(self, tmp_path, monkeypatch) -> None:
        """Fail-open control: a guard that is always on is a guard, not a check."""
        from pipeline.steps import step_build_elo

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        seed_sandbox_games(make_season_games(LIVE_SEASON, weeks=2))
        _sandbox_builder_class(monkeypatch, sandbox)

        step_build_elo()

        written = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert len(written) == 8, (
            f"the unstubbed step must persist its eight snapshots, got {len(written)}"
        )

    def test_a_season_with_no_completed_games_is_not_a_refusal(
        self, tmp_path, monkeypatch
    ) -> None:
        """R3's explicit edge case: zero computed rows persists zero and exits clean.

        This is not hypothetical. The live 2026 capture reaches the Elo step with no
        graded games in silver at all; a guard that could not tell "nothing to write"
        from "wrote nothing" would make the phase's own cold start un-runnable.
        """
        from pipeline.steps import step_build_elo

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        seed_sandbox_games(make_season_games(LIVE_SEASON, weeks=2, graded_weeks=()))
        _sandbox_builder_class(monkeypatch, sandbox)

        step_build_elo()  # must NOT raise

        assert read_sandbox_table(sandbox, "elo_game_snapshots").empty


class TestTheStepReachesBothVerbs:
    """An AST assertion, so the wiring cannot be quietly unhooked again."""

    def test_step_build_elo_calls_the_update_and_the_live_append(self) -> None:
        source = (REPO_ROOT / "pipeline" / "steps.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "step_build_elo"
        )
        calls = sorted(
            {
                node.func.attr
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
        )

        assert "update_current_season" in calls, calls
        assert "save_live_append" in calls, (
            "step_build_elo computes the current season and must PERSIST it. Without "
            f"the live append the step is green and the table is empty. Calls: {calls}"
        )
        assert "save_results" not in calls, (
            f"the weekly step must never reach the retired replace-all verb: {calls}"
        )
