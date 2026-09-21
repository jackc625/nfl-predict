"""The Elo write path is TWO named verbs, and replace-all is reachable through one.

WHAT THIS MODULE IS FOR (COLD-02, T-33-12 / T-33-16 / T-33-16c)
---------------------------------------------------------------
``EloBuilder.save_results`` wrote ``elo_game_snapshots`` with ``append_mode=False``
and the comment "Always replace, full rebuild". That is correct for a full rebuild and
catastrophic for the weekly path: ``pipeline/steps.step_build_elo`` runs every Friday
against a table holding 24 seasons of burn-in, and a replace-shaped write there would
leave one season where the chain used to be. The verb was reachable by default, and
this repository's record is that the default-reachable path is the one that eventually
fires.

So ``save_results`` is RETIRED, and the two behaviours it conflated are now two verbs
that say which one they are:

* ``save_full_rebuild(snapshots, start_season)`` -- replace the snapshot table, loudly
  and attributably, behind its own CLI flag.
* ``save_live_append(season, *, snapshots)`` -- upsert the snapshot table on
  ``game_id``.

ONE ELO ARTIFACT (Plan 33.2-05, D33.2-22, owner ratified 2026-09-21). Until that plan
both verbs also wrote ``games_with_elo``, ``elo_rating_history``, ``elo_ratings_current``
and ``elo_ratings.json`` -- four side stores of a legacy per-season pass that learned
no home-field advantage. The pass and the stores were deleted together, and this
module's intent -- "every Elo write goes through a declared entry point" -- is now
asserted over the one surviving artifact. Two things went BY NAMED RULING rather than by
re-expression: the ``save_ratings`` write (``EloRatingSystem.save_ratings`` no longer
exists, so there is no JSON write to route), and the three-frame signature (there is
one grain and one destination, so the keyword-only snapshot frame is the whole
signature, and a test below refuses the two deleted frame names coming back).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    per_season_row_digests,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_ELO_SOURCE = REPO_ROOT / "scripts" / "build_elo.py"

# Production modules that could plausibly reach an Elo write. A retired verb has to be
# retired everywhere, not only where it was most recently read.
_PRODUCTION_MODULES: tuple[str, ...] = (
    "scripts/build_elo.py",
    "pipeline/steps.py",
    "features/elo_features.py",
)


# The four artifacts D33.2-22 deleted. No write verb may produce any of them again.
DELETED_ELO_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "games_with_elo.parquet",
    "elo_rating_history.parquet",
    "elo_ratings_current.parquet",
    "elo_ratings.json",
)


def _live_snapshots(builder, season: int):
    """Build one season's snapshot frame.

    Deliberately ``build_season_frames`` and NOT ``update_current_season``: this module
    is about the WRITE verbs, and coupling it to the live update path would make it
    fail for reasons that belong to the update path.
    """
    return builder.build_season_frames(season).snapshots


def _deleted_artifacts_on_disk(sandbox: Path) -> list[str]:
    """Every deleted-by-ruling artifact anywhere under the sandbox silver layer."""
    return sorted(
        path.relative_to(sandbox).as_posix()
        for path in (sandbox / "silver").rglob("*")
        if path.name in DELETED_ELO_ARTIFACT_FILENAMES
    )


class TestTheTwoNamedVerbs:
    """The class surface itself: two verbs in, one verb out."""

    def test_the_builder_exposes_both_verbs_and_no_longer_exposes_save_results(
        self,
    ) -> None:
        from scripts.build_elo import EloBuilder

        assert hasattr(EloBuilder, "save_full_rebuild"), (
            "EloBuilder must expose save_full_rebuild -- the explicitly named, loudly "
            "attributed replace-everything verb."
        )
        assert hasattr(EloBuilder, "save_live_append"), (
            "EloBuilder must expose save_live_append -- the weekly verb that upserts "
            "rather than replaces."
        )
        assert not hasattr(EloBuilder, "save_results"), (
            "save_results is RETIRED. While the name exists, a caller can reach "
            "replace-all by default, which is exactly the reachability T-33-12 is "
            "about."
        )

    def test_save_live_append_takes_one_keyword_only_snapshot_frame(self) -> None:
        """One grain, one destination, one frame -- and it cannot be passed by position."""
        from scripts.build_elo import EloBuilder

        parameters = inspect.signature(EloBuilder.save_live_append).parameters
        assert list(parameters) == ["self", "season", "snapshots"], (
            f"save_live_append takes {list(parameters)}. D33.2-22 left one row table, "
            "so the verb takes the season and the snapshot frame and nothing else."
        )
        assert parameters["snapshots"].kind is inspect.Parameter.KEYWORD_ONLY, (
            "'snapshots' must be KEYWORD-ONLY, so a caller cannot hand the frame and "
            "the season in the wrong order."
        )

    def test_neither_deleted_frame_is_accepted(self) -> None:
        """The two frames D33.2-22 deleted cannot come back as parameters."""
        from scripts.build_elo import EloBuilder

        for verb in ("save_live_append", "save_full_rebuild"):
            parameters = inspect.signature(getattr(EloBuilder, verb)).parameters
            for name in ("games_with_elo", "rating_history", "processed_games"):
                assert name not in parameters, (
                    f"{verb} accepts '{name}', a frame for a side table D33.2-22 "
                    "deleted. A frame with nowhere to go is how a deleted table "
                    "quietly comes back."
                )

    def test_the_row_and_state_constants_state_the_split_rule(self) -> None:
        """Asserted on the DEFINING module, never only through the re-export."""
        from scripts.elo_generation import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS

        assert ELO_ROW_TABLES == ("elo_game_snapshots",)
        assert ELO_STATE_ARTIFACTS == ()
        assert not set(ELO_ROW_TABLES) & set(ELO_STATE_ARTIFACTS), (
            "an artifact cannot be both accumulated history and current state"
        )

    def test_the_constants_agree_with_the_phase33_manifest(self) -> None:
        """The split rule has ONE committed home the tests import, not two.

        Compared against Plan 33.2-05's appended slot; the Plan 33-03 slot above it
        stays as the record of the five-artifact set that was true before D33.2-22.
        """
        from scripts.elo_generation import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS
        from tests.phase33_state import (
            PLAN_33_2_05_ELO_ROW_TABLE_NAMES,
            PLAN_33_2_05_ELO_STATE_ARTIFACT_NAMES,
        )

        assert tuple(PLAN_33_2_05_ELO_ROW_TABLE_NAMES) == ELO_ROW_TABLES
        assert tuple(PLAN_33_2_05_ELO_STATE_ARTIFACT_NAMES) == ELO_STATE_ARTIFACTS


class TestTheWriteSetIsDeclaredHonestly:
    """``save_dataframe`` writes BOTH stores; a write-set that omits one is a lie."""

    def test_the_write_set_names_the_shared_duckdb(self) -> None:
        from tests.phase33_state import PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB

        entries = list(PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB)
        assert len(entries) >= 4, (
            "the Elo write set has at least four members: the snapshot table, the "
            f"generation pointer, the staged generation tree and the DuckDB. Got "
            f"{entries}."
        )
        assert any("nfl_predictions.duckdb" in entry for entry in entries), (
            "data/storage.save_dataframe defaults save_to_db=True, so every Elo write "
            "that routes through it touches the SHARED DuckDB store as well as the "
            "parquet. A write set that names only parquet paths under-declares the "
            "blast radius of both verbs."
        )

    def test_the_write_set_names_every_row_table_and_state_artifact(self) -> None:
        from scripts.elo_generation import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS
        from tests.phase33_state import PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB

        joined = "\n".join(PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB)
        for name in (*ELO_ROW_TABLES, *ELO_STATE_ARTIFACTS):
            assert name in joined, (
                f"'{name}' is written by the Elo verbs but is absent from "
                "PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB."
            )

    def test_the_write_set_names_no_deleted_artifact(self) -> None:
        from tests.phase33_state import PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB

        joined = "\n".join(PLAN_33_2_05_ELO_WRITE_SET_INCLUDING_DUCKDB)
        declared_deleted = [
            name for name in DELETED_ELO_ARTIFACT_FILENAMES if name in joined
        ]
        assert not declared_deleted, (
            f"the write set still declares {declared_deleted}, which no verb writes "
            "since D33.2-22. A write set that over-declares hides the real blast "
            "radius as surely as one that under-declares."
        )


class TestSaveFullRebuild:
    """Replace-everything, but only through the verb that says so."""

    def test_a_full_rebuild_writes_the_snapshot_table_and_nothing_deleted(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2024, weeks=2)
        builder = sandbox_builder(sandbox, games)

        snapshots = builder.build_all_ratings(start_season=2024)
        builder.save_full_rebuild(snapshots, start_season=2024)

        written = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert len(written) == len(snapshots) > 0, (
            f"the full rebuild wrote {len(written)} snapshot rows for "
            f"{len(snapshots)} computed."
        )
        assert _deleted_artifacts_on_disk(sandbox) == [], (
            "the full rebuild wrote an artifact D33.2-22 deleted: "
            f"{_deleted_artifacts_on_disk(sandbox)}"
        )

    def test_a_full_rebuild_logs_an_attributed_line(
        self, tmp_path, monkeypatch
    ) -> None:
        """A full rebuild must never be mistakable for a weekly run in a log.

        The logger is captured directly rather than through ``caplog``: this project
        logs through structlog, so a stdlib-handler assertion would pass or fail on the
        logging CONFIGURATION rather than on what the verb actually recorded.
        """
        import scripts.build_elo as build_elo_mod

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2024, weeks=2)
        builder = sandbox_builder(sandbox, games)
        snapshots = builder.build_all_ratings(start_season=2024)

        recorded: list[tuple[str, dict]] = []
        real_logger = build_elo_mod.logger

        class _Recorder:
            def info(self, event, **kwargs):
                recorded.append((event, kwargs))
                return real_logger.info(event, **kwargs)

            def __getattr__(self, name):
                return getattr(real_logger, name)

        monkeypatch.setattr(build_elo_mod, "logger", _Recorder())
        builder.save_full_rebuild(snapshots, start_season=2024)

        attributed = [
            (event, kwargs) for event, kwargs in recorded if "FULL REBUILD" in event
        ]
        assert attributed, (
            "the full-rebuild write must announce itself by name; a silent "
            "replace-everything is indistinguishable from a weekly append in a log. "
            f"Events seen: {[event for event, _ in recorded]}"
        )
        _, fields = attributed[0]
        assert fields.get("start_season") == 2024, (
            f"the attributed line must name the start season, got {fields}"
        )
        assert fields.get("elo_game_snapshots_rows") == len(snapshots), (
            "the attributed line must name the snapshot row count; a rebuild that "
            f"reports no row count cannot be reconciled afterwards. Got {fields}"
        )
        for name in (
            "games_with_elo_rows",
            "elo_rating_history_rows",
            "elo_ratings_current_rows",
        ):
            assert name not in fields, (
                f"the attributed line still reports {name}, for a side table "
                "D33.2-22 deleted. A row count for a table that is not written is a "
                "false record."
            )


class TestSaveLiveAppend:
    """The weekly verb: one frame, one table, upserted on game_id."""

    def test_the_snapshot_frame_lands_in_its_table_and_nothing_else_is_written(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2025, weeks=2)
        builder = sandbox_builder(sandbox, games)

        snapshots = _live_snapshots(builder, 2025)
        builder.save_live_append(2025, snapshots=snapshots)

        written = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert "home_elo_pre" in written.columns, (
            "elo_game_snapshots must carry the PRE-GAME snapshot grain"
        )
        assert len(written) == len(snapshots) > 0
        assert _deleted_artifacts_on_disk(sandbox) == [], (
            "the live append wrote an artifact D33.2-22 deleted: "
            f"{_deleted_artifacts_on_disk(sandbox)}"
        )

    def test_a_second_live_append_does_not_grow_the_row_table(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2025, weeks=2)
        builder = sandbox_builder(sandbox, games)

        snapshots = _live_snapshots(builder, 2025)
        builder.save_live_append(2025, snapshots=snapshots)
        first = read_sandbox_table(sandbox, "elo_game_snapshots")
        builder.save_live_append(2025, snapshots=snapshots)
        second = read_sandbox_table(sandbox, "elo_game_snapshots")

        assert len(second) == len(second.drop_duplicates(subset=["game_id"])), (
            "elo_game_snapshots gained duplicate game_id rows on the second append -- "
            "the upsert is behaving like an append."
        )
        assert per_season_row_digests(second) == per_season_row_digests(first), (
            "a repeat append of the same frame moved the table's rows."
        )

    def test_a_foreign_season_in_the_frame_is_refused_by_name(
        self, tmp_path, monkeypatch
    ) -> None:
        """A live append cannot rewrite another season."""
        import pandas as pd

        from scripts.build_elo import EloForeignSeasonRowsError

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = pd.concat(
            [make_season_games(2024, weeks=2), make_season_games(2025, weeks=2)],
            ignore_index=True,
        )
        builder = sandbox_builder(sandbox, games)

        poisoned = _live_snapshots(builder, 2025).copy()
        poisoned.loc[poisoned.index[0], "season"] = 2024

        with pytest.raises(EloForeignSeasonRowsError) as excinfo:
            builder.save_live_append(2025, snapshots=poisoned)

        message = str(excinfo.value)
        assert "2024" in message, message
        assert "2025" in message, message
        assert read_sandbox_table(sandbox, "elo_game_snapshots").empty, (
            "the refusal half-applied: rows were written before it raised."
        )


class TestTheJsonStateWriteIsDeletedByRuling:
    """``save_ratings`` is not re-expressed: D33.2-22 deleted it, and that is asserted."""

    def test_the_rating_system_has_no_json_write_or_read(self) -> None:
        from ratings.elo import EloRatingSystem

        for name in ("save_ratings", "load_ratings", "ratings_state"):
            assert not hasattr(EloRatingSystem, name), (
                f"EloRatingSystem.{name} is back. It served elo_ratings.json, a store "
                "D33.2-22 deleted; a write path to it is a production write reachable "
                "from anything that builds ratings, including a read-only audit."
            )


class TestTheRetiredVerbIsGoneFromProduction:
    """A source scan, with both of its controls."""

    @staticmethod
    def _existing_modules() -> list[str]:
        return [p for p in _PRODUCTION_MODULES if (REPO_ROOT / p).exists()]

    @staticmethod
    def _violations(source: str, label: str) -> list[str]:
        """Report every reference to the retired replace-all verb in *source*."""
        found: list[str] = []
        needle = "save_" + "results"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == needle:
                found.append(f"{label}:{node.lineno}: attribute access .{needle}")
            elif isinstance(node, ast.Name) and node.id == needle:
                found.append(f"{label}:{node.lineno}: name reference {needle}")
            elif isinstance(node, ast.FunctionDef) and node.name == needle:
                found.append(f"{label}:{node.lineno}: def {needle}")
        return found

    def test_the_scan_visits_a_non_empty_module_list(self) -> None:
        """Anti-vacuity: a scan over zero modules reports green while proving nothing."""
        present = self._existing_modules()
        assert present, "the retired-verb scan visited ZERO modules"
        assert "scripts/build_elo.py" in present
        assert "pipeline/steps.py" in present

    def test_no_production_module_reaches_the_retired_verb(self) -> None:
        violations: list[str] = []
        for relative in self._existing_modules():
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            violations.extend(self._violations(source, relative))

        assert not violations, (
            "a production module still reaches the retired replace-all verb "
            "(T-33-12):\n" + "\n".join(f"  - {line}" for line in violations)
        )

    def test_the_scan_reports_a_planted_call(self) -> None:
        """Fail-closed control: a scan only ever observed passing cannot be trusted."""
        planted = "builder.save_" + "results(frame)\n"
        reported = self._violations(planted, "<planted>")
        assert reported, "the retired-verb scan did not report a planted call"


class TestReplaceShapedWritesAreConfinedToTheFullRebuild:
    """AST assertions over ``scripts/build_elo.py`` itself."""

    @staticmethod
    def _tree() -> ast.Module:
        return ast.parse(BUILD_ELO_SOURCE.read_text(encoding="utf-8"))

    def test_every_append_mode_keyword_lies_inside_save_full_rebuild(self) -> None:
        tree = self._tree()
        functions = {
            node.name: (node.lineno, node.end_lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "save_full_rebuild" in functions, "save_full_rebuild is not defined"
        low, high = functions["save_full_rebuild"]

        append_mode_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "append_mode"
        ]
        outside = [line for line in append_mode_lines if not low <= line <= high]
        assert not outside, (
            "append_mode= appears OUTSIDE save_full_rebuild at lines "
            f"{outside}. append_mode=False is the replace-all shape; reachable from "
            "anywhere else it can replace 24 seasons of burn-in with one (T-33-12)."
        )

    def test_partition_cols_appears_nowhere(self) -> None:
        tree = self._tree()
        partition_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg == "partition_cols"
        ]
        assert not partition_lines, (
            f"partition_cols= reappeared at lines {partition_lines}. The partitioned "
            "write form wrote into the SHARED data/silver/season=YYYY/ root and "
            "appended a new file on every run -- the ~30.7x games_with_elo bloat "
            "FIX-01 / D-13 removed (T-33-16)."
        )

    def test_the_fix_01_comment_survived_the_split(self) -> None:
        source = BUILD_ELO_SOURCE.read_text(encoding="utf-8")
        assert "FIX-01" in source and "D-13" in source, (
            "the FIX-01 / D-13 comment explaining why partition_cols must never come "
            "back was dropped when save_results was split. The comment is the only "
            "record of why the partitioned form is forbidden."
        )
