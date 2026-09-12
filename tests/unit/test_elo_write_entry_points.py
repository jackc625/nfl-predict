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

* ``save_full_rebuild(processed_games, start_season)`` -- replace everything, loudly
  and attributably, behind its own CLI flag.
* ``save_live_append(season, *, snapshots, games_with_elo, rating_history)`` -- upsert
  the three ROW tables on ``game_id``, replace the two STATE artifacts.

THE THREE FRAMES ARE NOT ONE FRAME. The three row artifacts come from three different
sources at three different grains -- snapshots from the per-game pre-game capture,
``games_with_elo`` from the merged season frame, ``elo_rating_history`` from the Elo
system's own game history. A ``save_live_append(frame, season)`` that upserted one
frame into all three would silently write the wrong rows into two of them, which is
why the signature assertion below is a first-class test rather than a style check.
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


def _live_frames(builder, season: int):
    """Run the live update and return its three frames, without persisting."""
    update = builder.update_current_season(season=season)
    return update.snapshots, update.games_with_elo, update.rating_history


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

    def test_save_live_append_takes_three_separately_named_frames(self) -> None:
        """A single-frame signature cannot serve three different grains."""
        from scripts.build_elo import EloBuilder

        parameters = inspect.signature(EloBuilder.save_live_append).parameters
        for name in ("snapshots", "games_with_elo", "rating_history"):
            assert name in parameters, (
                f"save_live_append is missing the '{name}' frame. The three row tables "
                "come from three different source frames at three different grains; a "
                "generic single-frame signature would upsert one of them into all "
                "three."
            )
        for name in ("snapshots", "games_with_elo", "rating_history"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                f"'{name}' must be KEYWORD-ONLY. Three positional frames of the same "
                "type are three frames a caller can silently transpose."
            )

    def test_the_row_and_state_constants_state_the_split_rule(self) -> None:
        from scripts.build_elo import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS

        assert ELO_ROW_TABLES == (
            "elo_game_snapshots",
            "games_with_elo",
            "elo_rating_history",
        )
        assert ELO_STATE_ARTIFACTS == ("elo_ratings_current", "elo_ratings")
        assert not set(ELO_ROW_TABLES) & set(ELO_STATE_ARTIFACTS), (
            "an artifact cannot be both accumulated history and current state"
        )

    def test_the_constants_agree_with_the_phase33_manifest(self) -> None:
        """The split rule has ONE committed home the tests import, not two."""
        from scripts.build_elo import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS
        from tests.phase33_state import ELO_ROW_TABLE_NAMES, ELO_STATE_ARTIFACT_NAMES

        assert tuple(ELO_ROW_TABLE_NAMES) == ELO_ROW_TABLES
        assert tuple(ELO_STATE_ARTIFACT_NAMES) == ELO_STATE_ARTIFACTS


class TestTheWriteSetIsDeclaredHonestly:
    """``save_dataframe`` writes BOTH stores; a write-set that omits one is a lie."""

    def test_the_write_set_names_the_shared_duckdb(self) -> None:
        from tests.phase33_state import ELO_WRITE_SET_INCLUDING_DUCKDB

        entries = list(ELO_WRITE_SET_INCLUDING_DUCKDB)
        assert len(entries) >= 5, (
            "the Elo write set has at least five members: three row tables, two state "
            f"artifacts. Got {entries}."
        )
        assert any("nfl_predictions.duckdb" in entry for entry in entries), (
            "data/storage.save_dataframe defaults save_to_db=True, so every Elo write "
            "that routes through it touches the SHARED DuckDB store as well as the "
            "parquet. A write set that names only parquet paths under-declares the "
            "blast radius of both verbs."
        )

    def test_the_write_set_names_every_row_table_and_state_artifact(self) -> None:
        from scripts.build_elo import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS
        from tests.phase33_state import ELO_WRITE_SET_INCLUDING_DUCKDB

        joined = "\n".join(ELO_WRITE_SET_INCLUDING_DUCKDB)
        for name in (*ELO_ROW_TABLES, *ELO_STATE_ARTIFACTS):
            assert name in joined, (
                f"'{name}' is written by the Elo verbs but is absent from "
                "ELO_WRITE_SET_INCLUDING_DUCKDB."
            )


class TestSaveFullRebuild:
    """Replace-everything, but only through the verb that says so."""

    def test_a_full_rebuild_writes_all_five_artifacts(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2024, weeks=2)
        builder = sandbox_builder(sandbox, games)

        processed = builder.build_all_ratings(start_season=2024)
        builder.save_full_rebuild(processed, start_season=2024)

        silver = sandbox / "silver"
        for table in ("elo_game_snapshots", "games_with_elo", "elo_rating_history"):
            assert (silver / f"{table}.parquet").exists(), f"{table} was not written"
        assert (silver / "elo_ratings_current.parquet").exists()
        assert (silver / "elo_ratings.json").exists()

    def test_a_full_rebuild_logs_an_attributed_line(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        """A full rebuild must never be mistakable for a weekly run in a log."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2024, weeks=2)
        builder = sandbox_builder(sandbox, games)
        processed = builder.build_all_ratings(start_season=2024)

        with caplog.at_level("INFO"):
            builder.save_full_rebuild(processed, start_season=2024)

        text = caplog.text
        assert "FULL REBUILD" in text, (
            "the full-rebuild write must announce itself by name; a silent "
            "replace-everything is indistinguishable from a weekly append in a log."
        )
        assert "2024" in text, "the attributed line must name the start season"


class TestSaveLiveAppend:
    """The weekly verb: three frames, three tables, upserted on game_id."""

    def test_each_frame_lands_in_its_own_table(self, tmp_path, monkeypatch) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2025, weeks=2)
        builder = sandbox_builder(sandbox, games)

        snapshots, games_with_elo, rating_history = _live_frames(builder, 2025)
        builder.save_live_append(
            2025,
            snapshots=snapshots,
            games_with_elo=games_with_elo,
            rating_history=rating_history,
        )

        written_snapshots = read_sandbox_table(sandbox, "elo_game_snapshots")
        written_games = read_sandbox_table(sandbox, "games_with_elo")
        written_history = read_sandbox_table(sandbox, "elo_rating_history")

        assert "home_elo_pre" in written_snapshots.columns, (
            "elo_game_snapshots must carry the PRE-GAME snapshot grain"
        )
        assert "home_rating_post" in written_games.columns, (
            "games_with_elo must carry the merged per-game rating updates"
        )
        assert "home_change" in written_history.columns, (
            "elo_rating_history must carry the Elo system's own game history grain"
        )
        assert len(written_snapshots) == len(snapshots)
        assert len(written_history) == len(rating_history)

    def test_a_second_live_append_does_not_grow_any_row_table(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(2025, weeks=2)
        builder = sandbox_builder(sandbox, games)

        snapshots, games_with_elo, rating_history = _live_frames(builder, 2025)
        for _ in range(2):
            builder.save_live_append(
                2025,
                snapshots=snapshots,
                games_with_elo=games_with_elo,
                rating_history=rating_history,
            )

        for table in ("elo_game_snapshots", "games_with_elo", "elo_rating_history"):
            written = read_sandbox_table(sandbox, table)
            assert len(written) == len(written.drop_duplicates(subset=["game_id"])), (
                f"{table} gained duplicate game_id rows on the second append -- the "
                "upsert is behaving like an append."
            )

        assert per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        ) == per_season_row_digests(read_sandbox_table(sandbox, "elo_game_snapshots"))

    @pytest.mark.parametrize(
        "frame_name", ["snapshots", "games_with_elo", "rating_history"]
    )
    def test_a_foreign_season_in_any_frame_is_refused_by_name(
        self, tmp_path, monkeypatch, frame_name
    ) -> None:
        """Asserted once per frame: a live append cannot rewrite another season."""
        import pandas as pd

        from scripts.build_elo import EloForeignSeasonRowsError

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = pd.concat(
            [make_season_games(2024, weeks=2), make_season_games(2025, weeks=2)],
            ignore_index=True,
        )
        builder = sandbox_builder(sandbox, games)

        snapshots, games_with_elo, rating_history = _live_frames(builder, 2025)
        frames = {
            "snapshots": snapshots,
            "games_with_elo": games_with_elo,
            "rating_history": rating_history,
        }
        poisoned = frames[frame_name].copy()
        poisoned.loc[poisoned.index[0], "season"] = 2024
        frames[frame_name] = poisoned

        with pytest.raises(EloForeignSeasonRowsError) as excinfo:
            builder.save_live_append(2025, **frames)

        message = str(excinfo.value)
        assert "2024" in message, message
        assert "2025" in message, message


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
