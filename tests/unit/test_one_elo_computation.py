"""One Elo computation, one answer on disk (Plan 33.2-05, D33.2-22, SPEC R11).

WHAT WAS DELETED, AND WHY THIS MODULE EXISTS
--------------------------------------------
``EloBuilder.build_all_ratings`` used to run the canonical chain and then a SECOND,
legacy per-season pass: it reset the rating system and re-processed each season on its
own, so ``learn_home_field_advantage`` found no prior season and fell back to 48 every
year. That pass fed four side stores -- ``games_with_elo``, ``elo_rating_history``,
``elo_ratings_current`` and ``elo_ratings.json`` -- which differed from
``elo_game_snapshots`` by up to 9.7 rating points, and ``utils/similar_games.py`` read
them. D33.2-22 (owner ratified 2026-09-21) deleted the pass, the four stores, their
writers and the reader together. This module asserts that nothing reaches any of them.

WHY THE INSTRUMENT IS AN AST SCAN AND NOT A STRIPPED-TEXT GREP
--------------------------------------------------------------
Three of the four names are DuckDB TABLE NAMES and the fourth is a FILE PATH, so in this
codebase they appear as STRING LITERALS: ``_upsert_row_table("games_with_elo", frame)``
and ``root / "elo_ratings.json"`` are the exact forms the deleted writers took. A grep
that strips ``tokenize.STRING`` tokens -- to stop the mandated both-seams comment in
``scripts/build_elo.py`` tripping it -- would make exactly those two leftovers
INVISIBLE: it would remove the false failure and take the true positive with it, and
the scan would pass on precisely the leftover it exists to catch.

So the scan reads the PARSED tree. A string constant EQUAL to a deleted name is a hit
(so the call-site literals above are caught), and an identifier -- attribute, name,
imported name, parameter -- equal to one is a hit (so ``update.games_with_elo`` is
caught). Comments are not in the AST at all, and a docstring is one constant whose value
equals no table name, so the explanation the deletion requires and a passing scan stay
compatible. The two planted violations below are written in the literal-in-a-call form
for that reason.

FOUR CONTROLS, following ``tests/unit/test_freeze_parse_single_source.py``: non-vacuity
(the scanned name set has a declared length of five, and the scan parses a non-empty
file set), the assertion, a planted violation per name, and a no-false-positive case
built in the shape ``scripts/build_elo.py`` takes after the deletion.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The production roots the plan's consumer search covers. ``tests/`` is excluded here
# by design: test modules legitimately NAME the deleted artifacts in order to assert
# their absence, and this scan is about production code reaching them.
PRODUCTION_ROOTS: tuple[str, ...] = (
    "scripts",
    "features",
    "backtest",
    "models",
    "utils",
    "conf",
    "api",
    "pipeline",
    "data",
    "ratings",
    "audit",
    "web",
)

# The four deleted artifacts, matched by EQUALITY against string constants and
# identifiers.
DELETED_TABLE_NAMES: frozenset[str] = frozenset(
    {"games_with_elo", "elo_rating_history", "elo_ratings_current", "elo_ratings.json"}
)

# The deleted reader, matched as a SUBSTRING of identifiers and module names so
# ``utils.similar_games`` and ``get_similar_games_engine`` are both caught.
DELETED_READER_NAME: str = "similar_games"

# The whole scanned vocabulary: four tables plus the reader. Its length is asserted.
SCANNED_NAMES: tuple[str, ...] = (*sorted(DELETED_TABLE_NAMES), DELETED_READER_NAME)

# ``audit/elo_replay.py`` names the four tables in a DOCSTRING that forbids reading
# them; that is prose, not a reference, and the AST treats it as one unequal constant.
# No module is exempted: the no-exemption rule is what makes the scan's answer mean
# something.


def _string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _identifiers(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.Import):
            found.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(alias.asname or alias.name for alias in node.names)
    return found


def legacy_elo_references(source: str) -> list[str]:
    """Every reference in *source* to a deleted Elo artifact or the deleted reader.

    The importable form every control below drives, so the planted violations exercise
    this code path rather than a copy of its logic.
    """
    tree = ast.parse(source)
    constants = _string_constants(tree)
    identifiers = _identifiers(tree)
    hits = sorted(
        f"constant:{name}" for name in constants & DELETED_TABLE_NAMES
    ) + sorted(f"identifier:{name}" for name in identifiers & DELETED_TABLE_NAMES)
    hits += sorted(
        f"reader:{name}" for name in identifiers if DELETED_READER_NAME in name
    )
    return hits


def _production_files() -> list[Path]:
    return sorted(
        path
        for root in PRODUCTION_ROOTS
        if (REPO_ROOT / root).is_dir()
        for path in (REPO_ROOT / root).rglob("*.py")
    )


class TestNoProductionModuleReachesADeletedArtifact:
    """The four controls, then the assertion over every production module."""

    def test_the_scanned_vocabulary_has_five_names(self) -> None:
        """NON-VACUITY: four deleted tables plus the deleted reader."""
        assert len(SCANNED_NAMES) == 5, SCANNED_NAMES
        assert len(DELETED_TABLE_NAMES) == 4

    def test_the_scan_parses_a_non_empty_production_file_set(self) -> None:
        """NON-VACUITY: a scan over zero files reports green while proving nothing."""
        files = _production_files()
        assert len(files) > 100, f"only {len(files)} production files were found"
        names = {path.relative_to(REPO_ROOT).as_posix() for path in files}
        assert "scripts/build_elo.py" in names
        assert "scripts/elo_generation.py" in names
        assert "pipeline/steps.py" in names

    def test_no_production_module_references_a_deleted_artifact(self) -> None:
        violations: dict[str, list[str]] = {}
        for path in _production_files():
            hits = legacy_elo_references(path.read_text(encoding="utf-8"))
            if hits:
                violations[path.relative_to(REPO_ROOT).as_posix()] = hits
        assert violations == {}, (
            "production code still reaches an Elo artifact D33.2-22 deleted:\n"
            + "\n".join(f"  - {name}: {hits}" for name, hits in violations.items())
        )

    @pytest.mark.parametrize(
        ("name", "planted"),
        [
            # LOAD-BEARING: the literal-in-a-call form the deleted writers took.
            ("games_with_elo", 'builder._upsert_row_table("games_with_elo", frame)\n'),
            ("elo_ratings.json", 'path = silver_root / "elo_ratings.json"\n'),
            (
                "elo_rating_history",
                'save_dataframe(history, "elo_rating_history", layer="silver")\n',
            ),
            (
                "elo_ratings_current",
                'load_dataframe("elo_ratings_current", layer="silver")\n',
            ),
            (DELETED_READER_NAME, "from utils.similar_games import engine\n"),
        ],
        ids=lambda value: str(value).replace("\n", "")[:40],
    )
    def test_a_planted_violation_of_each_name_is_flagged(
        self, name: str, planted: str
    ) -> None:
        """PLANTED VIOLATION, one per scanned name."""
        hits = legacy_elo_references(planted)
        assert any(name in hit for hit in hits), (
            f"the scan did not flag a planted reference to {name!r}: {planted!r}"
        )

    def test_a_surviving_attribute_access_is_flagged(self) -> None:
        """The exact form pipeline/steps.py carried: ``update.games_with_elo``."""
        planted = "builder.save_live_append(s, games_with_elo=update.games_with_elo)\n"
        assert "identifier:games_with_elo" in legacy_elo_references(planted)

    def test_no_false_positive_on_comments_and_docstrings(self) -> None:
        """NO FALSE POSITIVE: the post-deletion shape of scripts/build_elo.py passes.

        A module whose COMMENT and DOCSTRING name all four tables and the reader is not
        flagged -- that is the both-seams explanation D33.2-22 requires, and the reason
        a stripped-text grep was rejected.
        """
        clean = textwrap.dedent(
            '''
            """Elo builder.

            games_with_elo, elo_rating_history, elo_ratings_current and
            elo_ratings.json were deleted with utils/similar_games.py (D33.2-22).
            """

            # D33.2-22: games_with_elo / elo_rating_history / elo_ratings_current /
            # elo_ratings.json and similar_games are gone. Restore both seams or none.
            ELO_SNAPSHOT_TABLE = "elo_game_snapshots"


            def build(snapshots):
                return snapshots
            '''
        )
        assert legacy_elo_references(clean) == []


class TestTheLegacyPassIsUnreachable:
    """``build_all_ratings`` runs one pass and no JSON state writer survives."""

    def test_build_all_ratings_does_not_call_the_legacy_pass(self) -> None:
        from scripts.build_elo import EloBuilder

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(EloBuilder.build_all_ratings))
        )
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert "process_seasons_chronologically" not in called
        assert "EloRatingSystem" not in called, (
            "build_all_ratings resets the rating system again -- the first step of the "
            "legacy second pass."
        )
        assert "build_elo_with_snapshots" in called, (
            "build_all_ratings no longer runs the canonical chain at all."
        )

    def test_the_builder_has_no_legacy_pass(self) -> None:
        from scripts.build_elo import EloBuilder

        assert not hasattr(EloBuilder, "process_seasons_chronologically")

    def test_the_rating_system_exposes_no_json_state_writer_or_reader(self) -> None:
        from ratings.elo import EloRatingSystem

        assert not hasattr(EloRatingSystem, "save_ratings")
        assert not hasattr(EloRatingSystem, "load_ratings")

    def test_the_per_season_primitive_is_kept(self) -> None:
        """The OPPOSITE guard: only the builder's plural wrapper went."""
        from ratings.elo import EloRatingSystem

        assert hasattr(EloRatingSystem, "process_season_chronologically")

    def test_the_live_update_is_snapshots_only(self) -> None:
        import dataclasses

        from scripts.build_elo import LiveSeasonUpdate

        fields = tuple(field.name for field in dataclasses.fields(LiveSeasonUpdate))
        assert fields == ("season", "snapshots"), fields

    def test_the_reader_module_is_gone(self) -> None:
        assert not (REPO_ROOT / "utils" / "similar_games.py").exists()
