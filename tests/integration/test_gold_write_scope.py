"""CR-01: the gold write mode must follow the BUILD'S SCOPE, never the other way round.

Plan 30-07 added ``replace_mode=True`` to ``FeatureMatrixBuilder.save_feature_matrices``
so a full rebuild could NARROW the gold schema. It was passed UNCONDITIONALLY, and the
same function is reachable with a season filter: ``build_features.py --season 2025``
produces a 2025-only matrix, and writing that under replace mode makes the slice the
whole table in BOTH stores (``create_table_from_df(if_exists="replace")`` and
``pm.save``), destroying 2002-2024 gold. ``PIPELINE.md`` and ``RUNBOOK.md`` both publish
that exact command.

The bug class these tests pin:

1. A SCOPED write must MERGE -- the seasons the slice did not carry survive, and the
   rows it did carry are replaced latest-wins on ``game_id``.
2. A FULL rebuild must still REPLACE -- that is what lets a rebuild drop columns, and it
   is the Phase-30 rung-3 behaviour that must not regress in the other direction.
3. A NARROWING scoped write must be REFUSED before it writes, because the merge path
   concatenates and ``pd.concat`` UNIONS columns, which would resurrect dropped columns
   as all-null across every retained historical row.

No test previously drove ``save_feature_matrices`` with a season subset against a
populated gold table -- ``tests/integration/test_storage_replace_mode.py`` proves the
storage PRIMITIVE writes the frame it is given, which is correct and orthogonal. That
gap is why the regression shipped.

All tests use tmp_path + a monkeypatched ParquetManager and a DuckDB-free write path, so
no real data lake is touched.
"""

from __future__ import annotations

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import ParquetManager
from scripts.build_features import FeatureMatrixBuilder

GOLD_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "elo_diff",
    "target_ats",
]


def _matrix(season: int, n: int = 3) -> pd.DataFrame:
    """A minimal gold-matrix-shaped frame for one season."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_W{i + 1:02d}_BUF@KC" for i in range(n)],
            "season": [season] * n,
            "week": list(range(1, n + 1)),
            "home_team": ["KC"] * n,
            "away_team": ["BUF"] * n,
            "elo_diff": [10.0 + i for i in range(n)],
            "target_ats": [1.0 * (i % 2) for i in range(n)],
        }
    )[GOLD_COLUMNS]


@pytest.fixture
def gold_lake(tmp_path, monkeypatch):
    """An isolated parquet-only lake, with DuckDB writes disabled.

    ``save_dataframe`` writes to both stores; the DuckDB half needs no coverage here
    (it takes the SAME ``combined_df``), and disabling it keeps the test hermetic.
    """
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(tmp_path)))

    real_save = storage_mod.save_dataframe

    def _parquet_only(*args, **kwargs):
        kwargs["save_to_db"] = False
        return real_save(*args, **kwargs)

    # build_features imports save_dataframe by name, so patch it there too.
    import scripts.build_features as bf_mod

    monkeypatch.setattr(bf_mod, "save_dataframe", _parquet_only)
    monkeypatch.setattr(storage_mod, "save_dataframe", _parquet_only)

    # load_dataframe's "auto" source tries DuckDB first; force parquet so the
    # reconciliation read sees this lake and not the developer's real database.
    real_load = storage_mod.load_dataframe

    def _parquet_load(table_name, layer="silver", source="auto", **kwargs):
        return real_load(table_name, layer=layer, source="parquet", **kwargs)

    monkeypatch.setattr(bf_mod, "load_dataframe", _parquet_load)

    return tmp_path


def _read_gold(tmp_path, target: str = "ats") -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "gold" / f"features_{target}.parquet")


@pytest.mark.integration
class TestScopedGoldWritePreservesHistory:
    """A per-season build must not truncate gold (CR-01)."""

    def test_season_scoped_write_preserves_other_seasons(self, gold_lake):
        """The seasons the slice did not carry must survive the write.

        This is the regression: with ``replace_mode=True`` passed unconditionally,
        writing a 2025-only matrix left gold containing ONLY 2025.
        """
        builder = FeatureMatrixBuilder()

        history = pd.concat(
            [_matrix(2022), _matrix(2023), _matrix(2024)], ignore_index=True
        )
        builder.save_feature_matrices({"ats": history})  # full rebuild

        assert set(_read_gold(gold_lake)["season"]) == {2022, 2023, 2024}

        # Now the scoped build that used to destroy it.
        builder.save_feature_matrices({"ats": _matrix(2025)}, target_season=2025)

        after = _read_gold(gold_lake)
        assert set(after["season"]) == {2022, 2023, 2024, 2025}, (
            f"A --season 2025 build truncated gold to {sorted(set(after['season']))}. "
            "replace_mode must follow the build's scope: a scoped build carries only "
            "its slice and must MERGE, not become the whole table."
        )
        assert len(after) == len(history) + 3, (
            f"Expected {len(history) + 3} rows after merging a 3-row 2025 slice into "
            f"{len(history)} rows of history, got {len(after)}"
        )

    def test_week_scoped_write_preserves_history(self, gold_lake):
        """``--week`` alone is also a scoped build and must merge, not replace."""
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": _matrix(2024)})

        week_slice = _matrix(2025, n=1)
        builder.save_feature_matrices({"ats": week_slice}, target_week=1)

        after = _read_gold(gold_lake)
        assert set(after["season"]) == {2024, 2025}, (
            f"A --week-scoped build truncated gold to {sorted(set(after['season']))}"
        )

    def test_scoped_rewrite_of_same_games_is_latest_wins_not_duplicated(
        self, gold_lake
    ):
        """Re-running the same scoped build must replace those rows, not append them.

        The merge path drops the incoming ``game_id``s from the existing table before
        concatenating, so a repeated current-week build is idempotent in cardinality
        and takes the NEW values.
        """
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices(
            {"ats": pd.concat([_matrix(2024), _matrix(2025)], ignore_index=True)}
        )
        before_rows = len(_read_gold(gold_lake))

        updated = _matrix(2025)
        updated["elo_diff"] = [99.0, 98.0, 97.0]
        builder.save_feature_matrices({"ats": updated}, target_season=2025)
        builder.save_feature_matrices({"ats": updated}, target_season=2025)

        after = _read_gold(gold_lake)
        assert len(after) == before_rows, (
            f"Repeating a scoped build multiplied rows: {before_rows} -> {len(after)}"
        )
        refreshed = after[after["season"] == 2025].sort_values("week")
        assert list(refreshed["elo_diff"]) == [99.0, 98.0, 97.0], (
            "Scoped re-write must be latest-wins on game_id; got stale values "
            f"{list(refreshed['elo_diff'])}"
        )
        assert set(after[after["season"] == 2024]["elo_diff"]) == {10.0, 11.0, 12.0}, (
            "The untouched season's values must not move during a scoped write"
        )


@pytest.mark.integration
class TestFullRebuildStillReplaces:
    """The Phase-30 rung-3 narrowing behaviour must not regress the other way."""

    def test_full_rebuild_replaces_and_can_drop_columns(self, gold_lake):
        """A full rebuild IS the table: dropped columns must not survive it.

        Under the append path a ``pd.concat`` union would write the dropped column
        back, all-null -- the failure Plan 30-07's replace_mode exists to prevent.
        """
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices(
            {"ats": pd.concat([_matrix(2023), _matrix(2024)], ignore_index=True)}
        )
        assert "elo_diff" in _read_gold(gold_lake).columns

        narrowed = _matrix(2024).drop(columns=["elo_diff"])
        builder.save_feature_matrices({"ats": narrowed})

        after = _read_gold(gold_lake)
        assert "elo_diff" not in after.columns, (
            "A full rebuild must REPLACE, so a dropped column cannot survive it "
            "(SPEC R3 / rung-3 narrowing)"
        )
        assert set(after["season"]) == {2024}, (
            "A full rebuild's frame IS the table -- it carried only 2024, so gold "
            f"must be only 2024; got {sorted(set(after['season']))}"
        )

    def test_full_rebuild_row_count_is_stable_across_repeats(self, gold_lake):
        """A repeated full rebuild must not multiply rows (SPEC R1 re-run)."""
        builder = FeatureMatrixBuilder()
        full = pd.concat([_matrix(2023), _matrix(2024)], ignore_index=True)
        builder.save_feature_matrices({"ats": full})
        builder.save_feature_matrices({"ats": full})

        assert len(_read_gold(gold_lake)) == len(full)


@pytest.mark.integration
class TestNarrowingScopedWriteIsRefused:
    """A scoped write cannot change the schema, and says so before writing."""

    def test_scoped_write_with_missing_column_raises(self, gold_lake):
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": _matrix(2024)})

        narrowed = _matrix(2025).drop(columns=["elo_diff"])
        with pytest.raises(ValueError, match="Refusing an incremental gold write"):
            builder.save_feature_matrices({"ats": narrowed}, target_season=2025)

    def test_refusal_happens_before_the_write(self, gold_lake):
        """The refusal must leave gold untouched -- a stop, not a diagnosis."""
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": _matrix(2024)})
        before = _read_gold(gold_lake)

        with pytest.raises(ValueError):
            builder.save_feature_matrices(
                {"ats": _matrix(2025).drop(columns=["elo_diff"])}, target_season=2025
            )

        pd.testing.assert_frame_equal(_read_gold(gold_lake), before)

    def test_scoped_write_onto_absent_gold_is_allowed(self, gold_lake):
        """No history to protect means nothing to refuse -- the table is created."""
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": _matrix(2025)}, target_season=2025)

        assert set(_read_gold(gold_lake)["season"]) == {2025}


@pytest.mark.integration
class TestBuildFeaturesCliScopeFlags:
    """WR-09: the documented full-rebuild flag must exist and be unambiguous."""

    def test_all_seasons_flag_is_accepted(self):
        """``RUNBOOK.md`` section 11 publishes ``--all-seasons``; argparse must know it."""
        import scripts.build_features as bf_mod

        source = (bf_mod.__file__ or "").replace("\\", "/")
        assert source, "could not locate scripts/build_features.py"

        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, source, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        assert "--all-seasons" in proc.stdout, (
            "scripts/build_features.py does not define --all-seasons, but "
            "RUNBOOK.md section 11 instructs the operator to pass it. argparse "
            "exits 2 on the published command, and the obvious correction "
            "(--season <YEAR>) is the SCOPED build."
        )

    def test_all_seasons_and_season_are_mutually_exclusive(self):
        import subprocess
        import sys

        import scripts.build_features as bf_mod

        proc = subprocess.run(
            [sys.executable, bf_mod.__file__, "--all-seasons", "--season", "2024"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2, (
            "--all-seasons (full replace) and --season (scoped merge) select "
            "different write modes and must not be combinable; argparse should "
            f"exit 2, got {proc.returncode}"
        )
