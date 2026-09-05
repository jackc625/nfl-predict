"""Integration test for the end-to-end Elo pipeline to gold features.

Verifies:
- EloBuilder produces snapshots with per-game pre-game Elo
- EloFeatureBuilder reads those snapshots and produces real features
- Elo columns have variance (std > 0), no constant 1500.0 values
- The whole round trip runs inside a SANDBOX and never touches production silver

THE DEFECT THIS MODULE USED TO CARRY (Plan 31-11, owner ruling C-MODIFIED)
--------------------------------------------------------------------------
Until 2026-09-05 the test below called::

    save_dataframe(snapshots_df, "elo_game_snapshots", layer="silver", append_mode=False)

with no ``tmp_path``, no redirected root, and the comment "(mimicking the real
pipeline)". It was not mimicking the pipeline; it WAS the pipeline. Every full
``pytest tests/integration`` run rebuilt Elo from 2018 against whatever nflverse
play-by-play was live that day and overwrote the production
``data/silver/elo_game_snapshots.parquet``.

It fired at least four times during Phase 31 -- the wave-1, wave-2 and wave-3 post-merge
gates and Plan 31-10's suite -- the last at 2026-09-04 22:05:25. The next full gold
rebuild (Plan 31-11 rung 1, 2026-09-04 22:36) then carried fourteen moved Elo columns
into the 2025 slice, and the phase HARD STOPPED on them. Nothing reported the write,
because the only boundary check the phase ran was ``git status --porcelain data/``,
which ``.gitignore:22`` makes structurally incapable of failing.

The assertions below are UNCHANGED. What changed is where the bytes land: a sandbox
``ParquetManager`` and ``DuckDBConnection`` under ``tmp_path``, plus the content-based
``data_boundary_guard`` fixture, which fails this test if a single tracked file under
``data/`` moves. Weakening the assertions to dodge the write would have thrown away the
coverage; redirecting the write keeps all of it.

WHY THE SEAM IS THE MODULE GLOBALS
----------------------------------
``data.storage.save_dataframe`` takes no root argument. It resolves its destination
through the module-level singletons ``_parquet_manager`` and ``_db_connection`` (see
``get_parquet_manager`` / ``get_db_connection``, data/storage.py:674-692). Those
singletons are therefore the only injectable seam, and ``monkeypatch.setattr`` restores
them when the test ends. ``tests/integration/test_gold_write_scope.py:72`` already uses
exactly this seam, so the pattern is the house one rather than a new invention.
"""

import pytest


def _silver_has_games() -> bool:
    """Check if silver layer has games data."""
    try:
        from data.storage import load_dataframe

        df = load_dataframe("games", layer="silver")
        return len(df) > 0
    except (FileNotFoundError, OSError, ValueError):
        return False


@pytest.mark.skipif(
    not _silver_has_games(),
    reason=(
        "live silver games not present at data/silver/games.parquet -- data/ is "
        "gitignored runtime state."
    ),
)
class TestEloPipelineToGold:
    """End-to-end test: build Elo snapshots, then build features from them."""

    def test_elo_pipeline_to_gold(self, tmp_path, monkeypatch, data_boundary_guard):
        """Run EloBuilder with snapshots, then build Elo features via
        EloFeatureBuilder. Assert Elo columns have variance (std > 0),
        no constant 1500.0, and produce meaningful differentiation.

        Every write in this test lands under ``tmp_path``. ``data_boundary_guard``
        digests every tracked file under ``data/`` before and after and fails the test
        if any of them moved, so a future edit that reintroduces the production write
        is caught by this test rather than by a gold rebuild three plans later.
        """
        from datetime import datetime

        import data.storage as storage_mod
        from data.storage import (
            DuckDBConnection,
            ParquetManager,
            load_dataframe,
            save_dataframe,
        )
        from features.elo_features import ELO_FEATURE_COLUMNS, EloFeatureBuilder
        from scripts.build_elo import EloBuilder

        # Step 0: read production games ONCE, read-only, BEFORE the redirect. The Elo
        # build needs real games to produce real ratings; copying them into the sandbox
        # keeps the input real while keeping every write out of production.
        production_games = load_dataframe("games", layer="silver")
        assert len(production_games) > 0, "silver games is empty"

        sandbox = tmp_path / "data"
        sandbox.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(sandbox))
        )
        monkeypatch.setattr(
            storage_mod,
            "_db_connection",
            DuckDBConnection(str(sandbox / "sandbox.duckdb")),
        )

        # Seed the sandbox so every subsequent load_dataframe resolves inside it.
        save_dataframe(production_games, "games", layer="silver", replace_mode=True)

        # Step 1: Build Elo with snapshots
        builder = EloBuilder()
        snapshots_df = builder.build_elo_with_snapshots(start_season=2018)

        assert len(snapshots_df) > 0, (
            "build_elo_with_snapshots should produce snapshots"
        )

        # Save snapshots to the SANDBOX silver layer. The production pipeline writes
        # data/silver/elo_game_snapshots.parquet here; a test must not.
        save_dataframe(
            snapshots_df, "elo_game_snapshots", layer="silver", append_mode=False
        )

        assert (sandbox / "silver" / "elo_game_snapshots.parquet").exists(), (
            "the snapshot write did not land in the sandbox, which means the redirect "
            "did not take and the production store is the one that was written."
        )

        # Step 2: Build features using EloFeatureBuilder
        games_df = load_dataframe("games", layer="silver")
        # Pick a recent season for feature building
        season_2024 = games_df[games_df["season"] == 2024]
        if len(season_2024) == 0:
            season_2024 = games_df[games_df["season"] == games_df["season"].max()]

        feature_builder = EloFeatureBuilder()
        features = feature_builder.build_features(
            season_2024,
            as_of_datetime=datetime(2025, 3, 1, 0, 0),
            target_season=int(season_2024["season"].iloc[0]),
        )

        # Step 3: Verify Elo columns exist and have variance
        for col in ELO_FEATURE_COLUMNS:
            assert col in features.columns, f"Missing Elo feature column: {col}"

        # Elo ratings should have variance (not all 1500.0)
        home_elo_std = features["home_elo"].astype(float).std()
        away_elo_std = features["away_elo"].astype(float).std()

        assert home_elo_std > 0, (
            f"home_elo should have variance (std > 0), got std={home_elo_std}. "
            f"Values: {features['home_elo'].unique()[:5]}"
        )
        assert away_elo_std > 0, (
            f"away_elo should have variance (std > 0), got std={away_elo_std}. "
            f"Values: {features['away_elo'].unique()[:5]}"
        )

        # No constant 1500.0 values (the old bug)
        home_elo_values = features["home_elo"].astype(float)
        all_1500 = (home_elo_values == 1500.0).all()
        assert not all_1500, (
            "home_elo should not be constant 1500.0 (old batch leakage bug)"
        )

        # elo_diff should have variance too
        elo_diff_std = features["elo_diff"].astype(float).std()
        assert elo_diff_std > 0, (
            f"elo_diff should have variance, got std={elo_diff_std}"
        )

        # elo_prob_home should be in (0, 1) range
        probs = features["elo_prob_home"].astype(float)
        assert probs.min() > 0, f"elo_prob_home min should be > 0, got {probs.min()}"
        assert probs.max() < 1, f"elo_prob_home max should be < 1, got {probs.max()}"


class TestThisModuleCannotWriteProductionSilver:
    """A source-level tripwire, so a reintroduced production write fails FAST.

    The behavioural guard above only fires when the (slow, data-dependent) end-to-end
    test actually runs. On a checkout without silver games it skips, and the module
    could then be edited back to writing production with nothing objecting. This check
    runs everywhere and costs nothing.
    """

    def test_no_storage_write_happens_before_the_sandbox_redirect(self) -> None:
        from pathlib import Path

        source = Path(__file__).read_text(encoding="utf-8")
        body = source.split("def test_elo_pipeline_to_gold", 1)[1].split(
            "\nclass TestThisModuleCannotWriteProductionSilver", 1
        )[0]

        # Match on the ARGUMENT, not on the call's line layout: `ruff format` wraps
        # this call across four lines, and a scan pinned to one spelling would report a
        # missing redirect that is right there.
        redirect_at = body.find('"_parquet_manager"')
        assert redirect_at != -1, (
            "test_elo_pipeline_to_gold no longer redirects data.storage._parquet_manager "
            "onto a sandbox root. Without that redirect every save_dataframe call in "
            "this module writes data/silver/ -- the exact defect that moved fourteen "
            "Elo columns in gold during Phase 31."
        )

        first_write = body.find("save_dataframe(")
        assert first_write > redirect_at, (
            "a save_dataframe call appears BEFORE the sandbox redirect, so it writes "
            "the production silver lake. Move the redirect above every write."
        )

    def test_the_boundary_guard_fixture_is_requested(self) -> None:
        from pathlib import Path

        source = Path(__file__).read_text(encoding="utf-8")
        assert "data_boundary_guard" in source, (
            "this module must request the content-based data_boundary_guard fixture. "
            "`git status --porcelain data/` cannot substitute for it: .gitignore:22 "
            "blankets data/, so that check returns empty whether the store is intact "
            "or overwritten."
        )
