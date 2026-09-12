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

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Plan 33-06 Task 3: the identity columns are INERT FOR GOLD, proven semantically.
#
# WHY THIS LIVES HERE. This module is where the gold-COLUMN claims live, and one
# home for them is worth more than a module per plan (Plan 33-12 owns the other
# half). The write-SCOPE arm above and this write-CONTENT arm answer the same kind
# of question about the same artifacts.
#
# WHY A SEMANTIC SCAN AND NOT A TOKEN SCAN. The obvious form -- grep `features/`
# for `stadium_id|neutral_site|season_type` and require zero hits -- is
# SELF-INVALIDATING here, because Plan 33-06 Task 2 deliberately ADDS
# `_get_venue_by_stadium_id` and `resolve_venue_for_game` to
# `features/contextual.py`, and the scan would flag this plan's own repair. Worse,
# the token form does not test the property that matters: a module may legitimately
# READ an identity column to route a lookup without that column ever becoming a
# model feature. Reading a value to decide WHICH STADIUM a game is at, and emitting
# that value as a number a model trains on, are different acts.
#
# The claim that matters is "these three names are not COLUMNS of any built model
# feature matrix", and that is asserted on the BUILT matrices' column sets.
#
# WHY IT MATTERS AT ALL. `generate_feature_matrices` derives its feature list as
# "every column that is not on the exclude list", so a column that survives
# `combine_features` BECOMES a model feature by default. The three identity columns
# are inert only because `combine_features` selects its base columns explicitly and
# does not carry them. That is a real property of real code, and it is one merge
# block away from being false -- which is exactly why it is asserted rather than
# assumed, and why the planted-violation control below exists.
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS = ("stadium_id", "neutral_site", "season_type")

# A matrix narrower than this is a build that produced nothing useful, and an
# absence claim over an empty column set is vacuously true. Asserted BEFORE the
# absence claim, so the proof cannot pass by producing no gold at all.
MINIMUM_MATRIX_COLUMNS = 20
MINIMUM_MATRIX_ROWS = 24


def _sandbox_sources(*, identity: bool = True, plant: str | None = None):
    """Feature sources in the real shape, with the identity columns on `games`.

    Two seasons of games so the expanding normalization has a prior season to
    bootstrap from, and one source per REQUIRED feature group so the LeakageGate's
    combined-matrix check passes on its own terms rather than being bypassed.
    """
    rows = []
    for season in (2023, 2024):
        for i in range(24):
            week = (i % 12) + 1
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_G{i:02d}",
                    "season": season,
                    "week": week,
                    "home_team": "KC" if i % 2 else "BUF",
                    "away_team": "BUF" if i % 2 else "KC",
                    "kickoff_et": pd.Timestamp(
                        f"{season}-09-{(i % 27) + 1:02d} 13:00",
                        tz="America/New_York",
                    ),
                    "home_score": 20 + (i % 14),
                    "away_score": 17 + (i % 11),
                    "venue": "Arrowhead Stadium",
                    "venue_roof": "outdoor",
                    "stadium_id": "KAN00",
                    "neutral_site": False,
                    "season_type": "Regular",
                }
            )
    games = pd.DataFrame(rows)
    if not identity:
        games = games.drop(columns=list(IDENTITY_COLUMNS))

    n = len(games)
    elo = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "home_elo": [1500.0 + i for i in range(n)],
            "away_elo": [1500.0 - i for i in range(n)],
            "elo_diff": [2.0 * i for i in range(n)],
            "elo_prob_home": 0.55,
            "elo_prob_away": 0.45,
            "hfa_used": 55.0,
        }
    )
    team_form = pd.DataFrame(
        [
            {
                "team": team,
                "target_season": int(row.season),
                "target_week": int(row.week),
                "side": side,
                "rolling_epa_per_play": 0.05 + (idx % 7) / 100,
            }
            for idx, row in enumerate(games.itertuples())
            for team in (row.home_team, row.away_team)
            for side in ("offense", "defense")
        ]
    ).drop_duplicates(subset=["team", "target_season", "target_week", "side"])
    weather = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "temp_f": 60.0,
            "wind_mph": 5.0,
            "precip_mm": 0.0,
            "is_outdoor": True,
            "weather_severity_score": [0.1 + (i % 5) / 10 for i in range(n)],
        }
    )
    market = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "snapshot_spread": [-3.0 + (i % 7) for i in range(n)],
            "snapshot_total": [44.0 + (i % 9) for i in range(n)],
            "snapshot_ml_prob_home_fair": 0.55,
            "spread_movement": 0.0,
            "total_movement": 0.0,
        }
    )

    sources = {
        "games": games,
        "team_form": team_form,
        "elo": elo,
        "contextual": pd.DataFrame(),
        "weather": weather,
        "market": market,
        "qb_tracking": pd.DataFrame(),
        "snaps": pd.DataFrame(),
        "injury": pd.DataFrame(),
    }

    if plant is not None:
        # THE PLANTED VIOLATION. A future edit adds an explicit merge block that
        # carries an identity column through combine_features. It is planted as a
        # SOURCE the generic contextual merge picks up, because that is the shape
        # such an edit would actually take (see the 28-06 snap/injury blocks).
        sources["contextual"] = pd.DataFrame(
            {"game_id": games["game_id"], plant: games[plant].to_numpy()}
        )

    return sources


def _build_into_sandbox(
    monkeypatch, *, identity: bool = True, plant: str | None = None
):
    """Run the REAL generate_feature_matrices over synthetic sources and save.

    ``load_all_feature_sources`` is the ONLY thing replaced -- everything from
    ``combine_features`` through the per-target split is the production code path,
    which is what makes the column sets below evidence about production rather than
    about the test's own arithmetic. The opponent adjuster is stubbed out so the
    build never reads the developer's real lake.
    """
    from datetime import UTC, datetime

    from scripts.build_features import FeatureMatrixBuilder

    # FAIL-CLOSED INTERLOCK, and it is here because the omission ALREADY HAPPENED
    # once during this plan's own execution: a test in this class forgot to request
    # `gold_lake`, `save_feature_matrices` ran against the real lake, and all three
    # production gold matrices were overwritten with 48 synthetic rows. The autouse
    # write guard reported it AFTER the fact, which is the right place for a
    # detector and the wrong place for a stop. This assertion is the stop.
    base = Path(storage_mod._parquet_manager.base_path).resolve()
    repo_data = (Path(__file__).resolve().parents[2] / "data").resolve()
    assert base != repo_data and repo_data not in base.parents, (
        f"REFUSING to build: the parquet manager still points at {base}, which is "
        "the PRODUCTION lake. This helper WRITES three gold matrices. Request the "
        "`gold_lake` fixture."
    )

    builder = FeatureMatrixBuilder()
    sources = _sandbox_sources(identity=identity, plant=plant)
    monkeypatch.setattr(
        builder, "load_all_feature_sources", lambda *a, **k: sources, raising=False
    )
    monkeypatch.setattr(
        builder.team_form_calc,
        "get_per_game_stats",
        lambda *a, **k: pd.DataFrame(),
        raising=False,
    )

    matrices = builder.generate_feature_matrices(
        as_of_datetime=datetime(2030, 1, 1, tzinfo=UTC)
    )
    builder.save_feature_matrices(matrices)
    return matrices


@pytest.mark.integration
class TestTheIdentityColumnsNeverReachAModelFeatureMatrix:
    """COLD-09 / D33-15: proven on the BUILT matrices, with both controls."""

    def test_the_three_matrices_are_built_and_non_vacuous(self, gold_lake, monkeypatch):
        """Asserted FIRST: an empty build would satisfy the absence claim for free."""
        _build_into_sandbox(monkeypatch)

        for target in ("wp", "ats", "ou"):
            frame = _read_gold(gold_lake, target)
            assert len(frame) >= MINIMUM_MATRIX_ROWS, (
                f"features_{target} has {len(frame)} rows. An absence claim over an "
                "empty matrix proves nothing."
            )
            assert len(frame.columns) >= MINIMUM_MATRIX_COLUMNS, (
                f"features_{target} has {len(frame.columns)} columns, fewer than the "
                f"{MINIMUM_MATRIX_COLUMNS} a real matrix carries."
            )

    def test_no_identity_column_is_in_any_built_matrix(self, gold_lake, monkeypatch):
        """The claim, asserted on column SETS -- not on a token search of features/."""
        _build_into_sandbox(monkeypatch)

        for target in ("wp", "ats", "ou"):
            frame = _read_gold(gold_lake, target)
            present = sorted(set(IDENTITY_COLUMNS) & set(frame.columns))
            assert not present, (
                f"identity column(s) {present!r} reached features_{target}. "
                "generate_feature_matrices treats every non-excluded column as a "
                "model feature, so an identity column that survives "
                "combine_features BECOMES one -- and Plan 33-14's expected change "
                "set for the gold rebuild would have to be re-derived."
            )

    def test_the_scan_flags_a_planted_violation(self, gold_lake, monkeypatch):
        """A scan never observed FIRING is indistinguishable from an unwired one."""
        _build_into_sandbox(monkeypatch, plant="neutral_site")

        flagged = {
            target: sorted(
                set(IDENTITY_COLUMNS) & set(_read_gold(gold_lake, target).columns)
            )
            for target in ("wp", "ats", "ou")
        }
        assert all(cols == ["neutral_site"] for cols in flagged.values()), (
            "the planted identity column was NOT flagged in every matrix: "
            f"{flagged!r}. The control proves the scan can fail; without it a green "
            "result means nothing."
        )

    def test_the_scan_does_not_flag_the_legitimate_routing_use(
        self, gold_lake, monkeypatch
    ):
        """READING stadium_id to resolve a venue is not EMITTING it as a feature.

        ``features/contextual.py`` reads ``stadium_id`` on purpose -- that is the
        whole COLD-09 repair -- and a token scan of ``features/`` would flag it. The
        semantic scan does not, because the routing use produces venue coordinates
        and a roof, never a column named ``stadium_id``. This test states that
        distinction and then demonstrates it.
        """
        from features import contextual

        source = Path(contextual.__file__ or "").read_text(encoding="utf-8")
        assert "stadium_id" in source, (
            "features/contextual.py no longer mentions stadium_id, so the "
            "no-false-positive control has nothing to control for -- the routing "
            "repair was reverted."
        )

        venue = contextual._get_venue_by_stadium_id("RIO00")
        assert venue is not None
        assert "stadium_id" in venue, "the ROUTING output carries the id, by design"

        matrices = _build_into_sandbox(monkeypatch)
        for target, frame in matrices.items():
            assert "stadium_id" not in frame.columns, (
                f"features_{target} carries stadium_id even though nothing planted "
                "it -- the routing read leaked into the feature matrix."
            )

    def test_a_build_without_the_identity_columns_produces_the_same_column_set(
        self, gold_lake, monkeypatch
    ):
        """The strongest form: the columns' PRESENCE in silver changes nothing.

        Plan 33-12 backfills these three across every season. If gold's column set
        is identical with and without them, that migration cannot move gold's shape
        -- which is the specific reassurance Plan 33-14's expected change set needs.
        """
        with_identity = _build_into_sandbox(monkeypatch)
        with_columns = {t: sorted(f.columns) for t, f in with_identity.items()}

        without_identity = _build_into_sandbox(monkeypatch, identity=False)
        without_columns = {t: sorted(f.columns) for t, f in without_identity.items()}

        assert with_columns == without_columns, (
            "adding the three identity columns to silver changed gold's column set. "
            "They are supposed to be inert for gold; if they are not, the Plan "
            "33-12 backfill is a gold-shape change and must be planned as one."
        )
