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

import os
import re
import shutil
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import ParquetManager
from features.contextual import NO_PRIOR_GAME_SIGNATURE
from features.injury import InjuryBuilder
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder
from scripts.build_features import FeatureMatrixBuilder
from tests import phase33_state

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

    # The Elo provenance supplier reads silver `games` and `elo_game_snapshots` through
    # features.elo_features' OWN `load_dataframe` binding. Left on "auto" it tries DuckDB
    # first and reads the developer's REAL database, so a sandbox build would date its
    # synthetic games against production snapshots (Plan 33.2-04: that is exactly how the
    # identity-column tests came to see "provenance matches ZERO game ids").
    import features.elo_features as elo_features_mod

    monkeypatch.setattr(elo_features_mod, "load_dataframe", _parquet_load)

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
class TestALiveSkipNeverDeletesAHistoricalGame:
    """33.2 review C1 CR-06 = B WR-01: a history refusal saves nothing but tomorrow's rows.

    D33.2-05: a history build that meets a refusal saves nothing, and the slate is predicted
    from the last good gold plus tomorrow's rows. The live full build used to be written in
    replace mode without the refused game, silently deleting it from the training table.
    """

    @staticmethod
    def _stored_history() -> pd.DataFrame:
        return pd.concat([_matrix(2011), _matrix(2025)], ignore_index=True)

    def test_the_stored_history_is_kept_and_only_tomorrows_rows_are_written(
        self, gold_lake
    ):
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": self._stored_history()})
        refused = "2011_W02_BUF@KC"
        tomorrow = _matrix(2026, n=2)

        # Tonight's build: the refused 2011 game excluded, history re-normalized, tomorrow in.
        tonight = pd.concat([self._stored_history(), tomorrow], ignore_index=True).loc[
            lambda f: f["game_id"] != refused
        ]
        tonight["elo_diff"] = tonight["elo_diff"] + 1000.0  # every re-built value moved
        builder.save_feature_matrices(
            {"ats": tonight},
            history_refused=frozenset({refused}),
            serve_game_ids=frozenset(tomorrow["game_id"]),
        )

        after = _read_gold(gold_lake).set_index("game_id")
        assert refused in after.index, (
            "the refused historical game was deleted from gold"
        )
        stored = self._stored_history().set_index("game_id")
        for game_id in stored.index:
            assert after.loc[game_id, "elo_diff"] == stored.loc[game_id, "elo_diff"], (
                f"history row {game_id} was rewritten by a build that met a refusal"
            )
        for game_id in tomorrow["game_id"]:
            assert after.loc[game_id, "elo_diff"] >= 1000.0, (
                "tomorrow's row not written"
            )
        assert len(after) == len(stored) + len(tomorrow)

    def test_when_the_columns_differ_the_refused_rows_are_carried_forward(
        self, gold_lake
    ):
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": self._stored_history()})
        refused = "2011_W02_BUF@KC"
        tonight = (
            pd.concat([self._stored_history(), _matrix(2026, n=1)], ignore_index=True)
            .loc[lambda f: f["game_id"] != refused]
            .assign(new_feature=1.0)
        )
        builder.save_feature_matrices(
            {"ats": tonight},
            history_refused=frozenset({refused}),
            serve_game_ids=frozenset({"2026_W01_BUF@KC"}),
        )

        after = _read_gold(gold_lake).set_index("game_id")
        assert refused in after.index, "the refused game was deleted from gold"
        assert "new_feature" in after.columns
        assert len(after) == len(tonight) + 1

    def test_a_clean_night_still_replaces_the_table(self, gold_lake):
        builder = FeatureMatrixBuilder()
        builder.save_feature_matrices({"ats": self._stored_history()})
        builder.save_feature_matrices({"ats": _matrix(2026)})
        assert set(_read_gold(gold_lake)["season"]) == {2026}


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


# The first Sunday of each sandbox season. Every sandbox game kicks off at 13:00 ET on
# its week's Sunday, so week order IS time order -- which the information-time gate
# requires of any honest schedule (a week-1 game dated after a week-2 game would make
# the Elo rank columns read a result from after the week-1 lock).
_SANDBOX_FIRST_SUNDAY: dict[int, str] = {2023: "2023-09-10", 2024: "2024-09-08"}


def _sandbox_games() -> pd.DataFrame:
    """Two seasons, twelve weeks, two games a week between two disjoint team pairs."""
    rows = []
    for season in (2023, 2024):
        first_sunday = pd.Timestamp(
            f"{_SANDBOX_FIRST_SUNDAY[season]} 13:00", tz="America/New_York"
        )
        for i in range(24):
            week = i // 2 + 1
            pair = ("BUF", "KC") if i % 2 == 0 else ("MIA", "NYJ")
            home, away = pair if week % 2 else pair[::-1]
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_G{i:02d}",
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": first_sunday + pd.Timedelta(weeks=week - 1),
                    "home_score": 20 + (i % 14),
                    "away_score": 17 + (i % 11),
                    "venue": "Arrowhead Stadium",
                    "venue_roof": "outdoor",
                    "stadium_id": "KAN00",
                    "neutral_site": False,
                    "season_type": "Regular",
                }
            )
    return pd.DataFrame(rows)


def _honest_elo_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """The sandbox games' Elo snapshots, produced by the REAL canonical chain.

    Not invented numbers: ``EloBuilder._process_chain`` walks the sandbox games exactly as
    it walks production silver, so every row is a pre-game capture of real sandbox
    results. That is what lets the information-time gate date each row (Plan 33.2-01) and
    value-check the undatable first-week rows against the start state -- the coverage
    rule is satisfied, never relaxed.
    """
    from scripts.build_elo import EloBuilder, build_snapshot_frame

    rows = EloBuilder()._process_chain(
        sorted(int(s) for s in games["season"].unique()),
        games=games,
        learn_from=games,
    )
    return build_snapshot_frame(rows)


def _elo_source_from(snapshots: pd.DataFrame) -> pd.DataFrame:
    """The Elo feature source, carrying the snapshot values the gold join carries."""
    return pd.DataFrame(
        {
            "game_id": snapshots["game_id"],
            "home_elo": snapshots["home_elo_pre"],
            "away_elo": snapshots["away_elo_pre"],
            "home_elo_uncertainty": snapshots["home_elo_uncertainty"],
            "away_elo_uncertainty": snapshots["away_elo_uncertainty"],
            "elo_diff": snapshots["home_elo_pre"] - snapshots["away_elo_pre"],
            "elo_prob_home": snapshots["elo_prob_home"],
            "elo_prob_away": 1.0 - snapshots["elo_prob_home"],
            "hfa_used": snapshots["hfa_used"],
        }
    )


def _sandbox_silver_weather(games: pd.DataFrame) -> pd.DataFrame:
    """One archived day-before bulletin row per game, the values the weather source carries.

    Since Plan 33.2-12 the weather builder is an information-time supplier: it dates each
    game from the silver ``weather`` row its one fence selected, so a sandbox that seeds the
    weather FEATURES must also seed the silver rows they came from. Each is issued at the
    12 UTC cycle of the ET day before kickoff, before the game's 18:00 ET lock.
    """
    kickoffs = pd.to_datetime(games["kickoff_et"], utc=True)
    issued = kickoffs.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(
        None
    ) - pd.Timedelta(hours=12)
    issued = issued.dt.tz_localize("UTC")
    return pd.DataFrame(
        {
            "game_id": games["game_id"].to_numpy(),
            "forecast_time": issued.to_numpy(),
            "game_time": kickoffs.to_numpy(),
            "forecast_issue_time": issued.to_numpy(),
            "weather_source": "historical_forecast",
            "is_outdoor": True,
            "weather_coverage": True,
            "temp_f": 60.0,
            "wind_mph": 5.0,
            "humidity_pct": 50.0,
        }
    )


def _sandbox_silver_odds(games: pd.DataFrame) -> pd.DataFrame:
    """One consensus odds row per game, CAPTURED the day before its lock.

    Since Plan 33.2-14 the market builder is an information-time supplier: a line counts for
    a game only with a recorded capture time (``created_at``) at or before its lock (owner
    ruling 2026-09-22), and the supplier dates each game from the silver ``odds_snapshot``
    rows it admitted. A sandbox that seeds the market FEATURES must also seed the silver
    rows they came from, each captured 24 hours before the game's 18:00 ET lock.
    """
    import utils.game_lock as lock_rule

    locks = lock_rule.lock_frame(games[["game_id", "kickoff_et"]])
    captured = pd.to_datetime(
        [locks[str(g)] - pd.Timedelta(days=1) for g in games["game_id"]], utc=True
    )
    return pd.DataFrame(
        {
            "game_id": games["game_id"].to_numpy(),
            "sportsbook": "consensus",
            "snapshot_ts": [str(locks[str(g)]) for g in games["game_id"]],
            "created_at": captured,
            "ml_home": -150.0,
            "ml_away": 130.0,
            "spread": -3.0,
            "total": 44.0,
        }
    )


def _sandbox_team_form(games: pd.DataFrame) -> pd.DataFrame:
    """The silver ``team_form_features`` shape: one row per (team, side, target week).

    Only the SECOND sandbox season carries a row. A rolling value exists only once a
    team has an earlier game whose result was known at the target's lock (Plan 33.2-14's
    team-form provenance checks exactly that), and the first season's week-1 games have
    none -- a row there would be a value built from nothing.

    IT CARRIES A SECOND METRIC (Plan 33.2-20), and that is not decoration. The
    opponent-adjusted merge REPLACES the raw EPA columns: it drops
    ``{side}_{off,def}_rolling_epa_per_play`` once the adjusted family lands. With
    ``rolling_epa_per_play`` as the fixture's ONLY metric, team_form's entire arrival set
    left the final matrices and the merge assertion refused the build -- correctly, on a
    frame that was unrealistically thin. Production silver team form carries about twenty
    rolling metrics and only the three raw EPA ones are replaced, so the sandbox now
    carries a metric that SURVIVES, exactly as production does.
    """
    later = games[games["season"] == games["season"].max()]
    return pd.DataFrame(
        [
            {
                "team": team,
                "target_season": int(row.season),
                "target_week": int(row.week),
                "side": side,
                "rolling_epa_per_play": 0.05 + (idx % 7) / 100,
                "rolling_success_rate": 0.44 + (idx % 5) / 100,
            }
            for idx, row in enumerate(later.itertuples())
            for team in (row.home_team, row.away_team)
            for side in ("offense", "defense")
        ]
    ).drop_duplicates(subset=["team", "target_season", "target_week", "side"])


#: The families a SYNTHETIC lake carries no source data for, and whose provenance
#: therefore has to be DECLARED rather than derived from a silver table that is not there.
#:
#: ``contextual`` is in the list even though its supplier needs no silver table, because
#: one of these tests builds from a games frame with the IDENTITY COLUMNS REMOVED -- that
#: is its whole subject -- and a frame with no ``stadium_id`` cannot resolve a venue, so
#: the real supplier refuses it by name (``UnknownStadiumError``). What that would be
#: testing is venue resolution, which ``tests/unit/test_builder_lock_cutoffs.py`` owns;
#: what THIS module tests is which columns reach a model matrix.
DECLARED_ABSENT_FAMILIES: tuple[str, ...] = (
    "contextual",
    "qb_tracking",
    "snaps",
    "injury",
)


def _declared_absent_frame(
    games: pd.DataFrame,
    signature: Mapping[str, float | None],
    *,
    per_team: bool = False,
) -> pd.DataFrame:
    """One row per game (or per TEAM per game) at the builder's own declared unknown values.

    The values come from the builder's ``no_information_signature()`` and are never
    retyped here, so a builder that changes what its unknown looks like changes this
    fixture with it rather than silently disagreeing with it.

    Args:
        games: The sandbox games frame.
        signature: The builder's declared ``column -> value`` unknown.
        per_team: Emit two rows per game carrying ``team``, the shape ``combine_features``
            expects from the QB source.
    """
    if per_team:
        frame = pd.DataFrame(
            {
                "game_id": [*games["game_id"], *games["game_id"]],
                "team": [*games["home_team"], *games["away_team"]],
            }
        )
    else:
        frame = pd.DataFrame({"game_id": games["game_id"].to_numpy()})
    for column, value in signature.items():
        frame[column] = np.nan if value is None else value
    return frame


def _declared_absent_provenance(games: pd.DataFrame) -> pd.DataFrame:
    """Every game undatable, with a NULL time -- exactly one row per game."""
    return pd.DataFrame(
        {
            "game_id": games["game_id"].to_numpy(),
            "basis": "no_information",
            "information_time": pd.Series([None] * len(games), dtype="object"),
        }
    )


def declare_absent_suppliers(builder, sources, monkeypatch) -> None:
    """Point the data-less families' suppliers at their honest, CHECKED declaration.

    NOT AN EXEMPTION, and the difference is the whole point. The gate still runs on each
    of these sources: it checks two-way coverage against the lock frame, refuses a
    duplicate or a missing row, and VALUE-CHECKS every declared ``no_information`` row
    against the builder's own signature. What is stubbed is only where the provenance
    comes from -- a synthetic lake has no silver snap_counts, injuries or play-by-play for
    the real supplier to read.
    """
    from scripts.build_features import SUPPLIER_ATTRIBUTES

    provenance = _declared_absent_provenance(sources["games"])
    for family in DECLARED_ABSENT_FAMILIES:
        supplier = getattr(builder, SUPPLIER_ATTRIBUTES[family])
        monkeypatch.setattr(
            supplier,
            "information_times",
            lambda games_df, _p=provenance, **kwargs: _p.copy(),
        )


def canonical_sandbox_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """The sandbox games under their CANONICAL ids (the sandbox's own ids are opaque).

    ``features.opponent_adj`` resolves every play-by-play id through the project's one
    converter, which cannot read ``2023_W01_G00``. The adjuster is therefore given the
    same games under the ids the converter produces.
    """
    return games.assign(
        game_id=[
            f"{s}_W{w:02d}_{a}@{h}"
            for s, w, a, h in zip(
                games["season"],
                games["week"],
                games["away_team"],
                games["home_team"],
                strict=True,
            )
        ]
    )


def sandbox_per_game_stats(games: pd.DataFrame) -> pd.DataFrame:
    """A flat per-game EPA pool for the sandbox's own games, one row per team per side.

    PLAN 33.2-20: this replaces an EMPTY frame. An empty per-game pool leaves the
    opponent-adjusted family checked as ``empty_unchecked`` and merged nowhere, and from
    Plan 33.2-20 that refuses the build -- correctly, because gold written without the
    family would be missing twelve columns with nothing saying so. The pool is flat (every
    EPA 0.0) because these tests assert about COLUMN SETS and identity leakage, not about
    adjustment values.
    """
    canonical = canonical_sandbox_schedule(games)
    rows = []
    for game in canonical.to_dict("records"):
        pbp_id = (
            f"{game['season']}_{game['week']:02d}_"
            f"{game['away_team']}_{game['home_team']}"
        )
        for team in (game["home_team"], game["away_team"]):
            for side in ("offense", "defense"):
                rows.append(
                    {
                        "game_id": pbp_id,
                        "season": game["season"],
                        "week": game["week"],
                        "team": team,
                        "side": side,
                        "epa_per_play": 0.0,
                        "pass_epa_per_play": 0.0,
                        "rush_epa_per_play": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _sandbox_sources(*, identity: bool = True, plant: str | None = None):
    """Feature sources in the real shape, with the identity columns on `games`.

    Two seasons of games so the expanding normalization has a prior season to
    bootstrap from, and one source per REQUIRED feature group so the LeakageGate's
    combined-matrix check passes on its own terms rather than being bypassed. The Elo
    source is built from honest snapshots of these games (``_honest_elo_snapshots``);
    ``_build_into_sandbox`` seeds the same snapshots into the sandbox silver layer so the
    information-time gate can date every Elo row.
    """
    games = _sandbox_games()
    elo = _elo_source_from(_honest_elo_snapshots(games))
    if not identity:
        games = games.drop(columns=list(IDENTITY_COLUMNS))

    n = len(games)
    team_form = _sandbox_team_form(games)
    weather = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "temp_f": 60.0,
            "wind_mph": 5.0,
            "precip_mm": 0.0,
            "is_outdoor": True,
            "weather_severity_score": [0.1 + (i % 5) / 10 for i in range(n)],
            # Every sandbox game carries an OBSERVED reading (the values above), so its
            # coverage flag is 1.0. The full builder's gold path refuses a merged
            # weather frame without this flag (Phase 33.1 WR-10) -- which is what had
            # these tests red BEFORE Phase 33.2's Elo gate ever ran.
            "weather_coverage": 1.0,
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

    # PLAN 33.2-20: A SYNTHETIC LAKE'S DATA-LESS FAMILIES DECLARE THEMSELVES.
    #
    # These four used to be EMPTY frames here. From Plan 33.2-20 an empty source frame
    # REFUSES the build -- ``_SOURCE_LOAD_ERRORS`` turns a FAILED load into an empty frame
    # too, so a silent skip lets a broken source read as a clean pass -- and the merge
    # assertion refuses a ``merged`` key that contributed no column either way.
    #
    # THE FIX IS NOT TO EXEMPT THEM. It is to say the true thing in the form the contract
    # provides: this lake has no snap-count, injury or play-by-play data, so every game is
    # the honest unknown. Each frame carries one row per game with every value at its own
    # BUILDER'S declared ``no_information_signature`` value -- read from the builder, never
    # retyped here -- and ``_declare_absent_suppliers`` gives each one an all-
    # ``no_information`` provenance frame. The gate then checks them exactly as it checks
    # every other source: two-way coverage, one row per game, and the declared values
    # against the frame. Nothing is skipped and nothing is excused.
    contextual = _declared_absent_frame(games, NO_PRIOR_GAME_SIGNATURE)
    qb_tracking = _declared_absent_frame(
        games, QBTracker().no_information_signature(), per_team=True
    )
    snap_builder = SnapCountBuilder()
    snaps = _declared_absent_frame(games, snap_builder.no_information_signature())
    injury = _declared_absent_frame(
        games, InjuryBuilder(snap_builder=snap_builder).no_information_signature()
    )

    sources = {
        "games": games,
        "team_form": team_form,
        "elo": elo,
        "contextual": contextual,
        "weather": weather,
        "market": market,
        "qb_tracking": qb_tracking,
        "snaps": snaps,
        "injury": injury,
    }

    if plant is not None:
        # THE PLANTED VIOLATION. A future edit adds an explicit merge block that
        # carries an identity column through combine_features. It is planted as a
        # SOURCE the generic contextual merge picks up, because that is the shape
        # such an edit would actually take (see the 28-06 snap/injury blocks).
        #
        # Since Plan 33.2-14 the contextual source is CHECKED by the information-time
        # gate, which value-checks every game whose contextual builder read no prior
        # game against the declared no-information values. The planted frame carries
        # those values so the gate accepts it and the plant reaches combine_features.
        sources["contextual"] = pd.DataFrame(
            {
                "game_id": games["game_id"],
                plant: games[plant].to_numpy(),
                **NO_PRIOR_GAME_SIGNATURE,
            }
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

    # Seed the sandbox SILVER layer with the games and their honest Elo snapshots: the
    # Elo information-time supplier dates each Elo row from these two tables, and a
    # sandbox with neither has nothing to date the Elo source by.
    seeded_games = _sandbox_games()
    storage_mod._parquet_manager.save(seeded_games, "silver/games.parquet")
    storage_mod._parquet_manager.save(
        _honest_elo_snapshots(seeded_games), "silver/elo_game_snapshots.parquet"
    )
    storage_mod._parquet_manager.save(
        _sandbox_silver_weather(seeded_games), "silver/weather.parquet"
    )
    # The market supplier dates each game from the silver odds rows captured at or before
    # its lock (Plan 33.2-14, owner ruling 2026-09-22).
    storage_mod._parquet_manager.save(
        _sandbox_silver_odds(seeded_games), "silver/odds_snapshot.parquet"
    )
    # The team-form supplier reads the silver rolling table to know which teams carry a
    # row for each game (Plan 33.2-14).
    storage_mod._parquet_manager.save(
        _sandbox_team_form(seeded_games), "silver/team_form_features.parquet"
    )

    builder = FeatureMatrixBuilder()
    sources = _sandbox_sources(identity=identity, plant=plant)
    # load_all_feature_sources lays team form out one row per game (Plan 33.2-14);
    # the replaced loader hands over the same shape.
    sources["team_form"] = builder._team_form_per_game(
        sources["team_form"], sources["games"]
    )
    monkeypatch.setattr(
        builder, "load_all_feature_sources", lambda *a, **k: sources, raising=False
    )
    # PLAN 33.2-20: a REAL per-game pool, not an empty frame. An empty pool leaves the
    # opponent-adjusted family merged nowhere and reported ``empty_unchecked``, which the
    # armed gate refuses -- correctly, since gold written without it would be twelve
    # columns short with nothing saying so. The adjuster gets the sandbox games under their
    # canonical ids, because it resolves every play-by-play id through the project's one
    # converter and the sandbox's own ids are opaque to it.
    from features.opponent_adj import OpponentAdjuster

    builder.opponent_adj = OpponentAdjuster(
        window=10,
        min_opponent_games=4,
        schedule_df=canonical_sandbox_schedule(seeded_games),
    )
    monkeypatch.setattr(
        builder.team_form_calc,
        "get_per_game_stats",
        lambda *a, **k: sandbox_per_game_stats(seeded_games),
        raising=False,
    )
    declare_absent_suppliers(builder, sources, monkeypatch)

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


# ---------------------------------------------------------------------------
# Plan 33-12 Task 2(a): P6 / Q-01 -- the identity migration moves no gold COLUMN,
# measured on a REAL full-history build from the migrated 16-column silver.
#
# WHY THIS LIVES HERE, BESIDE PLAN 33-06'S SEMANTIC SCAN. That scan proves the
# three identity names never become model features, using SYNTHETIC sources. It
# cannot say what the real matrices are WIDE, and the width is the number Plan
# 33-14's expected change set is anchored on. This is the other half of the same
# claim, on the same artifacts, in the same module (Plan 33-06's own block above
# says Plan 33-12 owns it).
#
# THE MEASUREMENT IS TAKEN **BEFORE** THE ELO RE-DERIVATION, DELIBERATELY. If the
# widths hold here, then whatever Plan 33-14's rebuild moves in the gold SCHEMA is
# attributable to the Elo change and not to this migration. IF THIS TEST FAILS,
# the "identity columns are INERT for gold" claim is WITHDRAWN and the rebuild
# needs a second, separately attributed rung -- exactly as 33-12-PLAN.md:207 says.
#
# WHAT WAS MEASURED, AND THE HONEST SHAPE OF THE RESULT. A full-history build from
# the migrated silver into a sandbox produced 194 / 195 / 194 columns and a column
# SET identical to production gold's, while 132 of those columns' VALUES differ
# from production gold. Zero columns added, zero removed. So the migration is
# schema-inert and value-active, and calling it simply "inert" would have been the
# comfortable half of the truth. The 132 is an UPPER BOUND on this migration's
# value effect, not an attribution: production gold predates several other pending
# input corrections, and separating them is Plan 33.1-07's and Plan 33-14's job,
# which is precisely why each runs as its own declared rung.
#
# WHY THE HEAVY ARM IS OPT-IN. The full-history build takes 577 s measured. The
# suite already runs about 23 minutes and the owner's standing instruction is to
# measure one thing rather than verify broadly, so a ten-minute tax on every run
# would buy a re-measurement of a number that cannot drift without some other test
# in this file going red first. The instrument is COMMITTED and re-runnable by
# name; what is skipped is paying for it on every unrelated run.
# ---------------------------------------------------------------------------

FULL_BUILD_ENV_FLAG = "NFL_RUN_FULL_GOLD_BUILD"

FULL_BUILD_SKIP_MESSAGE = (
    "the full-history sandbox gold build is OPT-IN: it takes about 577 s (measured "
    "2026-09-14) and the suite is already ~23 minutes. Run it by name with "
    "NFL_RUN_FULL_GOLD_BUILD=1 uv run python -m pytest "
    "tests/integration/test_gold_write_scope.py -k full_history -q . The values it "
    "produces are recorded in tests.phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD "
    "and SANDBOX_GOLD_DIGESTS_33_12, and the always-on tests in this class check "
    "those recorded values against today's production gold."
)

GOLD_TARGETS = ("wp", "ats", "ou")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _production_gold_widths() -> tuple[int, ...]:
    """Widths read through ``scripts/fingerprint_gold.py`` -- Plan 33-14's instrument.

    The same tool on both sides is the point: a width this plan measures with one
    instrument and the next plan compares with another is a comparison of
    instruments.
    """
    from scripts.fingerprint_gold import fingerprint_gold

    document = fingerprint_gold(Path("data"))
    return tuple(
        len(document[f"features_{target}"]["columns"]) for target in GOLD_TARGETS
    )


def _production_gold_digests() -> dict[str, str]:
    from tests.data_boundary import digest_file

    return {
        target: digest_file(Path("data") / "gold" / f"features_{target}.parquet")
        for target in GOLD_TARGETS
    }


@pytest.mark.integration
class TestTheIdentityMigrationMovesNoGoldColumn:
    """COLD-09 / P6 / Q-01, always-on half: the recorded reference is still true."""

    def test_the_recorded_reference_matches_todays_production_gold(self) -> None:
        """The committed reference and the live store must not drift apart silently.

        RE-ANCHORED by Plan 33-14 Task 5, which is the plan
        ``GOLD_REBUILD_NEWLY_RED_REASONS`` assigns this reconciliation to by
        name. This test was left DELIBERATELY RED by Plan 33.1-08, and its own
        message said not to adjust one number to match the other without
        deciding which is wrong.

        THE DECISION: the REFERENCE was stale, not gold.
        ``GOLD_WIDTHS_BEFORE_ELO_REBUILD`` records (194, 195, 194), which was
        true when Plan 33-12 measured it and stopped being true when Phase
        33.1's rung 1 added ``weather_coverage``. The one-column delta IS that
        flag. So the constant is a PRE-33.1 HISTORICAL RECORD rather than a
        mistake, and it is left BYTE-UNCHANGED; what moves is the reference this
        ASSERTION reads.

        Editing the constant to agree with today's gold would have destroyed a
        historical record to make an assertion pass -- the mirror of clearing a
        disclosure by making it green, and the second thing this phase asks the
        owner to confirm was not done.

        The full decision, with both references and the rationale, is recorded in
        ``tests.phase33_state.GOLD_WIDTH_REFERENCE_RECONCILIATION_33_14``.

        RE-POINTED AGAIN by Plan 33.2-12, in the same voice. The reference became
        stale in its turn because p332_ rung 4 rebuilt gold with an ATTRIBUTED width
        change: the two weather inputs no forecast can supply (``precip_mm``,
        ``raw_precip_mm``) left all three matrices. The assertion now reads
        ``current_ladder_widths()``, IMPORTED from the width module so there is one
        resolver, not two: the last Phase-33.2 ``P332_*_GOLD_WIDTH_DELTA`` slot's
        measured widths. ``GOLD_WIDTHS_AFTER_WEATHER_RUNG`` is left BYTE-UNCHANGED as
        the record of what gold was between Phase 33.1 and that rung.
        """
        from tests.unit.test_data_qa_gold_width import current_ladder_widths

        measured = _production_gold_widths()
        recorded = current_ladder_widths()
        superseded = tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD)
        assert measured == recorded, (
            f"production gold is {measured} wide but "
            f"the Phase-33.2 width ladder records {recorded}. Either gold was "
            "rebuilt with a schema change that nobody attributed, or this "
            "reference is stale in its turn. The superseded pre-33.1 reference "
            f"GOLD_WIDTHS_BEFORE_ELO_REBUILD reads {superseded} and is retained "
            "as a historical record; do not adjust one to match the other "
            "without deciding which is wrong."
        )

    def test_the_superseded_reference_is_retained_unedited(self) -> None:
        """The record was RE-POINTED, not rewritten.

        The failure mode this guards is the opposite of the one above: making
        the assertion pass by editing ``GOLD_WIDTHS_BEFORE_ELO_REBUILD`` to
        agree with today's gold. That would destroy the only committed statement
        of what gold was before Phase 33.1's rung 1, to save re-pointing one
        reference.
        """
        assert tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD) == (194, 195, 194), (
            "GOLD_WIDTHS_BEFORE_ELO_REBUILD is the PRE-33.1 record and must stay "
            f"(194, 195, 194). It reads "
            f"{tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD)!r}, which means "
            "somebody edited a historical record instead of re-pointing the "
            "assertion that reads it."
        )
        reconciliation = phase33_state.GOLD_WIDTH_REFERENCE_RECONCILIATION_33_14
        assert reconciliation["old_constant_left_byte_unchanged"] is True
        assert "REFERENCE" in str(reconciliation["which_was_wrong"])

    def test_no_identity_column_is_in_any_production_matrix(self) -> None:
        """The migration added a column to silver; none of the three reached gold."""
        for target in GOLD_TARGETS:
            columns = set(
                pd.read_parquet(
                    Path("data") / "gold" / f"features_{target}.parquet"
                ).columns
            )
            present = sorted(set(IDENTITY_COLUMNS) & columns)
            assert not present, (
                f"identity column(s) {present!r} are columns of features_{target}. "
                "generate_feature_matrices treats every non-excluded column as a "
                "model feature, so this is a model-input change, not a cosmetic one."
            )

    def test_the_recorded_widths_are_three_plausible_integers(self) -> None:
        """Anti-typo. A reference nobody can sanity-check is a reference nobody checks."""
        recorded = tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD)
        assert len(recorded) == 3
        assert all(isinstance(width, int) and width > 100 for width in recorded), (
            f"GOLD_WIDTHS_BEFORE_ELO_REBUILD is {recorded!r}; a real matrix carries "
            "well over a hundred columns."
        )

    def test_the_sandbox_digests_name_the_three_matrices_and_are_content_hashes(
        self,
    ) -> None:
        """The record says WHICH files the widths were read from, not merely the widths."""
        recorded = phase33_state.SANDBOX_GOLD_DIGESTS_33_12
        assert [name for name, _ in recorded] == [
            f"features_{target}.parquet" for target in GOLD_TARGETS
        ], f"SANDBOX_GOLD_DIGESTS_33_12 names {[n for n, _ in recorded]!r}"
        for name, digest in recorded:
            assert _HEX64.match(digest), (
                f"{name}'s recorded digest {digest!r} is not a 64-character sha256. "
                "A stat signature would mean the measurement rested on size and "
                "mtime, which D33-32 forbids for a verdict."
            )

    def test_the_measured_files_were_not_production_gold(self) -> None:
        """THE ANTI-FALLBACK CONTROL, and the reason the digests are recorded at all.

        A test that runs a sandboxed build and then reads PRODUCTION gold passes
        identically when the sandbox path did nothing whatever. The recorded
        sandbox digests must therefore NOT equal the production ones -- if they
        did, the widths above would be a description of files the build never
        wrote (Codex HIGH).
        """
        production = _production_gold_digests()
        for (name, sandbox_digest), target in zip(
            phase33_state.SANDBOX_GOLD_DIGESTS_33_12, GOLD_TARGETS, strict=True
        ):
            assert sandbox_digest != production[target], (
                f"the recorded sandbox digest for {name} is byte-identical to "
                f"production data/gold/features_{target}.parquet. Either the "
                "sandbox build silently fell back to the production root, or the "
                "digest was copied from the wrong file. Both make the width "
                "measurement meaningless."
            )

    @pytest.mark.skipif(
        not os.environ.get(FULL_BUILD_ENV_FLAG),
        reason=FULL_BUILD_SKIP_MESSAGE,
    )
    def test_a_full_history_sandbox_build_is_194_195_194_and_carries_no_identity_column(
        self, tmp_path, monkeypatch
    ) -> None:
        """The heavy arm: the real builder, the real migrated silver, a real sandbox.

        Three separate proofs that the SANDBOX is the thing being measured:

        1. the sandbox paths are INJECTED through the readers and the writer, and
           the production lake is asserted not to be the parquet root;
        2. each matrix file is asserted to EXIST and to carry an mtime at or after
           the build-start instant, so a pre-existing file cannot be read as
           though this build had produced it;
        3. the production ``data/`` tree is digested before and after and asserted
           IDENTICAL, so a silent fallback to the production root is a boundary
           violation rather than a passing test.
        """
        from scripts.fingerprint_gold import fingerprint_gold
        from tests.data_boundary import (
            PRODUCTION_DATA_ROOT,
            assert_tree_unchanged,
            digest_file,
            digest_tree,
        )

        before_tree = digest_tree(PRODUCTION_DATA_ROOT)

        sandbox = tmp_path / "lake"
        sandbox.mkdir()
        shutil.copytree(Path("data") / "silver", sandbox / "silver")

        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(sandbox))
        )
        import scripts.build_features as bf_mod

        real_save = storage_mod.save_dataframe

        def _parquet_only(*args, **kwargs):
            kwargs["save_to_db"] = False
            return real_save(*args, **kwargs)

        real_load = storage_mod.load_dataframe

        def _parquet_load(table_name, layer="silver", source="auto", **kwargs):
            return real_load(table_name, layer=layer, source="parquet", **kwargs)

        monkeypatch.setattr(bf_mod, "save_dataframe", _parquet_only)
        monkeypatch.setattr(storage_mod, "save_dataframe", _parquet_only)
        monkeypatch.setattr(bf_mod, "load_dataframe", _parquet_load)

        assert (
            Path(storage_mod._parquet_manager.base_path).resolve()
            != Path("data").resolve()
        ), "the parquet manager still points at the production lake"

        build_start_ns = time.time_ns()
        builder = FeatureMatrixBuilder()
        matrices = builder.generate_feature_matrices(
            as_of_datetime=datetime(2030, 1, 1, tzinfo=UTC)
        )
        builder.save_feature_matrices(matrices)

        measured_digests = []
        for target in GOLD_TARGETS:
            path = sandbox / "gold" / f"features_{target}.parquet"
            assert path.exists(), (
                f"the sandbox build produced no {path.name}; the width assertions "
                "below would otherwise have read production gold."
            )
            assert path.stat().st_mtime_ns >= build_start_ns, (
                f"{path.name} is OLDER than the build start instant, so it is not "
                "a file this build wrote."
            )
            measured_digests.append((path.name, digest_file(path)))

        document = fingerprint_gold(sandbox)
        widths = tuple(
            len(document[f"features_{target}"]["columns"]) for target in GOLD_TARGETS
        )
        assert widths == tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD), (
            f"a full-history build from the migrated 16-column silver produced "
            f"{widths}, not {tuple(phase33_state.GOLD_WIDTHS_BEFORE_ELO_REBUILD)}. "
            "The 'identity columns are INERT for gold' claim is WITHDRAWN: Plan "
            "33-14's rebuild now has two causes to attribute, not one, and needs a "
            "second separately-attributed rung."
        )

        for target in GOLD_TARGETS:
            columns = set(document[f"features_{target}"]["columns"])
            present = sorted(set(IDENTITY_COLUMNS) & columns)
            assert not present, (
                f"identity column(s) {present!r} reached the sandbox features_{target}."
            )

        assert [digest for _, digest in measured_digests] != [
            digest_file(Path("data") / "gold" / f"features_{target}.parquet")
            for target in GOLD_TARGETS
        ], "the sandbox matrices are byte-identical to production gold"

        assert_tree_unchanged(
            before_tree, digest_tree(PRODUCTION_DATA_ROOT), PRODUCTION_DATA_ROOT
        )


# ---------------------------------------------------------------------------
# Plan 33-12 Task 2(a2): DOWNSTREAM COMPATIBILITY, and the one column that is NOT
# merely tolerated.
#
# The review asked that every reader of data/silver/games.parquet be driven
# against the widened frame and against the narrow one and be shown to produce the
# same output. Driving it produced a finding that changes the shape of the answer,
# so the answer is split rather than averaged:
#
#   season_type and neutral_site ARE inert for the gold-reaching builders. Dropping
#   them changes no output cell.
#
#   stadium_id IS NOT INERT AND IS NOT OPTIONAL. Since D33.1-06 the contextual
#   builder routes EVERY game of EVERY season by its own stadium_id
#   (features/contextual.py:1268, :1593), and Plan 33.1-04's owner-assigned fix made
#   an unresolvable id a LOUD hard failure instead of an empty contextual frame. So
#   the honest assertion is not "every reader tolerates the new column" -- it is
#   that dropping it RAISES UnknownStadiumError BY NAME. That is a stronger
#   statement than tolerance, and it is the reason this migration had to run before
#   Plan 33.1-07 rather than after it.
# ---------------------------------------------------------------------------

INERT_IDENTITY_COLUMNS = ("season_type", "neutral_site")

# A small REAL slice of the migrated store: enough games to exercise the per-row
# loops, few enough that the whole class runs in a couple of seconds. Read-only.
COMPAT_SLICE_SEASON = 2024
COMPAT_SLICE_MAX_WEEK = 2


def _migrated_games_slice() -> pd.DataFrame:
    path = Path("data") / "silver" / "games.parquet"
    if not path.exists():
        pytest.skip(
            "data/silver/games.parquet is absent -- a fresh checkout has no data/ "
            "(it is gitignored). Run the Plan 33-12 migration to populate it."
        )
    frame = pd.read_parquet(path)
    if "stadium_id" not in frame.columns:
        pytest.skip(
            "data/silver/games.parquet carries no stadium_id, so the Plan 33-12 "
            "identity migration has not been run against this checkout."
        )
    return frame[
        (frame["season"] == COMPAT_SLICE_SEASON)
        & (frame["week"] <= COMPAT_SLICE_MAX_WEEK)
    ].copy()


@pytest.mark.integration
class TestTheWidenedSilverFrameIsSafeForItsReaders:
    """Task 2(a2). Driven, not asserted from a source scan."""

    def test_the_slice_is_non_vacuous(self) -> None:
        """Asserted first: identical output over zero rows proves nothing."""
        games = _migrated_games_slice()
        assert len(games) >= 24, (
            f"the compatibility slice holds {len(games)} games; a comparison over "
            "an empty frame is vacuously equal."
        )
        assert games["stadium_id"].notna().all()

    @pytest.mark.parametrize("builder_name", ["contextual", "market"])
    def test_the_gold_reaching_builders_ignore_season_type_and_neutral_site(
        self, builder_name: str
    ) -> None:
        """The 16-column frame and the 14-column one produce identical output."""
        from features.contextual import ContextualFeaturesCalculator
        from features.market_anchors import MarketAnchorFeaturesCalculator

        calculators = {
            "contextual": ContextualFeaturesCalculator,
            "market": MarketAnchorFeaturesCalculator,
        }
        as_of = datetime(2030, 1, 1, tzinfo=UTC)
        games = _migrated_games_slice()
        narrowed = games.drop(columns=list(INERT_IDENTITY_COLUMNS))

        wide = calculators[builder_name]().build_features(games, as_of)
        narrow = calculators[builder_name]().build_features(narrowed, as_of)

        assert not wide.empty, (
            f"the {builder_name} builder produced an empty frame, so the "
            "comparison below is vacuous."
        )
        pd.testing.assert_frame_equal(wide, narrow)

    def test_dropping_stadium_id_is_refused_by_name_rather_than_tolerated(self) -> None:
        """THE FINDING. stadium_id is load-bearing, not an optional extra column."""
        from features.contextual import (
            ContextualFeaturesCalculator,
            UnknownStadiumError,
        )

        games = _migrated_games_slice().drop(columns=["stadium_id"])
        with pytest.raises(UnknownStadiumError) as excinfo:
            ContextualFeaturesCalculator().build_features(
                games, datetime(2030, 1, 1, tzinfo=UTC)
            )
        assert "stadium_id" in str(excinfo.value), (
            "the refusal must NAME the column it could not route on; a generic "
            "message sends the operator back to the source to find out what broke."
        )

    def test_a_null_stadium_id_raises_a_named_refusal_rather_than_resolving_silently(
        self,
    ) -> None:
        """Nullability, answered in the direction the code actually takes.

        The review asked that a null ``stadium_id`` be tolerated. It is NOT, and
        that is correct rather than a defect: a null id means the game's venue is
        unknown, and the alternative to refusing is silently resolving to the home
        team's stadium -- which is exactly the misresolution
        HISTORICAL_NEUTRAL_MISRESOLUTION measures on 91 games. The migrated store
        carries ZERO nulls, so the refusal is unreachable in practice; it is
        asserted so that it stays unreachable by refusal rather than by luck.
        """
        from features.contextual import (
            ContextualFeaturesCalculator,
            UnknownStadiumError,
        )

        games = _migrated_games_slice()
        games.loc[games.index[0], "stadium_id"] = None
        with pytest.raises(UnknownStadiumError):
            ContextualFeaturesCalculator().build_features(
                games, datetime(2030, 1, 1, tzinfo=UTC)
            )
