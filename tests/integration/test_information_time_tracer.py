"""THE PHASE 33.2 TRACER: one game's lock, one source's information time, one refusal.

Phase 33.2, Plan 33.2-01 Task 3 (SPEC R1, R2; D33.2-01, D33.2-05).

WHAT THIS PROVES, IN ONE PATH
-----------------------------
The real ``FeatureMatrixBuilder.generate_feature_matrices`` -- not a stub of it -- runs
against a SANDBOXED data root seeded with a real, read-only slice of production silver,
and this module shows, end to end:

1. the lock rule: every game's lock comes from ``utils.game_lock`` through the one lock
   frame the build computes;
2. the sidecar provenance: the ``elo`` source supplies a per-row
   ``(game_id, basis, information_time)`` frame, never a column;
3. the gate: every ``per_row`` information time is at or before its game's lock, and the
   ``no_information`` rows are value-checked against the declared start-state signature;
4. the refusal: ONE game's information time moved to its lock plus one second stops the
   build, the refusal NAMES that game, and the sandbox gold directory is byte-identical to
   its pre-run digest -- the build stopped BEFORE any write (D33.2-05, history half);
5. the ``<=`` boundary: the same game moved to EXACTLY its lock builds, and produces the
   same gold as the clean run;
6. the write boundary: production ``data/`` and ``artifacts/`` are content-digested around
   the whole run. ``git status data/`` is not used anywhere in this module -- ``data/`` is
   gitignored, so that check is structurally incapable of failing (RESEARCH P11, COLD-05).

The remaining twenty-seven plans of the phase expand horizontally from this slice.

HOW NARROW THE PROOF IS -- READ THIS BEFORE QUOTING IT
------------------------------------------------------
This is NOT whole-build coverage and must never be quoted as such. Exactly ONE of the nine
``feature_sources`` registry keys (``elo``) is checked against the lock; ``games`` is the
base frame; of the other seven, those that loaded zero rows on this slice are reported
``empty_unchecked`` and the rest ``unregistered``. The opponent-adjusted family is merged
AFTER the Stage-1 loop, so it is out of the loop's reach entirely and is reported in its own
``post_stage1_sources`` set. ``checked_sources == ("elo",)`` is the CORRECT result for this
wave. Plans 33.2-12 .. 33.2-17 add the remaining suppliers, Plan 33.2-16 brings the
opponent-adjusted family into the loop, and Plan 33.2-20 arms the refusal that turns every
non-``checked`` set into a build failure.

WHY 2002, AND WHAT THE SLICE IS
-------------------------------
2002 is the only season whose Elo rows carry BOTH bases: week 1 comes from the synthetic 1500
start state with no contributing game (``no_information``) and weeks 2 onward are dated
(``per_row``), so mixed provenance is proved through the REAL build, not only in a unit
fixture. It also predates the team-form, snap, injury and market coverage of the owned data,
so those sources load ZERO rows against real data, exercising ``empty_unchecked`` with a
genuine empty frame.

The slice is the real 2002 rows of silver ``games``, ``elo_game_snapshots`` and
``weather_features``, copied read-only. ``weather_features`` is in the slice because the
build hard-requires its ``weather_severity_score`` column; without it the build stops on a
``KeyError`` unrelated to this plan. The build emits the WP gold matrix only: the ATS and
O/U targets are derived solely from market lines, and the owned odds data holds no 2002
line, so there is no honest ATS or O/U matrix for 2002 to emit.

The sandbox root arrives through the ``TRACER_SANDBOX`` environment variable (a fresh
temporary directory when it is unset), and ``data.storage``'s two module singletons are
pointed at it, which is the only injectable seam ``save_dataframe`` and ``load_dataframe``
have (``tests/fixtures/elo_sandbox.py``).

Run this module:  uv run pytest tests/integration/test_information_time_tracer.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import DuckDBConnection, ParquetManager, save_dataframe
from features.elo_features import EloFeatureBuilder
from features.provenance import CoverageReport, InformationTimeViolation
from scripts.build_features import FeatureMatrixBuilder
from scripts.fingerprint_gold import BUILD_CLOCK_COLUMNS
from tests.data_boundary import (
    PRODUCTION_ARTIFACTS_ROOT,
    PRODUCTION_DATA_ROOT,
    assert_tree_unchanged,
    content_digest_tree,
)
from utils import game_lock

pytestmark = pytest.mark.usefixtures("data_boundary_guard", "artifacts_boundary_guard")

TRACER_SEASON = 2002
SANDBOX_ENV = "TRACER_SANDBOX"

#: The real silver tables copied read-only into the sandbox, each filtered to the season.
#: Plan 33.2-12 added silver ``weather``: the weather builder is now an information-time
#: supplier and dates each game from the silver row its one fence selected. That table
#: carries no ``season`` column, so it is sliced by the season in its ``game_id``.
SEEDED_TABLES: tuple[str, ...] = (
    "games",
    "elo_game_snapshots",
    "weather_features",
    "weather",
)

#: The nine keys ``load_all_feature_sources`` populates (scripts/build_features.py).
REGISTRY_KEYS: tuple[str, ...] = (
    "games",
    "team_form",
    "elo",
    "contextual",
    "weather",
    "market",
    "qb_tracking",
    "snaps",
    "injury",
)

#: MEASURED on this slice (Plan 33.2-01 Task 3): which of the seven non-elo keys loaded
#: zero rows, and which loaded rows but have no provenance supplier yet.
EXPECTED_EMPTY_UNCHECKED: tuple[str, ...] = ("injury", "market", "snaps", "team_form")
#: Plan 33.2-12 registered ``weather`` (it owns the weather fence and its provenance), so
#: it moved from the unregistered set to ``checked_sources``. Plan 33.2-13 registered
#: ``qb_tracking`` (and ``injury``, which loads zero rows on this slice and so stays
#: ``empty_unchecked``), so ``qb_tracking`` moved to ``checked_sources`` too, after
#: ``weather`` in loop order.
EXPECTED_UNREGISTERED: tuple[str, ...] = ("contextual",)
EXPECTED_CHECKED: tuple[str, ...] = ("elo", "weather", "qb_tracking")


@dataclass
class TracerRun:
    """Everything the three sandbox builds observed, recorded once for the module."""

    sandbox: Path
    games: pd.DataFrame
    clean_matrices: dict[str, pd.DataFrame]
    clean_coverage: CoverageReport | None
    clean_provenance: pd.DataFrame
    planted_game: str
    planted_lock: pd.Timestamp
    planted_error: InformationTimeViolation | None
    gold_digest_before_planted: dict[str, str]
    gold_digest_after_planted: dict[str, str]
    at_lock_matrices: dict[str, pd.DataFrame]
    at_lock_error: BaseException | None = None


#: The REAL supplier, captured before any test wraps it.
_REAL_INFORMATION_TIMES = EloFeatureBuilder.information_times


def _seed(sandbox: Path) -> pd.DataFrame:
    """Copy the season's real rows of each seeded table from production, read-only."""
    for layer in ("bronze", "silver", "gold"):
        (sandbox / layer).mkdir(parents=True, exist_ok=True)
    games = pd.DataFrame()
    for table in SEEDED_TABLES:
        frame = pd.read_parquet(PRODUCTION_DATA_ROOT / "silver" / f"{table}.parquet")
        season = (
            frame["season"]
            if "season" in frame.columns
            else frame["game_id"].str[:4].astype(int)
        )
        frame = frame.loc[season == TRACER_SEASON].reset_index(drop=True)
        assert len(frame) > 0, f"production silver {table} has no {TRACER_SEASON} rows"
        save_dataframe(frame, table, layer="silver", replace_mode=True)
        if table == "games":
            games = frame
    return games


def _gold_digest(sandbox: Path) -> dict[str, str]:
    return content_digest_tree(sandbox / "gold")


def _provenance_patch(
    original: Callable[..., pd.DataFrame],
    sink: list[pd.DataFrame],
    move: tuple[str, object] | None,
) -> Callable[..., pd.DataFrame]:
    """Wrap the REAL ``information_times``: record its output, optionally move ONE row.

    The real supplier computes every row; the wrapper only records it, or -- for the
    planted and at-lock runs -- overwrites exactly one game's information time.
    """

    def _wrapped(
        self: EloFeatureBuilder, games_df: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        frame = original(self, games_df, **kwargs)
        if move is not None:
            game_id, when = move
            frame = frame.copy()
            frame["information_time"] = frame["information_time"].astype(object)
            frame.loc[frame["game_id"] == game_id, "information_time"] = when
        sink.append(frame)
        return frame

    return _wrapped


def _build(
    monkeypatch: pytest.MonkeyPatch,
    sink: list[pd.DataFrame],
    move: tuple[str, object] | None = None,
    *,
    save: bool,
) -> tuple[dict[str, pd.DataFrame], CoverageReport | None]:
    monkeypatch.setattr(
        EloFeatureBuilder,
        "information_times",
        _provenance_patch(_REAL_INFORMATION_TIMES, sink, move),
    )
    builder = FeatureMatrixBuilder()
    matrices = builder.generate_feature_matrices(target_season=TRACER_SEASON)
    if save:
        builder.save_feature_matrices(matrices, TRACER_SEASON, None)
    return matrices, builder.information_time_coverage


@pytest.fixture(scope="module")
def tracer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TracerRun]:
    """Run the three sandbox builds once, inside an explicit production digest bracket.

    The function-scoped ``data_boundary_guard`` / ``artifacts_boundary_guard`` opted into
    by ``pytestmark`` bracket each test; they cannot bracket a MODULE fixture, which runs
    before them. So this fixture brackets its own builds with the same instrument
    (``tests.data_boundary.content_digest_tree`` + ``assert_tree_unchanged``).
    """
    data_before = content_digest_tree(PRODUCTION_DATA_ROOT)
    artifacts_before = content_digest_tree(PRODUCTION_ARTIFACTS_ROOT)

    parent = os.environ.get(SANDBOX_ENV)
    base = (
        Path(tempfile.mkdtemp(prefix="p332_tracer_", dir=parent))
        if parent
        else tmp_path_factory.mktemp("p332_tracer")
    )
    sandbox = base / "data"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(SANDBOX_ENV, str(sandbox))
        sandbox = Path(os.environ[SANDBOX_ENV])
        sandbox.mkdir(parents=True, exist_ok=True)
        connection = DuckDBConnection(str(sandbox / "sandbox.duckdb"))
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(sandbox))
        )
        monkeypatch.setattr(storage_mod, "_db_connection", connection)
        try:
            games = _seed(sandbox)
            locks = game_lock.lock_frame(games)

            # 1. The clean build, saved to the SANDBOX gold layer.
            clean_sink: list[pd.DataFrame] = []
            clean_matrices, clean_coverage = _build(monkeypatch, clean_sink, save=True)
            clean_provenance = clean_sink[-1]

            # 2. The planted refusal: ONE per_row game moved to its lock + 1 second.
            planted_game = str(
                clean_provenance.loc[clean_provenance["basis"] == "per_row", "game_id"]
                .sort_values()
                .iloc[0]
            )
            planted_lock = cast(
                pd.Timestamp, pd.Timestamp(cast(Any, locks[planted_game]))
            )
            gold_before = _gold_digest(sandbox)
            planted_error: InformationTimeViolation | None = None
            try:
                _build(
                    monkeypatch,
                    [],
                    (planted_game, planted_lock + pd.Timedelta(seconds=1)),
                    save=True,
                )
            except InformationTimeViolation as exc:
                planted_error = exc
            gold_after = _gold_digest(sandbox)

            # 3. The at-lock control: the same game at EXACTLY its lock.
            at_lock_error: BaseException | None = None
            at_lock_matrices: dict[str, pd.DataFrame] = {}
            try:
                at_lock_matrices, _ = _build(
                    monkeypatch, [], (planted_game, planted_lock), save=False
                )
            except InformationTimeViolation as exc:
                at_lock_error = exc

            yield TracerRun(
                sandbox=sandbox,
                games=games,
                clean_matrices=clean_matrices,
                clean_coverage=clean_coverage,
                clean_provenance=clean_provenance,
                planted_game=planted_game,
                planted_lock=planted_lock,
                planted_error=planted_error,
                gold_digest_before_planted=gold_before,
                gold_digest_after_planted=gold_after,
                at_lock_matrices=at_lock_matrices,
                at_lock_error=at_lock_error,
            )
        finally:
            connection.close()

    assert_tree_unchanged(
        data_before, content_digest_tree(PRODUCTION_DATA_ROOT), PRODUCTION_DATA_ROOT
    )
    assert_tree_unchanged(
        artifacts_before,
        content_digest_tree(PRODUCTION_ARTIFACTS_ROOT),
        PRODUCTION_ARTIFACTS_ROOT,
    )


def _without_build_clock(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in BUILD_CLOCK_COLUMNS if c in frame.columns])


# ===========================================================================
# 1. The sandbox build is the REAL build, and it writes only the sandbox
# ===========================================================================


class TestTheSandboxBuild:
    def test_the_sandbox_root_came_from_the_environment_variable(
        self, tracer: TracerRun
    ) -> None:
        assert tracer.sandbox.name == "data"
        assert (tracer.sandbox / "silver" / "games.parquet").exists()
        assert PRODUCTION_DATA_ROOT.resolve() not in tracer.sandbox.resolve().parents

    def test_the_real_build_produced_the_2002_wp_gold_matrix(
        self, tracer: TracerRun
    ) -> None:
        wp = tracer.clean_matrices["wp"]
        assert set(wp["season"]) == {TRACER_SEASON}
        assert set(wp["game_id"]) == set(tracer.games["game_id"])
        assert len(wp) == len(tracer.games) == 267
        assert (tracer.sandbox / "gold" / "features_wp.parquet").exists()

    def test_ats_and_ou_are_absent_because_2002_has_no_owned_market_line(
        self, tracer: TracerRun
    ) -> None:
        """Their targets derive ONLY from market lines; no 2002 line exists to derive from."""
        assert "ats" not in tracer.clean_matrices
        assert "ou" not in tracer.clean_matrices
        assert "market" in EXPECTED_EMPTY_UNCHECKED


# ===========================================================================
# 2. The clean pass, and mixed provenance through the REAL build
# ===========================================================================


class TestTheCleanPass:
    def test_every_game_in_the_slice_has_an_elo_provenance_row(
        self, tracer: TracerRun
    ) -> None:
        prov = tracer.clean_provenance
        assert list(prov.columns) == ["game_id", "basis", "information_time"]
        assert set(prov["game_id"]) == set(tracer.games["game_id"])
        assert not prov["game_id"].duplicated().any()

    def test_every_per_row_time_is_at_or_before_its_lock(
        self, tracer: TracerRun
    ) -> None:
        locks = game_lock.lock_frame(tracer.games)
        per_row = tracer.clean_provenance.loc[
            tracer.clean_provenance["basis"] == "per_row"
        ]
        assert len(per_row) > 0
        for gid, when in zip(
            per_row["game_id"], per_row["information_time"], strict=True
        ):
            assert game_lock.is_admissible(when, locks[gid]), gid

    def test_mixed_provenance_week_1_undated_weeks_2_on_dated(
        self, tracer: TracerRun
    ) -> None:
        prov = tracer.clean_provenance.merge(
            tracer.games[["game_id", "week"]], on="game_id"
        )
        week1 = prov.loc[prov["week"] == 1]
        later = prov.loc[prov["week"] > 1]
        assert len(week1) == 16
        assert set(week1["basis"]) == {"no_information"}
        assert week1["information_time"].isna().all()
        assert set(later["basis"]) == {"per_row"}
        assert later["information_time"].notna().all()
        # The recorded split: 16 undated, 251 dated.
        assert prov["basis"].value_counts().to_dict() == {
            "per_row": 251,
            "no_information": 16,
        }

    def test_the_week_1_values_satisfy_the_declared_signature(
        self, tracer: TracerRun
    ) -> None:
        signature = EloFeatureBuilder().no_information_signature()
        week1 = tracer.games.loc[tracer.games["week"] == 1, "game_id"]
        wp = tracer.clean_matrices["wp"]
        assert len(wp.loc[wp["game_id"].isin(week1)]) == 16
        # The gold matrix is z-scored, so the signature is checked on silver's own values:
        snapshots = pd.read_parquet(
            tracer.sandbox / "silver" / "elo_game_snapshots.parquet"
        )
        raw = snapshots.loc[snapshots["game_id"].isin(week1)]
        assert (raw["home_elo_pre"] == signature["home_elo"]).all()
        assert (raw["away_elo_pre"] == signature["away_elo"]).all()
        assert (raw["home_elo_uncertainty"] == signature["home_elo_uncertainty"]).all()
        assert (raw["away_elo_uncertainty"] == signature["away_elo_uncertainty"]).all()


# ===========================================================================
# 2a. The narrow-coverage report: four sets, asserted -- never a bare pass
# ===========================================================================


class TestTheNarrowCoverageReport:
    """``checked_sources`` is exactly the sources registered so far, in loop order.

    Plan 33.2-01 registered ``elo``; Plan 33.2-12 registered ``weather``; Plan 33.2-13
    registered ``qb_tracking`` and ``injury``. Plans 33.2-14 .. 33.2-17 (the remaining suppliers), 33.2-16 (the opponent-adjusted family) and 33.2-20
    (the armed refusal) are the ones that shrink the other three sets to empty.
    """

    def test_exactly_the_registered_sources_were_checked(
        self, tracer: TracerRun
    ) -> None:
        assert tracer.clean_coverage is not None
        assert tracer.clean_coverage.checked_sources == EXPECTED_CHECKED

    def test_the_empty_and_unregistered_sets_are_named(self, tracer: TracerRun) -> None:
        report = tracer.clean_coverage
        assert report is not None
        assert report.empty_unchecked_sources == EXPECTED_EMPTY_UNCHECKED
        assert report.unregistered_sources == EXPECTED_UNREGISTERED

    def test_the_post_stage1_family_is_its_own_set(self, tracer: TracerRun) -> None:
        assert tracer.clean_coverage is not None
        assert tracer.clean_coverage.post_stage1_sources == ("opponent_adj",)

    def test_the_sets_partition_the_eight_non_games_keys(
        self, tracer: TracerRun
    ) -> None:
        report = tracer.clean_coverage
        assert report is not None
        sets = (
            report.checked_sources,
            report.empty_unchecked_sources,
            report.unregistered_sources,
        )
        flat = [key for group in sets for key in group]
        assert len(flat) == len(set(flat)), "a key was counted twice"
        assert set(flat) == set(REGISTRY_KEYS) - {"games"}


# ===========================================================================
# 3. The planted refusal, and 4. the at-lock control
# ===========================================================================


class TestThePlantedRefusal:
    def test_the_build_raised_an_information_time_violation(
        self, tracer: TracerRun
    ) -> None:
        assert isinstance(tracer.planted_error, InformationTimeViolation)

    def test_the_refusal_names_that_exact_game(self, tracer: TracerRun) -> None:
        assert tracer.planted_error is not None
        assert tracer.planted_game in str(tracer.planted_error)
        assert tracer.planted_error.details["game_ids"] == [tracer.planted_game]
        assert tracer.planted_error.details["source"] == "elo"

    def test_the_sandbox_gold_is_byte_identical_after_the_refused_run(
        self, tracer: TracerRun
    ) -> None:
        """The build stopped BEFORE any write (D33.2-05, history half)."""
        assert len(tracer.gold_digest_before_planted) > 0, "non-vacuity: gold existed"
        assert tracer.gold_digest_after_planted == tracer.gold_digest_before_planted


class TestTheAtLockControl:
    def test_the_at_lock_build_completed(self, tracer: TracerRun) -> None:
        assert tracer.at_lock_error is None
        assert "wp" in tracer.at_lock_matrices

    def test_the_at_lock_build_produces_the_same_gold_as_the_clean_run(
        self, tracer: TracerRun
    ) -> None:
        assert set(tracer.at_lock_matrices) == set(tracer.clean_matrices)
        for target, clean in tracer.clean_matrices.items():
            pd.testing.assert_frame_equal(
                _without_build_clock(tracer.at_lock_matrices[target]).reset_index(
                    drop=True
                ),
                _without_build_clock(clean).reset_index(drop=True),
            )
