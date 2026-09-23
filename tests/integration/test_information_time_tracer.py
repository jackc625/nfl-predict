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

HOW WIDE THE PROOF IS -- READ THIS BEFORE QUOTING IT
-----------------------------------------------------
WAS NARROW, AND IS NO LONGER (Plan 33.2-20). At Plan 33.2-01 exactly ONE of the nine
``feature_sources`` registry keys (``elo``) was checked here, ``games`` was skipped as the
base frame, four keys loaded zero rows and were reported ``empty_unchecked``, and the
opponent-adjusted family sat in its own ``post_stage1_sources`` set. ``checked_sources ==
("elo",)`` was the CORRECT result for that wave, and this section said so at length.

From Plan 33.2-20 this run checks all TEN declared sources -- the nine registry keys,
``games`` among them, plus the opponent-adjusted family -- and every other set is EMPTY. It
is now a whole-build proof for this slice, and the three sets being empty is the assertion
rather than the caveat: an empty source, an unregistered source or an unchecked post-Stage-1
family each REFUSE the build.

WHY 2002, AND WHAT THE SLICE IS
-------------------------------
2002 is the only season whose Elo rows carry BOTH bases: week 1 comes from the synthetic 1500
start state with no contributing game (``no_information``) and weeks 2 onward are dated
(``per_row``), so mixed provenance is proved through the REAL build, not only in a unit
fixture.

THE SEEDING CHANGED WITH THE GATE, AND THE DISTINCTION IS THE POINT (Plan 33.2-20). The
slice is the real 2002 rows of silver ``games``, ``elo_game_snapshots``,
``weather_features``, ``weather``, ``team_form_features`` and ``team_game_stats``, copied
read-only -- team form reaches back to 2002 since p332_ rung 8 widened its floor -- plus
SCHEMA-ONLY copies of ``snap_counts``, ``injuries`` and ``odds_snapshot``, of which
production genuinely holds no 2002 row (snaps start in 2013, injury reports in 2009, the
stored odds in 2018).

Those last three used to be absent from the sandbox altogether. That made ``load_dataframe``
raise, ``_SOURCE_LOAD_ERRORS`` turn each failure into an EMPTY frame, and the build proceed
with four sources contributing nothing -- which is exactly the shape Plan 33.2-20 turns into
a refusal, because a FAILED load and an absent one are indistinguishable at the frame. With
the table PRESENT and empty, each builder does its real job: it emits one row per game at
its declared unknown values, and the gate VALUE-CHECKS that declaration. "The table is
missing" and "the table has nothing for these games" are different facts, and only the
second is true of 2002.

``weather_features`` is in the slice because the build hard-requires its
``weather_severity_score`` column; without it the build stops on a ``KeyError`` unrelated to
this plan. All THREE gold matrices are emitted: p332_ rung 9 (Plan 33.2-19) removed the
line-derived ``target_ats`` / ``target_ou`` and selects ATS and O/U on ``home_margin`` and
``total_points``, both derived from the SCORES, which 2002 has. No market column reaches
gold, which is asserted directly rather than inferred from an absent matrix.

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
    # PLAN 33.2-20 added these two. Silver team form now reaches back to 2002 (p332_
    # rung 8 widened its floor), and ``team_game_stats`` is what the opponent-adjusted
    # family's per-game pool is built from.
    "team_form_features",
    "team_game_stats",
)

#: Tables the sandbox seeds with their real SCHEMA and ZERO rows, because production has
#: no rows of them for this season at all -- snap counts start in 2013, injury reports in
#: 2009 and the stored odds in 2018.
#:
#: WHY SEED THEM AT ALL (Plan 33.2-20). Without the table present, ``load_dataframe``
#: raises, ``_SOURCE_LOAD_ERRORS`` turns the source into an EMPTY FRAME, and from this plan
#: an empty source REFUSES the build. WITH the table present and empty, each builder does
#: what it is designed to do: it emits one row per game carrying its declared unknown
#: values, and the gate VALUE-CHECKS that declaration. The difference matters and is the
#: whole point of the change -- "the table is missing" and "the table has nothing for these
#: games" are different facts, and only the second is true of 2002.
EMPTY_SEEDED_TABLES: tuple[str, ...] = ("snap_counts", "injuries", "odds_snapshot")

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

#: EMPTY NOW (Plan 33.2-20), and the change is the plan's whole subject.
#:
#: Was: ``("injury", "market", "snaps", "team_form")`` -- MEASURED on this slice by Plan
#: 33.2-01, when four of the nine keys loaded ZERO rows because their silver tables were
#: not in the sandbox at all and ``_SOURCE_LOAD_ERRORS`` turned each failed load into an
#: empty frame. Plan 33.2-20 makes an empty source REFUSE the build, precisely because a
#: FAILED load is indistinguishable from an absent one at the frame -- so a slice that
#: reads as four empty sources is a slice nothing should be written from.
#:
#: The fix is not an exemption, it is the seeding: the sandbox now carries silver
#: ``team_form_features`` and ``team_game_stats`` (real 2002 rows -- rung 8 widened team
#: form's floor to 2002) and SCHEMA-ONLY copies of ``snap_counts``, ``injuries`` and
#: ``odds_snapshot``, which production genuinely has no 2002 rows of. With the tables
#: present, each builder emits one row per game at its declared unknown values and the gate
#: VALUE-CHECKS that declaration, exactly as it does for every dated source.
EXPECTED_EMPTY_UNCHECKED: tuple[str, ...] = ()
#: Plan 33.2-12 registered ``weather`` (it owns the weather fence and its provenance), so
#: it moved from the unregistered set to ``checked_sources``. Plan 33.2-13 registered
#: ``qb_tracking`` (and ``injury``, which loads zero rows on this slice and so stays
#: ``empty_unchecked``), so ``qb_tracking`` moved to ``checked_sources`` too, after
#: ``weather`` in loop order. Plan 33.2-14 registered ``contextual``, ``snaps`` and
#: ``team_form``: ``contextual`` loads rows on this slice and moved to ``checked_sources``
#: (between ``elo`` and ``weather`` in loop order); ``snaps`` and ``team_form`` load zero
#: rows here and stay ``empty_unchecked``. Nothing on this slice is unregistered any more.
EXPECTED_UNREGISTERED: tuple[str, ...] = ()
#: Plan 33.2-16 brought the opponent-adjusted family into the gate at its post-Stage-1 merge
#: site, so it is checked AFTER the Stage-1 loop and appended last. MEASURED on this slice: a
#: --season 2002 build reads 2001-2002 play-by-play (the scoped per-game branch has no 2018
#: floor); the 2001 games have no kickoff in silver games, cannot be timed and are never
#: admitted, and the 2002 games are adjusted at their locks -- so the family has rows and is
#: CHECKED.
#: ALL TEN (Plan 33.2-20): the nine registry keys, ``games`` among them, plus the
#: opponent-adjusted family. In LOOP ORDER -- ``games`` first because the Stage-1 loop
#: reaches it first, then the eight builder keys in registry order, then the family, which
#: is checked at its own post-Stage-1 merge site and so is appended last.
#:
#: ``games`` is here for the first time. It DEFINES the lock, so it cannot supply a
#: non-circular information time; Plan 33.2-20 gives it the second declared basis with its
#: VALUES checked against ``features.schedule_moves.facts_at_lock``, which lands it in
#: ``checked_sources`` like any other key rather than being skipped in silence.
EXPECTED_CHECKED: tuple[str, ...] = (
    "games",
    "team_form",
    "elo",
    "contextual",
    "weather",
    "market",
    "qb_tracking",
    "snaps",
    "injury",
    "opponent_adj",
)


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
        if "season" in frame.columns:
            season = frame["season"]
        elif "target_season" in frame.columns:
            season = frame["target_season"]
        else:
            season = frame["game_id"].str[:4].astype(int)
        frame = frame.loc[season == TRACER_SEASON].reset_index(drop=True)
        assert len(frame) > 0, f"production silver {table} has no {TRACER_SEASON} rows"
        save_dataframe(frame, table, layer="silver", replace_mode=True)
        if table == "games":
            games = frame

    for table in EMPTY_SEEDED_TABLES:
        frame = pd.read_parquet(PRODUCTION_DATA_ROOT / "silver" / f"{table}.parquet")
        assert len(frame) > 0, (
            f"production silver {table} is empty, so its schema is not"
        )
        save_dataframe(frame.iloc[0:0], table, layer="silver", replace_mode=True)
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

    def test_all_three_matrices_are_built_because_their_targets_come_from_the_scores(
        self, tracer: TracerRun
    ) -> None:
        """Was: ``test_ats_and_ou_are_absent_because_2002_has_no_owned_market_line``.

        PREMISE CORRECTED TO THE MEASUREMENT, not relaxed. That test asserted the ATS and
        O/U matrices were ABSENT for 2002, and its reason was true when it was written:
        both were gated and row-filtered on ``target_ats`` / ``target_ou``, which subtract
        a market line, and 2002 has no stored line at all.

        p332_ rung 9 (Plan 33.2-19) REMOVED those two columns with their market parents and
        selects each matrix on its trainer's own target instead -- ``home_margin`` for ATS
        and ``total_points`` for O/U, both derived from the SCORES. 2002 has scores, so all
        three matrices are built, and the market source is still the honest unknown: no
        2002 game carries a line, and none of the five market columns reaches gold.

        The subject the old test really carried -- that no 2002 gold value comes from a
        market line -- is asserted directly below, which is stronger than inferring it from
        an absent matrix.
        """
        assert set(tracer.clean_matrices) == {"wp", "ats", "ou"}
        for name, matrix in tracer.clean_matrices.items():
            assert len(matrix) == len(tracer.games) == 267, name
            market_columns = [
                column
                for column in matrix.columns
                if str(column).startswith("snapshot_")
                or str(column) in {"spread_movement", "total_movement"}
                or str(column) in {"target_ats", "target_ou"}
            ]
            assert market_columns == [], (
                f"{name} carries market-derived column(s) {market_columns}. No 2002 game "
                "has a stored line, and rung 9 removed every market column from gold."
            )
        # The market source is CHECKED (it emits one honest-unknown row per game) and it
        # is merged NOWHERE -- its registry disposition is exactly `checked_not_merged`.
        assert "market" in EXPECTED_CHECKED
        assert "market" not in EXPECTED_EMPTY_UNCHECKED


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
    registered ``qb_tracking`` and ``injury``; Plan 33.2-14 registered ``contextual``,
    ``snaps`` and ``team_form``. Plans 33.2-14 .. 33.2-17 (the remaining suppliers), 33.2-16 (the opponent-adjusted family) and 33.2-20
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
        """Was: ``post_stage1_sources == ("opponent_adj",)`` -- the family merged after Stage 1
        and out of the gate's reach. Plan 33.2-16 checks it at its merge site, so the set is
        empty and the family is reported in the ordinary sets (``checked_sources`` here)."""
        assert tracer.clean_coverage is not None
        assert tracer.clean_coverage.post_stage1_sources == ()

    def test_the_sets_partition_all_nine_registry_keys_and_the_family(
        self, tracer: TracerRun
    ) -> None:
        """Was: ``..._partition_the_eight_non_games_keys``, which subtracted ``games``.

        PREMISE CORRECTED TO THE MEASUREMENT. ``games`` was outside the partition because
        the Stage-1 loop skipped it in silence -- it DEFINES the lock, so it cannot supply
        a non-circular information time. Plan 33.2-20 gives it the SECOND declared basis
        with its values checked against ``features.schedule_moves.facts_at_lock``, so it is
        reported like any other key and the partition is over all nine plus the family.
        """
        report = tracer.clean_coverage
        assert report is not None
        sets = (
            report.checked_sources,
            report.empty_unchecked_sources,
            report.unregistered_sources,
        )
        flat = [key for group in sets for key in group]
        assert len(flat) == len(set(flat)), "a key was counted twice"
        # All nine registry keys plus the opponent-adjusted family, which is not a registry
        # key but is checked at its merge site since Plan 33.2-16.
        assert set(flat) == set(REGISTRY_KEYS) | {"opponent_adj"}
        assert "games" in report.checked_sources


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
