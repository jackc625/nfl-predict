"""SILVER **AND GOLD** rebuild from bronze with ZERO network calls (SPEC R3, Ruling W).

THIS IS THE ONLY TEST IN THE REPOSITORY THAT ASSERTS A CLAUDE.md HARD CONSTRAINT
END TO END, and it is deliberately heavy for that reason. It is ``slow``-marked so
the quick tier stays quick. Run it by name:

    uv run python -m pytest tests/integration/test_weather_rebuild_offline.py -q

WHY IT GOES ALL THE WAY TO GOLD
--------------------------------
SPEC R3's acceptance is "Rebuilding silver **and gold** from bronze afterwards
requires zero network calls", and the SPEC's ``## Constraints`` calls it a CLAUDE.md
hard constraint. A proof that stopped at the silver promotion would stop one step
short of the claim: the gold path runs three feature-build steps DOWNSTREAM of
silver -- ``step_build_contextual`` (``pipeline/steps.py:511-521``),
``step_build_weather_features`` (``:523-532``) and ``step_build_features``
(``:595-601``) -- and none of them would have been under the fixture. So ONE test
carries the whole chain:

    24 bronze snapshots
      -> silver ``weather``            (promote_corpus_to_silver)
      -> silver ``weather_features``   (step_build_weather_features)
      -> silver ``contextual_features``(step_build_contextual)
      -> gold features_wp / features_ats / features_ou  (step_build_features)

all of it inside ``sandbox_data_root``, and all of it under ``deny_network``.

Plan 33.1-07 Task 3 additionally runs the PRODUCTION gold rebuild under the same
fixture, so the property is proven on the real run as well as in the sandbox. That
requirement is cross-referenced in both plans so neither can drop it believing the
other has it.

THE REACHABILITY CONTROLS ARE WHAT MAKE THE GREEN MEAN ANYTHING
----------------------------------------------------------------
A rebuild that passes under a deny fixture proves only that this code path happened
not to call out -- unless the fixture is proven LIVE. There is therefore ONE control
per deny layer (archive, forecast, raw socket), each deliberately reaching for its
layer and asserting its distinctive error IS raised. A fixture that silently stopped
patching a layer fails HERE rather than passing silently everywhere.

A FOURTH control records something the first run of this module discovered: on
Windows ``asyncio.run`` cannot be used under ``deny_network`` at all, because
building a ProactorEventLoop opens a ``socket.socketpair()`` for the loop's own
self-pipe and the outer net fires on asyncio's plumbing before the coroutine
starts. The named-client controls therefore step their coroutine directly (see
``drive_once``). That collision is the outer net working, so it is pinned as a
control rather than worked around silently.

THE SANDBOX REFUSES BY NAME
----------------------------
``generate_feature_matrices`` reads its silver inputs through ``load_dataframe``.
Those table names are enumerated from the SOURCE (see
``silver_tables_read_by_generate_feature_matrices``), not typed here, and a sandbox
missing one of them REFUSES BY NAME -- so this test cannot pass by quietly building
less than it claims to build.

NOTHING HERE WRITES ``data/`` OR ``artifacts/``. ``monkeypatch`` alone is not a
sandbox; the writes go somewhere else, and ``data_boundary_guard`` proves it by
content rather than by intention.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import asyncio
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from scripts.backfill_historical_weather import (
    BACKFILL_BRONZE_GLOB,
    CORPUS_FIRST_SEASON,
    CORPUS_LAST_SEASON,
    load_pinned_game_facts,
    promote_corpus_to_silver,
)
from tests.fixtures.elo_sandbox import redirect_storage_to_sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_BRONZE = REPO_ROOT / "data" / "bronze"
PRODUCTION_SILVER = REPO_ROOT / "data" / "silver"
BUILD_FEATURES_SOURCE = REPO_ROOT / "scripts" / "build_features.py"

EXPECTED_CORPUS_ROWS = 6499
EXPECTED_SNAPSHOTS = 24
GOLD_TABLES = ("features_wp", "features_ats", "features_ou")

_SKIP_REASON = (
    "the weather_backfill bronze corpus is absent from data/bronze/ -- data/ is "
    "gitignored, so a fresh checkout does not have it. Produce it with:\n"
    "    uv run python -m scripts.backfill_historical_weather --bracket run"
)


def _production_snapshots() -> list[Path]:
    return sorted(PRODUCTION_BRONZE.glob(f"{BACKFILL_BRONZE_GLOB}.parquet"))


# DELIBERATELY NOT A MODULE-LEVEL SKIP. The corpus-dependent chain needs the 24
# gitignored snapshots, but the three REACHABILITY CONTROLS test the FIXTURE and
# must run on a fresh checkout -- they are what make every other deny-network claim
# in this repository mean anything, and a module-level skip would silence them
# exactly where nobody would notice.
requires_corpus = pytest.mark.skipif(
    len(_production_snapshots()) < EXPECTED_SNAPSHOTS, reason=_SKIP_REASON
)


# ---------------------------------------------------------------------------
# The sandbox's input contract, READ FROM THE SOURCE rather than typed.
# ---------------------------------------------------------------------------


def silver_tables_read_by_generate_feature_matrices() -> frozenset[str]:
    """Every silver table ``build_features.py`` loads through ``load_dataframe``.

    PARSED FROM THE SOURCE, not listed here. A hand-kept list goes stale exactly
    when a new source is added, and the failure it then produces is an empty
    left-merged column rather than an error -- which is the WR-10 defect
    ``load_all_feature_sources`` already carries a comment about.

    Only calls whose first argument is a STRING LITERAL and whose ``layer`` is
    ``"silver"`` are collected. ``load_dataframe(table_name, layer="gold")`` at
    ``:1935`` is a gold READ-BACK inside the save path and is correctly excluded by
    both conditions.
    """
    tree = ast.parse(BUILD_FEATURES_SOURCE.read_text(encoding="utf-8"))
    tables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "load_dataframe" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        layer = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "layer"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ),
            "silver",
        )
        if layer == "silver":
            tables.add(first.value)
    return frozenset(tables)


# The two tables this test DELETES from the sandbox before rebuilding, because they
# are what the rebuild is supposed to PRODUCE. Copying them in and then asserting
# they hold 6,499 rows would assert a property of `shutil.copy2`.
REBUILT_TABLES = frozenset({"weather", "weather_features"})


def lift_sandbox_games_to_post_wave_12(sandbox: Path) -> bool:
    """Give the sandbox's silver ``games`` a ``stadium_id``, if it has none.

    WHY THIS EXISTS, AND WHY IT IS NOT THIS PLAN PICKING ANOTHER PLAN'S DECISION.
    The SPEC's ``## Constraints`` states plainly that this phase RUNS AFTER Phase 33
    Wave 12, "which puts ``stadium_id`` into ``data/silver/games.parquet`` for all
    seasons". The tree is not in that state yet: production silver ``games`` is 15
    columns and carries no ``stadium_id``. Since D33.1-06 retired the home-team
    fallback, ``features.contextual.resolve_venue_for_game`` reads the game's own
    ``stadium_id`` and RAISES on ``None`` -- so ``step_build_contextual``, and
    therefore the whole gold build, cannot run against production silver today.
    MEASURED by this module on 2026-09-13: ``UnknownStadiumError: game
    '2018_W01_ATL@PHI' names stadium_id None``.

    That blocker is REAL and it belongs to Plan 33.1-07, which owns the choice
    between waiting for Wave 12 and joining ``stadium_id`` from the pinned feed.
    THIS FUNCTION DOES NOT MAKE THAT CHOICE. It puts the SANDBOX into the
    post-Wave-12 shape the SPEC says the phase assumes, so that R3's gold half can
    be proven in the environment the requirement describes, and it reports whether
    it had to -- see ``test_the_sandbox_records_which_silver_games_shape_it_found``.
    Production is untouched either way.

    The precedent is ``tests/conftest.per_game_roof_games``, whose
    ``with_silver_stadium_id`` flag exists for exactly this reason: "Ruling D2's two
    input shapes ... Asserting both is what stops this plan from depending on a
    guess about which state Wave 12 leaves the tree in."

    The id comes from ``load_pinned_game_facts`` -- the same pinned source
    ``scripts.backfill_historical_weather._reconcile_stadium_id`` reads, so the
    sandbox and the corpus cannot disagree about which stadium a game was at.

    Returns:
        True when the lift was applied (the tree was pre-Wave-12), False when the
        column was already there.
    """
    games_path = sandbox / "silver" / "games.parquet"
    games = pd.read_parquet(games_path, engine="pyarrow")
    if "stadium_id" in games.columns:
        return False

    seasons = list(range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1))
    facts = load_pinned_game_facts(seasons).set_index("game_id")["stadium_id"]
    games["stadium_id"] = games["game_id"].map(facts)

    unmapped = sorted(games.loc[games["stadium_id"].isna(), "game_id"])
    if unmapped:
        raise AssertionError(
            f"{len(unmapped)} sandbox game(s) could not be given a stadium_id from "
            f"the pinned feed. First ten: {unmapped[:10]}"
        )

    games.to_parquet(games_path, engine="pyarrow", index=False)
    return True


def scope_sandbox_games_to_the_corpus(sandbox: Path) -> int:
    """Cut the sandbox's silver ``games`` to the corpus seasons; return the rows dropped.

    FOUND BY PLAN 33.2-12, and pre-existing: production silver ``games`` now carries the
    live 2026 season (the Phase 33-18 capture), and the ERA5 corpus this class rebuilds
    covers 2002-2025 only, so ``step_build_weather_features`` refused the first 2026 game
    with no weather row (``2026_W02_DET@BUF``) -- measured at HEAD before any Plan 33.2-12
    change. The 2026 games are not part of the corpus this proof is about, so the sandbox
    is scoped to it rather than the step's refusal being loosened.
    """
    games_path = sandbox / "silver" / "games.parquet"
    games = pd.read_parquet(games_path, engine="pyarrow")
    scoped = games[games["season"].between(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON)]
    scoped.to_parquet(games_path, engine="pyarrow", index=False)
    return len(games) - len(scoped)


def populate_sandbox(sandbox: Path, silver_root: Path | None = None) -> list[Path]:
    """Copy the 24 bronze snapshots and every required silver table into *sandbox*.

    REFUSES BY NAME when a table ``generate_feature_matrices`` reads is absent from
    production, so a silently-narrower sandbox cannot pass this test by building
    less than it claims.

    Args:
        sandbox: The sandbox data root to populate.
        silver_root: Where to copy silver from. Defaults to production silver; the
            parameter exists so the refusal branch can be EXERCISED against an empty
            directory rather than left as an untested branch, and an untested
            refusal is indistinguishable from no refusal.

    Returns:
        The bronze snapshot paths copied into the sandbox.
    """
    silver_root = PRODUCTION_SILVER if silver_root is None else silver_root
    required = silver_tables_read_by_generate_feature_matrices()
    absent = sorted(
        name for name in required if not (silver_root / f"{name}.parquet").is_file()
    )
    if absent:
        raise AssertionError(
            "the sandbox cannot be populated: "
            f"{absent} -- table(s) generate_feature_matrices reads through "
            "load_dataframe are ABSENT from production silver. Building without "
            "them would produce a gold matrix with those families all-null while "
            "this test still reported success, which is the one failure mode a "
            "rebuild proof must not have. Required set, parsed from "
            f"scripts/build_features.py: {sorted(required)}"
        )

    (sandbox / "bronze").mkdir(parents=True, exist_ok=True)
    (sandbox / "silver").mkdir(parents=True, exist_ok=True)
    (sandbox / "gold").mkdir(parents=True, exist_ok=True)

    snapshots = _production_snapshots()
    for path in snapshots:
        shutil.copy2(path, sandbox / "bronze" / path.name)

    # EVERY silver table, not only the enumerated three: the on-the-fly builders
    # (Elo, market anchors, snaps, injuries) read their own tables and degrade to
    # empty frames when they cannot, so copying the whole layer keeps the rebuild
    # realistic rather than minimal. The enumerated three are the ones whose
    # ABSENCE is a refusal.
    for path in sorted(silver_root.glob("*.parquet")):
        if path.stem in REBUILT_TABLES:
            continue
        shutil.copy2(path, sandbox / "silver" / path.name)

    return snapshots


@pytest.fixture(scope="module")
def pinned_game_ids() -> frozenset[str]:
    seasons = list(range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1))
    return frozenset(load_pinned_game_facts(seasons)["game_id"])


# ---------------------------------------------------------------------------
# THE CHAIN.
# ---------------------------------------------------------------------------


class TestSilverAndGoldRebuildOffline:
    """Ruling W: bronze -> silver -> the two feature tables -> all three gold."""

    @requires_corpus
    @pytest.mark.slow
    def test_silver_and_gold_rebuild_from_bronze_with_zero_network_calls(
        self,
        tmp_path,
        monkeypatch,
        deny_network,
        pinned_game_ids,
        data_boundary_guard,
        artifacts_boundary_guard,
    ):
        from pipeline.steps import step_build_contextual, step_build_weather_features
        from scripts.build_features import FeatureMatrixBuilder

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        snapshots = populate_sandbox(sandbox)
        # The SPEC says this phase runs AFTER Wave 12, so the sandbox is put into
        # the shape the requirement assumes. Production is untouched, and whether
        # the lift was needed is recorded by its own test below.
        lift_sandbox_games_to_post_wave_12(sandbox)
        scope_sandbox_games_to_the_corpus(sandbox)
        assert len(snapshots) == EXPECTED_SNAPSHOTS, (
            f"the sandbox holds {len(snapshots)} bronze snapshots, not "
            f"{EXPECTED_SNAPSHOTS} -- one per season 2002 through 2025."
        )
        for name in REBUILT_TABLES:
            assert not (sandbox / "silver" / f"{name}.parquet").exists(), (
                f"silver/{name}.parquet was copied into the sandbox. It is what "
                "this test REBUILDS, so its presence would let the row-count "
                "assertion below pass on a copy."
            )

        # -- silver `weather`, from the 24 bronze snapshots and nothing else --
        promotion = promote_corpus_to_silver(base_path=sandbox)
        assert promotion["rows"] == EXPECTED_CORPUS_ROWS
        assert promotion["legacy_files_read"] == ()
        weather = pd.read_parquet(sandbox / "silver" / "weather.parquet")
        assert len(weather) == EXPECTED_CORPUS_ROWS
        assert frozenset(weather["game_id"]) == pinned_game_ids

        # -- the two feature tables the gold path reads, over that sandbox silver --
        step_build_weather_features()
        step_build_contextual()

        weather_features = pd.read_parquet(
            sandbox / "silver" / "weather_features.parquet"
        )
        assert len(weather_features) == EXPECTED_CORPUS_ROWS, (
            f"weather_features rebuilt to {len(weather_features)} rows, not "
            f"{EXPECTED_CORPUS_ROWS}. This table is the gold path's weather input; "
            "a short one silently left-merges to null."
        )
        assert (sandbox / "silver" / "contextual_features.parquet").is_file()

        # -- all three gold matrices --
        builder = FeatureMatrixBuilder()
        matrices = builder.generate_feature_matrices()
        builder.save_feature_matrices(matrices)

        for table in GOLD_TABLES:
            path = sandbox / "gold" / f"{table}.parquet"
            assert path.is_file(), (
                f"gold/{table}.parquet was not produced by the offline rebuild. "
                "SPEC R3 says silver AND gold rebuild from bronze; a chain that "
                "stops at silver does not assert the requirement."
            )
            frame = pd.read_parquet(path)
            assert len(frame) == EXPECTED_CORPUS_ROWS, (
                f"gold/{table}.parquet holds {len(frame)} rows, not "
                f"{EXPECTED_CORPUS_ROWS}."
            )
            assert frozenset(frame["game_id"]) == pinned_game_ids, (
                f"gold/{table}.parquet does not carry the pinned feed's game_id "
                "set exactly."
            )

    def test_a_sandbox_missing_a_required_silver_table_refuses_by_name(self, tmp_path):
        """The refuse-by-name guard fires, and NAMES the table.

        Without this the populate helper's refusal is an untested branch, and an
        untested refusal is indistinguishable from no refusal at all.
        """
        required = sorted(silver_tables_read_by_generate_feature_matrices())
        assert required, "the source scan found no silver load_dataframe calls"

        # A silver root holding EVERY required table except one, so the refusal
        # fires on a specific absence rather than on an empty directory -- an empty
        # directory would pass this assertion while the guard only knew how to say
        # "nothing here".
        victim = required[0]
        narrowed = tmp_path / "narrowed_silver"
        narrowed.mkdir()
        for name in required[1:]:
            (narrowed / f"{name}.parquet").write_bytes(b"")

        with pytest.raises(AssertionError) as excinfo:
            populate_sandbox(tmp_path / "sandbox", silver_root=narrowed)

        assert victim in str(excinfo.value), (
            "the refusal did not NAME the absent table. A refusal that says only "
            "'a table is missing' sends the reader back to the filesystem."
        )

    @requires_corpus
    def test_the_sandbox_records_which_silver_games_shape_it_found(
        self, tmp_path, data_boundary_guard
    ):
        """Which Wave-12 state the tree is in, recorded rather than assumed.

        THIS IS THE BLOCKER PLAN 33.1-07 OWNS, pinned as an assertion so that plan
        meets it as a fact rather than as prose in a SUMMARY. Since D33.1-06 retired
        the home-team fallback, ``resolve_venue_for_game`` raises on a ``None``
        ``stadium_id``, so ``step_build_contextual`` -- and with it the entire gold
        build -- cannot run against a pre-Wave-12 silver ``games`` table.

        The test is written to stay GREEN across the transition: it asserts only
        that the two states are told apart correctly, never that the tree is in one
        of them. A test that asserted "production is pre-Wave-12" would turn RED the
        day Wave 12 landed and would be a false tripwire.
        """
        production_games = pd.read_parquet(
            PRODUCTION_SILVER / "games.parquet", engine="pyarrow"
        )
        production_has_it = "stadium_id" in production_games.columns

        sandbox = tmp_path / "shape_probe"
        populate_sandbox(sandbox)
        lifted = lift_sandbox_games_to_post_wave_12(sandbox)

        assert lifted is not production_has_it, (
            "the lift and the production shape disagree: lifted="
            f"{lifted}, production_carries_stadium_id={production_has_it}. The "
            "lift must fire exactly when the column is absent."
        )

        after = pd.read_parquet(sandbox / "silver" / "games.parquet", engine="pyarrow")
        assert "stadium_id" in after.columns
        assert after["stadium_id"].notna().all(), (
            "the sandbox games table still carries a null stadium_id after the "
            "lift; step_build_contextual would raise on those rows by name."
        )

        # A second application is a no-op, so the helper is safe to call twice.
        assert lift_sandbox_games_to_post_wave_12(sandbox) is False

    def test_the_required_silver_tables_are_parsed_from_the_source(self):
        """The enumeration tracks ``build_features.py``, and is not a typed list."""
        required = silver_tables_read_by_generate_feature_matrices()

        assert "games" in required, (
            "the source scan lost `games`, which is the BASE frame every gold row "
            "is built from. The scan is broken, not the source."
        )
        assert "weather_features" in required, (
            "the source scan lost `weather_features`, which is the table this "
            "whole phase rebuilds."
        )
        assert "features_wp" not in required, (
            "the scan collected a GOLD table. load_dataframe(table_name, "
            "layer='gold') at build_features.py:1935 is a read-back inside the "
            "save path and must not be treated as an input."
        )


# ---------------------------------------------------------------------------
# PLAN 33.2-12 (p332_ rung 4): THE HISTORY PATH IS NOW THE MOS BRONZE.
#
# Since rung 4, 2002-2025 silver `weather` and `weather_features` are regenerated from the
# archived day-before NWS MOS bulletins in data/bronze/mos/ (scripts/weather_from_mos.py,
# `scripts.build_weather --all-seasons --from-bronze`), not promoted from the ERA5 corpus.
# The class above still proves the (quarantined) ERA5 promotion chain rebuilds offline; the
# class below proves the path production now uses does.
# ---------------------------------------------------------------------------

MOS_BRONZE = PRODUCTION_BRONZE / "mos"
MOS_REBUILT_TABLES = ("weather", "weather_features")

requires_mos_bronze = pytest.mark.skipif(
    not any(MOS_BRONZE.glob("mos_bulletins_raw_bronze_*.parquet")),
    reason=(
        "data/bronze/mos/ is absent (data/ is gitignored). Produce it with: "
        "    uv run python -m scripts.backfill_mos_forecasts --seasons 2002-2025 --apply"
    ),
)


class TestTheMosHistoryRebuildsOfflineByteIdentically:
    """Regenerating silver weather from bronze, network denied, is byte-reproducible."""

    @requires_mos_bronze
    @pytest.mark.slow
    def test_regeneration_is_offline_idempotent_and_reproduces_production(
        self, tmp_path, data_boundary_guard, artifacts_boundary_guard
    ):
        """The network is denied by the regeneration's own counting guard.

        Not layered on the module's ``deny_network`` fixture: that fixture replaces
        ``socket.socket`` itself, and the counting guard patches ``socket.socket.connect``
        so it can report HOW MANY connections were attempted, not merely that one was.
        """
        from scripts.weather_from_mos import deny_network as counting_guard
        from scripts.weather_from_mos import regenerate_history

        root = tmp_path / "mos_sandbox"
        (root / "silver").mkdir(parents=True)
        shutil.copytree(MOS_BRONZE, root / "bronze" / "mos")
        for table in ("games", *MOS_REBUILT_TABLES):
            shutil.copy2(
                PRODUCTION_SILVER / f"{table}.parquet",
                root / "silver" / f"{table}.parquet",
            )

        written: list[dict[str, bytes]] = []
        for _ in range(2):
            calls = [0]
            with counting_guard(calls):
                report = regenerate_history(root)
            assert calls[0] == 0, f"{calls[0]} network connection(s) were attempted"
            assert report.observation_rows == 0
            written.append(
                {
                    table: (root / "silver" / f"{table}.parquet").read_bytes()
                    for table in MOS_REBUILT_TABLES
                }
            )

        assert written[0] == written[1], (
            "a second regeneration over its own output wrote different bytes: the "
            "regeneration is not a function of the bronze it reads"
        )
        for table in MOS_REBUILT_TABLES:
            pd.testing.assert_frame_equal(
                pd.read_parquet(root / "silver" / f"{table}.parquet"),
                pd.read_parquet(PRODUCTION_SILVER / f"{table}.parquet"),
                obj=f"silver {table}: sandbox regeneration vs production",
            )


# ---------------------------------------------------------------------------
# THE REACHABILITY CONTROLS -- one per deny layer.
# ---------------------------------------------------------------------------


def drive_once(coroutine):
    """Step *coroutine* to its first suspension, WITHOUT an asyncio event loop.

    MEASURED, NOT STYLISTIC (Plan 33.1-06 Task 4). ``asyncio.run`` cannot be used
    under ``deny_network`` on Windows: building a ProactorEventLoop calls
    ``socket.socketpair()`` for the loop's own self-pipe
    (``asyncio/proactor_events.py:781``), so the OUTER NET fires on asyncio's
    plumbing before the coroutine is ever started, and the archive control fails
    with ``SocketOpenedUnderDenyNetwork`` instead of the archive error it exists to
    assert.

    That collision is not a defect in the fixture -- it is the outer net working,
    and it is incidental evidence that layer three genuinely catches paths nobody
    enumerated. But a control has to reach the layer it is testing. Each deny stub
    raises on its first step, so stepping the coroutine once is sufficient and
    needs no loop at all.

    Anything else that runs under ``deny_network`` and needs a real event loop has
    the same problem; that is recorded in the ``deny_network`` docstring.
    """
    try:
        coroutine.send(None)
    except StopIteration as stop:  # pragma: no cover - the stubs always raise
        return stop.value
    finally:
        coroutine.close()
    return None


class TestEveryDenyLayerIsProvenLive:
    """Each layer of ``deny_network`` is reached on purpose and asserted to fire.

    Without these, a green offline rebuild proves only that this code path happened
    not to call out. With only the named-client controls it proves less than SPEC R3
    claims, because R3 is a statement about EVERY path and a named-client patch is
    an enumeration.
    """

    def test_reachability_control_the_archive_fetch_layer_is_live(self, deny_network):
        import scripts.backfill_historical_weather as backfill

        with pytest.raises(deny_network["archive"]):
            drive_once(
                backfill.fetch_game_weather(
                    None, 40.0, -74.0, "2016-09-11", 13, "America/New_York"
                )
            )

    def test_reachability_control_the_archive_floor_probe_layer_is_live(
        self, deny_network
    ):
        """The SECOND archive entry point, which is not the same function.

        ``assert_archive_covers_corpus_floor`` reaches the archive through
        ``_probe_archive_day``. A fixture that patched only ``fetch_game_weather``
        would leave a live network path open through the coverage probe.
        """
        import scripts.backfill_historical_weather as backfill

        with pytest.raises(deny_network["archive"]):
            drive_once(
                backfill._probe_archive_day(
                    None, 40.0, -74.0, "2002-09-05", "America/New_York"
                )
            )

    def test_reachability_control_the_forecast_client_layer_is_live(self, deny_network):
        import scripts.ingest_weather as ingest

        with pytest.raises(deny_network["forecast"]):
            drive_once(
                ingest.fetch_game_forecast(
                    None, 40.0, -74.0, "2026-09-11", 13, "America/New_York"
                )
            )

    def test_the_outer_net_catches_asyncios_own_plumbing_on_windows(self, deny_network):
        """A fourth control, recording the collision the other three had to dodge.

        This is not decoration. It pins a real property of the three-layer fixture:
        the socket layer is outer enough to catch a path that is not a network call
        at all -- the event loop's internal self-pipe. Anyone who later 'fixes' the
        archive controls by reaching for ``asyncio.run`` again will find this test
        already explaining why that does not work.
        """
        with pytest.raises(deny_network["socket"]):
            asyncio.run(asyncio.sleep(0))

    def test_reachability_control_the_raw_socket_layer_is_live(self, deny_network):
        """The OUTER NET. Layers one and two are an enumeration; this one is not."""
        with pytest.raises(deny_network["socket"]):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_the_three_deny_layers_raise_three_distinct_types(self, deny_network):
        """A single shared error type could not tell the operator which layer fired."""
        types = [
            deny_network["archive"],
            deny_network["forecast"],
            deny_network["socket"],
        ]
        assert len(set(types)) == 3, (
            f"the deny layers do not raise distinct types: {types}. A control "
            "asserting a shared type would pass while two layers were unpatched."
        )
