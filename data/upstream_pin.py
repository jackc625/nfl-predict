"""Pinned upstream snapshots of the nflverse frames a gold rebuild consumes.

WHY THIS EXISTS
---------------
``CLAUDE.md`` states the constraint the whole project rests on: "any prediction must be
reproducible given the same input data snapshot". Until this module existed there WAS no
input data snapshot. A full gold rebuild reached nflverse LIVE at three call sites --
``features/team_form.py`` (play-by-play), ``features/qb_tracking.py`` (play-by-play and
depth charts) and ``scripts/ingest_games.py`` (schedules and play-by-play) -- with
``nflreadpy``'s default ``CacheMode.MEMORY``, which is per-process and therefore refetches
on every run.

Phase 31 measured the consequence rather than theorising it. Two gold builds of the same
code, 2026-08-22 and 2026-09-04, disagreed on sixteen columns
(``{home,away}_{off,def}_rolling_opp_adj_*``, ``{home,away}_qb_adjustment``,
``{home,away}_backup_quality_delta``) from season 2020 onward -- INSIDE the protected
2021-2024 holdout the deploy gate's frozen baseline was measured against. Nothing before
2020 moved, which is the signature of an nflverse play-by-play re-release, not of a code
change. The 2026-08-22 gold is unrecoverable and the frozen ATS baseline CLV no longer
matches what the same computation now yields. A pinned input is what makes that class of
loss impossible to repeat.

A FILESYSTEM CACHE IS NOT A PIN. ``nflreadpy``'s ``CacheMode.FILESYSTEM`` is keyed by a
``cache_duration`` (86400 seconds by default) and EXPIRES. It is a speed optimisation:
after a day it silently refetches, which is exactly the drift being fixed. This module
stores immutable, digest-recorded parquet snapshots under ``data/bronze/`` -- the layer
``CLAUDE.md`` and ``PIPELINE.md`` already define as "Raw snapshots (append-only,
timestamped)" -- and verifies their bytes on every read.

FAIL CLOSED, NEVER SILENTLY LIVE
--------------------------------
The pin is the DEFAULT. When a requested season is not pinned the loaders RAISE
:class:`UpstreamPinMissing` naming the missing seasons and the command that captures them.
They do NOT quietly fall back to the network, because a silent fallback reintroduces the
exact drift the pin exists to remove and does it invisibly.

Refetching live is available and is an EXPLICIT, LOUD opt-in: set the environment variable
named by :data:`LIVE_OPT_IN_ENV`. Every live fetch emits a
:class:`UpstreamPinBypassedWarning` through ``warnings.warn`` AND a structured log
warning, so a bypass appears in pytest output and in the run log rather than only in
someone's shell history.

:class:`UpstreamPinError` deliberately inherits from ``Exception`` and NOT from
``RuntimeError``, ``ValueError`` or ``ImportError``. Every one of the wired call sites sits
inside an ``except`` clause naming those types and returning an EMPTY frame on failure
(``features/qb_tracking.py`` does exactly that). A pin error caught by one of those
handlers would be converted into an empty play-by-play frame and a silently degraded gold
matrix -- a worse outcome than the drift. ``test_upstream_pin.py`` holds that inheritance
as a test.

TWO ZONES: SEALED AND LIVE
--------------------------
Phase 32 splits the pin in two along a season boundary. Seasons at or before
:data:`SEALED_THROUGH_SEASON` are SEALED: they are captured once into
``config/upstream_pin.json`` and their bytes never move again, so a diff on that file is
always a red flag. Season :data:`LIVE_ZONE_FIRST_SEASON` is LIVE: it is captured week by
week into ``config/upstream_live/<season>.json`` (see :mod:`data.upstream_live`), it grows
by design, and a diff on THAT file is always expected. The two records must never be one
file, because their diffs mean opposite things.

Seasons beyond the live zone belong to NO zone and are refused rather than silently
admitted. Promoting the boundary forward is a deliberate, one-way act that cannot be
rehearsed before the live season actually ends.

WHAT IS PINNED, AND WHAT IS NOT
-------------------------------
Pinned: ``load_pbp`` (column-narrowed, see :data:`PBP_PINNED_COLUMNS`), ``load_schedules``
(full frame) and ``load_depth_charts`` (full frame).

The play-by-play pin is NARROWED ON COLUMNS and complete on rows. A full-width 2002-2025
play-by-play vendoring is roughly 500 MB (20.7 MB for 2024 alone at 49,492 x 372); the
23 columns the builders actually read are 1.1-1.3 MB per season, about 30 MB in total.
The narrowing is enumerated and attributed below, and it is PROVEN rather than asserted:
the rung-1 fingerprint comparison rebuilds gold from this pin and requires zero non-clock
column moves against the live-built gold it replaces. A missing column would either raise
or move a value, and the ladder would report it.

NOT pinned, deliberately, and named here so the boundary of the reproducibility claim is
legible: ``nflreadpy.load_injuries`` and ``nflreadpy.load_snap_counts``. Those two already
write their own timestamped ``data/bronze/`` snapshots through ``scripts/ingest_injuries.py``
and ``scripts/ingest_snaps.py`` and reach gold through ``data/silver/``, so their inputs
are already frozen on disk; re-pinning them here would create a second, competing snapshot
of the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

# The committed provenance record. ``data/`` is gitignored, so the parquet snapshots
# themselves are NOT in git; this manifest is, which is what lets a fresh checkout say
# WHICH upstream revision a verdict was measured against even when the bytes are absent.
MANIFEST_PATH = Path("config/upstream_pin.json")

# The committed LOCK on the sealed half of that manifest. Same "committed record,
# gitignored bytes" asymmetry, one turn tighter: the manifest records WHAT was pinned, the
# lock records that the sealed half has not MOVED since it was locked. A hand edit to
# ``config/upstream_pin.json`` that never runs the capture tool shows only as a git diff,
# and a git diff is something a reader has to notice. Divergence from this file FAILS
# ``tests/unit/test_sealed_zone_refusal.py`` instead, which is something nobody can miss.
#
# SERIALISATION: JSON, not TOML, and the reason is recorded in the file's own
# ``format_note`` too. ``json.dumps(indent=2)`` is a standard-library writer that already
# produces this repository's committed-record style -- it is exactly how
# ``scripts/pin_upstream_snapshot.py`` writes ``config/upstream_pin.json``. The
# ``config/*.toml`` verdict records exist only because the standard library has NO TOML
# writer and the hand-pasted block IS the discipline there
# (``backtest/group_gate.py::render_verdict_toml``). The ``.lock`` extension is kept
# because it is the path ``32-CONTEXT.md`` and ``32-RESEARCH.md`` both name literally; the
# extension names the CONTRACT, not the serialisation.
SEALED_LOCK_PATH: Path = Path("config/upstream_pin.sealed.lock")

# Set this to any non-empty value to permit a live nflverse fetch. Nothing sets it
# implicitly; the pipeline, the tests and the rebuild all run with it unset.
LIVE_OPT_IN_ENV = "NFL_PREDICT_ALLOW_LIVE_UPSTREAM"

MANIFEST_SCHEMA_VERSION = 1
SEALED_LOCK_SCHEMA_VERSION: int = 1

# The zone boundary, with the consumer of each half named.
#
# SEALED_THROUGH_SEASON -- the last season the sealed pin owns. Read by
#   ``zone_for_season``, by ``scripts/capture_live_season.py`` (which refuses to write
#   anything at or below it) and, from Plan 32-02, by ``scripts/pin_upstream_snapshot.py``
#   (which refuses to REWRITE anything at or below it without an explicit override).
# LIVE_ZONE_FIRST_SEASON -- the one season the live, append-only, per-week zone owns. Read
#   by ``data/upstream_live.py`` and by the live capture CLI.
#
# WHY A LITERAL AND NOT ``nflreadpy.get_current_season()``: that helper flips to the new
# season on the Thursday following Labor Day. A computed boundary would therefore move the
# sealed zone SILENTLY and MID-WEEK -- a season that was immutable on Wednesday would
# become writable on Thursday, with nothing on record to attribute the change to. The
# boundary moves only when a human edits this line, which is the whole point: promoting a
# season from live to sealed is one-way and cannot be undone by re-running anything.
SEALED_THROUGH_SEASON: int = 2025
LIVE_ZONE_FIRST_SEASON: int = SEALED_THROUGH_SEASON + 1

ZONE_SEALED = "sealed"
ZONE_LIVE = "live"
ZONE_UNKNOWN = "unknown"

# The committed live-zone manifest directory, as TEXT. ``data/upstream_live.py`` owns the
# real ``Path`` constant; naming it here as a string keeps this module free of an
# import-time dependency on its own consumer (the only link runs the other way, through a
# function-local deferred import).
LIVE_MANIFEST_DIR_TEXT = "config/upstream_live"

# Every play-by-play column any gold-rebuild consumer reads, with the consumer named.
# Adding a consumer that needs a column outside this tuple requires re-capturing the pin;
# the failure mode is a loud KeyError inside the builder, not a silent wrong number.
PBP_PINNED_COLUMNS: tuple[str, ...] = (
    # Grain and identity -- all three consumers.
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    # scripts/ingest_games.py:173 fetch_pbp_data -- per-game result extraction.
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    # features/team_form.py -- fetch_pbp_data filter and calculate_team_game_stats.
    "play_type",
    "epa",
    "success",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "touchdown",
    "first_down",
    "cpoe",
    "fixed_drive",
    # features/qb_tracking.py -- compute_qb_game_stats.
    "passer_player_id",
    "qb_epa",
)

# Frames pinned whole. Schedules are ~285 rows x ~46 columns per season and depth charts
# a few thousand rows; narrowing either would buy nothing and would put a second column
# judgement in the reproducibility claim for no reason.
DATASET_COLUMNS: dict[str, tuple[str, ...] | None] = {
    "pbp": PBP_PINNED_COLUMNS,
    "schedules": None,
    "depth_charts": None,
}

# The bronze ``table_name`` each dataset is stored under, following the existing
# ``<name>_raw_bronze_<season>_W00_<UTCstamp>.parquet`` convention.
DATASET_TABLE_NAMES: dict[str, str] = {
    "pbp": "pbp",
    "schedules": "schedules",
    "depth_charts": "depth_charts",
}

DATASET_LOADERS: dict[str, str] = {
    "pbp": "nflreadpy.load_pbp",
    "schedules": "nflreadpy.load_schedules",
    "depth_charts": "nflreadpy.load_depth_charts",
}

_CHUNK_BYTES = 1 << 20


class UpstreamPinError(Exception):
    """Base class for every pin refusal.

    Inherits ``Exception`` and NOT ``RuntimeError``/``ValueError``/``ImportError`` on
    purpose. ``features/qb_tracking.py`` catches ``(ImportError, ValueError,
    RuntimeError)`` around its loaders and returns an EMPTY DataFrame; ``features/
    team_form.py`` and ``scripts/ingest_games.py`` catch similar tuples. A pin refusal
    that landed in one of those handlers would be converted into an empty play-by-play
    frame and a silently degraded gold matrix. Refusing loudly is the whole point, so the
    exception is deliberately outside every existing handler's reach.
    """


class UpstreamPinMissing(UpstreamPinError):
    """The pin does not cover a requested season and live fetching was not opted in to."""


class UpstreamPinCorrupt(UpstreamPinError):
    """A pinned file is absent, or its bytes no longer match the recorded digest."""


class UpstreamLiveCaptureMissing(UpstreamPinError):
    """The live zone has no capture matching the requested (season, week, sequence).

    Refuses a READ against the live manifest. The live zone is append-only and per-week,
    so "season 2026 is covered" is not the same claim as "week 6 of season 2026 was
    captured". Asking for a week nobody captured must name the weeks that DO exist and
    the command that would create the missing one, never fall back to the newest capture
    -- a replay that silently served a different week would reproduce the wrong number
    while reporting success.
    """


class UpstreamSeasonWindowRefused(UpstreamPinError):
    """A requested season window cannot be served as asked, and will not be approximated.

    Refuses a request whose SHAPE is unsatisfiable rather than whose bytes are missing --
    an as-of context that excludes every capture of a requested week, or a season span
    the two zones cannot jointly cover in the order it was asked for. Narrowing the window
    silently would hand a builder fewer seasons than it asked for and let it compute a
    rolling feature off a shorter history than its caller believes.
    """


class ZoneWriteRefused(UpstreamPinError):
    """A write was aimed at the wrong zone, and was refused before anything was written.

    Refuses a WRITE. The sealed zone and the live zone have opposite mutability
    contracts, so each capture tool refuses the other's seasons outright: the live
    capture will not touch a season at or below :data:`SEALED_THROUGH_SEASON`, and the
    sealed capture will not rewrite one without an explicit, recorded override. The
    refusal happens before any fetch, so a mis-aimed run costs nothing and changes
    nothing.
    """


class UpstreamLiveCorrupt(UpstreamPinCorrupt):
    """A live-zone manifest or capture is unreadable, absent, or no longer its own bytes.

    The live-zone sibling of :class:`UpstreamPinCorrupt`, and a SUBCLASS of it so every
    existing handler that already treats a corrupt pin as fatal treats a corrupt capture
    the same way. Raised for a manifest whose ``schema_version`` this build does not
    understand (a version refusal, never a migration) and for a capture file whose bytes
    no longer match the digest recorded for them.
    """


class UpstreamPinBypassedWarning(UserWarning):
    """A live nflverse fetch happened because the operator explicitly allowed it."""


def zone_for_season(season: int) -> str:
    """Return which pin zone *season* belongs to.

    Exactly three answers, and the third is not an error case to be tidied away later:

    * :data:`ZONE_SEALED` -- at or before :data:`SEALED_THROUGH_SEASON`. Immutable.
    * :data:`ZONE_LIVE` -- exactly :data:`LIVE_ZONE_FIRST_SEASON`. Append-only, per week.
    * :data:`ZONE_UNKNOWN` -- anything later. Deliberately owned by NEITHER zone: the
      one-way promotion that would move the boundary forward cannot be exercised until
      the live season actually ends, so a later season must REFUSE rather than be
      silently admitted to the live zone and captured under semantics nobody ratified.
    """
    if season <= SEALED_THROUGH_SEASON:
        return ZONE_SEALED
    if season == LIVE_ZONE_FIRST_SEASON:
        return ZONE_LIVE
    return ZONE_UNKNOWN


def digest_file(path: Path) -> str:
    """Return the sha256 of *path*'s bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_data_root() -> Path:
    """The data lake root, read from settings so ``DATA_ROOT_PATH`` still redirects it."""
    from conf.settings import get_settings

    return Path(get_settings().config.data.root_path)


def load_manifest(manifest_path: Path | str | None = None) -> dict | None:
    """Return the pin manifest, or ``None`` when no pin has been captured.

    A missing manifest is not an error here: it is the state of a fresh checkout, and the
    refusal that matters belongs at the point of a load, where the message can name the
    seasons that were actually asked for.
    """
    path = Path(manifest_path) if manifest_path is not None else MANIFEST_PATH
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        msg = (
            f"Upstream pin manifest at '{path}' declares schema_version {version!r}, but "
            f"this build understands version {MANIFEST_SCHEMA_VERSION}. Re-capture the "
            "pin with `python -m scripts.pin_upstream_snapshot` rather than reading a "
            "manifest whose meaning is not the meaning this code assigns it."
        )
        raise UpstreamPinCorrupt(msg)
    return manifest


def pinned_seasons(dataset: str, manifest: dict | None) -> list[int]:
    """Return the sorted seasons *dataset* is pinned for, or ``[]`` when it is not."""
    if not manifest:
        return []
    entry = manifest.get("datasets", {}).get(dataset)
    if not entry:
        return []
    return sorted(int(season) for season in entry.get("seasons", {}))


def load_sealed_lock(lock_path: Path | str | None = None) -> dict | None:
    """Return the sealed-zone lock, or ``None`` when none has been generated.

    A missing lock is not an error here, for the same reason a missing manifest is not:
    it is the state of a checkout that has never generated one. The refusal that matters
    belongs at the point of comparison, where the message can name the pairs that moved.

    An unrecognised ``schema_version`` is a HARD REFUSAL and never a migration, exactly
    as in :func:`load_manifest`. A lock read under the wrong meaning would compare two
    documents that agree on their keys and disagree on what those keys assert, which is
    worse than having no lock at all.
    """
    path = Path(lock_path) if lock_path is not None else SEALED_LOCK_PATH
    if not path.is_file():
        return None
    lock = json.loads(path.read_text(encoding="utf-8"))
    version = lock.get("schema_version")
    if version != SEALED_LOCK_SCHEMA_VERSION:
        msg = (
            f"Sealed-zone lock at '{path}' declares schema_version {version!r}, but this "
            f"build understands version {SEALED_LOCK_SCHEMA_VERSION}. Regenerate it with "
            "`python -m scripts.pin_upstream_snapshot --refresh-sealed-lock "
            '--sealed-rewrite-reason "<why>"` rather than reading a lock whose meaning '
            "is not the meaning this code assigns it."
        )
        raise UpstreamPinCorrupt(msg)
    return lock


def _sealed_pairs_from_manifest(manifest: dict) -> dict[tuple[str, int], dict]:
    """Every ``(dataset, season)`` in *manifest* whose season is in the SEALED zone."""
    pairs: dict[tuple[str, int], dict] = {}
    for dataset, record in manifest.get("datasets", {}).items():
        for season, entry in record.get("seasons", {}).items():
            if zone_for_season(int(season)) == ZONE_SEALED:
                pairs[(dataset, int(season))] = entry
    return pairs


def _sealed_pairs_from_lock(lock: dict) -> dict[tuple[str, int], dict]:
    """Every ``(dataset, season)`` the lock records, sealed-zone or not.

    Non-sealed entries are returned rather than filtered out: a live-zone season sitting
    in the SEALED lock is itself a problem to report, not noise to discard.
    """
    pairs: dict[tuple[str, int], dict] = {}
    for dataset, seasons in lock.get("datasets", {}).items():
        for season, entry in seasons.items():
            pairs[(dataset, int(season))] = entry
    return pairs


def sealed_lock_problems(lock: dict | None, manifest: dict | None) -> list[str]:
    """Return the divergences between the sealed lock and the pin manifest.

    EMPTY means intact -- the same contract, and the same reported shape, as
    ``scripts/pin_upstream_snapshot.py::verify``. A list is returned rather than a bool
    so the assertion that consumes it prints the offenders instead of printing ``False``.

    Only the SEALED span is compared. Live-zone seasons legitimately move week by week
    and are governed by ``config/upstream_live/<season>.json``; comparing them here would
    make the lock fail every week by design. A live-zone season PRESENT in the sealed
    lock is the opposite case and is reported, because the two records must never merge.
    """
    problems: list[str] = []
    if lock is None:
        problems.append(
            f"no sealed-zone lock at '{SEALED_LOCK_PATH}'. Generate it with "
            "`python -m scripts.pin_upstream_snapshot --refresh-sealed-lock "
            '--sealed-rewrite-reason "<why>"`.'
        )
    if manifest is None:
        problems.append(f"no upstream pin manifest at '{MANIFEST_PATH}'")
    if lock is None or manifest is None:
        return problems

    locked = _sealed_pairs_from_lock(lock)
    pinned = _sealed_pairs_from_manifest(manifest)

    for (dataset, season), entry in sorted(locked.items()):
        zone = zone_for_season(season)
        if zone != ZONE_SEALED:
            problems.append(
                f"{dataset} {season}: LIVE-ZONE SEASON IN SEALED LOCK (zone {zone}). The "
                "sealed lock and the live manifest have opposite mutability contracts; a "
                f"season above {SEALED_THROUGH_SEASON} belongs in "
                f"{LIVE_MANIFEST_DIR_TEXT}/{season}.json, never here."
            )
            continue
        if (dataset, season) not in pinned:
            problems.append(
                f"{dataset} {season}: MISSING FROM MANIFEST. The lock records this "
                f"sealed season but {MANIFEST_PATH} no longer pins it, so a sealed entry "
                "was deleted."
            )
            continue
        recorded = pinned[(dataset, season)]
        if entry.get("sha256") != recorded.get("sha256"):
            problems.append(
                f"{dataset} {season}: SHA256 MOVED\n"
                f"    locked   {entry.get('sha256')}\n"
                f"    manifest {recorded.get('sha256')}"
            )
        if entry.get("rows") != recorded.get("rows"):
            problems.append(
                f"{dataset} {season}: ROWS MOVED\n"
                f"    locked   {entry.get('rows')}\n"
                f"    manifest {recorded.get('rows')}"
            )
        if entry.get("bytes") != recorded.get("bytes"):
            problems.append(
                f"{dataset} {season}: BYTES MOVED\n"
                f"    locked   {entry.get('bytes')}\n"
                f"    manifest {recorded.get('bytes')}"
            )

    for dataset, season in sorted(pinned):
        if (dataset, season) not in locked:
            problems.append(
                f"{dataset} {season}: MISSING FROM LOCK. {MANIFEST_PATH} pins this "
                "sealed season but the lock does not record it, so either a sealed "
                "season was added without an attributed regeneration or the lock is "
                "stale."
            )
    return problems


def live_upstream_allowed() -> bool:
    """True when the operator has explicitly opted in to a live nflverse fetch."""
    return bool(os.environ.get(LIVE_OPT_IN_ENV, "").strip())


PIN_CLI_TEXT = ".venv/Scripts/python.exe -m scripts.pin_upstream_snapshot"
LIVE_CLI_TEXT = ".venv/Scripts/python.exe -m scripts.capture_live_season"


def sealed_rewrite_command(dataset: str, season: int, reason: str) -> str:
    """The command that re-captures an ALREADY-PINNED sealed season, with attribution.

    The plain capture command is the wrong advice for this case and has been since
    ``scripts/pin_upstream_snapshot.py::assert_write_allowed`` existed: the manifest
    entry already exists, and replacing an existing sealed entry is precisely what the
    sealed zone was built to refuse. Printing a command the tool would reject is worse
    than printing no advice at all -- an operator follows it, gets a second refusal, and
    learns that the refusals in this module are not to be taken literally.
    """
    return (
        f"{PIN_CLI_TEXT} --dataset {dataset} --seasons {season} {season} "
        f'--allow-sealed-rewrite --sealed-rewrite-reason "{reason}"'
    )


def _recovery_options(
    dataset: str, missing: list[int], manifest: dict | None = None
) -> list[str]:
    """The ordered recovery options for *missing*, one per zone actually present.

    A refusal that names the wrong tool is a refusal operators route around. The sealed
    and live zones have different capture commands, so the options are built from the
    zone of each uncovered season rather than from a single hard-coded line -- and,
    within the sealed zone, from whether the season is ALREADY PINNED, because those two
    cases now take different commands.
    """
    sealed = [season for season in missing if zone_for_season(season) == ZONE_SEALED]
    live = [season for season in missing if zone_for_season(season) == ZONE_LIVE]
    unknown = [season for season in missing if zone_for_season(season) == ZONE_UNKNOWN]

    covered = set(pinned_seasons(dataset, manifest))
    unpinned = sorted(season for season in sealed if season not in covered)
    already_pinned = sorted(season for season in sealed if season in covered)

    options: list[str] = []
    if unpinned:
        # Print the SPAN form only when every season in that span is being captured.
        # A span that stepped over an already-pinned season would be refused by the
        # write gate, which is the exact failure this reconciliation exists to remove.
        span_is_solid = max(unpinned) - min(unpinned) + 1 == len(unpinned)
        if span_is_solid:
            commands = (
                f"       {PIN_CLI_TEXT} --dataset {dataset} "
                f"--seasons {min(unpinned)} {max(unpinned)}"
            )
        else:
            commands = "\n".join(
                f"       {PIN_CLI_TEXT} --dataset {dataset} --seasons {season} {season}"
                for season in unpinned
            )
        options.append(
            f"Capture the missing SEALED season(s) into the pin (writes {MANIFEST_PATH} "
            "and timestamped snapshots under data/bronze/):\n" + commands
        )
    if already_pinned:
        commands = "\n".join(
            "       "
            + sealed_rewrite_command(
                dataset,
                season,
                "re-capturing a sealed season the pin records but cannot serve",
            )
            for season in already_pinned
        )
        options.append(
            "Re-capture the SEALED season(s) the manifest already records. This "
            "REPLACES a sealed entry, so it needs the attributed override:\n" + commands
        )
    if live:
        commands = "\n".join(
            f"       {LIVE_CLI_TEXT} --season {season} --week <W> --dataset {dataset}"
            for season in live
        )
        options.append(
            "Capture the missing LIVE-zone season(s) one week at a time (writes "
            f"{LIVE_MANIFEST_DIR_TEXT}/<season>.json and timestamped snapshots under "
            "data/bronze/). <W> is the week being PREDICTED, not the last week present "
            "in the data:\n" + commands
        )
    if unknown:
        options.append(
            "Season(s) "
            + ", ".join(str(season) for season in unknown)
            + " lie BEYOND the live zone, which ends at "
            f"{LIVE_ZONE_FIRST_SEASON}. NO tool captures them, deliberately: moving the "
            "boundary forward promotes a season from live to sealed and is one-way. It "
            "is a human edit to data.upstream_pin.SEALED_THROUGH_SEASON, made once the "
            "season has actually ended -- never a side effect of a load."
        )
    options.append(
        "Allow a live fetch for this run only, accepting that its output is NOT "
        f"reproducible:\n       {LIVE_OPT_IN_ENV}=1 <your command>\n"
        "     Every live fetch emits an UpstreamPinBypassedWarning and a logged warning."
    )
    return options


def _refusal_message(dataset: str, missing: list[int], manifest: dict | None) -> str:
    covered = pinned_seasons(dataset, manifest)
    covered_text = (
        f"{covered[0]}-{covered[-1]}" if covered else "nothing (no pin captured)"
    )
    zone_lines = "\n".join(
        f"  {season}  zone {zone_for_season(season)}" for season in missing
    )
    options = "\n".join(
        f"  {number}. {option}"
        for number, option in enumerate(
            _recovery_options(dataset, missing, manifest), start=1
        )
    )
    return (
        f"The upstream pin does not cover {dataset} season(s) "
        f"{', '.join(str(season) for season in missing)}. It covers {covered_text}.\n"
        "\n"
        "Each uncovered season, and the zone it would belong to:\n"
        f"{zone_lines}\n"
        "\n"
        "This build REFUSES to reach nflverse silently. A live fetch is what moved "
        "sixteen opponent-adjusted and QB columns from season 2020 onward between the "
        "2026-08-22 and 2026-09-04 gold builds, inside the protected 2021-2024 holdout, "
        "and it did so with nothing on record to attribute it to.\n"
        "\n"
        "Do ONE of these, deliberately:\n"
        f"{options}"
    )


def _pin_entry(dataset: str, season: int, manifest: dict) -> dict:
    return manifest["datasets"][dataset]["seasons"][str(season)]


# The two situations ``_read_pinned_frame`` refuses in, phrased as the ``--sealed-rewrite
# -reason`` an operator recovering from them would actually give. A worked example beats
# a ``<why>`` placeholder: the placeholder invites an empty-sounding reason, and a reason
# nobody can act on six months later is the thing the attribution requirement exists to
# stop.
_FRESH_CHECKOUT_REASON = (
    "fresh checkout: the committed record is present but the gitignored bytes are not"
)
_DIGEST_MOVED_REASON = (
    "the pinned bytes no longer match the digest recorded for them in the manifest"
)


def _recapture_advice(dataset: str, season: int, reason: str) -> str:
    """The literal re-capture command for *season*, and why it takes the form it does.

    Both of ``_read_pinned_frame``'s refusals fire for a season that IS in the manifest.
    For a sealed season that means the plain capture command they used to print is now
    REFUSED by ``scripts/pin_upstream_snapshot.py::assert_write_allowed`` -- the entry
    exists, and replacing an existing sealed entry is exactly what the zone prevents. So
    the advice is the override form, with a concrete reason for THIS situation rather
    than a ``<why>`` placeholder.
    """
    if zone_for_season(season) != ZONE_SEALED:
        return f"       {PIN_CLI_TEXT} --dataset {dataset} --seasons {season} {season}"
    return (
        f"       {sealed_rewrite_command(dataset, season, reason)}\n"
        "The override flags are required because the manifest entry ALREADY EXISTS, and "
        "rewriting an existing sealed entry is the thing the sealed zone exists to "
        "prevent; without them this command is refused."
    )


def _read_pinned_frame(
    dataset: str,
    season: int,
    manifest: dict,
    data_root: Path,
) -> pd.DataFrame:
    """Read one pinned season, verifying its bytes against the recorded digest first.

    The digest check runs BEFORE the parquet is parsed, for the reason
    ``utils.paths.reject_data_path`` states about its own ordering: a check that arrives
    after the work is a check nobody can afford to trust. A pin whose bytes have moved is
    not a pin, and reading it anyway would produce a number no manifest can account for.
    """
    entry = _pin_entry(dataset, season, manifest)
    path = data_root / entry["path"]
    if not path.is_file():
        msg = (
            f"The {dataset} pin for season {season} names '{path}', which does not "
            f"exist. {MANIFEST_PATH} is committed but data/ is gitignored, so a fresh "
            "checkout has the record without the bytes. Re-capture with:\n"
            + _recapture_advice(dataset, season, _FRESH_CHECKOUT_REASON)
        )
        raise UpstreamPinCorrupt(msg)

    actual = digest_file(path)
    if actual != entry["sha256"]:
        msg = (
            f"The {dataset} pin for season {season} at '{path}' does NOT match the "
            f"digest recorded in {MANIFEST_PATH}.\n"
            f"  recorded sha256 {entry['sha256']}\n"
            f"  actual   sha256 {actual}\n"
            "A pinned snapshot is immutable by definition. Either the file was "
            "rewritten, or the manifest belongs to a different capture. Re-capture the "
            "pin rather than trusting bytes no record accounts for:\n"
            + _recapture_advice(dataset, season, _DIGEST_MOVED_REASON)
        )
        raise UpstreamPinCorrupt(msg)

    return pd.read_parquet(path)


def _partition_seasons(
    dataset: str,
    seasons: list[int],
    manifest: dict | None,
    *,
    manifest_dir: Path | str | None = None,
) -> dict[str, list[int]]:
    """Split *seasons* into the zone each one can actually be SERVED from.

    Zone membership alone is not coverage. A sealed-zone season is ``sealed`` only when
    the sealed manifest pins it; a live-zone season is ``live`` only when its manifest
    exists AND a capture resolves for this dataset. Everything else -- including a
    live-zone season whose manifest records other datasets but not this one -- is
    ``unknown``, which is the set the all-or-nothing refusal and the live fetch both work
    from.
    """
    partition: dict[str, list[int]] = {"sealed": [], "live": [], "unknown": []}
    covered = set(pinned_seasons(dataset, manifest))

    # Only touch the live manifest directory when the request actually reaches into the
    # live zone: a pure sealed load must cost exactly what it cost before.
    live_covered: set[int] = set()
    if any(zone_for_season(season) == ZONE_LIVE for season in seasons):
        from data import upstream_live

        live_covered = upstream_live.live_covered_seasons(manifest_dir=manifest_dir)

    for season in seasons:
        zone = zone_for_season(season)
        if zone == ZONE_SEALED and manifest is not None and season in covered:
            partition["sealed"].append(season)
            continue
        if zone == ZONE_LIVE and season in live_covered:
            from data import upstream_live

            live_manifest = upstream_live.load_live_manifest(
                season, manifest_dir=manifest_dir
            )
            try:
                upstream_live.resolve_capture(live_manifest, dataset)
            except UpstreamLiveCaptureMissing:
                partition["unknown"].append(season)
                continue
            partition["live"].append(season)
            continue
        partition["unknown"].append(season)
    return partition


def _load(
    dataset: str,
    seasons: list[int],
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
    live_manifest_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return *dataset* for *seasons*, from whichever zone covers each one."""
    manifest = load_manifest(manifest_path)
    partition = _partition_seasons(
        dataset, seasons, manifest, manifest_dir=live_manifest_dir
    )
    missing = partition["unknown"]
    sealed = set(partition["sealed"])

    if seasons and not missing:
        from data import upstream_live

        root = Path(data_root) if data_root is not None else default_data_root()
        frames = []
        for season in seasons:
            if season in sealed:
                frames.append(_read_pinned_frame(dataset, season, manifest, root))
                continue
            live_manifest = upstream_live.load_live_manifest(
                season, manifest_dir=live_manifest_dir
            )
            entry = upstream_live.resolve_capture(live_manifest, dataset)
            frames.append(upstream_live.read_live_frame(entry, root))
        combined = (
            pd.concat(frames, ignore_index=True)
            if len(frames) > 1
            else frames[0].copy()
        )
        logger.info(
            "Loaded pinned upstream snapshot",
            dataset=dataset,
            seasons=seasons,
            sealed_seasons=partition["sealed"],
            live_seasons=partition["live"],
            rows=len(combined),
            captured_at_utc=(manifest or {}).get("captured_at_utc"),
        )
        return combined

    if not live_upstream_allowed():
        raise UpstreamPinMissing(_refusal_message(dataset, missing, manifest))

    message = (
        f"LIVE nflverse fetch for {dataset} season(s) "
        f"{', '.join(str(season) for season in seasons)}: the pin does not cover "
        f"{', '.join(str(season) for season in missing)} and {LIVE_OPT_IN_ENV} is set. "
        "The output of this run is NOT reproducible from a recorded snapshot."
    )
    warnings.warn(message, UpstreamPinBypassedWarning, stacklevel=3)
    logger.warning(
        "Upstream pin bypassed -- fetching live",
        dataset=dataset,
        seasons=seasons,
        missing_seasons=missing,
        sealed_seasons=partition["sealed"],
        live_seasons=partition["live"],
        opt_in_env=LIVE_OPT_IN_ENV,
    )

    # PIN-02. This line passed ``seasons`` -- the WHOLE request -- so one uncovered
    # season sent every covered one back to nflverse too, silently replacing pinned
    # bytes with live ones in the same frame. It fetches the GENUINELY UNCOVERED half
    # only. Nothing else covered stops coming from its zone.
    if not sealed and not partition["live"]:
        return _fetch_live(dataset, missing)

    # A MIXED request. Each uncovered season is fetched on its own so the frames can be
    # concatenated in the ORIGINALLY REQUESTED season order beside the zone-served ones,
    # with no need to guess how to split one multi-season live frame back apart.
    from data import upstream_live

    root = Path(data_root) if data_root is not None else default_data_root()
    frames = []
    for season in seasons:
        if season in sealed:
            frames.append(_read_pinned_frame(dataset, season, manifest, root))
        elif season in set(partition["live"]):
            live_manifest = upstream_live.load_live_manifest(
                season, manifest_dir=live_manifest_dir
            )
            entry = upstream_live.resolve_capture(live_manifest, dataset)
            frames.append(upstream_live.read_live_frame(entry, root))
        else:
            frames.append(_fetch_live(dataset, [season]))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()


def _fetch_live(dataset: str, seasons: list[int]) -> pd.DataFrame:
    """Fetch *dataset* from nflverse. Reached ONLY through the explicit opt-in above."""
    import nflreadpy as nfl

    if dataset == "pbp":
        return nfl.load_pbp(seasons).to_pandas()
    if dataset == "schedules":
        return nfl.load_schedules(seasons).to_pandas()
    if dataset == "depth_charts":
        frames = [nfl.load_depth_charts(season).to_pandas() for season in seasons]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    msg = f"Unknown upstream dataset {dataset!r}. Known: {sorted(DATASET_COLUMNS)}."
    raise UpstreamPinError(msg)


def load_pbp(
    seasons: list[int] | tuple[int, ...],
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
    live_manifest_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_pbp(seasons).to_pandas()``.

    Returns a pandas frame directly -- callers must NOT call ``.to_pandas()`` on it.
    Carries :data:`PBP_PINNED_COLUMNS` and every row of every requested season.

    ``live_manifest_dir`` redirects the LIVE zone's manifest lookup the same way
    ``manifest_path`` redirects the sealed one, so a test can exercise both zones inside
    ``tmp_path`` without touching the committed records.
    """
    return _load(
        "pbp",
        list(seasons),
        manifest_path=manifest_path,
        data_root=data_root,
        live_manifest_dir=live_manifest_dir,
    )


def load_schedules(
    seasons: list[int] | tuple[int, ...],
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
    live_manifest_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_schedules(seasons).to_pandas()``."""
    return _load(
        "schedules",
        list(seasons),
        manifest_path=manifest_path,
        data_root=data_root,
        live_manifest_dir=live_manifest_dir,
    )


def load_depth_charts(
    season: int,
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
    live_manifest_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_depth_charts(season).to_pandas()``."""
    return _load(
        "depth_charts",
        [season],
        manifest_path=manifest_path,
        data_root=data_root,
        live_manifest_dir=live_manifest_dir,
    )
