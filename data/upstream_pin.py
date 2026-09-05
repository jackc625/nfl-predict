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

# Set this to any non-empty value to permit a live nflverse fetch. Nothing sets it
# implicitly; the pipeline, the tests and the rebuild all run with it unset.
LIVE_OPT_IN_ENV = "NFL_PREDICT_ALLOW_LIVE_UPSTREAM"

MANIFEST_SCHEMA_VERSION = 1

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


class UpstreamPinBypassedWarning(UserWarning):
    """A live nflverse fetch happened because the operator explicitly allowed it."""


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


def live_upstream_allowed() -> bool:
    """True when the operator has explicitly opted in to a live nflverse fetch."""
    return bool(os.environ.get(LIVE_OPT_IN_ENV, "").strip())


def _refusal_message(dataset: str, missing: list[int], manifest: dict | None) -> str:
    covered = pinned_seasons(dataset, manifest)
    covered_text = (
        f"{covered[0]}-{covered[-1]}" if covered else "nothing (no pin captured)"
    )
    return (
        f"The upstream pin does not cover {dataset} season(s) "
        f"{', '.join(str(season) for season in missing)}. It covers {covered_text}.\n"
        "\n"
        "This build REFUSES to reach nflverse silently. A live fetch is what moved "
        "sixteen opponent-adjusted and QB columns from season 2020 onward between the "
        "2026-08-22 and 2026-09-04 gold builds, inside the protected 2021-2024 holdout, "
        "and it did so with nothing on record to attribute it to.\n"
        "\n"
        "Do ONE of these, deliberately:\n"
        f"  1. Capture the missing seasons into the pin (writes {MANIFEST_PATH} and "
        "timestamped snapshots under data/bronze/):\n"
        "       .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot "
        f"--dataset {dataset} --seasons {min(missing)} {max(missing)}\n"
        f"  2. Allow a live fetch for this run only, accepting that its output is NOT "
        "reproducible:\n"
        f"       {LIVE_OPT_IN_ENV}=1 <your command>\n"
        "     Every live fetch emits an UpstreamPinBypassedWarning and a logged warning."
    )


def _pin_entry(dataset: str, season: int, manifest: dict) -> dict:
    return manifest["datasets"][dataset]["seasons"][str(season)]


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
            "checkout has the record without the bytes. Re-capture with "
            f"`.venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --dataset "
            f"{dataset} --seasons {season} {season}`."
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
            "pin rather than trusting bytes no record accounts for."
        )
        raise UpstreamPinCorrupt(msg)

    return pd.read_parquet(path)


def _load(
    dataset: str,
    seasons: list[int],
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
) -> pd.DataFrame:
    """Return *dataset* for *seasons*, from the pin when it covers them."""
    manifest = load_manifest(manifest_path)
    covered = set(pinned_seasons(dataset, manifest))
    missing = sorted(season for season in seasons if season not in covered)

    if not missing and manifest is not None:
        root = Path(data_root) if data_root is not None else default_data_root()
        frames = [
            _read_pinned_frame(dataset, season, manifest, root) for season in seasons
        ]
        combined = (
            pd.concat(frames, ignore_index=True)
            if len(frames) > 1
            else frames[0].copy()
        )
        logger.info(
            "Loaded pinned upstream snapshot",
            dataset=dataset,
            seasons=seasons,
            rows=len(combined),
            captured_at_utc=manifest.get("captured_at_utc"),
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
        opt_in_env=LIVE_OPT_IN_ENV,
    )
    return _fetch_live(dataset, seasons)


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
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_pbp(seasons).to_pandas()``.

    Returns a pandas frame directly -- callers must NOT call ``.to_pandas()`` on it.
    Carries :data:`PBP_PINNED_COLUMNS` and every row of every requested season.
    """
    return _load(
        "pbp",
        list(seasons),
        manifest_path=manifest_path,
        data_root=data_root,
    )


def load_schedules(
    seasons: list[int] | tuple[int, ...],
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_schedules(seasons).to_pandas()``."""
    return _load(
        "schedules",
        list(seasons),
        manifest_path=manifest_path,
        data_root=data_root,
    )


def load_depth_charts(
    season: int,
    *,
    manifest_path: Path | str | None = None,
    data_root: Path | str | None = None,
) -> pd.DataFrame:
    """Pinned replacement for ``nflreadpy.load_depth_charts(season).to_pandas()``."""
    return _load(
        "depth_charts",
        [season],
        manifest_path=manifest_path,
        data_root=data_root,
    )
