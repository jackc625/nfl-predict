"""The content-addressed decision-input snapshot: what a decision SAW, stored when it decided (LDGR-09).

WHY THE INPUTS ARE STORED, NOT REBUILT
--------------------------------------
Gold is rebuilt nightly, and a gold row rebuilt after results land can differ from the row the
decision scored (NF-08). A replay that rebuilt its inputs from the lake would therefore be a replay
of a different decision. So each decision run stores ONE snapshot of exactly what it consumed --
taken from ``backtest.weekly_bet_list.DecisionBundle``, the one decision seam, never re-read
afterwards -- and every row of that run (all games, all targets) references it by digest:

  * ``candidates``  -- the candidates frame as handed to the selector;
  * ``schedule``    -- the week's schedule spine;
  * ``gold_wp`` / ``gold_ats`` / ``gold_ou`` -- each target's gold rows exactly as scored;
  * ``meta.json``   -- the resolved artifact ids, the chain-fit record path and sha256, season,
    week, the run instant, the library versions and the ``uv.lock`` sha256 (a dependency upgrade
    is the realistic way replay drifts, 34-RESEARCH H).

CONTENT-ADDRESSED, AND BY CANONICAL CONTENT
-------------------------------------------
Each part is digested with ``forward_ledger.canonical.frame_digest`` -- the LDGR-05 discipline
applied to a frame -- sorted by that part's stated UNIQUE key (:data:`SNAPSHOT_PART_KEYS`); a
duplicated key is refused by name, because a tie would let row order move the digest. The snapshot
digest is sha256 over the sorted ``[part, part digest]`` pairs plus the digest of the meta record,
and it names the directory: ``ledger/snapshots/<digest>/``. :func:`compute_snapshot_digest` produces
the same digest in memory, writing nothing, so the runner can stamp a row and apply first pick stands
before deciding whether anything needs writing.

WRITTEN BEFORE THE ROW, PROVEN, AND NEVER HALF-WRITTEN
-------------------------------------------------------
The writer stages every part in a private directory, re-reads each parquet and proves its digest
equals the in-memory frame's, then renames the staging directory to ``<digest>`` -- so a crash
leaves either no snapshot or a whole one, and a whole one that no row references is harmless (it is
re-verified and reused if the same inputs recur; 34-RESEARCH F1). An existing ``<digest>`` is
re-verified, never overwritten. The loader recomputes everything and names the part that differs:
each part's file bytes are checked against the sha256 recorded at write time (so a byte that
changes no value is still caught), then its canonical digest against the meta record, then the
snapshot digest against the directory name.

Every write is asked of ``data.write_sink.current_sink()`` first, so a dry run writes nothing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from data.write_sink import current_sink
from forward_ledger.canonical import CanonicalValueError, frame_digest

if TYPE_CHECKING:
    from backtest.weekly_bet_list import DecisionBundle

__all__ = [
    "META_FILENAME",
    "SNAPSHOTS_DIRNAME",
    "SNAPSHOT_PARTS",
    "SNAPSHOT_PART_KEYS",
    "DecisionSnapshot",
    "SnapshotDigestMismatchError",
    "SnapshotInputError",
    "SnapshotMissingError",
    "compute_snapshot_digest",
    "load_decision_snapshot",
    "write_decision_snapshot",
]

SNAPSHOTS_DIRNAME: str = "snapshots"
META_FILENAME: str = "meta.json"

# The model targets whose gold rows a snapshot stores, one part each.
_GOLD_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

SNAPSHOT_PARTS: tuple[str, ...] = (
    "candidates",
    "schedule",
    *(f"gold_{target}" for target in _GOLD_TARGETS),
)

# The literal sort key of each part -- the one ``frame_digest`` is ALWAYS called with. Each must
# identify a row uniquely: one candidate per game per target (the candidates builder emits one row
# per (game, target) with a model output), one schedule row per game, one gold row per game in the
# decided week.
SNAPSHOT_PART_KEYS: dict[str, tuple[str, ...]] = {
    "candidates": ("game_id", "target"),
    "schedule": ("game_id",),
    "gold_wp": ("game_id",),
    "gold_ats": ("game_id",),
    "gold_ou": ("game_id",),
}

# Domain separation for the snapshot digest, and the version of the meta record's shape.
SNAPSHOT_DOMAIN = b"nfl-ledger-snapshot-v1\n"
SNAPSHOT_FORMAT: int = 1

# The lock file whose sha256 pins every library version replay depends on (repo-relative; the
# scheduled task and the CLIs run from the repository root).
UV_LOCK_PATH: Path = Path("uv.lock")

# The distributions whose versions the meta record names beside the uv.lock digest.
_LIBRARY_DISTRIBUTIONS: tuple[str, ...] = (
    "pandas",
    "pyarrow",
    "numpy",
    "scikit-learn",
    "xgboost",
)

# The meta key recording each part file's sha256. It is written into meta.json but NOT digested:
# parquet bytes are not canonical content, and the in-memory digest must not depend on them.
_PART_FILES_KEY = "part_file_sha256"

if set(SNAPSHOT_PART_KEYS) != set(SNAPSHOT_PARTS):  # pragma: no cover
    msg = "SNAPSHOT_PART_KEYS must name exactly the parts of SNAPSHOT_PARTS"
    raise RuntimeError(msg)


class SnapshotDigestMismatchError(Exception):
    """A stored snapshot does not verify; ``part`` names what differs (a part, or ``meta``).

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a snapshot that fails to verify
    must never be degraded into "no inputs stored".
    """

    def __init__(self, part: str, detail: str) -> None:
        self.part = part
        super().__init__(f"decision snapshot part {part!r} does not verify: {detail}")


class SnapshotMissingError(Exception):
    """No snapshot directory exists for the requested digest."""


class SnapshotInputError(Exception):
    """A decision bundle cannot be snapshotted as given (named in the message)."""


@dataclass(frozen=True)
class DecisionSnapshot:
    """One verified stored snapshot.

    Attributes:
        digest: The snapshot digest (its directory name).
        frames: Each part's frame as read back, keyed by part name.
        meta: The digested meta record.
    """

    digest: str
    frames: dict[str, pd.DataFrame]
    meta: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_parts(bundle: DecisionBundle) -> dict[str, pd.DataFrame]:
    missing = [target for target in _GOLD_TARGETS if target not in bundle.gold_inputs]
    if missing:
        msg = (
            f"the decision bundle carries no gold rows for {missing}; a snapshot without the rows "
            "a target was scored from could never replay it"
        )
        raise SnapshotInputError(msg)
    return {
        "candidates": bundle.candidates,
        "schedule": bundle.schedule,
        **{f"gold_{target}": bundle.gold_inputs[target] for target in _GOLD_TARGETS},
    }


def _part_digest(part: str, frame: pd.DataFrame) -> str:
    try:
        return frame_digest(frame, SNAPSHOT_PART_KEYS[part])
    except CanonicalValueError as error:
        msg = f"decision snapshot part {part!r}: {error}"
        raise CanonicalValueError(msg) from error


def _decided_season_week(schedule: pd.DataFrame) -> tuple[int, int]:
    seasons = sorted({int(value) for value in schedule["season"].dropna()})
    weeks = sorted({int(value) for value in schedule["week"].dropna()})
    if len(seasons) != 1 or len(weeks) != 1:
        msg = (
            f"the bundle's schedule spans seasons {seasons} and weeks {weeks}; one snapshot "
            "records one decided (season, week)"
        )
        raise SnapshotInputError(msg)
    return seasons[0], weeks[0]


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: importlib_metadata.version(name) for name in _LIBRARY_DISTRIBUTIONS},
    }


def _digested_meta(
    bundle: DecisionBundle,
    part_digests: Mapping[str, str],
    meta_extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if bundle.run_instant.tzinfo is None:
        msg = f"the bundle's run instant {bundle.run_instant!r} is naive"
        raise SnapshotInputError(msg)
    season, week = _decided_season_week(bundle.schedule)
    chain_fit = Path(bundle.chain_fit_path)
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "season": season,
        "week": week,
        "run_instant": bundle.run_instant.isoformat(),
        "resolved": None if bundle.resolved is None else asdict(bundle.resolved),
        "chain_fit_path": chain_fit.as_posix(),
        "chain_fit_sha256": _file_sha256(chain_fit),
        "library_versions": _library_versions(),
        "uv_lock_sha256": _file_sha256(UV_LOCK_PATH),
        "part_keys": {part: list(SNAPSHOT_PART_KEYS[part]) for part in SNAPSHOT_PARTS},
        "part_digests": dict(part_digests),
        "extra": dict(meta_extra or {}),
    }


def _snapshot_digest(meta: Mapping[str, Any]) -> str:
    """sha256 over the sorted ``[part, part digest]`` pairs and the meta record's digest."""
    pairs = sorted([part, digest] for part, digest in meta["part_digests"].items())
    meta_digest = hashlib.sha256(_canonical_json(meta)).hexdigest()
    payload = [["parts", pairs], ["meta", meta_digest]]
    return hashlib.sha256(SNAPSHOT_DOMAIN + _canonical_json(payload)).hexdigest()


def _prepare(
    bundle: DecisionBundle, meta_extra: Mapping[str, Any] | None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], str]:
    parts = _bundle_parts(bundle)
    part_digests = {part: _part_digest(part, frame) for part, frame in parts.items()}
    meta = _digested_meta(bundle, part_digests, meta_extra)
    return parts, meta, _snapshot_digest(meta)


def compute_snapshot_digest(
    bundle: DecisionBundle, *, meta_extra: Mapping[str, Any] | None = None
) -> str:
    """The digest :func:`write_decision_snapshot` would produce for *bundle*. Writes nothing.

    Raises:
        CanonicalValueError: a part cannot be canonicalized (the part is named), including a
            duplicated sort key.
        SnapshotInputError: the bundle lacks a gold part, spans more than one (season, week), or
            carries a naive run instant.
    """
    return _prepare(bundle, meta_extra)[2]


def write_decision_snapshot(
    ledger_dir: Path | str,
    bundle: DecisionBundle,
    *,
    meta_extra: Mapping[str, Any] | None = None,
) -> str:
    """Store *bundle*'s decision inputs under ``<ledger_dir>/snapshots/<digest>/``; return the digest.

    Idempotent: an existing ``<digest>`` directory is re-verified and reused, never rewritten. A
    new one is staged privately, each part proven to read back to the frame the decision saw, and
    only then renamed into place; any failure removes the staging directory.

    Raises:
        Everything :func:`compute_snapshot_digest` raises, plus
        SnapshotDigestMismatchError: a written part does not read back to its frame, or an
            existing snapshot under this digest does not verify.
    """
    parts, meta, digest = _prepare(bundle, meta_extra)
    root = Path(ledger_dir) / SNAPSHOTS_DIRNAME
    target = root / digest
    if target.exists():
        load_decision_snapshot(ledger_dir, digest)
        return digest

    rows = sum(len(frame) for frame in parts.values())
    if not current_sink().authorize(str(target), "ledger_snapshot", rows):
        return digest

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{digest}-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        part_files: dict[str, str] = {}
        for part, frame in parts.items():
            path = staging / f"{part}.parquet"
            frame.to_parquet(path, index=False)
            if _part_digest(part, pd.read_parquet(path)) != meta["part_digests"][part]:
                raise SnapshotDigestMismatchError(
                    part,
                    "the written parquet does not read back to the frame the decision saw",
                )
            part_files[part] = _file_sha256(path)
        (staging / META_FILENAME).write_bytes(
            _canonical_json({**meta, _PART_FILES_KEY: part_files})
        )
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            # Another writer stored the same inputs first: keep theirs, after verifying it.
            shutil.rmtree(staging)
            load_decision_snapshot(ledger_dir, digest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return digest


def load_decision_snapshot(ledger_dir: Path | str, digest: str) -> DecisionSnapshot:
    """Read and VERIFY the snapshot *digest*: every part's bytes and content, and the digest itself.

    Raises:
        SnapshotMissingError: no directory exists for *digest*.
        SnapshotDigestMismatchError: naming the part (or ``meta``) that does not verify.
    """
    directory = Path(ledger_dir) / SNAPSHOTS_DIRNAME / digest
    if not directory.is_dir():
        msg = f"no decision snapshot {digest!r} under {directory.parent.as_posix()}"
        raise SnapshotMissingError(msg)

    try:
        stored = json.loads((directory / META_FILENAME).read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotDigestMismatchError("meta", f"unreadable: {error}") from error
    if not isinstance(stored, dict) or not isinstance(
        stored.get(_PART_FILES_KEY), dict
    ):
        raise SnapshotDigestMismatchError("meta", "not a snapshot meta record")
    part_files: dict[str, Any] = stored.pop(_PART_FILES_KEY)
    meta: dict[str, Any] = stored
    try:
        recomputed = _snapshot_digest(meta)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise SnapshotDigestMismatchError("meta", f"malformed: {error}") from error
    if recomputed != digest:
        raise SnapshotDigestMismatchError(
            "meta", f"the record digests to {recomputed}, not to its directory name"
        )

    frames: dict[str, pd.DataFrame] = {}
    for part in SNAPSHOT_PARTS:
        path = directory / f"{part}.parquet"
        if not path.is_file():
            raise SnapshotDigestMismatchError(part, "the part file is missing")
        if _file_sha256(path) != part_files.get(part):
            raise SnapshotDigestMismatchError(
                part, "its bytes differ from the bytes written with the snapshot"
            )
        frame = pd.read_parquet(path)
        if _part_digest(part, frame) != meta["part_digests"].get(part):
            raise SnapshotDigestMismatchError(
                part, "its canonical content differs from the recorded digest"
            )
        frames[part] = frame
    return DecisionSnapshot(digest=digest, frames=frames, meta=meta)
