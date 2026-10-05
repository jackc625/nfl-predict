"""The reproduction key: which upstream capture, which gold and which odds a decision read (LDGR-09).

WHAT IT IS FOR
--------------
A ledger row is re-derived from its stored decision-input snapshot (``forward_ledger.snapshots``),
never from a rebuild of the lake. The reproduction key says WHERE those inputs came from, so a
replay can tell "the inputs are the ones the decision read" from "the lake has moved since":

  * ``upstream_capture_key`` -- the Phase-32 live capture identity ``(season, week, sequence,
    sha256)`` of every dataset in the season's live manifest, resolved exactly the way the build
    reads it (the active as-of address when one is set, else the newest capture);
  * ``gold_generation_key`` -- the identity of the gold build, taken over the BYTES of the three
    gold matrices, so it does not depend on ``feature_timestamp`` and survives backlog 999.4's
    planned removal of that column;
  * ``odds_snapshot_digest`` -- the canonical digest of the odds rows of the decided games that
    were ADMISSIBLE at the decision instant (recorded capture instant at or before it).

ONE FORMULA, TWO TIERS
----------------------
``tests/gold_generation.py`` already computes the gold generation key for the test tier. Production
code never imports from ``tests/``, so the formula is re-implemented here and a test proves the two
agree byte for byte (34-RESEARCH A10). Change one and that test fails.

Reads only: this module writes nothing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from backtest.ou_divergence import odds_information_time
from data.schemas import LIVE_ODDS_CAPTURE_KEY
from data.upstream_live import AsOfCapture, current_as_of, resolve_capture
from forward_ledger.canonical import frame_digest

__all__ = [
    "GOLD_MATRIX_LABELS",
    "ReproKey",
    "ReproKeyUnavailableError",
    "build_repro_key",
    "gold_generation_key",
    "odds_snapshot_digest",
    "upstream_capture_key",
]

# The three gold matrices, as the repo-relative labels the test-tier formula hashes. The LABEL is
# fixed whatever directory the bytes are read from, so a key computed over a copy of gold equals
# the key of the gold it copies.
GOLD_MATRIX_LABELS: dict[str, str] = {
    "wp": "data/gold/features_wp.parquet",
    "ats": "data/gold/features_ats.parquet",
    "ou": "data/gold/features_ou.parquet",
}

_READ_CHUNK_BYTES = 1 << 20


class ReproKeyUnavailableError(Exception):
    """A reproduction-key component cannot be established, so no key is returned.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a key that silently described
    fewer inputs than the decision read would be a key that lies about its row.
    """


@dataclass(frozen=True)
class ReproKey:
    """The three input identities stamped on a ledger row (all non-null strings)."""

    upstream_capture_key: str
    gold_generation_key: str
    odds_snapshot_digest: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gold_generation_key(gold_dir: Path | str) -> str:
    """The content identity of the three gold matrices under *gold_dir*.

    The ``tests/gold_generation.py`` formula: one ``<label>=<sha256 of the file's bytes>`` line per
    matrix, the lines sorted, sha256 over their newline-joined ASCII.

    Raises:
        ReproKeyUnavailableError: a matrix is absent. A key over the remaining two would claim to
            describe gold it never read.
    """
    directory = Path(gold_dir)
    lines: list[str] = []
    for target, label in GOLD_MATRIX_LABELS.items():
        path = directory / f"features_{target}.parquet"
        if not path.is_file():
            msg = (
                f"gold matrix {path.as_posix()} is absent; a gold generation key over the "
                "remaining matrices would describe gold it never read"
            )
            raise ReproKeyUnavailableError(msg)
        lines.append(f"{label}={_file_sha256(path)}")
    joined = "\n".join(sorted(lines))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def odds_snapshot_digest(
    odds: pd.DataFrame, game_ids: Iterable[str], decided_at: datetime
) -> str:
    """The canonical digest of the odds rows of *game_ids* admissible at *decided_at*.

    Admissible means a recorded capture instant (``created_at``, read through the decision path's
    own ``odds_information_time``) at or before *decided_at*. A row captured later, or with no
    recorded instant, was not information the decision had, and does not move the digest. Rows are
    ordered by the live store's row identity, ``LIVE_ODDS_CAPTURE_KEY``.

    Raises:
        ReproKeyUnavailableError: *decided_at* is naive.
    """
    if decided_at.tzinfo is None:
        msg = f"decided_at {decided_at!r} is naive; admissibility is judged between instants"
        raise ReproKeyUnavailableError(msg)
    wanted = {str(game_id) for game_id in game_ids}
    of_games = cast(
        "pd.DataFrame", odds[odds["game_id"].astype(str).isin(sorted(wanted))]
    )
    known_at = odds_information_time(of_games)
    admissible = cast("pd.DataFrame", of_games[known_at <= pd.Timestamp(decided_at)])
    return frame_digest(admissible, LIVE_ODDS_CAPTURE_KEY)


def upstream_capture_key(
    manifest: dict[str, Any] | None, *, as_of: AsOfCapture | None = None
) -> str:
    """Canonical JSON naming ``[season, week, sequence, sha256]`` for every live dataset.

    Each dataset's capture is resolved through ``data.upstream_live.resolve_capture`` at the as-of
    in force (``current_as_of``: *as_of*, then an ``as_of_capture`` block, then the environment),
    else the newest capture -- exactly the address the build read.

    Raises:
        ReproKeyUnavailableError: the manifest records no dataset at all.
        data.upstream_pin.UpstreamLiveCaptureMissing: a dataset has no capture at the address.
    """
    record = manifest or {}
    datasets = sorted(record.get("datasets", {}))
    if not datasets:
        msg = (
            "the live manifest records no dataset, so no upstream capture can be named"
        )
        raise ReproKeyUnavailableError(msg)
    active = current_as_of(explicit=as_of)
    season = record.get("season")
    identities: dict[str, list[Any]] = {}
    for dataset in datasets:
        entry = resolve_capture(
            record,
            dataset,
            week=None if active is None else active.week,
            sequence=None if active is None else active.sequence,
        )
        identities[dataset] = [
            season,
            entry["week"],
            entry["sequence"],
            entry["sha256"],
        ]
    return json.dumps(
        identities, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


def build_repro_key(
    *,
    manifest: dict[str, Any] | None,
    gold_dir: Path | str,
    odds: pd.DataFrame,
    game_ids: Iterable[str],
    decided_at: datetime,
    as_of: AsOfCapture | None = None,
) -> ReproKey:
    """The three identities of one decision's inputs, assembled into its :class:`ReproKey`."""
    return ReproKey(
        upstream_capture_key=upstream_capture_key(manifest, as_of=as_of),
        gold_generation_key=gold_generation_key(gold_dir),
        odds_snapshot_digest=odds_snapshot_digest(odds, game_ids, decided_at),
    )
