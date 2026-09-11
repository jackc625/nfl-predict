"""The LIVE half of the two-zone upstream pin: an append-only, per-week capture record.

WHY THIS EXISTS
---------------
``data/upstream_pin.py`` froze the seasons a gold rebuild reads and made reproducibility
a property the build actually has. It froze them as a SEALED set: capture once, never
move. That contract is exactly right for a finished season and exactly wrong for the one
currently being played. A live season's upstream frame grows every week, and nflverse
re-releases corrections into weeks already recorded. A single immutable snapshot of 2026
would be stale within seven days; re-capturing it in place would silently overwrite the
bytes a published prediction was made from, which is the loss the pin exists to prevent.

So the live season gets its own record, with the opposite mutability contract: a list of
captures per ``(season, dataset, week)``, APPENDED to and never rewritten. Reading week 6
back means asking for week 6's capture, not for "the current state of 2026".

TWO FILES WHOSE DIFFS MEAN OPPOSITE THINGS
------------------------------------------
``config/`` is committed and ``data/`` is gitignored, for both zones: the record of WHICH
upstream revision a verdict was measured against survives a fresh checkout even when the
parquet bytes do not.

A diff on ``config/upstream_pin.json`` is ALWAYS a red flag -- the sealed zone moved, and
something has to account for it. A diff on ``config/upstream_live/<season>.json`` is
ALWAYS expected -- it is what a week passing looks like. Those two signals cannot share a
file without destroying both: a reviewer who has learned to expect the second stops
reading the first. They are separate files on purpose and must stay separate.

WHAT A CAPTURE IS LABELLED WITH
-------------------------------
The ``week`` on a capture labels the week being PREDICTED, not the last week present in
the data. RATIFIED by the owner on 2026-09-11 (D32-13, ruling "Select: predicted"), and
one-way: from Phase 34 onward the ledger's reproduction key is ``(season, week,
sequence)``, so changing the meaning after the first row exists silently reinterprets
every prior capture and every recorded replay address, with no migration able to tell
which convention a row was written under. Replay of a Week-6 row is therefore literally
"give me week 6's capture", with no arithmetic in the address to get wrong.

What the capture CONTAINS is a separate, recorded fact -- never inferred from the label.
:data:`WEEK_LABEL_MEANS` states the convention in the manifest itself, and
``content_through_week`` records the maximum week value OBSERVED in the frame. The two
are not related by arithmetic: each NFL week opens with a Thursday night game played
before the project's Friday 6 PM ET freeze, so a week-6 capture normally already contains
week-6 rows. Measured 2026-09-10: live ``load_pbp([2026])`` held 166 week-1 rows on the
Thursday of week 1.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from data.upstream_pin import (
    DATASET_LOADERS,
    UpstreamLiveCaptureMissing,
    UpstreamLiveCorrupt,
    digest_file,
)
from utils import get_logger

logger = get_logger(__name__)

LIVE_MANIFEST_SCHEMA_VERSION: int = 1

# The committed live-zone record, one file per live season. Separate from
# ``config/upstream_pin.json`` because their diffs mean opposite things -- see the module
# docstring. ``data.upstream_pin.LIVE_MANIFEST_DIR_TEXT`` is the same path as text, used
# there only to keep refusal messages free of an import back into this module.
LIVE_MANIFEST_DIR: Path = Path("config/upstream_live")

# The frame column ``content_through_week`` is OBSERVED from.
WEEK_COLUMN = "week"

# The bucket a frame with no ``week`` column falls into, for the per-week digests Plan
# 32-04 adds. Named here so the sentinel is one constant rather than a repeated literal.
NO_WEEK_COLUMN_BUCKET = "__no_week_column__"

# Recorded verbatim into every capture entry, so the convention travels with the bytes
# rather than living only in this docstring. D32-13, ratified 2026-09-11.
WEEK_LABEL_MEANS = (
    "the week being PREDICTED, not the last week present in the data (D32-13)"
)

LIVE_SOURCE = "nflverse (github.com/nflverse) via nflreadpy"


def live_manifest_path(season: int, *, manifest_dir: Path | str | None = None) -> Path:
    """Return the committed manifest path for *season*."""
    directory = Path(manifest_dir) if manifest_dir is not None else LIVE_MANIFEST_DIR
    return directory / f"{season}.json"


def empty_live_manifest(season: int) -> dict:
    """Return a fresh, capture-less live manifest for *season*."""
    return {
        "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
        "season": season,
        "zone": "live",
        "source": LIVE_SOURCE,
        "datasets": {},
    }


def load_live_manifest(
    season: int,
    *,
    manifest_dir: Path | str | None = None,
) -> dict | None:
    """Return *season*'s live manifest, or ``None`` when nothing has been captured yet.

    An absent file is not an error here, for the same reason
    ``data.upstream_pin.load_manifest`` says: it is the state of a fresh checkout, and
    the refusal that matters belongs at the point of a load where the message can name
    the week that was actually asked for.

    A manifest whose ``schema_version`` this build does not understand is a HARD refusal
    and never a migration -- reading it anyway would assign a meaning to fields that
    nobody wrote under that meaning.
    """
    path = live_manifest_path(season, manifest_dir=manifest_dir)
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != LIVE_MANIFEST_SCHEMA_VERSION:
        msg = (
            f"Live-zone manifest at '{path}' declares schema_version {version!r}, but "
            f"this build understands version {LIVE_MANIFEST_SCHEMA_VERSION}. Re-capture "
            "the season with `python -m scripts.capture_live_season` rather than "
            "reading a manifest whose meaning is not the meaning this code assigns it."
        )
        raise UpstreamLiveCorrupt(msg)
    return manifest


def write_live_manifest(
    manifest: dict,
    *,
    manifest_dir: Path | str | None = None,
) -> Path:
    """Write *manifest* to its season's committed path and return that path."""
    path = live_manifest_path(int(manifest["season"]), manifest_dir=manifest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote live-zone manifest",
        season=manifest.get("season"),
        path=str(path),
        datasets=sorted(manifest.get("datasets", {})),
    )
    return path


def live_covered_seasons(*, manifest_dir: Path | str | None = None) -> set[int]:
    """Return every season with a live manifest that holds at least one capture.

    A manifest file with no captures in it is NOT coverage: it is the shape a season has
    after ``empty_live_manifest`` and before anything was fetched, and treating it as
    covered would send a reader to a capture list that cannot answer.
    """
    directory = Path(manifest_dir) if manifest_dir is not None else LIVE_MANIFEST_DIR
    if not directory.is_dir():
        return set()

    covered: set[int] = set()
    for candidate in sorted(directory.glob("*.json")):
        try:
            season = int(candidate.stem)
        except ValueError:
            continue
        manifest = load_live_manifest(season, manifest_dir=directory)
        if manifest is None:
            continue
        datasets = manifest.get("datasets", {})
        if any(record.get("captures") for record in datasets.values()):
            covered.add(season)
    return covered


def append_capture(manifest: dict, dataset: str, entry: dict) -> dict:
    """APPEND *entry* to *dataset*'s capture list in *manifest*, and return *manifest*.

    Append-only holds literally, not by convention: no existing list element is read,
    rewritten or reordered here. That is the contrast with the sealed tool, which
    REPLACES a season's manifest entry and orphans the parquet it superseded.

    The sequence is unconditionally ``1`` in this plan -- the real per-week sequence rule
    (D32-14: a second capture of the same week appends a new sequence) is Plan 32-04's,
    and lands here.
    """
    record = manifest.setdefault("datasets", {}).setdefault(
        dataset,
        {"loader": DATASET_LOADERS[dataset], "captures": []},
    )
    record["loader"] = DATASET_LOADERS[dataset]
    appended = dict(entry)
    appended["sequence"] = 1
    record["captures"].append(appended)
    return manifest


def captures_for(
    manifest: dict | None,
    dataset: str,
    *,
    week: int | None = None,
) -> list[dict]:
    """Return *dataset*'s captures in stored (append) order, optionally for one week."""
    if not manifest:
        return []
    record = manifest.get("datasets", {}).get(dataset)
    if not record:
        return []
    captures = list(record.get("captures", []))
    if week is None:
        return captures
    return [capture for capture in captures if capture.get("week") == week]


def _capture_inventory(manifest: dict | None, dataset: str) -> str:
    """The ``(week, sequence, captured_at_utc)`` triples that DO exist, as text."""
    captures = captures_for(manifest, dataset)
    if not captures:
        return "  (none -- this dataset has no capture in the live zone at all)"
    return "\n".join(
        f"  (week {capture.get('week')}, sequence {capture.get('sequence')}, "
        f"{capture.get('captured_at_utc')})"
        for capture in captures
    )


def resolve_capture(
    manifest: dict | None,
    dataset: str,
    *,
    week: int | None = None,
    sequence: int | None = None,
) -> dict:
    """Return the one capture a read should serve, or refuse by name.

    With no *week*, the NEWEST capture wins: the highest ``(week, sequence)``. With a
    *week*, only that week's captures are eligible, and an absent one REFUSES rather than
    falling back to the newest -- a replay that silently served a different week would
    reproduce the wrong number while reporting success.
    """
    season = (manifest or {}).get("season", "<season>")
    candidates = captures_for(manifest, dataset, week=week)
    if sequence is not None:
        candidates = [
            capture for capture in candidates if capture.get("sequence") == sequence
        ]

    if not candidates:
        asked = f"{dataset} season {season}"
        if week is not None:
            asked += f" week {week}"
        if sequence is not None:
            asked += f" sequence {sequence}"
        msg = (
            f"The live zone has no capture for {asked}.\n"
            "\n"
            "Captures that DO exist for this dataset, as (week, sequence, "
            "captured_at_utc):\n"
            f"{_capture_inventory(manifest, dataset)}\n"
            "\n"
            "The live zone is append-only and per-week, so covering a season is not the "
            "same as covering a week, and a missing week is NEVER served from a "
            "different one. Capture it deliberately:\n"
            "       .venv/Scripts/python.exe -m scripts.capture_live_season "
            f"--season {season} --week {week if week is not None else '<W>'} "
            f"--dataset {dataset}"
        )
        raise UpstreamLiveCaptureMissing(msg)

    return max(
        candidates,
        key=lambda capture: (capture.get("week") or 0, capture.get("sequence") or 0),
    )


def read_live_frame(entry: dict, data_root: Path | str) -> pd.DataFrame:
    """Read one capture, verifying its bytes against the recorded digest FIRST.

    The ordering is copied from ``data.upstream_pin._read_pinned_frame`` and the reason
    is the same: a check that arrives after the work is a check nobody can afford to
    trust. A capture whose bytes have moved is not the capture a prediction was made
    from, and parsing it anyway would produce a number no manifest can account for.
    """
    path = Path(data_root) / entry["path"]
    if not path.is_file():
        msg = (
            f"The live capture for week {entry.get('week')} sequence "
            f"{entry.get('sequence')} names '{path}', which does not exist. "
            f"{LIVE_MANIFEST_DIR}/ is committed but data/ is gitignored, so a fresh "
            "checkout has the record without the bytes. Re-capture with "
            "`.venv/Scripts/python.exe -m scripts.capture_live_season`."
        )
        raise UpstreamLiveCorrupt(msg)

    actual = digest_file(path)
    if actual != entry["sha256"]:
        msg = (
            f"The live capture at '{path}' does NOT match the digest recorded for it.\n"
            f"  recorded sha256 {entry['sha256']}\n"
            f"  actual   sha256 {actual}\n"
            "A capture is immutable by definition; the live zone appends new ones, it "
            "never rewrites an old one. Either the file was rewritten, or the manifest "
            "entry belongs to a different capture. Refusing rather than trusting bytes "
            "no record accounts for."
        )
        raise UpstreamLiveCorrupt(msg)

    return pd.read_parquet(path)


def _content_through_week(frame: pd.DataFrame) -> int | None:
    """The maximum OBSERVED week in *frame*, or ``None`` when there is none to observe.

    OBSERVED, never computed from the label. ``content_max_week == week - 1`` is FALSE in
    the ordinary case: the Thursday night game that opens each NFL week is played before
    the Friday 6 PM ET freeze, so a week-N capture normally already carries week-N rows.
    Nothing in this phase may assert a computed relationship between the two.
    """
    if WEEK_COLUMN not in frame.columns or frame.empty:
        return None
    observed = pd.to_numeric(frame[WEEK_COLUMN], errors="coerce").dropna()
    if observed.empty:
        return None
    return int(observed.max())


def build_capture_entry(
    dataset: str,
    season: int,
    week: int,
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    path: Path,
    data_root: Path | str,
) -> dict:
    """Describe one capture for the manifest.

    This is ``scripts.pin_upstream_snapshot.capture_season``'s entry EXTENDED -- the
    shared keys keep the same names and meanings -- with the live zone's four additions:
    the ``week`` being predicted, the ``sequence`` that distinguishes a re-capture of it
    (assigned by :func:`append_capture`), the convention string, and the observed
    content depth.
    """
    return {
        "week": week,
        "sequence": None,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "path": Path(path).relative_to(Path(data_root)).as_posix(),
        "sha256": digest_file(Path(path)),
        "bytes": Path(path).stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
        "upstream_width": int(raw.shape[1]),
        "week_label_means": WEEK_LABEL_MEANS,
        "content_through_week": _content_through_week(frame),
    }
