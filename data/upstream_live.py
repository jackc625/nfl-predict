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

import hashlib
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

# The bucket a frame with no ``week`` column falls into. Named here so the sentinel is
# one constant rather than a repeated literal.
NO_WEEK_COLUMN_BUCKET = "__no_week_column__"

# The version of the PER-WEEK DIGEST SHAPE (D32-06), recorded in every capture entry.
#
# D32-06 is rated COSTLY and the shape is frozen here on purpose: it is written into
# every committed capture entry for the whole season, and changing it mid-season would
# mean earlier revision verdicts were computed under a different rule, so the season's
# revision history would stop being comparable end to end. The version stamp is what lets
# a reader of a 2026 entry say which rule produced it instead of assuming.
WEEK_DIGEST_SCHEMA_VERSION: int = 1

# The identity columns every digest is ordered by, in priority order. See
# :func:`week_digests` for why ordering by IDENTITY (and never by value) is the whole
# point: a corrected ``epa`` must move that column's digest and nothing else's.
DIGEST_SORT_KEYS: tuple[str, ...] = ("game_id",)

# The three things a ``week_partition`` can record. Each is an explicit OBSERVATION
# written into the capture entry, never an inference a later reader has to make.
WEEK_PARTITION_PER_WEEK = "per-week (one bucket per observed week value)"
WEEK_PARTITION_WHOLE_FRAME = "whole-frame (dataset has no week column)"
WEEK_PARTITION_EMPTY = "empty (zero rows captured)"

# Recorded verbatim into every capture entry, so the convention travels with the bytes
# rather than living only in this docstring. D32-13, ratified 2026-09-11.
WEEK_LABEL_MEANS = (
    "the week being PREDICTED, not the last week present in the data (D32-13)"
)

# The version of the week-label CONVENTION itself, recorded beside the convention string
# in every capture entry. D32-13 is rated ONE-WAY: from Phase 34 the ledger's reproduction
# key is ``(season, week, sequence)``, so a later change of meaning would silently
# reinterpret every prior capture. Stamping the version is what would let a reader of a
# 2026 entry say which convention it was written under instead of having to assume.
WEEK_LABEL_SEMANTICS_VERSION: int = 1

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


def next_sequence(manifest: dict | None, dataset: str, week: int) -> int:
    """Return the sequence a NEW capture of ``(dataset, week)`` should carry.

    ``max(existing sequences for that (dataset, week)) + 1``, or ``1`` when that week has
    never been captured.

    SEQUENCES ARE PER ``(dataset, week)``, NOT GLOBAL. "Give me week 6 sequence 2" is
    therefore unambiguous without also knowing how many times every OTHER week of that
    dataset was captured -- which is what a global counter would require, and what would
    make a Phase-34 ledger row's reproduction key unreadable on its own. A dataset
    captured for the first time in week 9 starts that week at sequence 1, exactly like
    every other week.
    """
    existing = [
        capture.get("sequence") or 0
        for capture in captures_for(manifest, dataset, week=week)
    ]
    return max(existing) + 1 if existing else 1


def append_capture(manifest: dict, dataset: str, entry: dict) -> dict:
    """APPEND *entry* to *dataset*'s capture list in *manifest*, and return *manifest*.

    THIS IS WHERE D32-14's APPEND-ONLY PROPERTY LIVES. Append-only holds literally, not
    by convention: the only mutation of the capture list is a single ``list.append``.
    Nothing here reads, rewrites, reorders or index-assigns an existing element, so an
    earlier capture's ``path``, ``sha256`` and recorded digests are untouched by
    definition rather than by care.

    That is the deliberate contrast with ``scripts/pin_upstream_snapshot.py``, whose
    docstring (36-39) records the sealed tool's behaviour: capturing an already-pinned
    season REPLACES that season's manifest entry and orphans the parquet it superseded.
    The sealed zone can afford that because a sealed season is captured once; a live
    season is captured every week, and a Saturday re-run after a failed Friday must be a
    recorded fact rather than an erasure.

    The sequence comes from :func:`next_sequence` -- per ``(dataset, week)``, never
    global.
    """
    record = manifest.setdefault("datasets", {}).setdefault(
        dataset,
        {"loader": DATASET_LOADERS[dataset], "captures": []},
    )
    record["loader"] = DATASET_LOADERS[dataset]
    appended = dict(entry)
    appended["sequence"] = next_sequence(manifest, dataset, int(entry["week"]))
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
    """The ``(week, sequence, captured_at_utc)`` triples that DO exist, as text.

    The compact ``(week, sequence)`` pair list is rendered too, and deliberately: an
    operator who addressed a sequence that does not exist needs the set of addresses that
    DO, in the exact form they would retype, not only a prose triple per line.
    """
    captures = captures_for(manifest, dataset)
    if not captures:
        return "  (none -- this dataset has no capture in the live zone at all)"
    triples = "\n".join(
        f"  (week {capture.get('week')}, sequence {capture.get('sequence')}, "
        f"{capture.get('captured_at_utc')})"
        for capture in captures
    )
    pairs = ", ".join(
        f"({capture.get('week')}, {capture.get('sequence')})" for capture in captures
    )
    return f"{triples}\n  addressable (week, sequence) pairs: {pairs}"


def resolve_capture(
    manifest: dict | None,
    dataset: str,
    *,
    week: int | None = None,
    sequence: int | None = None,
) -> dict:
    """Return the one capture a read should serve, or refuse by name.

    Three addressing modes, and each one is a different question (D32-14):

    * *week* and *sequence* both given -- THAT exact capture, or a refusal listing the
      ``(week, sequence)`` pairs that do exist. This is Phase 34's replay address.
    * *week* given, *sequence* omitted -- the HIGHEST sequence for that week. "Default
      reads take the newest capture for that week": a Saturday re-run supersedes the
      Friday one for ordinary reads while the Friday one stays addressable forever.
    * both omitted -- the highest ``(week, sequence)`` overall, i.e. the newest capture.

    An absent week REFUSES rather than falling back to the newest -- a replay that
    silently served a different week would reproduce the wrong number while reporting
    success. The same holds for an absent sequence within a week that does exist.
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


def _weeks_present(frame: pd.DataFrame) -> list[int]:
    """Every distinct week OBSERVED in *frame*, sorted ascending.

    An observation, exactly like :func:`_content_through_week`, and for the same reason:
    a live capture is not a contiguous window. A frame can hold a week's Thursday game
    with the rest of that week still unplayed, and nflverse can append a postseason week
    into a frame whose regular-season weeks are already complete.
    """
    if WEEK_COLUMN not in frame.columns or frame.empty:
        return []
    observed = pd.to_numeric(frame[WEEK_COLUMN], errors="coerce").dropna()
    return sorted({int(value) for value in observed})


def _digest_ordering(frame: pd.DataFrame) -> pd.DataFrame:
    """Return *frame* in the canonical row order every digest is taken over.

    Ordered by the capture's IDENTITY columns (:data:`DIGEST_SORT_KEYS`, i.e. ``game_id``)
    with a STABLE sort, falling back to the frame's full column order when the dataset
    carries no identity key at all.

    Ordering by identity and never by value is load-bearing. A digest taken over rows in
    upstream's arrival order would report a revision every time nflverse happened to emit
    the same rows in a different order. A digest taken over rows sorted by their CONTENT
    would be worse: correcting one ``epa`` value would move that row's position, and every
    other column's digest inside that week would move with it -- so a routine stat
    correction would look identical to a wholesale recompute, which is exactly the
    calibration failure D32-06 exists to prevent.

    Within one ``game_id`` the upstream row order is PRESERVED (``kind="mergesort"`` is
    stable) and is itself part of the captured content: for play-by-play that order is the
    play order.
    """
    keys = [column for column in DIGEST_SORT_KEYS if column in frame.columns]
    if not keys:
        keys = list(frame.columns)
    if not keys:
        return frame
    return frame.sort_values(by=keys, kind="mergesort")


def week_digests(frame: pd.DataFrame) -> dict[str, dict]:
    """Digest *frame* PER WEEK and PER COLUMN (D32-06). The shape is frozen.

    Returns a mapping from the string form of each distinct observed ``week`` value to::

        {"rows": <int>, "frame_sha256": <hex>, "columns": {<column>: <hex>, ...}}

    * The bucket key is ``str(int(week))``, so it survives a JSON round trip as a stable
      key -- a JSON object key is a string either way, and an integer key would come back
      as one anyway without saying so.
    * A frame with NO ``week`` column -- ``depth_charts`` can be one -- yields the single
      bucket :data:`NO_WEEK_COLUMN_BUCKET` over the whole frame. That is a recorded
      observation, never an inference.
    * A frame with zero rows yields ``{}``. The entry's ``week_partition`` says which of
      those two cases produced an unusual map.

    THE DIGEST RENDERING METHOD, NAMED EXPLICITLY because D32-06 is costly and every
    committed 2026 capture entry is written under it: each slice is first put into the
    canonical row order (:func:`_digest_ordering`), then rendered through
    ``pandas.util.hash_pandas_object(..., index=False)`` -- which produces one uint64 per
    row (or per element, for a single column) and is independent of the DataFrame's index
    -- and that array's ``.tobytes()`` is fed to the standard library's sha256 constructor
    (the ONE call in this module, in the nested ``_digest`` below). ``frame_sha256``
    digests the whole ordered slice; each ``columns`` entry digests that one column WITHIN
    the same ordered slice, so a per-column digest and the frame digest always describe the
    same rows in the same order.

    The one hasher in this phase is ``data.upstream_pin.digest_file``, which hashes FILE
    BYTES. This hashes a frame's VALUES, which is a different subject: two faithful
    parquet writes of the same frame can differ in bytes (compression, metadata, row-group
    layout) while carrying identical data, so a byte digest cannot answer "did week 3
    change". No second file hasher is introduced.
    """

    def _digest(subject: pd.DataFrame | pd.Series) -> str:
        rendered = pd.util.hash_pandas_object(subject, index=False).to_numpy().tobytes()
        return hashlib.sha256(rendered).hexdigest()

    def _bucket(slice_: pd.DataFrame) -> dict:
        ordered = _digest_ordering(slice_)
        return {
            "rows": len(ordered),
            "frame_sha256": _digest(ordered),
            "columns": {
                str(column): _digest(ordered[column]) for column in ordered.columns
            },
        }

    if frame.empty:
        return {}
    if WEEK_COLUMN not in frame.columns:
        return {NO_WEEK_COLUMN_BUCKET: _bucket(frame)}

    weeks = pd.to_numeric(frame[WEEK_COLUMN], errors="coerce")
    unreadable = int(weeks.isna().sum())
    if unreadable:
        from scripts.pin_upstream_snapshot import PinCaptureError

        msg = (
            f"{unreadable} of {len(frame)} captured row(s) carry a '{WEEK_COLUMN}' value "
            "that is not a number, so they belong to no week bucket. Recording the "
            "capture anyway would digest FEWER rows than were written, and the missing "
            "rows would never appear in any revision verdict. Refusing rather than "
            "silently narrowing what the digest covers."
        )
        raise PinCaptureError(msg)

    return {
        str(int(week)): _bucket(frame.loc[weeks == week])
        for week in sorted(weeks.unique())
    }


def week_partition_of(frame: pd.DataFrame) -> str:
    """Which of the three :data:`WEEK_PARTITION_PER_WEEK` shapes *frame* produced."""
    if frame.empty:
        return WEEK_PARTITION_EMPTY
    if WEEK_COLUMN not in frame.columns:
        return WEEK_PARTITION_WHOLE_FRAME
    return WEEK_PARTITION_PER_WEEK


def _assert_one_season(dataset: str, season: int, frame: pd.DataFrame) -> None:
    """Refuse a capture whose frame carries rows for a season it does not claim.

    ``nflreadpy.load_schedules`` downloads ONE monolithic ``games.parquet`` covering
    1999-2026 and filters it in memory, so a schedules frame that reached here unfiltered
    would put twenty-seven SEALED seasons' rows inside the LIVE season's digest. Every
    postseason append would then look like a revision of everything, and the sealed zone's
    bytes would be riding in a record whose whole premise is that it changes weekly.

    :class:`scripts.pin_upstream_snapshot.PinCaptureError` and not
    :class:`data.upstream_pin.UpstreamPinError`: this is a capture-tool failure, reachable
    only from the capture CLI, never from a builder's read path. The import is deferred to
    the failure branch so the ``data`` package keeps no module-scope dependency on
    ``scripts``.
    """
    if "season" not in frame.columns or frame.empty:
        return
    observed = pd.to_numeric(frame["season"], errors="coerce").dropna()
    leaked = sorted({int(value) for value in observed} - {int(season)})
    if not leaked:
        return

    from scripts.pin_upstream_snapshot import PinCaptureError

    msg = (
        f"Refusing to record a live {dataset} capture for season {season}: the frame also "
        f"carries rows for season(s) {', '.join(str(value) for value in leaked)}.\n"
        "\n"
        "nflreadpy.load_schedules downloads one monolithic games.parquet covering every "
        "season and filters in memory, so an unfiltered frame reaching here is the "
        "ORDINARY failure, not an edge case. Digesting it would put those seasons' bytes "
        "inside the live zone's per-week digest, and every postseason append would then "
        "report a revision of all of them.\n"
        "\n"
        f"Filter the frame to season {season} before capturing it. Nothing was recorded."
    )
    raise PinCaptureError(msg)


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
    (assigned by :func:`append_capture`), the convention string, the observed content
    depth, and the per-week/per-column digests D32-06 freezes.
    """
    _assert_one_season(dataset, season, frame)
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
        # THE LABEL AND THE CONTENT ARE TWO SEPARATE RECORDED FACTS, AND NEITHER IS
        # COMPUTED FROM THE OTHER.
        #
        # ``week`` and ``week_label_means`` state the CONVENTION (fixed, D32-13).
        # ``content_through_week`` and ``weeks_present`` state what was OBSERVED in the
        # bytes. 32-RESEARCH.md measured, on 2026-09-10, that live ``load_pbp([2026])``
        # already held 166 week-1 rows on the THURSDAY of week 1 -- before any Friday
        # 6 PM ET freeze. So ``content_max_week == week - 1`` is FALSE in the ordinary
        # case, not merely in an edge case, and NO test in this phase may assert any
        # arithmetic relationship between the label and the content.
        "week_label_means": WEEK_LABEL_MEANS,
        "week_label_semantics_version": WEEK_LABEL_SEMANTICS_VERSION,
        "content_through_week": _content_through_week(frame),
        "weeks_present": _weeks_present(frame),
        # D32-06. The shape is frozen for the season; see week_digests's docstring for
        # the rendering method and WEEK_DIGEST_SCHEMA_VERSION for why the stamp is here.
        "week_digest_schema_version": WEEK_DIGEST_SCHEMA_VERSION,
        "week_partition": week_partition_of(frame),
        "week_digests": week_digests(frame),
    }
