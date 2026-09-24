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

import contextlib
import contextvars
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pandas as pd

from data.upstream_pin import (
    DATASET_LOADERS,
    UpstreamLiveCaptureMissing,
    UpstreamLiveCorrupt,
    UpstreamPinError,
    digest_file,
)
from data.write_sink import current_sink
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
#
# VERSION 2 (WR-06/WR-01 of the 32-REVIEW, 2026-09-11). Version 1's ordering fell back to
# sorting by EVERY COLUMN -- i.e. BY VALUE -- whenever no identity key was present, which
# is the exact calibration failure :func:`_digest_ordering`'s own docstring forbids, and
# which the production ``depth_charts`` capture took. Version 2 gives every dataset an
# identity key, refuses to value-sort, and records the ordering it used on every bucket.
#
# THE BUMP IS SAFE TO MAKE MID-SEASON ONLY BECAUSE ``data.live_revision`` NOW READS THIS
# STAMP: a v2 map diffed against a v1 map rules UNKNOWN rather than reporting a
# whole-season revision. That guard (WR-06) landed first, deliberately. The already
# committed 2026 week-1 entries stay stamped 1 and are not rewritten -- an append-only
# record is not re-rendered under a later rule.
#
# G-32-90 CORRECTED THE ``depth_charts`` IDENTITY KEY AND THIS CONSTANT STAYED 2. That is a
# RULING, not an oversight, and it is recorded here because a constant that sat still through
# a rule change looks like one to anyone who only sees the diff. When the correction landed
# (2026-09-11) NO ``week_digest_schema_version: 2`` entry had been written anywhere -- all
# three committed 2026 capture entries are stamped 1 -- so version 2 has only ever meant the
# CORRECTED rule, and no capture exists that a reader could mistake for one written under the
# uncorrected five-column key. A bump to 3 would record a distinction no data can express: it
# would create a version number labelling an EMPTY set of entries while implying some record
# had been written under the five-column rule.
#
# WHAT MAKES THAT ARGUMENT EXPIRE, stated because it is the load-bearing condition: it holds
# only because the correction landed BEFORE the next live capture. Once a version-2 entry
# exists, changing the digest shape or an identity key needs a THIRD version and leaves the
# season reading under two rules -- which is exactly the end-to-end incomparability this
# constant exists to prevent.
WEEK_DIGEST_SCHEMA_VERSION: int = 2

# THE IDENTITY COLUMNS EACH DATASET'S DIGEST IS ORDERED BY, in priority order.
#
# PER DATASET, and that is the WR-01 fix. Ordering by IDENTITY and never by VALUE is the
# whole point of D32-06 -- a corrected ``epa`` must move that column's digest and nothing
# else's -- but version 1 held ONE tuple, ``("game_id",)``, and value-sorted anything that
# did not carry it. ``depth_charts`` carries no ``game_id``, so the single 509,781-row
# production bucket was ordered by all twelve of its columns, by value. Measured on that
# exact column layout: ONE corrected ``player_name`` moved SEVEN of the twelve column
# digests, making a routine roster correction indistinguishable from a wholesale recompute.
#
# ``depth_charts`` has no game grain; its identity is the ROSTER SLOT on a given date --
# ``dt``, ``team``, ``gsis_id``, ``pos_id``, ``pos_slot`` -- with ``espn_id`` behind it as a
# TIE-BREAKER rather than as part of that identity. That is why ``espn_id`` goes LAST: the
# ordering stays readable as "date, team, player, position, slot" with a disambiguator behind
# it. All six columns are present in the committed 2026 capture's column list.
#
# THE KEY MUST BE UNIQUE ON REAL DATA, and the version of this block that shipped with WR-01
# argued the opposite: it said the identity columns "are not required to be UNIQUE: ties keep
# upstream's arrival order, exactly as rows within one ``game_id`` already do". THAT
# EQUIVALENCE IS THE ERROR, and it is corrected here rather than merely worked around, because
# the comment is where the defect was argued for and leaving it standing would re-invite it.
# Within one ``game_id`` the tie order is the PLAY order: captured content, carrying meaning,
# and a digest that preserves it is preserving something real. Within a roster slot on a date
# the tie order is nothing but the order nflverse happened to emit two rows the key cannot
# tell apart, so preserving it publishes a digest that MOVES ON A RE-ORDER with no underlying
# change.
#
# MEASURED (G-32-90, 2026-09-11) on the committed 2026 capture, 509,781 rows: the five-column
# key left 2,163 duplicate rows in 2,023 tie groups, EVERY one of them a group where
# ``gsis_id`` is NULL -- 27,837 rows carry no ``gsis_id`` at all -- and inside those groups
# ``espn_id``, ``pos_rank`` and ``player_name`` differ. So an upstream re-order moved THREE
# column digests with no underlying change: the same false-positive class WR-01 was fixing,
# reduced from 100% of rows to 0.42% rather than eliminated. The arithmetic that closes it:
# with ``espn_id`` in the key there are 0 duplicate rows, and ``espn_id`` has 0 nulls across
# all 509,781 rows, so the key is TOTAL on this dataset rather than merely better.
#
# THE STANDING REQUIREMENT THIS KEY NOW CARRIES: a ``depth_charts`` key must be UNIQUE on the
# real capture, not merely plausible on a fixture. Three hand-built rows with distinct
# ``gsis_id`` values are unique under any prefix of this key, so only real bytes can measure
# it -- ``tests/unit/test_live_zone_capture_boundary.py``'s
# ``TestTheDepthChartsIdentityKeyIsUniqueOnTheRealCapture`` does, against the committed
# capture, so a later edit cannot quietly reintroduce ties.
DIGEST_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "pbp": ("game_id",),
    "schedules": ("game_id",),
    "depth_charts": ("dt", "team", "gsis_id", "pos_id", "pos_slot", "espn_id"),
}

# The ordering fallback used when a dataset is not named above, so a frame whose dataset
# this module was not told about still gets an identity attempt rather than a value sort.
DIGEST_SORT_KEYS_DEFAULT: tuple[str, ...] = ("game_id",)

# Recorded on EVERY digest bucket, so the order a digest was taken in is a stated fact
# rather than something a later reader has to re-derive from this module's source at the
# version the entry was written under.
DIGEST_ORDERING_UNORDERED = (
    "unordered -- no identity key column present, so the rows were NOT sorted and the "
    "per-column map is omitted"
)

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

# The key a live-revision verdict rides under, INSIDE the capture entry it was computed
# from (D32-08). See :func:`attach_verdict` for why it is not a separate file.
CAPTURE_VERDICT_KEY = "revision"


# ---------------------------------------------------------------------------
# THE AS-OF: WHICH CAPTURE A READ SERVES (D32-15)
# ---------------------------------------------------------------------------
#
# ``features/team_form.py``, ``features/qb_tracking.py`` and ``scripts/ingest_games.py``
# each call the pin INDEPENDENTLY, at different depths inside their own call chains. A
# capture-selection keyword threaded through those three builders -- and through
# everything that calls them -- has one path that gets missed, and that path silently
# reads today's newest bytes and returns a number that looks entirely normal. There is no
# traceback and no empty frame to notice.
#
# So capture selection is a PROCESS-LEVEL fact with a per-call override, never a threaded
# keyword: the three call shapes stay exactly as they are and still reach the same as-of.
#
# WHAT PHASE 32 SHIPS, AND WHAT IT DELIBERATELY DOES NOT. D32-15 describes the
# process-level form as "an explicit context manager, or a CLI flag that sets it once for a
# rebuild". Phase 32 ships the context manager (:func:`as_of_capture`), the per-call
# ``as_of=`` keyword on ``data.upstream_pin``'s loaders, and the environment form below. It
# deliberately does NOT add an ``--as-of`` flag to ``scripts/build_features.py`` or
# ``scripts/ingest_games.py``: those belong to the live run Phase 33 stands up, and
# :data:`AS_OF_ENV` already supplies the "set it once for a whole rebuild" affordance
# without this phase reaching into two CLIs it is not otherwise touching.
# ``data.upstream_pin.LIVE_OPT_IN_ENV`` is the established precedent for a process-level
# pin setting carried by the environment.
AS_OF_ENV = "NFL_PREDICT_UPSTREAM_AS_OF"

# The two accepted spellings, quoted into every refusal so the message teaches the form.
AS_OF_ACCEPTED_FORMS = (
    "'<week>' (for example '6') or '<week>:<sequence>' (for example '6:2')"
)


@dataclass(frozen=True)
class AsOfCapture:
    """WHICH live capture a read serves: a week, and optionally one exact sequence.

    ``week`` is the week being PREDICTED, never the last week present in the data
    (D32-13, ratified 2026-09-11). That is what makes a replay addressable with no
    arithmetic: reproducing a Week-6 ledger row is literally ``AsOfCapture(week=6)``.
    Under the rejected convention the address would have been "the capture whose content
    ends at week 5, unless the Thursday game had already landed", which is a computation
    with a wrong answer available.

    ``sequence`` is ``None`` for the ordinary read -- the NEWEST capture of that week,
    which is D32-14's default: a Saturday re-capture supersedes the Friday one for
    ordinary reads while the Friday one stays addressable forever. A set ``sequence``
    addresses one exact capture and is Phase 34's replay address.

    AN AS-OF APPLIES ONLY TO LIVE-ZONE SEASONS. A sealed season has exactly one pinned
    file per dataset, and an as-of never changes which bytes it reads -- there is no
    second candidate for it to choose between. Setting an as-of around a sealed load is
    therefore a no-op on the values, by construction rather than by care, and
    ``tests/unit/test_upstream_pin.py`` holds that as a test over two different as-ofs.
    """

    week: int
    sequence: int | None = None

    def render(self) -> str:
        """The as-of in the exact form :data:`AS_OF_ENV` accepts.

        The round trip is the point: an as-of stamped into a run log can be pasted
        straight back into the environment variable to re-read the same capture.
        """
        if self.sequence is None:
            return str(self.week)
        return f"{self.week}:{self.sequence}"


# The process-level as-of. A ``ContextVar`` and NOT a module-level mutable global: a value
# set inside a thread or an asyncio task must not leak sideways into an unrelated one, and
# this repository runs pytest with ``asyncio_mode = "auto"``, which makes that a live
# concern rather than a hypothetical.
_AS_OF_VAR: contextvars.ContextVar[AsOfCapture | None] = contextvars.ContextVar(
    "nfl_predict_upstream_as_of",
    default=None,
)


def validate_capture_address(week: int, sequence: int | None = None) -> None:
    """Refuse a ``(week, sequence)`` that could never be addressed again (WR-08).

    THE ONE DEFINITION OF WHAT AN ADDRESS MAY BE, so the three ways of naming a capture --
    ``--week`` on the capture CLI, :func:`as_of_capture`, and :func:`parse_as_of` reading
    :data:`AS_OF_ENV` -- cannot disagree about it. They DID disagree: ``parse_as_of``
    refuses any component below 1, while ``--week`` was a bare ``type=int`` and
    ``as_of_capture`` validated nothing at all. So ``--week 0`` and ``--week -3`` were
    accepted and became the capture's PERMANENT replay key ``(season, week, sequence)`` --
    the key D32-13 rates ONE-WAY -- and that capture could then never be addressed through
    the environment variable, which is the "set it once for a whole rebuild" affordance
    Phase 33 is meant to use.

    ``--week 0`` was worse than merely unreachable: it writes a bronze file named
    ``<table>_raw_bronze_<season>_W00_<stamp>.parquet``, which is the SEALED tool's own
    naming convention (``scripts/pin_upstream_snapshot`` passes ``week=0``), so a live
    capture would have been filed under a sealed-looking name. ``--week -3`` produced the
    malformed path segment ``W-3``.

    Raises:
        UpstreamPinError: If *week* or *sequence* is below 1.
    """
    for name, value in (("week", week), ("sequence", sequence)):
        if value is None:
            continue
        if int(value) < 1:
            msg = (
                f"{name} {value} is not a {name}. The week labels the week being "
                "PREDICTED (D32-13) and, with the sequence, becomes this capture's "
                f"PERMANENT replay key (season, week, sequence). {AS_OF_ENV} refuses "
                "anything below 1, so a capture recorded at this address could never be "
                f"read back through it.\n"
                f"\n"
                f"Accepted forms for an as-of: {AS_OF_ACCEPTED_FORMS}."
            )
            raise UpstreamPinError(msg)


@contextlib.contextmanager
def as_of_capture(week: int, sequence: int | None = None) -> Iterator[AsOfCapture]:
    """Read the live zone as of *week* for the duration of the block.

    Every pinned load inside the block -- through any of the three builder call chains,
    at any depth, with their call shapes unmodified -- resolves the live zone to that
    capture.

    The token is RESET in a ``finally``, so the as-of never survives the block, including
    when the body raises. An as-of that outlived its block would silently re-address an
    unrelated later load, which is the same class of wrong-bytes-that-look-normal failure
    the process-level context exists to prevent.

    The address is validated through :func:`validate_capture_address` (WR-08), so the
    context manager and :data:`AS_OF_ENV` agree on what an as-of may be. They did not:
    ``parse_as_of`` refused any component below 1 and this function refused nothing, so
    ``as_of_capture(0)`` built an address the environment variable could not express.
    """
    validate_capture_address(week, sequence)
    active = AsOfCapture(week, sequence)
    token = _AS_OF_VAR.set(active)
    try:
        yield active
    finally:
        _AS_OF_VAR.reset(token)


def parse_as_of(raw: str) -> AsOfCapture:
    """Parse :data:`AS_OF_ENV`'s value, or REFUSE it.

    Accepts ``"6"`` (the newest capture of week 6) and ``"6:2"`` (week 6, sequence 2).
    Anything else -- a non-integer, a non-positive number, an empty string, more than one
    colon -- raises :class:`data.upstream_pin.UpstreamPinError`.

    IT NEVER FALLS BACK TO THE NEWEST CAPTURE. A typo'd as-of that silently read today's
    bytes is exactly the D32-15 hazard: the run would finish, report success, and produce
    a number computed from a capture nobody asked for -- and the number would look
    entirely normal.
    """

    def _refuse() -> NoReturn:
        msg = (
            f"{AS_OF_ENV}={raw!r} is not a capture address.\n"
            "\n"
            f"Accepted forms: {AS_OF_ACCEPTED_FORMS}.\n"
            "  <week>      the week being PREDICTED (D32-13), not the last week present "
            "in the data\n"
            "  <sequence>  one exact re-capture of that week; omit it to read the newest "
            "capture for the week\n"
            "\n"
            "Refusing rather than falling back to the newest capture. A mistyped as-of "
            "that silently read today's bytes would finish the run, report success, and "
            "return a number that looks entirely normal -- computed from a capture "
            "nobody asked for."
        )
        raise UpstreamPinError(msg)

    text = raw.strip()
    parts = text.split(":")
    if not text or len(parts) > 2:
        _refuse()
    try:
        values = [int(part) for part in parts]
    except ValueError:
        _refuse()
    if any(value < 1 for value in values):
        _refuse()
    return AsOfCapture(values[0], values[1] if len(values) == 2 else None)


def current_as_of(*, explicit: AsOfCapture | None = None) -> AsOfCapture | None:
    """The as-of in force, by ONE precedence order defined in ONE place.

    1. *explicit* -- the per-call ``as_of=`` keyword, when given.
    2. :func:`as_of_capture`'s context variable, when set. An explicit ``with`` block is
       more specific than a process-wide setting, so it wins over the environment.
    3. :data:`AS_OF_ENV`, when set and non-blank, through :func:`parse_as_of`. A blank
       value is NOT an as-of, mirroring ``data.upstream_pin.live_upstream_allowed``.
    4. ``None`` -- read the newest capture, which is D32-14's default read.

    Every consumer calls this function. No consumer re-implements the order: two
    implementations of a precedence rule are two rules, and the one that disagrees is the
    one that silently reads the wrong capture.
    """
    if explicit is not None:
        return explicit
    scoped = _AS_OF_VAR.get()
    if scoped is not None:
        return scoped
    raw = os.environ.get(AS_OF_ENV, "")
    if raw.strip():
        return parse_as_of(raw)
    return None


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
    """Write *manifest* ATOMICALLY to its season's committed path; return that path.

    WHY ATOMIC (WR-04). :func:`append_capture`'s docstring argues that append-only "holds
    literally, not by convention: the only mutation of the capture list is a single
    ``list.append``". That was true IN MEMORY and false ON DISK: this function used
    ``path.write_text``, a TRUNCATING, non-atomic rewrite of the file that holds EVERY
    capture of the season. An interruption or a disk-full part way through left a truncated
    JSON document, and :func:`load_live_manifest` would then raise ``JSONDecodeError`` for
    every subsequent read -- the whole season's committed record gone, including the
    verdicts :func:`attach_verdict` exists to keep welded to their bytes.

    ``data/sealed_probe_log.py`` makes exactly this argument when it chooses JSONL append
    over a re-rendered array: "A record whose integrity rests on a serialiser
    round-tripping identically for a whole season is not a record". This file was the
    counterexample. It cannot become append-only -- a verdict is attached to an entry that
    was appended earlier in the same run, so the document genuinely is re-rendered -- but
    the WRITE can be made all-or-nothing, which removes the truncation outcome entirely.

    The pattern is the repository's existing one,
    ``pipeline/execution_log.py::write_execution_log_atomic``: serialise fully, write to a
    temp file in the SAME directory (so ``os.replace`` is a rename within one filesystem
    and therefore atomic), then replace. A crash leaves either the old complete document or
    the new one, never a torn one.

    WHAT THIS DOES NOT FIX, recorded rather than implied: two CONCURRENT captures (the
    Friday scheduler plus an ad-hoc run) still each load the manifest, append, and write,
    with no lock -- so the second write discards the first's entry. The atomic write makes
    that a LOST ENTRY rather than a CORRUPT FILE, which is recoverable by re-capturing the
    week; a truncated document is not recoverable at all. A real fix needs an advisory lock
    around the load-append-write span, which is a larger change than this one and is not
    attempted here.
    """
    path = live_manifest_path(int(manifest["season"]), manifest_dir=manifest_dir)
    # The write sink (Plan 33.2-27 Task 2b): the manifest is COMMITTED; a dry run records the
    # write and leaves the file untouched.
    if not current_sink().authorize(path.as_posix(), "write_live_manifest"):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise BEFORE opening anything: a serialiser that raises must not be able to
    # leave even a temp file behind, and it certainly must not have truncated the target.
    payload = json.dumps(manifest, indent=2) + "\n"

    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # os.replace and not Path.replace: the same deliberate choice
        # pipeline/execution_log.py records, for the same cross-platform reason.
        os.replace(temporary, path)  # noqa: PTH105
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
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


def identity_keys_for(dataset: str | None) -> tuple[str, ...]:
    """The identity columns *dataset*'s digest is ordered by (:data:`DIGEST_SORT_KEYS`)."""
    if dataset is None:
        return DIGEST_SORT_KEYS_DEFAULT
    return DIGEST_SORT_KEYS.get(dataset, DIGEST_SORT_KEYS_DEFAULT)


def _digest_ordering(
    frame: pd.DataFrame, dataset: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Return *frame* in its canonical row order, and the ORDERING that produced it.

    Ordered by *dataset*'s IDENTITY columns (:data:`DIGEST_SORT_KEYS`) with a STABLE sort.
    When the frame carries NONE of them the rows are left UNSORTED and the ordering is
    reported as :data:`DIGEST_ORDERING_UNORDERED`; the caller then omits the per-column map,
    because per-column digests over an arbitrary row order do not mean what this docstring
    says they mean.

    ORDERING BY IDENTITY AND NEVER BY VALUE IS LOAD-BEARING. A digest taken over rows in
    upstream's arrival order would report a revision every time nflverse happened to emit
    the same rows in a different order. A digest taken over rows sorted by their CONTENT
    would be worse: correcting one ``epa`` value would move that row's position, and every
    other column's digest inside that week would move with it -- so a routine stat
    correction would look identical to a wholesale recompute, which is exactly the
    calibration failure D32-06 exists to prevent.

    WR-01: version 1 of this function said exactly that and then did it anyway. Its
    fallback was ``keys = list(frame.columns)`` -- a sort by every column, by value -- and
    that was not a dead branch, it was the branch the real ``depth_charts`` production
    capture took. Measured on the committed 2026 column layout: one corrected
    ``player_name`` moved SEVEN of twelve column digests. The fix is to give each dataset a
    real identity key and to REFUSE to value-sort rather than to fall back to one.

    THE SECOND FAILURE ON THE SAME LINE, also fixed here: ``sort_values`` over every column
    includes object columns, so one mixed-type object column (a str beside a non-NaN float)
    raised ``TypeError`` from inside :func:`week_digests` -- which runs AFTER the bronze
    bytes are written, so it failed the capture and orphaned a snapshot. A sort that cannot
    be performed is now recorded as UNORDERED rather than raised: the ordering is a
    property of the digest, not a reason to lose a week's capture (D32-07).

    Within one identity value the upstream row order is PRESERVED (``kind="mergesort"`` is
    stable) and is itself part of the captured content: for play-by-play that order is the
    play order.
    """
    keys = [column for column in identity_keys_for(dataset) if column in frame.columns]
    if not keys:
        return frame, DIGEST_ORDERING_UNORDERED
    try:
        ordered = frame.sort_values(by=keys, kind="mergesort")
    except TypeError:
        # An identity column holding mixed incomparable types. Recorded, never raised --
        # see this docstring's second paragraph from the end.
        return frame, DIGEST_ORDERING_UNORDERED
    return ordered, "identity: " + ", ".join(keys)


def week_digests(frame: pd.DataFrame, dataset: str | None = None) -> dict[str, dict]:
    """Digest *frame* PER WEEK and PER COLUMN (D32-06). The shape is frozen.

    Returns a mapping from the string form of each distinct observed ``week`` value to::

        {"rows": <int>, "ordering": <str>, "frame_sha256": <hex>,
         "columns": {<column>: <hex>, ...}}

    * *dataset* selects the identity columns the rows are ordered by
      (:data:`DIGEST_SORT_KEYS`). It is optional so an ad-hoc caller can digest a frame
      without naming it, in which case :data:`DIGEST_SORT_KEYS_DEFAULT` applies.
    * ``ordering`` is NEW IN SCHEMA VERSION 2 and rides on EVERY bucket, including the
      ordinary one. It names the identity columns the slice was sorted by, or reports
      :data:`DIGEST_ORDERING_UNORDERED`. A key that appeared only on the unusual branch
      would make the bucket shape depend on which branch produced it, which is the exact
      property this module refuses everywhere else.
    * An UNORDERED bucket carries an EMPTY ``columns`` map. Per-column digests over an
      arbitrary row order would be numbers that do not answer "did this column change",
      and publishing them under a name that claims they do is worse than publishing
      nothing. ``rows`` and ``frame_sha256`` still detect the movement; what is lost is
      only the per-column ATTRIBUTION, and ``ordering`` says so on the bucket itself.
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
        ordered, ordering = _digest_ordering(slice_, dataset)
        columns = (
            {}
            if ordering == DIGEST_ORDERING_UNORDERED
            else {str(column): _digest(ordered[column]) for column in ordered.columns}
        )
        return {
            "rows": len(ordered),
            "ordering": ordering,
            "frame_sha256": _digest(ordered),
            "columns": columns,
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
        # ``dataset`` is PASSED, not inferred from the frame's columns (WR-01). Which
        # identity key a digest was ordered by has to be a property of the dataset being
        # captured, or a frame that happened to arrive missing a column would silently
        # switch ordering rules mid-season.
        "week_digests": week_digests(frame, dataset),
    }


def attach_verdict(entry: dict, verdict: dict) -> dict:
    """Write *verdict* onto *entry* under :data:`CAPTURE_VERDICT_KEY`, and return *entry*.

    D32-08, and the whole reason the verdict is not its own file: A FINDING AND THE BYTES
    IT WAS COMPUTED FROM CAN NEVER DRIFT APART. The verdict is written into the SAME
    entry, and therefore in the SAME :func:`write_live_manifest` call, as the digest it
    was ruled from. Two files would be two writes, and two writes can half-fail -- leaving
    a verdict about bytes no entry records, or an entry whose verdict is about a different
    capture. Neither is distinguishable after the fact from an honest pair.

    *entry* is MUTATED in place rather than copied, deliberately. The caller holds the
    entry that is already inside the manifest (``append_capture`` appended it), so a copy
    would be attached to an object the manifest write never sees -- the verdict would
    simply not appear in the committed record, silently and with no error.

    A SECOND ATTACH IS REFUSED. Re-ruling an entry that already carries a verdict would
    restate an old finding under whatever rule is current now, and the committed record
    would then claim the later ruling was the one drawn at capture time. That is the same
    hazard :func:`data.live_revision.as_record` avoids by storing ``is_revision`` rather
    than leaving it to be recomputed: what a detector CONCLUDED when it looked is a fact,
    and facts are appended, never edited.

    Raises:
        UpstreamLiveCorrupt: If *entry* is not a mapping, or already carries a verdict.
    """
    if not isinstance(entry, dict):
        msg = (
            f"a capture entry must be a mapping to carry a verdict, got "
            f"{type(entry).__name__}. Nothing was attached."
        )
        raise UpstreamLiveCorrupt(msg)

    if CAPTURE_VERDICT_KEY in entry:
        msg = (
            f"the capture entry for week {entry.get('week')} sequence "
            f"{entry.get('sequence')} already carries a {CAPTURE_VERDICT_KEY!r} block, so "
            "a second verdict is REFUSED.\n"
            "\n"
            "A re-ruled entry would silently restate an earlier finding under a later "
            "rule, and the committed record would then report the new ruling as the one "
            "drawn when the capture was taken. The live zone appends captures; it never "
            "re-rules one. Capture the week again if a fresh ruling is wanted -- that is "
            "a new entry, with its own sequence, and both stay addressable."
        )
        raise UpstreamLiveCorrupt(msg)

    entry[CAPTURE_VERDICT_KEY] = verdict
    return entry
