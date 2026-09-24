"""Capture one week of the LIVE nflverse season into the append-only live-zone pin.

This is the live-zone mirror of ``scripts/pin_upstream_snapshot.py``. That tool owns the
SEALED zone: capture a finished season once, never move it. This one owns the season
currently being played: fetch the datasets a gold rebuild reads, write a timestamped
snapshot into ``data/bronze/``, and APPEND a capture record to the committed manifest at
``config/upstream_live/<season>.json``.

The two tools refuse each other's seasons outright, before any fetch. Asking this one for
a sealed season raises ``ZoneWriteRefused`` and names the tool that does own it, because
the failure a mis-aimed capture would cause -- silently rewriting the bytes a published
verdict was measured against -- is not one a warning can undo.

WHAT THE WEEK MEANS
-------------------
``--week`` is the week being PREDICTED, not the last week present in the data. RATIFIED
by the owner on 2026-09-11 (D32-13). A capture labelled week 6 is the capture the week-6
predictions were made from; what it actually contains is recorded separately, as the
manifest's observed ``content_through_week``. Do not compute one from the other: the
Thursday night game that opens each NFL week is played before the Friday 6 PM ET freeze,
so a week-6 capture normally already carries week-6 rows.

USAGE
-----
Capture every dataset for the week being predicted::

    .venv/Scripts/python.exe -m scripts.capture_live_season --season 2026 --week 3

Capture one dataset only::

    .venv/Scripts/python.exe -m scripts.capture_live_season --season 2026 --week 3 \
        --dataset pbp

THE TWO DETECTORS RIDE ALONG (D32-07, D32-08, D32-09)
------------------------------------------------------
Every capture runs both: the LIVE ruling, which diffs this capture against the previous
one over the weeks already consumed, and the SEALED probe, which asks whether nflverse
re-released a season the pin froze. The live verdict is written INTO the capture entry it
was computed from, so a finding and the bytes behind it can never drift apart. The sealed
verdict leaves exactly ONE line per run in the committed
``config/upstream_probe_log.jsonl`` -- clean and UNKNOWN alike, because a season of lines
is the proof the detector was alive and a gap in them is the evidence it was not.

NEITHER CAN FAIL A CAPTURE AND NEITHER CAN GO QUIET. A probe error, a malformed payload
or an unreadable bet list is recorded as an explicit UNKNOWN carrying its reason -- never
a silent skip and never a clean. See :func:`run_detectors`.

Re-run both over the captures already recorded, fetching and writing nothing::

    .venv/Scripts/python.exe -m scripts.capture_live_season --season 2026 --detect-only

THE EXIT CODES ARE A PINNED CONTRACT: ``0`` clean, ``1`` the capture itself failed, ``2``
usage, ``3`` a CRITICAL verdict was recorded, ``4`` an UNKNOWN one. A capture ALWAYS
returns ``0`` when it captured, whatever the verdict; only ``--detect-only`` returns the
distinguishable code, because it is the mode an external check calls.

NOT WIRED INTO THE PIPELINE, DELIBERATELY (D32-04). Phase 32 ships this as a standalone,
fully-tested CLI. Phase 33 wires it as the new FIRST step of the DATA phase, ahead of
``ingest_games``, when it stands up the live run it can actually observe firing. An
unwired capture is a capture nobody runs, so that obligation is carried forward
explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from data.live_revision import (
    GRADED_SOURCE_UNRESOLVED,
    LIVE_REVISION_SCHEMA_VERSION,
    detect_live_revision,
    verdict_severity_rank,
)
from data.live_revision import VERDICT_KEYS as LIVE_VERDICT_KEYS
from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    DEFAULT_SEVERITY,
    VERDICT_SCHEMA_VERSION,
    RevisionEventClass,
    RevisionSeverity,
    severity_rank,
)
from data.sealed_probe import (
    DATASET_RELEASE_TAGS,
    PROBE_STRATEGY_CONTENT,
    SEALED_PROBE_STRATEGY,
    SealedProbeUnavailable,
    fetch_release_assets,
    probe_sealed,
    seed_signatures,
)
from data.sealed_probe import VERDICT_KEYS as SEALED_VERDICT_KEYS
from data.sealed_probe_log import (
    SEALED_PROBE_LOG_PATH,
    SealedProbeLogCorrupt,
    append_probe_entry,
    probe_log_staleness,
)
from data.storage import save_bronze_snapshot
from data.upstream_live import (
    CAPTURE_VERDICT_KEY,
    LIVE_MANIFEST_DIR,
    append_capture,
    attach_verdict,
    build_capture_entry,
    captures_for,
    empty_live_manifest,
    load_live_manifest,
    validate_capture_address,
    write_live_manifest,
)
from data.upstream_pin import (
    DATASET_COLUMNS,
    DATASET_TABLE_NAMES,
    LIVE_ZONE_FIRST_SEASON,
    SEALED_LOCK_PATH,
    SEALED_THROUGH_SEASON,
    ZONE_LIVE,
    UpstreamPinCorrupt,
    UpstreamPinError,
    UpstreamSeasonWindowRefused,
    ZoneWriteRefused,
    default_data_root,
    digest_file,
    load_sealed_lock,
    zone_for_season,
)
from scripts import pin_upstream_snapshot
from scripts.pin_upstream_snapshot import PinCaptureError
from utils import get_logger

logger = get_logger(__name__)

# The exit-code vocabulary, declared once and NEVER renumbered: a caller that learned
# "3 means CRITICAL" from one release must not be silently retrained by the next.
# ``0`` clean, ``1`` a real finding and ``2`` a usage error continue
# ``scripts/pin_upstream_snapshot.py``'s existing codes.
#
# THE FIVE INTEGERS ARE A PINNED CONTRACT, asserted literally by
# ``tests/integration/test_detect_only.py::TestTheExitCodeContractIsPinned``. D32-09 makes
# the code the machine-readable half of "an external check can act on this", and a code
# that is only described in prose is one refactor away from silently becoming 0 -- which
# reads to every caller as "nothing to report". See :func:`exit_code_for` for the mapping
# and for why a CRITICAL outranks an UNKNOWN.
EXIT_OK = 0
EXIT_CAPTURE_FAILED = 1
EXIT_USAGE = 2
EXIT_CRITICAL = 3
EXIT_UNKNOWN = 4

# How long to wait for the clock second to advance past a colliding snapshot name, and
# how many times. 15 x 0.2 s = 3 s, which comfortably outlasts the one-second window the
# collision can occupy while still failing fast if something else is wrong.
_COLLISION_POLL_SECONDS = 0.2
_COLLISION_POLL_ATTEMPTS = 15

# ---------------------------------------------------------------------------
# THE DETECTORS' OWN CONSTANTS (D32-07, D32-08, D32-09)
# ---------------------------------------------------------------------------

# The two rhythms a probe-log line can have been written by, recorded ON the line so the
# committed record can tell an ad-hoc check from the scheduled weekly run.
PROBE_LOG_MODE_CAPTURE = "capture"
PROBE_LOG_MODE_DETECT_ONLY = "detect-only"


def probe_log_mode_partial_capture(completed: int, total: int) -> str:
    """The mode a run records when the capture FAILED partway through (WR-07).

    D32-08's whole reading of the committed log is that "a season of lines is itself the
    proof the detector was alive, and a GAP in them is the evidence that it was not". A gap
    that actually meant "the capture of one dataset raised" would make that reading wrong,
    so a failed run still appends its line -- and the line says so, rather than being
    indistinguishable from a clean weekly run.

    ``mode`` is the natural home for the fact: it is already the field that distinguishes
    one rhythm from another, so a reader who has learned to read it learns nothing new.
    """
    return f"{PROBE_LOG_MODE_CAPTURE} (partial -- the run failed after {completed} of {total} dataset(s))"


# How old the newest probe-log line may be before this run says the detector appears to
# have stopped. The capture runs weekly (Friday), so 7 days is the cadence and the extra 3
# are slack: a run moved from Friday to Monday is an ordinary schedule slip, and a warning
# that fires on one is a warning nobody reads by week three (PITFALLS B3). A gap larger
# than that is the evidence PITFALLS F2 asks for -- a dead detector and a healthy system
# are otherwise the same observable -- and pointing at it costs one file read.
SEALED_PROBE_CADENCE_DAYS: float = 10.0

# THE PROBE-LOG LINE'S FROZEN KEY SET, built from an EXPLICIT list and never from the raw
# HTTP response (T-32-06). Every value written is either computed here or lifted by name
# out of the sealed verdict, so no upstream-supplied field can ride into a committed file
# unnoticed.
#
# The first eleven are the sealed verdict's own :data:`data.sealed_probe.VERDICT_KEYS`;
# the last three are this run's context. Plan 32-08 named THIRTEEN and did not name
# ``verdict_schema_version``; it is carried anyway, because every other committed record in
# this phase stamps the rule that produced it and a line that cannot say which rule wrote
# it is exactly the line a reader a season later cannot use. The count is published here
# rather than described, so it is measurable rather than asserted.
PROBE_LOG_ENTRY_KEYS: tuple[str, ...] = (
    "verdict_schema_version",
    "probed_at_utc",
    "event_class",
    "severity",
    "checked",
    "expected",
    "unresolved",
    "strategy",
    "findings",
    "reason",
    "rate_limit_remaining",
    "mode",
    "season",
    "datasets",
)


# WHEN EACH DATASET'S SEASON BECOMES REQUESTABLE FROM nflreadpy, per dataset. Measured
# against nflreadpy 0.1.5's own gates (``nflreadpy/utils_date.py``) on 2026-09-10.
#
# An operator who hits a season-window refusal needs the DATE, not just the failure: the
# difference between "wait until Thursday" and "this will never work" is the difference
# between a five-minute pause and an afternoon of debugging.
SEASON_WINDOW_HINTS: dict[str, str] = {
    "pbp": (
        "nflreadpy gates load_pbp on get_current_season(), which flips to the new season "
        "on the THURSDAY FOLLOWING LABOR DAY. Until then the season does not exist "
        "upstream at all. (Measured: on 2026-09-09 load_pbp([2026]) raised; on 2026-09-10 "
        "it returned 166 week-1 rows.)"
    ),
    "depth_charts": (
        "nflreadpy gates load_depth_charts on get_current_season(roster=True), which "
        "flips on MARCH 15. A season requested before March 15 of its own year does not "
        "exist upstream."
    ),
    "schedules": (
        "nflreadpy does NOT gate load_schedules on a season window at all -- it downloads "
        "one monolithic games.parquet and filters in memory, returning ZERO ROWS rather "
        "than raising for a season it has no rows for. A season-window refusal from this "
        "dataset therefore means nflreadpy's behaviour changed; investigate rather than "
        "waiting for a date."
    ),
}

# The substring nflreadpy's season-window ValueError carries, for both gated datasets:
# ``ValueError("Season must be between 1999 and 2026")`` from load_pbp and
# ``ValueError("Season must be between 2001 and 2026")`` from load_depth_charts.
_SEASON_WINDOW_MARKER = "Season must be between"


def fetch_live_guarded(dataset: str, season: int) -> pd.DataFrame:
    """Fetch one season from nflverse, translating EVERY failure into the pin's family.

    THIS IS THE FETCH BOUNDARY, and it exists because of one measured hazard.
    ``data/upstream_pin.py``'s module docstring states that ``UpstreamPinError`` inherits
    ``Exception`` and deliberately NOT ``RuntimeError`` / ``ValueError`` / ``ImportError``,
    because every wired call site (``features/qb_tracking.py``, ``features/team_form.py``,
    ``scripts/ingest_games.py``) catches those types and returns an EMPTY frame. And
    ``nflreadpy`` raises exactly those types: ``ValueError`` for a season outside its
    window or an unparseable download, ``ConnectionError`` for a transport failure. A raw
    ``nflreadpy`` exception escaping into a call site is therefore INDISTINGUISHABLE from
    "upstream had no data" -- which is the precise failure the pin's hierarchy exists to
    prevent, and which matters doubly here because D32-03 makes "genuinely zero rows" a
    recordable fact.

    Two outcomes, and they are different facts:

    * A ``ValueError`` whose text matches the season-window shape becomes
      :class:`data.upstream_pin.UpstreamSeasonWindowRefused`, carrying that dataset's
      :data:`SEASON_WINDOW_HINTS` so the operator learns the DATE the season becomes
      requestable.
    * Any other ``ValueError``, and any ``ConnectionError`` or ``OSError``, becomes the
      base :class:`data.upstream_pin.UpstreamPinError` with the dataset, the season and
      the original message in its text.

    Both are chained ``from exc``, so the original traceback survives. Nothing is
    swallowed and nothing is converted into an empty frame: an empty capture is something
    upstream SAYS, never something an exception handler decides on its behalf.
    """
    try:
        return pin_upstream_snapshot.fetch_live(dataset, season)
    except ValueError as exc:
        hint = SEASON_WINDOW_HINTS.get(
            dataset, "No season-window hint for this dataset."
        )
        if _SEASON_WINDOW_MARKER in str(exc):
            msg = (
                f"Refusing the live {dataset} capture for season {season}: nflreadpy "
                f"reports that season is OUTSIDE its own window.\n"
                f"  upstream said: {exc}\n"
                "\n"
                "WHY THIS IS A REFUSAL AND NOT AN EMPTY CAPTURE: an empty capture records "
                "that upstream HAD nothing for this moment, which is a fact about the "
                "season. This is upstream saying the season is not addressable yet, which "
                "is a fact about the request. Recording it as an empty capture would pin a "
                "claim nobody made.\n"
                "\n"
                f"WHEN IT BECOMES REQUESTABLE: {hint}\n"
                "\n"
                "Do ONE of these, deliberately:\n"
                "  1. Wait until the date above, then re-run:\n"
                "       .venv/Scripts/python.exe -m scripts.capture_live_season "
                f"--season {season} --week <W> --dataset {dataset}\n"
                "  2. Capture a dataset whose window is already open -- the three do NOT "
                "open together:\n"
                "       .venv/Scripts/python.exe -m scripts.capture_live_season "
                f"--season {season} --week <W> --dataset schedules"
            )
            raise UpstreamSeasonWindowRefused(msg) from exc

        msg = (
            f"The live {dataset} fetch for season {season} failed inside nflreadpy with a "
            f"ValueError: {exc}\n"
            "\n"
            "Re-raised in the UpstreamPinError family on purpose. Every wired call site "
            "catches ValueError and returns an EMPTY frame, so letting this one through "
            "unchanged would turn a fetch failure into a silently degraded gold matrix "
            "that no manifest could account for. Nothing was written."
        )
        raise UpstreamPinError(msg) from exc
    except OSError as exc:
        # ConnectionError is a subclass of OSError, so this covers nflreadpy's documented
        # transport failure as well as any local filesystem failure inside its cache.
        msg = (
            f"The live {dataset} fetch for season {season} failed in transport: "
            f"{type(exc).__name__}: {exc}\n"
            "\n"
            "A TRANSPORT FAILURE IS NOT AN EMPTY SEASON. Re-raised in the UpstreamPinError "
            "family so it cannot be caught by a call site's (ImportError, ValueError, "
            "RuntimeError, ConnectionError) handler and converted into an empty frame -- a "
            "dead network and a season with no data would otherwise be the same "
            "observable. Nothing was written; re-run the capture."
        )
        raise UpstreamPinError(msg) from exc


def _utc_stamp_now() -> str:
    """The UTC stamp ``data.storage.save_bronze_snapshot`` will put in the filename.

    The format string is duplicated from ``data/storage.py:1009`` DELIBERATELY and is the
    only copy in this module, so a test can pin the clock here and get the same second
    the writer is about to use. Making this the single seam is what lets the same-second
    collision be exercised deterministically instead of raced.
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _bronze_snapshot_glob(table_name: str, season: int, week: int) -> str:
    """The filename pattern ``save_bronze_snapshot`` writes for this (season, week)."""
    return f"{table_name}_raw_bronze_{season}_W{week:02d}_*.parquet"


def _stamp_of(path: Path) -> str:
    """The UTC stamp segment of a bronze snapshot filename."""
    return path.stem.rsplit("_", 1)[-1]


def _reserve_distinct_bronze_second(
    bronze_dir: Path,
    table_name: str,
    season: int,
    week: int,
) -> None:
    """Wait until the current clock second yields a filename nothing already owns.

    ``data/storage.py::save_bronze_snapshot`` builds its filename from a SECOND-resolution
    UTC stamp, and its own comment names the resulting collision as a known defect that
    was out of scope there. The live zone is the first caller for which it is not
    academic: it passes a real ``week``, and two captures of the same ``(season, week)``
    inside one second would produce the SAME path and the second would silently overwrite
    the first -- breaking ``data/bronze/``'s append-only contract literally, and
    destroying the bytes a published week-N prediction was made from.

    So this polls: while any existing snapshot for this ``(table_name, season, week)``
    carries the stamp :func:`_utc_stamp_now` would produce, sleep and re-check. If the
    second never advances within the budget, REFUSE.

    :class:`scripts.pin_upstream_snapshot.PinCaptureError` is the right family here and
    NOT :class:`data.upstream_pin.UpstreamPinError`. This is a capture-tool failure at top
    level -- it cannot reach a builder, because builders read the manifest and never call
    the capture CLI -- and ``PinCaptureError`` is already what the sealed capture tool
    raises for "the snapshot could not be captured faithfully, so nothing was recorded
    for it". Using the read-refusal family for a write-tool failure would put a
    capture-tool bug in the same bucket as a missing pin.
    """
    pattern = _bronze_snapshot_glob(table_name, season, week)
    for _attempt in range(_COLLISION_POLL_ATTEMPTS):
        stamp = _utc_stamp_now()
        colliding = [
            candidate
            for candidate in sorted(bronze_dir.glob(pattern))
            if _stamp_of(candidate) == stamp
        ]
        if not colliding:
            return
        time.sleep(_COLLISION_POLL_SECONDS)

    stamp = _utc_stamp_now()
    colliding = [
        candidate
        for candidate in sorted(bronze_dir.glob(pattern))
        if _stamp_of(candidate) == stamp
    ]
    msg = (
        f"Refusing to capture {table_name} season {season} week {week}: the bronze "
        f"snapshot name for this clock second is already taken by "
        f"{', '.join(str(path) for path in colliding)}.\n"
        "\n"
        "CAUSE: data/storage.py::save_bronze_snapshot builds its filename from a "
        "SECOND-resolution UTC stamp "
        "(<table>_raw_bronze_<season>_W<week>_<YYYYmmddTHHMMSS>.parquet), so two captures "
        "of the same (season, week) inside one second collide and the second would "
        "OVERWRITE the first. data/bronze/ is append-only by contract: a recorded "
        "snapshot is the evidence a superseded verdict was measured against, so it is "
        "never rewritten.\n"
        "\n"
        f"Nothing was fetched into a file and the live manifest is byte-unchanged. "
        f"RECOVERY: re-run the capture; the next second yields a distinct name:\n"
        f"       .venv/Scripts/python.exe -m scripts.capture_live_season "
        f"--season {season} --week {week}"
    )
    raise PinCaptureError(msg)


def _pinned_basis_digest(dataset: str, season: int) -> str:
    """Digest TODAY's upstream ``(dataset, season)`` ON THE SEALED LOCK'S OWN BASIS.

    THE BASIS IS THE WHOLE POINT, and getting it wrong is the one mistake that turns this
    probe into wallpaper. ``data.sealed_probe.probe_sealed`` rules a content pair by
    comparing the digest it is handed against the lock's ``sha256`` -- and that ``sha256``
    is :func:`data.upstream_pin.digest_file` over the parquet THIS PROJECT wrote, after
    ``narrow`` and after nflreadpy filtered the monolithic asset to one season. It is NOT
    the digest of the raw upstream asset stream. The two are different bytes BY
    CONSTRUCTION, so handing ``probe_sealed`` a raw-stream digest would report a move on
    every content pair, forever -- a detector that cries wolf on all of them is exactly as
    useless as one that stays silent. ``probe_sealed`` is pure and cannot check which
    basis it was given, which is why its docstring states the contract and why this
    function is the place that honours it.

    SO THE BASIS IS REPRODUCED BY THE PIN'S OWN CAPTURE PATH, not by a re-implementation:
    ``fetch_live`` -> ``narrow`` -> ``save_bronze_snapshot`` -> ``digest_file``, which is
    ``scripts.pin_upstream_snapshot.capture_season`` line for line, minus the manifest
    write. Agreement is therefore by construction rather than by care.

    MEASURED 2026-09-11, and the reason this approach is usable at all: re-writing the
    already-pinned ``schedules`` 2010 frame through ``save_bronze_snapshot`` into a
    scratch directory reproduced the lock's recorded
    ``720167aa...cd9bb9`` EXACTLY. The parquet writer is deterministic for a fixed frame
    and a fixed pyarrow, so an unchanged upstream digests to an unchanged value. (A
    pyarrow upgrade that changed the encoding WOULD move every content pair at once. That
    shows up as a whole-dataset finding rather than a per-season one, which is a
    distinguishable shape -- and it is recorded, not acted on, exactly like every other
    sealed finding.)

    ``data.sealed_probe.content_digest_for`` is deliberately NOT used, and it is named here
    so nobody reaches for it later: it digests the raw published stream, which is the wrong
    basis above, and there is no second baseline in the lock to compare a raw digest
    against. For the same reason a metadata hit is NOT escalated to it -- ``probe_sealed``'s
    metadata branch rules on the recorded ``upstream_updated_at`` / ``upstream_size`` pair
    and never looks at ``content_digests``, so an escalation could not change any verdict
    and would only spend a 20 MB download to produce a number nothing reads.

    THE SCRATCH WRITE NEVER TOUCHES THE DATA LAKE. The parquet goes into a
    :class:`tempfile.TemporaryDirectory` that is deleted before this function returns, so
    the probe cannot add a file to ``data/bronze/`` -- a detector that wrote into the
    append-only archive it watches would be corrupting its own evidence.

    Raises:
        SealedProbeUnavailable: On any fetch, narrow or write failure. The probe's own
            family and NOT :class:`data.upstream_pin.UpstreamPinError`, deliberately: a
            transport failure while PROBING must never reach the capture as a pin refusal
            and stop the weekly run, because the run reads pinned bytes and is provably
            unaffected by whether GitHub answered.
    """
    try:
        raw = pin_upstream_snapshot.fetch_live(dataset, season)
        frame = pin_upstream_snapshot.narrow(dataset, raw)
    except (OSError, ValueError, RuntimeError, ImportError, PinCaptureError) as exc:
        msg = (
            f"could not fetch the current upstream {dataset} frame for season {season} to "
            f"digest it on the pin's basis: {type(exc).__name__}: {exc}. The probe could "
            "not see upstream for this pair, so it is UNRESOLVED rather than clean."
        )
        raise SealedProbeUnavailable(msg) from exc

    with tempfile.TemporaryDirectory(prefix="sealed-probe-") as scratch:
        try:
            path = save_bronze_snapshot(
                frame,
                table_name=DATASET_TABLE_NAMES[dataset],
                season=season,
                week=0,
                base_path=Path(scratch),
            )
            return digest_file(path)
        except (OSError, ValueError, KeyError) as exc:
            msg = (
                f"could not render the current upstream {dataset} frame for season "
                f"{season} on the pin's basis: {type(exc).__name__}: {exc}."
            )
            raise SealedProbeUnavailable(msg) from exc


def _content_strategy_pairs(lock: dict | None) -> list[tuple[str, int]]:
    """Every ``(dataset, season)`` in *lock* whose strategy rules on CONTENT."""
    pairs: list[tuple[str, int]] = []
    for dataset, seasons in sorted((lock or {}).get("datasets", {}).items()):
        if SEALED_PROBE_STRATEGY.get(dataset) != PROBE_STRATEGY_CONTENT:
            continue
        pairs.extend(
            (dataset, season) for season in sorted(int(value) for value in seasons)
        )
    return pairs


class SealedProbeRun:
    """The sealed half's per-RUN state: one lock, one probe, one memoised verdict.

    ONE PROBE PER RUN, NOT ONE PER DATASET. A three-dataset capture rules on the same 76
    pinned pairs three times over, and re-fetching for each would spend three times the
    rate limit to answer the same question three times -- then write one log line about it
    anyway (D32-08 makes the line per RUN). So the verdict is computed once and
    :meth:`remember`ed, and every later caller in the same run gets that exact mapping
    back. The memo holds an UNKNOWN as readily as a clean one: a run whose probe failed
    must not retry the failure once per dataset and record whichever attempt happened to
    land last.

    The lock is loaded ONCE for the same reason, and a lock that will not load is not an
    exception this class raises -- it is recorded on the verdict by the guard in
    :func:`run_detectors`, because the sealed lock is the detector's input and a detector
    that cannot read its own input is UNKNOWN, never clean.
    """

    def __init__(
        self,
        *,
        lock_path: Path | str | None = None,
        session: object | None = None,
    ) -> None:
        self.lock_path = lock_path
        self.session = session
        self._lock: dict | None = None
        self._lock_loaded = False
        self._lock_error: BaseException | None = None
        self._verdict: dict | None = None

    @property
    def completed(self) -> dict | None:
        """The verdict this run already produced, or ``None`` if it has not probed yet."""
        return self._verdict

    @property
    def cached_lock(self) -> dict | None:
        """The lock IF it already loaded, without attempting to load it.

        The accessor a failure handler uses: reading the lock inside the handler that
        catches a lock failure would raise from inside the guard.
        """
        return self._lock

    def remember(self, verdict: dict) -> None:
        """Record *verdict* as this run's answer, including a guarded UNKNOWN."""
        self._verdict = verdict

    @property
    def lock_error(self) -> BaseException | None:
        """The refusal :meth:`lock` met, or ``None``. Recorded, never raised (WR-02)."""
        return self._lock_error

    def lock(self) -> dict | None:
        """The sealed lock, loaded at most once per run. NEVER raises for a bad lock.

        WR-02. This class's own docstring already promised it -- "a lock that will not load
        is not an exception this class raises" -- and it was false for the single most
        likely lock failure. ``load_sealed_lock`` raises :class:`UpstreamPinCorrupt` (an
        ``UpstreamPinError`` subclass) on an unrecognised ``schema_version``, and
        :func:`run_detectors`'s deliberate ``except UpstreamPinError: raise`` clause sits
        AHEAD of the broad guard that would have recorded it -- so it escaped, unwound
        through :func:`capture_live_dataset` AFTER the bronze snapshot had been written and
        BEFORE ``write_live_manifest``, and the weekly run exited ``EXIT_CAPTURE_FAILED``.
        The week was not recorded at all and an orphan parquet was left in the append-only
        archive, for a reason that had nothing to do with the bytes just fetched. A
        hand-edited or future-versioned lock took the whole live zone offline -- which is
        precisely what D32-07 forbids: "IT CANNOT RAISE FOR A DETECTOR REASON".

        ONLY :class:`UpstreamPinCorrupt` IS CONVERTED, and the narrowness is the point. It
        is the one ``UpstreamPinError`` that can originate in the DETECTOR'S OWN INPUT. Any
        other member of that family reaching :func:`run_detectors` came from the CAPTURE or
        READ path, where a pin refusal genuinely means the bytes are compromised, and the
        WR-10 re-raise must still carry it out untouched.

        The refusal is RECORDED on :attr:`lock_error`, not swallowed: :meth:`probe`
        short-circuits to an UNKNOWN verdict carrying its text, so the committed probe log
        says the detector could not read its own input. A detector that cannot read its
        input is UNKNOWN, never clean.
        """
        if not self._lock_loaded:
            path = self.lock_path if self.lock_path is not None else SEALED_LOCK_PATH
            try:
                self._lock = load_sealed_lock(path)
            except UpstreamPinCorrupt as exc:
                self._lock_error = exc
                self._lock = None
            self._lock_loaded = True
        return self._lock

    def probe(self, *, now: datetime | None = None) -> dict:
        """Fetch what the ruling needs and return one frozen sealed verdict.

        Per-tag and per-pair failures are resolved INTO the verdict rather than raised:
        an unfetchable tag becomes ``None`` in ``assets_by_tag`` and an unobtainable
        content digest is simply absent, and ``probe_sealed`` reports both as UNRESOLVED
        pairs naming what it missed. That is strictly more informative than one exception
        for the whole run -- ``checked < expected`` still forces UNKNOWN, so nothing can
        read as clean, but the line says WHICH pairs went unseen.
        """
        lock = self.lock()

        # WR-02. The lock is the ruling's ONLY baseline, so a lock that would not load
        # means there is nothing to rule against and no point spending three GitHub
        # requests to discover that. Recorded as UNKNOWN carrying the refusal's own text
        # -- never raised, and never clean.
        if self._lock_error is not None:
            return _unknown_sealed_verdict(self._lock_error, lock=None, now=now)

        rate_limits: dict[str, int | None] = {}

        assets_by_tag: dict[str, dict[str, dict] | None] = {}
        for tag in sorted(set(DATASET_RELEASE_TAGS.values())):
            try:
                assets_by_tag[tag] = fetch_release_assets(
                    tag, session=self.session, rate_limits=rate_limits
                )
            except SealedProbeUnavailable as exc:
                logger.warning(
                    "Sealed probe could not read a release tag",
                    tag=tag,
                    error=str(exc),
                )
                assets_by_tag[tag] = None

        content_digests: dict[tuple[str, int], str] = {}
        for dataset, season in _content_strategy_pairs(lock):
            try:
                content_digests[(dataset, season)] = _pinned_basis_digest(
                    dataset, season
                )
            except SealedProbeUnavailable as exc:
                logger.warning(
                    "Sealed probe could not digest a content pair on the pin's basis",
                    dataset=dataset,
                    season=season,
                    error=str(exc),
                )

        observed = [value for value in rate_limits.values() if value is not None]
        return probe_sealed(
            lock,
            assets_by_tag=assets_by_tag,
            content_digests=content_digests,
            now=now,
            # The LOWEST figure observed across the tags, because the calls run in
            # sequence and the lowest is the most recent reading of the headroom left.
            rate_limit_remaining=min(observed) if observed else None,
        )


def _unknown_sealed_verdict(
    exc: BaseException,
    *,
    lock: dict | None,
    now: datetime | None = None,
) -> dict:
    """The sealed verdict a FAILED probe records: UNKNOWN, carrying its reason.

    Built from :data:`data.sealed_probe.VERDICT_KEYS` so a guarded line and a ruled line
    are the same shape in the committed log, and so ``checked`` and ``expected`` are
    present on both -- a verdict recorded without its coverage is a claim nobody can audit.
    ``checked`` is 0 and ``expected`` is every pair the lock records, which is the honest
    statement of what a failed probe looked at.
    """
    pairs = sum(len(seasons) for seasons in (lock or {}).get("datasets", {}).values())
    verdict = dict.fromkeys(SEALED_VERDICT_KEYS)
    verdict.update(
        {
            "verdict_schema_version": VERDICT_SCHEMA_VERSION,
            "probed_at_utc": (now or datetime.now(UTC)).isoformat(),
            "event_class": str(RevisionEventClass.UNKNOWN),
            "severity": str(DEFAULT_SEVERITY[RevisionEventClass.UNKNOWN]),
            "expected": pairs,
            "checked": 0,
            "unresolved": [],
            "strategy": {},
            "findings": [],
            "reason": (
                f"the sealed probe FAILED before it could rule on any of the {pairs} "
                f"pinned pair(s): {type(exc).__name__}: {exc}. Recorded as UNKNOWN rather "
                "than skipped, because a detector that went quiet and a detector that "
                "found nothing are otherwise the same observable."
            ),
            "rate_limit_remaining": None,
        }
    )
    return verdict


def _unknown_live_verdict(
    exc: BaseException,
    *,
    dataset: str,
    season: int,
    current_entry: dict,
) -> dict:
    """The live verdict a FAILED ruling records: UNKNOWN, ``correction_owed`` ``None``.

    ``None`` and never ``False``: ``False`` asserts that no correction is owed, and a
    ruling that failed is by definition unable to make that assertion. The key set is
    :data:`data.live_revision.VERDICT_KEYS`, so a guarded verdict and a ruled one stay
    diffable inside the same season's manifest.
    """
    verdict = dict.fromkeys(LIVE_VERDICT_KEYS)
    verdict.update(
        {
            "verdict_schema_version": VERDICT_SCHEMA_VERSION,
            "live_revision_schema_version": LIVE_REVISION_SCHEMA_VERSION,
            "dataset": dataset,
            "season": int(season),
            "week": current_entry.get("week"),
            "sequence": current_entry.get("sequence"),
            "event_class": str(RevisionEventClass.UNKNOWN),
            "severity": str(DEFAULT_SEVERITY[RevisionEventClass.UNKNOWN]),
            "diff": None,
            CORRECTION_OWED: None,
            CORRECTION_OWED_SCOPE: None,
            "graded_weeks": None,
            "graded_weeks_source": GRADED_SOURCE_UNRESOLVED,
            "reason": (
                f"the live-revision ruling for {dataset} season {season} FAILED: "
                f"{type(exc).__name__}: {exc}. Nothing can honestly be said about whether "
                "this capture moved an already-graded week, so the obligation is recorded "
                "as undecided rather than absent."
            ),
        }
    )
    return verdict


def run_detectors(
    *,
    dataset: str,
    season: int,
    current_entry: dict,
    prior_entry: dict | None = None,
    sealed: SealedProbeRun | None = None,
    graded_output_dir: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    """Run BOTH detectors and return ``{"live": verdict, "sealed": verdict}``.

    THE ONE PLACE EITHER DETECTOR RUNS. The scheduled capture, an ad-hoc ``--detect-only``
    check and the test suite all arrive here, so "the ad-hoc check agrees with the capture"
    is true by construction rather than by two implementations happening to match.

    IT CANNOT RAISE FOR A DETECTOR REASON, and that is D32-07's hardest constraint. The
    detector runs INSIDE the thing it is watching: a probe network error, a malformed
    upstream payload or an unresolvable graded set must not fail a capture that reads and
    writes PINNED bytes and is provably unaffected by anything upstream did. A detector
    that can stop the weekly run is a liability rather than an instrument, and it earns an
    override flag within a month.

    AND IT CANNOT GO QUIET, which pulls the other way. A bare ``except: pass`` would
    satisfy the paragraph above and destroy the detector: a swallowed failure and a clean
    week are the same observable (PITFALLS F2). The only thing that reconciles the two is
    recording an explicit UNKNOWN carrying the failure's own text -- which is what each
    guard below does, on a verdict whose key set is identical to a ruled one's.

    THE TWO HALVES ARE GUARDED INDEPENDENTLY so one going blind does not blind the other.
    A GitHub outage must still leave a live verdict; an unreadable bet list must still
    leave a sealed one.

    ``except UpstreamPinError: raise`` SITS BEFORE EVERY BROAD HANDLER (WR-10), WITH ONE
    NAMED CARVE-OUT (WR-02). A pin refusal is the one failure that must never be degraded:
    ``data/upstream_pin.py``'s
    hierarchy exists because every wired call site converts the ordinary exception types
    into an empty frame, and ``scripts/ingest_games.py:308-320`` records that exact defect
    reaching production -- a bare ``except Exception`` there caught the pin's refusal, the
    ingest logged one warning line and wrote silver ``games`` with no scores merged. A
    broad handler around a detector would do the same thing in the one place nobody is
    watching.

    THE CARVE-OUT, AND WHY IT IS NOT A HOLE IN THAT RULE. The clause above is about pin
    refusals raised by the CAPTURE and READ paths -- the bytes this run is actually
    handling. It was ALSO catching :class:`data.upstream_pin.UpstreamPinCorrupt` raised by
    ``SealedProbeRun.lock()``, which is not the same thing at all: that is the DETECTOR
    failing to read ITS OWN INPUT, and letting it out failed a capture that had already
    written its bronze bytes, leaving an orphan parquet and no manifest entry. So the
    sealed lock read is converted to a verdict INSIDE
    :meth:`SealedProbeRun.lock`, at the point where the origin is still known, rather than
    by widening this clause -- which could not tell the two origins apart. Every other
    ``UpstreamPinError`` still leaves here untouched.

    THE CADENCE AND COVERAGE RULING (32-CONTEXT.md leaves both to the planner; recorded
    here because a discretion nobody wrote down is a discretion the next reader must
    re-litigate):

    * THE SEALED PROBE RUNS ON EVERY CAPTURE AND COVERS ALL THREE SEALED DATASETS.
      Measured: the full three-dataset hybrid costs about 931 KB and 0.8 s per run against
      an unauthenticated 60-request/hour ceiling, and covering play-by-play alone would
      save about 695 KB and 0.2 s. Neither the bytes, the seconds nor the rate limit
      distinguishes the options at any plausible cadence, so the tie is broken on what the
      committed log MEANS: one rhythm gives every line one meaning and makes a gap in the
      record unambiguous, where two cadences would leave "no entry this week" ambiguous
      between "did not run" and "was not due" -- and that ambiguity is the entire thing
      D32-08's log exists to remove.
    * ALL THREE DATASETS ARE CAPTURED EVERY WEEK (the same discretion, one layer out).
      A replay of any week is then TOTAL: every dataset a gold rebuild reads is addressable
      at that week, with no dataset carrying gaps a later reader would have to interpolate
      across. Capturing only what changed would make the manifest a record of the
      capturer's judgement rather than of the season.

    Args:
        dataset: The dataset being ruled on.
        season: The live season.
        current_entry: The capture entry just appended, or the newest existing one under
            ``--detect-only``. ``{}`` when the dataset has never been captured.
        prior_entry: The capture entry immediately before it, or ``None``.
        sealed: The run's :class:`SealedProbeRun`. ``None`` builds a fresh one, which
            probes upstream for real.
        graded_output_dir: The bet-list directory the graded-week record is read from.
            ``None`` uses the configured default.
        now: An injected timezone-aware instant for the sealed verdict's stamp.

    Returns:
        ``{"live": <live verdict>, "sealed": <sealed verdict>}``.
    """
    try:
        # Imported HERE and not at module scope: ``data.graded_weeks`` binds
        # ``api.cache``'s grading vocabulary at import time, and ``api`` is the highest
        # layer in this repository while a capture CLI sits near the bottom. The deferral
        # is also what lets a test monkeypatch ``data.graded_weeks.graded_weeks_record``
        # and have this call site see it.
        from data.graded_weeks import GradedWeeksUnavailable, graded_weeks_record

        try:
            graded: object = graded_weeks_record(season, output_dir=graded_output_dir)
        except GradedWeeksUnavailable as unavailable:
            # Caught AT THE CALL SITE and passed INTO the ruling, so the unresolved
            # branch is explicit. ``detect_live_revision`` reads an exception as "the
            # graded set could not be resolved" and rules UNKNOWN with the obligation
            # undecided; letting the default ``graded=None`` stand instead would say the
            # same thing by omission, which is how an omission becomes an empty set.
            graded = unavailable

        live_verdict = detect_live_revision(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            prior_entry=prior_entry,
            graded=graded,
        )
    except UpstreamPinError:
        # WR-10. See this function's docstring: a pin refusal is re-raised, never
        # degraded, and this clause is placed BEFORE the broad handler on purpose.
        raise
    except Exception as exc:  # noqa: BLE001 - a detector failure is RECORDED, not raised
        live_verdict = _unknown_live_verdict(
            exc, dataset=dataset, season=season, current_entry=current_entry
        )
        logger.warning(
            "Live-revision detector failed; recorded as UNKNOWN",
            dataset=dataset,
            season=season,
            error=str(exc),
        )

    sealed_run = sealed if sealed is not None else SealedProbeRun()
    sealed_verdict = sealed_run.completed
    if sealed_verdict is None:
        try:
            sealed_verdict = sealed_run.probe(now=now)
        except UpstreamPinError:
            # WR-10 again, and for the same reason. The order is asserted at source level
            # by this plan's own verification, not merely intended.
            raise
        except Exception as exc:  # noqa: BLE001 - a detector failure is RECORDED
            # ``_cached_lock`` and not ``lock()``: the lock may be the very thing that
            # failed, and re-reading it here would raise INSIDE the handler that exists to
            # stop this failure from escaping.
            sealed_verdict = _unknown_sealed_verdict(exc, lock=sealed_run.cached_lock)
            logger.warning(
                "Sealed probe failed; recorded as UNKNOWN",
                season=season,
                error=str(exc),
            )
        sealed_run.remember(sealed_verdict)

    return {"live": live_verdict, "sealed": sealed_verdict}


def probe_log_entry(
    sealed_verdict: dict,
    *,
    mode: str,
    season: int,
    datasets: list[str],
) -> dict:
    """Render one committed probe-log line from *sealed_verdict*.

    Built key by key from :data:`PROBE_LOG_ENTRY_KEYS` and never from the raw HTTP
    response (T-32-06), so an upstream-supplied field cannot ride into a committed file
    unnoticed. A key the verdict does not carry is written as ``None`` rather than
    omitted: a line whose key set depends on which branch produced it is not a line a
    reader can diff a season later.
    """
    context = {
        "mode": mode,
        "season": int(season),
        "datasets": sorted(datasets),
    }
    return {
        key: context[key] if key in context else sealed_verdict.get(key)
        for key in PROBE_LOG_ENTRY_KEYS
    }


def record_probe_run(
    sealed_verdict: dict,
    *,
    mode: str,
    season: int,
    datasets: list[str],
    sealed_log: Path | str | None = None,
    now: datetime | None = None,
) -> None:
    """Append EXACTLY ONE line for this run, clean and UNKNOWN alike (D32-08).

    Called once per RUN and never once per dataset. A season of lines is itself the proof
    the detector was alive, so a clean verdict is as worth appending as a finding -- and a
    GAP is the evidence that it was not.

    The staleness check runs FIRST, and warns when the previous line is older than
    :data:`SEALED_PROBE_CADENCE_DAYS`. A gap is already in the record; pointing at it
    costs one file read and turns a fact nobody would look for into a line in this run's
    output.

    NOTHING HERE CAN FAIL THE RUN. An unreadable log BLOCKS a READER
    (``data.sealed_probe_log``'s contract, and rightly: a truncated write must not look
    like a clean slate to an auditor) -- but blocking the CAPTURE on its own detector's
    bookkeeping is the D32-07 prohibition, so the failure is warned about loudly and the
    append is still attempted.
    """
    path = sealed_log if sealed_log is not None else SEALED_PROBE_LOG_PATH

    try:
        staleness = probe_log_staleness(
            path=path, max_age_days=SEALED_PROBE_CADENCE_DAYS, now=now
        )
        if staleness["entries"] == 0:
            logger.info(
                "Sealed probe log is empty; this run writes its first line",
                path=str(path),
            )
        elif staleness["stale"]:
            logger.warning(
                "Sealed probe appears NOT to have run since its last recorded line",
                path=str(path),
                latest_probed_at_utc=staleness["latest_probed_at_utc"],
                age_days=round(float(staleness["age_days"]), 2),
                cadence_days=SEALED_PROBE_CADENCE_DAYS,
                entries=staleness["entries"],
            )
    except (SealedProbeLogCorrupt, ValueError, OSError) as exc:
        logger.warning(
            "Sealed probe log could not be read for a staleness check",
            path=str(path),
            error=str(exc),
        )

    try:
        append_probe_entry(
            probe_log_entry(
                sealed_verdict, mode=mode, season=season, datasets=datasets
            ),
            path=path,
        )
    except (SealedProbeLogCorrupt, OSError) as exc:
        logger.warning(
            "Sealed probe line could NOT be appended; this run leaves a gap in the record",
            path=str(path),
            error=str(exc),
        )


def announce_verdicts(verdicts: dict[str, dict], *, season: int) -> None:
    """Emit a WARNING for every CRITICAL verdict. LOUD, and BLOCKING NOTHING (D32-09).

    The keyword fields are keyword fields and never an f-string payload, so the structured
    record stays queryable -- the ordering and the discipline are
    ``data/upstream_pin.py``'s: warn where a human reads it first, then log where the run
    record keeps it.

    NOTHING HERE RAISES AND NOTHING CHANGES AN EXIT CODE. The capture is a WRITE whose
    correctness does not depend on the verdict: it recorded the bytes upstream served and
    digested them. Failing it on a sealed finding would make a scheduler treat a correct
    step as a broken one, and blocking a run that is demonstrably correct is how a detector
    earns an override flag and then gets ignored. ``config/upstream_pin.json`` is not
    touched here either: re-freezing would erase the evidence the probe exists to produce.
    """
    for name, verdict in sorted(verdicts.items()):
        if verdict_severity_rank(verdict) < severity_rank(RevisionSeverity.CRITICAL):
            continue

        findings = verdict.get("findings") or []
        seasons_by_dataset: dict[str, list[int]] = {}
        for finding in findings:
            seasons_by_dataset.setdefault(str(finding.get("dataset")), []).append(
                finding.get("season")
            )
        scope = verdict.get(CORRECTION_OWED_SCOPE) or {}
        if not seasons_by_dataset and scope:
            seasons_by_dataset[str(scope.get("dataset"))] = [scope.get("season")]

        datasets = sorted(seasons_by_dataset)
        seasons = sorted(
            {value for values in seasons_by_dataset.values() for value in values},
            key=lambda value: (value is None, value),
        )
        print(
            f"UPSTREAM REVISION ({name}, {verdict.get('event_class')}): "
            f"dataset(s) {', '.join(datasets) or '(none named)'} season(s) "
            f"{', '.join(str(value) for value in seasons) or '(none named)'}\n"
            f"  {verdict.get('reason')}\n"
            "  The run is UNAFFECTED: it reads pinned bytes, so this is recorded and not "
            "blocking.",
            file=sys.stderr,
        )
        logger.warning(
            "Upstream revision detected",
            verdict=name,
            event_class=verdict.get("event_class"),
            severity=verdict.get("severity"),
            live_season=int(season),
            datasets=datasets,
            seasons=seasons,
            moved=[
                f"{finding.get('dataset')} {finding.get('season')} "
                f"{finding.get('strategy')}"
                for finding in findings
            ],
            reason=verdict.get("reason"),
            blocking=False,
        )


# The exit code each SEVERITY rank maps to, DERIVED through
# ``data.revision_events.severity_rank`` rather than written out by hand. Building the
# table this way is what makes the precedence in :func:`exit_code_for` a property of the
# frozen ladder instead of a second, silently divergent copy of it.
_EXIT_BY_SEVERITY_RANK: dict[int, int] = {
    severity_rank(RevisionSeverity.INFORMATIONAL): EXIT_OK,
    severity_rank(RevisionSeverity.WARNING): EXIT_UNKNOWN,
    severity_rank(RevisionSeverity.CRITICAL): EXIT_CRITICAL,
}


def exit_code_for(verdicts: dict[str, dict]) -> int:
    """The ONE place a set of verdicts becomes a process exit code (D32-09).

    ``0`` when the loudest verdict is informational, ``3`` when any is CRITICAL, ``4``
    when none is critical and any is UNKNOWN. ``1`` is never returned from here: it means
    the CAPTURE ITSELF failed, and a detector verdict is not a capture failure. ``2`` is
    argparse's.

    PRECEDENCE, FOR THE ADJACENCY THE RULE IS LEAST OBVIOUS ABOUT: when one run carries
    BOTH a CRITICAL and an UNKNOWN, the CRITICAL wins. A confirmed move in bytes this
    project's numbers depend on is strictly more actionable than a probe that could not
    see, and nothing is lost by the choice -- BOTH verdicts are recorded in full, in the
    committed probe log and in the capture entry, whichever integer the process returns.
    The integer is a routing hint for an external check, never the record.

    The comparison goes through ``data.revision_events.severity_rank`` (via
    :func:`data.live_revision.verdict_severity_rank`, which applies it to a verdict) and
    never through a hand-written one. :class:`data.revision_events.RevisionSeverity` is a
    ``StrEnum``, so ``"critical" < "informational"`` is a legal string comparison that
    silently returns the WRONG order -- alphabetical, not loudness -- and routing a
    CRITICAL as the quietest thing in the run would be a one-character mistake.
    """
    ranks = [verdict_severity_rank(verdict) for verdict in verdicts.values()]
    loudest = max(ranks, default=severity_rank(RevisionSeverity.INFORMATIONAL))
    return _EXIT_BY_SEVERITY_RANK[loudest]


def refuse_an_unopened_live_season(season: int) -> None:
    """Refuse a live season after the first until the season before it is over (step 27b).

    OPENING a season needs no human -- the owner's requirement is that August needs no manual
    step -- but it is never guessed from the calendar either: season ``N`` may be written
    only once the RECORDED schedule holds season ``N - 1``'s Super Bowl. So the daily run can
    capture next season's schedule the first day it is due, and nothing can capture a season
    two ahead, or one whose predecessor is still being played.

    Raises:
        ZoneWriteRefused: naming the season and the missing Super Bowl. Nothing is fetched.
    """
    if season <= LIVE_ZONE_FIRST_SEASON:
        return
    from utils.current_slate import season_is_complete

    if season_is_complete(season - 1):
        return
    msg = (
        f"Refusing to open live season {season}: the recorded schedule holds no Super Bowl "
        f"for season {season - 1}, so that season is not over and {season} cannot have "
        "begun. A live season opens only after the one before it ends. Nothing was fetched "
        f"and nothing was written. If season {season - 1} has in fact ended, refresh its "
        f"schedule first: `uv run python -m scripts.ingest_games --season {season - 1}`"
    )
    raise ZoneWriteRefused(msg)


#: The datasets EMPTY BY DEFINITION until a season's first game kicks off: no game played, no
#: plays. D32-03's empty capture for these is proved by the recorded schedule (step 27c).
EMPTY_UNTIL_FIRST_KICKOFF: frozenset[str] = frozenset({"pbp"})

#: The capture-entry key that says WHY an empty capture was recorded in place of a refusal.
EMPTY_CAPTURE_BASIS_KEY = "empty_capture_basis"


def _capture_instant() -> datetime:
    """The moment this capture asks upstream: the one clock the empty-season rule reads."""
    return datetime.now(UTC)


def empty_before_first_kickoff(
    dataset: str,
    season: int,
    refusal: UpstreamSeasonWindowRefused,
    *,
    data_root: Path,
    manifest_dir: Path,
) -> tuple[pd.DataFrame, str] | None:
    """The empty capture a season-window refusal stands for, or ``None`` when it proves nothing.

    Step 27c, applying D32-03. nflverse serves no play-by-play for a new season until the
    Thursday after Labor Day, and nflreadpy REFUSES the request until then -- but a season
    opener locks the day before it is played, so its lock-day run met that refusal and the
    build could not read the season at all. D32-03 already says what this moment is: "upstream
    had nothing for this dataset at this moment", recorded as an explicit empty capture.

    The refusal alone never proves that -- it is a fact about the REQUEST, which is why
    :func:`fetch_live_guarded` raises it. The proof is the RECORDED SCHEDULE: when it holds the
    season and its first kickoff is still after this capture's instant, no game has been played,
    so there are no plays and the empty capture is the truth. Anything else keeps the refusal:
    a dataset that is not empty-until-kickoff, a schedule not recorded (or unreadable), and --
    the case that matters -- a season with a game already kicked off, where upstream serving
    nothing is an outage, not an empty season.

    The empty frame carries the column set and dtypes the pin already serves for the PREVIOUS
    season's play-by-play, so every builder that concatenates the two reads the prior season
    exactly as before (an untyped empty frame would turn integer columns to ``object`` in the
    concatenation). What upstream itself served is recorded separately as width 0.

    Returns:
        ``(empty_frame, basis)``, or ``None`` when the refusal must stand.

    Raises:
        UpstreamPinError: the previous season's play-by-play cannot be read from the pin.
    """
    if dataset not in EMPTY_UNTIL_FIRST_KICKOFF:
        return None
    from utils.current_slate import SlateResolutionError, first_recorded_kickoff

    instant = _capture_instant()
    try:
        first_kickoff = first_recorded_kickoff(season)
    except SlateResolutionError:
        return None
    if first_kickoff is None or first_kickoff <= instant:
        return None

    from data import upstream_pin

    prior = upstream_pin.load_pbp(
        [season - 1], data_root=data_root, live_manifest_dir=manifest_dir
    )
    upstream_said = refusal.__cause__ if refusal.__cause__ is not None else refusal
    basis = (
        f"D32-03 empty capture: nflreadpy refused {dataset} season {season} as outside its "
        f"window ('{upstream_said}') at {instant.isoformat()}, and the recorded schedule's first "
        f"{season} kickoff is {first_kickoff.isoformat()}, after that instant -- no game had "
        "been played, so there were no plays. Columns and dtypes are the previous season's."
    )
    return prior.iloc[0:0].copy(), basis


def capture_live_dataset(
    dataset: str,
    season: int,
    week: int,
    *,
    data_root: Path,
    manifest_dir: Path,
    sealed: SealedProbeRun | None = None,
    graded_output_dir: Path | str | None = None,
) -> dict:
    """Fetch, narrow, write, verify and RECORD one live-zone dataset for one week.

    THE LIVE PATH NARROWS WITH THE SEALED ZONE'S OWN ALLOWLIST (discretion item 7,
    decided here and recorded). Play-by-play is narrowed by
    ``scripts.pin_upstream_snapshot.narrow`` against
    ``data.upstream_pin.PBP_PINNED_COLUMNS`` -- IMPORTED, never re-implemented. A live
    capture carrying a different column set from the sealed zone's would make a mixed
    ``[2025, 2026]`` frame RAGGED at the season boundary, and that mixed request is
    exactly the shape ``features/team_form.py`` issues (it prepends ``all_seasons[0] - 1``
    to every request). One allowlist means one column set on both sides of the join.

    ``narrow`` PRESERVES ABSENCE: an allowlisted column upstream did not supply is left
    absent rather than materialised as all-null, because ``features/team_form.py`` branches
    on ``"cpoe" in group.columns`` and inventing the column would change what gets
    computed. Research measured all 23 pinned columns present in live 2026 today, so that
    branch is NOT exercised by current evidence -- which is why it is held by an explicit
    test instead of an assumption. The risk is not hypothetical: ``depth_charts`` went from
    a 15-column schema (<=2024) to a 12-column one (>=2025) with ZERO name overlap, inside
    the sealed pin's own coverage.

    THE ORDER OF THE LAST FOUR STEPS IS THE CONTRACT, not an implementation detail. The
    bronze bytes are written, :func:`scripts.pin_upstream_snapshot.assert_round_trip_faithful`
    proves they are the frame that was fetched, the entry (with its digests) is built from
    them, and ONLY THEN is the capture appended to the manifest and the manifest written.

    The reason is the one ``data.upstream_pin._read_pinned_frame`` gives for its own
    digest-before-parse ordering: a check that arrives after the work is a check nobody
    can afford to trust. An interruption anywhere between the bronze write and the
    manifest write leaves an UNREFERENCED bronze file and a byte-unchanged
    ``config/upstream_live/<season>.json`` -- a recoverable state, because the next run
    simply writes a new snapshot and nothing ever pointed at the orphan. The reverse
    order would leave a manifest entry pointing at bytes nobody verified, which is
    indistinguishable from a faithful capture at read time and is exactly what the digest
    exists to make impossible.

    THE DETECTORS SIT BETWEEN THE APPEND AND THE WRITE, and cannot fail any of the above:
    :func:`run_detectors` records an explicit UNKNOWN for any failure of its own rather
    than raising, so a GitHub outage or an unreadable bet list costs a verdict and never a
    capture. See :func:`run_detectors` for why that guard is not a swallow.
    """
    zone = zone_for_season(season)
    if zone != ZONE_LIVE:
        msg = (
            f"Refusing to write a LIVE capture for season {season}: it is in the "
            f"{zone!r} zone, not {ZONE_LIVE!r}. The live zone starts at season "
            f"{LIVE_ZONE_FIRST_SEASON}; everything at or before {SEALED_THROUGH_SEASON} "
            "is sealed and immutable. That season belongs to "
            "scripts.pin_upstream_snapshot, which owns the SEALED zone.\n"
            "\n"
            "Nothing was fetched and nothing was written. The two zones have opposite "
            "mutability contracts, so a capture aimed at the wrong one is refused "
            "before it can rewrite bytes a published verdict was measured against."
        )
        raise ZoneWriteRefused(msg)
    refuse_an_unopened_live_season(season)

    empty_basis: str | None = None
    try:
        raw = fetch_live_guarded(dataset, season)
        frame = pin_upstream_snapshot.narrow(dataset, raw)
    except UpstreamSeasonWindowRefused as refusal:
        # Step 27c: before the season's first kickoff the refusal stands for an empty
        # season, proved by the recorded schedule. Otherwise it is re-raised untouched.
        empty = empty_before_first_kickoff(
            dataset, season, refusal, data_root=data_root, manifest_dir=manifest_dir
        )
        if empty is None:
            raise
        frame, empty_basis = empty
        raw = pd.DataFrame()  # upstream served nothing: no rows and no columns

    # D32-03: AN EMPTY LIVE CAPTURE IS RECORDED, NOT REFUSED.
    #
    # There is deliberately no zero-row guard here, in contrast with
    # ``scripts.pin_upstream_snapshot.capture_season``, which refuses one outright. 2026
    # play-by-play is LEGITIMATELY empty before the season's first game, and
    # ``features/team_form.py`` asks for ``[2025, 2026]`` together -- so an empty 2026
    # half is exactly what a live fetch would have returned, and recording it is what
    # makes the live zone able to answer for that moment at all. "Upstream had nothing
    # for this dataset at this moment" is a fact worth pinning: it is digested,
    # attributable and addressable like any other capture, with ``rows: 0`` and an empty
    # ``week_digests`` map whose ``week_partition`` says which case produced it.
    #
    # Step 27c: nflreadpy REFUSES a new season's play-by-play until the Thursday after Labor
    # Day, so before the opener the empty season arrives as a refusal, not as zero rows. The
    # refusal alone claims nothing about the season; the recorded schedule does, and
    # :func:`empty_before_first_kickoff` records the empty capture only on that proof.
    #
    # The SEALED zone keeps its refusal unchanged. A sealed season that came back empty
    # is not a fact about the world, it is a failed capture, and pinning it would starve
    # every builder that reads it for as long as the pin stands.

    table_name = DATASET_TABLE_NAMES[dataset]
    bronze_dir = Path(data_root) / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    _reserve_distinct_bronze_second(bronze_dir, table_name, season, week)
    existing_paths = set(
        bronze_dir.glob(_bronze_snapshot_glob(table_name, season, week))
    )

    try:
        path = save_bronze_snapshot(
            frame,
            table_name=table_name,
            season=season,
            week=week,
            base_path=data_root,
            # WR-04. The live zone is the caller whose append-only contract is
            # LOAD-BEARING: these bronze bytes are the evidence a published week-N
            # prediction was measured against, and the manifest entry written below
            # records a digest OF THEM. An overwrite would destroy a previous week's
            # evidence while leaving that entry claiming it was verified.
            exclusive=True,
        )
    except FileExistsError as collision:
        # WR-04. ``save_bronze_snapshot`` now creates the file EXCLUSIVELY, so a
        # second-resolution collision is refused BEFORE any byte of the previous snapshot
        # is touched -- including across two processes, where the pre-write reservation
        # cannot help because each process's check happens before either writes. The
        # previous snapshot is INTACT here; the message below says so.
        msg = (
            f"The live capture for {dataset} season {season} week {week} COLLIDED with an "
            f"existing bronze snapshot: {collision}.\n"
            "\n"
            "This is the second-resolution filename collision data/storage.py names, "
            "reached despite the pre-write reservation -- which means the reservation and "
            "the write disagreed, or a second process is capturing the same (season, "
            "week) concurrently.\n"
            "\n"
            "NOTHING WAS LOST AND NOTHING WAS RECORDED. The file is created exclusively, "
            "so the earlier snapshot's bytes are intact and the live manifest is "
            "byte-unchanged: no entry claims these bytes were verified. RECOVERY: re-run "
            "the capture; the next second yields a distinct name:\n"
            f"       .venv/Scripts/python.exe -m scripts.capture_live_season "
            f"--season {season} --week {week}"
        )
        raise PinCaptureError(msg) from collision

    if path in existing_paths:
        # UNREACHABLE while save_bronze_snapshot creates exclusively, and kept as
        # defence in depth: if that exclusivity is ever weakened, this notices that the
        # append-only archive was overwritten rather than letting it pass silently.
        msg = (
            f"The live capture for {dataset} season {season} week {week} wrote OVER an "
            f"existing bronze snapshot at '{path}'. data/bronze/ is append-only by "
            "contract, so those bytes were the evidence some earlier verdict was "
            "measured against and they are now gone.\n"
            "\n"
            "The exclusive create in data/storage.py::save_bronze_snapshot should have "
            "made this impossible, so reaching it means that guard was removed or "
            "bypassed. NOTHING was recorded: the live manifest is byte-unchanged, so no "
            "entry claims these bytes were verified."
        )
        raise PinCaptureError(msg)

    pin_upstream_snapshot.assert_round_trip_faithful(frame, path)

    entry = build_capture_entry(dataset, season, week, raw, frame, path, data_root)
    if empty_basis is not None:
        entry[EMPTY_CAPTURE_BASIS_KEY] = empty_basis
    manifest = load_live_manifest(season, manifest_dir=manifest_dir) or (
        empty_live_manifest(season)
    )
    manifest = append_capture(manifest, dataset, entry)

    # THE DETECTOR RUNS BEFORE THE MANIFEST IS WRITTEN, AND THE MANIFEST IS WRITTEN ONCE
    # (D32-08). The verdict is attached to the entry the digests were just computed from,
    # so the finding and the bytes it was ruled on land in the SAME committed write and
    # can never drift apart. Running the detector after the write would need a second
    # write, and two writes can half-fail.
    #
    # The prior entry is the newest capture of this dataset BEFORE the one just appended.
    # ``captures_for`` returns APPEND order, which is the order the runs happened -- the
    # same positional reading ``data.sealed_probe_log.latest_entry`` uses, and for the same
    # reason: re-sorting by a recorded timestamp would let a wrong clock reorder history.
    captures = captures_for(manifest, dataset)
    recorded = captures[-1]
    prior_entry = captures[-2] if len(captures) >= 2 else None
    verdicts = run_detectors(
        dataset=dataset,
        season=season,
        current_entry=recorded,
        prior_entry=prior_entry,
        sealed=sealed,
        graded_output_dir=graded_output_dir,
    )
    attach_verdict(recorded, verdicts["live"])

    write_live_manifest(manifest, manifest_dir=manifest_dir)

    logger.info(
        "Captured live upstream week",
        dataset=dataset,
        season=season,
        week=week,
        sequence=recorded["sequence"],
        rows=recorded["rows"],
        content_through_week=recorded["content_through_week"],
        path=recorded["path"],
        revision_event_class=verdicts["live"]["event_class"],
        sealed_event_class=verdicts["sealed"]["event_class"],
    )
    return recorded


def run_capture(
    datasets: list[str],
    season: int,
    week: int,
    *,
    data_root: Path,
    manifest_dir: Path,
    sealed_lock: Path | str | None = None,
    sealed_log: Path | str | None = None,
    graded_output_dir: Path | str | None = None,
) -> int:
    """Capture every requested dataset for one week. Returns an exit code.

    ALWAYS ``EXIT_OK`` WHEN THE CAPTURE ITSELF SUCCEEDED, whatever the detectors found
    (D32-09). A capture is a WRITE whose correctness does not depend on any verdict: it
    fetched what upstream served, proved the bytes round-trip, and recorded them. A
    non-zero exit here would make a scheduler treat a correct step as a failed one, and a
    weekly job that reports failure every week it is working is a job whose exit code
    stops being read. ``--detect-only`` is the mode an external check calls, and it is the
    mode whose exit code carries meaning -- see :func:`run_detect_only`.

    ONE sealed probe and ONE probe-log line per run, held by the single
    :class:`SealedProbeRun` threaded through every dataset.

    THE LINE IS APPENDED EVEN WHEN THE CAPTURE FAILS (WR-07). The dataset loop sits inside
    a ``try``/``finally``, because a dataset raising -- a season-window refusal, a bronze
    collision, ``_assert_one_season`` -- used to let the exception escape to ``main``'s
    handler with the probe-log line never written, even though :class:`SealedProbeRun` had
    already probed upstream and was holding a complete verdict, possibly a CRITICAL finding
    about a sealed re-release. That discarded a real finding AND forged the one signal
    D32-08's log exists to provide: "a season of lines is itself the proof the detector was
    alive, and a GAP in them is the evidence that it was not". A gap that actually meant
    "one dataset's capture raised" makes that reading wrong for the whole season.

    The partial run's line is MARKED as partial through
    :func:`probe_log_mode_partial_capture`, so it is not silently indistinguishable from a
    complete weekly run either.
    """
    sealed = SealedProbeRun(lock_path=sealed_lock)
    verdicts: dict[str, dict] = {}
    captured: list[str] = []
    try:
        for dataset in datasets:
            entry = capture_live_dataset(
                dataset,
                season,
                week,
                data_root=data_root,
                manifest_dir=manifest_dir,
                sealed=sealed,
                graded_output_dir=graded_output_dir,
            )
            captured.append(dataset)
            verdicts[f"live:{dataset}"] = entry[CAPTURE_VERDICT_KEY]
            print(
                f"{dataset}: season {season} week {week} sequence {entry['sequence']} -- "
                f"{entry['rows']} row(s), {len(entry['columns'])} column(s), content "
                f"through week {entry['content_through_week']}, {entry['path']} "
                f"[{entry[CAPTURE_VERDICT_KEY]['event_class']}]"
            )
    finally:
        # The run's ONE line, appended whether the loop completed or raised. Guarded so a
        # failure in the logging can never REPLACE the capture's own exception -- a
        # traceback naming the log writer instead of the bronze collision that actually
        # happened would be strictly less useful than the one being unwound.
        partial_verdict = sealed.completed
        if partial_verdict is not None and len(captured) != len(datasets):
            try:
                record_probe_run(
                    partial_verdict,
                    mode=probe_log_mode_partial_capture(len(captured), len(datasets)),
                    season=season,
                    datasets=datasets,
                    sealed_log=sealed_log,
                )
            except Exception as log_failure:  # noqa: BLE001 - never mask the real error
                logger.warning(
                    "Could not append the probe-log line for a failed capture",
                    season=season,
                    error=str(log_failure),
                )

    sealed_verdict = sealed.completed
    if sealed_verdict is not None:
        verdicts["sealed"] = sealed_verdict
        record_probe_run(
            sealed_verdict,
            mode=PROBE_LOG_MODE_CAPTURE,
            season=season,
            datasets=datasets,
            sealed_log=sealed_log,
        )
        print(
            f"sealed probe: {sealed_verdict['event_class']} -- checked "
            f"{sealed_verdict['checked']} of {sealed_verdict['expected']} pinned pair(s)"
        )

    announce_verdicts(verdicts, season=season)
    print(f"Wrote {manifest_dir}/{season}.json")
    return EXIT_OK


def run_detect_only(
    datasets: list[str],
    season: int,
    *,
    manifest_dir: Path,
    sealed_lock: Path | str | None = None,
    sealed_log: Path | str | None = None,
    graded_output_dir: Path | str | None = None,
    as_json: bool = False,
) -> int:
    """Re-run the WHOLE detector over captures already in hand. Returns an exit code.

    ONE CODE PATH, NOT TWO. This calls the same :func:`run_detectors` the capture calls,
    over the same entries, so the answer it gives is the answer the capture gave -- proved
    directly by ``tests/integration/test_detect_only.py::
    TestDetectOnlyIsTheSameCodePath::test_it_reproduces_the_verdict_the_capture_recorded``
    rather than assumed. A second implementation for the ad-hoc question would be a second
    place for the ruling to drift, inside a detector, where drift looks like calm.

    IT DIFFERS ONLY IN WHAT IT DOES NOT DO: no dataset content is fetched, no snapshot is
    written, no capture is appended and no live manifest is written. The live half
    compares the two newest EXISTING captures of each dataset -- which are exactly the two
    the in-capture ruling compared when the newer of them was taken. Fewer than two
    captures gives ``no_prior_capture``, exactly as a first real capture does.

    IT DOES STILL PROBE, AND IT DOES STILL APPEND ITS ONE LINE. A probe that ran and left
    no trace reintroduces the dead-detector ambiguity D32-08's log closes; the line records
    ``mode: "detect-only"`` so the ad-hoc rhythm and the weekly one stay distinguishable in
    the committed record.
    """
    manifest = load_live_manifest(season, manifest_dir=manifest_dir)
    sealed = SealedProbeRun(lock_path=sealed_lock)
    verdicts: dict[str, dict] = {}

    for dataset in datasets:
        existing = captures_for(manifest, dataset)
        current = existing[-1] if existing else {}
        prior = existing[-2] if len(existing) >= 2 else None
        verdicts[f"live:{dataset}"] = run_detectors(
            dataset=dataset,
            season=season,
            current_entry=current,
            prior_entry=prior,
            sealed=sealed,
            graded_output_dir=graded_output_dir,
        )["live"]

    sealed_verdict = sealed.completed
    if sealed_verdict is not None:
        verdicts["sealed"] = sealed_verdict
        record_probe_run(
            sealed_verdict,
            mode=PROBE_LOG_MODE_DETECT_ONLY,
            season=season,
            datasets=datasets,
            sealed_log=sealed_log,
        )

    if as_json:
        print(json.dumps(verdicts, indent=2, sort_keys=True, default=str))
    else:
        for name, verdict in sorted(verdicts.items()):
            print(
                f"{name}: {verdict['event_class']} ({verdict['severity']})\n"
                f"  {verdict['reason']}"
            )

    announce_verdicts(verdicts, season=season)
    return exit_code_for(verdicts)


def run_seed_sealed_signatures(
    *,
    sealed_lock: Path | str | None = None,
    ruled_by: str | None,
    allow_reseed: bool = False,
    session: object | None = None,
    now: datetime | None = None,
) -> int:
    """Write the one-time upstream metadata baseline onto the sealed lock (Plan 32-09).

    A BASELINE WRITE, NOT A RUN. It captures nothing, rules on nothing and appends no
    probe-log line: a line in that log means "the detector ran and this is what it saw",
    and a seeding pass saw a baseline it was in the act of creating.

    IT REFUSES WITHOUT AN ATTRIBUTED ``--ruled-by``, and it refuses to overwrite an
    already-seeded baseline without ``--allow-reseed``. Both are ``EXIT_USAGE``: the
    command as invoked cannot be carried out, and the fix is a flag rather than an
    investigation. Re-seeding silently would compare today's upstream against today's
    upstream and report clean forever -- erasing the very divergence a probe exists to
    find.

    The content-strategy digests are computed on the pin's own basis and passed in, so
    ``seed_signatures`` can use them as an agreement CHECK. Stamping a baseline as agreed
    over an unruled content divergence would date-stamp a lie; a pair whose digest could
    not be obtained is simply not checked, and says so.
    """
    attributed = (ruled_by or "").strip()
    if not attributed:
        print(
            "ERROR: --seed-sealed-signatures requires --ruled-by NAME. An unattributed "
            "baseline write is indistinguishable from suppression six months later: the "
            "lock would record that upstream agreed, with nobody having said they looked.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    path = Path(sealed_lock) if sealed_lock is not None else SEALED_LOCK_PATH
    lock = load_sealed_lock(path)
    if lock is None:
        print(
            f"ERROR: there is no sealed-zone lock at '{path}' to seed. Generate it first "
            "with `python -m scripts.pin_upstream_snapshot --refresh-sealed-lock "
            '--sealed-rewrite-reason "<why>"`.',
            file=sys.stderr,
        )
        return EXIT_USAGE

    rate_limits: dict[str, int | None] = {}
    assets_by_tag: dict[str, dict[str, dict] | None] = {}
    for tag in sorted(set(DATASET_RELEASE_TAGS.values())):
        try:
            assets_by_tag[tag] = fetch_release_assets(
                tag, session=session, rate_limits=rate_limits
            )
        except SealedProbeUnavailable as exc:
            print(
                f"SEED REFUSED: could not read release tag {tag!r}: {exc}",
                file=sys.stderr,
            )
            return EXIT_CAPTURE_FAILED

    content_digests: dict[tuple[str, int], str] = {}
    for dataset, season in _content_strategy_pairs(lock):
        try:
            content_digests[(dataset, season)] = _pinned_basis_digest(dataset, season)
        except SealedProbeUnavailable as exc:
            logger.warning(
                "Seeding could not check a content pair against the pin's basis",
                dataset=dataset,
                season=season,
                error=str(exc),
            )

    try:
        updated = seed_signatures(
            lock,
            assets_by_tag=assets_by_tag,
            content_digests=content_digests,
            seeded_at_utc=(now or datetime.now(UTC)).isoformat(),
            seeded_by=attributed,
            overwrite=allow_reseed,
        )
    except ValueError as refusal:
        print(f"SEED REFUSED:\n{refusal}", file=sys.stderr)
        return EXIT_USAGE
    except SealedProbeUnavailable as refusal:
        print(f"SEED REFUSED:\n{refusal}", file=sys.stderr)
        return EXIT_CAPTURE_FAILED

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    seeded = sum(
        1
        for seasons in updated.get("datasets", {}).values()
        for entry in seasons.values()
        if entry.get("upstream_updated_at") is not None
    )
    unseeded = len(updated.get("signatures_unseeded") or [])
    print(
        f"Seeded {seeded} upstream signature(s) into {path} "
        f"({unseeded} pair(s) left unseeded)\n  ruled by: {attributed}"
    )
    return EXIT_OK


def _addressable_week(raw: str) -> int:
    """``--week``'s type. Refuses a week no as-of could ever address again (WR-08).

    ``type=int`` alone accepted ``--week 0`` and ``--week -3``, and the value became the
    capture's PERMANENT replay key ``(season, week, sequence)`` -- the key D32-13 rates
    ONE-WAY. ``data.upstream_live.parse_as_of`` refuses any component below 1, so such a
    capture could never be read back through ``AS_OF_ENV``, which is the "set it once for a
    whole rebuild" affordance Phase 33 is meant to use.

    The rule is NOT re-typed here: it delegates to
    :func:`data.upstream_live.validate_capture_address`, the one definition the context
    manager and the environment variable also use. Two spellings of one rule is two rules,
    and the one that disagrees is the one that records an unaddressable capture.

    Raised as ``argparse.ArgumentTypeError`` so it surfaces as a USAGE error (exit 2) with
    the parser's own formatting, rather than as a capture failure -- nothing was fetched
    and nothing was written.
    """
    try:
        week = int(raw)
    except ValueError as exc:
        msg = f"--week {raw!r} is not an integer."
        raise argparse.ArgumentTypeError(msg) from exc
    try:
        validate_capture_address(week)
    except UpstreamPinError as exc:
        raise argparse.ArgumentTypeError(f"--week {week}: {exc}") from exc
    return week


def build_parser() -> argparse.ArgumentParser:
    summary = (__doc__ or "Capture one live-season week.").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help=(
            f"The live season to capture (the live zone starts at {LIVE_ZONE_FIRST_SEASON}; "
            "a later season opens once the one before it has ended)."
        ),
    )
    parser.add_argument(
        "--week",
        type=_addressable_week,
        default=None,
        help=(
            "The week being PREDICTED -- not the last week present in the data. "
            "The capture records what it actually contains separately. Must be 1 or "
            "greater: it becomes this capture's permanent replay key. Required for a "
            "capture; meaningless for --detect-only and --seed-sealed-signatures, which "
            "write no capture."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_COLUMNS),
        action="append",
        default=None,
        help="Capture only this dataset (repeatable). Default: all of them.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=LIVE_MANIFEST_DIR,
        help=f"Live manifest directory (default: {LIVE_MANIFEST_DIR})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data lake root (default: the configured data root)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help=(
            "Re-run BOTH detectors over the captures already recorded and exit. Fetches "
            "no dataset content, writes no capture and writes no manifest. Returns "
            f"{EXIT_CRITICAL} on a CRITICAL and {EXIT_UNKNOWN} on an UNKNOWN, so an "
            "external check can act on the code."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the --detect-only verdicts as JSON instead of a human summary.",
    )
    parser.add_argument(
        "--sealed-lock",
        type=Path,
        default=None,
        help=f"Sealed-zone lock path (default: {SEALED_LOCK_PATH})",
    )
    parser.add_argument(
        "--sealed-log",
        type=Path,
        default=None,
        help=f"Committed sealed-probe log path (default: {SEALED_PROBE_LOG_PATH})",
    )
    parser.add_argument(
        "--seed-sealed-signatures",
        action="store_true",
        help=(
            "Write the one-time upstream metadata baseline onto the sealed lock and exit. "
            "Requires --ruled-by; refuses an already-seeded baseline without "
            "--allow-reseed. Captures nothing and appends no probe-log line."
        ),
    )
    parser.add_argument(
        "--ruled-by",
        type=str,
        default=None,
        help="WHO is writing the baseline. Required by --seed-sealed-signatures.",
    )
    parser.add_argument(
        "--allow-reseed",
        action="store_true",
        help=(
            "Permit REPLACING an already-recorded upstream baseline. Without it a "
            "re-seed is refused, because seeding over a divergence erases it."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_root = args.data_root if args.data_root is not None else default_data_root()
    datasets = sorted(set(args.dataset)) if args.dataset else sorted(DATASET_COLUMNS)

    try:
        if args.seed_sealed_signatures:
            return run_seed_sealed_signatures(
                sealed_lock=args.sealed_lock,
                ruled_by=args.ruled_by,
                allow_reseed=args.allow_reseed,
            )

        if args.detect_only:
            return run_detect_only(
                datasets,
                args.season,
                manifest_dir=Path(args.manifest_dir),
                sealed_lock=args.sealed_lock,
                sealed_log=args.sealed_log,
                as_json=args.json,
            )

        if args.week is None:
            print(
                "ERROR: --week is required for a capture. It is the week being "
                "PREDICTED, not the last week present in the data. Pass --detect-only to "
                "re-run the detectors over the captures already recorded instead.",
                file=sys.stderr,
            )
            return EXIT_USAGE

        return run_capture(
            datasets,
            args.season,
            args.week,
            data_root=Path(data_root),
            manifest_dir=Path(args.manifest_dir),
            sealed_lock=args.sealed_lock,
            sealed_log=args.sealed_log,
        )
    except ZoneWriteRefused as error:
        # A season this tool does not own is an operator mistake, not a capture failure:
        # nothing was fetched and nothing was written.
        print(f"LIVE CAPTURE REFUSED:\n{error}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:  # noqa: BLE001 - the CLI boundary turns any failure into a code
        print(f"LIVE CAPTURE FAILED: {error}", file=sys.stderr)
        return EXIT_CAPTURE_FAILED


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    sys.exit(main())
