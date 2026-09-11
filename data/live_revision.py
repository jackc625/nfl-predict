"""The LIVE-capture revision ruling: what a capture-to-capture difference MEANS.

WHAT THIS MODULE IS
-------------------
A PURE ruling over two capture entries that are ALREADY IN HAND. It is D32-05's FREE half:
"diff each 2026 capture against the previous one over the weeks already consumed -- both
frames are already in hand". It fetches nothing and downloads nothing, because there is
nothing to fetch; ``data/upstream_live.py`` already recorded, inside each committed capture
entry, the per-week and per-column digests D32-06 froze. This module only compares them and
rules on what the comparison means.

It performs NO I/O of any kind -- no network call, no file read, no parquet read. That is
asserted at source level by ``tests/unit/test_revision_severity.py``, not merely intended.
Purity is what makes the ruling testable offline against synthetic graded sets, which is
what D32-11 requires of the escalation branch built here.

WHO CALLS IT
------------
``scripts/capture_live_season.py`` (Plan 32-08) and nothing else. That caller resolves the
graded-week record and passes it in; see :func:`detect_live_revision` on why the graded read
is a PARAMETER and not an import.

THE ONE CALIBRATION THAT DECIDES WHETHER THIS DETECTOR IS READ
---------------------------------------------------------------
`.planning/research/PITFALLS.md` B3. Both failure modes are silent in opposite directions:

* A revision alert EVERY WEEK means the detector is miscalibrated, and it is wallpaper by
  week three -- the first genuinely important finding then arrives into an inbox nobody
  reads. nflverse restates the running week's play-by-play routinely, and a live season
  grows a whole new week every Friday.
* NO revision alert ALL SEASON means the detector is dead (PITFALLS F2), and a dead detector
  and a healthy system are the same observable.

So the predicate in :attr:`WeekDiff.is_revision` is load-bearing, and the ruling below
resolves every ambiguity LOUD: a downgraded CRITICAL and a genuinely calm week look
identical in a log, and only one of those two mistakes is recoverable after the fact.

ASCII only, no emoji (CLAUDE.md hard constraint).

Run the tests:  .venv/Scripts/python.exe -m pytest tests/unit/test_revision_severity.py -q
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    DEFAULT_SEVERITY,
    VERDICT_SCHEMA_VERSION,
    RevisionEventClass,
    RevisionSeverity,
    severity_rank,
)

__all__ = [
    "CORRECTION_DISCHARGED_BY",
    "GRADED_SOURCE_UNRESOLVED",
    "GRADED_SOURCE_UNSTATED",
    "LIVE_REVISION_SCHEMA_VERSION",
    "VERDICT_KEYS",
    "WeekDiff",
    "as_record",
    "compare_week_digests",
    "detect_live_revision",
    "verdict_severity_rank",
]


# The version of the DIFF AND VERDICT SHAPE this module writes, stamped onto every record
# it produces. It sits beside -- and is deliberately separate from --
# ``data.upstream_live.WEEK_DIGEST_SCHEMA_VERSION`` (the shape of the INPUT) and
# ``data.revision_events.VERDICT_SCHEMA_VERSION`` (the shape every verdict class in this
# project shares). Three stamps because three shapes can move independently: the digest
# could gain a field without the ruling changing, and the ruling could gain a field without
# the shared verdict envelope changing. A reader of a 2026 entry can then say which rule
# produced it rather than assuming.
LIVE_REVISION_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class WeekDiff:
    """What moved between two captures: week by week, column by column, row count by row count.

    Every collection is SORTED, and the two mappings are built in sorted key order. That is
    not tidiness: the rendered form of this object rides inside a committed capture entry in
    ``config/upstream_live/<season>.json``, so an unstable iteration order would show up in
    a review diff as though the data had moved when nothing had. Two runs over the same two
    digest maps must render byte-identically.

    The sort is the plain lexicographic sort of the STRING bucket key, not a numeric sort of
    the week. The bucket set is not all numbers -- a dataset with no ``week`` column records
    the single ``data.upstream_live.NO_WEEK_COLUMN_BUCKET`` sentinel -- and a comparator that
    has to special-case the sentinel is a second rule with a wrong answer available. The
    string key is what ``week_digests`` froze, so the string sort is the one order that
    always exists.
    """

    weeks_changed: tuple[str, ...] = ()
    weeks_added: tuple[str, ...] = ()
    weeks_removed: tuple[str, ...] = ()
    columns_changed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rows_changed: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def is_revision(self) -> bool:
        """Whether this difference is a REVISION rather than ordinary in-season growth.

        THE LOAD-BEARING PREDICATE. `.planning/research/PITFALLS.md` B3, by name.

        ``weeks_added`` is IGNORED ENTIRELY, and that single exclusion is the difference
        between a detector that gets read and one that gets filtered to a folder. A NEW week
        appearing in a later capture is exactly what a live season does every Friday: week 6
        is captured, then week 7 is captured and contains a week-7 bucket that week 6's
        capture could not have had. There is nothing to report about it. A detector that
        fired on it would fire on EVERY capture of every week of the season, and by week
        three the owner would have learned to ignore the one line that eventually matters.

        A week present in BOTH captures whose digest moved IS a revision: upstream restated
        something it had already published.

        A week REMOVED is a revision too, and a LOUDER claim than a changed one: upstream
        withdrawing a week it previously published says the earlier publication should not
        have been relied on at all, which is strictly stronger than saying its values have
        been corrected. It is called out here rather than left implicit from
        ``weeks_removed`` appearing in the ``or`` below, because the reason is not visible
        in the code.
        """
        return bool(self.weeks_changed or self.weeks_removed)


def _bucket_columns(bucket: object) -> dict[str, str]:
    """The ``columns`` sub-map of one ``week_digests`` bucket, defensively."""
    if not isinstance(bucket, dict):
        return {}
    columns = bucket.get("columns")
    if not isinstance(columns, dict):
        return {}
    return {str(name): columns[name] for name in columns}


def _changed_columns(prior_bucket: object, current_bucket: object) -> tuple[str, ...]:
    """The sorted column names whose digest differs between two buckets.

    A column present in ONE bucket's map and absent from the other's counts as CHANGED, not
    as a non-comparison. A column that appeared is new information inside a week that was
    already published; a column that vanished is information withdrawn from it. Reporting
    either as "nothing to compare" would let a schema move through the detector in silence,
    which is the quiet direction this module refuses everywhere.
    """
    prior_columns = _bucket_columns(prior_bucket)
    current_columns = _bucket_columns(current_bucket)
    missing = object()
    return tuple(
        sorted(
            name
            for name in set(prior_columns) | set(current_columns)
            if prior_columns.get(name, missing) != current_columns.get(name, missing)
        )
    )


def _row_counts(prior_bucket: object, current_bucket: object) -> tuple[int, int] | None:
    """The ``(prior_rows, current_rows)`` pair when the row count moved, else ``None``.

    Reported INDEPENDENTLY of the column digests. A week can gain or lose rows while every
    surviving column's digest stays comparable, and the row count is the one signal that
    still says something when a dataset carries no column map at all.
    """
    prior_rows = prior_bucket.get("rows") if isinstance(prior_bucket, dict) else None
    current_rows = (
        current_bucket.get("rows") if isinstance(current_bucket, dict) else None
    )
    if prior_rows == current_rows:
        return None
    return (int(prior_rows or 0), int(current_rows or 0))


def _frame_digest(bucket: object) -> object:
    """One bucket's whole-slice digest, or ``None`` when the bucket is not a mapping."""
    return bucket.get("frame_sha256") if isinstance(bucket, dict) else None


def compare_week_digests(prior: dict, current: dict) -> WeekDiff:
    """Diff two ``data.upstream_live.week_digests`` maps. Pure, and never mutates either.

    Args:
        prior: The EARLIER capture's ``week_digests`` map -- a mapping from the string week
            bucket to ``{"rows", "frame_sha256", "columns": {name: sha}}``.
        current: The LATER capture's map, in the same shape.

    Returns:
        A :class:`WeekDiff`. A week is reported CHANGED when any of its three observables
        moved -- the whole-slice digest, the row count, or any column digest. All three are
        checked rather than trusting ``frame_sha256`` alone: the frame digest is the
        authoritative one, but a diff that reported a week as unchanged while also reporting
        a moved row count for it would be an internally contradictory record, and the two
        cheap extra comparisons remove that possibility entirely.
    """
    prior_weeks = set(prior)
    current_weeks = set(current)
    shared = sorted(prior_weeks & current_weeks)

    columns_changed: dict[str, tuple[str, ...]] = {}
    rows_changed: dict[str, tuple[int, int]] = {}
    changed: list[str] = []

    for week in shared:
        prior_bucket = prior[week]
        current_bucket = current[week]

        moved_columns = _changed_columns(prior_bucket, current_bucket)
        if moved_columns:
            columns_changed[week] = moved_columns

        moved_rows = _row_counts(prior_bucket, current_bucket)
        if moved_rows is not None:
            rows_changed[week] = moved_rows

        digest_moved = _frame_digest(prior_bucket) != _frame_digest(current_bucket)
        if digest_moved or moved_columns or moved_rows is not None:
            changed.append(week)

    return WeekDiff(
        weeks_changed=tuple(changed),
        weeks_added=tuple(sorted(current_weeks - prior_weeks)),
        weeks_removed=tuple(sorted(prior_weeks - current_weeks)),
        columns_changed=columns_changed,
        rows_changed=rows_changed,
    )


def as_record(diff: WeekDiff) -> dict:
    """Render *diff* as the JSON-serialisable block that rides inside a capture entry.

    The FULL diff is recorded, not a summary. A record that said only "week 3 changed" would
    force every later reader to take the finding on trust, with no way to ask which columns
    moved or by how many rows -- and the frames it was computed from are the previous
    capture's, which a reader months later has no cheap way to recompute. Recording the whole
    thing costs a few hundred bytes per capture and makes the entry self-supporting.

    ``is_revision`` is rendered as a stored field rather than left for a reader to recompute
    from the collections, so the committed record says what the detector CONCLUDED at the
    moment it looked. If the predicate is ever retuned, the old entries keep reporting the
    conclusion that was actually drawn under the old rule instead of silently re-deciding
    themselves under the new one.
    """
    return {
        "live_revision_schema_version": LIVE_REVISION_SCHEMA_VERSION,
        "weeks_changed": list(diff.weeks_changed),
        "weeks_added": list(diff.weeks_added),
        "weeks_removed": list(diff.weeks_removed),
        "columns_changed": {
            week: list(columns) for week, columns in diff.columns_changed.items()
        },
        "rows_changed": {week: list(pair) for week, pair in diff.rows_changed.items()},
        "is_revision": diff.is_revision,
    }


# ---------------------------------------------------------------------------
# THE RULING: what the difference MEANS, how loud it is, and what it OWES.
# ---------------------------------------------------------------------------

# The FOURTEEN keys every verdict carries, on EVERY branch. Frozen here the way
# ``data/sealed_probe.py`` freezes ``VERDICT_KEYS`` and for the same reason: a verdict rides
# inside its own capture entry in ``config/upstream_live/<season>.json`` (D32-08), which
# accumulates all season, so a line whose key set depends on which branch produced it is not
# a line a reader can diff a season later. Four of the names -- ``verdict_schema_version``,
# ``event_class``, ``severity``, ``reason`` -- are deliberately the SAME names the sealed
# probe uses for the same meanings, so the two verdict classes read as one vocabulary in the
# committed record rather than as two dialects.
#
# Adding a key means bumping ``data.revision_events.VERDICT_SCHEMA_VERSION``; renaming one
# makes the already-written entries stop meaning what they said.
VERDICT_KEYS: tuple[str, ...] = (
    "verdict_schema_version",
    "live_revision_schema_version",
    "dataset",
    "season",
    "week",
    "sequence",
    "event_class",
    "severity",
    "diff",
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    "graded_weeks",
    "graded_weeks_source",
    "reason",
)


# Who discharges the obligation this module only RAISES. Recorded as a value inside the
# scope rather than described in a sentence, so "which corrections are still owed, and by
# whom" stays a query.
CORRECTION_DISCHARGED_BY: str = "phase-34 ledger correction block"


# The provenance string recorded when the graded set could NOT be resolved. It is a
# deliberate non-answer: a verdict must never be able to report a source that implies it
# read something when it did not.
GRADED_SOURCE_UNRESOLVED: str = "unresolved -- the graded-week record could not be resolved; see this verdict's reason"

# The provenance string recorded when a caller supplied weeks but no source of its own.
GRADED_SOURCE_UNSTATED: str = (
    "unstated -- the caller supplied a graded-week set without naming where it looked"
)


def _resolve_graded(graded: object) -> tuple[list[int] | None, str, str | None]:
    """Interpret the *graded* argument. Returns ``(weeks, source, failure)``.

    ``failure`` is ``None`` only when the graded set was genuinely resolved. Everything else
    -- an absent argument, a raised :class:`data.graded_weeks.GradedWeeksUnavailable`, a
    record that reports itself unresolved, a shape this function does not recognise --
    returns a failure sentence and ``weeks`` of ``None``.

    A MISSING ARGUMENT IS A FAILURE, NOT AN EMPTY SET. That is the single most important
    line in this function. ``graded=None`` is the default only so the parameter can be named
    at the call site; a caller that forgot to resolve the graded record has told this module
    NOTHING, and reading "nothing" as "nothing is graded" is exactly the silent downgrade
    D32-11 exists to prevent -- a CRITICAL becomes an informational and the log looks calm.
    An EXPLICITLY RECORDED empty set (``{"weeks": [], "resolved": True, ...}``, which is what
    ``data.graded_weeks.graded_weeks_record`` returns for an absent store) is a different
    input and is honoured as the genuine empty answer it is.
    """
    if graded is None:
        return (
            None,
            GRADED_SOURCE_UNRESOLVED,
            "no graded-week record was supplied to the detector, so which weeks have "
            "already been graded is unknown. An absent argument is NOT an empty set: "
            "reading it as one would silently downgrade a revision of a settled week.",
        )

    if isinstance(graded, BaseException):
        return (None, GRADED_SOURCE_UNRESOLVED, str(graded) or repr(graded))

    if not isinstance(graded, dict):
        return (
            None,
            GRADED_SOURCE_UNRESOLVED,
            f"the graded-week record has an unrecognised type "
            f"({type(graded).__name__}); the detector expects the mapping "
            "data.graded_weeks.graded_weeks_record returns, or the "
            "GradedWeeksUnavailable it raised.",
        )

    weeks = graded.get("weeks")
    if not graded.get("resolved") or weeks is None:
        stated = graded.get("reason")
        return (
            None,
            GRADED_SOURCE_UNRESOLVED,
            str(stated)
            if stated
            else "the graded-week record reports itself unresolved and named no reason.",
        )

    return (
        sorted(int(week) for week in weeks),
        str(graded.get("source") or GRADED_SOURCE_UNSTATED),
        None,
    )


def _week_number(bucket_key: str) -> int | None:
    """The integer week a digest bucket key names, or ``None`` when it names no week.

    ``data.upstream_live.week_digests`` keys its buckets with ``str(int(week))`` for a
    dataset that has a ``week`` column, and with the single ``__no_week_column__`` sentinel
    for one that does not. The sentinel is the case this function exists to mark: it cannot
    be intersected with a graded week set, and pretending it can -- in either direction --
    is a wrong answer. The sentinel is recognised by FAILING to parse rather than by being
    compared against an imported constant, so this module keeps no import of
    ``data.upstream_live`` and stays free of pandas.
    """
    try:
        return int(bucket_key)
    except (TypeError, ValueError):
        return None


def _verdict(
    *,
    dataset: str,
    season: int,
    current_entry: dict,
    event_class: RevisionEventClass,
    diff: dict | None,
    correction_owed: bool | None,
    correction_owed_scope: dict | None,
    graded_weeks: list[int] | None,
    graded_weeks_source: str,
    reason: str,
) -> dict:
    """Assemble one verdict. THE ONLY PLACE A SEVERITY IS EVER SET.

    ``severity`` is ``DEFAULT_SEVERITY[event_class]`` and nothing else, on every branch. No
    caller and no branch above may pass one in, so a severity cannot drift out of the frozen
    vocabulary ``data/revision_events.py`` committed (T-32-34). The severity table is a
    CALIBRATION and may be retuned within a schema version; the event class is a FACT and may
    not. Keeping the two bound here is what lets Phase 35 retune the volume of the whole
    detector in one edit without rewriting any history of what was observed.

    ``graded_weeks`` and ``graded_weeks_source`` ride on EVERY verdict, including ``clean``
    and ``no_prior_capture``. A record can therefore never claim that no weeks were graded
    without also saying where it looked -- the same discipline the sealed probe's
    ``checked`` / ``expected`` pair enforces, and the thing that makes an empty answer
    auditable instead of merely plausible.
    """
    return {
        "verdict_schema_version": VERDICT_SCHEMA_VERSION,
        "live_revision_schema_version": LIVE_REVISION_SCHEMA_VERSION,
        "dataset": dataset,
        "season": int(season),
        "week": current_entry.get("week"),
        "sequence": current_entry.get("sequence"),
        "event_class": str(event_class),
        "severity": str(DEFAULT_SEVERITY[event_class]),
        "diff": diff,
        CORRECTION_OWED: correction_owed,
        CORRECTION_OWED_SCOPE: correction_owed_scope,
        "graded_weeks": graded_weeks,
        "graded_weeks_source": graded_weeks_source,
        "reason": reason,
    }


def detect_live_revision(
    *,
    dataset: str,
    season: int,
    current_entry: dict,
    prior_entry: dict | None = None,
    graded: object = None,
) -> dict:
    """Rule on one live capture against the one before it. Pure; mutates neither entry.

    Args:
        dataset: The upstream dataset this capture is of, recorded on the verdict.
        season: The live season, recorded on the verdict and on any owed-correction scope.
        current_entry: The capture entry just built, in the shape
            ``data.upstream_live.build_capture_entry`` produces.
        prior_entry: The most recent EARLIER capture entry of the same dataset, or ``None``
            when this is the first one.
        graded: The record ``data.graded_weeks.graded_weeks_record`` returned, or the
            :class:`data.graded_weeks.GradedWeeksUnavailable` it raised.

            IT IS A PARAMETER AND NOT AN IMPORT, for two reasons. It keeps this module pure
            and testable offline against synthetic graded sets, which is what D32-11 asks of
            the escalation branch. And it keeps the ONE seam Phase 34 repoints inside
            ``data/graded_weeks.py`` where D32-11 put it -- a second reader here would be a
            second place to forget when LDGR-01 relocates the store, and it would be forgotten
            inside a detector, where a silent failure looks exactly like "nothing to report".
            ``32-08`` is the caller that resolves it and passes it in.

    Returns:
        A mapping whose keys are exactly :data:`VERDICT_KEYS`.

    THE RULING, IN ORDER. The order is the substance, not a formatting choice:

    1. NO PRIOR CAPTURE -> ``no_prior_capture``. This is the planner's ruling on CONTEXT's
       open discretion item ("the detector's behaviour on the very FIRST 2026 capture, when
       there is no previous capture to diff against. Expected: an explicit
       ``no_prior_capture`` verdict, not a clean one"). A first capture and a capture that was
       COMPARED AND FOUND IDENTICAL are different facts. Calling the first one ``clean`` would
       put PITFALLS F2's ambiguity -- a dead detector and a healthy system as the same
       observable -- back for the whole opening week of the season, which is the week with
       the least other evidence available to contradict it.

    2. THE GRADED SET IS UNRESOLVED -> ``unknown``, with ``correction_owed`` exactly ``None``
       and the underlying failure text carried in ``reason``. Evaluated BEFORE the diff is
       ruled on, deliberately: a ruling made without knowing what was graded has exactly one
       possible error, and it is the quiet one. ``None`` and not ``False`` -- ``False`` is a
       claim that nothing is owed, which is a claim this branch is by definition unable to
       make.

    3. THE TWO ENTRIES WERE DIGESTED UNDER DIFFERENT ``week_digest_schema_version``
       STAMPS -> ``unknown``, ``correction_owed`` exactly ``None`` (WR-06). Two maps
       produced by different rules are not comparable, and reading their difference as
       movement would report a whole-season revision -- escalating to a CRITICAL with a
       FABRICATED obligation on every graded week -- on the run after any digest-shape
       bump. The diff is recorded and deliberately not ruled on, the same way branch 2
       records it. Evaluated before the diff is interpreted, and after branch 2 because
       "what was graded" is the more fundamental unknown of the two.

    4. NOT A REVISION -> ``clean``, nothing owed. In-season growth lands here; see
       :attr:`WeekDiff.is_revision`.

    5. A MOVED OR REMOVED WEEK IS IN THE GRADED SET -> ``live_revision_graded``,
       ``correction_owed`` ``True``, and a ``correction_owed_scope`` naming the season and
       the intersecting weeks as sorted ints. The scope is QUERYABLE DATA and not prose: a
       later phase must be able to select every capture entry whose ``correction_owed`` is
       true, and the weeks it owes on, without parsing a sentence.

       WHAT THIS PHASE DELIBERATELY DOES NOT DO. It does not build, write or schedule the
       correction block, and it never re-grades anything. ``.planning/research/
       ARCHITECTURE.md`` section 5.4 rules that a corrected score arriving after a bet was
       graded can NEVER re-grade it; the only honest handling is an appended correction
       block, and that block is Phase 34's to write. Phase 32 raises the obligation; Phase 34
       discharges it.

    6. A MOVED BUCKET CANNOT BE ATTRIBUTED TO A WEEK, AND SOMETHING IS GRADED -> ``unknown``,
       ``correction_owed`` ``None``. A dataset with no ``week`` column digests as one
       whole-frame bucket, so when that bucket moves the detector genuinely CANNOT say which
       weeks it touched. Ruling it ``live_revision`` would assert that no graded week moved,
       which is not something this branch knows; ruling it ``live_revision_graded`` would
       assert one did, which it also does not know, and would fire a CRITICAL every week a
       whole-frame dataset churns -- PITFALLS B3's alert-fatigue failure. ``unknown`` is the
       claim that is actually true, it is louder than the ordinary case (F2) and quieter than
       a confirmed one, and ``correction_owed`` of ``None`` says the obligation is undecided
       rather than absent. With NOTHING graded there is no uncertainty to record and the
       ordinary case applies. Step 5 is checked first throughout: a definite finding always
       outranks an uncertain one.

    7. OTHERWISE -> ``live_revision``, nothing owed. The ordinary in-season restatement,
       kept quiet on purpose so that the five loud branches above stay worth reading.
    """
    graded_weeks, graded_source, graded_failure = _resolve_graded(graded)

    if prior_entry is None:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.NO_PRIOR_CAPTURE,
            diff=None,
            correction_owed=False,
            correction_owed_scope=None,
            graded_weeks=graded_weeks,
            graded_weeks_source=graded_source,
            reason=(
                f"there was no previous capture of {dataset} for season {season} to diff "
                "against, so nothing can be said about movement. This is NOT a clean "
                "verdict: 'nothing to compare' and 'compared and found no change' are "
                "different claims, and a season of entries has to be able to tell them "
                "apart after the fact."
            ),
        )

    diff = compare_week_digests(
        prior_entry.get("week_digests") or {},
        current_entry.get("week_digests") or {},
    )
    diff_record = as_record(diff)

    if graded_failure is not None:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.UNKNOWN,
            # The diff is RECORDED but deliberately NOT RULED ON. It cost nothing to compute
            # and a later reader may well want it; what this branch refuses to do is draw a
            # severity from it without knowing what was graded.
            diff=diff_record,
            correction_owed=None,
            correction_owed_scope=None,
            graded_weeks=None,
            graded_weeks_source=graded_source,
            reason=(
                f"the graded-week state for season {season} could not be resolved, so "
                f"whether this {dataset} capture moved an already-graded week is unknown "
                f"and no severity can honestly be drawn from the diff. Underlying failure: "
                f"{graded_failure}"
            ),
        )

    # WR-06. THE TWO ENTRIES MUST HAVE BEEN DIGESTED UNDER THE SAME RULE, and until now
    # nothing checked. ``data.upstream_live.WEEK_DIGEST_SCHEMA_VERSION`` exists precisely
    # because "changing it mid-season would mean earlier revision verdicts were computed
    # under a different rule, so the season's revision history would stop being comparable
    # end to end" -- but the stamp was written for readers and never read by the one
    # component that depends on it. A bump (Phase 32's CR/WR fix cycle performed exactly
    # one) would make the first capture after it diff a v2 map against a v1 map: every
    # shared week's digest differs, the detector reports a WHOLE-SEASON revision, and it
    # escalates to a CRITICAL with ``correction_owed`` True for every graded week -- a
    # fabricated obligation, and PITFALLS B3's alert-fatigue failure delivered in one run.
    #
    # Checked AFTER the graded branch and BEFORE the diff is interpreted: the diff is
    # RECORDED (it cost nothing and a later reader may want it) but no severity is drawn
    # from it, which is the same shape the graded-failure branch above uses for the same
    # reason. ``correction_owed`` is None and never False -- False asserts nothing is
    # owed, which is a claim this branch cannot make.
    prior_version = prior_entry.get("week_digest_schema_version")
    current_version = current_entry.get("week_digest_schema_version")
    if prior_version != current_version:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.UNKNOWN,
            diff=diff_record,
            correction_owed=None,
            correction_owed_scope=None,
            graded_weeks=graded_weeks,
            graded_weeks_source=graded_source,
            reason=(
                f"the previous {dataset} capture was digested under week-digest schema "
                f"{prior_version!r} and this one under {current_version!r}, so the two "
                "maps were produced by different rules and are not comparable. No "
                "movement can honestly be read from their difference: every shared week "
                "would appear to have moved because the DIGEST changed, not because the "
                "bytes did. The diff is recorded but deliberately NOT ruled on. A "
                "like-for-like comparison resumes at the next capture, once both sides "
                "were digested under the current rule."
            ),
        )

    if not diff.is_revision:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.CLEAN,
            diff=diff_record,
            correction_owed=False,
            correction_owed_scope=None,
            graded_weeks=graded_weeks,
            graded_weeks_source=graded_source,
            reason=(
                f"the {dataset} capture was compared week by week against the previous one "
                "and no week present in both moved or disappeared."
                + (
                    f" {len(diff.weeks_added)} new week bucket(s) appeared "
                    f"({', '.join(diff.weeks_added)}), which is ordinary in-season growth "
                    "and is not a revision."
                    if diff.weeks_added
                    else ""
                )
            ),
        )

    moved = sorted(set(diff.weeks_changed) | set(diff.weeks_removed))
    graded_set = set(graded_weeks or [])
    intersecting = sorted(
        {
            week
            for week in (_week_number(key) for key in moved)
            if week is not None and week in graded_set
        }
    )

    if intersecting:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.LIVE_REVISION_GRADED,
            diff=diff_record,
            correction_owed=True,
            correction_owed_scope={
                "season": int(season),
                "weeks": intersecting,
                "dataset": dataset,
                "reason": (
                    f"upstream restated or withdrew {dataset} data for season {season} "
                    f"week(s) {', '.join(str(week) for week in intersecting)}, whose bets "
                    "were already graded, so the settled record was computed from bytes "
                    "that upstream no longer publishes."
                ),
                "discharged_by": CORRECTION_DISCHARGED_BY,
            },
            graded_weeks=graded_weeks,
            graded_weeks_source=graded_source,
            reason=(
                f"the {dataset} capture moved or dropped week(s) "
                f"{', '.join(str(week) for week in intersecting)} of season {season}, "
                "which are ALREADY GRADED. A corrected score arriving after a bet was "
                "graded can never re-grade it, so a ledger correction block is OWED and is "
                f"recorded here as data for the {CORRECTION_DISCHARGED_BY} to discharge."
            ),
        )

    unattributable = [key for key in moved if _week_number(key) is None]
    if unattributable and graded_set:
        return _verdict(
            dataset=dataset,
            season=season,
            current_entry=current_entry,
            event_class=RevisionEventClass.UNKNOWN,
            diff=diff_record,
            correction_owed=None,
            correction_owed_scope=None,
            graded_weeks=graded_weeks,
            graded_weeks_source=graded_source,
            reason=(
                f"the {dataset} capture moved bucket(s) {', '.join(unattributable)}, which "
                "belong to no week, so the detector cannot say whether an already-graded "
                f"week of season {season} was touched. "
                f"{len(graded_set)} week(s) are graded, so the question is live. Recorded "
                "as undecided rather than answered either way: claiming no graded week "
                "moved would be the silent downgrade, and claiming one did would fire a "
                "confirmed finding this branch has no basis for."
            ),
        )

    return _verdict(
        dataset=dataset,
        season=season,
        current_entry=current_entry,
        event_class=RevisionEventClass.LIVE_REVISION,
        diff=diff_record,
        correction_owed=False,
        correction_owed_scope=None,
        graded_weeks=graded_weeks,
        graded_weeks_source=graded_source,
        reason=(
            f"the {dataset} capture restated week(s) {', '.join(moved)} of season "
            f"{season}, none of which has been graded. nflverse restates the running "
            "week's data routinely, so this is the ordinary in-season case and is recorded "
            "at the quiet volume on purpose."
        ),
    )


def verdict_severity_rank(verdict: dict) -> int:
    """Rank *verdict*'s severity on the frozen ladder -- the ONLY sanctioned comparison.

    A caller that needs to know which of several verdicts is loudest (``32-08`` picks one
    exit code for a run that ruled on several datasets; Phase 35 routes by volume) must not
    compare the ``severity`` strings directly. :class:`data.revision_events.RevisionSeverity`
    is a ``StrEnum``, so ``"critical" < "informational"`` is a perfectly legal string
    comparison that silently returns the WRONG order -- alphabetical, not loudness. Routing
    a CRITICAL as though it were the quietest is precisely the failure this module spends
    every branch above avoiding, and it would be a one-character mistake at a call site.

    ``severity_rank`` refuses anything outside the vocabulary, so a verdict carrying a
    severity that drifted raises here rather than sorting silently into the wrong position.
    """
    return severity_rank(RevisionSeverity(verdict["severity"]))
