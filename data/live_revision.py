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

__all__ = [
    "LIVE_REVISION_SCHEMA_VERSION",
    "WeekDiff",
    "as_record",
    "compare_week_digests",
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
# RED-phase skeleton for Task 2. The ruling lands in the GREEN commit.
# ---------------------------------------------------------------------------

VERDICT_KEYS: tuple[str, ...] = ()


def detect_live_revision(
    *,
    dataset: str,
    season: int,
    current_entry: dict,
    prior_entry: dict | None = None,
    graded: object = None,
) -> dict:
    """Rule on one live capture. Not yet implemented (RED phase)."""
    return {}
