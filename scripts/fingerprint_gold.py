"""Fingerprint the gold feature matrices per column, per season (D-Q5).

A full gold rebuild reaches nflreadpy LIVE (``features/team_form.py:35,118``;
``features/qb_tracking.py:710,776``) with no cache configured, so an upstream
play-by-play or depth-chart revision lands in the same artifact as whatever
change the rebuild was actually run for. Without a control, that upstream drift
is silently absorbed and misattributed.

This tool turns that invisible confound into a recorded observation. Run it
BEFORE a rebuild, run it again AFTER, then ``--compare BEFORE AFTER`` to get the
exact list of columns that moved and the seasons in which each moved.

It is strictly read-only with respect to ``data/`` -- the JSON output must be
written somewhere else (the quick task keeps it under the phase's gitignored
``artifacts/`` directory).
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from conf.settings import get_settings

GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")


def _column_bytes(series: pd.Series) -> bytes:
    """Return a deterministic byte encoding of *series* values.

    Numeric columns hash their raw IEEE-754 / integer representation, which is
    exact; anything else falls back to a repr-joined string so the encoding is
    still stable across runs.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype="uint8").tobytes()
    if pd.api.types.is_integer_dtype(series):
        return series.to_numpy(dtype="int64").tobytes()
    if pd.api.types.is_float_dtype(series):
        return series.to_numpy(dtype="float64").tobytes()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").to_numpy(dtype="int64").tobytes()
    return "\x1f".join(repr(value) for value in series.to_numpy()).encode("utf-8")


_DISCRETE_INDICATOR_PREDICATE = None


def _discrete_indicator_predicate():
    """Return ``FeatureMatrixBuilder._is_discrete_indicator``, imported lazily and once.

    The predicate is imported rather than re-expressed. CR-02's exemption is
    deliberately a VALUE test, not a name test (``scripts/build_features.py``
    comment at the exemption), so a second copy of the rule here would be free to
    drift away from the one the rebuild actually applies -- the 29-06 second-list
    failure mode in a different costume (D30-02). The import is deferred because
    ``scripts.build_features`` pulls the whole feature stack and this module is
    also used as a plain fingerprint reader.
    """
    global _DISCRETE_INDICATOR_PREDICATE
    if _DISCRETE_INDICATOR_PREDICATE is None:
        from scripts.build_features import FeatureMatrixBuilder

        _DISCRETE_INDICATOR_PREDICATE = FeatureMatrixBuilder._is_discrete_indicator
    return _DISCRETE_INDICATOR_PREDICATE


def _column_meta(series: pd.Series) -> dict:
    """Return the per-column facts an attribution needs to say WHY a column moved.

    ``_column_bytes`` already makes a dtype change or a null-count change move the
    per-season hash, so the adjacency is satisfied today -- but only IMPLICITLY, and
    an implicit signal cannot be reported. Recording the three facts explicitly lets
    ``attribute_rung`` distinguish "this column's values moved" from "this column's
    storage moved", and lets rung 1 attribute a move to CR-02 without consulting a
    name list.

    Discreteness is evaluated ONCE over the whole column, never per season: a column
    that is continuous overall but happens to be constant within one season would be
    misclassified as discrete for that season.
    """
    is_discrete = _discrete_indicator_predicate()
    try:
        discrete = bool(is_discrete(series))
    except (TypeError, ValueError):
        # Unhashable or non-comparable values (never numeric features) are not
        # indicators. Record the fact rather than letting the fingerprint fail.
        discrete = False
    return {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "discrete_indicator": discrete,
    }


def fingerprint_matrix(df: pd.DataFrame) -> dict:
    """Return per-column, per-season hashes plus shape metadata for *df*.

    Each cell hashes the column's values ordered by ``game_id`` within the
    season, so a row-order change alone never registers as drift.

    The ``columns`` map and every existing key are unchanged, so fingerprint
    documents written before Plan 30-04 stay comparable. The growth is additive:
    a sibling ``column_meta`` map carrying each column's dtype, null count and
    CR-02 discreteness.
    """
    seasons = sorted(int(season) for season in df["season"].dropna().unique())
    columns: dict[str, dict[str, str]] = {}

    ordered = df.sort_values("game_id")
    for season in seasons:
        subset = ordered[ordered["season"] == season]
        for column in subset.columns:
            digest = hashlib.sha256(_column_bytes(subset[column])).hexdigest()[:16]
            columns.setdefault(column, {})[str(season)] = digest

    return {
        "rows": len(df),
        "width": int(df.shape[1]),
        "seasons": seasons,
        "rows_per_season": {
            str(season): int((df["season"] == season).sum()) for season in seasons
        },
        "columns": columns,
        "column_meta": {
            str(column): _column_meta(series) for column, series in df.items()
        },
    }


def fingerprint_gold(base_path: Path | None = None) -> dict:
    """Fingerprint every gold feature matrix."""
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    result: dict[str, dict] = {}
    for matrix in GOLD_MATRICES:
        path = root / "gold" / f"{matrix}.parquet"
        if not path.exists():
            result[matrix] = {"missing": True}
            continue
        result[matrix] = fingerprint_matrix(pd.read_parquet(path, engine="pyarrow"))
    return result


def compare_fingerprints(before: dict, after: dict) -> dict:
    """Return, per matrix, the columns whose hash moved and in which seasons.

    A column counts as MOVED when its values moved, OR when its dtype moved, OR
    when its null count moved. The last two matter because a column can be
    value-identical and still be a different artifact -- an ``int64`` that became
    a ``float64``, or a column that gained a NaN somewhere the season hashes
    happen not to separate. Such a column appears in ``columns_changed`` with an
    EMPTY season list, and ``column_details[column]["reasons"]`` says which of
    the three moved.

    Documents written before Plan 30-04 carry no ``column_meta``, so the dtype and
    null-count comparisons are simply absent for them and the report is exactly
    what it always was.
    """
    report: dict[str, dict] = {}
    for matrix in GOLD_MATRICES:
        b = before.get(matrix, {})
        a = after.get(matrix, {})
        b_cols = b.get("columns", {})
        a_cols = a.get("columns", {})
        b_meta = b.get("column_meta", {})
        a_meta = a.get("column_meta", {})

        changed: dict[str, list[str]] = {}
        details: dict[str, dict] = {}
        for column in sorted(set(b_cols) & set(a_cols)):
            seasons = [
                season
                for season in sorted(set(b_cols[column]) | set(a_cols[column]))
                if b_cols[column].get(season) != a_cols[column].get(season)
            ]
            bm = b_meta.get(column, {})
            am = a_meta.get(column, {})

            reasons: list[str] = []
            if seasons:
                reasons.append("values")
            if bm and am:
                if bm.get("dtype") != am.get("dtype"):
                    reasons.append("dtype")
                if bm.get("null_count") != am.get("null_count"):
                    reasons.append("null_count")

            if not reasons:
                continue

            changed[column] = seasons
            details[column] = {
                "seasons": seasons,
                "reasons": reasons,
                "dtype_before": bm.get("dtype"),
                "dtype_after": am.get("dtype"),
                "null_count_before": bm.get("null_count"),
                "null_count_after": am.get("null_count"),
                "discrete_indicator_before": bm.get("discrete_indicator"),
                "discrete_indicator_after": am.get("discrete_indicator"),
            }

        report[matrix] = {
            "width_before": b.get("width"),
            "width_after": a.get("width"),
            "rows_before": b.get("rows"),
            "rows_after": a.get("rows"),
            "rows_per_season_before": b.get("rows_per_season", {}),
            "rows_per_season_after": a.get("rows_per_season", {}),
            "columns_added": sorted(set(a_cols) - set(b_cols)),
            "columns_removed": sorted(set(b_cols) - set(a_cols)),
            "columns_changed": changed,
            "column_details": details,
        }
    return report


# ---------------------------------------------------------------------------
# Per-rung cause attribution (SPEC R1, Plan 30-04)
# ---------------------------------------------------------------------------

# The ONE named cause each rung of the D30-17 rebuild ladder is allowed to move
# columns for. A moved column that cannot be attributed to its rung's cause FAILS
# the step -- the whole point of running four separate rebuilds instead of one.
RUNG_CAUSES: dict[int, str] = {
    1: "CR-02",
    2: "WR-06",
    3: "line_movement drop",
    4: "N-01",
}

# The per-build clock stamp, and the ONLY column exempted by name anywhere in this
# module. It is different IN KIND from every other column in a gold matrix:
# ``scripts/build_features.py`` writes ``datetime.now(UTC)`` into it once per build,
# so it takes exactly one distinct value per build and MUST move on every rebuild by
# construction. It records WHEN the frame was built; it measures nothing about the
# games in the frame, and no model consumes it (``models/temporal.py`` excludes it).
#
# Two consequences follow, and both are why it gets its own category rather than a
# tolerance:
#
# 1. Filing it under the upstream-drift candidate cause would be FALSE. That note
#    tells a reader to go find the nflreadpy revision that moved the column; for a
#    clock there is none, and the reader burns the search anyway.
# 2. Counting it as a moved value makes rung 3's empty-changed-set criterion
#    structurally UNSATISFIABLE -- no correct rebuild can ever satisfy it, so the
#    criterion stops discriminating between a right rebuild and a wrong one.
#
# Plan 30-06 already reported it this way at rung 1, by hand, in its SUMMARY. Owner
# ruling D30-OWNER-08 makes the instrument do it, at EVERY rung. This exempts ONE
# named build artifact; every other column is judged exactly as before.
BUILD_CLOCK_COLUMNS = ("feature_timestamp",)

# Rungs whose failures may legitimately be upstream drift rather than a wrong fix.
# Rung 4 is DELIBERATELY excluded: SPEC R2 makes an unexplained 2021-2024 move a
# hard blocker, and offering an escape there would let the phase talk itself past
# the one control it exists to run.
_UPSTREAM_ESCAPE_RUNGS = (1, 2, 3)

# Rungs at which a PROVEN value-preserving dtype change may be attributed. Rung 4 is
# excluded for the same reason it is excluded from the upstream-drift escape: SPEC R2
# makes an unexplained 2021-2024 move a hard blocker, and the one control this phase
# rests on does not get a new way to be talked past. The exclusion is conservative
# rather than necessary -- a reproduced per-season hash is a PROOF, not a hypothesis --
# but widening rung 4's criterion is not this change's business (D30-OWNER-08).
_DTYPE_PROOF_RUNGS = (1, 2, 3)

# The reasons a column may have moved and still be a candidate for the dtype proof.
# A null-count move is a real change in the data and is never storage-only, so a
# column carrying it is never offered to the proof.
_VALUE_PRESERVING_REASONS = frozenset({"values", "dtype"})

_UPSTREAM_DRIFT_NOTE = (
    "CANDIDATE CAUSE: upstream drift. A full gold rebuild reads nflreadpy LIVE with "
    "no cache configured, so a play-by-play or depth-chart revision published between "
    "two rungs lands in this rung's artifact. Check the nflreadpy revision date for the "
    "affected column BEFORE concluding the {cause} fix is wrong."
)


# The single case convention every column set is normalized to before comparison.
# A fingerprint document is JSON, whose key ordering carries no meaning, and gold
# column names are lower-case by construction. Normalizing BOTH sides means a
# verdict cannot differ between two runs over identical data, and an upstream
# rename that changes only capitalization is reported as a RENAME rather than as a
# simultaneous add and remove.
def _canonical(name: str) -> str:
    """Return *name* under the module's one case convention (lower-case)."""
    return name.lower()


def _canonical_map(names) -> dict[str, str]:
    """Map canonical name -> the original spelling, for a list of column names."""
    return {_canonical(name): name for name in names}


def _line_movement_columns(column_names) -> list[str]:
    """Return the ``line_movement`` family present in *column_names*, sorted.

    Derived from ``backtest.signal_lift.group_columns`` -- the ONE group registry.
    A second list of the family's names inside this module is precisely the 29-06
    failure mode that D30-02 exists to prevent, so there is none.
    """
    from backtest.signal_lift import group_columns

    frame = pd.DataFrame(columns=pd.Index(list(column_names)))
    return group_columns(frame, "line_movement")


def _per_season_digests(
    frame: pd.DataFrame, column: str, recode_to: str | None = None
) -> dict[str, str]:
    """Return *column*'s per-season digests under ``fingerprint_matrix``'s exact rule.

    The row ordering, the season grouping and the byte encoding are the SAME ones that
    wrote the documents being compared. Re-hashing under any other rule would produce
    digests that prove nothing about them.

    With *recode_to*, the column is cast to that dtype first -- which is what makes a
    dtype change checkable at all: a digest cannot be re-encoded, only values can.
    """
    ordered = frame.sort_values("game_id")
    series = cast("pd.Series", ordered[column])
    if recode_to is not None:
        series = series.astype(recode_to)
    seasons = ordered["season"].to_numpy()
    return {
        str(season): hashlib.sha256(
            _column_bytes(cast("pd.Series", series[seasons == season]))
        ).hexdigest()[:16]
        for season in sorted(int(value) for value in frame["season"].dropna().unique())
    }


def _gold_frame_loader(base_path: Path | None = None):
    """Return a lazy, cached ``matrix -> DataFrame | None`` reader over live gold.

    The dtype proof needs the AFTER frame's VALUES, and no fingerprint document
    carries them. Reading them is strictly READ-ONLY with respect to ``data/``, and
    the proof's first step requires what it read to reproduce the AFTER document's own
    digests -- so a frame that has moved on since the document was written proves
    nothing, rather than proving the wrong thing.
    """
    cache: dict[str, pd.DataFrame | None] = {}

    def load(matrix: str) -> pd.DataFrame | None:
        if matrix not in cache:
            root = (
                Path(base_path)
                if base_path is not None
                else Path(get_settings().config.data.root_path)
            )
            path = root / "gold" / f"{matrix}.parquet"
            cache[matrix] = (
                pd.read_parquet(path, engine="pyarrow") if path.exists() else None
            )
        return cache[matrix]

    return load


def _prove_value_preserving_dtype(
    matrix: str,
    column: str,
    meta: dict,
    before: dict | None,
    after: dict | None,
    frame_loader,
) -> bool:
    """Return True only when *column*'s dtype move is PROVEN to preserve every value.

    "It is only a dtype change" is an assertion, and an assertion is not evidence.
    The attribution has to be EARNED, in three steps, and any one of them failing
    leaves the column exactly where it was -- a moved column, judged by its rung's
    ordinary criterion:

    1. **Identity.** Re-hashing the loaded frame's column AS IT STANDS must reproduce
       the AFTER document's per-season digests. Without this the proof could be run
       against some other frame that merely has a column of the same name.
    2. **Losslessness.** Re-encode to the prior dtype, then encode BACK, and require
       the frame's own values to return. A truncating cast (``float64`` 1.5 ->
       ``int32`` 1) can reproduce a prior hash while the value genuinely moved; this
       is the hole that closes.
    3. **Reproduction.** The PRIOR per-season digests must return EXACTLY, in every
       season. This is the proof; steps 1 and 2 only make it mean what it says.

    Plan 30-07 ran exactly this argument by hand for ``home_win`` and reproduced the
    rung-2 digest in 24 of 24 seasons (D30-DEFER-11). Nothing here is special-cased to
    that column, or to any column: the rule is about proof, not about a name. A dtype
    pair whose re-encode cannot even be attempted is not a forgiven change -- it is one
    the instrument cannot check, and it stays a failure.
    """
    dtype_before = meta.get("dtype_before")
    dtype_after = meta.get("dtype_after")
    if not dtype_before or not dtype_after or dtype_before == dtype_after:
        return False
    if not set(meta.get("reasons") or []) <= _VALUE_PRESERVING_REASONS:
        return False
    if before is None or after is None or frame_loader is None:
        return False

    before_names = _canonical_map(before.get(matrix, {}).get("columns", {}))
    after_names = _canonical_map(after.get(matrix, {}).get("columns", {}))
    if column not in before_names or column not in after_names:
        return False
    before_digests = before[matrix]["columns"][before_names[column]]
    after_digests = after[matrix]["columns"][after_names[column]]

    try:
        frame = frame_loader(matrix)
    except (OSError, ValueError):
        return False
    if frame is None or not {"game_id", "season"} <= set(frame.columns):
        return False
    live_names = _canonical_map(frame.columns)
    if column not in live_names:
        return False
    name = live_names[column]

    try:
        if _per_season_digests(frame, name) != after_digests:
            return False
        recoded = frame[name].astype(dtype_before)
        if not recoded.astype(dtype_after).equals(frame[name]):
            return False
        if _per_season_digests(frame, name, recode_to=dtype_before) != before_digests:
            return False
    except (TypeError, ValueError, OverflowError, KeyError):
        return False
    return True


def _expected_signature(rung: int, before: dict | None = None) -> dict:
    """Return the predicted ``compare_fingerprints`` diff shape for *rung*.

    When *before* (the pre-rung fingerprint document) is supplied, rung 3's expected
    removed-set is DERIVED per matrix from that document's column list rather than
    described. That is the difference between "every removed column looks like a
    line-movement column" and "the removed set IS the line-movement family".
    """
    if rung not in RUNG_CAUSES:
        msg = f"Unknown rung {rung!r}. Must be one of {sorted(RUNG_CAUSES)}."
        raise ValueError(msg)

    signature = {
        "rung": rung,
        "cause": RUNG_CAUSES[rung],
        "columns_added": "empty",
        "columns_removed": "empty",
        "columns_changed": "",
        "rows": "unchanged",
        "width": "unchanged",
    }

    if rung == 1:
        signature["columns_changed"] = (
            "restricted to columns whose values are indicator levels "
            "(FeatureMatrixBuilder._is_discrete_indicator is True)"
        )
    elif rung == 2:
        signature["columns_changed"] = (
            "any column whose values were imputed or clipped; seasons broad, "
            "because the prior-seasons-only bounds move everywhere"
        )
    elif rung == 3:
        signature["columns_changed"] = (
            "EMPTY -- dropping columns must not move a surviving value"
        )
        signature["width"] = "reduced by exactly the number of removed columns"
        if before is not None:
            signature["columns_removed"] = {
                matrix: _line_movement_columns(
                    before.get(matrix, {}).get("columns", {})
                )
                for matrix in GOLD_MATRICES
            }
        else:
            signature["columns_removed"] = (
                "exactly the line_movement family, identical in all three matrices"
            )
    else:
        signature["columns_changed"] = (
            "every changed column's season list equals exactly the seasons that "
            "gained rows; any other season BLOCKS the phase (SPEC R2)"
        )
        signature["rows"] = "strictly increased"

    return signature


def _matrix_verdict() -> dict:
    """Return an empty per-matrix verdict slot."""
    return {
        "ok": True,
        "attributed": [],
        "unattributed": [],
        # The per-build clock, reported as its own category so it is neither an
        # explanation for anything nor evidence of anything.
        "build_clock": [],
        # Dtype moves that EARNED their attribution by reproducing the prior
        # per-season hash, each carrying the two dtypes by name.
        "value_preserving_dtype": [],
        # Dtype moves offered to the proof that did NOT reproduce. Recorded so the
        # verdict shows the proof was attempted; these columns stay in the changed
        # set and are judged by the rung's ordinary criterion.
        "dtype_proof_failed": [],
        "renamed_case_only": [],
        "failures": [],
    }


def _normalized_diff(detail: dict) -> dict:
    """Return one matrix's diff with every column set canonicalized and deduped."""
    added = _canonical_map(detail.get("columns_added", []))
    removed = _canonical_map(detail.get("columns_removed", []))

    # A name present on both sides under the case convention is a RENAME, not a
    # simultaneous add and remove.
    renames = sorted(set(added) & set(removed))

    changed_raw = detail.get("columns_changed", {})
    details_raw = detail.get("column_details")

    changed = {_canonical(name): list(seasons) for name, seasons in changed_raw.items()}
    details = (
        None
        if details_raw is None
        else {_canonical(name): value for name, value in details_raw.items()}
    )

    return {
        "added": sorted(name for name in added if name not in renames),
        "removed": sorted(name for name in removed if name not in renames),
        "renames": [[removed[name], added[name]] for name in renames],
        "changed": changed,
        "details": details,
        # Filled by _split_value_preserving_dtype. A column here MOVED -- it just
        # moved provably in storage only, so it still counts as the rebuild having
        # done something, and never as an unexplained value change.
        "dtype_preserved": [],
    }


def _split_build_clock(diff: dict, verdict: dict) -> None:
    """Move BUILD_CLOCK_COLUMNS out of the changed set into their own category."""
    clock = {_canonical(name) for name in BUILD_CLOCK_COLUMNS}
    for column in sorted(name for name in diff["changed"] if name in clock):
        verdict["build_clock"].append(column)
        del diff["changed"][column]


def _split_value_preserving_dtype(
    matrix: str,
    diff: dict,
    verdict: dict,
    before: dict | None,
    after: dict | None,
    frame_loader,
) -> None:
    """Move PROVEN value-preserving dtype changes out of the changed set."""
    if diff["details"] is None:
        return
    for column in sorted(diff["changed"]):
        meta = diff["details"].get(column) or {}
        if not meta.get("dtype_before") or meta.get("dtype_before") == meta.get(
            "dtype_after"
        ):
            continue
        if _prove_value_preserving_dtype(
            matrix, column, meta, before, after, frame_loader
        ):
            verdict["value_preserving_dtype"].append(
                {
                    "column": column,
                    "dtype_before": meta["dtype_before"],
                    "dtype_after": meta["dtype_after"],
                }
            )
            diff["dtype_preserved"].append(column)
            del diff["changed"][column]
        else:
            verdict["dtype_proof_failed"].append(column)


def _grown_seasons(detail: dict) -> list[str]:
    """Return the seasons whose row count increased, sorted."""
    before = detail.get("rows_per_season_before") or {}
    after = detail.get("rows_per_season_after") or {}
    return sorted(
        season
        for season in set(before) | set(after)
        if int(after.get(season, 0)) > int(before.get(season, 0))
    )


def attribute_rung(
    report: dict,
    rung: int,
    before: dict | None = None,
    after: dict | None = None,
    frame_loader=None,
) -> dict:
    """Attribute every moved column in *report* to *rung*'s one named cause.

    Returns a structured VERDICT rather than raising, so the caller decides
    severity. The verdict carries:

    - ``ok``      -- False when anything at all is unexplained. An unattributed
                     moved column FAILS the step (SPEC R1); so does a diff that
                     moved nothing, because WR-06 plus CR-02 plus a fifteen-column
                     drop must move something and an empty diff means the rebuild
                     did not do what it claimed.
    - ``blocking`` -- True only for a rung-4 anomaly (SPEC R2) and for a rung-3
                     non-empty changed set. At rungs 1-3 an unattributed column is
                     a FINDING: it may be upstream nflreadpy drift, and every
                     message at those rungs says so.

    Two categories are split out of the changed set before any rung criterion sees
    it, and each is reported in its own verdict slot rather than silently forgiven:

    - ``build_clock`` -- see ``BUILD_CLOCK_COLUMNS``. A per-build clock moves on every
      rebuild by construction, so it is neither an explanation nor evidence.
    - ``value_preserving_dtype`` -- a dtype move that reproduced the PRIOR per-season
      hash exactly under re-encoding. This requires *after* and *frame_loader*,
      because no fingerprint document carries the values a re-encode needs; without
      them the proof cannot run and the column stays in the changed set. That
      fail-closed default is deliberate: an unverifiable change is not a forgiven one.

    The comparison is deterministic with respect to both column ordering and case:
    every column set is normalized to a sorted set under ``_canonical`` on BOTH
    sides before anything is compared, and the emitted attributed / unattributed
    sets are canonical and sorted. A verdict that changed with JSON key order would
    differ between two runs over identical data.
    """
    signature = _expected_signature(rung, before=before)
    cause = RUNG_CAUSES[rung]
    upstream = (
        " " + _UPSTREAM_DRIFT_NOTE.format(cause=cause)
        if rung in _UPSTREAM_ESCAPE_RUNGS
        else ""
    )

    # The STRONG expected removed-set: derived per matrix from the pre-drop document
    # by _expected_signature. It was already being computed and recorded in the
    # signature, and was then never handed to the matrix judge -- which is what made
    # rung 3's partial-drop arm vacuous (D30-DEFER-12). It is handed over now.
    signature_removed = signature.get("columns_removed")
    derived_removed = signature_removed if isinstance(signature_removed, dict) else None

    matrices: dict[str, dict] = {}
    failures: list[str] = []
    blocking = False

    for matrix in sorted(report):
        detail = report[matrix]
        verdict = _matrix_verdict()
        matrices[matrix] = verdict

        def fail(message: str, verdict: dict = verdict, matrix: str = matrix) -> None:
            verdict["ok"] = False
            verdict["failures"].append(message + upstream)
            failures.append(f"{matrix}: {message}{upstream}")

        if detail.get("width_before") is None or detail.get("width_after") is None:
            fail(
                f"matrix {matrix} is absent from one or both fingerprint documents, so "
                f"rung {rung} cannot be attributed at all"
            )
            continue

        diff = _normalized_diff(detail)
        _split_build_clock(diff, verdict)
        if rung in _DTYPE_PROOF_RUNGS:
            _split_value_preserving_dtype(
                matrix, diff, verdict, before, after, frame_loader
            )

        for original_before, original_after in diff["renames"]:
            verdict["renamed_case_only"].append([original_before, original_after])
            fail(
                f"case-only rename '{original_before}' -> '{original_after}'. The column "
                "survived, but its spelling moved, and nothing in this rung's cause "
                "renames a column"
            )

        blocking |= _attribute_one_matrix(
            rung,
            detail,
            diff,
            verdict,
            fail,
            expected_removed=derived_removed.get(matrix)
            if derived_removed is not None
            else None,
        )

    ok = all(verdict["ok"] for verdict in matrices.values())
    return {
        "rung": rung,
        "cause": cause,
        "ok": ok,
        "blocking": bool(blocking),
        "signature": signature,
        "matrices": matrices,
        "failures": failures,
    }


def _attribute_one_matrix(
    rung, detail, diff, verdict, fail, expected_removed=None
) -> bool:
    """Apply *rung*'s predicted signature to one matrix. Returns whether it blocks."""
    cause = RUNG_CAUSES[rung]
    width_before = detail["width_before"]
    width_after = detail["width_after"]
    rows_before = detail.get("rows_before")
    rows_after = detail.get("rows_after")
    blocking = False

    diff_is_empty = (
        not diff["added"]
        and not diff["removed"]
        and not diff["renames"]
        and not diff["changed"]
        and not diff["dtype_preserved"]
        and rows_before == rows_after
    )
    if diff_is_empty:
        fail(
            f"rung {rung} ({cause}) moved no column and changed no row count. An empty "
            "diff means the rebuild did not do what it claimed"
        )
        if rung == 4:
            blocking = True

    if rung == 3:
        expected_removed = _rung3_expected_removed(diff, expected_removed)
        for column in diff["removed"]:
            if column in expected_removed:
                verdict["attributed"].append(column)
            else:
                verdict["unattributed"].append(column)
                fail(
                    f"column '{column}' was REMOVED at rung 3 but is not a member of the "
                    "line_movement family derived from backtest.signal_lift.group_columns"
                )
        for column in sorted(set(expected_removed) - set(diff["removed"])):
            fail(
                f"column '{column}' is a line_movement family member but was NOT removed "
                "at rung 3 -- the drop is PARTIAL"
            )
        for column in diff["added"]:
            fail(f"column '{column}' was ADDED at rung 3; the drop adds nothing")
        if width_after != width_before - len(diff["removed"]):
            fail(
                f"width moved {width_before} -> {width_after}, which is not "
                f"{width_before} minus the {len(diff['removed'])} removed columns"
            )
        if diff["changed"]:
            blocking = True
            moved = ", ".join(sorted(diff["changed"]))
            fail(
                "rung 3 moved surviving values, which dropping columns must never do: "
                f"{moved}. That means the dropped columns were participating in some "
                "whole-frame statistic"
            )
        if rows_before != rows_after:
            fail(f"rows moved {rows_before} -> {rows_after}; the drop adds no row")
        return blocking

    if rung == 4:
        blocking |= _attribute_rung4(detail, diff, verdict, fail)
        return blocking

    # Rungs 1 and 2 share their structural expectations: nothing added, nothing
    # removed, width unchanged. They differ only in which changed columns count as
    # explained.
    for column in diff["added"]:
        fail(f"column '{column}' was ADDED at rung {rung}; {cause} adds no column")
    for column in diff["removed"]:
        fail(f"column '{column}' was REMOVED at rung {rung}; {cause} removes no column")
    if width_before != width_after:
        fail(
            f"width moved {width_before} -> {width_after} at rung {rung}; {cause} "
            "changes no column count"
        )
    if rows_before != rows_after:
        fail(
            f"rows moved {rows_before} -> {rows_after} at rung {rung}; {cause} adds no row"
        )

    if rung == 1:
        _attribute_rung1(diff, verdict, fail)
    else:
        # WR-06 refits every imputation and winsorization bound on the strictly-prior
        # seasons, so ANY column that had a value imputed or clipped may move. The
        # rung's discipline is structural (nothing added, removed or resized) plus the
        # empty-diff refusal above, not a per-column allow-list.
        verdict["attributed"].extend(sorted(diff["changed"]))

    return blocking


def _rung3_expected_removed(diff: dict, derived: list[str] | None = None) -> list[str]:
    """Return the expected rung-3 removed set, canonical and sorted.

    With *derived* -- the per-matrix family read out of the PRE-DROP fingerprint
    document by ``_expected_signature`` -- this is the EXACT set the drop must remove,
    so the two loops in ``_attribute_one_matrix`` together form an EQUALITY: a removed
    column outside it is unattributed, and a member of it that survived is a PARTIAL
    drop.

    The fallback below filters ``diff["removed"]`` by the family predicate, which makes
    it a SUBSET of the observed set BY CONSTRUCTION -- so the partial-drop arm could
    never fire, and a family removed from two matrices and retained in a third sailed
    through on width arithmetic a partial drop satisfies trivially (D30-DEFER-12). It
    is kept only for a caller that supplied no BEFORE document, where it is the most
    the tool can say.
    """
    if derived is not None:
        return sorted({_canonical(name) for name in derived})
    return [_canonical(name) for name in _line_movement_columns(diff["removed"])]


def _attribute_rung1(diff: dict, verdict: dict, fail) -> None:
    """CR-02 exempts DISCRETE INDICATORS from winsorization, and nothing else."""
    if diff["details"] is None:
        for column in sorted(diff["changed"]):
            verdict["unattributed"].append(column)
        if diff["changed"]:
            fail(
                "this fingerprint pair carries no column_details, so rung 1 cannot tell "
                "a discrete indicator from a measurement. Re-run scripts/fingerprint_gold.py "
                "on both sides so column_meta is recorded"
            )
        return

    for column in sorted(diff["changed"]):
        meta = diff["details"].get(column, {})
        discrete = bool(meta.get("discrete_indicator_before")) or bool(
            meta.get("discrete_indicator_after")
        )
        if discrete:
            verdict["attributed"].append(column)
        else:
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' moved at rung 1 but is not a discrete indicator, so "
                "CR-02's winsorization exemption cannot explain it"
            )


def _attribute_rung4(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """N-01 may move the re-synced seasons and NOTHING else (SPEC R2)."""
    blocking = False
    width_before = detail["width_before"]
    width_after = detail["width_after"]
    rows_before = detail.get("rows_before")
    rows_after = detail.get("rows_after")

    for column in diff["added"]:
        blocking = True
        fail(f"column '{column}' was ADDED at rung 4; the re-sync adds no column")
    for column in diff["removed"]:
        blocking = True
        fail(f"column '{column}' was REMOVED at rung 4; the re-sync removes no column")
    if width_before != width_after:
        blocking = True
        fail(
            f"width moved {width_before} -> {width_after} at rung 4; the re-sync changes "
            "no column count"
        )
    if diff["renames"]:
        blocking = True

    grown = _grown_seasons(detail)
    if rows_after is None or rows_before is None or rows_after <= rows_before:
        blocking = True
        fail(
            f"rows moved {rows_before} -> {rows_after} at rung 4, but the N-01 re-sync "
            "exists to ADD rows; a re-sync that adds none proves nothing"
        )
    if not grown:
        blocking = True
        fail(
            "no season gained rows at rung 4, so there is no re-synced season to attribute "
            "a move to"
        )
    if not diff["changed"]:
        blocking = True
        fail(
            "no column moved at rung 4 even though rows were expected to arrive; a "
            "re-synced season must move that season's column hashes"
        )

    allowed = set(grown)
    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        outside = [season for season in seasons if season not in allowed]
        if outside:
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' moved in season(s) {', '.join(outside)}, outside the "
                f"re-synced season(s) {', '.join(grown) or '(none)'}. The WR-06 fix is "
                "INCOMPLETE -- a whole-frame statistic is still reaching prior seasons -- "
                "and the phase is BLOCKED (SPEC R2). Do not proceed to the gate"
            )
        else:
            verdict["attributed"].append(column)

    return blocking


def _print_attribution(verdict: dict) -> None:
    """Print an attribution verdict in the shape a human reads at checkpoint 2."""
    status = (
        "OK" if verdict["ok"] else ("BLOCKED" if verdict["blocking"] else "FINDING")
    )
    print(f"rung {verdict['rung']} ({verdict['cause']}): {status}")
    for matrix, detail in verdict["matrices"].items():
        print(f"  {matrix}:")
        print(f"    attributed:   {detail['attributed']}")
        print(f"    unattributed: {detail['unattributed']}")
        if detail.get("build_clock"):
            print(f"    build clock:  {detail['build_clock']} (moves every build)")
        for preserved in detail.get("value_preserving_dtype", []):
            print(
                f"    dtype only:   {preserved['column']} "
                f"{preserved['dtype_before']} -> {preserved['dtype_after']} "
                "(prior per-season hash reproduced exactly)"
            )
        if detail.get("dtype_proof_failed"):
            print(
                f"    dtype UNPROVEN: {detail['dtype_proof_failed']} "
                "(re-encode did not reproduce the prior hash)"
            )
        if detail["renamed_case_only"]:
            print(f"    case renames: {detail['renamed_case_only']}")
    if verdict["failures"]:
        print("  FAILURES:")
        for message in verdict["failures"]:
            print(f"    - {message}")
    print()


def _print_comparison(report: dict) -> None:
    for matrix, detail in report.items():
        print(f"{matrix}:")
        print(
            f"  width {detail['width_before']} -> {detail['width_after']}   "
            f"rows {detail['rows_before']} -> {detail['rows_after']}"
        )
        if detail["columns_added"]:
            print(f"  columns ADDED:   {detail['columns_added']}")
        if detail["columns_removed"]:
            print(f"  columns REMOVED: {detail['columns_removed']}")
        changed = detail["columns_changed"]
        print(f"  columns CHANGED: {len(changed)}")
        for column, seasons in changed.items():
            print(f"    {column}: seasons {','.join(seasons)}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description="Fingerprint gold feature matrices per column, per season"
    )
    parser.add_argument(
        "--out", type=Path, help="Write the fingerprint JSON to this path"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="Compare two previously written fingerprint JSON documents",
    )
    parser.add_argument(
        "--attribute-rung",
        type=int,
        choices=sorted(RUNG_CAUSES),
        metavar="N",
        help=(
            "Attribute the --compare diff to rebuild rung N's one named cause "
            f"({', '.join(f'{k}={v}' for k, v in sorted(RUNG_CAUSES.items()))}). "
            "Exits 1 when the verdict BLOCKS the phase, 3 when it is a non-blocking "
            "finding, 0 when every moved column is attributed."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point for gold fingerprinting."""
    args = build_parser().parse_args()

    if args.attribute_rung is not None:
        if not args.compare:
            print(
                "ERROR: --attribute-rung requires --compare BEFORE AFTER",
                file=sys.stderr,
            )
            sys.exit(2)
        before = json.loads(args.compare[0].read_text(encoding="utf-8"))
        after = json.loads(args.compare[1].read_text(encoding="utf-8"))
        verdict = attribute_rung(
            compare_fingerprints(before, after),
            args.attribute_rung,
            before=before,
            after=after,
            frame_loader=_gold_frame_loader(),
        )
        _print_attribution(verdict)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            # The run timestamp lives on the VERDICT document, never on the
            # fingerprint document: a fingerprint carrying a timestamp could not be
            # byte-compared across two runs, and that byte-comparison is what makes
            # "unchanged gold hashes identically" checkable. Recording it here is what
            # makes the upstream-drift hypothesis checkable after the fact -- an
            # unattributed column is only diagnosable against the nflreadpy revision
            # date if the rung's run time is on record.
            document = {
                **verdict,
                "attributed_at": datetime.now(UTC).isoformat(),
                "before_document": str(args.compare[0]),
                "after_document": str(args.compare[1]),
            }
            args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        if verdict["blocking"]:
            sys.exit(1)
        if not verdict["ok"]:
            sys.exit(3)
        return

    if args.compare:
        before = json.loads(args.compare[0].read_text(encoding="utf-8"))
        after = json.loads(args.compare[1].read_text(encoding="utf-8"))
        report = compare_fingerprints(before, after)
        _print_comparison(report)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    if args.out is None:
        print("ERROR: --out is required unless --compare is given", file=sys.stderr)
        sys.exit(2)

    fingerprint = fingerprint_gold()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    for matrix, detail in fingerprint.items():
        if detail.get("missing"):
            print(f"{matrix}: MISSING")
            continue
        print(
            f"{matrix}: rows={detail['rows']} width={detail['width']} "
            f"seasons={detail['seasons'][0]}-{detail['seasons'][-1]}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
