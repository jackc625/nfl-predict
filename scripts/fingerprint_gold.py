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
written somewhere else, and that is ENFORCED rather than merely asserted:
``--out`` is checked through ``utils.paths.reject_data_path`` before any gold is
read (WR-07). Phase 30 keeps its documents under the gitignored
``outputs/fingerprints/`` directory.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from conf.settings import get_settings
from utils.paths import reject_data_path

GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")

# The per-build clock stamp, and the ONLY column exempted by name anywhere in this
# module. It is different IN KIND from every other column in a gold matrix:
# ``scripts/build_features.py:570`` executes
# ``combined_features["feature_timestamp"] = datetime.now(UTC)`` on EVERY build, so the
# column takes exactly one distinct value per build and MUST move in every season of
# every rebuild by construction. ``tests/phase30_state.py:768-774`` records the measured
# consequence: two full-history builds on unchanged inputs "produce gold that is
# identical in every column EXCEPT feature_timestamp". It records WHEN the frame was
# built; it measures nothing about the games in the frame, and no model consumes it
# (``models/temporal.py`` excludes it).
#
# Three consequences follow, and they are why it gets its own CATEGORY rather than a
# tolerance:
#
# 1. Filing it under the upstream-drift candidate cause would be FALSE. That note
#    tells a reader to go find the nflreadpy revision that moved the column; for a
#    clock there is none, and the reader burns the search anyway.
# 2. Counting it as a moved value makes rung 3's empty-changed-set criterion
#    structurally UNSATISFIABLE -- no correct rebuild can ever satisfy it, so the
#    criterion stops discriminating between a right rebuild and a wrong one. The same
#    holds for any Phase-31 rung condition phrased as "zero moved columns" (REVIEW-CLOCK).
#    The condition that IS reachable, and that this constant is what makes expressible,
#    is "zero NON-CLOCK moves, with the moved set exactly EQUAL to the registered clock
#    set" -- see ``non_clock_moves`` / ``build_clock_moves`` on every comparison report.
# 3. This is a CLASSIFICATION, not a SUPPRESSION. A clock column that moves is still
#    REPORTED in ``columns_changed``, carrying move kind ``build_clock``; it is merely
#    attributed to the build clock rather than to the data. And the set is REGISTERED
#    rather than inferred from a name, so an unregistered timestamp-LOOKING column that
#    moves is a non-clock move by definition and cannot be waved through by resembling
#    one.
#
# Plan 30-06 already reported it this way at rung 1, by hand, in its SUMMARY. Owner
# ruling D30-OWNER-08 makes the instrument do it, at EVERY rung. This exempts ONE
# named build artifact; every other column is judged exactly as before.
BUILD_CLOCK_COLUMNS = ("feature_timestamp",)


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


def _column_meta_for_season(series: pd.Series) -> dict:
    """Return the STORAGE facts that are meaningful for a single season's slice.

    Exactly two of ``_column_meta``'s three facts appear here, and the omission is
    the point. Dtype and null count are properties of the bytes a season holds, so
    each is well-defined on a slice. Discreteness is NOT: ``_column_meta``'s
    docstring records that a column which is continuous overall but happens to be
    constant within one season would be misclassified as an indicator for that
    season, and nothing about slicing per season changes that reasoning.
    """
    return {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
    }


def fingerprint_matrix(df: pd.DataFrame) -> dict:
    """Return per-column, per-season hashes plus shape metadata for *df*.

    Each cell hashes the column's values ordered by ``game_id`` within the
    season, so a row-order change alone never registers as drift.

    The ``columns`` map and every existing key are unchanged, so fingerprint
    documents written before Plan 30-04 stay comparable. The growth is additive:
    a sibling ``column_meta`` map carrying each column's dtype, null count and
    CR-02 discreteness.

    Plan 31-03 grows it once more, under the SAME discipline: a second sibling
    ``column_meta_by_season`` carrying each column's dtype and null count PER
    SEASON. Without it, ``compare_fingerprints`` could only report a storage-level
    move as "changed, seasons: []" -- and Plan 31-11's hard stop measures a strict
    2021-2024 slice, so a move carrying no season is exactly the case that either
    trips that tripwire spuriously or slips past it. Discreteness is deliberately
    NOT in the per-season map; see ``_column_meta_for_season``.
    """
    seasons = sorted(int(season) for season in df["season"].dropna().unique())
    columns: dict[str, dict[str, str]] = {}
    meta_by_season: dict[str, dict[str, dict]] = {}

    ordered = df.sort_values("game_id")
    for season in seasons:
        subset = ordered[ordered["season"] == season]
        for column in subset.columns:
            series = cast("pd.Series", subset[column])
            digest = hashlib.sha256(_column_bytes(series)).hexdigest()[:16]
            columns.setdefault(column, {})[str(season)] = digest
            meta_by_season.setdefault(str(column), {})[str(season)] = (
                _column_meta_for_season(series)
            )

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
        "column_meta_by_season": meta_by_season,
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


def _is_build_clock(column: str) -> bool:
    """True when *column* is a REGISTERED per-build clock (see BUILD_CLOCK_COLUMNS).

    Membership is the whole test. A column that merely LOOKS like a timestamp is not
    a clock here, because a name heuristic is exactly how a real data move gets waved
    through wearing a clock's costume (T-31-13b).
    """
    return _canonical(column) in {_canonical(name) for name in BUILD_CLOCK_COLUMNS}


def _storage_move_seasons(b_by: dict, a_by: dict, field: str) -> list[str]:
    """Return the seasons in which *field* differs between two per-season meta maps.

    Returns an empty list when EITHER side is absent, which is the pre-Plan-31
    document case: an old document carries no ``column_meta_by_season``, so there is
    nothing to attribute a storage move to and the report says exactly that instead
    of inventing seasons.
    """
    if not b_by or not a_by:
        return []
    return [
        season
        for season in sorted(set(b_by) | set(a_by))
        if (b_by.get(season) or {}).get(field) != (a_by.get(season) or {}).get(field)
    ]


def compare_fingerprints(before: dict, after: dict) -> dict:
    """Return, per matrix, the columns whose hash moved and in which seasons.

    A column counts as MOVED when its values moved, OR when its dtype moved, OR
    when its null count moved. The last two matter because a column can be
    value-identical and still be a different artifact -- an ``int64`` that became
    a ``float64``, or a column that gained a NaN somewhere the season hashes
    happen not to separate.

    Every moved column carries a ``move_kind`` in ``column_details``, and its season
    list is the UNION of the seasons its values moved in and the seasons its STORAGE
    moved in:

    * ``values``      -- at least one per-season hash moved. The strongest claim, so
                         it wins whenever it applies: a column that moved in both ways
                         is a value move, never softened to a storage one.
    * ``storage``     -- only dtype or null count moved. Before Plan 31-03 this case
                         was reported with an EMPTY season list, which is unjudgeable
                         against the strict 2021-2024 slice Plan 31-11 measures. It is
                         now attributed per season from ``column_meta_by_season``.
    * ``build_clock`` -- the column is in ``BUILD_CLOCK_COLUMNS``. A CLASSIFICATION,
                         not a suppression: it stays in ``columns_changed`` and is
                         merely attributed to the build clock rather than to the data.

    Two disjoint lists are emitted alongside, ``non_clock_moves`` and
    ``build_clock_moves``, whose union is exactly the moved set. They are what makes a
    zero-move condition EXPRESSIBLE against a build that stamps ``datetime.now(UTC)``
    on every row: "zero non-clock moves, moved set equal to the registered clock set"
    is reachable, where "zero moved columns" never is (REVIEW-CLOCK).

    Documents written before Plan 30-04 carry no ``column_meta``, and documents written
    before Plan 31-03 carry no ``column_meta_by_season``. Each comparison is simply
    absent for them and the report is exactly what it always was.
    """
    report: dict[str, dict] = {}
    for matrix in GOLD_MATRICES:
        b = before.get(matrix, {})
        a = after.get(matrix, {})
        b_cols = b.get("columns", {})
        a_cols = a.get("columns", {})
        b_meta = b.get("column_meta", {})
        a_meta = a.get("column_meta", {})
        b_by_season = b.get("column_meta_by_season", {})
        a_by_season = a.get("column_meta_by_season", {})

        changed: dict[str, list[str]] = {}
        details: dict[str, dict] = {}
        for column in sorted(set(b_cols) & set(a_cols)):
            value_seasons = [
                season
                for season in sorted(set(b_cols[column]) | set(a_cols[column]))
                if b_cols[column].get(season) != a_cols[column].get(season)
            ]
            bm = b_meta.get(column, {})
            am = a_meta.get(column, {})
            b_by = b_by_season.get(column, {})
            a_by = a_by_season.get(column, {})

            dtype_seasons = _storage_move_seasons(b_by, a_by, "dtype")
            null_seasons = _storage_move_seasons(b_by, a_by, "null_count")

            reasons: list[str] = []
            if value_seasons:
                reasons.append("values")
            if (bm and am and bm.get("dtype") != am.get("dtype")) or dtype_seasons:
                reasons.append("dtype")
            if (
                bm and am and bm.get("null_count") != am.get("null_count")
            ) or null_seasons:
                reasons.append("null_count")

            if not reasons:
                continue

            storage_seasons = sorted(set(dtype_seasons) | set(null_seasons))
            seasons = sorted(set(value_seasons) | set(storage_seasons))
            if _is_build_clock(column):
                move_kind = "build_clock"
            elif value_seasons:
                move_kind = "values"
            else:
                move_kind = "storage"

            changed[column] = seasons
            details[column] = {
                "seasons": seasons,
                "seasons_values": value_seasons,
                "seasons_storage": storage_seasons,
                "move_kind": move_kind,
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
            "non_clock_moves": sorted(
                column for column in changed if not _is_build_clock(column)
            ),
            "build_clock_moves": sorted(
                column for column in changed if _is_build_clock(column)
            ),
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

# Where every rung document lives. ``outputs/`` is gitignored, so NO fingerprint
# document is a committed artifact: the committed record is this code plus the tests,
# and the measured hashes are transcribed into the running plan's SUMMARY.
FINGERPRINT_DIR = Path("outputs/fingerprints")

# The document names Phase 30 already wrote into FINGERPRINT_DIR. They are the record
# of the phase that produced the standing gold, they cannot be regenerated (the builds
# that produced them reached nflreadpy LIVE), and a Phase-31 run that overwrote one
# would destroy evidence rather than add any (T-31-11).
PHASE30_RUNG_DOCUMENTS = tuple(f"rung{rung}.json" for rung in range(5))

# The prefix Phase-31 rung documents carry so they cannot collide with the above.
PHASE31_RUNG_PREFIX = "p31_"


# ---------------------------------------------------------------------------
# PHASE 33.1'S RUNG -- A PREFIX-AWARE CAUSE TABLE, NOT A WIDENED INTEGER SPACE
# (Plan 33.1-07 Task 1, Ruling N).
#
# THE THREE REASONS THIS IS A PREFIX AND NOT A NEW INTEGER, on the record so the
# choice reads as a decision rather than as a number somebody picked:
#
# 1. Taking rung integer 5 would land in `_expected_signature`'s `else` branch,
#    which is rung 4's signature -- "rows strictly increased" plus a season-list
#    rule -- and THIS rung's rows are UNCHANGED at 6,499. It would be judged
#    against the wrong prediction, and a judge applying the wrong prediction is
#    worse than no judge, because its verdict still reads as one.
# 2. Taking rung integer 5 under a fresh prefix would make `require_rung_ladder`
#    demand `p331_rung0.json` through `p331_rung4.json` -- four documents
#    describing rebuilds this phase does not run.
# 3. Plan 33-14 wants to extend the SAME closed dict for the Elo rung. Racing for
#    an integer now leaves the two phases colliding later; a prefix gives 33-14
#    the same seam instead of a contested key.
#
# So the cause LOOKUP became prefix-aware. Phase 30's four `RUNG_CAUSES` entries
# are byte-untouched, Phase 31's `p31_` ladder is unaffected, and
# `require_rung_ladder(dir, 1, "p331_")` demands exactly `p331_rung0.json` -- the
# fingerprint of CURRENT gold taken BEFORE the rebuild, which is precisely the
# control it exists to enforce.
# ---------------------------------------------------------------------------

PHASE331_RUNG_PREFIX: str = "p331_"
PHASE331_RUNG: int = 1

# The ONE column this rung adds, by NAME. The width integer alone cannot say
# WHICH column arrived, and a build that added an unrelated column while omitting
# the flag satisfies +1 exactly as well as the right one does.
PHASE331_ADDED_COLUMN: str = "weather_coverage"

# THE COMPOUND LABEL. The SPEC prohibition this phase carries reads: "MUST NOT
# label a gold rebuild 'the weather rung' if it also carries Elo, bye-window or
# ats_edge changes -- that re-creates the attribution failure this phase exists to
# prevent." The same reasoning binds the rung's own name: it has THREE causes, and
# a label naming one of them is the mislabelling, not a shorthand for it.
PHASE331_RUNG_CAUSE: str = (
    "COMPOUND (three causes, never 'the weather rung'): (1) real ERA5 weather "
    "replacing the fabricated 65.0F constant across 2002-2025 [R5/D33.1-07]; "
    "(2) the all-seasons stadium_id routing correction [D33.1-06], which moves "
    "the venue/travel/timezone/elevation family for the 1,153 games that "
    "resolved to the wrong stadium; and (3) restored 2025 coverage -- the 207 "
    "games the two silver feature tables were missing, which the rebuild adds "
    "independently of any weather or routing change"
)

# ---------------------------------------------------------------------------
# THE FOLLOW-UP RUNG (Plan 33.1-07, declared 2026-09-14 AFTER the diagnosis).
#
# Ruling N2 names exactly two legitimate moves when a changed column falls
# outside rung 1's declared families: STOP, or declare a NEW family in a FOLLOW-UP
# RUNG. It explicitly forbids the third -- a footnote on rung 1 -- and it forbids
# widening rung 1's declaration after the diff has been seen, because a signature
# edited after the observation is a transcription wearing a prediction's clothes.
#
# Rung 1 returned ok=False with 45 unattributed columns, identical in all three
# matrices. The owner ordered the residual DIAGNOSED before anything was declared;
# the investigation ran eight controlled rebuilds and is recorded in full at
# `.planning/phases/33.1-.../33.1-07-GROUP1-DIAGNOSIS.md`. This rung is the second
# legitimate move, taken on that evidence rather than as a guess.
#
# IT JUDGES THE SAME TRANSITION. No rebuild happened between rung 1 and this rung
# and none will: `p331_rung0.json -> p331_rung1.json` is re-judged under a SECOND
# declaration that adds three families rung 1 did not carry. That is why there is
# no `p331_rung2.json` fingerprint document -- writing one would assert a rebuild
# that did not occur. `require_rung_ladder(dir, 2, "p331_")` demands rung 0 and
# rung 1, which is exactly right: both exist, and the chain is intact.
# ---------------------------------------------------------------------------

PHASE331_FOLLOWUP_RUNG: int = 2

PHASE331_FOLLOWUP_RUNG_CAUSE: str = (
    "FOLLOW-UP to the Phase-33.1 compound rung, declaring the three residual "
    "groups rung 1 refused (45 columns, identical in all three matrices). It "
    "re-judges the SAME p331_rung0 -> p331_rung1 transition and rebuilds "
    "nothing. (1) STALE-BASELINE CARRY-FORWARD -- 40 rolling team-form, "
    "opponent-adjusted and Elo-momentum columns that moved in season 2024 (24 in "
    "2024 only, 16 in 2024 and 2025 through the prior-season normalisation "
    "channel). The move PREDATES this rung: the Plan 33-12 sandbox gold, built "
    "five and a half hours earlier from the OLD silver, already carries 40/40 of "
    "the new values and 0/40 of rung 0's. ITS TRIGGER IS NOT ESTABLISHED and its "
    "MAGNITUDE IS PERMANENTLY UNMEASURABLE -- see PHASE331_FOLLOWUP_GROUP1_. "
    "(2) THE MISLABELLING PROHIBITION FIRING CORRECTLY -- 4 columns that satisfy "
    "the 2025 staleness predicate but belong to prohibited families, where the "
    "prohibition takes precedence by design. That is the guard working, not a "
    "defect. (3) A GENUINE WEATHER-FAMILY WIDENING -- raw_weather_severity, the "
    "un-normalized copy of a WEATHER_FEATURE_COLUMNS member, absent from that "
    "registry and therefore never reachable by rung 1's family 1"
)

RUNG_CAUSES_BY_PREFIX: dict[str, dict[int, str]] = {
    # Phase 30's ladder, and every unprefixed caller. Referenced, NOT copied: a
    # second spelling of the four entries is the second-list failure mode D30-02
    # exists to prevent, and the negative control in
    # tests/integration/test_gold_rebuild_attribution.py proves the new table did
    # not capture the old integer.
    "": RUNG_CAUSES,
    # PHASE 31 SHARES PHASE 30'S CAUSES ON PURPOSE, and the reason is that a
    # prefix has TWO jobs which happen to coincide here and not there. It
    # name-spaces the DOCUMENTS (so `p31_rung0.json` cannot overwrite Phase 30's
    # unregenerable `rung0.json`), and it selects the CAUSE TABLE. Phase 31's
    # ladder re-runs Phase 30's named causes over new inputs, so only the first
    # job applies; Phase 33.1's rung has a cause of its own, so both do. Written
    # as a reference to the same dict rather than a copy -- a second spelling of
    # the four entries is the second-list failure mode D30-02 exists to prevent.
    PHASE31_RUNG_PREFIX: RUNG_CAUSES,
    # TWO rungs under this one prefix, and they are NOT two rebuilds. Rung 1 is
    # the compound gold rebuild; rung 2 is the follow-up DECLARATION over the same
    # transition. Ruling N2's remedy for a legitimate out-of-family move is a new
    # family in a follow-up rung, so the follow-up must be a real ladder entry
    # with its own cause -- not a second key on rung 1's.
    PHASE331_RUNG_PREFIX: {
        PHASE331_RUNG: PHASE331_RUNG_CAUSE,
        PHASE331_FOLLOWUP_RUNG: PHASE331_FOLLOWUP_RUNG_CAUSE,
        # Rung 3 is a REAL rebuild again, unlike rung 2. Its cause is declared
        # further down, beside its families, because it could not be written
        # until rung 1's OUTPUT had been measured -- which is what found the
        # three input defects it corrects. The entry is filled in immediately
        # after that declaration so this table stays the one place a prefix's
        # rungs are enumerated.
    },
}

# The season(s) the 207 previously-absent rows belong to. MEASURED before the
# rebuild (tests.phase33_state.PHASE331_STALENESS_GAME_IDS carries the exact ids)
# and committed, because the rebuild is what CLOSES the gap -- afterwards the set
# is unmeasurable, which is the same argument `p331_rung0.json` rests on.
PHASE331_STALENESS_SEASONS: tuple[int, ...] = (2025,)

# The columns `combine_features` (scripts/build_features.py:558) strips or merges
# ON rather than merging IN, so they are carried by the frame and contributed by
# no builder as a feature.
PHASE331_CONTEXTUAL_MERGE_KEYS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
)

# The gold-name shape `FeatureMatrixBuilder._get_team_features`
# (scripts/build_features.py:803) constructs for the team-form rolling family --
# `{home,away}_{off,def}_rolling_{metric}`. That family is what the R12
# bye-window rule in `features/team_form.py` governs, so it is the bye-window
# cause's footprint in gold.
#
# A PREDICATE HERE IS SAFE WHERE A NAME HEURISTIC FOR THE BUILD CLOCK WOULD NOT
# BE, and the asymmetry is the reason `_is_build_clock` insists on a REGISTERED
# set. A name test that GRANTS an exemption lets a real move through wearing a
# costume; this one WIDENS a refusal, so its failure mode is an extra refusal
# rather than an absorbed change. The markers deliberately exclude
# `_rolling_snap_share_`, which is the snap-count family and not team form.
_PHASE331_TEAM_FORM_ROLLING_MARKERS: tuple[str, ...] = (
    "_off_rolling_",
    "_def_rolling_",
)


def phase331_weather_family() -> tuple[str, ...]:
    """DECLARED FAMILY 1: the weather columns, from the ONE registry that owns them.

    Derived from ``features.weather.WEATHER_FEATURE_COLUMNS`` rather than
    re-listed here. The import is deferred for the same reason
    ``_discrete_indicator_predicate``'s is: this module is also used as a plain
    fingerprint reader and must not pull the feature stack to do that.
    """
    from features.weather import WEATHER_FEATURE_COLUMNS

    return tuple(WEATHER_FEATURE_COLUMNS)


_PHASE331_VENUE_FAMILY_CACHE: tuple[str, ...] | None = None


def _derive_phase331_venue_family() -> tuple[str, ...]:
    """Run the contextual builder once and return the columns it ACTUALLY emits.

    DERIVED, NEVER TYPED. ``tests/unit/test_data_qa_gold_width.py``'s docstring
    already records what a second hand-written list of a family costs: the list
    and the predicate that did the work drift apart, and every assertion resting
    on the list is then pinning a fiction. The contextual builder emits its column
    set by composing several factory dicts inside one loop, so the only honest way
    to name that set is to run it.

    The probe frame is ONE synthetic game built from the builder's OWN venue
    records -- a real ``stadium_id`` and a real home team -- because
    ``_resolve_venue_id_for_game`` RAISES on an id absent from ``data/venues.json``
    rather than falling back to the home team's stadium (D33.1-06). The
    situational spot-flag block reloads the season schedule from silver and
    degrades to neutral flags on any load error, so the emitted COLUMN SET is the
    same with or without ``data/`` -- which is what lets this run on a fresh
    checkout.
    """
    from features.contextual import ContextualFeaturesCalculator

    calculator = ContextualFeaturesCalculator()
    record = next(
        venue
        for venue in calculator.venues_data["venues"]
        if venue.get("stadium_id") and venue.get("home_teams")
    )
    team = record["home_teams"][0]
    probe = pd.DataFrame(
        [
            {
                "game_id": "2024_W01_PROBE",
                "season": 2024,
                "week": 1,
                "home_team": team,
                "away_team": team,
                "kickoff_et": pd.Timestamp(
                    "2024-09-08 13:00:00", tz="America/New_York"
                ),
                "stadium_id": record["stadium_id"],
                "home_score": 0.0,
                "away_score": 0.0,
            }
        ]
    )
    emitted = calculator.build_features(probe, datetime(2024, 9, 9, tzinfo=UTC))
    merge_keys = set(PHASE331_CONTEXTUAL_MERGE_KEYS)
    family = tuple(
        sorted(column for column in emitted.columns if column not in merge_keys)
    )
    if not family:
        msg = (
            "the contextual builder emitted no feature column at all, so DECLARED "
            "FAMILY 2 would be empty and the Phase-33.1 rung would refuse every "
            "venue/travel/timezone/elevation move it exists to attribute. "
            "Refusing to derive an empty family rather than judging against one."
        )
        raise ValueError(msg)
    return family


def phase331_venue_family() -> tuple[str, ...]:
    """DECLARED FAMILY 2, derived once and cached for the process."""
    global _PHASE331_VENUE_FAMILY_CACHE
    if _PHASE331_VENUE_FAMILY_CACHE is None:
        _PHASE331_VENUE_FAMILY_CACHE = _derive_phase331_venue_family()
    return _PHASE331_VENUE_FAMILY_CACHE


def phase331_prohibited_families(
    columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, tuple[str, ...]]:
    """The three families the SPEC prohibition names, resolved over *columns*.

    Two of the three are EXACT NAMES and ignore *columns* entirely:

    * ``elo`` -- ``features.elo_features.ELO_FEATURE_COLUMNS``, the tuple
      ``combine_features`` itself merges by (scripts/build_features.py:616).
    * ``ats_edge`` -- one exact name. It is not in gold today; the guard is
      forward-looking, because Plan 33-10's points-scale repair is a separate
      cause and a separate rung.

    ``bye_window`` is a PREDICATE over the team-form rolling shape, so it can only
    be enumerated against a universe. *columns* is that universe -- a diff's
    changed set at attribution time, or the two DECLARED families when the caller
    is checking disjointness. It defaults to the declared families precisely so
    that disjointness check is NON-VACUOUS: the predicate is resolved against
    exactly the set it must not intersect.

    NOT IN ANY OF THE THREE: ``home_off_bye`` / ``away_off_bye``. They are
    contextual-emitted, so they are inside DECLARED FAMILY 2 by construction, and
    the bye-WINDOW cause is the rolling-window selector in ``features/team_form.py``
    rather than the off-bye rest flag. Stated here so a later reader does not
    "fix" it by moving them.
    """
    from features.elo_features import ELO_FEATURE_COLUMNS

    universe = (
        tuple(columns)
        if columns is not None
        else (*phase331_weather_family(), *phase331_venue_family())
    )
    return {
        "elo": tuple(ELO_FEATURE_COLUMNS),
        "bye_window": tuple(
            sorted(
                column
                for column in universe
                if any(
                    marker in _canonical(column)
                    for marker in _PHASE331_TEAM_FORM_ROLLING_MARKERS
                )
            )
        ),
        "ats_edge": ("ats_edge",),
    }


def _phase331_prohibited_family(column: str) -> str | None:
    """The prohibited family *column* belongs to, or None."""
    canonical = _canonical(column)
    for label, names in phase331_prohibited_families([canonical]).items():
        if canonical in {_canonical(name) for name in names}:
            return label
    return None


# THE EXPECTED CHANGE SET, DECLARED AS A MODULE CONSTANT.
#
# A constant rather than something assembled inside `_expected_signature`'s
# branch, because it is this phase's PRE-DECLARED prediction and it has to be
# readable in committed source BEFORE the rebuild runs. `T-33.1-43` is the threat
# it answers: a signature edited after the diff is seen converts a prediction into
# a transcription.
#
# EVERY FAMILY IS ENUMERABLE OR SOURCE-DERIVED (Ruling N2). The first draft
# declared family 3 as "any column that moves because 207 rows gained coverage",
# which is a CAUSE STORY, not an allow-list: it cannot be evaluated against a
# diff, so any moved column can be argued into it afterwards. This repository has
# the worked example -- `_attribute_rung2` below is a blanket attribution whose
# own comment records that it "cannot FAIL on a moved column", and that rung 2's
# first attempt "attributed perfectly cleanly -- ok, zero unattributed -- while
# having silently destroyed 18 columns". So family 3 is a ROW-SCOPED PREDICATE
# instead: a changed column outside families 1 and 2 is attributable ONLY IF its
# per-season digests are byte-identical in every season except 2025. A column that
# moved in 2019 cannot have moved because 2025 gained rows, and under this
# predicate it is UNATTRIBUTED and blocks.
PHASE331_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE331_RUNG,
    "prefix": PHASE331_RUNG_PREFIX,
    "cause": PHASE331_RUNG_CAUSE,
    "columns_added": (PHASE331_ADDED_COLUMN,),
    "columns_removed": "empty",
    "columns_changed": (
        "restricted to three DECLARED families, each ENUMERABLE or SOURCE-DERIVED: "
        "(1) the weather family named by features.weather.WEATHER_FEATURE_COLUMNS; "
        "(2) PHASE331_VENUE_FAMILY_COLUMNS, derived at first use from the columns "
        "features.contextual.ContextualFeaturesCalculator.build_features actually "
        "emits, minus the merge keys -- the family D33.1-06 widened SPEC R5's "
        "acceptance to include, and where Ruling H puts venue_cold_climate; and "
        "(3) the 2025 staleness repair, which is a ROW population rather than a "
        "column family and therefore gets a ROW-SCOPED predicate -- a changed "
        "column outside (1) and (2) is attributable only if its per-season digests "
        "are byte-identical in every season except "
        f"{PHASE331_STALENESS_SEASONS}"
    ),
    "rows": "unchanged",
    "width": "increased by exactly one, the named coverage flag",
    "declared_families": ("weather", "venue", "staleness_2025"),
    "family_mechanisms": {
        "weather": "source-derived constant",
        "venue": "source-derived constant",
        "staleness_2025": "row-scoped per-season-digest predicate",
    },
    # Ruling N2's second half, stated where the judge can be read. `ok` must be
    # True UNCONDITIONALLY. `unattributed_with_reason` still exists and is still
    # populated -- an explanation is worth having -- but it SUPPLEMENTS the check
    # and can never substitute for it. Where an out-of-family move turns out to be
    # legitimate, the correct act is a NEW declared family in a follow-up rung,
    # not a footnote on this one.
    "ok_required_unconditionally": True,
}


# ---------------------------------------------------------------------------
# THE FOLLOW-UP RUNG'S THREE DECLARED FAMILIES.
#
# Each is EXACT NAMES plus a SEASON RESTRICTION, or a SOURCE-DERIVED mapping.
# Ruling N2's discipline is unchanged and is not relaxed by the fact that this
# declaration is written after the diff: the families below are enumerable, they
# DISCRIMINATE (a member that moved in a season outside its declared restriction
# is still UNATTRIBUTED and still blocks), and a column that is in no family is
# refused exactly as it was at rung 1. What a follow-up rung buys is that the
# unknown is BOUNDED AND WRITTEN DOWN rather than hidden inside a
# plausible-looking bucket -- which is the whole reason "declare a new family"
# is a legitimate move and "annotate past it" is not.
# ---------------------------------------------------------------------------

# GROUP 1 -- THE STALE-BASELINE CARRY-FORWARD. 40 columns, measured from
# `p331_rung1_attribution.json` and identical in all three matrices.
#
# WHAT IS SETTLED, each by a controlled rebuild (33.1-07-GROUP1-DIAGNOSIS.md):
#   * The build is DETERMINISTIC. A fresh full rebuild reproduced every column of
#     every season byte-for-byte except `feature_timestamp`, and pinning
#     `as_of_datetime` 4.5 months earlier was byte-identical too.
#   * The move PREDATES this rung. The Plan 33-12 sandbox gold, built 02:55:48 on
#     2026-09-14 from the OLD 6,292-row weather silver -- five and a half hours
#     BEFORE the rung -- already carries 40/40 of the new 2024 values and 0/40 of
#     rung 0's. So it is NOT caused by the real weather, NOT by the stadium_id
#     routing correction and NOT by the 2025 coverage restore.
#   * ELIMINATED by controlled rebuild, each by measurement rather than argument:
#     wall-clock `as_of` default; population-dependent normalisation; the Wave-12
#     identity migration (reverting it moves only the rest-days family); the
#     builder code itself (the pre-rung tree at bc31982 run against today's data
#     gives 40/40 match to current gold); row order (reversing 2024's rows moves
#     146 columns, not 40); the pinned upstream play-by-play (all 76
#     `config/upstream_pin.json` entries re-hash clean); and DuckDB/parquet
#     divergence.
#
# WHAT IS NOT SETTLED, stated without softening:
#   * THE TRIGGER IS NOT ESTABLISHED. It is localised to 2026-09-12 08:36 ->
#     2026-09-14 02:55 (Phase 33's waves 9-12), but every observable input and all
#     builder code are byte-identical across that window, and BOTH the old and the
#     current code produce today's values from today's data. The pre-rung 2024
#     values are not reproducible from any surviving state.
#   * THE MAGNITUDE IS PERMANENTLY UNMEASURABLE. `p331_rung0.json` holds per-season
#     digests, not values, and no copy of the pre-rung gold survives. A digest can
#     say "different"; it can never say "how different". The sharp
#     2020-2023-clean / 2024-moved boundary argues against float noise, but that is
#     an argument, not a measurement, and it is not recorded as one.
#
# WHY 2024, measured: team-form silver covers only 2020-2025 so 2002-2019 are a
# structural 0.0 constant and CANNOT move; `features/team_form.py:714` hardcodes
# `range(2018, 2025)`, making 2024 the last opponent-adjusted season; and season
# 2025's normalisation bootstraps on season 2024, which is the 24-only / 16-both
# split. The declared seasons below are exactly those two and no others.
PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS: tuple[str, ...] = (
    "away_def_rolling_opp_adj_epa_per_play",
    "away_def_rolling_opp_adj_pass_epa",
    "away_def_rolling_opp_adj_rush_epa",
    "away_def_rolling_pass_success_rate",
    "away_def_rolling_red_zone_td_rate",
    "away_def_rolling_rush_success_rate",
    "away_def_rolling_success_rate",
    "away_def_rolling_third_down_conversion_rate",
    "away_elo_momentum",
    "away_off_rolling_avg_drive_start_yardline",
    "away_off_rolling_cpoe",
    "away_off_rolling_neutral_pace",
    "away_off_rolling_neutral_pass_rate",
    "away_off_rolling_opp_adj_epa_per_play",
    "away_off_rolling_opp_adj_pass_epa",
    "away_off_rolling_opp_adj_rush_epa",
    "away_off_rolling_pass_success_rate",
    "away_off_rolling_red_zone_td_rate",
    "away_off_rolling_rush_success_rate",
    "away_off_rolling_success_rate",
    "away_off_rolling_third_down_conversion_rate",
    "home_def_rolling_opp_adj_epa_per_play",
    "home_def_rolling_opp_adj_pass_epa",
    "home_def_rolling_opp_adj_rush_epa",
    "home_def_rolling_pass_success_rate",
    "home_def_rolling_rush_success_rate",
    "home_def_rolling_success_rate",
    "home_def_rolling_third_down_conversion_rate",
    "home_elo_momentum",
    "home_off_rolling_avg_drive_start_yardline",
    "home_off_rolling_cpoe",
    "home_off_rolling_neutral_pass_rate",
    "home_off_rolling_opp_adj_epa_per_play",
    "home_off_rolling_opp_adj_pass_epa",
    "home_off_rolling_opp_adj_rush_epa",
    "home_off_rolling_pass_success_rate",
    "home_off_rolling_red_zone_td_rate",
    "home_off_rolling_rush_success_rate",
    "home_off_rolling_success_rate",
    "home_off_rolling_third_down_conversion_rate",
)

# The ONLY seasons a Group-1 column may move in. 2024 is where the carry-forward
# sits; 2025 is reachable ONLY through season 2025's normalisation bootstrap on
# season 2024, which Run E4 of the diagnosis demonstrated directly. A Group-1
# column that moved in 2019 is NOT this cause and is UNATTRIBUTED -- the same
# discrimination rung 1's row-scoped staleness predicate carries.
PHASE331_FOLLOWUP_STALE_BASELINE_SEASONS: tuple[int, ...] = (2024, 2025)

# GROUP 2 -- THE MISLABELLING PROHIBITION FIRING CORRECTLY, not a defect.
#
# These four satisfy rung 1's staleness predicate exactly (they moved in 2025 and
# in no other season) and would have been attributed to the 2025 coverage restore
# -- except that they belong to PROHIBITED families, and
# `_attribute_phase331` checks the prohibition FIRST, by design, so a prohibited
# column can never be absorbed by a declared family. Rung 1 refusing them is the
# guard doing its job. Declaring them here does not weaken it: the prohibition
# still fires for every prohibited column NOT named below, and these four keep a
# season restriction of exactly 2025.
PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS: tuple[str, ...] = (
    "elo_prob_home",
    "home_def_rolling_red_zone_td_rate",
    "home_elo",
    "home_elo_uncertainty",
)

PHASE331_FOLLOWUP_PROHIBITED_2025_SEASONS: tuple[int, ...] = (2025,)

# GROUP 3 -- A GENUINE WEATHER-FAMILY WIDENING, expressed as a SOURCE-DERIVED
# MAPPING rather than a bare name.
#
# `raw_weather_severity` moved in ALL 24 seasons, which is exactly what replacing
# a fabricated 65.0F constant with real ERA5 observations does to a weather
# column. It was refused only because it is absent from
# `features.weather.WEATHER_FEATURE_COLUMNS`: `scripts/build_features.py:1819`
# mints it as a verbatim un-normalized COPY of `weather_severity_score` after
# imputation and before normalisation, so it never passes through the registry
# rung 1's family 1 is derived from.
#
# The value is the WEATHER_FEATURE_COLUMNS member each widening column copies, and
# the predicate CHECKS it at attribution time against the live registry. That is
# what makes this source-derived rather than a second hand-written list: an entry
# whose source is not a weather column is refused, so the family cannot be
# extended into a general-purpose bucket.
PHASE331_FOLLOWUP_WEATHER_WIDENING: dict[str, str] = {
    "raw_weather_severity": "weather_severity_score",
}

PHASE331_FOLLOWUP_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE331_FOLLOWUP_RUNG,
    "prefix": PHASE331_RUNG_PREFIX,
    "cause": PHASE331_FOLLOWUP_RUNG_CAUSE,
    # The SAME transition. This rung rebuilds nothing, so its structural
    # prediction is rung 1's structural prediction, unchanged.
    "judges": (
        "the SAME p331_rung0 -> p331_rung1 transition, re-judged under a second "
        "declaration. No rebuild happened between the two rungs and none will, "
        "which is why there is no p331_rung2.json fingerprint document"
    ),
    "no_new_rebuild": True,
    "columns_added": (PHASE331_ADDED_COLUMN,),
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "increased by exactly one, the named coverage flag",
    "columns_changed": (
        "rung 1's three families, PLUS three declared residual families: "
        "(4) PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS -- 40 exact names "
        f"restricted to seasons {PHASE331_FOLLOWUP_STALE_BASELINE_SEASONS}, whose "
        "TRIGGER IS NOT ESTABLISHED and whose MAGNITUDE IS PERMANENTLY "
        "UNMEASURABLE; (5) PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS -- 4 exact "
        f"names restricted to seasons {PHASE331_FOLLOWUP_PROHIBITED_2025_SEASONS}, "
        "the mislabelling prohibition firing correctly; and (6) "
        "PHASE331_FOLLOWUP_WEATHER_WIDENING -- a source-derived mapping from each "
        "un-normalized copy to the WEATHER_FEATURE_COLUMNS member it copies, "
        "checked against the live registry at attribution time"
    ),
    "declared_families": (
        "carried_at_rung_1",
        "stale_baseline_2024",
        "prohibited_family_2025",
        "weather_widening",
    ),
    "family_mechanisms": {
        "carried_at_rung_1": "source-derived constant",
        "stale_baseline_2024": "enumerated names with a season restriction",
        "prohibited_family_2025": "enumerated names with a season restriction",
        "weather_widening": "source-derived constant",
    },
    # UNCHANGED from rung 1, and deliberately restated rather than inherited: a
    # follow-up rung that quietly relaxed the unconditional check would be the
    # footnote Ruling N2 forbids, wearing a rung's clothes.
    "ok_required_unconditionally": True,
    "group1_trigger": "NOT ESTABLISHED",
    "group1_magnitude": "PERMANENTLY UNMEASURABLE",
}


# ---------------------------------------------------------------------------
# RUNG 3 -- THE INPUT-CORRECTION RUNG, DECLARED BEFORE ANYTHING WAS REBUILT
# (Plan 33.1-07 Task 4, declared 2026-09-14).
#
# WHY THERE IS A THIRD RUNG AT ALL. Rung 1 rebuilt gold on the corrected
# weather. Measuring rung 1's OUTPUT -- which is what Task 4 exists to do --
# found three defects in the INPUTS that rung, and every rung before it, had
# been consuming. All three discard real data that exists on disk:
#
#   1. RAIN THAT FELL WAS DISCARDED. `calculate_precipitation_features` refused
#      when `precip_prob` was absent. That is a FORECAST probability; ERA5
#      reanalysis never reports one. So the gate fired on all 4,847 outdoor
#      games and threw away `precip_mm`, which is present on all 6,499 rows.
#      Twelve precipitation columns and the seven composites that read
#      `precip_impact_score` were NULL for every outdoor game -- the games where
#      rain is the entire point.
#   2. THE COVERAGE FLAG WAS INERT. Silver carried `weather_coverage` = 1.0 on
#      all 6,499 rows; gold recorded 0.0 on all 6,499 -- the value that MEANS no
#      observation. The flag was z-scored, and the expanding std of a constant
#      column is zero, so every row collapsed to the neutral 0.0.
#   3. SEASON 2025'S TEAM STRENGTH WAS FAKE. `range(2018, 2025)` stops at 2024,
#      so the twelve `*_rolling_opp_adj_*` columns carried 2 distinct values
#      across the 285 games of 2025 -- against 285 in both 2023 and 2024. 2025
#      is the season feeding the live 2026 predictions.
#
# WHY ONE RUNG AND NOT THREE. The OWNER ruled on 2026-09-14: fix all three, then
# ONE rebuild. The tradeoff they were given and accepted is stated plainly --
# three causes in one rebuild cannot be separated afterwards. That mattered when
# the frozen pre-correction baseline was the thing being protected; the STANDING
# OWNER RULING of the same date voids that baseline, so what one rebuild costs
# here is separability between three corrections that all point the same way,
# and what it buys is ending the phase with inputs that are actually correct.
#
# THIS RUNG'S BASELINE IS NOT STALE, and that is the difference from rung 1.
# `p331_rung2.json` is a fingerprint of gold as rung 1 left it -- freshly built,
# freshly attributed, 38 minutes of controlled rebuilds already run over it by
# the Group-1 diagnosis. Rung 1's own baseline was five and a half hours old and
# carried the unexplained carry-forward the follow-up rung had to declare.
#
# THERE IS NO `p331_rung2` REBUILD, so where does its fingerprint come from? The
# follow-up rung rebuilt nothing -- it re-judged the rung0 -> rung1 transition --
# so gold at rung 2 IS gold at rung 1. `p331_rung2.json` is therefore a fresh
# MEASUREMENT of current gold rather than a copy of rung 1's document, taken
# before this rung rebuilds anything. `require_rung_ladder(dir, 3, "p331_")`
# demands rungs 0, 1 and 2, and all three then exist.
# ---------------------------------------------------------------------------

PHASE331_RUNG3: int = 3

PHASE331_RUNG3_CAUSE: str = (
    "COMPOUND (three INPUT corrections, never 'the precipitation rung'): "
    "(1) precipitation derived from ERA5's measured precip_mm instead of being "
    "discarded for want of a forecast probability the archive never reports, "
    "which moves the twelve precipitation columns and the seven composites that "
    "read precip_impact_score across every season; (2) the weather_coverage "
    "flag preserved at its recorded level instead of being z-scored to 0.0 -- "
    "the value that means NO OBSERVATION -- on all 6,499 rows; and (3) season "
    "2025's twelve opponent-adjusted team-strength columns built from real "
    "play-by-play instead of imputed, because the per-game season pool was the "
    "hardcoded range(2018, 2025) and stopped at 2024"
)

# DECLARED FAMILY 3A and 3B reuse rung 1's and the follow-up's source-derived
# mechanisms unchanged: `phase331_weather_family()` for the registry columns and
# `PHASE331_FOLLOWUP_WEATHER_WIDENING` for the un-normalized copy. They are
# REFERENCED rather than re-listed, which is the same second-list discipline
# D30-02 exists to enforce.

# DECLARED FAMILY 3C -- the 2025 team-strength repair. TWELVE EXACT NAMES plus a
# SEASON RESTRICTION of 2025 and nothing else.
#
# THE SEASON RESTRICTION IS MEASURED, NOT ASSUMED. Adding 2025 to the per-game
# pool could in principle have moved every season: `features/opponent_adj.py`
# computes `league_avg_def` as a WHOLE-FRAME mean over the fetched rows, and a
# wider pool moves that scalar (measured: -0.003941 -> -0.002092). A read-only
# probe ran the REAL adjustment stage under both pools before this declaration
# was written and compared the per-game and the rolling outputs:
#
#     per-game rows compared 7,476   moved 0   max delta 0.0
#     rolling  rows compared 8,064   moved 0   max delta 0.0
#
# ZERO rows moved outside 2025, and the reason is itself a finding recorded in
# the SUMMARY: the opponent adjustment is INERT. `opp_adj_<metric>` equals the
# raw metric for all 8,564 rows, because the `has_enough` gate never fires, so
# the league average it would have been added to never reaches the output. That
# is a PRE-EXISTING defect, outside this plan's scope, deliberately NOT fixed
# here -- fixing it would move all of 2018-2024 and blow past this declaration.
#
# So the restriction to 2025 is a measurement of what this change does, and a
# member that moved in 2019 is UNATTRIBUTED and blocks, exactly as it would be
# if it were undeclared.
PHASE331_RUNG3_TEAM_STRENGTH_2025_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{unit}_rolling_opp_adj_{metric}"
    for side in ("home", "away")
    for unit in ("off", "def")
    for metric in ("epa_per_play", "pass_epa", "rush_epa")
)

PHASE331_RUNG3_TEAM_STRENGTH_SEASONS: tuple[int, ...] = (2025,)

PHASE331_RUNG3_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE331_RUNG3,
    "prefix": PHASE331_RUNG_PREFIX,
    "cause": PHASE331_RUNG3_CAUSE,
    # NOTHING IS ADDED HERE, and that is the structural difference from rungs 1
    # and 2. Those predicted the coverage flag ARRIVING; this rung corrects what
    # three existing columns families CONTAIN. A build that added a column would
    # be doing something this rung did not declare.
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to three DECLARED families, each ENUMERABLE or "
        "SOURCE-DERIVED: (1) the weather family named by "
        "features.weather.WEATHER_FEATURE_COLUMNS, any season, because replacing "
        "a discarded measurement with a real one moves every season it was "
        "discarded in; (2) PHASE331_FOLLOWUP_WEATHER_WIDENING, the "
        "source-derived mapping from raw_weather_severity to the registry "
        "column it copies, checked against the live registry at attribution "
        "time; and (3) PHASE331_RUNG3_TEAM_STRENGTH_2025_COLUMNS -- twelve exact "
        "names restricted to season "
        f"{PHASE331_RUNG3_TEAM_STRENGTH_SEASONS}, measured by a read-only probe "
        "of the real adjustment stage under both season pools BEFORE this "
        "declaration was written"
    ),
    "declared_families": (
        "weather",
        "weather_widening",
        "team_strength_2025",
    ),
    "family_mechanisms": {
        "weather": "source-derived constant",
        "weather_widening": "source-derived constant",
        "team_strength_2025": "enumerated names with a season restriction",
    },
    # Ruling N2, restated rather than inherited for the same reason the
    # follow-up rung restates it: a rung that quietly relaxed the unconditional
    # check would be a footnote wearing a rung's clothes.
    "ok_required_unconditionally": True,
    "owner_ruling": (
        "the owner ruled on 2026-09-14 that all three input defects be fixed "
        "and ONE rebuild run, having been told and having accepted that three "
        "causes in one rebuild cannot be separated afterwards"
    ),
    "declared_before_the_rebuild": True,
}

# Registered into the ONE table that enumerates a prefix's rungs. Written here
# rather than inline above because the cause string could not exist until rung
# 1's output had been measured; the table above carries a pointer so a reader
# scanning it does not conclude the prefix has two rungs.
RUNG_CAUSES_BY_PREFIX[PHASE331_RUNG_PREFIX][PHASE331_RUNG3] = PHASE331_RUNG3_CAUSE


# ---------------------------------------------------------------------------
# PHASE 33'S WAVE-14 LADDER -- A SECOND PREFIX, FOR THE REASON RECORDED AT :398
# (Plan 33-14 Task 3, D33-35, owner ruling of 2026-09-14).
#
# The prefix-aware block twelve screens up anticipated this exactly: "Plan 33-14
# wants to extend the SAME closed dict for the Elo rung. Racing for an integer
# now leaves the two phases colliding later; a prefix gives 33-14 the same seam
# instead of a contested key." This is that extension. Phase 30's four
# `RUNG_CAUSES` entries, `PHASE31_RUNG_PREFIX` and every `PHASE331_*` symbol are
# BYTE-UNTOUCHED, and the negative control in
# tests/integration/test_gold_rebuild_attribution.py -- rung 1 with NO prefix
# still resolves to CR-02 -- keeps passing, which is what proves this addition
# did not capture the old integer.
#
# TWO RUNGS, ONE CAUSE EACH, AND A RUNG 0 THAT IS NOT A REBUILD. Rung 0
# fingerprints gold exactly as Phase 33.1 left it, taken BEFORE anything is
# rebuilt: it is the control Phase 33.1 could not take in time, and the one
# artifact that cannot be recovered once rung 1 has overwritten gold.
# `require_rung_ladder(dir, 1, "p33_")` demands exactly `p33_rung0.json`.
#
# WHY TWO REBUILDS AND NOT ONE. Two separate causes sit between Phase 33.1's gold
# and the gold Plan 33-15 re-fits on. Merging them would make every column both
# can reach unattributable -- which is the failure Phase 33.1 spent a dedicated
# diagnosis recovering from, where 45 columns came back unattributed and the
# trigger for 40 of them is PERMANENTLY UNMEASURABLE because no copy of the prior
# gold survives. The owner was shown that precedent and ruled for the ladder.
#
# THE CLI NEEDS NO CHANGE. `--attribute-rung`'s choices come from
# `sorted(RUNG_CAUSES)`, which already accepts 1 and 2 under any prefix; adding a
# second way to say the same thing would be the drift this module keeps refusing.
# ---------------------------------------------------------------------------

PHASE33_RUNG_PREFIX: str = "p33_"
PHASE33_EXEMPTION_RUNG: int = 1
PHASE33_ELO_RUNG: int = 2

# DECLARED, AND DELIBERATELY NOT REGISTERED. Plan 33-14 provided for a third rung
# only on the branch where Task 1's measurement showed the precipitation
# partition fix moves a gold column. IT DOES NOT: the premise was measured in two
# independent halves and recorded in `tests.phase33_state.PRECIP_PARTITION_PREMISE`
# -- 0 of 6,499 rows moved across all twelve precipitation columns, maximum
# absolute delta 0.0, with a pre-fix control reporting the same zero -- and
# `PRECIP_PARTITION_PROMOTED_TO_ITS_OWN_RUNG` is False. The owner ruled
# `two-rung-ladder` on 2026-09-14 on that evidence.
#
# The integer and its cause are recorded so a later reader can see that the third
# rung was CONSIDERED and ruled out by measurement, rather than overlooked. It is
# absent from `RUNG_CAUSES_BY_PREFIX` and has no attribution function, because
# D33-35 forbids running an empty-diff rung to prove a null and shipping the
# machinery for a rebuild that will never run is the code analogue of exactly
# that. Registering it would also make `require_rung_ladder(dir, 3, "p33_")`
# demand a document describing a rebuild nobody performed.
PHASE33_PRECIP_RUNG: int = 3

PHASE33_EXEMPTION_RUNG_CAUSE: str = (
    "THE NORMALIZATION-EXEMPTION WIDENING (D33-34(a), .planning/WINDOWS.md rows "
    "33 and 40), and NOTHING else: the level-preservation exemption in "
    "scripts/build_features.normalize_combined_features widens from a filter "
    "that could only ever yield the single column `weather_coverage` to a "
    "PREDICATE resolving the nineteen weather indicator flags and the six "
    "sibling `*_coverage` columns alongside it. Those twenty-five reached gold "
    "z-scored into many distinct decimals -- `is_snow` carried 274 distinct "
    "values and `wind_moderate` 5,667, so the same snowy game read differently "
    "in week 3 than in week 15 -- and the six coverage siblings carried the "
    "mirror defect, where an uncovered row read far CLOSER to the covered level "
    "than to the uncovered one. No column is added, none removed, no row moves"
)

PHASE33_ELO_RUNG_CAUSE: str = (
    "THE ELO RE-DERIVATION (Plan 33-13, D33-34(c)), and NOTHING else: gold is "
    "rebuilt on the canonical 2002-2025 Elo chain re-derived in Wave 13, "
    "replacing the fabricated 0.0 Elo that 4,288 of 6,499 gold rows carried. It "
    "moves raw Elo, the derived differences and probabilities, the uncertainty "
    "pair, the rank and percentile columns and the momentum columns. No column "
    "is added, none removed, no row moves"
)

PHASE33_PRECIP_RUNG_CAUSE: str = (
    "NOT RUN. The precipitation one-hot partition fix on the live-forecast path "
    "(D33-34(b), WINDOWS.md row 39) would have taken this rung had it moved any "
    "gold column. It does not: measured READ-ONLY over the whole historical "
    "corpus under the fixed code, 0 of 6,499 rows moved in any of the twelve "
    "precipitation columns, because the 1,652 rows carrying a precip_prob are "
    "exactly the indoor games -- which take the dome branch and never reach the "
    "band helper -- while all 4,847 outdoor games carry no probability and take "
    "the mm-only branch the fix leaves byte-unchanged"
)

_PHASE33_LEVEL_PRESERVED_CACHE: tuple[str, ...] | None = None


def _gold_column_names(base_path: Path | None = None) -> tuple[str, ...]:
    """Every column name present in any gold matrix, read from the parquet SCHEMA.

    The schema rather than the frame: this is a NAME question, and reading three
    195-column frames to answer it would cost seconds for nothing. Strictly
    read-only with respect to ``data/``.
    """
    import pyarrow.parquet as pq

    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    names: set[str] = set()
    for matrix in GOLD_MATRICES:
        path = root / "gold" / f"{matrix}.parquet"
        if path.exists():
            names |= set(pq.read_schema(path).names)
    return tuple(sorted(names))


def phase33_level_preserved_family(base_path: Path | None = None) -> tuple[str, ...]:
    """THE EXEMPTION RUNG'S DECLARED FAMILY, resolved through the BUILDER's predicate.

    NOT A SECOND LIST. ``FeatureMatrixBuilder._level_preserved_columns`` is split
    into its two arms precisely so this function can call the SAME two arms over
    the same inputs the build sees, rather than re-expressing the rule here --
    which is the D30-02 second-list failure mode, and which
    ``_discrete_indicator_predicate`` twenty screens up already refuses for the
    CR-02 predicate on identical grounds.

    The two arms need different universes, and each gets the one it is about:

    * The SUFFIX arm is a NAME question, so it runs over the gold column set.
    * The INDICATOR arm is a VALUE question about the frame as it ENTERS
      normalization, so it runs over silver ``weather_features`` -- the frame the
      gold build merges. Judging discreteness on GOLD would be meaningless:
      normalization is the very thing that destroys it, which is the defect this
      rung exists to fix.

    Raises:
        ValueError: when the resolved family is empty, which would make the rung
            refuse every move it exists to attribute.
    """
    global _PHASE33_LEVEL_PRESERVED_CACHE
    if _PHASE33_LEVEL_PRESERVED_CACHE is not None:
        return _PHASE33_LEVEL_PRESERVED_CACHE

    from data.storage import load_dataframe
    from features.weather import WEATHER_FEATURE_COLUMNS_BY_BUILDER
    from scripts.build_features import FeatureMatrixBuilder

    weather = load_dataframe("weather_features", "silver", "parquet")
    declared = set(WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"])
    preserved = [
        column
        for column in weather.columns
        if column != "game_id" and column in declared
    ]

    family = tuple(
        sorted(
            set(
                FeatureMatrixBuilder._level_preserved_suffix_columns(
                    _gold_column_names(base_path)
                )
            )
            | set(
                FeatureMatrixBuilder._level_preserved_indicator_columns(
                    weather, preserved
                )
            )
        )
    )
    if not family:
        msg = (
            "the level-preservation predicate resolved to NO column, so the "
            "Phase-33 exemption rung would refuse every move it exists to "
            "attribute. Refusing to derive an empty family rather than judging "
            "against one."
        )
        raise ValueError(msg)
    _PHASE33_LEVEL_PRESERVED_CACHE = family
    return family


def phase33_elo_family() -> tuple[str, ...]:
    """THE ELO RUNG'S DECLARED FAMILY, from the ONE registry that owns it.

    Derived from ``features.elo_features.ELO_FEATURE_COLUMNS`` -- the tuple
    ``combine_features`` itself merges by -- rather than re-listed.

    NO WIDENING IS NEEDED, and that is a CORRECTION to Plan 33-14's own
    instruction, which said to derive this from the registry "and widen to the
    rank, percentile and momentum columns that registry does not name". MEASURED:
    the registry names all fourteen, ranks, percentiles and momentum included.
    The widening the plan provided for would have been a second hand-written list
    beside a registry that was already complete.

    The claim is CHECKED rather than assumed, because it is the one that matters:
    ``_add_rank_features`` (``features/elo_features.py:153-258``) computes ranks
    over the same-week snapshot POPULATION, so one game's corrected Elo moves the
    rank columns of EVERY OTHER GAME in that week. Rank, percentile and momentum
    therefore propagate FURTHER than the two raw Elo columns, and a family that
    omitted them would under-declare the rung's blast radius. If the registry
    ever stops naming them this refuses rather than silently narrowing.

    Raises:
        ValueError: when the registry names no rank, percentile or momentum
            column, so the declared family would under-state the blast radius.
    """
    from features.elo_features import ELO_FEATURE_COLUMNS

    family = tuple(ELO_FEATURE_COLUMNS)
    for marker in ("rank", "percentile", "momentum"):
        if not any(marker in _canonical(column) for column in family):
            msg = (
                f"features.elo_features.ELO_FEATURE_COLUMNS names no {marker!r} "
                "column, so the Phase-33 Elo rung's declared family would "
                "under-state its blast radius. Same-week ranks are computed over "
                "the whole snapshot population (features/elo_features.py:153-258), "
                "so one game's corrected Elo moves the rank columns of every other "
                "game that week. Refusing rather than declaring a family that "
                "cannot cover what the rung moves."
            )
            raise ValueError(msg)
    return family


PHASE33_EXEMPTION_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE33_EXEMPTION_RUNG,
    "prefix": PHASE33_RUNG_PREFIX,
    "cause": PHASE33_EXEMPTION_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to ONE declared family, resolved through the BUILDER's own "
        "level-preservation predicate rather than a second list: the columns "
        "phase33_level_preserved_family() names, pinned by name in "
        "tests.phase33_state.GOLD_LEVEL_PRESERVED_COLUMNS_33_14 and asserted "
        "equal in BOTH directions against live gold. ANY season, because a "
        "normalization change moves every season the column was normalized in"
    ),
    "declared_families": ("level_preserved",),
    "family_mechanisms": {
        "level_preserved": "predicate shared with the builder, pinned by name",
    },
    "declared_before_the_rebuild": True,
    "owner_ruling": (
        "the owner ruled 'two-rung-ladder' on 2026-09-14, having been shown "
        "Task 1's measured precipitation premise, both per-rung expected change "
        "sets, the re-anchored widths (195, 196, 195) and the statement that "
        "tripwire 2 stays red"
    ),
}

PHASE33_ELO_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE33_ELO_RUNG,
    "prefix": PHASE33_RUNG_PREFIX,
    "cause": PHASE33_ELO_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to ONE declared family, SOURCE-DERIVED from "
        "features.elo_features.ELO_FEATURE_COLUMNS: raw Elo, the derived "
        "differences and probabilities, the uncertainty pair, the rank and "
        "percentile columns and the momentum columns"
    ),
    "declared_families": ("elo",),
    "family_mechanisms": {"elo": "source-derived constant"},
    "declared_before_the_rebuild": True,
    # THE SLICE EXPECTATION IS A PREDICTION, NOT A GUARANTEE, and saying so in
    # the signature is what stops a later reader treating it as a promise. The
    # Elo builder RESETS at the start of its requested range and processes
    # chronologically (scripts/build_elo.py:142-205), so a corrupted 2018-start
    # chain can legitimately differ THROUGHOUT later seasons; and same-week ranks
    # are computed over the whole snapshot population, so one game's corrected
    # Elo moves the rank columns of every other game that week. Propagation past
    # the declared slice set is REPORTED WITH A WRITTEN EXPLANATION, never
    # hard-failed -- blocking on it would block a CORRECT rebuild for disagreeing
    # with a guess. The owner accepted exactly this framing on 2026-09-14.
    "expected_slices_are_a_prediction": True,
    "slice_propagation_is_reported_not_fatal": True,
}

PHASE33_PRECIP_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE33_PRECIP_RUNG,
    "prefix": PHASE33_RUNG_PREFIX,
    "cause": PHASE33_PRECIP_RUNG_CAUSE,
    "run": False,
    "why_not_run": (
        "Task 1 MEASURED the premise in two independent halves rather than "
        "asserting it. VALUE HALF: the full weather builder driven read-only over "
        "the whole historical corpus under the fixed code moved 0 of 6,499 rows "
        "in all twelve precipitation columns, maximum absolute delta 0.0, with a "
        "pre-fix control reporting the same zero. STRUCTURAL HALF: the gold build "
        "loads silver weather_features and never invokes the weather calculator. "
        "D33-35 forbids running an empty-diff rung to prove a null, and the owner "
        "ruled 'two-rung-ladder' on that measurement"
    ),
}

RUNG_CAUSES_BY_PREFIX[PHASE33_RUNG_PREFIX] = {
    PHASE33_EXEMPTION_RUNG: PHASE33_EXEMPTION_RUNG_CAUSE,
    PHASE33_ELO_RUNG: PHASE33_ELO_RUNG_CAUSE,
}


# ---------------------------------------------------------------------------
# RUNG 2 BECAME A DECLARATION-ONLY RUNG (owner ruling `skip-rung-2-as-already-
# landed`, 2026-09-14, at Plan 33-14's Task-4 blocking-human checkpoint).
#
# WHAT HAPPENED, MEASURED. Rung 1 was declared to carry ONE cause -- the
# normalization-exemption widening -- and it did move exactly the 25 of 26
# level-preserved columns it was supposed to. It ALSO moved eighteen columns
# outside that family: all fourteen `ELO_FEATURE_COLUMNS` members, plus the four
# situational spot flags that `features.contextual._look_ahead_flag` and
# `_letdown_flag` derive from opponent Elo. Rows carrying a fabricated 0.0 Elo
# fell from 4,288 to 16 -- and those 16 are ALL `(2002, 1)`, the burn-in
# boundary where no prior game exists, which is structurally explicable rather
# than fabricated.
#
# WHY, AND IT IS A PLANNING-ORDER FINDING RATHER THAN A RUNG THAT MISBEHAVED.
# Plan 33-13 re-derived the canonical Elo chain into SILVER during Wave 13.
# Gold was not rebuilt between that re-derivation and rung 1, so rung 1 was
# simply the FIRST gold build to consume it. The ladder separates causes by
# ORDER OF CODE CHANGE; this cause was a DATA change that had already landed.
# No ordering WITHIN this plan could have separated them -- only a gold rebuild
# taken BEFORE Wave 13 could have, which is a cross-plan ordering call nobody
# made.
#
# SO THERE IS NO SECOND REBUILD, AND CONSTRUCTING ONE WOULD BE DISHONEST TWICE
# OVER. A re-run of `--all-seasons` would now move nothing but the build clock:
# D33-35 forbids running an empty-diff rung to prove a null BY NAME, and
# `_phase33_structure`'s empty-diff refusal would read it as proof the rebuild
# did not do what it claimed. Constructing a synthetic rebuild to absorb the
# eighteen columns has the shape of absorbing a disclosure, which this phase
# prohibits.
#
# THE PRECEDENT IS PHASE 33.1'S FOLLOW-UP RUNG, and this reuses its shape
# exactly: a real ladder entry with its own cause that RE-JUDGES the SAME
# transition and rebuilds nothing. `p33_rung0.json -> p33_rung1.json` is
# re-judged under a SECOND declaration adding the two families rung 1 did not
# carry. That is why there is no `p33_rung2.json` fingerprint document --
# writing one would assert a rebuild that did not occur --- and
# `require_rung_ladder(dir, 2, "p33_")` demands rungs 0 and 1, which is exactly
# right: both exist and the chain is intact.
#
# WHAT IS AND IS NOT LOST. The two causes' column families are DISJOINT: 25
# level-preserved against 14 Elo plus 4 Elo-derived. Every moved column is still
# attributable to exactly ONE named cause, and nothing is unexplained. What was
# lost is the ladder's MECHANISM (two rebuilds, one cause each), not its PURPOSE
# (per-column attribution). This is NOT the Phase-33.1 residual, where 45
# columns came back unattributed and the trigger for 40 is PERMANENTLY
# UNMEASURABLE.
#
# NOTHING PRE-DECLARED IS EDITED. `PHASE33_ELO_RUNG_CAUSE` and
# `PHASE33_ELO_RUNG_EXPECTED_SIGNATURE` above stay BYTE-UNCHANGED as the record
# of what was predicted before either rebuild ran. The follow-up declaration
# below is ADDITIVE and says in its own fields that it was authored AFTER the
# diff -- which is precisely the distinction between a follow-up rung and a
# retroactive edit (Ruling N2, T-33.1-43).
# ---------------------------------------------------------------------------

# THE FOUR ELO-DERIVED SITUATIONAL COLUMNS, enumerated with the source that
# makes them attributable to the Elo cause rather than asserted to be.
#
# `features/contextual.py` computes both flags from OPPONENT ELO:
# `_look_ahead_flag` reads the next opponent's freeze-known Elo via
# `_freeze_known_elo` and compares it against this week's opponent Elo with
# `ELO_SPOT_STEP`; `_letdown_flag` does the same over the previous game. A
# changed Elo chain therefore changes WHICH games count as a trap or a letdown
# spot. They are neither weather nor coverage columns, so rung 1's exemption
# cannot reach them, and they are not in `ELO_FEATURE_COLUMNS`, so the Elo
# family cannot either -- which is why they need naming rather than inferring.
PHASE33_ELO_DERIVED_SITUATIONAL_COLUMNS: tuple[str, ...] = (
    "home_look_ahead_spot",
    "away_look_ahead_spot",
    "home_letdown_spot",
    "away_letdown_spot",
)

PHASE33_ELO_RUNG_FOLLOWUP_SIGNATURE: dict[str, object] = {
    "rung": PHASE33_ELO_RUNG,
    "prefix": PHASE33_RUNG_PREFIX,
    "cause": PHASE33_ELO_RUNG_CAUSE,
    "judges": (
        "the SAME p33_rung0 -> p33_rung1 transition, re-judged under a second "
        "declaration. No second rebuild happened and none will, which is why "
        "there is no p33_rung2.json fingerprint document"
    ),
    "no_new_rebuild": True,
    "authored_after_the_diff": True,
    "supersedes": (
        "PHASE33_ELO_RUNG_EXPECTED_SIGNATURE, which predicted rung 2 would be "
        "its own rebuild. That prediction is kept BYTE-UNCHANGED in source as "
        "the record of what was declared before either rebuild ran; only the "
        "framing is superseded, and the Elo causal family it named is unchanged "
        "and correct"
    ),
    "owner_ruling": (
        "the owner ruled `skip-rung-2-as-already-landed` on 2026-09-14, having "
        "been shown the exemption rung's attributed set (25 of 26), the 18 "
        "explained out-of-family columns, the widths (195, 196, 195) at every "
        "point, the fabricated-zero fall from 4,288 to 16 rows (all of them "
        "2002 week 1), and both findings. `run-rung-2-anyway` was rejected on "
        "the evidence as an empty-diff rung run to prove a null"
    ),
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "rung 1's declared family, PLUS two declared families rung 1 did not "
        "carry: (1) `elo` -- features.elo_features.ELO_FEATURE_COLUMNS, "
        "source-derived, covering raw Elo, the derived differences and "
        "probabilities, the uncertainty pair, the rank and percentile columns "
        "and the momentum columns; and (2) `elo_derived_situational` -- the "
        "four spot flags features.contextual derives from opponent Elo against "
        "ELO_SPOT_STEP. Everything else is refused exactly as at rung 1"
    ),
    "declared_families": (
        "carried_at_rung_1",
        "elo",
        "elo_derived_situational",
    ),
    "family_mechanisms": {
        "carried_at_rung_1": "predicate shared with the builder, pinned by name",
        "elo": "source-derived constant",
        "elo_derived_situational": "enumerated names with a source justification",
    },
    # NO SEASON RESTRICTION ON THE ELO FAMILIES, and its absence is a
    # declaration. `build_elo_with_snapshots` resets at the start of its range
    # and processes chronologically, and same-week ranks are computed over the
    # whole snapshot population, so a season restriction would refuse a CORRECT
    # rebuild for disagreeing with a guess. The OBSERVED slice set was all 24
    # seasons against a declared 2002-2017 plus (2018, 1); that propagation is
    # REPORTED and explained in tests.phase33_state, never gated here. The owner
    # accepted this framing on 2026-09-14.
    "expected_slices_are_a_prediction": True,
    "slice_propagation_is_reported_not_fatal": True,
    "ok_required_unconditionally": True,
}


# ---------------------------------------------------------------------------
# PHASE 33.2'S LADDER -- THE `p332_` PREFIX (Plan 33.2-08 Task 4, D33.2-20).
#
# NINE rungs, one cause each, numbered once for the whole phase in
# 33.2-08-PLAN.md <rung_allocation>: 1 odds correction (33.2-08), 2 international
# venues (33.2-09), 3 schedule moves (33.2-10), 4 day-before weather (33.2-12),
# 5 builder cutoffs (33.2-14), 6 snap/injury feeds (33.2-15), 7 opponent adjustment
# (33.2-16), 8 placeholders and the widened window (33.2-18), 9 market columns out
# of gold (33.2-19).
#
# THE REGISTRATION SHAPE, owned by this plan and consumed by every later rung:
#
# * Each rung declares exactly three names: PHASE332_{SUBJECT}_RUNG,
#   PHASE332_{SUBJECT}_RUNG_CAUSE and PHASE332_{SUBJECT}_RUNG_EXPECTED_SIGNATURE.
# * Registration is INCREMENTAL. This block assigns the cause table ONCE as a
#   single-key dict; every later rung adds ONE key to that dict in its own block,
#   its rung constant mapped to its cause constant (the :1100 precedent),
#   plus one entry in each of the two dispatch tables. Nothing pre-declares all
#   nine: a pre-declared cause that later changes is a cause table that lies.
# * TWO DISPATCH TABLES, PHASE332_RUNG_SIGNATURES and PHASE332_RUNG_ATTRIBUTORS
#   (the second is declared beside its first attributor, further down, because a
#   table cannot name a function before it exists). Both REFUSE an unregistered
#   rung by name.
# * ONE `p332_` branch at each of the two dispatch sites (`_expected_signature`,
#   `_attribute_one_matrix`), reading those tables. THE BRANCH IS LOAD-BEARING,
#   measured 2026-09-20: a registered prefix with no branch does NOT refuse -- it
#   falls through to the generic path, whose rung semantics are PHASE 30's keyed by
#   rung NUMBER. Rung 1 there attributes a changed column only when it was already
#   a discrete indicator (`_attribute_rung1`), and the market columns this rung
#   moves are continuous, so a CORRECT rebuild would report every one of them
#   unattributed. An unregistered prefix refuses in `_rung_causes`; an undispatched
#   one prints a verdict. Only the second is dangerous.
# ---------------------------------------------------------------------------

PHASE332_RUNG_PREFIX: str = "p332_"

PHASE332_ODDS_RUNG: int = 1

# THE COUNTS BELOW WERE EMITTED BY scripts/repair_odds_snapshot.py ON THE TABLE AS
# FOUND (2026-09-21, the Task-2 and Task-3 --apply runs), never carried from the plan
# or the CONTEXT. 33.2-CONTEXT.md D33.2-08 item 4 says "10" sign conflicts and "some"
# 1970 timestamps; the detectors printed SIGN_CONFLICTS_FOUND= 6 and
# EPOCH_1970_FOUND= 1855, and the CONTEXT is deliberately left unedited.
PHASE332_ODDS_RUNG_CAUSE: str = (
    "THE SILVER odds_snapshot CORRECTION of Plan 33.2-08 (SPEC R12, D33.2-08 item 4, "
    "D33.2-23), and NOTHING else: the 6 spread-sign conflicts its detector found "
    "(SIGN_CONFLICTS_FOUND= 6) settled against the owned odds_timeline first and "
    "ESPN's public odds archive second -- 7 values corrected with a citation "
    "(RESOLVED= 7), 5 set to unknown with a recorded reason (NULLED_WITH_REASON= 5), "
    "no row dropped; the disputed 28.5 total on 2023_W18_NYJ@NE confirmed against "
    "the archived DraftKings close; the 1970 created_at family (EPOCH_1970_FOUND= "
    "1855, CREATED_AT_REPAIRED= 0, CREATED_AT_NULLED= 1855) set to NULL; and the "
    "stray partition folders removed (STRAY_DIRS_REMOVED= 8) after nothing "
    "legitimate and missing was found to recover (RECOVERED= 0). Only the market "
    "columns of the games whose spread or moneyline changed can move. No column is "
    "added, none removed, no row moves"
)

# Where a corrected VALUE lands in gold. Keyed by gold column, valued by the
# odds_snapshot columns it is computed from (features.market_anchors.
# MarketAnchorFeaturesCalculator.build_features). created_at reaches no gold column.
PHASE332_ODDS_MARKET_SOURCES: dict[str, tuple[str, ...]] = {
    "snapshot_spread": ("spread",),
    "snapshot_total": ("total",),
    "snapshot_ml_prob_home_fair": ("ml_home", "ml_away"),
}

# DISCLOSED AT RUN TIME, NOT DECLARED BEFORE IT (owner ruling 2026-09-21, the rung-1
# attribution checkpoint). The first rung-1 comparison moved `target_ats` in exactly the
# seasons `snapshot_spread` moved, and it was missing from the pre-declared column list
# above. It is the arithmetic CHILD of that gold column -- `target_ats = point_differential
# - snapshot_spread` (scripts.build_features.create_target_variables) -- so it is the SAME
# cause, not a second one. The owner said yes to adding it, disclosed as found at run
# time. It is kept in its own table, beside and never inside the pre-declared one, so the
# record still shows what was declared before the rebuild and what was learned from it.
# Keyed by child gold column, valued by the parent gold column it is computed from.
PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN: dict[str, str] = {
    "target_ats": "snapshot_spread",
}

PHASE332_ODDS_RUN_TIME_DISCLOSURE: str = (
    "target_ats was NOT in the column list declared before the rebuild; it was found at "
    "run time (2026-09-21) and added by owner ruling. It is final margin minus the gold "
    "snapshot_spread column (target_ats = point_differential - snapshot_spread), so it "
    "moves wherever that column moves -- including 2022 and 2025, which hold no corrected "
    "game, because the gold snapshot_spread it subtracts is the NORMALIZED value and "
    "normalization carries a corrected spread into later seasons. It is attributed only "
    "where its parent snapshot_spread was itself attributed in the same matrix, and only "
    "in seasons where snapshot_spread moved"
)

# The committed correction record the attribution reads its corrected games from --
# ONE source for which values changed, never a second list here.
PHASE332_ODDS_CORRECTION_RECORD: Path = Path("config/odds_corrections.toml")

PHASE332_ODDS_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_ODDS_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_ODDS_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the three market columns snapshot_spread, snapshot_total and "
        "snapshot_ml_prob_home_fair, each only when a value of its own odds_snapshot "
        "source was CORRECTED OR NULLED in config/odds_corrections.toml (a confirmed "
        "value changes nothing), and only in seasons ON OR AFTER the earliest season "
        "holding such a value. The season rule is mechanical, not a guess: gold's "
        "normalization is expanding (compute_prior_season_stats over prior seasons, "
        "expanding_normalize over earlier weeks), so a corrected value moves its own "
        "season and feeds every later season's statistics, and can reach no earlier one"
    ),
    "declared_families": ("market",),
    "family_mechanisms": {
        "market": (
            "gold column -> odds_snapshot source columns, read against the committed "
            "correction record"
        ),
    },
    "declared_before_the_rebuild": True,
    "owner_ruling": (
        "the owner ratified both locked rulings on 2026-09-21 ('Confirm both' = "
        "ratify-locked): delete the stray folders, recovering zero rows; null an "
        "unsettleable price with a recorded reason and keep the game"
    ),
    # Everything above this key was declared before the rebuild ran and is unchanged.
    "disclosed_at_run_time": PHASE332_ODDS_RUN_TIME_DISCLOSURE,
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX] = {
    PHASE332_ODDS_RUNG: PHASE332_ODDS_RUNG_CAUSE
}

PHASE332_RUNG_SIGNATURES: dict[int, dict[str, object]] = {
    PHASE332_ODDS_RUNG: PHASE332_ODDS_RUNG_EXPECTED_SIGNATURE
}

# ---------------------------------------------------------------------------
# RETAKEN BASELINES (owner ruling 2026-09-21: "retake a stale baseline, never widen a
# rung's cause"; binding on every p332_ rung).
#
# A rung is judged against gold rebuilt from the CURRENT inputs with ONLY that rung's
# own cause undone. The first rung-0 document (`p332_rung0.json`) was a fingerprint of
# gold as it stood on disk, last built 2026-09-14 -- and Plan 33.2-05 had re-sorted
# `elo_game_snapshots` after that (commit bfe3b24, value-preserving). 2002 week-1 teams
# all rate 1500 and `_add_rank_features` breaks exact ties by row order, so a rebuild
# from today's inputs moves 2002 Elo rank/percentile (and 2003, through prior-season
# normalization) whatever the rung does. That movement is the earlier plan's, not the
# rung's.
#
# So a rung that retakes its baseline registers the retaken document here, built in a
# SCRATCH data root (never production data/) from today's inputs minus its own cause.
# The original rung document is KEPT, never overwritten: the difference between the two
# is recorded as a MEASURED pre-rung carry-in, attributed to the plan that caused it.
# A rung absent from this table is judged against its ladder predecessor as before.
# ---------------------------------------------------------------------------

PHASE332_RETAKEN_BASELINES: dict[int | str, str] = {
    PHASE332_ODDS_RUNG: f"{PHASE332_RUNG_PREFIX}rung0_retaken.json",
}

PHASE332_RETAKEN_BASELINE_REASONS: dict[int | str, str] = {
    PHASE332_ODDS_RUNG: (
        "RETAKEN 2026-09-21 by owner ruling. Gold rebuilt with `scripts/build_features.py "
        "--through-season 2025` (the rung's own build path and gates) in a SCRATCH data "
        "root holding a copy of today's production inputs, with ONLY the 12 odds values "
        "config/odds_corrections.toml marks corrected or nulled put back to their "
        "recorded old_value -- i.e. today's inputs minus exactly rung 1's cause. The "
        "original p332_rung0.json (gold on disk, last built 2026-09-14) is kept unchanged "
        "and was stale against its inputs"
    ),
}

# The measured cause of the difference between the original and the retaken rung-0
# documents. It is a carry-in from BEFORE the ladder's first rung, not part of any rung.
PHASE332_RUNG0_CARRY_IN_CAUSE: str = (
    "PRE-LADDER CARRY-IN, MEASURED 2026-09-21: Plan 33.2-05's value-preserving re-sort of "
    "silver elo_game_snapshots into (season, kickoff_et, game_id) order (commit bfe3b24), "
    "made after gold was last built (2026-09-14). Ratings are unchanged game by game, but "
    "every 2002 week-1 team is tied at 1500 and _add_rank_features breaks exact ties by "
    "row order, so 2002 Elo rank/percentile values change and carry into 2003 through "
    "season-to-season normalization. Recorded here, NOT inside rung 1"
)


# ---------------------------------------------------------------------------
# EXTRA STEPS: a separately-attributed rebuild that is NOT one of a ladder's numbered rungs
# (first used by Plan 33.2-10's orchestrator-assigned surface-classification fix, step
# "3b"). The numbered rungs are allocated once for the whole phase, so a fix that must run
# between two of them cannot take a number without colliding with a later plan's rung.
#
# An extra step has a STRING id (never an int, so it can never equal a numbered rung) and
# names the numbered rung it FOLLOWS. The ladder order is then: every numbered rung k, and
# directly after it every extra step that follows k. So the step is judged against the rung
# it follows, and the next numbered rung is judged against the step. Its cause lives in its
# own table rather than in RUNG_CAUSES_BY_PREFIX, whose integer keys the numbered-rung
# tooling sorts, takes the max of, and ranges over.
#
# Keyed by prefix, so the ladder helpers read them without an `if prefix ==` branch.
# ---------------------------------------------------------------------------

EXTRA_STEPS_BY_PREFIX: dict[str, dict[str, int]] = {}
EXTRA_STEP_CAUSES_BY_PREFIX: dict[str, dict[str, str]] = {}


def _is_extra_step(rung: int | str, prefix: str) -> bool:
    return isinstance(rung, str) and rung in EXTRA_STEPS_BY_PREFIX.get(prefix, {})


def _ladder_predecessors(rung: int | str, prefix: str = "") -> list[int | str]:
    """Every ladder entry that must exist before *rung*, in ladder order.

    A numbered rung N needs rungs 0 .. N-1 and every extra step following one of them. An
    extra step following F needs rungs 0 .. F, every extra step following an earlier rung,
    and every extra step that also follows F but was REGISTERED before it: steps sharing a
    rung run in registration order (step 3c, Plan 33.2-12, follows rung 3 after step 3b,
    so it is judged against step 3b and rung 4 against step 3c).

    Raises:
        ValueError: *rung* is a string that names no registered extra step.
    """
    steps = EXTRA_STEPS_BY_PREFIX.get(prefix, {})
    registered = list(steps)
    if isinstance(rung, str):
        if rung not in steps:
            msg = (
                f"Unknown extra step {rung!r} under rung prefix {prefix!r}. "
                f"Registered: {sorted(steps)}."
            )
            raise ValueError(msg)
        numbered = list(range(steps[rung] + 1))
        position = registered.index(rung)
        earlier = [
            s
            for s, follows in steps.items()
            if follows < steps[rung]
            or (follows == steps[rung] and registered.index(s) < position)
        ]
    else:
        numbered = list(range(rung))
        earlier = [s for s, follows in steps.items() if follows < rung]
    order = {k: (k, 0, 0) for k in numbered} | {
        s: (steps[s], 1, registered.index(s)) for s in earlier
    }
    return sorted(order, key=order.__getitem__)


def _step_cause(rung: int | str, prefix: str = "") -> str:
    """The cause of a numbered rung or of an extra step under *prefix*."""
    if _is_extra_step(rung, prefix):
        return EXTRA_STEP_CAUSES_BY_PREFIX[prefix][rung]
    return _rung_causes(prefix)[rung]


def phase332_baseline_document_path(directory: Path | str, rung: int | str) -> Path:
    """The document *rung* of the `p332_` ladder is judged AGAINST.

    The retaken baseline when the rung registered one in ``PHASE332_RETAKEN_BASELINES``,
    otherwise its ladder predecessor -- the last entry of ``_ladder_predecessors``, so a
    numbered rung that follows an extra step is judged against that step, and an extra
    step against the rung it follows. A registered baseline that is absent is REFUSED
    rather than silently replaced by the predecessor: falling back would judge the rung
    against the stale document the retake exists to replace.

    Raises:
        MissingPredecessorFingerprintError: a registered retaken baseline is absent.
    """
    name = PHASE332_RETAKEN_BASELINES.get(rung)
    if name is None:
        predecessor = _ladder_predecessors(rung, PHASE332_RUNG_PREFIX)[-1]
        return rung_document_path(directory, predecessor, PHASE332_RUNG_PREFIX)
    path = Path(directory) / name
    if not path.exists():
        msg = (
            f"Refusing to attribute p332_ rung {rung}: its registered retaken baseline "
            f"'{path}' does not exist. Build it in a scratch data root from today's "
            "inputs minus only this rung's cause; do not fall back to the stale "
            "predecessor document."
        )
        raise MissingPredecessorFingerprintError(msg)
    return path


def _phase332_table(kind: str, rung: int | str) -> tuple[dict, str]:
    """The `p332_` dispatch table of *kind* ("signatures" or "attributors") for *rung*.

    A numbered rung reads PHASE332_RUNG_*; a registered extra step reads
    PHASE332_EXTRA_STEP_*. ``_phase332_table_entry`` then refuses an unregistered entry.
    """
    extra = _is_extra_step(rung, PHASE332_RUNG_PREFIX)
    if kind == "signatures":
        if extra:
            return PHASE332_EXTRA_STEP_SIGNATURES, "PHASE332_EXTRA_STEP_SIGNATURES"
        return PHASE332_RUNG_SIGNATURES, "PHASE332_RUNG_SIGNATURES"
    if extra:
        return PHASE332_EXTRA_STEP_ATTRIBUTORS, "PHASE332_EXTRA_STEP_ATTRIBUTORS"
    return PHASE332_RUNG_ATTRIBUTORS, "PHASE332_RUNG_ATTRIBUTORS"


def _phase332_table_entry(table: dict, rung: int | str, table_name: str):
    """Read *rung* from a `p332_` dispatch table, REFUSING by name when it is absent.

    A rung that silently fell through to a default would be judged against another
    phase's semantics -- the failure the per-site branch exists to prevent.
    """
    try:
        return table[rung]
    except KeyError:
        msg = (
            f"p332_ rung {rung} is not registered in {table_name}. Every p332_ rung "
            "registers ONE entry in PHASE332_RUNG_SIGNATURES AND one in "
            "PHASE332_RUNG_ATTRIBUTORS (Plan 33.2-08 <owned_protocol_rung_registration>); "
            "a rung missing from either is refused rather than judged by a default."
        )
        raise ValueError(msg) from None


def phase332_odds_corrected_seasons(
    record_path: Path | str = PHASE332_ODDS_CORRECTION_RECORD,
) -> dict[str, int]:
    """Earliest season holding a changed value of each market column's source.

    Read from the committed correction record, so the attribution and the record
    cannot disagree about which games were corrected. A ``confirmed`` entry changed
    nothing and is excluded; a ``corrected`` or ``nulled`` one moved a stored value.

    Returns:
        ``gold column -> earliest season``, for the market columns that have at least
        one changed source value. A column absent here had NOTHING corrected.
    """
    import tomllib

    record = tomllib.loads(Path(record_path).read_text(encoding="utf-8"))
    earliest: dict[str, int] = {}
    for entry in record.get("correction", []):
        if entry.get("disposition") not in ("corrected", "nulled"):
            continue
        season = int(str(entry["game_id"])[:4])
        for gold_column, sources in PHASE332_ODDS_MARKET_SOURCES.items():
            if entry["column"] in sources:
                earliest[gold_column] = min(season, earliest.get(gold_column, season))
    return earliest


def _rung_causes(prefix: str = "") -> dict[int, str]:
    """The cause table *prefix* names.

    An UNKNOWN prefix is a refusal rather than a silent fall-back to Phase 30's
    causes: a typo that resolved to CR-02 would judge this rung against the wrong
    prediction and still print a verdict.
    """
    if prefix not in RUNG_CAUSES_BY_PREFIX:
        msg = (
            f"Unknown rung prefix {prefix!r}. Must be one of "
            f"{sorted(RUNG_CAUSES_BY_PREFIX)}."
        )
        raise ValueError(msg)
    return RUNG_CAUSES_BY_PREFIX[prefix]


def __getattr__(name: str):
    """Expose ``PHASE331_VENUE_FAMILY_COLUMNS`` as a lazily-derived constant.

    It READS like the module constant it is, and it COSTS nothing until something
    asks for it -- deriving it means constructing the contextual builder and
    running it over one synthetic game, and this module is also imported as a
    plain fingerprint reader that has no business pulling the feature stack.
    """
    if name == "PHASE331_VENUE_FAMILY_COLUMNS":
        return phase331_venue_family()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


class MissingPredecessorFingerprintError(RuntimeError):
    """A rung was asked to attribute before its ladder predecessor existed.

    A NAMED error rather than a bare ``FileNotFoundError`` traceback, because the two
    say different things. ``FileNotFoundError`` says a path was wrong; this says the
    LADDER WAS RUN OUT OF ORDER, which is a different mistake with a different fix.
    """


def rung_document_path(
    directory: Path | str, rung: int | str, prefix: str = ""
) -> Path:
    """Return the fingerprint document path for *rung* under *directory*.

    The *prefix* is what keeps two phases' ladders apart in one gitignored directory:
    Phase 30 wrote ``rung0.json`` .. ``rung4.json``, and Phase 31 writes
    ``p31_rung0.json`` .. under ``PHASE31_RUNG_PREFIX``.
    """
    return Path(directory) / f"{prefix}rung{rung}.json"


def require_rung_ladder(
    directory: Path | str, rung: int, prefix: str = ""
) -> list[Path]:
    """Verify every document rung 0 .. *rung* - 1 exists, and return them.

    RUNG 0 MUST EXIST. D31-09 describes rung 1 as a full rebuild proving the build
    still reproduces CURRENT gold -- which is only checkable against a fingerprint of
    current gold taken BEFORE rung 1 overwrote it. Run rung 1 first and that baseline
    is gone for good: the rebuild has already replaced the artifact it was supposed to
    be compared against, and no later step can recover it.

    The whole chain is required, not merely the immediate predecessor. A ladder is an
    ORDER, and a rung-2 attribution resting on a rung-1 document that was itself never
    judged against rung 0 is a chain with a link missing in the middle.

    Raises:
        MissingPredecessorFingerprintError: naming the first absent document.
    """
    verified: list[Path] = []
    for predecessor in _ladder_predecessors(rung, prefix):
        path = rung_document_path(directory, predecessor, prefix)
        if not path.exists():
            msg = (
                f"Refusing to attribute rung {rung}: its ladder predecessor "
                f"'{path}' does not exist. The rungs are an ORDER -- rung 0 is the "
                "fingerprint of CURRENT gold, taken BEFORE any rebuild overwrites "
                "it, and once a rebuild has run that baseline cannot be recovered. "
                f"Write it first with `--rung {predecessor}`"
                + (f" --rung-prefix {prefix}" if prefix else "")
                + ", then re-run this attribution."
            )
            raise MissingPredecessorFingerprintError(msg)
        verified.append(path)
    return verified


def assert_ladder_is_recoverable(
    directory: Path | str, rung: int, prefix: str = ""
) -> None:
    """Refuse BEFORE a rebuild if this ladder's baseline cannot still be recovered.

    ``require_rung_ladder`` already raises when a predecessor document is absent
    -- but it raises at ATTRIBUTION time, which is AFTER the rebuild has
    overwritten the gold the missing document was supposed to describe. At that
    moment the refusal is a diagnosis of an unrecoverable state; run here, before
    anything is rebuilt, it is a refusal the operator can still act on, because
    the recovery it names -- re-fingerprint CURRENT gold -- is only possible while
    current gold still stands.

    ``outputs/`` is gitignored, so an absent document is a reachable state rather
    than a hypothetical.

    THE PARSE CHECK IS THE ADDITION. A truncated or half-written JSON document
    passes ``exists()`` and fails at the comparison later -- at exactly the same
    unrecoverable moment an absent one would. Existence is not readability.

    Raises:
        MissingPredecessorFingerprintError: naming the first unusable document and
            carrying the command that rewrites it.
    """
    recovery = f"`--rung 0 --rung-prefix {prefix}`" if prefix else "`--rung 0`"
    for predecessor in _ladder_predecessors(rung, prefix):
        path = rung_document_path(directory, predecessor, prefix)
        preamble = (
            f"Refusing to REBUILD under rung prefix {prefix!r}: its ladder "
            f"baseline '{path}' "
        )
        remedy = (
            " Once a rebuild has run, the gold this document describes is gone and "
            "the attribution cannot be recovered at all. Re-write it NOW, while "
            f"current gold still stands, with {recovery}, then re-run the rebuild."
        )
        if not path.exists():
            raise MissingPredecessorFingerprintError(
                preamble + "does not exist." + remedy
            )
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MissingPredecessorFingerprintError(
                preamble + f"exists but does not parse as JSON ({error})." + remedy
            ) from error


# Rungs whose failures may legitimately be upstream drift rather than a wrong fix.
# Rung 4 is DELIBERATELY excluded: SPEC R2 makes an unexplained 2021-2024 move a
# hard blocker, and offering an escape there would let the phase talk itself past
# the one control it exists to run.
_UPSTREAM_ESCAPE_RUNGS = (1, 2, 3)

# Rungs at which a PROVEN value-preserving dtype change may be attributed. ALL FOUR,
# deliberately -- unlike _UPSTREAM_ESCAPE_RUNGS above, which stops at 3.
#
# The two are not the same kind of thing, and that is the whole reason they differ.
# Upstream drift is a HYPOTHESIS: the judge cannot check it, so it hands the reader a
# candidate cause to go verify, and offering that at rung 4 would let the phase talk
# itself past the one control it exists to run. The dtype rule is a PROOF, executed
# in-process, and it must clear three independent conditions before it attributes
# anything -- the loaded frame reproduces the AFTER document's digests (identity), the
# re-encode round-trips back to the frame's own values (losslessness, which is what
# stops a truncating float64 1.5 -> int32 1 cast from reproducing a hash while the
# value moved), and the PRIOR per-season digests return exactly in every season. A
# column that clears all three is byte-identical in value and therefore cannot be
# evidence of a whole-frame statistic leaking into a prior season, which is precisely
# what SPEC R2's control is looking for.
#
# Excluding rung 4 would not make the control stricter in any way that carries
# information; it would make it fire falsely on a replace_mode dtype restoration -- the
# exact mechanism seen at rung 3, where the append path's own silent upcast was undone.
# A control that fires falsely gets overridden, and overrides erode a gate faster than
# a well-proven exemption does. Owner ruled (D30-OWNER-10) after Plan 30-18 flagged its
# own conservative choice as arguably wrong on the merits.
_DTYPE_PROOF_RUNGS = (1, 2, 3, 4)

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


def _per_slice_digests(frame: pd.DataFrame, columns) -> dict[str, str]:
    """Return ONE digest per ``(season, week)`` slice over *columns*.

    ``_per_season_digests``'s idiom EXTENDED TO A WEEK KEY, not a second hasher:
    the same ``game_id`` ordering and the same ``_column_bytes`` encoding, so a
    slice digest is comparable in kind with everything else this module writes.

    WHY A WEEK KEY EXISTS AT ALL (Plan 33-14, cross-AI review finding). A
    SEASON-only constant cannot express "2018 week 1". Phase 33's Elo rung
    declares an expected change set of every season 2002-2017 PLUS the single
    pair ``(2018, 1)``, and a per-season instrument can only either claim all of
    2018 moved or claim none of it did. Both are false, and a declaration that
    cannot be evaluated is not a declaration.

    WHY ONE DIGEST PER SLICE RATHER THAN ONE PER COLUMN PER SLICE. The question a
    slice set answers is "did this rung reach these rows", which is a ROW
    question. WHICH columns moved is already answered per season by
    ``compare_fingerprints``, and the two together say more than either alone.

    Args:
        frame: A gold matrix carrying ``game_id``, ``season`` and ``week``.
        columns: The columns whose values the digest covers. Columns absent from
            *frame* are skipped, so a family naming a column this matrix does not
            carry degrades to the columns it does.

    Returns:
        A map from ``"{season}|{week}"`` to a 16-character digest.
    """
    present = [column for column in columns if column in frame.columns]
    ordered = frame.sort_values("game_id")
    digests: dict[str, str] = {}
    keys = list(
        zip(ordered["season"].to_numpy(), ordered["week"].to_numpy(), strict=True)
    )
    for season, week in sorted({(int(s), int(w)) for s, w in keys}):
        mask = (ordered["season"] == season) & (ordered["week"] == week)
        subset = ordered.loc[mask]
        payload = b"".join(
            _column_bytes(cast("pd.Series", subset[column])) for column in present
        )
        digests[f"{season}|{week}"] = hashlib.sha256(payload).hexdigest()[:16]
    return digests


def gold_slice_digests(columns, base_path: Path | None = None) -> dict[str, dict]:
    """Per-matrix ``(season, week)`` slice digests over *columns*, read-only.

    The companion to ``fingerprint_gold`` for a ladder that has to say WHICH
    ``(season, week)`` slices a rung moved. Like rung 0, it can only be taken
    while the gold it describes still stands.

    Args:
        columns: The causal column family to digest.
        base_path: Data root override; defaults to the configured one.

    Returns:
        ``{matrix: {"season|week": digest}}``, with a ``{"missing": True}`` entry
        for any matrix absent from disk.
    """
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
        result[matrix] = _per_slice_digests(
            pd.read_parquet(path, engine="pyarrow"), columns
        )
    return result


def changed_slices(before: dict, after: dict) -> dict[str, list[tuple[int, int]]]:
    """Return, per matrix, the ``(season, week)`` slices whose digest moved.

    Args:
        before: A ``gold_slice_digests`` document taken before the rebuild.
        after: The same, taken after.

    Returns:
        ``{matrix: [(season, week), ...]}``, sorted.
    """
    moved: dict[str, list[tuple[int, int]]] = {}
    for matrix in GOLD_MATRICES:
        b = before.get(matrix) or {}
        a = after.get(matrix) or {}
        if b.get("missing") or a.get("missing"):
            moved[matrix] = []
            continue
        keys = sorted(
            {
                (int(key.split("|")[0]), int(key.split("|")[1]))
                for key in set(b) | set(a)
            }
        )
        moved[matrix] = [
            (season, week)
            for season, week in keys
            if b.get(f"{season}|{week}") != a.get(f"{season}|{week}")
        ]
    return moved


def collapse_whole_seasons(
    moved: list[tuple[int, int]], universe: list[tuple[int, int]]
) -> list[tuple[int, int | None]]:
    """Collapse a moved slice list to ``(season, None)`` where EVERY week moved.

    The declared sets in ``tests.phase33_state`` use ``(season, None)`` to mean a
    whole season, so an observed set recorded only as concrete pairs could never
    be compared against them -- every pair would read as "outside the declared
    set" and demand an explanation for a season that was declared in full. This
    is what makes the two sets commensurable.

    A season is collapsed ONLY when every ``(season, week)`` slice the matrix
    carries for it moved. A season that moved in some weeks and not others keeps
    its individual pairs, because "most of 2019 moved" is a different fact from
    "2019 moved" and the difference is exactly what a partial move looks like.

    Args:
        moved: The observed moved slices.
        universe: Every ``(season, week)`` slice present at all.

    Returns:
        The collapsed set, sorted, with ``(season, None)`` for a whole season.
    """
    moved_set = set(moved)
    by_season: dict[int, set[int]] = {}
    for season, week in universe:
        by_season.setdefault(season, set()).add(week)

    collapsed: list[tuple[int, int | None]] = []
    for season in sorted(by_season):
        weeks = by_season[season]
        moved_weeks = {week for s, week in moved_set if s == season}
        if not moved_weeks:
            continue
        if moved_weeks == weeks:
            collapsed.append((season, None))
        else:
            collapsed.extend((season, week) for week in sorted(moved_weeks))
    return collapsed


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


def _expected_signature(
    rung: int, before: dict | None = None, prefix: str = ""
) -> dict:
    """Return the predicted ``compare_fingerprints`` diff shape for *rung*.

    When *before* (the pre-rung fingerprint document) is supplied, rung 3's expected
    removed-set is DERIVED per matrix from that document's column list rather than
    described. That is the difference between "every removed column looks like a
    line-movement column" and "the removed set IS the line-movement family".

    *prefix* selects the CAUSE TABLE (Ruling N). It defaults to the empty prefix, so
    every call written before Plan 33.1-07 resolves exactly where it always did --
    the negative control in the attribution tests asserts that rung 1 with no prefix
    is still CR-02, so the new table cannot have captured the old integer.
    """
    causes = _rung_causes(prefix)
    if rung not in causes and not _is_extra_step(rung, prefix):
        msg = (
            f"Unknown rung {rung!r} under rung prefix {prefix!r}. Must be one of "
            f"{sorted(causes)} or a registered extra step "
            f"{sorted(EXTRA_STEPS_BY_PREFIX.get(prefix, {}))}."
        )
        raise ValueError(msg)

    if prefix == PHASE332_RUNG_PREFIX:
        # Phase 33.2's rungs are judged against their OWN declared signature, read
        # from the dispatch table and returned as a COPY so a caller cannot edit
        # the prediction it is about to be judged against. Never the generic
        # Phase-30 path below, which keys rung semantics by NUMBER.
        table, name = _phase332_table("signatures", rung)
        return dict(_phase332_table_entry(table, rung, name))

    if prefix == PHASE33_RUNG_PREFIX:
        # Returned as a COPY, for the reason the Phase-33.1 branch below records:
        # a caller must not be able to edit the prediction it is about to be
        # judged against.
        if rung == PHASE33_ELO_RUNG:
            # The FOLLOW-UP signature, not the pre-declared one. Rung 2 became a
            # declaration-only rung by owner ruling; the pre-declared signature
            # stays byte-unchanged in source as the record of what was predicted.
            return dict(PHASE33_ELO_RUNG_FOLLOWUP_SIGNATURE)
        return dict(PHASE33_EXEMPTION_RUNG_EXPECTED_SIGNATURE)

    if prefix == PHASE331_RUNG_PREFIX:
        # The PRE-DECLARED change set, returned as a copy so a caller cannot edit
        # the prediction it is about to be judged against (T-33.1-43). The
        # follow-up rung has its OWN signature; rung 1's is byte-untouched by it,
        # which is the difference between a follow-up rung and a widened
        # declaration (Ruling N2).
        if rung == PHASE331_RUNG3:
            return dict(PHASE331_RUNG3_EXPECTED_SIGNATURE)
        if rung == PHASE331_FOLLOWUP_RUNG:
            return dict(PHASE331_FOLLOWUP_EXPECTED_SIGNATURE)
        return dict(PHASE331_EXPECTED_SIGNATURE)

    signature = {
        "rung": rung,
        "cause": causes[rung],
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
        # Whether this rung's criterion actually DISCRIMINATES per column. Rung 2's
        # cause (WR-06 refits every bound) admits any moved column, so its
        # attribution is a blanket one and it sets this False. `ok: True` there means
        # "nothing contradicted the rung's structural signature", NOT "every column
        # was individually explained" -- the distinction rung 2's first attempt made
        # expensive, attributing cleanly while having flattened 18 columns (WR-11).
        "discriminating": True,
        "attributed": [],
        "unattributed": [],
        # Per moved column: its move kind and the seasons attributed to it, so a rung
        # report can say "this column's storage moved in 2025 only" instead of "this
        # column changed, seasons: []". Populated for EVERY moved column, including
        # ones later split out into build_clock or value_preserving_dtype -- the split
        # changes how a column is judged, never whether it was reported.
        "move_kinds": {},
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


def _move_kind(column: str, meta: dict | None, seasons: list[str]) -> str:
    """Return *column*'s move kind, trusting the registered clock set over the document.

    A document may already carry ``move_kind`` (Plan 31-03 onward). It is honoured for
    ``values`` and ``storage`` only -- a ``build_clock`` claim is NEVER taken on the
    document's word, because a document that could name any column a clock would defeat
    the registered set the classification exists to be driven by (T-31-13b).
    """
    if _is_build_clock(column):
        return "build_clock"
    recorded = (meta or {}).get("move_kind")
    if recorded in ("values", "storage"):
        return recorded
    reasons = (meta or {}).get("reasons") or (["values"] if seasons else [])
    return "values" if "values" in reasons else "storage"


def _record_move_kinds(diff: dict, verdict: dict) -> None:
    """Record every moved column's kind and attributed seasons on *verdict*."""
    details = diff["details"] or {}
    for column, seasons in diff["changed"].items():
        meta = details.get(column) or {}
        verdict["move_kinds"][column] = {
            "kind": _move_kind(column, meta, list(seasons)),
            "seasons": sorted(seasons),
            "seasons_values": sorted(meta.get("seasons_values") or []),
            "seasons_storage": sorted(meta.get("seasons_storage") or []),
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
    ladder_directory: Path | str | None = None,
    rung_prefix: str = "",
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

    With *ladder_directory*, the rung refuses to attribute at all until every earlier
    rung document exists under *rung_prefix* -- see ``require_rung_ladder``. It is
    opt-in on the function because the tests judge hand-built reports that have no
    ladder on disk; the CLI always supplies it, so an operator cannot run the ladder
    out of order.

    Raises:
        MissingPredecessorFingerprintError: when *ladder_directory* is supplied and a
            predecessor rung document is absent.
    """
    if ladder_directory is not None:
        require_rung_ladder(ladder_directory, rung, rung_prefix)

    signature = _expected_signature(rung, before=before, prefix=rung_prefix)
    cause = _step_cause(rung, rung_prefix)
    # THE PHASE-33.1 RUNG GETS NO UPSTREAM-DRIFT ESCAPE, and the suppression is
    # scoped to the PREFIX rather than expressed by editing `_UPSTREAM_ESCAPE_RUNGS`
    # -- that tuple is Phase 30's record and rung 1 legitimately carries the escape
    # there. This rung rebuilds from SILVER, not from nflreadpy, so offering an
    # "upstream revision" candidate cause would send a reader hunting for something
    # that cannot be the explanation.
    upstream = (
        " " + _UPSTREAM_DRIFT_NOTE.format(cause=cause)
        if rung in _UPSTREAM_ESCAPE_RUNGS
        and rung_prefix not in (PHASE331_RUNG_PREFIX, PHASE332_RUNG_PREFIX)
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
        _record_move_kinds(diff, verdict)
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
            rung_prefix=rung_prefix,
        )

    ok = all(verdict["ok"] for verdict in matrices.values())
    non_clock_moves, build_clock_moves = _summarize_moves(matrices)
    return {
        "rung": rung,
        "cause": cause,
        "ok": ok,
        "blocking": bool(blocking),
        "signature": signature,
        "matrices": matrices,
        # The two DISJOINT lists, unioned across matrices, whose union is the whole
        # moved set. A rung condition can be asserted against either: "zero non-clock
        # moves" is reachable for a full rebuild, "zero moved columns" is not.
        "non_clock_moves": non_clock_moves,
        "build_clock_moves": build_clock_moves,
        "failures": failures,
    }


def _summarize_moves(matrices: dict) -> tuple[list[str], list[str]]:
    """Return (non_clock_moves, build_clock_moves) unioned across every matrix.

    Both are canonical and sorted, so a verdict cannot differ between two runs over
    identical data whose JSON key order happens to differ.
    """
    moved = {
        column for verdict in matrices.values() for column in verdict["move_kinds"]
    }
    return (
        sorted(column for column in moved if not _is_build_clock(column)),
        sorted(column for column in moved if _is_build_clock(column)),
    )


def _attribute_one_matrix(
    rung, detail, diff, verdict, fail, expected_removed=None, rung_prefix: str = ""
) -> bool:
    """Apply *rung*'s predicted signature to one matrix. Returns whether it blocks."""
    if rung_prefix == PHASE332_RUNG_PREFIX:
        table, name = _phase332_table("attributors", rung)
        attributor = _phase332_table_entry(table, rung, name)
        return attributor(detail, diff, verdict, fail)

    if rung_prefix == PHASE33_RUNG_PREFIX:
        if rung == PHASE33_ELO_RUNG:
            return _attribute_phase33_elo(detail, diff, verdict, fail)
        return _attribute_phase33_exemption(detail, diff, verdict, fail)

    if rung_prefix == PHASE331_RUNG_PREFIX:
        if rung == PHASE331_RUNG3:
            return _attribute_phase331_rung3(detail, diff, verdict, fail)
        if rung == PHASE331_FOLLOWUP_RUNG:
            return _attribute_phase331_followup(detail, diff, verdict, fail)
        return _attribute_phase331(detail, diff, verdict, fail)

    cause = _rung_causes(rung_prefix)[rung]
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
        _attribute_rung2(diff, verdict, fail)

    return blocking


_PHASE331_MISLABELLING_PROHIBITION = (
    "The SPEC prohibition this rung carries reads: MUST NOT label a gold rebuild "
    "'the weather rung' if it also carries Elo, bye-window or ats_edge changes -- "
    "that re-creates the attribution failure this phase exists to prevent. A "
    "column from one of those three families is therefore UNATTRIBUTED here, not "
    "absorbed. The remedy is a SEPARATE rung for the separate cause, never a "
    "footnote on this one."
)


def _phase331_structure(detail: dict, diff: dict, fail) -> bool:
    """The STRUCTURAL half of the Phase-33.1 prediction, shared by both rungs.

    EXTRACTED rather than copied when the follow-up rung was declared. The two
    rungs judge the SAME p331_rung0 -> p331_rung1 transition, so they must make
    the SAME structural prediction -- exactly one added column and it is the NAMED
    coverage flag, nothing removed, width +1, rows unchanged, and a non-empty
    diff. Two copies of that prediction could drift apart, and a follow-up rung
    whose structural claim had quietly diverged from the rung it follows would be
    judging a different rebuild than the one that ran.

    The added column is asserted by NAME in BOTH directions -- an unexpected
    addition fails, and so does the flag's ABSENCE -- so a build that added
    something unrelated while omitting the flag cannot satisfy the integer and
    pass.

    Returns:
        Whether this matrix BLOCKS the phase on structure alone.
    """
    blocking = False
    width_before = detail["width_before"]
    width_after = detail["width_after"]
    rows_before = detail.get("rows_before")
    rows_after = detail.get("rows_after")

    expected_added = [_canonical(PHASE331_ADDED_COLUMN)]
    for column in diff["added"]:
        if column not in expected_added:
            blocking = True
            fail(
                f"column '{column}' was ADDED at the Phase-33.1 rung, which adds "
                f"exactly ONE column -- the coverage flag "
                f"'{PHASE331_ADDED_COLUMN}'. The width integer alone cannot say "
                "WHICH column arrived, which is why the addition is pinned by name"
            )
    for column in expected_added:
        if column not in diff["added"]:
            fail(
                f"the coverage flag '{PHASE331_ADDED_COLUMN}' was NOT added at the "
                "Phase-33.1 rung. Without it a NULL observation is indistinguishable "
                "from a measured one, which is the whole point of the rung (SPEC R5)"
            )
    for column in diff["removed"]:
        blocking = True
        fail(
            f"column '{column}' was REMOVED at the Phase-33.1 rung; none of its "
            "three causes removes a column"
        )

    if width_after != width_before + len(expected_added):
        fail(
            f"width moved {width_before} -> {width_after} at the Phase-33.1 rung, "
            f"which is not {width_before} plus the "
            f"{len(expected_added)} declared added column(s)"
        )
    if rows_before != rows_after:
        fail(
            f"rows moved {rows_before} -> {rows_after} at the Phase-33.1 rung. The "
            "rung re-derives the SAME games -- the 207 restored rows are a silver "
            "staleness repair that gold already carried -- so a row-count move "
            "means the rebuild did something this rung did not declare"
        )

    if (
        not diff["changed"]
        and not diff["added"]
        and not diff["removed"]
        and not diff["renames"]
        and not diff["dtype_preserved"]
    ):
        fail(
            "the Phase-33.1 rung moved no column and added none. Replacing a "
            "fabricated 65.0F constant with real ERA5 observations across 4,847 "
            "outdoor games MUST move something; an empty diff means the rebuild "
            "did not do what it claimed"
        )

    return blocking


def _attribute_phase331(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """The Phase-33.1 rung: one added column, three declared families, `ok` or STOP.

    STRUCTURE FIRST. Exactly one column is added and it is the NAMED coverage flag;
    nothing is removed; width moves by exactly +1; rows are UNCHANGED, because this
    rung re-derives the same 6,499 games rather than adding any. The added column is
    asserted by NAME in BOTH directions -- an unexpected addition fails, and so does
    the flag's ABSENCE -- so a build that added something unrelated while omitting
    the flag cannot satisfy the integer and pass.

    THEN THE THREE FAMILIES, in order:

    1. A column in the PROHIBITED families (Elo, the team-form rolling family the
       bye-window rule governs, ``ats_edge``) is UNATTRIBUTED and BLOCKS, with the
       SPEC prohibition quoted. This is checked FIRST and the families are proven
       disjoint from the declared ones by a test, so a prohibited column cannot be
       absorbed by a declared family.
    2. ``features.weather.WEATHER_FEATURE_COLUMNS`` -- family 1.
    3. ``phase331_venue_family()`` -- family 2, the contextual builder's own emitted
       set. Ruling H puts ``venue_cold_climate`` here, not in the weather family:
       it is derived from the stadium's geography and from no weather observation.
    4. Everything else falls to the ROW-SCOPED staleness predicate: attributable
       ONLY IF every season it moved in is in ``PHASE331_STALENESS_SEASONS``. The
       season lists come from ``compare_fingerprints``, which derives them from the
       same per-season digests ``_per_season_digests`` computes -- so "byte-identical
       in 2002-2024" is a measurement here, not a description. A column that moved in
       2019 cannot have moved because 2025 gained rows, and it is UNATTRIBUTED.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {"weather": [], "venue": [], "staleness_2025": []}
    blocking = _phase331_structure(detail, diff, fail)
    weather = {_canonical(name) for name in phase331_weather_family()}
    venue = {_canonical(name) for name in phase331_venue_family()}
    staleness = {str(season) for season in PHASE331_STALENESS_SEASONS}

    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])

        prohibited = _phase331_prohibited_family(column)
        if prohibited is not None:
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' moved at the Phase-33.1 rung but belongs to the "
                f"PROHIBITED '{prohibited}' family. "
                + _PHASE331_MISLABELLING_PROHIBITION
            )
            continue

        if column in weather:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["weather"].append(column)
            continue
        if column in venue:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["venue"].append(column)
            continue

        outside = [season for season in seasons if season not in staleness]
        if seasons and not outside:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["staleness_2025"].append(column)
            continue

        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at the Phase-33.1 rung in season(s) "
            f"{', '.join(seasons) or '(none attributed)'} and belongs to NEITHER "
            "declared column family. The only remaining cause is the 2025 staleness "
            "repair, which is a ROW population: it can move a column in "
            f"{', '.join(sorted(staleness))} and in no other season, because those "
            "are the only rows that were absent. "
            + (
                f"This column ALSO moved in {', '.join(outside)}, which no declared "
                "cause reaches."
                if outside
                else "This column carries NO attributed season at all, so nothing "
                "can be said about when it moved."
            )
            + " Do NOT annotate past it: `ok` must be True unconditionally "
            "(Ruling N2), and a legitimate out-of-family move is a NEW declared "
            "family in a follow-up rung, never a footnote on this one"
        )

    return blocking


def _phase331_season_restricted(
    column: str,
    seasons: list[str],
    allowed: tuple[int, ...],
    family: str,
    verdict: dict,
    fail,
    note: str,
) -> bool:
    """Attribute *column* to *family* only if every season it moved in is *allowed*.

    THE SEASON RESTRICTION IS WHAT MAKES AN ENUMERATED FAMILY DISCRIMINATE. A
    bare name list would accept its members in ANY season, which is precisely the
    non-discriminating shape Ruling N2 refuses -- ``_attribute_rung2``'s own
    comment records that its blanket predicate "cannot FAIL on a moved column"
    while 18 columns had been silently destroyed. Under this helper a Group-1
    column that moved in 2019 is UNATTRIBUTED and blocks, exactly as an
    undeclared column would be, because 2019 is not a season the declared cause
    can reach.

    Returns:
        Whether the column was ATTRIBUTED.
    """
    allowed_labels = {str(season) for season in allowed}
    outside = [season for season in seasons if season not in allowed_labels]
    if seasons and not outside:
        verdict["attributed"].append(column)
        verdict["changed_by_family"][family].append(column)
        return True

    verdict["unattributed"].append(column)
    fail(
        f"column '{column}' is DECLARED in the Phase-33.1 follow-up rung's "
        f"'{family}' family, but that family is restricted to season(s) "
        f"{', '.join(sorted(allowed_labels))} and this column moved in "
        f"{', '.join(seasons) or '(no season at all)'}. "
        + note
        + " A declared family with no season restriction would accept its members "
        "in ANY season, which is the non-discriminating shape Ruling N2 refuses. "
        "The remedy is to understand the column, never to widen the restriction "
        "after the diff is seen."
    )
    return False


def _attribute_phase331_followup(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """The FOLLOW-UP rung: rung 1's families PLUS the three declared residual groups.

    WHAT THIS RUNG IS. Rung 1 returned ``ok=False`` with 45 unattributed columns.
    Ruling N2 permits exactly two responses -- STOP, or declare a NEW family in a
    FOLLOW-UP RUNG -- and forbids the third, a footnote on rung 1. The owner
    ordered the residual diagnosed first; the investigation ran eight controlled
    rebuilds, and this rung is the second response taken on that evidence. It
    re-judges the SAME transition and rebuilds nothing.

    WHAT IT DOES NOT DO. It does not widen rung 1's declaration: rung 1's
    signature, its three families and its verdict are byte-untouched, and
    ``attribute_rung(report, PHASE331_RUNG, rung_prefix=...)`` still returns
    ``ok=False`` on this diff. That is the whole difference between a follow-up
    rung and a retroactive edit, and a test asserts it.

    THE ORDER, and why the prohibition check moves:

    1. The three FOLLOW-UP families, by EXACT NAME plus a SEASON RESTRICTION.
       They are checked FIRST because two of them are populated by columns rung 1
       refused ON the prohibition -- checking the prohibition first would refuse
       the very columns this rung exists to declare.
    2. The PROHIBITED families, for everything NOT named above. Unchanged in
       force: an Elo, bye-window or ``ats_edge`` column that is not one of the 44
       enumerated names is still UNATTRIBUTED and still BLOCKS, with the SPEC
       prohibition quoted. Declaring 44 names does not open the family.
    3. Rung 1's own three families -- weather, venue, and the row-scoped 2025
       staleness predicate -- which attribute the 86 columns rung 1 already
       attributed, recorded here as ``carried_at_rung_1`` rather than re-derived
       into a new bucket.
    4. Everything else is UNATTRIBUTED and fails, exactly as at rung 1.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {
        "carried_at_rung_1": [],
        "stale_baseline_2024": [],
        "prohibited_family_2025": [],
        "weather_widening": [],
    }
    blocking = _phase331_structure(detail, diff, fail)

    weather = {_canonical(name) for name in phase331_weather_family()}
    venue = {_canonical(name) for name in phase331_venue_family()}
    staleness = {str(season) for season in PHASE331_STALENESS_SEASONS}
    stale_baseline = {
        _canonical(name) for name in PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS
    }
    prohibited_2025 = {
        _canonical(name) for name in PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS
    }
    widening = {
        _canonical(name): _canonical(source)
        for name, source in PHASE331_FOLLOWUP_WEATHER_WIDENING.items()
    }

    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])

        if column in stale_baseline:
            if not _phase331_season_restricted(
                column,
                seasons,
                PHASE331_FOLLOWUP_STALE_BASELINE_SEASONS,
                "stale_baseline_2024",
                verdict,
                fail,
                "The carry-forward sits in 2024 and reaches 2025 ONLY through "
                "season 2025's normalisation bootstrap on season 2024. Seasons "
                "2002-2019 are structurally incapable of moving (team-form silver "
                "starts in 2020) and 2020-2023 were measured byte-identical, so a "
                "move anywhere else is a DIFFERENT cause wearing a declared name.",
            ):
                blocking = True
            continue

        if column in prohibited_2025:
            if not _phase331_season_restricted(
                column,
                seasons,
                PHASE331_FOLLOWUP_PROHIBITED_2025_SEASONS,
                "prohibited_family_2025",
                verdict,
                fail,
                "These four are declared BECAUSE they satisfy rung 1's 2025 "
                "staleness predicate and were refused only on the prohibited-family "
                "check, which fires first by design. A move outside 2025 would mean "
                "they are not that, and the prohibition would be right again.",
            ):
                blocking = True
            continue

        if column in widening:
            source = widening[column]
            if source in weather:
                verdict["attributed"].append(column)
                verdict["changed_by_family"]["weather_widening"].append(column)
                continue
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' is declared in the follow-up rung's "
                f"weather-widening family as an un-normalized copy of "
                f"'{source}', but '{source}' is NOT in "
                "features.weather.WEATHER_FEATURE_COLUMNS. The widening family is "
                "SOURCE-DERIVED: each entry is attributable only because the column "
                "it copies is a registered weather column. An entry whose source is "
                "not a weather column would turn the family into a "
                "general-purpose bucket, which is exactly what Ruling N2 forbids"
            )
            continue

        prohibited = _phase331_prohibited_family(column)
        if prohibited is not None:
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' moved at the Phase-33.1 FOLLOW-UP rung but "
                f"belongs to the PROHIBITED '{prohibited}' family and is NOT one of "
                "the names this rung declares. " + _PHASE331_MISLABELLING_PROHIBITION
            )
            continue

        if column in weather or column in venue:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["carried_at_rung_1"].append(column)
            continue

        outside = [season for season in seasons if season not in staleness]
        if seasons and not outside:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["carried_at_rung_1"].append(column)
            continue

        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at the Phase-33.1 FOLLOW-UP rung in "
            f"season(s) {', '.join(seasons) or '(none attributed)'} and belongs to "
            "NEITHER rung 1's three families NOR any of the three this rung "
            "declares. A follow-up rung declares a BOUNDED residual measured off a "
            "diagnosis; it is not an open bucket, and an undeclared column is "
            "refused here exactly as it was at rung 1. `ok` must be True "
            "unconditionally (Ruling N2)"
        )

    return blocking


def _phase331_rung3_structure(detail: dict, diff: dict, fail) -> bool:
    """The STRUCTURAL half of rung 3's prediction: the SHAPE does not move at all.

    NOT SHARED WITH ``_phase331_structure``, and the difference is the point.
    Rungs 1 and 2 predicted a column ARRIVING -- the coverage flag -- so their
    structural claim is width +1. Rung 3 corrects what three existing column
    families CONTAIN, so its claim is that nothing is added, nothing is removed
    and the width does not move. Sharing one helper would have meant one of the
    two rungs judging against the other's prediction.

    Returns:
        Whether this matrix BLOCKS the phase on structure alone.
    """
    blocking = False
    width_before = detail["width_before"]
    width_after = detail["width_after"]
    rows_before = detail.get("rows_before")
    rows_after = detail.get("rows_after")

    for column in diff["added"]:
        blocking = True
        fail(
            f"column '{column}' was ADDED at the Phase-33.1 rung 3, which adds "
            "NO column. This rung corrects what three existing families "
            "CONTAIN -- discarded rainfall, a z-scored coverage flag and an "
            "unbuilt 2025 -- and none of those three arrivals is a new column"
        )
    for column in diff["removed"]:
        blocking = True
        fail(
            f"column '{column}' was REMOVED at the Phase-33.1 rung 3; none of "
            "its three causes removes a column"
        )

    if width_after != width_before:
        fail(
            f"width moved {width_before} -> {width_after} at the Phase-33.1 "
            "rung 3, which predicts an UNCHANGED width"
        )
    if rows_before != rows_after:
        fail(
            f"rows moved {rows_before} -> {rows_after} at the Phase-33.1 rung 3. "
            "The rung re-derives the SAME games from the SAME silver row "
            "population -- only the values three families carry change -- so a "
            "row-count move means the rebuild did something this rung did not "
            "declare"
        )

    if (
        not diff["changed"]
        and not diff["added"]
        and not diff["removed"]
        and not diff["renames"]
        and not diff["dtype_preserved"]
    ):
        fail(
            "the Phase-33.1 rung 3 moved no column at all. Twenty weather "
            "columns were NULL for 4,847 outdoor games and the coverage flag "
            "read 0.0 on all 6,499 rows; recovering them MUST move something, "
            "and an empty diff means the rebuild did not do what it claimed"
        )

    return blocking


def _attribute_phase331_rung3(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 3: three input corrections, three declared families, `ok` or STOP.

    THE ORDER, and why the team-strength family is checked FIRST:

    1. ``PHASE331_RUNG3_TEAM_STRENGTH_2025_COLUMNS``, by EXACT NAME plus a
       SEASON RESTRICTION of 2025. Checked first because all twelve match the
       ``_off_rolling_`` / ``_def_rolling_`` marker and therefore belong to the
       PROHIBITED bye-window family. Checking the prohibition first would refuse
       the very columns this rung exists to declare -- the same ordering, and
       the same reason, as the follow-up rung's two enumerated families.
    2. ``PHASE331_FOLLOWUP_WEATHER_WIDENING``, the SOURCE-DERIVED mapping from
       an un-normalized copy to the registry column it copies, re-checked
       against the live registry rather than trusted.
    3. The PROHIBITED families, for everything not named above. Unchanged in
       force: an Elo, bye-window or ``ats_edge`` column that is not one of the
       twelve is still UNATTRIBUTED and still BLOCKS.
    4. ``features.weather.WEATHER_FEATURE_COLUMNS``, any season. The
       precipitation correction replaces a value that was discarded in every
       season, so a season restriction here would be a prediction this rung has
       no basis for.
    5. Everything else is UNATTRIBUTED and fails.

    THERE IS NO ROW-SCOPED STALENESS PREDICATE AT THIS RUNG. Rung 1 carried one
    because it restored 207 absent rows; this rung restores none -- the silver
    row population is identical on both sides -- so offering that escape would
    hand a moved column a cause that cannot be true here.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {
        "weather": [],
        "weather_widening": [],
        "team_strength_2025": [],
    }
    blocking = _phase331_rung3_structure(detail, diff, fail)

    weather = {_canonical(name) for name in phase331_weather_family()}
    team_strength = {
        _canonical(name) for name in PHASE331_RUNG3_TEAM_STRENGTH_2025_COLUMNS
    }
    widening = {
        _canonical(name): _canonical(source)
        for name, source in PHASE331_FOLLOWUP_WEATHER_WIDENING.items()
    }

    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])

        if column in team_strength:
            if not _phase331_season_restricted(
                column,
                seasons,
                PHASE331_RUNG3_TEAM_STRENGTH_SEASONS,
                "team_strength_2025",
                verdict,
                fail,
                "Season 2025 was the ONLY season the per-game pool omitted, and "
                "a read-only probe of the real adjustment stage under both "
                "pools measured ZERO moved rows in 2018-2024 (7,476 per-game "
                "and 8,064 rolling rows compared, max delta 0.0) BEFORE this "
                "declaration was written. A move in another season is a "
                "DIFFERENT cause wearing a declared name.",
            ):
                blocking = True
            continue

        if column in widening:
            source = widening[column]
            if source in weather:
                verdict["attributed"].append(column)
                verdict["changed_by_family"]["weather_widening"].append(column)
                continue
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' is declared in the weather-widening family "
                f"as an un-normalized copy of '{source}', but '{source}' is NOT "
                "in features.weather.WEATHER_FEATURE_COLUMNS. The family is "
                "SOURCE-DERIVED: each entry is attributable only because the "
                "column it copies is a registered weather column"
            )
            continue

        prohibited = _phase331_prohibited_family(column)
        if prohibited is not None:
            blocking = True
            verdict["unattributed"].append(column)
            fail(
                f"column '{column}' moved at the Phase-33.1 rung 3 but belongs "
                f"to the PROHIBITED '{prohibited}' family and is NOT one of the "
                "twelve names this rung declares. " + _PHASE331_MISLABELLING_PROHIBITION
            )
            continue

        if column in weather:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["weather"].append(column)
            continue

        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at the Phase-33.1 rung 3 in season(s) "
            f"{', '.join(seasons) or '(none attributed)'} and belongs to NONE "
            "of its three declared families. This rung restores no absent row, "
            "so there is no row-scoped staleness cause to fall back on -- "
            "unlike rung 1, where 207 rows genuinely arrived. `ok` must be True "
            "unconditionally (Ruling N2): the remedy is to understand the "
            "column, never to widen this declaration after the diff is seen"
        )

    return blocking


def _phase33_structure(detail: dict, diff: dict, fail, label: str, what: str) -> bool:
    """The STRUCTURAL half both Phase-33 rungs predict: the SHAPE does not move.

    Shared by the two rungs because they make the SAME structural claim, unlike
    Phase 33.1's rungs 1 and 3 -- which is exactly why that phase needed two
    helpers and this one needs one. Neither Phase-33 rung adds a column, removes
    one, changes a width or moves a row: one re-runs a normalization stage and
    the other re-derives an existing family's values.

    A structural surprise BLOCKS, while an out-of-family column move is a
    FINDING. The asymmetry is deliberate and it is the plan's: a column arriving
    or leaving is a change nobody declared and cannot be explained after the
    fact, whereas a moved column outside the declared family is required to carry
    a written explanation and is reported rather than gated.

    Args:
        detail: One matrix's ``compare_fingerprints`` entry.
        diff: The normalized diff for that matrix.
        fail: The per-matrix failure recorder.
        label: How to name this rung in a message.
        what: One clause saying what the rung must have moved, used by the
            empty-diff refusal.

    Returns:
        Whether this matrix BLOCKS the phase on structure alone.
    """
    blocking = False
    width_before = detail["width_before"]
    width_after = detail["width_after"]
    rows_before = detail.get("rows_before")
    rows_after = detail.get("rows_after")

    for column in diff["added"]:
        blocking = True
        fail(
            f"column '{column}' was ADDED at {label}, which adds NO column. This "
            "rung changes what an EXISTING family contains; an arriving column "
            "is a structural change nobody declared"
        )
    for column in diff["removed"]:
        blocking = True
        fail(f"column '{column}' was REMOVED at {label}; this rung removes none")

    if width_after != width_before:
        fail(
            f"width moved {width_before} -> {width_after} at {label}, which "
            "predicts an UNCHANGED width"
        )
    if rows_before != rows_after:
        fail(
            f"rows moved {rows_before} -> {rows_after} at {label}. The rung "
            "re-derives the SAME games from the SAME silver row population, so a "
            "row-count move means the rebuild did something this rung did not "
            "declare"
        )

    if (
        not diff["changed"]
        and not diff["added"]
        and not diff["removed"]
        and not diff["renames"]
        and not diff["dtype_preserved"]
    ):
        fail(
            f"{label} moved no column at all. {what} MUST move something, and an "
            "empty diff means the rebuild did not do what it claimed"
        )

    return blocking


def _attribute_phase33_exemption(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """The exemption rung: ONE declared family, resolved through the builder's predicate.

    A moved column inside ``phase33_level_preserved_family()`` is attributed. A
    moved column outside it is UNATTRIBUTED and fails, but does NOT block: the
    plan's instruction is to PARTITION rather than gate, with every out-of-family
    move carrying a written explanation in
    ``tests.phase33_state.GOLD_REBUILD_UNEXPLAINED_CHANGES_EXEMPTION``. A rung
    that hard-failed here would stop the ladder on the very observation it exists
    to record.

    NO SEASON RESTRICTION, and its absence is a declaration rather than an
    oversight. A normalization change moves every season the column was
    normalized in, so restricting the family to a season list would be a
    prediction this rung has no basis for -- the same reasoning Phase 33.1's
    rung 3 recorded for its own weather family.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {"level_preserved": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "the Phase-33 exemption rung",
        "Twenty-five level-bearing columns were z-scored into many distinct "
        "decimals -- is_snow alone carried 274 -- and returning them to their "
        "recorded levels",
    )
    family = {_canonical(name) for name in phase33_level_preserved_family()}

    for column in sorted(diff["changed"]):
        if column in family:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["level_preserved"].append(column)
            continue
        seasons = sorted(diff["changed"][column])
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at the Phase-33 exemption rung in "
            f"season(s) {', '.join(seasons) or '(none attributed)'} but is NOT a "
            "member of the level-preservation family the builder's own predicate "
            "resolves. The rung's ONE cause is the level exemption widening, "
            "which cannot reach a column it does not exempt. This is a FINDING "
            "rather than a block: record it in "
            "GOLD_REBUILD_UNEXPLAINED_CHANGES_EXEMPTION with a written "
            "explanation naming the causal column and why the normalization "
            "change could reach it. Do NOT construct a second rung to absorb "
            "it -- inventing a rung for an unexpected change is the shape of "
            "absorbing a disclosure"
        )

    return blocking


def _attribute_phase33_elo(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """The Elo rung, as a DECLARATION-ONLY follow-up over the rung-1 transition.

    WHAT THIS RUNG IS. It re-judges the SAME ``p33_rung0 -> p33_rung1``
    transition under a second declaration, and rebuilds nothing. Rung 1's own
    declaration and its verdict are BYTE-UNTOUCHED by it --
    ``attribute_rung(report, PHASE33_EXEMPTION_RUNG, ...)`` still returns
    ``ok=False`` on this diff with the same eighteen unattributed columns. That
    is the whole difference between a follow-up rung and a retroactive edit, and
    it is Phase 33.1's precedent applied unchanged.

    THE THREE FAMILIES:

    1. ``elo`` -- ``features.elo_features.ELO_FEATURE_COLUMNS``, source-derived,
       which already names the rank, percentile and momentum columns. Those are
       the ones that propagate FURTHER than raw Elo, because ``_add_rank_features``
       computes ranks over the same-week snapshot POPULATION, so one game's
       corrected Elo moves the rank columns of every other game that week.
    2. ``elo_derived_situational`` -- the four spot flags
       ``features.contextual`` derives from opponent Elo against
       ``ELO_SPOT_STEP``. The SAME cause, one step downstream. They are in
       neither the Elo registry nor the exemption family, which is exactly why
       they must be named rather than inferred.
    3. ``carried_at_rung_1`` -- the level-preservation family, recorded under its
       own label rather than re-derived into a new bucket, so a reader can see at
       a glance which columns this rung EXPLAINS and which it merely INHERITS.

    Everything else is UNATTRIBUTED and fails, exactly as at rung 1. A follow-up
    rung declares a BOUNDED residual measured off a diagnosis; it is not an open
    bucket.

    NO SEASON RESTRICTION ON EITHER ELO FAMILY, and the reason is load-bearing.
    ``build_elo_with_snapshots`` RESETS the Elo system at the start of its
    requested range and processes chronologically
    (``scripts/build_elo.py:142-205``), so a corrupted 2018-start chain can
    legitimately differ throughout later seasons -- and it did: the observed
    slice set is all 24 seasons against a declared 2002-2017 plus ``(2018, 1)``.
    A season restriction would refuse a CORRECT rebuild for disagreeing with a
    guess. The slice comparison happens against the DECLARED set in
    ``tests.phase33_state``, where every out-of-set slice carries a written
    explanation; it is deliberately not a gate here.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {
        "carried_at_rung_1": [],
        "elo": [],
        "elo_derived_situational": [],
    }
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "the Phase-33 Elo rung",
        "Replacing a fabricated 0.0 Elo on 4,288 of 6,499 rows with the "
        "re-derived 2002-2025 chain",
    )
    elo = {_canonical(name) for name in phase33_elo_family()}
    situational = {_canonical(name) for name in PHASE33_ELO_DERIVED_SITUATIONAL_COLUMNS}
    carried = {_canonical(name) for name in phase33_level_preserved_family()}

    for column in sorted(diff["changed"]):
        if column in elo:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["elo"].append(column)
            continue
        if column in situational:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["elo_derived_situational"].append(column)
            continue
        if column in carried:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["carried_at_rung_1"].append(column)
            continue
        seasons = sorted(diff["changed"][column])
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved across the p33_rung0 -> p33_rung1 "
            f"transition in season(s) {', '.join(seasons) or '(none attributed)'} "
            "and belongs to NEITHER rung 1's level-preservation family NOR "
            "either of the two this follow-up rung declares. A follow-up rung "
            "declares a BOUNDED residual measured off a diagnosis; it is not an "
            "open bucket, and an undeclared column is refused here exactly as it "
            "was at rung 1. Record it in GOLD_REBUILD_UNEXPLAINED_CHANGES_ELO "
            "with a written explanation. Do NOT construct another rung to absorb "
            "it -- inventing a rung for an unexpected change is the shape of "
            "absorbing a disclosure"
        )

    return blocking


def _attribute_p332_odds(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 1 of the `p332_` ladder: the silver odds correction's OWN attribution.

    A changed column is attributed ONLY when it is one of the market columns whose
    odds_snapshot source had a value corrected or nulled in the committed record,
    AND every season it moved in is on or after the earliest season holding such a
    value (gold normalization is expanding, so a corrected value can reach its own
    and later seasons and no earlier one). Anything else is UNATTRIBUTED and fails:
    it is never absorbed, and the cause is never widened to fit it.

    NOT `_attribute_rung1`, deliberately. Phase 30's rung 1 attributes a column only
    when it was already a discrete indicator; these market columns are continuous,
    so that judge would report a correct rebuild as entirely unattributed.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"market": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "p332_ rung 1 (the odds correction)",
        "Correcting and nulling spread and moneyline values in silver odds_snapshot",
    )
    earliest = {
        _canonical(column): season
        for column, season in phase332_odds_corrected_seasons().items()
    }
    children = {
        _canonical(child): _canonical(parent)
        for child, parent in PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN.items()
    }

    # Parents first, so a disclosed child can be judged against its parent's verdict.
    for column in sorted(set(diff["changed"]) - set(children)):
        seasons = sorted(diff["changed"][column])
        floor = earliest.get(column)
        if floor is not None and seasons and all(int(s) >= floor for s in seasons):
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["market"].append(column)
            continue
        verdict["unattributed"].append(column)
        if floor is None:
            why = (
                "it is not a market column whose odds_snapshot source was corrected "
                "or nulled in config/odds_corrections.toml"
            )
        else:
            why = (
                f"it moved in season(s) before {floor}, the earliest season holding a "
                "corrected value of its source, which expanding normalization cannot reach"
            )
        fail(
            f"column '{column}' moved at p332_ rung 1 in season(s) "
            f"{', '.join(seasons) or '(none attributed)'}, but {why}. The rung's ONE "
            "cause is the silver odds correction; do NOT widen it to fit this diff"
        )

    # Run-time-disclosed children (PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN): attributed
    # ONLY when the parent gold column was attributed in THIS matrix and every season the
    # child moved in is a season the parent moved in. A child moving without its parent
    # has no arithmetic path from the correction and is unattributed.
    for column in sorted(set(diff["changed"]) & set(children)):
        seasons = sorted(diff["changed"][column])
        parent = children[column]
        parent_seasons = set(diff["changed"].get(parent, []))
        if (
            parent in verdict["attributed"]
            and seasons
            and set(seasons) <= parent_seasons
        ):
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["market"].append(column)
            continue
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at p332_ rung 1 in season(s) "
            f"{', '.join(seasons) or '(none attributed)'}, but it is disclosed only as "
            f"the arithmetic child of '{parent}', which was not attributed here in "
            "those seasons. The rung's ONE cause is the silver odds correction; do NOT "
            "widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_ATTRIBUTORS: dict[int, Callable[..., bool]] = {
    PHASE332_ODDS_RUNG: _attribute_p332_odds
}


# ---------------------------------------------------------------------------
# p332_ RUNG 2 -- THE 2025 INTERNATIONAL VENUES (Plan 33.2-09 Task 3, SPEC R8 venue half).
#
# Registered per Plan 33.2-08's <owned_protocol_rung_registration>: three names, one
# incremental cause line, one entry in each dispatch table, and NO new `if prefix ==`
# branch -- the `p332_` branch at each dispatch site reads the tables. Without the table
# entries the registered prefix would fall through to Phase 30's `_attribute_rung2`,
# which still prints a verdict.
#
# DECLARED BEFORE THE REBUILD RAN. The explainable set is DERIVED from source rather than
# listed by hand (see `phase332_stadium_dependent_columns`): the contextual builder is the
# only gold builder that reads `stadium_id`, and the columns that are a function of it are
# found by running that builder over one synthetic game placed at every venue in
# data/venues.json and keeping the columns whose value differs between venues.
#
# WEATHER IS NOT IN THE SET, AND THAT IS THE PREDICTION. Gold's weather columns are read
# from silver `weather_features` by game_id; the gold build never reads `stadium_id` for
# them, and this rung changes no silver weather row. So the weather family must NOT move
# here. The seven games' silver weather rows still describe the US stadiums the feed
# named (the Sao Paulo and Berlin games are marked indoor and carry no weather at all);
# that is replaced at rung 4 by the day-before forecast backfill, which reads the
# corrected `stadium_id` -- and keeping the two causes apart is why this rung runs first.
# ---------------------------------------------------------------------------

PHASE332_VENUE_RUNG: int = 2

PHASE332_VENUE_CORRECTION_RECORD: Path = Path(
    "config/international_venue_corrections.toml"
)

# The one season the correction touches. Gold normalization is expanding within a
# season and bootstraps from the prior season only, and 2025 is the last season a
# ladder rung builds (--through-season 2025), so a 2025 venue change can reach no other
# season.
PHASE332_VENUE_CORRECTED_SEASON: int = 2025

PHASE332_VENUE_RUNG_CAUSE: str = (
    "THE 2025 INTERNATIONAL VENUE CORRECTION of Plan 33.2-09 (SPEC R8 venue half), and "
    "NOTHING else: the seven 2025 games played outside the United States "
    "(2025_W01_KC@LAC, 2025_W04_MIN@PIT, 2025_W05_MIN@CLE, 2025_W06_DEN@NYJ, "
    "2025_W07_LA@JAX, 2025_W10_ATL@IND, 2025_W11_WAS@MIA), which silver games recorded "
    "at the US home team's own stadium, corrected in silver to the venue each was played "
    "at (Sao Paulo, Dublin, London x3, Berlin, Madrid) against the cited record "
    "config/international_venue_corrections.toml -- stadium_id, venue and venue_roof "
    "moved together, neutral_site unchanged -- plus the two venue records that "
    "correction needed, Croke Park (DUB00) and the Olympiastadion (BER00), added to "
    "data/venues.json. Only the stadium-dependent contextual columns can move, only in "
    "season 2025. No column is added, none removed, no row moves"
)

_PHASE332_STADIUM_DEPENDENT_CACHE: tuple[str, ...] | None = None


def _derive_phase332_stadium_dependent_columns() -> tuple[str, ...]:
    """The gold columns that are a FUNCTION OF ``stadium_id``, found by running the builder.

    One synthetic game -- fixed teams, fixed kickoff -- is placed at EVERY venue in
    ``data/venues.json`` and built in one frame. A column whose value differs between two
    of those rows can only differ because the venue did, so it reads ``stadium_id``; a
    column constant across all of them does not. Deriving rather than listing is the
    Phase-33.1 precedent (``_derive_phase331_venue_family``): a hand list and the code
    that does the work drift apart.
    """
    from features.contextual import ContextualFeaturesCalculator

    calculator = ContextualFeaturesCalculator()
    venues = [v for v in calculator.venues_data["venues"] if v.get("stadium_id")]
    home_team, away_team = "KC", "BUF"
    probe = pd.DataFrame(
        [
            {
                "game_id": f"2024_W01_PROBE_{venue['stadium_id']}",
                "season": 2024,
                "week": 1,
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_et": pd.Timestamp(
                    "2024-09-08 13:00:00", tz="America/New_York"
                ),
                "stadium_id": venue["stadium_id"],
                "home_score": 0.0,
                "away_score": 0.0,
            }
            for venue in venues
        ]
    )
    emitted = calculator.build_features(probe, datetime(2024, 9, 9, tzinfo=UTC))
    merge_keys = set(PHASE331_CONTEXTUAL_MERGE_KEYS)
    dependent = tuple(
        sorted(
            column
            for column in emitted.columns
            if column not in merge_keys and emitted[column].nunique(dropna=False) > 1
        )
    )
    if not dependent:
        msg = (
            "no contextual column varied across the venue probe, so the stadium-dependent "
            "set would be empty and p332_ rung 2 would refuse every move it exists to "
            "attribute. Refusing to derive an empty set."
        )
        raise ValueError(msg)
    return dependent


def phase332_stadium_dependent_columns() -> tuple[str, ...]:
    """The derived stadium-dependent set, computed once per process."""
    global _PHASE332_STADIUM_DEPENDENT_CACHE
    if _PHASE332_STADIUM_DEPENDENT_CACHE is None:
        _PHASE332_STADIUM_DEPENDENT_CACHE = _derive_phase332_stadium_dependent_columns()
    return _PHASE332_STADIUM_DEPENDENT_CACHE


PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_VENUE_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_VENUE_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the STADIUM-DEPENDENT columns -- derived by "
        "phase332_stadium_dependent_columns(), which places one synthetic game at every "
        "venue in data/venues.json, runs features.contextual.ContextualFeaturesCalculator."
        "build_features, and keeps each emitted column whose value differs between venues "
        "-- and each only in season 2025 and in no other season"
    ),
    "rows_changed": (
        "before normalization only the seven corrected games can differ; after it, a "
        "stadium-dependent column that expanding_normalize rescales can also move on OTHER "
        "2025 rows, because the within-season expanding mean and standard deviation for "
        "2025 then include a corrected game (the earliest is week 1, so any 2025 row may "
        "move in such a column). A column normalization leaves at its level (the discrete "
        "indicators) can move on the seven games only. Measured row by row at run time "
        "against a copy of the before-gold, never inferred from the per-season digests"
    ),
    "weather": (
        "NOT expected to move: gold weather is read from silver weather_features by "
        "game_id and this rung changes no silver weather row. The seven games' weather is "
        "replaced at rung 4 (day-before forecasts), which reads the corrected stadium_id"
    ),
    "declared_families": ("stadium_dependent",),
    "family_mechanisms": {
        "stadium_dependent": (
            "source-derived: the contextual builder's emitted columns that vary across "
            "one probe game placed at every venue in data/venues.json"
        ),
    },
    "declared_before_the_rebuild": True,
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_VENUE_RUNG] = (
    PHASE332_VENUE_RUNG_CAUSE
)

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21, "retake a stale
# baseline, never widen a rung's cause"). Rung 2 registers NO retaken baseline because the
# check below found nothing to retake: it is judged against its ladder predecessor.
PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION_DOCUMENT: str = (
    f"{PHASE332_RUNG_PREFIX}rung2_baseline_confirm.json"
)

PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-21 before rung 2 ran. Gold was rebuilt with `scripts/build_features.py "
    "--through-season 2025` in a SCRATCH data root (DATA_ROOT_PATH and DUCKDB_PATH pointed at "
    "a copy of today's production data/) from today's inputs minus exactly rung 2's cause -- "
    "the silver games repair not yet applied and data/venues.json still at 60 records -- and "
    "its fingerprint equals p332_rung1.json on EVERY non-clock column of all three matrices "
    "(only feature_timestamp, the build clock, differs). The production data/ tree was "
    "digest-identical (463 files) before and after that build, and to the digest taken just "
    "after rung 1's rebuild. So nothing moved between rungs 1 and 2: no carry-in, no retake"
)


def _attribute_p332_venue(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 2 of the `p332_` ladder: the 2025 international venue correction's OWN judge.

    A changed column is attributed ONLY when it is in the derived stadium-dependent set
    AND the seasons it moved in are exactly {2025}. Anything else is UNATTRIBUTED and
    fails; it is never absorbed and the cause is never widened to fit it.

    NOT `_attribute_rung2`, deliberately: that is Phase 30's WR-06 judge, a blanket
    attribution whose own comment records that it cannot fail on a moved column.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"stadium_dependent": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "p332_ rung 2 (the 2025 international venues)",
        "Moving seven 2025 games to the venues they were played at",
    )
    dependent = {_canonical(column) for column in phase332_stadium_dependent_columns()}
    corrected = {str(PHASE332_VENUE_CORRECTED_SEASON)}
    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        if column in dependent and set(seasons) == corrected:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["stadium_dependent"].append(column)
            continue
        verdict["unattributed"].append(column)
        why = (
            f"it moved in season(s) other than {PHASE332_VENUE_CORRECTED_SEASON}, which a "
            "2025 venue correction cannot reach"
            if column in dependent
            else "it is not a stadium-dependent column (it does not vary with the venue "
            "in the derived probe)"
        )
        fail(
            f"column '{column}' moved at p332_ rung 2 in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The rung's ONE cause is the "
            "2025 international venue correction; do NOT widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_SIGNATURES[PHASE332_VENUE_RUNG] = PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE
PHASE332_RUNG_ATTRIBUTORS[PHASE332_VENUE_RUNG] = _attribute_p332_venue


# ---------------------------------------------------------------------------
# p332_ RUNG 3 -- EMERGENCY SCHEDULE MOVES (Plan 33.2-10 Task 4, SPEC R8, D33.2-04, D33.2-21).
#
# Registered per Plan 33.2-08's <owned_protocol_rung_registration>: three names, one
# incremental cause line, one entry in each dispatch table, and NO new `if prefix ==`
# branch. The table entries are sharpest here: without them the generic path's rung 3
# expects a `line_movement` column REMOVAL (Phase 30's rung 3), so a correct rebuild that
# removes nothing would be reported against another phase's cause.
#
# DECLARED BEFORE THE REBUILD RAN. The explainable columns are DERIVED, not listed: each
# kind of move that is reverted (a REAL move whose verdict is post_lock in
# config/schedule_moves.toml) reaches a known family of contextual columns, and a venue
# move reaches exactly the stadium-dependent set rung 2 already derives from source. Today
# the table holds ONE real post-lock move, a venue move, so only that family can move.
#
# WEATHER IS NOT IN THE SET, AND THAT IS THE PREDICTION. Gold weather is read from silver
# weather_features by game_id and this rung writes no silver row, so no weather column can
# move here. The neutralised game's WEATHER LOCATION is served by the same accessor to the
# weather writer (scripts.ingest_weather resolves a game's venue through facts_at_lock), and
# rung 4 (Plan 33.2-12) regenerates silver weather through it -- the observation stored
# today was taken at the post-move venue and is replaced there, not here.
# ---------------------------------------------------------------------------

PHASE332_SCHEDULE_MOVE_RUNG: int = 3

PHASE332_SCHEDULE_MOVE_TABLE: Path = Path("config/schedule_moves.toml")

PHASE332_SCHEDULE_MOVE_RUNG_CAUSE: str = (
    "THE EMERGENCY SCHEDULE-MOVE NEUTRALISATION of Plan 33.2-10 (SPEC R8, D33.2-04, "
    "D33.2-21), and NOTHING else: every game whose emergency move was announced after its "
    "lock is rebuilt with the facts as they stood before the move, read through the one "
    "accessor features.schedule_moves.facts_at_lock from the owner-ratified evidence table "
    "config/schedule_moves.toml -- today exactly one game, 2003_W08_MIA@LAC, moved by the "
    "Cedar fire from Qualcomm Stadium (SDG00) to Sun Devil Stadium (PHO99) with no dated "
    "report before its lock, so its contextual features return to SDG00. Only the "
    "schedule-fact columns of the reverted moves' kinds can move, only in seasons on or "
    "after the earliest season holding such a move. No column is added, none removed, no "
    "row moves"
)

# The contextual columns a reverted DATE reaches: the weekday family, the rest family of
# the game's own two teams, and the time-zone travel columns (a date can cross a DST
# change). Unused while the table holds no post-lock date move; stated so a later verdict
# change is judged by a declared set rather than a guessed one.
PHASE332_SCHEDULE_DATE_FAMILY: tuple[str, ...] = (
    "thursday_game",
    "monday_game",
    "saturday_game",
    "short_week",
    "game_day_of_week",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "both_short_rest",
    "home_short_rest",
    "away_short_rest",
    "home_off_bye",
    "away_off_bye",
    "away_timezone_diff_hours",
    "away_abs_timezone_diff_hours",
    "away_travel_fatigue_score",
    "away_cross_country_travel",
    "away_eastward_travel",
    "away_westward_travel",
)

# A reverted WEEK moves the date too, and the two week-position columns.
PHASE332_SCHEDULE_WEEK_ONLY_COLUMNS: tuple[str, ...] = (
    "season_progress",
    "late_season",
)


def phase332_post_lock_real_moves(
    table_path: Path | str = PHASE332_SCHEDULE_MOVE_TABLE,
) -> dict[str, tuple[str, ...]]:
    """``game_id -> the kinds of its REAL moves that are reverted`` at its lock.

    Read through ``features.schedule_moves`` -- the same loader and the same selection
    the builders use -- so the attribution and the build cannot disagree about which
    games were neutralised.
    """
    from features.schedule_moves import load_schedule_moves

    reverted: dict[str, tuple[str, ...]] = {}
    for game_id, moves in load_schedule_moves(table_path).items():
        last_pre_lock = max(
            (m.move_index for m in moves if m.verdict == "pre_lock"), default=0
        )
        after = [m for m in moves if m.move_index > last_pre_lock]
        if any(m.verdict == "post_lock" for m in after):
            kinds = tuple(sorted({m.what_moved for m in after if m.is_real_move}))
            if kinds:
                reverted[game_id] = kinds
    return reverted


def phase332_schedule_fact_columns(
    table_path: Path | str = PHASE332_SCHEDULE_MOVE_TABLE,
) -> tuple[str, ...]:
    """The gold columns rung 3 may move: the union of the reverted move kinds' families."""
    kinds = {k for ks in phase332_post_lock_real_moves(table_path).values() for k in ks}
    columns: set[str] = set()
    if "venue" in kinds:
        columns |= set(phase332_stadium_dependent_columns())
    if kinds & {"date", "week"}:
        columns |= set(PHASE332_SCHEDULE_DATE_FAMILY)
    if "week" in kinds:
        columns |= set(PHASE332_SCHEDULE_WEEK_ONLY_COLUMNS)
    return tuple(sorted(columns))


def phase332_schedule_move_earliest_season(
    table_path: Path | str = PHASE332_SCHEDULE_MOVE_TABLE,
) -> int | None:
    """The earliest season holding a reverted move, or None when nothing is reverted."""
    games = phase332_post_lock_real_moves(table_path)
    return min((int(game_id[:4]) for game_id in games), default=None)


PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_SCHEDULE_MOVE_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_SCHEDULE_MOVE_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the SCHEDULE-FACT columns of the reverted moves' kinds -- derived by "
        "phase332_schedule_fact_columns(): a venue move reaches the stadium-dependent set "
        "(phase332_stadium_dependent_columns), a date move the weekday, own-team rest and "
        "time-zone travel columns, a week move those plus season_progress and late_season. "
        "Today only a venue move is reverted. Each column only in seasons ON OR AFTER the "
        "earliest season holding a reverted move (2003): the build's imputation medians and "
        "winsorization bounds are fitted on ALL strictly-prior seasons, so a changed value "
        "can reach every later season and no earlier one"
    ),
    "rows_changed": (
        "before normalization only the neutralised game can differ. After it: a rescaled "
        "column can also move on other rows of the same season sorted at or after that "
        "game (expanding within-season statistics), on the next season's early rows (the "
        "prior-season bootstrap), and on any later season's rows through the strictly-prior "
        "imputation and winsorization fits. A level-preserved indicator can move on the "
        "neutralised game only. Measured row by row at run time against a copy of the "
        "before-gold, never inferred from the per-season digests"
    ),
    "weather": (
        "NOT expected to move: gold weather is read from silver weather_features by game_id "
        "and this rung writes no silver row. The neutralised game's weather LOCATION is "
        "served by facts_at_lock to scripts.ingest_weather's venue resolution; rung 4 "
        "(Plan 33.2-12) regenerates silver weather through it"
    ),
    "declared_families": ("schedule_fact",),
    "family_mechanisms": {
        "schedule_fact": (
            "source-derived: the contextual columns each reverted move kind reaches, read "
            "against the owner-ratified config/schedule_moves.toml through "
            "features.schedule_moves"
        ),
    },
    "declared_before_the_rebuild": True,
    "owner_ruling": (
        "2026-09-21, 'Approve as written' (ratify-all): the evidence table stands as "
        "committed in 64abce6 -- 52 before-lock moves keep their real facts and "
        "2003_W08_MIA@LAC stays decided after the lock and is neutralised"
    ),
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_SCHEDULE_MOVE_RUNG] = (
    PHASE332_SCHEDULE_MOVE_RUNG_CAUSE
)

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21, "retake a stale
# baseline, never widen a rung's cause"). Rung 3 registers NO retaken baseline because the
# check below found nothing to retake: it is judged against its ladder predecessor.
PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION_DOCUMENT: str = (
    f"{PHASE332_RUNG_PREFIX}rung3_baseline_confirm.json"
)

PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-21 before rung 3 ran. Gold was rebuilt with `scripts/build_features.py "
    "--through-season 2025` in a SCRATCH data root (DATA_ROOT_PATH and DUCKDB_PATH pointed at "
    "a copy of today's production data/) from today's inputs minus exactly rung 3's cause -- "
    "the code at commit 64abce6, before features.schedule_moves was wired into the contextual "
    "builder -- and its fingerprint equals p332_rung2.json on EVERY non-clock column of all "
    "three matrices (only feature_timestamp, the build clock, differs). The production data/ "
    "tree was digest-identical (463 files) before and after that build, and production gold "
    "fingerprinted identical to p332_rung2.json. So nothing moved between rungs 2 and 3: no "
    "carry-in, no retake"
)


def _attribute_p332_schedule_move(
    detail: dict, diff: dict, verdict: dict, fail
) -> bool:
    """Rung 3 of the `p332_` ladder: the schedule-move neutralisation's OWN judge.

    A changed column is attributed ONLY when it is in the derived schedule-fact set AND
    every season it moved in is on or after the earliest season holding a reverted move.
    Anything else is UNATTRIBUTED and fails; it is never absorbed and the cause is never
    widened to fit it.

    NOT the generic rung-3 path, deliberately: that is Phase 30's, which expects a
    `line_movement` column removal this rung does not make.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"schedule_fact": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "p332_ rung 3 (the emergency schedule moves)",
        "Rebuilding post-lock-moved games with their pre-move facts",
    )
    explainable = {_canonical(column) for column in phase332_schedule_fact_columns()}
    floor = phase332_schedule_move_earliest_season()
    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        in_range = (
            floor is not None
            and bool(seasons)
            and all(int(s) >= floor for s in seasons)
        )
        if column in explainable and in_range:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["schedule_fact"].append(column)
            continue
        verdict["unattributed"].append(column)
        why = (
            f"it moved in season(s) before {floor}, the earliest season holding a "
            "reverted move, which the strictly-prior fits cannot reach"
            if column in explainable
            else "it is not a schedule-fact column of any reverted move's kind"
        )
        fail(
            f"column '{column}' moved at p332_ rung 3 in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The rung's ONE cause is the "
            "schedule-move neutralisation; do NOT widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_SIGNATURES[PHASE332_SCHEDULE_MOVE_RUNG] = (
    PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE
)
PHASE332_RUNG_ATTRIBUTORS[PHASE332_SCHEDULE_MOVE_RUNG] = _attribute_p332_schedule_move


# ---------------------------------------------------------------------------
# p332_ EXTRA STEP 3b -- SURFACE CLASSIFICATION (Plan 33.2-10, orchestrator-assigned; the
# deferred-items entry "surface_mismatch treats natural and hybrid grass ... as synthetic").
#
# NOT A NUMBERED RUNG. Rungs 4-9 are allocated to later plans, so this fix -- one cause,
# its own rebuild, its own attribution (D33.2-20) -- is registered as the EXTRA STEP "3b",
# which follows rung 3: it is judged against p332_rung3.json, and rung 4 will be judged
# against p332_rung3b.json (see EXTRA_STEPS_BY_PREFIX).
#
# DECLARED BEFORE THE REBUILD RAN. The only gold column the surface class reaches is
# `surface_mismatch` (features.contextual._compute_surface_mismatch). The games whose value
# changes are DERIVED, not listed: every silver game is classified under the retired
# two-spelling rule and under features.contextual.SURFACE_CLASS_BY_SPELLING, and the games
# whose mismatch differs are the reclassified set. No team's home venue uses a reclassified
# spelling, so every reclassified game is played at a venue spelled "Grass", "Hybrid Grass"
# or "Desso GrassMaster" (London, Mexico City, Frankfurt, Munich, Sao Paulo, Dublin,
# Berlin); "RealGrass" (Texas Stadium) is artificial turf and does not change.
# ---------------------------------------------------------------------------

PHASE332_SURFACE_STEP: str = "3b"

PHASE332_SURFACE_STEP_FOLLOWS: int = PHASE332_SCHEDULE_MOVE_RUNG

# The rule this step retires, kept so the reclassified set can be recomputed from source.
PHASE332_SURFACE_STEP_RETIRED_GRASS_SPELLINGS: frozenset[str] = frozenset(
    {"Bermuda Grass", "Kentucky Bluegrass"}
)

PHASE332_SURFACE_STEP_COLUMNS: tuple[str, ...] = ("surface_mismatch",)

PHASE332_SURFACE_STEP_CAUSE: str = (
    "THE SURFACE-CLASSIFICATION FIX of Plan 33.2-10 extra step 3b (orchestrator-assigned, "
    "deferred-items entry of Plan 33.2-09), and NOTHING else: "
    "features.contextual.SURFACE_CLASS_BY_SPELLING now classifies every distinct surface "
    "spelling in data/venues.json -- Grass, Hybrid Grass and Desso GrassMaster as grass "
    "(natural or hybrid), RealGrass and the other turf products as synthetic -- and an "
    "unlisted spelling raises instead of defaulting to synthetic, so surface_mismatch is "
    "recomputed at the international and hybrid-pitch venues the two-spelling rule misread. "
    "Only surface_mismatch can move, only in seasons on or after the earliest season "
    "holding a reclassified game. No column is added, none removed, no row moves"
)

_PHASE332_SURFACE_RECLASSIFIED_CACHE: dict[str, int] | None = None


def phase332_surface_reclassified_games() -> dict[str, int]:
    """``game_id -> season`` for every silver game whose surface_mismatch the fix changes.

    Each game is classified twice with the builder's own inputs -- its venue at the lock
    (features.schedule_moves.facts_at_lock, the same accessor the builder reads) and the
    away team's home venue (the calculator's team-to-venue map) -- once under the retired
    two-spelling rule and once under features.contextual.surface_is_grass. Reads silver
    ``games`` READ-ONLY and is computed once per process.
    """
    global _PHASE332_SURFACE_RECLASSIFIED_CACHE
    if _PHASE332_SURFACE_RECLASSIFIED_CACHE is None:
        from data.storage import load_dataframe
        from features.contextual import ContextualFeaturesCalculator, surface_is_grass

        calculator = ContextualFeaturesCalculator()
        surfaces = {
            venue["venue_id"]: venue["surface"]
            for venue in calculator.venues_data["venues"]
        }
        games = load_dataframe("games", layer="silver")
        changed: dict[str, int] = {}
        for _, game in games.iterrows():
            away_home = calculator.team_venues.get(game["away_team"])
            if not away_home:
                continue
            venue_id = calculator._resolve_venue_id_for_game(game)
            away_surface, game_surface = surfaces[away_home], surfaces[venue_id]
            retired = PHASE332_SURFACE_STEP_RETIRED_GRASS_SPELLINGS
            old = (away_surface in retired) != (game_surface in retired)
            new = surface_is_grass(away_surface) != surface_is_grass(game_surface)
            if old != new:
                changed[str(game["game_id"])] = int(game["season"])
        _PHASE332_SURFACE_RECLASSIFIED_CACHE = changed
    return dict(_PHASE332_SURFACE_RECLASSIFIED_CACHE)


def phase332_surface_step_earliest_season(through_season: int = 2025) -> int | None:
    """The earliest season (at most *through_season*) holding a reclassified game."""
    seasons = [
        season
        for season in phase332_surface_reclassified_games().values()
        if season <= through_season
    ]
    return min(seasons, default=None)


PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_SURFACE_STEP,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_SURFACE_STEP_CAUSE,
    "follows_rung": PHASE332_SURFACE_STEP_FOLLOWS,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to surface_mismatch, the one gold column the surface class reaches, "
        "and only in seasons ON OR AFTER the earliest season holding a reclassified game "
        "(derived by phase332_surface_reclassified_games from silver games and the two "
        "rules): a changed value moves its own season and, through the prior-season "
        "bootstrap and the strictly-prior fits, later seasons, never an earlier one"
    ),
    "rows_changed": (
        "before normalization only the reclassified games can differ. surface_mismatch is "
        "a varying binary flag, so gold z-scores it: after normalization it can also move "
        "on other rows of a reclassified game's season sorted at or after that game, and "
        "on the next season's early rows. Measured row by row at run time against a copy "
        "of the before-gold"
    ),
    "weather": "NOT expected to move: the surface class feeds no weather column",
    "declared_families": ("surface_class",),
    "family_mechanisms": {
        "surface_class": (
            "source-derived: every silver game classified under the retired two-spelling "
            "rule and under features.contextual.SURFACE_CLASS_BY_SPELLING"
        ),
    },
    "declared_before_the_rebuild": True,
}

EXTRA_STEPS_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[PHASE332_SURFACE_STEP] = (
    PHASE332_SURFACE_STEP_FOLLOWS
)
EXTRA_STEP_CAUSES_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[
    PHASE332_SURFACE_STEP
] = PHASE332_SURFACE_STEP_CAUSE

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21). Step 3b registers NO
# retaken baseline: it is judged against rung 3, the rung it follows.
PHASE332_SURFACE_STEP_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-21 before step 3b ran. p332_rung3.json IS gold rebuilt from today's "
    "inputs minus exactly this step's cause: the rung-3 rebuild ran on the same inputs "
    "with the code at commit 0757605, which differs from this step's code only by the "
    "surface classification. Between that rebuild and this step the production data/ tree "
    "did not change (463 files, digest-identical to the state rung 3 left), and production "
    "gold fingerprinted identical to p332_rung3.json. No carry-in, no retake"
)


def _attribute_p332_surface(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Extra step 3b of the `p332_` ladder: the surface-classification fix's OWN judge.

    A changed column is attributed ONLY when it is surface_mismatch AND every season it
    moved in is on or after the earliest season holding a reclassified game. Anything
    else is UNATTRIBUTED and fails; the cause is never widened to fit it.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"surface_class": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "p332_ step 3b (the surface classification)",
        "Reclassifying natural and hybrid grass pitches",
    )
    explainable = {_canonical(column) for column in PHASE332_SURFACE_STEP_COLUMNS}
    floor = phase332_surface_step_earliest_season()
    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        in_range = (
            floor is not None
            and bool(seasons)
            and all(int(s) >= floor for s in seasons)
        )
        if column in explainable and in_range:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["surface_class"].append(column)
            continue
        verdict["unattributed"].append(column)
        why = (
            f"it moved in season(s) before {floor}, the earliest season holding a "
            "reclassified game"
            if column in explainable
            else "the surface class reaches no column but surface_mismatch"
        )
        fail(
            f"column '{column}' moved at p332_ step 3b in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The step's ONE cause is the "
            "surface classification; do NOT widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


# The `p332_` extra-step dispatch tables: one entry per registered step, read by
# _phase332_table exactly as PHASE332_RUNG_SIGNATURES / _ATTRIBUTORS are for numbered rungs.
PHASE332_EXTRA_STEP_SIGNATURES: dict[str, dict[str, object]] = {
    PHASE332_SURFACE_STEP: PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE
}
PHASE332_EXTRA_STEP_ATTRIBUTORS: dict[str, Callable[..., bool]] = {
    PHASE332_SURFACE_STEP: _attribute_p332_surface
}


# ---------------------------------------------------------------------------
# p332_ EXTRA STEP 3c -- KICKOFF HOUR CORRECTION (Plan 33.2-12, orchestrator-assigned; the
# deferred-items entry "68 silver kickoffs in 2002-2005 are stored at 09:00 ET for Monday
# and Thursday NIGHT games", found by Plan 33.2-10).
#
# NOT A NUMBERED RUNG. It runs BEFORE rung 4 because rung 4's forecast-hour selection reads
# the kickoff hour. It follows rung 3 and was registered after step 3b, so it is judged
# against p332_rung3b.json and rung 4 is judged against p332_rung3c.json.
#
# DECLARED BEFORE ANY PRODUCTION WRITE. Silver `games.kickoff_et` moves +12 hours (09:00 ->
# 21:00 ET, same date) for the 68 games of config/kickoff_hour_corrections.toml. The date
# is unchanged, so no lock and no weekday moves (thursday_game, monday_game, short_week and
# game_day_of_week cannot move). Silver weather is NOT regenerated here (that is rung 4),
# and gold weather is read from silver weather_features by game id, so no weather column
# can move. The one gold reader of the kickoff HOUR is the contextual rest-day count:
# features.contextual.calculate_rest_days floors (current - previous kickoff).days, so a
# 12-hour shift can move the rest count of a corrected game itself and of each of its two
# teams' NEXT game -- and with it the columns derived from rest (rest_advantage, the
# short-rest flags, off_bye). The games whose rest count moves are DERIVED, not listed:
# phase332_kickoff_hour_rest_changes calls that same function under both clocks.
# ---------------------------------------------------------------------------

PHASE332_KICKOFF_HOUR_STEP: str = "3c"

PHASE332_KICKOFF_HOUR_STEP_FOLLOWS: int = PHASE332_SCHEDULE_MOVE_RUNG

# Every gold column computed from the contextual rest-day count (features.contextual,
# the per-game loop that calls calculate_rest_days). NOT the weekday family: the date
# does not move.
PHASE332_KICKOFF_HOUR_STEP_COLUMNS: tuple[str, ...] = (
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "both_short_rest",
    "home_short_rest",
    "away_short_rest",
    "home_off_bye",
    "away_off_bye",
)

PHASE332_KICKOFF_HOUR_STEP_CAUSE: str = (
    "THE KICKOFF-HOUR CORRECTION of Plan 33.2-12 extra step 3c (orchestrator-assigned, "
    "deferred-items entry of Plan 33.2-10), and NOTHING else: silver games.kickoff_et of the "
    "68 2002-2005 Monday and Thursday night games in config/kickoff_hour_corrections.toml "
    "moves from 09:00 to 21:00 ET on the same date (a 12-hour AM/PM error in the feed, each "
    "game cited to its archived box score or official gamebook), applied at every ingest by "
    "scripts.ingest_games.resolve_kickoff_gametime. No lock, weekday or weather value moves. "
    "Only the rest-day family can move -- home_rest_days, away_rest_days, rest_advantage, "
    "both_short_rest, home_short_rest, away_short_rest, home_off_bye, away_off_bye -- and "
    "only in seasons on or after the earliest season holding a game whose rest count "
    "changes. No column is added, none removed, no row moves"
)

_PHASE332_KICKOFF_REST_CHANGES_CACHE: dict[str, int] | None = None


def phase332_kickoff_hour_rest_changes() -> dict[str, int]:
    """``game_id -> season`` for every silver game whose home or away rest count moves.

    For each recorded game, its two teams are scored at that game and at each team's next
    game, with ``features.contextual.ContextualFeaturesCalculator.calculate_rest_days`` --
    the builder's own function -- once with every recorded game at its feed clock and once
    at its corrected clock. Both clocks come from the record, so the answer is the same
    whether silver has been repaired yet or not. No other game's rest can move: a rest
    count reads only the game's own kickoff and its team's previous one, and a 12-hour shift
    cannot reorder a team's games. Reads silver ``games`` READ-ONLY; computed once.
    """
    global _PHASE332_KICKOFF_REST_CHANGES_CACHE
    if _PHASE332_KICKOFF_REST_CHANGES_CACHE is None:
        from zoneinfo import ZoneInfo

        from data.storage import load_dataframe
        from features.contextual import ContextualFeaturesCalculator
        from scripts.ingest_games import load_kickoff_hour_corrections
        from utils.date_utils import kickoff_wall_clock_et

        eastern = ZoneInfo("America/New_York")
        games = load_dataframe("games", layer="silver").reset_index(drop=True)
        corrections = load_kickoff_hour_corrections()
        position_of = {gid: i for i, gid in enumerate(games["game_id"])}
        base = games["kickoff_et"].map(kickoff_wall_clock_et)

        def clocks(field: str) -> pd.Series:
            series = base.copy()
            for game_id, correction in corrections.items():
                series.at[position_of[game_id]] = datetime.fromisoformat(
                    f"{correction.gameday} {getattr(correction, field)}"
                ).replace(tzinfo=eastern)
            return series

        feed, fixed = clocks("feed_gametime"), clocks("corrected_gametime")
        calculator = ContextualFeaturesCalculator()
        changed: dict[str, int] = {}
        for game_id in corrections:
            position = position_of[game_id]
            for team in (
                games.at[position, "home_team"],
                games.at[position, "away_team"],
            ):
                plays = games.index[
                    (games["home_team"] == team) | (games["away_team"] == team)
                ]
                later = [i for i in plays if fixed[i] > fixed[position]]
                scored = [position] + (
                    [min(later, key=fixed.__getitem__)] if later else []
                )
                for i in scored:
                    before = calculator.calculate_rest_days(
                        team, feed[i], games, kickoffs=feed
                    )
                    after = calculator.calculate_rest_days(
                        team, fixed[i], games, kickoffs=fixed
                    )
                    if before != after:
                        changed[str(games.at[i, "game_id"])] = int(
                            games.at[i, "season"]
                        )
        _PHASE332_KICKOFF_REST_CHANGES_CACHE = changed
    return dict(_PHASE332_KICKOFF_REST_CHANGES_CACHE)


def phase332_kickoff_hour_step_earliest_season(
    through_season: int = 2025,
) -> int | None:
    """The earliest season (at most *through_season*) holding a game whose rest moves."""
    seasons = [
        season
        for season in phase332_kickoff_hour_rest_changes().values()
        if season <= through_season
    ]
    return min(seasons, default=None)


PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_KICKOFF_HOUR_STEP,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_KICKOFF_HOUR_STEP_CAUSE,
    "follows_rung": PHASE332_KICKOFF_HOUR_STEP_FOLLOWS,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the eight rest-day columns (PHASE332_KICKOFF_HOUR_STEP_COLUMNS), "
        "and only in seasons ON OR AFTER the earliest season holding a game whose rest count "
        "moves (derived by phase332_kickoff_hour_rest_changes from silver games and the "
        "record): a changed value moves its own season and, through the prior-season "
        "bootstrap and the strictly-prior fits, later seasons, never an earlier one"
    ),
    "rows_changed": (
        "before normalization only the games phase332_kickoff_hour_rest_changes derives "
        "can differ. The rest columns are continuous or varying flags, so gold z-scores "
        "them within each season: after normalization they can also move on the other rows "
        "of an affected game's season sorted at or after that game (the rescaled same-season "
        "neighbours) and on the next season's early rows. Measured row by row at run time "
        "against a copy of the before-gold"
    ),
    "weather": (
        "NOT expected to move: silver weather is not regenerated at this step, and gold "
        "weather is read from silver weather_features by game id"
    ),
    "declared_families": ("rest_days",),
    "family_mechanisms": {
        "rest_days": (
            "source-derived: every recorded game's two teams scored at that game and at "
            "their next game with features.contextual's calculate_rest_days, under the feed "
            "clock and the corrected clock"
        ),
    },
    "declared_before_the_rebuild": True,
}

EXTRA_STEPS_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[
    PHASE332_KICKOFF_HOUR_STEP
] = PHASE332_KICKOFF_HOUR_STEP_FOLLOWS
EXTRA_STEP_CAUSES_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[
    PHASE332_KICKOFF_HOUR_STEP
] = PHASE332_KICKOFF_HOUR_STEP_CAUSE

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21). Step 3c registers NO
# retaken baseline: it is judged against step 3b, the entry before it.
PHASE332_KICKOFF_HOUR_STEP_BASELINE_CONFIRMATION_DOCUMENT: str = (
    f"{PHASE332_RUNG_PREFIX}step3c_baseline_confirm.json"
)

PHASE332_KICKOFF_HOUR_STEP_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-21 before step 3c wrote anything. Gold was rebuilt with "
    "`scripts/build_features.py --through-season 2025` in a SCRATCH data root "
    "(DATA_ROOT_PATH and DUCKDB_PATH pointed at a copy of today's production data/, the "
    "code exported from commit 5d7eefd, before the kickoff correction existed) from today's "
    "inputs minus exactly this step's cause, and its fingerprint equals p332_rung3b.json on "
    "EVERY non-clock column of all three matrices (only feature_timestamp, the build clock, "
    "differs). Plan 33.2-11 changed data/ between step 3b and this step only under "
    "bronze/mos/, which no gold builder reads. The production data/ tree was "
    "digest-identical (1107 files) before and after that build. No carry-in, no retake"
)


def _attribute_p332_kickoff_hour(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Extra step 3c of the `p332_` ladder: the kickoff-hour correction's OWN judge.

    A changed column is attributed ONLY when it is one of the eight rest-day columns AND
    every season it moved in is on or after the earliest season holding a game whose rest
    count moves. Anything else is UNATTRIBUTED and fails; the cause is never widened.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"rest_days": []}
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        "p332_ step 3c (the kickoff-hour correction)",
        "Correcting the 2002-2005 night-game kickoff hours",
    )
    explainable = {_canonical(column) for column in PHASE332_KICKOFF_HOUR_STEP_COLUMNS}
    floor = phase332_kickoff_hour_step_earliest_season()
    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        in_range = (
            floor is not None
            and bool(seasons)
            and all(int(s) >= floor for s in seasons)
        )
        if column in explainable and in_range:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["rest_days"].append(column)
            continue
        verdict["unattributed"].append(column)
        why = (
            f"it moved in season(s) before {floor}, the earliest season holding a game "
            "whose rest count moves"
            if column in explainable
            else "the kickoff hour reaches no gold column but the rest-day family"
        )
        fail(
            f"column '{column}' moved at p332_ step 3c in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The step's ONE cause is the "
            "kickoff-hour correction; do NOT widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_EXTRA_STEP_SIGNATURES[PHASE332_KICKOFF_HOUR_STEP] = (
    PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE
)
PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_KICKOFF_HOUR_STEP] = (
    _attribute_p332_kickoff_hour
)


# ---------------------------------------------------------------------------
# p332_ RUNG 4 -- THE DAY-BEFORE WEATHER FORECAST (Plan 33.2-12, SPEC R6, D33.2-02 / D33.2-12
# / D33.2-13). Registered per Plan 33.2-08's <owned_protocol_rung_registration>: three names,
# one incremental cause-table append, one entry in EACH dispatch table, no new prefix branch.
#
# DECLARED BEFORE THE REBUILD. The rung's one cause is the silver regeneration of Plan
# 33.2-12 Task 1 plus the two forecast-less inputs leaving gold (Task 2). Every gold column
# the weather builder emits can move, for every outdoor game, in every season 2002-2025 --
# the widest rung of the ladder -- and exactly two columns leave all three matrices. The
# explainable family is DERIVED from the builder's own declaration
# (features.weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]) plus the one gold column the
# build copies from it (raw_weather_severity, scripts/build_features.py: the un-normalized
# weather_severity_score kept for display), never typed from the diff.
#
# JUDGED AGAINST p332_rung3c.json, the entry before it (extra step 3c follows step 3b).
# ---------------------------------------------------------------------------

PHASE332_WEATHER_RUNG: int = 4

#: The two weather inputs no forecast can supply, removed from gold through the registry
#: group ``weather_unsupplied`` (backtest.signal_lift) by scripts.build_features.
PHASE332_WEATHER_RUNG_REMOVED_COLUMNS: tuple[str, ...] = ("precip_mm", "raw_precip_mm")

#: Gold columns the build DERIVES from a weather-builder column (not emitted by the builder).
PHASE332_WEATHER_DERIVED_COPIES: tuple[str, ...] = ("raw_weather_severity",)

#: The accepted live-versus-history provider mismatch (RESEARCH 6.6; config/mos_tolerance.py,
#: owner heads-up of 2026-09-21): recorded with the rung, never hidden.
PHASE332_WEATHER_RUNG_PROVIDER_MISMATCH: str = (
    "ACCEPTED, MEASURED train/serve difference: history (2002-2025) is now the archived NWS "
    "MOS bulletin while live 2026 games are served from Open-Meteo. Measured 2026-09-15 at one "
    "common valid hour across 18 stadium stations: temperature agrees to +0.37 F (effectively "
    "exact); wind carries a +1.56 mph systematic offset (bulletins higher, maximum gap 6.5 "
    "mph). The gold wind_calm / wind_moderate boundary sits at 5.0 mph, so on a genuinely calm "
    "day the offset can flip the band between training and serving, and wind_moderate is an "
    "ATS-selected feature today. For Plan 33.2-23: the re-fit's candidate set may reasonably "
    "prefer the continuous wind_mph / wind_impact_score over the hard-edged wind one-hots for "
    "this reason"
)

PHASE332_WEATHER_RUNG_CAUSE: str = (
    "THE DAY-BEFORE WEATHER FORECAST of Plan 33.2-12 (SPEC R6, D33.2-02), and NOTHING else: "
    "silver weather and weather_features for 2002-2025 regenerated from the archived day-before "
    "12 UTC NWS MOS bulletins (data/bronze/mos/), replacing the ERA5 reanalysis observations, "
    "with precip_mm and raw_precip_mm removed from gold as forecast-less inputs. Only the "
    "weather-builder columns and the one column the build copies from them "
    "(raw_weather_severity) can move, in any season; exactly precip_mm and raw_precip_mm are "
    "removed from all three matrices, no column is added and no row moves"
)


def phase332_weather_columns() -> frozenset[str]:
    """The rung's explainable family: the full weather builder's columns plus their copies.

    DERIVED from ``features.weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]`` -- the ONE
    declaration the builder asserts its own frame against -- plus
    ``PHASE332_WEATHER_DERIVED_COPIES``. Canonical (lower-case) names.
    """
    from features.weather import WEATHER_FEATURE_COLUMNS_BY_BUILDER

    return frozenset(
        _canonical(column)
        for column in (
            *WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"],
            *PHASE332_WEATHER_DERIVED_COPIES,
        )
    )


PHASE332_WEATHER_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_WEATHER_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_WEATHER_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": PHASE332_WEATHER_RUNG_REMOVED_COLUMNS,
    "rows": "unchanged",
    "width": "each matrix exactly two columns narrower (precip_mm, raw_precip_mm)",
    "columns_changed": (
        "restricted to the weather family (phase332_weather_columns: the full weather "
        "builder's declared columns and raw_weather_severity), in ANY season 2002-2025: every "
        "outdoor game's weather is a new value, and every dome keeps its genuine indoor state"
    ),
    "rows_changed": (
        "every outdoor game's weather-family values; after normalization the rescaled "
        "weather columns also move on every other row of each season and, through the "
        "strictly-prior fits, on later seasons. A discrete indicator that now takes both "
        "values (extreme_weather, which was constant on the observations) becomes "
        "level-preserved by the builder's own predicate. Measured row by row at run time "
        "against a copy of the before-gold"
    ),
    "weather": "EXPECTED to move: this rung IS the weather",
    "declared_families": ("weather",),
    "family_mechanisms": {
        "weather": (
            "source-derived: features.weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER['full'] plus "
            "the build's raw_weather_severity copy"
        ),
    },
    "provider_mismatch": PHASE332_WEATHER_RUNG_PROVIDER_MISMATCH,
    "declared_before_the_rebuild": True,
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_WEATHER_RUNG] = (
    PHASE332_WEATHER_RUNG_CAUSE
)

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21). Rung 4 registers NO
# retaken baseline: it is judged against step 3c, the entry before it.
PHASE332_WEATHER_RUNG_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-21 before rung 4 rebuilt gold. p332_rung3c.json IS gold rebuilt from "
    "today's inputs minus exactly this rung's cause: step 3c's rebuild ran on the same inputs "
    "with the code at commit 0c67856 (before the weather fence, the regeneration and the "
    "weather_unsupplied drop existed), and between that rebuild and this rung the production "
    "data/ tree changed ONLY in this rung's own declared silver write (silver/weather.parquet, "
    "silver/weather_features.parquet and their DuckDB copy -- digest bracket "
    "outputs/p332_rung4_before.json). No carry-in, no retake"
)


def _attribute_p332_weather(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 4 of the `p332_` ladder: the day-before weather forecast's OWN judge.

    STRUCTURE: no column may be added; the removed set must be EXACTLY the two declared
    forecast-less columns; the width must fall by exactly that many; rows must not move. Any
    structural surprise BLOCKS. VALUES: a changed column is attributed only when it is in the
    derived weather family. Anything else is UNATTRIBUTED and fails; the cause is never
    widened to fit it.

    NOT the generic rung-4 path, deliberately: that is Phase 30's N-01 re-sync, keyed by rung
    NUMBER, and a registered-but-undispatched p332_ rung 4 would be judged by it.

    Returns:
        Whether this matrix BLOCKS the phase.
    """
    verdict["changed_by_family"] = {"weather": []}
    label = "p332_ rung 4 (the day-before weather forecast)"
    blocking = False
    removed_expected = {_canonical(c) for c in PHASE332_WEATHER_RUNG_REMOVED_COLUMNS}
    for column in diff["added"]:
        blocking = True
        fail(f"column '{column}' was ADDED at {label}, which adds none")
    removed = {_canonical(c) for c in diff["removed"]}
    if removed != removed_expected:
        blocking = True
        fail(
            f"{label} removed {sorted(removed)}, not exactly the declared forecast-less "
            f"columns {sorted(removed_expected)}"
        )
    if detail["width_after"] != detail["width_before"] - len(removed_expected):
        blocking = True
        fail(
            f"width moved {detail['width_before']} -> {detail['width_after']} at {label}, "
            f"which predicts exactly {len(removed_expected)} columns fewer"
        )
    if detail.get("rows_before") != detail.get("rows_after"):
        blocking = True
        fail(
            f"rows moved {detail.get('rows_before')} -> {detail.get('rows_after')} at "
            f"{label}; the rung re-derives the same games"
        )
    if not diff["changed"] and not removed:
        fail(f"{label} moved no column at all; the weather must move")

    explainable = phase332_weather_columns()
    for column in sorted(diff["changed"]):
        if column in explainable:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["weather"].append(column)
            continue
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at {label} in season(s) "
            f"{', '.join(sorted(diff['changed'][column])) or '(none)'}, but it is not a "
            "weather-family column. The rung's ONE cause is the day-before forecast; do NOT "
            "widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_SIGNATURES[PHASE332_WEATHER_RUNG] = (
    PHASE332_WEATHER_RUNG_EXPECTED_SIGNATURE
)
PHASE332_RUNG_ATTRIBUTORS[PHASE332_WEATHER_RUNG] = _attribute_p332_weather


# ---------------------------------------------------------------------------
# p332_ EXTRA STEP 4b -- A RETRACTABLE ROOF IS UNKNOWN AT THE LOCK (Plan 33.2-14,
# orchestrator-assigned; owner ruling 2026-09-21, "Treat as unknown at lock", recorded in the
# phase's deferred-items.md).
#
# NOT A NUMBERED RUNG. Rungs 5-9 are allocated to later plans, so this fix -- one cause, its
# own rebuild, its own attribution (D33.2-20) -- is the EXTRA STEP "4b": a string id that
# follows rung 4 and cannot collide with any numbered rung. It is judged against its RETAKEN
# baseline (below), and rung 5 is judged against p332_rung4b.json.
#
# THE LEAK. Rung 4 applied Phase 33.1's per-game roof rule, so 620 games played with a
# retractable roof CLOSED became domes with no weather (2019_W18_BUF@HOU among them). Whether a
# retractable roof closes is decided close to kickoff, often BECAUSE of the weather, so the
# realized state is post-lock information. D33.2-04's "roof known at the lock" covers the FIXED
# roof type only.
#
# DECLARED BEFORE THE REBUILD. The fix is in the silver regeneration alone
# (scripts.weather_from_mos no longer reads a game's realized roof), and gold weather is read
# from silver weather_features by game id, so only the weather family can move: the 620 games'
# values (619 gain their forecast; 2019_W18_BUF@HOU becomes an honest absence) plus the
# rescaled same-season neighbours of every season that holds one (every season 2002-2025
# does) and, through the strictly-prior fits, later seasons. The model-visible "this roof may
# close" flag is venue_retractable (features.contextual, the venue's FIXED type at the lock
# venue), which this step does not move. No column is added, none removed, no row moves.
# ---------------------------------------------------------------------------

PHASE332_RETRACTABLE_ROOF_STEP: str = "4b"

PHASE332_RETRACTABLE_ROOF_STEP_FOLLOWS: int = PHASE332_WEATHER_RUNG

#: Games whose rung-4 silver row the realized roof decided, measured read-only 2026-09-21
#: (scripts.weather_from_mos at commit fda8727: closed_roof_game_ids over 2002-2025).
PHASE332_RETRACTABLE_ROOF_STEP_REDECIDED_GAMES: int = 620

#: The one redecided game whose bulletin is absent from the archive (Plan 33.2-11): it becomes
#: an absence (NULL plus the coverage flag), never a stand-in run.
PHASE332_RETRACTABLE_ROOF_STEP_ARCHIVE_GAP: str = "2019_W18_BUF@HOU"

PHASE332_RETRACTABLE_ROOF_STEP_CAUSE: str = (
    "A RETRACTABLE ROOF IS UNKNOWN AT THE LOCK, p332_ extra step 4b of Plan 33.2-14 "
    "(orchestrator-assigned; owner ruling 2026-09-21), and NOTHING else: the 2002-2025 silver "
    "weather regeneration (scripts.weather_from_mos) no longer reads any game's realized "
    "open/closed roof, so every game at a retractable-roof stadium carries the day-before "
    "forecast exactly as an outdoor game does -- 620 games rung 4 had turned into domes, 619 "
    "gaining their bulletin and 2019_W18_BUF@HOU (a confirmed archive gap) becoming an honest "
    "absence -- while fixed-roof domes stay domes and the venue's retractability stays "
    "model-visible through venue_retractable. Only the weather-builder columns and "
    "raw_weather_severity can move, in any season 2002-2025. No column is added, none removed, "
    "no row moves"
)

PHASE332_RETRACTABLE_ROOF_STEP_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_RETRACTABLE_ROOF_STEP,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_RETRACTABLE_ROOF_STEP_CAUSE,
    "follows_rung": PHASE332_RETRACTABLE_ROOF_STEP_FOLLOWS,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the weather family (phase332_weather_columns: the full weather "
        "builder's declared columns and raw_weather_severity), in any season 2002-2025: every "
        "season holds at least one redecided game"
    ),
    "rows_changed": (
        "before normalization only the 620 redecided games can differ. The rescaled weather "
        "columns are z-scored within each season from that season's earlier rows and fitted "
        "on prior seasons, so after normalization they also move on the rows of a redecided "
        "game's season sorted after it (the rescaled same-season neighbours) and on later "
        "seasons. Measured row by row at run time against a copy of the before-gold"
    ),
    "weather": "EXPECTED to move: this step re-decides which games weather applies to",
    "declared_families": ("weather",),
    "family_mechanisms": {
        "weather": (
            "source-derived: features.weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER['full'] plus "
            "the build's raw_weather_severity copy"
        ),
    },
    "model_visible_roof_flag": (
        "venue_retractable (features.contextual.encode_venue_features, the lock venue's FIXED "
        "roof type in data/venues.json); NOT expected to move"
    ),
    "declared_before_the_rebuild": True,
}

EXTRA_STEPS_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[
    PHASE332_RETRACTABLE_ROOF_STEP
] = PHASE332_RETRACTABLE_ROOF_STEP_FOLLOWS
EXTRA_STEP_CAUSES_BY_PREFIX.setdefault(PHASE332_RUNG_PREFIX, {})[
    PHASE332_RETRACTABLE_ROOF_STEP
] = PHASE332_RETRACTABLE_ROOF_STEP_CAUSE

# THE BASELINE IS RETAKEN, NOT ASSUMED (owner ruling 2026-09-21: "retake a stale baseline,
# never widen a rung's cause"). Plan 33.2-13 changed the injury and QB builders (code only,
# no gold rebuild) after rung 4 was built, so gold rebuilt from TODAY's inputs differs from
# p332_rung4.json by that plan's effect whatever step 4b does. Step 4b is therefore judged
# against gold rebuilt in a SCRATCH data root from today's inputs minus only its own cause,
# and the difference between p332_rung4.json and that baseline is recorded as Plan 33.2-13's
# measured effect -- never inside step 4b.
PHASE332_RETRACTABLE_ROOF_STEP_BASELINE_DOCUMENT: str = (
    f"{PHASE332_RUNG_PREFIX}rung4_retaken.json"
)

PHASE332_RETAKEN_BASELINES[PHASE332_RETRACTABLE_ROOF_STEP] = (
    PHASE332_RETRACTABLE_ROOF_STEP_BASELINE_DOCUMENT
)
PHASE332_RETAKEN_BASELINE_REASONS[PHASE332_RETRACTABLE_ROOF_STEP] = (
    "RETAKEN 2026-09-21 by the standing ruling. Gold rebuilt with `scripts/build_features.py "
    "--through-season 2025` in a SCRATCH data root (DATA_ROOT_PATH and DUCKDB_PATH pointed at "
    "a copy of today's production data/, the code exported from commit fda8727 -- Plan "
    "33.2-13's last commit, before step 4b existed), i.e. today's inputs minus exactly step "
    "4b's cause. The production data/ tree was digest-identical (1107 files) before and after "
    "that build. p332_rung4.json is kept unchanged"
)

#: THE CARRY-IN'S CAUSE: Plan 33.2-13's builder changes, measured between p332_rung4.json and
#: the retaken baseline. Recorded here, NOT inside step 4b (and not inside rung 5, whose
#: baseline is step 4b's gold): it is the injury and QB half of the builder-cutoff move, and
#: it lands in production gold at step 4b's rebuild.
PHASE332_STEP4B_CARRY_IN_CAUSE: str = (
    "PRE-STEP-4b CARRY-IN: Plan 33.2-13's injury and QB builders moved onto each game's own "
    "lock (commits dfe241e, 2c36173, 501e47c -- code only, no gold rebuild, made after rung 4 "
    "was built): injury reports admitted only when timed at or before the lock, 2025+ injury "
    "rows the honest unknown (upstream dropped date_modified), the latest admitted report per "
    "player, exact neutral availability for a team that admitted nothing; 2025 starting QBs "
    "from the depth chart published at or before the lock; play-by-play admitted per game "
    "when that game ended by the lock; the per-(season, week, lock) rolling cache. It is the "
    "injury and starting-QB / QB-play-by-play half of rung 5's declared builder-cutoff cause, "
    "measured before step 4b rather than folded into it"
)

#: DECLARED BEFORE THE CARRY-IN WAS MEASURED (2026-09-21; this commit precedes the first read
#: of the scratch fingerprint): the columns the injury and QB builders emit into gold.
#: Canonical names. A carry-in column outside this set halts the ladder for investigation.
PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "injury": (
        "away_availability_coverage",
        "away_availability_fraction",
        "away_backup_quality_delta",
        "away_date_modified_coverage",
        "away_injury_coverage",
        "away_qb_out_flag",
        "home_availability_coverage",
        "home_availability_fraction",
        "home_backup_quality_delta",
        "home_date_modified_coverage",
        "home_injury_coverage",
        "home_qb_out_flag",
    ),
    "qb": ("away_qb_adjustment", "home_qb_adjustment"),
}


def phase332_step4b_carry_in(
    fingerprint_dir: Path | str = FINGERPRINT_DIR,
) -> dict[str, object]:
    """The measured pre-step-4b carry-in: p332_rung4.json against the retaken baseline.

    Returns the non-clock columns that moved (unioned over the three matrices, with their
    seasons), the per-builder split against ``PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS``,
    every moved column OUTSIDE the prediction (which must be empty) and the per-matrix
    structure (added / removed columns and rows before and after).
    """
    directory = Path(fingerprint_dir)
    original = json.loads(
        rung_document_path(
            directory, PHASE332_WEATHER_RUNG, PHASE332_RUNG_PREFIX
        ).read_text(encoding="utf-8")
    )
    retaken = json.loads(
        (directory / PHASE332_RETRACTABLE_ROOF_STEP_BASELINE_DOCUMENT).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(original, retaken)
    moved = _phase332_moved_seasons(report)
    predicted = {
        builder: {_canonical(c) for c in columns}
        for builder, columns in PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS.items()
    }
    union: set[str] = set().union(*predicted.values())
    structure = {
        matrix: {
            "added": list(report[matrix]["columns_added"]),
            "removed": list(report[matrix]["columns_removed"]),
            "rows": [report[matrix]["rows_before"], report[matrix]["rows_after"]],
        }
        for matrix in GOLD_MATRICES
    }
    return {
        "moved_seasons": moved,
        "by_builder": {
            builder: sorted(c for c in moved if c in columns)
            for builder, columns in predicted.items()
        },
        "outside_prediction": sorted(c for c in moved if c not in union),
        "structure": structure,
    }


def _attribute_p332_retractable_roof(
    detail: dict, diff: dict, verdict: dict, fail
) -> bool:
    """Extra step 4b of the `p332_` ladder: the retractable-roof fix's OWN judge.

    STRUCTURE: nothing added, nothing removed, width and rows unchanged (a surprise BLOCKS).
    VALUES: a changed column is attributed only when it is in the derived weather family.
    Anything else is UNATTRIBUTED and fails; the cause is never widened to fit it.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    verdict["changed_by_family"] = {"weather": []}
    label = "p332_ step 4b (a retractable roof is unknown at the lock)"
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        label,
        "Giving the retractable-roof games their day-before forecast",
    )
    explainable = phase332_weather_columns()
    for column in sorted(diff["changed"]):
        if column in explainable:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["weather"].append(column)
            continue
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' moved at {label} in season(s) "
            f"{', '.join(sorted(diff['changed'][column])) or '(none)'}, but it is not a "
            "weather-family column. The step's ONE cause is the retractable-roof rule; do "
            "NOT widen it to fit this diff"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_EXTRA_STEP_SIGNATURES[PHASE332_RETRACTABLE_ROOF_STEP] = (
    PHASE332_RETRACTABLE_ROOF_STEP_EXPECTED_SIGNATURE
)
PHASE332_EXTRA_STEP_ATTRIBUTORS[PHASE332_RETRACTABLE_ROOF_STEP] = (
    _attribute_p332_retractable_roof
)


# ---------------------------------------------------------------------------
# p332_ RUNG 5 -- EVERY BUILDER SELECTION CUTOFF ON THE PER-GAME LOCK (Plan 33.2-14 Task 3,
# SPEC R5, D33.2-01). Registered per Plan 33.2-08's <owned_protocol_rung_registration>: three
# names, one incremental cause-table append, one entry in EACH dispatch table, no new prefix
# branch.
#
# ONE CAUSE, EIGHT SELECTIONS. The rung is one rule change applied uniformly -- "a builder
# admits a row only when it was known at or before the target game's own lock" -- to the
# injury, starting-QB identity, QB play-by-play, letdown, rest-days, market-odds, snap-window
# and team-form selections. The breadth is one rule, not a bundle (D33.2-20). The day-before
# forecast fence is NOT in it: that selection is rung 4's cause and Plan 33.2-12's alone.
#
# DECLARED BEFORE THE REBUILD, PER BUILDER, from read-only measurement and not from the diff:
#
# * injury, qb -- EMPTY at this rung. Plan 33.2-13's builders moved these values, and their
#   movement already landed in production gold at extra step 4b, recorded there as Plan
#   33.2-13's MEASURED CARRY-IN (PHASE332_STEP4B_CARRY_IN_*). Rung 5 is judged against
#   p332_rung4b.json, which already carries it.
# * contextual (letdown, rest days), snaps -- EMPTY. Measured read-only 2026-09-21 over
#   2002-2025 (the previous executor's probe of the pre- and post-change builders): outputs
#   byte-identical. A team's previous game ends days before its next lock on every ordinary
#   schedule; a row here would have to be a NAMED rescheduled game.
# * team_form -- EMPTY, per D33.2-01's measurement: 0 games in 2002-2026 whose week-keyed
#   prior-game inputs include a result that ended after their day-before lock. The window is
#   per team and a team plays at most once a week. A team-form movement is a FINDING.
# * market -- the five market columns and their two arithmetic children. By owner ruling
#   2026-09-22 ("Only real capture times") a line counts only with a RECORDED capture time
#   (created_at) at or before the lock; every stored 2018-2025 line fails (created_at NULL,
#   or the 2026-09-05 backfill), so every 2018-2025 market value becomes the honest unknown.
#   target_ats and target_ou are final margin / total minus the gold snapshot_spread /
#   snapshot_total, so they move with them (not model inputs; disclosed here, before the
#   rebuild). Seasons: 2018-2025 only -- silver odds_snapshot holds no row before 2018, so
#   earlier games already carried no line.
#
# JUDGED AGAINST p332_rung4b.json, the entry before it (no retake: see the confirmation).
# ---------------------------------------------------------------------------

PHASE332_CUTOFF_RUNG: int = 5

PHASE332_CUTOFF_RUNG_CAUSE: str = (
    "EVERY BUILDER SELECTION CUTOFF MOVED FROM A FRAME-WIDE GLOBAL CLOCK TO THE TARGET GAME'S "
    "OWN DAY-BEFORE LOCK, p332_ rung 5 of Plan 33.2-14 (SPEC R5, D33.2-01), and NOTHING else. "
    "It is ONE rule change applied uniformly to eight selections, not eight unrelated "
    "corrections: injury reports, starting-QB identity and QB play-by-play (Plan 33.2-13), "
    "and the letdown flag, the rest-days window, the market-odds selection, the snap window "
    "and the team-form rolling window (this plan). A prior game counts only once it ENDED "
    "(kickoff plus the declared four hours) at or before the lock, a report or depth chart "
    "only when timed at or before it, and a market line only with a RECORDED capture time "
    "(created_at) at or before it (owner ruling 2026-09-22) -- so every stored 2018-2025 "
    "line, whose only time is a manufactured snapshot_ts label, becomes the honest unknown. "
    "The injury and QB half already landed in production gold at step 4b as Plan 33.2-13's "
    "measured carry-in, and the contextual, snap and team-form windows agree with their "
    "week-keyed predecessors on every ordinary schedule, so only the five market columns and "
    "their two arithmetic children (target_ats, target_ou) can move, in seasons 2018-2025. No "
    "column is added, none removed, no row moves"
)

#: The first season silver odds_snapshot holds any row (measured read-only 2026-09-22: 2,140
#: rows over 2018-2025, one consensus line per game). A market value can move in this season
#: and, through expanding normalization, in later ones -- never earlier.
PHASE332_CUTOFF_RUNG_EARLIEST_MARKET_SEASON: int = 2018

#: The five compressed market columns (features.market_anchors.MARKET_FEATURE_COLUMNS).
PHASE332_CUTOFF_RUNG_MARKET_COLUMNS: tuple[str, ...] = (
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
)

#: The market columns' ARITHMETIC CHILDREN in gold (scripts.build_features.
#: create_target_variables): child -> the gold market column it subtracts. Declared BEFORE the
#: rebuild (compare rung 1, where target_ats had to be disclosed at run time). Not model
#: inputs (models/temporal.py lists both as id columns).
PHASE332_CUTOFF_RUNG_MARKET_CHILDREN: dict[str, str] = {
    "target_ats": "snapshot_spread",
    "target_ou": "snapshot_total",
}

#: THE PREDICTION, PER BUILDER (Plan 33.2-14 Task 3; Codex MEDIUM). Declared before the rebuild.
#: A builder whose subset is EMPTY must move nothing; a moved column outside every subset halts
#: the rung for investigation. Canonical names.
PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER: dict[str, tuple[str, ...]] = {
    "injury": (),
    "qb": (),
    "contextual": (),
    "snaps": (),
    "team_form": (),
    "market": (
        *PHASE332_CUTOFF_RUNG_MARKET_COLUMNS,
        *PHASE332_CUTOFF_RUNG_MARKET_CHILDREN,
    ),
}


def phase332_cutoff_builder_columns() -> dict[str, frozenset[str]]:
    """The gold columns each builder whose cutoff this rung moves can reach, by builder.

    DERIVED from each builder's own declarations, never typed from a diff: the injury and QB
    columns from ``PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS``; the contextual selections'
    columns (rest days and the letdown flag) from ``features.contextual.
    NO_PRIOR_GAME_SIGNATURE``; the snap and team-form columns from their builders'
    ``no_information_signature``; the market columns and their two children declared above.
    Used to BISECT a moved column to the builder that produced it.
    """
    from features.contextual import NO_PRIOR_GAME_SIGNATURE
    from features.snaps import SnapCountBuilder
    from features.team_form import ROLLING_COLUMNS

    snaps = SnapCountBuilder.__new__(SnapCountBuilder)
    team_form = {
        f"{prefix}_{side}_{column}"
        for prefix in ("home", "away")
        for side in ("off", "def")
        for column in ROLLING_COLUMNS
    }
    families = {
        **{
            builder: set(columns)
            for builder, columns in PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS.items()
        },
        "contextual": set(NO_PRIOR_GAME_SIGNATURE),
        "snaps": set(SnapCountBuilder.no_information_signature(snaps)),
        "team_form": team_form,
        "market": {
            *PHASE332_CUTOFF_RUNG_MARKET_COLUMNS,
            *PHASE332_CUTOFF_RUNG_MARKET_CHILDREN,
        },
    }
    return {
        builder: frozenset(_canonical(c) for c in columns)
        for builder, columns in families.items()
    }


def phase332_cutoff_moved_by_builder(moved: Iterable[str]) -> dict[str, list[str]]:
    """*moved* columns split by the builder that emits them; ``unmapped`` = none of the six."""
    families = phase332_cutoff_builder_columns()
    split: dict[str, list[str]] = {builder: [] for builder in families}
    split["unmapped"] = []
    for column in sorted(_canonical(c) for c in moved):
        owners = [b for b, columns in families.items() if column in columns]
        split[owners[0] if owners else "unmapped"].append(column)
    return split


PHASE332_CUTOFF_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_CUTOFF_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_CUTOFF_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the market builder's five columns and their two arithmetic children "
        "(target_ats, target_ou), each only in seasons on or after 2018 (the first season any "
        "stored line exists; expanding normalization reaches its own and later seasons, never "
        "an earlier one); a child only where its parent moved in the same matrix and seasons. "
        "The injury, QB, contextual, snap and team-form subsets are EMPTY"
    ),
    "rows_changed": (
        "every 2018-2025 game's market values become the honest unknown (NULL in the builder's "
        "frame), which the gold build's existing missing handling carries into the same "
        "representation 2002-2017 games already hold (no line); spread_movement and "
        "total_movement were already that representation for every game (one stored snapshot "
        "per game gave a 0.0 movement), so they are expected not to move"
    ),
    "predicted_by_builder": PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER,
    "weather": "NOT expected to move: the forecast fence is rung 4's cause",
    "declared_families": ("market",),
    "family_mechanisms": {
        "market": (
            "features.market_anchors.admissible_market_rows: created_at at or before the "
            "game's lock (owner ruling 2026-09-22); gold children by create_target_variables"
        ),
    },
    "owner_ruling": (
        "2026-09-22, Plan 33.2-14 market-odds checkpoint, Option B 'Only real capture "
        "times': a market line is admissible only with a genuinely recorded capture time at "
        "or before the game's lock; the fabricated snapshot_ts stamps are never information "
        "times; target_ats / target_ou move with the market columns"
    ),
    "declared_before_the_rebuild": True,
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_CUTOFF_RUNG] = (
    PHASE332_CUTOFF_RUNG_CAUSE
)

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21). Rung 5 registers NO retaken
# baseline: p332_rung4b.json is production gold as step 4b wrote it, and nothing but rung 5's
# own cause has changed since.
PHASE332_CUTOFF_RUNG_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-22 before rung 5 rebuilt gold. p332_rung4b.json IS gold rebuilt from "
    "today's inputs minus exactly this rung's cause: step 4b's rebuild (2026-09-21, code "
    "5e190c9) is the last production gold write, the production data/ tree has been "
    "digest-identical since (1107 files: outputs/p332_rung5_probe_before.json, taken after "
    "that write, equals outputs/p332_rung5_scratch_prod_before.json), and every code change "
    "since is rung 5's own cause (the contextual, snap, team-form and market selections and "
    "their registration: fc1a5dd, dabcc83, 009126f). No carry-in, no retake"
)


def _attribute_p332_cutoff(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 5 of the `p332_` ladder: the builder-cutoff move's OWN judge.

    STRUCTURE: nothing added, nothing removed, width and rows unchanged (a surprise BLOCKS).
    VALUES: a market column is attributed only when every season it moved in is on or after
    ``PHASE332_CUTOFF_RUNG_EARLIEST_MARKET_SEASON``; a market CHILD only when its parent was
    attributed in this matrix and the child's seasons are a subset of the parent's. Anything
    else -- including any column of a builder whose predicted subset is EMPTY -- is
    UNATTRIBUTED and fails, naming the builder it belongs to so the surprise is bisected
    without a second rung. The cause is never widened to fit it.

    NOT the generic rung-5 path, deliberately: that is Phase 30's, keyed by rung NUMBER, and a
    registered-but-undispatched p332_ rung 5 would be judged by it and still print a verdict.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    label = "p332_ rung 5 (every builder selection cutoff on the per-game lock)"
    verdict["changed_by_family"] = {
        builder: [] for builder in PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER
    }
    blocking = _phase33_structure(
        detail,
        diff,
        fail,
        label,
        "Moving every builder selection onto each game's own lock",
    )
    floor = PHASE332_CUTOFF_RUNG_EARLIEST_MARKET_SEASON
    market = {_canonical(c) for c in PHASE332_CUTOFF_RUNG_MARKET_COLUMNS}
    children = {
        _canonical(child): _canonical(parent)
        for child, parent in PHASE332_CUTOFF_RUNG_MARKET_CHILDREN.items()
    }
    owners = phase332_cutoff_builder_columns()

    def _owner(column: str) -> str:
        found = [b for b, columns in owners.items() if column in columns]
        return found[0] if found else "none of the six builders whose cutoff moved"

    def _refuse(column: str, seasons: list[str], why: str) -> None:
        verdict["unattributed"].append(column)
        fail(
            f"column '{column}' (builder: {_owner(column)}) moved at {label} in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The rung's ONE cause is the "
            "builder-cutoff move and its prediction was declared per builder before the "
            "rebuild; do NOT widen it to fit this diff"
        )

    for column in sorted(set(diff["changed"]) - set(children)):
        seasons = sorted(diff["changed"][column])
        if column in market and seasons and all(int(s) >= floor for s in seasons):
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["market"].append(column)
            continue
        if column in market:
            _refuse(
                column,
                seasons,
                f"it moved in a season before {floor}, the first season any stored line "
                "exists, which expanding normalization cannot reach",
            )
        else:
            _refuse(column, seasons, "its builder's predicted subset is EMPTY")

    for column in sorted(set(diff["changed"]) & set(children)):
        seasons = sorted(diff["changed"][column])
        parent = children[column]
        parent_seasons = set(diff["changed"].get(parent, []))
        if (
            parent in verdict["attributed"]
            and seasons
            and set(seasons) <= parent_seasons
        ):
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["market"].append(column)
            continue
        _refuse(
            column,
            seasons,
            f"it is declared only as the arithmetic child of '{parent}', which was not "
            "attributed here in those seasons",
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_SIGNATURES[PHASE332_CUTOFF_RUNG] = PHASE332_CUTOFF_RUNG_EXPECTED_SIGNATURE
PHASE332_RUNG_ATTRIBUTORS[PHASE332_CUTOFF_RUNG] = _attribute_p332_cutoff


# ---------------------------------------------------------------------------
# p332_ RUNG 6 -- THE SNAP AND INJURY FEEDS WIRED AND POPULATED (Plan 33.2-15 Task 4, D33.2-16).
# Registered per Plan 33.2-08's <owned_protocol_rung_registration>: three names, one
# incremental cause-table append, one entry in EACH dispatch table, no new prefix branch.
#
# DECLARED BEFORE THE REBUILD, from the prospective-stamp ruling and not from "the feeds now
# have data":
#
# * snaps -- the 20 snap columns MOVE, in 2025 only. Silver snap_counts stopped at 2024, so
#   every 2025 game's window was the last eight 2024 games (team_form's dynamic window reaches
#   the prior season when the current one has nothing): 2025 weeks 3 and 15 read the SAME
#   values. Real 2025 snaps replace that. A snap count's information time is its game's END
#   instant (Plan 33.2-14's rule), so every one is admissible at later games' locks. Week-1
#   rows may not move (their window is still the 2024 tail). 2026 is NOT in gold under the
#   owner's --through-season 2025 ruling, so 2025 is the only season that can move.
# * injury, qb -- EMPTY. Every 2025 game was already basis="no_information" at rung 5 (upstream
#   dropped date_modified), and the 2025 rows this rung's ingest wrote carry the release
#   asset's 2026-09-07 stamp, AFTER every 2025 lock, so the builder admits none of them. The
#   capture rule becomes load-bearing only where a capture precedes a lock -- the forward
#   daily run -- and no such game is in a history rebuild. An injury movement would mean a
#   post-lock stamp was ADMITTED: a named finding, never a diff to absorb. availability_coverage
#   depends only on whether prior snap shares EXIST, which they did before (the 2024 tail) and
#   do after.
# * no pre-2025 row moves and no width changes: the generic per-season imputer is untouched
#   (the pre-coverage case is Plan 33.2-17's cause at rung 8), the empty-source guard never
#   fires on a full build (neither frame is empty), and every statistic that 2025 enters is
#   strictly-prior or within-season.
#
# JUDGED AGAINST p332_rung5.json (no retake: see the confirmation).
# ---------------------------------------------------------------------------

PHASE332_FEED_RUNG: int = 6

PHASE332_FEED_RUNG_CAUSE: str = (
    "THE SNAP AND INJURY FEEDS WIRED, p332_ rung 6 of Plan 33.2-15 (D33.2-16), and NOTHING else: "
    "both ingests run for 2025 and 2026, every row validated through a real silver schema "
    "(SnapCountSchema / InjurySchema, via validate_bronze_to_silver) and stamped with its "
    "release asset's publication time as capture provenance (upstream_captured_at), and the "
    "median fill for a family whose source frame is EMPTY replaced by NaN. Only the 20 snap "
    "columns can move, only in 2025 (real 2025 snaps replace the 2024 window every 2025 game "
    "read); the injury and QB columns cannot, because a 2025 row captured in 2026 carries a "
    "stamp after every 2025 lock and admits nothing. No column is added, none removed, no row "
    "moves"
)

#: The seasons a moved column may move in. 2026 is listed because the ingest populated it, but
#: gold stops at 2025 under the owner's --through-season ruling, so the PREDICTION is 2025 only.
PHASE332_FEED_RUNG_ALLOWED_SEASONS: tuple[str, ...] = ("2025", "2026")
PHASE332_FEED_RUNG_PREDICTED_SEASONS: tuple[str, ...] = ("2025",)


def phase332_feed_snap_columns() -> tuple[str, ...]:
    """The 20 snap columns, DERIVED from the snap builder's own ``no_information_signature``."""
    from features.snaps import SnapCountBuilder

    snaps = SnapCountBuilder.__new__(SnapCountBuilder)
    return tuple(
        sorted(_canonical(c) for c in SnapCountBuilder.no_information_signature(snaps))
    )


#: THE PREDICTION, PER BUILDER. Declared before the rebuild. ``snaps`` is derived at import
#: from the builder's declaration; the empty subsets are recorded, not omitted, so a movement
#: there arrives as a named finding.
PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER: dict[str, tuple[str, ...]] = {
    "snaps": phase332_feed_snap_columns(),
    "injury": (),
    "qb": (),
}


def phase332_feed_builder_columns() -> dict[str, frozenset[str]]:
    """The gold columns each feed builder can reach: snaps from its signature, injury / qb from
    ``PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS`` (the declared injury and QB families)."""
    return {
        "snaps": frozenset(phase332_feed_snap_columns()),
        "injury": frozenset(
            _canonical(c) for c in PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS["injury"]
        ),
        "qb": frozenset(
            _canonical(c) for c in PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS["qb"]
        ),
    }


def phase332_feed_moved_by_builder(moved: Iterable[str]) -> dict[str, list[str]]:
    """*moved* columns split by the feed builder that emits them; ``unmapped`` = none of them."""
    families = phase332_feed_builder_columns()
    split: dict[str, list[str]] = {builder: [] for builder in families}
    split["unmapped"] = []
    for column in sorted(_canonical(c) for c in moved):
        owners = [b for b, columns in families.items() if column in columns]
        split[owners[0] if owners else "unmapped"].append(column)
    return split


#: THE INJURY-FRESHNESS LIMITATION and the ESPN ruling, recorded together in the rung record
#: (Plan 33.2-15's espn_ruling). The single source is data.upstream_asset_stamp.
PHASE332_FEED_RUNG_ESPN_RULING: str = (
    "The ESPN injuries endpoint is NOT adopted as a model input and NOT archived: it carries game "
    "status and prose rather than the official practice-participation report the model is "
    "trained on (train/serve mismatch), it is current-season only so 2009-2024 cannot be "
    "rebuilt from it (a mid-history character change), and it is an undocumented endpoint with "
    "no published contract. The daily nflverse file stands and the limitation is stated"
)

PHASE332_FEED_RUNG_EXPECTED_SIGNATURE: dict[str, object] = {
    "rung": PHASE332_FEED_RUNG,
    "prefix": PHASE332_RUNG_PREFIX,
    "cause": PHASE332_FEED_RUNG_CAUSE,
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged",
    "width": "unchanged",
    "columns_changed": (
        "restricted to the snap builder's 20 columns, each only in 2025 (2026 is not in gold); "
        "the injury and QB subsets are EMPTY -- an injury movement means a post-lock capture "
        "stamp was admitted"
    ),
    "rows_changed": (
        "2025 games whose snap window now reads real 2025 snaps instead of the last eight 2024 "
        "games; week-1 rows may not move, since their window is still the 2024 tail"
    ),
    "predicted_by_builder": PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER,
    "predicted_seasons": PHASE332_FEED_RUNG_PREDICTED_SEASONS,
    "allowed_seasons": PHASE332_FEED_RUNG_ALLOWED_SEASONS,
    "declared_families": ("snaps",),
    "family_mechanisms": {
        "snaps": (
            "features.snaps.SnapCountBuilder over silver snap_counts, now populated for "
            "2025-2026 by scripts/ingest_snaps.py through SnapCountSchema"
        ),
    },
    "prospective_stamp_ruling": (
        "the release asset's updated_at is CAPTURE provenance for the file fetched now, a "
        "row's information time only where it is at or before that row's game's lock; a "
        "backfill of a completed season admits nothing on that basis"
    ),
    "disclosed": (
        "the first 2025 injury upsert exposed a silver-writer defect (a stored "
        "datetime64[us, UTC] column met fresh datetime64[ns, UTC] rows and was written as "
        "TEXT); data.storage.upsert_silver now aligns the pair and refuses an object result, "
        "and silver injuries.date_modified was repaired once from the stored strings, verified "
        "row-for-row against the Phase-28 bronze capture (81,408 pre-2025 rows identical). "
        "Values are unchanged, so no gold column can move from it"
    ),
    "injury_freshness_limitation": (
        "A Wednesday 18:00 ET lock for a Thursday game sees the nflverse injury file as it "
        "stood at roughly 08:38 ET that morning (measured 2026-09-15), so it misses that "
        "Wednesday's practice report"
    ),
    "espn_ruling": PHASE332_FEED_RUNG_ESPN_RULING,
    "declared_before_the_rebuild": True,
}

RUNG_CAUSES_BY_PREFIX[PHASE332_RUNG_PREFIX][PHASE332_FEED_RUNG] = (
    PHASE332_FEED_RUNG_CAUSE
)

# THE BASELINE WAS CONFIRMED, NOT ASSUMED (owner ruling 2026-09-21). No retake: p332_rung5.json is
# production gold as rung 5 wrote it, production data/ was digest-identical to rung 5's
# after-state when this rung began, and every code change since is rung 6's own cause.
PHASE332_FEED_RUNG_BASELINE_CONFIRMATION: str = (
    "CONFIRMED 2026-09-22 before rung 6 wrote anything. p332_rung5.json IS gold rebuilt from "
    "today's inputs minus exactly this rung's cause: rung 5's rebuild is the last production "
    "gold write, production data/ was digest-identical to outputs/p332_rung5_after.json (1107 "
    "files, verified immediately before outputs/p332_rung6_before.json was taken), and every "
    "code change since b090892 is rung 6's own (the stamp module and schemas, the gated and "
    "stamped ingests, the empty-source guard, the empty-snap-table fix, the silver-writer "
    "alignment). No carry-in, no retake"
)


def _attribute_p332_feed(detail: dict, diff: dict, verdict: dict, fail) -> bool:
    """Rung 6 of the `p332_` ladder: the snap and injury feeds' OWN judge.

    STRUCTURE: nothing added, nothing removed, width and rows unchanged (a surprise BLOCKS).
    VALUES: a snap column is attributed only when every season it moved in is in
    ``PHASE332_FEED_RUNG_ALLOWED_SEASONS``. Anything else is UNATTRIBUTED and fails, naming its
    builder: a pre-2025 snap movement means a second cause leaked into the rung, and ANY injury
    or QB movement means a post-lock capture stamp was admitted. The cause is never widened.

    NOT the generic rung-6 path, deliberately: a registered-but-undispatched p332_ rung would be
    judged by Phase 30's rung-number-keyed path and still print a verdict.

    Returns:
        Whether this matrix BLOCKS the phase (a structural surprise only).
    """
    label = "p332_ rung 6 (the snap and injury feeds wired and populated)"
    verdict["changed_by_family"] = {
        builder: [] for builder in PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER
    }
    blocking = _phase33_structure(
        detail, diff, fail, label, "Wiring the snap and injury feeds"
    )
    snaps = set(PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER["snaps"])
    allowed = set(PHASE332_FEED_RUNG_ALLOWED_SEASONS)
    owners = phase332_feed_builder_columns()

    for column in sorted(diff["changed"]):
        seasons = sorted(diff["changed"][column])
        if column in snaps and seasons and set(seasons) <= allowed:
            verdict["attributed"].append(column)
            verdict["changed_by_family"]["snaps"].append(column)
            continue
        owner = next(
            (b for b, columns in owners.items() if column in columns),
            "none of the feed builders",
        )
        verdict["unattributed"].append(column)
        if column in snaps:
            why = (
                "it moved in a season before 2025, which only a SECOND cause could reach "
                "(the generic per-season imputer is Plan 33.2-17's)"
            )
        elif owner in ("injury", "qb"):
            why = (
                "the injury and QB subsets are EMPTY: a movement means a capture stamp "
                "after its game's lock was ADMITTED"
            )
        else:
            why = "it belongs to no builder this rung's cause reaches"
        fail(
            f"column '{column}' (builder: {owner}) moved at {label} in season(s) "
            f"{', '.join(seasons) or '(none)'}, but {why}. The rung's ONE cause is the "
            "feed wiring and its prediction was declared before the rebuild; do NOT widen it"
        )
    verdict["attributed"].sort()
    return blocking


PHASE332_RUNG_SIGNATURES[PHASE332_FEED_RUNG] = PHASE332_FEED_RUNG_EXPECTED_SIGNATURE
PHASE332_RUNG_ATTRIBUTORS[PHASE332_FEED_RUNG] = _attribute_p332_feed


def _attribute_rung2(diff: dict, verdict: dict, fail) -> None:
    """WR-06 may MOVE any imputed or clipped column; it may not FLATTEN one.

    WR-06 refits every imputation and winsorization bound on the strictly-prior
    seasons, so ANY column that had a value imputed or clipped may move. There is no
    per-column allow-list to check against -- the rung's discipline is structural
    (nothing added, removed or resized) plus the empty-diff refusal.

    That makes rung 2 a blanket attribution, and a blanket attribution cannot FAIL on
    a moved column. The phase measured what that costs: rung 2's first attempt
    "attributed perfectly cleanly -- ok, zero unattributed -- while having silently
    destroyed 18 columns", through a degenerate ``q01 == q99`` clip that flattened
    each of them to a constant.

    The one health claim the rung CAN make from what it already records is the one
    that failure mode trips: refitting a bound moves a measurement, it does not turn
    it into a constant. So a column that became indicator-valued is unattributed and
    fails here, rather than being disclosed in prose after the fact.

    The rung still does not DISCRIMINATE among the columns it attributes, and says so
    -- see ``discriminating`` on the verdict, which ``_print_attribution`` renders as
    ATTRIBUTED (NOT HEALTH-CHECKED) so a reader does not mistake it for the clean
    per-column verdict rung 1 gives.
    """
    verdict["discriminating"] = False

    details = diff["details"] or {}
    for column in sorted(diff["changed"]):
        meta = details.get(column, {})
        if _became_indicator(meta):
            verdict["unattributed"].append(column)
            _fail_became_indicator(
                column,
                2,
                "WR-06 refits bounds; it does not FLATTEN a measurement.",
                fail,
            )
        else:
            verdict["attributed"].append(column)


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


def _became_indicator(meta: dict) -> bool:
    """True when a column was a MEASUREMENT before the rung and an indicator after.

    ``FeatureMatrixBuilder._is_discrete_indicator`` returns True iff every non-null
    value is one of ``-1.0 / 0.0 / 1.0``, so a CONSTANT column at any of those three
    levels satisfies it. Gold's feature columns are expanding-window z-scores, which
    means a column the rebuild FLATTENS lands at a constant 0.0 and therefore
    acquires ``discrete_indicator_after == True``.

    That is a destroyed column wearing an exemption's costume, and it is not
    hypothetical here: rung 2's first attempt flattened columns through a degenerate
    ``q01 == q99`` clip. Recording the transition separately is what lets a rung
    refuse to attribute a column to a rule that only ever applied to columns which
    ALREADY were indicators.
    """
    return not bool(meta.get("discrete_indicator_before")) and bool(
        meta.get("discrete_indicator_after")
    )


def _fail_became_indicator(column: str, rung: int, rule: str, fail) -> None:
    """Report a measurement that became an indicator-valued (constant) column."""
    fail(
        f"column '{column}' was NOT a discrete indicator before rung {rung} and IS "
        f"one after. {rule} The likely cause is a degenerate clip FLATTENING the "
        "column to a constant -- the rung-2 first-attempt failure mode -- and a "
        "destroyed column must not attribute to itself"
    )


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
        # Attribute on the BEFORE side ONLY. CR-02's exemption is about columns the
        # winsorizer was always going to skip, i.e. ones that were ALREADY indicators
        # when the rung started. A column that BECAME one is a new fact about the
        # data, not an instance of the exemption -- and reading the two sides with
        # `or` let a flattened column supply its own excuse, coming back
        # `ok: True, unattributed: []`.
        if bool(meta.get("discrete_indicator_before")):
            verdict["attributed"].append(column)
            continue

        verdict["unattributed"].append(column)
        if _became_indicator(meta):
            _fail_became_indicator(
                column,
                1,
                "CR-02 exempts columns that are ALREADY indicators; it never turns a "
                "measurement into one.",
                fail,
            )
        else:
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
        elif _became_indicator((diff["details"] or {}).get(column, {})):
            blocking = True
            verdict["unattributed"].append(column)
            _fail_became_indicator(
                column,
                4,
                "A re-sync ADDS ROWS to a season; it does not flatten a column "
                "that already had values.",
                fail,
            )
        else:
            verdict["attributed"].append(column)

    return blocking


def _platform_record() -> dict[str, str]:
    """The facts that make a cross-machine digest disagreement DIAGNOSABLE.

    Recorded instead of introducing an epsilon. ``scripts/fingerprint_gold.py``
    already has a comparison convention -- EXACT bytes, via ``_column_bytes``'s
    IEEE-754 encoding -- and every prior phase's committed diff was produced
    under it. A second convention here would make this diff incomparable with
    Phase 30's, Phase 31's and Phase 33.1's, which is a worse failure than the
    one rounding would prevent. Rounding would also mask exactly the small
    systematic shift a bad derivation produces.
    """
    import platform

    import numpy as np

    blas = "unknown"
    try:
        config = np.__config__.get_info("blas_opt")  # type: ignore[attr-defined]
        blas = str(config.get("libraries", config)) if config else "unknown"
    except (AttributeError, KeyError, TypeError):
        try:
            blas = str(
                np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]  # type: ignore[attr-defined]
            )
        except (AttributeError, KeyError, TypeError):
            blas = "unknown"

    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "blas": blas,
    }


def _toml_escape(value: str) -> str:
    """Escape *value* for a TOML basic string."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _toml_array(values) -> str:
    """Render *values* as a single-line TOML array of basic strings."""
    return "[" + ", ".join(f'"{_toml_escape(str(v))}"' for v in values) + "]"


def _slice_label(entry) -> str:
    """Render a ``(season, week)`` slice as a lossless string.

    TOML has no null, so ``(2018, None)`` cannot round-trip as an array. A whole
    season is rendered as ``"2018"`` and a single week as ``"2018|1"`` -- which
    reads correctly and, unlike a sentinel integer, cannot be mistaken for week
    zero.
    """
    season, week = entry
    return str(season) if week is None else f"{season}|{week}"


def write_phase33_rebuild_diff(out_path: Path | str) -> Path:
    """Emit the COMMITTED per-rung record of what this phase's ladder moved.

    ``data/gold/`` and ``outputs/`` are both gitignored, so this file is the only
    way a fresh checkout can say WHAT each rung moved. That is the whole reason
    it is committed and the fingerprint documents are not.

    THE DECLARATIONS ARE READ FROM ``tests.phase33_state``, not restated here.
    That module is this repository's WITNESS pattern -- constants only, no
    project imports, no I/O, importable from any tier -- and it is where the
    expected and observed sets were recorded before and after the rebuild. The
    alternative is a second spelling of every declaration inside the generator,
    which is the D30-02 failure mode. The import is deferred so this module can
    still be used as a plain fingerprint reader.

    Args:
        out_path: Where to write the TOML. Refused if it points under ``data/``.

    Returns:
        The path written.
    """
    import tests.phase33_state as state

    out_path = reject_data_path(
        Path(out_path), what="the rebuild diff", suggestion="config/"
    )

    before = json.loads(
        rung_document_path(FINGERPRINT_DIR, 0, PHASE33_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        rung_document_path(FINGERPRINT_DIR, 1, PHASE33_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)

    added: set[str] = set()
    removed: set[str] = set()
    for matrix in GOLD_MATRICES:
        added |= set(report[matrix]["columns_added"])
        removed |= set(report[matrix]["columns_removed"])
    seasons_before = set(before["features_ou"]["seasons"])
    seasons_after = set(after["features_ou"]["seasons"])

    platform_record = _platform_record()
    baseline_widths = tuple(state.GOLD_WIDTHS_BEFORE_PHASE33_LADDER)
    after_widths = tuple(state.GOLD_WIDTHS_AFTER_EXEMPTION_RUNG)

    rungs: list[dict] = [
        {
            "rung": 0,
            "cause": (
                "BASELINE, NOT A REBUILD. A fingerprint of gold exactly as Phase "
                "33.1 left it, taken BEFORE anything was rebuilt. It is the "
                "control Phase 33.1 could not take in time, and the one artifact "
                "that cannot be recovered once rung 1 has overwritten gold."
            ),
            "rebuilt": False,
            "added_columns": [],
            "removed_columns": [],
            "added_seasons": [],
            "removed_seasons": [],
            "expected_slices": [],
            "observed_slices": [],
            "causal_columns": [],
            "widths_before": baseline_widths,
            "widths_after": baseline_widths,
            "attribution": "n/a -- a baseline is not attributed; it is what the rungs are attributed AGAINST",
            "unexplained": {},
        },
        {
            "rung": 1,
            "cause": PHASE33_EXEMPTION_RUNG_CAUSE,
            "rebuilt": True,
            "added_columns": sorted(added),
            "removed_columns": sorted(removed),
            "added_seasons": sorted(seasons_after - seasons_before),
            "removed_seasons": sorted(seasons_before - seasons_after),
            "expected_slices": [
                _slice_label(s)
                for s in state.GOLD_REBUILD_EXPECTED_CHANGED_SLICES_EXEMPTION
            ],
            "observed_slices": [
                _slice_label(s)
                for s in state.GOLD_REBUILD_OBSERVED_CHANGED_SLICES_EXEMPTION
            ],
            "causal_columns": sorted(state.GOLD_REBUILD_CAUSAL_COLUMNS_EXEMPTION),
            "widths_before": baseline_widths,
            "widths_after": after_widths,
            "attribution": (
                "FINDING, not BLOCKED. 25 of the pinned 26 level-preserved "
                "columns attributed; weather_coverage is the declared-but-"
                "UNCHANGED member because Phase 33.1 already exempted it. 18 "
                "columns moved outside the family and every one carries a "
                "written explanation below."
            ),
            "unexplained": {
                k: v
                for k, v in state.GOLD_REBUILD_UNEXPLAINED_CHANGES_EXEMPTION.items()
                if isinstance(k, str)
            },
        },
        {
            "rung": 2,
            "cause": PHASE33_ELO_RUNG_CAUSE,
            "rebuilt": False,
            "added_columns": [],
            "removed_columns": [],
            "added_seasons": [],
            "removed_seasons": [],
            "expected_slices": [
                _slice_label(s) for s in state.GOLD_REBUILD_EXPECTED_CHANGED_SLICES_ELO
            ],
            "observed_slices": [
                _slice_label(s) for s in state.GOLD_REBUILD_OBSERVED_CHANGED_SLICES_ELO
            ],
            "causal_columns": sorted(
                set(state.GOLD_REBUILD_CAUSAL_COLUMNS_ELO)
                | set(PHASE33_ELO_DERIVED_SITUATIONAL_COLUMNS)
            ),
            "widths_before": baseline_widths,
            "widths_after": after_widths,
            "attribution": (
                "OK, 0 unattributed. A DECLARATION-ONLY rung: it re-judges the "
                "SAME p33_rung0 -> p33_rung1 transition under a second "
                "declaration and rebuilds nothing, so there is no p33_rung2.json "
                "fingerprint document. 14 Elo columns, 4 Elo-derived situational "
                "columns, 25 carried from rung 1."
            ),
            "unexplained": {
                k: v
                for k, v in state.GOLD_REBUILD_UNEXPLAINED_CHANGES_ELO.items()
                if isinstance(k, str)
            },
        },
    ]

    lines: list[str] = [
        "# " + "=" * 75,
        "# config/phase33_gold_rebuild_diff.toml -- the per-rung record of the",
        "# Phase-33 Wave-14 gold rebuild ladder (prefix p33_). Plan 33-14, R4.",
        "#",
        "# GENERATOR OUTPUT. Produced by",
        '#   python -c "import scripts.fingerprint_gold as f;'
        " f.write_phase33_rebuild_diff('config/phase33_gold_rebuild_diff.toml')\"",
        "# Do NOT hand-edit any value below: re-run the generator. A hand-edited",
        "# value is indistinguishable from a tampered one.",
        "#",
        "# WHY THIS FILE IS COMMITTED WHEN THE FINGERPRINTS ARE NOT. data/gold/ and",
        "# outputs/ are both gitignored, so a fresh checkout cannot read either the",
        "# gold this ladder moved or the documents that measured it. This file is the",
        "# only place the measurement survives a clone.",
        "#",
        "# NO EPSILON AND NO ROUNDING. The comparison convention is EXACT bytes, via",
        "# _column_bytes' IEEE-754 encoding, and every prior phase's committed diff was",
        "# produced under it. A second convention here would make this diff",
        "# incomparable with Phase 30's, Phase 31's and Phase 33.1's. The platform is",
        "# recorded instead, so a cross-machine disagreement is diagnosable rather than",
        "# mysterious.",
        "#",
        "# AN OMITTED LIST IS NOT AN EMPTY LIST. Every added/removed column and season",
        "# list is PRESENT below, empty where nothing moved, because 'absent' and 'we",
        "# measured nothing' must not read the same.",
        "#",
        "# ASCII only, no emoji (CLAUDE.md hard constraint).",
        "# " + "=" * 75,
        "",
        f'generated_at = "{datetime.now(UTC).isoformat()}"',
        f'rung_prefix = "{PHASE33_RUNG_PREFIX}"',
        'float_tolerance = "none -- EXACT byte comparison of the IEEE-754 encoding'
        " produced by scripts.fingerprint_gold._column_bytes. No epsilon and no"
        ' rounding step is applied anywhere in this ladder."',
        "",
        "[platform]",
    ]
    for key, value in sorted(platform_record.items()):
        lines.append(f'{key} = "{_toml_escape(value)}"')

    for entry in rungs:
        lines.extend(
            [
                "",
                f"[rung.{entry['rung']}]",
                f"rung = {entry['rung']}",
                f'prefix = "{PHASE33_RUNG_PREFIX}"',
                f"rebuilt = {'true' if entry['rebuilt'] else 'false'}",
                f'cause = "{_toml_escape(str(entry["cause"]))}"',
                f"added_columns = {_toml_array(entry['added_columns'])}",
                f"removed_columns = {_toml_array(entry['removed_columns'])}",
                f"added_seasons = {_toml_array(entry['added_seasons'])}",
                f"removed_seasons = {_toml_array(entry['removed_seasons'])}",
                f"expected_slices = {_toml_array(entry['expected_slices'])}",
                f"observed_slices = {_toml_array(entry['observed_slices'])}",
                f"causal_columns = {_toml_array(entry['causal_columns'])}",
                f"widths_before = {_toml_array(entry['widths_before'])}",
                f"widths_after = {_toml_array(entry['widths_after'])}",
                f'attribution = "{_toml_escape(str(entry["attribution"]))}"',
                "",
                f"[rung.{entry['rung']}.unexplained]",
            ]
        )
        for column, why in sorted(entry["unexplained"].items()):
            lines.append(f'{column} = "{_toml_escape(why)}"')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_phase332_rebuild_diff(
    out_path: Path | str, fingerprint_dir: Path | str = FINGERPRINT_DIR
) -> Path:
    """Emit the COMMITTED per-rung record of the `p332_` ladder (rungs 0 .. 6, steps 3b, 3c, 4b).

    ``data/gold/`` and ``outputs/`` are both gitignored, so this file is the only
    place a fresh checkout can read what the ladder moved. Unlike
    ``write_phase33_rebuild_diff`` it reads nothing from ``tests.phase33_state``:
    this ladder's witnesses are appended AFTER this file is generated, so the
    attribution is recomputed here from the two fingerprint documents and the
    declared signature -- the same judge ``attribute_rung`` applies.

    A rung whose diff is EMPTY is recorded as declared-but-not-run (the Phase-33
    precip precedent), never as a rung that ran.

    Later `p332_` rungs extend this writer with one entry each. Rung 2 (Plan 33.2-09) is
    appended by ``_phase332_venue_rung_lines`` when its document exists.

    Args:
        out_path: Where to write the TOML. Refused if it points under ``data/``.
        fingerprint_dir: The directory holding the ``p332_rung*.json`` ladder.

    Returns:
        The path written.
    """
    out_path = reject_data_path(
        Path(out_path), what="the rebuild diff", suggestion="config/"
    )
    require_rung_ladder(fingerprint_dir, PHASE332_ODDS_RUNG + 1, PHASE332_RUNG_PREFIX)
    original_rung0_path = rung_document_path(fingerprint_dir, 0, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, PHASE332_ODDS_RUNG)
    original_rung0 = json.loads(original_rung0_path.read_text(encoding="utf-8"))
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    retaken = baseline_path != original_rung0_path
    # The pre-ladder carry-in: original rung 0 -> the retaken baseline. Recorded, never
    # attributed to rung 1 (PHASE332_RUNG0_CARRY_IN_CAUSE).
    carry_in: dict[str, list[str]] = {}
    if retaken:
        carry_report = compare_fingerprints(original_rung0, before)
        for matrix in GOLD_MATRICES:
            for column, seasons in carry_report[matrix]["columns_changed"].items():
                if not _is_build_clock(column):
                    merged = set(carry_in.get(column, [])) | set(seasons)
                    carry_in[column] = sorted(merged)
    after = json.loads(
        rung_document_path(
            fingerprint_dir, PHASE332_ODDS_RUNG, PHASE332_RUNG_PREFIX
        ).read_text(encoding="utf-8")
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report,
        PHASE332_ODDS_RUNG,
        before=before,
        after=after,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )

    widths_before = [before[m]["width"] for m in GOLD_MATRICES]
    widths_after = [after[m]["width"] for m in GOLD_MATRICES]
    moved = verdict["non_clock_moves"]
    ran = bool(moved)
    seasons_by_column: dict[str, list[str]] = {}
    for matrix in GOLD_MATRICES:
        for column, seasons in report[matrix]["columns_changed"].items():
            if not _is_build_clock(column):
                merged = set(seasons_by_column.get(column, [])) | set(seasons)
                seasons_by_column[column] = sorted(merged)
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )

    lines: list[str] = [
        "# " + "=" * 75,
        "# config/phase332_gold_rebuild_diff.toml -- the per-rung record of the",
        "# Phase-33.2 gold rebuild ladder (prefix p332_). Plan 33.2-08 opens it.",
        "#",
        "# GENERATOR OUTPUT. Produced by",
        '#   python -c "import scripts.fingerprint_gold as f;'
        " f.write_phase332_rebuild_diff('config/phase332_gold_rebuild_diff.toml')\"",
        "# Do NOT hand-edit any value below: re-run the generator.",
        "#",
        "# NO EPSILON AND NO ROUNDING: EXACT bytes of the IEEE-754 encoding produced by",
        "# scripts.fingerprint_gold._column_bytes, as every prior ladder. ASCII only.",
        "# " + "=" * 75,
        "",
        f'generated_at = "{datetime.now(UTC).isoformat()}"',
        f'rung_prefix = "{PHASE332_RUNG_PREFIX}"',
        'float_tolerance = "none -- EXACT byte comparison"',
        "",
        "[platform]",
    ]
    for key, value in sorted(_platform_record().items()):
        lines.append(f'{key} = "{_toml_escape(value)}"')
    widths_original = [original_rung0[m]["width"] for m in GOLD_MATRICES]
    lines.extend(
        [
            "",
            "[rung.0]",
            "rung = 0",
            f'prefix = "{PHASE332_RUNG_PREFIX}"',
            "rebuilt = false",
            'cause = "BASELINE, NOT A REBUILD. Gold as it stood on disk after BOTH silver '
            "odds corrections (Tasks 2 and 3) landed and before the rung-1 rebuild "
            'overwrote it (last built 2026-09-14)."',
            f'document = "{original_rung0_path.name}"',
            f"widths = {_toml_array(widths_original)}",
        ]
    )
    if retaken:
        lines.extend(
            [
                "superseded_as_rung1_baseline = true",
                "",
                "[rung.0.retaken]",
                f'document = "{baseline_path.name}"',
                "rebuilt = true",
                f'why = "{_toml_escape(PHASE332_RETAKEN_BASELINE_REASONS[PHASE332_ODDS_RUNG])}"',
                f"widths = {_toml_array(widths_before)}",
                "",
                "[rung.0.carry_in]",
                f'from_document = "{original_rung0_path.name}"',
                f'to_document = "{baseline_path.name}"',
                f'cause = "{_toml_escape(PHASE332_RUNG0_CARRY_IN_CAUSE)}"',
                f"moved_columns = {_toml_array(sorted(carry_in))}",
                "",
                "[rung.0.carry_in.moved_seasons]",
            ]
        )
        for column, seasons in sorted(carry_in.items()):
            lines.append(f"{column} = {_toml_array(seasons)}")
    lines.extend(
        [
            "",
            f"[rung.{PHASE332_ODDS_RUNG}]",
            f"rung = {PHASE332_ODDS_RUNG}",
            f'prefix = "{PHASE332_RUNG_PREFIX}"',
            f"rebuilt = {'true' if ran else 'false'}",
            f'baseline_document = "{baseline_path.name}"',
            f'cause = "{_toml_escape(PHASE332_ODDS_RUNG_CAUSE)}"',
            f"declared_columns = {_toml_array(sorted(PHASE332_ODDS_MARKET_SOURCES))}",
            "disclosed_at_run_time_columns = "
            f"{_toml_array(sorted(PHASE332_ODDS_RUN_TIME_DISCLOSED_CHILDREN))}",
            f'disclosed_at_run_time = "{_toml_escape(PHASE332_ODDS_RUN_TIME_DISCLOSURE)}"',
            f"widths_before = {_toml_array(widths_before)}",
            f"widths_after = {_toml_array(widths_after)}",
            f"moved_columns = {_toml_array(moved)}",
            f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
            f"unattributed_columns = {_toml_array(unattributed)}",
            f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
            f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
            f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[PHASE332_ODDS_RUNG].__name__}"',
        ]
    )
    if not ran:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the rung is '
            'recorded as declared-but-not-run rather than as a rung that ran"'
        )
    lines.extend(["", f"[rung.{PHASE332_ODDS_RUNG}.moved_seasons]"])
    for column, seasons in sorted(seasons_by_column.items()):
        lines.append(f"{column} = {_toml_array(seasons)}")

    venue_document = rung_document_path(
        fingerprint_dir, PHASE332_VENUE_RUNG, PHASE332_RUNG_PREFIX
    )
    if venue_document.exists():
        lines.extend(_phase332_venue_rung_lines(fingerprint_dir))

    schedule_document = rung_document_path(
        fingerprint_dir, PHASE332_SCHEDULE_MOVE_RUNG, PHASE332_RUNG_PREFIX
    )
    if schedule_document.exists():
        lines.extend(_phase332_schedule_move_rung_lines(fingerprint_dir))

    surface_document = rung_document_path(
        fingerprint_dir, PHASE332_SURFACE_STEP, PHASE332_RUNG_PREFIX
    )
    if surface_document.exists():
        lines.extend(_phase332_surface_step_lines(fingerprint_dir))

    kickoff_document = rung_document_path(
        fingerprint_dir, PHASE332_KICKOFF_HOUR_STEP, PHASE332_RUNG_PREFIX
    )
    if kickoff_document.exists():
        lines.extend(_phase332_kickoff_hour_step_lines(fingerprint_dir))

    weather_document = rung_document_path(
        fingerprint_dir, PHASE332_WEATHER_RUNG, PHASE332_RUNG_PREFIX
    )
    if weather_document.exists():
        lines.extend(_phase332_weather_rung_lines(fingerprint_dir))

    retractable_document = rung_document_path(
        fingerprint_dir, PHASE332_RETRACTABLE_ROOF_STEP, PHASE332_RUNG_PREFIX
    )
    if retractable_document.exists():
        lines.extend(_phase332_retractable_roof_step_lines(fingerprint_dir))

    cutoff_document = rung_document_path(
        fingerprint_dir, PHASE332_CUTOFF_RUNG, PHASE332_RUNG_PREFIX
    )
    if cutoff_document.exists():
        lines.extend(_phase332_cutoff_rung_lines(fingerprint_dir))

    feed_document = rung_document_path(
        fingerprint_dir, PHASE332_FEED_RUNG, PHASE332_RUNG_PREFIX
    )
    if feed_document.exists():
        lines.extend(_phase332_feed_rung_lines(fingerprint_dir))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _phase332_moved_seasons(report: dict) -> dict[str, list[str]]:
    """Non-clock column -> the seasons it moved in, unioned over the three matrices."""
    seasons_by_column: dict[str, list[str]] = {}
    for matrix in GOLD_MATRICES:
        for column, seasons in report[matrix]["columns_changed"].items():
            if not _is_build_clock(column):
                merged = set(seasons_by_column.get(column, [])) | set(seasons)
                seasons_by_column[column] = sorted(merged)
    return seasons_by_column


def _phase332_venue_rung_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` rung 2 (Plan 33.2-09), recomputed from the ladder.

    Judged by the same `attribute_rung` call the CLI makes, against the rung's baseline
    document (its ladder predecessor: PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION records
    why no retake was needed).
    """
    require_rung_ladder(fingerprint_dir, PHASE332_VENUE_RUNG + 1, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(
        fingerprint_dir, PHASE332_VENUE_RUNG
    )
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(
            fingerprint_dir, PHASE332_VENUE_RUNG, PHASE332_RUNG_PREFIX
        ).read_text(encoding="utf-8")
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report,
        PHASE332_VENUE_RUNG,
        before=before,
        after=after,
        rung_prefix=PHASE332_RUNG_PREFIX,
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_VENUE_RUNG_EXPECTED_SIGNATURE
    rung = PHASE332_VENUE_RUNG
    lines = [
        "",
        f"[rung.{rung}]",
        f"rung = {rung}",
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        f'baseline_confirmation = "{_toml_escape(PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION)}"',
        "baseline_confirmation_document = "
        f'"{PHASE332_VENUE_RUNG_BASELINE_CONFIRMATION_DOCUMENT}"',
        f'cause = "{_toml_escape(PHASE332_VENUE_RUNG_CAUSE)}"',
        f'correction_record = "{PHASE332_VENUE_CORRECTION_RECORD.as_posix()}"',
        "declared_columns = "
        f"{_toml_array(sorted(phase332_stadium_dependent_columns()))}",
        f"declared_seasons = {_toml_array([str(PHASE332_VENUE_CORRECTED_SEASON)])}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f'weather = "{_toml_escape(str(signature["weather"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[rung].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the rung is '
            'recorded as declared-but-not-run rather than as a rung that ran"'
        )
    lines.extend(["", f"[rung.{rung}.moved_seasons]"])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_schedule_move_rung_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` rung 3 (Plan 33.2-10), recomputed from the ladder.

    Judged by the same `attribute_rung` call the CLI makes, against the rung's baseline
    document (its ladder predecessor: PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION
    records why no retake was needed).
    """
    rung = PHASE332_SCHEDULE_MOVE_RUNG
    require_rung_ladder(fingerprint_dir, rung + 1, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, rung)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, rung, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, rung, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_SCHEDULE_MOVE_RUNG_EXPECTED_SIGNATURE
    floor = phase332_schedule_move_earliest_season()
    lines = [
        "",
        f"[rung.{rung}]",
        f"rung = {rung}",
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION)}"',
        "baseline_confirmation_document = "
        f'"{PHASE332_SCHEDULE_MOVE_RUNG_BASELINE_CONFIRMATION_DOCUMENT}"',
        f'cause = "{_toml_escape(PHASE332_SCHEDULE_MOVE_RUNG_CAUSE)}"',
        f'move_table = "{PHASE332_SCHEDULE_MOVE_TABLE.as_posix()}"',
        "post_lock_real_moves = "
        f"{_toml_array(sorted(phase332_post_lock_real_moves()))}",
        f"declared_columns = {_toml_array(list(phase332_schedule_fact_columns()))}",
        f"declared_season_floor = {floor if floor is not None else 0}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f'weather = "{_toml_escape(str(signature["weather"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[rung].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the rung is '
            'recorded as declared-but-not-run rather than as a rung that ran"'
        )
    lines.extend(["", f"[rung.{rung}.moved_seasons]"])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_surface_step_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` extra step 3b (Plan 33.2-10), recomputed.

    Judged by the same `attribute_rung` call the CLI makes, against the rung the step
    follows (PHASE332_SURFACE_STEP_BASELINE_CONFIRMATION records why no retake was needed).
    """
    step = PHASE332_SURFACE_STEP
    require_rung_ladder(fingerprint_dir, step, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, step)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, step, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, step, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_SURFACE_STEP_EXPECTED_SIGNATURE
    floor = phase332_surface_step_earliest_season()
    reclassified = sorted(
        game
        for game, season in phase332_surface_reclassified_games().items()
        if season <= 2025
    )
    lines = [
        "",
        f'[rung."{step}"]',
        f'rung = "{step}"',
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        "extra_step = true",
        f"follows_rung = {PHASE332_SURFACE_STEP_FOLLOWS}",
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_SURFACE_STEP_BASELINE_CONFIRMATION)}"',
        f'cause = "{_toml_escape(PHASE332_SURFACE_STEP_CAUSE)}"',
        f"declared_columns = {_toml_array(list(PHASE332_SURFACE_STEP_COLUMNS))}",
        f"declared_season_floor = {floor if floor is not None else 0}",
        f"reclassified_games_through_2025 = {_toml_array(reclassified)}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_EXTRA_STEP_ATTRIBUTORS[step].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the step is '
            'recorded as declared-but-not-run rather than as a step that ran"'
        )
    lines.extend(["", f'[rung."{step}".moved_seasons]'])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_kickoff_hour_step_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` extra step 3c (Plan 33.2-12), recomputed.

    Judged by the same `attribute_rung` call the CLI makes, against step 3b, the ladder
    entry before it (PHASE332_KICKOFF_HOUR_STEP_BASELINE_CONFIRMATION records why no retake
    was needed).
    """
    step = PHASE332_KICKOFF_HOUR_STEP
    require_rung_ladder(fingerprint_dir, step, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, step)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, step, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, step, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_KICKOFF_HOUR_STEP_EXPECTED_SIGNATURE
    floor = phase332_kickoff_hour_step_earliest_season()
    rest_moved = sorted(
        game
        for game, season in phase332_kickoff_hour_rest_changes().items()
        if season <= 2025
    )
    lines = [
        "",
        f'[rung."{step}"]',
        f'rung = "{step}"',
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        "extra_step = true",
        f"follows_rung = {PHASE332_KICKOFF_HOUR_STEP_FOLLOWS}",
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_KICKOFF_HOUR_STEP_BASELINE_CONFIRMATION)}"',
        f'cause = "{_toml_escape(PHASE332_KICKOFF_HOUR_STEP_CAUSE)}"',
        f"declared_columns = {_toml_array(list(PHASE332_KICKOFF_HOUR_STEP_COLUMNS))}",
        f"declared_season_floor = {floor if floor is not None else 0}",
        'correction_record = "config/kickoff_hour_corrections.toml"',
        f"rest_count_moved_games_through_2025 = {_toml_array(rest_moved)}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_EXTRA_STEP_ATTRIBUTORS[step].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the step is '
            'recorded as declared-but-not-run rather than as a step that ran"'
        )
    lines.extend(["", f'[rung."{step}".moved_seasons]'])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_weather_rung_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` rung 4 (Plan 33.2-12), recomputed.

    Judged by the same `attribute_rung` call the CLI makes, against step 3c, the ladder entry
    before it. Carries the accepted live-versus-history provider mismatch with the rung.
    """
    rung = PHASE332_WEATHER_RUNG
    require_rung_ladder(fingerprint_dir, rung, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, rung)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, rung, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, rung, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    removed = sorted({c for m in GOLD_MATRICES for c in report[m]["columns_removed"]})
    signature = PHASE332_WEATHER_RUNG_EXPECTED_SIGNATURE
    lines = [
        "",
        f"[rung.{rung}]",
        f"rung = {rung}",
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        f"rebuilt = {'true' if verdict['non_clock_moves'] or removed else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_WEATHER_RUNG_BASELINE_CONFIRMATION)}"',
        f'cause = "{_toml_escape(PHASE332_WEATHER_RUNG_CAUSE)}"',
        f"declared_removed_columns = {_toml_array(list(PHASE332_WEATHER_RUNG_REMOVED_COLUMNS))}",
        f"declared_family = {_toml_array(sorted(phase332_weather_columns()))}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"removed_columns = {_toml_array(removed)}",
        f"moved_columns = {_toml_array(verdict['non_clock_moves'])}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[rung].__name__}"',
        f'provider_mismatch = "{_toml_escape(PHASE332_WEATHER_RUNG_PROVIDER_MISMATCH)}"',
        "live_minus_history_wind_offset_mph = 1.56",
        "wind_band_edge_mph = 5.0",
    ]
    lines.extend(["", f"[rung.{rung}.moved_seasons]"])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_retractable_roof_step_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` extra step 4b (Plan 33.2-14), recomputed.

    Judged by the same ``attribute_rung`` call the CLI makes, against the RETAKEN baseline
    (``PHASE332_RETAKEN_BASELINES``). The carry-in between p332_rung4.json and that baseline
    -- Plan 33.2-13's measured effect -- is recorded in its own table, beside the step and
    never inside it.
    """
    step = PHASE332_RETRACTABLE_ROOF_STEP
    require_rung_ladder(fingerprint_dir, step, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, step)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, step, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, step, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_RETRACTABLE_ROOF_STEP_EXPECTED_SIGNATURE
    carry_in = phase332_step4b_carry_in(fingerprint_dir)
    carry_seasons = cast("dict[str, list[str]]", carry_in["moved_seasons"])
    carry_by_builder = cast("dict[str, list[str]]", carry_in["by_builder"])
    carry_outside = cast("list[str]", carry_in["outside_prediction"])
    lines = [
        "",
        f'[rung."{step}"]',
        f'rung = "{step}"',
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        "extra_step = true",
        f"follows_rung = {PHASE332_RETRACTABLE_ROOF_STEP_FOLLOWS}",
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_retaken_why = "
        f'"{_toml_escape(PHASE332_RETAKEN_BASELINE_REASONS[step])}"',
        f'cause = "{_toml_escape(PHASE332_RETRACTABLE_ROOF_STEP_CAUSE)}"',
        f"declared_family = {_toml_array(sorted(phase332_weather_columns()))}",
        f"redecided_games = {PHASE332_RETRACTABLE_ROOF_STEP_REDECIDED_GAMES}",
        f'archive_gap_game = "{PHASE332_RETRACTABLE_ROOF_STEP_ARCHIVE_GAP}"',
        'owner_ruling = "2026-09-21: the open/closed state of a retractable roof is NOT '
        'known at the lock (Treat as unknown at lock)"',
        f'model_visible_roof_flag = "{_toml_escape(str(signature["model_visible_roof_flag"]))}"',
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_EXTRA_STEP_ATTRIBUTORS[step].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the step is '
            'recorded as declared-but-not-run rather than as a step that ran"'
        )
    lines.extend(["", f'[rung."{step}".moved_seasons]'])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    lines.extend(
        [
            "",
            f'[rung."{step}".carry_in]',
            'from_document = "'
            + rung_document_path(
                fingerprint_dir, PHASE332_WEATHER_RUNG, PHASE332_RUNG_PREFIX
            ).name
            + '"',
            f'to_document = "{baseline_path.name}"',
            f'cause = "{_toml_escape(PHASE332_STEP4B_CARRY_IN_CAUSE)}"',
            f"moved_columns = {_toml_array(sorted(carry_seasons))}",
            f"outside_prediction = {_toml_array(carry_outside)}",
            "",
            f'[rung."{step}".carry_in.predicted_by_builder]',
        ]
    )
    for builder, columns in sorted(PHASE332_STEP4B_CARRY_IN_PREDICTED_COLUMNS.items()):
        lines.append(f"{builder} = {_toml_array(list(columns))}")
    lines.extend(["", f'[rung."{step}".carry_in.moved_by_builder]'])
    for builder, columns in sorted(carry_by_builder.items()):
        lines.append(f"{builder} = {_toml_array(columns)}")
    lines.extend(["", f'[rung."{step}".carry_in.moved_seasons]'])
    for column, seasons in sorted(carry_seasons.items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_cutoff_rung_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` rung 5 (Plan 33.2-14 Task 3), recomputed.

    Judged by the same ``attribute_rung`` call the CLI makes, against step 4b, the entry
    before it. Records the prediction PER BUILDER beside the measured split, so a surprise is
    bisected to its builder from the committed record alone.
    """
    rung = PHASE332_CUTOFF_RUNG
    require_rung_ladder(fingerprint_dir, rung, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, rung)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, rung, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, rung, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_CUTOFF_RUNG_EXPECTED_SIGNATURE
    lines = [
        "",
        f"[rung.{rung}]",
        f"rung = {rung}",
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_CUTOFF_RUNG_BASELINE_CONFIRMATION)}"',
        f'cause = "{_toml_escape(PHASE332_CUTOFF_RUNG_CAUSE)}"',
        f'owner_ruling = "{_toml_escape(str(signature["owner_ruling"]))}"',
        f"earliest_market_season = {PHASE332_CUTOFF_RUNG_EARLIEST_MARKET_SEASON}",
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[rung].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the rung is '
            'recorded as declared-but-not-run rather than as a rung that ran"'
        )
    lines.extend(["", f"[rung.{rung}.predicted_by_builder]"])
    for builder, columns in PHASE332_CUTOFF_RUNG_PREDICTED_BY_BUILDER.items():
        lines.append(f"{builder} = {_toml_array(list(columns))}")
    lines.extend(["", f"[rung.{rung}.moved_by_builder]"])
    for builder, columns in phase332_cutoff_moved_by_builder(moved).items():
        lines.append(f"{builder} = {_toml_array(columns)}")
    lines.extend(["", f"[rung.{rung}.market_children]"])
    for child, parent in PHASE332_CUTOFF_RUNG_MARKET_CHILDREN.items():
        lines.append(f'{child} = "{parent}"')
    lines.extend(["", f"[rung.{rung}.moved_seasons]"])
    for column, seasons in sorted(_phase332_moved_seasons(report).items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _phase332_feed_rung_lines(fingerprint_dir: Path | str) -> list[str]:
    """The committed record of `p332_` rung 6 (Plan 33.2-15 Task 4), recomputed from the ladder.

    Judged by the same ``attribute_rung`` call the CLI makes, against rung 5, the entry before
    it. Records the prediction PER BUILDER beside the measured split, the predicted and moved
    seasons, and -- together, as the plan requires -- the injury-freshness limitation and the
    ESPN ruling.
    """
    rung = PHASE332_FEED_RUNG
    require_rung_ladder(fingerprint_dir, rung, PHASE332_RUNG_PREFIX)
    baseline_path = phase332_baseline_document_path(fingerprint_dir, rung)
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(
        rung_document_path(fingerprint_dir, rung, PHASE332_RUNG_PREFIX).read_text(
            encoding="utf-8"
        )
    )
    report = compare_fingerprints(before, after)
    verdict = attribute_rung(
        report, rung, before=before, after=after, rung_prefix=PHASE332_RUNG_PREFIX
    )
    moved = verdict["non_clock_moves"]
    unattributed = sorted(
        {c for detail in verdict["matrices"].values() for c in detail["unattributed"]}
    )
    signature = PHASE332_FEED_RUNG_EXPECTED_SIGNATURE
    moved_seasons = _phase332_moved_seasons(report)
    lines = [
        "",
        f"[rung.{rung}]",
        f"rung = {rung}",
        f'prefix = "{PHASE332_RUNG_PREFIX}"',
        f"rebuilt = {'true' if moved else 'false'}",
        f'baseline_document = "{baseline_path.name}"',
        "baseline_confirmation = "
        f'"{_toml_escape(PHASE332_FEED_RUNG_BASELINE_CONFIRMATION)}"',
        f'cause = "{_toml_escape(PHASE332_FEED_RUNG_CAUSE)}"',
        f'rows_changed = "{_toml_escape(str(signature["rows_changed"]))}"',
        f'prospective_stamp_ruling = "{_toml_escape(str(signature["prospective_stamp_ruling"]))}"',
        f'disclosed = "{_toml_escape(str(signature["disclosed"]))}"',
        "injury_freshness_limitation = "
        f'"{_toml_escape(str(signature["injury_freshness_limitation"]))}"',
        f'espn_ruling = "{_toml_escape(PHASE332_FEED_RUNG_ESPN_RULING)}"',
        f"predicted_seasons = {_toml_array(list(PHASE332_FEED_RUNG_PREDICTED_SEASONS))}",
        f"allowed_seasons = {_toml_array(list(PHASE332_FEED_RUNG_ALLOWED_SEASONS))}",
        "moved_season_union = "
        f"{_toml_array(sorted({s for seasons in moved_seasons.values() for s in seasons}))}",
        f"widths_before = {_toml_array([before[m]['width'] for m in GOLD_MATRICES])}",
        f"widths_after = {_toml_array([after[m]['width'] for m in GOLD_MATRICES])}",
        f"moved_columns = {_toml_array(moved)}",
        f"build_clock_moves = {_toml_array(verdict['build_clock_moves'])}",
        f"unattributed_columns = {_toml_array(unattributed)}",
        f"attribution_ok = {'true' if verdict['ok'] else 'false'}",
        f"attribution_blocking = {'true' if verdict['blocking'] else 'false'}",
        f'attributor = "{PHASE332_RUNG_ATTRIBUTORS[rung].__name__}"',
    ]
    if not moved:
        lines.append(
            'why_not_run = "the rebuild moved no non-clock column, so the rung is '
            'recorded as declared-but-not-run rather than as a rung that ran"'
        )
    lines.extend(["", f"[rung.{rung}.predicted_by_builder]"])
    for builder, columns in PHASE332_FEED_RUNG_PREDICTED_BY_BUILDER.items():
        lines.append(f"{builder} = {_toml_array(list(columns))}")
    lines.extend(["", f"[rung.{rung}.moved_by_builder]"])
    for builder, columns in phase332_feed_moved_by_builder(moved).items():
        lines.append(f"{builder} = {_toml_array(columns)}")
    lines.extend(["", f"[rung.{rung}.moved_seasons]"])
    for column, seasons in sorted(moved_seasons.items()):
        lines.append(f"{column} = {_toml_array(seasons)}")
    return lines


def _print_attribution(verdict: dict) -> None:
    """Print an attribution verdict in the shape a human reads at checkpoint 2.

    A rung whose criterion does not discriminate per column prints
    ATTRIBUTED (NOT HEALTH-CHECKED) rather than OK. ``RUNBOOK.md`` tells the
    operator to judge by the exit code, and the exit code is unchanged -- this
    stops the PRINTED report from reading like a clean per-column verdict when the
    rung is structurally incapable of giving one (WR-11).
    """
    blanket = any(
        detail.get("discriminating") is False for detail in verdict["matrices"].values()
    )
    if not verdict["ok"]:
        status = "BLOCKED" if verdict["blocking"] else "FINDING"
    elif blanket:
        status = "ATTRIBUTED (NOT HEALTH-CHECKED)"
    else:
        status = "OK"
    print(f"rung {verdict['rung']} ({verdict['cause']}): {status}")
    print(f"  non-clock moves: {verdict.get('non_clock_moves', [])}")
    print(f"  build-clock moves: {verdict.get('build_clock_moves', [])}")
    for matrix, detail in verdict["matrices"].items():
        print(f"  {matrix}:")
        print(f"    attributed:   {detail['attributed']}")
        print(f"    unattributed: {detail['unattributed']}")
        for column, move in sorted(detail.get("move_kinds", {}).items()):
            # "storage moved in 2025 only" rather than "changed, seasons: []".
            seasons = ",".join(move["seasons"]) or "(no season attributed)"
            print(f"    moved:        {column} [{move['kind']}] seasons {seasons}")
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
            "finding, 0 when every moved column is attributed. REFUSES when any "
            "earlier rung document is missing from --fingerprint-dir, so the ladder "
            "cannot be run out of order."
        ),
    )
    parser.add_argument(
        "--rung",
        type=int,
        metavar="N",
        help=(
            "Write this run's fingerprint as rung N's document, deriving --out as "
            "<fingerprint-dir>/<rung-prefix>rungN.json. Mutually exclusive with --out."
        ),
    )
    parser.add_argument(
        "--rung-prefix",
        default="",
        metavar="PREFIX",
        help=(
            "Name-space every rung document written or required by this run, AND "
            "select the cause table. Phase 31 uses "
            f"'{PHASE31_RUNG_PREFIX}' so its ladder cannot overwrite the Phase-30 "
            f"documents {', '.join(PHASE30_RUNG_DOCUMENTS)}, which record the phase "
            "that produced the standing gold and cannot be regenerated; Phase 33.1 "
            f"uses '{PHASE331_RUNG_PREFIX}' for the same reason and because its rung "
            "is judged against its own declared signature rather than Phase 30's "
            f"(known prefixes: {sorted(RUNG_CAUSES_BY_PREFIX)})."
        ),
    )
    parser.add_argument(
        "--fingerprint-dir",
        type=Path,
        default=FINGERPRINT_DIR,
        metavar="DIR",
        help=f"Directory holding the rung ladder (default: {FINGERPRINT_DIR})",
    )
    return parser


def _is_ladder_run(args) -> bool:
    """True when this ``--attribute-rung`` invocation is a LADDER run.

    The ordering refusal must fire on a ladder and stay silent on an ad-hoc
    comparison, and the difference is whether the run CLAIMS to be a rung of one:

    * ``--rung-prefix`` is a declaration. A run that name-spaces its documents is
      running a named ladder, and Plan 31-11 always passes ``p31_``.
    * Otherwise, the BEFORE document must literally BE rung N-1's document under
      ``--fingerprint-dir``. Phase 30's ``--compare rung1.json rung2.json
      --attribute-rung 2`` is a ladder by that test; ``RUNBOOK.md``'s documented
      ``--compare before.json after.json --attribute-rung 2`` is not, and demanding a
      rung-0 baseline of it would refuse a command that never claimed to be a rung --
      a refusal that fires on the wrong thing is the kind operators learn to override.
    """
    if args.rung_prefix:
        return True
    expected = rung_document_path(
        args.fingerprint_dir, args.attribute_rung - 1, args.rung_prefix
    )
    try:
        return Path(args.compare[0]).resolve() == expected.resolve()
    except OSError:
        return False


def main() -> None:
    """CLI entry point for gold fingerprinting."""
    parser = build_parser()
    args = parser.parse_args()

    # --rung DERIVES --out; supplying both would leave which one wins unstated, and a
    # rung document that landed somewhere other than the ladder is a rung document the
    # next rung's predecessor check will not find.
    if args.rung is not None:
        if args.out is not None:
            parser.error(
                "--rung derives --out from --fingerprint-dir and --rung-prefix; pass "
                "one or the other, not both."
            )
        args.out = rung_document_path(args.fingerprint_dir, args.rung, args.rung_prefix)

    # WR-07: this module's docstring says it "is strictly read-only with respect to
    # data/ -- the JSON output must be written somewhere else", and nothing enforced it:
    # --out accepted any path, and main() mkdir'd the parent and wrote. The house guard
    # runs FIRST, before any gold is read or hashed, for the reason
    # backtest.group_gate._reject_data_path states -- a refusal that arrives after the
    # work is a refusal nobody can afford to trust.
    if args.out is not None:
        args.out = reject_data_path(
            args.out,
            what="the fingerprint document",
            suggestion="outputs/fingerprints/",
        )

    if args.attribute_rung is not None:
        if not args.compare:
            print(
                "ERROR: --attribute-rung requires --compare BEFORE AFTER",
                file=sys.stderr,
            )
            sys.exit(2)
        before = json.loads(args.compare[0].read_text(encoding="utf-8"))
        after = json.loads(args.compare[1].read_text(encoding="utf-8"))
        try:
            verdict = attribute_rung(
                compare_fingerprints(before, after),
                args.attribute_rung,
                before=before,
                after=after,
                frame_loader=_gold_frame_loader(),
                ladder_directory=(
                    args.fingerprint_dir if _is_ladder_run(args) else None
                ),
                rung_prefix=args.rung_prefix,
            )
        except MissingPredecessorFingerprintError as error:
            # A named refusal, printed as a sentence rather than a traceback: the
            # operator ran the ladder out of order, which is a different mistake from
            # mistyping a path and has a different fix.
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)
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
