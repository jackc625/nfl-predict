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


def rung_document_path(directory: Path | str, rung: int, prefix: str = "") -> Path:
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
    for predecessor in range(rung):
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
    for predecessor in range(rung):
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
    if rung not in causes:
        msg = (
            f"Unknown rung {rung!r} under rung prefix {prefix!r}. Must be one of "
            f"{sorted(causes)}."
        )
        raise ValueError(msg)

    if prefix == PHASE331_RUNG_PREFIX:
        # The PRE-DECLARED change set, returned as a copy so a caller cannot edit
        # the prediction it is about to be judged against (T-33.1-43). The
        # follow-up rung has its OWN signature; rung 1's is byte-untouched by it,
        # which is the difference between a follow-up rung and a widened
        # declaration (Ruling N2).
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
    cause = _rung_causes(rung_prefix)[rung]
    # THE PHASE-33.1 RUNG GETS NO UPSTREAM-DRIFT ESCAPE, and the suppression is
    # scoped to the PREFIX rather than expressed by editing `_UPSTREAM_ESCAPE_RUNGS`
    # -- that tuple is Phase 30's record and rung 1 legitimately carries the escape
    # there. This rung rebuilds from SILVER, not from nflreadpy, so offering an
    # "upstream revision" candidate cause would send a reader hunting for something
    # that cannot be the explanation.
    upstream = (
        " " + _UPSTREAM_DRIFT_NOTE.format(cause=cause)
        if rung in _UPSTREAM_ESCAPE_RUNGS and rung_prefix != PHASE331_RUNG_PREFIX
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
    if rung_prefix == PHASE331_RUNG_PREFIX:
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
