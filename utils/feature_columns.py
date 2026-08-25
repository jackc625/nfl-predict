"""Column-role constants shared by feature building and model training.

One module owns the answer to "what ROLE does this gold column play", so the
feature builder and the walk-forward splitter cannot drift apart about it.

``DISPLAY_ONLY_COLUMNS`` names the six un-normalized ``raw_*`` weather
passthroughs. ``scripts/build_features.py`` writes them purely so the API cache
can surface a human-meaningful value -- ``api/cache.py`` reads
``raw_weather_severity`` and ``raw_wind_mph`` for display -- and each duplicates
a normalized twin that IS a model feature.

**Why they are excluded: the SCALE, not constancy.** They are deliberately
withheld from ``expanding_normalize``, so they reach a model un-normalized and
with un-neutralised nulls, alongside a z-scored twin carrying the same
measurement. That is the whole argument, and it does not depend on their
distribution.

An earlier version of this note argued from constancy instead, and the phase's
own rebuild falsified it. On Plan 30-03's frozen pre-Phase-30 fixture
``tests/fixtures/gold/features_ats_pre_phase30.parquet`` all six ARE constant
(``nunique == 1``). On the accepted rung-4 gold they are NOT, in all three
matrices over 6,499 rows:

===========================  ==============  ==========================
column                       fixture         ``data/gold/features_*``
===========================  ==============  ==========================
``raw_temp_f``               nunique 1       nunique 15
``raw_wind_mph``             nunique 1       nunique 10
``raw_precip_prob``          nunique 1       nunique 4
``raw_precip_mm``            nunique 1       nunique 2
``raw_humidity_pct``         nunique 1       nunique 10, 6,214 nulls
``raw_weather_severity``     nunique 1       nunique 4
===========================  ==============  ==========================

They are excluded from ``expanding_normalize`` but NOT from
``handle_missing_data_and_outliers``, so WR-06's per-season prior-median
imputation and per-season winsorization moved them. Their having become varying
is not a reason to re-admit them -- it is precisely why the un-normalized path
now matters, and it is asserted in ``tests/unit/test_temporal_display_columns.py``
against LIVE gold so the claim cannot silently go stale again.

Two consumers, one constant:

* ``scripts.build_features`` excludes them from ``expanding_normalize`` so they
  are never z-scored -- they exist to be READ, not modelled.
* ``models.temporal.WalkForwardSplitter._feature_cols`` excludes them from the
  MODEL feature set.

**Scope of that second exclusion: FORWARD-LOOKING only.** It governs what a
FUTURE fit can select. It says nothing about an artifact already on disk, and
one deployed artifact violates the invariant it would otherwise state. The
RETAINED O/U model ``ou_20260326_163930`` -- the v1.0 pre-Elo model the Phase-30
gate refused to replace -- lists three of the six among its 25 features:

    raw_precip_mm, raw_precip_prob, raw_wind_mph

``scripts/generate_current_week_predictions.py`` slices gold by that saved
feature list, so those three ARE model inputs in production today, at raw
(un-normalized) scale, on values the Phase-30 rebuild moved from constant to
varying. The artifact predates Plan 30-15 and the gate refusal that retained it
is sound; re-fitting it to close this is a GATE decision, not a cleanup. Until
some future re-fit passes the gate, the honest statement is the one above: the
exclusion is forward-looking, and there is one deployed artifact it does not
reach. ``tests/unit/test_temporal_display_columns.py`` pins that residue
explicitly, so a NEW offender cannot be added quietly and the known one cannot
be forgotten.

The second exclusion did not exist until Plan 30-15 (owner ruling D30-OWNER-04).
The consequence was concrete rather than theoretical. ``data/silver/weather.parquet``
holds fourteen rows, all season 2025, so the pre-rebuild ``raw_humidity_pct``
constant of 54.0 was a future-to-past broadcast wearing the costume of a
measurement. When WR-06 (Plan 30-06) correctly refused to keep fabricating it,
the column became honestly NaN for 6,214 of 6,263 gold rows -- and because it was
excluded from normalization but NOT from the feature set, its NaNs were never
mapped to the neutral 0.0 z-score either. It reached
``BaseTrainer.select_features``, where the WP ``LogisticRegression`` raised
``ValueError: Input X contains NaN`` and errored every test in
``tests/integration/test_group_gate_determinism.py`` -- the Stage-1 orchestrator.

The repair is EXCLUSION, never fabrication. Imputing the column would
re-introduce the exact 2025-broadcast-backwards leak WR-06 removed; dropping it
would change gold width and corrupt the Phase-30 rung ladder's attribution.

Add a display column HERE and both consumers pick it up. Never restate these
names at a call site -- ``tests/unit/test_temporal_display_columns.py`` fails if
you do.
"""

from __future__ import annotations

DISPLAY_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "raw_wind_mph",
        "raw_temp_f",
        "raw_precip_prob",
        "raw_precip_mm",
        "raw_humidity_pct",
        "raw_weather_severity",
    }
)
"""Un-normalized display passthroughs: written to gold, never selected by a fit.

"Never selected by a FIT" is the exact claim. It is not "never a model input":
the retained ``ou_20260326_163930`` artifact, fitted before Plan 30-15, consumes
``raw_precip_mm`` / ``raw_precip_prob`` / ``raw_wind_mph`` in production. See the
module docstring.
"""

NORMALIZATION_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "feature_timestamp",
)
"""Identifier / outcome columns that expanding-window normalization must skip.

A DIFFERENT category from the display columns: these are keys and raw outcomes,
and ``models.temporal`` already handles them separately via its ``id_cols``.
"""


def display_only_columns() -> frozenset[str]:
    """Return the display-only column names.

    Reads the module global at call time so both consumers observe the same
    value -- including when a test patches it to prove a newly-added display
    column reaches both without a second edit.
    """
    return DISPLAY_ONLY_COLUMNS


def normalization_exclude_columns() -> list[str]:
    """Columns ``expanding_normalize`` must never z-score.

    The identifier half is structural (keys and outcomes); the display half is
    derived from :data:`DISPLAY_ONLY_COLUMNS`, so the six names are stated once
    in this module and nowhere else.
    """
    return [*NORMALIZATION_IDENTIFIER_COLUMNS, *sorted(DISPLAY_ONLY_COLUMNS)]
