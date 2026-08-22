"""Column-role constants shared by feature building and model training.

One module owns the answer to "what ROLE does this gold column play", so the
feature builder and the walk-forward splitter cannot drift apart about it.

``DISPLAY_ONLY_COLUMNS`` names the six un-normalized ``raw_*`` weather
passthroughs. ``scripts/build_features.py`` writes them purely so the API cache
can surface a human-meaningful value -- ``api/cache.py`` reads
``raw_weather_severity`` and ``raw_wind_mph`` for display -- and each duplicates
a normalized twin that IS a model feature. They therefore carry no independent
signal: measured against Plan 30-03's frozen pre-Phase-30 fixture
``tests/fixtures/gold/features_ats_pre_phase30.parquet``, all six are constant
(``nunique == 1``) BOTH before and after the Phase-30 rebuild.

Two consumers, one constant:

* ``scripts.build_features`` excludes them from ``expanding_normalize`` so they
  are never z-scored -- they exist to be READ, not modelled.
* ``models.temporal.WalkForwardSplitter._feature_cols`` excludes them from the
  MODEL feature set, so a display column can never become a model input.

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
"""Un-normalized display passthroughs: written to gold, never a model feature."""

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
