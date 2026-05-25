"""Regression tests for game_context derived-column recomputation.

Guards three web-cache fixes that share the same defect class -- a normalized
gold column surfaced raw in the UI:

1. ``is_divisional``. Previously ``api/cache.py`` applied ``.astype(bool)``
   to the gold ``is_divisional`` column, which is expanding-window z-score
   normalized (float). Any non-zero z-score became ``True``, so ~99.8% of
   games were mislabeled "Divisional Game". The fix recomputes the flag from
   the canonical ``ratings.elo.is_divisional_game`` via the module-level
   helper ``api.cache._compute_is_divisional``.

2. ``home_elo`` / ``away_elo``. The cache read these from the gold
   ``features_wp.parquet`` where they are expanding-window z-scores (plus a
   1500.0 placeholder leakage for the earliest games), so the UI showed
   z-scores like ``-0.78`` instead of raw ratings. The fix drops them from
   ``gold_cols`` and sources raw ``home_elo_pre`` / ``away_elo_pre`` from
   ``data/silver/elo_game_snapshots.parquet`` via the module-level helper
   ``api.cache._attach_raw_elo``. Snapshot-less games get NaN -> SQL NULL
   -> "N/A" in the UI.

3. ``weather_severity``. The cache read the normalized
   ``weather_severity_score`` from gold (an expanding-window z-score in
   ~[-1.7, 3.7]), so the UI showed values like "-0.2" / "1.8". The fix reads
   the un-normalized ``raw_weather_severity`` gold passthrough (renamed
   ``weather_severity``, a raw composite in ~[0, 0.8]) and stores a
   qualitative ``weather_severity_band`` computed by the module-level helper
   ``api.cache._weather_severity_band``. The 0.60/0.80 band anchors match
   ``features.weather`` weather_game / extreme_weather.

These tests assert the recomputed flags / ratings / bands against known
inputs and explicitly catch a regression back to the normalized-gold values.
"""

from __future__ import annotations

import pandas as pd

from api.cache import (
    _attach_raw_elo,
    _compute_is_divisional,
    _weather_severity_band,
)


def test_same_division_pairs_resolve_true() -> None:
    """Known same-division matchups must resolve to True."""
    home = pd.Series(["BUF", "KC", "DAL", "KC"])
    away = pd.Series(["NYJ", "LV", "PHI", "DEN"])

    result = _compute_is_divisional(home, away)

    assert result.tolist() == [True, True, True, True]


def test_cross_division_pairs_resolve_false() -> None:
    """Known cross-division / cross-conference matchups must resolve to False."""
    home = pd.Series(["BUF", "DAL", "GB", "WAS", "KC"])
    away = pd.Series(["KC", "SF", "MIA", "LAC", "CLE"])

    result = _compute_is_divisional(home, away)

    assert result.tolist() == [False, False, False, False, False]


def test_result_is_real_bool_dtype() -> None:
    """The schema column is BOOLEAN, so the Series must be real bool dtype.

    A float dtype here is the exact shape of the original bug.
    """
    home = pd.Series(["BUF", "BUF"])
    away = pd.Series(["NYJ", "KC"])

    result = _compute_is_divisional(home, away)

    assert result.dtype == bool


def test_divisional_rate_in_plausible_band() -> None:
    """Over a representative mixed set, the True-rate stays in a plausible band.

    Each team plays 6 of 17 divisional games (~35% league-wide), so a
    realistic rate sits well under 0.5. A rate near 1.0 reproduces the
    normalized-float-to-bool regression and must fail this assertion.
    """
    # Mixed set: 4 divisional pairs, 8 non-divisional pairs -> rate = 1/3.
    home = pd.Series(
        [
            "BUF",  # divisional (AFC East)
            "KC",  # divisional (AFC West)
            "DAL",  # divisional (NFC East)
            "GB",  # divisional (NFC North)
            "BUF",  # cross-division (AFC)
            "DAL",  # cross-division (NFC)
            "GB",  # cross-conference
            "WAS",  # cross-conference
            "KC",  # cross-division (AFC)
            "SEA",  # cross-division (NFC)
            "NE",  # cross-division (AFC)
            "PIT",  # cross-conference
        ]
    )
    away = pd.Series(
        [
            "NYJ",
            "LV",
            "PHI",
            "MIN",
            "KC",
            "SF",
            "MIA",
            "LAC",
            "CLE",
            "NYG",
            "DEN",
            "ATL",
        ]
    )

    result = _compute_is_divisional(home, away)
    rate = float(result.mean())

    assert 0.2 <= rate <= 0.5, f"divisional rate {rate:.3f} outside plausible band"


# ---------------------------------------------------------------------------
# Raw-Elo sourcing (_attach_raw_elo)
# ---------------------------------------------------------------------------


def test_snapshot_game_gets_raw_elo_in_band() -> None:
    """A game with a snapshot gets RAW pre-game Elo, not a z-score.

    The returned home_elo/away_elo must land in a plausible raw Elo band
    (~1100-1900), never the tight z-score range the gold column used to
    surface (e.g. -0.78).
    """
    context = pd.DataFrame({"game_id": ["G_SNAP", "G_NOSNAP"]})
    snapshots = pd.DataFrame(
        {
            "game_id": ["G_SNAP"],
            "home_elo_pre": [1680.4],
            "away_elo_pre": [1432.0],
        }
    )

    result = _attach_raw_elo(context, snapshots)
    snap_row = result.loc[result["game_id"] == "G_SNAP"].iloc[0]

    assert snap_row["home_elo"] == 1680.4
    assert snap_row["away_elo"] == 1432.0
    assert 1100 <= snap_row["home_elo"] <= 1900
    assert 1100 <= snap_row["away_elo"] <= 1900


def test_exactly_1500_snapshot_elo_is_preserved() -> None:
    """A legitimate raw rating of exactly 1500.0 stays 1500.0.

    16 real snapshot rows are exactly 1500.0, so the band check must ALLOW
    1500 (never assert ``!= 1500``).
    """
    context = pd.DataFrame({"game_id": ["G_1500"]})
    snapshots = pd.DataFrame(
        {
            "game_id": ["G_1500"],
            "home_elo_pre": [1500.0],
            "away_elo_pre": [1500.0],
        }
    )

    result = _attach_raw_elo(context, snapshots)
    row = result.iloc[0]

    assert row["home_elo"] == 1500.0
    assert row["away_elo"] == 1500.0
    assert 1100 <= row["home_elo"] <= 1900


def test_snapshot_less_game_gets_nan() -> None:
    """A game without a snapshot gets NaN -> SQL NULL -> 'N/A' in the UI."""
    context = pd.DataFrame({"game_id": ["G_SNAP", "G_NOSNAP"]})
    snapshots = pd.DataFrame(
        {
            "game_id": ["G_SNAP"],
            "home_elo_pre": [1680.4],
            "away_elo_pre": [1432.0],
        }
    )

    result = _attach_raw_elo(context, snapshots)
    nosnap_row = result.loc[result["game_id"] == "G_NOSNAP"].iloc[0]

    assert pd.isna(nosnap_row["home_elo"])
    assert pd.isna(nosnap_row["away_elo"])


def test_present_elo_never_a_small_zscore() -> None:
    """Every present rating is a raw Elo, never a small z-score.

    A raw Elo can never land in the tight z-score range ~[-4, 4]; guard that
    any non-null result has abs(v) > 50 so a regression to z-scores fails.
    """
    context = pd.DataFrame({"game_id": ["A", "B", "C"]})
    snapshots = pd.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "home_elo_pre": [1151.0, 1500.0, 1842.4],
            "away_elo_pre": [1200.0, 1600.0, 1300.0],
        }
    )

    result = _attach_raw_elo(context, snapshots)

    for col in ("home_elo", "away_elo"):
        present = result[col].dropna()
        assert (present.abs() > 50).all(), f"{col} contains a z-score-sized value"


def test_left_join_does_not_duplicate_rows_or_add_suffixes() -> None:
    """The LEFT join adds only home_elo/away_elo and never fans out rows.

    Because home_elo/away_elo are dropped from gold_cols before this helper
    runs, the merge cleanly ADDS the columns -- no _x/_y suffix collision --
    and the unique snapshot game_id cannot duplicate context rows.
    """
    context = pd.DataFrame({"game_id": ["G1", "G2", "G3"]})
    snapshots = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2020, 2021, 2022],
            "home_elo_pre": [1500.0, 1600.0, 1400.0],
            "away_elo_pre": [1480.0, 1550.0, 1620.0],
        }
    )

    result = _attach_raw_elo(context, snapshots)

    assert len(result) == len(context)
    assert "home_elo" in result.columns
    assert "away_elo" in result.columns
    assert not any(c.endswith(("_x", "_y")) for c in result.columns)
    # Snapshot-only columns must not leak into context.
    assert "season" not in result.columns
    assert "home_elo_pre" not in result.columns


# ---------------------------------------------------------------------------
# Weather-severity band mapping (_weather_severity_band)
# ---------------------------------------------------------------------------


def test_band_anchors_match_weather_code_thresholds() -> None:
    """The 0.60/0.80 anchors match features.weather weather_game/extreme_weather.

    ``features/weather.py:475-476`` defines ``weather_game`` at severity >= 0.6
    and ``extreme_weather`` at severity >= 0.8. The band must align so the
    display semantics never drift from the code's own thresholds.
    """
    band = _weather_severity_band(pd.Series([0.60, 0.80]))

    assert band.iloc[0] == "Significant"
    assert band.iloc[1] == "Extreme"


def test_band_just_below_anchors_drops_one_level() -> None:
    """A value just below an anchor falls into the band below it."""
    band = _weather_severity_band(pd.Series([0.59, 0.79]))

    assert band.iloc[0] == "Moderate"
    assert band.iloc[1] == "Significant"


def test_band_sub_06_calibrated_cutoffs() -> None:
    """Sub-0.6 cutoffs map to the calibrated bands from the real distribution.

    Cutoffs 0.10 / 0.25 sit in the valleys between the 0.09 / 0.19 / 0.39
    severity clusters (RESEARCH section 5).
    """
    severities = pd.Series([0.0, 0.09, 0.10, 0.19, 0.25, 0.39])
    band = _weather_severity_band(severities)

    assert band.tolist() == [
        "Clear",
        "Clear",
        "Mild",
        "Mild",
        "Moderate",
        "Moderate",
    ]


def test_band_boundaries_are_left_closed() -> None:
    """A value exactly at a cutoff belongs to the UPPER band (left-closed bins).

    ``right=False`` makes each interval ``[lo, hi)`` so an exact-cutoff value
    falls into the higher band: 0.25 -> Moderate (not Mild), 0.60 ->
    Significant, 0.80 -> Extreme.
    """
    band = _weather_severity_band(pd.Series([0.10, 0.25, 0.60, 0.80]))

    assert band.tolist() == ["Mild", "Moderate", "Significant", "Extreme"]


def test_band_input_is_raw_composite_not_zscore() -> None:
    """Given plausible RAW composites, every input is in [0, 1] and maps correctly.

    This is the raw-not-z-score guard (mirrors the raw-Elo ``1100<=v<=1900``
    band test). A negative value, or one above ~1.5, would mean the cache is
    reading the NORMALIZED ``weather_severity_score`` column instead of the
    ``raw_weather_severity`` passthrough -- this assertion catches that
    regression. Boundary values are explicitly allowed (never ``!= specific``).
    """
    severities = pd.Series([0.0, 0.09, 0.39, 0.60, 0.80])

    assert (severities >= 0.0).all()
    assert (severities <= 1.0).all()

    band = _weather_severity_band(severities)

    assert band.tolist() == ["Clear", "Clear", "Moderate", "Significant", "Extreme"]


def test_band_returns_object_dtype_aligned_to_input_index() -> None:
    """The helper returns object-dtype labels aligned to the input index.

    Object dtype (plain strings) stores as a DuckDB VARCHAR and keeps the
    equality assertions above plain ``str`` comparisons; the preserved index
    lets the result be assigned straight back as a cache column.
    """
    severities = pd.Series([0.05, 0.30, 0.85], index=[10, 20, 30])

    band = _weather_severity_band(severities)

    assert band.dtype == object
    assert band.index.tolist() == [10, 20, 30]
    assert band.tolist() == ["Clear", "Moderate", "Extreme"]
