"""A NULL weather cell survives the imputer AND the normalizer, per builder.

WHAT IS BEING PINNED
--------------------
SPEC R5, second half: "No weather column in any gold matrix contains an imputed
numeric stand-in." Plan 33.1-04 Task 1 stopped `features/weather.py` producing
one. TWO mechanisms downstream would put it straight back, and neither is named
in the SPEC, the CONTEXT or the research:

* `FeatureMatrixBuilder._impute_game_level_features` fills a NaN with a
  prior-seasons MEDIAN. SPEC prohibition 1 names that by name -- "a seasonal
  average or a venue mean is the same defect wearing a better label".
* `features.normalization.expanding_normalize` ends with `normalized.fillna(0.0)`.
  That fallback exists for a position whose STATISTIC was unavailable (an early
  week with fewer than `min_periods` points and no prior-season bootstrap).
  Applied to a position whose VALUE was absent it fabricates a neutral reading
  for a measurement that does not exist. Two causes were collapsed into one
  fill; this module pins them apart.

RULING K1 -- THE SET IS PER BUILDER (Codex 33.1-04 MEDIUM)
-----------------------------------------------------------
The two weather builders do not emit the same columns: the full builder
contributes 47, the compressed one 4. ONE broad constant would become an
assertion about whichever builder a test happened to exercise, while the other
silently median-filled. So `missing_preserving_columns` is a MAPPING keyed by
builder identity, derived from the frame that was actually merged.

The proof below is a CONSUMPTION assertion, not a declaration assertion: the
argument is captured at the REAL `expanding_normalize` call site and compared to
that builder's entry. A test that only read the constant would pass while one
call site received the other builder's set -- which is the exact defect Ruling K1
exists to prevent.

FOUR CONTROLS
-------------
1. NON-VACUITY: the recorder sees at least one real call, asserted by its own
   call count, and both call sites (single-season and batch) are exercised.
2. THE ASSERTIONS: the preserved cell stays NaN through both mechanisms.
3. A PLANTED VIOLATION: clearing the entry makes the build RAISE, per builder,
   so the seam is proven fail-closed rather than silently inert.
4. NO FALSE POSITIVE: an UNPRESERVED control column is still median-imputed and
   still falls back to the neutral 0.0 z-score, and the new parameter defaults
   to empty so every existing caller is byte-preserved.

Nothing here reads or writes a data store. Every frame is constructed in memory.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.build_features as build_features_module
from features.normalization import expanding_normalize
from features.weather import WEATHER_FEATURE_COLUMNS_BY_BUILDER
from scripts.build_features import BUILDER_KEYS, FeatureMatrixBuilder

WEATHER_COLUMN = "weather_severity_score"
# Deliberately NOT prefixed home_/away_: that prefix routes a column to
# `_impute_team_features`, which is a different mechanism with a different
# fixture shape. The control has to travel the SAME path as the preserved
# column for its survival to mean anything.
CONTROL_COLUMN = "control_metric"


def _normalizable_frame() -> pd.DataFrame:
    """One season, eight weeks, one preserved column and one control column.

    Both columns carry a NaN at the SAME position, so any difference in what
    comes back out is a difference the preserving set made and nothing else.
    """
    return pd.DataFrame(
        {
            "season": [2020] * 8,
            "week": list(range(1, 9)),
            WEATHER_COLUMN: [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
            CONTROL_COLUMN: [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
        }
    )


class TestTheNormalizerExemptionIsOptInAndNarrow:
    """Controls 2 and 4, at the normalizer."""

    def test_a_preserved_nan_survives_and_the_control_does_not(self):
        frame = _normalizable_frame()
        result = expanding_normalize(
            frame.copy(),
            feature_cols=[WEATHER_COLUMN, CONTROL_COLUMN],
            preserve_missing_cols=[WEATHER_COLUMN],
        )
        assert np.isnan(result[WEATHER_COLUMN].iloc[3])
        assert np.isfinite(result[CONTROL_COLUMN].iloc[3]), (
            "a column OUTSIDE the declared set must keep today's behaviour "
            "exactly, or the exemption is not proven narrow"
        )

    def test_without_the_parameter_the_same_position_is_filled(self):
        """CONTROL 4: the new behaviour is OPT-IN, not a silent global change."""
        frame = _normalizable_frame()
        result = expanding_normalize(
            frame.copy(), feature_cols=[WEATHER_COLUMN, CONTROL_COLUMN]
        )
        assert np.isfinite(result[WEATHER_COLUMN].iloc[3])

    def test_a_present_value_is_transformed_identically(self):
        """Preservation applies to ABSENCE, never to the transform."""
        frame = _normalizable_frame()
        preserved = expanding_normalize(
            frame.copy(),
            feature_cols=[WEATHER_COLUMN],
            preserve_missing_cols=[WEATHER_COLUMN],
        )
        plain = expanding_normalize(frame.copy(), feature_cols=[WEATHER_COLUMN])
        for position in (0, 1, 2, 4, 5, 6, 7):
            assert (
                abs(
                    preserved[WEATHER_COLUMN].iloc[position]
                    - plain[WEATHER_COLUMN].iloc[position]
                )
                < 1e-12
            )

    def test_the_default_is_empty(self):
        import inspect

        signature = inspect.signature(expanding_normalize)
        default = signature.parameters["preserve_missing_cols"].default
        assert tuple(default) == ()


_ROWS = 8
_VALUES = [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]

# The one preserved column each builder's assertions read. Chosen as the first
# NUMERIC entry of that builder's declared set rather than hard-coded, so the
# test cannot quietly assert about a column the builder stopped emitting.
_PRESERVED_UNDER_TEST = {
    key: next(c for c in columns if c != "weather_condition")
    for key, columns in WEATHER_FEATURE_COLUMNS_BY_BUILDER.items()
}


def _weather_only_frame(builder_key: str) -> pd.DataFrame:
    """Exactly what `combine_features` merges: game_id plus that builder's set."""
    frame = pd.DataFrame(
        {"game_id": [f"2020_W{week:02d}_A@B" for week in range(1, _ROWS + 1)]}
    )
    for column in WEATHER_FEATURE_COLUMNS_BY_BUILDER[builder_key]:
        if column == "weather_condition":
            frame[column] = ["Clear"] * _ROWS
            continue
        frame[column] = _VALUES
    return frame


def _merged_frame(builder_key: str) -> pd.DataFrame:
    """A combined-matrix-shaped frame carrying *builder_key*'s weather columns."""
    frame = pd.DataFrame(
        {
            "game_id": [f"2020_W{week:02d}_A@B" for week in range(1, _ROWS + 1)],
            "season": [2020] * _ROWS,
            "week": list(range(1, _ROWS + 1)),
            CONTROL_COLUMN: _VALUES,
        }
    )
    weather = _weather_only_frame(builder_key).drop(columns=["game_id"])
    return pd.concat([frame, weather], axis=1)


def _builder_with_merged_weather(builder_key: str) -> FeatureMatrixBuilder:
    builder = FeatureMatrixBuilder()
    resolved = builder.record_missing_preserving_columns(
        _weather_only_frame(builder_key)
    )
    assert resolved == builder_key, (
        "the builder key is RESOLVED from the merged frame's own columns; if it "
        "resolved to the other builder the rest of this test would be asserting "
        "about the wrong set"
    )
    return builder


class TestTheImputerExemptionIsOptInAndNarrow:
    """Controls 2 and 4, at the game-level imputer."""

    @pytest.mark.parametrize("builder_key", BUILDER_KEYS)
    def test_a_preserved_nan_survives_the_game_level_median(self, builder_key):
        builder = _builder_with_merged_weather(builder_key)
        frame = _merged_frame(builder_key)
        processed = builder.handle_missing_data_and_outliers(frame)

        preserved = _PRESERVED_UNDER_TEST[builder_key]
        assert np.isnan(processed[preserved].iloc[3]), (
            f"{preserved} was median-imputed. A seasonal median for an absent "
            "observation is SPEC prohibition 1 -- the same defect wearing a "
            "better label."
        )
        assert np.isfinite(processed[CONTROL_COLUMN].iloc[3]), (
            "the control column outside the declared set must still be imputed"
        )


class TestTheSeamFailsClosedPerBuilder:
    """CONTROL 3, asserted separately for each builder key."""

    @pytest.mark.parametrize("builder_key", BUILDER_KEYS)
    def test_an_emptied_entry_raises_naming_the_attribute_and_the_builder(
        self, builder_key
    ):
        builder = _builder_with_merged_weather(builder_key)
        builder.missing_preserving_columns[builder_key] = ()

        with pytest.raises(ValueError) as excinfo:
            builder.handle_missing_data_and_outliers(_merged_frame(builder_key))

        message = str(excinfo.value)
        assert "missing_preserving_columns" in message
        assert builder_key in message, (
            "an exemption that is live for one builder and inert for the other "
            "is worse than none, because half the evidence says it works -- so "
            "the refusal has to name WHICH builder it is refusing for"
        )


class TestEachCallSiteReceivesItsOwnBuildersEntry:
    """CONTROL 1 and the Ruling K1 proof: captured from the REAL calls."""

    @pytest.mark.parametrize("builder_key", BUILDER_KEYS)
    @pytest.mark.parametrize("target_season", [2020, None])
    def test_the_recorded_argument_equals_that_builders_entry(
        self, monkeypatch, builder_key, target_season
    ):
        recorded: list[tuple[str, ...]] = []
        original = build_features_module.expanding_normalize

        def recorder(*args, **kwargs):
            recorded.append(tuple(kwargs.get("preserve_missing_cols", ())))
            return original(*args, **kwargs)

        monkeypatch.setattr(build_features_module, "expanding_normalize", recorder)

        builder = _builder_with_merged_weather(builder_key)
        builder.normalize_combined_features(
            _merged_frame(builder_key), target_season=target_season
        )

        assert recorded, (
            "the recorder saw ZERO calls, so this test would pass while "
            "asserting nothing about either call site"
        )
        expected = builder.missing_preserving_columns[builder_key]
        for observed in recorded:
            assert observed == tuple(expected)


class TestTheSeamIsCommitted:
    def test_the_state_slot_records_both_mechanisms_and_both_builders(self):
        from tests.phase33_state import MISSING_PRESERVING_SEAM as seam

        assert len(seam["mechanisms"]) >= 2
        assert seam["default_is_empty"] is True
        assert seam["fail_closed"] is True
        assert sorted(seam["preserved_set_size_by_builder"]) == [
            "compressed",
            "full",
        ]
        for size in seam["preserved_set_size_by_builder"].values():
            assert size > 0
