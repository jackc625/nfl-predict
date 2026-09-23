"""Display-only gold columns are never model features (Plan 30-15, D30-OWNER-04).

``scripts/build_features.py`` writes six un-normalized ``raw_*`` weather
passthroughs purely so the API cache can surface a human-meaningful value
(``api/cache.py:1250-1259`` reads ``raw_weather_severity`` and ``raw_wind_mph``).
Each duplicates a normalized twin that IS a model feature, and each is
deliberately withheld from ``expanding_normalize`` -- so a display column reaches
a model un-normalized and with un-neutralised nulls, beside a z-scored twin
carrying the same measurement. That SCALE argument is the reason for the
exclusion.

It is not a constancy argument, and an earlier version of this module said it
was. All six are constant (``nunique == 1``) on Plan 30-03's frozen pre-Phase-30
fixture ``tests/fixtures/gold/features_ats_pre_phase30.parquet`` and NONE of them
is constant on the rung-4 gold that shipped -- they are excluded from
``expanding_normalize`` but not from ``handle_missing_data_and_outliers``, so
WR-06's per-season imputation and winsorization moved them.
``TestRealGold.test_display_columns_are_not_constant_on_live_gold`` measures that
on LIVE gold rather than on the fixture, which is what would have caught the
claim going stale.

They were nevertheless model inputs. ``build_features`` excluded them from
``expanding_normalize`` only; ``models.temporal.WalkForwardSplitter._feature_cols``
excluded ID and target columns and then took every remaining numeric column.
That latent defect became visible the moment WR-06 (Plan 30-06) stopped
fabricating ``raw_humidity_pct`` -- ``data/silver/weather.parquet`` holds
fourteen rows, all season 2025, so the pre-rebuild constant 54.0 was a
future-to-past broadcast -- and the honestly-NaN column reached
``BaseTrainer.select_features``, where the WP ``LogisticRegression`` raised
``ValueError: Input X contains NaN`` and errored all five
``tests/integration/test_group_gate_determinism.py`` tests.

The contract these tests pin:

* The six names live in exactly ONE place, ``utils.feature_columns``.
* ``WalkForwardSplitter`` subtracts them from the model feature set.
* ``build_features`` derives its normalization exclusion from the same constant,
  so a display column added in future reaches BOTH consumers without a second
  edit -- proved by patching the constant and re-reading both, not by comparing
  two hand-written lists.
* Neither consumer restates a display name as a string literal (a copy that can
  drift is exactly what this plan removes).
* The fix is EXCLUSION, never fabrication: ``raw_humidity_pct`` stays NaN in
  gold.
* The exclusion is FORWARD-LOOKING. It governs what a fit can select and has no
  reach over an artifact already on disk, so the residue -- the retained O/U
  model that consumes three of the six -- is pinned by name and count rather
  than wished away.

WHAT PHASE 33.2 MOVED, AND WHY NOTHING HERE WAS RELAXED (Plan 33.2-20)
----------------------------------------------------------------------
Every anchor below was measured on rung-4 PHASE-30 gold. Three independent
Phase-33.2 causes moved them, and the re-anchor keeps them apart because they are
separate facts, each recorded in ``tests/phase33_state`` beside the Phase-30 slots
rather than instead of them.

1. ``raw_precip_mm`` LEFT GOLD. p332_ rung 4 (Plan 33.2-12, SPEC R6) replaced the
   ERA5 reanalysis with the archived day-before FORECAST, and a forecast reports a
   probability, never observed millimetres -- so ``precip_mm`` and its display
   sibling leave every matrix through the ``weather_unsupplied`` registry group.
   ``DISPLAY_ONLY_COLUMNS`` still names SIX and that is correct: it is the
   CONSUMER's exclusion list, and a column absent from gold is excluded trivially
   rather than wrongly. The tests below now distinguish "named for exclusion" from
   "present in gold" instead of conflating them, and assert the difference is
   exactly ``raw_precip_mm``.
2. THE WEATHER VALUES ARE DIFFERENT DATA. A forecast is not a reanalysis, so every
   distinct count and null count moved. The nulls are now the 1,030 fixed-roof
   domes plus the 74 uncovered games -- except ``raw_weather_severity``, a
   COMPOSITE that is a genuine 0.0 for a covered dome, so only the 74 are null.
   That is Ruling J's split, seen on the display siblings.
3. THE CLEAN BUILD ADDED THE 17 PLAYED 2026 GAMES (6,499 -> 6,516).

AND ONE ASSERTION WAS SUPERSEDED BY AN OWNER RULING, not re-anchored.
``test_the_wp_feature_set_is_free_of_nan`` pinned gold to carry no NaN in any WP
feature column. The owner ruled twice on 2026-09-22 that a value nothing honest can
fill stays BLANK -- p332_ step 7b (a within-season gap is filled only from games
ended by the gap's own lock) and step 8d (a cell whose expanding statistic could not
be formed is blank, never the neutral 0.0) -- so 160 of the 175 WP feature columns
now carry NaN, 93,864 cells, deliberately. Those rulings rest on the models taking a
blank NATIVELY, and for WP that is the in-fold imputer plus ``_was_missing``
indicator in ``models/trainers/wp_trainer.py``. The claim the assertion was really
making -- that a WP fit does not raise ``Input X contains NaN`` -- is therefore
asserted at the TRAINER, where the treatment lives, rather than deleted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from tests import phase33_state
from utils.feature_columns import (
    DISPLAY_ONLY_COLUMNS,
    NORMALIZATION_IDENTIFIER_COLUMNS,
    display_only_columns,
    normalization_exclude_columns,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLD_DIR = _REPO_ROOT / "data" / "gold"
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

# The six un-normalized weather passthroughs, restated ONCE here so a silent
# edit to the shared constant has to argue with a test.
_EXPECTED_DISPLAY_COLUMNS = frozenset(
    {
        "raw_wind_mph",
        "raw_temp_f",
        "raw_precip_prob",
        "raw_precip_mm",
        "raw_humidity_pct",
        "raw_weather_severity",
    }
)

# ``raw_humidity_pct`` after the Plan 30-06 rung-2 rebuild: honestly unpopulated for
# every row outside season 2025, the ONLY season ``data/silver/weather.parquet`` covers
# (fourteen observations). Within 2025 the column is filled by WR-06's documented
# self-fit path -- a season with no prior slice to fit from uses its own -- which never
# crosses a season boundary backwards and is therefore not the future-to-past broadcast
# WR-06 removed.
#
# That cross-season claim is the ONLY temporal claim made here. Whether the within-2025
# fit is POINT-IN-TIME -- whether a given 2025 row's imputed value is free of information
# dated after that row's own game -- is NOT established, and is registered as
# D30-DEFER-14 rather than assumed in either direction. It is a documentation question
# and not a gate question only because the column reaches no model: Plan 30-15 excluded
# all six raw_* display siblings from the model feature set under D30-OWNER-04. If the
# weather family is ever backfilled (D30-DEFER-09, an ingestion phase), it becomes a gate
# question and must be settled first.
#
# Pinned as an ANCHOR, not a tolerance: the whole point of Plan 30-15 is
# that the consumer was fixed and the DATA was left alone. A legitimate historical
# weather backfill (D30-DEFER-09, an ingestion phase, explicitly out of Phase-30 scope)
# must update this number deliberately.
_RAW_HUMIDITY_NAN_ROWS = 6214

# Rung 4 (Plan 30-08, the N-01 re-sync) grew gold from 6,263 to 6,499 rows: the DuckDB
# copy of silver ``games`` was 207 rows behind its authoritative parquet, and once it was
# re-synced the rebuild carried season 2025 from weeks 1-4 (49 rows) to weeks 1-22 (285).
# The row count is a COMPANION anchor and it moved for that sanctioned reason.
#
# ``_RAW_HUMIDITY_NAN_ROWS`` did NOT move, and that is the load-bearing fact: all 236 new
# rows are season-2025 games, 2025 is the only season ``data/silver/weather.parquet``
# covers, and 6,499 - 285 == 6,214 exactly. The column is still honestly unpopulated
# everywhere there is no upstream observation, and still imputed nowhere. The test below
# now also asserts that identity directly, so a future legitimate row addition cannot
# make this anchor stale without the claim itself being re-examined.
_GOLD_ROWS = 6499
_GOLD_2025_ROWS = 285

# RE-ANCHORED (Plan 33.2-20). The Phase-30 constants above are kept as the record of
# the gold they measured; the live assertions read the slots below. See the module
# docstring for the three causes.
_LIVE_GOLD_ROWS = phase33_state.P332_20_GOLD_ROWS_AFTER_CLEAN_BUILD
_DISPLAY_COLUMNS_IN_GOLD = frozenset(phase33_state.P332_20_DISPLAY_COLUMNS_IN_GOLD)
_DISPLAY_COLUMNS_NOT_IN_GOLD = frozenset(
    phase33_state.P332_20_DISPLAY_COLUMNS_NOT_IN_GOLD
)

# Measured on the accepted rung-4 gold, identically in all three matrices. Every one
# is > 1, so the six are NOT constant after the Phase-30 rebuild -- which is what
# falsified the old constancy rationale in ``utils/feature_columns.py``. They are
# withheld from ``expanding_normalize`` but not from
# ``handle_missing_data_and_outliers``, so WR-06's per-season prior-median imputation
# and per-season winsorization moved them.
#
# Anchors, not tolerances. A legitimate weather backfill (D30-DEFER-09) moves these
# deliberately and must re-read the module rationale while doing so.
_LIVE_GOLD_DISPLAY_NUNIQUE = {
    "raw_humidity_pct": 10,
    "raw_precip_mm": 2,
    "raw_precip_prob": 4,
    "raw_temp_f": 15,
    "raw_weather_severity": 4,
    "raw_wind_mph": 10,
}

# The FORWARD-LOOKING exclusion's residue in production, recorded rather than implied.
# ``ou_20260326_163930`` is the v1.0 pre-Elo O/U model: the Phase-30 gate refused its
# replacement (paired -0.487007, p=9.24e-12) and RETAINED it, and it was fitted long
# before Plan 30-15 excluded display columns from the feature set. Its saved
# feature_list.json is what ``scripts/generate_current_week_predictions.py`` slices gold
# by, so these three reach the model at raw scale today.
#
# Closing it is a re-fit that must pass the gate, not a cleanup. WP
# (``wp_20260824_113325``) and ATS (``ats_20260605_220128``) are clean.
#
# A HISTORICAL RECORD since Plan 33.2-25's swap: none of those three serves any more.
_PHASE30_DEPLOYED_DISPLAY_RESIDUE = {
    "ou_20260326_163930": ["raw_precip_mm", "raw_precip_prob", "raw_wind_mph"],
}


def _splitter(target_col: str = "target_wp") -> WalkForwardSplitter:
    return WalkForwardSplitter(
        config=TemporalSplitConfig(
            train_seasons=[2018, 2019],
            hp_val_seasons=[2020],
            holdout_seasons=[2021],
        ),
        target_col=target_col,
    )


def _synthetic_frame() -> pd.DataFrame:
    """A frame shaped like gold: id cols, real features, the six display cols."""
    rows = 8
    frame = pd.DataFrame(
        {
            "game_id": [f"2021_{i:04d}" for i in range(rows)],
            "season": 2021,
            "week": np.arange(rows) + 1,
            "home_team": "KC",
            "away_team": "DET",
            "target_wp": (np.arange(rows) % 2).astype(int),
            "elo_diff": np.linspace(-100.0, 100.0, rows),
            "rest_advantage": np.linspace(-3.0, 3.0, rows),
        }
    )
    for name in sorted(_EXPECTED_DISPLAY_COLUMNS):
        frame[name] = 1.0
    # The column that actually breaks LogisticRegression on rebuilt gold.
    frame["raw_humidity_pct"] = np.nan
    return frame


class TestSharedDisplayConstant:
    """One constant names the display columns; nothing restates them."""

    def test_constant_names_exactly_the_six_raw_weather_passthroughs(self) -> None:
        assert DISPLAY_ONLY_COLUMNS == _EXPECTED_DISPLAY_COLUMNS

    def test_accessor_returns_the_constant_itself_not_a_copy(self) -> None:
        assert display_only_columns() is DISPLAY_ONLY_COLUMNS

    def test_normalization_exclusion_display_half_is_exactly_the_constant(
        self,
    ) -> None:
        exclusion = set(normalization_exclude_columns())
        assert exclusion - set(NORMALIZATION_IDENTIFIER_COLUMNS) == DISPLAY_ONLY_COLUMNS

    def test_identifier_half_is_unchanged_by_this_plan(self) -> None:
        """The eight identifier columns are a different category and stay put."""
        assert NORMALIZATION_IDENTIFIER_COLUMNS == (
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "feature_timestamp",
        )

    @pytest.mark.parametrize("name", sorted(_EXPECTED_DISPLAY_COLUMNS))
    def test_temporal_does_not_restate_a_display_name_as_a_literal(
        self, name: str
    ) -> None:
        source = (_REPO_ROOT / "models" / "temporal.py").read_text(encoding="utf-8")
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source

    @pytest.mark.parametrize(
        "name",
        sorted(_EXPECTED_DISPLAY_COLUMNS - {"raw_weather_severity"}),
    )
    def test_build_features_does_not_restate_a_display_name_as_a_literal(
        self, name: str
    ) -> None:
        source = (_REPO_ROOT / "scripts" / "build_features.py").read_text(
            encoding="utf-8"
        )
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source

    def test_build_features_names_raw_weather_severity_only_to_produce_it(
        self,
    ) -> None:
        """The one legitimate literal: the assignment that CREATES the column."""
        source = (_REPO_ROOT / "scripts" / "build_features.py").read_text(
            encoding="utf-8"
        )
        assert source.count('"raw_weather_severity"') == 1
        assert 'processed_features["raw_weather_severity"]' in source


class TestWalkForwardSplitterExcludesDisplayColumns:
    """The model feature set never contains a display column."""

    def test_no_display_column_survives_into_the_feature_set(self) -> None:
        cols = _splitter()._feature_cols(_synthetic_frame())
        assert not (set(cols) & DISPLAY_ONLY_COLUMNS)

    def test_ordinary_numeric_features_are_untouched(self) -> None:
        cols = _splitter()._feature_cols(_synthetic_frame())
        assert "elo_diff" in cols
        assert "rest_advantage" in cols

    def test_generated_splits_carry_no_display_column(self) -> None:
        """The exclusion reaches TrainTestSplit, not just the helper."""
        base = _synthetic_frame()
        earlier = base.assign(season=2020, game_id=base["game_id"] + "_a")
        frame = pd.concat([earlier, base], ignore_index=True)
        splits = list(_splitter().generate_splits(frame))
        assert splits, "expected one holdout split"
        for split in splits:
            assert not (set(split.train_data.columns) & DISPLAY_ONLY_COLUMNS)
            assert not (set(split.test_data.columns) & DISPLAY_ONLY_COLUMNS)
            assert not split.train_data.isna().to_numpy().any()

    def test_train_val_split_carries_no_display_column(self) -> None:
        base = _synthetic_frame()
        earlier = base.assign(season=2018, game_id=base["game_id"] + "_a")
        frame = pd.concat([earlier, base.assign(season=2020)], ignore_index=True)
        split = _splitter().get_train_val_split(frame)
        assert not (set(split.train_data.columns) & DISPLAY_ONLY_COLUMNS)
        assert not (set(split.test_data.columns) & DISPLAY_ONLY_COLUMNS)


class TestOneConstantTwoConsumers:
    """A display column added in future needs exactly one edit."""

    def test_a_new_display_column_reaches_both_consumers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import feature_columns

        widened = frozenset(DISPLAY_ONLY_COLUMNS | {"raw_sentinel_pct"})
        monkeypatch.setattr(feature_columns, "DISPLAY_ONLY_COLUMNS", widened)

        # Consumer 1: the normalization exclusion (scripts.build_features).
        assert "raw_sentinel_pct" in normalization_exclude_columns()

        # Consumer 2: the model feature set (models.temporal).
        frame = _synthetic_frame()
        frame["raw_sentinel_pct"] = 2.0
        assert "raw_sentinel_pct" not in _splitter()._feature_cols(frame)


@pytest.mark.skipif(
    not (_GOLD_DIR / "features_wp.parquet").exists(),
    reason="data/gold not built on this checkout",
)
class TestRealGold:
    """Measured against the on-disk gold, not a synthetic stand-in.

    Written against rung-2 gold by Plan 30-15; re-anchored to rung-4 gold by Plan
    30-08, whose N-01 re-sync legitimately grew the frame from 6,263 to 6,499 rows.
    The re-anchoring moved the COMPANION row count only -- the claim these tests
    exist for, that the six display columns leave the feature set at the CONSUMER
    while the data is left alone, is unchanged and its NaN anchor did not move.
    """

    @pytest.mark.parametrize(
        ("table", "target_col"),
        [
            ("features_wp", "target_wp"),
            ("features_ats", "target_ats"),
            ("features_ou", "target_ou"),
        ],
    )
    def test_exactly_the_six_display_columns_leave_the_feature_set(
        self, table: str, target_col: str
    ) -> None:
        """Every display column PRESENT in gold is excluded, and nothing else is.

        RE-ANCHORED (Plan 33.2-20), not relaxed. This compared against all six names
        and measured five, because rung 4 removed ``raw_precip_mm`` from gold with
        ``precip_mm`` -- a forecast has no observed millimetres. A column that is not
        in the frame cannot be subtracted from the frame's own numeric columns, so
        the six-name comparison was asking gold a question about a column gold no
        longer has.

        The claim is unchanged and is now stated exactly: the excluded set equals the
        display columns THE FRAME CARRIES, the absent ones are named rather than
        absorbed, and ``kept - naive`` is still empty so nothing is smuggled in.
        """
        frame = pd.read_parquet(_GOLD_DIR / f"{table}.parquet")
        splitter = _splitter(target_col=target_col)

        kept = set(splitter._feature_cols(frame))
        naive_exclude = set(splitter.id_cols) | {target_col}
        naive = {
            c
            for c in frame.select_dtypes(include=["number"]).columns
            if c not in naive_exclude
        }

        in_gold = DISPLAY_ONLY_COLUMNS & set(frame.columns)
        assert in_gold == _DISPLAY_COLUMNS_IN_GOLD, sorted(in_gold)
        assert DISPLAY_ONLY_COLUMNS - in_gold == _DISPLAY_COLUMNS_NOT_IN_GOLD, (
            "the set of display columns gold no longer carries has changed. Since "
            "rung 4 it is exactly raw_precip_mm; a second absentee means another "
            "column left gold and nobody said so."
        )
        assert naive - kept == in_gold
        assert kept - naive == set()

    def test_the_wp_fit_survives_the_nan_the_owner_rulings_put_in_gold(self) -> None:
        """SUPERSEDED BY A RULING, and re-stated where the treatment now lives.

        This asserted that no WP feature column carries NaN -- the concrete failure
        that blocked the Stage-1 orchestrator was ``ValueError: Input X contains NaN``
        out of the WP ``LogisticRegression``. The owner ruled twice on 2026-09-22 that
        a value nothing honest can fill stays BLANK (p332_ steps 7b and 8d), so gold
        now carries NaN in 160 of 175 WP feature columns BY DESIGN. Re-anchoring the
        count would pin a number that says nothing; deleting the node would drop the
        only guard on the failure it was written for.

        So the claim moves to where the rulings put the treatment: the WP pipeline's
        in-fold median imputer with its ``_was_missing`` indicator. A NaN-carrying
        frame is driven through it here. If that pair is ever removed, this fails with
        the original error, which is exactly the coverage the old assertion gave.
        """
        frame = pd.read_parquet(_GOLD_DIR / "features_wp.parquet")
        splitter = _splitter()
        columns = splitter._feature_cols(frame)
        features = frame[columns]

        measured_columns = len(columns)
        with_nulls = int((features.isna().sum() > 0).sum())
        assert measured_columns == phase33_state.P332_20_WP_FEATURE_COLUMNS
        assert with_nulls == phase33_state.P332_20_WP_FEATURE_COLUMNS_WITH_NULLS, (
            f"{with_nulls} of {measured_columns} WP feature columns carry NaN; the "
            "clean build measured "
            f"{phase33_state.P332_20_WP_FEATURE_COLUMNS_WITH_NULLS}. The blanks are "
            "the owner's step-7b and step-8d rulings, so a move here is a change to "
            "what those rulings left blank, not a leak."
        )
        assert with_nulls > 0, (
            "non-vacuity: with no NaN in the frame the pipeline assertion below would "
            "pass against a pipeline that cannot handle one"
        )

        from models.trainers.wp_trainer import (
            MISSING_INDICATOR_SUFFIX,
            WP_PIPELINE_STEP_NAMES,
            WPTrainer,
        )

        pipeline = WPTrainer()._create_model({})
        assert tuple(pipeline.named_steps) == WP_PIPELINE_STEP_NAMES
        assert "imputer" in pipeline.named_steps
        assert "missing_indicator" in pipeline.named_steps
        assert MISSING_INDICATOR_SUFFIX == "_was_missing"

        sample = features.head(200).to_numpy(dtype=float)
        labels = frame["home_win"].head(200).to_numpy(dtype=int)
        assert bool(pd.isna(sample).any()), "fixture sanity: the sample carries NaN"
        # The original failure was an exception, so the assertion is that fitting
        # RETURNS. A bare fit that raises fails this node with that exception.
        pipeline.fit(sample, labels)
        assert pipeline.predict_proba(sample).shape == (len(sample), 2)

    def test_raw_humidity_pct_was_excluded_not_imputed(self) -> None:
        """Proof the CONSUMER was fixed and the DATA was left alone.

        RE-ANCHORED (Plan 33.2-20), and the CLAIM is unchanged: the column is null
        wherever there is no reading and is imputed nowhere. What moved is where
        "no reading" falls. At Phase 30 the upstream table covered season 2025 alone,
        so the null set was "every season but 2025". Since p332_ rung 4 and step 4b
        the source is the day-before forecast over the whole history, so the null set
        is the population weather cannot describe: the 1,030 fixed-roof domes plus the
        74 uncovered games.

        That is asserted as an IDENTITY rather than as a count -- a row is null here
        exactly when its weather family is absent -- so a legitimate row addition
        cannot make it stale, and a single imputed cell still fails it.
        """
        frame = pd.read_parquet(_GOLD_DIR / "features_wp.parquet")
        assert len(frame) == _LIVE_GOLD_ROWS
        nulls = dict(phase33_state.P332_20_DISPLAY_NULLS_AFTER_CLEAN_BUILD)
        assert int(frame["raw_humidity_pct"].isna().sum()) == nulls["raw_humidity_pct"]

        # The identity: null exactly where no observation applies. `weather_coverage`
        # 0.0 is an uncovered game and `weather_affects_game` 0.0 a fixed-roof dome;
        # neither has a humidity reading, and every other row does.
        absent = (frame["weather_coverage"] == 0.0) | (
            frame["weather_affects_game"] == 0.0
        )
        assert bool(frame.loc[absent, "raw_humidity_pct"].isna().all()), (
            "a game with no observation carries a humidity value, so something "
            "imputed it"
        )
        assert not bool(frame.loc[~absent, "raw_humidity_pct"].isna().any()), (
            "a game WITH an observation carries no humidity, so the passthrough "
            "dropped a reading it had"
        )
        assert int(absent.sum()) == nulls["raw_humidity_pct"]

    @pytest.mark.parametrize("table", ["features_wp", "features_ats", "features_ou"])
    def test_the_display_columns_are_still_present_in_gold(self, table: str) -> None:
        """Exclusion happens at the consumer; no column is dropped BY THE EXCLUSION.

        RE-ANCHORED (Plan 33.2-20). The claim is that the model-feature exclusion does
        not reach gold, and it still holds. ``raw_precip_mm`` is nevertheless gone --
        removed by rung 4's registry group because a forecast has no observed
        millimetres, which is a DATA decision and not this exclusion -- so the
        assertion names the one absentee instead of asserting a superset that is no
        longer true. An unexplained sixth absentee still fails.
        """
        frame = pd.read_parquet(_GOLD_DIR / f"{table}.parquet")
        assert set(frame.columns) >= _DISPLAY_COLUMNS_IN_GOLD
        assert DISPLAY_ONLY_COLUMNS - set(frame.columns) == _DISPLAY_COLUMNS_NOT_IN_GOLD

    @pytest.mark.parametrize("table", ["features_wp", "features_ats", "features_ou"])
    def test_display_columns_are_not_constant_on_live_gold(self, table: str) -> None:
        """The exclusion's rationale is SCALE, and this is why it cannot be constancy.

        ``utils.feature_columns`` once justified the exclusion by claiming all six
        columns are constant both before and after the Phase-30 rebuild. The
        "before" half is true on the frozen fixture; the "after" half is false on
        every matrix that shipped. The columns are withheld from
        ``expanding_normalize`` but NOT from ``handle_missing_data_and_outliers``,
        so WR-06's per-season prior-median imputation and per-season winsorization
        moved them.

        Measuring the FIXTURE alone is what let the stale claim through, so this
        reads LIVE gold. The anchors are exact, not tolerances: a legitimate
        weather backfill (D30-DEFER-09) must move them deliberately.

        RE-ANCHORED (Plan 33.2-20) for a legitimate change of exactly that kind, and
        the rationale is UNCHANGED: p332_ rung 4 replaced the ERA5 reanalysis with the
        archived day-before forecast, so these are different measurements of different
        things and every distinct count moved. The counts went UP on every column
        (e.g. raw_temp_f 15 -> 100), which is the forecast carrying real per-game
        variation where the reanalysis had been broadcast from fourteen rows. The
        falsified "all six are constant" claim is NOT restored.
        """
        frame = pd.read_parquet(_GOLD_DIR / f"{table}.parquet")

        measured = {
            name: int(frame[name].nunique())
            for name in sorted(_DISPLAY_COLUMNS_IN_GOLD)
        }
        assert measured == dict(
            phase33_state.P332_20_DISPLAY_NUNIQUE_AFTER_CLEAN_BUILD
        ), (
            f"{table}: display-column distributions moved. utils/feature_columns.py "
            "argues the exclusion from SCALE (un-normalized, un-neutralised nulls), "
            "not from constancy -- re-read that rationale before re-anchoring, and "
            "do not restore the falsified 'all six are constant' claim."
        )

        for name, distinct in measured.items():
            assert distinct > 1, (
                f"{table}.{name} is constant on live gold. That is not a reason to "
                "re-admit it to the feature set -- the exclusion is about scale -- "
                "but it does mean this anchor and the module rationale are stale."
            )

    def test_the_display_columns_carry_nulls_exactly_where_weather_is_absent(
        self,
    ) -> None:
        """The un-neutralised-null half of the scale argument, measured.

        RE-ANCHORED (Plan 33.2-20), and the un-neutralised-null claim is STRONGER than
        before: at Phase 30 only ``raw_humidity_pct`` carried nulls, so the argument
        rested on one column. Since rung 4 four of the five carry them, and the fifth
        does not for a stated reason -- ``raw_weather_severity`` is a COMPOSITE and a
        covered dome genuinely scores 0.0 there, which is Ruling J's split seen on the
        display siblings. Both populations are asserted, so a change to either fails.
        """
        frame = pd.read_parquet(_GOLD_DIR / "features_ats.parquet")
        nulls = {
            name: int(frame[name].isna().sum())
            for name in sorted(_DISPLAY_COLUMNS_IN_GOLD)
        }
        recorded = dict(phase33_state.P332_20_DISPLAY_NULLS_AFTER_CLEAN_BUILD)
        assert nulls == recorded

        # The four MEASUREMENTS are null for the domes AND the uncovered games; the
        # composite only for the uncovered ones. Stated as the arithmetic, so the
        # difference between the two populations cannot be re-pinned away.
        uncovered = int((frame["weather_coverage"] == 0.0).sum())
        domes = int((frame["weather_affects_game"] == 0.0).sum())
        assert recorded["raw_weather_severity"] == uncovered
        for name in (
            "raw_humidity_pct",
            "raw_precip_prob",
            "raw_temp_f",
            "raw_wind_mph",
        ):
            assert recorded[name] == uncovered + domes, name


@pytest.mark.skipif(
    not (_ARTIFACTS_DIR / "latest.json").exists(),
    reason=(
        "artifacts/ is gitignored and absent on this checkout, so the deployed-artifact "
        "residue control did not run -- a green suite here does NOT include it"
    ),
)
class TestDeployedArtifactResidue:
    """The exclusion is FORWARD-LOOKING, and the residue is recorded, not implied.

    ``models.temporal.WalkForwardSplitter._feature_cols`` decides what a FIT can
    select. It has no reach over an artifact already written. The Phase-30 gate
    refused the O/U candidate and RETAINED ``ou_20260326_163930``, the v1.0
    pre-Elo model, which was fitted long before Plan 30-15 and lists three of the
    six display columns among its 25 features.
    ``scripts/generate_current_week_predictions.py`` slices gold by that saved
    list, so those three ARE model inputs in production right now, at raw
    (un-normalized) scale, on values the rebuild moved from constant to varying.

    That is a real gap and it is not closed by a re-fit here -- re-fitting O/U is
    a GATE decision, and the refusal that retained this artifact is sound. What
    was missing was any recorded statement of it. These tests are that record:
    the residue is pinned EXACTLY, so a NEW offender fails and the known one
    cannot quietly become permanent-by-forgetting.

    RE-ANCHORED after Plan 33.2-25's swap. The residue above was closed the intended
    way, by re-fits: production now serves ``P332_25B_SWAP_ARTIFACT_IDS``, and
    MEASURED 2026-09-23 none of the three consumes a display column, so the live
    residue is ``{}``. ``_PHASE30_DEPLOYED_DISPLAY_RESIDUE`` stays as the record of
    what the Phase-30 O/U model consumed.
    """

    @staticmethod
    def _deployed_residue() -> dict[str, list[str]]:
        import json

        manifest = json.loads(
            (_ARTIFACTS_DIR / "latest.json").read_text(encoding="utf-8")
        )
        residue: dict[str, list[str]] = {}
        for target in ("wp", "ats", "ou"):
            artifact = manifest[target]
            feature_list_path = _ARTIFACTS_DIR / artifact / "feature_list.json"
            if not feature_list_path.exists():
                continue
            names = json.loads(feature_list_path.read_text(encoding="utf-8"))
            offending = sorted(set(names) & display_only_columns())
            if offending:
                residue[artifact] = offending
        return residue

    def test_the_deployed_residue_is_exactly_the_recorded_one(self) -> None:
        assert self._deployed_residue() == {}, (
            "A deployed artifact consumes display-only columns. That means a fit "
            "selected a display column despite Plan 30-15, or an artifact predating "
            "it was promoted -- investigate before re-anchoring."
        )

    def test_the_manifest_names_the_swapped_artifacts_and_each_has_a_feature_list(
        self,
    ) -> None:
        """Non-vacuity for the ``{}`` above: ``_deployed_residue`` skips a missing list."""
        import json

        manifest = json.loads(
            (_ARTIFACTS_DIR / "latest.json").read_text(encoding="utf-8")
        )
        assert manifest == dict(phase33_state.P332_25B_SWAP_ARTIFACT_IDS)
        for target in ("wp", "ats", "ou"):
            assert (_ARTIFACTS_DIR / manifest[target] / "feature_list.json").is_file()

    def test_a_freshly_selected_feature_set_could_not_produce_the_residue(
        self,
    ) -> None:
        """The forward-looking half: today's selector cannot pick these up.

        Pins the two halves of the claim together -- the residue exists BECAUSE
        the artifact predates the exclusion, not because the exclusion leaks.
        """
        frame = _synthetic_frame()
        for name in _PHASE30_DEPLOYED_DISPLAY_RESIDUE["ou_20260326_163930"]:
            assert name in frame.columns
            assert name not in _splitter(target_col="target_wp")._feature_cols(frame)
