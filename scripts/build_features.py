#!/usr/bin/env python3
"""
Unified Feature Building Pipeline

This script combines all feature sources into complete feature matrices:
- Team form metrics (rolling 4-week EPA, success rates)
- Elo ratings (with uncertainty and home field advantage)
- Contextual features (travel, weather, venue, rest)
- Weather features (wind, temperature, precipitation - outdoor only)
- Market anchor features (opening/snapshot lines, devigged probabilities)

Feature processing includes:
- Missing data imputation and outlier handling (winsorization)
- Expanding-window normalization (per-season, with prior-season bootstrap)
- Information-time gate: each source's per-game information time against that
  game's own lock (features.provenance, D33.2-01), plus the combined-matrix
  LeakageGate
- Separate feature matrices for WP, ATS, and O/U targets
- Storage in gold layer for model consumption

Usage:
    python scripts/build_features.py --season 2024 --week 1
    python scripts/build_features.py --season 2024  # All weeks in season
    python scripts/build_features.py  # All available data
    python scripts/build_features.py --season 2024 --as-of 2024-10-04T18:00:00
"""

import argparse
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.contextual import ContextualFeaturesCalculator, UnknownStadiumError
from features.elo_features import EloFeatureBuilder
from features.injury import InjuryBuilder
from features.market_anchors import MarketAnchorFeaturesCalculator
from features.normalization import compute_prior_season_stats, expanding_normalize
from features.opponent_adj import OpponentAdjuster
from features.protocol import InformationTimeProvider
from features.provenance import (
    PROVENANCE_COLUMNS,
    CoverageReport,
    InformationTimeGate,
    build_lock_frame,
    refuse_provenance_columns,
)
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder
from features.team_form import TeamFormCalculator
from features.validation import LeakageGate, LeakageViolation
from features.weather import (
    WEATHER_COVERAGE_COLUMN,
    WEATHER_FEATURE_COLUMNS_BY_BUILDER,
    WeatherFeaturesCalculator,
)
from scripts.fingerprint_gold import BUILD_CLOCK_COLUMNS
from utils import get_logger
from utils.date_utils import ET
from utils.exceptions import DataIngestionError
from utils.feature_columns import normalization_exclude_columns

logger = get_logger(__name__)

# Every optional-source guard below catches THIS tuple, and nothing else.
#
# This is not a line-movement problem. ``DataIngestionError`` derives from
# ``NFLPredictException`` and therefore from ``Exception`` directly -- it is NOT a
# subclass of ValueError, OSError or FileNotFoundError -- and ``load_dataframe``
# raises it (``data/storage.py:923``) whenever a table is absent from both DuckDB
# and parquet. Every optional silver source below guards a ``load_dataframe`` call
# or a builder that makes one, so before CR-03 they ALL shared the same false
# graceful-degradation contract: on a fresh clone the guard could not fire, the
# error escaped, and gold could not be built at all -- which falsifies the
# degradation contract this module asserts in prose for the line-movement builder.
#
# One constant is what stops the twelve sites drifting apart again.
#
# ONE TYPE IS DELIBERATELY ABSENT, AND ITS ABSENCE IS LOAD-BEARING (Plan 33-04,
# T-33-18b). ``features.elo_features.ProvisionalSnapshotAsTrainingInputError`` -- the
# refusal of a PROVISIONAL Elo row as a training input -- derives from ``RuntimeError``,
# which is not in this tuple and not a superclass of anything in it, so it PROPAGATES to
# the caller instead of being converted into an empty Elo frame. That conversion is
# exactly the failure the refusal exists to prevent: a gold build with no Elo columns and
# a green exit is indistinguishable from one that worked.
#
# The exclusion is by TYPE and deliberately NOT by an ``except
# ProvisionalSnapshotAsTrainingInputError: raise`` handler. A handler would have to be
# kept in step with the exception hierarchy forever, and adding one here would make this
# module import the refusal it is supposed to know nothing about.
# ``tests/unit/test_provisional_training_refusal.py`` asserts the exclusion member by
# member, asserts by AST that no handler in this file names the refusal or catches bare
# ``Exception``, and drives a build whose Elo source raises it to prove the error reaches
# the caller rather than producing zero rows -- with a control showing that an ordinary
# ``ValueError`` still degrades to an empty frame, which is the behaviour that stays.
_SOURCE_LOAD_ERRORS = (
    DataIngestionError,
    ValueError,
    KeyError,
    TypeError,
    FileNotFoundError,
    OSError,
)

# The Phase-29 signal group this build removes from gold (SPEC R3, D29-07-01).
#
# WHAT DELIBERATELY STAYS, so a reviewer does not read any of it as an oversight:
# ``features/line_movement.py`` stays in the tree, the ``line_movement`` leakage
# keyword stays in ``features/validation.py``, and ``line_movement`` stays
# registered in ``backtest.signal_lift._GROUP_PREDICATE``. Three reasons, all of
# them about the registry:
#
#   1. The registry's own comment block warns against editing it to fix a symptom
#      -- that is the 29-06 trap (a deny-list that cannot name a group which does
#      not exist yet) re-armed for the next phase that widens gold.
#   2. A committed test asserts the registration, and reddening it buys nothing.
#   3. With the entry retained, ``group_columns`` on post-drop gold returns an
#      EMPTY list, ``excluded_columns`` adds nothing, and the family is excluded
#      again AUTOMATICALLY if it ever returns.
_LINE_MOVEMENT_GROUP = "line_movement"

# Families merged AFTER the Stage-1 information-time loop, so they are not
# ``feature_sources`` registry keys at all and cannot be reached by registering a
# provenance supplier. Reported in their OWN set of the CoverageReport rather than
# folded into the unregistered keys, so a structural gap is not hidden inside a
# bookkeeping one. The opponent-adjusted family is merged in
# ``generate_feature_matrices`` after Stage 1; Plan 33.2-16 registers it and
# removes it from this tuple.
POST_STAGE1_SOURCES: tuple[str, ...] = ("opponent_adj",)

# THE TWO WEATHER BUILDER IDENTITIES (Ruling K1, Plan 33.1-04).
#
# `features.weather` exposes two builders that do NOT emit the same weather
# columns: `build_weather_features` (the "full" builder, which writes silver
# `weather_features` and therefore feeds gold) and `build_features` (the
# "compressed" FeatureBuilder-Protocol builder). The missing-preserving set
# below is keyed by these names because ONE broad set would become an assertion
# about whichever builder a test happened to exercise, while the other silently
# median-filled -- and half the evidence saying it works is worse than none.
#
# The order matches `WEATHER_FEATURE_COLUMNS_BY_BUILDER`, and a test asserts
# the two agree, so a third builder cannot appear on one side only.
BUILDER_KEYS: tuple[str, ...] = ("full", "compressed")

# THE NAME-BASED HALF OF THE LEVEL-PRESERVATION PREDICATE (Plan 33-14 Task 3,
# D33-34(a), .planning/WINDOWS.md rows 33 and 40).
#
# A column whose name ends with this suffix answers "is there an observation
# behind this row". Its LEVELS ARE ITS MEANING, constant or not, and that is
# precisely why the suffix arm carries NO varying-levels requirement while the
# weather arm does: `weather_coverage` is a CONSTANT 1.0 on the corrected corpus,
# and the whole defect Phase 33.1 spent a rung on was that constant being
# z-scored to 0.0 -- the value that means NO OBSERVATION. A rule that demanded
# two levels here would re-break the exact column it was written to protect.
LEVEL_PRESERVED_COLUMN_SUFFIX: str = "_coverage"


def drop_feature_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return *df* without any column belonging to *group*.

    The column set is obtained from ``backtest.signal_lift.group_columns`` -- the
    ONE group registry the Phase-28 screen, the Phase-29 screen and the Phase-30
    gate all already read (D30-02). This module therefore carries no second list
    of the family's names, which is exactly the failure mode that let Phase 29's
    fifteen columns fall into the Phase-28 baseline.

    The import is deferred rather than module-level because ``backtest.signal_lift``
    pulls in ``backtest.diagnose`` and all three trainers, i.e. the whole model
    stack -- far too heavy for a data-layer build script to import eagerly, and a
    layering inversion besides. ``scripts/fingerprint_gold.py`` defers the identical
    import for the identical reason.

    Raises:
        ValueError: when *group* matches NO column in *df*. A drop expressed as a
            predicate can be misspelled, and a misspelled predicate removes nothing
            while every downstream width count and presence check reads exactly as
            it would after a successful drop. Refusing is what stops this build
            emitting gold that only LOOKS dropped.
    """
    from backtest.signal_lift import group_columns

    columns = group_columns(df, group)
    if not columns:
        msg = (
            f"drop_feature_group('{group}') matched NO column of the "
            f"{len(df.columns)}-column frame it was asked to drop from. A drop that "
            "removes nothing is indistinguishable downstream from a drop that "
            "worked, so this build refuses to emit gold that only LOOKS dropped. "
            "Check the group predicate in backtest.signal_lift._GROUP_PREDICATE "
            "against the frame's actual column names."
        )
        raise ValueError(msg)
    return df.drop(columns=columns)


class FeatureMatrixBuilder:
    """
    Build unified feature matrices from all feature sources.

    Combines team form, Elo, contextual, weather, and market features
    into complete feature matrices ready for model training.
    """

    # WR-06: the minimum number of non-null values a FIT SOURCE must hold before it
    # can support an imputation median or a winsorization bound. The threshold is
    # preserved verbatim from the pre-WR-06 shape (``notna().sum() > 10``); what
    # moved is WHAT it is applied to -- the strictly-prior slice the statistic is
    # actually estimated from, rather than the whole multi-season column.
    _MIN_FIT_POINTS = 10

    def __init__(self):
        """Initialize feature matrix builder."""
        self.logger = get_logger(__name__)

        # Feature calculators
        self.team_form_calc = TeamFormCalculator()
        self.elo_calc = EloFeatureBuilder()
        self.contextual_calc = ContextualFeaturesCalculator()
        self.weather_calc = WeatherFeaturesCalculator()
        self.market_calc = MarketAnchorFeaturesCalculator()
        self.qb_tracker = QBTracker()
        self.opponent_adj = OpponentAdjuster(window=10, min_opponent_games=4)

        # Snaps are constructed (and invoked) BEFORE injuries (D-09 build order):
        # InjuryBuilder consumes the SnapCountBuilder per-position prior shares as
        # its availability weights via the constructor handoff (review #6), which
        # LOCKS the snaps-before-injuries order rather than relying on an implicit
        # re-load. The contextual situational-spot features (Plan 28-04) already
        # flow through self.contextual_calc; no separate builder is needed here.
        self.snap_builder = SnapCountBuilder()
        self.injury_builder = InjuryBuilder(snap_builder=self.snap_builder)

        # Leakage gate for hard-fail validation of the COMBINED matrix
        self.leakage_gate = LeakageGate()

        # The last build's information-time coverage (four named sets). Set by
        # ``_check_information_times``; None until a build has run Stage 1.
        self.information_time_coverage: CoverageReport | None = None

        # Feature processing parameters
        self.outlier_percentiles = (1, 99)  # Winsorization bounds
        self.min_games_for_stats = 10  # Minimum games for normalization

        # WR-06: the machine-readable self-fit flag. Maps a column name to the
        # seasons whose imputation median / winsorization bounds were fitted on
        # their OWN rows because no usable strictly-prior slice existed. Reset at
        # the start of every ``handle_missing_data_and_outliers`` call and logged at
        # its end, so the flag is observable in a build log AND assertable by a
        # test. A self-fit outside the earliest data-bearing season is a per-column
        # coverage floor -- a fact worth surfacing, which a log line alone would
        # never make checkable.
        self.self_fit_seasons: dict[str, list[int]] = {}

        # RULING K1: the per-builder missing-preserving set. `None` until a
        # weather frame is merged, and the weather branch of `combine_features`
        # is its ONLY writer. The mapping carries exactly ONE key -- the builder
        # whose frame was actually merged -- because a build merges one weather
        # frame, and declaring an entry for a builder that did not run would be
        # asserting about a set nothing consumed.
        self.missing_preserving_columns: dict[str, tuple[str, ...]] | None = None
        self.active_builder_key: str | None = None

    # ------------------------------------------------------------------
    # Ruling K1: the per-builder missing-preserving seam
    # ------------------------------------------------------------------

    def record_missing_preserving_columns(self, weather_features: pd.DataFrame) -> str:
        """Declare the preserving set from the weather frame that was MERGED.

        DERIVED, never hand-listed. The set is the merged frame's own columns
        minus ``game_id``, cross-checked against
        ``features.weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER`` so a hand edit
        to either side is a failure rather than a silent divergence.

        The builder identity is RESOLVED from those columns rather than passed
        in, because the caller reads the frame off silver and does not otherwise
        know which builder wrote it.

        Args:
            weather_features: The weather frame about to be merged, carrying
                ``game_id`` plus one builder's feature columns.

        Returns:
            The resolved builder key.

        Raises:
            ValueError: The frame's columns are not a subset of either declared
                builder entry, so no set can be derived from them honestly.
        """
        merged = [c for c in weather_features.columns if c != "game_id"]
        declared_union = set().union(
            *(set(v) for v in WEATHER_FEATURE_COLUMNS_BY_BUILDER.values())
        )
        undeclared = sorted(set(merged) - declared_union)
        if undeclared:
            msg = (
                "the merged weather frame carries columns that belong to no "
                f"declared builder entry: {undeclared}. Update features.weather's "
                "family tuples rather than widening this resolver -- a set "
                "derived from an unrecognised frame is a set nobody declared."
            )
            raise ValueError(msg)

        # RESOLVE BY BEST MATCH, not by subset.
        #
        # A subset test looks tighter and is wrong here: the two entries OVERLAP
        # (`weather_severity_score` and `wind_mph` are in both) and each has
        # columns the other lacks, so a frame carrying, say, `temp_f` AND
        # `is_outdoor` is a subset of NEITHER entry while every one of its
        # columns is declared. A synthetic build frame looks exactly like that,
        # and refusing it would turn a legitimate build into a declaration
        # error.
        #
        # Only the winning entry's columns are preserved. The rest keep today's
        # behaviour, which is the conservative direction: the exemption stays
        # narrow and opt-in rather than widening itself to whatever arrived.
        builder_key = max(
            BUILDER_KEYS,
            key=lambda key: len(
                set(merged) & set(WEATHER_FEATURE_COLUMNS_BY_BUILDER[key])
            ),
        )
        declared = WEATHER_FEATURE_COLUMNS_BY_BUILDER[builder_key]
        self.missing_preserving_columns = {
            builder_key: tuple(c for c in merged if c in set(declared))
        }
        self.active_builder_key = builder_key
        logger.info(
            "Recorded the missing-preserving weather columns",
            builder=builder_key,
            preserved=len(self.missing_preserving_columns[builder_key]),
        )
        return builder_key

    def _preserved_weather_columns(self) -> tuple[str, ...]:
        """The active builder's preserving set. FAIL-CLOSED, per builder.

        An exemption that silently does nothing is worse than none, because it
        reads as a guarantee and behaves as a comment -- the same failure mode
        ``stamp_weather_source_on_existing_rows`` records for a silently-ignored
        ``base_path``. An exemption that is live for one builder and inert for
        the other is worse still, because half the evidence says it works, so
        the refusal names WHICH builder it is refusing for.

        Returns:
            The preserved column names, or an empty tuple when no weather frame
            was merged at all (a build with no weather source has nothing to
            preserve and nothing to fabricate).

        Raises:
            ValueError: A weather frame WAS merged and its entry is absent,
                ``None`` or empty.
        """
        if self.missing_preserving_columns is None:
            return ()
        builder_key = self.active_builder_key
        if builder_key is None:
            msg = (
                "missing_preserving_columns is set but active_builder_key is "
                "None, so no call site can tell which builder's entry to read"
            )
            raise ValueError(msg)
        preserved = self.missing_preserving_columns.get(builder_key)
        if not preserved:
            msg = (
                "missing_preserving_columns has no usable entry for builder "
                f"{builder_key!r}. A weather frame WAS merged, so the "
                "prior-seasons median and the neutral 0.0 z-score would both "
                "run over the weather family and put back the numeric stand-in "
                "SPEC prohibition 1 forbids. Refusing rather than imputing."
            )
            raise ValueError(msg)
        return tuple(preserved)

    def load_all_feature_sources(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all feature sources from silver layer.

        Args:
            target_season: Specific season to load
            target_week: Specific week to load
            as_of_datetime: Time-fence cutoff for builders that need it
                (QBTracker, OpponentAdjuster). Defaults to ``datetime.now(ET)``.

        Returns:
            Dictionary with all feature DataFrames
        """
        # CR-01: the default MUST be tz-aware. A naive ``datetime.now()`` is a LOCAL
        # wall clock, and every downstream consumer re-labels it as UTC without
        # shifting it (``ensure_utc_aware`` reinterprets, and the since-retired
        # kickoff-versus-now fence tz_localized). On the owner's ET machine
        # the Friday 18:05 ET orchestrator slot became 18:05Z == 14:05 ET, silently
        # fencing OUT the Friday-6PM-ET freeze snapshot that the whole D-12 cadence
        # exists to capture; east of UTC the same bug points the other way and LEAKS.
        # ET is the project's canonical wall clock (kickoffs, the Friday freeze), so
        # that is what "now" means here; being aware, it CONVERTS correctly instead
        # of being reinterpreted.
        if as_of_datetime is None:
            as_of_datetime = datetime.now(ET)
        logger.info(
            "Loading all feature sources",
            target_season=target_season,
            target_week=target_week,
        )

        feature_sources = {}

        try:
            # Core game data.
            #
            # WR-10: filter by season when a season is given, and narrow by week only
            # when a week is ALSO given. This previously required BOTH, while every
            # builder's ``build_features`` filters on each independently and main()
            # declares --season and --week as independent options -- so a season-only
            # build left most games unfiltered here, produced builder rows only for
            # the target season, and left the rest NaN after the left merge, where
            # the whole-column median then filled them.
            #
            # The defect was FOUND on the Phase-29 coverage flag, whose median is 1.0
            # -- so the fill stamped every uncovered game as covered. That family has
            # since left gold (SPEC R3), but the filter defect is general to every
            # left-merged source and the fix stays.
            games_df = load_dataframe("games", layer="silver")
            if target_season:
                games_df = games_df[games_df["season"] == target_season]
                if target_week:
                    games_df = games_df[games_df["week"] == target_week]
            feature_sources["games"] = games_df
            logger.info("Loaded games data", records=len(games_df))

            # Team form features
            try:
                team_form_df = load_dataframe("team_form_features", layer="silver")
                if target_season and target_week:
                    team_form_df = team_form_df[
                        (team_form_df["target_season"] == target_season)
                        & (team_form_df["target_week"] == target_week)
                    ]
                feature_sources["team_form"] = team_form_df
                logger.info("Loaded team form features", records=len(team_form_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load team form features", error=str(e))
                feature_sources["team_form"] = pd.DataFrame()

            # Elo features (computed on-the-fly via EloFeatureBuilder)
            try:
                elo_df = self.elo_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["elo"] = elo_df
                logger.info("Built Elo features", records=len(elo_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build Elo features", error=str(e))
                feature_sources["elo"] = pd.DataFrame()

            # Contextual features (computed on-the-fly for Phase 5 additions)
            try:
                contextual_df = self.contextual_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["contextual"] = contextual_df
                logger.info("Built contextual features", records=len(contextual_df))
            except UnknownStadiumError:
                # A STADIUM-RESOLUTION REFUSAL IS NOT A SOURCE-LOAD FAILURE, and
                # this handler exists only because the type system cannot say so
                # (owner ruling 2026-09-13, found by Plan 33.1-03).
                #
                # `UnknownStadiumError` subclasses `ValueError`, which is a
                # member of `_SOURCE_LOAD_ERRORS`, so without this line the
                # loudest thing R1's resolver can say -- "this game names a
                # stadium I have no record for" -- became a WARNING and an EMPTY
                # contextual frame. Gold would then be built with no
                # venue / travel / rest / situational family at all, and the run
                # would exit 0. A green run is precisely what that defect
                # produces, which is the milestone invariant this protects.
                #
                # SURGICAL on purpose. Removing `ValueError` from
                # `_SOURCE_LOAD_ERRORS` would move eleven other optional-source
                # guards whose degradation contract is deliberate and separately
                # tested, and re-basing the exception would change what every
                # existing `except ValueError` around the resolver catches.
                raise
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build contextual features", error=str(e))
                feature_sources["contextual"] = pd.DataFrame()

            # Weather features.
            #
            # source="parquet", NOT the default "auto" (code review WR-02). This is
            # the read that FEEDS GOLD, and it was the one read left on "auto" after
            # Plan 33.1-02 hardened the two raw-table reads. "auto" tries DuckDB
            # FIRST (data/storage.py:1091-1096) and db.table_exists is LAYER-BLIND,
            # so a DuckDB table named weather_features under any layer wins over
            # data/silver/weather_features.parquet -- which is the exact mechanism
            # this phase's root-cause analysis blames for 6,485 of 6,499 gold rows
            # sitting at a fabricated mild-temperature default. The two stores agree
            # today (6,499 rows, identical columns, weather_coverage sums equal), so
            # naming the source changes nothing now and closes the way back.
            try:
                weather_df = load_dataframe("weather_features", "silver", "parquet")
                if target_season and target_week:
                    weather_df = weather_df[
                        (weather_df["season"] == target_season)
                        & (weather_df["week"] == target_week)
                    ]
                feature_sources["weather"] = weather_df
                logger.info("Loaded weather features", records=len(weather_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load weather features", error=str(e))
                feature_sources["weather"] = pd.DataFrame()

            # Market anchor features (computed on-the-fly via MarketAnchorFeaturesCalculator)
            try:
                market_df = self.market_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["market"] = market_df
                logger.info("Built market anchor features", records=len(market_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build market anchor features", error=str(e))
                feature_sources["market"] = pd.DataFrame()

            # QB tracking features (computed via QBTracker, not loaded from silver)
            try:
                qb_features_df = self.qb_tracker.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["qb_tracking"] = qb_features_df
                logger.info("Loaded QB tracking features", records=len(qb_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load QB tracking features", error=str(e))
                feature_sources["qb_tracking"] = pd.DataFrame()

            # Snap-count features (computed via SnapCountBuilder). CRITICAL build
            # order (D-09): snaps are invoked BEFORE injuries so the per-position
            # prior snap shares are available to the injury availability metric.
            try:
                snap_features_df = self.snap_builder.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["snaps"] = snap_features_df
                logger.info("Built snap-count features", records=len(snap_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build snap-count features", error=str(e))
                feature_sources["snaps"] = pd.DataFrame()

            # Injury features (computed via InjuryBuilder AFTER snaps; the builder
            # draws its D-09 availability weights from the constructor-injected
            # SnapCountBuilder's per-position prior shares, review #6).
            try:
                injury_features_df = self.injury_builder.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                )
                feature_sources["injury"] = injury_features_df
                logger.info("Built injury features", records=len(injury_features_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to build injury features", error=str(e))
                feature_sources["injury"] = pd.DataFrame()

            # SEAM 1 of 2 for the Phase-29 line-movement family is DELIBERATELY
            # ABSENT here (SPEC R3, D29-07-01). The family used to be registered at
            # this point and merged by an explicit block in ``combine_features``;
            # BOTH have been removed, because ``combine_features`` has no generic
            # loop and leaving either one behind resurrects the columns on the next
            # rebuild. The paid ``odds_timeline`` archive and
            # ``features/line_movement.py`` are untouched -- what left is gold, not
            # the data or the builder.

            return feature_sources

        except _SOURCE_LOAD_ERRORS as e:
            logger.error("Failed to load feature sources", error=str(e))
            raise

    def combine_features(
        self, feature_sources: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all feature sources into unified feature matrix.

        Args:
            feature_sources: Dictionary with all feature DataFrames

        Returns:
            Combined feature matrix
        """
        logger.info("Combining all feature sources")

        # Start with games as base
        games_df = feature_sources["games"].copy()

        if len(games_df) == 0:
            logger.error("No games data available")
            return pd.DataFrame()

        # Initialize combined features with game identifiers
        combined_features = games_df[
            ["game_id", "season", "week", "home_team", "away_team"]
        ].copy()

        # Add game results if available (for target creation)
        if "home_score" in games_df.columns:
            combined_features["home_score"] = games_df["home_score"]
        if "away_score" in games_df.columns:
            combined_features["away_score"] = games_df["away_score"]

        # Merge each feature source
        feature_counts = {}

        # Team form features (need to handle home/away separately)
        team_form_df = feature_sources.get("team_form", pd.DataFrame())
        if len(team_form_df) > 0:
            home_form = self._get_team_features(
                team_form_df, combined_features, "home_team", "home"
            )
            away_form = self._get_team_features(
                team_form_df, combined_features, "away_team", "away"
            )
            combined_features = combined_features.merge(
                home_form, on="game_id", how="left"
            )
            combined_features = combined_features.merge(
                away_form, on="game_id", how="left"
            )
            feature_counts["team_form"] = len(
                [col for col in combined_features.columns if "form_" in col]
            )

        # Elo features (game-level, already has home/away columns from EloFeatureBuilder)
        elo_df = feature_sources.get("elo", pd.DataFrame())
        if len(elo_df) > 0:
            from features.elo_features import ELO_FEATURE_COLUMNS

            merge_cols = ["game_id"] + [
                c for c in ELO_FEATURE_COLUMNS if c in elo_df.columns
            ]
            combined_features = combined_features.merge(
                elo_df[merge_cols], on="game_id", how="left"
            )
            feature_counts["elo"] = len(
                [col for col in combined_features.columns if "elo_" in col]
            )

        # Contextual features (game-level)
        contextual_df = feature_sources.get("contextual", pd.DataFrame())
        if len(contextual_df) > 0:
            merge_cols = ["game_id"]
            contextual_features = contextual_df.drop(
                columns=["season", "week", "home_team", "away_team"], errors="ignore"
            )
            combined_features = combined_features.merge(
                contextual_features, on=merge_cols, how="left"
            )
            feature_counts["contextual"] = len(
                [col for col in contextual_features.columns if col != "game_id"]
            )

        # Weather features (game-level)
        weather_df = feature_sources.get("weather", pd.DataFrame())
        if len(weather_df) > 0:
            merge_cols = ["game_id"]
            weather_features = weather_df.drop(
                columns=["season", "week"], errors="ignore"
            )
            # RULING K1: the ONE writer of the missing-preserving set, placed
            # at the merge so the set is derived from the frame that actually
            # arrived rather than from a list somebody maintains by hand.
            self.record_missing_preserving_columns(weather_features)
            combined_features = combined_features.merge(
                weather_features, on=merge_cols, how="left"
            )
            feature_counts["weather"] = len(
                [col for col in weather_features.columns if col != "game_id"]
            )

        # Market anchor features (game-level, compressed 5 features)
        market_df = feature_sources.get("market", pd.DataFrame())
        if len(market_df) > 0:
            market_feature_cols = [
                "snapshot_spread",
                "snapshot_total",
                "snapshot_ml_prob_home_fair",
                "spread_movement",
                "total_movement",
            ]
            merge_cols = ["game_id"] + [
                c for c in market_feature_cols if c in market_df.columns
            ]
            combined_features = combined_features.merge(
                market_df[merge_cols], on="game_id", how="left"
            )
            feature_counts["market"] = len(
                [c for c in market_feature_cols if c in market_df.columns]
            )

        # QB adjustment features (one value per team per game)
        qb_df = feature_sources.get("qb_tracking", pd.DataFrame())
        if len(qb_df) > 0 and "qb_adjustment" in qb_df.columns:
            # Merge home QB adjustment
            home_qb = qb_df[["game_id", "team", "qb_adjustment"]].copy()
            home_qb = home_qb.merge(
                combined_features[["game_id", "home_team"]].drop_duplicates(),
                left_on=["game_id", "team"],
                right_on=["game_id", "home_team"],
                how="inner",
            )
            home_qb = home_qb[["game_id", "qb_adjustment"]].rename(
                columns={"qb_adjustment": "home_qb_adjustment"}
            )
            combined_features = combined_features.merge(
                home_qb, on="game_id", how="left"
            )

            # Merge away QB adjustment
            away_qb = qb_df[["game_id", "team", "qb_adjustment"]].copy()
            away_qb = away_qb.merge(
                combined_features[["game_id", "away_team"]].drop_duplicates(),
                left_on=["game_id", "team"],
                right_on=["game_id", "away_team"],
                how="inner",
            )
            away_qb = away_qb[["game_id", "qb_adjustment"]].rename(
                columns={"qb_adjustment": "away_qb_adjustment"}
            )
            combined_features = combined_features.merge(
                away_qb, on="game_id", how="left"
            )

            feature_counts["qb_tracking"] = 2  # home + away qb_adjustment

        # Snap-count features (game-level; SnapCountBuilder already emits the
        # home_/away_-expanded columns, so merge on game_id like the elo block).
        # combine_features has NO generic loop over feature_sources keys -- without
        # this explicit block the registered snap columns pass the LeakageGate but
        # are SILENTLY DROPPED from gold (review #1, orchestrator-verified).
        snaps_df = feature_sources.get("snaps", pd.DataFrame())
        if len(snaps_df) > 0:
            snap_cols = [c for c in snaps_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                snaps_df[["game_id", *snap_cols]], on="game_id", how="left"
            )
            feature_counts["snaps"] = len(snap_cols)

        # Injury features (game-level; InjuryBuilder already emits the home_/away_-
        # expanded columns). Same rationale as the snap block (review #1): merge on
        # game_id so the columns actually reach all three gold matrices.
        injury_df = feature_sources.get("injury", pd.DataFrame())
        if len(injury_df) > 0:
            injury_cols = [c for c in injury_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                injury_df[["game_id", *injury_cols]], on="game_id", how="left"
            )
            feature_counts["injury"] = len(injury_cols)

        # SEAM 2 of 2 for the Phase-29 line-movement family is DELIBERATELY ABSENT
        # here (SPEC R3, D29-07-01). This is where an explicit merge block used to
        # sit, and it is the seam that actually landed the columns in gold: a
        # source registered in ``feature_sources`` but not merged here passes the
        # LeakageGate and is then silently dropped (the 28-06 lesson). Read in
        # reverse, that is exactly why removing only the registration would not
        # have been enough on its own, and why removing only this block would not
        # either -- a later reader restoring one seam must restore both, and
        # ``_enforce_line_movement_dropped`` will remove the result anyway.

        # Add feature timestamp (tz-aware UTC; the storage layer rejects naive
        # datetimes, and feature_timestamp is persisted into every gold matrix)
        combined_features["feature_timestamp"] = datetime.now(UTC)

        logger.info(
            "Combined all features",
            total_games=len(combined_features),
            total_features=len(combined_features.columns),
            feature_breakdown=feature_counts,
        )

        return combined_features

    def _information_time_suppliers(self) -> dict[str, object]:
        """The builder behind each ``feature_sources`` registry key.

        This maps a key to the object that BUILDS it; it does not decide which keys are
        checked. Admission to the gate is ``isinstance(builder,
        InformationTimeProvider)`` -- a structural fact about the builder, not a
        hand-kept list -- so a builder that gains the two provenance members is checked
        with no edit here.
        """
        return {
            "team_form": self.team_form_calc,
            "elo": self.elo_calc,
            "contextual": self.contextual_calc,
            "weather": self.weather_calc,
            "market": self.market_calc,
            "qb_tracking": self.qb_tracker,
            "snaps": self.snap_builder,
            "injury": self.injury_builder,
        }

    def _check_information_times(
        self,
        feature_sources: dict[str, pd.DataFrame],
        lock_frame: pd.Series,
        target_season: int | None,
        target_week: int | None,
    ) -> CoverageReport:
        """Stage 1: every registered source's per-game information time vs its lock.

        A STAGED ROLLOUT, NEVER AN EXEMPTION. At Plan 33.2-01 only ``elo`` satisfies
        ``InformationTimeProvider``. The remaining keys gain suppliers in the plans that
        make them lock-honest (33.2-12 .. 33.2-17); Plan 33.2-16 brings the
        opponent-adjusted family into the loop; and Plan 33.2-20 arms the refusal of any
        key with no provenance once every source has one. Until then the unchecked keys
        are NAMED in the CoverageReport, which is logged at every build -- no source is
        allow-listed, excepted or run in a report-only mode, and nothing here downgrades
        a refusal to a warning.

        Returns:
            The four-set CoverageReport, also stored on
            ``self.information_time_coverage``.

        Raises:
            features.provenance.InformationTimeViolation: from the gate, naming the
                source and game(s). Deliberately outside ``_SOURCE_LOAD_ERRORS``.
        """
        gate = InformationTimeGate()
        suppliers = self._information_time_suppliers()
        games_df = feature_sources["games"]
        empty_unregistered: list[str] = []
        unregistered: list[str] = []

        for source_name, source_df in feature_sources.items():
            if source_name == "games":
                continue  # the base frame, not a builder output

            supplier = suppliers.get(source_name)
            if isinstance(supplier, InformationTimeProvider):
                # An empty frame is reported by the gate BEFORE it reads provenance, so
                # the supplier is not asked to date a source that failed to load.
                provenance = (
                    supplier.information_times(
                        games_df, target_season=target_season, target_week=target_week
                    )
                    if len(source_df) > 0
                    else pd.DataFrame({column: [] for column in PROVENANCE_COLUMNS})
                )
                gate.check(
                    source_name,
                    source_df,
                    provenance,
                    lock_frame,
                    no_information_signature=supplier.no_information_signature(),
                )
            elif len(source_df) == 0:
                empty_unregistered.append(source_name)
            else:
                unregistered.append(source_name)

        report = CoverageReport(
            checked_sources=gate.checked_sources,
            empty_unchecked_sources=tuple(
                sorted({*gate.empty_unchecked_sources, *empty_unregistered})
            ),
            unregistered_sources=tuple(sorted(unregistered)),
            post_stage1_sources=POST_STAGE1_SOURCES,
        )
        self.information_time_coverage = report
        logger.info(
            "Information-time gate coverage (Stage 1). ONLY checked_sources were "
            "checked against each game's lock; this is not whole-build coverage.",
            checked_sources=list(report.checked_sources),
            empty_unchecked_sources=list(report.empty_unchecked_sources),
            unregistered_sources=list(report.unregistered_sources),
            post_stage1_sources=list(report.post_stage1_sources),
        )
        return report

    def _enforce_line_movement_dropped(
        self, combined_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Guarantee the Phase-29 line-movement family does not reach gold.

        SPEC R3 / D29-07-01. On the intended path this finds nothing and returns
        the frame untouched, because BOTH seams that could land the family have
        been removed -- the ``feature_sources`` registration and the explicit
        ``combine_features`` merge block. That is the structural removal, and it
        is the one that matters.

        This is nevertheless not dead code, and the ``if`` is not a formality.
        ``combine_features`` has NO generic loop over ``feature_sources``, so a
        seam restored by a later edit lands its columns in gold SILENTLY -- the
        LeakageGate passes them and nothing else looks. This is the one place that
        would notice, and it sits before ``handle_missing_data_and_outliers``, so a
        reinstated family is removed before any imputation or winsorization can see
        it (which is what makes removing the WR-10 neutral-default guard safe).

        NOTE ON THE EMPTY CASE, because the asymmetry is deliberate.
        ``drop_feature_group`` REFUSES a zero match -- a drop asked to remove
        something and removing nothing is indistinguishable downstream from one
        that worked. A build whose seams are gone was never asking, so it must not
        raise; the refusal guards the drop, and the seam removal guards the build.
        """
        # Deferred for the same reason as in ``drop_feature_group``: importing
        # ``backtest.signal_lift`` eagerly pulls the whole model stack into a
        # data-layer build script.
        from backtest.signal_lift import group_columns

        present = group_columns(combined_features, _LINE_MOVEMENT_GROUP)
        if not present:
            return combined_features

        logger.warning(
            "Line-movement columns reached the combined matrix and were dropped "
            "before gold; a removed merge/registration seam has returned",
            columns=present,
            count=len(present),
        )
        return drop_feature_group(combined_features, _LINE_MOVEMENT_GROUP)

    def _get_team_features(
        self,
        team_form_df: pd.DataFrame,
        games_df: pd.DataFrame,
        team_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Get team form features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's offensive features
            team_off = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "offense")
            ]

            # Get team's defensive features
            team_def = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "defense")
            ]

            game_features = {"game_id": game_id}

            # Add offensive features
            if len(team_off) > 0:
                off_row = team_off.iloc[0]
                for col in off_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_off_{col}"
                        game_features[feature_name] = off_row[col]

            # Add defensive features
            if len(team_def) > 0:
                def_row = team_def.iloc[0]
                for col in def_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_def_{col}"
                        game_features[feature_name] = def_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def _get_team_elo_features(
        self, elo_df: pd.DataFrame, games_df: pd.DataFrame, team_col: str, prefix: str
    ) -> pd.DataFrame:
        """Get Elo features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's Elo rating
            team_elo = elo_df[
                (elo_df["team"] == team)
                & (elo_df["season"] == season)
                & (elo_df["week"] == week)
            ]

            game_features = {"game_id": game_id}

            if len(team_elo) > 0:
                elo_row = team_elo.iloc[0]
                for col in elo_row.index:
                    if col not in ["team", "season", "week"]:
                        feature_name = f"{prefix}_{col}"
                        game_features[feature_name] = elo_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def handle_missing_data_and_outliers(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Handle missing data and outliers with winsorization.

        WR-06: every distributional statistic below is fitted on the seasons
        STRICTLY BEFORE the season it is applied to. Before this, the imputation
        median and the q01/q99 bounds were computed over the whole multi-season
        frame, so season 2025 could move a 2021 feature value -- the temporal
        boundary this phase turns on, and the reason SPEC R2's byte-identity
        control exists.

        DELIBERATE DIVERGENCE FROM THE HOUSE PRECEDENT. ``features.normalization.
        compute_prior_season_stats`` fits on season Y-1 ONLY; this fits on ALL
        seasons before Y. Both are point-in-time; they differ in sample size, and
        D30-16's own rationale asks for a full-season-or-more sample so
        winsorization stays stable -- a q01/q99 estimate from roughly 285 games is
        materially noisier than one from thousands. The two also run at different
        stages on different statistics (mean and standard deviation for a z-score
        bootstrap, versus median and quantiles for imputation and clipping), so
        they are NOT required to agree. Do not "fix" one to match the other.

        ACCEPTED RESIDUAL, documented rather than left unstated: within-season
        lookahead remains. A self-fitting season's week-1 bound still sees that
        season's week 18, and ``_impute_team_features``' team mean and season mean
        are still within-season. D30-16 accepts both, and neither can be moved by
        adding a LATER season's rows -- which is exactly what SPEC R2 asserts.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from processing

        Returns:
            Processed feature matrix
        """
        logger.info(
            "Handling missing data and outliers", features=len(features_df.columns)
        )

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        processed_df = features_df.copy()

        # Get numeric feature columns
        feature_cols = [
            col for col in processed_df.columns if col not in target_columns
        ]
        numeric_cols = (
            processed_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        missing_stats = {}
        outlier_stats = {}

        # WR-06: the per-season passes, computed ONCE. ``season`` sits in the
        # target-column exclusion list above, which removes it from ``numeric_cols``
        # but leaves the column in the frame -- so a per-season pass can group on it
        # directly. Each entry is (season, season_mask, strictly_prior_mask).
        self.self_fit_seasons = {}
        season_passes = self._season_passes(processed_df)
        earliest_season = season_passes[0][0] if season_passes else None

        # The WR-10 neutral-default branch that used to open this loop is GONE with
        # the family it guarded (SPEC R3). It filled the Phase-29 line-movement
        # columns from the builder's own neutral defaults rather than from a
        # median, because the median of the coverage flag is 1.0 and a median fill
        # therefore fabricated coverage. Nothing is weakened by its removal: the
        # SPEC R3 drop runs on the COMBINED matrix, before this method is ever
        # called, so a family reinstated by a returning seam is already gone by the
        # time any imputation could see it. Keeping an unreachable guard that names
        # a builder this module no longer imports would be a false statement about
        # what the code does.
        # RULING K1, evaluated ONCE and FAIL-CLOSED. Read before the loop so a
        # merged weather frame with no usable entry refuses here rather than
        # after silently median-filling the first weather column it meets.
        preserved_weather_columns = set(self._preserved_weather_columns())

        for col in numeric_cols:
            original_missing = processed_df[col].isna().sum()

            # THE WEATHER EXEMPTION (SPEC prohibition 1). Neither imputer runs
            # for a column in the active builder's preserving set: a seasonal or
            # venue median for an absent observation is the same defect wearing
            # a better label, which is exactly what the WR-10 neutral-default
            # branch removed from the top of this loop was doing for the
            # line-movement family. The NaN is the answer, and it survives
            # expanding_normalize too.
            #
            # The exemption is from IMPUTATION only, and deliberately not from
            # winsorization: a MEASURED temperature has genuine outliers, and
            # `_is_discrete_indicator` below already exempts `weather_coverage`
            # by the CR-02 rule, which is why the coverage flag cannot be
            # clipped into a constant on a single-season build. Missing-handling
            # and outlier-handling have been independent since CR-02 and stay so.
            preserve_this_column = col in preserved_weather_columns
            if preserve_this_column and original_missing > 0:
                missing_stats[col] = original_missing

            # Handle missing data
            if original_missing > 0 and not preserve_this_column:
                # For team-based features, use team's season average
                if any(prefix in col for prefix in ["home_", "away_"]):
                    processed_df[col] = self._impute_team_features(processed_df, col)
                else:
                    # WR-06 surface 1: for game-level features this was
                    # ``processed_df[col].median()`` over the WHOLE frame, so a 2002
                    # gap was filled from a statistic that saw 2025. It is now a
                    # per-season, strictly-prior median.
                    processed_df[col] = self._impute_game_level_features(
                        processed_df, col, season_passes, earliest_season
                    )

                missing_stats[col] = original_missing

            # CR-02: a DISCRETE INDICATOR has no outliers to winsorize, and
            # clipping one destroys the distinction it exists to encode.
            #
            # The defect was found on a rare binary coverage flag. Its guard against
            # median-imputation used to sit inside `if original_missing > 0:` and end
            # in `continue`, so in the normal case -- a builder emitting a row for
            # every game, original_missing == 0 -- the guard never ran and execution
            # fell straight into the unconditional winsorization below.
            #
            # On a single covered season the minority level was far below 1% (271 of
            # 272 games), so q01 == q99 == 1.0 and the clip stamped EVERY minority row
            # with the majority level: a `--season 2023` build produced gold whose
            # flag was a constant 1.0, destroying the column's variance before
            # expanding_normalize saw it. The full-history rebuild happened to be safe
            # (q01 = 0.0 at ~13% minority), which is why the published gold was
            # unaffected and why nothing caught it. It is a GENERAL defect for any
            # rare binary flag -- `saturday_game` is a live candidate, and it is why
            # this exemption outlives the family it was found on.
            #
            # The old shape was also internally inconsistent: the `continue` skipped
            # winsorization entirely whenever the column DID have NaNs, so the same
            # column was winsorized or not depending on whether a gap happened to
            # exist. Missing-handling and outlier-handling are now independent.
            #
            # The test is deliberately a VALUE test, not a name test: any column
            # whose values are all indicator levels is discrete, however it is
            # spelled. A continuous column is NOT exempted merely because it belongs
            # to a family whose flag is -- a totals level or a drift measurement has
            # genuine outliers and stays winsorized.
            #
            # WR-06 note on WHERE this test sits. It is evaluated ONCE per column,
            # over the whole frame, OUTSIDE the per-season loop below. That is a
            # correctness requirement, not tidiness: a column that is {0, 1} overall
            # is discrete and a per-season evaluation would still classify it
            # correctly, but a column that is continuous overall yet happens to be
            # constant at one of -1 / 0 / 1 within a single season would be
            # misclassified as an indicator for that season and would silently
            # escape winsorization there (T-30-29). It is evaluated AFTER the
            # missing-handling above, exactly as it was before this rewrite, so the
            # predicate still sees the already-imputed column its docstring names.
            if self._is_discrete_indicator(processed_df[col]):
                continue

            # Handle outliers with winsorization, per season, on strictly-prior
            # bounds (WR-06 surface 2). The pre-WR-06 shape computed one q01/q99 pair
            # from the whole frame and clipped every row against it.
            #
            # Every season's bound is fitted on a PRE-CLIP SNAPSHOT of the column,
            # taken once here -- after this column's missing-handling, before any
            # season is clipped. Reading the live frame inside the loop instead
            # would make season Y's bound depend on the ALREADY-CLIPPED values of
            # the seasons before it, and for a column whose early history is a
            # neutral constant that cascade is fatal: the first season whose prior
            # slice is dominated by the neutral value gets a degenerate bound, is
            # flattened to that constant, which makes the NEXT season's prior slice
            # even more constant, and the column can never recover even after real
            # data arrives. Measured on the real matrices, the live-frame form
            # destroyed 18 columns outright -- the whole Phase-28 injury
            # availability family and the whole Phase-29 line-movement family.
            # The snapshot is also the statistically correct source: a quantile
            # estimated from already-winsorized data understates its own tail.
            fit_values = self._column(processed_df, col).copy()

            outliers_count = 0
            for season, season_mask, prior_mask in season_passes:
                fit_source, self_fit = self._season_fit_source(
                    fit_values, season, season_mask, prior_mask, earliest_season
                )

                # The pre-WR-06 minimum-data-points condition, now applied to the fit
                # source rather than to the whole column.
                if fit_source.notna().sum() <= self._MIN_FIT_POINTS:
                    continue

                lower_bound = fit_source.quantile(self.outlier_percentiles[0] / 100)
                upper_bound = fit_source.quantile(self.outlier_percentiles[1] / 100)
                if pd.isna(lower_bound) or pd.isna(upper_bound):
                    continue

                # A DEGENERATE bound is not a winsorization, it is an erasure. When
                # the fit source is dominated by one value -- a neutral default over
                # seasons the feature's upstream source does not cover -- q01 and
                # q99 collapse onto that value, and clipping to [c, c] overwrites
                # every genuine observation in the season with c. That is exactly
                # CR-02's finding, generalized from indicator columns to any column
                # with a constant-dominated prehistory: the clip removes no outlier,
                # it removes the feature. A season with no informative prior bound
                # is left unclipped, and the next season -- whose prior slice now
                # contains this season's real spread -- gets a real bound.
                if lower_bound >= upper_bound:
                    continue

                if self_fit:
                    self._record_self_fit(col, season)

                season_values = processed_df.loc[season_mask, col]
                season_outliers = int(
                    (
                        (season_values < lower_bound) | (season_values > upper_bound)
                    ).sum()
                )
                if season_outliers > 0:
                    processed_df.loc[season_mask, col] = season_values.clip(
                        lower=lower_bound, upper=upper_bound
                    )
                    outliers_count += season_outliers

            if outliers_count > 0:
                outlier_stats[col] = outliers_count

        logger.info(
            "Completed missing data and outlier handling",
            missing_imputed=len(missing_stats),
            outliers_winsorized=len(outlier_stats),
            discrete_indicators_exempt=sum(
                1 for c in numeric_cols if self._is_discrete_indicator(processed_df[c])
            ),
            total_features_processed=len(numeric_cols),
            # WR-06: the self-fit flag, made observable in a build log. A season
            # other than the earliest appearing here is a per-column coverage floor
            # -- the column's upstream source simply starts later -- and is a finding
            # worth reading, not an error.
            self_fit_columns=len(self.self_fit_seasons),
            self_fit_seasons=sorted(
                {
                    season
                    for seasons in self.self_fit_seasons.values()
                    for season in seasons
                }
            ),
        )

        return processed_df

    # ------------------------------------------------------------------
    # WR-06 helpers: prior-seasons-only fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _column(df: pd.DataFrame, col: str) -> pd.Series:
        """Return ``df[col]`` narrowed to a Series.

        pandas types ``df[col]`` as ``Series | DataFrame`` because a DUPLICATED
        column label yields a frame. The gold matrices carry no duplicate labels, so
        the frame arm is unreachable here; stating that once beats repeating a cast
        at every call site, and it keeps the WR-06 helpers honestly typed.
        """
        values = df[col]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return values

    @staticmethod
    def _season_passes(df: pd.DataFrame) -> list[tuple]:
        """Return ``[(season, season_mask, strictly_prior_mask), ...]`` ascending.

        Computed once per call and reused across every column, because the masks are
        the expensive part of a per-season pass over two hundred columns.

        A frame with no ``season`` column is never the production path -- the column
        is carried through ``combine_features`` and is required by
        ``_impute_team_features`` -- but some unit frames omit it. Such a frame
        degrades to ONE self-fitting pass over the whole frame, i.e. to the
        pre-WR-06 whole-frame behaviour, rather than to no processing at all.
        """
        if "season" not in df.columns:
            everything = pd.Series(True, index=df.index)
            return [(None, everything, pd.Series(False, index=df.index))]

        seasons = sorted(df["season"].dropna().unique())
        return [
            (season, df["season"] == season, df["season"] < season)
            for season in seasons
        ]

    def _record_self_fit(self, col: str, season) -> None:
        """Record that *col*'s statistic for *season* was fitted on its own rows."""
        if season is None:
            return
        seasons = self.self_fit_seasons.setdefault(col, [])
        if int(season) not in seasons:
            seasons.append(int(season))

    def _season_fit_source(
        self,
        values: pd.Series,
        season,
        season_mask: pd.Series,
        prior_mask: pd.Series,
        earliest_season,
    ) -> tuple[pd.Series, bool]:
        """Return ``(fit_source, self_fit)`` for *season*.

        The fit source is the strictly-prior slice. The earliest data-bearing season
        has none, and a column whose upstream source starts mid-history has an EMPTY
        prior slice in its first populated season -- a naive prior-only rewrite would
        hand both of them NaN bounds or raise (T-30-55). Both cases fall back to the
        season's own rows under the SAME documented flag, so a per-column coverage
        floor is handled by the general rule rather than by a per-family exception.
        """
        if season != earliest_season:
            prior = values.loc[prior_mask]
            if prior.notna().sum() > self._MIN_FIT_POINTS:
                return prior, False
        return values.loc[season_mask], True

    def _impute_game_level_features(
        self,
        df: pd.DataFrame,
        col: str,
        season_passes: list[tuple],
        earliest_season,
    ) -> pd.Series:
        """Fill a game-level column's gaps with a prior-seasons-only median (WR-06).

        Replaces ``df[col].fillna(df[col].median())``, whose median saw every future
        season. A season with no usable fit source at all keeps its NaNs rather than
        borrowing a value from the future: that is the deliberate consequence of the
        fix, not an oversight, and downstream ``expanding_normalize`` already maps an
        un-normalizable position to the neutral 0.0 z-score.

        Like the winsorization pass, the median is fitted on the column as it
        ENTERED this pass -- ``source`` below -- never on the partially-filled
        frame. One rule for both passes, and no season's statistic can depend on
        values another season's fill just wrote.
        """
        source = self._column(df, col)
        result = source.copy()

        for season, season_mask, prior_mask in season_passes:
            season_values = result.loc[season_mask]
            if not season_values.isna().any():
                continue

            fit_source, self_fit = self._season_fit_source(
                source, season, season_mask, prior_mask, earliest_season
            )
            median_value = float(fit_source.median())
            if np.isnan(median_value):
                continue

            if self_fit:
                self._record_self_fit(col, season)
            result.loc[season_mask] = season_values.fillna(median_value)

        return result

    def normalize_combined_features(
        self,
        processed_features: pd.DataFrame,
        target_season: int | None = None,
    ) -> pd.DataFrame:
        """Expanding-window normalization (replaces within-season Z-scores).

        EXTRACTED from ``generate_feature_matrices`` by Plan 33.1-04 Task 2,
        verbatim apart from the new missing-preserving argument. (Spelling that
        argument's name out here would make it look like a third pass site to
        the source scan that counts them, which is exactly the kind of hollow
        hit a scan over prose produces.) The extraction is what makes Ruling
        K1's CONSUMPTION test possible: the
        argument at each real call site can now be captured without driving the
        whole build, which would write gold.

        Args:
            processed_features: The combined matrix, after missing-data and
                outlier handling.
            target_season: Single-season mode when set, batch mode when None.

        Returns:
            The normalized frame.
        """
        # Identifier columns plus the display-only raw_* passthroughs. The
        # display half is DERIVED from utils.feature_columns, the single place
        # those names are stated, so the same list also governs which columns
        # models.temporal keeps out of the MODEL feature set (Plan 30-15 /
        # D30-OWNER-04). A display column added there needs no edit here.
        exclude_cols = normalization_exclude_columns()
        feature_cols = [
            col for col in processed_features.columns if col not in exclude_cols
        ]
        numeric_feature_cols = (
            processed_features[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        # RULING K1: read the entry for the builder that actually ran, ONCE,
        # and hand the SAME entry to both call sites below. Fail-closed: a
        # merged weather frame with no usable entry raises here rather than
        # letting the neutral 0.0 z-score put the stand-in back.
        preserve_by_builder = {
            self.active_builder_key
            or BUILDER_KEYS[0]: self._preserved_weather_columns()
        }
        active_builder = self.active_builder_key or BUILDER_KEYS[0]

        # THE COVERAGE FLAG IS NOT A MEASUREMENT, so it is not z-scored
        # (Plan 33.1-07 Task 4). DERIVED from the set the merged frame actually
        # contributed, never hardcoded: the compressed builder emits no coverage
        # flag at all, so a build over that frame must pass an EMPTY set rather
        # than name a column nothing contributed.
        #
        # WHY THIS EXEMPTION EXISTS, measured rather than argued: silver
        # weather_features carried weather_coverage = 1.0 on all 6,499 rows and
        # gold recorded 0.0 on all 6,499. The expanding std of a constant column
        # is zero, safe_std clips to 1e-8, and (1.0 - 1.0) / 1e-8 is 0.0 -- which
        # is the value features.weather._absent_observation_features writes to
        # mean NO OBSERVATION. The one column whose whole purpose is to tell an
        # absence from a mild day was reporting absence for every game that had
        # a real ERA5 observation behind it.
        #
        # It is the same argument CR-02 already makes about winsorization one
        # stage earlier -- "clipping one destroys the distinction it exists to
        # encode" -- applied to the transform that runs next.
        # THE INTENT IS ASSERTED, NOT MERELY FILTERED FOR (code review WR-10).
        #
        # The filter below is correctly non-WIDEABLE -- it can only ever yield the one
        # named column. Its mirror failure was silent: if weather_coverage ever LEAVES
        # the preserving set (a builder change, a renamed family tuple), the generator
        # yields an EMPTY tuple, expanding_normalize z-scores the flag back to 0.0 on
        # every row, and NOTHING raises. That is the defect this phase spent a rung
        # fixing, restored by omission -- and 0.0 is the code's own word for NO
        # OBSERVATION, so it would be asserting absence for every game with a real ERA5
        # reading behind it.
        #
        # _preserved_weather_columns twenty screens up already refuses BY NAME when its
        # entry is missing for the active builder. This is the same discipline at the
        # one site that consumes it. The compressed builder legitimately contributes no
        # coverage flag, so the refusal is scoped to the FULL builder -- the one that
        # writes gold.
        active_preserved = preserve_by_builder[active_builder]
        if (
            active_builder == BUILDER_KEYS[0]
            and WEATHER_COVERAGE_COLUMN not in active_preserved
        ):
            msg = (
                f"{WEATHER_COVERAGE_COLUMN!r} is not in the {active_builder!r} builder's "
                f"preserving set ({sorted(active_preserved)}), so the level exemption "
                "would be an EMPTY tuple and the coverage flag would be z-scored back to "
                "0.0 -- the value that means NO OBSERVATION -- on every row. Refusing "
                "rather than silently rebuilding gold with the defect this phase removed."
            )
            raise ValueError(msg)
        # THE ONE-COLUMN FILTER BECAME A PREDICATE (Plan 33-14 Task 3, D33-34(a)).
        #
        # It read `tuple(c for c in active_preserved if c == WEATHER_COVERAGE_COLUMN)`
        # -- correctly non-wideable, and one column wide. The owner ruled on
        # 2026-09-14 that .planning/WINDOWS.md rows 33 and 40 are TAKEN before
        # Wave 15's re-fit rather than carried past it, so the exemption now
        # resolves the nineteen weather indicator flags and the six sibling
        # `*_coverage` columns alongside the flag it already carried.
        #
        # THE REFUSAL ABOVE IS DELIBERATELY UNCHANGED. It fires on the PRESERVING
        # SET losing the coverage flag, which is still a real defect and still
        # worth refusing on, and it fires BEFORE this line.
        #
        # See `_level_preserved_columns` for why the two arms carry different
        # rules, and why the value arm requires more than one level.
        preserve_level_cols = self._level_preserved_columns(
            processed_features, active_preserved
        )
        logger.info(
            "Resolved the level-preserved columns",
            builder=active_builder,
            preserved_levels=len(preserve_level_cols),
            columns=list(preserve_level_cols),
        )

        # Compute prior-season stats for bootstrap and normalize
        if target_season:
            # Single-season mode: compute prior stats once
            prior_stats = compute_prior_season_stats(
                processed_features, numeric_feature_cols, target_season - 1
            )
            return expanding_normalize(
                processed_features,
                feature_cols=numeric_feature_cols,
                group_col="season",
                sort_cols=["season", "week"],
                min_periods=4,
                prior_season_stats=prior_stats,
                preserve_missing_cols=preserve_by_builder[active_builder],
                preserve_level_cols=preserve_level_cols,
            )

        # Batch mode: compute prior-season stats per season
        seasons = sorted(processed_features["season"].unique())
        normalized_parts = []
        for s in seasons:
            season_df = processed_features[processed_features["season"] == s].copy()
            prior_stats = compute_prior_season_stats(
                processed_features, numeric_feature_cols, s - 1
            )
            norm_part = expanding_normalize(
                season_df,
                feature_cols=numeric_feature_cols,
                group_col="season",
                sort_cols=["season", "week"],
                min_periods=4,
                prior_season_stats=prior_stats,
                preserve_missing_cols=preserve_by_builder[active_builder],
                preserve_level_cols=preserve_level_cols,
            )
            normalized_parts.append(norm_part)
        return pd.concat(normalized_parts, ignore_index=False)

    @staticmethod
    def _is_discrete_indicator(series: pd.Series) -> bool:
        """True when a column's values are indicator levels, not measurements.

        CR-02. Winsorization answers "is this value an implausible extreme of a
        continuous distribution". That question is meaningless for a column whose
        values are drawn from {-1, 0, 1} -- a coverage flag, a sign, a boolean
        game-context marker. For such a column the 1st and 99th percentiles are
        simply the modal level whenever the minority level is rarer than 1%, so
        the clip does not remove an outlier: it OVERWRITES every minority row with
        the majority level and leaves a constant.

        That is how ``--season 2023`` produced gold whose
        ``line_movement_coverage`` was a constant 1.0, encoding "we measured this
        game's trajectory" for games that have none -- the exact conflation the
        flag exists to prevent.

        Args:
            series: The (already imputed) numeric feature column.

        Returns:
            True iff every non-null value is one of -1.0, 0.0 or 1.0.
        """
        values = pd.unique(series.dropna())
        return len(values) > 0 and set(values.tolist()) <= {-1.0, 0.0, 1.0}

    # ------------------------------------------------------------------
    # THE LEVEL-PRESERVATION PREDICATE (Plan 33-14 Task 3, D33-34(a)).
    #
    # It replaces a filter that could only ever yield ONE column --
    # `weather_coverage` -- with a predicate over the frame's own columns,
    # closing .planning/WINDOWS.md rows 33 and 40 against this phase.
    #
    # WHAT ROWS 33 AND 40 REGISTER, measured rather than argued. Nineteen weather
    # yes/no flags reach gold z-scored into many distinct decimals -- `is_snow`
    # carries 274 distinct values, `wind_moderate` 5,667 -- so the same snowy game
    # gets a different number in week 3 than in week 15, and the level that WAS
    # the meaning is gone. Six sibling `*_coverage` flags carry the mirror defect:
    # `home_injury_coverage` reads a constant 0.0 across 2002-2008, which are
    # genuinely UNCOVERED, while 2009-2024 range -15.97 to +0.207 -- so an
    # uncovered row reads a value far CLOSER to the covered level than to the
    # uncovered one.
    #
    # A WIDENING PREDICATE IS THE DANGEROUS DIRECTION, and it is split into two
    # arms with DIFFERENT rules for that reason. A predicate that widens a
    # REFUSAL fails safe; this one widens an EXEMPTION, so a column it wrongly
    # selects is a column that silently stops being normalized and reaches the
    # re-fit as a raw level.
    #
    #   * The SUFFIX arm is NAME-based and has no varying-levels requirement. A
    #     `*_coverage` column's levels are its meaning by construction, and the
    #     canonical case is a CONSTANT one.
    #   * The WEATHER arm is VALUE-based and DOES require at least two distinct
    #     non-null levels. Without that clause the predicate selects three columns
    #     nobody declared -- `precip_prob`, `raw_precip_prob` and
    #     `extreme_weather` -- each of which carries exactly ONE non-null value on
    #     today's corpus and is therefore "discrete" only by accident of the data.
    #     Two of the three are continuous PROBABILITIES: exempting them would be
    #     threat T-33-81 realised, and worse, the exemption would FLICKER between
    #     generations, since a 2026 live forecast supplying real probabilities
    #     makes the column continuous again and silently drops it back out.
    #
    # The resolved set is asserted EQUAL to
    # `tests.phase33_state.GOLD_LEVEL_PRESERVED_COLUMNS_33_14` in BOTH directions
    # against live gold -- an unexpected member and a missing member both fail --
    # so a widening exemption cannot quietly exempt a column nobody declared.
    # ------------------------------------------------------------------

    @classmethod
    def _is_varying_discrete_indicator(cls, series: pd.Series) -> bool:
        """True when *series* is a discrete indicator carrying MORE THAN ONE level.

        COMPOSED on ``_is_discrete_indicator`` rather than restating its
        ``{-1, 0, 1}`` rule, which would be the D30-02 second-list failure mode
        inside a single class.

        Args:
            series: A pre-normalization numeric feature column.

        Returns:
            True iff every non-null value is an indicator level AND at least two
            distinct non-null values are present.
        """
        return cls._is_discrete_indicator(series) and series.dropna().nunique() >= 2

    @staticmethod
    def _level_preserved_suffix_columns(columns) -> tuple[str, ...]:
        """The NAME arm: every column whose name ends with the coverage suffix.

        Args:
            columns: Any iterable of column names.

        Returns:
            The matching names, sorted and de-duplicated.
        """
        return tuple(
            sorted(
                {
                    str(column)
                    for column in columns
                    if str(column).endswith(LEVEL_PRESERVED_COLUMN_SUFFIX)
                }
            )
        )

    @classmethod
    def _level_preserved_indicator_columns(
        cls, frame: pd.DataFrame, active_preserved
    ) -> tuple[str, ...]:
        """The VALUE arm: the active builder's weather flags, judged on *frame*.

        Args:
            frame: The PRE-NORMALIZATION frame. Discreteness is a property of the
                values as they enter normalization; judging it on gold would be
                meaningless, because normalization is what destroys it.
            active_preserved: The active builder's preserved weather column set.

        Returns:
            The matching names, sorted and de-duplicated.
        """
        return tuple(
            sorted(
                {
                    column
                    for column in active_preserved
                    if column in frame.columns
                    and cls._is_varying_discrete_indicator(frame[column])
                }
            )
        )

    @classmethod
    def _level_preserved_columns(
        cls, frame: pd.DataFrame, active_preserved
    ) -> tuple[str, ...]:
        """The columns returned at their RECORDED LEVEL rather than z-scored.

        A ``classmethod`` rather than an instance method on purpose: it needs no
        builder state, and ``scripts.fingerprint_gold.phase33_level_preserved_family``
        resolves the rung's declared family through THESE SAME two arms rather
        than through a second hand-written list.

        Args:
            frame: The pre-normalization frame.
            active_preserved: The active builder's preserved weather column set.

        Returns:
            The union of the two arms, sorted and de-duplicated.
        """
        return tuple(
            sorted(
                set(cls._level_preserved_suffix_columns(frame.columns))
                | set(cls._level_preserved_indicator_columns(frame, active_preserved))
            )
        )

    def _impute_team_features(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Impute missing team features using team's season average.

        WR-14: the two ``col.replace("home_", "")`` / ``col.replace("away_", "")``
        statements that used to sit here were no-ops -- ``str`` is immutable and
        the results were discarded -- so they looked like they computed a base
        column name and did not. Nothing downstream ever needed one: the imputation
        works on ``col`` itself and only needs to know WHICH team column to group
        by. The dead lines are gone rather than "fixed", because there was no bug
        to fix, only a false suggestion that a base name was in play.

        WR-06 SURFACE 3, named by neither the SPEC nor CONTEXT (T-30-28). Both
        last-resort fallbacks below used to be ``df[col].median()`` over the WHOLE
        frame, and this is the branch every ``home_*`` / ``away_*`` column takes --
        the large majority of features. Any 2021-2024 row reaching either of them
        WOULD move when the N-01 re-sync adds 2025 rows, failing SPEC R2's
        byte-identity control for a cause unrelated to the two surfaces D30-16
        names. Both now fit on the strictly-prior seasons.

        The two WITHIN-SEASON statistics -- the team mean and the season mean -- are
        DELIBERATELY left exactly as they are. Neither can be moved by adding a
        later season's rows, so neither threatens SPEC R2, and converting them would
        be a larger behavioural change than D30-16 authorises in the
        highest-blast-radius file in this phase.
        """
        season_passes = self._season_passes(df)
        earliest_season = season_passes[0][0] if season_passes else None

        if col.startswith("home_"):
            team_col = "home_team"
        elif col.startswith("away_"):
            team_col = "away_team"
        else:
            # Not a team feature. The dispatch in handle_missing_data_and_outliers
            # routes on ``"home_" in col`` while this method routes on
            # ``col.startswith("home_")``, so a column carrying the substring
            # anywhere but the front lands here. WR-06 surface 3, first site.
            return self._impute_game_level_features(
                df, col, season_passes, earliest_season
            )

        result = self._column(df, col).copy()

        # For each team with missing data, use their season average
        for season, season_mask, prior_mask in season_passes:
            season_data = df.loc[season_mask]
            # Computed LAZILY, and only when the two within-season statistics have
            # both come back NaN: computing it eagerly would record a self-fit for
            # every season of a late-arriving column, whose prior slices are empty
            # but whose fallback is never actually reached.
            prior_median = None
            prior_median_computed = False

            for team in season_data[team_col].unique():
                team_mask = season_mask & (df[team_col] == team)
                team_values = df.loc[team_mask, col]

                if team_values.isna().any():
                    team_mean = team_values.mean()
                    if pd.isna(team_mean):
                        # Use season average if team has no data
                        team_mean = season_data[col].mean()
                    if pd.isna(team_mean):
                        # WR-06 surface 3, second site: the last resort is the
                        # median of the seasons STRICTLY BEFORE this one, not of
                        # the whole frame.
                        if not prior_median_computed:
                            prior_median = self._prior_season_median(
                                self._column(df, col),
                                col,
                                season,
                                season_mask,
                                prior_mask,
                                earliest_season,
                            )
                            prior_median_computed = True
                        if prior_median is None:
                            # Nothing at or before this season can fill the
                            # gap. Leave it NaN rather than borrow from the
                            # future -- the deliberate consequence of WR-06.
                            continue
                        team_mean = prior_median

                    result.loc[team_mask & result.isna()] = team_mean

        return result

    def _prior_season_median(
        self,
        values: pd.Series,
        col: str,
        season,
        season_mask: pd.Series,
        prior_mask: pd.Series,
        earliest_season,
    ) -> float | None:
        """Return the strictly-prior-seasons median for *col* in *season* (WR-06)."""
        fit_source, self_fit = self._season_fit_source(
            values, season, season_mask, prior_mask, earliest_season
        )
        median_value = float(fit_source.median())
        if np.isnan(median_value):
            return None
        if self_fit:
            self._record_self_fit(col, season)
        return float(median_value)

    def normalize_features_within_seasons(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Apply Z-score normalization within seasons.

        .. deprecated::
            This method uses full-season mean/std which leaks future data.
            Use ``expanding_normalize`` from ``features.normalization`` instead.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from normalization

        Returns:
            Normalized feature matrix
        """
        warnings.warn(
            "normalize_features_within_seasons is deprecated due to future data leakage. "
            "Use expanding_normalize from features.normalization instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info("Normalizing features within seasons")

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        normalized_df = features_df.copy()

        # Get numeric feature columns to normalize
        feature_cols = [
            col for col in normalized_df.columns if col not in target_columns
        ]
        numeric_cols = (
            normalized_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        normalization_stats = {}

        # Normalize within each season
        for season in normalized_df["season"].unique():
            season_mask = normalized_df["season"] == season
            season_data = normalized_df.loc[season_mask]

            if len(season_data) < self.min_games_for_stats:
                logger.warning(
                    "Insufficient data for normalization",
                    season=season,
                    games=len(season_data),
                )
                continue

            season_normalized_cols = 0

            for col in numeric_cols:
                col_data = season_data[col]

                if col_data.notna().sum() >= self.min_games_for_stats:
                    mean_val = col_data.mean()
                    std_val = col_data.std()

                    if std_val > 0:  # Avoid division by zero
                        normalized_df.loc[season_mask, col] = (
                            col_data - mean_val
                        ) / std_val
                        season_normalized_cols += 1

            normalization_stats[season] = season_normalized_cols

        logger.info(
            "Completed within-season normalization",
            seasons_processed=len(normalization_stats),
            avg_features_per_season=np.mean(list(normalization_stats.values())),
        )

        return normalized_df

    def create_target_variables(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for WP, ATS, and O/U prediction.

        Args:
            features_df: Feature matrix with game scores

        Returns:
            Feature matrix with target variables added
        """
        logger.info("Creating target variables")

        target_df = features_df.copy()

        # Only create targets if we have scores
        if (
            "home_score" not in target_df.columns
            or "away_score" not in target_df.columns
        ):
            logger.warning("No score data available for target creation")
            return target_df

        # Convert score columns to numeric (they may be stored as strings)
        target_df["home_score"] = pd.to_numeric(
            target_df["home_score"], errors="coerce"
        )
        target_df["away_score"] = pd.to_numeric(
            target_df["away_score"], errors="coerce"
        )

        # Remove games with missing scores
        score_mask = target_df["home_score"].notna() & target_df["away_score"].notna()
        target_df = target_df[score_mask].copy()

        # Win Probability target (1 = home win, 0 = away win)
        target_df["target_wp"] = (
            target_df["home_score"] > target_df["away_score"]
        ).astype(int)
        # Trainer-expected column: strict binary (ties = 0, same as trainer derivation)
        target_df["home_win"] = target_df["target_wp"].copy()

        # Handle ties (rare in NFL)
        ties = target_df["home_score"] == target_df["away_score"]
        if ties.sum() > 0:
            logger.info("Found tied games", count=ties.sum())
            # For WP regression target, ties as 0.5
            target_df.loc[ties, "target_wp"] = 0.5
            # For classifier target, ties stay 0 (not a home win)

        # Point differential (for ATS calculation if spreads available)
        target_df["point_differential"] = (
            target_df["home_score"] - target_df["away_score"]
        )
        # Trainer-expected column (alias for ATS regression target)
        target_df["home_margin"] = target_df["point_differential"].copy()

        # Total points (for O/U calculation if totals available)
        target_df["total_points"] = target_df["home_score"] + target_df["away_score"]

        # ATS target (requires market data)
        if "snapshot_spread" in target_df.columns:
            # ATS = actual margin - spread (positive = home team covered)
            target_df["target_ats"] = (
                target_df["point_differential"] - target_df["snapshot_spread"]
            )
            target_df["home_covered_spread"] = (target_df["target_ats"] > 0).astype(int)

        # O/U target (requires market data)
        if "snapshot_total" in target_df.columns:
            # O/U = actual total - market total (positive = over)
            target_df["target_ou"] = (
                target_df["total_points"] - target_df["snapshot_total"]
            )
            target_df["game_went_over"] = (target_df["target_ou"] > 0).astype(int)

        logger.info(
            "Created target variables",
            wp_targets=target_df["target_wp"].notna().sum(),
            ats_targets=target_df.get("target_ats", pd.Series()).notna().sum(),
            ou_targets=target_df.get("target_ou", pd.Series()).notna().sum(),
        )

        return target_df

    def generate_feature_matrices(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate complete feature matrices for all prediction targets.

        Pipeline flow:
        1. Load feature sources, then build ONE per-game lock frame
        2. Stage 1: the information-time gate -- each registered source's
           per-game information time against that game's own lock
        3. Combine features (then the gold dtype guard)
        4. Stage 2: LeakageGate.validate_combined_matrix
        5. Handle missing data and outliers
        6. Expanding-window normalization (replaces within-season Z-scores)
        7. Create target variables
        8. Split into per-target matrices

        Args:
            target_season: Specific season to process.
            target_week: Specific week to process.
            as_of_datetime: The cutoff handed to the builders that still take an
                ``as_of_datetime`` parameter. It is NOT the fence: the fence is
                the per-game lock frame. Defaults to ``datetime.now(ET)`` --
                tz-AWARE, see the CR-01 note on ``load_all_feature_sources``.

        Returns:
            Dictionary with feature matrices for each target.
        """
        # THIS DEFAULT IS NO LONGER A FENCE (Phase 33.2, D33.2-01). The fence is the
        # per-game lock frame built below; nothing in Stage 1 reads this value. It
        # survives ONLY as the ``as_of_datetime`` argument the builders still take
        # (QBTracker, OpponentAdjuster, ...), whose cutoffs move onto the lock in
        # Plans 33.2-12 / 33.2-13. CR-01 still applies to it: tz-aware ET, never a
        # naive local clock.
        if as_of_datetime is None:
            as_of_datetime = datetime.now(ET)

        logger.info(
            "Generating feature matrices",
            target_season=target_season,
            target_week=target_week,
            as_of_datetime=str(as_of_datetime),
        )

        try:
            # Load all feature sources
            feature_sources = self.load_all_feature_sources(
                target_season, target_week, as_of_datetime=as_of_datetime
            )

            # -- ONE lock frame (game_id -> tz-aware lock), before any per-source
            #    work (RESEARCH 3.3). A game with no kickoff has no lock and the
            #    whole build refuses, naming it.
            lock_frame = build_lock_frame(feature_sources["games"])

            # -- Stage 1: the information-time gate (SPEC R2, D33.2-01) --
            # The kickoff-versus-now loop that stood here is DELETED, not disabled.
            self._check_information_times(
                feature_sources, lock_frame, target_season, target_week
            )

            # Combine features
            combined_features = self.combine_features(feature_sources)

            # -- The sidecar is never a column: no datetime-typed or
            #    provenance-named column may enter a gold matrix.
            refuse_provenance_columns(
                combined_features, build_clock_columns=BUILD_CLOCK_COLUMNS
            )

            if len(combined_features) == 0:
                logger.error("No features to process")
                return {}

            # -- SPEC R3: the line-movement family must not reach gold --
            # Runs on the COMBINED matrix, before every downstream stage and
            # therefore long before the gold write.
            combined_features = self._enforce_line_movement_dropped(combined_features)

            # -- Replace raw EPA with opponent-adjusted EPA --
            # OpponentAdjuster needs per-game stats (with game_id, raw EPA),
            # not the rolling averages from the silver table.
            games_df = feature_sources["games"]
            try:
                # THE SEASON POOL COMES FROM THE FRAME ABOVE (Plan 33.1-07
                # Task 4). It used to come from a hardcoded range(2018, 2025)
                # inside the calculator, which stopped at 2024 and left season
                # 2025's twelve opponent-adjusted columns carrying 2 distinct
                # values across 285 games. The caller already holds the games
                # frame, so it names the coverage rather than making the
                # calculator re-derive the same fact from a second store read.
                per_game_stats = self.team_form_calc.get_per_game_stats(
                    as_of_datetime,
                    target_season=target_season,
                    seasons=None
                    if target_season is not None
                    else games_df["season"].dropna().tolist(),
                )
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                logger.warning(
                    "Failed to get per-game stats for opponent adjustment",
                    error=str(e),
                )
                per_game_stats = pd.DataFrame()
            if len(per_game_stats) > 0:
                try:
                    adjusted_df = self.opponent_adj.build_features(
                        games_df,
                        as_of_datetime,
                        target_season=target_season,
                        target_week=target_week,
                        team_game_stats=per_game_stats,
                    )

                    if len(adjusted_df) > 0:
                        # Merge opponent-adjusted features using same _get_team_features pattern
                        for prefix, team_col in [
                            ("home", "home_team"),
                            ("away", "away_team"),
                        ]:
                            adj_team = self._get_team_features(
                                adjusted_df, combined_features, team_col, prefix
                            )
                            # Only keep the rolling_opp_adj_* columns (not duplicate other rolling cols)
                            opp_adj_cols = [
                                c for c in adj_team.columns if "opp_adj" in c
                            ]
                            if opp_adj_cols:
                                adj_team = adj_team[["game_id", *opp_adj_cols]]
                                combined_features = combined_features.merge(
                                    adj_team, on="game_id", how="left"
                                )

                        # Drop old raw EPA columns that are now replaced by opp_adj versions
                        raw_epa_suffixes = [
                            "rolling_epa_per_play",
                            "rolling_pass_epa_per_play",
                            "rolling_rush_epa_per_play",
                        ]
                        cols_to_drop = []
                        for pfx in ["home", "away"]:
                            for side in ["off", "def"]:
                                for suffix in raw_epa_suffixes:
                                    col_name = f"{pfx}_{side}_{suffix}"
                                    if col_name in combined_features.columns:
                                        cols_to_drop.append(col_name)

                        if cols_to_drop:
                            combined_features = combined_features.drop(
                                columns=cols_to_drop
                            )
                            logger.info(
                                "Replaced raw EPA with opponent-adjusted EPA",
                                dropped_columns=cols_to_drop,
                                n_dropped=len(cols_to_drop),
                            )
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(
                        "Failed to apply opponent adjustment, keeping raw EPA",
                        error=str(e),
                    )

            # -- Stage 2: Combined matrix validation --
            try:
                self.leakage_gate.validate_combined_matrix(
                    combined_features, as_of_datetime
                )
            except LeakageViolation as e:
                self.leakage_gate.write_diagnostic_report(e)
                raise

            # Handle missing data and outliers
            processed_features = self.handle_missing_data_and_outliers(
                combined_features
            )

            # Preserve the un-normalized composite severity for display before
            # normalization runs (mirrors the raw_wind_mph / raw_precip_mm
            # family). The normalized weather_severity_score stays the model
            # feature; this raw sibling is excluded below so it is never
            # z-scored and the cache can surface a human-meaningful value.
            processed_features["raw_weather_severity"] = processed_features[
                "weather_severity_score"
            ]

            normalized_features = self.normalize_combined_features(
                processed_features, target_season=target_season
            )

            # Create target variables
            final_features = self.create_target_variables(normalized_features)

            # Generate separate matrices for each prediction target
            feature_matrices = {}

            # Base feature columns (exclude identifiers and targets)
            exclude_cols = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
                "target_wp",
                "target_ats",
                "target_ou",
                "home_win",
                "home_margin",
                "point_differential",
                "total_points",
                "home_covered_spread",
                "game_went_over",
            ]

            feature_cols = [
                col for col in final_features.columns if col not in exclude_cols
            ]

            # Metadata columns to carry through for target derivation
            meta_cols = ["game_id", "season", "week", "feature_timestamp"]
            score_cols = [
                c for c in ["home_score", "away_score"] if c in final_features.columns
            ]

            # Win Probability matrix
            wp_target_cols = [
                c for c in ["target_wp", "home_win"] if c in final_features.columns
            ]
            wp_matrix = final_features[
                [*meta_cols, *score_cols, *wp_target_cols, *feature_cols]
            ].copy()
            feature_matrices["wp"] = wp_matrix

            # ATS matrix (only include games with spread data)
            if "target_ats" in final_features.columns:
                ats_target_cols = [
                    c
                    for c in ["target_ats", "home_margin", "point_differential"]
                    if c in final_features.columns
                ]
                ats_games = final_features["target_ats"].notna()
                ats_matrix = final_features.loc[
                    ats_games,
                    [*meta_cols, *score_cols, *ats_target_cols, *feature_cols],
                ].copy()
                feature_matrices["ats"] = ats_matrix

            # O/U matrix (only include games with total data)
            if "target_ou" in final_features.columns:
                ou_target_cols = [
                    c
                    for c in ["target_ou", "total_points"]
                    if c in final_features.columns
                ]
                ou_games = final_features["target_ou"].notna()
                ou_matrix = final_features.loc[
                    ou_games,
                    [*meta_cols, *score_cols, *ou_target_cols, *feature_cols],
                ].copy()
                feature_matrices["ou"] = ou_matrix

            # The opponent-adjusted family and the targets join AFTER the guard
            # above, so each final matrix is guarded again before it is returned.
            for matrix in feature_matrices.values():
                refuse_provenance_columns(
                    matrix, build_clock_columns=BUILD_CLOCK_COLUMNS
                )

            logger.info(
                "Generated feature matrices",
                wp_games=len(feature_matrices.get("wp", [])),
                ats_games=len(feature_matrices.get("ats", [])),
                ou_games=len(feature_matrices.get("ou", [])),
                total_features=len(feature_cols),
            )

            return feature_matrices

        except _SOURCE_LOAD_ERRORS as e:
            logger.error("Failed to generate feature matrices", error=str(e))
            raise

    @staticmethod
    def _reject_narrowing_incremental_write(
        table_name: str, matrix_df: pd.DataFrame
    ) -> None:
        """Refuse an incremental gold write that would resurrect dropped columns.

        The incremental path concatenates the incoming slice onto the rows of the
        existing table it does not replace, and ``pd.concat`` UNIONS columns. So a
        slice built under a NARROWER schema than the one on disk would write the
        removed columns back, all-null for every retained historical row -- the
        exact shape ``check_gold_integrity``'s all-null check fails on, for a
        reason that looks nothing like the cause.

        A schema change is a full-rebuild operation. Refusing here is a stop
        BEFORE the write, not a diagnosis afterwards.

        An absent or unreadable gold table is not an error: there is no history to
        protect and the append path will simply create the table.
        """
        try:
            existing = load_dataframe(table_name, layer="gold")
        except (DataIngestionError, FileNotFoundError, OSError) as e:
            logger.info(
                "No existing gold table to reconcile against; incremental write "
                "will create it",
                table_name=table_name,
                error=str(e),
            )
            return

        if existing.empty:
            return

        incoming_cols = set(matrix_df.columns)
        existing_cols = set(existing.columns)
        if incoming_cols == existing_cols:
            return

        msg = (
            f"Refusing an incremental gold write for '{table_name}': the incoming "
            f"frame has {len(incoming_cols)} columns against {len(existing_cols)} "
            f"on disk (missing here: {sorted(existing_cols - incoming_cols)}; new "
            f"here: {sorted(incoming_cols - existing_cols)}). An incremental write "
            "concatenates, and pd.concat UNIONS columns, so this would resurrect "
            "dropped columns as all-null across all retained history. Run a FULL "
            "rebuild (no --season / --week) to change the gold schema."
        )
        raise ValueError(msg)

    def save_feature_matrices(
        self,
        feature_matrices: dict[str, pd.DataFrame],
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> None:
        """
        Save feature matrices to gold layer.

        The WRITE MODE follows the BUILD'S SCOPE, and the two modes are not
        interchangeable (CR-01):

        * FULL REBUILD (``target_season is None and target_week is None``) --
          the build carried every season, so the frame in hand IS the table.
          Written with ``replace_mode=True``: a single self-contained Parquet
          file, no directory partitioning, and no append/dedup merge. This is
          the only mode that can NARROW a schema, which is what the Phase-30
          rung-3 column drop needed (SPEC R3, T-30-05, D-10). A repeated full
          rebuild is idempotent.

        * INCREMENTAL (a ``--season`` and/or ``--week`` build) -- the build
          carried only that slice. Written with ``replace_mode=False``, i.e.
          ``save_dataframe``'s append path, which drops the incoming
          ``game_id``s from the existing table and concatenates: latest-wins
          on ``game_id``, full history preserved.

        Replace mode on an incremental build would write the slice AS the whole
        table in BOTH stores -- ``create_table_from_df(if_exists="replace")``
        and ``pm.save`` -- destroying every season the slice did not carry. It
        is therefore selected from the scope rather than passed unconditionally.

        Because the incremental path concatenates, and ``pd.concat`` UNIONS
        columns, a NARROWING incremental write would silently resurrect dropped
        columns as all-null. That case is refused rather than guessed at: it
        needs a full rebuild.

        Args:
            feature_matrices: Dictionary with feature matrices
            target_season: Season the matrices were built for. Not None means
                the build is scoped, so the write merges instead of replacing.
            target_week: Week the matrices were built for. Same effect.
        """
        full_rebuild = target_season is None and target_week is None
        logger.info(
            "Saving feature matrices to gold layer",
            target_season=target_season,
            target_week=target_week,
            full_rebuild=full_rebuild,
            write_mode="replace" if full_rebuild else "merge",
        )

        for target, matrix_df in feature_matrices.items():
            if len(matrix_df) == 0:
                logger.warning("Empty feature matrix", target=target)
                continue

            # Coerce non-numeric feature columns (string artifacts from append)
            non_meta = [
                c
                for c in matrix_df.columns
                if c
                not in [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "feature_timestamp",
                    "weather_condition",
                ]
            ]
            for col in non_meta:
                if matrix_df[col].dtype == object:
                    matrix_df[col] = pd.to_numeric(matrix_df[col], errors="coerce")

            # Drop non-numeric string columns that aren't features
            if "weather_condition" in matrix_df.columns:
                matrix_df = matrix_df.drop(columns=["weather_condition"])

            # Drop columns that are 100% NaN (offense-only metrics on defense side)
            all_nan_cols = [
                c
                for c in matrix_df.columns
                if matrix_df[c].isna().all() and c not in ["game_id"]
            ]
            if all_nan_cols:
                matrix_df = matrix_df.drop(columns=all_nan_cols)
                logger.info(
                    "Dropped all-NaN columns",
                    target=target,
                    dropped=all_nan_cols,
                )

            table_name = f"features_{target}"

            # Write the gold matrix as a single self-contained file (no
            # directory partitioning). The gold matrices are one-row-per-game
            # tables; carrying game_id, they dedup cleanly with latest-wins.
            #
            # The prior partition_cols=["season"] if target_season else None
            # re-introduced the shared-root partitioned-append antipattern that
            # 25c364f eradicated from every silver builder: pq.write_to_dataset
            # writes season=YYYY/ partition directories into the SHARED
            # data/gold/ root, where all three matrices (features_wp/ats/ou)
            # collide in the same season=YYYY/ directory and each current-week
            # run appends a NEW hash-named parquet instead of overwriting --
            # silently multiplying gold cardinality and cross-contaminating the
            # three matrices.
            #
            # replace_mode on a FULL REBUILD: the passed frame IS the table
            # (SPEC R3, T-30-05).
            #
            # Removing partition_cols alone left save_dataframe's DEFAULT
            # append_mode=True path, which reads the existing gold table, drops the
            # rows whose game_id appears in the new frame -- on a full rebuild that
            # is ALL of them, leaving an empty but still full-width frame -- and
            # then concatenates. A pd.concat UNIONS columns even when one operand
            # has zero rows. Every previous rebuild in this project only ADDED
            # columns, where that union is a harmless no-op, which is why the append
            # path has never misbehaved. The Phase-30 rung-3 drop is the first
            # rebuild that REMOVES columns, and under append mode its 194-column
            # in-memory frame would have been written back 209 columns wide with the
            # fifteen dropped columns present and entirely NULL -- failing
            # check_gold_integrity's all-null check for a reason that looks nothing
            # like the actual cause.
            #
            # Replace mode makes a repeated full rebuild idempotent, which is what
            # SPEC R1's byte-identical re-run acceptance needs. The partitioned-append
            # antipattern above stays excluded in BOTH modes: replace_mode forces
            # partition_cols to None, and the merge branch below passes none either.
            # (CR-01, D-10)
            #
            # It is NOT applied to a scoped build. `--season 2025` produces a
            # 285-row, 2025-only matrix, and replace mode would write that AS the
            # gold table in both DuckDB and parquet, destroying 2002-2024. A scoped
            # build merges instead: latest-wins on game_id, full history preserved.
            if not full_rebuild:
                self._reject_narrowing_incremental_write(table_name, matrix_df)

            save_dataframe(
                matrix_df,
                table_name=table_name,
                layer="gold",
                replace_mode=full_rebuild,
            )

            logger.info(
                "Saved feature matrix",
                target=target,
                table_name=table_name,
                records=len(matrix_df),
                features=len(matrix_df.columns) - 5,
            )  # Exclude metadata columns


def main():
    """Build unified feature matrices."""
    parser = argparse.ArgumentParser(description="Build unified feature matrices")
    # --all-seasons is the EXPLICIT name for the full historical rebuild, which is
    # also what a bare invocation does. It exists so the destructive mode can be
    # ASKED FOR by name rather than reached by omission: it is the mode that writes
    # with replace_mode=True, i.e. the frame in hand becomes the gold table.
    # RUNBOOK.md section 11's attributed-rebuild procedure publishes this flag
    # (WR-09), and before this it did not exist -- argparse rejected the documented
    # command with exit 2, and the obvious "correction" an operator would reach for
    # was --season <YEAR>, which is the SCOPED build (CR-01).
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    scope_group.add_argument(
        "--all-seasons",
        action="store_true",
        help=(
            "Full historical rebuild over every season (the default when no "
            "--season is given). This is the only mode that REPLACES the gold "
            "tables, and so the only mode that can change the gold schema."
        ),
    )
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--as-of",
        type=str,
        help="As-of datetime (ISO format) for time-fence enforcement",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save feature matrices to gold layer (default: on)",
    )
    # The bare --save flag was a no-op (store_true with default=True can never
    # turn saving OFF). --no-save is the real toggle for a read-only build.
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Build the feature matrices without writing them to the gold layer",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate feature matrices after building",
    )

    args = parser.parse_args()

    # --all-seasons names the FULL rebuild, so it cannot also carry a week scope.
    # argparse's mutually-exclusive group already rejects --all-seasons --season.
    if args.all_seasons and args.week is not None:
        parser.error(
            "--all-seasons is the full historical rebuild and cannot be combined "
            "with --week. Use --season <YEAR> --week <N> for a scoped build."
        )

    # Parse --as-of datetime if provided.
    #
    # CR-01: a bare ISO string on the command line means ET -- the freeze this
    # project fences on is "Friday 6 PM ET", and an operator typing
    # `--as-of 2024-10-04T18:00:00` means 18:00 ET, not 18:00 UTC. Attaching ET
    # here makes the value tz-AWARE, so `ensure_utc_aware` downstream CONVERTS it
    # (22:00Z) instead of re-labelling it (18:00Z == 14:00 ET, four hours early).
    # An explicit offset in the string is honoured as given.
    as_of_dt = None
    if args.as_of:
        as_of_dt = datetime.fromisoformat(args.as_of)
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=ET)

    logger.info("Building unified feature matrices", season=args.season, week=args.week)

    try:
        # Initialize feature matrix builder
        builder = FeatureMatrixBuilder()

        # Generate feature matrices
        feature_matrices = builder.generate_feature_matrices(
            target_season=args.season,
            target_week=args.week,
            as_of_datetime=as_of_dt,
        )

        if not feature_matrices:
            logger.warning("No feature matrices generated")
            return

        # Display summary
        print("\nFeature matrices summary:")
        print("=" * 60)

        total_features = 0
        for target, matrix_df in feature_matrices.items():
            feature_count = len(matrix_df.columns) - 5  # Exclude metadata
            total_features = feature_count  # They should all have same feature count

            print(f"{target.upper()} Matrix:")
            print(f"  Games: {len(matrix_df)}")
            print(f"  Features: {feature_count}")

            if len(matrix_df) > 0:
                # Show target distribution
                target_col = f"target_{target}"
                if target_col in matrix_df.columns:
                    if target == "wp":
                        home_wins = (matrix_df[target_col] == 1).sum()
                        away_wins = (matrix_df[target_col] == 0).sum()
                        ties = (matrix_df[target_col] == 0.5).sum()
                        print(
                            f"  Home wins: {home_wins}, Away wins: {away_wins}, Ties: {ties}"
                        )
                    else:
                        target_mean = matrix_df[target_col].mean()
                        target_std = matrix_df[target_col].std()
                        print(
                            f"  Target mean: {target_mean:.3f}, std: {target_std:.3f}"
                        )

        print(f"\nTotal unique features across all matrices: {total_features}")

        # Sample feature types
        if feature_matrices:
            sample_matrix = next(iter(feature_matrices.values()))
            feature_cols = [
                col
                for col in sample_matrix.columns
                if col
                not in [
                    "game_id",
                    "season",
                    "week",
                    "feature_timestamp",
                    "target_wp",
                    "target_ats",
                    "target_ou",
                ]
            ]

            feature_types = {}
            for col in feature_cols:
                if "elo_" in col:
                    feature_types.setdefault("Elo", []).append(col)
                elif any(
                    x in col
                    for x in ["home_off_", "away_off_", "home_def_", "away_def_"]
                ):
                    feature_types.setdefault("Team Form", []).append(col)
                elif any(
                    x in col
                    for x in ["travel_", "rest_", "venue_", "thursday_", "short_"]
                ):
                    feature_types.setdefault("Contextual", []).append(col)
                elif any(x in col for x in ["wind_", "temp_", "precip_", "weather_"]):
                    feature_types.setdefault("Weather", []).append(col)
                elif any(
                    x in col
                    for x in ["ml_", "spread_", "total_", "opening_", "snapshot_"]
                ):
                    feature_types.setdefault("Market", []).append(col)
                else:
                    feature_types.setdefault("Other", []).append(col)

            print("\nFeature breakdown by type:")
            for ftype, fcols in feature_types.items():
                print(f"  {ftype}: {len(fcols)} features")

        # Save feature matrices if requested
        if args.save:
            builder.save_feature_matrices(feature_matrices, args.season, args.week)
            logger.info("Saved all feature matrices to gold layer")

        logger.info("Feature matrix building completed successfully")

    except _SOURCE_LOAD_ERRORS as e:
        logger.error("Failed to build feature matrices", error=str(e))
        raise


if __name__ == "__main__":
    main()
