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
import dataclasses
import sys
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.contextual import (
    ContextualFeaturesCalculator,
    UnknownStadiumError,
    UnknownSurfaceError,
)
from features.elo_features import EloFeatureBuilder
from features.injury import INJURY_FEATURE_COLUMNS, InjuryBuilder
from features.market_anchors import MarketAnchorFeaturesCalculator
from features.normalization import (
    LockOrderedStatisticUnavailableError,
    compute_prior_season_stats,
    expanding_normalize,
)
from features.opponent_adj import (
    OPP_ADJ_SOURCE_NAME,
    OpponentAdjuster,
    opponent_adjusted_gold_columns,
)
from features.point_in_time_fill import (
    UNTIMED_LOCK_NS,
    admitted_mean,
    imputation_game_timing,
    row_timing,
)
from features.protocol import InformationTimeProvider
from features.provenance import (
    PROVENANCE_COLUMNS,
    CoverageReport,
    InformationTimeGate,
    ProvenanceCoverageError,
    SourceCheckState,
    build_lock_frame,
    derive_post_stage1_gap,
    refuse_provenance_columns,
)
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder
from features.team_form import (
    OFFENSE_ONLY_ROLLING_COLUMNS,
    TeamFormCalculator,
    source_limited_gold_columns,
)
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

# The weather inputs no forecast can supply (Plan 33.2-12, SPEC R6): ``precip_mm`` and
# ``raw_precip_mm``. The archived day-before bulletins carry no millimetre amount, so both
# are NULL on every historical row. Unlike line_movement, they DO arrive in the combined
# matrix on every build (silver ``weather_features`` still carries them), and they leave
# through the same drop -- the registry, never a hand-written list.
_WEATHER_UNSUPPLIED_GROUP = "weather_unsupplied"

# THE BETTING LINE (Plan 33.2-19, p332_ rung 9, D33.2-03): ``snapshot_spread``,
# ``snapshot_total``, ``snapshot_ml_prob_home_fair``, ``spread_movement`` and
# ``total_movement``. No betting line is a model input for any target, and the exclusion
# comes from the ONE registry rather than the hand-written list that used to sit in
# ``combine_features``.
#
# THE SEAM SPLIT IS A DECISION, NOT AN OVERSIGHT, and it is the one difference from
# line_movement. The MERGE block was removed; the ``feature_sources["market"]``
# REGISTRATION was RETAINED. The market source is where the R2 information-time gate
# checks the lock-fenced odds selection in ``features/market_anchors.py``, and the same
# calculator feeds grading and CLV through ``pipeline/steps.py::step_build_market_anchors``.
# Deleting the registration would retire that check silently. So after rung 9 the market
# source is CHECKED by the gate and MERGED nowhere; Plan 33.2-20 records exactly that as
# the key's ``checked_not_merged`` disposition and asserts its inverse -- zero market
# columns in the final frame.
_MARKET_GROUP = "market"

# EVERY group removed from the combined matrix before gold, in drop order. ONE mechanism
# for all three (``_enforce_groups_dropped``), never three mechanisms.
GOLD_DROPPED_GROUPS: tuple[str, ...] = (
    _LINE_MOVEMENT_GROUP,
    _WEATHER_UNSUPPLIED_GROUP,
    _MARKET_GROUP,
)

# The dropped groups whose presence means a REMOVED SEAM HAS RETURNED, so finding any
# column is an anomaly worth a warning rather than the expected path. line_movement lost
# BOTH its registration and its merge block; market kept its registration and lost its
# merge block, so a market column in the combined matrix still means the merge seam came
# back. weather_unsupplied arrives from silver on every build and is dropped every build.
_SEAM_REMOVED_GROUPS: frozenset[str] = frozenset({_LINE_MOVEMENT_GROUP, _MARKET_GROUP})

# Families merged AFTER the Stage-1 information-time loop and NOT checked by the gate at
# their merge site: a structural gap, reported in its OWN set of the CoverageReport rather
# than folded into the unregistered keys, so it is not hidden inside a bookkeeping one.
#
# EMPTY SINCE PLAN 33.2-16. The opponent-adjusted family is still merged after Stage 1 (it
# needs the combined frame's games and the per-game play-by-play pool), but it is now
# CHECKED by the information-time gate AT THAT MERGE SITE (``_merge_opponent_adjusted``,
# source name ``features.opponent_adj.OPP_ADJ_SOURCE_NAME``), and reported in the ordinary
# ``checked_sources`` / ``empty_unchecked_sources`` sets like any registry key. A family
# added after Stage 1 without that check belongs back in this tuple, by name.
#
# NO LONGER THE REPORT'S SOURCE OF TRUTH (Plan 33.2-20). The CoverageReport's
# ``post_stage1_sources`` is now DERIVED per build by
# ``features.provenance.derive_post_stage1_gap`` -- the declared post-Stage-1 families not
# yet in ``checked_sources`` -- rather than read from this literal. The reason is the
# failure mode a literal cannot survive: the same edit that removed a merge-site gate check
# would set this tuple to ``()``, and the build would report full coverage of a family
# nothing checked. Derivation makes the gap a CONSEQUENCE of the check not running.
# The constant remains as the documentation of the shape, and a test pins it empty.
POST_STAGE1_SOURCES: tuple[str, ...] = ()

# THE FEATURE-SOURCE REGISTRY the Stage-1 information-time loop walks: each
# ``feature_sources`` key mapped to the ``FeatureMatrixBuilder`` attribute that BUILDS it.
# Module-level (Plan 33.2-03) so a caller can COUNT the sources without constructing the
# builder, whose constructor instantiates every calculator: ``pipeline.live_skip`` derives its
# round cap from ``FEATURE_SOURCE_KEYS`` rather than from a literal, because plans
# 33.2-12 .. 33.2-17 grow this set. Add a source HERE and ``_information_time_suppliers`` and
# the cap both follow.
#
# THE REGISTRATION IS COMPLETE, AND FROM HERE IT IS A HARD REFUSAL (Plan 33.2-20).
#
# Every one of the nine ``feature_sources`` keys now has a provenance disposition, and the
# TWO-WAY refusal is ARMED: ``features.provenance.REGISTRY_KEY_DISPOSITIONS`` is a dict the
# gate READS, and ``InformationTimeGate.assert_registry_is_fully_disposed`` raises when its
# key set differs from the live registry in EITHER direction. A future source must supply
# ``information_times()`` and carry a ledger row BEFORE it can be registered. That is a
# refusal, not a convention: a tenth key added without a row stops the build by name.
#
# THE LEDGER LIVES IN ``features/provenance.py``, NOT HERE, and the move is the point. This
# comment used to BE the registration record, rewritten from memory at each plan, and two
# keys went missing from it that way: ``team_form`` entirely, and ``games`` undisposed. A
# comment cannot be read by the code it describes; a dict can, and is.
#
# THERE IS NO EXEMPTION MECHANISM OF ANY KIND. No allow-list. No labelled exception. No
# report-only mode. No environment variable and no flag that downgrades a refusal to a
# warning. A source is checked, or the build refuses and names it -- there is no third
# outcome, and ``tests/unit/test_no_check_exemptions.py`` is the standing scan that keeps it
# that way over the parsed trees of this module, ``features/provenance.py`` and
# ``scripts/validate_features.py``.
#
# TWO DISPOSITIONS ARE NOT EXEMPTIONS, and both say so where a reader meets them:
#
#     games   takes the SECOND declared basis (``no_information``) because it DEFINES the
#             lock and so cannot supply a non-circular information time -- and it pays that
#             basis's price: its VALUES are checked, against
#             features.schedule_moves.facts_at_lock, per game
#             (``InformationTimeGate.check_games``). A mismatch refuses the build exactly as
#             a post-lock information time does for any other source.
#     market  is ``checked_not_merged``: the source is built, registered and CHECKED like
#             every other key, and Plan 33.2-19 removed only its MERGE seam under D33.2-03.
#             Its arrival assertion is the INVERSE of the others -- an empty arrival set and
#             zero market-predicate columns in any final matrix -- so it carries an
#             assertion of its own rather than being excused from one.
#
# Registration is STRUCTURAL (``isinstance(builder, InformationTimeProvider)``), so the
# mapping below is the wiring, not the switch. ``opponent_adj`` is deliberately NOT a tenth
# key: it is merged AFTER the Stage-1 loop (``_merge_opponent_adjusted``), is checked by the
# gate at that merge site (Plan 33.2-16), and is declared in
# ``features.provenance.POST_STAGE1_FAMILY_DISPOSITIONS``. An exact nine-key equality cannot
# see it, which is why the final refusal demands exactly TEN checked names.
SUPPLIER_ATTRIBUTES: dict[str, str] = {
    "team_form": "team_form_calc",
    "elo": "elo_calc",
    "contextual": "contextual_calc",
    "weather": "weather_calc",
    "market": "market_calc",
    "qb_tracking": "qb_tracker",
    "snaps": "snap_builder",
    "injury": "injury_builder",
}
FEATURE_SOURCE_KEYS: tuple[str, ...] = tuple(SUPPLIER_ATTRIBUTES)


def drop_excluded_games(
    feature_sources: dict[str, pd.DataFrame], excluded_game_ids: frozenset[str]
) -> dict[str, pd.DataFrame]:
    """*feature_sources* without the games a LIVE run has already dropped (D33.2-05).

    Every frame carrying a ``game_id`` column loses the excluded games' rows, so the base
    ``games`` frame, the lock frame built from it and every provenance frame a supplier derives
    from it agree on the narrowed set, and the information-time gate never re-reads a game the
    live-skip policy has already recorded and excluded. A frame with no ``game_id`` (a team-keyed
    source) is returned as-is: it cannot name a game, and its rows reach gold only through a merge
    onto the narrowed ``games`` frame.

    An EMPTY set returns the SAME frame objects -- a history build, which never excludes
    anything, is byte-for-byte the build it was.
    """
    if not excluded_game_ids:
        return feature_sources
    wanted = sorted(excluded_game_ids)
    return {
        name: (
            frame.loc[~frame["game_id"].astype(str).isin(wanted)].reset_index(drop=True)
            if "game_id" in frame.columns
            else frame
        )
        for name, frame in feature_sources.items()
    }


def scope_games_through_season(
    games_df: pd.DataFrame,
    through_season: int | None,
    *,
    target_season: int | None = None,
    target_week: int | None = None,
) -> pd.DataFrame:
    """*games_df* cut to seasons ``<= through_season`` -- the LADDER-RUNG full rebuild.

    OWNER RULING 2026-09-21 (Plan 33.2-08 Task 4 checkpoint, Option A). Gold has covered
    2002-2025 since before the 2026 capture reached silver. A plain ``--all-seasons``
    rebuild now ALSO builds 2026, and every rung of the Phase-33.2 ``p332_`` ladder
    (Plans 33.2-08 .. 33.2-19) must move gold for ONE declared cause (D33.2-20) -- a
    season appearing mid-ladder would be a second cause. So each rung rebuilds with
    ``--through-season 2025``, and 2026 enters gold exactly once, in Plan 33.2-20's
    production history build.

    It narrows ONLY the base ``games`` frame, which is what every builder, the lock
    frame and the information-time gate are driven from. Nothing else changes: the
    build is still a FULL rebuild written with ``replace_mode=True``, and no gate is
    skipped or relaxed -- a 2002-2025 game that would refuse under ``--all-seasons``
    refuses here too.

    ``None`` (the default) is the unbounded full rebuild, returned as the SAME object.

    Raises:
        ValueError: when combined with a season or week scope (it is a bound on the
            FULL rebuild, not a scoped build), or when the bound leaves no game.
    """
    if through_season is None:
        return games_df
    if target_season is not None or target_week is not None:
        msg = (
            "through_season bounds the FULL rebuild and cannot be combined with "
            f"target_season={target_season!r} / target_week={target_week!r}: a scoped "
            "build merges its slice, a full rebuild replaces the table."
        )
        raise ValueError(msg)
    scoped = games_df[games_df["season"] <= through_season]
    if len(scoped) == 0:
        msg = (
            f"through_season={through_season} leaves no game in silver games (seasons "
            f"present: {sorted(games_df['season'].dropna().unique().tolist())})."
        )
        raise ValueError(msg)
    return scoped


def scope_full_rebuild_to_elo_coverage(
    games_df: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """*games_df* cut to the games an Elo snapshot can date -- the FULL rebuild only.

    WHAT THIS RESOLVES, AND WHY IT IS NOT A RELAXATION (Plan 33.2-20).

    Under the owned provenance contract a game in the Elo SOURCE frame with no
    ``elo_game_snapshots`` row gets no provenance row, and the gate's two-way coverage
    refuses it by name. That is correct and stays. But ``--all-seasons`` loads EVERY silver
    game, and silver now carries the whole unplayed 2026 schedule: MEASURED 2026-09-22, 240
    of the 272 2026 games are beyond the provisional week and have no snapshot, because
    pre-game Elo for a game five months out does not exist. A full history rebuild would
    refuse on all 240.

    The two honest routes were: scope the build to games that CAN carry a snapshot, or have
    the provisional-snapshot path cover them. The second is rejected on its merits --
    generating a "provisional" pre-game rating for a week-18 game before week 3 is played
    is a fabricated number wearing a real column's name, which is the defect this whole
    phase removes. So the build is SCOPED, and the coverage rule is untouched: every game
    that IS in the build still needs a snapshot and a provenance row, checked exactly as
    before.

    THE SCOPE IS BOUNDED SO IT CAN NEVER BECOME A SILENT HISTORY-DROPPER. A game is removed
    only when it has NO snapshot AND is UNPLAYED (no result). A game WITH a result and no
    snapshot is a real coverage defect -- the Elo chain skipped a game that has been played
    -- and it RAISES here, naming the games, rather than being quietly dropped along with
    the future ones. MEASURED on production silver: every one of the 6,499 games through
    2025 has a snapshot, so the refusal arm is live and the drop arm reaches only the future.

    FULL REBUILD ONLY. A scoped build (``--season`` / ``--week``) is untouched, exactly as
    ``scope_games_through_season`` is: the live daily path builds a week whose provisional
    snapshots were written moments earlier, and a missing one there must still refuse rather
    than silently produce an empty build.

    Args:
        games_df: The base games frame.
        snapshots: The ``elo_game_snapshots`` table (``game_id`` column).

    Returns:
        *games_df* without the unplayed, snapshot-less games. The SAME object when there
        are none.

    Raises:
        ProvenanceCoverageError: a PLAYED game has no Elo snapshot.
    """
    if len(games_df) == 0:
        return games_df

    dated = {str(g) for g in snapshots["game_id"]} if len(snapshots) else set()
    missing = ~games_df["game_id"].astype(str).isin(dated)
    if not bool(missing.any()):
        return games_df

    played = pd.Series(False, index=games_df.index)
    for column in ("home_score", "away_score"):
        if column in games_df.columns:
            played |= games_df[column].notna()

    played_without_a_snapshot = games_df.loc[missing & played, "game_id"]
    if len(played_without_a_snapshot) > 0:
        names = sorted(str(g) for g in played_without_a_snapshot)
        more = f" (+{len(names) - 10} more)" if len(names) > 10 else ""
        msg = (
            f"{len(names)} PLAYED game(s) have no elo_game_snapshots row: "
            f"{names[:10]}{more}. A played "
            "game with no pre-game Elo is a coverage defect in the Elo chain, not a game "
            "in the future -- it is refused here rather than dropped with the unplayed "
            "ones. Rebuild the Elo chain; the coverage rule is not relaxed."
        )
        raise ProvenanceCoverageError(
            msg,
            {
                "source": "elo",
                "violation_type": "played_game_without_a_snapshot",
                "game_ids": names,
            },
        )

    scoped = games_df.loc[~missing]
    dropped = games_df.loc[missing]
    logger.info(
        "Scoped the full rebuild to the games an Elo snapshot can date: unplayed games "
        "beyond the provisional week carry no pre-game Elo and are not built",
        games_before=len(games_df),
        games_after=len(scoped),
        dropped=len(dropped),
        dropped_by_season={
            int(season): int(count)
            for season, count in dropped.groupby("season").size().items()
        }
        if "season" in dropped.columns
        else {},
    )
    return scoped


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

# THE EMPTY-SOURCE-FAMILY GUARD (Plan 33.2-15, D33.2-16).
#
# WHAT IT ENDS. A family whose SOURCE FRAME carries nothing -- zero rows (the source failed to
# load or build), or rows whose every value column is NULL (the builder read an empty silver
# table) -- used to reach the generic imputer `_impute_team_features`, whose last resort is the
# strictly-prior-seasons median, and the neutral 0.0 z-score after it. Measured on the snap
# family: 2025 weeks 3 and 15 produced IDENTICAL snap values in columns the ATS and O/U models
# actually consume, because no current snap data existed and something plausible was put in
# its place. A historical median wearing a real column's name is a fabricated input. So an
# empty family's value columns are NaN, are EXCLUDED from the generic imputer and are preserved
# as NaN through normalization; the injury family's existing `*_coverage` flags read 0.0 (no
# information). Absence then survives into the models the way it should: WP's fold-fitted
# imputation pipeline (models/trainers/wp_trainer.py) median-imputes INSIDE the fold and appends
# a `_was_missing` indicator per column; ATS/O-U XGBoost take NaN natively. Do NOT restore a
# build-time fill "as a convenience".
#
# WHAT IT DELIBERATELY DOES NOT TOUCH, and who owns it. The guard is keyed on the SOURCE FRAME
# being empty, never on a season literal, so on a full-history rebuild (neither frame empty) it
# moves nothing; it is the LIVE-path property D33.2-16 requires. Its PER-SEASON half -- a family
# with no value for one whole season while other seasons have them (the seasons before snaps' or
# injury's first covered season) -- is Plan 33.2-17's, below: THE WHOLE-FAMILY-SEASON GUARD.
EMPTY_SOURCE_GUARDED_FAMILIES: tuple[str, ...] = ("snaps", "injury")

# THE WHOLE-FAMILY-SEASON GUARD (Plan 33.2-17 Task 2, D33.2-08 item 2; the per-season half the
# empty-source guard above deliberately left).
#
# A family that DECLARES A COVERAGE FLAG -- snaps (``snap_coverage``) and injury
# (``injury_coverage`` / ``availability_coverage``); the opponent-adjusted family's values are
# never imputed at all (FLAG_GUARDED_NAN_COLUMNS) -- and that has NO value in a column for a
# whole season is BEFORE that family's coverage (or its feed was absent that season). Its values
# stay NaN, its flags say so, and the NaN survives normalization: nothing -- not even the
# strictly-prior-seasons median -- stands in for a season with no data. Keyed on the season
# carrying no value in the column, NEVER on a season literal: the first covered season emerges
# from what upstream supplied. A WITHIN-SEASON gap in such a family is still filled, by the
# point-in-time rule of p332_ step 7b. On today's history the prior median already found nothing
# for the pre-coverage seasons (no earlier season has data either); what this ends is a FUTURE
# season with prior data and none of its own being filled with a borrowed median -- exactly how
# 2025 weeks 3 and 15 once came out identical.
COVERAGE_FLAGGED_FAMILIES: tuple[str, ...] = EMPTY_SOURCE_GUARDED_FAMILIES

# THE FLAG-GUARDED NaN COLUMNS (Plan 33.2-16, SPEC R9 / R10).
#
# The twelve ``*_rolling_opp_adj_*`` values are NaN exactly where the opponent adjustment could
# not reach -- below the minimum opponent history, before a team's first covered game, and in
# the seasons the per-game play-by-play pool does not cover -- and every such row carries its
# ``*_rolling_opp_adj_coverage`` flag at 0.0. That NaN is the ANSWER, so it is excluded from the
# generic imputer and survives normalization, like an empty family's (above).
#
# WHY IT MUST NOT BE IMPUTED, measured rather than argued: ``_impute_team_features`` fills a gap
# with the team's WITHIN-SEASON mean, so a 2018 week-3 row the adjustment could not reach would
# be filled from that team's weeks 4-17 -- a look-ahead the day-before lock forbids, and one the
# retired raw-EPA fall-through never had. Its last resort, the neutral 0.0 z-score, would state
# "exactly average" about a value nobody computed. WP's fold-fitted pipeline median-imputes
# INSIDE the fold and appends a ``_was_missing`` indicator; ATS/O-U XGBoost take NaN natively;
# and the four flags say the absence out loud to every model.
FLAG_GUARDED_NAN_COLUMNS: tuple[str, ...] = opponent_adjusted_gold_columns()[0]


def source_family_is_empty(frame: pd.DataFrame, value_columns: Sequence[str]) -> bool:
    """True when a family's source frame carries no value at all.

    Zero rows, or not one non-null value in any of the family's VALUE columns (its
    ``*_coverage`` flags excluded). A frame whose builder emitted its documented unknown values
    -- the injury builder's neutral defaults with the coverage flags at 0.0 -- is NOT empty: those
    are values the information-time gate value-checks, not an absence.

    Args:
        frame: The family's ``feature_sources`` frame.
        value_columns: The family's value (non-flag) columns.
    """
    if len(frame) == 0:
        return True
    present = [column for column in value_columns if column in frame.columns]
    return not present or not bool(frame[present].notna().to_numpy().any())


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


def _enforce_groups_dropped(
    combined_features: pd.DataFrame,
    groups: tuple[str, ...] = GOLD_DROPPED_GROUPS,
) -> pd.DataFrame:
    """Guarantee no column of any group in *groups* reaches gold.

    GENERALISED from the Phase-29 line-movement drop (Plan 33.2-12) rather than
    copied: ONE body, one registry lookup per group, so the weather inputs no
    forecast supplies (``weather_unsupplied``) and, later, the market columns
    (Plan 33.2-19) leave through the same mechanism as line movement.

    LINE MOVEMENT (SPEC R3 / D29-07-01). On the intended path its lookup finds
    nothing, because BOTH seams that could land the family have been removed --
    the ``feature_sources`` registration and the explicit ``combine_features``
    merge block. That is the structural removal, and it is the one that matters.
    WEATHER_UNSUPPLIED arrives from silver ``weather_features`` on every build and
    is dropped every build; that is the expected path, logged at info.

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

    for group in groups:
        present = group_columns(combined_features, group)
        if not present:
            continue
        if group in _SEAM_REMOVED_GROUPS:
            logger.warning(
                "A dropped group's columns reached the combined matrix and were "
                "dropped before gold; a removed merge/registration seam has returned",
                group=group,
                columns=present,
                count=len(present),
            )
        else:
            logger.info(
                "Dropped a group that must not reach gold",
                group=group,
                columns=present,
                count=len(present),
            )
        combined_features = drop_feature_group(combined_features, group)
    return combined_features


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

        # WR-06's machine-readable flag, RENAMED by p332_ extra step 8b for what it
        # now records: a column name mapped to the seasons whose winsorization was
        # SKIPPED because no usable strictly-prior slice existed. Until step 8b those
        # seasons fitted their bounds on their OWN rows instead, and the flag was
        # called ``self_fit_seasons``; the owner ruled on 2026-09-22 that a bound
        # fitted on the season it clips is post-lock information (D33.2-01), so the
        # seasons are now left unclipped and the log says so. Reset at the start of
        # every ``handle_missing_data_and_outliers`` call and logged at its end, so
        # it is observable in a build log AND assertable by a test. An entry outside
        # the earliest data-bearing season is a per-column coverage floor -- a fact
        # worth surfacing, which a log line alone would never make checkable.
        self.unclipped_seasons: dict[str, list[int]] = {}

        # RULING K1: the per-builder missing-preserving set. `None` until a
        # weather frame is merged, and the weather branch of `combine_features`
        # is its ONLY writer. The mapping carries exactly ONE key -- the builder
        # whose frame was actually merged -- because a build merges one weather
        # frame, and declaring an entry for a builder that did not run would be
        # asserting about a set nothing consumed.
        self.missing_preserving_columns: dict[str, tuple[str, ...]] | None = None
        self.active_builder_key: str | None = None

        # PLAN 33.2-15: the families whose SOURCE FRAME was empty in the last
        # ``combine_features`` call, mapped to their VALUE columns. Their columns are NaN, the
        # generic imputer skips them and normalization preserves the NaN (see
        # EMPTY_SOURCE_GUARDED_FAMILIES). Reset on every ``combine_features`` call.
        self.empty_source_families: dict[str, tuple[str, ...]] = {}

        # P332_ EXTRA STEP 7b (owner ruling 2026-09-22): when each game's RESULT existed and
        # its lock, ``game_id``-indexed (``features.point_in_time_fill``). Set by
        # ``combine_features`` from the games frame it combines; the imputer admits a value
        # into a fill only when its game had ended by the gap's lock. ``None`` = nothing can
        # be timed, so no within-season statistic is read at all.
        self.imputation_timing: pd.DataFrame | None = None

        # ...and the cells the imputer had to leave BLANK because nothing known at the lock
        # could fill them, although the season carries values for the column. Column ->
        # row index. Reset on every ``handle_missing_data_and_outliers`` call; normalization
        # keeps these cells NaN instead of turning them into the neutral 0.0 z-score.
        self.imputation_left_blank: dict[str, pd.Index] = {}

        # PLAN 33.2-20: what each merge block ADDED to the combined frame, keyed by the
        # declared source name. Reset on every ``combine_features`` call and read by
        # ``InformationTimeGate.assert_merge_dispositions`` on the FINAL matrices.
        # RECORDED, never inferred: ``team_form`` and ``qb_tracking`` both arrive renamed,
        # so a match against a source frame's own column names would reject correct gold.
        self.merge_arrivals: dict[str, tuple[str, ...]] = {}

    def _record_arrival(
        self, source_name: str, before: Sequence[str], frame: pd.DataFrame
    ) -> None:
        """Record the columns *source_name*'s merge block added, as a set difference.

        Called with the frame's columns as they stood BEFORE the block and the frame
        AFTER it. Repeated calls for one key UNION, so a source merged in two steps (the
        two QB merges, the team-form layout and its coverage flags) is recorded once and
        completely. An empty result is a RECORD that the block added nothing, which is a
        different fact from never having been called.
        """
        had = set(map(str, before))
        added = {str(column) for column in frame.columns} - had
        self.merge_arrivals[source_name] = tuple(
            sorted(set(self.merge_arrivals.get(source_name, ())) | added)
        )

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
        *,
        through_season: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all feature sources from silver layer.

        Args:
            target_season: Specific season to load
            target_week: Specific week to load
            as_of_datetime: The cutoff argument the FeatureBuilder Protocol still carries.
                It is NOT a fence for QBTracker or InjuryBuilder (Plan 33.2-13), nor for
                the contextual, snap, team-form and market-anchor builders (Plan
                33.2-14), nor for OpponentAdjuster (Plan 33.2-16): each selects at every game's
                own lock. Defaults to ``datetime.now(ET)``.
            through_season: Last season a FULL rebuild carries (the ladder-rung
                bound, see ``scope_games_through_season``). ``None`` = every season.

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
            games_df = scope_games_through_season(
                games_df,
                through_season,
                target_season=target_season,
                target_week=target_week,
            )
            # PLAN 33.2-20: the FULL rebuild carries only the games an Elo snapshot can
            # date. An unplayed game months out has no pre-game Elo and never will until
            # it is played; a PLAYED game with no snapshot still refuses by name. A scoped
            # build is untouched, so the live daily path's behaviour is unchanged.
            if target_season is None and target_week is None:
                games_df = scope_full_rebuild_to_elo_coverage(
                    games_df, self.elo_calc._load_snapshots()
                )
            feature_sources["games"] = games_df
            logger.info("Loaded games data", records=len(games_df))

            # ONE lock per game for the builders that select at their game's own lock
            # (Plan 33.2-13: QBTracker, InjuryBuilder), built ONCE here and handed in,
            # never derived per row. A game with no kickoff has no lock: that refusal
            # (MissingKickoffError) is re-raised by the outer handler below rather than
            # turning any source into an empty frame. generate_feature_matrices builds
            # the gate's own lock frame again AFTER a live run's exclusions, from the
            # same rule, so the two cannot disagree for any game both contain.
            source_locks = build_lock_frame(games_df)

            # Team form features, laid out ONE ROW PER GAME (Plan 33.2-14). Silver
            # ``team_form_features`` is keyed by (team, side, target week) and carries no
            # game_id, so the information-time gate could not match it to a lock; the
            # per-game frame -- exactly the home/away columns ``combine_features`` used
            # to lay out at merge time, built by the same ``_get_team_features`` -- is what
            # reaches gold, so it is what the gate checks.
            try:
                team_form_df = load_dataframe("team_form_features", layer="silver")
                # WR-10's shape, as the games block above (Plan 33.2-27): a season alone
                # narrows, and a week narrows further. It used to need BOTH, so a
                # season-only build left this source unfiltered.
                if target_season:
                    team_form_df = team_form_df[
                        team_form_df["target_season"] == target_season
                    ]
                    if target_week:
                        team_form_df = team_form_df[
                            team_form_df["target_week"] == target_week
                        ]
                team_form_df = self._team_form_per_game(team_form_df, games_df)
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
                    lock_frame=source_locks,
                )
                feature_sources["contextual"] = contextual_df
                logger.info("Built contextual features", records=len(contextual_df))
            except (UnknownStadiumError, UnknownSurfaceError):
                # UnknownSurfaceError (Plan 33.2-10 step 3b) is a LookupError, which the
                # handler below does not catch either; it is named here so its route to
                # the caller is stated rather than incidental.
                #
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
                # WR-10's shape (Plan 33.2-27): a season alone narrows, a week further.
                if target_season:
                    weather_df = weather_df[weather_df["season"] == target_season]
                    if target_week:
                        weather_df = weather_df[weather_df["week"] == target_week]
                # SCOPED TO THIS BUILD'S GAMES (Plan 33.2-12). The frame is left-merged
                # onto `games`, so a row for a game outside the build never reached gold;
                # but the information-time gate checks the SOURCE frame one-to-one
                # against a lock per game, and a silver row for a game this build does
                # not carry (a 2026 game under --through-season 2025) has no lock.
                weather_df = weather_df[weather_df["game_id"].isin(games_df["game_id"])]
                # THE FRAME FOLLOWS THE FENCE (Plan 33.2-20). The weather VALUES come off
                # silver ``weather_features`` while the PROVENANCE is derived from silver
                # ``weather`` through the one fence, so the two could disagree -- and on
                # the first clean 2026 build they did, for one game whose forecast was
                # captured two days AFTER it was played. A row whose forecast the fence
                # cannot date now carries no measurement, through the same
                # absent-observation shape the games abroad take.
                weather_df = self.weather_calc.enforce_fence_on_feature_frame(
                    weather_df, games_df
                )
                feature_sources["weather"] = weather_df
                logger.info("Loaded weather features", records=len(weather_df))
            except _SOURCE_LOAD_ERRORS as e:
                logger.warning("Failed to load weather features", error=str(e))
                feature_sources["weather"] = pd.DataFrame()

            # Market anchor features (computed on-the-fly via MarketAnchorFeaturesCalculator).
            # Each game's lines are those CAPTURED at or before its own lock (Plan 33.2-14,
            # owner ruling 2026-09-22): the recorded created_at, never the snapshot_ts label.
            try:
                market_df = self.market_calc.build_features(
                    games_df,
                    as_of_datetime,
                    target_season=target_season,
                    target_week=target_week,
                    lock_frame=source_locks,
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
                    lock_frame=source_locks,
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
                    lock_frame=source_locks,
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
                    lock_frame=source_locks,
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

        # P332_ EXTRA STEP 7b: time every game once, from the frame being combined, so the
        # imputer can tell which games had ended by a gap's lock.
        self.imputation_timing = imputation_game_timing(games_df)

        # ARRIVAL IS RECORDED HERE, PER MERGE BLOCK, AND NEVER INFERRED FROM A SOURCE
        # FRAME'S COLUMN NAMES (Plan 33.2-20). Two sources arrive RENAMED -- ``team_form``
        # through ``_get_team_features`` with ``home_``/``away_`` prefixes, and
        # ``qb_tracking``'s ``qb_adjustment`` as ``home_qb_adjustment`` /
        # ``away_qb_adjustment`` -- so a name match against the source frame would reject
        # correct gold for both, the same false rejection the reviews found for ``market``.
        # What each block ADDED is the only honest answer, and only the block knows it.
        self.merge_arrivals = {}

        # Initialize combined features with game identifiers
        combined_features = games_df[
            ["game_id", "season", "week", "home_team", "away_team"]
        ].copy()

        # Add game results if available (for target creation)
        if "home_score" in games_df.columns:
            combined_features["home_score"] = games_df["home_score"]
        if "away_score" in games_df.columns:
            combined_features["away_score"] = games_df["away_score"]

        # ``games`` is the BASE frame, so its arrivals are the identity columns
        # combine_features starts from -- which is why its merge disposition is ``merged``
        # like any other key rather than a special case.
        self._record_arrival("games", [], combined_features)

        # Merge each feature source
        feature_counts = {}

        # Team form features. ``load_all_feature_sources`` hands them in already laid
        # out one row per game (Plan 33.2-14); a team-keyed frame (a caller that built
        # ``feature_sources`` by hand) is laid out here exactly as before.
        team_form_df = feature_sources.get("team_form", pd.DataFrame())
        _before = list(combined_features.columns)
        if len(team_form_df) > 0 and "game_id" in team_form_df.columns:
            combined_features = combined_features.merge(
                team_form_df, on="game_id", how="left"
            )
            feature_counts["team_form"] = len(
                [col for col in combined_features.columns if "form_" in col]
            )
        elif len(team_form_df) > 0:
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
        combined_features = self._flag_source_limited_team_form(combined_features)
        # The source-limited coverage flags are DERIVED from the team-form values laid out
        # above, so they are team_form's arrivals too: the snapshot spans both steps.
        self._record_arrival("team_form", _before, combined_features)

        # Elo features (game-level, already has home/away columns from EloFeatureBuilder)
        elo_df = feature_sources.get("elo", pd.DataFrame())
        _before = list(combined_features.columns)
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
        self._record_arrival("elo", _before, combined_features)

        # Contextual features (game-level)
        contextual_df = feature_sources.get("contextual", pd.DataFrame())
        _before = list(combined_features.columns)
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
        self._record_arrival("contextual", _before, combined_features)

        # Weather features (game-level)
        weather_df = feature_sources.get("weather", pd.DataFrame())
        _before = list(combined_features.columns)
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
        self._record_arrival("weather", _before, combined_features)

        # ``market`` ARRIVES NOWHERE, and that is RECORDED rather than left absent (Plan
        # 33.2-20). An absent record and an empty one are different facts -- "nobody said"
        # against "the block added nothing" -- and only the second is this key's correct
        # ``checked_not_merged`` state. The assertion for it is the INVERSE of every other
        # key's: the arrival set must be empty AND no final matrix may carry a column
        # matched by backtest.signal_lift._GROUP_PREDICATE["market"].
        self._record_arrival(
            "market", list(combined_features.columns), combined_features
        )

        # -- THE MARKET MERGE SEAM IS GONE (Plan 33.2-19, p332_ rung 9, D33.2-03) --
        #
        # A ``market_feature_cols`` list stood here naming ``snapshot_spread``,
        # ``snapshot_total``, ``snapshot_ml_prob_home_fair``, ``spread_movement`` and
        # ``total_movement``, and a merge that landed them in the combined frame. It was a
        # SECOND answer to "which columns are the betting line", beside
        # ``backtest.signal_lift._GROUP_PREDICATE["market"]``, and a second answer drifts.
        # The merge is removed and the exclusion comes from that ONE registry, through
        # ``_enforce_groups_dropped`` -- the same drop mechanism line_movement and
        # weather_unsupplied use.
        #
        # THE REGISTRATION IS RETAINED ON PURPOSE. ``feature_sources["market"]`` is still
        # built and still registered above, because that is where the R2 information-time
        # gate checks the lock-fenced odds selection in ``features/market_anchors.py``,
        # which grading and CLV still depend on. Removing the registration as well would
        # retire that check silently and would leave Plan 33.2-20's nine-key
        # ``REGISTRY_KEY_DISPOSITIONS`` ledger disagreeing with the live registry. So the
        # market source is CHECKED and MERGED NOWHERE: Plan 33.2-20 records that as its
        # ``checked_not_merged`` disposition.
        #
        # A LATER READER WHO RESTORES THE MERGE lands five line columns in the combined
        # frame. ``_enforce_groups_dropped`` removes them, with a WARNING naming the
        # returned seam, before ``handle_missing_data_and_outliers`` sees them -- which is
        # why ``market`` is in ``_SEAM_REMOVED_GROUPS``. Nothing else would notice:
        # ``combine_features`` has no generic loop over ``feature_sources``.

        # QB adjustment features (one value per team per game)
        qb_df = feature_sources.get("qb_tracking", pd.DataFrame())
        _before = list(combined_features.columns)
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
        # RENAMED ON ARRIVAL: the source column is ``qb_adjustment`` and what lands is
        # ``home_qb_adjustment`` / ``away_qb_adjustment``. A source-name match would find
        # neither and reject correct gold.
        self._record_arrival("qb_tracking", _before, combined_features)

        # Snap-count features (game-level; SnapCountBuilder already emits the
        # home_/away_-expanded columns, so merge on game_id like the elo block).
        # combine_features has NO generic loop over feature_sources keys -- without
        # this explicit block the registered snap columns pass the LeakageGate but
        # are SILENTLY DROPPED from gold (review #1, orchestrator-verified).
        snaps_df = feature_sources.get("snaps", pd.DataFrame())
        _before = list(combined_features.columns)
        if len(snaps_df) > 0:
            snap_cols = [c for c in snaps_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                snaps_df[["game_id", *snap_cols]], on="game_id", how="left"
            )
            feature_counts["snaps"] = len(snap_cols)
        self._record_arrival("snaps", _before, combined_features)

        # Injury features (game-level; InjuryBuilder already emits the home_/away_-
        # expanded columns). Same rationale as the snap block (review #1): merge on
        # game_id so the columns actually reach all three gold matrices.
        injury_df = feature_sources.get("injury", pd.DataFrame())
        _before = list(combined_features.columns)
        if len(injury_df) > 0:
            injury_cols = [c for c in injury_df.columns if c != "game_id"]
            combined_features = combined_features.merge(
                injury_df[["game_id", *injury_cols]], on="game_id", how="left"
            )
            feature_counts["injury"] = len(injury_cols)
        self._record_arrival("injury", _before, combined_features)

        # PLAN 33.2-15: a family whose source frame carries nothing reads as UNKNOWN (NaN, its
        # coverage flags at 0.0), never as a historical median. Laid out here, after the two
        # merges, so the matrix width does not depend on whether a source loaded.
        #
        # Its columns count as that family's arrivals: they are the family's answer, laid
        # out under its own names. (An empty source now REFUSES the build at Stage 1, so a
        # passing build never reaches here with a family in that state -- but the record is
        # kept honest rather than left to that assumption.)
        _before_layout = list(combined_features.columns)
        combined_features = self._lay_out_empty_source_families(
            combined_features, feature_sources
        )
        for family in self.empty_source_families:
            self._record_arrival(family, _before_layout, combined_features)

        # SEAM 2 of 2 for the Phase-29 line-movement family is DELIBERATELY ABSENT
        # here (SPEC R3, D29-07-01). This is where an explicit merge block used to
        # sit, and it is the seam that actually landed the columns in gold: a
        # source registered in ``feature_sources`` but not merged here passes the
        # LeakageGate and is then silently dropped (the 28-06 lesson). Read in
        # reverse, that is exactly why removing only the registration would not
        # have been enough on its own, and why removing only this block would not
        # either -- a later reader restoring one seam must restore both, and
        # ``_enforce_groups_dropped`` will remove the result anyway.

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

    def _source_family_columns(
        self, family: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(value_columns, coverage_flag_columns)`` a guarded family emits into gold.

        DERIVED from each builder's own declaration, never typed again here: the snap
        builder's ``no_information_signature`` (every snap column, ``snap_coverage`` among
        them since Plan 33.2-17) and ``features.injury.INJURY_FEATURE_COLUMNS`` expanded
        home/away. A ``*_coverage`` column is a flag.
        """
        if family == "snaps":
            columns = tuple(self.snap_builder.no_information_signature())
        elif family == "injury":
            columns = tuple(
                f"{prefix}_{column}"
                for prefix in ("home", "away")
                for column in INJURY_FEATURE_COLUMNS
            )
        else:
            msg = f"{family!r} is not an empty-source-guarded family"
            raise KeyError(msg)
        flags = tuple(c for c in columns if c.endswith(LEVEL_PRESERVED_COLUMN_SUFFIX))
        values = tuple(c for c in columns if c not in flags)
        return values, flags

    def _lay_out_empty_source_families(
        self,
        combined_features: pd.DataFrame,
        feature_sources: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Give every guarded family whose SOURCE FRAME is empty its honest unknown.

        Its value columns are NaN (added when the merge above never ran because the frame had
        zero rows) and its ``*_coverage`` flags are 0.0, and the family is recorded in
        ``self.empty_source_families`` so the generic imputer skips it and normalization keeps
        the NaN. A family whose frame carries values is untouched. See
        ``EMPTY_SOURCE_GUARDED_FAMILIES`` for why, and for what is deliberately left to Plan
        33.2-17.
        """
        self.empty_source_families = {}
        for family in EMPTY_SOURCE_GUARDED_FAMILIES:
            frame = feature_sources.get(family, pd.DataFrame())
            values, flags = self._source_family_columns(family)
            if not source_family_is_empty(frame, values):
                continue
            self.empty_source_families[family] = values
            for column in values:
                combined_features[column] = np.nan
            for column in flags:
                combined_features[column] = 0.0
            logger.warning(
                "A feature family's source frame is EMPTY: its columns read as unknown "
                "(NaN, coverage 0.0) and are excluded from imputation",
                family=family,
                source_rows=len(frame),
                value_columns=len(values),
            )
        return combined_features

    @staticmethod
    def _flag_source_limited_team_form(frame: pd.DataFrame) -> pd.DataFrame:
        """Give each source-limited team-form metric its coverage flag (Plan 33.2-17 Task 2).

        ``features.team_form.SOURCE_LIMITED_ROLLING_COLUMNS`` names the metrics the pinned
        play-by-play does not carry in every season (``rolling_cpoe`` before 2006). The flag is
        1.0 where the laid-out value exists and 0.0 where it does not -- including a game with
        no team-form row at all -- and the whole-family-season guard keeps the unmeasured value
        NaN through normalization. Added only when the value column is present.
        """
        for value, flag in source_limited_gold_columns():
            if value in frame.columns:
                frame[flag] = frame[value].notna().astype(float)
        return frame

    def _coverage_flagged_value_columns(self, frame: pd.DataFrame) -> frozenset[str]:
        """The value columns of every flag-declaring family whose flags are in *frame*.

        Plan 33.2-17's whole-family-season guard reads this: for these columns a season with
        no value stays NaN instead of taking the strictly-prior-seasons median. Derived from
        the builders' own column lists (``_source_family_columns``), never from a name pattern.
        """
        columns: set[str] = set()
        for family in COVERAGE_FLAGGED_FAMILIES:
            values, flags = self._source_family_columns(family)
            if any(flag in frame.columns for flag in flags):
                columns.update(c for c in values if c in frame.columns)
        # The team-form metrics the pinned play-by-play lacks in some seasons, each with its
        # own flag (``_flag_source_limited_team_form``).
        columns.update(
            value
            for value, flag in source_limited_gold_columns()
            if value in frame.columns and flag in frame.columns
        )
        return frozenset(columns)

    def _empty_source_value_columns(self) -> tuple[str, ...]:
        """Every value column of every family recorded empty by the last combine, sorted."""
        recorded = getattr(self, "empty_source_families", {}) or {}
        return tuple(sorted({c for columns in recorded.values() for c in columns}))

    def _information_time_suppliers(self) -> dict[str, object]:
        """The builder behind each ``feature_sources`` registry key.

        This maps a key to the object that BUILDS it; it does not decide which keys are
        checked. Admission to the gate is ``isinstance(builder,
        InformationTimeProvider)`` -- a structural fact about the builder, not a
        hand-kept list -- so a builder that gains the two provenance members is checked
        with no edit here.
        """
        return {
            key: getattr(self, attribute)
            for key, attribute in SUPPLIER_ATTRIBUTES.items()
        }

    def _check_information_times(
        self,
        feature_sources: dict[str, pd.DataFrame],
        lock_frame: pd.Series,
        target_season: int | None,
        target_week: int | None,
    ) -> CoverageReport:
        """Stage 1: every registered source's per-game information time vs its lock.

        A STAGED ROLLOUT, NEVER AN EXEMPTION. Admission is structural
        (``isinstance(builder, InformationTimeProvider)``), so this docstring is the
        RECORD of which keys have a supplier, not a switch:

        * REGISTERED: ``elo`` (Plan 33.2-01); ``weather`` (Plan 33.2-12 -- the selector
          owns the weather fence and is the only thing that knows which bulletin a game
          used); ``qb_tracking`` and ``injury`` (Plan 33.2-13 -- both select at each
          game's own lock and report per-row provenance: the depth-chart ``dt`` and the
          latest admitted game end for QB, the latest admitted report time for injury,
          and ``no_information`` wherever nothing was admitted); ``contextual``,
          ``snaps`` and ``team_form`` (Plan 33.2-14 -- each admits a prior game only once
          it ENDED at or before the target game's lock, and reports the end of the latest
          game it actually read); ``market`` (Plan 33.2-14 -- a line counts only with a
          recorded capture time, ``created_at``, at or before the lock, and reports the
          latest such capture; the ``snapshot_ts`` label is never an information time).
        * The full nine-key ledger, with the ``games`` disposition, sits beside
          ``SUPPLIER_ATTRIBUTES``. The post-Stage-1 opponent-adjusted family is NOT walked
          here: Plan 33.2-16 checks it at its own merge site
          (``_merge_opponent_adjusted``), which adds it to this report afterwards.

        No source is EXEMPTED, only not yet reached: the unchecked keys are NAMED in the
        CoverageReport, which is logged at every build, and Plan 33.2-20 arms the refusal
        of any key with no provenance once the list is complete. Nothing is allow-listed,
        excepted or run in a report-only mode, and nothing here downgrades a refusal to a
        warning.

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

        # THE TWO-WAY REFUSAL, armed (Plan 33.2-20). Before the loop, so a key with no
        # ledger row stops the build here rather than after ten minutes of source loads.
        gate.assert_registry_is_fully_disposed(feature_sources)

        for source_name, source_df in feature_sources.items():
            if source_name == "games":
                # NOT A SKIP. ``games`` DEFINES the lock, so it takes the second declared
                # basis and its VALUES are checked against
                # features.schedule_moves.facts_at_lock. It lands in checked_sources like
                # any other key; see InformationTimeGate.check_games.
                gate.check_games(source_df)
                continue

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

        empty_unchecked = tuple(
            sorted({*gate.empty_unchecked_sources, *empty_unregistered})
        )
        report = CoverageReport(
            checked_sources=gate.checked_sources,
            empty_unchecked_sources=empty_unchecked,
            unregistered_sources=tuple(sorted(unregistered)),
            # DERIVED, never a literal (Plan 33.2-20). A declared post-Stage-1 family that
            # has not been checked YET is in this set; ``_record_opponent_adjusted_coverage``
            # re-derives it after the merge-site check runs. A family whose check was
            # removed therefore STAYS here and the final refusal names it, which a literal
            # ``()`` could not express -- the same edit that removed the check would have
            # emptied the literal.
            post_stage1_sources=derive_post_stage1_gap(gate.checked_sources),
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

        # AN EMPTY SOURCE IS NAMED AND REFUSES (Plan 33.2-20). Plan 33.2-01 froze
        # ``SourceCheckState.EMPTY_UNCHECKED`` as a REPORT state for exactly this moment;
        # this plan makes it refuse.
        #
        # WHY IT CANNOT STAY A SKIP: ``_SOURCE_LOAD_ERRORS`` converts a FAILED source load
        # into an EMPTY frame with a warning, so a broken source is indistinguishable from
        # an absent one at the frame. Skipping an empty frame therefore lets a load failure
        # read as a clean pass and exit green -- the exact shape of defect this whole phase
        # exists to remove. ``EMPTY_UNCHECKED`` is a report state and never a way to DECLARE
        # a source unchecked: the ``InformationBasis`` vocabulary still has exactly two
        # members and gains no third.
        if empty_unchecked:
            msg = (
                f"{len(empty_unchecked)} feature source(s) loaded ZERO rows and could not "
                f"be checked against any lock: {list(empty_unchecked)}. "
                "scripts.build_features._SOURCE_LOAD_ERRORS turns a FAILED source load "
                "into an empty frame, so an empty source is indistinguishable from a "
                "broken one here -- and a build that proceeds past either exits green with "
                "the source missing from gold. Fix the source; there is no way to declare "
                "it unchecked."
            )
            raise ProvenanceCoverageError(
                msg,
                {
                    "violation_type": "empty_unchecked_source",
                    "empty_unchecked": list(empty_unchecked),
                },
            )
        return report

    # The builder's seam onto the module-level ``_enforce_groups_dropped``: the SAME
    # function object, bound as a static method, rather than a forwarding wrapper that
    # re-declared the signature and the default.
    #
    # WHY IT MATTERS BEYOND TIDINESS (Plan 33.2-19). The structural check that the ONE
    # production call site passes all three dropped groups reads the GROUP ARGUMENT off
    # the parsed call -- the bare fact that the call exists proves nothing, because with
    # the market merge deleted end-state gold is clean whether or not ``market`` is in
    # that tuple. A forwarding wrapper is a SECOND call whose argument is its own
    # parameter name, so the check saw two calls and could not say which was production.
    # One binding, one call, one readable argument.
    _enforce_groups_dropped = staticmethod(_enforce_groups_dropped)

    def _team_form_per_game(
        self, team_form_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Silver team form laid out one row per game: home columns, then away columns.

        The same two ``_get_team_features`` calls and the same two left merges
        ``combine_features`` made, in the same order, so every gold value and column is
        unchanged; only WHEN the layout happens moved (to load time, where the
        information-time gate can match each row to its game's lock).
        """
        if len(team_form_df) == 0 or len(games_df) == 0:
            return pd.DataFrame()
        base = games_df[["game_id", "season", "week", "home_team", "away_team"]]
        home = self._get_team_features(team_form_df, base, "home_team", "home")
        away = self._get_team_features(team_form_df, base, "away_team", "away")
        return (
            pd.DataFrame(base[["game_id"]])
            .merge(home, on="game_id", how="left")
            .merge(away, on="game_id", how="left")
        )

    @staticmethod
    def _get_team_features(
        team_form_df: pd.DataFrame,
        games_df: pd.DataFrame,
        team_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Get team form features for home or away team.

        Copies ONLY the ``rolling_*`` columns of the team's (season, week, side) row, renamed
        ``{prefix}_off_{col}`` / ``{prefix}_def_{col}``. A static method so the
        opponent-adjusted layout (and its coupling test) runs this exact code without a
        builder instance.

        THE DEFENCE SIDE SKIPS THE OFFENCE-ONLY METRICS (p332_ extra step 8e, owner ruling
        2026-09-22). ``features.team_form.OFFENSE_ONLY_ROLLING_COLUMNS`` names the four the
        calculator never populates for a defence row, so their defensive copies had never
        held a measured value in ANY season; gold copied them anyway and normalization
        turned the all-NaN block into a flat 0.0, which a model reads as "exactly average".
        The skip reads that ONE registry rather than a second list here, so a fifth
        offence-only metric leaves gold automatically and a metric that stops being
        offence-only returns automatically. The OFFENSIVE copies are untouched: the metrics
        are offence-only, not absent.
        """
        offense_only = set(OFFENSE_ONLY_ROLLING_COLUMNS)
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
                    if col.startswith("rolling_") and col not in offense_only:
                        feature_name = f"{prefix}_def_{col}"
                        game_features[feature_name] = def_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    # ------------------------------------------------------------------
    # The opponent-adjusted family: merged after Stage 1, CHECKED at its merge site
    # (Plan 33.2-16)
    # ------------------------------------------------------------------

    @staticmethod
    def _lay_out_opponent_adjusted(
        adjusted_df: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """The adjuster's (team, side, week) rows laid out per game, through BOTH filters.

        ``_get_team_features`` copies only ``rolling_*`` columns, and only columns containing
        ``opp_adj`` are kept -- the two filters a family column must pass to reach gold
        (``features.opponent_adj.OPP_ADJ_COVERAGE_COLUMN`` is named for exactly that). What
        survives is returned as-is, keyed by ``game_id``; nothing is filled here.
        """
        laid_out = pd.DataFrame(games[["game_id"]])
        for prefix, team_col in (("home", "home_team"), ("away", "away_team")):
            adj_team = FeatureMatrixBuilder._get_team_features(
                adjusted_df, games, team_col, prefix
            )
            # Only keep the rolling_opp_adj_* columns (not duplicate other rolling cols)
            opp_adj_cols = [c for c in adj_team.columns if "opp_adj" in c]
            if opp_adj_cols:
                laid_out = laid_out.merge(
                    adj_team[["game_id", *opp_adj_cols]], on="game_id", how="left"
                )
        return laid_out

    @staticmethod
    def opponent_adjusted_per_game(
        adjusted_df: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """The family's per-game source frame: ``game_id``, twelve values and four flags.

        :meth:`_lay_out_opponent_adjusted`, then completed to the declared gold columns: a
        game whose team carried no rolling row is the flagged unknown -- its values NaN and
        its ``*_rolling_opp_adj_coverage`` flags 0.0 -- which is exactly what
        ``OpponentAdjuster.no_information_signature`` declares and the gate value-checks.
        """
        frame = FeatureMatrixBuilder._lay_out_opponent_adjusted(adjusted_df, games)
        values, flags = opponent_adjusted_gold_columns()
        for column in values:
            if column not in frame.columns:
                frame[column] = np.nan
        for column in flags:
            frame[column] = (
                frame[column].fillna(0.0).astype(float)
                if column in frame.columns
                else 0.0
            )
        return frame[["game_id", *values, *flags]]

    def _record_opponent_adjusted_coverage(self, state: SourceCheckState) -> None:
        """Add the opponent-adjusted family to the build's CoverageReport, BY NAME.

        Stage 1 builds the report before this family exists; the merge site's gate check
        then files it under ``checked_sources`` or ``empty_unchecked_sources`` like any
        registry key, so a narrow build still reads as narrow.
        """
        report = self.information_time_coverage
        if report is None:
            return
        if state is SourceCheckState.CHECKED:
            report = dataclasses.replace(
                report, checked_sources=(*report.checked_sources, OPP_ADJ_SOURCE_NAME)
            )
        else:
            report = dataclasses.replace(
                report,
                empty_unchecked_sources=tuple(
                    sorted({*report.empty_unchecked_sources, OPP_ADJ_SOURCE_NAME})
                ),
            )
        # RE-DERIVED, not left at its Stage-1 value (Plan 33.2-20): the family has now been
        # reached, so the gap closes here and only here. A build whose merge-site check did
        # not run never gets to this line, so the gap stays open and the final refusal names
        # it.
        report = dataclasses.replace(
            report, post_stage1_sources=derive_post_stage1_gap(report.checked_sources)
        )
        self.information_time_coverage = report
        logger.info(
            "Information-time gate coverage after the post-Stage-1 merge",
            source=OPP_ADJ_SOURCE_NAME,
            state=state.value,
            checked_sources=list(report.checked_sources),
            empty_unchecked_sources=list(report.empty_unchecked_sources),
        )

    def _merge_opponent_adjusted(
        self,
        combined_features: pd.DataFrame,
        games_df: pd.DataFrame,
        lock_frame: pd.Series,
        as_of_datetime: datetime,
        target_season: int | None,
        target_week: int | None,
    ) -> pd.DataFrame:
        """Replace raw EPA with opponent-adjusted EPA, CHECKED by the information-time gate.

        THE REGISTRATION SITE (Plan 33.2-16, SPEC R2). The family is merged after Stage 1, so
        it is not a ``feature_sources`` key; it is checked HERE, against the build's one lock
        frame, before its columns join the matrix. A per-row time after a game's lock stops
        the build naming the game -- ``InformationTimeViolation`` is not in the handler below
        -- and so does an unresolvable play-by-play id (``OpponentResolutionError``, a
        ``RuntimeError`` in neither tuple). The handler keeps only the pre-existing
        degradation for an ordinary failure, which leaves raw EPA under its RAW names.

        Returns:
            *combined_features* with the twelve ``*_rolling_opp_adj_*`` values and the four
            ``*_rolling_opp_adj_coverage`` flags merged in and the six raw EPA columns
            dropped -- or unchanged when there is no per-game pool to adjust.
        """
        gate = InformationTimeGate()
        signature = self.opponent_adj.no_information_signature()
        empty_provenance = pd.DataFrame({column: [] for column in PROVENANCE_COLUMNS})

        # OpponentAdjuster needs per-game stats (with game_id, raw EPA),
        # not the rolling averages from the silver table.
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

        if len(per_game_stats) == 0:
            state = gate.check(
                OPP_ADJ_SOURCE_NAME,
                pd.DataFrame(),
                empty_provenance,
                lock_frame,
                no_information_signature=signature,
            )
            self._record_opponent_adjusted_coverage(state)
            return combined_features

        try:
            adjusted_df = self.opponent_adj.build_features(
                games_df,
                as_of_datetime,
                target_season=target_season,
                target_week=target_week,
                team_game_stats=per_game_stats,
            )
            if len(adjusted_df) == 0:
                state = gate.check(
                    OPP_ADJ_SOURCE_NAME,
                    pd.DataFrame(),
                    empty_provenance,
                    lock_frame,
                    no_information_signature=signature,
                )
                self._record_opponent_adjusted_coverage(state)
                return combined_features

            games = combined_features[
                ["game_id", "season", "week", "home_team", "away_team"]
            ]
            family = self.opponent_adjusted_per_game(adjusted_df, games)
            state = gate.check(
                OPP_ADJ_SOURCE_NAME,
                family,
                self.opponent_adj.information_times(
                    games, target_season=target_season, target_week=target_week
                ),
                lock_frame,
                no_information_signature=signature,
            )
            self._record_opponent_adjusted_coverage(state)
            _before = list(combined_features.columns)
            combined_features = combined_features.merge(
                family, on="game_id", how="left"
            )
            self._record_arrival(OPP_ADJ_SOURCE_NAME, _before, combined_features)

            # Drop old raw EPA columns that are now replaced by opp_adj versions
            raw_epa_suffixes = [
                "rolling_epa_per_play",
                "rolling_pass_epa_per_play",
                "rolling_rush_epa_per_play",
            ]
            cols_to_drop = [
                f"{pfx}_{side}_{suffix}"
                for pfx in ("home", "away")
                for side in ("off", "def")
                for suffix in raw_epa_suffixes
                if f"{pfx}_{side}_{suffix}" in combined_features.columns
            ]
            if cols_to_drop:
                combined_features = combined_features.drop(columns=cols_to_drop)
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
        return combined_features

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

        THE IMPUTATION IS POINT-IN-TIME (p332_ extra step 7b, owner ruling
        2026-09-22). D30-16 accepted a within-season residual: the team mean and the
        season mean used to be computed over the WHOLE season, so a week-3 gap was
        filled from weeks 4-17. Under the day-before lock (D33.2-01) that is post-lock
        information, and it is gone: a gap is filled only from games that had ENDED by
        its own lock (``features.point_in_time_fill``), then from the strictly-prior
        seasons, and where nothing honest exists it stays blank
        (``self.imputation_left_blank``) through normalization too.

        THE WINSORIZATION IS POINT-IN-TIME TOO (p332_ extra step 8b, owner ruling
        2026-09-22 "Skip trim, first season"). Until that step a season with no
        strictly-prior fit -- the earliest season, or a column's first populated season --
        fitted its q01/q99 bounds on its OWN whole season, so a week-1 value was clipped
        by a bound that had read week 18. That was the residual D30-16 accepted, recorded
        here as accepted, and routed to the owner by step 7b rather than folded into it.
        It is gone: bounds come from strictly-prior seasons only, a season with no such
        fit is left UNCLIPPED (exactly as a degenerate bound already was) and recorded in
        ``self.unclipped_seasons``, and clipping begins the following season. NOTHING in
        this method now fits a statistic on the season it is applied to.

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

        # P332_ EXTRA STEP 7b: every row timed ONCE (when its game ended, its lock), and the
        # blank register reset -- it describes THIS call's frame only.
        self.imputation_left_blank = {}
        timed_rows = row_timing(processed_df, self.imputation_timing)
        # PLAN 33.2-17: a flag-declaring family's whole season with no value stays NaN.
        whole_season_guarded = self._coverage_flagged_value_columns(processed_df)

        # WR-06: the per-season passes, computed ONCE. ``season`` sits in the
        # target-column exclusion list above, which removes it from ``numeric_cols``
        # but leaves the column in the frame -- so a per-season pass can group on it
        # directly. Each entry is (season, season_mask, strictly_prior_mask).
        # P332_ EXTRA STEP 8b: the seasons left UNCLIPPED because no strictly-prior slice
        # could be fitted. It replaces the self-fit log, which recorded the seasons that
        # fitted a bound on their own rows -- the behaviour this step removed.
        self.unclipped_seasons = {}
        season_passes = self._season_passes(processed_df)
        # Still read by the IMPUTER below, whose earliest-season rule step 7b settled and
        # step 8b does not touch. The winsorization pass no longer needs it at all.
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
        # PLAN 33.2-15: a family whose SOURCE FRAME was empty is excluded from imputation
        # outright (EMPTY_SOURCE_GUARDED_FAMILIES). Its NaN is the answer. PLAN 33.2-16: so is
        # an opponent-adjusted value the adjustment could not reach (FLAG_GUARDED_NAN_COLUMNS).
        empty_family_columns = set(self._empty_source_value_columns()) | set(
            FLAG_GUARDED_NAN_COLUMNS
        )

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
            preserve_this_column = (
                col in preserved_weather_columns or col in empty_family_columns
            )
            if preserve_this_column and original_missing > 0:
                missing_stats[col] = original_missing

            # Handle missing data
            if original_missing > 0 and not preserve_this_column:
                source_values = self._column(processed_df, col).copy()
                # For team-based features, the team's mean over its games ended by the lock
                if any(prefix in col for prefix in ["home_", "away_"]):
                    processed_df[col] = self._impute_team_features(
                        processed_df,
                        col,
                        timed_rows,
                        leave_empty_seasons=col in whole_season_guarded,
                    )
                else:
                    # WR-06 surface 1: for game-level features this was
                    # ``processed_df[col].median()`` over the WHOLE frame, so a 2002
                    # gap was filled from a statistic that saw 2025. It is now a
                    # per-season, strictly-prior median (and, step 7b, a pre-lock mean
                    # where no prior season can be fitted).
                    processed_df[col] = self._impute_game_level_features(
                        processed_df, col, season_passes, earliest_season, timed_rows
                    )
                self._record_imputation_blanks(
                    processed_df,
                    col,
                    source_values,
                    include_empty_seasons=col in whole_season_guarded,
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
                fit_source, no_prior_fit = self._season_fit_source(
                    fit_values, prior_mask
                )

                # P332_ EXTRA STEP 8b: a season with no usable strictly-prior fit is left
                # UNCLIPPED rather than clipped against its own rows. The season is
                # RECORDED, because "this column was not winsorized here" is a fact a
                # reader of the build log needs, and the next season -- whose prior slice
                # now holds this one -- gets a real bound.
                if no_prior_fit:
                    self._record_unclipped(col, season)
                    continue

                # The pre-WR-06 minimum-data-points condition, now applied to the fit
                # source rather than to the whole column. Unchanged by step 8b: it is the
                # same threshold the refusal above is decided on.
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
                # UNCHANGED by step 8b, including its silence: the degenerate skip was
                # never recorded in the log above and is not recorded now. That log
                # answers "which seasons had no strictly-prior fit", and widening it to
                # every constant-dominated prior slice would bury the answer.
                if lower_bound >= upper_bound:
                    continue

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
            # WR-06's observable flag, renamed by p332_ extra step 8b for what it now
            # records: the seasons left UNCLIPPED because no strictly-prior slice could
            # be fitted. A season other than the earliest appearing here is a per-column
            # coverage floor -- the column's upstream source simply starts later -- and is
            # a finding worth reading, not an error.
            unclipped_columns=len(self.unclipped_seasons),
            unclipped_seasons=sorted(
                {
                    season
                    for seasons in self.unclipped_seasons.values()
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
        degrades to ONE pass whose strictly-prior mask is empty. Under p332_ extra
        step 8b that pass has no bound to fit and the frame is left UNWINSORIZED
        (it used to self-fit, i.e. degrade to the pre-WR-06 whole-frame behaviour).
        The imputation half of the pass is unaffected and still runs.
        """
        if "season" not in df.columns:
            everything = pd.Series(True, index=df.index)
            return [(None, everything, pd.Series(False, index=df.index))]

        seasons = sorted(df["season"].dropna().unique())
        return [
            (season, df["season"] == season, df["season"] < season)
            for season in seasons
        ]

    def _record_unclipped(self, col: str, season) -> None:
        """Record that *col*'s *season* was left UNCLIPPED for want of a prior fit."""
        if season is None:
            return
        seasons = self.unclipped_seasons.setdefault(col, [])
        if int(season) not in seasons:
            seasons.append(int(season))

    def _season_fit_source(
        self,
        values: pd.Series,
        prior_mask: pd.Series,
    ) -> tuple[pd.Series, bool]:
        """Return ``(fit_source, no_prior_fit)`` for one season's winsorization bound.

        The fit source is the STRICTLY-PRIOR slice, and nothing else.

        P332_ EXTRA STEP 8b (owner ruling 2026-09-22, "Skip trim, first season"). This
        used to fall back to the season's OWN rows whenever no strictly-prior slice could
        be fitted -- the earliest data-bearing season, and a column whose upstream source
        starts mid-history (T-30-55) -- under a ``self_fit`` flag. That bound saw the whole
        season, so a week-1 value was clipped by a statistic that had read week 18. Under
        the day-before lock (D33.2-01) that is post-lock information, and it was the last
        statistic in this method still reading it.

        There is no fallback now. A season with no strictly-prior fit is reported as such
        and the caller leaves it UNCLIPPED -- exactly the treatment a degenerate bound
        already gets -- so clipping begins the FOLLOWING season, from past seasons only.
        The minimum-fit-points rule is unchanged and is what "usable" means here.
        """
        prior = values.loc[prior_mask]
        return prior, prior.notna().sum() <= self._MIN_FIT_POINTS

    def _impute_game_level_features(
        self,
        df: pd.DataFrame,
        col: str,
        season_passes: list[tuple],
        earliest_season,
        timed_rows: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> pd.Series:
        """Fill a game-level column's gaps from what was known at each gap's lock.

        WR-06: the fill is the median of the seasons STRICTLY BEFORE the gap's season,
        replacing ``df[col].fillna(df[col].median())``, whose median saw every future
        season. The median is fitted on the column as it ENTERED this pass -- ``source``
        below -- never on the partially-filled frame, so no season's statistic can depend
        on values another season's fill just wrote.

        P332_ EXTRA STEP 7b (owner ruling 2026-09-22). Where no prior season can be fitted
        (the earliest season, or a column whose data starts later) this used to fall back
        to a median over the gap's OWN whole season -- every later game that season
        included. It now takes the mean over that season's games that had ENDED by the
        gap's own lock, and where there is none the gap stays NaN: never a value read from
        a game after the lock.

        Args:
            df: The combined frame.
            col: The column to fill.
            season_passes: ``_season_passes(df)``.
            earliest_season: The first season in *df* (kept for the caller's signature;
                the strictly-prior slice of the earliest season is empty by construction).
            timed_rows: ``features.point_in_time_fill.row_timing`` for *df*; derived from
                ``self.imputation_timing`` when omitted.
        """
        del earliest_season  # the strictly-prior slice already encodes it
        source = self._column(df, col)
        result = source.copy()
        ends, locks = timed_rows if timed_rows is not None else self._timed_rows(df)
        missing = source.isna().to_numpy()

        for _season, season_mask, prior_mask in season_passes:
            season_values = result.loc[season_mask]
            if not season_values.isna().any():
                continue

            prior_median = self._strict_prior_median(source, prior_mask)
            if prior_median is not None:
                result.loc[season_mask] = season_values.fillna(prior_median)
                continue

            in_season = season_mask.to_numpy()
            season_source = source[in_season]
            season_ends = ends[in_season]
            for position in np.flatnonzero(in_season & missing):
                fill = admitted_mean(season_source, season_ends, locks[position])
                if not np.isnan(fill):
                    result.iloc[position] = fill

        return result

    def _normalization_row_locks(self, frame: pd.DataFrame) -> pd.Series:
        """Each row's LOCK, for the normalization window (p332_ extra step 8c).

        The lock comes from the ONE rule (``utils.game_lock`` through
        ``features.point_in_time_fill``, the same timing the imputer reads); this
        method derives nothing. A row the timing does not cover carries the untimed
        sentinel, and a game with no lock has no window -- so the whole frame is
        refused BY NAME rather than normalized against week order, which is the leak
        this step removed.

        Args:
            frame: The frame about to be normalized.

        Returns:
            A ``Series`` of UTC nanosecond locks, indexed like *frame*.

        Raises:
            LockOrderedStatisticUnavailableError: naming how many rows could not be
                timed, and the first of them.
        """
        _ends, locks = row_timing(frame, self.imputation_timing)
        untimed = locks == UNTIMED_LOCK_NS
        if bool(untimed.any()):
            ids = (
                frame.loc[untimed, "game_id"].astype(str).tolist()
                if "game_id" in frame.columns
                else []
            )
            msg = (
                f"{int(untimed.sum())} row(s) of the frame about to be normalized "
                f"carry no lock (e.g. {ids[:5] or 'the frame has no game_id column'}), "
                "so their expanding statistics cannot be ordered by lock instant. "
                "Refusing rather than falling back to week order (p332_ extra step 8c, "
                "D33.2-01)."
            )
            raise LockOrderedStatisticUnavailableError(msg)
        return pd.Series(locks, index=frame.index, name="lock_ns")

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
        # PLAN 33.2-15: an empty family's NaN survives normalization too, instead of becoming
        # the neutral 0.0 z-score. Appended AFTER the weather entry, and only when a family
        # was recorded empty, so a build with no empty family passes exactly the weather set.
        preserve_missing = tuple(preserve_by_builder[active_builder]) + tuple(
            column
            for column in self._empty_source_value_columns()
            if column not in set(preserve_by_builder[active_builder])
        )
        # PLAN 33.2-16: the opponent-adjusted values' NaN is the flagged unknown too.
        preserve_missing = preserve_missing + tuple(
            column
            for column in FLAG_GUARDED_NAN_COLUMNS
            if column not in preserve_missing
        )

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

        # P332_ EXTRA STEP 8c: the lock every expanding statistic below is ordered by,
        # resolved ONCE for the whole frame and then sliced per season, so the batch
        # and single-season paths cannot disagree about a row's window.
        row_locks = self._normalization_row_locks(processed_features)

        # Compute prior-season stats for bootstrap and normalize
        if target_season:
            # Single-season mode: compute prior stats once
            prior_stats = compute_prior_season_stats(
                processed_features, numeric_feature_cols, target_season - 1
            )
            return self._restore_imputation_blanks(
                expanding_normalize(
                    processed_features,
                    feature_cols=numeric_feature_cols,
                    group_col="season",
                    sort_cols=["season", "week"],
                    min_periods=4,
                    prior_season_stats=prior_stats,
                    preserve_missing_cols=preserve_missing,
                    preserve_level_cols=preserve_level_cols,
                    row_locks=row_locks,
                )
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
                preserve_missing_cols=preserve_missing,
                preserve_level_cols=preserve_level_cols,
                row_locks=row_locks.loc[season_df.index],
            )
            normalized_parts.append(norm_part)
        # P332_ EXTRA STEP 7b: a recorded blank stays blank (never the neutral 0.0).
        return self._restore_imputation_blanks(
            pd.concat(normalized_parts, ignore_index=False)
        )

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

    def _impute_team_features(
        self,
        df: pd.DataFrame,
        col: str,
        timed_rows: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        leave_empty_seasons: bool = False,
    ) -> pd.Series:
        """Impute a missing team feature from what was known at the gap's own lock.

        P332_ EXTRA STEP 7b (owner ruling 2026-09-22, "Fill from earlier games"). This used
        to fill a gap with the team's mean over the WHOLE season and then the whole-season
        league mean -- so a week-3 gap was filled from weeks 4-17, which under the
        day-before lock (D33.2-01) is post-lock information. Every statistic below now
        reads only games that had ENDED by the gap's lock (kickoff plus the declared game
        duration, admitted at or before the lock -- the builders' own timing,
        ``features.point_in_time_fill``):

        1. the team's mean over its earlier games this season, on the SAME side of the
           ball the column describes (a ``home_*`` column averages the team's home rows,
           as it always has);
        2. if the team has none, the league's mean over the season's games ended by the
           lock;
        3. then the median of the seasons STRICTLY BEFORE this one (WR-06 surface 3) --
           never a self-fit on the gap's own season;
        4. and where none of those exists the gap stays NaN. The caller records it in
           ``imputation_left_blank`` and normalization keeps it blank.

        WR-14: the no-op ``col.replace(...)`` statements that once sat here are gone; the
        imputation only needs to know WHICH team column to group by.

        Args:
            df: The combined frame.
            col: The column to fill.
            timed_rows: ``features.point_in_time_fill.row_timing`` for *df*; derived from
                ``self.imputation_timing`` when omitted.
            leave_empty_seasons: Plan 33.2-17's whole-family-season guard. When True, a season
                in which *col* has no value at all is left NaN -- not even the
                strictly-prior-seasons median fills it. Set for the value columns of a
                family that declares a coverage flag.
        """
        season_passes = self._season_passes(df)
        earliest_season = season_passes[0][0] if season_passes else None
        if timed_rows is None:
            timed_rows = self._timed_rows(df)

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
                df, col, season_passes, earliest_season, timed_rows
            )

        source = self._column(df, col)
        result = source.copy()
        ends, locks = timed_rows
        teams = df[team_col].to_numpy()
        missing = source.isna().to_numpy()

        for _season, season_mask, prior_mask in season_passes:
            in_season = season_mask.to_numpy()
            gaps = np.flatnonzero(in_season & missing)
            if len(gaps) == 0:
                continue
            # A season with no value at all for this column is a coverage floor, not a
            # within-season gap: nothing inside it can be admitted.
            season_has_values = bool((in_season & ~missing).any())
            if leave_empty_seasons and not season_has_values:
                # Before a flag-declaring family's coverage: the honest unknown, never a
                # borrowed median (Plan 33.2-17, COVERAGE_FLAGGED_FAMILIES).
                continue
            # Computed LAZILY, and at most once per season.
            prior_median: float | None = None
            prior_median_computed = False

            for position in gaps:
                fill = float("nan")
                if season_has_values:
                    team_rows = in_season & (teams == teams[position])
                    fill = admitted_mean(
                        source[team_rows], ends[team_rows], locks[position]
                    )
                    if np.isnan(fill):
                        fill = admitted_mean(
                            source[in_season], ends[in_season], locks[position]
                        )
                if np.isnan(fill):
                    if not prior_median_computed:
                        prior_median = self._strict_prior_median(source, prior_mask)
                        prior_median_computed = True
                    if prior_median is None:
                        # Nothing known at this lock can fill the gap. Leave it NaN rather
                        # than borrow from a later game.
                        continue
                    fill = prior_median
                result.iloc[position] = fill

        return result

    def _timed_rows(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-row ``(end_ns, lock_ns)`` for *df* from ``self.imputation_timing``."""
        return row_timing(df, getattr(self, "imputation_timing", None))

    def _strict_prior_median(
        self, values: pd.Series, prior_mask: pd.Series
    ) -> float | None:
        """The median of the seasons STRICTLY BEFORE this one, or None (WR-06, step 7b).

        Unlike ``_season_fit_source`` -- which the winsorization pass still uses -- this
        NEVER falls back to the season's own rows: that fallback is a whole-season read,
        and an imputed value must never see a game after its lock. A prior slice with too
        few values to support a median (``_MIN_FIT_POINTS``) yields None.
        """
        prior = values.loc[prior_mask]
        if prior.notna().sum() <= self._MIN_FIT_POINTS:
            return None
        return float(prior.median())

    def _record_imputation_blanks(
        self,
        df: pd.DataFrame,
        col: str,
        source_values: pd.Series,
        *,
        include_empty_seasons: bool = False,
    ) -> None:
        """Record the cells of *col* the imputer had to leave blank.

        Step 7b: a cell is recorded when it is still NaN after imputation AND its season
        carries at least one value for the column -- a within-season gap nothing known at its
        lock could fill.

        Plan 33.2-17 (*include_empty_seasons*): for a flag-declaring family a season with NO
        value is recorded too -- the family's coverage floor, NaN beside a false flag, kept
        blank through normalization. For every other family such a season is a coverage floor
        this rule does not own, and is not recorded.
        """
        still_missing = self._column(df, col).isna() & source_values.isna()
        if not still_missing.any():
            return
        if include_empty_seasons:
            self.imputation_left_blank[col] = df.index[still_missing.to_numpy()]
            return
        if "season" in df.columns:
            seasons_with_values = set(df.loc[source_values.notna(), "season"])
            still_missing &= df["season"].isin(seasons_with_values)
        elif not source_values.notna().any():
            return
        if still_missing.any():
            self.imputation_left_blank[col] = df.index[still_missing.to_numpy()]

    def _restore_imputation_blanks(self, normalized: pd.DataFrame) -> pd.DataFrame:
        """Put back the NaN of every recorded blank cell after normalization (step 7b).

        ``expanding_normalize`` excludes a NaN from its statistics and then writes the
        neutral 0.0 z-score into it. For a recorded blank that 0.0 would read as "exactly
        average" about a value nobody could know at the lock, so the cell is NaN again --
        blank all the way into gold. Every model reads it as missing: WP's fold-fitted
        ``_was_missing`` indicator (D33.2-08 item 2) and XGBoost's native missing branch.
        """
        for col, rows in getattr(self, "imputation_left_blank", {}).items():
            if col in normalized.columns:
                present = rows.intersection(normalized.index)
                normalized.loc[present, col] = np.nan
        return normalized

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

        # Labels are computed on the PLAYED games only. An unplayed game is KEPT and rejoins
        # below with every label blank (Plan 33.2-27): this line used to DELETE it, so the
        # slate the daily run predicts never reached gold. A history build has no unplayed game,
        # so its output, dtypes included, is unchanged.
        score_mask = target_df["home_score"].notna() & target_df["away_score"].notna()
        unplayed = target_df[~score_mask]
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

        # -- THE FOUR LINE-DERIVED TARGET COLUMNS ARE GONE (Plan 33.2-19, rung 9) --
        #
        # ``target_ats = point_differential - snapshot_spread`` and
        # ``target_ou = total_points - snapshot_total`` stood here, with
        # ``home_covered_spread`` and ``game_went_over`` as their boolean children. They
        # were arithmetic children of exactly the columns rung 9 removes, so they cannot
        # stay as they are; and they were never what their names said. They ran AFTER
        # normalization, so they subtracted a Z-SCORED market value, not a line in points
        # (recorded by Plan 33.2-08's rung-1 attribution). Since rung 5 every gold market
        # value has been the neutral 0.0, so ``target_ats`` equalled ``point_differential``
        # and ``target_ou`` equalled ``total_points`` for every game 2002-2025.
        #
        # MEASURED BEFORE REMOVING THEM, because "no reader" is a claim, not an
        # assumption: all four are ID/EXCLUDED columns in ``models/temporal._DEFAULT_ID_COLS``
        # -- never model inputs -- and no trainer, backtest, API or web module reads any of
        # them. The ATS trainer's target is ``home_margin`` and the O/U trainer's is
        # ``total_points`` (``models/trainers/ats_trainer.py``, ``ou_trainer.py``), both of
        # which are derived from the SCORES and are untouched here. ``home_covered_spread``
        # and ``game_went_over`` never reached a gold matrix at all: they were excluded from
        # the feature set and named in no target list.
        #
        # THE ALTERNATIVE WAS RECOMPUTING THEM FROM THE RAW SILVER LINE IN POINTS -- a
        # closing line is legitimate in an OUTCOME LABEL, since D33.2-01 governs model
        # INPUTS -- and it was REJECTED on measurement: only 2,140 of 6,499 games carry a
        # stored line at all, so ``final_features["target_ats"].notna()`` would have cut the
        # ATS matrix from 6,499 rows to ~2,140. The matrices are selected on the real
        # trainer targets below instead.

        logger.info(
            "Created target variables",
            wp_targets=target_df["target_wp"].notna().sum(),
            ats_targets=target_df["home_margin"].notna().sum(),
            ou_targets=target_df["total_points"].notna().sum(),
            unplayed_kept=len(unplayed),
        )

        if len(unplayed) > 0:
            target_df = pd.concat([target_df, unplayed]).loc[features_df.index]
        return target_df

    def generate_feature_matrices(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
        *,
        excluded_game_ids: frozenset[str] = frozenset(),
        through_season: int | None = None,
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
            excluded_game_ids: Games a LIVE run has already dropped under the live-skip rule
                (D33.2-05, Plan 33.2-03). Removed from every source BEFORE the lock frame and
                the information-time gate, so a re-run after a skip does not re-refuse a game
                that is already recorded and excluded. Empty by default, and a HISTORY build
                never passes it: history must stop on any violation, not skip.
            through_season: Last season a FULL rebuild carries -- the ``p332_`` ladder
                rungs pass 2025 (owner ruling 2026-09-21; ``scope_games_through_season``).
                It narrows the base games frame BEFORE the lock frame, so every gate
                below runs unchanged over the games that remain. ``None`` = every season.

        Returns:
            Dictionary with feature matrices for each target.
        """
        # THIS DEFAULT IS NO LONGER A FENCE (Phase 33.2, D33.2-01). The fence is the
        # per-game lock frame built below; nothing in Stage 1 reads this value. It
        # survives ONLY as the ``as_of_datetime`` argument the builders still take.
        # QBTracker and InjuryBuilder no longer read it (Plan 33.2-13 moved both onto
        # each game's lock), and neither does OpponentAdjuster (Plan 33.2-16). CR-01 still
        # applies to it: tz-aware ET, never a naive local clock.
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
                target_season,
                target_week,
                as_of_datetime=as_of_datetime,
                through_season=through_season,
            )
            feature_sources = drop_excluded_games(feature_sources, excluded_game_ids)

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

            # -- SPEC R3 / R6: the dropped groups must not reach gold (line movement,
            # and the weather inputs no forecast supplies). Runs on the COMBINED
            # matrix, before every downstream stage and therefore long before the
            # gold write.
            combined_features = self._enforce_groups_dropped(
                combined_features, GOLD_DROPPED_GROUPS
            )

            # -- Replace raw EPA with opponent-adjusted EPA, CHECKED by the gate at this
            #    post-Stage-1 merge site (Plan 33.2-16) --
            combined_features = self._merge_opponent_adjusted(
                combined_features,
                feature_sources["games"],
                lock_frame,
                as_of_datetime,
                target_season,
                target_week,
            )

            # -- Stage 2: Combined matrix validation --
            try:
                self.leakage_gate.validate_combined_matrix(
                    combined_features, as_of_datetime
                )
            except LeakageViolation as e:
                # THE PRODUCTION TREE IS NAMED HERE, not defaulted inside the writer
                # (Plan 33.2-20). ``write_diagnostic_report`` used to default to
                # ``outputs/diagnostics/``, so every test that drove a build into this
                # refusal wrote into the production tree. The default is gone and this,
                # the ONE production caller, says where the report goes.
                self.leakage_gate.write_diagnostic_report(
                    e, output_dir=project_root / "outputs" / "diagnostics"
                )
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

            # Base feature columns (exclude identifiers and targets).
            #
            # ``target_ats``, ``target_ou``, ``home_covered_spread`` and
            # ``game_went_over`` left the build at rung 9 and are RETAINED here on
            # purpose: this list is a guard, not a description of gold, and a name that
            # stays in it costs nothing while a name dropped from it would silently admit
            # a reintroduced line-derived column into the feature set.
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

            # ATS matrix, SELECTED ON ITS TRAINER'S OWN TARGET (Plan 33.2-19, rung 9).
            # ``home_margin`` is what ``ATSTrainer._get_target_column`` reads, and it is
            # derived from the scores. Was: gated and row-filtered on ``target_ats``, the
            # line-derived column rung 9 removes -- which would have left this matrix
            # unbuilt, and which (recomputed from the raw silver line) would have cut it
            # from 6,499 rows to the ~2,140 games that carry a stored line at all.
            #
            # EVERY GAME, PLAYED OR NOT (Plan 33.2-27), exactly as the WP matrix. Keeping only
            # rows with a label removed every unplayed game, so tomorrow's slate never reached
            # the ATS and O/U tables and could not be predicted. An unplayed game is in a full
            # build only through a PROVISIONAL Elo snapshot, which every trainer refuses at its
            # gold-loading boundary, so such a row is served and never trained on. Historical
            # rows are unchanged: every played game carries its label.
            if "home_margin" in final_features.columns:
                ats_target_cols = [
                    c
                    for c in ["home_margin", "point_differential"]
                    if c in final_features.columns
                ]
                ats_matrix = final_features[
                    [*meta_cols, *score_cols, *ats_target_cols, *feature_cols]
                ].copy()
                feature_matrices["ats"] = ats_matrix

            # O/U matrix, on ``total_points`` -- ``OUTrainer._get_target_column``'s own
            # target, also derived from the scores. Every game, for the same reason.
            if "total_points" in final_features.columns:
                ou_target_cols = [
                    c for c in ["total_points"] if c in final_features.columns
                ]
                ou_matrix = final_features[
                    [*meta_cols, *score_cols, *ou_target_cols, *feature_cols]
                ].copy()
                feature_matrices["ou"] = ou_matrix

            # The opponent-adjusted family and the targets join AFTER the guard
            # above, so each final matrix is guarded again before it is returned.
            for matrix in feature_matrices.values():
                refuse_provenance_columns(
                    matrix, build_clock_columns=BUILD_CLOCK_COLUMNS
                )

            # -- THE ONE FINAL REFUSAL (Plan 33.2-20), after the post-Stage-1 merge and
            #    before any gold write --
            #
            # Two assertions, both over what is ABOUT TO BE WRITTEN rather than over an
            # intermediate frame:
            #
            #   1. the merge dispositions, per declared source, on each FINAL matrix --
            #      after _enforce_groups_dropped and every other column-removing step, so
            #      it certifies the matrices themselves. Registration proves nothing about
            #      arrival: combine_features has no generic loop, so a source registered
            #      but not merged passes the gate and is then silently dropped.
            #   2. the coverage report, which must show all TEN declared sources checked
            #      and nothing left in any of the three unchecked sets.
            #
            # There is no path that reports either failure without refusing.
            InformationTimeGate.assert_merge_dispositions(
                feature_matrices, self.merge_arrivals
            )
            coverage = self.information_time_coverage
            if coverage is None:
                msg = (
                    "no information-time CoverageReport exists for this build, so nothing "
                    "records which sources were checked. Refusing to write gold."
                )
                raise ProvenanceCoverageError(
                    msg, {"violation_type": "missing_coverage_report"}
                )
            InformationTimeGate.refuse_incomplete_coverage(coverage)

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


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (extracted so its argument wiring is unit-testable)."""
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
    # --through-season is the LADDER-RUNG form of the full rebuild (owner ruling
    # 2026-09-21, Plan 33.2-08). It is still a full rebuild -- same write path
    # (replace_mode=True), same gates -- bounded to seasons <= YEAR. Every p332_
    # rung (Plans 33.2-08 .. 33.2-19) passes 2025, so the unplayed 2026 season does
    # not enter gold as a second cause mid-ladder; Plan 33.2-20 is the one build
    # that adds 2026. It is never the default.
    parser.add_argument(
        "--through-season",
        type=int,
        metavar="YEAR",
        help=(
            "Full rebuild that stops at season YEAR: it REPLACES gold exactly like "
            "--all-seasons, through the same gates. The Phase-33.2 gold ladder rungs "
            "pass 2025 so 2026 cannot enter gold mid-ladder. Not combinable with "
            "--season or --week. Default: no bound."
        ),
    )
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and cross-validate the CLI arguments (*argv* defaults to ``sys.argv``)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # --all-seasons names the FULL rebuild, so it cannot also carry a week scope.
    # argparse's mutually-exclusive group already rejects --all-seasons --season.
    if args.all_seasons and args.week is not None:
        parser.error(
            "--all-seasons is the full historical rebuild and cannot be combined "
            "with --week. Use --season <YEAR> --week <N> for a scoped build."
        )
    # --through-season bounds the FULL rebuild; --season / --week select the scoped,
    # MERGING build, a different write mode, so the pair is refused, never guessed.
    if args.through_season is not None and (
        args.season is not None or args.week is not None
    ):
        parser.error(
            "--through-season bounds the full rebuild and cannot be combined with "
            "--season or --week (those select the SCOPED, merging build)."
        )
    return args


def main(argv: list[str] | None = None):
    """Build unified feature matrices."""
    args = parse_args(argv)

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

    logger.info(
        "Building unified feature matrices",
        season=args.season,
        week=args.week,
        through_season=args.through_season,
    )

    try:
        # Initialize feature matrix builder
        builder = FeatureMatrixBuilder()

        # Generate feature matrices
        feature_matrices = builder.generate_feature_matrices(
            target_season=args.season,
            target_week=args.week,
            as_of_datetime=as_of_dt,
            through_season=args.through_season,
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

        # THE COVERAGE THIS BUILD ACHIEVED, PRINTED (Plan 33.2-20). Five lines from the
        # FINAL CoverageReport, so a run's information-time coverage is a fact on stdout
        # rather than something to be inferred from a green exit.
        #
        # CHECKED_SOURCES is the NON-VACUOUS one: 10 means the nine registry keys AND the
        # opponent-adjusted family were each checked against every game's own lock.
        # INFORMATION_TIME_VIOLATIONS is only ever printed as 0, because a violation raises
        # long before this point -- it is the line that says so out loud rather than an
        # absence a reader has to interpret.
        # ``getattr`` because a test may hand ``main`` a recording stub rather than the
        # real builder. It is not a way for the real build to skip these lines: the
        # builder sets the attribute in its constructor, and Task 3's verify FAILS on an
        # ABSENT line as well as on a wrong one, so a build that printed nothing would be
        # caught rather than read as a pass.
        coverage = getattr(builder, "information_time_coverage", None)
        if coverage is not None:
            print(f"\nCHECKED_SOURCES= {len(coverage.checked_sources)}")
            print(f"EMPTY_UNCHECKED= {len(coverage.empty_unchecked_sources)}")
            print(f"UNREGISTERED= {len(coverage.unregistered_sources)}")
            print(f"POST_STAGE1_UNCHECKED= {len(coverage.post_stage1_sources)}")
            print("INFORMATION_TIME_VIOLATIONS= 0")
            print(f"CHECKED_SOURCE_NAMES= {sorted(coverage.checked_sources)}")

        logger.info("Feature matrix building completed successfully")

    except _SOURCE_LOAD_ERRORS as e:
        logger.error("Failed to build feature matrices", error=str(e))
        raise


if __name__ == "__main__":
    main()
