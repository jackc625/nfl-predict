"""
Market Anchor Features Calculator

This module calculates market-based features from betting odds:
- Opening line capture and storage (earliest available odds)
- Current line at each game's own day-before-kickoff lock (D33.2-01)
- Moneyline to probability conversions with devig
- Line movement tracking (opening vs snapshot)
- Market efficiency indicators
- STRICT no-leakage policy (no closing lines ever used)

Market anchors provide:
- Sharp money indicators (line movement patterns)
- Public vs sharp betting patterns
- Market inefficiency detection
- Baseline probabilities for model comparison

WHEN A LINE WAS KNOWN: ITS RECORDED CAPTURE TIME, NEVER ``snapshot_ts`` (owner ruling
2026-09-22, Plan 33.2-14, "Only real capture times"). A market line is admissible for a game
only when its row carries a GENUINELY RECORDED capture instant -- ``created_at``, which the
live capture path stamps with the moment the response was observed (``scripts/ingest_odds.py``)
-- at or before that game's lock (``utils.game_lock.is_admissible``: at-lock admissible, one
second later not). ``snapshot_ts`` is a LABEL and is never read as an information time: every
stored 2018-2024 row carries one manufactured constant per season (18:00 ET on September 19,
after 210 week-1/2 games had been played), 2025 carries the retired preceding-Friday freeze,
and the live path writes the game's own lock into it. Plan 33.2-08 nulled the 1970
``created_at`` family and the 2025 rows' ``created_at`` is the 2026-09-05 backfill, so no stored
2018-2025 line qualifies: those games are the honest unknown -- every market value NULL,
``basis="no_information"`` checked against a signature of NULLs, never a 0.0 or 0.5 stand-in.
Live 2026 lines captured before their lock are admitted normally.
"""

import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import pandas as pd

import utils.game_lock as lock_rule
from conf.settings import get_settings
from data.storage import load_dataframe
from features.provenance import PROVENANCE_COLUMNS, InformationBasis
from utils import get_logger
from utils.probability_utils import (
    devig_probabilities,
    moneyline_to_probability,
)

logger = get_logger(__name__)

#: The odds column that records WHEN a line was seen: the observed capture instant the live
#: capture path writes (``scripts/ingest_odds.py``). THE information time of a market line.
MARKET_CAPTURE_TIME_COLUMN: str = "created_at"

#: The compressed market columns the gold build merges (``scripts/build_features.py``).
MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
)


def admissible_market_rows(odds_df: pd.DataFrame, locks: pd.Series) -> pd.DataFrame:
    """The odds rows known at or before THEIR OWN game's lock, by recorded capture time.

    THIS SITE NO LONGER FEEDS ANY MODEL INPUT, AND THE FENCE STILL APPLIES (Plan
    33.2-19, p332_ rung 9, D33.2-03). Rung 9 removed the five market columns from every
    gold matrix, so nothing selected here is fitted on any more. What it still feeds is
    GRADING and CLV, through ``pipeline/steps.py::step_build_market_anchors`` and the
    R2 information-time gate that checks this source -- which is why the
    ``feature_sources["market"]`` registration was deliberately RETAINED when the merge
    seam was deleted. A grading input built from post-lock information is still a defect,
    so the lock fence below is NOT relaxed and must not be: "it is only used for grading"
    is a reason to keep the fence, not to drop it.

    A row is admitted only when (1) its game has a lock in *locks* and (2) its
    ``MARKET_CAPTURE_TIME_COLUMN`` value is present and ``utils.game_lock.is_admissible``
    against that lock -- the ONE admissibility rule, reached as a module attribute at call
    time. A NULL capture time is NOT admitted: an unknown time is never assumed early, and
    ``snapshot_ts`` is never consulted in its place (see the module docstring).

    Args:
        odds_df: Stored odds rows (``game_id``, ``sportsbook``, ``created_at``, line columns).
        locks: ``game_id`` -> tz-aware lock (``utils.game_lock.lock_frame``).

    Returns:
        The admitted rows, with ``information_time`` = the row's capture instant (tz-aware
        UTC). Empty, with that column, when nothing is admitted.

    Raises:
        ValueError: a capture time that carries no timezone (the strict parser refuses a
            naive instant rather than relabelling it).
    """
    if MARKET_CAPTURE_TIME_COLUMN not in odds_df.columns or len(odds_df) == 0:
        empty = odds_df.iloc[0:0].copy()
        empty["information_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        return empty

    lock_by_game = {str(game_id): lock for game_id, lock in locks.items()}
    admitted_positions: list[int] = []
    for position, (game_id, captured) in enumerate(
        zip(
            odds_df["game_id"].astype(str),
            odds_df[MARKET_CAPTURE_TIME_COLUMN],
            strict=True,
        )
    ):
        lock = lock_by_game.get(game_id)
        if lock is None or pd.isna(captured):
            continue
        if lock_rule.is_admissible(captured, lock):
            admitted_positions.append(position)

    admitted = odds_df.iloc[admitted_positions].copy()
    admitted["information_time"] = pd.to_datetime(
        admitted[MARKET_CAPTURE_TIME_COLUMN], utc=True
    )
    return admitted


def _as_float(value: object) -> float:
    """*value* as a float, NULL (NaN) when it is missing."""
    return float(cast("float", value)) if pd.notna(value) else float("nan")


def _difference(latest: object, earliest: object) -> float:
    """``latest - earliest`` when both are known, otherwise NULL (never a 0.0 stand-in)."""
    if pd.notna(latest) and pd.notna(earliest):
        return float(cast("float", latest)) - float(cast("float", earliest))
    return float("nan")


class MarketAnchorFeaturesCalculator:
    """
    Calculate market anchor features from betting odds data.

    Features calculated:
    - Opening line features (earliest available odds)
    - Snapshot line features (each game's own day-before-kickoff lock)
    - Line movement indicators (opening to snapshot)
    - Devigged probability calculations
    - Market efficiency metrics
    - Sharp vs public money indicators
    """

    def __init__(self):
        """Initialize market anchor features calculator."""
        self.settings = get_settings()

        # Devig method from configuration
        self.devig_method = self.settings.config.models.features.market.devig_method

        # Standard juice assumptions
        self.standard_juice = -110

        # Minimum time thresholds for opening lines
        self.min_opening_hours = 24  # Minimum hours before game for "opening" line
        self.snapshot_day = "Friday"  # Day of week for snapshot
        self.snapshot_hour = 18  # Hour (6 PM ET)

    def identify_opening_lines(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify opening lines from historical odds data.

        Opening lines are defined as the earliest available odds
        that are at least 24 hours before kickoff.

        Args:
            odds_df: DataFrame with historical odds

        Returns:
            DataFrame with opening line data
        """
        logger.info("Identifying opening lines", total_records=len(odds_df))

        try:
            # Load games data to get kickoff times
            games_df = load_dataframe("games", layer="silver")

            # Coerce snapshot_ts to a tz-aware (UTC) datetime. On-disk
            # odds_snapshot stores snapshot_ts as an ISO STRING (object dtype --
            # _normalize_parquet_datetime_columns string-formats it), so the raw
            # column cannot be subtracted from the tz-aware datetime kickoff_et.
            # The canonical build_features path already coerces (see
            # build_features below); this deprecated path did not, which crashed
            # build_market_anchors.py with "unsupported operand -: Timestamp and
            # str" for every season. (FIX-01, D-13)
            #
            # THROUGH THE ONE PARSE PATH, not a bare pd.to_datetime (WR-01). The bare
            # call infers its format from the FIRST element and coerces every row in
            # the other spelling to NaT: measured on live silver, 1,855 of 2,140 rows
            # (the "2024-09-19T18:00:00-04:00" spelling) were destroyed by the version
            # that used to be here. NaT then fails the >= min_opening_hours fence, so
            # 87% of games silently lost their odds and fell through to neutral market
            # defaults. build_features was fixed in Plan 31-xx; these two deprecated
            # methods were not, and pipeline/steps.py::step_build_market_anchors calls
            # them on every scheduled run.
            odds_df = odds_df.copy()
            odds_df["snapshot_ts"] = self._parse_snapshot_column(odds_df["snapshot_ts"])

            # Merge with games to get kickoff times
            odds_with_kickoff = odds_df.merge(
                games_df[["game_id", "kickoff_et"]], on="game_id", how="left"
            )

            # Calculate hours before kickoff (both operands tz-aware now)
            odds_with_kickoff["hours_before_kickoff"] = (
                odds_with_kickoff["kickoff_et"] - odds_with_kickoff["snapshot_ts"]
            ).dt.total_seconds() / 3600

            # Filter to odds that are at least min_opening_hours before kickoff
            early_odds = odds_with_kickoff[
                odds_with_kickoff["hours_before_kickoff"] >= self.min_opening_hours
            ].copy()

            if len(early_odds) == 0:
                logger.warning("No early odds found for opening line identification")
                return pd.DataFrame()

            # For each game and sportsbook, find the earliest (opening) line
            opening_lines = []

            for (game_id, sportsbook), group in early_odds.groupby(
                ["game_id", "sportsbook"]
            ):
                # Sort by snapshot time (earliest first)
                group_sorted = group.sort_values("snapshot_ts")
                opening_line = group_sorted.iloc[0]

                opening_lines.append(
                    {
                        "game_id": game_id,
                        "sportsbook": sportsbook,
                        "opening_snapshot_ts": opening_line["snapshot_ts"],
                        "hours_before_kickoff": opening_line["hours_before_kickoff"],
                        # Opening odds
                        "opening_ml_home": opening_line.get("ml_home"),
                        "opening_ml_away": opening_line.get("ml_away"),
                        "opening_spread": opening_line.get("spread"),
                        "opening_spread_ju_home": opening_line.get("spread_ju_home"),
                        "opening_spread_ju_away": opening_line.get("spread_ju_away"),
                        "opening_total": opening_line.get("total"),
                        "opening_total_over_ju": opening_line.get("total_over_ju"),
                        "opening_total_under_ju": opening_line.get("total_under_ju"),
                    }
                )

            opening_df = pd.DataFrame(opening_lines)

            logger.info(
                "Identified opening lines",
                games=opening_df["game_id"].nunique(),
                sportsbooks=opening_df["sportsbook"].nunique(),
                total_records=len(opening_df),
            )

            return opening_df

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to identify opening lines", error=str(e))
            raise

    def select_snapshot_lines_at_lock(
        self, odds_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Select each game's latest odds known at or before THAT game's own lock.

        The lock is ``utils.game_lock.lock_frame`` over the games' own kickoffs:
        18:00 ET on the ET calendar day before kickoff, at-lock admissible
        (D33.2-01). It is PER GAME. This replaces a selection that derived ONE
        Friday for the whole frame from the latest snapshot in it -- right for a
        single game-week and wrong for any frame spanning two, because a Thursday
        game and the following Sunday's games do not lock at the same instant.

        A row is known when it was CAPTURED (``created_at``), never when its
        ``snapshot_ts`` label says (owner ruling 2026-09-22, Plan 33.2-14): the same
        ``admissible_market_rows`` rule the gold builder uses, and the freshest
        admissible capture per sportsbook wins.

        Odds rows for games absent from *games_df* are not selected: a game with no
        schedule row has no kickoff and therefore no lock, and this method builds
        lines only for the games it was asked about.

        Args:
            odds_df: DataFrame with odds data (``game_id``, ``sportsbook``,
                ``snapshot_ts`` and the line columns).
            games_df: The games whose lines are wanted, carrying ``game_id`` and a
                tz-aware ``kickoff_et`` (silver ``games`` shape).

        Returns:
            DataFrame with one snapshot line per (game, sportsbook); its
            ``cutoff_time`` column is that game's own lock.

        Raises:
            utils.game_lock.MissingKickoffError: when a wanted game has no kickoff.
        """
        import utils.game_lock as lock_rule

        logger.info("Selecting snapshot lines at lock", total_records=len(odds_df))

        try:
            # Coerce snapshot_ts to a tz-aware (UTC) datetime -- on-disk
            # odds_snapshot stores it as an ISO string (object dtype), which
            # cannot drive .max()/.weekday() or a datetime comparison. Same
            # deprecated-path coercion gap as identify_opening_lines. (FIX-01)
            #
            # Through the ONE parse path for the same reason spelled out in
            # identify_opening_lines: a bare pd.to_datetime NaT'd 1,855 of 2,140 live
            # rows, and here a NaT also drops out of the <= cutoff comparison (WR-01).
            wanted_ids = sorted(set(games_df["game_id"].astype(str)))
            odds_df = cast(
                "pd.DataFrame",
                odds_df[odds_df["game_id"].astype(str).isin(wanted_ids)],
            ).copy()
            odds_df["snapshot_ts"] = self._parse_snapshot_column(odds_df["snapshot_ts"])

            # One lock per game, from the ONE rule. The lock is an ET instant, so
            # comparing it with the tz-aware (UTC) snapshot_ts compares instants:
            # an 18:00 ET lock is 22:00/23:00 UTC, never a UTC-localized 18:00
            # (the WR-02 lesson this method has carried since Phase 20).
            present = games_df[
                games_df["game_id"]
                .astype(str)
                .isin(sorted(set(odds_df["game_id"].astype(str))))
            ]
            locks = lock_rule.lock_frame(
                cast("pd.DataFrame", present[["game_id", "kickoff_et"]])
            )
            odds_df["cutoff_time"] = (
                odds_df["game_id"].astype(str).map(cast("Any", locks))
            )

            # Admitted by RECORDED CAPTURE TIME at or before the game's own lock
            # (at-lock admissible, one second later not); snapshot_ts is a label.
            pre_cutoff_odds = admissible_market_rows(odds_df, locks)

            if len(pre_cutoff_odds) == 0:
                logger.warning("No odds found before cutoff time")
                return pd.DataFrame()

            # For each game and sportsbook, find the latest line before cutoff
            snapshot_lines = []

            for (game_id, sportsbook), group in pre_cutoff_odds.groupby(
                ["game_id", "sportsbook"]
            ):
                # The freshest admissible CAPTURE (latest first)
                group_sorted = group.sort_values("information_time", ascending=False)
                snapshot_line = group_sorted.iloc[0]

                snapshot_lines.append(
                    {
                        "game_id": game_id,
                        "sportsbook": sportsbook,
                        "snapshot_ts": snapshot_line["snapshot_ts"],
                        "cutoff_time": snapshot_line["cutoff_time"],
                        # Snapshot odds
                        "snapshot_ml_home": snapshot_line.get("ml_home"),
                        "snapshot_ml_away": snapshot_line.get("ml_away"),
                        "snapshot_spread": snapshot_line.get("spread"),
                        "snapshot_spread_ju_home": snapshot_line.get("spread_ju_home"),
                        "snapshot_spread_ju_away": snapshot_line.get("spread_ju_away"),
                        "snapshot_total": snapshot_line.get("total"),
                        "snapshot_total_over_ju": snapshot_line.get("total_over_ju"),
                        "snapshot_total_under_ju": snapshot_line.get("total_under_ju"),
                    }
                )

            snapshot_df = pd.DataFrame(snapshot_lines)

            logger.info(
                "Selected snapshot lines at lock",
                games=snapshot_df["game_id"].nunique(),
                sportsbooks=snapshot_df["sportsbook"].nunique(),
                total_records=len(snapshot_df),
            )

            return snapshot_df

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to select snapshot lines at lock", error=str(e))
            raise

    def calculate_devigged_probabilities(
        self, odds_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate devigged probabilities from odds.

        Args:
            odds_data: Dictionary with odds data

        Returns:
            Dictionary with devigged probabilities
        """
        try:
            probabilities = {}

            # Moneyline probabilities
            ml_home = odds_data.get("ml_home")
            ml_away = odds_data.get("ml_away")

            if ml_home is not None and ml_away is not None:
                prob_home_raw = moneyline_to_probability(ml_home)
                prob_away_raw = moneyline_to_probability(ml_away)

                # Devig moneyline probabilities
                prob_home_fair, prob_away_fair = devig_probabilities(
                    prob_home_raw, prob_away_raw, method=self.devig_method
                )

                probabilities.update(
                    {
                        "ml_prob_home_raw": prob_home_raw,
                        "ml_prob_away_raw": prob_away_raw,
                        "ml_prob_home_fair": prob_home_fair,
                        "ml_prob_away_fair": prob_away_fair,
                        "ml_vig": (prob_home_raw + prob_away_raw) - 1.0,
                    }
                )

            # Spread probabilities
            spread_ju_home = odds_data.get("spread_ju_home", self.standard_juice)
            spread_ju_away = odds_data.get("spread_ju_away", self.standard_juice)

            if spread_ju_home is not None and spread_ju_away is not None:
                spread_prob_home_raw = moneyline_to_probability(spread_ju_home)
                spread_prob_away_raw = moneyline_to_probability(spread_ju_away)

                # Devig spread probabilities
                spread_prob_home_fair, spread_prob_away_fair = devig_probabilities(
                    spread_prob_home_raw, spread_prob_away_raw, method=self.devig_method
                )

                probabilities.update(
                    {
                        "spread_prob_home_raw": spread_prob_home_raw,
                        "spread_prob_away_raw": spread_prob_away_raw,
                        "spread_prob_home_fair": spread_prob_home_fair,
                        "spread_prob_away_fair": spread_prob_away_fair,
                        "spread_vig": (spread_prob_home_raw + spread_prob_away_raw)
                        - 1.0,
                    }
                )

            # Total probabilities
            total_over_ju = odds_data.get("total_over_ju", self.standard_juice)
            total_under_ju = odds_data.get("total_under_ju", self.standard_juice)

            if total_over_ju is not None and total_under_ju is not None:
                total_prob_over_raw = moneyline_to_probability(total_over_ju)
                total_prob_under_raw = moneyline_to_probability(total_under_ju)

                # Devig total probabilities
                total_prob_over_fair, total_prob_under_fair = devig_probabilities(
                    total_prob_over_raw, total_prob_under_raw, method=self.devig_method
                )

                probabilities.update(
                    {
                        "total_prob_over_raw": total_prob_over_raw,
                        "total_prob_under_raw": total_prob_under_raw,
                        "total_prob_over_fair": total_prob_over_fair,
                        "total_prob_under_fair": total_prob_under_fair,
                        "total_vig": (total_prob_over_raw + total_prob_under_raw) - 1.0,
                    }
                )

            return probabilities

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to calculate devigged probabilities", error=str(e))
            return {}

    def calculate_line_movement(
        self, opening_data: dict[str, Any], snapshot_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate line movement from opening to snapshot.

        Args:
            opening_data: Opening line data
            snapshot_data: Snapshot line data

        Returns:
            Dictionary with line movement features
        """
        try:
            movement = {}

            # Moneyline movement
            opening_ml_home = opening_data.get("opening_ml_home")
            snapshot_ml_home = snapshot_data.get("snapshot_ml_home")

            if opening_ml_home is not None and snapshot_ml_home is not None:
                ml_movement = snapshot_ml_home - opening_ml_home
                movement["ml_home_movement"] = ml_movement
                movement["ml_home_moved_toward_favorite"] = (
                    1.0 if ml_movement < 0 else 0.0
                )

            opening_ml_away = opening_data.get("opening_ml_away")
            snapshot_ml_away = snapshot_data.get("snapshot_ml_away")

            if opening_ml_away is not None and snapshot_ml_away is not None:
                ml_movement_away = snapshot_ml_away - opening_ml_away
                movement["ml_away_movement"] = ml_movement_away

            # Spread movement
            opening_spread = opening_data.get("opening_spread")
            snapshot_spread = snapshot_data.get("snapshot_spread")

            if opening_spread is not None and snapshot_spread is not None:
                spread_movement = snapshot_spread - opening_spread
                movement["spread_movement"] = spread_movement
                movement["spread_moved_toward_home"] = (
                    1.0 if spread_movement < 0 else 0.0
                )
                movement["spread_moved_toward_away"] = (
                    1.0 if spread_movement > 0 else 0.0
                )
                movement["spread_moved"] = 1.0 if abs(spread_movement) > 0.5 else 0.0

            # Total movement
            opening_total = opening_data.get("opening_total")
            snapshot_total = snapshot_data.get("snapshot_total")

            if opening_total is not None and snapshot_total is not None:
                total_movement = snapshot_total - opening_total
                movement["total_movement"] = total_movement
                movement["total_moved_up"] = 1.0 if total_movement > 0 else 0.0
                movement["total_moved_down"] = 1.0 if total_movement < 0 else 0.0
                movement["total_moved"] = 1.0 if abs(total_movement) > 0.5 else 0.0

            # Overall movement indicators
            movements = [
                movement.get("ml_home_movement", 0),
                movement.get("spread_movement", 0),
                movement.get("total_movement", 0),
            ]

            significant_movements = sum(1 for m in movements if abs(m) > 0.5)
            movement["significant_line_movement"] = (
                1.0 if significant_movements >= 2 else 0.0
            )

            return movement

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to calculate line movement", error=str(e))
            return {}

    def calculate_market_efficiency_indicators(
        self, opening_probs: dict[str, float], snapshot_probs: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate market efficiency indicators.

        Args:
            opening_probs: Opening line probabilities
            snapshot_probs: Snapshot line probabilities

        Returns:
            Dictionary with market efficiency features
        """
        try:
            efficiency = {}

            # Probability movement (indication of information flow)
            for market in ["ml", "spread", "total"]:
                opening_home_key = f"{market}_prob_home_fair"
                snapshot_home_key = f"{market}_prob_home_fair"

                if (
                    opening_home_key in opening_probs
                    and snapshot_home_key in snapshot_probs
                ):
                    prob_change = abs(
                        snapshot_probs[snapshot_home_key]
                        - opening_probs[opening_home_key]
                    )
                    efficiency[f"{market}_prob_movement"] = prob_change

            # Vig comparison (market competition)
            for market in ["ml", "spread", "total"]:
                opening_vig_key = f"{market}_vig"
                snapshot_vig_key = f"{market}_vig"

                if (
                    opening_vig_key in opening_probs
                    and snapshot_vig_key in snapshot_probs
                ):
                    vig_change = (
                        snapshot_probs[snapshot_vig_key]
                        - opening_probs[opening_vig_key]
                    )
                    efficiency[f"{market}_vig_change"] = vig_change
                    efficiency[f"{market}_vig_increased"] = (
                        1.0 if vig_change > 0 else 0.0
                    )

            # Sharp money indicators (reverse line movement)
            # This occurs when the line moves against public betting patterns
            if "ml_home_movement" in efficiency and "spread_movement" in efficiency:
                # If moneyline and spread move in opposite directions, it may indicate sharp action
                ml_mov = efficiency.get("ml_home_movement", 0)
                spread_mov = efficiency.get("spread_movement", 0)

                if ml_mov * spread_mov < 0:  # Opposite directions
                    efficiency["reverse_line_movement"] = 1.0
                else:
                    efficiency["reverse_line_movement"] = 0.0

            return efficiency

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(
                "Failed to calculate market efficiency indicators", error=str(e)
            )
            return {}

    def create_consensus_lines(self, lines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create consensus lines from multiple sportsbooks.

        Args:
            lines_df: DataFrame with lines from multiple sportsbooks

        Returns:
            DataFrame with consensus lines per game
        """
        logger.info("Creating consensus lines", total_records=len(lines_df))

        try:
            consensus_lines = []

            # The lines frame is the output of identify_opening_lines /
            # select_snapshot_lines_at_lock, which prefix every odds column with
            # ``opening_`` or ``snapshot_`` (e.g. ``opening_ml_home``,
            # ``snapshot_spread``). The original code read the bare ``ml_home`` /
            # ``ml_away`` / ``spread`` / ``total`` names that no longer exist after
            # that upstream rename, raising KeyError 'ml_home' for every call. Detect
            # the present prefix and read the columns that actually exist. (D-11-E)
            prefix = (
                "opening_" if "opening_ml_home" in lines_df.columns else "snapshot_"
            )
            ml_home_col = f"{prefix}ml_home"
            ml_away_col = f"{prefix}ml_away"
            spread_col = f"{prefix}spread"
            total_col = f"{prefix}total"

            for game_id, group in lines_df.groupby("game_id"):
                # Calculate consensus for each market
                consensus = {"game_id": game_id}

                # Moneyline consensus (median)
                ml_home_values = group[ml_home_col].dropna()
                ml_away_values = group[ml_away_col].dropna()

                if len(ml_home_values) > 0:
                    consensus["consensus_ml_home"] = ml_home_values.median()
                if len(ml_away_values) > 0:
                    consensus["consensus_ml_away"] = ml_away_values.median()

                # Spread consensus (median)
                spread_values = group[spread_col].dropna()
                if len(spread_values) > 0:
                    consensus["consensus_spread"] = spread_values.median()

                # Total consensus (median)
                total_values = group[total_col].dropna()
                if len(total_values) > 0:
                    consensus["consensus_total"] = total_values.median()

                # Count of sportsbooks
                consensus["num_sportsbooks"] = len(group)

                # Range indicators (market disagreement)
                if len(spread_values) > 1:
                    consensus["spread_range"] = (
                        spread_values.max() - spread_values.min()
                    )
                if len(total_values) > 1:
                    consensus["total_range"] = total_values.max() - total_values.min()

                consensus_lines.append(consensus)

            consensus_df = pd.DataFrame(consensus_lines)

            logger.info(
                "Created consensus lines",
                games=len(consensus_df),
                avg_sportsbooks=consensus_df["num_sportsbooks"].mean(),
            )

            return consensus_df

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to create consensus lines", error=str(e))
            raise

    def build_market_anchor_features(
        self,
        games_df: pd.DataFrame,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """
        Build market anchor features for all games (full/uncompressed output).

        .. deprecated::
            Use :meth:`build_features` instead (conforms to FeatureBuilder Protocol).

        Args:
            games_df: DataFrame with game information
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for

        Returns:
            DataFrame with market anchor features
        """
        warnings.warn(
            "build_market_anchor_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building market anchor features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load odds data
            odds_df = load_dataframe("odds_snapshot", layer="silver")
            logger.info("Loaded odds data", odds_records=len(odds_df))

            # Filter to target if specified
            if target_season and target_week:
                # Filter games
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

                # Filter odds data to matching games
                game_ids = games_df["game_id"].tolist()
                odds_df = odds_df[odds_df["game_id"].isin(game_ids)]

            if len(odds_df) == 0:
                logger.warning("No odds data available for market anchor features")
                # Return empty features
                return self._create_empty_market_features(games_df)

            # Identify opening and snapshot lines
            # Only lines CAPTURED at or before their game's lock may open a market,
            # by the same rule the snapshot selection and the gold builder use.
            wanted = games_df[
                games_df["game_id"]
                .astype(str)
                .isin(set(odds_df["game_id"].astype(str)))
            ]
            admitted_odds = admissible_market_rows(
                odds_df,
                lock_rule.lock_frame(
                    cast("pd.DataFrame", wanted[["game_id", "kickoff_et"]])
                ),
            ).drop(columns=["information_time"])
            opening_lines_df = self.identify_opening_lines(admitted_odds)
            snapshot_lines_df = self.select_snapshot_lines_at_lock(odds_df, games_df)

            # Create consensus lines
            if len(opening_lines_df) > 0:
                opening_consensus_df = self.create_consensus_lines(opening_lines_df)
            else:
                opening_consensus_df = pd.DataFrame()

            if len(snapshot_lines_df) > 0:
                snapshot_consensus_df = self.create_consensus_lines(snapshot_lines_df)
            else:
                snapshot_consensus_df = pd.DataFrame()

            # Build features for each game
            market_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                season = game["season"]
                week = game["week"]

                # Basic game identifiers
                game_features = {"game_id": game_id, "season": season, "week": week}

                # Get opening consensus data
                opening_data = {}
                if len(opening_consensus_df) > 0:
                    opening_game = opening_consensus_df[
                        opening_consensus_df["game_id"] == game_id
                    ]
                    if len(opening_game) > 0:
                        opening_data = opening_game.iloc[0].to_dict()

                # Get snapshot consensus data
                snapshot_data = {}
                if len(snapshot_consensus_df) > 0:
                    snapshot_game = snapshot_consensus_df[
                        snapshot_consensus_df["game_id"] == game_id
                    ]
                    if len(snapshot_game) > 0:
                        snapshot_data = snapshot_game.iloc[0].to_dict()

                # Calculate features if we have market data
                if opening_data or snapshot_data:
                    # Opening line features
                    if opening_data:
                        opening_probs = self.calculate_devigged_probabilities(
                            opening_data
                        )
                        for key, value in opening_probs.items():
                            game_features[f"opening_{key}"] = value

                        # Add raw opening odds
                        game_features.update(
                            {
                                "opening_ml_home": opening_data.get(
                                    "consensus_ml_home"
                                ),
                                "opening_ml_away": opening_data.get(
                                    "consensus_ml_away"
                                ),
                                "opening_spread": opening_data.get("consensus_spread"),
                                "opening_total": opening_data.get("consensus_total"),
                                "opening_num_sportsbooks": opening_data.get(
                                    "num_sportsbooks", 0
                                ),
                            }
                        )

                    # Snapshot line features
                    if snapshot_data:
                        snapshot_probs = self.calculate_devigged_probabilities(
                            snapshot_data
                        )
                        for key, value in snapshot_probs.items():
                            game_features[f"snapshot_{key}"] = value

                        # Add raw snapshot odds
                        game_features.update(
                            {
                                "snapshot_ml_home": snapshot_data.get(
                                    "consensus_ml_home"
                                ),
                                "snapshot_ml_away": snapshot_data.get(
                                    "consensus_ml_away"
                                ),
                                "snapshot_spread": snapshot_data.get(
                                    "consensus_spread"
                                ),
                                "snapshot_total": snapshot_data.get("consensus_total"),
                                "snapshot_num_sportsbooks": snapshot_data.get(
                                    "num_sportsbooks", 0
                                ),
                            }
                        )

                    # Line movement features
                    if opening_data and snapshot_data:
                        movement_features = self.calculate_line_movement(
                            opening_data, snapshot_data
                        )
                        game_features.update(movement_features)

                        # Market efficiency features
                        if "opening_probs" in locals() and "snapshot_probs" in locals():
                            efficiency_features = (
                                self.calculate_market_efficiency_indicators(
                                    opening_probs, snapshot_probs
                                )
                            )
                            game_features.update(efficiency_features)

                    # Market availability indicators
                    game_features.update(
                        {
                            "has_opening_lines": 1.0 if opening_data else 0.0,
                            "has_snapshot_lines": 1.0 if snapshot_data else 0.0,
                            "has_line_movement": 1.0
                            if (opening_data and snapshot_data)
                            else 0.0,
                        }
                    )

                else:
                    # No market data available - set defaults
                    game_features.update(self._default_market_features())

                market_features.append(game_features)

            # Convert to DataFrame
            features_df = pd.DataFrame(market_features)

            logger.info(
                "Built market anchor features",
                features_count=len(features_df),
                games_with_lines=int(features_df["has_snapshot_lines"].sum()),
                games_with_movement=int(features_df["has_line_movement"].sum()),
            )

            return features_df

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error("Failed to build market anchor features", error=str(e))
            raise

    def _create_empty_market_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create market features with default values when no odds data available."""
        empty_features = []

        for _, game in games_df.iterrows():
            game_features = {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
            }
            game_features.update(self._default_market_features())
            empty_features.append(game_features)

        return pd.DataFrame(empty_features)

    def _default_market_features(self) -> dict[str, float]:
        """Return default market features when no odds data is available."""
        return {
            # Market availability
            "has_opening_lines": 0.0,
            "has_snapshot_lines": 0.0,
            "has_line_movement": 0.0,
            # Default probabilities (50/50)
            "snapshot_ml_prob_home_fair": 0.5,
            "snapshot_ml_prob_away_fair": 0.5,
            "snapshot_spread_prob_home_fair": 0.5,
            "snapshot_spread_prob_away_fair": 0.5,
            "snapshot_total_prob_over_fair": 0.5,
            "snapshot_total_prob_under_fair": 0.5,
            # No movement
            "ml_home_movement": 0.0,
            "spread_movement": 0.0,
            "total_movement": 0.0,
            "spread_moved": 0.0,
            "total_moved": 0.0,
            "significant_line_movement": 0.0,
            # Default market efficiency
            "reverse_line_movement": 0.0,
            # Default vig
            "snapshot_ml_vig": 0.05,
            "snapshot_spread_vig": 0.05,
            "snapshot_total_vig": 0.05,
        }

    def validate_market_anchor_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate market anchor features for data quality.

        Args:
            features_df: Market anchor features DataFrame

        Returns:
            True if validation passes
        """
        if len(features_df) == 0:
            logger.error("No market anchor features found")
            return False

        # Check for required columns
        required_cols = ["game_id", "season", "week", "has_snapshot_lines"]
        missing_cols = set(required_cols) - set(features_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check probability ranges (should be 0-1)
        prob_cols = [
            col for col in features_df.columns if "prob" in col and "fair" in col
        ]
        for col in prob_cols:
            if col in features_df.columns:
                probs = features_df[col].dropna()
                if len(probs) > 0:
                    if probs.min() < 0 or probs.max() > 1:
                        logger.warning(
                            f"Probabilities outside 0-1 range for {col}",
                            min_prob=probs.min(),
                            max_prob=probs.max(),
                        )

        # Check spread ranges (should be reasonable)
        spread_cols = [
            col
            for col in features_df.columns
            if "spread" in col and "movement" not in col
        ]
        for col in spread_cols:
            if col in features_df.columns:
                spreads = features_df[col].dropna()
                if len(spreads) > 0:
                    if spreads.min() < -30 or spreads.max() > 30:
                        logger.warning(
                            f"Spreads outside reasonable range for {col}",
                            min_spread=spreads.min(),
                            max_spread=spreads.max(),
                        )

        # Check total ranges (should be 20-80 typically)
        total_cols = [
            col
            for col in features_df.columns
            if "total" in col and "movement" not in col and "prob" not in col
        ]
        for col in total_cols:
            if col in features_df.columns:
                totals = features_df[col].dropna()
                if len(totals) > 0:
                    if totals.min() < 15 or totals.max() > 100:
                        logger.warning(
                            f"Totals outside reasonable range for {col}",
                            min_total=totals.min(),
                            max_total=totals.max(),
                        )

        # Check that no closing lines are present
        closing_cols = [col for col in features_df.columns if "closing" in col.lower()]
        if closing_cols:
            logger.error(
                "CRITICAL: Closing line data detected - violates no-leakage policy",
                closing_columns=closing_cols,
            )
            return False

        # Summary statistics
        games_with_lines = int(features_df["has_snapshot_lines"].sum())
        games_with_movement = int(features_df["has_line_movement"].sum())

        logger.info(
            "Market anchor features validation completed",
            records=len(features_df),
            games_with_lines=games_with_lines,
            games_with_movement=games_with_movement,
            market_columns=len(
                [
                    col
                    for col in features_df.columns
                    if col not in ["game_id", "season", "week"]
                ]
            ),
        )

        return True

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods (compressed output)
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        lock_frame: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build compressed market anchor features (5 features), each game at its OWN lock.

        Conforms to the FeatureBuilder Protocol. A game's lines are the odds rows
        CAPTURED at or before that game's lock (``admissible_market_rows``: the recorded
        ``created_at`` against ``utils.game_lock``, at-lock admissible). ``snapshot_ts`` is a
        label and is never read here (owner ruling 2026-09-22, Plan 33.2-14). A game with
        no admitted line is the honest unknown: every market value NULL.

        AFTER PLAN 33.2-19 THIS STOPS FEEDING A MODEL INPUT (D33.2-03: no betting line is a
        model input) and becomes grading / CLV only. The fence still applies then: a
        grading input built from post-lock information is still a defect.

        Output columns:
            game_id, snapshot_spread, snapshot_total,
            snapshot_ml_prob_home_fair, spread_movement, total_movement

        Args:
            games_df: DataFrame of games to build features for (``game_id`` and a
                tz-aware ``kickoff_et``).
            as_of_datetime: Carried for the ``FeatureBuilder`` Protocol ONLY. It is NOT a
                fence and no selection reads it.
            target_season: Optional season filter.
            target_week: Optional week filter.
            lock_frame: The build's ``game_id`` -> lock frame, built ONCE by the caller.
                When ``None`` it is built here from the target games.

        Returns:
            DataFrame with exactly 6 columns (game_id + 5 features).

        Raises:
            utils.game_lock.MissingKickoffError: a target game has no kickoff.
        """
        del as_of_datetime  # the Protocol's argument; every game is fenced at its lock
        logger.info(
            "Building compressed market anchor features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            if target_season and target_week:
                games_df = cast(
                    "pd.DataFrame",
                    games_df[
                        (games_df["season"] == target_season)
                        & (games_df["week"] == target_week)
                    ],
                ).copy()

            admitted = self._admitted_for(games_df, lock_frame)

            compressed_rows: list[dict[str, object]] = []
            # Games with no line captured at or before their lock, COUNTED. The unknown is
            # a NULL, which no reader can mistake for a real line -- but a reader should
            # still learn how many games carried none.
            unknown: list[str] = []
            admitted_by_game = {
                str(game_id): rows for game_id, rows in admitted.groupby("game_id")
            }

            for game_id in games_df["game_id"]:
                game_odds = admitted_by_game.get(str(game_id))
                if game_odds is None or len(game_odds) == 0:
                    unknown.append(str(game_id))
                    compressed_rows.append(
                        self._default_compressed_market_features(game_id)
                    )
                    continue
                compressed_rows.append(self._compress_game_lines(game_id, game_odds))

            features_df = pd.DataFrame(
                compressed_rows, columns=["game_id", *MARKET_FEATURE_COLUMNS]
            )

            if unknown:
                seasons = sorted(
                    {str(gid).split("_")[0] for gid in unknown if "_" in str(gid)}
                )
                logger.info(
                    "Market anchors are the honest unknown for games with no line "
                    "captured at or before their lock",
                    games_unknown=len(unknown),
                    games_total=len(features_df),
                    seasons_affected=seasons,
                    first_examples=unknown[:5],
                )

            logger.info(
                "Built compressed market anchor features",
                features_count=len(features_df),
                games_unknown=len(unknown),
            )

            return features_df

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(
                "Failed to build compressed market anchor features",
                error=str(e),
            )
            raise

    def _admitted_for(
        self, games_df: pd.DataFrame, lock_frame: pd.Series | None
    ) -> pd.DataFrame:
        """The odds rows admitted for *games_df*, memoised for ``information_times``.

        The provenance supplier re-reads THIS frame (what the features were computed from)
        rather than re-deriving a selection, so it reports what was actually admitted.
        """
        if len(games_df) == 0:
            self._admitted = pd.DataFrame(columns=["game_id", "information_time"])
            self._admitted_game_ids: frozenset[str] = frozenset()
            return self._admitted

        odds_df = load_dataframe("odds_snapshot", layer="silver")
        logger.info("Loaded odds data", odds_records=len(odds_df))
        wanted = {str(g) for g in games_df["game_id"]}
        odds_df = cast(
            "pd.DataFrame",
            odds_df[odds_df["game_id"].astype(str).isin(sorted(wanted))],
        )
        locks = (
            lock_frame
            if lock_frame is not None
            else lock_rule.lock_frame(
                cast("pd.DataFrame", games_df[["game_id", "kickoff_et"]])
            )
        )
        self._admitted = admissible_market_rows(odds_df, locks)
        self._admitted_game_ids = frozenset(wanted)
        return self._admitted

    def _compress_game_lines(
        self, game_id: object, game_odds: pd.DataFrame
    ) -> dict[str, object]:
        """One game's five market values from its ADMITTED lines (consensus across books).

        Opening line: the earliest admitted capture per sportsbook. Snapshot line: the
        freshest admitted capture per sportsbook. A value the admitted lines cannot supply
        is NULL -- never a 0.5 probability or a 0.0 movement stand-in.
        """
        opening_lines = (
            game_odds.sort_values("information_time")
            .groupby("sportsbook")
            .first()
            .reset_index()
        )
        snapshot_lines = (
            game_odds.sort_values("information_time", ascending=False)
            .groupby("sportsbook")
            .first()
            .reset_index()
        )

        snap_spread = snapshot_lines["spread"].dropna().median()
        snap_total = snapshot_lines["total"].dropna().median()
        open_spread = opening_lines["spread"].dropna().median()
        open_total = opening_lines["total"].dropna().median()

        prob_home_fair = float("nan")
        snap_ml_home_vals = snapshot_lines["ml_home"].dropna()
        snap_ml_away_vals = snapshot_lines["ml_away"].dropna()
        if len(snap_ml_home_vals) > 0 and len(snap_ml_away_vals) > 0:
            prob_home_raw = moneyline_to_probability(int(snap_ml_home_vals.median()))
            prob_away_raw = moneyline_to_probability(int(snap_ml_away_vals.median()))
            prob_home_fair, _ = devig_probabilities(
                prob_home_raw, prob_away_raw, method=self.devig_method
            )

        return {
            "game_id": game_id,
            "snapshot_spread": _as_float(snap_spread),
            "snapshot_total": _as_float(snap_total),
            "snapshot_ml_prob_home_fair": float(prob_home_fair),
            "spread_movement": _difference(snap_spread, open_spread),
            "total_movement": _difference(snap_total, open_total),
        }

    # ------------------------------------------------------------------
    # InformationTimeProvider (features.protocol, Plan 33.2-01's owned contract)
    # ------------------------------------------------------------------

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game with no line captured at or before its lock: every market value NULL."""
        return dict.fromkeys(MARKET_FEATURE_COLUMNS)

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """One provenance row per game: the latest CAPTURE among the lines it admitted.

        Read from the admitted frame :meth:`build_features` computed its values from (the
        memo), re-derived through the same ``admissible_market_rows`` rule only when called
        for games that build did not cover. The time is always a recorded ``created_at``,
        never a ``snapshot_ts`` label. A game that admitted no line is
        ``basis="no_information"`` with a NULL time, value-checked against
        :meth:`no_information_signature`.

        Returns:
            A frame with exactly ``PROVENANCE_COLUMNS``.
        """
        target = games_df
        if target_season and target_week:
            target = cast(
                "pd.DataFrame",
                games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ],
            )
        wanted = {str(g) for g in target["game_id"]}
        memo_ids: frozenset[str] = getattr(self, "_admitted_game_ids", frozenset())
        admitted = (
            self._admitted
            if memo_ids and wanted <= memo_ids
            else self._admitted_for(target, None)
        )
        latest: dict[str, pd.Timestamp] = {}
        if len(admitted) > 0:
            by_game = admitted.groupby(admitted["game_id"].astype(str))
            latest = {
                str(game_id): pd.Timestamp(cast("Any", when))
                for game_id, when in by_game["information_time"].max().items()
            }
        records: list[dict[str, Any]] = []
        for game_id in target["game_id"]:
            when = latest.get(str(game_id))
            records.append(
                {
                    "game_id": str(game_id),
                    "basis": (
                        InformationBasis.PER_ROW.value
                        if when is not None
                        else InformationBasis.NO_INFORMATION.value
                    ),
                    "information_time": when,
                }
            )
        return pd.DataFrame(records, columns=list(PROVENANCE_COLUMNS))

    @staticmethod
    def _parse_snapshot_column(column: pd.Series) -> pd.Series:
        """Parse a stored ``snapshot_ts`` column to tz-aware UTC, REFUSING the unparseable.

        Delegates every value to
        ``scripts.ingest_historical_odds.normalize_snapshot_ts`` -- the ONE parse path
        clause 3's type trap requires, which anchors a naive value in Eastern rather than
        UTC and handles both spellings the live column holds.

        A value that cannot be parsed RAISES, naming the offending values. It is never
        coerced to NaT: a NaT fails the fence comparison, and a row silently dropped from
        the fence becomes a game with no odds and a neutral-default market anchor. That
        is precisely the failure this method exists to end, and it is a failure that
        looks exactly like "this game had no line".

        Args:
            column: The stored ``snapshot_ts`` values, in any shape the column holds.

        Returns:
            A tz-aware UTC ``datetime64`` series, index-aligned with *column*.

        Raises:
            ValueError: naming every distinct unparseable value.
        """
        from scripts.ingest_historical_odds import normalize_snapshot_ts

        parsed: list[Any] = []
        unparseable: dict[str, str] = {}
        for value in column:
            try:
                parsed.append(normalize_snapshot_ts(value))
            except (ValueError, TypeError) as error:
                unparseable.setdefault(repr(value), str(error))
                parsed.append(pd.NaT)

        if unparseable:
            detail = "; ".join(f"{value}: {why}" for value, why in unparseable.items())
            msg = (
                f"{len(unparseable)} distinct snapshot_ts value(s) could not be parsed, "
                "so the market-anchor time fence cannot be applied to them. They are NOT "
                "coerced to NaT: a NaT fails the fence, the game loses its odds, and its "
                f"market anchors silently become the neutral default. Offenders: {detail}"
            )
            raise ValueError(msg)

        return pd.to_datetime(pd.Series(parsed, index=column.index), utc=True)

    def _default_compressed_market_features(self, game_id: object) -> dict[str, object]:
        """A game with no line captured at or before its lock: the HONEST UNKNOWN.

        Every value NULL, exactly ``no_information_signature``. This used to be a 0.5
        probability and 0.0 movements, stand-ins a reader could not tell from a real
        even-money line that never moved (owner ruling 2026-09-22).
        """
        return {
            "game_id": game_id,
            **dict.fromkeys(MARKET_FEATURE_COLUMNS, float("nan")),
        }

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get compressed market features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Carried for the Protocol; not a fence (the game's lock is).

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            # The game's lock needs its kickoff, so the game is read from silver games; a
            # kickoff-less stand-in row would have no lock and is refused by the rule.
            games = load_dataframe("games", layer="silver")
            games_df = cast(
                "pd.DataFrame", games[games["game_id"].astype(str) == str(game_id)]
            )
            if len(games_df) == 0:
                logger.warning("No schedule row for game", game_id=game_id)
                return {}
            result = self.build_features(games_df, as_of_datetime)

            if len(result) == 0:
                logger.warning(
                    "No market features for game",
                    game_id=game_id,
                )
                return {}

            row = result.iloc[0]
            return {col: float(row[col]) for col in result.columns if col != "game_id"}

        except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.error(
                "Failed to get market features for game",
                game_id=game_id,
                error=str(e),
            )
            return {}
