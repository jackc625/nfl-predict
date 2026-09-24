"""
Market Anchor Features Calculator

This module calculates five compressed market features per game from betting odds, each game
fenced at its own day-before-kickoff lock (D33.2-01): the snapshot spread and total, the
devigged home-win probability, and the opening-to-snapshot spread and total movements. No
closing line is ever used. Since Plan 33.2-19 none of them is a model input; the source stays
registered with the information-time gate.

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
    gold matrix, so nothing selected here is fitted on any more. What it still feeds is the
    R2 information-time gate that checks this source -- which is why the
    ``feature_sources["market"]`` registration was deliberately RETAINED when the merge
    seam was deleted. (The deprecated ``step_build_market_anchors`` path that wrote the
    unread silver ``market_anchor_features`` table is RETIRED, 33.2 review batch 3.) An
    input built from post-lock information is still a defect, so the lock fence below is
    NOT relaxed and must not be: "it is not a model input any more" is a reason to keep
    the fence, not to drop it.

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
    Calculate the five compressed market anchor features from betting odds data.

    Each game's lines are the odds rows captured at or before its own lock; from them come
    the snapshot spread/total, the devigged home-win probability and the opening-to-snapshot
    movements (:meth:`build_features`). The deprecated wide-output path
    (``build_market_anchor_features`` and its silver ``market_anchor_features`` table, read
    by no production code) is RETIRED (33.2 review batch 3, the retirement Plan 33.2-27
    routed here).
    """

    def __init__(self):
        """Initialize market anchor features calculator."""
        self.settings = get_settings()

        # Devig method from configuration
        self.devig_method = self.settings.config.models.features.market.devig_method

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

        # EACH BOOK is de-vigged on its own pair, and the median is taken over the resulting
        # PROBABILITIES (33.2 review B WR-12). The median of the raw American odds crossed the
        # +/-100 discontinuity -- -105 and +100 gave -2, "probability" 0.0196.
        priced = snapshot_lines.dropna(subset=["ml_home", "ml_away"])
        fair_by_book = [
            devig_probabilities(
                moneyline_to_probability(int(home)),
                moneyline_to_probability(int(away)),
                method=self.devig_method,
            )[0]
            for home, away in zip(priced["ml_home"], priced["ml_away"], strict=True)
        ]
        prob_home_fair = (
            float(pd.Series(fair_by_book).median()) if fair_by_book else float("nan")
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
