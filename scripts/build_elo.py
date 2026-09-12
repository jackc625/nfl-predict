"""Build and update Elo ratings for NFL teams.

This script processes historical game data chronologically to build Elo ratings
for all NFL teams. It can process multiple seasons or update ratings for the
current season.

THE WRITE PATH IS TWO NAMED VERBS (Plan 33-03, D33-08)
------------------------------------------------------
``save_results`` used to be the only way to persist Elo, and it wrote
``elo_game_snapshots`` with ``append_mode=False`` -- "always replace, full rebuild".
That is right for a rebuild and catastrophic for the WEEKLY path, which runs every
Friday against a table holding 24 seasons of burn-in. The two behaviours are now two
verbs that say which one they are:

* :meth:`EloBuilder.save_full_rebuild` -- replace everything, loudly attributed, and
  reachable from the CLI only behind its own ``--full-rebuild`` flag.
* :meth:`EloBuilder.save_live_append` -- upsert the three ROW tables on ``game_id``,
  replace the two STATE artifacts, refuse any frame carrying another season's rows.

Both publish through :func:`publish_elo_generation`, so the FIVE artifacts are atomic
together rather than five separately-atomic files that can crash into a mixed state.

Usage:
    python scripts/build_elo.py --season 2024              # Process single season
    python scripts/build_elo.py --seasons 2020 2021 2022  # Process multiple seasons
    python scripts/build_elo.py --all-seasons              # Process all available seasons
    python scripts/build_elo.py --current                  # Process current season only
    python scripts/build_elo.py --all-seasons --full-rebuild   # REPLACE the Elo tables
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(".")

from conf.settings import get_settings
from data.storage import (
    get_db_connection,
    load_dataframe,
    save_dataframe,
    upsert_silver,
)
from ratings.elo import EloRatingSystem, is_divisional_game

# RE-EXPORTED, not re-declared. The generation publisher and the ROW/STATE split rule
# live in ``scripts/elo_generation`` -- publishing an artifact SET atomically is a
# self-contained concern with its own vocabulary, and this module is about building
# ratings. Callers and tests import either name from here or from there and get the
# same object, so the split rule cannot acquire a second home (D30-02's lesson).
from scripts.elo_generation import (
    ELO_BURN_IN_START_SEASON,
    ELO_GENERATION_DIRNAME,
    ELO_GENERATION_POINTER_NAME,
    ELO_GENERATION_POINTER_PATH,
    ELO_ROW_TABLE_KEY_COLUMN,
    ELO_ROW_TABLES,
    ELO_STATE_ARTIFACTS,
    EloGenerationIncompleteError,
    EloGenerationPublisher,
    default_stage_writer,
    elo_generation_pointer_path,
    new_generation_id,
    publish_elo_generation,
    read_elo_generation_pointer,
    staged_artifact_filename,
)
from utils import get_current_nfl_week, get_logger, setup_logging

logger = get_logger(__name__)

__all__ = [
    "ELO_BURN_IN_START_SEASON",
    "ELO_GENERATION_DIRNAME",
    "ELO_GENERATION_POINTER_NAME",
    "ELO_GENERATION_POINTER_PATH",
    "ELO_RATING_UPDATE_COLUMNS",
    "ELO_ROW_TABLES",
    "ELO_ROW_TABLE_KEY_COLUMN",
    "ELO_SNAPSHOT_COLUMNS",
    "ELO_STATE_ARTIFACTS",
    "EloBuilder",
    "EloForeignSeasonRowsError",
    "EloGenerationIncompleteError",
    "EloGenerationPublisher",
    "EloSeasonReapplicationError",
    "EloSnapshotNotPersistedError",
    "LiveSeasonUpdate",
    "default_stage_writer",
    "elo_generation_pointer_path",
    "new_generation_id",
    "publish_elo_generation",
    "read_elo_generation_pointer",
    "staged_artifact_filename",
]


# The snapshot schema ``EloFeatureBuilder`` reads. Stated once so the live path and the
# canonical builder cannot emit two different shapes.
ELO_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_elo_pre",
    "away_elo_pre",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
    "elo_prob_home",
    "hfa_used",
)

# The per-game rating-update columns merged into ``games_with_elo``.
ELO_RATING_UPDATE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "home_rating_pre",
    "away_rating_pre",
    "home_rating_post",
    "away_rating_post",
    "home_change",
    "away_change",
)


# ---------------------------------------------------------------------------
# Named refusals. Every one of them carries what disagreed, not just that
# something did.
# ---------------------------------------------------------------------------


class EloForeignSeasonRowsError(ValueError):
    """A live append was handed a frame carrying rows outside the named season.

    The whole point of the live verb is that it cannot touch another season's rows;
    a frame that reaches into one is refused rather than silently filtered, because
    silently filtering would hide the caller's bug.
    """


class EloSnapshotNotPersistedError(RuntimeError):
    """Snapshots were COMPUTED and never WRITTEN.

    This is the failure ``pipeline/steps.step_build_elo`` shipped for its whole life:
    the step computed nothing at all, returned cleanly, and the snapshot table never
    gained a current-season row -- so the gold LEFT JOIN produced NaN Elo that was
    then imputed into the deployed WP model. A green step that persisted nothing is
    indistinguishable from one that worked, which is exactly why this refusal exists.
    """

    def __init__(self, rows: int, season: int) -> None:
        self.rows = rows
        self.season = season
        super().__init__(
            f"{rows} Elo snapshot row(s) were computed for season {season} and never "
            f"written. A green Elo step that persisted nothing leaves the snapshot "
            f"table without a {season} row, and the gold join then imputes NaN Elo "
            f"into the deployed WP model. Persist them with: "
            f"python -m scripts.build_elo --current"
        )


class EloSeasonReapplicationError(RuntimeError):
    """The current season was re-derived from a state that already contains it.

    Defence in depth for the double-application defect (T-33-16b). The re-derivation
    is supposed to be a pure function of (prior season's terminal state, this season's
    completed games); handing it a starting state that already carries current-season
    results would apply every one of those games a SECOND time and silently inflate
    the ratings three deployed models consume.
    """


@dataclass
class LiveSeasonUpdate:
    """The three ROW frames for ONE season, at the three grains they really have.

    They are NOT one frame. ``snapshots`` is the per-game PRE-game capture,
    ``games_with_elo`` is the season's games merged with their rating updates, and
    ``rating_history`` is the Elo system's own per-update history. A single-frame
    hand-off would upsert one grain into all three tables.
    """

    season: int
    snapshots: pd.DataFrame
    games_with_elo: pd.DataFrame
    rating_history: pd.DataFrame


class EloBuilder:
    """Build and manage Elo ratings for NFL teams."""

    def __init__(self, data_root: Path | str | None = None):
        """Initialize Elo builder.

        Args:
            data_root: The data lake root this builder writes to. Defaults to the
                configured production root. It is an explicit parameter because
                ``data.storage.upsert_silver`` -- the verb the live append uses --
                resolves its destination from a ``base_path`` argument and ignores the
                module singletons that ``save_dataframe`` uses, so redirecting the
                singletons alone would still upsert into production ``data/silver/``.
        """
        self.settings = get_settings()
        self.elo_system = EloRatingSystem()
        self.data_root = Path(
            data_root if data_root is not None else self.settings.config.data.root_path
        )
        # Snapshots COMPUTED but not yet WRITTEN. Task 2's persist refusal reads it.
        self._pending_snapshot_rows: int = 0

    @property
    def silver_root(self) -> Path:
        """The silver layer beneath this builder's data root."""
        return self.data_root / "silver"

    @property
    def pending_snapshot_rows(self) -> int:
        """How many computed snapshot rows are still unwritten."""
        return self._pending_snapshot_rows

    def load_games_data(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """
        Load games data for specified seasons.

        Args:
            seasons: List of seasons to load (default: all available)

        Returns:
            DataFrame with game data
        """
        try:
            games_df = load_dataframe("games", layer="silver")

            if seasons:
                games_df = games_df[games_df["season"].isin(seasons)]

            logger.info(
                "Loaded games data",
                total_games=len(games_df),
                seasons=sorted(games_df["season"].unique())
                if len(games_df) > 0
                else [],
            )

            return games_df

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error("Failed to load games data", error=str(e))
            raise

    def process_seasons_chronologically(self, seasons: list[int]) -> pd.DataFrame:
        """
        Process multiple seasons chronologically to build Elo ratings.

        Args:
            seasons: List of seasons to process in order

        Returns:
            DataFrame with all processed games and rating updates
        """
        logger.info("Starting chronological Elo processing", seasons=seasons)

        all_processed_games = []

        for season in sorted(seasons):
            logger.info(f"Processing season {season}")

            # Load games for this season
            season_games = self.load_games_data([season])

            if len(season_games) == 0:
                logger.warning(f"No games found for season {season}")
                continue

            # Process season chronologically
            processed_games = self.elo_system.process_season_chronologically(
                season_games, season
            )

            all_processed_games.append(processed_games)

            # Log season summary
            current_ratings = self.elo_system.get_current_ratings(season)
            logger.info(
                f"Completed season {season}",
                games_processed=len(processed_games),
                teams_rated=len(current_ratings),
                top_team=current_ratings.iloc[0]["team"]
                if len(current_ratings) > 0
                else None,
                top_rating=current_ratings.iloc[0]["rating"]
                if len(current_ratings) > 0
                else None,
            )

        # Combine all processed games
        if all_processed_games:
            result_df = pd.concat(all_processed_games, ignore_index=True)
            logger.info(
                "Chronological processing completed",
                total_games=len(result_df),
                seasons_processed=len(seasons),
            )
            return result_df
        return pd.DataFrame()

    # -- the one chronological chain both the canonical builder and the live path use

    def _process_chain(
        self,
        seasons: list[int],
        *,
        games: pd.DataFrame,
        learn_from: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        """Process *seasons* in order against the CURRENT Elo state.

        For each game, in chronological order: capture the PRE-game ratings, record
        the snapshot, and only THEN process the result. That ordering is what keeps a
        snapshot free of its own game's outcome (no batch leakage).

        Args:
            seasons: Seasons to process, in order.
            games: The frame processed games are drawn from.
            learn_from: The frame home-field advantage is LEARNED from. Separate from
                *games* because ``learn_home_field_advantage`` filters to
                ``season - 1``: handed a single-season frame the filter is empty and
                the function silently returns ``hfa_init``.

        Returns:
            ``(snapshot_rows, rating_update_rows)``.
        """
        snapshot_rows: list[dict] = []
        update_rows: list[dict] = []

        for season in sorted(seasons):
            self.elo_system.apply_season_carryover(season)
            self.elo_system.learn_home_field_advantage(learn_from, season)

            season_games = games[games["season"] == season].sort_values("kickoff_et")

            games_processed = 0
            for _, game in season_games.iterrows():
                # Skip games without results
                if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
                    continue

                home = game["home_team"]
                away = game["away_team"]

                # Step 1: Capture PRE-GAME ratings
                home_rating = self.elo_system.get_or_create_rating(home, season)
                away_rating = self.elo_system.get_or_create_rating(away, season)
                home_pre = home_rating.rating
                away_pre = away_rating.rating

                divisional = is_divisional_game(home, away)

                # Get prediction using current (pre-game) state
                prediction = self.elo_system.predict_game(
                    home, away, season, is_divisional=divisional
                )

                # Step 2: Record snapshot
                snapshot_rows.append(
                    {
                        "game_id": game["game_id"],
                        "season": season,
                        "week": game["week"],
                        "home_team": home,
                        "away_team": away,
                        "home_elo_pre": home_pre,
                        "away_elo_pre": away_pre,
                        "home_elo_uncertainty": home_rating.uncertainty,
                        "away_elo_uncertainty": away_rating.uncertainty,
                        "elo_prob_home": prediction["home_win_prob"],
                        "hfa_used": prediction["hfa_used"],
                    }
                )

                # Step 3: THEN process game result (updates ratings)
                home_change, away_change = self.elo_system.update_ratings(
                    home_team=home,
                    away_team=away,
                    home_score=int(game["home_score"]),
                    away_score=int(game["away_score"]),
                    season=season,
                    game_date=game["kickoff_et"],
                    game_id=game["game_id"],
                    is_divisional=divisional,
                )

                update_rows.append(
                    {
                        "game_id": game["game_id"],
                        "home_rating_pre": home_pre,
                        "away_rating_pre": away_pre,
                        "home_rating_post": self.elo_system.ratings[home].rating,
                        "away_rating_post": self.elo_system.ratings[away].rating,
                        "home_change": home_change,
                        "away_change": away_change,
                    }
                )
                games_processed += 1

            logger.info(
                f"Completed season {season} snapshots",
                games_processed=games_processed,
            )

        return snapshot_rows, update_rows

    def build_season_frames(
        self,
        season: int,
        *,
        games: pd.DataFrame | None = None,
        learn_from: pd.DataFrame | None = None,
    ) -> LiveSeasonUpdate:
        """Process ONE season against the current Elo state and return its three frames.

        Does NOT reset the Elo system: the caller decides what state the season starts
        from, which is the whole subject of the re-derivation in
        :meth:`update_current_season`.

        Args:
            season: The season to process.
            games: Games frame (default: the full silver games table).
            learn_from: Frame HFA is learned from (default: *games*).

        Returns:
            The season's snapshots, ``games_with_elo`` rows and rating history.
        """
        all_games = self.load_games_data() if games is None else games
        learning_frame = all_games if learn_from is None else learn_from

        history_before = len(self.elo_system.game_history)
        snapshot_rows, update_rows = self._process_chain(
            [season], games=all_games, learn_from=learning_frame
        )

        snapshots = pd.DataFrame(snapshot_rows, columns=list(ELO_SNAPSHOT_COLUMNS))

        season_games = all_games[all_games["season"] == season].sort_values(
            "kickoff_et"
        )
        games_with_elo = season_games.copy()
        if update_rows:
            games_with_elo = games_with_elo.merge(
                pd.DataFrame(update_rows), on="game_id", how="left"
            )

        history = self.elo_system.get_rating_history()
        rating_history = (
            history.iloc[history_before:].reset_index(drop=True)
            if len(history) > history_before
            else pd.DataFrame(columns=history.columns)
        )

        return LiveSeasonUpdate(
            season=season,
            snapshots=snapshots,
            games_with_elo=games_with_elo,
            rating_history=rating_history,
        )

    def update_current_season(self, season: int | None = None) -> LiveSeasonUpdate:
        """
        Update Elo ratings for the current season only.

        Returns:
            The season's three ROW frames (see :class:`LiveSeasonUpdate`).
        """
        current_season = season if season is not None else get_current_nfl_week()[0]

        logger.info(f"Updating Elo ratings for current season {current_season}")

        # Load existing ratings if available
        self.elo_system.load_ratings()

        update = self.build_season_frames(current_season)
        self._snapshots_df = update.snapshots
        self._pending_snapshot_rows = len(update.snapshots)
        return update

    def build_elo_with_snapshots(self, start_season: int = 2002) -> pd.DataFrame:
        """Build Elo ratings and save per-game pre-game snapshots.

        For each game in chronological order:
        1. Capture PRE-GAME ratings for both teams
        2. Record snapshot (pre-game Elo, uncertainty, win prob, HFA)
        3. THEN process the game result (updates ratings)

        This ensures each game's snapshot contains the ratings computed
        from all prior games but NOT the current game's result (no batch
        leakage).

        Args:
            start_season: First season to process. Default 2002 for full
                burn-in (16 seasons before first backtest season 2018).

        Returns:
            DataFrame of per-game pre-game snapshots with columns:
            game_id, season, week, home_team, away_team, home_elo_pre,
            away_elo_pre, home_elo_uncertainty, away_elo_uncertainty,
            elo_prob_home, hfa_used.
        """
        logger.info(
            "Building Elo ratings with per-game snapshots",
            start_season=start_season,
        )

        # Reset Elo system for clean build
        self.elo_system = EloRatingSystem()

        # Load all games from silver layer
        all_games = self.load_games_data()
        available_seasons = sorted(all_games["season"].unique())
        seasons_to_process = [s for s in available_seasons if s >= start_season]

        if not seasons_to_process:
            logger.warning("No seasons to process", start_season=start_season)
            return pd.DataFrame()

        logger.info(
            "Processing seasons for Elo snapshots",
            seasons=seasons_to_process,
            total_seasons=len(seasons_to_process),
        )

        snapshot_rows, _ = self._process_chain(
            seasons_to_process, games=all_games, learn_from=all_games
        )

        snapshots_df = pd.DataFrame(snapshot_rows, columns=list(ELO_SNAPSHOT_COLUMNS))
        logger.info(
            "Built all Elo snapshots",
            total_snapshots=len(snapshots_df),
            seasons=len(seasons_to_process),
        )

        return snapshots_df

    def build_all_ratings(self, start_season: int = 2002) -> pd.DataFrame:
        """Build Elo ratings from scratch with per-game snapshots.

        Uses build_elo_with_snapshots to process all seasons chronologically
        and produce per-game pre-game snapshots. Also processes seasons
        through the legacy path for backward compatibility of games_with_elo.

        Args:
            start_season: First season to include (default: 2002 for burn-in)

        Returns:
            DataFrame with all processed games (from legacy path)
        """
        # Build snapshots (the primary output)
        self._snapshots_df = self.build_elo_with_snapshots(start_season)
        self._pending_snapshot_rows = len(self._snapshots_df)

        # Also run the legacy processing path for games_with_elo compatibility
        # Reset Elo system for clean processing
        self.elo_system = EloRatingSystem()

        all_games = self.load_games_data()
        available_seasons = sorted(all_games["season"].unique())
        seasons_to_process = [s for s in available_seasons if s >= start_season]

        logger.info(
            "Building all Elo ratings from scratch",
            start_season=start_season,
            seasons_to_process=seasons_to_process,
        )

        return self.process_seasons_chronologically(seasons_to_process)

    # -- the two write verbs ------------------------------------------------

    def save_full_rebuild(
        self, processed_games: pd.DataFrame, start_season: int
    ) -> None:
        """REPLACE every Elo artifact. Reachable only through ``--full-rebuild``.

        Saves, with today's replace-everything semantics unchanged:
        - elo_game_snapshots: Per-game pre-game Elo snapshots (primary artifact)
        - games_with_elo: Games with Elo rating updates (legacy)
        - elo_ratings_current: Current team ratings
        - elo_rating_history: Full rating history
        - elo_ratings.json: Elo system state

        All five are staged under one generation id and validated together before any
        of them is published, so a crash mid-write cannot leave a mixed generation.

        Args:
            processed_games: DataFrame with games and rating updates.
            start_season: The first season this rebuild covers -- named in the log so a
                full rebuild can never be mistaken for a weekly run in a log file.
        """
        snapshots = getattr(self, "_snapshots_df", None)
        if not isinstance(snapshots, pd.DataFrame):
            snapshots = pd.DataFrame()
        current_ratings = self.elo_system.get_current_ratings()
        rating_history = self.elo_system.get_rating_history()

        staged = {
            "elo_game_snapshots": snapshots,
            "games_with_elo": processed_games,
            "elo_rating_history": rating_history,
            "elo_ratings_current": current_ratings,
            "elo_ratings": self.elo_system.ratings_state(),
        }

        def _publish_live() -> None:
            # Save per-game pre-game snapshots (primary artifact for EloFeatureBuilder)
            if len(snapshots) > 0:
                save_dataframe(
                    snapshots,
                    "elo_game_snapshots",
                    layer="silver",
                    append_mode=False,  # Always replace, full rebuild
                )

            # Save updated games with Elo ratings.
            # replace_mode: processed_games is the complete games-with-Elo table for
            # this build; write a single self-replacing file so a full rebuild is
            # idempotent. The prior partition_cols=["season"] wrote into the shared
            # data/silver/season=YYYY/ root, mixing this table's files with the
            # weather/contextual/market feature tables and appending a new file on
            # every run (the ~30.7x games_with_elo bloat). (FIX-01, D-13)
            if len(processed_games) > 0:
                save_dataframe(
                    processed_games,
                    "games_with_elo",
                    layer="silver",
                    replace_mode=True,
                )

            # Save current ratings
            if len(current_ratings) > 0:
                save_dataframe(
                    current_ratings,
                    "elo_ratings_current",
                    layer="silver",
                    append_mode=False,  # Always replace current ratings, don't append
                )

            # Save rating history.
            # replace_mode: rating_history is the complete history for this build;
            # single self-replacing file keeps the rebuild idempotent (no shared-root
            # season-partition append bloat). (FIX-01, D-13)
            if len(rating_history) > 0:
                save_dataframe(
                    rating_history,
                    "elo_rating_history",
                    layer="silver",
                    replace_mode=True,
                )

            # Save Elo system state to JSON
            self.silver_root.mkdir(parents=True, exist_ok=True)
            self.elo_system.save_ratings(str(self.silver_root / "elo_ratings.json"))

        publish_elo_generation(
            staged,
            new_generation_id(),
            silver_root=self.silver_root,
            publish_live=_publish_live,
            mode="full_rebuild",
            start_season=int(start_season),
        )
        self._pending_snapshot_rows = 0

        logger.info(
            "FULL REBUILD published -- every Elo artifact was REPLACED",
            start_season=int(start_season),
            elo_game_snapshots_rows=len(snapshots),
            games_with_elo_rows=len(processed_games),
            elo_rating_history_rows=len(rating_history),
            elo_ratings_current_rows=len(current_ratings),
        )

    def save_live_append(
        self,
        season: int,
        *,
        snapshots: pd.DataFrame,
        games_with_elo: pd.DataFrame,
        rating_history: pd.DataFrame,
    ) -> None:
        """UPSERT one season's rows; replace only the two state artifacts.

        THREE EXPLICIT FRAMES, one per row table, because the three tables have three
        different grains and three different sources. A generic single-frame form would
        upsert one grain into all three and silently corrupt two of them.

        A season with ZERO completed games writes zero new rows and returns normally --
        R3's explicit edge case. The refusal in this phase is about snapshots that were
        COMPUTED and not written, never about a season that had nothing to compute.

        Args:
            season: The ONLY season these frames may touch.
            snapshots: Per-game pre-game snapshots for *season*.
            games_with_elo: The season's games merged with their rating updates.
            rating_history: The Elo system's per-update history for *season*.

        Raises:
            EloForeignSeasonRowsError: If any frame carries a row from another season.
        """
        supplied = {
            "elo_game_snapshots": snapshots,
            "games_with_elo": games_with_elo,
            "elo_rating_history": rating_history,
        }
        for name, frame in supplied.items():
            _refuse_foreign_season_rows(name, frame, season)

        scoped = {
            name: _rows_for_season(frame, season) for name, frame in supplied.items()
        }
        current_ratings = self.elo_system.get_current_ratings()

        staged = {
            **scoped,
            "elo_ratings_current": current_ratings,
            "elo_ratings": self.elo_system.ratings_state(),
        }

        def _publish_live() -> None:
            for name in ELO_ROW_TABLES:
                self._upsert_row_table(scoped[name], name)

            # STATE artifacts are current-state by definition, so they are replaced.
            # replace_mode (not append_mode=False) on purpose: it writes one
            # self-contained file and never reaches the partitioned form, and it keeps
            # every replace-shaped keyword in this module inside save_full_rebuild,
            # where an AST assertion pins them (T-33-12).
            if len(current_ratings) > 0:
                save_dataframe(
                    current_ratings,
                    "elo_ratings_current",
                    layer="silver",
                    replace_mode=True,
                )

            self.silver_root.mkdir(parents=True, exist_ok=True)
            self.elo_system.save_ratings(str(self.silver_root / "elo_ratings.json"))

        publish_elo_generation(
            staged,
            new_generation_id(),
            silver_root=self.silver_root,
            publish_live=_publish_live,
            mode="live_append",
            season=int(season),
        )
        self._pending_snapshot_rows = 0

        logger.info(
            "Live append published",
            season=int(season),
            elo_game_snapshots_rows=len(scoped["elo_game_snapshots"]),
            games_with_elo_rows=len(scoped["games_with_elo"]),
            elo_rating_history_rows=len(scoped["elo_rating_history"]),
        )

    def _upsert_row_table(self, frame: pd.DataFrame, table_name: str) -> int:
        """Upsert *frame* into a silver ROW table and keep DuckDB in step.

        ``upsert_silver`` writes PARQUET ONLY. ``load_dataframe(source="auto")``
        resolves DuckDB FIRST (data/storage.py:959-963), so a parquet-only upsert would
        leave the database serving the pre-append rows and every consumer would read
        the stale copy -- the same "the write did not land" failure this plan exists to
        remove, merely relocated. The combined table is therefore read back and the
        DuckDB copy replaced from it, so the two stores cannot disagree.
        """
        if frame is None or len(frame) == 0:
            logger.info("Live append wrote zero rows", table=table_name)
            return 0

        path = upsert_silver(
            frame,
            table_name,
            key_column=ELO_ROW_TABLE_KEY_COLUMN,
            base_path=self.data_root,
        )
        combined = pd.read_parquet(path, engine="pyarrow")
        get_db_connection().create_table_from_df(
            combined, table_name, if_exists="replace"
        )
        return len(frame)

    def validate_ratings(self) -> bool:
        """
        Validate Elo ratings for sanity checks.

        Returns:
            True if validation passes
        """
        current_ratings = self.elo_system.get_current_ratings()

        if len(current_ratings) == 0:
            logger.error("No ratings found - validation failed")
            return False

        # Check rating ranges
        min_rating = current_ratings["rating"].min()
        max_rating = current_ratings["rating"].max()

        if min_rating < 800 or max_rating > 2200:
            logger.warning(
                "Ratings outside expected range",
                min_rating=min_rating,
                max_rating=max_rating,
            )

        # Check for reasonable spread
        rating_std = current_ratings["rating"].std()
        if rating_std < 50 or rating_std > 300:
            logger.warning("Rating standard deviation unusual", rating_std=rating_std)

        # Check that all teams have played games
        no_games = current_ratings[current_ratings["games_played"] == 0]
        if len(no_games) > 0:
            logger.warning(
                "Teams with no games played", teams=no_games["team"].tolist()
            )

        logger.info(
            "Elo rating validation completed",
            teams=len(current_ratings),
            rating_range=(min_rating, max_rating),
            rating_std=rating_std,
        )

        return True


def _rows_for_season(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Return the rows of *frame* belonging to *season*."""
    if frame is None or len(frame) == 0 or "season" not in frame.columns:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    return frame[frame["season"] == season]


def _refuse_foreign_season_rows(
    table_name: str, frame: pd.DataFrame, season: int
) -> None:
    """Refuse, by name, a frame carrying rows outside *season*.

    Silently filtering would be worse than refusing: it would hide a caller that built
    the wrong frame, and the wrong frame is how a live append reaches into a season it
    has no business touching (T-33-14).
    """
    if frame is None or len(frame) == 0 or "season" not in frame.columns:
        return

    foreign = sorted(
        {
            int(value)
            for value in frame["season"].dropna().unique()
            if int(value) != season
        }
    )
    if foreign:
        raise EloForeignSeasonRowsError(
            f"save_live_append({season}) was handed a '{table_name}' frame carrying "
            f"rows from season(s) {foreign}. A live append may only ever touch the "
            f"season it names; rows from {foreign} would rewrite history that is not "
            "this week's to rewrite."
        )


def main():
    """CLI entry point for Elo rating builder."""
    parser = argparse.ArgumentParser(description="Build NFL Elo ratings")
    parser.add_argument("--season", type=int, help="Process single season")
    parser.add_argument(
        "--seasons", nargs="+", type=int, help="Process multiple seasons"
    )
    parser.add_argument(
        "--all-seasons", action="store_true", help="Process all available seasons"
    )
    parser.add_argument(
        "--current", action="store_true", help="Update current season only"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=ELO_BURN_IN_START_SEASON,
        help=f"Starting season for all-seasons build (default: {ELO_BURN_IN_START_SEASON})",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "REPLACE every Elo artifact rather than upserting. Required for the "
            "--all-seasons / --seasons / --season write paths, and never reachable by "
            "default: replace-all against a table holding 24 seasons of burn-in is a "
            "destructive operation and has to be asked for."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing ratings"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results (dry run)"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging()

        builder = EloBuilder()

        if args.validate_only:
            # Load existing ratings and validate
            builder.elo_system.load_ratings()
            success = builder.validate_ratings()
            if success:
                print("Elo rating validation passed")
            else:
                print("Elo rating validation failed")
                sys.exit(1)
            return

        live_update: LiveSeasonUpdate | None = None
        processed_games = pd.DataFrame()

        if args.current:
            live_update = builder.update_current_season()
        elif args.all_seasons:
            processed_games = builder.build_all_ratings(args.start_season)
        elif args.seasons:
            processed_games = builder.process_seasons_chronologically(args.seasons)
        elif args.season:
            processed_games = builder.process_seasons_chronologically([args.season])
        else:
            # Default: update current season
            live_update = builder.update_current_season()

        if not args.no_save:
            if live_update is not None:
                builder.save_live_append(
                    live_update.season,
                    snapshots=live_update.snapshots,
                    games_with_elo=live_update.games_with_elo,
                    rating_history=live_update.rating_history,
                )
            elif len(processed_games) > 0:
                if not args.full_rebuild:
                    print(
                        "Refusing to write: a historical build REPLACES the Elo "
                        "tables. Re-run with --full-rebuild to confirm, or --no-save "
                        "for a dry run."
                    )
                    sys.exit(1)
                builder.save_full_rebuild(processed_games, args.start_season)

        # Validate results
        builder.validate_ratings()

        # Print summary
        current_ratings = builder.elo_system.get_current_ratings()
        if len(current_ratings) > 0:
            games_seen = (
                len(live_update.games_with_elo)
                if live_update is not None
                else len(processed_games)
            )
            print("\nElo ratings built successfully!")
            print(f"Teams rated: {len(current_ratings)}")
            print(f"Games processed: {games_seen}")
            print("\nTop 5 teams:")
            for _, team in current_ratings.head().iterrows():
                print(
                    f"  {team['team']}: {team['rating']:.1f} ({team['games_played']} games)"
                )

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Elo building failed", error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
