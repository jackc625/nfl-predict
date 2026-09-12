"""Elo Feature Builder

This module creates Elo-based features for game prediction models by
looking up pre-computed per-game Elo snapshots from the silver layer.

The snapshots are produced by scripts/build_elo.py which processes all
games chronologically, recording each team's pre-game Elo BEFORE the
game result is applied. This eliminates batch leakage where end-of-season
ratings were previously assigned to every game.

Conforms to the FeatureBuilder Protocol with mandatory as_of_datetime
parameter for time-fence enforcement.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from data.storage import load_dataframe
from features.protocol import FeatureBuilder  # noqa: F401 (documents conformance)

# IMPORTED, never re-declared. ``scripts/build_elo`` owns the snapshot schema because it
# WRITES it; the flag's name, the table's name and the back-compat fill therefore have one
# home and cannot drift into two spellings that agree only until one of them changes. The
# same import-the-primitive seam ``models/train.py`` uses for the feature-group vocabulary
# (D30-02's lesson, after a second locally-declared group list silently broke a baseline).
from scripts.build_elo import (
    ELO_SNAPSHOT_TABLE,
    PROVISIONAL_COLUMN,
    ensure_provisional_flag,
)
from utils import get_logger
from utils.exceptions import DataIngestionError

logger = get_logger(__name__)

# How many offending game ids a refusal message names before it summarises the rest.
# An operator needs to know WHICH games to wait on; a 272-id wall of text is how a
# message stops being read.
_MAX_NAMED_GAME_IDS = 12

__all__ = [
    "ELO_FEATURE_COLUMNS",
    "EloFeatureBuilder",
    "ProvisionalSnapshotAsTrainingInputError",
    "assert_no_provisional_training_rows",
    "ensure_provisional_flag",
    "load_elo_snapshots",
    "provisional_row_counts",
]

# Elo feature column names -- extended with momentum, rank, percentile
ELO_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob_home",
    "elo_prob_away",
    "hfa_used",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
    "home_elo_momentum",
    "away_elo_momentum",
    "home_elo_rank",
    "away_elo_rank",
    "home_elo_percentile",
    "away_elo_percentile",
]


class ProvisionalSnapshotAsTrainingInputError(RuntimeError):
    """A PROVISIONAL Elo snapshot row reached a model TRAINING input.

    A provisional row is the pre-game Elo of a game that has NOT been played. Serving it
    is correct -- that is what it exists for. TRAINING on it is not: the row is a
    placeholder, and a model fitted against placeholders has learned the placeholder.

    WHY THIS IS A ``RuntimeError`` AND THAT CHOICE IS LOAD-BEARING. Three separate
    handler tuples in this repository would otherwise swallow it:
    ``scripts/build_features._SOURCE_LOAD_ERRORS`` converts a caught source-load failure
    into an EMPTY FRAME and logs a warning, and all four trainers catch
    ``(FileNotFoundError, OSError, ValueError, KeyError)`` around their gold load and
    ``return``. A refusal caught by any of them refuses nothing: it produces a green run
    that trained on nothing, or a gold build with no Elo at all -- which is the exact
    silent-failure shape Phase 33 exists to remove. ``RuntimeError`` is deliberately
    absent from every one of those tuples, and a test asserts the exclusion member by
    member rather than trusting this docstring.
    """


def load_elo_snapshots() -> pd.DataFrame:
    """THE ONE read seam for ``elo_game_snapshots``.

    Every consumer goes through here so the back-compat fill happens exactly once. A
    snapshot parquet written before ``is_provisional`` existed reads back WITH the column,
    ``False`` for every row -- not as a PyArrow schema-union error, and not as a null
    third state each call site then has to decide about.

    Returns:
        The snapshot table, flag-aligned.
    """
    return ensure_provisional_flag(load_dataframe(ELO_SNAPSHOT_TABLE, layer="silver"))


def provisional_row_counts(snapshots: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Count PROVISIONAL rows by ``(season, week)``.

    Exposure has to be OBSERVABLE and not merely refusable. A refusal fires once, at the
    boundary, on the run that would have trained; a count logged at every build is what
    lets somebody notice that week 4's provisional rows never got replaced -- BEFORE a
    trainer refuses and the week's run is simply lost.

    Args:
        snapshots: Any snapshot-shaped frame. A PRE-FLAG frame reports ``{}`` rather than
            raising: no flag means no provisional rows, which is the historical truth.

    Returns:
        ``{(season, week): count}`` for the weeks that have at least one provisional row.
    """
    if snapshots is None or len(snapshots) == 0:
        return {}
    if PROVISIONAL_COLUMN not in snapshots.columns:
        return {}

    flagged = snapshots.loc[snapshots[PROVISIONAL_COLUMN].fillna(False).astype(bool)]
    if len(flagged) == 0 or not {"season", "week"}.issubset(flagged.columns):
        return {}

    grouped = flagged.groupby(["season", "week"]).size()
    return {
        (int(season), int(week)): int(count)
        for (season, week), count in grouped.items()
    }


def _resolve_backing_snapshots(
    snapshots: pd.DataFrame | None, context: str
) -> pd.DataFrame | None:
    """The snapshot rows a gold frame rests on, or ``None`` if they cannot be resolved.

    ``None`` is NOT a bypass. A provisional row can only exist in the snapshot table, so
    a checkout with no snapshot table has nothing to refuse -- there is no state in which
    an unresolvable table hides a provisional row. The warning is logged so the
    unresolved case is visible rather than assumed.
    """
    if snapshots is not None:
        return ensure_provisional_flag(snapshots)
    try:
        return load_elo_snapshots()
    except (
        DataIngestionError,
        FileNotFoundError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        logger.warning(
            "Could not resolve backing Elo snapshots for the provisional-row check",
            context=context,
            error=str(exc),
        )
        return None


def _provisional_refusal_message(offending: pd.DataFrame, context: str) -> str:
    """The refusal text: which games, which weeks, and what to do about it."""
    ids = (
        sorted(str(value) for value in offending["game_id"].tolist())
        if "game_id" in offending.columns
        else []
    )
    shown = ids[:_MAX_NAMED_GAME_IDS]
    more = f" (+{len(ids) - len(shown)} more)" if len(ids) > len(shown) else ""

    if {"season", "week"}.issubset(offending.columns):
        weeks = sorted(
            {
                (int(season), int(week))
                for season, week in zip(
                    offending["season"].tolist(), offending["week"].tolist()
                )
            }
        )
    else:
        weeks = []

    return (
        f"{len(offending)} PROVISIONAL Elo snapshot row(s) reached a TRAINING input at "
        f"{context}: {shown}{more}. Affected (season, week): {weeks or 'unknown'}. A "
        "provisional row is the pre-game Elo of a game that has NOT been played -- "
        "correct to SERVE and wrong to train on, because a model fitted against a "
        "placeholder has learned the placeholder. RECOVERY: re-run training after those "
        "results land and the real snapshots replace the provisional rows in place "
        "(python -m scripts.build_elo --current, then rebuild gold), or drive the "
        "SERVING path, which legitimately consumes them."
    )


def assert_no_provisional_training_rows(
    frame: pd.DataFrame,
    context: str,
    *,
    snapshots: pd.DataFrame | None = None,
) -> None:
    """Refuse *frame* as a training input if it rests on any PROVISIONAL snapshot row.

    THE GUARD LIVES HERE AND IS CALLED AT FOUR PLACES. Every model training entry point
    loads gold DIRECTLY -- three through ``load_dataframe("features_*", layer="gold")``
    and one through ``pd.read_parquet`` -- and not one of them passes through
    :meth:`EloFeatureBuilder.build_features`. A guard placed only in the feature builder
    would protect none of them. One implementation, four call sites, pinned by
    ``tests/phase33_state.TRAINER_GOLD_LOAD_SITES``; a second copy is the drift surface
    this repository has been bitten by three times.

    GOLD DOES NOT CARRY THE FLAG, so the check cannot simply read a column. The Elo join
    takes an explicit seven-column subset that excludes ``is_provisional`` (NF-08's column
    claim), which is why a gold frame is resolved back to its BACKING snapshot rows by
    ``game_id`` and those are what get checked.

    Args:
        frame: The frame about to be trained on. May be a gold matrix (no flag, resolved
            by ``game_id``) or a snapshot-shaped frame (flag read directly).
        context: Where the check fired, e.g. ``"train:wp"``. Named in the message,
            because "a provisional row reached training" without a location is a fact
            nobody can act on.
        snapshots: Backing snapshots to check against (default: the silver table through
            the one read seam). Explicit only so a caller that already holds the frame
            need not re-read it.

    Raises:
        ProvisionalSnapshotAsTrainingInputError: Naming the offending ``game_id``s, the
            affected ``(season, week)`` pairs and the recovery. Also raised, FAIL-CLOSED,
            when the frame carries neither the flag nor ``game_id`` -- an integrity gate
            that cannot verify must refuse rather than wave the frame through.
    """
    if frame is None or len(frame) == 0:
        return

    if PROVISIONAL_COLUMN in frame.columns:
        candidates = ensure_provisional_flag(frame)
    else:
        if "game_id" not in frame.columns:
            raise ProvisionalSnapshotAsTrainingInputError(
                f"the frame handed to {context} carries neither "
                f"'{PROVISIONAL_COLUMN}' nor 'game_id', so whether it rests on "
                "provisional Elo cannot be established. Refusing rather than assuming: "
                "an integrity gate that cannot verify and proceeds anyway is a gate that "
                "reports green on exactly the inputs it was built to catch."
            )
        backing = _resolve_backing_snapshots(snapshots, context)
        if backing is None or len(backing) == 0:
            return
        wanted = frame["game_id"].drop_duplicates()
        candidates = backing.loc[backing["game_id"].isin(wanted)]

    offending = candidates.loc[candidates[PROVISIONAL_COLUMN]]
    if len(offending) == 0:
        return

    raise ProvisionalSnapshotAsTrainingInputError(
        _provisional_refusal_message(offending, context)
    )


class EloFeatureBuilder:
    """Build Elo-based features from pre-computed snapshots.

    Satisfies the FeatureBuilder Protocol via structural subtyping.
    Features are derived from per-game Elo snapshots stored in the silver
    layer (elo_game_snapshots), NOT recomputed on the fly.

    The as_of_datetime parameter is accepted for protocol conformance but
    filtering is handled by the snapshot join -- each game's snapshot already
    contains only pre-game information by construction.
    """

    def __init__(self) -> None:
        """Initialize Elo feature builder with empty snapshot cache."""
        self._snapshots_df: pd.DataFrame | None = None

    def _load_snapshots(self) -> pd.DataFrame:
        """Load per-game Elo snapshots from silver layer.

        Loads elo_game_snapshots produced by scripts/build_elo.py and
        caches the result for repeated calls within the same session.

        Returns:
            DataFrame with per-game pre-game Elo snapshots.

        Raises:
            ValueError: If snapshots are not found in the silver layer.
        """
        if self._snapshots_df is not None:
            return self._snapshots_df

        logger.info("Loading Elo game snapshots from silver layer")
        # THROUGH THE ONE READ SEAM, so a pre-flag parquet arrives flag-aligned rather
        # than leaving this cache holding the only un-filled copy in the process.
        self._snapshots_df = load_elo_snapshots()
        logger.info(
            "Loaded Elo snapshots",
            total_snapshots=len(self._snapshots_df),
            seasons=sorted(self._snapshots_df["season"].unique().tolist())
            if len(self._snapshots_df) > 0
            else [],
        )
        return self._snapshots_df

    # ---- Derived feature methods ----

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        snapshots: pd.DataFrame,
        lookback: int = 4,
    ) -> pd.DataFrame:
        """Add Elo momentum features for home and away teams.

        Momentum measures the Elo change per game over a rolling window.
        For each team, it looks at the last `lookback` games (or fewer if
        the team has played fewer games in the season) and computes:
            momentum = (newest_elo - oldest_elo) / window_size

        Returns NaN for a team's first game of the season (no prior games).

        Args:
            df: DataFrame with games (must have game_id, season, week,
                home_team, away_team columns and Elo columns from snapshot merge).
            snapshots: Full elo_game_snapshots DataFrame.
            lookback: Number of prior games for momentum window (default 4).

        Returns:
            DataFrame with home_elo_momentum and away_elo_momentum added.
        """
        home_momentum = []
        away_momentum = []

        for _, game in df.iterrows():
            season = game["season"]
            week = game["week"]

            for team, elo_list in [
                (game["home_team"], home_momentum),
                (game["away_team"], away_momentum),
            ]:
                # Find all prior games for this team in this season
                team_home = snapshots[
                    (snapshots["season"] == season)
                    & (snapshots["week"] < week)
                    & (snapshots["home_team"] == team)
                ][["week", "home_elo_pre"]].rename(columns={"home_elo_pre": "team_elo"})
                team_away = snapshots[
                    (snapshots["season"] == season)
                    & (snapshots["week"] < week)
                    & (snapshots["away_team"] == team)
                ][["week", "away_elo_pre"]].rename(columns={"away_elo_pre": "team_elo"})
                prior_games = pd.concat(
                    [team_home, team_away], ignore_index=True
                ).sort_values("week")

                if len(prior_games) == 0:
                    elo_list.append(np.nan)
                else:
                    window = prior_games.tail(lookback)
                    oldest_elo = window.iloc[0]["team_elo"]
                    newest_elo = window.iloc[-1]["team_elo"]
                    momentum = (newest_elo - oldest_elo) / len(window)
                    elo_list.append(momentum)

        df = df.copy()
        df["home_elo_momentum"] = home_momentum
        df["away_elo_momentum"] = away_momentum
        return df

    def _add_rank_features(
        self,
        df: pd.DataFrame,
        snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Elo rank and percentile features for home and away teams.

        For each game, determines the latest pre-game Elo for all teams
        as of that game's week, then ranks them 1-32 (1 = highest Elo).
        Percentile = (32 - rank + 1) / 32, so rank 1 = 1.0, rank 32 ~= 0.03.

        SAME-WEEK PROVISIONAL ROWS DO FEED THESE FOUR COLUMNS, AND THAT IS CORRECT
        (NF-08's ROW claim, Plan 33-04). The week selector below is ``week <= week``, and
        the frame handed in is the FULL snapshots table -- so for a LIVE week the ranking
        is computed partly FROM provisional rows. Pre-game Elo is captured BEFORE the
        game, so week N's snapshot is a valid input for ranking at week N; the
        alternative, selecting ``week < W``, would rank this week's teams on last week's
        standings. ``_add_momentum_features`` is unaffected either way: it selects
        strictly prior weeks, so no provisional row can enter a rolling momentum window
        as a zero-delta game. Both halves are asserted as VALUES in
        ``tests/unit/test_elo_gold_features.py`` rather than inferred from these
        selectors.

        THE CONSEQUENCE THE ORIGINAL COMMENT DOES NOT STATE -- a REPRODUCIBILITY HAZARD,
        recorded here rather than left for a reviewer to discover. A gold row's rank and
        percentile columns built on FRIDAY can differ from the same row rebuilt after the
        results land, because the provisional row is replaced in place by the real one and
        the ranking then resolves against different values. Nothing about that is a leak;
        it does mean these four columns are not a pure function of the game alone, they
        are a function of the game AND when the build ran. Handed to PHASE 34's replay
        work, which owns forward reproducibility. COLD-01's value-by-value comparison does
        NOT catch it: that comparison is scoped to the seven JOINED snapshot columns
        (``tests/phase33_state.ELO_GOLD_JOIN_SUBSET``), and these four are DERIVED after
        the join rather than joined.

        Args:
            df: DataFrame with games (must have game_id, season, week,
                home_team, away_team columns).
            snapshots: Full elo_game_snapshots DataFrame.

        Returns:
            DataFrame with home_elo_rank, away_elo_rank,
            home_elo_percentile, away_elo_percentile added.
        """
        home_ranks = []
        away_ranks = []
        home_pcts = []
        away_pcts = []

        # Cache rankings by (season, week) to avoid recomputation
        rank_cache: dict[tuple[int, int], dict[str, int]] = {}

        for _, game in df.iterrows():
            season = game["season"]
            week = game["week"]
            cache_key = (season, week)

            if cache_key not in rank_cache:
                # Build a mapping of team -> latest pre-game Elo as of this week
                season_snaps = snapshots[snapshots["season"] == season]
                # Include current week's games (pre-game Elo is captured BEFORE
                # the game, so week N's snapshot is valid for ranking at week N)
                week_snaps = season_snaps[season_snaps["week"] <= week]

                team_elos: dict[str, float] = {}

                # Process home teams
                for _, snap in week_snaps.iterrows():
                    home_t = snap["home_team"]
                    away_t = snap["away_team"]
                    snap_week = snap["week"]

                    # Keep the latest week's Elo for each team
                    if home_t not in team_elos or snap_week >= team_elos.get(
                        f"_week_{home_t}", -1
                    ):
                        team_elos[home_t] = snap["home_elo_pre"]
                        team_elos[f"_week_{home_t}"] = snap_week

                    if away_t not in team_elos or snap_week >= team_elos.get(
                        f"_week_{away_t}", -1
                    ):
                        team_elos[away_t] = snap["away_elo_pre"]
                        team_elos[f"_week_{away_t}"] = snap_week

                # Remove internal tracking keys
                clean_elos = {
                    k: v for k, v in team_elos.items() if not k.startswith("_week_")
                }

                # Rank: 1 = highest Elo
                sorted_teams = sorted(
                    clean_elos.items(), key=lambda x: x[1], reverse=True
                )
                n_teams = len(sorted_teams)
                team_rank = {
                    team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)
                }
                rank_cache[cache_key] = team_rank

            team_rank = rank_cache[cache_key]
            n_teams = len(team_rank)

            if n_teams == 0:
                # No Elo snapshots cover this (season, week): the snapshot table
                # only spans the rated era (2018+), so pre-2018 burn-in weeks have
                # no teams to rank. Emit NaN ("no Elo rank") instead of dividing by
                # zero. These games are Elo burn-in and are never model inputs --
                # training/validation start at season 2018, where snapshots always
                # exist and n_teams > 0, so this branch never affects model data.
                home_ranks.append(float("nan"))
                away_ranks.append(float("nan"))
                home_pcts.append(float("nan"))
                away_pcts.append(float("nan"))
                continue

            home_r = team_rank.get(game["home_team"], n_teams)
            away_r = team_rank.get(game["away_team"], n_teams)

            home_ranks.append(home_r)
            away_ranks.append(away_r)
            home_pcts.append((n_teams - home_r + 1) / n_teams)
            away_pcts.append((n_teams - away_r + 1) / n_teams)

        df = df.copy()
        df["home_elo_rank"] = home_ranks
        df["away_elo_rank"] = away_ranks
        df["home_elo_percentile"] = home_pcts
        df["away_elo_percentile"] = away_pcts
        return df

    # ---- Protocol-conforming methods ----

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build Elo features by joining games with pre-computed snapshots.

        Looks up per-game Elo snapshots from the silver layer and joins
        them to the input games DataFrame. Each snapshot contains the
        pre-game Elo ratings (computed from all prior games only).

        Args:
            games_df: DataFrame with game records (must have game_id column).
            as_of_datetime: Time-fence cutoff (accepted for protocol
                conformance; filtering is inherent in snapshot construction).
            target_season: Optional filter to a specific season.
            target_week: Optional filter to a specific week.

        Returns:
            DataFrame with Elo feature columns added.
        """
        logger.info(
            "Building Elo features from snapshots",
            games=len(games_df),
            as_of_datetime=str(as_of_datetime),
        )

        # Apply optional filters
        filtered_df = games_df.copy()
        if target_season is not None:
            filtered_df = filtered_df[filtered_df["season"] == target_season]
        if target_week is not None:
            filtered_df = filtered_df[filtered_df["week"] == target_week]

        # Load pre-computed snapshots
        snapshots = self._load_snapshots()

        # Provisional exposure is RECORDED at every build, not only refused at the
        # training boundary. A refusal fires once, on the run that would have trained; a
        # count in the run log is what lets somebody notice that a week's provisional
        # rows were never replaced before that run is lost.
        counts = provisional_row_counts(snapshots)
        if counts:
            logger.warning(
                "Elo snapshots include PROVISIONAL rows (correct to SERVE, refused as a "
                "TRAINING input)",
                provisional_rows=sum(counts.values()),
                by_season_week={
                    f"{season}_W{week:02d}": count
                    for (season, week), count in sorted(counts.items())
                },
            )

        # Select only the columns we need from snapshots for the join
        snapshot_cols = [
            "game_id",
            "home_elo_pre",
            "away_elo_pre",
            "home_elo_uncertainty",
            "away_elo_uncertainty",
            "elo_prob_home",
            "hfa_used",
        ]
        # Only keep columns that exist in the snapshots
        available_cols = [c for c in snapshot_cols if c in snapshots.columns]
        snapshot_subset = snapshots[available_cols]

        # Merge snapshots onto games by game_id (left join to keep all games)
        merged = filtered_df.merge(snapshot_subset, on="game_id", how="left")

        # Rename snapshot columns to feature names
        merged = merged.rename(
            columns={
                "home_elo_pre": "home_elo",
                "away_elo_pre": "away_elo",
            }
        )

        # Compute derived columns
        merged["elo_diff"] = merged["home_elo"] - merged["away_elo"]
        merged["elo_prob_away"] = 1.0 - merged["elo_prob_home"]

        # Log any games without snapshots (future games or missing data)
        missing_count = merged["home_elo"].isna().sum()
        if missing_count > 0:
            logger.warning(
                "Games without Elo snapshots (future games or missing data)",
                missing_count=missing_count,
                total_games=len(merged),
            )

        # Add derived features: momentum and rank/percentile
        merged = self._add_momentum_features(merged, snapshots)
        merged = self._add_rank_features(merged, snapshots)

        logger.info("Built Elo features from snapshots", games=len(merged))
        return merged

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get Elo features for a single game from pre-computed snapshots.

        Looks up the game's pre-game Elo snapshot from the silver layer.

        Args:
            game_id: Unique game identifier (e.g., '2024_01_BUF_MIA').
            as_of_datetime: Time-fence cutoff (accepted for protocol
                conformance).

        Returns:
            Dictionary mapping feature names to values.

        Raises:
            ValueError: If game_id is not found in snapshots.
        """
        snapshots = self._load_snapshots()
        game_snap = snapshots[snapshots["game_id"] == game_id]

        if len(game_snap) == 0:
            raise ValueError(
                f"No Elo snapshot found for game_id={game_id}. "
                "Ensure build_elo.py has been run with --all-seasons."
            )

        snap = game_snap.iloc[0]
        season = int(snap["season"])
        week = int(snap["week"])

        # Base features from snapshot
        features = {
            "home_elo": float(snap["home_elo_pre"]),
            "away_elo": float(snap["away_elo_pre"]),
            "elo_diff": float(snap["home_elo_pre"] - snap["away_elo_pre"]),
            "elo_prob_home": float(snap["elo_prob_home"]),
            "elo_prob_away": float(1.0 - snap["elo_prob_home"]),
            "hfa_used": float(snap["hfa_used"]),
            "home_elo_uncertainty": float(snap["home_elo_uncertainty"]),
            "away_elo_uncertainty": float(snap["away_elo_uncertainty"]),
        }

        # Compute momentum for home and away teams
        home_team = snap["home_team"]
        away_team = snap["away_team"]
        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            team_home = snapshots[
                (snapshots["season"] == season)
                & (snapshots["week"] < week)
                & (snapshots["home_team"] == team)
            ][["week", "home_elo_pre"]].rename(columns={"home_elo_pre": "team_elo"})
            team_away = snapshots[
                (snapshots["season"] == season)
                & (snapshots["week"] < week)
                & (snapshots["away_team"] == team)
            ][["week", "away_elo_pre"]].rename(columns={"away_elo_pre": "team_elo"})
            prior_games = pd.concat(
                [team_home, team_away], ignore_index=True
            ).sort_values("week")

            if len(prior_games) == 0:
                features[f"{prefix}_elo_momentum"] = float("nan")
            else:
                window = prior_games.tail(4)
                oldest_elo = window.iloc[0]["team_elo"]
                newest_elo = window.iloc[-1]["team_elo"]
                features[f"{prefix}_elo_momentum"] = float(
                    (newest_elo - oldest_elo) / len(window)
                )

        # Compute rank and percentile
        season_snaps = snapshots[
            (snapshots["season"] == season) & (snapshots["week"] <= week)
        ]
        team_elos: dict[str, float] = {}
        for _, s in season_snaps.iterrows():
            team_elos[s["home_team"]] = s["home_elo_pre"]
            team_elos[s["away_team"]] = s["away_elo_pre"]

        sorted_teams = sorted(team_elos.items(), key=lambda x: x[1], reverse=True)
        n_teams = len(sorted_teams)
        team_rank = {t: r + 1 for r, (t, _) in enumerate(sorted_teams)}

        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            rank = team_rank.get(team, n_teams)
            features[f"{prefix}_elo_rank"] = float(rank)
            features[f"{prefix}_elo_percentile"] = float((n_teams - rank + 1) / n_teams)

        return features
