"""Build the v2 ("repaired") gold matrices into ``outputs/gold_v2/``, beside the live gold.

This script reuses the live builder through a SUBCLASS that changes three seams:

* ``opponent_adj`` is the windowed adjuster (``features.opponent_adj_windowed``) and
  ``qb_tracker`` the replacement-level tracker (``features.qb_v2``);
* ``normalize_combined_features`` is the identity, so gold keeps raw levels (Elo points,
  success rates, mph) instead of per-season z-scores;
* ``_merge_opponent_adjusted`` keeps the six raw rolling EPA columns (the live team-form rolling averages, untouched by the v2 window) the live merge drops.

``generate_feature_matrices`` RETURNS the matrices and writes nothing; it runs under a
``RecordingSink`` and the build refuses if any storage primitive asked to write. Two v2-only
families -- market-implied team ratings and the league scoring level -- are then merged on
``game_id``, each checked first by its OWN ``InformationTimeGate`` against the lock frame
(never the builder's coverage report, whose final refusal demands exactly ten checked
names). The six derived inputs are added, the combined-matrix ``LeakageGate`` runs on each
matrix, and the three matrices are written with plain ``DataFrame.to_parquet``.

``build_gold_v2`` NEVER calls ``save_dataframe`` / ``save_feature_matrices``: they also
replace the layer-blind DuckDB table ``features_{t}`` the live predictor reads. It writes
nothing under ``data/``.

THE ONE SANCTIONED WAY V2 GOLD REACHES ``data/gold`` (MIDSEASON-SWITCH-RECORD.md section 4) is
``FeatureMatrixBuilderV2Live``: the same build, the same two families and the same checks, run by
the nightly gold build (``pipeline.steps.build_and_save_gold``) and written there by the parent's
``save_feature_matrices`` -- parquet and the DuckDB table together. The nightly build picks it
only while ``refit.generations.in_force_recipe`` answers the installed v2 recipe, so v2 gold and
the v2 models can only reach production together.

Usage:
    uv run python -m scripts.build_gold_v2 [--out-dir outputs/gold_v2]

ASCII only, no emoji.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.study_1a import refuse_data_dir
from data.storage import load_dataframe
from data.upstream_pin import load_schedules
from data.write_sink import RecordingSink, active_sink
from features.league_level import league_points_level
from features.market_rating import (
    market_rating_no_information_signature,
    market_ratings,
)
from features.opponent_adj_windowed import WindowedOpponentAdjuster
from features.provenance import (
    InformationBasis,
    InformationTimeGate,
    SourceCheckState,
    build_lock_frame,
    refuse_provenance_columns,
)
from features.qb_v2 import QBTrackerV2
from features.validation import LeakageGate
from refit.records import code_commit
from scripts.build_features import FeatureMatrixBuilder
from scripts.fingerprint_gold import BUILD_CLOCK_COLUMNS
from utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = Path("outputs/gold_v2")
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
# 2001 cannot join (silver starts in 2002), so 2002 has no market prior.
NFLVERSE_SEASONS: tuple[int, ...] = tuple(range(2001, 2027))

#: The six raw rolling EPA columns (live team-form averages, untouched by the v2 window) the live opponent-adjusted merge drops; v2 keeps them.
RAW_ROLLING_EPA_COLUMNS: tuple[str, ...] = tuple(
    f"{venue}_{side}_rolling_{metric}"
    for venue in ("home", "away")
    for side in ("off", "def")
    for metric in ("epa_per_play", "pass_epa_per_play", "rush_epa_per_play")
)

V2_DERIVED_COLUMNS: tuple[str, ...] = (
    "success_rate_net",
    "opp_adj_epa_net",
    "qb_value_diff",
    "off_epa_sum",
    "def_epa_sum",
    "pace_sum",
)

#: The league level of a game with nothing admissible at its lock is NaN.
LEAGUE_LEVEL_NO_INFORMATION_SIGNATURE: dict[str, float | None] = {
    "league_points_level": None
}

#: Label columns ``create_target_variables`` adds. The live build runs the LeakageGate on
#: the combined matrix BEFORE they exist; ``home_margin`` would trip its ``margin`` keyword.
_LABEL_COLUMNS: tuple[str, ...] = (
    "target_wp",
    "home_win",
    "home_margin",
    "point_differential",
    "total_points",
)


class FeatureMatrixBuilderV2(FeatureMatrixBuilder):
    """The live builder with the windowed adjuster, the v2 QB tracker and raw levels."""

    def __init__(self) -> None:
        super().__init__()
        self.opponent_adj = WindowedOpponentAdjuster(window=10, min_opponent_games=4)
        self.qb_tracker = QBTrackerV2()

    def normalize_combined_features(
        self,
        processed_features: pd.DataFrame,
        target_season: int | None = None,
    ) -> pd.DataFrame:
        """No per-season z-score: v2 gold keeps every column at its raw level."""
        return processed_features

    def _merge_opponent_adjusted(
        self,
        combined_features: pd.DataFrame,
        games_df: pd.DataFrame,
        lock_frame: pd.Series,
        as_of_datetime: datetime,
        target_season: int | None,
        target_week: int | None,
    ) -> pd.DataFrame:
        """The live merge, then the raw rolling EPA columns it dropped merged back."""
        present = [c for c in RAW_ROLLING_EPA_COLUMNS if c in combined_features.columns]
        stash = combined_features.loc[:, ["game_id", *present]]
        merged = super()._merge_opponent_adjusted(
            combined_features,
            games_df,
            lock_frame,
            as_of_datetime,
            target_season,
            target_week,
        )
        dropped = [c for c in present if c not in merged.columns]
        if not dropped:
            return merged
        return merged.merge(
            stash.loc[:, ["game_id", *dropped]],
            on="game_id",
            how="left",
            validate="one_to_one",
        )


def add_v2_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """*frame* with the six v2 inputs computed from its per-side columns."""
    return frame.assign(
        success_rate_net=(
            frame["home_off_rolling_success_rate"]
            - frame["home_def_rolling_success_rate"]
        )
        - (
            frame["away_off_rolling_success_rate"]
            - frame["away_def_rolling_success_rate"]
        ),
        opp_adj_epa_net=(
            frame["home_off_rolling_opp_adj_epa_per_play"]
            - frame["home_def_rolling_opp_adj_epa_per_play"]
        )
        - (
            frame["away_off_rolling_opp_adj_epa_per_play"]
            - frame["away_def_rolling_opp_adj_epa_per_play"]
        ),
        qb_value_diff=frame["home_qb_adjustment"] - frame["away_qb_adjustment"],
        off_epa_sum=frame["home_off_rolling_opp_adj_epa_per_play"]
        + frame["away_off_rolling_opp_adj_epa_per_play"],
        def_epa_sum=frame["home_def_rolling_opp_adj_epa_per_play"]
        + frame["away_def_rolling_opp_adj_epa_per_play"],
        pace_sum=frame["home_off_rolling_neutral_pace"]
        + frame["away_off_rolling_neutral_pace"],
    )


def _provenance(family: pd.DataFrame) -> pd.DataFrame:
    """The gate's sidecar for a family: per_row where a time exists, else no_information."""
    times = family["information_time"].reset_index(drop=True)
    return pd.DataFrame(
        {
            "game_id": family["game_id"].reset_index(drop=True),
            "basis": np.where(
                times.notna(),
                InformationBasis.PER_ROW.value,
                InformationBasis.NO_INFORMATION.value,
            ),
            "information_time": times,
        }
    )


def _checked_family_values(
    name: str,
    family: pd.DataFrame,
    signature: Mapping[str, float | None],
    game_ids: set[str],
    lock_frame: pd.Series,
) -> pd.DataFrame:
    """The family's value columns for *game_ids*, after its own information-time check."""
    rows = family.loc[family["game_id"].isin(sorted(game_ids))].reset_index(drop=True)
    missing = game_ids - set(rows["game_id"])
    if missing:
        msg = f"family {name!r} has no row for {len(missing)} gold game(s): {sorted(missing)[:10]}"
        raise ValueError(msg)
    values = rows.drop(columns=["information_time"])
    state = InformationTimeGate().check(
        name,
        values,
        _provenance(rows),
        lock_frame,
        no_information_signature=signature,
    )
    if state is not SourceCheckState.CHECKED:
        msg = f"family {name!r} was not checked by the information-time gate ({state})"
        raise RuntimeError(msg)
    return values


def add_v2_families(
    matrices: dict[str, pd.DataFrame], games: pd.DataFrame, nflverse: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Merge the market-rating and league-level families, then the derived columns.

    Only games with a kickoff have a lock, so only they feed the families. Every gold game
    must get a row from each family; each family is checked by a fresh
    ``InformationTimeGate`` before it is merged. ``information_time`` never reaches gold.
    """
    dated = games.loc[games["kickoff_et"].notna()].reset_index(drop=True)
    lock_frame = build_lock_frame(dated)
    game_ids = {str(g) for matrix in matrices.values() for g in matrix["game_id"]}
    families = {
        "market_rating": (
            market_ratings(dated, nflverse),
            market_rating_no_information_signature(),
        ),
        "league_level": (
            league_points_level(dated),
            LEAGUE_LEVEL_NO_INFORMATION_SIGNATURE,
        ),
    }
    out = dict(matrices)
    for name, (family, signature) in families.items():
        values = _checked_family_values(name, family, signature, game_ids, lock_frame)
        out = {
            target: _merge_family(name, target, matrix, values)
            for target, matrix in out.items()
        }
    return {target: add_v2_derived_columns(matrix) for target, matrix in out.items()}


def _merge_family(
    name: str, target: str, matrix: pd.DataFrame, values: pd.DataFrame
) -> pd.DataFrame:
    """*matrix* with the family's values left-merged on ``game_id``; no column is overwritten."""
    clash = (set(values.columns) & set(matrix.columns)) - {"game_id"}
    if clash:
        msg = f"family {name!r} would overwrite {sorted(clash)} in the {target} matrix"
        raise ValueError(msg)
    return matrix.merge(values, on="game_id", how="left", validate="one_to_one")


def check_v2_matrices(matrices: dict[str, pd.DataFrame], built_at: datetime) -> None:
    """Check each finished v2 matrix: the combined-matrix ``LeakageGate`` and the dtype guard.

    The gate runs with the label columns dropped (the live build runs it before they exist);
    ``refuse_provenance_columns`` refuses any time-typed or provenance-named column.

    Raises:
        LeakageViolation: a matrix carries a leakage keyword or misses a required group.
        InformationTimeViolation: a matrix carries a time or a provenance column.
    """
    leakage_gate = LeakageGate()
    for matrix in matrices.values():
        features = matrix.drop(columns=[c for c in _LABEL_COLUMNS if c in matrix])
        leakage_gate.validate_combined_matrix(features, built_at)
        refuse_provenance_columns(matrix, build_clock_columns=BUILD_CLOCK_COLUMNS)


class FeatureMatrixBuilderV2Live(FeatureMatrixBuilderV2):
    """The v2 builder whose matrices are finished v2 gold: the families added and checked.

    The one sanctioned way v2 gold reaches ``data/gold``: the nightly build saves what this
    returns with the parent's ``save_feature_matrices``. ``build_gold_v2`` keeps using the bare
    ``FeatureMatrixBuilderV2`` and adds the families itself, so they are never added twice.
    """

    def generate_feature_matrices(
        self,
        target_season: int | None = None,
        target_week: int | None = None,
        as_of_datetime: datetime | None = None,
        *,
        excluded_game_ids: frozenset[str] = frozenset(),
        through_season: int | None = None,
        unplayed_through_lock: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """The v2 build, then the market-rating and league-level families, then the checks.

        An empty build is returned as is: there is nothing to add the families to.
        """
        matrices = super().generate_feature_matrices(
            target_season=target_season,
            target_week=target_week,
            as_of_datetime=as_of_datetime,
            excluded_game_ids=excluded_game_ids,
            through_season=through_season,
            unplayed_through_lock=unplayed_through_lock,
        )
        if not matrices:
            return matrices
        matrices = add_v2_families(
            matrices,
            load_dataframe("games", layer="silver"),
            load_schedules(list(NFLVERSE_SEASONS)),
        )
        check_v2_matrices(matrices, datetime.now(UTC))
        return matrices


def build_gold_v2(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Build the v2 matrices, check them, and write them and a manifest to *out_dir*.

    Raises:
        ValueError: *out_dir* is under ``data/`` (checked before any work).
        RuntimeError: a storage primitive asked to write during the build, or the build
            did not return all three matrices.
    """
    from pipeline import live_skip

    target_dir = refuse_data_dir(
        out_dir, what="the v2 gold", default_dir=DEFAULT_OUT_DIR
    )
    built_at = datetime.now(UTC)
    sink = RecordingSink()
    with active_sink(sink):
        matrices = FeatureMatrixBuilderV2().generate_feature_matrices(
            excluded_game_ids=live_skip.excluded_games(),
            unplayed_through_lock=built_at,
        )
        _refuse_recorded_writes(sink)
        if set(matrices) != set(TARGETS):
            msg = f"the build returned matrices {sorted(matrices)}, not {list(TARGETS)}"
            raise RuntimeError(msg)
        games = load_dataframe("games", layer="silver")
        nflverse = load_schedules(list(NFLVERSE_SEASONS))
        matrices = add_v2_families(matrices, games, nflverse)
        _refuse_recorded_writes(sink)

    check_v2_matrices(matrices, built_at)

    target_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        matrices[target].to_parquet(
            target_dir / f"features_{target}.parquet", index=False
        )
    shapes = {
        target: {"rows": len(matrix), "columns": len(matrix.columns)}
        for target, matrix in matrices.items()
    }
    manifest = {
        "built_at_utc": built_at.isoformat(),
        "git_commit": code_commit(PROJECT_ROOT),
        "matrices": shapes,
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote the v2 gold", out_dir=str(target_dir), shapes=shapes)
    return matrices


def _refuse_recorded_writes(sink: RecordingSink) -> None:
    """Refuse the build as soon as any storage primitive asked to write."""
    if sink.intended_writes:
        msg = (
            "the v2 build asked a storage primitive to write, and it may write nothing "
            f"outside its own output directory: {sink.intended_writes}"
        )
        raise RuntimeError(msg)


def build_parser() -> argparse.ArgumentParser:
    """The CLI: one option, where the v2 gold goes."""
    parser = argparse.ArgumentParser(
        description="Build the v2 gold matrices into a directory outside data/."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory, never under data/ (default {DEFAULT_OUT_DIR.as_posix()})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the v2 gold into ``--out-dir`` and print one line per matrix."""
    args = build_parser().parse_args(argv)
    matrices = build_gold_v2(args.out_dir)
    for target in TARGETS:
        matrix = matrices[target]
        print(f"features_{target}: {len(matrix)} rows x {len(matrix.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
