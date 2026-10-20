"""Cutoff-scoped re-fit training: every finished game before a cutoff, frozen recipe (Phase 37).

WHY A CUTOFF PATH EXISTS BESIDE THE SEASON-GRANULAR TRAINERS (SPEC R3, D40-06, research Q9)
-----------------------------------------------------------------------------------------
``models.train`` fits on whole completed seasons: its final fit reads
``conf.season_partition`` (2002-2025, clamped at ``LATEST_COMPLETED_SEASON``) and the walk-forward
masks are ``season < holdout``. Neither can train on a finished 2026 game, and neither can stop at
an instant inside a season. The first re-fit (Tue 2026-11-03) must do both, and Plan 37-09's
re-measure runs this same function once per walk-forward fold. So this module selects rows by
KICKOFF INSTANT and reuses the trainers' own methods (``select_features``, ``_create_model``,
``_get_default_params``, ``_predict_raw``, the converters, ``save``) without editing the trainers,
the partition or ``models.train``.

THE RULES, AS PRE-REGISTERED (``ROLLING-REFIT-PREREGISTRATION.md`` sections 7-8)
------------------------------------------------------------------------------
* A training row is a game from 2002 that kicked off STRICTLY before the cutoff and has a final
  score. A game at the cutoff is out. Each excluded game is counted by reason
  (``no_final_score``, ``no_gold_row``), and the row count must equal the finished games minus
  those exclusions.
* Labels come from silver scores by the gold label formula
  (``forward_ledger.settle.realized_values_from_scores``) and must equal gold's label wherever
  gold has one. No training row may rest on a PROVISIONAL Elo snapshot (COLD-02).
* Rows are put in (season, week, game_id) order before anything else: XGBoost's row subsampling
  draws by position, so a shuffled frame would be a different model (research Pitfall 6).
* The input pool is gold minus the standing excluded groups; inputs are re-picked by the trainers'
  own rule within the caps WP 20 / ATS 25 / O/U 25, with each trainer's fixed default settings.
* NO SEARCH OF ANY KIND. Nothing here tunes, opens a study or changes a setting (RFIT-07).
* Every fit runs with the thread pools pinned to ONE thread, measured in force
  (``models.train.pinned_thread_pool``): an unpinned XGBoost fit was measured to move a prediction
  by up to 3.98 points (research Q7). The pin sits in :func:`fit_target_at_cutoff`, the lowest
  public fitting entry point, because the determinism tests and Plan 37-09's folds call it
  directly.

THE CALIBRATION RULE (D-26) AND ITS OWNER NOTE
----------------------------------------------
The calibration season C is the latest season every one of whose games kicked off before the
cutoff (2025 for every 2026 cutoff; S-1 for a fold whose cutoff is season S's first kickoff). A
helper model, with inputs re-picked on seasons 2002..C-1, is fitted on 2002..C-1 and predicts C;
the WP Platt calibrator (isotonic fallback) and the ATS / O/U normal residual converter are fitted
on those C predictions only, so no calibrator ever sees predictions of its own training rows. The
shipped model then re-picks its inputs and fits on every eligible row. Owner note: this differs
from the row-19 models' rule (calibrated on 2023, inputs picked on 2002-2022), so a re-fit
generation changes two things at once versus the frozen arm; the owner chose this knowingly.

Artifacts are saved with ``update_latest=False`` (``BaseTrainer.save``): nothing here writes
``artifacts/latest.json``; the swap is a separate, integrity-checked step.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.refit_preregistration_2026 import (
    EXCLUDED_FEATURE_GROUPS,
    GROUP_VERDICT_NORMALIZED_SHA256,
    INPUT_CAPS,
    REFIT_RECIPE_ID,
    THREAD_LIMIT,
    TRAINING_START_SEASON,
)
from backtest.signal_lift import select_group_columns
from features.elo_features import (
    ProvisionalSnapshotAsTrainingInputError,
    assert_no_provisional_training_rows,
)
from forward_ledger.repro_key import gold_generation_key
from forward_ledger.settle import load_silver_games, realized_values_from_scores
from models.calibrate import ProbabilityCalibrator
from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.train import (
    GOLD_GENERATION_DIGEST_METADATA_KEY,
    GROUP_VERDICT_DIGEST_METADATA_KEY,
    THREAD_LIMIT_METADATA_KEY,
    pinned_thread_pool,
    verdict_exclusion,
)
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import BaseTrainer, threads_in_force
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from refit.witness import PREREGISTRATION_SHA256
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "EXCLUSION_NO_FINAL_SCORE",
    "EXCLUSION_NO_GOLD_ROW",
    "EligibleRows",
    "GenerationTraining",
    "RefitThreadCountError",
    "RefitTrainingError",
    "TargetFit",
    "calibration_season",
    "eligible_rows",
    "fit_target_at_cutoff",
    "predict_rows",
    "train_generation",
]

EXCLUSION_NO_FINAL_SCORE: str = "no_final_score"
EXCLUSION_NO_GOLD_ROW: str = "no_gold_row"

# The pre-registered integrity-check names this module refuses under
# (backtest.refit_preregistration_2026.INTEGRITY_CHECKS).
_CHECK_TRAINING_SCOPE = "training_scope"
_CHECK_INPUT_POOL = "input_pool_and_caps"
_CHECK_FIXED_SETTINGS = "fixed_settings"
_CHECK_CALIBRATION = "out_of_sample_calibration"

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
_TRAINER_FOR: dict[str, type[WPTrainer] | type[ATSTrainer] | type[OUTrainer]] = {
    "wp": WPTrainer,
    "ats": ATSTrainer,
    "ou": OUTrainer,
}
_ROW_ORDER: list[str] = ["season", "week", "game_id"]
_EXCLUDE_GROUPS_PROVENANCE = "verdict"


class RefitTrainingError(Exception):
    """A re-fit's training refused, naming the pre-registered check it failed.

    Inherits bare ``Exception``: several call sites in this repository catch ``ValueError`` /
    ``KeyError`` and degrade quietly, and a refused re-fit degraded into "carry on" would train on
    rows the record forbids.

    Attributes:
        check: The pre-registered integrity-check name (for example ``training_scope``).
    """

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


class RefitThreadCountError(RefitTrainingError):
    """A re-fit was asked to run at a thread count other than the pre-registered one."""


@dataclass(frozen=True)
class EligibleRows:
    """The training rows of one target at one cutoff, in canonical order.

    Attributes:
        frame: Gold rows of every eligible game, labelled from silver, sorted by
            (season, week, game_id) with a fresh 0..n-1 index.
        exclusions: ``{no_final_score: ids, no_gold_row: ids}``, each sorted.
        cutoff_utc: The cutoff, as a UTC instant.
    """

    frame: pd.DataFrame
    exclusions: dict[str, tuple[str, ...]]
    cutoff_utc: datetime


@dataclass(frozen=True)
class TargetFit:
    """One target fitted to one cutoff. Producing it writes nothing.

    Attributes:
        trainer: The trainer carrying the fitted model, inputs, calibration object and metadata,
            ready for ``trainer.save``.
        metadata: The trainer's metadata dict (the same object ``trainer.metadata`` holds).
        eligible: The rows it was fitted on and the recorded exclusions.
        calibration_game_ids: The season-C games the calibrator / converter was fitted on.
    """

    trainer: BaseTrainer
    metadata: dict[str, Any]
    eligible: EligibleRows
    calibration_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class GenerationTraining:
    """The three targets trained to one cutoff.

    Attributes:
        artifact_ids: ``{target: artifact directory name}``; empty when nothing was saved.
        metadata: ``{target: metadata dict}``.
        gold_generation_digest: The content identity of the gold the fits read.
        no_final_score_ids: The sorted union of every target's ``no_final_score`` exclusions,
            which the attempt record carries so a later attempt of the date holds them (R10).
    """

    artifact_ids: dict[str, str]
    metadata: dict[str, dict[str, Any]]
    gold_generation_digest: str
    no_final_score_ids: tuple[str, ...]


def _as_utc(cutoff_utc: datetime) -> datetime:
    """The cutoff as an aware UTC instant; a naive one is refused, never guessed."""
    if cutoff_utc.tzinfo is None:
        raise RefitTrainingError(
            _CHECK_TRAINING_SCOPE,
            f"cutoff {cutoff_utc!r} is naive; a training cutoff is an instant and must carry "
            "its zone",
        )
    return cutoff_utc.astimezone(UTC)


def _target_column(target: str) -> str:
    """The trainer's own label column for *target*, read from the trainer, never restated."""
    if target not in _TRAINER_FOR:
        msg = f"unknown target {target!r}; expected one of {list(_TARGETS)}"
        raise ValueError(msg)
    # The method returns a literal and reads no instance state, so the trainer's config
    # validation (and its log line) is not run just to learn a column name.
    return object.__new__(_TRAINER_FOR[target])._get_target_column()


def _kicked_off_before(games: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Silver games from the training start that kicked off strictly before *cutoff*."""
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    in_scope = (kickoff < pd.Timestamp(cutoff)) & (
        games["season"] >= TRAINING_START_SEASON
    )
    return games.loc[in_scope]


def _numeric(column: Any) -> pd.Series:
    """*column* as numbers, anything unreadable as missing."""
    return cast("pd.Series", pd.to_numeric(column, errors="coerce"))


def _columns(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The sub-frame of *names*, typed as the frame it is."""
    return cast("pd.DataFrame", frame[names])


def _label(frame: pd.DataFrame, column: str) -> pd.Series:
    """The label column *column*, typed as the series it is."""
    return cast("pd.Series", frame[column])


def _has_final_score(games: pd.DataFrame) -> pd.Series:
    return _numeric(games["home_score"]).notna() & _numeric(games["away_score"]).notna()


def eligible_rows(
    gold: pd.DataFrame, games: pd.DataFrame, cutoff_utc: datetime, target: str
) -> EligibleRows:
    """The training rows of *target* at *cutoff_utc* (SPEC R3 eligibility).

    Args:
        gold: The target's gold matrix (any row order).
        games: Silver ``games`` (``game_id``, ``season``, tz-aware ``kickoff_et``, scores).
        cutoff_utc: The scheduled cutoff, tz-aware.
        target: ``wp``, ``ats`` or ``ou``.

    Returns:
        The labelled rows in canonical order and the counted exclusions.

    Raises:
        RefitTrainingError: ``training_scope`` -- a naive cutoff, duplicate gold rows, a gold
            label that disagrees with silver's scores (naming the game), no eligible row, a
            season with finished games but no gold row at all, or a row count that does not
            equal the finished games minus the recorded exclusions.
    """
    cutoff = _as_utc(cutoff_utc)
    label_column = _target_column(target)

    duplicated = gold["game_id"].astype(str).duplicated()
    if duplicated.any():
        ids = sorted(gold.loc[duplicated, "game_id"].astype(str).unique())
        raise RefitTrainingError(
            _CHECK_TRAINING_SCOPE,
            f"{target} gold carries duplicate rows for {ids[:10]}; one game is one row",
        )

    kicked = _kicked_off_before(games, cutoff)
    scored = _has_final_score(kicked)
    finished = kicked.loc[scored]
    no_final_score = tuple(sorted(kicked.loc[~scored, "game_id"].astype(str)))

    finished_ids = set(finished["game_id"].astype(str))
    gold_ids = set(gold["game_id"].astype(str))
    no_gold_row = tuple(sorted(finished_ids - gold_ids))

    frame = gold.loc[gold["game_id"].astype(str).isin(sorted(finished_ids))].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    labels = realized_values_from_scores(finished)[target]
    silver_label = frame["game_id"].map(labels).astype(float)

    if label_column in frame.columns:
        gold_label = _numeric(frame[label_column])
        disagree = gold_label.notna() & (gold_label != silver_label)
        if disagree.any():
            ids = sorted(frame.loc[disagree, "game_id"])
            raise RefitTrainingError(
                _CHECK_TRAINING_SCOPE,
                f"{target} gold label {label_column!r} disagrees with the silver scores "
                f"for {ids[:10]} ({len(ids)} game(s)); the label formula and gold must agree",
            )
    frame[label_column] = silver_label

    if frame.empty:
        raise RefitTrainingError(
            _CHECK_TRAINING_SCOPE,
            f"{target} has no eligible training row before {cutoff.isoformat()}",
        )

    missing_seasons = sorted(
        {int(season) for season in finished["season"]}
        - {int(season) for season in frame["season"]}
    )
    if missing_seasons:
        raise RefitTrainingError(
            _CHECK_TRAINING_SCOPE,
            f"{target} gold has no row for any game of season(s) {missing_seasons}, "
            "which have finished games: this is not the full-history gold build",
        )

    expected = len(finished_ids) - len(no_gold_row)
    if len(frame) != expected:
        raise RefitTrainingError(
            _CHECK_TRAINING_SCOPE,
            f"{target} has {len(frame)} training rows but {len(finished_ids)} finished "
            f"games minus {len(no_gold_row)} without a gold row is {expected}",
        )

    frame = frame.sort_values(_ROW_ORDER, kind="mergesort").reset_index(drop=True)
    return EligibleRows(
        frame=frame,
        exclusions={
            EXCLUSION_NO_FINAL_SCORE: no_final_score,
            EXCLUSION_NO_GOLD_ROW: no_gold_row,
        },
        cutoff_utc=cutoff,
    )


def calibration_season(games: pd.DataFrame, cutoff_utc: datetime) -> int:
    """The latest season every one of whose games kicked off before the cutoff (D-26).

    Raises:
        RefitTrainingError: ``out_of_sample_calibration`` -- no season from the training start
            is complete before the cutoff.
    """
    cutoff = _as_utc(cutoff_utc)
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    last_kickoff = cast("pd.Series", kickoff.groupby(games["season"].astype(int)).max())
    complete_mask = (last_kickoff < pd.Timestamp(cutoff)).to_numpy(dtype=bool)
    complete = [
        season
        for season in last_kickoff.index.to_numpy(dtype=int)[complete_mask].tolist()
        if season >= TRAINING_START_SEASON
    ]
    if not complete:
        raise RefitTrainingError(
            _CHECK_CALIBRATION,
            f"no season from {TRAINING_START_SEASON} is complete before {cutoff.isoformat()}",
        )
    return max(complete)


def _frozen_pool(gold: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Gold minus the record's excluded groups, refused if the ratified verdict disagrees."""
    groups, digest = verdict_exclusion()
    if sorted(groups) != sorted(EXCLUDED_FEATURE_GROUPS):
        raise RefitTrainingError(
            _CHECK_INPUT_POOL,
            f"the ratified verdict excludes {list(groups)}, the record "
            f"{list(EXCLUDED_FEATURE_GROUPS)}",
        )
    if digest != GROUP_VERDICT_NORMALIZED_SHA256:
        raise RefitTrainingError(
            _CHECK_INPUT_POOL,
            f"the group verdict file hashes to {digest}, not the frozen "
            f"{GROUP_VERDICT_NORMALIZED_SHA256}",
        )
    return select_group_columns(
        gold, None, exclude_groups=EXCLUDED_FEATURE_GROUPS
    ), digest


def _fit_calibration(
    trainer: BaseTrainer, predictions: np.ndarray, actuals: np.ndarray
) -> str:
    """Fit the target's calibration object on season-C predictions; return its method name.

    The fitted object is assigned on the trainer, exactly as the trainers' own
    ``train_and_evaluate`` assigns it, because ``BaseTrainer.save`` writes ``calibrator.pkl``
    only from ``trainer.calibrator`` and ``converter_params`` only from the converters: an
    object left unassigned would make serving fall back to a hardcoded spread.
    """
    if isinstance(trainer, WPTrainer):
        result = ProbabilityCalibrator(
            primary_method="platt", fallback_method="isotonic"
        ).calibrate_probabilities(raw_probabilities=predictions, true_labels=actuals)
        trainer.calibrator = result.calibrator
        return str(result.method)
    if isinstance(trainer, ATSTrainer):
        trainer.residual_converter = trainer._fit_residual_converter(
            actuals, predictions
        )
        return trainer.residual_converter.distribution_type
    if isinstance(trainer, OUTrainer):
        trainer.total_converter = trainer._fit_total_converter(actuals, predictions)
        return trainer.total_converter.distribution_type
    msg = f"no calibration rule for trainer {type(trainer).__name__}"
    raise TypeError(msg)


def _new_trainer(target: str, helper_seasons: list[int], season_c: int) -> BaseTrainer:
    """A trainer whose split is the cutoff-relative D-26 one: helper seasons, then C, then C+1.

    Only the split's season lists are read by this module (the feature pool's id columns and the
    recorded seasons); no walk-forward stage of the trainer is ever run.
    """
    return _TRAINER_FOR[target](
        config=TemporalSplitConfig(
            train_seasons=helper_seasons,
            hp_val_seasons=[season_c],
            holdout_seasons=[season_c + 1],
        )
    )


def fit_target_at_cutoff(
    target: str,
    gold: pd.DataFrame,
    games: pd.DataFrame,
    cutoff_utc: datetime,
    *,
    snapshots: pd.DataFrame | None = None,
) -> TargetFit:
    """Fit *target* on every eligible row before *cutoff_utc* by the frozen recipe. No I/O.

    The whole body -- eligibility, the D-26 helper fit and calibration, the final selection and
    fit -- runs inside ``pinned_thread_pool(THREAD_LIMIT)``. A nested pin from
    :func:`train_generation` is harmless.

    Args:
        target: ``wp``, ``ats`` or ``ou``.
        gold: The target's gold matrix.
        games: Silver ``games``.
        cutoff_utc: The scheduled cutoff, tz-aware.
        snapshots: Backing Elo snapshots for the provisional-row guard (None reads the silver
            table through its own seam).

    Returns:
        The :class:`TargetFit`, its trainer ready for ``save``.

    Raises:
        RefitTrainingError: naming the pre-registered check that refused.
    """
    with pinned_thread_pool(THREAD_LIMIT):
        cutoff = _as_utc(cutoff_utc)
        pool_gold, verdict_digest = _frozen_pool(gold)
        eligible = eligible_rows(pool_gold, games, cutoff, target)
        frame = eligible.frame
        try:
            assert_no_provisional_training_rows(
                frame, f"refit:{target}", snapshots=snapshots
            )
        except ProvisionalSnapshotAsTrainingInputError as exc:
            raise RefitTrainingError(_CHECK_TRAINING_SCOPE, str(exc)) from exc

        season_c = calibration_season(games, cutoff)
        helper_seasons = list(range(TRAINING_START_SEASON, season_c))
        if not helper_seasons:
            raise RefitTrainingError(
                _CHECK_CALIBRATION,
                f"calibration season {season_c} leaves no helper season before it",
            )
        trainer = _new_trainer(target, helper_seasons, season_c)
        label_column = trainer._get_target_column()
        pool = WalkForwardSplitter(trainer.config, label_column)._feature_cols(frame)
        cap = INPUT_CAPS[target]
        params = trainer._get_default_params()

        helper_rows = frame.loc[frame["season"] < season_c]
        calibration_rows = frame.loc[frame["season"] == season_c]
        if helper_rows.empty or calibration_rows.empty:
            raise RefitTrainingError(
                _CHECK_CALIBRATION,
                f"{target}: {len(helper_rows)} helper rows and {len(calibration_rows)} "
                f"season-{season_c} rows; both must be non-empty",
            )
        helper_inputs = trainer.select_features(
            _columns(helper_rows, pool),
            _label(helper_rows, label_column),
            max_features=cap,
        )
        helper_model = trainer._create_model(params)
        helper_model.fit(helper_rows[helper_inputs], helper_rows[label_column])
        helper_predictions = np.asarray(
            trainer._predict_raw(
                helper_model, _columns(calibration_rows, helper_inputs)
            ),
            dtype=float,
        )
        calibration_method = _fit_calibration(
            trainer,
            helper_predictions,
            calibration_rows[label_column].to_numpy(dtype=float),
        )

        selected = trainer.select_features(
            _columns(frame, pool), _label(frame, label_column), max_features=cap
        )
        model = trainer._create_model(params)
        model.fit(frame[selected], frame[label_column])
        trainer.feature_names = selected
        trainer.model = model
        if isinstance(trainer, WPTrainer):
            # D33.1-R1: the WP Pipeline IS the preprocessing; one object on the way to disk.
            trainer.preprocessing = model
        thread_count = threads_in_force()

    calibration_ids = tuple(calibration_rows["game_id"].astype(str))
    trainer.metadata = {
        "target": target,
        "feature_names": list(selected),
        "refit_recipe": REFIT_RECIPE_ID,
        "cutoff_utc": cutoff.isoformat(),
        "training_rows": len(frame),
        "exclusions": {
            reason: list(ids) for reason, ids in eligible.exclusions.items()
        },
        "selected_inputs": list(selected),
        "calibration": {
            "season": season_c,
            "helper_seasons": helper_seasons,
            "helper_inputs": list(helper_inputs),
            "game_ids": list(calibration_ids),
            "method": calibration_method,
        },
        "best_params": params,
        "exclude_groups": list(EXCLUDED_FEATURE_GROUPS),
        "exclude_groups_provenance": _EXCLUDE_GROUPS_PROVENANCE,
        GROUP_VERDICT_DIGEST_METADATA_KEY: verdict_digest,
        THREAD_LIMIT_METADATA_KEY: thread_count,
        "preregistration_sha256": dict(PREREGISTRATION_SHA256),
    }
    logger.info(
        "Re-fit target fitted to cutoff",
        target=target,
        cutoff_utc=cutoff.isoformat(),
        training_rows=len(frame),
        n_inputs=len(selected),
        calibration_season=season_c,
    )
    return TargetFit(
        trainer=trainer,
        metadata=trainer.metadata,
        eligible=eligible,
        calibration_game_ids=calibration_ids,
    )


def predict_rows(fit: TargetFit, rows: pd.DataFrame) -> np.ndarray:
    """*fit*'s predictions for *rows*, as float64: WP calibrated probability, ATS margin, O/U total.

    WP is computed exactly as ``backtest.diagnose.score_deployed_artifacts`` computes it from the
    saved artifact: the raw probability, then the calibrator's ``predict``.
    """
    trainer = fit.trainer
    raw = np.asarray(
        trainer._predict_raw(trainer.model, _columns(rows, trainer.feature_names)),
        dtype=float,
    )
    if isinstance(trainer, WPTrainer):
        return np.asarray(trainer.calibrator.predict(raw), dtype=np.float64)
    return raw.astype(np.float64)


def _hold_no_score(games: pd.DataFrame, held: frozenset[str]) -> pd.DataFrame:
    """*games* with the scores of the *held* ids blanked, in memory only (R10)."""
    if not held:
        return games
    held_games = games.copy()
    mask = held_games["game_id"].astype(str).isin(sorted(held))
    held_games.loc[mask, ["home_score", "away_score"]] = np.nan
    return held_games


def _expected_training_rows(
    games: pd.DataFrame, cutoff: datetime, exclusions: Mapping[str, tuple[str, ...]]
) -> int:
    """Finished games from the training start to the cutoff, minus the no-gold-row exclusions."""
    kicked = _kicked_off_before(games, cutoff)
    finished = int(_has_final_score(kicked).sum())
    return finished - len(exclusions[EXCLUSION_NO_GOLD_ROW])


def train_generation(
    cutoff_utc: datetime,
    *,
    gold_dir: Path | str,
    silver_dir: Path | str,
    artifacts_dir: Path | str,
    thread_limit: int = THREAD_LIMIT,
    snapshots: pd.DataFrame | None = None,
    save: bool = True,
    held_no_score_ids: frozenset[str] = frozenset(),
) -> GenerationTraining:
    """Train WP, ATS and O/U to *cutoff_utc* and save them, never touching ``latest.json``.

    Args:
        cutoff_utc: The date's scheduled cutoff (the same on every attempt of the date).
        gold_dir: The directory holding the three gold matrices.
        silver_dir: The directory holding silver ``games.parquet``.
        artifacts_dir: Where the three artifacts are written.
        thread_limit: Must equal the pre-registered ``THREAD_LIMIT``; checked before anything
            is read.
        snapshots: Backing Elo snapshots for the provisional-row guard.
        save: Write the artifacts (False trains and returns the record only).
        held_no_score_ids: The ``no_final_score`` games of the date's FIRST attempt. Their
            scores are blanked in memory, so a result landing between attempts never changes
            the date's training rows (pre-registered, record section 8).

    Returns:
        The :class:`GenerationTraining` record.

    Raises:
        RefitThreadCountError: ``thread_limit`` is not the pre-registered count.
        RefitTrainingError: a pre-registered training check refused.
    """
    if thread_limit != THREAD_LIMIT:
        raise RefitThreadCountError(
            _CHECK_FIXED_SETTINGS,
            f"a re-fit runs at thread count {THREAD_LIMIT}; {thread_limit} was requested",
        )
    cutoff = _as_utc(cutoff_utc)
    games = _hold_no_score(load_silver_games(silver_dir), held_no_score_ids)
    gold_root = Path(gold_dir)
    digest = gold_generation_key(gold_root)

    artifact_ids: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    no_final_score: set[str] = set()
    for target in _TARGETS:
        gold = pd.read_parquet(gold_root / f"features_{target}.parquet")
        with pinned_thread_pool(THREAD_LIMIT):
            fit = fit_target_at_cutoff(target, gold, games, cutoff, snapshots=snapshots)
        expected = _expected_training_rows(games, cutoff, fit.eligible.exclusions)
        if fit.metadata["training_rows"] != expected:
            raise RefitTrainingError(
                _CHECK_TRAINING_SCOPE,
                f"{target} trained on {fit.metadata['training_rows']} rows; the finished "
                f"games minus the recorded exclusions are {expected}",
            )
        fit.metadata[GOLD_GENERATION_DIGEST_METADATA_KEY] = digest
        no_final_score.update(fit.eligible.exclusions[EXCLUSION_NO_FINAL_SCORE])
        metadata[target] = fit.metadata
        if save:
            artifact_ids[target] = fit.trainer.save(Path(artifacts_dir)).name

    return GenerationTraining(
        artifact_ids=artifact_ids,
        metadata=metadata,
        gold_generation_digest=digest,
        no_final_score_ids=tuple(sorted(no_final_score)),
    )
