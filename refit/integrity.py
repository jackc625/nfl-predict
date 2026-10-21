"""The pre-registered integrity checks a re-fit must pass before it goes live (RFIT-06, SPEC R5).

GATE-FREE BY OWNER RULING, NEVER CHECK-FREE
-------------------------------------------
The record (``ROLLING-REFIT-PREREGISTRATION.md`` section 10) rules the re-fit gate-free: "the
shadow arm is the control and the season close is the verdict", so no performance test stands
between a re-fit and the live arm. A broken model must still never go live, so every check here is
an ENGINEERING fact -- the record was not edited, the rows were the rule's rows, the inputs and
settings are the frozen ones, the calibration never saw its own rows, the re-measure finished, the
models score, the next decision run can stamp them -- and none reads a result.

THE ORDER IS THE RECORD'S
-------------------------
The checks run in ``backtest.refit_preregistration_2026.INTEGRITY_CHECKS`` order and the first
failure refuses the attempt by name (:class:`IntegrityRefusal`); :func:`run_integrity_checks` runs
every pre-swap check in that order. The last two, ``clean_code_commit`` and ``bundle_validation``,
belong to the swap itself (``refit.swap``): one must hold before the permanent install intent is
written, the other is the swap's own validation. ``recipe_fingerprint`` is also run by the hook
BEFORE any training (P4); running it again here costs one hash of two files.

WHAT THE TRAINING CHECKS READ
-----------------------------
The training checks re-read what ``refit.training.train_generation`` RECORDED for each target (its
metadata, the dict saved into the artifact's ``metadata.json``) against the frozen record and the
caller's ``games`` frame -- the same frame the training read, with any held no-score game already
masked (R10) -- so a recorded fact that disagrees with the rule refuses the install by name.

THE INTEGRITY FRAME AND THE DRY STAMP (D-34, D-25, research Pitfall 1)
---------------------------------------------------------------------
:func:`integrity_frame` is the stored 2026 gold rows of the NFL week that just finished plus every
gold row of a slate built the same day; nothing is built for a future slate. Each new model must
load, read only gold columns and return one finite value per row (:func:`check_scores_frame`).
:func:`check_dry_stamp_and_validate` then decides that frame as the next decision run will -- the
generation's own bet-rule record, the served ids resolved as ``latest.json`` will resolve them,
the week's first decision run as the instant -- and passes every row through the ledger's stamping
step and new-row validation with NO snapshot, copy or ledger write, proving the next decision run
accepts the new ids.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from backtest.diagnose import score_deployed_artifacts
from backtest.refit_preregistration_2026 import (
    AA_FIRST_WEEK,
    EV_FLOORS,
    EXCLUDED_FEATURE_GROUPS,
    GROUP_VERDICT_NORMALIZED_SHA256,
    INPUT_CAPS,
    INTEGRITY_CHECKS,
    MIN_USABLE_REMEASURE_ROWS,
    MODEL_SETTINGS,
    REFIT_DATE_LISTS,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    THREAD_LIMIT,
    TRAINING_START_SEASON,
)
from backtest.signal_lift import excluded_columns
from backtest.weekly_bet_list import build_weekly_decision_bundle, load_bet_rule_record
from data.upstream_live import load_live_manifest
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1
from forward_ledger.repro_key import build_repro_key
from forward_ledger.runner import (
    _has_verdict_rows,
    _silver_odds,
    _verdict_scope_or_none,
)
from forward_ledger.settle import load_silver_games
from forward_ledger.snapshots import compute_snapshot_digest
from forward_ledger.stamps import stamp_ledger_rows
from forward_ledger.store import _validate_new_row, read_entries
from models.artifacts import ResolvedArtifacts, load_model_artifact
from models.train import GROUP_VERDICT_DIGEST_METADATA_KEY, THREAD_LIMIT_METADATA_KEY
from refit.aa import week_first_decision_run
from refit.training import RefitTrainingError, calibration_season
from refit.witness import fingerprint_mismatches

if TYPE_CHECKING:
    from refit.generations import Generation
    from refit.remeasure import BetRuleNumbers
    from refit.training import GenerationTraining

__all__ = [
    "IntegrityRefusal",
    "check_aa_proof",
    "check_dry_stamp_and_validate",
    "check_fixed_settings",
    "check_input_pool_and_caps",
    "check_out_of_sample_calibration",
    "check_recipe_fingerprint",
    "check_remeasure_complete",
    "check_scores_frame",
    "check_training_scope",
    "integrity_frame",
    "run_integrity_checks",
]

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
_ROW_ORDER: list[str] = ["season", "week", "game_id"]

_RECIPE_FINGERPRINT = "recipe_fingerprint"
_AA_PROOF = "aa_proof"
_TRAINING_SCOPE = "training_scope"
_INPUT_POOL_AND_CAPS = "input_pool_and_caps"
_FIXED_SETTINGS = "fixed_settings"
_OUT_OF_SAMPLE_CALIBRATION = "out_of_sample_calibration"
_REMEASURE_COMPLETE = "remeasure_complete"
_SCORES_FRAME = "scores_frame"
_DRY_STAMP_AND_VALIDATE = "dry_stamp_and_validate"

# The scored column every target's output carries (backtest.diagnose.score_deployed_artifacts):
# WP's calibrated probability, ATS's margin, O/U's total.
_SCORED_COLUMN = "model_prob"

# The training's recorded exclusion of a finished game with no gold row (refit.training).
_NO_GOLD_ROW = "no_gold_row"


class IntegrityRefusal(Exception):
    """A re-fit refused by one pre-registered integrity check, named in ``check``.

    Inherits bare ``Exception``: several call sites in this repository catch ``ValueError`` /
    ``KeyError`` and degrade quietly, and a refused re-fit degraded into "carry on" would put a
    model the record forbids on the live arm. The hook records the attempt ``refused`` with this
    check, and the FAILED push names it.

    Attributes:
        check: One of ``INTEGRITY_CHECKS``.
    """

    def __init__(self, check: str, message: str) -> None:
        if check not in INTEGRITY_CHECKS:
            msg = (
                f"{check!r} is not a pre-registered integrity check {INTEGRITY_CHECKS}"
            )
            raise ValueError(msg)
        super().__init__(f"{check}: {message}")
        self.check = check


def _target_metadata(
    training: GenerationTraining, target: str, check: str
) -> Mapping[str, Any]:
    """The metadata *training* recorded for *target*; a target it did not record is refused."""
    metadata = training.metadata.get(target)
    if not metadata:
        raise IntegrityRefusal(check, f"the training recorded no {target} model")
    return metadata


def _as_utc(instant: datetime, check: str) -> datetime:
    if instant.tzinfo is None:
        raise IntegrityRefusal(check, f"the cutoff {instant!r} is naive")
    return instant.astimezone(UTC)


# ---------------------------------------------------------------------------
# 1. recipe_fingerprint, 2. aa_proof
# ---------------------------------------------------------------------------


def check_recipe_fingerprint(repo_root: Path) -> None:
    """The running checkout's two record files hash to the pre-registered witness (P4)."""
    mismatched = fingerprint_mismatches(repo_root)
    if mismatched:
        raise IntegrityRefusal(
            _RECIPE_FINGERPRINT,
            f"{mismatched} are missing or no longer hash to the pre-registered witness "
            "(refit.witness); a re-fit never runs on an edited record",
        )


def check_aa_proof(
    list_week: int,
    aa_records: Sequence[Mapping[str, Any]],
    *,
    first_refit: bool,
) -> None:
    """For the first re-fit only: the list in force was selected by a clean A/A week (section 5).

    The list of A/A week *list_week* is in force only when that week's first recorded A/A result
    is clean and every week from 8 before it has a first result that is not; after the first
    re-fit installs, the A/A check is never applied again.
    """
    if not first_refit:
        return
    if list_week not in REFIT_DATE_LISTS:
        raise IntegrityRefusal(
            _AA_PROOF,
            f"no pre-registered date list is selected by A/A week {list_week}",
        )
    first_by_week: dict[int, Mapping[str, Any]] = {}
    for result in aa_records:
        if result["season"] == REFIT_SEASON:
            first_by_week.setdefault(int(result["week"]), result)
    for week in range(AA_FIRST_WEEK, list_week + 1):
        result = first_by_week.get(week)
        if result is None:
            raise IntegrityRefusal(
                _AA_PROOF,
                f"A/A week {week} has no recorded result, so the list of week {list_week} "
                "was not selected by the A/A rule",
            )
        if week < list_week and result["clean"]:
            raise IntegrityRefusal(
                _AA_PROOF,
                f"A/A week {week} was recorded clean, so the list in force is week {week}'s, "
                f"not week {list_week}'s",
            )
        if week == list_week and not result["clean"]:
            raise IntegrityRefusal(
                _AA_PROOF,
                f"A/A week {list_week} was not recorded clean (reason "
                f"{result.get('reason')!r}), so it selects no date list",
            )


# ---------------------------------------------------------------------------
# 3. training_scope, 4. input_pool_and_caps, 5. fixed_settings, 6. out_of_sample_calibration
# ---------------------------------------------------------------------------


def _finished_before(games: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Games from the training start that kicked off strictly before *cutoff* with a final score."""
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    home = cast("pd.Series", pd.to_numeric(games["home_score"], errors="coerce"))
    away = cast("pd.Series", pd.to_numeric(games["away_score"], errors="coerce"))
    mask = (
        (kickoff < pd.Timestamp(cutoff))
        & (games["season"].astype(int) >= TRAINING_START_SEASON)
        & home.notna()
        & away.notna()
    )
    return cast("pd.DataFrame", games.loc[mask])


def check_training_scope(
    training: GenerationTraining, games: pd.DataFrame, cutoff_utc: datetime
) -> None:
    """Every target trained to the date's own cutoff on exactly the rule's row count (section 8).

    The count is the finished games kicked off strictly before the cutoff from 2002, minus the
    recorded ``no_gold_row`` exclusions; a row at or after the cutoff, or one row too many or too
    few, refuses.
    """
    cutoff = _as_utc(cutoff_utc, _TRAINING_SCOPE)
    finished = len(_finished_before(games, cutoff))
    for target in _TARGETS:
        metadata = _target_metadata(training, target, _TRAINING_SCOPE)
        recorded_cutoff = metadata.get("cutoff_utc")
        if recorded_cutoff != cutoff.isoformat():
            raise IntegrityRefusal(
                _TRAINING_SCOPE,
                f"{target} was trained to cutoff {recorded_cutoff!r}, not the date's "
                f"{cutoff.isoformat()!r}",
            )
        no_gold_row = metadata.get("exclusions", {}).get(_NO_GOLD_ROW)
        if no_gold_row is None:
            raise IntegrityRefusal(
                _TRAINING_SCOPE, f"{target} recorded no {_NO_GOLD_ROW!r} exclusions"
            )
        expected = finished - len(no_gold_row)
        if metadata.get("training_rows") != expected:
            raise IntegrityRefusal(
                _TRAINING_SCOPE,
                f"{target} trained on {metadata.get('training_rows')!r} rows; the {finished} "
                f"finished games before the cutoff minus {len(no_gold_row)} without a gold "
                f"row are {expected}",
            )


def check_input_pool_and_caps(training: GenerationTraining) -> None:
    """Every selected input comes from the frozen pool and each target stays within its cap."""
    for target in _TARGETS:
        metadata = _target_metadata(training, target, _INPUT_POOL_AND_CAPS)
        inputs = [str(name) for name in metadata.get("feature_names") or []]
        if not inputs:
            raise IntegrityRefusal(
                _INPUT_POOL_AND_CAPS, f"{target} recorded no selected input"
            )
        if len(inputs) > INPUT_CAPS[target]:
            raise IntegrityRefusal(
                _INPUT_POOL_AND_CAPS,
                f"{target} selected {len(inputs)} inputs, over its cap {INPUT_CAPS[target]}",
            )
        forbidden = excluded_columns(
            pd.DataFrame(columns=pd.Index(inputs)), EXCLUDED_FEATURE_GROUPS
        )
        if forbidden:
            raise IntegrityRefusal(
                _INPUT_POOL_AND_CAPS,
                f"{target} selected {forbidden}, which belong to the excluded groups "
                f"{list(EXCLUDED_FEATURE_GROUPS)}",
            )
        if sorted(metadata.get("exclude_groups") or []) != sorted(
            EXCLUDED_FEATURE_GROUPS
        ):
            raise IntegrityRefusal(
                _INPUT_POOL_AND_CAPS,
                f"{target} excluded the groups {metadata.get('exclude_groups')!r}, not the "
                f"record's {list(EXCLUDED_FEATURE_GROUPS)}",
            )
        if (
            metadata.get(GROUP_VERDICT_DIGEST_METADATA_KEY)
            != GROUP_VERDICT_NORMALIZED_SHA256
        ):
            raise IntegrityRefusal(
                _INPUT_POOL_AND_CAPS,
                f"{target} read a group verdict hashing to "
                f"{metadata.get(GROUP_VERDICT_DIGEST_METADATA_KEY)!r}, not the frozen "
                f"{GROUP_VERDICT_NORMALIZED_SHA256}",
            )


def check_fixed_settings(training: GenerationTraining) -> None:
    """Every model carries exactly the fixed settings, at thread count 1, under the re-fit recipe.

    The settings are compared with the record's ``MODEL_SETTINGS`` (the trainers' defaults as
    frozen), so a tuned or edited setting -- the trace any search would leave -- refuses.
    """
    for target in _TARGETS:
        metadata = _target_metadata(training, target, _FIXED_SETTINGS)
        if metadata.get("refit_recipe") != REFIT_RECIPE_ID:
            raise IntegrityRefusal(
                _FIXED_SETTINGS,
                f"{target} records recipe {metadata.get('refit_recipe')!r}, not "
                f"{REFIT_RECIPE_ID!r}",
            )
        settings = metadata.get("best_params")
        if settings != MODEL_SETTINGS[target]:
            raise IntegrityRefusal(
                _FIXED_SETTINGS,
                f"{target} was fitted with {settings!r}, not the fixed settings "
                f"{MODEL_SETTINGS[target]!r}",
            )
        threads = metadata.get(THREAD_LIMIT_METADATA_KEY)
        if threads != THREAD_LIMIT:
            raise IntegrityRefusal(
                _FIXED_SETTINGS,
                f"{target} recorded thread count {threads!r}; a re-fit runs at "
                f"{THREAD_LIMIT}",
            )


def check_out_of_sample_calibration(
    training: GenerationTraining, games: pd.DataFrame
) -> None:
    """Each calibrator was fitted only on its helper's predictions of season C (D-26).

    Season C must be the latest season wholly before the cutoff, the helper must have trained on
    2002 through C-1 only, and every calibration game must be a season-C game of *games* -- never
    one of the helper's own training rows.
    """
    season_of = dict(
        zip(
            games["game_id"].astype(str),
            games["season"].astype(int),
            strict=True,
        )
    )
    for target in _TARGETS:
        metadata = _target_metadata(training, target, _OUT_OF_SAMPLE_CALIBRATION)
        calibration = metadata.get("calibration") or {}
        recorded_cutoff = metadata.get("cutoff_utc")
        try:
            expected_season = calibration_season(
                games, datetime.fromisoformat(str(recorded_cutoff))
            )
        except (RefitTrainingError, ValueError) as exc:
            raise IntegrityRefusal(
                _OUT_OF_SAMPLE_CALIBRATION,
                f"{target}: no calibration season follows from the recorded cutoff "
                f"{recorded_cutoff!r}: {exc}",
            ) from exc
        season = calibration.get("season")
        if season != expected_season:
            raise IntegrityRefusal(
                _OUT_OF_SAMPLE_CALIBRATION,
                f"{target} calibrated on season {season!r}; the latest season wholly before "
                f"the cutoff is {expected_season}",
            )
        helper = list(calibration.get("helper_seasons") or [])
        if helper != list(range(TRAINING_START_SEASON, expected_season)):
            raise IntegrityRefusal(
                _OUT_OF_SAMPLE_CALIBRATION,
                f"{target}'s helper trained on seasons {helper}, not "
                f"{TRAINING_START_SEASON}..{expected_season - 1}",
            )
        game_ids = [str(game_id) for game_id in calibration.get("game_ids") or []]
        if not game_ids:
            raise IntegrityRefusal(
                _OUT_OF_SAMPLE_CALIBRATION, f"{target} recorded no calibration game"
            )
        foreign = sorted(
            game_id for game_id in game_ids if season_of.get(game_id) != expected_season
        )
        if foreign:
            raise IntegrityRefusal(
                _OUT_OF_SAMPLE_CALIBRATION,
                f"{target} calibrated on {foreign[:10]}, which are not season-"
                f"{expected_season} games: a calibrator may only see games its helper never "
                "trained on",
            )


# ---------------------------------------------------------------------------
# 7. remeasure_complete
# ---------------------------------------------------------------------------


def check_remeasure_complete(numbers: BetRuleNumbers) -> None:
    """Section 9 finished for all three targets, 100+ usable rows each, at the fixed EV floors."""
    for target in _TARGETS:
        usable = numbers.rows_by_target.get(target, {}).get("usable")
        if usable is None or usable < MIN_USABLE_REMEASURE_ROWS:
            raise IntegrityRefusal(
                _REMEASURE_COMPLETE,
                f"{target} has {usable!r} usable re-measure rows; fewer than "
                f"{MIN_USABLE_REMEASURE_ROWS} is a refusal, never a no-bets generation",
            )
        if target not in numbers.edge_tier_thresholds:
            raise IntegrityRefusal(
                _REMEASURE_COMPLETE, f"{target} has no re-measured edge-tier pair"
            )
    if dict(numbers.ev_floors) != EV_FLOORS:
        raise IntegrityRefusal(
            _REMEASURE_COMPLETE,
            f"the EV floors {dict(numbers.ev_floors)} are not the fixed {EV_FLOORS}",
        )


# ---------------------------------------------------------------------------
# The integrity frame (D-34, D-25), 8. scores_frame, 9. dry_stamp_and_validate
# ---------------------------------------------------------------------------


def integrity_frame(
    gold_dir: Path | str,
    games: pd.DataFrame,
    *,
    finished_week: int,
    slate_game_ids: frozenset[str],
) -> dict[str, pd.DataFrame]:
    """Per target, the stored 2026 gold rows of *finished_week* plus every row of today's slate.

    The finished week's games are read from the recorded schedule (*games*); the slate is the one
    built the same day (a catch-up on a prediction day), empty otherwise. Nothing is built: only
    rows already stored in gold are returned, in (season, week, game_id) order.
    """
    in_week = (games["season"].astype(int) == REFIT_SEASON) & (
        games["week"].astype(int) == finished_week
    )
    wanted = sorted(
        set(games.loc[in_week, "game_id"].astype(str))
        | {str(game_id) for game_id in slate_game_ids}
    )
    frames: dict[str, pd.DataFrame] = {}
    for target in _TARGETS:
        gold = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
        rows = gold.loc[gold["game_id"].astype(str).isin(wanted)]
        frames[target] = rows.sort_values(_ROW_ORDER, kind="mergesort").reset_index(
            drop=True
        )
    return frames


def check_scores_frame(
    artifact_ids: Mapping[str, str],
    frame: Mapping[str, pd.DataFrame],
    artifacts_dir: Path | str,
) -> None:
    """Each new model loads, reads only gold columns, and scores every frame row finitely.

    WP's output must also lie within [0, 1]. An empty frame proves nothing and is refused.
    """
    root = Path(artifacts_dir)
    for target in _TARGETS:
        version = artifact_ids[target]
        rows = frame.get(target)
        if rows is None or rows.empty:
            raise IntegrityRefusal(
                _SCORES_FRAME, f"the integrity frame holds no {target} row to score"
            )
        try:
            loaded = load_model_artifact(target, version=version, artifacts_dir=root)
        except Exception as exc:  # named in the refusal, never swallowed
            raise IntegrityRefusal(
                _SCORES_FRAME,
                f"{version!r} does not load: {type(exc).__name__}: {exc}",
            ) from exc
        missing = [name for name in loaded["feature_list"] if name not in rows.columns]
        if missing:
            raise IntegrityRefusal(
                _SCORES_FRAME,
                f"{version!r} reads {missing[:10]}, which are not gold columns",
            )
        try:
            scored = score_deployed_artifacts(
                target, gold_df=rows, artifacts_dir=root, version=version
            )
        except Exception as exc:  # named in the refusal, never swallowed
            raise IntegrityRefusal(
                _SCORES_FRAME,
                f"{version!r} could not score the frame: {type(exc).__name__}: {exc}",
            ) from exc
        values = scored[_SCORED_COLUMN].to_numpy(dtype=float)
        if len(values) != len(rows):
            raise IntegrityRefusal(
                _SCORES_FRAME,
                f"{version!r} returned {len(values)} values for {len(rows)} rows",
            )
        not_finite = ~np.isfinite(values)
        if not_finite.any():
            ids = sorted(rows.loc[not_finite, "game_id"].astype(str))
            raise IntegrityRefusal(
                _SCORES_FRAME, f"{version!r} returned a non-finite value for {ids[:10]}"
            )
        if target == "wp" and ((values < 0.0) | (values > 1.0)).any():
            raise IntegrityRefusal(
                _SCORES_FRAME,
                f"{version!r} returned a win probability outside [0, 1]",
            )


def _served_ids(generation: Generation, artifacts_dir: Path) -> ResolvedArtifacts:
    """The ids the next decision run will resolve once ``latest.json`` names this generation.

    The four pointers are the generation's; the converter is read from the new blend's own
    binding, exactly as ``models.artifacts.resolve_production_artifacts`` reads it.
    """
    # Imported lazily, as models.artifacts does: the blender pulls in pandas and scipy.
    from models.blending import MarketBlender

    blender = MarketBlender.from_artifacts(artifacts_dir, version=generation.ids.blend)
    return ResolvedArtifacts(
        wp=generation.ids.wp,
        ats=generation.ids.ats,
        ou=generation.ids.ou,
        blend=generation.ids.blend,
        converter=blender.market_probability_artifact_id,
    )


def _decision_weeks(
    games: pd.DataFrame,
    season: int,
    finished_week: int,
    slate_game_ids: frozenset[str],
) -> list[int]:
    """The finished week, and the slate's week when a slate was built today."""
    in_season = games.loc[games["season"].astype(int) == season]
    week_of = dict(
        zip(
            in_season["game_id"].astype(str),
            in_season["week"].astype(int),
            strict=True,
        )
    )
    unknown = sorted(
        str(game_id) for game_id in slate_game_ids if str(game_id) not in week_of
    )
    if unknown:
        msg = f"the slate games {unknown[:10]} are not in the {season} schedule"
        raise ValueError(msg)
    return sorted(
        {finished_week} | {week_of[str(game_id)] for game_id in slate_game_ids}
    )


def check_dry_stamp_and_validate(
    *,
    season: int,
    finished_week: int,
    slate_game_ids: frozenset[str],
    generation: Generation,
    artifacts_dir: Path | str,
    gold_dir: Path | str,
    silver_dir: Path | str,
    ledger_dir: Path | str,
    manifest_dir: Path | str | None = None,
) -> None:
    """Decide the integrity frame as the next decision run will, stamp it and validate it. NO write.

    For the finished week (and the slate's week, when one exists) the decision is built through
    the generation-aware bet rule -- the generation's ids as served, its bet-rule record, the
    fits and edge thresholds ``load_bet_rule_record`` reads from it, the week's first decision
    run as the instant -- then the reproduction key and the snapshot digest are computed in
    memory, every row is stamped ``recipe-2026-refit-v1`` with the generation's ids as the only
    registered ids, and every row goes through ``forward_ledger.store._validate_new_row``. No
    snapshot, artifact copy or ledger write is ever made.

    Args:
        season: The re-fit season.
        finished_week: The NFL week that just finished.
        slate_game_ids: The games of a slate built today (empty on a no-games day).
        generation: The candidate generation (its ``bet_rule_path`` is
            ``refit.records.bet_rule_record_path(ledger_dir, sha)``).
        artifacts_dir, gold_dir, silver_dir: The stores the decision reads.
        ledger_dir: The ledger, read only (its verdict rows decide the scope rule).
        manifest_dir: The live-manifest directory (``config/upstream_live`` when None).

    Raises:
        IntegrityRefusal: ``dry_stamp_and_validate``, naming what the decision, the stamping or
            the validation refused.
    """
    try:
        games = load_silver_games(silver_dir)
        weeks = _decision_weeks(games, season, finished_week, slate_game_ids)
        resolved = _served_ids(generation, Path(artifacts_dir))
        fits, thresholds = load_bet_rule_record(generation.bet_rule_path)
        entries = read_entries(Path(ledger_dir))
        scope = _verdict_scope_or_none()
        ledger_has_verdict_rows = _has_verdict_rows(entries)
        odds = _silver_odds(silver_dir)
        manifest = load_live_manifest(season, manifest_dir=manifest_dir)
        for week in weeks:
            decided_at = week_first_decision_run(games, season, week)
            bundle = build_weekly_decision_bundle(
                season,
                week,
                artifacts_dir=Path(artifacts_dir),
                gold_dir=Path(gold_dir),
                silver_dir=Path(silver_dir),
                now=decided_at,
                excluded_game_ids=frozenset(),
                resolved=resolved,
                chain_fit_path=generation.bet_rule_path,
                fits=fits,
                edge_thresholds=thresholds,
            )
            if bundle.frame.empty:
                raise IntegrityRefusal(
                    _DRY_STAMP_AND_VALIDATE,
                    f"the dry decision of {season} week {week} decided nothing, so it proves "
                    "nothing about the next decision run",
                )
            repro = build_repro_key(
                manifest=manifest,
                gold_dir=gold_dir,
                odds=odds,
                game_ids=sorted(bundle.frame["game_id"].astype(str).unique()),
                decided_at=decided_at,
            )
            stamped = stamp_ledger_rows(
                bundle.frame,
                resolved=bundle.resolved,
                repro=repro,
                snapshot_digest=compute_snapshot_digest(bundle),
                scope=scope,
                ledger_has_verdict_rows=ledger_has_verdict_rows,
                recipe_id=REFIT_RECIPE_ID,
                registered_ids=(generation.ids,),
            )
            columns = cast("pd.DataFrame", stamped[list(IMMUTABLE_COLUMNS_V1)])
            for row in columns.to_dict(orient="records"):
                _validate_new_row(row)
    except IntegrityRefusal:
        raise
    except Exception as exc:  # named in the refusal, never swallowed
        raise IntegrityRefusal(
            _DRY_STAMP_AND_VALIDATE,
            f"the next decision run would refuse the new ids: {type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# The pre-swap checks, in the record's order
# ---------------------------------------------------------------------------


def run_integrity_checks(
    *,
    repo_root: Path,
    list_week: int,
    aa_records: Sequence[Mapping[str, Any]],
    first_refit: bool,
    training: GenerationTraining,
    games: pd.DataFrame,
    cutoff_utc: datetime,
    numbers: BetRuleNumbers,
    generation: Generation,
    season: int,
    finished_week: int,
    slate_game_ids: frozenset[str],
    artifacts_dir: Path | str,
    gold_dir: Path | str,
    silver_dir: Path | str,
    ledger_dir: Path | str,
    manifest_dir: Path | str | None = None,
) -> None:
    """Run every pre-swap check in ``INTEGRITY_CHECKS`` order; the first failure refuses.

    ``clean_code_commit`` and ``bundle_validation`` follow in ``refit.swap.install_generation``.

    Raises:
        IntegrityRefusal: naming the first check that failed.
    """
    check_recipe_fingerprint(repo_root)
    check_aa_proof(list_week, aa_records, first_refit=first_refit)
    check_training_scope(training, games, cutoff_utc)
    check_input_pool_and_caps(training)
    check_fixed_settings(training)
    check_out_of_sample_calibration(training, games)
    check_remeasure_complete(numbers)
    model_ids = {target: generation.ids.model_id_for(target) for target in _TARGETS}
    frame = integrity_frame(
        gold_dir, games, finished_week=finished_week, slate_game_ids=slate_game_ids
    )
    check_scores_frame(model_ids, frame, artifacts_dir)
    check_dry_stamp_and_validate(
        season=season,
        finished_week=finished_week,
        slate_game_ids=slate_game_ids,
        generation=generation,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        ledger_dir=ledger_dir,
        manifest_dir=manifest_dir,
    )
