"""The recorded owner install of the v2 models (MIDSEASON-SWITCH-RECORD.md section 4).

WHAT ONE CALL DOES
------------------
:func:`run_owner_install` is the one owner command of the 2026 mid-season switch
(``scripts/install_v2.py``). In order:

1. **Preconditions** (:func:`install_preconditions`): the v2 recipe is not already in force, and
   no 2026 game of weeks 1-9 is still before its lock, so no week-9 decision can be stamped with
   a v2 model. A refusal here builds and writes nothing.
2. **A clean commit** (``refit.swap._code_commit_problem``): the permanent install record names the
   code that ran, so an unknown or dirty commit is refused before anything is built.
3. **v2 gold**: the shared gold matrices are rebuilt with ``FeatureMatrixBuilderV2Live``.
4. **Training, once** (:func:`train_v2_models`): each target's fixed v2 recipe on every game from
   2002 that kicked off strictly before the install instant and has a final score. No search.
5. **The three artifacts are saved**; WP with ``preprocessing=model``, as the live WP is saved, so
   serving reads one object.
6. **The blend is tuned** (:func:`tune_v2_blend`): the v2 walk-forward, the extended corpus and
   the converter's extended slopes, all at the install instant and taken as ``backtest/study_1a.py``
   takes them; the weights are then fitted by ``refit.remeasure.tune_generation_blend`` (the live
   re-fit's ``backtest.tune.run_blend_tuning``, which writes the blend artifact), not by the
   ``MarketBlender().tune_weights`` call Study 1a's ``_decision_tests`` makes.
7. **The tuned weights must equal Study 1a's earned weights** (``V2_EARNED_WEIGHTS``, 0 / 0 / 0).
   Any other weight is a new finding for the owner to rule on, never something to install.
8. **The bet-rule record** (:func:`v2_bet_rule_payload`): no EV floor and no edge band, because a
   model with weight 0 has no edge to band.
9. **The intent** (``install_intent``) is appended durably to ``ledger/refit/owner_installs.jsonl``;
   a declined write refuses the swap (a served manifest no record explains would be unexplainable).
10. **The swap**: ``models.artifacts.replace_manifest`` writes the four pointers in one write after
    validating the whole bundle (the converter is the blend's bound one).
11. **The installed event.**

A REFUSAL PUTS THE V1 GOLD BACK
-------------------------------
Any exception from the v2 gold build until ``latest.json`` names the v2 ids -- a refusal, a crash,
an interrupt -- rebuilds the v1 gold (``FeatureMatrixBuilder``) and is re-raised, so the v1 models
in force never score v2 gold. ``latest.json`` is unchanged: it is only written by
``replace_manifest``. Artifact directories a stopped attempt saved stay on disk unserved; nothing
here deletes anything.

A FAILED SWAP IS RECORDED AS REFUSED
------------------------------------
Every failure after the intent is written and before ``latest.json`` names the v2 ids appends a
``refused`` event with the intent's ``install_date`` (the class name only, never exception text),
so the never-served intent never counts as a generation (``refit.generations``). ``install_date``
is computed once per run, the ET calendar date of the install instant.

A FAILURE AFTER THE WRITE IS AN INSTALL, NOT A REFUSAL
------------------------------------------------------
``replace_manifest`` writes ``latest.json`` before it returns, so an exception or an interrupt can
arrive after the write landed. Whenever ``latest.json`` names every v2 pointer
(:func:`_manifest_serves`), the v2 models are serving: the ``installed`` event is appended (never
``refused``, which would hide a served generation and make every later ``in_force_recipe`` raise),
and the v1 gold is NOT rebuilt (the v2 gold on disk is the gold the served models read). An
exception is logged and the install returns its outcome; an interrupt (a ``BaseException`` that
is not an ``Exception``) is re-raised after the event is appended.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import utils.game_lock as lock_rule
from backtest.midseason_switch_2026 import (
    DAY_BEFORE_RETIRED_FROM,
    INSTALLED_RECIPE,
    RECORD_PATH,
    V2_EARNED_WEIGHTS,
)
from backtest.refit_preregistration_2026 import MARKET_CONVERTER_ID, THREAD_LIMIT
from backtest.tune import BlendTuningRun
from forward_ledger.repro_key import gold_generation_key
from forward_ledger.settle import load_silver_games
from models.artifacts import (
    BUNDLE_MANIFEST_KEYS,
    ArtifactBundleInvalidError,
    replace_manifest,
    resolve_production_artifacts,
    save_model_artifact,
)
from models.market_probability import load_market_probability_artifact
from models.train import (
    GOLD_GENERATION_DIGEST_METADATA_KEY,
    THREAD_LIMIT_METADATA_KEY,
    pinned_thread_pool,
)
from models.v2_recipe import (
    V2_INPUTS,
    V2_LABEL,
    V2_TARGETS,
    V2_TRAIN_START,
    build_v2_estimator,
)
from models.v2_trainer import fit_v2, walk_forward_v2
from refit.generations import in_force_recipe
from refit.records import (
    append_owner_install,
    bet_rule_relpath,
    code_commit,
    write_bet_rule_record,
)
from refit.remeasure import (
    BET_RULE_RECORD_KIND,
    extended_corpus,
    extended_slopes,
    tune_generation_blend,
)
from refit.swap import _code_commit_problem
from refit.training import eligible_rows
from refit.witness import REPO_ROOT, _normalized_sha256
from scripts.build_features import FeatureMatrixBuilder
from scripts.build_gold_v2 import FeatureMatrixBuilderV2Live
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "SWITCH_RECORD_PATHS",
    "InstallOutcome",
    "InstallRefused",
    "gold_history_content_sha256",
    "install_preconditions",
    "run_owner_install",
    "switch_record_sha256",
    "train_v2_models",
    "tune_v2_blend",
    "v2_bet_rule_payload",
    "v2_feature_importances",
    "v2_model_metadata",
]

#: The two files of the switch record, repo-relative: the record every v2 artifact names.
SWITCH_RECORD_PATHS: tuple[str, str] = (
    RECORD_PATH,
    "backtest/midseason_switch_2026.py",
)

#: What sanctions the install, on its intent.
_RULING: str = f"{RECORD_PATH} section 4"

#: How the bet-rule record says it was installed (a cadence re-fit's record names its date).
_INSTALL_PATH: str = "owner"

# The five ids an install names: the four manifest pointers and the blend's bound converter.
_ID_KEYS: tuple[str, ...] = ("wp", "ats", "ou", "blend", "converter")

# The ``check`` a ``refused`` event names: the bundle failed validation, or the swap failed.
_CHECK_BUNDLE_VALIDATION: str = "bundle_validation"
_CHECK_SWAP: str = "swap"

#: The command that rebuilds gold for the recipe in force (v1 until the swap writes).
_GOLD_RECOVERY_COMMAND: str = "uv run python -m scripts.rebuild_gold"


class InstallRefused(RuntimeError):
    """The owner install stopped before ``latest.json`` was written; the reason is the message."""


@dataclass(frozen=True)
class InstallOutcome:
    """What a successful install put in force.

    Attributes:
        ids: wp, ats, ou, blend and converter of the v2 generation.
        bet_rule_sha256: The sha256 naming its bet-rule record.
        blend_weights: The tuned blend weights (the earned 0 / 0 / 0).
    """

    ids: dict[str, str]
    bet_rule_sha256: str
    blend_weights: dict[str, float]


def _as_utc(instant: datetime) -> datetime:
    """*instant* as aware UTC; a naive instant is refused, never guessed."""
    if instant.tzinfo is None or instant.utcoffset() is None:
        msg = f"the install needs an aware instant, got the naive {instant!r}"
        raise ValueError(msg)
    return instant.astimezone(UTC)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def install_preconditions(
    games: pd.DataFrame, now_utc: datetime, in_force: str
) -> None:
    """Refuse the install unless it is the first v2 install and week 9 has fully locked.

    Args:
        games: Silver ``games`` (``game_id``, ``season``, ``week``, tz-aware ``kickoff_et``).
        now_utc: The install instant, tz-aware.
        in_force: The recipe of the generation ``latest.json`` names.

    Raises:
        InstallRefused: the v2 recipe is already in force; the schedule names no 2026 game before
            the switch week; or such a game's lock is after *now_utc*.
        ValueError: *now_utc* is naive.
    """
    now = _as_utc(now_utc)
    if in_force == INSTALLED_RECIPE:
        msg = (
            f"the {INSTALLED_RECIPE} recipe is already in force; the v2 models are installed "
            f"once ({_RULING}), so a second install refuses"
        )
        raise InstallRefused(msg)

    season, first_retired_week = DAY_BEFORE_RETIRED_FROM
    before_switch = games.loc[
        (games["season"].astype(int) == season)
        & (games["week"].astype(int) < first_retired_week)
    ]
    if before_switch.empty:
        msg = (
            f"the schedule names no {season} game of weeks 1-{first_retired_week - 1}, so the "
            f"install cannot know week {first_retired_week - 1} has locked"
        )
        raise InstallRefused(msg)
    locks = lock_rule.lock_frame(before_switch)
    pending = cast("pd.Series", locks.loc[locks > pd.Timestamp(now)]).sort_values()
    if not pending.empty:
        last_game = str(pending.index[-1])
        last_lock = str(pending.iloc[-1])
        msg = (
            f"{len(pending)} game(s) of {season} weeks 1-{first_retired_week - 1} still lock "
            f"after {now.isoformat()}, the last {last_game} at {last_lock}; the install may "
            f"run only after week {first_retired_week - 1}'s last lock, so no week-"
            f"{first_retired_week - 1} decision is stamped with a v2 model"
        )
        raise InstallRefused(msg)


# ---------------------------------------------------------------------------
# Training, metadata, the blend and the bet rule
# ---------------------------------------------------------------------------


def train_v2_models(
    gold_dir: Path, games: pd.DataFrame, cutoff_utc: datetime
) -> dict[str, tuple[Pipeline, int]]:
    """Fit each target's fixed v2 recipe on its eligible rows from ``V2_TRAIN_START``.

    The rows are ``refit.training.eligible_rows`` at *cutoff_utc* (every game that kicked off
    strictly before it and has a final score, labelled from silver), exactly as one fold of
    ``models.v2_trainer.walk_forward_v2`` trains. Writes nothing.

    Returns:
        ``{target: (fitted pipeline, training row count)}``.

    Raises:
        refit.training.RefitTrainingError: the rows break the training scope (a partial gold).
    """
    cutoff = _as_utc(cutoff_utc)
    trained: dict[str, tuple[Pipeline, int]] = {}
    with pinned_thread_pool(THREAD_LIMIT):
        for target in V2_TARGETS:
            gold = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
            eligible = eligible_rows(gold, games, cutoff, target).frame
            train = eligible.loc[eligible["season"].astype(int) >= V2_TRAIN_START]
            trained[target] = (fit_v2(target, train), len(train))
    return trained


def switch_record_sha256(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """``{path: sha256}`` of the switch record's two files, CRLF folded to LF (``refit.witness``)."""
    return {
        path: _normalized_sha256(Path(repo_root) / path) for path in SWITCH_RECORD_PATHS
    }


def _estimator_text(pipeline: Pipeline) -> str:
    """The recipe's steps in order, as their ``repr``: what the artifact was fitted with."""
    return " | ".join(f"{name}={step!r}" for name, step in pipeline.steps)


def _best_params(target: str, pipeline: Pipeline) -> dict[str, float]:
    """The final estimator's one setting, read from the recipe (WP ``C``, ATS / O/U ``alpha``)."""
    final = pipeline.named_steps["model"]
    if target == "wp":
        return {"C": float(final.C)}
    return {"alpha": float(final.alpha)}


#: The last season of the gold history the served v2 models were fitted on, whatever the season
#: in progress appends nightly: the history key below covers seasons up to this one.
_GOLD_HISTORY_LAST_SEASON: int = 2025

#: The build clock column, dropped from the history key: every rebuild moves it.
_BUILD_CLOCK_COLUMN: str = "feature_timestamp"


def gold_history_content_sha256(gold_dir: Path) -> str:
    """sha256 of the 2002-2025 rows of the three gold matrices, by content.

    The v2 models are fitted on these completed seasons AND on the 2026 games played before the
    install instant; the key deliberately covers only seasons up to 2025, because the nightly build
    recomputes the season in progress, so a 2026 append or a 2026 recomputation cannot move it. A
    change therefore means a feature value of a completed season changed under the served models;
    the 2026 rows they were also fitted on are not covered by this key. The recipe
    is the one ``tests/unit/test_final_fit_entry_point.py`` restates beside
    ``ROW19_SERVED_MODELS_GOLD_HISTORY_CONTENT_SHA256``: per matrix (wp, ats, ou), its production
    path ``data/gold/features_<t>.parquet``, then the season <= 2025 rows sorted by ``game_id``
    with the build clock dropped -- ``game_id`` joined by newlines, then each other column in
    name order as its name and its values cast to float64.
    """
    digest = hashlib.sha256()
    for target in V2_TARGETS:
        frame = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
        history = (
            frame.loc[frame["season"] <= _GOLD_HISTORY_LAST_SEASON]
            .drop(columns=[_BUILD_CLOCK_COLUMN])
            .sort_values("game_id")
            .reset_index(drop=True)
        )
        digest.update(f"data/gold/features_{target}.parquet\n".encode("ascii"))
        digest.update("\n".join(history["game_id"]).encode("ascii"))
        for column in sorted(c for c in history.columns if c != "game_id"):
            digest.update(f"\n{column}\n".encode("ascii"))
            digest.update(history[column].astype("float64").to_numpy().tobytes())
    return digest.hexdigest()


def v2_feature_importances(target: str, model: Pipeline) -> dict[str, float]:
    """``{input: |coefficient|}`` of a fitted v2 pipeline, keyed by ``V2_INPUTS[target]``.

    The inputs are standardised before the linear model, so the absolute coefficients are
    comparable across inputs. WP's logistic regression has one row of coefficients (the home-win
    class); ridge has a flat vector. This is the ``feature_importances`` mapping
    ``api.cache._load_feature_importances`` serves to the model-insights page.

    Raises:
        ValueError: the fitted model has a different number of coefficients than the recipe has
            inputs (the imputer dropped an all-missing input), so no name can be trusted.
    """
    inputs = V2_INPUTS[target]
    coefficients = np.asarray(model.named_steps["model"].coef_, dtype=float)
    if coefficients.ndim == 2:
        coefficients = coefficients[0]
    if coefficients.shape != (len(inputs),):
        msg = (
            f"the fitted {target} model has {coefficients.size} coefficients for the "
            f"{len(inputs)} recipe inputs {list(inputs)}; an all-missing input was dropped"
        )
        raise ValueError(msg)
    return {
        name: abs(float(value))
        for name, value in zip(inputs, coefficients, strict=True)
    }


def v2_model_metadata(
    target: str,
    *,
    cutoff_utc: datetime,
    training_rows: int,
    gold_generation_digest: str,
    gold_history_digest: str,
    commit: str,
    feature_importances: Mapping[str, float],
) -> dict[str, object]:
    """The ``metadata.json`` of one v2 artifact.

    It carries what ``models.artifacts.replace_manifest`` validates (``target``, the gold digest),
    what ``backtest.tune.read_source_recipes`` reads (``best_params``, ``exclude_groups``) and what
    ``api.cache._load_feature_importances`` serves (``feature_importances``, from
    :func:`v2_feature_importances`), and the 2002-2025 gold history it was fitted on
    (``gold_history_content_sha256``, :func:`gold_history_content_sha256`). It names the recipe
    under ``v2_recipe`` -- never under a ledger stamp column's name.

    Raises:
        ValueError: *feature_importances* is not keyed by exactly the target's recipe inputs.
    """
    if set(feature_importances) != set(V2_INPUTS[target]):
        msg = (
            f"the {target} feature importances name {sorted(feature_importances)}, not the "
            f"recipe inputs {list(V2_INPUTS[target])}"
        )
        raise ValueError(msg)
    estimator = build_v2_estimator(target)
    return {
        "target": target,
        "feature_names": list(V2_INPUTS[target]),
        "feature_importances": {
            name: float(feature_importances[name]) for name in V2_INPUTS[target]
        },
        "v2_recipe": INSTALLED_RECIPE,
        "cutoff_utc": _as_utc(cutoff_utc).isoformat(),
        "training_rows": int(training_rows),
        "train_start": V2_TRAIN_START,
        "label": V2_LABEL[target],
        "estimator": _estimator_text(estimator),
        "best_params": _best_params(target, estimator),
        "exclude_groups": [],
        THREAD_LIMIT_METADATA_KEY: THREAD_LIMIT,
        GOLD_GENERATION_DIGEST_METADATA_KEY: gold_generation_digest,
        "gold_history_content_sha256": gold_history_digest,
        "code_commit": commit,
        "switch_record_sha256": switch_record_sha256(),
    }


def tune_v2_blend(
    *,
    artifacts_dir: Path,
    silver_dir: Path,
    gold_dir: Path,
    games: pd.DataFrame,
    cutoff_utc: datetime,
    model_ids: Mapping[str, str],
    gold_generation_digest: str,
) -> BlendTuningRun:
    """Tune the v2 generation's blend: the v2 walk-forward, then the per-generation blend tune.

    The walk-forward, the extended corpus and the converter's extended slopes are taken as
    ``backtest/study_1a.py`` takes them, at the install instant. The weights are fitted by
    ``refit.remeasure.tune_generation_blend`` (``backtest.tune.run_blend_tuning``, the live
    re-fit's blend tune), not by Study 1a's ``MarketBlender().tune_weights`` call in
    ``_decision_tests``; the blend is written into *artifacts_dir*, bound to
    ``MARKET_CONVERTER_ID``, its provenance naming *model_ids*.
    """
    predictions = walk_forward_v2(gold_dir, games, cutoff_utc)
    corpus = extended_corpus(silver_dir, games, cutoff_utc)
    slopes = extended_slopes(
        load_market_probability_artifact(MARKET_CONVERTER_ID, artifacts_dir)
    )
    return tune_generation_blend(
        artifacts_dir,
        silver_dir,
        gold_dir,
        artifact_ids=model_ids,
        predictions=predictions,
        corpus=corpus,
        slopes=slopes,
        gold_generation=gold_generation_digest,
    )


def v2_bet_rule_payload(
    *,
    artifact_ids: Mapping[str, str],
    cutoff_utc: datetime,
    blend_weights: Mapping[str, float],
) -> dict[str, object]:
    """The v2 generation's bet-rule record: no EV floor, no edge band, no residual SD.

    The ``record_kind`` is the one ``backtest.weekly_bet_list.load_bet_rule_record`` reads; each
    ``tune_fit`` block records an explicit null floor (a recorded "no honest floor", never a
    missing key), and every edge-tier threshold is null, so the website shows no band.

    Raises:
        ValueError: *artifact_ids* lacks one of the five ids, or *blend_weights* a target.
    """
    missing = [key for key in _ID_KEYS if not artifact_ids.get(key)]
    if missing:
        msg = f"the v2 bet-rule record needs artifact ids for {missing}"
        raise ValueError(msg)
    missing_weights = [target for target in V2_TARGETS if target not in blend_weights]
    if missing_weights:
        msg = f"the v2 bet-rule record needs blend weights for {missing_weights}"
        raise ValueError(msg)

    tune_fit = {
        target: {
            "ev_floor_t": None,
            "frozen_sd": None,
            "lean": None,
            "season_bias_by_season": {},
            "sd_correction_bias_by_season": {},
            "calibration_gate_passed": None,
        }
        for target in V2_TARGETS
    }
    return {
        "record_kind": BET_RULE_RECORD_KIND,
        "refit_recipe": INSTALLED_RECIPE,
        "install_path": _INSTALL_PATH,
        "artifact_ids": {key: str(artifact_ids[key]) for key in _ID_KEYS},
        "cutoff_utc": _as_utc(cutoff_utc).isoformat(),
        "blend_weights": {
            target: float(blend_weights[target]) for target in V2_TARGETS
        },
        "edge_tier_thresholds": dict.fromkeys(V2_TARGETS),
        "tune_fit": tune_fit,
        "targets": {target: {"fallback_trigger": None} for target in V2_TARGETS},
    }


# ---------------------------------------------------------------------------
# The install
# ---------------------------------------------------------------------------


def _tuned_weights(run: BlendTuningRun) -> dict[str, float]:
    weights = run.result.weights
    return {
        target: float(getattr(weights, f"{target}_model_weight"))
        for target in V2_TARGETS
    }


def _save_v2_models(
    trained: Mapping[str, tuple[Pipeline, int]],
    *,
    artifacts_dir: Path,
    cutoff_utc: datetime,
    gold_generation_digest: str,
    gold_history_digest: str,
    commit: str,
) -> dict[str, str]:
    """Save the three models (never touching ``latest.json``); return their artifact ids."""
    ids: dict[str, str] = {}
    for target in V2_TARGETS:
        model, rows = trained[target]
        metadata = v2_model_metadata(
            target,
            cutoff_utc=cutoff_utc,
            training_rows=rows,
            gold_generation_digest=gold_generation_digest,
            gold_history_digest=gold_history_digest,
            commit=commit,
            feature_importances=v2_feature_importances(target, model),
        )
        directory = save_model_artifact(
            model,
            target,
            metadata,
            list(V2_INPUTS[target]),
            # The live WP is saved as one object under both names (D33.1-R1).
            preprocessing=model if target == "wp" else None,
            artifacts_dir=artifacts_dir,
        )
        ids[target] = directory.name
    return ids


def _refused_check(failure: BaseException) -> str:
    """The check a failed swap names: the bundle's validation, or the swap itself."""
    if isinstance(failure, ArtifactBundleInvalidError):
        return _CHECK_BUNDLE_VALIDATION
    return _CHECK_SWAP


def _append_refused(
    ledger_dir: Path, install_date: str, failure: BaseException
) -> bool:
    """Record that the swap did not write; False (logged) when the record itself failed."""
    event = {
        "event": "refused",
        "install_date": install_date,
        "check": _refused_check(failure),
        "exception_class": type(failure).__name__,
    }
    try:
        written = append_owner_install(ledger_dir, event)
    # Broad on purpose, and not a swallow: the record's failure is logged by class, and the
    # swap's own failure -- the one the owner must see -- is re-raised by the caller.
    except Exception as record_error:  # noqa: BLE001 - logged, then the swap failure re-raised
        logger.error(
            "The refused event of an owner install was not written",
            exception_class=type(record_error).__name__,
        )
        return False
    return written


def _append_installed(
    ledger_dir: Path, install_date: str, ids: Mapping[str, str]
) -> None:
    """Record that the swap wrote; a declined write is logged (the intent names the served ids)."""
    installed = {
        "event": "installed",
        "install_date": install_date,
        "after_ids": dict(ids),
    }
    if not append_owner_install(ledger_dir, installed):
        logger.warning(
            "The installed event was not written; the intent already names the served ids"
        )


def _manifest_serves(artifacts_dir: Path, pointers: Mapping[str, str]) -> bool:
    """True only when ``latest.json`` names every one of *pointers* with exactly its id.

    Only the swap writes the v2 ids, so a manifest naming them proves the swap landed, whatever
    failed after its write. Never raises: it is asked inside exception handlers, where a new
    exception would replace the failure the owner must see. An absent, unreadable or unparseable
    manifest is False, and so are empty *pointers* (nothing intended is nothing served).
    """
    if not pointers:
        return False
    try:
        manifest = json.loads((Path(artifacts_dir) / "latest.json").read_text())
    # Broad on purpose: whatever stops the read, the manifest proves no swap (the contract above).
    except Exception:  # noqa: BLE001 - an unreadable manifest is False, never an error
        return False
    if not isinstance(manifest, dict):
        return False
    return all(manifest.get(key) == value for key, value in pointers.items())


def _swap_in_v2(
    *,
    artifacts_dir: Path,
    ledger_dir: Path,
    gold_dir: Path,
    silver_dir: Path,
    games: pd.DataFrame,
    now: datetime,
    install_date: str,
    commit: str,
    intended_pointers: dict[str, str],
) -> InstallOutcome:
    """Steps 4-11 on the v2 gold: train, save, tune, refuse or record, intent, swap, installed.

    *intended_pointers* is filled with the four manifest pointers just before the swap, so the
    caller can tell a swap that landed from one that never wrote.
    """
    digest = gold_generation_key(gold_dir)
    history_digest = gold_history_content_sha256(gold_dir)
    trained = train_v2_models(gold_dir, games, now)
    model_ids = _save_v2_models(
        trained,
        artifacts_dir=artifacts_dir,
        cutoff_utc=now,
        gold_generation_digest=digest,
        gold_history_digest=history_digest,
        commit=commit,
    )
    run = tune_v2_blend(
        artifacts_dir=artifacts_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        games=games,
        cutoff_utc=now,
        model_ids=model_ids,
        gold_generation_digest=digest,
    )
    weights = _tuned_weights(run)
    earned = {target: float(V2_EARNED_WEIGHTS[target]) for target in V2_TARGETS}
    if weights != earned:
        msg = (
            f"the tuned blend weights {weights} are not Study 1a's earned weights {earned} "
            f"({_RULING}): a nonzero weight is a new finding for the owner to rule on, never "
            f"installed silently. Nothing was recorded; the tuned blend {run.artifact_dir.name} "
            "and the three models stay on disk unserved"
        )
        raise InstallRefused(msg)

    ids = {
        **model_ids,
        "blend": run.artifact_dir.name,
        "converter": run.converter_artifact_id,
    }
    payload = v2_bet_rule_payload(
        artifact_ids=ids, cutoff_utc=now, blend_weights=weights
    )
    _path, sha = write_bet_rule_record(ledger_dir, payload)

    before = dataclasses.asdict(resolve_production_artifacts(artifacts_dir))
    intent: dict[str, Any] = {
        "event": "install_intent",
        "install_date": install_date,
        "cutoff_utc": now.isoformat(),
        "ruling": _RULING,
        "recipe": INSTALLED_RECIPE,
        "ids": ids,
        "bet_rule_path": bet_rule_relpath(sha),
        "bet_rule_sha256": sha,
        "code_commit": commit,
        "gold_generation_digest": digest,
        "before_ids": before,
    }
    if not append_owner_install(ledger_dir, intent):
        msg = (
            "the install intent was not written (the write sink declined it), so latest.json "
            "was not touched"
        )
        raise InstallRefused(msg)

    pointers = {key: ids[key] for key in BUNDLE_MANIFEST_KEYS}
    intended_pointers.update(pointers)
    outcome = InstallOutcome(ids=ids, bet_rule_sha256=sha, blend_weights=weights)
    try:
        replace_manifest(pointers, artifacts_dir)
    except BaseException as failure:
        if _manifest_serves(artifacts_dir, pointers):
            # The write landed and something after it failed: the v2 models serve, so this is
            # an install, never a refusal (a refused event would hide a served generation).
            _append_installed(ledger_dir, install_date, ids)
            if not isinstance(failure, Exception):
                raise
            logger.error(
                "The swap wrote latest.json and then failed; the install stands",
                exception_class=type(failure).__name__,
                error=str(failure),
            )
            return outcome
        recorded = _append_refused(ledger_dir, install_date, failure)
        if not isinstance(failure, Exception):
            raise
        unrecorded = (
            ""
            if recorded
            else (
                " The refused event was NOT written, so the intent will count as a "
                "generation that never served; tell the owner."
            )
        )
        msg = (
            f"{_refused_check(failure)}: latest.json was not written: "
            f"{type(failure).__name__}: {failure}.{unrecorded}"
        )
        raise InstallRefused(msg) from failure
    _append_installed(ledger_dir, install_date, ids)
    return outcome


def _restore_v1_gold(
    build_gold: Callable[[type], None], failure: BaseException
) -> None:
    """Rebuild the v1 gold after a stopped install, naming the recovery if that fails too."""
    logger.error(
        "The owner install stopped; rebuilding the v1 gold",
        exception_class=type(failure).__name__,
    )
    try:
        build_gold(FeatureMatrixBuilder)
    # Broad on purpose: whatever stopped the rebuild, the owner must be told the gold is not
    # v1 and how to rebuild it; both failures are named in the refusal, never swallowed.
    except Exception as restore_error:  # noqa: BLE001 - named in the refusal raised below
        msg = (
            f"the install stopped ({type(failure).__name__}: {failure}) and the v1 gold rebuild "
            f"then failed ({type(restore_error).__name__}: {restore_error}). latest.json still "
            f"serves the v1 models, so rebuild their gold with `{_GOLD_RECOVERY_COMMAND}` "
            "before the next daily run"
        )
        raise InstallRefused(msg) from failure


def run_owner_install(
    *,
    artifacts_dir: Path,
    ledger_dir: Path,
    gold_dir: Path,
    silver_dir: Path,
    now_utc: datetime,
    build_gold: Callable[[type], None],
) -> InstallOutcome:
    """Install the v2 models once, recorded, or refuse with ``latest.json`` and gold back on v1.

    The order is the module docstring's. Nothing before the v2 gold build writes anything.

    Args:
        artifacts_dir: The artifacts root: ``latest.json``, the converter, the new artifacts.
        ledger_dir: The forward ledger directory holding ``refit/``.
        gold_dir: Where *build_gold* writes the three gold matrices.
        silver_dir: The silver lake (``games``, the odds tables the blend corpus reads).
        now_utc: The install instant, tz-aware: the training cutoff, and the ET calendar day of
            every event this run records.
        build_gold: Rebuilds and saves the shared gold with the builder class it is given.

    Returns:
        The :class:`InstallOutcome` now in force.

    Raises:
        InstallRefused: a precondition, the code commit, the tuned weights, the intent write or
            the swap refused; ``latest.json`` was not written.
        ValueError: *now_utc* is naive.
        BaseException: an interrupt after ``latest.json`` named the v2 ids propagates, with the
            ``installed`` event appended and the v2 gold kept (module docstring).
    """
    now = _as_utc(now_utc)
    install_date = now.astimezone(lock_rule.ET).date().isoformat()
    artifacts = Path(artifacts_dir)
    ledger = Path(ledger_dir)

    games = load_silver_games(silver_dir)
    install_preconditions(games, now, in_force_recipe(artifacts, ledger))
    commit = code_commit(REPO_ROOT)
    problem = _code_commit_problem(commit)
    if problem is not None or commit is None:
        msg = f"clean_code_commit: {problem}"
        raise InstallRefused(msg)

    intended_pointers: dict[str, str] = {}
    try:
        build_gold(FeatureMatrixBuilderV2Live)
        outcome = _swap_in_v2(
            artifacts_dir=artifacts,
            ledger_dir=ledger,
            gold_dir=Path(gold_dir),
            silver_dir=Path(silver_dir),
            games=games,
            now=now,
            install_date=install_date,
            commit=commit,
            intended_pointers=intended_pointers,
        )
    except BaseException as failure:
        if _manifest_serves(artifacts, intended_pointers):
            # The v2 models serve and the v2 gold is on disk: a v1 rebuild would feed them the
            # wrong gold, so nothing is restored.
            logger.error(
                "The owner install stopped after latest.json named the v2 ids; the v2 gold stays",
                exception_class=type(failure).__name__,
            )
            raise
        _restore_v1_gold(build_gold, failure)
        raise

    logger.info("Installed the v2 models", ids=outcome.ids, install_date=install_date)
    return outcome
