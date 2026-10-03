"""Unified training entry point for WP, ATS, and O/U models.

Single command to train all models:
    python -m models.train --target all

Individual targets:
    python -m models.train --target wp
    python -m models.train --target ats
    python -m models.train --target ou

This module addresses MODL-08 (baseline comparison against market closing line)
and integrates all prior work (Plans 01-03) into a single runnable pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

# D30-01/D30-02: the feature-group vocabulary and the selection primitive are IMPORTED from
# backtest.signal_lift, never re-declared here. This models -> backtest direction mirrors the
# import-the-primitive seam at models/deploy_gate.py:99 (D24-13): a second, locally-declared
# group list is exactly what silently broke the Phase-28 baseline in 29-06 (T-30-15), because
# the two copies can drift without anything failing.
#
# select_group_columns returns a FRESH in-memory copy and never mutates data/gold, which is why
# a Stage-2 exclusion needs no second gold-shaped artifact (D30-01's rejected alternative).
from backtest.signal_lift import ALL_REGISTERED_GROUPS, select_group_columns
from conf.season_partition import default_season_partition
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.base import threads_in_force
from models.trainers.final_fit import apply_final_fit_to_trainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)

# Valid target choices
_VALID_TARGETS = ("wp", "ats", "ou", "all")

# ---------------------------------------------------------------------------
# THE GOLD-GENERATION MARKER (Phase 33 Wave 15, D33-25 / R7).
#
# The metadata key a re-fit writes to DECLARE, explicitly, that its training distribution
# included the corrected historical weather record. It is DEFINED and TESTED by
# `tests/unit/test_weather_bridge_expiry.py`, which deliberately does NOT write it --
# writing it in Phase 33.1 would have been claiming a re-fit happened. This module writes
# it, and only when `--gold-generation` is supplied.
#
# THE VALUE COMES FROM THE COMMAND LINE, NOT FROM AN IMPORT, and the reason is worth
# recording: the ONE producer of the key is `tests.gold_generation.gold_generation_key`,
# and a production module importing from the tests package to reach it would be the wrong
# direction. The operator measures it once, records it in
# `tests.phase33_state.GOLD_GENERATION_AT_REFIT`, and passes it in; a test then asserts
# that record equals the ladder's own generation AND the live key, so the marker is
# provably the generation the gold rebuild produced rather than a string somebody typed.
#
# ABSENT, NEVER EMPTY. Omitting the flag leaves the key out of metadata entirely. An
# ordinary training run claims nothing about a weather generation, and an empty-string
# marker would be a claim that reads as a non-claim.
# ---------------------------------------------------------------------------
GOLD_GENERATION_METADATA_KEY = "trained_on_real_weather_generation"

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED RE-FIT (D33.2-17 / R13, Plan 33.2-23).
#
# `--tune` is an EXPLICIT opt-in to the two-arm search, and it is NOT the same thing as the
# default (which already tunes). Three things change under it and only under it, so every
# other caller -- scripts.retrain_models, backtest.engine, the Friday orchestrator -- is
# byte-identical:
#
#   * the trainer opts into `use_phase332_tuning()`: the pre-registered budget, a
#     RandomSampler baseline over the same objective and budget, and adoption only on a
#     margin cleared on a season neither arm saw;
#   * the verdict's own digest is recorded beside the exclusion even when the list was
#     typed (the exclusion itself is DERIVED from the verdict on EVERY path when
#     --exclude-groups is omitted -- see resolve_exclusion);
#   * `--gold-generation` becomes REQUIRED. A re-fit whose artifact cannot name the gold it
#     was trained on is a re-fit nobody can reproduce, and this module must not import the
#     tests package to measure it for itself (see GOLD_GENERATION_METADATA_KEY).
# ---------------------------------------------------------------------------

#: The ratified Stage-1 feature-group verdict, read verbatim -- the same file and the same
#: key `scripts/promote_models.py` reads, so what the rule decided and what gets trained
#: cannot diverge by a typo. Anchored to the REPOSITORY, exactly as promote_models anchors
#: it, so what gets trained never depends on the directory the operator stood in.
GROUP_GATE_VERDICT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "group_gate_verdict.toml"
)

#: Metadata keys the pre-registered re-fit writes into `metadata.json`. They live there and
#: NOT in the tuning sidecar because they describe the TRAINING RUN -- what data, which
#: verdict, which study identity, how many threads -- rather than the search.
GOLD_GENERATION_DIGEST_METADATA_KEY = "gold_generation_digest"
GROUP_VERDICT_DIGEST_METADATA_KEY = "group_verdict_digest"
TUNING_STUDY_TAG_METADATA_KEY = "tuning_study_tag"
THREAD_LIMIT_METADATA_KEY = "thread_limit"


def _normalized_sha256(path: Path) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED bytes.

    Normalized because this repository has ``core.autocrlf=true`` and no
    ``.gitattributes``, so a raw-byte digest of a tracked text file would hold only on the
    platform that measured it.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verdict_exclusion() -> tuple[tuple[str, ...], str]:
    """Return the ratified exclusion list and the verdict file's digest.

    DERIVED, never transcribed (D24-07, and the same precedence
    ``scripts/promote_models._resolve_exclude_groups`` applies). The digest travels with it
    so an artifact records WHICH verdict shaped its feature set, not merely that one did.

    Returns:
        ``(groups, digest)``.

    Raises:
        FileNotFoundError: If no ratified verdict exists. This is a STOP rather than an
            empty list: without the verdict every feature group is trained, including the
            ones the frozen rule DROPPED, and the run would silently reverse a ratified
            owner decision.
    """
    if not GROUP_GATE_VERDICT_PATH.exists():
        msg = (
            f"no ratified Stage-1 verdict at '{GROUP_GATE_VERDICT_PATH}'. Training "
            "refuses to run on EVERY feature group, including any the frozen rule "
            "dropped: that would silently reverse a ratified decision. Restore the "
            "verdict, or pass --exclude-groups explicitly (with a provenance other than "
            "'verdict') to state the exclusion deliberately."
        )
        raise FileNotFoundError(msg)
    with GROUP_GATE_VERDICT_PATH.open("rb") as handle:
        verdict = tomllib.load(handle)
    if "excluded_groups" not in verdict:
        msg = f"'{GROUP_GATE_VERDICT_PATH}' has no 'excluded_groups' key; refusing."
        raise KeyError(msg)
    groups = tuple(str(group) for group in verdict["excluded_groups"])
    return groups, _normalized_sha256(GROUP_GATE_VERDICT_PATH)


@contextmanager
def pinned_thread_pool(thread_limit: int | None) -> Iterator[None]:
    """Apply the thread pin THIS module records, and prove it is in force.

    A33.2-review WR-02: ``--thread-limit`` (and ``--tune``, which defaults it) used to be
    written into every artifact's metadata while the pin itself was applied only by
    ``scripts/train_models.py``. A run started as ``python -m models.train --tune`` -- the
    invocation this module's docstring names as canonical -- therefore recorded
    ``thread_limit: 1`` while XGBoost ran on every core.

    The pin is now applied here, at RUNTIME, through ``threadpoolctl``: every numerical
    library this module needs is already loaded by its own imports, so the runtime pin
    reaches every pool (the environment-variable half, which must precede the numpy
    import, stays in ``scripts/train_models.py``). After applying it the pools are MEASURED,
    and a pool that did not take the pin is a refusal rather than a false record.

    Args:
        thread_limit: The count to pin, or None to leave the pools exactly as they are.

    Raises:
        RuntimeError: If, after pinning, the loaded pools do not all report
            ``thread_limit``.
    """
    if thread_limit is None:
        yield
        return
    with threadpool_limits(limits=thread_limit):
        in_force = threads_in_force()
        if in_force != thread_limit:
            msg = (
                f"thread pin requested at {thread_limit} but the loaded pools report "
                f"{in_force} (None = no pool loaded, or the pools disagree). Refusing to "
                "record a pin that is not in force."
            )
            raise RuntimeError(msg)
        yield


def parse_exclude_groups(raw: str) -> tuple[str, ...]:
    """Parse the ``--exclude-groups`` scalar token into a tuple of group names.

    The comprehension's ``if g.strip()`` filter is LOAD-BEARING, not defensive tidying:
    ``"".split(",")`` returns ``[""]`` -- a one-element list holding the empty string -- which
    would reach ``group_columns`` as an unregistered name and raise. Under the naive form the
    DEFAULT invocation (the one that must reproduce today's behaviour byte-for-byte) would
    hard-fail. The same filter absorbs a trailing comma and any all-whitespace token.

    Args:
        raw: The raw comma-separated flag value (``""`` by default).

    Returns:
        The parsed group names, empty when nothing was requested.
    """
    return tuple(g.strip() for g in raw.split(",") if g.strip())


def resolve_exclusion(
    raw: str | None, provenance: str
) -> tuple[tuple[str, ...], str, str | None]:
    """Resolve the feature-group exclusion, its provenance and the verdict digest.

    The ONE resolution every training run goes through, tuned or not:

      * ``--exclude-groups`` omitted (``raw is None``): the owner's ratified verdict,
        provenance ``"verdict"``. A missing or unreadable verdict REFUSES (see
        :func:`verdict_exclusion`) rather than training on every group.
      * an explicit list with provenance ``"verdict"``: it must equal the verdict file's
        list, otherwise REFUSE -- a typed list may not claim the verdict's authority.
      * an explicit list with any other provenance: taken as typed (``""`` states
        "exclude nothing" deliberately), no digest.

    Args:
        raw: The raw ``--exclude-groups`` value, or None when the flag was not given.
        provenance: The ``--exclude-groups-provenance`` value.

    Returns:
        ``(groups, provenance, verdict_digest_or_None)``.

    Raises:
        FileNotFoundError: The verdict is needed and absent.
        ValueError: A typed list claims provenance ``"verdict"`` but differs from it.
    """
    if raw is None:
        groups, digest = verdict_exclusion()
        return groups, "verdict", digest
    groups = parse_exclude_groups(raw)
    if provenance != "verdict":
        return groups, provenance, None
    verdict_groups, digest = verdict_exclusion()
    if sorted(groups) != sorted(verdict_groups):
        msg = (
            f"--exclude-groups {list(groups)} is labelled provenance 'verdict' but the "
            f"ratified verdict at '{GROUP_GATE_VERDICT_PATH}' excludes "
            f"{list(verdict_groups)}. Refusing: omit --exclude-groups to use the verdict, "
            "or label a different list 'override'."
        )
        raise ValueError(msg)
    return groups, provenance, digest


def compute_market_baseline(
    games_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    """Compute market-only baseline metrics for comparison.

    Treats closing line implied probabilities as a "model" and evaluates
    with the same metrics as our trained models. This answers the question:
    "Is our model better than just following the market?"

    For WP: Devig closing moneylines to get fair home-win probability,
    then compute accuracy, Brier score, and log loss.

    For ATS: The market spread implies 50/50 probability by definition.
    Report how often the home team actually covers the closing spread.

    For O/U: The market total implies 50/50 probability by definition.
    Report how often the game actually goes over the closing total.

    Args:
        games_df: Games DataFrame with game_id, home_score, away_score.
        closing_odds_df: Closing odds with game_id, ml_home, ml_away,
            spread, total.
        target: One of "wp", "ats", "ou".

    Returns:
        Dict with keys: market_accuracy, market_brier (WP only),
        market_logloss (WP only), n_games, n_excluded.
    """
    # Merge PLAYED games with closing odds on game_id. Gold carries the live slate's
    # unplayed games, whose NaN scores would otherwise grade as away wins / no cover /
    # under (NaN > NaN is False) -- code review WR-04.
    played = games_df.dropna(subset=["home_score", "away_score"])
    merged = played.merge(
        closing_odds_df, on="game_id", how="inner", suffixes=("", "_odds")
    )

    if target == "wp":
        return _compute_wp_baseline(merged, len(games_df))
    if target == "ats":
        return _compute_ats_baseline(merged, len(games_df))
    if target == "ou":
        return _compute_ou_baseline(merged, len(games_df))
    msg = f"Unknown target: {target}. Must be one of: wp, ats, ou"
    raise ValueError(msg)


def _compute_wp_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute WP market baseline using devigged closing moneylines.

    Devigging removes the bookmaker's margin (vig/juice) so that
    implied probabilities sum to 1.0 instead of > 1.0.

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, market_brier, market_logloss, n_games, n_excluded.
    """
    # Exclude rows with missing moneylines
    valid = merged.dropna(subset=["ml_home", "ml_away"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.0,
            "market_brier": 1.0,
            "market_logloss": float("inf"),
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    # Convert moneylines to raw implied probabilities
    home_raw = valid["ml_home"].apply(lambda x: moneyline_to_probability(int(x)))
    away_raw = valid["ml_away"].apply(lambda x: moneyline_to_probability(int(x)))

    # Devig: proportional method (divide by overround)
    total_raw = home_raw + away_raw
    fair_prob_home = home_raw / total_raw

    # Actual outcomes
    actual_home_win = (valid["home_score"] > valid["away_score"]).astype(int)

    # Compute metrics
    market_accuracy = float(
        accuracy_score(actual_home_win, (fair_prob_home > 0.5).astype(int))
    )
    market_brier = float(brier_score_loss(actual_home_win, fair_prob_home))
    market_logloss = float(
        log_loss(actual_home_win, np.clip(fair_prob_home.values, 0.01, 0.99))
    )

    return {
        "market_accuracy": market_accuracy,
        "market_brier": market_brier,
        "market_logloss": market_logloss,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def _compute_ats_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute ATS market baseline.

    The market spread is set to equalize action, meaning the implied
    probability of either side covering is ~50%. The baseline reports
    how often the home team actually covers the closing spread.

    Home covers when: (home_score - away_score) + spread > 0
    (where spread is negative for home favorite).

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, n_games, n_excluded.
    """
    valid = merged.dropna(subset=["spread"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.5,
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    actual_margin = valid["home_score"] - valid["away_score"]
    # Home covers if actual margin + spread > 0
    # Spread convention: negative means home is favored
    actual_cover = ((actual_margin + valid["spread"]) > 0).astype(int)

    # Market baseline accuracy: how often does the closing spread
    # correctly predict the cover side? Since spread implies 50/50,
    # the baseline is just the empirical cover rate.
    market_accuracy = float(actual_cover.mean())

    return {
        "market_accuracy": market_accuracy,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def _compute_ou_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute O/U market baseline.

    The market total is set to equalize action, meaning the implied
    probability of over/under is ~50%. The baseline reports how often
    the game actually goes over the closing total.

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, n_games, n_excluded.
    """
    valid = merged.dropna(subset=["total"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.5,
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    actual_total = valid["home_score"] + valid["away_score"]
    actual_over = (actual_total > valid["total"]).astype(int)

    # Market baseline accuracy: how often does the game go over?
    # Since the market implies 50/50, this is the empirical over rate.
    market_accuracy = float(actual_over.mean())

    return {
        "market_accuracy": market_accuracy,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def train_target(
    target: str,
    features_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame | None = None,
    config: TemporalSplitConfig | None = None,
    artifacts_dir: Path = Path("artifacts"),
    tune: bool = True,
    exclude_groups: tuple[str, ...] = (),
    exclude_groups_provenance: str = "none",
    gold_generation: str | None = None,
    preregistered_search: bool = False,
    group_verdict_digest: str | None = None,
    thread_limit: int | None = None,
) -> dict[str, Any]:
    """Train a single model target and optionally compute market baseline.

    Instantiates the appropriate trainer, runs the walk-forward evaluation, THEN runs the
    explicit FINAL FIT over every completed season in the committed partition rule, saves
    artifacts, and computes the market baseline if odds are available.

    Args:
        target: One of "wp", "ats", "ou".
        features_df: Feature matrix with ID columns, features, and target column.
        closing_odds_df: Optional closing odds for CLV and market baseline.
        config: Temporal split configuration. Defaults to default split.
        artifacts_dir: Root directory for saving model artifacts.
        tune: When True (default), tune hyperparameters via Optuna. When False,
            perform a straight re-fit with default params and no Optuna sweep
            (D24-12), threaded down to trainer.train_and_evaluate(tune=False).
        exclude_groups: The feature groups ALREADY removed from ``features_df`` by the
            caller. Recorded in the artifact's metadata, not applied here.
        exclude_groups_provenance: Where that list came from -- ``"verdict"`` (the ratified
            Stage-1 verdict), ``"override"`` (hand-typed on the command line) or ``"none"``.
        gold_generation: The gold generation key this run's features were read from, or
            None. When supplied it is written into the artifact's metadata under
            :data:`GOLD_GENERATION_METADATA_KEY`; when omitted the key is ABSENT.

    Returns:
        Dict with keys: model_metrics, market_baseline, artifact_path.

    Note:
        WR-04: the exclusion is applied by ``main()`` in memory between the parquet read and
        this call, and it used to be LOGGED and then dropped -- ``BaseTrainer.metadata``
        recorded the target, params, season results and config and nothing else. CLAUDE.md
        requires that "any prediction must be reproducible given the same input data
        snapshot", and re-running from an artifact's own metadata reproduced a DIFFERENT
        feature set, because the exclusion was recoverable only from the git-tracked verdict
        file plus knowledge of which commit was current. The phase went to some trouble to
        make the exclusion DERIVED rather than transcribed; recording the derived value in the
        artifact is what makes that benefit reach a later auditor.
    """
    # Instantiate the appropriate trainer
    trainers = {
        "wp": WPTrainer,
        "ats": ATSTrainer,
        "ou": OUTrainer,
    }

    if target not in trainers:
        msg = f"Unknown target: {target}. Must be one of: {list(trainers.keys())}"
        raise ValueError(msg)

    trainer_class = trainers[target]
    trainer = trainer_class(config=config)

    if tune and preregistered_search:
        # D33.2-17 / R13: the PRE-REGISTERED two-arm search. It is the Stage-2 identity
        # (so everything the comment below says still holds) PLUS the pre-registered
        # budget, the RandomSampler baseline and the outer-season adoption gate.
        trainer.use_phase332_tuning()
    elif tune:
        # SPEC R5 / T-30-02: this entry point IS the Stage-2 candidate train that
        # scripts/promote_models STEP 1 invokes, so a tuned run here must genuinely search --
        # a fresh per-phase study identity, storage outside data/, and a hard failure if zero
        # new trials ran. The opt-in is explicit and scoped to this call site on purpose:
        # backtest.engine.run_backtest also trains with tune=True, and giving IT a fresh study
        # changes the parameters it lands on and drifts the frozen v2.1 AUDIT-REPORT anchors
        # (verified empirically during Plan 30-01). Those callers keep the legacy identity.
        trainer.use_phase30_tuning()

    logger.info("Starting training", target=target)

    # Train and evaluate
    model_metrics = trainer.train_and_evaluate(features_df, closing_odds_df, tune=tune)

    # WR-04: record WHAT was withheld from the frame, and whether that list was ratified or
    # hand-typed, in the artifact itself. Written after train_and_evaluate (which assigns
    # self.metadata wholesale) and before save, so it lands in the saved metadata.json.
    trainer.metadata["exclude_groups"] = list(exclude_groups)
    trainer.metadata["exclude_groups_provenance"] = exclude_groups_provenance

    # ------------------------------------------------------------------
    # THE FINAL FIT (D33.1-01 / D33.1-02, Phase 33 Wave 15). THE ORDER IS LOAD-BEARING.
    #
    # WHY IT IS HERE AT ALL. `WalkForwardSplitter.generate_splits` builds every fold as
    # `season < holdout_season` and each concrete trainer keeps the LAST fold's model, so
    # without this call the shipped artifact stops one holdout season short of the corpus
    # WHATEVER the partition says -- and no choice of the three `--config-*-seasons`
    # lists can change that, because widening `train_seasons` does not touch the fold mask.
    #
    # WHY THIS ORDER AND NOT ANOTHER. `apply_final_fit_to_trainer` assigns
    # `trainer.model`, `trainer.preprocessing` and the two final-fit metadata keys, and
    # `BaseTrainer.save` reads exactly those three. Producing a `FinalFitResult` and NOT
    # applying it would leave the save path persisting the LAST FOLD's object while the
    # record described the final fit -- threat T-33.1-65d, named in that module's own
    # docstring.
    #
    # WHY THE PARTITION COMES FROM THE RULE AND NOT FROM THE FLAGS. The three
    # `--config-*-seasons` arguments describe the WALK-FORWARD FOLDS.
    # `partition.final_fit` is a FOURTH set that deliberately overlaps them, so
    # constructing a `SeasonPartition` out of the flags would silently fit the shipped
    # model on the selection window. Reading `conf.season_partition` directly is what
    # keeps one rule, read once.
    #
    # THE RECORDED COST, stated rather than hidden: the calibration component is carried
    # across BY REFERENCE and is never refitted, so the shipped model's calibrator was
    # fitted against a NARROWER model than the one that ships. That biases mildly toward
    # UNDER-confidence -- a conservative and statable error.
    # ------------------------------------------------------------------
    partition = default_season_partition()
    final_fit_result = trainer.final_fit(features_df, partition)
    apply_final_fit_to_trainer(trainer, final_fit_result)

    # The weather-generation marker (D33-25 / R7). Written AFTER the final fit, because
    # `apply_final_fit_to_trainer` writes into the same metadata dict, and BEFORE the
    # save, because that is where it has to land to reach metadata.json. Absent when the
    # caller supplied nothing -- see GOLD_GENERATION_METADATA_KEY.
    if gold_generation:
        trainer.metadata[GOLD_GENERATION_METADATA_KEY] = gold_generation
        # D33.2-17 / R13: the SAME value under the name an auditor looks for. The key
        # above is the gold-weather bridge's flip predicate and means "this run DECLARES
        # it trained on the corrected weather generation"; this one is the plain
        # provenance fact -- which gold generation these features came from -- and every
        # re-fit artifact must carry it whether or not the bridge ever reads it.
        trainer.metadata[GOLD_GENERATION_DIGEST_METADATA_KEY] = gold_generation

    if group_verdict_digest:
        trainer.metadata[GROUP_VERDICT_DIGEST_METADATA_KEY] = group_verdict_digest
    if preregistered_search:
        trainer.metadata[TUNING_STUDY_TAG_METADATA_KEY] = trainer.tuning_study_tag
    if thread_limit is not None:
        # Recorded because Plan 33.2-22 MEASURED the XGBoost legs returning different
        # answers at different OpenMP thread counts, by enough to move a verdict. A number
        # produced under an unrecorded thread count is one nobody else can reproduce.
        trainer.metadata[THREAD_LIMIT_METADATA_KEY] = thread_limit

    # Save artifacts
    artifact_path = trainer.save(artifacts_dir)
    logger.info("Artifacts saved", target=target, path=str(artifact_path))

    # Compute market baseline if closing odds available
    market_baseline = None
    if closing_odds_df is not None:
        # Build a games_df from features for market baseline
        games_cols = ["game_id", "home_score", "away_score"]
        if all(col in features_df.columns for col in games_cols):
            games_df = features_df[games_cols].copy()
        elif "game_id" in features_df.columns:
            games_df = features_df[["game_id"]].copy()
            if "home_score" in features_df.columns:
                games_df["home_score"] = features_df["home_score"]
            if "away_score" in features_df.columns:
                games_df["away_score"] = features_df["away_score"]
        else:
            # Try using the index as game_id
            games_df = features_df.reset_index()
            if "game_id" not in games_df.columns:
                games_df = games_df.rename(columns={"index": "game_id"})

        if "home_score" in games_df.columns and "away_score" in games_df.columns:
            market_baseline = compute_market_baseline(
                games_df, closing_odds_df, target=target
            )
            logger.info(
                "Market baseline computed",
                target=target,
                baseline=market_baseline,
            )

    return {
        "model_metrics": model_metrics,
        "market_baseline": market_baseline,
        "artifact_path": str(artifact_path),
        # None on every path but the pre-registered one, where it is the per-target margin
        # verdict this plan must publish rather than absorb.
        "adoption_record": trainer.adoption_record,
    }


def _pooled_brier(holdout_predictions: pd.DataFrame | None) -> float | None:
    """Mean squared error of the pooled out-of-sample WP probabilities, or None.

    Args:
        holdout_predictions: The trainer's per-game out-of-sample record
            (``prediction`` / ``actual``), or None.

    Returns:
        The Brier score, or None when there is no out-of-sample prediction to score.
    """
    if holdout_predictions is None or holdout_predictions.empty:
        return None
    prediction = holdout_predictions["prediction"].to_numpy(dtype=float)
    actual = holdout_predictions["actual"].to_numpy(dtype=float)
    return float(np.mean((prediction - actual) ** 2))


def print_summary(results: dict[str, dict]) -> None:
    """Print a clean summary table of training results to stdout.

    Shows per-target metrics including accuracy, Brier score, ECE, CLV,
    and market baseline accuracy. Followed by per-season breakdown.

    Args:
        results: Dict mapping target name to training results dict.
    """
    print()
    print("=" * 70)
    print("  Training Summary")
    print("=" * 70)
    print()

    # Header row
    header = (
        f"{'Target':<8} {'Seasons':<10} {'Accuracy':<10} "
        f"{'Brier':<8} {'ECE':<8} {'CLV':<10} {'Market Acc':<10}"
    )
    print(header)
    print("-" * 70)

    for target, result in results.items():
        model = result.get("model_metrics", {})
        baseline = result.get("market_baseline")
        metadata = model.get("metadata", {})

        # Extract season range from season_results. Guard the single-season case (avoid a
        # "2024-24" render) and the bare % 100 modulo, which silently assumes all seasons
        # share a century (IN-03). Display-only string for the training summary table.
        season_results = model.get("season_results", [])
        if season_results:
            seasons = [s["season"] for s in season_results]
            lo, hi = min(seasons), max(seasons)
            season_range = f"{lo}" if lo == hi else f"{lo}-{hi}"
        else:
            season_range = "-"

        # Extract metrics based on target type
        if target == "wp":
            # WP has accuracy from season metrics
            accuracies = [s.get("accuracy", 0) for s in season_results]
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            # A33.2-review IN-03: this column printed the ECE under the Brier heading. The
            # Brier score is now computed from the trainer's own pooled out-of-sample
            # predictions, and is "-" when there are none -- never another metric.
            brier = _pooled_brier(model.get("holdout_predictions"))
            brier_str = f"{brier:.3f}" if brier is not None else "-"
            ece_str = f"{metadata.get('ece', '-'):.3f}" if "ece" in metadata else "-"
        else:
            # ATS/OU have MAE from season metrics
            avg_accuracy = 0.0
            brier_str = "-"
            ece_str = "-"

        # CLV
        clv_summary = metadata.get("clv_summary", {})
        mean_clv = clv_summary.get("mean_clv")
        clv_str = f"{mean_clv:+.4f}" if mean_clv is not None else "-"

        # Market baseline
        market_acc = baseline.get("market_accuracy") if baseline else None
        market_str = f"{market_acc:.3f}" if market_acc is not None else "-"

        # Format accuracy
        acc_str = f"{avg_accuracy:.3f}" if avg_accuracy > 0 else "-"

        row = (
            f"{target.upper():<8} {season_range:<10} {acc_str:<10} "
            f"{brier_str:<8} {ece_str:<8} {clv_str:<10} {market_str:<10}"
        )
        print(row)

    # Per-season breakdown
    print()
    print("-" * 70)
    print("  Per-Season Breakdown")
    print("-" * 70)
    print()

    for target, result in results.items():
        model = result.get("model_metrics", {})
        season_results = model.get("season_results", [])

        if not season_results:
            continue

        print(f"  {target.upper()}:")
        for sr in season_results:
            season = sr.get("season", "?")
            n_games = sr.get("n_games", 0)

            if target == "wp":
                acc = sr.get("accuracy", 0)
                print(f"    {season}: {n_games} games, accuracy={acc:.3f}")
            else:
                mae = sr.get("mae", 0)
                rmse = sr.get("rmse", 0)
                print(f"    {season}: {n_games} games, MAE={mae:.2f}, RMSE={rmse:.2f}")

        print()

    print("=" * 70)
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``models.train`` argument parser.

    Extracted from ``main()`` so the argv surface is testable without running a train
    (the house shape used by tests/unit/test_friday_pipeline_cli.py).

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Train NFL prediction models with walk-forward temporal validation.",
        prog="python -m models.train",
    )
    parser.add_argument(
        "--target",
        choices=list(_VALID_TARGETS),
        default="all",
        help="Which model(s) to train. Default: all",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving model artifacts. Default: artifacts/",
    )
    parser.add_argument(
        "--no-clv",
        action="store_true",
        help="Skip CLV computation (train without closing odds).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Straight re-fit with existing default params; skip Optuna tuning (D24-12).",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help=(
            "Train every target. Equivalent to --target all, which is already the "
            "default; it exists so a re-fit command reads as the deliberate act it is "
            "rather than relying on a default."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Run the PRE-REGISTERED two-arm search (D33.2-17 / R13). This is NOT the same "
            "as the default, which also tunes: under this flag the trial budget and the "
            "search space come from config/tuning_preregistration.py, a RandomSampler "
            "baseline runs FIRST over the same objective and the same budget, and the "
            "searched winner is adopted only if it beats that baseline by the "
            "pre-registered per-target margin on a season neither arm saw. The "
            "feature-group exclusion is DERIVED from the ratified verdict and "
            "--gold-generation becomes REQUIRED. Cannot be combined with --no-tune."
        ),
    )
    parser.add_argument(
        "--thread-limit",
        type=int,
        default=None,
        help=(
            "Pin the BLAS/OpenMP thread pool to this many threads for the whole run and "
            "record the value in each artifact's metadata. Plan 33.2-22 MEASURED the "
            "XGBoost legs returning different answers at different thread counts, by "
            "enough to move a verdict, so an unpinned run is one nobody else can "
            "reproduce. Omitting it leaves the pool unpinned, which is the previous "
            "behaviour exactly. The pin is applied here at runtime (threadpoolctl) and "
            "measured before it is recorded; scripts/train_models.py additionally sets "
            "the environment variables BEFORE numpy is imported. Under --tune it "
            "defaults to config.tuning_preregistration.PINNED_THREAD_COUNT."
        ),
    )
    # SITE 5 of the season partition (RESEARCH 11.1), and the one that matters most for what
    # a FUTURE run records. R6's target names "all three model configs", which read
    # holdout_seasons [2021..2024] -- but those live in artifacts/<id>/metadata.json, the
    # RECORD of a past training run, and D33.1-04 PROHIBITS editing them. The source-side
    # surrogate is these three defaults plus TemporalSplitConfig.default(): together they
    # decide what the next run WRITES into a new metadata.json.
    #
    # DERIVED from conf.season_partition (SPEC R6, D33.1-03). They used to be three string
    # literals reading "2018,2019" / "2020" / "2021,2022,2023,2024".
    _partition = default_season_partition()
    _train_default = ",".join(str(season) for season in _partition.selection)
    _hp_val_default = ",".join(str(season) for season in _partition.hp_val)
    _holdout_default = ",".join(str(season) for season in _partition.holdout)
    parser.add_argument(
        "--config-train-seasons",
        type=str,
        default=_train_default,
        help=f"Comma-separated training seasons. Default: {_train_default}",
    )
    parser.add_argument(
        "--config-hp-val-seasons",
        type=str,
        default=_hp_val_default,
        help=f"Comma-separated HP validation seasons. Default: {_hp_val_default}",
    )
    parser.add_argument(
        "--config-holdout-seasons",
        type=str,
        default=_holdout_default,
        help=f"Comma-separated holdout seasons. Default: {_holdout_default}",
    )
    parser.add_argument(
        "--exclude-groups",
        type=str,
        default=None,
        help=(
            "Comma-separated feature GROUPS to drop from the frame before training "
            f"(D30-01). Registered vocabulary: {', '.join(ALL_REGISTERED_GROUPS)}. "
            "OMITTED, the groups the owner's ratified verdict "
            "(config/group_gate_verdict.toml) rules out are excluded, and a missing "
            "verdict refuses; pass '' to state 'exclude nothing' deliberately. "
            "An unregistered name is a hard failure, never a silent no-op. A registered "
            "group with zero columns present IS a legal no-op. A single scalar token, "
            "matching the --config-*-seasons convention (deliberately not nargs='+', "
            "which is an argv foot-gun under PowerShell when followed by another flag)."
        ),
    )
    parser.add_argument(
        "--exclude-groups-provenance",
        type=str,
        choices=("verdict", "override", "none"),
        default="none",
        help=(
            "Where --exclude-groups came from, recorded verbatim in the artifact's "
            "metadata (WR-04): 'verdict' (DERIVED from the ratified Stage-1 verdict), "
            "'override' (hand-typed on the command line) or 'none'. "
            "scripts/promote_models already resolves this and passes it through, so an "
            "auditor reading a promoted artifact can tell a ratified exclusion from a "
            "typed one without reconstructing which commit was current."
        ),
    )
    parser.add_argument(
        "--gold-generation",
        type=str,
        default=None,
        help=(
            "The gold GENERATION KEY this run's features were read from, recorded "
            f"verbatim in the artifact's metadata under '{GOLD_GENERATION_METADATA_KEY}' "
            "(D33-25 / R7). Measure it with `tests.gold_generation.gold_generation_key()` "
            "and record it in tests.phase33_state.GOLD_GENERATION_AT_REFIT rather than "
            "typing it; a test asserts that record equals both the gold-rebuild ladder's "
            "own generation and the live key. OMITTING THE FLAG LEAVES THE KEY ABSENT, "
            "not empty -- an ordinary training run claims nothing about a weather "
            "generation, and the 2026 gold-weather bridge's flip predicate reads the "
            "marker as an explicit declaration of INTENT."
        ),
    )
    return parser


def main() -> None:
    """Main entry point for unified model training.

    Usage:
        python -m models.train --target all
        python -m models.train --target wp
        python -m models.train --target ats --artifacts-dir custom/path
        python -m models.train --target ats --exclude-groups line_movement
    """
    parser = build_parser()

    args = parser.parse_args()

    if args.tune and args.no_tune:
        parser.error(
            "--tune (the pre-registered two-arm search) and --no-tune (skip tuning "
            "entirely) contradict each other; pass one or neither."
        )

    # D30-01: the Stage-2 feature-group exclusion, applied IN MEMORY between the parquet read
    # and train_target. One resolution for every path (see resolve_exclusion).
    exclude_groups, exclude_groups_provenance, group_verdict_digest = resolve_exclusion(
        args.exclude_groups, args.exclude_groups_provenance
    )
    if args.exclude_groups is None:
        print(
            f"  Exclusion list DERIVED from the ratified Stage-1 verdict "
            f"({GROUP_GATE_VERDICT_PATH}): {list(exclude_groups)}"
        )
    elif exclude_groups_provenance != "verdict":
        print(
            "  [OVERRIDE] --exclude-groups was supplied on the COMMAND LINE; this "
            "list is NOT the ratified Stage-1 verdict."
        )
    thread_limit: int | None = args.thread_limit

    if args.tune:
        from config.tuning_preregistration import PINNED_THREAD_COUNT

        if not args.gold_generation:
            parser.error(
                "--gold-generation is REQUIRED with --tune. A re-fit whose artifact "
                "cannot name the gold generation it was trained on is a re-fit nobody "
                "can reproduce. Measure it with tests.gold_generation.gold_generation_key"
                "() and pass it; this module deliberately does not import the tests "
                "package to measure it for itself."
            )
        if group_verdict_digest is None:
            # A typed list still records WHICH verdict was current beside it.
            _, group_verdict_digest = verdict_exclusion()
        if thread_limit is None:
            thread_limit = PINNED_THREAD_COUNT

    # Parse season lists
    train_seasons = [int(s.strip()) for s in args.config_train_seasons.split(",")]
    hp_val_seasons = [int(s.strip()) for s in args.config_hp_val_seasons.split(",")]
    holdout_seasons = [int(s.strip()) for s in args.config_holdout_seasons.split(",")]

    config = TemporalSplitConfig(
        train_seasons=train_seasons,
        hp_val_seasons=hp_val_seasons,
        holdout_seasons=holdout_seasons,
    )

    # Determine targets to train
    if args.all_targets or args.target == "all":
        targets = ["wp", "ats", "ou"]
    else:
        targets = [args.target]

    # Load closing odds (unless --no-clv)
    closing_odds_df = None
    if not args.no_clv:
        odds_path = Path("data/silver/odds_snapshot.parquet")
        if odds_path.exists():
            # ONE row per game, through the backtest engine's reader (code review WR-04):
            # the live store holds one row per BOOK per capture (nine per 2026 game), and
            # every consumer here joins on game_id. Same dedupe and LAR->LA game_id
            # normalisation as the engine (aaa3b34).
            from backtest.engine import BacktestEngine

            closing_odds_df = BacktestEngine()._load_closing_odds()
            logger.info(
                "Loaded closing odds",
                n_games=len(closing_odds_df),
                path=str(odds_path),
            )
        else:
            logger.warning(
                "Closing odds not found, skipping CLV",
                path=str(odds_path),
            )

    # Train each target, under the pin that gets recorded (WR-02).
    all_results: dict[str, dict] = {}

    with pinned_thread_pool(thread_limit):
        for target in targets:
            # Load feature matrix
            features_path = Path(f"data/gold/features_{target}.parquet")
            if not features_path.exists():
                logger.error(
                    "Feature matrix not found",
                    target=target,
                    path=str(features_path),
                )
                print(
                    f"ERROR: Feature matrix not found at {features_path}. "
                    f"Run the feature build pipeline first.",
                    file=sys.stderr,
                )
                continue

            features_df = pd.read_parquet(features_path)

            # COLD-02 / T-33-18: the FOURTH gold-loading boundary, and the one that bypasses
            # `load_dataframe` entirely. Refuse a PROVISIONAL Elo row as a TRAINING input
            # here, immediately after the read and BEFORE the feature-group exclusion below --
            # a provisional row excluded from the column set is still in the rows being fitted.
            #
            # Judged over every row up to the LAST season this run fits (code review WR-01).
            # Every fold trains on `season < holdout`, tests on the holdout, and the final fit
            # takes `partition.final_fit`, so no row past that season is ever fitted. During
            # the season, gold always carries the next slate's provisional rows (unplayed,
            # live season); refusing on them blocked every re-fit until the season ended.
            from features.elo_features import assert_no_provisional_training_rows

            last_fit_season = max(
                *config.train_seasons,
                *config.hp_val_seasons,
                *config.holdout_seasons,
                *default_season_partition().final_fit,
            )
            assert_no_provisional_training_rows(
                features_df[features_df["season"] <= last_fit_season],
                f"train:{target}",
            )
            logger.info(
                "Loaded features",
                target=target,
                n_rows=len(features_df),
                n_cols=len(features_df.columns),
            )

            # D30-01: apply the Stage-2 feature-group exclusion in memory, BEFORE train_target.
            #
            # The call is SKIPPED entirely on the empty default -- calling select_group_columns
            # with no explicit exclude_groups is NOT equivalent, because its default is the
            # Phase-28 GROUPS deny-list and would silently strip three whole families.
            #
            # group=None is the baseline-leg semantic: every column minus the excluded groups'
            # columns, nothing re-admitted. select_group_columns returns a fresh copy, so
            # data/gold is never touched (HARD BOUNDARY) and no second gold-shaped artifact is
            # needed. train_target's signature is unchanged -- it already accepts any DataFrame.
            #
            # The before/after counts are logged around the call so the exclusion's real effect on
            # the feature set is visible in the run log rather than inferred from a downstream
            # artifact.
            if exclude_groups:
                n_cols_before = len(features_df.columns)
                logger.info(
                    "Applying feature-group exclusion",
                    target=target,
                    exclude_groups=list(exclude_groups),
                    n_cols_before=n_cols_before,
                )
                features_df = select_group_columns(
                    features_df, None, exclude_groups=exclude_groups
                )
                n_cols_after = len(features_df.columns)
                logger.info(
                    "Feature-group exclusion applied",
                    target=target,
                    exclude_groups=list(exclude_groups),
                    n_cols_before=n_cols_before,
                    n_cols_after=n_cols_after,
                    n_cols_dropped=n_cols_before - n_cols_after,
                )

            result = train_target(
                target=target,
                features_df=features_df,
                closing_odds_df=closing_odds_df,
                config=config,
                artifacts_dir=args.artifacts_dir,
                tune=not args.no_tune,
                exclude_groups=exclude_groups,
                exclude_groups_provenance=exclude_groups_provenance,
                gold_generation=args.gold_generation,
                preregistered_search=args.tune,
                group_verdict_digest=group_verdict_digest,
                thread_limit=thread_limit,
            )
            all_results[target] = result

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("No models were trained. Check feature matrix availability.")

    print_run_record(all_results, thread_limit)


def print_run_record(results: dict[str, dict], thread_limit: int | None) -> None:
    """Print the machine-readable record of what this run actually did.

    THE MARGIN VERDICT IS PUBLISHED, PER TARGET, WHETHER OR NOT IT WAS CLEARED. A search
    that failed its pre-registered bar is a FINDING to state plainly, not a result to
    soften -- so a target with no recorded verdict prints ``unrecorded`` rather than
    nothing, and ``unrecorded`` is a failure signal for the caller that reads this.

    Args:
        results: Per-target training results from :func:`train_target`.
        thread_limit: The pinned thread count, or None when the run was unpinned.
    """
    print(f"TRAINED= {len(results)}")
    print(f"THREAD_LIMIT= {thread_limit if thread_limit is not None else 'unpinned'}")
    verdicts = []
    for target, result in results.items():
        record = result.get("adoption_record")
        if not record:
            verdicts.append(f"{target}:unrecorded")
            continue
        state = "cleared" if record["margin_cleared"] else "not-cleared"
        verdicts.append(
            f"{target}:{state}:adopted={record['adopted_arm']}"
            f":gap={record['outer_gap']:+.6f}"
            f":margin={record['margin']}"
            f":outer_season={record['outer_season']}"
        )
    print("MARGIN_VERDICT= " + " ".join(verdicts))
    for target, result in results.items():
        print(f"ARTIFACT= {target} {Path(result['artifact_path']).name}")


if __name__ == "__main__":
    main()
