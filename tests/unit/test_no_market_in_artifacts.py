"""No market line reached the Phase-33.2 re-fit artifacts, and no dead model was consulted.

WHAT IS ASSERTED, AND WHY IT IS THE WRITTEN FILE
------------------------------------------------
The CLEAN-02 F-1 lesson, applied again: a SOURCE SCAN proves what the code intends and the
WRITTEN ARTIFACT proves what happened. So every assertion here opens the files the save path
actually produced --

  * ``feature_list.json`` for the feature set. That is the CANONICAL copy
    (``models/artifacts.py`` writes it there); ``metadata.json`` carries a convenience
    ``feature_names`` key which is NOT what serving reads;
  * ``metadata.json`` for the gold generation, the group verdict, the study tag and the
    thread pin;
  * ``{target}_params.json`` ``tuning_metadata`` for the search record.

-- and never the trainer source.

WHY A MARKET LINE MUST NOT BE THERE (D33.2-03). All three DEPLOYED models were fitted with a
betting line inside their selected feature set -- WP ``snapshot_spread``; ATS
``snapshot_spread`` plus ``snapshot_ml_prob_home_fair``; O/U ``snapshot_total``, its rank-1
input of 25 -- and for 2018-2025 those are CLOSING lines, which did not exist at the
day-before-kickoff lock. The line stays available for pricing, grading and CLV. It stops
being a thing a model is fitted on.

THE TWO CONTROLS, so a green run is not mistaken for a wired-up check:
  * NON-VACUITY -- every feature list is non-empty. An empty list would satisfy an absence
    assertion perfectly while proving nothing;
  * PLANTED VIOLATION -- the same scan, run over a synthetic artifact carrying
    ``snapshot_spread``, MUST fail.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from backtest.signal_lift import _MARKET_SUFFIXES, _is_market_col
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


def _artifact_dir(artifact_id: str) -> Path:
    return ARTIFACTS_ROOT / artifact_id


def _feature_list(artifact_id: str) -> list[str]:
    return json.loads(
        (_artifact_dir(artifact_id) / "feature_list.json").read_text(encoding="utf-8")
    )


def _metadata(artifact_id: str) -> dict[str, Any]:
    return json.loads(
        (_artifact_dir(artifact_id) / "metadata.json").read_text(encoding="utf-8")
    )


def _tuning_metadata(target: str, artifact_id: str) -> dict[str, Any]:
    params = json.loads(
        (_artifact_dir(artifact_id) / f"{target}_params.json").read_text(
            encoding="utf-8"
        )
    )
    return params["tuning_metadata"]


def market_columns_in(feature_list: list[str]) -> list[str]:
    """Return the betting-line columns present in *feature_list*.

    The suffix predicate is IMPORTED from ``backtest.signal_lift`` rather than re-derived.
    A second copy of the rule here would be a second contract wearing one name -- and the
    rule is subtler than it looks: it matches by exact SUFFIX so that ``total_points`` (the
    O/U target) and ``rolling_total_epa`` are not swept up by a bare ``"total" in col``.
    """
    return [column for column in feature_list if _is_market_col(column)]


class TestNoMarketLineInTheWrittenFeatureLists:
    """D33.2-03 / R13, asserted on the artifacts that were actually written."""

    def test_the_three_refit_artifacts_exist(self) -> None:
        assert len(phase33_state.P332_23_REFIT_ARTIFACT_IDS) == 3
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            directory = _artifact_dir(artifact_id)
            assert directory.is_dir(), (
                f"the re-fit artifact {artifact_id} recorded for target '{target}' is not "
                f"on disk at {directory}. artifacts/ is gitignored, so this is a checkout "
                "without the re-fit rather than a defect in it."
            )
            assert (directory / "feature_list.json").is_file()
            assert (directory / "metadata.json").is_file()
            assert (directory / f"{target}_params.json").is_file()

    def test_no_written_feature_list_contains_a_market_line(self) -> None:
        found = {
            artifact_id: market_columns_in(_feature_list(artifact_id))
            for _, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS
        }
        assert all(not columns for columns in found.values()), (
            f"a betting line reached a re-fit artifact's WRITTEN feature list: {found}. "
            "Under D33.2-03 no market line of any timing is a model input for any target."
        )

    def test_every_feature_list_is_non_empty(self) -> None:
        """NON-VACUITY: an empty list would pass the absence check while proving nothing."""
        sizes = {
            artifact_id: len(_feature_list(artifact_id))
            for _, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS
        }
        assert all(size > 0 for size in sizes.values()), sizes

    def test_the_scan_catches_a_planted_violation(self, tmp_path: Path) -> None:
        """FAIL-CLOSED control on the scan itself."""
        planted = ["home_elo", "snapshot_spread", "rest_days_home"]
        assert market_columns_in(planted) == ["snapshot_spread"]
        # And the near-misses the suffix rule deliberately does NOT sweep up.
        assert market_columns_in(["total_points", "rolling_total_epa"]) == []

    def test_the_suffix_rule_is_the_canonical_one(self) -> None:
        """No second declaration of what a market column is."""
        assert set(_MARKET_SUFFIXES) == {
            "snapshot_spread",
            "snapshot_total",
            "snapshot_ml_prob_home_fair",
            "spread_movement",
            "total_movement",
        }

    def test_the_market_probability_blend_input_is_not_a_model_input(self) -> None:
        """Plan 33.2-21's market-probability artifact is a BLEND input, never a fit input.

        It lives under ``artifacts/`` beside the models, so it is exactly the thing a
        careless scan would either be satisfied by or confused by. Neither: no re-fit
        artifact's feature list names it, and it is not one of the three re-fit ids.
        """
        refit_ids = {
            artifact_id for _, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS
        }
        assert not any(
            artifact_id.startswith("market_probability") for artifact_id in refit_ids
        )
        for _, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            assert not any(
                "market_probability" in column for column in _feature_list(artifact_id)
            )


class TestNoPreCorrectionComparator:
    """The SPEC prohibition: no dead model or frozen gate baseline was kept or consulted."""

    def test_no_metadata_names_a_pre_correction_artifact_id(self) -> None:
        """The written record names no OTHER artifact directory.

        Every artifact directory under ``artifacts/`` that is not one of this re-fit's
        three is either a pre-correction model or Plan 33.2-21's blend input. A re-fit that
        named one in its own record would be a re-fit that compared itself to a dead model.
        """
        refit_ids = {
            artifact_id for _, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS
        }
        others = sorted(
            path.name
            for path in ARTIFACTS_ROOT.iterdir()
            if path.is_dir() and path.name not in refit_ids
        )
        assert others, (
            "no other artifact directory exists at all, so this check cannot discriminate"
        )
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            directory = _artifact_dir(artifact_id)
            blob = "\n".join(
                (directory / name).read_text(encoding="utf-8")
                for name in (
                    "metadata.json",
                    "feature_list.json",
                    f"{target}_params.json",
                )
            )
            named = [other for other in others if other in blob]
            assert named == [], (
                f"the re-fit artifact {artifact_id} names other artifact(s) in its own "
                f"written record: {named}. Pre-correction models are dead by standing "
                "owner ruling and are never a comparator."
            )

    def test_no_metadata_names_a_gate_baseline(self) -> None:
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            directory = _artifact_dir(artifact_id)
            blob = "\n".join(
                (directory / name).read_text(encoding="utf-8")
                for name in ("metadata.json", f"{target}_params.json")
            )
            for forbidden in ("gate.toml", "gate_baseline", "incumbent"):
                assert forbidden not in blob, (
                    f"{artifact_id} names {forbidden!r} in its written record; the "
                    "frozen gate baseline is not an input to this fit."
                )

    def test_the_fit_reads_no_other_artifact_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The READ SET is asserted, not the intent.

        A real (tiny-budget) WP fit runs under an audit hook that records every file the
        process opens. The assertion is that nothing under the production ``artifacts/``
        tree -- no pre-correction model, no ``latest.json``, no blend input -- was read.
        Asserting the code's intention instead would prove only that nobody wrote an
        obvious load; this proves nothing was loaded.
        """
        import pandas as pd

        from models.train import train_target
        from models.trainers import base as base_trainer

        monkeypatch.setattr(
            base_trainer, "TRIAL_BUDGET_BY_TARGET", {"wp": 4, "ats": 4, "ou": 4}
        )
        monkeypatch.setattr(base_trainer, "TUNING_STORAGE_DIR", tmp_path / "optuna")

        features = pd.read_parquet(REPO_ROOT / "data/gold/features_wp.parquet")

        opened: list[str] = []
        _record_into(opened)
        try:
            train_target(
                target="wp",
                features_df=features,
                closing_odds_df=None,
                artifacts_dir=tmp_path / "artifacts",
                tune=True,
                preregistered_search=True,
            )
        finally:
            _record_into(None)

        production_reads = sorted(
            {
                path
                for path in opened
                if Path(path).resolve().is_relative_to(ARTIFACTS_ROOT)
            }
        )
        assert opened, "the audit hook recorded nothing at all; it proves nothing"
        assert production_reads == [], (
            "the fit touched the production artifacts tree: "
            f"{production_reads}. Nothing under artifacts/ is an input to a re-fit -- not "
            "a pre-correction model, not latest.json, not the blend input."
        )


# ---------------------------------------------------------------------------
# The audit hook. It is installed ONCE and is inert unless a test arms it: audit hooks
# cannot be removed for the life of the interpreter, so the recorder is a module-level
# switch rather than an uninstallable hook.
# ---------------------------------------------------------------------------

_RECORDER: list[str] | None = None
_HOOK_INSTALLED = False


def _record_into(sink: list[str] | None) -> None:
    """Arm (or disarm) the open-event recorder, installing the hook on first use."""
    global _RECORDER, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    _RECORDER = sink


def _audit(event: str, args: tuple) -> None:
    """Record the path of every ``open`` the process performs while armed."""
    if _RECORDER is None or event != "open":
        return
    target = args[0]
    if isinstance(target, (str, Path)):
        _RECORDER.append(str(target))


class TestTheWrittenTuningRecord:
    """The search record each artifact PUBLISHES -- started counts, outer season, digests."""

    def test_both_arms_started_exactly_the_pre_registered_budget(self) -> None:
        from config.tuning_preregistration import TRIAL_BUDGET_BY_TARGET

        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            for arm in ("tpe", "random"):
                assert (
                    record["arms"][arm]["trials_started"]
                    == (TRIAL_BUDGET_BY_TARGET[target])
                ), (target, arm, record["arms"][arm]["trials_started"])

    def test_completed_and_pruned_are_reported_and_floored_by_nothing(self) -> None:
        """A published fact, deliberately NOT a pass bar.

        Under ``HyperbandPruner`` most trials are pruned by design, so a completed-trial
        floor could not be relied on to pass on a correct run -- and two arms with
        different samplers and different crc32 brackets will not complete the same number.
        """
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            for arm in ("tpe", "random"):
                arm_record = record["arms"][arm]
                assert isinstance(arm_record["trials_completed"], int)
                assert isinstance(arm_record["trials_pruned"], int)
                assert isinstance(arm_record["trials_failed"], int)
                started = arm_record["trials_started"]
                assert (
                    arm_record["trials_completed"]
                    + arm_record["trials_pruned"]
                    + arm_record["trials_failed"]
                    == started
                ), (
                    "the three terminal counts do not close against the started count, so "
                    "one of them describes something other than this study"
                )

    def test_the_two_arms_are_two_distinct_studies(self) -> None:
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            assert (
                record["arms"]["tpe"]["study_name"]
                != record["arms"]["random"]["study_name"]
            )
            assert record["arms"]["random"]["sampler"] == "RandomSampler"
            assert record["arms"]["tpe"]["sampler"] == "TPESampler"

    def test_the_outer_season_is_the_first_holdout_season_and_is_not_2025(self) -> None:
        from conf.season_partition import default_season_partition
        from config.tuning_preregistration import EXCLUDED_OUTER_SEASONS

        expected = default_season_partition().holdout[0]
        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            assert record["outer_season"] == expected
            assert record["outer_season"] not in EXCLUDED_OUTER_SEASONS

    def test_the_search_space_digest_matches_the_committed_pre_registration(
        self,
    ) -> None:
        from config.tuning_preregistration import search_space_digest

        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            assert record["search_space_digest"] == search_space_digest(target), (
                f"{artifact_id} searched a space whose digest does not match the "
                "committed pre-registration, so the two arms cannot be said to have "
                "searched the declared space"
            )

    def test_the_margin_verdict_is_recorded_per_target(self) -> None:
        """A failure is a FINDING, published, never a silent fall-through."""
        from config.tuning_preregistration import (
            BEAT_RANDOM_MARGIN_BY_TARGET,
            NOT_CLEARED_ARM,
            STUDY_ARM_TPE,
        )

        for target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            record = _tuning_metadata(target, artifact_id)
            assert record["margin"] == BEAT_RANDOM_MARGIN_BY_TARGET[target]
            assert isinstance(record["margin_cleared"], bool)
            assert record["adopted_arm"] in (STUDY_ARM_TPE, NOT_CLEARED_ARM)
            if not record["margin_cleared"]:
                assert record["adopted_arm"] == NOT_CLEARED_ARM
                assert record["not_cleared_rule"], (
                    "the bar was missed and the pre-registered rule that was applied is "
                    "not recorded beside it"
                )

    def test_each_artifact_records_the_gold_it_was_trained_on_and_the_thread_pin(
        self,
    ) -> None:
        for _target, artifact_id in phase33_state.P332_23_REFIT_ARTIFACT_IDS:
            metadata = _metadata(artifact_id)
            assert (
                metadata["gold_generation_digest"]
                == phase33_state.P332_20_CLEAN_BUILD_GOLD_GENERATION
            )
            assert (
                metadata["group_verdict_digest"]
                == phase33_state.P332_22_GROUP_VERDICT_FILE_SHA256
            )
            assert (
                metadata["tuning_study_tag"] == phase33_state.P332_23_TUNING_STUDY_TAG
            )
            assert metadata["thread_limit"] == phase33_state.P332_23_THREAD_LIMIT
            assert list(metadata["exclude_groups"]) == list(
                phase33_state.P332_22_EXCLUDED_GROUPS
            )
            assert metadata["exclude_groups_provenance"] == "verdict"
