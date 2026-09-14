"""The 2026 gold-weather bridge cannot become permanent (33.1-SPEC.md R7, D33-25).

WHAT THE BRIDGE IS. ``features.weather.WEATHER_GOLD_DEFAULT_SEASONS`` holds the gold
weather family at its ABSENT-OBSERVATION default for every season it names -- today,
exactly 2026. It was written by Phase 33 Wave 9 Task 5 (commit ``ed42df3``) under an owner
ruling taken at that plan's Task-4 blocking checkpoint. THIS PHASE DOES NOT AUTHOR IT:
``33.1-SPEC.md`` ``## Boundaries`` assigns the authoring to that wave by name. This module
BOUNDS it.

WHY IT NEEDS BOUNDING. The switch exists for ONE reason, recorded in its own block comment:
the deployed O/U artifact was fitted on gold whose weather columns do not vary at all inside
its training window, so live 2026 weather would arrive out of distribution. That is a
TRAIN/SERVE AGREEMENT decision. The moment a deployed model has actually seen weather vary,
the reason expires -- and a dated switch whose reason has expired but which nobody removes
is how a bridge becomes a permanent, undisclosed default.

WHAT THIS MODULE IS NOT. It is NOT a tripwire. A tripwire encodes an owner-accepted fact and
must STAY red. This module must PASS NOW and FAIL LATER, which is the opposite, and
``TestThisModuleIsNotATripwire`` asserts it is absent from
``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`` (33.1-SPEC.md prohibition 7).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_MANIFEST_PATH = REPO_ROOT / "artifacts" / "latest.json"
LIVE_ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

# ONE name for the switch, in ONE place, so a failure message and an assertion cannot
# drift apart.
SWITCH_MODULE = "features.weather"
SWITCH_SYMBOL = "WEATHER_GOLD_DEFAULT_SEASONS"
FLIP_CONDITION_SYMBOL = "WEATHER_GOLD_DEFAULT_FLIP_CONDITION"

# The metadata key a re-fit writes to DECLARE, explicitly, that its training distribution
# included the corrected historical weather record. Defined and tested here; WRITTEN by
# Phase 33 Wave 15. Writing it in this phase would be claiming a re-fit happened.
WEATHER_GENERATION_MARKER_KEY = "trained_on_real_weather_generation"

# The Phase 33 Wave 9 task that authored the switch, named so a reader of a failure
# message can find the decision rather than only the symbol.
SWITCH_AUTHORING_TASK = (
    ".planning/phases/33-live-cold-start-forward-temporal-integrity/33-09-PLAN.md "
    "Task 5, gated on the Task-4 blocking owner checkpoint (D33-25)"
)


def _weather_module() -> Any:
    """The live ``features.weather`` module, imported fresh rather than bound at import."""
    return importlib.import_module(SWITCH_MODULE)


def switch_is_present() -> bool:
    """Whether the 2026 gold-default switch exists in committed source right now.

    Asserted rather than assumed anywhere it matters. A test that silently passes because a
    symbol is absent is a green test asserting nothing, which is the exact failure the
    quarantine scan's non-vacuity control exists to stop.
    """
    return hasattr(_weather_module(), SWITCH_SYMBOL)


def _held_values_agree(left: Any, right: Any) -> bool:
    """Whether two held-state values are the same, treating NaN as equal to NaN."""
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return bool(left == right)


def defaulted_weather_columns() -> frozenset[str]:
    """The gold weather columns the 2026 bridge actually holds at a default.

    DERIVED FROM THE SWITCH'S OWN HELD-STATE PRODUCER, never from a list written here. A
    second copy of the defaulted set would drift from the bridge exactly when it matters --
    a column added to the family and not to the copy is a column the tripwire stops
    watching, with nothing to notice.

    HOW THE DERIVATION WORKS. The switch routes a held game through
    ``WeatherFeaturesCalculator._absent_observation_features``. Called under both
    applicability values, every column whose held value is the SAME in both is a column the
    bridge is defaulting; the one column whose held value DIFFERS is
    ``weather_affects_game``, which carries the real per-game roof fact through the hold
    unchanged and is therefore NOT defaulted. Deriving the carve-out rather than spelling it
    is what keeps this correct if the applicability family ever grows.

    WHY THE FULL BUILDER AND NOT THE COMPRESSED ONE. The full builder is what writes
    ``data/silver/weather_features.parquet``, which is what feeds GOLD -- and a deployed
    artifact's ``feature_list`` names gold columns. The compressed builder is the
    FeatureBuilder-Protocol serving path and contributes no gold column beyond the two it
    shares with the full builder.

    Falls back to ``features.weather.WEATHER_FEATURE_COLUMNS`` when the switch is absent, so
    the predicate stays evaluatable for whoever builds a switch later.
    """
    weather = _weather_module()
    if not switch_is_present():
        return frozenset(weather.WEATHER_FEATURE_COLUMNS)
    calculator = weather.WeatherFeaturesCalculator()
    held_outdoor = calculator._absent_observation_features(is_outdoor=True)
    held_indoor = calculator._absent_observation_features(is_outdoor=False)
    return frozenset(
        column
        for column, value in held_outdoor.items()
        if _held_values_agree(value, held_indoor[column])
    )


def moved_pointers(manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The ``(target, artifact_id)`` pairs in *manifest* that differ from this phase's close.

    The TRIGGER, not the predicate. There is no point evaluating artifact semantics while
    the pointers are unchanged, and a moved pointer by itself says nothing about whether the
    newly-pointed artifact consumes weather.
    """
    recorded = dict(phase33_state.DEPLOYED_POINTERS_AT_PHASE_331_CLOSE)
    moved: list[tuple[str, str]] = []
    for target, recorded_id in recorded.items():
        current_id = manifest.get(target)
        if current_id is not None and current_id != recorded_id:
            moved.append((target, str(current_id)))
    return tuple(moved)


def _read_artifact_feature_list(
    artifacts_root: Path | str, artifact_id: str
) -> frozenset[str]:
    """The selected feature list of *artifact_id*, or an empty set when it is unreadable."""
    path = Path(artifacts_root) / artifact_id / "feature_list.json"
    if not path.is_file():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("features", ())
    return frozenset(str(name) for name in payload)


def _read_artifact_metadata(
    artifacts_root: Path | str, artifact_id: str
) -> dict[str, Any]:
    """The metadata mapping of *artifact_id*, or an empty mapping when it is unreadable."""
    path = Path(artifacts_root) / artifact_id / "metadata.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def flip_condition_is_met(
    manifest: Mapping[str, Any], artifacts_root: Path | str
) -> bool:
    """Whether the 2026 bridge's flip condition is satisfied by *manifest*.

    FIRST DRAFT -- POINTER MOVEMENT ONLY. Replaced in this task's GREEN commit; kept here
    only long enough for the anti-assertion below to fail against it, so the false positive
    Ruling V rejects is PROVEN live rather than described.
    """
    del artifacts_root
    return bool(moved_pointers(manifest))


def _write_synthetic_artifact(
    root: Path,
    artifact_id: str,
    *,
    features: tuple[str, ...],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a synthetic artifact directory under *root*. Never the real ones."""
    directory = root / artifact_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_list.json").write_text(
        json.dumps(list(features)), encoding="utf-8"
    )
    (directory / "metadata.json").write_text(
        json.dumps(dict(metadata or {})), encoding="utf-8"
    )


def _live_manifest() -> dict[str, Any]:
    """The live production manifest, READ-ONLY. This module never writes it."""
    return json.loads(LIVE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_with_one_pointer_moved(target: str, artifact_id: str) -> dict[str, Any]:
    """A SYNTHETIC manifest mapping with exactly one pointer moved."""
    manifest = dict(phase33_state.DEPLOYED_POINTERS_AT_PHASE_331_CLOSE)
    manifest[target] = artifact_id
    return manifest


class TestTheSwitchPresenceIsRecorded:
    """The tree's REAL state is asserted, never inferred from a green run."""

    def test_the_switch_exists_in_committed_source(self) -> None:
        """Branch A: ``WEATHER_GOLD_DEFAULT_SEASONS`` is present, so R7 is satisfiable."""
        assert switch_is_present(), (
            f"{SWITCH_SYMBOL} is absent from {SWITCH_MODULE}. It is authored by "
            f"{SWITCH_AUTHORING_TASK}. With the switch absent there is nothing for this "
            "module to bound, and R7 must be recorded UNMET / INAPPLICABLE by a named "
            "refusal rather than satisfied by a green test that constrains no committed "
            "switch code."
        )

    def test_the_recorded_disposition_matches_the_measured_switch_state(self) -> None:
        """``WEATHER_BRIDGE_R7_DISPOSITION`` records the branch that was actually taken."""
        disposition = phase33_state.WEATHER_BRIDGE_R7_DISPOSITION
        assert disposition["switch_present"] is switch_is_present()
        assert disposition["status"] == "satisfied"
        assert disposition["branch"] == "A"

    def test_the_switch_holds_exactly_the_2026_season(self) -> None:
        """The bridge is one season wide. A widened hold is a different decision."""
        weather = _weather_module()
        assert set(getattr(weather, SWITCH_SYMBOL)) == {2026}

    def test_the_flip_condition_names_the_refit_rather_than_a_future_phase(
        self,
    ) -> None:
        """The removal condition is THIS phase's re-fit -- Wave 15 -- not Phase 37."""
        weather = _weather_module()
        condition = getattr(weather, FLIP_CONDITION_SYMBOL)
        assert "Wave 15" in condition
        assert "Phase 37" not in condition


class TestTheSwitchIsBounded:
    """BRANCH A ONLY. The live tree does NOT satisfy the flip condition -- yet."""

    def test_the_live_manifest_does_not_satisfy_the_flip_condition(self) -> None:
        """Green today. Red once a deployed model has actually seen weather vary."""
        manifest = _live_manifest()
        met = flip_condition_is_met(manifest, LIVE_ARTIFACTS_ROOT)
        assert not met, _bounding_failure_message(manifest)

    def test_the_bounding_failure_message_names_the_switch(self) -> None:
        """The failure a later reader will actually see names the switch and what to do."""
        message = _bounding_failure_message(_live_manifest())
        assert SWITCH_SYMBOL in message
        assert SWITCH_MODULE in message
        assert "must be REMOVED" in message


def _bounding_failure_message(manifest: Mapping[str, Any]) -> str:
    """The message the bounding assertion prints once the flip condition is met."""
    moved = moved_pointers(manifest)
    clauses = []
    for target, artifact_id in moved:
        selected = _read_artifact_feature_list(LIVE_ARTIFACTS_ROOT, artifact_id)
        intersecting = sorted(selected & defaulted_weather_columns())
        metadata = _read_artifact_metadata(LIVE_ARTIFACTS_ROOT, artifact_id)
        marker = metadata.get(WEATHER_GENERATION_MARKER_KEY)
        clauses.append(
            f"  {target} -> {artifact_id}: intersecting defaulted weather columns "
            f"{intersecting or 'none'}; {WEATHER_GENERATION_MARKER_KEY}={marker!r}"
        )
    return (
        f"THE 2026 GOLD-WEATHER BRIDGE HAS OUTLIVED ITS REASON. {SWITCH_SYMBOL} in "
        f"{SWITCH_MODULE} still holds season 2026 at the absent-observation default, but a "
        "deployed artifact now consumes the corrected weather record.\n"
        f"Flip condition: {getattr(_weather_module(), FLIP_CONDITION_SYMBOL, '(absent)')}\n"
        f"Moved pointers and which clause fired:\n" + "\n".join(clauses) + "\n"
        f"{SWITCH_SYMBOL} must be REMOVED, not re-dated. A later date is the same hold "
        "wearing a new number, and the reason for the hold -- a deployed model that had "
        "never seen weather vary -- has expired."
    )


class TestTheFlipPredicateFiresOnArtifactSemantics:
    """The fail-first proof, in BOTH clauses, so neither is dead code."""

    def test_clause_one_an_intersecting_defaulted_weather_column_satisfies_it(
        self, tmp_path: Path
    ) -> None:
        """A moved pointer to an artifact selecting a DEFAULTED weather column: True."""
        _write_synthetic_artifact(
            tmp_path,
            "wp_synthetic_weatherful",
            features=("elo_diff", "raw_temp_f"),
        )
        manifest = _manifest_with_one_pointer_moved("wp", "wp_synthetic_weatherful")
        assert flip_condition_is_met(manifest, tmp_path) is True

    def test_clause_two_the_generation_marker_alone_satisfies_it(
        self, tmp_path: Path
    ) -> None:
        """A moved pointer to a marker-bearing artifact with ZERO weather features: True."""
        _write_synthetic_artifact(
            tmp_path,
            "wp_synthetic_marked",
            features=("elo_diff", "home_rest_days"),
            metadata={
                WEATHER_GENERATION_MARKER_KEY: (
                    phase33_state.GOLD_GENERATION_AFTER_WEATHER_RUNG
                )
            },
        )
        manifest = _manifest_with_one_pointer_moved("wp", "wp_synthetic_marked")
        assert flip_condition_is_met(manifest, tmp_path) is True

    def test_an_unmoved_manifest_never_reaches_the_predicate(
        self, tmp_path: Path
    ) -> None:
        """Pointers unchanged: False, and no artifact is read."""
        manifest = dict(phase33_state.DEPLOYED_POINTERS_AT_PHASE_331_CLOSE)
        assert flip_condition_is_met(manifest, tmp_path) is False


class TestAMovedPointerAloneDoesNotSatisfyTheCondition:
    """THE ANTI-ASSERTION, and the whole point of Ruling V.

    Under the first draft's pointer-movement predicate this case returned True. It is the
    LIVE shape of a Wave-15 promotion of a still-weatherless WP -- ``wp_20260824_113325``
    and ``ats_20260605_220128`` each select ZERO weather columns today -- and a True here
    would instruct the removal of a guard whose reason still held exactly.
    ``models/prediction_pipeline.py:705-716`` reads ONLY the artifact's selected
    ``feature_list``, so such an artifact serves precisely as a weather-blind model does.
    """

    def test_a_weatherless_markerless_promotion_does_not_satisfy_it(
        self, tmp_path: Path
    ) -> None:
        """Pointer moved, artifact weather-blind and unmarked: FALSE."""
        _write_synthetic_artifact(
            tmp_path,
            "wp_synthetic_weatherless",
            features=("elo_diff", "home_rest_days", "is_divisional"),
            metadata={"train_seasons": [2002, 2025]},
        )
        manifest = _manifest_with_one_pointer_moved("wp", "wp_synthetic_weatherless")
        assert flip_condition_is_met(manifest, tmp_path) is False

    def test_the_two_deployed_weatherless_artifacts_are_the_recorded_live_case(
        self,
    ) -> None:
        """WP and ATS select zero defaulted weather columns; O/U selects seventeen."""
        defaulted = defaulted_weather_columns()
        measured = {
            target: len(set(features) & defaulted)
            for target, features in (
                phase33_state.DEPLOYED_FEATURE_LISTS_AT_PHASE_331_CLOSE.items()
            )
        }
        assert measured == (
            phase33_state.DEPLOYED_DEFAULTED_WEATHER_SELECTION_AT_PHASE_331_CLOSE
        )
        assert measured["wp"] == 0
        assert measured["ats"] == 0
        assert measured["ou"] == 17


class TestDefaultedWeatherColumnsComeFromTheSwitch:
    """The defaulted set is DERIVED, never spelled in this module."""

    def test_the_set_is_non_empty(self) -> None:
        """A predicate over an empty column set would be vacuously False forever."""
        assert defaulted_weather_columns()

    def test_the_applicability_column_is_not_defaulted(self) -> None:
        """``weather_affects_game`` carries the real roof fact THROUGH the hold."""
        assert "weather_affects_game" not in defaulted_weather_columns()

    def test_the_coverage_flag_is_defaulted(self) -> None:
        """``weather_coverage`` is forced to 0.0 by the hold, so it IS defaulted."""
        weather = _weather_module()
        assert weather.WEATHER_COVERAGE_COLUMN in defaulted_weather_columns()

    def test_the_set_is_a_subset_of_the_declared_weather_family(self) -> None:
        """Nothing outside the module's own weather declaration can enter the set."""
        weather = _weather_module()
        assert defaulted_weather_columns() <= set(weather.WEATHER_FEATURE_COLUMNS)


class TestThisModuleIsNotATripwire:
    """A tripwire must stay RED. This module must pass NOW and fail LATER.

    Adding any node id defined here to ``DELIBERATE_TRIPWIRE_NODE_IDS`` would neuter the one
    instrument that stops the 2026 bridge becoming permanent (33.1-SPEC.md prohibition 7):
    the list is the register of failures the owner has ACCEPTED, so a member of it is a test
    nobody is expected to act on.
    """

    def test_no_node_id_from_this_module_is_in_the_deliberate_list(self) -> None:
        """Every node id this module defines is absent from the tripwire register."""
        module_path = "tests/unit/test_weather_bridge_expiry.py"
        listed = [
            node_id
            for node_id in phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS
            if node_id.startswith(module_path)
        ]
        assert not listed, (
            f"{module_path} node ids were added to DELIBERATE_TRIPWIRE_NODE_IDS: {listed}. "
            "That register is for tests the owner has accepted as permanently red; this "
            "module is the opposite -- it passes now and must fail once the 2026 bridge "
            "outlives its reason."
        )

    def test_the_tripwire_tuple_still_holds_exactly_five_entries(self) -> None:
        """The five registered reds are unchanged by this plan."""
        assert len(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS) == 5


class TestTheLiveManifestIsNeverWritten:
    """The predicate takes a manifest MAPPING; the live file is read-only."""

    def test_the_live_manifest_still_holds_the_recorded_pointers(self) -> None:
        """Reading it changed nothing, and the recorded pointers ARE the live ones."""
        manifest = _live_manifest()
        for target, artifact_id in phase33_state.DEPLOYED_POINTERS_AT_PHASE_331_CLOSE:
            assert manifest[target] == artifact_id
