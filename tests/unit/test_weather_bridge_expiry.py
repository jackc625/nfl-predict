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


def _artifact_declares_a_post_rung_generation(metadata: Mapping[str, Any]) -> bool:
    """Whether *metadata* carries a generation marker naming corrected-weather gold.

    "At or after this phase's rung" cannot be written as an inequality: a gold generation
    key is a CONTENT DIGEST, and content digests do not order. The testable form is "the
    marker names a generation that is not the uncaptured pre-rung sentinel" -- which is the
    only pre-rung generation this repository can name at all, because nobody captured a real
    key before the rebuild overwrote the bytes.
    """
    marker = metadata.get(WEATHER_GENERATION_MARKER_KEY)
    if not isinstance(marker, str) or not marker.strip():
        return False
    return marker != phase33_state.GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED


def flip_condition_is_met(
    manifest: Mapping[str, Any], artifacts_root: Path | str
) -> bool:
    """Whether the 2026 bridge's flip condition is satisfied by *manifest*.

    THE CONDITION IS ARTIFACT SEMANTICS, WITH A MOVED POINTER AS THE TRIGGER ONLY. Each step
    of the argument corrects the one before it, and a reader who sees only the conclusion
    will simplify it straight back to the version that is wrong.

    "WAVE 15 RAN" IS WRONG. Wave 15 may promote nothing -- two refusals is the gate working,
    as Phase 30 already demonstrated -- and if it promotes nothing then every deployed model
    is still weather-blind and holding the 2026 gold weather family at its default is still
    the CORRECT train/serve agreement.

    "A POINTER MOVED" IS ALSO WRONG, AND IN THE DANGEROUS DIRECTION.
    ``models/prediction_pipeline.py:705-716`` reads ONLY the artifact's selected
    ``feature_list`` (``games_data[wp_features]``), so a promoted artifact with zero weather
    features serves EXACTLY as a weather-blind model does. That is the live case, not a
    hypothetical: ``wp_20260824_113325`` and ``ats_20260605_220128`` each select zero
    defaulted weather columns today. Under a pointer predicate, a Wave-15 promotion of a
    still-weatherless WP would flip the condition and this tripwire would demand the bridge
    be REMOVED while its reason still held -- a false positive that removes a guard.

    SO THE PREDICATE IS SEMANTICS, IN TWO STAGES:

      1. THE TRIGGER. A pointer differs from ``DEPLOYED_POINTERS_AT_PHASE_331_CLOSE``. If
         none has, return False without reading any artifact -- there is nothing to
         evaluate.
      2. THE PREDICATE, on the artifact that moved. True if EITHER its selected feature list
         intersects ``defaulted_weather_columns()``, OR its metadata carries the
         ``trained_on_real_weather_generation`` marker. Clause 1 is
         necessary-and-sufficient for HARM; clause 2 exists because it is not sufficient for
         INTENT -- a model deliberately re-fit on corrected weather that still selected none
         is a model whose training distribution changed, and the marker is how Wave 15 says
         so explicitly.

    Args:
        manifest: A manifest MAPPING, never a path. The live ``artifacts/latest.json`` is
            read-only here and the tests build synthetic mappings.
        artifacts_root: Directory holding the artifact directories to read.

    Returns:
        True once a deployed artifact actually consumes, or declares that it was trained on,
        the corrected weather record.
    """
    moved = moved_pointers(manifest)
    if not moved:
        return False
    defaulted = defaulted_weather_columns()
    for _target, artifact_id in moved:
        selected = _read_artifact_feature_list(artifacts_root, artifact_id)
        if selected & defaulted:
            return True
        if _artifact_declares_a_post_rung_generation(
            _read_artifact_metadata(artifacts_root, artifact_id)
        ):
            return True
    return False


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

    def test_the_held_season_set_is_now_EMPTY(self) -> None:
        """INVERTED 2026-09-14 by Plan 33-15 Task 4, on the owner's `remove the bridge`.

        WHAT IT ASSERTED: ``set(WEATHER_GOLD_DEFAULT_SEASONS) == {2026}`` -- "the bridge
        is one season wide; a widened hold is a different decision". That was the right
        assertion while the hold existed.

        WHAT IT ASSERTS NOW: the set is EMPTY, so NO season is held. The flip condition
        said the switch "must be REMOVED, not re-dated", and an empty held-season set is
        that removal -- a narrower assertion than the old one, not a weaker one, because
        ``{2026}`` and ``{2027}`` would both have satisfied "one season wide" and neither
        satisfies this.

        WHY THE SYMBOL SURVIVES the removal is recorded in ``features/weather.py``'s own
        dated block: deleting the two names would take the record of the ruling with them
        and would break the derivation :func:`defaulted_weather_columns` performs from the
        module's own held-state producer.
        """
        weather = _weather_module()
        assert set(getattr(weather, SWITCH_SYMBOL)) == set(), (
            "the 2026 gold-weather hold is still in force. Plan 33-15 Task 4 removed it "
            "on the owner's ruling of 2026-09-14, after all three targets were re-fit on "
            "corrected-weather gold and promoted."
        )

    def test_no_season_is_held_at_the_gold_default_any_more(self) -> None:
        """The BEHAVIOURAL half: an empty set is only a removal if the predicate agrees."""
        weather = _weather_module()
        for season in (2025, 2026, 2027):
            held = weather._game_is_held_at_gold_default(
                {"season": season}, f"{season}_W01_BUF@KC"
            )
            assert held is False, season

    def test_the_flip_condition_names_the_refit_rather_than_a_future_phase(
        self,
    ) -> None:
        """The removal condition is THIS phase's re-fit -- Wave 15 -- not Phase 37."""
        weather = _weather_module()
        condition = getattr(weather, FLIP_CONDITION_SYMBOL)
        assert "Wave 15" in condition
        assert "Phase 37" not in condition


class TestTheSwitchIsBounded:
    """THE BOUND HELD, THE CONDITION WAS MET, AND THE SWITCH WAS REMOVED.

    INVERTED 2026-09-14 by Plan 33-15 Task 4, on the owner's ruling `remove the bridge`.

    WHAT THIS CLASS ASSERTED, and it did its job: while the hold was in force, the live
    manifest did NOT satisfy the flip condition, and the moment it did this class went RED
    with a message naming the switch, its module, its flip-condition string, the pointer
    that moved and which clause fired. It fired exactly once, on the promotion of
    ``wp_20260914_221745`` / ``ats_20260914_221751`` / ``ou_20260914_221756``, which is the
    event it was built for.

    WHAT IT ASSERTS NOW: the condition IS met AND the switch is gone. Those two together
    are the only end state the flip condition permits -- "REMOVED, not re-dated" -- so a
    future edit that re-introduces a held season while the condition stands met fails here
    again. The bounding proof itself is PRESERVED below as a historical assertion rather
    than deleted, because a removed guard proves nothing about the removal it was meant to
    force.

    IT IS STILL NOT A TRIPWIRE. It is GREEN under a corrected assertion, not deliberately
    red, so ``DELIBERATE_TRIPWIRE_NODE_IDS`` stays at FIVE -- ``TestThisModuleIsNotATripwire``
    below asserts both halves of that.
    """

    def test_the_live_manifest_now_SATISFIES_the_flip_condition(self) -> None:
        """The inversion. Three deployed artifacts declare the corrected generation."""
        manifest = _live_manifest()
        assert flip_condition_is_met(manifest, LIVE_ARTIFACTS_ROOT) is True, (
            "the live manifest no longer satisfies the flip condition. If a rollback "
            "restored the pre-Plan-33-15 pointers, the 2026 hold must be RESTORED with "
            "its own dated ruling rather than this assertion being relaxed."
        )

    def test_the_switch_was_removed_rather_than_re_dated(self) -> None:
        """THE PAIR THAT MATTERS: condition met AND hold gone, asserted together."""
        weather = _weather_module()
        assert flip_condition_is_met(_live_manifest(), LIVE_ARTIFACTS_ROOT) is True
        assert set(getattr(weather, SWITCH_SYMBOL)) == set(), (
            "the flip condition is MET and a held season is still in force. That is the "
            "one state the condition forbids by name: the switch must be REMOVED, not "
            "re-dated, and a later date is the same hold wearing a new number."
        )

    def test_the_flip_condition_records_that_it_was_met(self) -> None:
        """The removal is DATED and names what met it, or it is an unexplained deletion."""
        condition = getattr(_weather_module(), FLIP_CONDITION_SYMBOL)
        assert "MET AND REMOVED" in condition
        assert "2026-09-14" in condition
        for artifact in (
            "wp_20260914_221745",
            "ats_20260914_221751",
            "ou_20260914_221756",
        ):
            assert artifact in condition, artifact

    def test_the_bounding_failure_message_still_names_the_switch(self) -> None:
        """PRESERVED AS A HISTORICAL ASSERTION, with its reason recorded here.

        This message is what a reader WOULD have been shown when the bound fired, and it
        is what they WILL be shown if a future phase re-introduces a hold and this module
        has to bound one again. Deleting it once the guard had done its job would leave
        the next bridge with no message at all, which is how the first one nearly became
        permanent.
        """
        message = _bounding_failure_message(_live_manifest(), LIVE_ARTIFACTS_ROOT)
        assert SWITCH_SYMBOL in message
        assert SWITCH_MODULE in message
        assert "must be REMOVED" in message

    def test_the_bounding_assertion_FIRES_against_a_simulated_post_flip_state(
        self, tmp_path: Path
    ) -> None:
        """THE PLANTED CONTROL. Driven against a post-flip state, this assertion FAILS.

        Without it the bounding test is a green assertion that has only ever seen the state
        it passes on -- indistinguishable from one that is not wired up. The control runs the
        SAME assertion expression against a SYNTHETIC manifest and artifact directory, never
        the real ones, and asserts both that it raises and that the message a later reader
        will actually be shown names the switch, its module, its flip-condition string, the
        pointer that moved and which clause fired.
        """
        _write_synthetic_artifact(
            tmp_path,
            "ou_synthetic_post_flip",
            features=("snapshot_total", "raw_temp_f", "wind_mph"),
            metadata={
                WEATHER_GENERATION_MARKER_KEY: (
                    phase33_state.GOLD_GENERATION_AFTER_WEATHER_RUNG
                )
            },
        )
        manifest = _manifest_with_one_pointer_moved("ou", "ou_synthetic_post_flip")

        raised: AssertionError | None = None
        try:
            met = flip_condition_is_met(manifest, tmp_path)
            assert not met, _bounding_failure_message(manifest, tmp_path)
        except AssertionError as error:
            raised = error

        assert raised is not None, (
            "the bounding assertion did NOT fire against a simulated post-flip state. A "
            "tripwire that cannot be made to fail proves nothing about the state it "
            "passes on."
        )
        message = str(raised)
        assert SWITCH_SYMBOL in message
        assert SWITCH_MODULE in message
        assert getattr(_weather_module(), FLIP_CONDITION_SYMBOL) in message
        assert "ou -> ou_synthetic_post_flip" in message
        assert "raw_temp_f" in message
        assert WEATHER_GENERATION_MARKER_KEY in message
        assert "must be REMOVED" in message
        assert "not re-dated" in message

    def test_the_live_tree_is_untouched_by_the_planted_control(self) -> None:
        """The control wrote only into ``tmp_path``.

        INVERTED with the class above: the live answer is now True, and the claim this
        test carries is unchanged -- that the planted control did not CHANGE it. The
        control writes a synthetic artifact into ``tmp_path`` and a synthetic manifest
        mapping in memory, so the live reading either side of it must be whatever the live
        tree actually says, which since the Plan 33-15 promotion is True.
        """
        manifest = _live_manifest()
        assert flip_condition_is_met(manifest, LIVE_ARTIFACTS_ROOT) is True


def _bounding_failure_message(
    manifest: Mapping[str, Any], artifacts_root: Path | str
) -> str:
    """The message the bounding assertion prints once the flip condition is met."""
    moved = moved_pointers(manifest)
    clauses = []
    for target, artifact_id in moved:
        selected = _read_artifact_feature_list(artifacts_root, artifact_id)
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
    """The predicate takes a manifest MAPPING; the live file is read-only.

    RE-POINTED 2026-09-14 by Plan 33-15 Task 4. The claim is unchanged -- THIS MODULE
    never writes the live manifest -- but its fixed reference had to move, because the
    manifest itself moved for the first time since Phase 33.1's close.

    RE-POINTED AGAIN 2026-09-23 by Plan 33.2-25 Task 4, for the same reason: the
    owner-accepted batched swap (SPEC R13) installed the corrected artifacts, so the
    recorded END STATE is now ``P332_25B_SWAP_ARTIFACT_IDS``. Phase 33's
    ``POST_GATE_ARTIFACT_MANIFEST`` is retained unedited as the record of what it installed.

    RE-POINTED AGAIN 2026-10-04 for WINDOWS row 19 (quick task 261003-vke): the swap of the
    re-fits on the neutral-site Elo gold installed ``ROW19_SWAP_ARTIFACT_IDS``, now the
    recorded END STATE; ``P332_25B_SWAP_ARTIFACT_IDS`` stays as the 2026-09-23 record.
    """

    def test_the_live_manifest_holds_the_POST_PROMOTION_pointers(self) -> None:
        """Reading it changed nothing, and the recorded END STATE is what is live."""
        manifest = _live_manifest()
        for target, artifact_id in phase33_state.ROW19_SWAP_ARTIFACT_IDS:
            assert manifest[target] == artifact_id, target

    def test_the_phase_331_close_record_is_RETAINED_unedited(self) -> None:
        """A superseded record is preserved, never rewritten to agree with today.

        ``DEPLOYED_POINTERS_AT_PHASE_331_CLOSE`` is the fixed half of the flip predicate's
        TRIGGER -- "a pointer differs from what Phase 33.1 left" -- and editing it to
        match the post-promotion manifest would make the trigger permanently unable to
        fire, which is the one edit the bridge machinery must never accept.
        """
        recorded = dict(phase33_state.DEPLOYED_POINTERS_AT_PHASE_331_CLOSE)

        assert recorded == {
            "wp": "wp_20260824_113325",
            "ats": "ats_20260605_220128",
            "ou": "ou_20260326_163930",
        }
        live = _live_manifest()
        moved = [t for t, old in recorded.items() if live[t] != old]
        assert sorted(moved) == ["ats", "ou", "wp"], moved
