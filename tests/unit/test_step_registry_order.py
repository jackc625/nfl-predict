"""The cache-population step is registered NON-CRITICAL and strictly AFTER its dependencies.

WHAT THIS PINS, AND WHY IT IS PINNED BY INDEX
----------------------------------------------
SPEC R9 / D31-29 require three separate things of the Friday cache step, and each of them is a
property of the REGISTRY rather than of the step body:

1. **Adjacency.** The step must run strictly AFTER ``generate_recommendations`` (which writes the
   durable bet-list artifact) and AFTER ``export_artifacts``, so the blob the population reads is
   the one just written. Asserted by comparing INDICES in the built registry, never by looking a
   name up and trusting that the list is in the order the source file appears to be in. A future
   insertion that moves the cache step above either dependency turns this red.

2. **Non-criticality.** A cache failure must degrade the run, not fail it: discarding good
   prediction output because a downstream convenience failed is the outcome R9 explicitly refuses.
   Asserted directly on the built ``StepDefinition``.

3. **Phase.** The step belongs to ``PipelinePhase.PREDICTIONS``, so ``--predictions-only`` includes
   it. A DATA-phase registration would run it before any prediction existed.

WHY THE ORDER CHECK IS A FUNCTION, NOT AN INLINE ASSERT
--------------------------------------------------------
``order_violations`` is exposed so the guard can be shown to FAIL. A check that has never been
observed failing is a check nobody knows the shape of, and this repository has its own case of a
guard that passed while the real behaviour differed. ``test_the_order_check_reports_a_violation_...``
feeds it a deliberately reordered copy of the real registry and asserts it reports the violation --
a permanent, committed demonstration rather than a one-time experiment recorded in prose.

WHAT THIS DELIBERATELY DOES NOT CLAIM
--------------------------------------
Registering the step non-critical routes its failure to ``alert_degraded_completion``, and
``pipeline/alert.py`` documents alerts as LOG-ONLY by default -- the email and messaging channels
are empirically inert. So the alert is NOT the protection against a silently stale bet list. The
protection is the ``/bets`` hard-block, which a reader cannot miss. That is asserted in
``tests/api/test_bets_page.py``, not here.

Selectors (``-k``): registered, order, violation, non_critical, phase, unique, not_vacuous.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pipeline.steps import (
    PipelinePhase,
    StepDefinition,
    build_step_registry,
)

# ---------------------------------------------------------------------------
# The names this guard reasons about
# ---------------------------------------------------------------------------

# The step Plan 31-18 registers. Named as a constant so a rename is a deliberate edit here rather
# than a silent no-op in a string comparison.
CACHE_STEP = "populate_web_cache"

# The two steps whose output the cache step reads. ``generate_recommendations`` writes the durable
# bet-list artifact; ``export_artifacts`` writes the prediction JSON. Both must precede it.
CACHE_STEP_DEPENDENCIES: tuple[str, ...] = (
    "generate_recommendations",
    "export_artifacts",
)

# The step Phase 33 Plan 33-07 registers, and the step it must precede. Same
# constant-not-literal discipline as the pair above, for the same reason.
CAPTURE_STEP = "capture_live_season"

# ``ingest_games`` reads the season nflverse is currently serving. The capture records what
# nflverse served AT THIS INSTANT, so it has to happen BEFORE the ingest consumes it --
# a capture taken afterwards is a record of a slightly different upstream than the one the
# run actually used, which is the one thing a capture exists to make impossible.
CAPTURE_STEP_SUCCESSORS: tuple[str, ...] = ("ingest_games",)


def _index_by_name(registry: list[StepDefinition]) -> dict[str, int]:
    """Position of every step in the built registry, keyed by name."""
    return {step.name: position for position, step in enumerate(registry)}


def order_violations(registry: list[StepDefinition]) -> list[str]:
    """Every adjacency rule *registry* breaks, as human-readable strings.

    Empty means the cache step sits strictly after BOTH dependencies. The comparison is on INDEX,
    so a reordering is caught even though every name is still present -- which is the failure mode
    a name-membership check cannot see.
    """
    positions = _index_by_name(registry)

    if CACHE_STEP not in positions:
        return [f"{CACHE_STEP!r} is absent from the registry entirely"]

    absent = [name for name in CACHE_STEP_DEPENDENCIES if name not in positions]
    if absent:
        return [f"dependency {name!r} is absent from the registry" for name in absent]

    return [
        (
            f"{CACHE_STEP!r} is at index {positions[CACHE_STEP]} but must run strictly AFTER "
            f"{name!r} at index {positions[name]}; the population would read a blob written by "
            "the PREVIOUS run"
        )
        for name in CACHE_STEP_DEPENDENCIES
        if positions[CACHE_STEP] <= positions[name]
    ]


def capture_order_violations(registry: list[StepDefinition]) -> list[str]:
    """Every adjacency rule *registry* breaks around the live-season capture step.

    The same INDEX-comparison shape ``order_violations`` uses, and exposed for the same
    reason: the two fail-closed controls below feed it a deliberately broken copy of the
    REAL registry and assert it reports the break. A guard nobody has watched fail is a
    guard whose shape nobody knows.
    """
    positions = _index_by_name(registry)

    if CAPTURE_STEP not in positions:
        return [f"{CAPTURE_STEP!r} is absent from the registry entirely"]

    absent = [name for name in CAPTURE_STEP_SUCCESSORS if name not in positions]
    if absent:
        return [f"successor {name!r} is absent from the registry" for name in absent]

    return [
        (
            f"{CAPTURE_STEP!r} is at index {positions[CAPTURE_STEP]} but must run strictly "
            f"BEFORE {name!r} at index {positions[name]}; a capture taken after the ingest "
            "records a different upstream than the one the run consumed"
        )
        for name in CAPTURE_STEP_SUCCESSORS
        if positions[CAPTURE_STEP] >= positions[name]
    ]


def _capture_step(registry: list[StepDefinition]) -> StepDefinition:
    """The registered capture step, or an assertion failure naming what was found instead."""
    matches = [step for step in registry if step.name == CAPTURE_STEP]
    assert len(matches) == 1, (
        f"expected exactly one {CAPTURE_STEP!r} entry in the registry, found "
        f"{len(matches)}. Registered names: {[step.name for step in registry]}"
    )
    return matches[0]


def _cache_step(registry: list[StepDefinition]) -> StepDefinition:
    """The registered cache step, or an assertion failure naming what was found instead."""
    matches = [step for step in registry if step.name == CACHE_STEP]
    assert len(matches) == 1, (
        f"expected exactly one {CACHE_STEP!r} entry in the registry, found {len(matches)}. "
        f"Registered names: {[step.name for step in registry]}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_cache_step_is_registered() -> None:
    """A cache-population step exists in the built registry (SPEC R9 boundary)."""
    registry = build_step_registry()
    step = _cache_step(registry)
    assert callable(step.callable)
    assert step.description, (
        "the cache step carries no description for the dry-run listing"
    )


def test_the_registry_walk_is_not_vacuous() -> None:
    """Both dependencies are really registered, so the order assertion has something to compare.

    Without this, a registry that lost ``export_artifacts`` entirely would make the ordering
    comparison trivially satisfiable rather than false.
    """
    positions = _index_by_name(build_step_registry())
    for name in CACHE_STEP_DEPENDENCIES:
        assert name in positions, (
            f"dependency {name!r} is not in the registry; the adjacency guard would be comparing "
            "against nothing"
        )


def test_every_registered_step_name_is_unique() -> None:
    """A duplicate name would make the index lookup ambiguous and the ordering claim meaningless."""
    names = [step.name for step in build_step_registry()]
    assert len(names) == len(set(names)), f"duplicate step name(s) in registry: {names}"


# ---------------------------------------------------------------------------
# Adjacency, by INDEX
# ---------------------------------------------------------------------------


def test_the_cache_step_runs_strictly_after_both_of_its_dependencies() -> None:
    """SPEC R9 adjacency: the blob the cache reads is the one this run just wrote."""
    violations = order_violations(build_step_registry())
    assert not violations, "\n".join(violations)


def test_the_order_check_reports_a_violation_when_the_cache_step_is_moved_before_a_dependency() -> (
    None
):
    """The guard is shown to be able to FAIL, on a reordered copy of the REAL registry.

    A check that has never been seen failing is a check whose shape nobody knows. This feeds
    ``order_violations`` a registry in which the cache step has been lifted to the front and
    asserts it reports BOTH dependencies -- so a future insertion that reorders the real registry
    cannot pass silently.
    """
    registry = build_step_registry()
    step = _cache_step(registry)
    reordered = [step] + [entry for entry in registry if entry.name != CACHE_STEP]

    violations = order_violations(reordered)
    assert len(violations) == len(CACHE_STEP_DEPENDENCIES), (
        f"a registry with the cache step moved to the front reported {len(violations)} "
        f"violation(s); expected one per dependency: {violations}"
    )
    for name in CACHE_STEP_DEPENDENCIES:
        assert any(repr(name) in message for message in violations), (
            f"the reordered registry did not report {name!r} as violated: {violations}"
        )


def test_the_order_check_reports_the_cache_step_going_missing() -> None:
    """Deleting the step is reported as an absence rather than passing as 'no violations'."""
    registry = [step for step in build_step_registry() if step.name != CACHE_STEP]
    violations = order_violations(registry)
    assert violations == [f"{CACHE_STEP!r} is absent from the registry entirely"], (
        violations
    )


# ---------------------------------------------------------------------------
# Non-criticality and phase
# ---------------------------------------------------------------------------


def test_the_cache_step_is_registered_non_critical() -> None:
    """SPEC R9 boundary: a cache failure degrades the run; it never discards good predictions."""
    step = _cache_step(build_step_registry())
    assert step.critical is False, (
        "the cache step is registered critical -- a cache failure would abort a run whose "
        "prediction work already succeeded, which is the outcome SPEC R9 explicitly refuses"
    )


def test_the_cache_step_is_not_retryable() -> None:
    """Retry is reserved for the three network-touching ingest steps.

    A cache rebuild is local work: retrying it would repeat a multi-second rebuild on a failure
    that a retry cannot fix, and ``TRANSIENT_EXCEPTIONS`` includes ``OSError``, which is exactly
    the class a genuinely broken cache path raises.
    """
    step = _cache_step(build_step_registry())
    assert step.retryable is False


def test_the_cache_step_belongs_to_the_predictions_phase() -> None:
    """A DATA-phase registration would run the population before any prediction existed."""
    step = _cache_step(build_step_registry())
    assert step.phase is PipelinePhase.PREDICTIONS


# ---------------------------------------------------------------------------
# The live-season capture step (Phase 33, Plan 33-07; discharges D32-04)
# ---------------------------------------------------------------------------


def test_the_capture_step_is_registered_first() -> None:
    """It is step 0 of the whole registry, not merely somewhere in the DATA phase.

    Asserted on index 0 rather than on "before ingest_games" alone, because the capture is
    the record of what upstream served THIS RUN and anything that reads upstream before it
    has been captured is unattributable.
    """
    registry = build_step_registry()
    assert registry[0].name == CAPTURE_STEP, [step.name for step in registry[:3]]


def test_the_capture_step_runs_strictly_before_the_ingest_it_records() -> None:
    violations = capture_order_violations(build_step_registry())
    assert not violations, "\n".join(violations)


def test_the_capture_order_check_reports_a_violation_when_it_is_moved_after_the_ingest() -> (
    None
):
    """The guard is shown to be able to FAIL, on a reordered copy of the REAL registry."""
    registry = build_step_registry()
    step = _capture_step(registry)
    without = [entry for entry in registry if entry.name != CAPTURE_STEP]
    reordered = [*without, step]

    violations = capture_order_violations(reordered)
    assert len(violations) == len(CAPTURE_STEP_SUCCESSORS), violations
    for name in CAPTURE_STEP_SUCCESSORS:
        assert any(repr(name) in message for message in violations), violations


def test_the_capture_order_check_reports_the_capture_step_going_missing() -> None:
    """Deleting the step is reported as an absence rather than passing as 'no violations'."""
    registry = [step for step in build_step_registry() if step.name != CAPTURE_STEP]
    violations = capture_order_violations(registry)
    assert violations == [f"{CAPTURE_STEP!r} is absent from the registry entirely"], (
        violations
    )


def test_the_capture_step_belongs_to_the_data_phase() -> None:
    """A PREDICTIONS registration would capture after the data it is meant to attest to."""
    step = _capture_step(build_step_registry())
    assert step.phase is PipelinePhase.DATA


def test_the_capture_step_is_critical_and_retryable() -> None:
    """T-33-36: a capture failure must STOP the run, and a network blip must not cause one.

    CRITICAL because a failed capture means the live zone has no row for this week, and
    every downstream DATA step would then run against last week's zone with nothing saying
    so -- availability traded for a silent unattributable run, which is the trade this
    phase exists to refuse.

    RETRYABLE for the same reason the three ingest steps are: it fetches over the network,
    and ``TRANSIENT_EXCEPTIONS`` is exactly the class a dropped connection raises. The
    append-only manifest makes a retry safe -- a second capture of the same week is a new
    sequence entry, not a rewrite.
    """
    step = _capture_step(build_step_registry())
    assert step.critical is True
    assert step.retryable is True
    assert step.max_retries == 3


def test_the_non_critical_set_is_exactly_the_declared_four() -> None:
    """The non-critical registrations are pinned as a SET, so a sixth cannot appear unnoticed.

    ``critical=False`` is a licence to fail quietly. Three steps carried it before Plan 31-18
    (weather ingest, weather features, output verification); the cache step is the fourth.

    ``build_market_anchors`` is the FIFTH, added under WR-13 and DECLARED here rather than
    inherited by accident, which is what this test exists to force. It was ``critical=True``, so a
    raise inside ``build_market_anchor_features`` aborted the entire Friday run -- including the
    prediction and bet-list work after it -- and what it writes, silver
    ``market_anchor_features``, is read by no production code (``AUTOMATION.md``): the gold build
    calls ``MarketAnchorFeaturesCalculator.build_features`` directly. Killing a run over a table
    nothing consumes is the wrong direction.

    If that step ever gains a consumer, its criticality has to be revisited with it.

    STILL FIVE AFTER PHASE 33 PLAN 33-07, and the three steps that plan added are declared
    here as absences rather than left to be inferred: ``capture_live_season``,
    ``verify_gold_currency`` and ``verify_prediction_currency`` are all ``critical=True``.
    Each one exists to STOP a run -- an unattributable upstream, a stale gold matrix, a
    prediction file whose rows are last week's -- and a gate registered non-critical is a
    gate that logs a warning while the run publishes anyway.

    FOUR AFTER PLAN 33-18. ``build_weather_features`` LEFT the set on owner ruling W1 of
    2026-09-15. As a non-critical step its refusal degraded the run and ``build_features``
    rebuilt gold from a weather-features table that did not cover the week being predicted;
    ``tests/integration/test_weather_features_refusal_stops_pipeline.py`` pins that the
    refusal now stops the run before gold is built.
    """
    non_critical = {step.name for step in build_step_registry() if not step.critical}
    assert non_critical == {
        "ingest_weather",
        "verify_output_files",
        "build_market_anchors",
        CACHE_STEP,
    }, f"the non-critical set moved: {sorted(non_critical)}"
