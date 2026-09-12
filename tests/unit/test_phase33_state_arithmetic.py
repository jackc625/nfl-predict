"""The collected-count arithmetic closes, so a quietly deleted test is a detectable event.

WHAT THIS DEFENDS AGAINST (T-33-08)
-----------------------------------
A phase that adds tests makes the suite's summary line move. If the only record of the
move is the line itself, then deleting an inconvenient test and adding a convenient one
leaves the line unchanged and the deletion invisible. The defence is ARITHMETIC over a
COLLECTED count:

    PRE_PHASE_COLLECTED + sum(per-plan counters) == POST_PHASE_COLLECTED

Every term is recorded independently by the plan that measured it, so the equation can
only close if nothing went missing. That is the whole point: the sum is not a summary of
the line, it is an independent claim the line has to agree with.

THE COUNTERS COUNT COLLECTED NODES
----------------------------------
``PER_PLAN_TEST_COUNT_SLOTS`` names eighteen immutable per-plan slots and its comment
states the unit: a ``@pytest.mark.parametrize`` case counts ONCE PER GENERATED NODE,
because the arithmetic is over collection and pytest collects one node per case. A plan
that counted test FUNCTIONS instead would under-report by the parametrize multiplier and
the equation would fail to close for a reason that has nothing to do with a deleted test
(T-33-08c).

THE AGGREGATE IS CHECKED AGAINST ITS PARTS, NOT TRUSTED
-------------------------------------------------------
``TESTS_ADDED_BY_PHASE_33`` is appended ONCE by Plan 33-18 at closure. It is asserted to
equal the sum of the eighteen per-plan slots, so it is a derived number with a check
rather than a nineteenth independent claim nobody reconciles.

THE POST-PHASE HALF SKIPS UNTIL THE SLOTS EXIST, AND THAT SKIP IS ITSELF CONTROLLED
----------------------------------------------------------------------------------
Plans 33-03 .. 33-18 have not run yet. The post-phase test therefore skips with a pinned
message until every slot is present -- and a companion test drives the same resolver and
asserts the skip happens, so the guard is known to be wired up rather than merely never
observed.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re

import pytest

from tests import phase32_state, phase33_state

# `5 failed, 4001 passed, 9 skipped, 14 xfailed` -> [("5", "failed"), ...]. The category
# word is captured rather than assumed positionally, because pytest omits categories it
# has none of (the api tier's line is a bare `367 passed`) and a positional parse would
# silently mis-assign them.
_COUNT_RE = re.compile(r"(\d+)\s+([a-z]+)")

_EXPECTED_SLOT_NAMES = tuple(f"TESTS_ADDED_33_{index:02d}" for index in range(1, 19))

POST_PHASE_PENDING_SKIP = (
    "the post-phase collected-count arithmetic cannot be checked yet: "
    "tests/phase33_state.py does not carry every per-plan slot named in "
    "PER_PLAN_TEST_COUNT_SLOTS and POST_PHASE_COLLECTED, both of which are appended by "
    "the plans that measure them (33-03 .. 33-18). This is a control that did NOT run "
    "on this checkout, not a control that passed."
)


def parse_counts(summary_line: str) -> dict[str, int]:
    """Parse a pytest summary line into ``{category: count}``.

    Args:
        summary_line: A line such as ``"5 failed, 4001 passed, 9 skipped, 14 xfailed"``.

    Returns:
        The category counts, with absent categories simply absent.
    """
    return {category: int(count) for count, category in _COUNT_RE.findall(summary_line)}


def _slots_present() -> list[str]:
    """The per-plan slot names that exist on ``tests.phase33_state`` right now."""
    return [
        name
        for name in phase33_state.PER_PLAN_TEST_COUNT_SLOTS
        if hasattr(phase33_state, name)
    ]


def _resolve_post_phase_or_skip() -> tuple[dict[str, int], int]:
    """Return the per-plan counters and POST_PHASE_COLLECTED, or skip with a pinned message.

    Returns:
        ``({slot_name: count}, post_phase_collected)``.

    Raises:
        Skipped: when any per-plan slot, or POST_PHASE_COLLECTED, is not yet appended.
    """
    present = _slots_present()
    if len(present) != len(phase33_state.PER_PLAN_TEST_COUNT_SLOTS) or not hasattr(
        phase33_state, "POST_PHASE_COLLECTED"
    ):
        pytest.skip(POST_PHASE_PENDING_SKIP)
    return (
        {name: int(getattr(phase33_state, name)) for name in present},
        int(phase33_state.POST_PHASE_COLLECTED),
    )


# ---------------------------------------------------------------------------
# The pre-phase line, checked against itself.
# ---------------------------------------------------------------------------


def test_the_pre_phase_failure_set_sums_to_the_collected_count() -> None:
    """5 + 4001 + 9 + 14 == 4029, asserted rather than left to the reader.

    A collected count that does not close against the failure set means a tier was
    dropped from the sum -- which is exactly how a three-tier measurement goes wrong,
    and exactly the kind of error that is invisible in prose.
    """
    counts = parse_counts(phase33_state.PRE_PHASE_FAILURE_SET)
    assert len(counts) == 4, (
        f"PRE_PHASE_FAILURE_SET parsed into {counts!r}; four categories were expected "
        f"from {phase33_state.PRE_PHASE_FAILURE_SET!r}."
    )
    assert sum(counts.values()) == phase33_state.PRE_PHASE_COLLECTED, (
        f"PRE_PHASE_FAILURE_SET sums to {sum(counts.values())} but "
        f"PRE_PHASE_COLLECTED is {phase33_state.PRE_PHASE_COLLECTED}. Parsed: {counts!r}."
    )


def test_the_three_tier_lines_sum_category_by_category_to_the_failure_set() -> None:
    """The instrument is part of the measurement, so the split must reconcile.

    ``PRE_PHASE_FAILURE_SET`` is the SUM of three separately-run tiers. Summing them
    category by category -- rather than only checking the grand total -- catches a tier
    line that was transcribed with the right total and the wrong distribution.
    """
    tier_counts = [parse_counts(line) for line in phase33_state.PRE_PHASE_TIER_LINES]
    assert len(tier_counts) == 3, phase33_state.PRE_PHASE_TIER_LINES

    summed: dict[str, int] = {}
    for counts in tier_counts:
        for category, count in counts.items():
            summed[category] = summed.get(category, 0) + count

    expected = parse_counts(phase33_state.PRE_PHASE_FAILURE_SET)
    assert summed == expected, (
        "the three tier lines do not sum to PRE_PHASE_FAILURE_SET category by "
        f"category.\n  tiers sum to: {summed!r}\n  failure set: {expected!r}\n"
        f"  tier lines:  {phase33_state.PRE_PHASE_TIER_LINES!r}"
    )


def test_the_phase33_baseline_is_not_inherited_from_phase32() -> None:
    """Phase 33 MEASURED its own baseline; it did not adopt the last number written down.

    Commit ``6f6af82`` added 66 tests AFTER ``tests/phase32_state.py``'s final slot was
    appended -- Phase 32's own Nyquist-validation additions. Inheriting the 3,963 would
    have been asserting against the last number somebody recorded rather than against
    the suite that exists, which is the exact failure these manifests are built to stop.
    """
    assert phase33_state.PRE_PHASE_FAILURE_SET != phase32_state.PRE_PHASE_FAILURE_SET, (
        "tests/phase33_state.PRE_PHASE_FAILURE_SET is byte-identical to Phase 32's. "
        "Commit 6f6af82 added 66 tests after phase32_state's final slot was appended, "
        "so an identical line means Phase 33's baseline was copied rather than "
        "measured."
    )
    assert (
        sum(parse_counts(phase32_state.PRE_PHASE_FAILURE_SET).values())
        < phase33_state.PRE_PHASE_COLLECTED
    )


# ---------------------------------------------------------------------------
# The per-plan slot protocol.
# ---------------------------------------------------------------------------


def test_the_per_plan_slots_name_the_eighteen_plans_in_order() -> None:
    """One immutable slot per plan, named for its plan, in plan order.

    The names are the contract: Plan 33-07 appends ``TESTS_ADDED_33_07`` and nothing
    else, so an append lands in exactly one place and a reader can tell at a glance
    which plans have reported.
    """
    assert phase33_state.PER_PLAN_TEST_COUNT_SLOTS == _EXPECTED_SLOT_NAMES, (
        "PER_PLAN_TEST_COUNT_SLOTS does not name the eighteen per-plan counters in "
        f"plan order.\n  got:      {phase33_state.PER_PLAN_TEST_COUNT_SLOTS!r}\n"
        f"  expected: {_EXPECTED_SLOT_NAMES!r}"
    )
    assert len(set(phase33_state.PER_PLAN_TEST_COUNT_SLOTS)) == 18


def test_the_slots_that_exist_so_far_are_a_contiguous_prefix() -> None:
    """Plans land in order, so the slots present must be 33-01 .. 33-NN with no holes.

    A hole means a plan finished without recording its count -- the term that would
    later be missing from the closing sum, discovered at phase close instead of at the
    plan that dropped it.
    """
    present = _slots_present()
    expected_prefix = list(phase33_state.PER_PLAN_TEST_COUNT_SLOTS[: len(present)])
    assert present == expected_prefix, (
        "the per-plan counters present on tests/phase33_state.py are not a contiguous "
        f"prefix of PER_PLAN_TEST_COUNT_SLOTS.\n  present:  {present!r}\n"
        f"  expected: {expected_prefix!r}\nA hole means a plan finished without "
        "recording its collected-node count."
    )
    assert present, (
        "no per-plan counter exists at all. Plan 33-01 appended TESTS_ADDED_33_01, so "
        "an empty list means the manifest lost a slot."
    )


def test_the_closing_aggregate_is_not_seeded_before_closure() -> None:
    """``TESTS_ADDED_BY_PHASE_33`` is Plan 33-18's slot, appended ONCE, at closure.

    Seeding it at zero and having each plan increment it would edit one slot eighteen
    times -- in the phase whose own audit record that append-once protocol IS. The
    protocol cannot be the first thing the phase breaks.

    This assertion is deliberately CONDITIONAL on the slots: it forbids the aggregate
    only while the eighteen parts are incomplete. Once Plan 33-18 has appended every
    slot, the aggregate is expected and is checked against the parts below.
    """
    if len(_slots_present()) == len(phase33_state.PER_PLAN_TEST_COUNT_SLOTS):
        return
    assert not hasattr(phase33_state, "TESTS_ADDED_BY_PHASE_33"), (
        "TESTS_ADDED_BY_PHASE_33 exists while the eighteen per-plan slots are still "
        f"incomplete (present: {_slots_present()!r}). That aggregate is appended ONCE "
        "by Plan 33-18 at closure; a value written earlier can only become correct by "
        "being edited, which is the append-once violation this manifest exists to "
        "demonstrate against."
    )


# ---------------------------------------------------------------------------
# The post-phase half -- pending until Plan 33-18, with its skip controlled.
# ---------------------------------------------------------------------------


def test_the_post_phase_collected_count_closes_against_the_per_plan_sum() -> None:
    """T-33-08: PRE_PHASE_COLLECTED + sum(per-plan counters) == POST_PHASE_COLLECTED.

    Reads the per-plan slots BY NAME from ``PER_PLAN_TEST_COUNT_SLOTS`` rather than a
    running aggregate, and separately asserts the closing aggregate equals the sum of
    its parts -- so the aggregate is checked rather than trusted.
    """
    counters, post_phase_collected = _resolve_post_phase_or_skip()
    total_added = sum(counters.values())

    assert phase33_state.PRE_PHASE_COLLECTED + total_added == post_phase_collected, (
        "the Phase-33 collected-count arithmetic does NOT close.\n"
        f"  PRE_PHASE_COLLECTED:   {phase33_state.PRE_PHASE_COLLECTED}\n"
        f"  sum(per-plan slots):   {total_added}  ({counters!r})\n"
        f"  POST_PHASE_COLLECTED:  {post_phase_collected}\n"
        "A shortfall means tests that existed before the phase are no longer collected "
        "-- deleted, renamed out of collection, or lost to a collection error -- and "
        "the summary line absorbed it silently."
    )

    assert hasattr(phase33_state, "TESTS_ADDED_BY_PHASE_33"), (
        "every per-plan slot is present but TESTS_ADDED_BY_PHASE_33 is not. Plan 33-18 "
        "appends the sum ONCE, at closure."
    )
    assert int(phase33_state.TESTS_ADDED_BY_PHASE_33) == total_added, (
        f"TESTS_ADDED_BY_PHASE_33 is {phase33_state.TESTS_ADDED_BY_PHASE_33} but the "
        f"eighteen per-plan slots sum to {total_added}. The aggregate is a DERIVED "
        "number and must agree with its parts; a disagreement is reported, never "
        "resolved by editing whichever one looks wrong."
    )


def test_the_pending_skip_guard_actually_fires() -> None:
    """Fail-closed control on the resolver: the skip is issued, with the pinned message.

    A guard that has only ever been observed NOT firing is indistinguishable from a
    guard that is not wired up. While Plans 33-03 .. 33-18 have not run, this drives the
    real resolver and asserts it steps aside by NAME rather than passing quietly.

    Once the phase closes this control becomes vacuous by design, so it asserts the
    complementary fact instead: the resolver returns a complete set of slots.
    """
    if len(_slots_present()) == len(
        phase33_state.PER_PLAN_TEST_COUNT_SLOTS
    ) and hasattr(phase33_state, "POST_PHASE_COLLECTED"):
        counters, _collected = _resolve_post_phase_or_skip()
        assert len(counters) == 18
        return

    with pytest.raises(pytest.skip.Exception) as excinfo:
        _resolve_post_phase_or_skip()
    assert str(excinfo.value) == POST_PHASE_PENDING_SKIP
