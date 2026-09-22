"""Gold-width tripwire parity test (IN-03, Plan 28-06 / 29-06 / 30-07).

``scripts.data_qa.GOLD_FEATURE_MATRICES`` hardcodes the expected column count of each
gold feature matrix as a deliberate audit tripwire: any stray/dropped column trips it
rather than passing silently. This test asserts the tripwire is itself HONEST -- the
declared expected width equals the real on-disk column count of each rebuilt matrix.

If a future builder legitimately adds or removes a column, this test fails until the
operator UPDATES ``GOLD_FEATURE_MATRICES`` to the new empirically-counted width
(Pitfall 6 -- never silence the tripwire by widening tolerances).

The second test is the reason the first is not enough. A width tripwire on its own
says only that the total moved by N; it cannot say WHICH N columns. Phase 29 pinned
its +15 widening to a named family here; Phase 30 rung 3 (SPEC R3, D29-07-01) removes
exactly that family again, so the same fifteen names now pin a NARROWING and the
tripwire returns to the Phase-28 widths. The direction inverted; the discipline did
not.

The named list is cross-checked against ``backtest.signal_lift.group_columns`` -- the
ONE group registry the screen, the drop and the rung-3 attribution all read (D30-02)
-- so a hand-written list here cannot silently drift away from the predicate that
actually did the removing.

PHASE 33.1 (Plan 33.1-07) adds ONE column on top of the Phase-28 width: the weather
coverage flag. The discipline is unchanged and the direction inverted again -- a
WIDENING this time -- so the Phase-30 residual assertion moved from ``== 0`` to the
length of a NAMED tuple, with the reason recorded in that test's docstring rather
than the assertion simply being relaxed. ``PHASE_28_WIDTHS`` and
``PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS`` are left exactly as they are: both are
historical records of what earlier phases did, and a historical record that gets
edited every time the present changes is not a record.

PHASE 33.2 (Plan 33.2-12, owner of the gold-width pin protocol) moves widths at four ladder
rungs: rung 4 removes the two forecast-less weather columns, rungs 7 and 8 add coverage
flags, rung 9 removes the market columns. Rather than re-pinning these tests at every rung,
each width-moving rung appends ONE ``P332_*_GOLD_WIDTH_DELTA`` slot to the append-once
manifest (``tests/phase33_state.py``) naming the columns it added and removed and the widths
it MEASURED before and after, and re-pins ``GOLD_FEATURE_MATRICES`` to its measured widths.
The resolvers below discover those slots by NAME -- the suffix is the discovery contract --
so the residual assertions follow the manifest and a later rung edits no test. The chain
test pins the arithmetic and the names; ``PHASE_28_WIDTHS``,
``PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS``, ``PHASE_331_ADDED_WEATHER_COLUMNS`` and
``test_the_phase33_ladder_moved_no_width`` stay byte-unchanged as the records they are.
"""

from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import group_columns
from features.weather import WEATHER_COVERAGE_COLUMN
from scripts.data_qa import GOLD_FEATURE_MATRICES
from tests import phase33_state

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"

# The Phase-29 line-movement family (Plan 29-06 / SIG-04): the seven D-09 totals
# features, the shared coverage flag, and the seven Tier (a) spread siblings -- 15
# game-level columns per matrix. Phase 29 added them; Plan 30-07 removes them.
PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS = (
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    "line_movement_coverage",
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)

# The Phase-28 widths Phase 29 widened FROM and Phase 30 rung 3 returns TO.
PHASE_28_WIDTHS = {"features_wp": 194, "features_ats": 195, "features_ou": 194}

# The ONE column Phase 33.1 adds on top of the Phase-28 width (Plan 33.1-07).
#
# NAMED, NOT COUNTED. The Phase-30 residual assertion below used to read
# `== 0`, which was exactly right while the tripwire had returned to the
# Phase-28 width and nothing else had moved it. Phase 33.1's rung 1 added the
# weather coverage flag, so the residual is now +1 BY DECISION rather than by
# accident -- and the difference between those two is the whole reason this
# tuple exists instead of the integer 1.
#
# The flag is imported from the ONE module that owns the name rather than
# spelled here, so a rename cannot leave this file pinning a column that no
# longer exists while the integer still matches.
PHASE_331_ADDED_WEATHER_COLUMNS = (WEATHER_COVERAGE_COLUMN,)

# The market columns that must survive the LINE-MOVEMENT drop. ``snapshot_total`` /
# ``snapshot_spread`` are the freeze anchors and ``total_movement`` / ``spread_movement``
# are the pre-existing MarketAnchor columns; all four are near-misses for a careless
# substring prune, which is what the node below still guards against.
#
# THEY ARE NO LONGER BASELINE FEATURES IN GOLD (Plan 33.2-19, p332_ rung 9, D33.2-03).
# They left all three matrices with the rest of the betting line, through the ``market``
# group -- a DIFFERENT family from line_movement, and the point of the node below is that
# the two predicates are told apart: line_movement matches none of these four and market
# matches all of them, so their removal is attributable to the market group alone.
MARKET_SURVIVORS = (
    "snapshot_total",
    "snapshot_spread",
    "total_movement",
    "spread_movement",
)

MATRIX_ORDER = ("features_wp", "features_ats", "features_ou")


# ---------------------------------------------------------------------------
# THE PHASE-33.2 WIDTH-DELTA RESOLVERS (Plan 33.2-12, <owned_protocol_gold_width_pin> P4).
# ---------------------------------------------------------------------------


def ladder_order(rung: int | str) -> tuple[int, str]:
    """Sort key for a `p332_` ladder id: a numbered rung, then the steps that follow it.

    The ladder's EXTRA STEPS carry string ids (``3b``, ``7b``, ``8c``, ``8e``) and its
    numbered rungs carry ints, so ``sorted(slots, key=lambda s: s["rung"])`` -- what this
    resolver used before p332_ extra step 8e -- raises ``TypeError`` the moment an extra
    step moves a width, as step 8e does (-8). Splitting the id into its leading integer
    and its suffix orders 4, 7, 8, "8e", 9 exactly as the ladder runs them, which is the
    order the chain arithmetic below depends on. A slot with an unparseable id is a
    manifest error and raises rather than sorting somewhere arbitrary.
    """
    text = str(rung)
    digits = ""
    for character in text:
        if not character.isdigit():
            break
        digits += character
    if not digits:
        msg = f"ladder id {rung!r} does not start with a rung number"
        raise ValueError(msg)
    return (int(digits), text[len(digits) :])


def phase332_width_deltas() -> list[dict]:
    """Every ``P332_*_GOLD_WIDTH_DELTA`` slot in the manifest, in ladder order."""
    slots = [
        getattr(phase33_state, name)
        for name in dir(phase33_state)
        if name.startswith("P332_") and name.endswith("_GOLD_WIDTH_DELTA")
    ]
    return sorted(slots, key=lambda slot: ladder_order(slot["rung"]))


def current_ladder_widths() -> tuple[int, int, int]:
    """The widths the ladder has reached: the last slot's ``widths_after``.

    ``phase33_state.GOLD_WIDTHS_AFTER_ELO_REBUILD`` -- the Phase-33 ladder's close -- when no
    Phase-33.2 rung has moved a width yet.
    """
    slots = phase332_width_deltas()
    if not slots:
        return tuple(phase33_state.GOLD_WIDTHS_AFTER_ELO_REBUILD)
    return tuple(slots[-1]["widths_after"])


def phase332_net_width_delta() -> int:
    """The net column change of every Phase-33.2 rung (identical in all three matrices)."""
    return sum(len(s["added"]) - len(s["removed"]) for s in phase332_width_deltas())


def width_chain_violations(
    slots: list[dict], baseline: tuple[int, int, int]
) -> list[str]:
    """Every arithmetic break in a chain of width-delta slots; empty when it holds.

    The first slot's ``widths_before`` must equal *baseline*, each later slot's
    ``widths_before`` its predecessor's ``widths_after``, and each slot's ``widths_after``
    its ``widths_before`` plus ``len(added)`` minus ``len(removed)`` in every matrix.
    """
    violations: list[str] = []
    expected_before = tuple(baseline)
    for slot in slots:
        before, after = tuple(slot["widths_before"]), tuple(slot["widths_after"])
        if before != expected_before:
            violations.append(
                f"rung {slot['rung']}: widths_before {before} != {expected_before}"
            )
        delta = len(slot["added"]) - len(slot["removed"])
        if after != tuple(width + delta for width in before):
            violations.append(
                f"rung {slot['rung']}: widths_after {after} != {before} + {delta}"
            )
        expected_before = after
    return violations


@pytest.mark.parametrize(
    ("table_name", "expected_width"), list(GOLD_FEATURE_MATRICES.items())
)
def test_gold_matrix_width_matches_tripwire(
    table_name: str, expected_width: int
) -> None:
    """Each rebuilt gold matrix's real column count equals its tripwire value."""
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    actual_width = pd.read_parquet(path).shape[1]
    assert actual_width == expected_width, (
        f"{table_name}: real column count {actual_width} != "
        f"GOLD_FEATURE_MATRICES expected {expected_width}. If the change is "
        f"intentional, update GOLD_FEATURE_MATRICES in scripts/data_qa.py to the "
        f"new empirically-counted width (IN-03, never silence the tripwire)."
    )


def test_the_named_removed_set_agrees_with_the_group_registry() -> None:
    """The hand-written list above IS the registry's line_movement family (D30-02).

    Without this, the list here and the predicate that performed the drop could
    drift apart and the delta assertion below would be pinning a fiction. Checked
    against a header-only frame so it asserts about the NAMES, not about gold.
    """
    frame = pd.DataFrame(
        columns=pd.Index([*PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS, *MARKET_SURVIVORS])
    )

    assert set(group_columns(frame, "line_movement")) == set(
        PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS
    ), (
        "the named removed set and backtest.signal_lift's line_movement predicate "
        "disagree -- one of them has drifted, and the delta assertion below would "
        "be pinning a set nothing actually removed"
    )
    assert len(PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS) == 15


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_phase_30_narrowing_is_exactly_the_line_movement_family(
    table_name: str,
) -> None:
    """The Phase-30 width delta is accounted for, column by column.

    The width tripwire on its own only says the total moved by 15; it cannot say
    the 15 are the intended columns. This pins the delta to the NAMED
    line-movement family, so a build that removed an unrelated column while
    incidentally leaving a line-movement one behind would fail here even though
    the integer still matched.

    Both halves are asserted: the fifteen are ABSENT from the rebuilt matrix, and
    the tripwire sits at the Phase-28 width plus the columns Phase 33.1 added.

    THE RESIDUAL WAS ZERO AND IS NOW +1, UPDATED WITH A REASON RATHER THAN
    SILENCED (Plan 33.1-07 Task 4). Phase 30's narrowing returned each matrix to
    exactly its Phase-28 width, so a zero residual was the right assertion for as
    long as nothing else moved the width. Phase 33.1's rung 1 then added the
    weather coverage flag -- the column that lets a reader tell "no weather
    record" from "the weather was mild" -- so the residual is now +1 BY DECISION.

    The expected residual is ``len(PHASE_331_ADDED_WEATHER_COLUMNS)`` rather than
    the integer 1, and the NAME is asserted separately below. That is the
    difference between "the width moved by one" and "the width moved by one, and
    the one is the column we meant": a build that added an unrelated column while
    omitting the flag satisfies the integer exactly as well as the right one does.

    AND NOW THE PHASE-33.2 NET DELTA, MOVED WITH A REASON (Plan 33.2-12). The ladder's
    width-moving rungs each record the NAMED columns they added and removed in a
    ``P332_*_GOLD_WIDTH_DELTA`` manifest slot; rung 4 removed the two weather inputs no
    forecast can supply (``precip_mm``, ``raw_precip_mm``), so the residual is the
    Phase-33.1 flag plus the summed Phase-33.2 net delta. The names are asserted by
    ``test_the_phase_332_width_deltas_chain_and_are_pinned_by_name``.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = list(pd.read_parquet(path).columns)
    surviving = [c for c in PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS if c in columns]
    assert not surviving, (
        f"{table_name} still carries line-movement columns {surviving} -- the SPEC "
        f"R3 drop is PARTIAL. A partial drop is worse than none: the three matrices "
        f"then disagree about the candidate feature set."
    )

    residual_delta = GOLD_FEATURE_MATRICES[table_name] - PHASE_28_WIDTHS[table_name]
    expected_residual = (
        len(PHASE_331_ADDED_WEATHER_COLUMNS) + phase332_net_width_delta()
    )
    assert residual_delta == expected_residual, (
        f"{table_name}: the tripwire reads "
        f"{GOLD_FEATURE_MATRICES[table_name]}, which is {residual_delta} columns "
        f"from the Phase-28 width {PHASE_28_WIDTHS[table_name]}. Removing the "
        f"{len(PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS)}-column line-movement family "
        f"returns the matrix to exactly its Phase-28 width, and Phase 33.1 adds "
        f"{len(PHASE_331_ADDED_WEATHER_COLUMNS)} on top of that "
        f"({list(PHASE_331_ADDED_WEATHER_COLUMNS)}) and the Phase-33.2 ladder a net "
        f"{phase332_net_width_delta()}. A residual other than "
        f"{expected_residual} means something ELSE changed the gold width. "
        f"Identify it before updating the tripwire."
    )


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_the_phase_331_widening_is_pinned_to_the_named_coverage_flag(
    table_name: str,
) -> None:
    """The +1 is pinned to a NAME, in both directions (Plan 33.1-07 Task 4).

    Direction one: the named column is PRESENT in the matrix. Direction two: the
    width equals the Phase-28 width plus exactly the length of the named tuple.

    Asserting only the second would pass for a build that added an unrelated
    column while omitting the flag; asserting only the first would pass for a
    build that added the flag AND something else. Together they say the width
    moved by these columns and no others.

    Removing ``weather_coverage`` from a copy of a matrix makes THIS test fail by
    NAME, where the width tripwire alone would only have reported an integer.

    The width half now adds the Phase-33.2 net delta (Plan 33.2-12): the flag is still
    present and still counted, and the rungs after it are accounted for by the manifest's
    named width-delta slots rather than by a relaxed integer.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = set(pd.read_parquet(path).columns)
    missing = [c for c in PHASE_331_ADDED_WEATHER_COLUMNS if c not in columns]
    assert not missing, (
        f"{table_name} does not carry {missing}. Without the coverage flag a NULL "
        f"observation is indistinguishable from a measured one, which is the whole "
        f"point of the Phase-33.1 rung (SPEC R5) -- and the width integer alone "
        f"cannot say WHICH column arrived."
    )

    assert (
        GOLD_FEATURE_MATRICES[table_name]
        == PHASE_28_WIDTHS[table_name]
        + len(PHASE_331_ADDED_WEATHER_COLUMNS)
        + phase332_net_width_delta()
    ), (
        f"{table_name}: the tripwire reads {GOLD_FEATURE_MATRICES[table_name]}, "
        f"which is not the Phase-28 width {PHASE_28_WIDTHS[table_name]} plus the "
        f"{len(PHASE_331_ADDED_WEATHER_COLUMNS)} named Phase-33.1 column(s) "
        f"{list(PHASE_331_ADDED_WEATHER_COLUMNS)} plus the Phase-33.2 net "
        f"{phase332_net_width_delta()}."
    )


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_the_market_survivors_are_not_taken_with_the_family(table_name: str) -> None:
    """The line-movement drop must be a family removal, not a suffix sweep.

    ``snapshot_total`` ends in ``total`` and ``spread_movement`` contains ``spread``; a
    substring-based prune would take all four of these with the fifteen. THAT suffix
    discipline is the node's real and unchanged subject, and it is asserted here AT THE
    PREDICATE: ``line_movement`` matches none of the four, ``market`` matches all four.

    IT NO LONGER READS GOLD, because the four are no longer IN gold and that is by
    decision (Plan 33.2-19, p332_ rung 9, D33.2-03): they left through the ``market``
    group, not through the line-movement drop, and the check now proves exactly that
    attribution. Was: the four asserted PRESENT in every matrix, which held only while
    no betting line had been removed from the model inputs.

    The predicate is asked about NAMES on a header-only frame, so this node says nothing
    about which gold exists; ``test_the_phase_332_width_deltas_chain_and_are_pinned_by_name``
    is what asserts their absence from the live matrices.
    """
    del table_name  # the assertion is about the predicate, not about a matrix
    frame = pd.DataFrame(
        columns=pd.Index([*PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS, *MARKET_SURVIVORS])
    )

    taken_by_line_movement = set(group_columns(frame, "line_movement")) & set(
        MARKET_SURVIVORS
    )
    assert taken_by_line_movement == set(), (
        f"the line-movement predicate matches market columns {sorted(taken_by_line_movement)}; "
        "a family drop that reaches them is the substring sweep this node exists to catch"
    )
    assert set(group_columns(frame, "market")) == set(MARKET_SURVIVORS), (
        "the market predicate does not match exactly the four survivors, so their "
        "removal from gold is not attributable to the market group alone"
    )


# ---------------------------------------------------------------------------
# THE PHASE-33 WAVE-14 LADDER'S WIDTH CLAIM (Plan 33-14 Task 5).
#
# ONE HOME FOR THE WIDTH NUMBERS, not a second. `GOLD_FEATURE_MATRICES` in
# scripts/data_qa.py already reads 195/196/195 and needs no edit: neither rung
# added or removed a column, which is itself the claim being asserted here.
# `PHASE_28_WIDTHS` and `PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS` above are
# historical records and stay exactly as they are.
# ---------------------------------------------------------------------------


def test_the_phase33_ladder_moved_no_width() -> None:
    """Both ends of the ladder read the same triple, and it is the live one.

    The exemption rung re-runs a normalization stage and the Elo rung is a
    declaration over the same transition -- neither adds or removes a column. A
    width move at either end would be a structural change nobody declared, and
    it is the one thing the per-rung diff cannot explain after the fact because
    `data/gold/` is gitignored and one-way.
    """
    before = tuple(phase33_state.GOLD_WIDTHS_BEFORE_PHASE33_LADDER)
    after_exemption = tuple(phase33_state.GOLD_WIDTHS_AFTER_EXEMPTION_RUNG)
    after_elo = tuple(phase33_state.GOLD_WIDTHS_AFTER_ELO_REBUILD)

    assert before == after_exemption == after_elo, (
        f"the ladder's width triples disagree: before {before}, after the "
        f"exemption rung {after_exemption}, after the Elo rung {after_elo}. "
        "Neither rung declared a width change."
    )
    assert before == tuple(phase33_state.GOLD_WIDTHS_AFTER_WEATHER_RUNG), (
        f"the ladder's baseline {before} is not Phase 33.1's close "
        f"{tuple(phase33_state.GOLD_WIDTHS_AFTER_WEATHER_RUNG)}, so something "
        "rebuilt gold between the two phases without attributing it."
    )


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_the_live_matrix_width_matches_the_ladder_record(table_name: str) -> None:
    """The recorded triple describes the store it is a record FOR.

    Asserted against LIVE gold rather than against another constant, because two
    constants agreeing with each other says nothing about the artifact.

    The recorded triple is ``current_ladder_widths()`` since Plan 33.2-12: the last
    Phase-33.2 width-delta slot's measured ``widths_after``, or the Phase-33 ladder's close
    while no Phase-33.2 rung has moved a width.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    recorded = dict(zip(MATRIX_ORDER, current_ladder_widths(), strict=True))
    actual = pd.read_parquet(path).shape[1]
    assert actual == recorded[table_name], (
        f"{table_name} is {actual} columns wide but the ladder records "
        f"{recorded[table_name]}. Either a rung moved a width nobody declared, "
        "or gold has been rebuilt since the ladder closed."
    )


# ---------------------------------------------------------------------------
# THE PHASE-33.2 WIDTH CHAIN (Plan 33.2-12, <owned_protocol_gold_width_pin> P4(b)).
# ---------------------------------------------------------------------------


def test_the_phase_332_width_deltas_chain_and_are_pinned_by_name() -> None:
    """The manifest's width deltas chain, name their columns, and equal the live pin.

    * The chain arithmetic holds from the Phase-33 ladder's close.
    * Every ``added`` name is PRESENT and every ``removed`` name ABSENT in each live matrix.
    * ``GOLD_FEATURE_MATRICES`` equals ``current_ladder_widths()``.

    Controls: at least one slot is discovered once rung 4 has run (non-vacuity), and a
    planted chain whose ``widths_after`` is off by one yields a violation.
    """
    slots = phase332_width_deltas()
    assert slots, "non-vacuity: no P332_*_GOLD_WIDTH_DELTA slot was discovered"

    baseline = tuple(phase33_state.GOLD_WIDTHS_AFTER_ELO_REBUILD)
    assert width_chain_violations(slots, baseline) == []

    for table_name in MATRIX_ORDER:
        path = GOLD_DIR / f"{table_name}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not built yet -- run scripts.build_features first")
        columns = set(pd.read_parquet(path).columns)
        for slot in slots:
            assert set(slot["added"]) <= columns, (table_name, slot["rung"])
            assert not set(slot["removed"]) & columns, (table_name, slot["rung"])

    assert (
        tuple(GOLD_FEATURE_MATRICES[name] for name in MATRIX_ORDER)
        == current_ladder_widths()
    )

    planted = [
        {
            **slots[0],
            "widths_after": tuple(w + 1 for w in slots[0]["widths_after"]),
        }
    ]
    assert width_chain_violations(planted, baseline) != []


def test_the_ladder_order_puts_an_extra_step_after_the_rung_it_follows() -> None:
    """``ladder_order`` is what lets an EXTRA STEP move a width (p332_ step 8e).

    The ladder's extra steps carry string ids and its numbered rungs carry ints, so the
    slot list cannot be ordered by the raw ``rung`` value at all once a string-id step
    appends one. This pins the order the chain arithmetic depends on, and the refusal
    for an id that names no rung.
    """
    assert sorted([9, "8e", 4, "8b", 8], key=ladder_order) == [4, 8, "8b", "8e", 9]
    with pytest.raises(ValueError):
        ladder_order("rung8")
