"""One edge-band ruler PER UNIT (Phase 33, plan 33-17 Task 1; CLEAN-01, D33-22).

WHAT THIS MODULE PINS
---------------------
``utils.edge_tier`` used to apply ONE threshold pair -- ``0.05 / 0.02`` -- to three
incompatible units: a WP probability delta, an ATS point margin and an O/U ratio. Its own
module docstring named that defect (DEF-31-17) rather than hiding it, and recorded that Plan
31-17 renamed the helper without repairing it. The repair is here: ``edge_tier`` now takes a
REQUIRED ``target`` and reads the frozen per-target pair from the Phase-33 pre-registration.

WHY ``target`` HAS NO DEFAULT
-----------------------------
A defaulted target would let a call site that forgets apply WP's probability thresholds to an
ATS point edge -- which IS the defect being removed, reintroduced as a default. So omitting it
is a ``TypeError`` from the interpreter itself, the one refusal nobody can forget to write.

WHY WP's 23 POINTS ARE REUSED RATHER THAN RE-RECORDED
-----------------------------------------------------
``tests/api/test_cache_betting._EDGE_TIER_SNAPSHOT`` was recorded by running the two retired
helpers side by side BEFORE the Phase-31 collapse. WP's frozen pair is UNCHANGED at 0.05 /
0.02 by design (D33-20), so that tuple survives byte-for-byte and keeps its evidentiary value
instead of being rewritten with a new expectation. This module imports it rather than copying
it, because a copy is a second place the evidence can drift.

ATS and O/U have no pre-collapse behaviour to preserve -- before this plan they had no
per-target band at all -- so their grids are recorded fresh in ``tests.phase33_state``, derived
from the frozen pairs by a throwaway reference band and never by calling the function they
check.

THE RULER IS NOW THE CORRECTION (Plan 33.2-26, SPEC R14)
---------------------------------------------------------
``utils.edge_tier`` reads ``backtest.corrected_cold_start_constants``, which supersedes the 11761c7
pairs. WP's corrected pair is UNCHANGED at 0.05 / 0.02 (the anchor), so its 23-point pre-collapse
snapshot is KEPT and asserted as before. ATS's and O/U's pairs moved, so their recorded grids --
measured against the 11761c7 pairs -- keep their VALUES and are re-labelled here by a throwaway
reference band on the CORRECTED pair, never by calling the function they check; a control proves
the recorded 11761c7 labels are NOT what the live band produces. A target whose corrected pair is
``None`` is UNBANDED: ``edge_tier`` returns ``None``, distinct from ``"low"``.

Run this module:  uv run pytest tests/unit/test_edge_tier_per_target.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

import backtest.cold_start_constants as superseded
import utils.edge_tier as edge_tier_module
from backtest.corrected_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from tests.api.test_cache_betting import _EDGE_TIER_SNAPSHOT
from tests.phase33_state import ATS_EDGE_TIER_GRID, OU_EDGE_TIER_GRID
from utils.edge_tier import (
    EDGE_TIER_HIGH_THRESHOLD,
    EDGE_TIER_LABELS,
    EDGE_TIER_MEDIUM_THRESHOLD,
    EDGE_TIER_TARGETS,
    UnknownEdgeTargetError,
    edge_tier,
    edge_tier_series,
)


def _reference_band(value: float, pair: tuple[float, float]) -> str:
    """A THROWAWAY reference band, never the function under test: STRICT ``>`` on |value|."""
    high, medium = pair
    magnitude = abs(value)
    if magnitude > high:
        return "high"
    if magnitude > medium:
        return "medium"
    return "low"


def _corrected_pair(target: str) -> tuple[float, float]:
    pair = EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    assert pair is not None, f"{target} has no corrected threshold"
    return pair


def _relabelled(
    grid: tuple[tuple[float, str], ...], target: str
) -> tuple[tuple[float, str], ...]:
    """The recorded grid's VALUES, labelled by the reference band on the CORRECTED pair."""
    pair = _corrected_pair(target)
    return tuple((value, _reference_band(value, pair)) for value, _label in grid)


# The three grids in one place, so every per-target assertion below iterates the SAME
# structure and a target added to the vocabulary without a grid fails loudly. WP's snapshot is
# kept verbatim (its corrected pair is unchanged); ATS's and O/U's are re-labelled.
_GRIDS: dict[str, tuple[tuple[float, str], ...]] = {
    "wp": _EDGE_TIER_SNAPSHOT,
    "ats": _relabelled(ATS_EDGE_TIER_GRID, "ats"),
    "ou": _relabelled(OU_EDGE_TIER_GRID, "ou"),
}


# ---------------------------------------------------------------------------
# 1. The target is REQUIRED
# ---------------------------------------------------------------------------


def test_omitting_the_target_is_a_type_error_and_not_a_silent_wp_band() -> None:
    """D33-22: no default. The interpreter refuses before any threshold is read.

    A defaulted target is the defect this requirement removes, wearing a different hat: a
    call site that forgets would band an ATS POINT edge against a WP PROBABILITY pair and
    publish the result without anything raising.
    """
    with pytest.raises(TypeError):
        edge_tier(0.03)  # type: ignore[call-arg]


def test_the_signature_records_the_absence_of_a_default() -> None:
    """The same claim, asserted on the signature rather than on the call.

    Separate from the TypeError above because they can fail for different reasons: a default
    could be added while some other TypeError still fired, and then the test above would stay
    green while the guarantee had gone.
    """
    parameter = inspect.signature(edge_tier).parameters["target"]
    assert parameter.default is inspect.Parameter.empty


def test_the_series_form_also_requires_a_target() -> None:
    """The vectorized entry point must not be the way round the scalar one's requirement."""
    with pytest.raises(TypeError):
        edge_tier_series(pd.Series([0.03]))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 1b. A target with NO honest threshold is UNBANDED (Plan 33.2-26, SPEC R14)
# ---------------------------------------------------------------------------


def _without_threshold(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> dict[str, tuple[float, float] | None]:
    planted = dict(EDGE_TIER_THRESHOLDS_BY_TARGET)
    planted[target] = None
    monkeypatch.setattr(edge_tier_module, "_THRESHOLDS_BY_TARGET", planted)
    return planted


def test_a_none_threshold_is_unbanded_never_a_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` is not unpacked into a TypeError and is not another target's pair."""
    _without_threshold(monkeypatch, "ats")
    for value in (10.0, 0.5, 0.0, None, np.nan):
        assert edge_tier(value, "ats") is None
    assert list(edge_tier_series(pd.Series([10.0, None]), "ats")) == [None, None]
    # The other targets keep their bands, and "low" stays a band.
    assert edge_tier(None, "wp") == "low"
    assert edge_tier(0.01, "wp") == "low"
    assert edge_tier(1.0, "ou") == "high"


def test_unbanded_is_distinct_from_low() -> None:
    """The control: with a threshold, a small edge IS banded "low" rather than left unbanded."""
    assert edge_tier(0.0, "ats") == "low"
    assert edge_tier(0.0, "ats") is not None


# ---------------------------------------------------------------------------
# 2. The three recorded grids
# ---------------------------------------------------------------------------


def test_wp_reproduces_all_twenty_three_pre_collapse_points_under_the_new_signature() -> (
    None
):
    """The Phase-31 evidence survives the signature change (T-33-85) AND the correction.

    WP's corrected pair is still 0.05 / 0.02 -- the values the collapsed helper already used --
    so every one of these 23 labels must be identical. A single mismatch means a published label
    on ``/`` and ``/betting`` has moved.
    """
    assert _corrected_pair("wp") == (0.05, 0.02)
    assert len(_EDGE_TIER_SNAPSHOT) == 23
    mismatches = [
        (value, expected, edge_tier(value, "wp"))
        for value, expected in _EDGE_TIER_SNAPSHOT
        if edge_tier(value, "wp") != expected
    ]
    assert not mismatches, (
        "the pre-collapse WP grid no longer reproduces under edge_tier(value, 'wp'): "
        f"{mismatches}"
    )


@pytest.mark.parametrize("target", ["ats", "ou"])
def test_each_line_target_reproduces_its_own_recorded_grid(target: str) -> None:
    """ATS in POINTS and O/U as a RATIO, each on its own frozen pair.

    Before this plan both were banded against WP's probability pair, which put 98.80% of ATS
    games in "high" (1,074 of 1,087). The grids are what make the new rule checkable at every
    boundary rather than only in aggregate.
    """
    grid = _GRIDS[target]
    assert len(grid) == 23
    mismatches = [
        (value, expected, edge_tier(value, target))
        for value, expected in grid
        if edge_tier(value, target) != expected
    ]
    assert not mismatches, f"{target} grid mismatches: {mismatches}"


@pytest.mark.parametrize(
    ("target", "grid"),
    [("ats", ATS_EDGE_TIER_GRID), ("ou", OU_EDGE_TIER_GRID)],
)
def test_the_live_band_is_not_the_superseded_11761c7_band(
    target: str, grid: tuple[tuple[float, str], ...]
) -> None:
    """The control that the ORIGINAL pair is not what the live band applies.

    The recorded grid IS the 11761c7 band (re-derived here from the original pair), and the live
    band disagrees with it on at least one of its own boundary points.
    """
    original = superseded.EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    assert [label for _v, label in grid] == [
        _reference_band(value, original) for value, _label in grid
    ]
    assert EDGE_TIER_THRESHOLDS_BY_TARGET[target] != original
    assert [edge_tier(value, target) for value, _label in grid] != [
        label for _v, label in grid
    ]


def test_the_three_grids_are_not_the_same_grid() -> None:
    """Anti-vacuity: a per-target ruler that answered identically everywhere would be one ruler.

    Taken on the ATS grid, whose values are POINTS. Against ATS's own pair the 23 rows span all
    three labels. Against WP's probability pair the MIDDLE BAND COLLAPSES ENTIRELY -- 22 of the
    23 read "high" and the only survivor is the exact zero, because a point margin above 0.05
    points is very nearly every margin there is. That collapse is the defect in miniature: it is
    the same shape as the measured 98.80% "high" share over the pinned population.
    """
    under_ats = [edge_tier(value, "ats") for value, _label in _GRIDS["ats"]]
    under_wp = [edge_tier(value, "wp") for value, _label in ATS_EDGE_TIER_GRID]
    assert under_ats != under_wp
    assert "medium" not in under_wp, (
        "the ATS grid no longer demonstrates the defect it was chosen to demonstrate; under "
        f"WP's probability pair it still bands {sorted(set(under_wp))} with a middle band"
    )
    assert under_wp.count("high") == 22, under_wp
    assert set(under_ats) == set(EDGE_TIER_LABELS)


# ---------------------------------------------------------------------------
# 3. The six boundaries -- STRICT, so a value AT a threshold bands LOWER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_a_value_exactly_at_the_high_threshold_bands_medium(target: str) -> None:
    """Three of the six boundary assertions -- one per target, at its own HIGH threshold."""
    high, _medium = EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    assert edge_tier(high, target) == "medium"
    assert edge_tier(-high, target) == "medium"


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_a_value_exactly_at_the_medium_threshold_bands_low(target: str) -> None:
    """The other three -- one per target, at its own MEDIUM threshold.

    Together with the test above these are the six the plan asks for. An accidental ``>=``
    anywhere in the rule moves a published label and fails here.
    """
    _high, medium = EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    assert edge_tier(medium, target) == "low"
    assert edge_tier(-medium, target) == "low"


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_just_above_each_threshold_bands_up(target: str) -> None:
    """The control for the two above: the bands are not simply collapsed downward.

    Without this, a rule that answered "low" for everything would satisfy both boundary tests.
    """
    high, medium = EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    assert edge_tier(high * 1.01, target) == "high"
    assert edge_tier(medium * 1.01, target) == "medium"


# ---------------------------------------------------------------------------
# 4. The closed vocabulary and the unknown-target refusal
# ---------------------------------------------------------------------------


def test_the_target_vocabulary_is_exactly_the_frozen_mapping_s_keys() -> None:
    """One source. A fourth target invented here would have no frozen pair to read."""
    assert sorted(EDGE_TIER_TARGETS) == sorted(EDGE_TIER_THRESHOLDS_BY_TARGET)
    assert sorted(EDGE_TIER_TARGETS) == ["ats", "ou", "wp"]


def test_an_unknown_target_refuses_by_name_and_names_the_valid_set() -> None:
    """Naming BOTH the offending value and the vocabulary is what makes the refusal actionable.

    A refusal that says only "unknown target" sends the reader to the source to find out what
    the alternatives are, which is the shape this repository treats as worse than no message.
    """
    with pytest.raises(UnknownEdgeTargetError) as excinfo:
        edge_tier(0.03, "spread")
    message = str(excinfo.value)
    assert "spread" in message
    for known in ("ats", "ou", "wp"):
        assert known in message


def test_the_series_form_refuses_an_unknown_target_even_on_an_empty_series() -> None:
    """The refusal is on the TARGET, not on the data.

    A validation that only fired per element would pass silently on an empty column -- the
    exact case a weekly slate with no priced games produces.
    """
    with pytest.raises(UnknownEdgeTargetError):
        edge_tier_series(pd.Series([], dtype=float), "spread")


# ---------------------------------------------------------------------------
# 5. edge_tier_series is a DISPATCH, checked by AST rather than by substring
# ---------------------------------------------------------------------------


def test_the_series_form_performs_no_comparison_of_its_own() -> None:
    """Zero ``ast.Compare`` nodes in its parsed body (Codex LOW, folded into the plan).

    This is what keeps "exactly one function computes the edge band" checkable. It is an AST
    walk and NOT a substring search for ``>`` or ``<``: a type annotation, a docstring, a
    comment or an f-string would false-positive a text scan, and a check that cries wolf is a
    check the next author deletes.
    """
    source = textwrap.dedent(inspect.getsource(edge_tier_series))
    compares = [
        node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Compare)
    ]
    assert compares == [], (
        "edge_tier_series performs its own comparison; the band would then have two "
        f"definitions that can drift apart (lines {[n.lineno for n in compares]})"
    )


def test_the_ast_check_would_fire_on_a_planted_comparison() -> None:
    """Fail-closed control. A check only ever observed passing cannot be told from a dead one."""
    planted = textwrap.dedent(
        """
        def edge_tier_series(edge, target):
            return edge.map(lambda value: "high" if value > 0.05 else "low")
        """
    )
    compares = [
        node for node in ast.walk(ast.parse(planted)) if isinstance(node, ast.Compare)
    ]
    assert compares, "the AST probe found no comparison in a body that plainly has one"


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_the_series_form_agrees_with_the_scalar_one_value_for_value(
    target: str,
) -> None:
    """The structural claim above, confirmed behaviourally on the target's own grid."""
    values = [value for value, _label in _GRIDS[target]]
    expected = [label for _value, label in _GRIDS[target]]
    assert list(edge_tier_series(pd.Series(values), target)) == expected


# ---------------------------------------------------------------------------
# 6. The degenerate series, per target (Antigravity MEDIUM, folded into the plan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_an_empty_series_returns_an_empty_series_and_does_not_raise(
    target: str,
) -> None:
    """A weekly slate with no priced game is a real input, not a malformed one."""
    result = edge_tier_series(pd.Series([], dtype=float), target)
    assert len(result) == 0


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_an_all_null_series_bands_low_throughout(target: str) -> None:
    """The absent-edge-is-low contract, stated in the helper and asserted per target.

    52 of the 1,139 games in the pinned population carry no stored market line at all, so all
    three edges are NULL on them. The contract is what stops the two call sites answering that
    case differently.
    """
    result = edge_tier_series(pd.Series([None, None, None], dtype=object), target)
    assert list(result) == ["low", "low", "low"]


@pytest.mark.parametrize("target", ["ats", "ou", "wp"])
def test_a_nan_never_silently_produces_a_band_by_comparison(target: str) -> None:
    """NaN compares False against every threshold, so an unguarded rule would answer "low"
    for the wrong reason -- and would answer "high" the moment the comparisons were inverted.

    Asserted alongside a REAL value in the same series so the check cannot pass on a frame
    that produced nothing at all.
    """
    high, _medium = EDGE_TIER_THRESHOLDS_BY_TARGET[target]
    result = edge_tier_series(pd.Series([np.nan, high * 2.0, None]), target)
    assert list(result) == ["low", "high", "low"]
    assert edge_tier(np.nan, target) == "low"
    assert edge_tier(None, target) == "low"


# ---------------------------------------------------------------------------
# 7. The two legacy constants still resolve, and they resolve to WP's pair
# ---------------------------------------------------------------------------


def test_the_legacy_module_constants_are_repointed_at_the_wp_entry() -> None:
    """They survive because ``tests/unit/test_phase33_preregistration.py`` binds to them.

    Re-pointed rather than re-typed: a second literal 0.05 would be a second source for a
    number the pre-registration froze, which is the drift this repository has been bitten by
    three times.
    """
    assert tuple(EDGE_TIER_THRESHOLDS_BY_TARGET["wp"]) == (
        EDGE_TIER_HIGH_THRESHOLD,
        EDGE_TIER_MEDIUM_THRESHOLD,
    )
