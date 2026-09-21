"""The standing guard: the O/U target has NO eligibility gate (D33.2-24, Plan 33.2-06).

D33.2-24 deleted ``OUStrategy``'s under-OR-high-total UNION, together with the high-total
boundary it compared against, so all three targets are EV-only: the calibrated chain runs on every
candidate and the EV floor alone decides. This module fails if a gate comes back.

It has TWO halves, because one half alone cannot prove the claim:

  * STRUCTURAL -- an AST scan of ``OUStrategy``'s own members for an eligibility-shaped name (a
    regime helper, a union helper, a boundary, a sub-population label). It catches the deleted
    shape returning as a new member. It CANNOT catch a gate written INSIDE the existing
    ``eligibility`` method: the old gate lived exactly there, behind the same protocol-shaped
    surface ``ATSStrategy`` exposes, so a new one could too without adding a member or changing a
    method set.
  * BEHAVIORAL -- :func:`assert_no_eligibility_gate` asks ``eligibility`` itself, for an over and
    an under candidate at closing totals below, at and above the retired 48.0, and requires
    ``None`` every time, plus the sideless ``"no_bet_side"`` reason ``ATSStrategy`` gives. A
    planted subclass that hides a gate inside ``eligibility`` -- no new member, identical public
    method set -- is flagged by it, and is asserted NOT to be flagged by the structural half, so
    this module demonstrates the exact gap the behavioral half closes.

Each half carries its controls: non-vacuity, the assertion, a planted violation, and a
no-false-positive case on ``ATSStrategy`` (the target that never had a gate).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

from backtest.selector_strategies import (
    NO_SUBPOPULATION_LABEL,
    OU_SIDES,
    ATSStrategy,
    OUStrategy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_STRATEGIES_PATH = REPO_ROOT / "backtest" / "selector_strategies.py"

# The closing totals the behavioral probe walks: below, AT and above the retired high-total
# boundary (48.0). A gate keyed on the total -- in either direction, strict or not -- changes the
# answer for at least one of them.
_RETIRED_BOUNDARY = 48.0
_PROBE_TOTALS: tuple[float, ...] = (44.5, _RETIRED_BOUNDARY, 51.5)

# The sideless reason every production target reports (``ATSStrategy.eligibility``).
_SIDELESS_REASON = "no_bet_side"

# Eligibility-shaped member names. ``_totals_regime`` and ``_union_arms`` are the two members
# D33.2-24 deleted; the rest are the shapes a returning gate would plausibly take. ``eligibility``
# and ``eligibility_label`` are the Protocol's own members and do not match (neither contains
# ``_eligible``).
_GATE_SHAPED_NAME = re.compile(r"regime|union|_eligible|subpop|boundary", re.IGNORECASE)

_FIXTURE_SD = 13.0
_FIXTURE_BIAS = {2021: -1.0}


# ---------------------------------------------------------------------------
# The structural half
# ---------------------------------------------------------------------------


def _class_node(source: str, class_name: str) -> ast.ClassDef:
    """The ``ClassDef`` named *class_name* in *source*, failing loudly when it is absent."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    msg = f"class {class_name!r} not found in the scanned source"
    raise AssertionError(msg)


def class_member_names(source: str, class_name: str) -> set[str]:
    """Every member *class_name* defines or reaches: methods, class attributes, ``self.<x>``.

    ``self.<x>`` is collected whether it is assigned or only READ, so a gate that compares against
    an attribute a base class or a later edit supplies is still seen. Read from the PARSED tree,
    never from text, so a docstring or comment that names a retired member -- which
    ``OUStrategy``'s docstring deliberately does -- cannot trip the scan.
    """
    node = _class_node(source, class_name)
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
        elif isinstance(item, ast.Assign):
            names.update(t.id for t in item.targets if isinstance(t, ast.Name))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
    names.update(
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
    )
    return names


def gate_shaped_members(source: str, class_name: str) -> set[str]:
    """The members of *class_name* whose NAME has an eligibility-gate shape."""
    return {
        name
        for name in class_member_names(source, class_name)
        if _GATE_SHAPED_NAME.search(name)
    }


def _public_decision_methods(cls: type) -> set[str]:
    """The public callables a strategy exposes -- the decision surface the core dispatches to."""
    return {
        name
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_") and callable(member)
    }


class TestTheStructuralHalf:
    """Four controls: non-vacuity, the assertion, a planted violation, no false positive."""

    def test_the_scan_reads_the_real_ou_strategy(self) -> None:
        """NON-VACUITY: the scan finds OUStrategy and the members it certainly has."""
        members = class_member_names(
            _STRATEGIES_PATH.read_text(encoding="utf-8"), "OUStrategy"
        )
        assert {"eligibility", "eligibility_label", "resolve_bet_side"} <= members
        assert {"frozen_sd", "season_bias_by_season", "_sim"} <= members

    def test_ou_strategy_defines_no_gate_shaped_member(self) -> None:
        """THE ASSERTION: no regime, union, boundary or sub-population member on OUStrategy."""
        offenders = gate_shaped_members(
            _STRATEGIES_PATH.read_text(encoding="utf-8"), "OUStrategy"
        )
        assert offenders == set(), (
            f"OUStrategy defines eligibility-shaped member(s) {sorted(offenders)}. D33.2-24 "
            "deleted the O/U eligibility gate and nothing replaces it."
        )
        # The live class agrees with its source: neither deleted member is reachable.
        assert not hasattr(OUStrategy, "_totals_regime")
        assert not hasattr(OUStrategy, "_union_arms")
        assert "high_total_boundary" not in inspect.signature(OUStrategy).parameters

    def test_a_planted_gate_member_is_flagged(self) -> None:
        """PLANTED VIOLATION: the shape D33.2-24 removed is still detectable."""
        planted = """
            class OUStrategy:
                def _totals_regime(self, closing_total):
                    return "high" if closing_total > self.high_total_boundary else "not_high"

                def eligibility(self, row, bet_side):
                    return None
        """
        assert gate_shaped_members(planted, "OUStrategy") == {
            "_totals_regime",
            "high_total_boundary",
        }

    def test_ats_strategy_is_not_flagged(self) -> None:
        """NO FALSE POSITIVE: the target that never had a gate passes the same scan."""
        source = _STRATEGIES_PATH.read_text(encoding="utf-8")
        assert class_member_names(source, "ATSStrategy")
        assert gate_shaped_members(source, "ATSStrategy") == set()

    def test_the_public_decision_surface_matches_ats(self) -> None:
        """A SHAPE check, not proof of absence: OUStrategy decides through ATSStrategy's surface.

        Equal method sets cannot prove there is no gate -- the deleted gate lived INSIDE the
        shared ``eligibility`` method -- which is why the behavioral half below exists.
        """
        assert _public_decision_methods(OUStrategy) == _public_decision_methods(
            ATSStrategy
        )


# ---------------------------------------------------------------------------
# The behavioral half
# ---------------------------------------------------------------------------


def _probe_row(closing_total: float, side: str | None) -> dict[str, Any]:
    """One candidate row at *closing_total*, carrying the fields either target could read.

    The model total is placed three points to the side's direction, so the row is a genuine
    candidate of that side rather than a bare dict the strategy never looks at.
    """
    offset = {"over": 3.0, "under": -3.0, None: 0.0}[side]
    return {
        "game_id": f"2021_W01_PROBE@{closing_total}",
        "season": 2021,
        "week": 1,
        "model_total": closing_total + offset,
        "closing_total": closing_total,
        "model_spread": 3.0,
        "closing_spread": 1.0,
    }


def eligibility_gate_violations(strategy: Any) -> list[str]:
    """Every probe on which *strategy*'s ``eligibility`` behaves like a gate.

    A sided candidate must be eligible (``None``) at every probed total, on both O/U sides; a
    sideless one must get the sideless reason ``ATSStrategy`` gives. Anything else is a gate.
    """
    violations: list[str] = []
    for closing_total in _PROBE_TOTALS:
        for side in OU_SIDES:
            reason = strategy.eligibility(_probe_row(closing_total, side), side)
            if reason is not None:
                violations.append(
                    f"{side} at closing total {closing_total}: rejected as {reason!r}"
                )
        sideless = strategy.eligibility(_probe_row(closing_total, None), None)
        if sideless != _SIDELESS_REASON:
            violations.append(
                f"sideless at closing total {closing_total}: {sideless!r}, "
                f"expected {_SIDELESS_REASON!r}"
            )
    return violations


def assert_no_eligibility_gate(strategy: Any) -> None:
    """Fail when *strategy*'s ``eligibility`` gates any sided candidate on the probe grid."""
    violations = eligibility_gate_violations(strategy)
    assert violations == [], (
        f"{type(strategy).__name__}.eligibility behaves as an eligibility gate "
        f"(D33.2-24 deleted the O/U one): {violations}"
    )


class _PlantedInlineGate(OUStrategy):
    """A gate hidden INSIDE ``eligibility``: no new member, the identical public method set.

    It refuses a low-total over -- the exact case the deleted UNION refused -- and delegates
    otherwise. Nothing about it is visible to a member-name scan or to method-set parity.
    """

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        if bet_side == "over" and float(row["closing_total"]) < _RETIRED_BOUNDARY:
            return "not_subpop"
        return super().eligibility(row, bet_side)


def _ou_strategy() -> OUStrategy:
    """The real O/U strategy -- after D33.2-24 its constructor takes no boundary."""
    return OUStrategy(frozen_sd=_FIXTURE_SD, season_bias_by_season=_FIXTURE_BIAS)


class TestTheBehavioralHalf:
    """Four controls: non-vacuity, the assertion, a planted violation, no false positive."""

    def test_the_probe_grid_is_not_vacuous(self) -> None:
        """NON-VACUITY: both sides at three totals straddling 48.0, and real sided candidates."""
        assert len(_PROBE_TOTALS) == 3
        assert min(_PROBE_TOTALS) < _RETIRED_BOUNDARY < max(_PROBE_TOTALS)
        assert _RETIRED_BOUNDARY in _PROBE_TOTALS
        assert set(OU_SIDES) == {"over", "under"}
        strategy = _ou_strategy()
        for closing_total in _PROBE_TOTALS:
            for side in OU_SIDES:
                assert (
                    strategy.resolve_bet_side(_probe_row(closing_total, side)) == side
                )
            assert strategy.resolve_bet_side(_probe_row(closing_total, None)) is None

    def test_the_real_ou_strategy_has_no_eligibility_gate(self) -> None:
        """THE ASSERTION: every sided O/U candidate is eligible; the label names no slice."""
        strategy = _ou_strategy()
        assert_no_eligibility_gate(strategy)
        for closing_total in _PROBE_TOTALS:
            for side in (*OU_SIDES, None):
                row = _probe_row(closing_total, side)
                assert strategy.eligibility_label(row, side) == NO_SUBPOPULATION_LABEL

    def test_a_gate_planted_inside_eligibility_is_flagged(self) -> None:
        """PLANTED VIOLATION, and the gap it proves: behavior catches what structure cannot."""
        planted = _PlantedInlineGate(
            frozen_sd=_FIXTURE_SD, season_bias_by_season=_FIXTURE_BIAS
        )
        with pytest.raises(AssertionError, match="eligibility gate"):
            assert_no_eligibility_gate(planted)
        assert eligibility_gate_violations(planted) == [
            "over at closing total 44.5: rejected as 'not_subpop'"
        ]

        # The structural half does NOT see it -- no new member, identical public surface.
        source = inspect.getsource(_PlantedInlineGate)
        assert gate_shaped_members(source, "_PlantedInlineGate") == set()
        assert _public_decision_methods(_PlantedInlineGate) == _public_decision_methods(
            ATSStrategy
        )

    def test_ats_strategy_is_not_flagged(self) -> None:
        """NO FALSE POSITIVE: the target that never had a gate passes the same probe."""
        strategy = ATSStrategy(frozen_sd=_FIXTURE_SD, season_bias_by_season={2021: 0.5})
        assert eligibility_gate_violations(strategy) == []
        assert_no_eligibility_gate(strategy)
