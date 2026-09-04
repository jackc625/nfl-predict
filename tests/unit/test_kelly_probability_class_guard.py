"""The Kelly probability guard, at the level of the CLASS of defect (SPEC R10, T-31-44/45/45b).

Phase 31, plan 31-10, Task 3.

WHAT WENT WRONG, AND WHY REPAIRING THE INSTANCE WOULD NOT BE ENOUGH.

  ``backtest/simulation.py`` sized the spread target's Kelly stake off ``implied + edge``, where
  ``edge`` was a POINTS DISTANCE. The sum crossed 1.0 for every strong signal, and
  ``calculate_kelly_fraction`` answered a value at or outside the unit interval with a SILENT
  ``return 0.0``. So the wrong input produced a plausible-looking output -- a zero stake, which
  reads as "no edge here" -- instead of a failure. That is why the defect survived a whole
  milestone and was found by counting zero-staked bets rather than by anything going red.

  Swapping the probability at that one call site repairs the instance. It leaves the CLASS intact:
  a points quantity reaching a probability argument stays possible for any target added later, and
  a boundary value stays a silent refusal. This module closes the class in three ways.

THE THREE ASSERTIONS, AND WHY EACH IS NEEDED.

  1. BEHAVIOURAL -- for each of the three targets, drive the real selector and the real simulator
     over a hand-built week and CAPTURE, with a spy, every probability that actually reaches a
     Kelly entry point. Assert each is strictly inside ``(0, 1)`` and that the captured count per
     target is greater than zero, so the assertion cannot pass by capturing nothing. Inferring the
     probabilities from the recorded stakes instead would only re-derive them from the same code.
  2. BOUNDARY -- ``0.0`` and ``1.0`` each RAISE, and the message names the offending value; so does
     a value outside the interval. The raise replaces the silent zero at BOTH public entry points,
     and the ``calculate_optimal_bet_size`` case is exercised with an edge SMALL ENOUGH to trigger
     that method's own early return, because validating only the low-level method would leave a
     whole class of invalid public input silently accepted (REVIEW-KELLY).
  3. STRUCTURAL -- an AST scan over the three modules that size bets, enumerating every call site
     that passes a probability into a Kelly entry point and checking where that value came from. A
     re-introduced ``implied + edge`` fails here even if no behavioural test happens to reach it.

Run this module:  .venv/Scripts/python.exe -m pytest tests/unit/test_kelly_probability_class_guard.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.simulation import BettingSimulator, SimulationConfig
from utils import kelly_criterion as kelly_module
from utils.kelly_criterion import KellyCalculator

REPO_ROOT = Path(__file__).resolve().parents[2]

# The three modules that decide a stake. Any Kelly call site lives in one of them; a fourth would
# be a second sizing path, which SPEC R4's source scan already forbids.
_SIZING_MODULES = (
    "backtest/simulation.py",
    "backtest/bet_selector.py",
    "backtest/selector_strategies.py",
)

# The two PUBLIC Kelly entry points and the parameter each takes the probability under.
_KELLY_ENTRY_POINTS = {
    "calculate_optimal_bet_size": "model_prob",
    "calculate_kelly_fraction": "win_probability",
}

# Names that hold an EDGE or an IMPLIED PROBABILITY. A probability may never be derived by adding
# one of these to anything: that sum is the defect, whatever the result is later called.
_FORBIDDEN_OPERANDS = frozenset({"edge", "implied", "points_edge", "market_prob"})

_FIXTURE_OU_SD = 13.0
_FIXTURE_OU_BIAS = {2021: -1.0}
_FIXTURE_ATS_SD = 13.0
_FIXTURE_ATS_BIAS = {2021: 0.0}
_BANKROLL = 10_000.0


# ---------------------------------------------------------------------------
# The spy
# ---------------------------------------------------------------------------


class _KellyProbabilitySpy:
    """Captures every probability handed to a Kelly entry point, then calls through.

    Patched onto the CLASS, so it sees every calculator any code path builds -- including the one
    ``BetSelector`` constructs privately and the one ``BettingSimulator.simulate`` builds per run.
    Neither is reachable from a test otherwise, and a spy that could only see an injected
    calculator would miss exactly the call sites this guard exists for.
    """

    def __init__(self) -> None:
        self.captured: list[tuple[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for method_name, param in _KELLY_ENTRY_POINTS.items():
            original = getattr(KellyCalculator, method_name)

            def _wrapped(
                inner_self,
                *args,
                _original=original,
                _method=method_name,
                _param=param,
                **kwargs,
            ):
                value = kwargs[_param] if _param in kwargs else args[0]
                self.captured.append((_method, value))
                return _original(inner_self, *args, **kwargs)

            monkeypatch.setattr(KellyCalculator, method_name, _wrapped)

    @property
    def values(self) -> list[Any]:
        return [value for _method, value in self.captured]


# ---------------------------------------------------------------------------
# Fixture weeks, one per target
# ---------------------------------------------------------------------------


def _results_like(frames: dict[str, pd.DataFrame]):
    backtest_results = MagicMock()
    backtest_results.all_predictions = frames
    backtest_results.all_clv = dict.fromkeys(frames, pd.DataFrame())
    return backtest_results


def _wp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2021_W01_A@B",
                "season": 2021,
                "week": 1,
                "model_prob": 0.70,
                "actual": 1,
                "ml_home": -150,
                "ml_away": 130,
                "spread": -3.0,
                "total": 45.0,
                "has_closing_odds": True,
            },
            {
                "game_id": "2021_W01_C@D",
                "season": 2021,
                "week": 1,
                "model_prob": 0.28,
                "actual": 0,
                "ml_home": 145,
                "ml_away": -165,
                "spread": 3.0,
                "total": 45.0,
                "has_closing_odds": True,
            },
        ]
    )


def _ats_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2021_W01_E@F",
                "season": 2021,
                "week": 1,
                "model_spread": 7.0,
                "actual": 10.0,
                "ml_home": -110,
                "ml_away": -110,
                "spread": -3.0,
                "total": 45.0,
                "has_closing_odds": True,
            },
            {
                "game_id": "2021_W01_G@H",
                "season": 2021,
                "week": 1,
                "model_spread": -9.0,
                "actual": -12.0,
                "ml_home": -110,
                "ml_away": -110,
                "spread": -3.0,
                "total": 45.0,
                "has_closing_odds": True,
            },
        ]
    )


def _ou_candidates() -> list[dict[str, Any]]:
    return [
        {
            "game_id": "2021_W01_I@J",
            "season": 2021,
            "week": 1,
            "model_total": 38.0,
            "closing_total": 45.0,
            "actual": 40.0,
        },
        {
            "game_id": "2021_W01_K@L",
            "season": 2021,
            "week": 1,
            "model_total": 36.0,
            "closing_total": 47.0,
            "actual": 39.0,
        },
    ]


def _ou_selector():
    from backtest.bet_selector import BetSelector

    return BetSelector(
        frozen_sd=_FIXTURE_OU_SD,
        season_bias_by_season=_FIXTURE_OU_BIAS,
        ev_floor_t=0.0,
        bankroll=_BANKROLL,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
    )


def _ats_selector():
    from backtest.bet_selector import BetSelector
    from backtest.selector_strategies import ATSStrategy

    return BetSelector(
        frozen_sd=_FIXTURE_ATS_SD,
        season_bias_by_season=_FIXTURE_ATS_BIAS,
        ev_floor_t=0.0,
        bankroll=_BANKROLL,
        strategies=[
            ATSStrategy(
                frozen_sd=_FIXTURE_ATS_SD, season_bias_by_season=_FIXTURE_ATS_BIAS
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Behavioural: every probability that REACHES Kelly, captured per target
# ---------------------------------------------------------------------------


class TestEveryKellyProbabilityIsStrictlyInsideTheUnitInterval:
    """Captured with a spy on the calculator class, never inferred from the resulting stakes."""

    def _assert_all_strictly_interior(
        self, spy: _KellyProbabilitySpy, target: str
    ) -> None:
        assert spy.captured, (
            f"no probability reached a Kelly entry point for target {target!r}; the assertion "
            "below would pass vacuously"
        )
        for method, value in spy.captured:
            assert isinstance(value, float), (
                f"[{target}] {method} received a non-float {value!r}"
            )
            assert 0.0 < value < 1.0, (
                f"[{target}] {method} received {value!r}, which is not strictly inside (0, 1)"
            )

    def test_winner_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The winner path sizes on the side-correct probability, and it is measured here."""
        spy = _KellyProbabilitySpy()
        spy.install(monkeypatch)
        BettingSimulator(SimulationConfig()).simulate(
            _results_like({"wp": _wp_frame()}), pd.DataFrame()
        )
        self._assert_all_strictly_interior(spy, "wp")

    def test_spread_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The spread path sizes on the calibrated cover probability (D31-04)."""
        spy = _KellyProbabilitySpy()
        spy.install(monkeypatch)
        BettingSimulator(SimulationConfig(), ats_bet_selector=_ats_selector()).simulate(
            _results_like({"ats": _ats_frame()}), pd.DataFrame()
        )
        self._assert_all_strictly_interior(spy, "ats")

    def test_totals_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The totals path sizes on the calibrated P(side) (BET-02)."""
        spy = _KellyProbabilitySpy()
        spy.install(monkeypatch)
        result = _ou_selector().select(_ou_candidates())
        assert result.selected
        self._assert_all_strictly_interior(spy, "ou")


# ---------------------------------------------------------------------------
# 2. Boundary: a value at or outside the interval is a LOUD failure
# ---------------------------------------------------------------------------


def _calculator(confidence_threshold: float = 0.02) -> KellyCalculator:
    return KellyCalculator(
        starting_bankroll=_BANKROLL,
        max_bet_pct=0.05,
        base_unit_size=100.0,
        default_kelly_fraction=0.25,
        confidence_threshold=confidence_threshold,
    )


class TestBoundaryValuesRaiseRatherThanReturnZero:
    """T-31-45: the silent ``return 0.0`` is what hid the defect, so it is gone."""

    @pytest.mark.parametrize("probability", [0.0, 1.0, 1.4, -0.2])
    def test_calculate_kelly_fraction_raises_and_names_the_value(
        self, probability: float
    ) -> None:
        """Each offending value raises, and the message contains it.

        Naming the value is what makes the failure actionable: "a probability was invalid" sends
        the reader looking, and the whole point is that the wrong number is hard to spot.
        """
        with pytest.raises(kelly_module.KellyProbabilityError) as exc:
            _calculator().calculate_kelly_fraction(probability, -110)
        assert repr(probability) in str(exc.value) or str(probability) in str(exc.value)

    @pytest.mark.parametrize("probability", [0.0, 1.0, 1.4, -0.2])
    def test_calculate_optimal_bet_size_raises_before_its_early_return(
        self, probability: float
    ) -> None:
        """REVIEW-KELLY: the edge is deliberately small enough to trigger the early return.

        ``market_odds`` of +10000 implies a market probability near 0.0099, so a ``model_prob`` of
        1.4 produces an edge far above the threshold -- which is why the threshold is raised here
        instead, to 2.0, so the early return WOULD fire for every parametrised value. Before the
        fix each of these returned a tidy zero-stake KellyResult reading "Edge too small or
        negative" and gave the caller no signal at all.
        """
        calculator = _calculator(confidence_threshold=2.0)
        with pytest.raises(kelly_module.KellyProbabilityError):
            calculator.calculate_optimal_bet_size(
                model_prob=probability, market_odds=-110
            )

    def test_the_early_return_really_would_have_fired(self) -> None:
        """The negative control for the test above: at that threshold a VALID probability returns 0.

        Without this, the raising test could be passing because the edge was large, not because
        validation runs first -- and the whole claim is about ordering.
        """
        result = _calculator(confidence_threshold=2.0).calculate_optimal_bet_size(
            model_prob=0.99, market_odds=-110
        )
        assert result.recommended_bet == 0.0
        assert "Edge too small" in result.reasoning

    def test_validation_precedes_the_edge_comparison_in_source(self) -> None:
        """The ordering is pinned in the source, not only in behaviour (T-31-45b)."""
        source = inspect.getsource(KellyCalculator.calculate_optimal_bet_size)
        body = source.split('"""', 2)[-1]
        validate_at = body.index("_validate_kelly_probability")
        early_return_at = body.index("edge <= self.confidence_threshold")
        assert validate_at < early_return_at

    def test_a_valid_probability_still_sizes_a_bet(self) -> None:
        """The positive control: the guard rejects only what it is meant to reject."""
        result = _calculator().calculate_optimal_bet_size(
            model_prob=0.60, market_odds=-110
        )
        assert result.recommended_bet > 0.0

    def test_the_error_is_a_valueerror_subclass(self) -> None:
        """An existing ``except ValueError`` still catches it; it is not a new vocabulary."""
        assert issubclass(kelly_module.KellyProbabilityError, ValueError)


# ---------------------------------------------------------------------------
# 3. Structural: no call site derives a probability from an edge
# ---------------------------------------------------------------------------


def _is_forbidden_expression(node: ast.AST) -> bool:
    """True when ``node`` adds an edge or an implied probability to something.

    Keyed on the OPERANDS rather than on the name of the result: renaming ``kelly_model_prob``
    would defeat a substring scan while leaving the defect exactly where it was.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    for operand in (node.left, node.right):
        if isinstance(operand, ast.Name) and operand.id in _FORBIDDEN_OPERANDS:
            return True
        if (
            isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Name)
            and operand.func.id == "moneyline_to_probability"
        ):
            return True
    return False


def _kelly_call_sites() -> list[dict[str, Any]]:
    """Every Kelly call site in the three sizing modules, with its probability argument.

    Each entry carries the module, the enclosing function, the line, the argument expression, and
    every assignment made to that argument's NAME inside the same function -- which is what lets
    the check follow a bare name back to the expression that produced it.
    """
    sites: list[dict[str, Any]] = []
    for relative in _SIZING_MODULES:
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                param = _KELLY_ENTRY_POINTS.get(node.func.attr)
                if param is None:
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords}
                argument = keywords.get(param) or (node.args[0] if node.args else None)
                assert argument is not None, (
                    f"{relative}:{node.lineno} calls {node.func.attr} with no probability"
                )
                assignments: list[ast.AST] = []
                if isinstance(argument, ast.Name):
                    for inner in ast.walk(function):
                        if isinstance(inner, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == argument.id
                            for t in inner.targets
                        ):
                            assignments.append(inner.value)
                sites.append(
                    {
                        "module": relative,
                        "function": function.name,
                        "line": node.lineno,
                        "entry_point": node.func.attr,
                        "argument": argument,
                        "assignments": assignments,
                    }
                )
    return sites


class TestNoCallSiteDerivesAProbabilityFromAnEdge:
    """T-31-44: the class of defect is closed structurally, not one instance at a time."""

    def test_the_scan_found_call_sites(self) -> None:
        """A structural scan that found nothing passes vacuously, so the count is asserted."""
        sites = _kelly_call_sites()
        assert len(sites) >= 2, (
            f"the Kelly call-site scan examined {len(sites)} sites across {_SIZING_MODULES}; "
            "fewer than two means the scan is not finding the sizing paths"
        )

    def test_every_kelly_probability_comes_from_a_calibrated_source(self) -> None:
        """No argument, and no assignment feeding one, adds an edge to anything.

        Failure names the module, the enclosing function and the line, so the report says WHERE
        rather than only THAT.
        """
        offenders: list[str] = []
        for site in _kelly_call_sites():
            where = (
                f"{site['module']}:{site['line']} in {site['function']}() -> "
                f"{site['entry_point']}"
            )
            if _is_forbidden_expression(site["argument"]):
                offenders.append(
                    f"{where}: the argument itself is an edge-derived expression"
                )
            for assignment in site["assignments"]:
                if _is_forbidden_expression(assignment):
                    offenders.append(
                        f"{where}: the value it passes is assigned from an edge-derived expression"
                    )
        assert offenders == [], (
            "a points-distance quantity can reach a Kelly probability argument:\n"
            + "\n".join(offenders)
        )

    def test_the_scan_can_actually_fail(self) -> None:
        """The negative control, on a synthetic module rather than on the real ones.

        Without it, a scan that silently matched nothing would look identical to a clean codebase.
        The executor separately re-introduced the real expression once in
        ``backtest/simulation.py``, confirmed this test went red, and reverted it before committing.
        """
        synthetic = ast.parse(
            "def size(self, edge, odds):\n"
            "    implied = moneyline_to_probability(odds)\n"
            "    prob = implied + edge\n"
            "    return self.calculate_optimal_bet_size(model_prob=prob, market_odds=odds)\n"
        )
        function = next(
            node for node in ast.walk(synthetic) if isinstance(node, ast.FunctionDef)
        )
        assignments = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "prob" for t in node.targets)
        ]
        assert any(_is_forbidden_expression(value) for value in assignments)
