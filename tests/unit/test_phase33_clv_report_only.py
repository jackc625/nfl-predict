"""CLV reaches no selection, sizing or refusal decision. It stays report-only.

WHAT IS BEING DEFENDED (T-33-09)
--------------------------------
Closing-line value is this project's most seductive number. Phase 31 measured WP's CLV over
its own selected 2025 bets at ``+0.0918`` with ``p 6.8e-34`` while the same bets' ROI was
indistinguishable from zero -- the CLV-to-ROI divergence that is the milestone's headline
finding. A number that strong and that uncorrelated with money is exactly the number
somebody eventually uses to pick bets, and the moment it does, the report stops being a
report: the selection becomes partly a function of the metric used to judge it, and no
later reader can separate the two.

So CLV is REPORT-ONLY, by decision (D27-06, D27-12, D31: ``CLV_P_VALUE_IS_REPORT_ONLY``),
and this module makes the decision structural rather than remembered.

WHAT A SOURCE SCAN IS AND IS NOT
--------------------------------
A source scan is a STRUCTURAL guarantee that a SHAPE is impossible. It is never acceptance
evidence for a behaviour. This module proves that no selection module reads CLV outside a
pinned inventory, and that no CLV value is consumed by a decision-shaped expression. It
proves nothing about whether any particular bet was well chosen.

``models/deploy_gate.py`` IS DELIBERATELY OUTSIDE THE SCANNED SET
-----------------------------------------------------------------
In the deploy gate, CLV is the SUBJECT. ``clv_non_regression_passes:183``,
``_pooled_floor_reasons:593`` and ``evaluate_target:737`` exist to judge a candidate model's
closing-line value against the incumbent's -- that is the gate's whole definition, ruled
and re-ruled across Phases 25, 30 and 31. A scan that flagged the gate would be a scan a
future author had to weaken, and a weakened guard asserts nothing. The exclusion is RECORDED
in ``tests/phase33_state.CLV_OUT_OF_SCOPE_MODULES`` with its reason, and a test below
asserts both that the gate is out of scope and that its reason is stated -- an unexplained
exclusion is indistinguishable from an oversight.

TWO SCANS, BECAUSE "NO REFERENCE AT ALL" IS NOT THE TRUE PROPERTY
-----------------------------------------------------------------
An earlier draft of this gate asserted that NO scanned module references a CLV symbol at
all. Measured against live source, that cannot pass and must never be made to:
``backtest/bet_selector.py`` legitimately imports ``clv_significance`` and builds
``SelectionResult.clv_report`` -- AFTER the selection and sizing loops have run, from the
bets already chosen, into a field nothing reads back. Deleting that would delete the report,
not the risk. Weakening the scan until it passed would have been the worse answer; the
honest one is that the property has two halves:

1. INVENTORY (``clv_sites`` / ``scan_module_for_clv``). Every CLV reference in every scanned
   module is collected and asserted EQUAL, in both directions, to the pinned
   ``CLV_REPORT_ONLY_SITES``. The default for a reference nobody pinned is REJECT, so a new
   read anywhere in a selection module fails by name. A vanished site fails too: an
   exemption that no longer matches anything is cover nobody is entitled to.
2. DECISION POSITION (``scan_module_for_clv_decisions``). No CLV-tainted value may be
   CONSUMED BY A DECISION anywhere in a scanned module -- including inside the pinned sites.
   Tainting propagates to fixpoint, so laundering a CLV value through two locals does not
   escape it. A presence check (``if not x``, ``x is None``) reads whether CLV was measured,
   not what it was, and is not a decision about a bet; that boundary is drawn mechanically
   and stated here rather than left to judgement.

FOUR CONTROLS per scan, following ``tests/unit/test_p31_constants_isolation.py``'s form:
non-vacuity, the assertion, a PLANTED violation, and a no-false-positive control.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# Calls that RANK, FILTER or CHOOSE. A CLV value reaching one of these is deciding which
# bets survive or in what order they are taken, whatever the surrounding prose says.
RANKING_CALLS = frozenset(
    {
        "sorted",
        "min",
        "max",
        "filter",
        "sort_values",
        "nlargest",
        "nsmallest",
        "query",
        "rank",
        "idxmax",
        "idxmin",
        "argmax",
        "argmin",
        "where",
        "mask",
        "clip",
    }
)

# Assignment targets that ARE a selection, sizing or refusal outcome. A CLV value anywhere
# in the value of one of these has reached the decision by definition.
DECISION_TARGET_FRAGMENTS = (
    "stake",
    "size",
    "unit",
    "kelly",
    "fraction",
    "wager",
    "edge",
    "eligib",
    "select",
    "refus",
    "reject",
    "floor",
    "threshold",
    "admit",
    "qualif",
)


def _is_clv_symbol(name: object) -> bool:
    """True when *name* is a CLV identifier the scan treats as a read."""
    if not isinstance(name, str):
        return False
    return name in phase33_state.CLV_SYMBOLS or name.startswith(
        phase33_state.CLV_SYMBOL_PREFIX
    )


def _is_clv_value_key(name: object) -> bool:
    """True when *name* is a per-bet record key holding a CLV VALUE.

    ``"clv"`` is the per-bet column this codebase actually carries (``bet_selector`` sets
    it, ``weekly_bet_list`` copies it onto the row), and it does not match the ``clv_``
    prefix rule. Leaving it out would leave the dominant shape unscanned by the
    decision-position half, which is the half that matters.
    """
    return isinstance(name, str) and name in phase33_state.CLV_VALUE_KEYS


def clv_sites(path: Path) -> list[tuple[str, str, str]]:
    """Every CLV reference in *path*, as ``(innermost_scope, kind, symbol)``.

    The scope is a qualified name (``BetSelector._clv_report``) rather than a line number
    on purpose: a line number makes the pinned inventory brittle against any edit above it,
    and the question the inventory answers is WHERE IN THE DESIGN a CLV value is read, not
    on which line.

    Args:
        path: A Python source file.

    Returns:
        Distinct sites, sorted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[tuple[str, str, str]] = set()

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            label = inner or "<module>"
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if _is_clv_symbol(alias.name):
                        found.add((label, "importfrom", alias.name))
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if _is_clv_symbol(alias.name.split(".")[-1]):
                        found.add((label, "import", alias.name))
            elif isinstance(child, ast.Name) and _is_clv_symbol(child.id):
                found.add((label, "name", child.id))
            elif isinstance(child, ast.Attribute) and _is_clv_symbol(child.attr):
                found.add((label, "attr", child.attr))
            elif isinstance(child, ast.Constant) and _is_clv_symbol(child.value):
                found.add((label, "str", child.value))
            visit(child, inner)

    visit(tree, "")
    return sorted(found)


def scan_module_for_clv(path: Path) -> list[str]:
    """Every CLV reference in *path*, as human-readable hits.

    The importable-callable form the planted-violation controls drive, so the controls
    exercise the real code path rather than a copy of its logic.

    Args:
        path: A Python source file.

    Returns:
        One string per site, naming the module, the scope, the kind and the symbol.
    """
    label = path.as_posix()
    return [
        f"{label}: {scope} reads {symbol!r} ({kind})"
        for scope, kind, symbol in clv_sites(path)
    ]


# ---------------------------------------------------------------------------
# The decision-position half.
# ---------------------------------------------------------------------------


def _tainted_expression(node: ast.AST, tainted: set[str]) -> bool:
    """True when *node* reads a CLV symbol, a CLV record key, or a tainted local."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and (
            _is_clv_symbol(inner.id) or inner.id in tainted
        ):
            return True
        if isinstance(inner, ast.Attribute) and (
            _is_clv_symbol(inner.attr) or inner.attr in tainted
        ):
            return True
        if isinstance(inner, ast.Constant) and (
            _is_clv_symbol(inner.value) or _is_clv_value_key(inner.value)
        ):
            return True
    return False


def _assigned_names(node: ast.AST) -> list[str]:
    """The simple target names an assignment-like *node* binds."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(
        node,
        (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension),
    ):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(
                element.id for element in target.elts if isinstance(element, ast.Name)
            )
    return names


def _taint_closure(tree: ast.AST) -> set[str]:
    """Names that hold a CLV-derived value, propagated to fixpoint.

    Laundering a CLV value through two intermediate locals must not escape the scan, and a
    single pass in source order would miss a binding introduced after its first use (a
    helper defined below its caller). Iterating to fixpoint costs nothing on files this
    size and removes the ordering question entirely.
    """
    tainted: set[str] = set()
    while True:
        grown = False
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.comprehension),
            ):
                continue
            value = node.iter if isinstance(node, ast.comprehension) else node.value
            if value is None or not _tainted_expression(value, tainted):
                continue
            for name in _assigned_names(node):
                if name not in tainted:
                    tainted.add(name)
                    grown = True
        if not grown:
            return tainted


def _is_presence_test(test: ast.expr, tainted: set[str]) -> bool:
    """True when *test* asks WHETHER a CLV value exists, not WHAT it is.

    ``if not clv_values:`` and ``if report is None:`` read measuredness. That is a fact
    about the instrument, not about a bet, and it is what a report renderer legitimately
    branches on. The boundary is drawn on SHAPE -- a bare name, a ``not`` of a bare name, or
    an ``is``/``is not`` comparison against ``None`` -- so it cannot be widened by argument.
    """
    if isinstance(test, ast.Name):
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _is_presence_test(test.operand, tainted)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], (ast.Is, ast.IsNot)):
            operands = [test.left, test.comparators[0]]
            return any(
                isinstance(operand, ast.Constant) and operand.value is None
                for operand in operands
            )
    if isinstance(test, ast.BoolOp):
        return all(_is_presence_test(value, tainted) for value in test.values)
    return False


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def scan_module_for_clv_decisions(path: Path) -> list[str]:
    """Every place in *path* where a CLV value is CONSUMED BY A DECISION.

    Args:
        path: A Python source file.

    Returns:
        Human-readable hits, each naming the line and the decision shape.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tainted = _taint_closure(tree)
    label = path.as_posix()
    hits: list[str] = []

    def flag(node: ast.AST, shape: str) -> None:
        hits.append(f"{label}:{getattr(node, 'lineno', 0)}: {shape}")

    for node in ast.walk(tree):
        # A branch on the VALUE of a CLV number.
        tests: list[ast.expr] = []
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            tests = [node.test]
        elif isinstance(node, ast.comprehension):
            tests = list(node.ifs)
        for test in tests:
            if _tainted_expression(test, tainted) and not _is_presence_test(
                test, tainted
            ):
                flag(node, "branches on a CLV value")

        # Comparison or arithmetic: the value itself is being used.
        if isinstance(node, ast.Compare) and _tainted_expression(node, tainted):
            against_none = any(
                isinstance(operand, ast.Constant) and operand.value is None
                for operand in [node.left, *node.comparators]
            )
            if not against_none:
                flag(node, "compares a CLV value")
        if isinstance(node, ast.BinOp) and _tainted_expression(node, tainted):
            flag(node, "does arithmetic on a CLV value")

        # Ranking, filtering or choosing.
        if isinstance(node, ast.Call) and _call_name(node) in RANKING_CALLS:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(_tainted_expression(argument, tainted) for argument in arguments):
                flag(node, f"passes a CLV value to {_call_name(node)}()")

        # An assignment whose TARGET is a decision outcome.
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
            if value is not None and _tainted_expression(value, tainted):
                for name in _assigned_names(node):
                    lowered = name.lower()
                    if any(
                        fragment in lowered for fragment in DECISION_TARGET_FRAGMENTS
                    ):
                        flag(node, f"assigns a CLV value into {name!r}")

    return sorted(set(hits))


# ---------------------------------------------------------------------------
# Resolving the scanned set.
# ---------------------------------------------------------------------------


def scanned_modules() -> list[Path]:
    """The selection / sizing / refusal modules this gate covers.

    Every entry of ``CLV_REPORT_ONLY_MODULES`` must EXIST. A module named in the constant
    that is not on disk is a RECORDING ERROR, never a pass -- an earlier draft of this plan
    named ``backtest/kelly_criterion.py``, which does not exist in this repository, and a
    tolerant resolver would have scanned four modules while claiming five.
    """
    resolved: list[Path] = []
    missing: list[str] = []
    for relative in phase33_state.CLV_REPORT_ONLY_MODULES:
        candidate = REPO_ROOT / relative
        if candidate.is_file():
            resolved.append(candidate)
        else:
            missing.append(relative)
    assert not missing, (
        "module(s) named in tests/phase33_state.CLV_REPORT_ONLY_MODULES are not on disk:\n"
        + "\n".join(f"  - {name}" for name in missing)
        + "\n\nA path recorded here and absent from the tree is a recording error, not a "
        "clean scan: the scan would visit a shorter list and still report green."
    )
    return resolved


# ---------------------------------------------------------------------------
# Control 1: non-vacuity.
# ---------------------------------------------------------------------------


def test_the_scan_visits_a_non_empty_module_list() -> None:
    """Anti-vacuity: every assertion below is over this list, so it must not be empty."""
    modules = scanned_modules()
    assert modules, (
        "the CLV scan visited ZERO modules. Every assertion below would then pass while "
        "proving nothing."
    )
    assert len(modules) == len(phase33_state.CLV_REPORT_ONLY_MODULES) >= 5


def test_the_scanned_set_covers_the_real_selection_and_sizing_surface() -> None:
    """The modules were resolved against live ``git ls-files``, not from memory.

    Named individually because the set is the scope of the whole guard: a selection module
    absent from it is a module where CLV may do anything at all.
    """
    relative = set(phase33_state.CLV_REPORT_ONLY_MODULES)
    for expected in (
        "backtest/weekly_bet_list.py",
        "backtest/bet_selector.py",
        "utils/bet_selector.py",
        "utils/kelly_criterion.py",
        "scripts/generate_current_week_predictions.py",
    ):
        assert expected in relative, (
            f"{expected} is not in CLV_REPORT_ONLY_MODULES, so CLV is unscanned there."
        )


def test_the_deploy_gate_is_out_of_scope_and_the_reason_is_recorded() -> None:
    """An unexplained exclusion is indistinguishable from an oversight.

    CLV is the deploy gate's SUBJECT; the gate exists to judge a candidate's closing-line
    value against the incumbent's. That is why it is excluded, and the reason must travel
    with the exclusion rather than living only in a reviewer's memory.
    """
    excluded = dict(phase33_state.CLV_OUT_OF_SCOPE_MODULES)
    assert "models/deploy_gate.py" in excluded
    reason = excluded["models/deploy_gate.py"]
    assert "subject" in reason.lower(), reason
    assert len(reason) > 80, reason
    assert "models/deploy_gate.py" not in set(phase33_state.CLV_REPORT_ONLY_MODULES)
    assert (REPO_ROOT / "models" / "deploy_gate.py").is_file(), (
        "models/deploy_gate.py is recorded as deliberately out of scope but does not "
        "exist, so the recorded reason describes nothing."
    )
    # And the exclusion is stated in this module's own docstring, where a reader of the
    # gate will actually meet it.
    assert "models/deploy_gate.py" in (__doc__ or "")


# ---------------------------------------------------------------------------
# Control 2: the assertions.
# ---------------------------------------------------------------------------


def test_every_clv_reference_is_a_pinned_report_only_site() -> None:
    """Both directions against ``CLV_REPORT_ONLY_SITES``: unpinned is REJECT, stale is too.

    An EXTRA site is a new CLV read in a selection module -- the T-33-09 event. A MISSING
    site is an exemption that no longer matches anything, which is cover nobody is entitled
    to and which hides the fact that the design changed.
    """
    measured: set[tuple[str, str, str, str]] = set()
    for path in scanned_modules():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for scope, kind, symbol in clv_sites(path):
            measured.add((relative, scope, kind, symbol))

    expected = {tuple(site) for site in phase33_state.CLV_REPORT_ONLY_SITES}

    extra = measured - expected
    assert not extra, (
        "UNPINNED CLV READ(S) in a selection / sizing / refusal module (T-33-09):\n"
        + "\n".join(f"  - {site}" for site in sorted(extra))
        + "\n\nCLV is REPORT-ONLY (D27-06 / D27-12). If a new read is genuinely a report "
        "and not a decision, it is recorded in "
        "tests/phase33_state.CLV_REPORT_ONLY_SITES under the append protocol, with its "
        "reason -- and it must still pass the decision-position scan below."
    )

    missing = expected - measured
    assert not missing, (
        "pinned CLV site(s) no longer exist:\n"
        + "\n".join(f"  - {site}" for site in sorted(missing))
        + "\n\nA pinned site that matches nothing is a stale exemption. Either the design "
        "changed and the inventory must be re-recorded under the append protocol, or a "
        "report was deleted -- and deleting the report is not the same as removing the "
        "risk."
    )

    assert measured == expected


def test_no_clv_value_is_consumed_by_a_decision() -> None:
    """The property the inventory alone cannot give: CLV never REACHES a decision.

    Pinning a site says a read is legitimate today. This says the read cannot BECOME a
    decision tomorrow without the suite going red -- including inside the pinned sites,
    which is where a future author would most naturally add the branch.
    """
    hits: list[str] = []
    for path in scanned_modules():
        hits.extend(scan_module_for_clv_decisions(path))

    assert not hits, (
        "CLV VALUE REACHED A DECISION (T-33-09):\n"
        + "\n".join(f"  - {line}" for line in hits)
        + "\n\nClosing-line value is report-only. Phase 31 measured WP's CLV over its own "
        "selected 2025 bets at +0.0918 (p 6.8e-34) while the same bets' ROI was "
        "indistinguishable from zero: a metric that strong and that uncorrelated with "
        "money must not choose, size or refuse a bet, or the selection becomes partly a "
        "function of the number used to judge it."
    )


# ---------------------------------------------------------------------------
# Control 3: PLANTED violations.
# ---------------------------------------------------------------------------


def test_the_inventory_scan_flags_a_planted_clv_read(tmp_path: Path) -> None:
    """Fail-closed control: a planted CLV read IS reported, naming module and symbol.

    Each recognised shape is planted, so no branch of the collector is untested: the
    string-key form (how records are read in this codebase), the bare name, the attribute,
    and the import.
    """
    planted = tmp_path / "planted_selector.py"
    planted.write_text(
        "\n".join(
            [
                "from backtest.diagnose import clv_significance",
                "",
                "",
                "def choose(candidate, report):",
                '    x = candidate["clv_delta_values"]',
                "    y = clv_delta_values",
                "    z = report.clv_non_regression_passes",
                "    return x, y, z, clv_significance",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hits = scan_module_for_clv(planted)
    joined = "\n".join(hits)
    assert hits, (
        "the CLV inventory scan did NOT flag a module planted with four CLV reads. The "
        "real assertion above is therefore a check that has only ever been observed "
        "passing, which is indistinguishable from one that cannot fail."
    )
    assert "clv_delta_values" in joined, hits
    assert "clv_non_regression_passes" in joined, hits
    assert "clv_significance" in joined, hits
    assert "planted_selector.py" in joined, hits
    assert ("choose", "str", "clv_delta_values") in clv_sites(planted)


def test_the_decision_scan_flags_planted_decision_shapes(tmp_path: Path) -> None:
    """Fail-closed control on the stronger half: each decision shape is proven live.

    Five shapes, planted one per line: a branch on the value, a comparison, arithmetic, a
    ranking call, and an assignment into a sizing target. A shape nobody has watched fire
    is a shape the scan may not actually recognise.
    """
    planted = tmp_path / "planted_decisions.py"
    planted.write_text(
        "\n".join(
            [
                "def decide(rows, candidate):",
                '    value = candidate["clv"]',
                "    if value > 0.0:",
                "        pass",
                "    laundered = value",
                "    stake = laundered * 2.0",
                "    ranked = sorted(rows, key=lambda row: laundered)",
                "    return stake, ranked",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hits = scan_module_for_clv_decisions(planted)
    joined = "\n".join(hits)
    assert hits, (
        "the decision-position scan did NOT flag a module that branches on, compares, "
        "multiplies, ranks by and sizes from a CLV value."
    )
    assert "branches on a CLV value" in joined, hits
    assert "compares a CLV value" in joined, hits
    assert "does arithmetic on a CLV value" in joined, hits
    assert "sorted()" in joined, hits
    assert "'stake'" in joined, hits


def test_the_taint_survives_being_laundered_through_locals(tmp_path: Path) -> None:
    """A CLV value copied through two locals is still a CLV value.

    The obvious way past a naive scan is one rename. Fixpoint propagation is what closes
    it, and this is the control that shows the closure works rather than merely exists.
    """
    planted = tmp_path / "planted_launder.py"
    planted.write_text(
        "\n".join(
            [
                "def size_it(candidate):",
                "    first = candidate.clv_delta_values",
                "    second = first",
                "    third = second",
                "    units = third * 0.25",
                "    return units",
                "",
            ]
        ),
        encoding="utf-8",
    )

    joined = "\n".join(scan_module_for_clv_decisions(planted))
    assert "'units'" in joined, joined
    assert "does arithmetic on a CLV value" in joined, joined


# ---------------------------------------------------------------------------
# Control 4: no false positives.
# ---------------------------------------------------------------------------


def test_a_report_renderer_that_only_summarises_clv_is_not_flagged(
    tmp_path: Path,
) -> None:
    """Fail-open control: the legitimate report-only shape stays clean.

    This is ``BetSelector._clv_report``'s shape, reduced: gather the per-bet values AFTER
    the bets are chosen, branch only on whether any exist, summarise, attach. A scan that
    reddened on it would be one a later plan had to weaken, and the report would be deleted
    to satisfy the guard -- which removes the disclosure and none of the risk.
    """
    legitimate = tmp_path / "legitimate_report.py"
    legitimate.write_text(
        "\n".join(
            [
                "from backtest.diagnose import clv_significance",
                "",
                "",
                "def report_for(selected):",
                "    if not selected:",
                "        return None",
                '    clv_values = [r["clv"] for r in selected if r["clv"] is not None]',
                "    if not clv_values:",
                "        return None",
                "    report = dict(clv_significance(clv_values))",
                '    report["metric"] = "REPORT-ONLY"',
                "    return report",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert scan_module_for_clv_decisions(legitimate) == []
    # It IS an inventory site -- a read that must be pinned -- and it is NOT a decision.
    assert scan_module_for_clv(legitimate)


def test_a_selection_module_with_no_clv_at_all_is_clean(tmp_path: Path) -> None:
    """Fail-open control: ordinary EV-based selection and sizing is untouched.

    The guard forbids CLV reaching a decision, not decisions. ``utils/kelly_criterion.py``
    and ``backtest/weekly_bet_list.py`` are exactly this shape today and must stay silent.
    """
    clean = tmp_path / "clean_selector.py"
    clean.write_text(
        "\n".join(
            [
                "def select(rows, ev_floor):",
                "    eligible = [row for row in rows if row['ev'] >= ev_floor]",
                "    ranked = sorted(eligible, key=lambda row: row['ev'], reverse=True)",
                "    stake = 0.25 * ranked[0]['ev'] if ranked else 0.0",
                "    closing_line_weight = 0.7",
                "    return ranked, stake, closing_line_weight",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert scan_module_for_clv(clean) == []
    assert scan_module_for_clv_decisions(clean) == []


def test_the_symbol_and_phrase_constants_are_populated() -> None:
    """The constants the whole gate reads from are non-empty and shaped as documented."""
    assert len(phase33_state.CLV_SYMBOLS) >= 5
    assert phase33_state.CLV_SYMBOL_PREFIX == "clv_"
    assert "clv" in phase33_state.CLV_VALUE_KEYS
    assert phase33_state.CLV_REPORT_ONLY_SITES
    for site in phase33_state.CLV_REPORT_ONLY_SITES:
        assert len(site) == 4, site
        assert all(isinstance(field, str) and field for field in site), site


def test_every_recorded_clv_symbol_is_recognised_by_the_scanner() -> None:
    """A symbol in the constant that the recogniser does not match is dead weight.

    Deliberately NOT parameterised over the constant. A ``parametrize`` argument is
    evaluated at COLLECTION time, so a decorator reading ``phase33_state.CLV_SYMBOLS``
    turns an absent constant into a collection ERROR for the whole module rather than a
    failing test -- and a collection error is not a red test, it is a module that did not
    run. The loop keeps the per-symbol failure message without that hazard.
    """
    unrecognised = [
        symbol for symbol in phase33_state.CLV_SYMBOLS if not _is_clv_symbol(symbol)
    ]
    assert not unrecognised, (
        f"CLV_SYMBOLS contains {unrecognised!r} but the scanner does not recognise "
        "them, so recording them gives false assurance."
    )
