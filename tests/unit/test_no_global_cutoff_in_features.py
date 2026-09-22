"""No builder selection in ``features/`` compares against a frame-wide scalar cutoff.

Plan 33.2-14 Task 2 (SPEC R5, D33.2-01). Every builder now selects each game's inputs at
that game's OWN lock. What this scan retires is the SHAPE the old fences had: a frame
column compared against ONE scalar -- the same instant for every game -- that arrived as a
function parameter (``as_of_datetime``, whose production value was ``datetime.now(ET)``).

WHY THE SCAN IS FOR A SHAPE AND NOT FOR ``datetime.now``. The cutoff arrives as a PARAMETER
and ``datetime.now(ET)`` lives in ``scripts/build_features.py``, outside ``features/``. A scan
for the literal ``datetime.now`` in ``features/`` passed BEFORE this plan and proves nothing
(``33.2-RESEARCH.md`` section 5). So the scan follows the value instead:

* TAINT SOURCES: a function parameter named in ``FRAME_WIDE_CUTOFF_PARAMETERS`` (the
  vocabulary every frame-wide fence in this tree used), and any ``*.now()`` / ``utcnow()``
  call. Taint flows through assignment.
* PER-GAME VALUES: a subscript of a lock mapping (``locks[game_id]``, ``lock_frame[...]``),
  a call to the lock rule (``game_lock(...)``, ``lock_frame(...)``), and a ``min(...)`` with
  at least one per-game argument. A per-game value is never tainted.
* THE FLAGGED SHAPE: a comparison with a tainted, non-per-game operand on either side (a
  comparison with a literal is not a selection and is ignored).

The ``min`` rule is the declared NO-FALSE-POSITIVE control (2): Plan 33.2-12 gives the weather
fence exactly ``forecast time <= min(the game's lock, the build instant)`` -- the per-game
lock bounded by when the build ran, so a pre-lock build cannot admit a bulletin that does
not exist yet. Its bound is DERIVED FROM the per-game lookup, so it is not the frame-wide
scalar cutoff R5 retires; a scan that flagged it would put this plan and rung 4 in
contradiction, and the cheapest route to green would be weakening whichever came second.

THE FOUR STRUCTURAL CONTROLS (``tests/unit/test_freeze_parse_single_source.py`` shape):
non-vacuity (a declared module count, and a non-empty set of comparisons resolved in every
module); the assertion; a planted violation that IS flagged; and two no-false-positive
cases that are NOT (the per-game lookup, and the ``min(lock, build instant)`` form).

``features/weather.py`` is IN the list: scanning a module is not owning it, and rung 4's
fence must satisfy the same scan as every other builder.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The builder modules whose selections must be per-game. DECLARED, never globbed, so a
#: module cannot join or leave the scan by accident.
SCANNED_MODULES: tuple[str, ...] = (
    "features/contextual.py",
    "features/injury.py",
    "features/market_anchors.py",
    "features/qb_tracking.py",
    "features/snaps.py",
    "features/team_form.py",
    "features/weather.py",
)
DECLARED_MODULE_COUNT = 7

#: The parameter names every frame-wide fence in this tree received its cutoff through.
FRAME_WIDE_CUTOFF_PARAMETERS: frozenset[str] = frozenset(
    {
        "as_of",
        "as_of_datetime",
        "as_of_utc",
        "build_instant",
        "cutoff",
        "cutoff_time",
        "cutoff_ts",
        "now",
    }
)

#: Calls that return a per-game lock.
_LOCK_RULE_CALLS: frozenset[str] = frozenset({"game_lock", "lock_frame"})

#: Calls that read the process clock.
_CLOCK_CALLS: frozenset[str] = frozenset({"now", "utcnow", "today"})


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _mentions_lock(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "lock" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "lock" in node.attr.lower()
    return False


@dataclass
class _Scope:
    tainted: set[str] = field(default_factory=set)
    per_game: set[str] = field(default_factory=set)

    def is_per_game(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.per_game
        if isinstance(node, ast.Subscript):
            return _mentions_lock(node.value) or self.is_per_game(node.value)
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _LOCK_RULE_CALLS:
                return True
            if name == "min":
                return any(self.is_per_game(arg) for arg in node.args)
            return any(self.is_per_game(arg) for arg in node.args) and not any(
                self.is_tainted(arg) for arg in node.args
            )
        if isinstance(node, ast.Attribute):
            return self.is_per_game(node.value)
        return False

    def is_tainted(self, node: ast.AST) -> bool:
        """True when *node* evaluates to (something derived from) the frame-wide cutoff.

        A SUBSCRIPT is tainted only through its VALUE, never through its selector: a
        frame filtered by a mask built from the cutoff is still a frame (its values are
        not the cutoff), so ``frame[keep]["game_id"] == game_id`` is not the shape --
        ``frame["ts"] <= cutoff`` is, and the filter that built ``keep`` is flagged
        where it is written.
        """
        if self.is_per_game(node):
            return False
        if isinstance(node, ast.Name):
            return node.id in self.tainted
        if isinstance(node, ast.Subscript):
            return self.is_tainted(node.value)
        if isinstance(node, ast.Call) and _call_name(node) in _CLOCK_CALLS:
            return True
        return any(self.is_tainted(child) for child in ast.iter_child_nodes(node))


def _targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        return names
    if isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(
        node.target, ast.Name
    ):
        return [node.target.id]
    if isinstance(node, ast.NamedExpr):
        return [node.target.id]
    return []


def _function_violations(
    function: ast.FunctionDef | ast.AsyncFunctionDef, inherited: _Scope
) -> tuple[list[int], int]:
    """(line numbers of flagged comparisons, comparisons resolved) for one function."""
    scope = _Scope(set(inherited.tainted), set(inherited.per_game))
    arguments = function.args
    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ):
        if argument.arg in FRAME_WIDE_CUTOFF_PARAMETERS:
            scope.tainted.add(argument.arg)
        else:
            scope.tainted.discard(argument.arg)
            scope.per_game.discard(argument.arg)

    own_nodes = [
        node
        for node in ast.walk(function)
        if node is not function and hasattr(node, "lineno")
    ]
    nested = [
        node
        for node in own_nodes
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
    ]
    inside_nested = {
        id(child) for inner in nested for child in ast.walk(inner) if child is not inner
    }
    body = sorted(
        (node for node in own_nodes if id(node) not in inside_nested),
        key=lambda node: (node.lineno, node.col_offset),
    )

    for node in body:
        value = getattr(node, "value", None)
        for name in _targets(node):
            if value is None:
                continue
            if scope.is_per_game(value):
                scope.per_game.add(name)
                scope.tainted.discard(name)
            elif scope.is_tainted(value):
                scope.tainted.add(name)
                scope.per_game.discard(name)

    flagged: list[int] = []
    comparisons = 0
    for node in body:
        if not isinstance(node, ast.Compare):
            continue
        comparisons += 1
        operands = [node.left, *node.comparators]
        if any(isinstance(op, ast.Constant) for op in operands):
            continue
        if any(scope.is_tainted(op) for op in operands):
            flagged.append(node.lineno)

    for inner in nested:
        if isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef):
            inner_flagged, inner_count = _function_violations(inner, scope)
            flagged.extend(inner_flagged)
            comparisons += inner_count
    return flagged, comparisons


def scan_source(source: str) -> tuple[list[int], int]:
    """(flagged comparison lines, comparisons resolved) over every top-level function."""
    tree = ast.parse(source)
    flagged: list[int] = []
    comparisons = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parents = [
            parent
            for parent in ast.walk(tree)
            if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef)
            and parent is not node
            and node in ast.walk(parent)
        ]
        if parents:
            continue  # nested: scanned inside its enclosing function
        found, count = _function_violations(node, _Scope())
        flagged.extend(found)
        comparisons += count
    return sorted(set(flagged)), comparisons


def _module_source(module: str) -> str:
    return (REPO_ROOT / module).read_text(encoding="utf-8")


class TestTheScanIsNotVacuous:
    def test_the_module_list_has_its_declared_length(self) -> None:
        assert len(SCANNED_MODULES) == DECLARED_MODULE_COUNT
        assert len(set(SCANNED_MODULES)) == DECLARED_MODULE_COUNT
        assert "features/weather.py" in SCANNED_MODULES

    def test_every_scanned_module_exists(self) -> None:
        for module in SCANNED_MODULES:
            assert (REPO_ROOT / module).is_file(), module

    def test_every_module_resolves_a_non_empty_set_of_comparisons(self) -> None:
        for module in SCANNED_MODULES:
            _, comparisons = scan_source(_module_source(module))
            assert comparisons > 0, (
                f"{module}: no comparison resolved -- a vacuous pass"
            )


class TestNoFrameWideScalarCutoffRemains:
    def test_no_scanned_module_compares_against_a_frame_wide_cutoff(self) -> None:
        offenders = {
            module: lines
            for module in SCANNED_MODULES
            if (lines := scan_source(_module_source(module))[0])
        }
        assert offenders == {}, (
            "a builder selection still compares a frame column against ONE scalar cutoff "
            f"derived from a parameter (module: lines): {offenders}. Select at each "
            "game's own lock (utils.game_lock) instead"
        )


class TestTheScanDiscriminates:
    def test_control_a_planted_parameter_cutoff_is_flagged(self) -> None:
        planted = textwrap.dedent(
            """
            def select(frame, cutoff):
                return frame[frame["ts"] < cutoff]
            """
        )
        flagged, _ = scan_source(planted)
        assert flagged == [3]

    def test_control_a_cutoff_derived_from_the_parameter_is_flagged(self) -> None:
        planted = textwrap.dedent(
            """
            import pandas as pd

            def select(frame, as_of_datetime):
                cutoff_ts = pd.Timestamp(as_of_datetime).tz_convert("UTC")
                return frame[frame["ts"] <= cutoff_ts]
            """
        )
        flagged, _ = scan_source(planted)
        assert flagged == [6]

    def test_control_the_process_clock_is_flagged(self) -> None:
        planted = textwrap.dedent(
            """
            from datetime import datetime

            def select(frame):
                return frame[frame["ts"] <= datetime.now()]
            """
        )
        flagged, _ = scan_source(planted)
        assert flagged == [5]

    def test_no_false_positive_a_per_game_lock_lookup(self) -> None:
        compliant = textwrap.dedent(
            """
            def select(frame, lock_frame, game_id):
                return frame[frame["ts"] <= lock_frame[game_id]]
            """
        )
        flagged, comparisons = scan_source(compliant)
        assert (flagged, comparisons) == ([], 1)

    def test_no_false_positive_the_lock_bounded_by_the_build_instant(self) -> None:
        """Rung 4's weather fence form: min(per-game lock, build instant) is per-game."""
        compliant = textwrap.dedent(
            """
            def select(frame, lock_frame, game_id, build_instant):
                bound = min(lock_frame[game_id], build_instant)
                return frame[frame["ts"] <= bound]

            def select_inline(frame, lock_frame, game_id, build_instant):
                return frame[frame["ts"] <= min(lock_frame[game_id], build_instant)]
            """
        )
        flagged, comparisons = scan_source(compliant)
        assert (flagged, comparisons) == ([], 2)

    def test_no_false_positive_a_frame_filtered_by_a_cutoff_mask(self) -> None:
        """Only the comparison against the cutoff is the shape, not later uses of the frame."""
        planted = textwrap.dedent(
            """
            def select(frame, cutoff, game_id):
                keep = frame["ts"] <= cutoff
                frame = frame[keep]
                return frame[frame["game_id"] == game_id]
            """
        )
        flagged, _ = scan_source(planted)
        assert flagged == [3]

    def test_a_bare_build_instant_is_still_flagged(self) -> None:
        """The min() rule is not a blanket pass for the build instant on its own."""
        planted = textwrap.dedent(
            """
            def select(frame, build_instant):
                return frame[frame["ts"] <= build_instant]
            """
        )
        flagged, _ = scan_source(planted)
        assert flagged == [3]
