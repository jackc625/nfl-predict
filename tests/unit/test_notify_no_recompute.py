"""No notify builder recomputes a stored figure (Phase 35 NOTIF-05, SPEC R6).

The digest and the push builders SELECT and FORMAT stored values; every EV, stake, return, payout,
rate and count was computed at cache build (or written by the run) and is only displayed here. Unit
formatting (a stored fraction shown as a percent) lives in the templates and ``api.presentation``,
never in a ``notify/*.py`` module.

The scan walks the AST of every ``notify/*.py`` file and flags:

* a binary operation (``+ - * / // % **`` ...) with an operand that names a figure field -- as a bare
  name (``per_bet_ev``), an attribute (``row.per_bet_ev``) or a subscript key
  (``row["per_bet_ev"]``);
* a ``sum(...)`` or ``.mean(...)`` / ``.sum(...)`` call whose arguments reference a figure field.

A negative control proves the scanner flags each of those forms, so a passing scan is evidence and
not a scanner that sees nothing. If the scan ever finds a real recomputation, the fix is to move the
formatting to a template or ``api.presentation`` -- never to loosen the scan.

Run this module:  uv run python -m pytest tests/unit/test_notify_no_recompute.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

NOTIFY_DIR = Path(__file__).resolve().parents[2] / "notify"

#: The stored figures no notify module may do arithmetic on.
FIGURE_FIELDS: frozenset[str] = frozenset(
    {
        "per_bet_ev",
        "stake_units",
        "flat_stake",
        "payout_flat",
        "realized_units",
        "hit_rate",
        "flat_return_units",
        "actual_return_units",
        "expected_return_units",
        "bets_graded",
        "wins",
        "losses",
        "pushes",
    }
)

_AGGREGATE_CALLS: frozenset[str] = frozenset({"sum", "mean"})


def _names_a_figure(node: ast.AST) -> bool:
    """True when *node* is a figure field as a name, an attribute or a constant subscript key."""
    if isinstance(node, ast.Name):
        return node.id in FIGURE_FIELDS
    if isinstance(node, ast.Attribute):
        return node.attr in FIGURE_FIELDS
    if isinstance(node, ast.Subscript):
        key = node.slice
        return isinstance(key, ast.Constant) and key.value in FIGURE_FIELDS
    return False


def _references_a_figure(node: ast.AST) -> bool:
    """True when any node under *node* names a figure field."""
    return any(_names_a_figure(child) for child in ast.walk(node))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def recomputations(source: str, filename: str = "<memory>") -> list[str]:
    """Every figure recomputation in *source*, as ``<file>:<line>: <code>``."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            flagged = _names_a_figure(node.left) or _names_a_figure(node.right)
        elif isinstance(node, ast.Call) and _call_name(node) in _AGGREGATE_CALLS:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            receiver = [node.func.value] if isinstance(node.func, ast.Attribute) else []
            flagged = any(_references_a_figure(arg) for arg in [*arguments, *receiver])
        else:
            continue
        if flagged:
            found.append(f"{filename}:{node.lineno}: {ast.unparse(node)}")
    return found


def test_no_notify_module_does_arithmetic_on_a_figure() -> None:
    modules = sorted(NOTIFY_DIR.glob("*.py"))
    assert modules, "no notify module was found, so the scan proves nothing"

    found = [
        hit
        for module in modules
        for hit in recomputations(module.read_text(encoding="utf-8"), module.name)
    ]

    assert found == [], "notify recomputes a stored figure:\n" + "\n".join(found)


def test_the_scan_flags_a_recomputation() -> None:
    """Negative control: each recomputation form is flagged, and a plain selection is not."""
    source = "\n".join(
        [
            'percent = row["per_bet_ev"] * 100',
            "percent = row.per_bet_ev * 100",
            'total = sum(r["payout_flat"] for r in rows)',
            "count = len(rows)",
            'chosen = [row for row in rows if row["arm"] == "live"]',
        ]
    )

    found = recomputations(source)

    assert [hit.split(": ", 1)[1] for hit in found] == [
        "row['per_bet_ev'] * 100",
        "row.per_bet_ev * 100",
        "sum((r['payout_flat'] for r in rows))",
    ]
