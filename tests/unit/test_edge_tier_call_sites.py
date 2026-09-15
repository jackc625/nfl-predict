"""NO production call site omits the edge band's ``target`` (Plan 33-17 Task 1; T-33-84).

WHY A SCAN AND NOT ONLY A SIGNATURE
------------------------------------
``edge_tier``'s ``target`` has no default, so a DIRECT call that omits it raises ``TypeError``
at once. That is not the whole risk. The band also reaches a column through
``Series.apply(edge_tier, ...)``, where the target travels as a forwarded keyword and an
omission surfaces only when the frame is non-empty -- so on an empty weekly slate it would not
surface at all. This module reads both call shapes out of the source and asserts the target is
supplied at every one.

WHY AST AND NOT TEXT (Codex LOW, folded into the plan)
-------------------------------------------------------
A missing argument is detected by inspecting an ``ast.Call``'s ``args`` and ``keywords``, never
by pattern-matching source text. A regex over a call spanning three lines, or one that happens
to mention ``target`` in a neighbouring comment, gives an answer nobody can trust in either
direction.

WHY A COMMITTED SET
--------------------
``tests.phase33_state.EDGE_TIER_CALL_SITES`` records the SIX sites that exist. Comparing
against it makes a SEVENTH site a failure as loudly as a broken one: a new consumer means the
band's reach has changed, and ``EDGE_TIER_DISPLAY_ONLY_CORRECTED``'s conclusion -- that the
band moves a printed label and changes NO BET -- must be re-established rather than inherited.

THE FOUR CONTROLS, in the shape ``tests/unit/test_p31_constants_isolation.py`` established:
non-vacuity over the scanned file list, the assertion itself, a PLANTED omission proving the
scan fires, and a no-false-positive case.

Run this module:  uv run pytest tests/unit/test_edge_tier_call_sites.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.phase33_state import EDGE_TIER_CALL_SITES

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two production consumers, taken from the committed set rather than re-listed, so the
# scanned file list and the expected site list cannot drift apart.
SCANNED_FILES: tuple[str, ...] = tuple(
    dict.fromkeys(path for path, _c, _t in EDGE_TIER_CALL_SITES)
)

_BANDING_NAMES: frozenset[str] = frozenset({"edge_tier", "edge_tier_series"})


def _callee_name(node: ast.AST) -> str | None:
    """The bare name a call's ``func`` resolves to, for a Name or an Attribute tail."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_target(value: ast.AST | None) -> str | None:
    """A string constant, or None for anything this scan will not vouch for."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def banding_call_sites(
    source: str, label: str
) -> list[tuple[str, int, str, str | None]]:
    """Every edge-band call in *source*, with the target it supplies or None.

    Two call shapes are recognised, because both are live in the tree:

      DIRECT    ``edge_tier_series(frame["wp_edge"], "wp")`` -- the target is the second
                positional argument or a ``target=`` keyword.
      DISPATCH  ``frame["wp_edge"].apply(edge_tier, target="wp")`` -- the band travels as a
                bare function OBJECT and the target rides pandas' forwarded keywords, so the
                call's own ``func`` is ``apply`` and the band is one of its arguments.

    A returned target of None is the finding: the site bands a column against SOME pair without
    saying which unit it is in.

    Args:
        source: Python source text.
        label: A name for the source, used in the returned tuples.

    Returns:
        ``(label, lineno, callee, target_or_None)`` per site, in source order.
    """
    found: list[tuple[str, int, str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue

        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        callee = _callee_name(node.func)

        if callee in _BANDING_NAMES:
            positional = node.args[1] if len(node.args) > 1 else None
            target = _constant_target(positional) or _constant_target(
                keywords.get("target")
            )
            found.append((label, node.lineno, callee, target))
            continue

        dispatched = [
            argument.id
            for argument in node.args
            if isinstance(argument, ast.Name) and argument.id in _BANDING_NAMES
        ]
        if not dispatched:
            continue
        # ``Series.apply(func, **kwargs)`` forwards kwargs to func, so ``target=`` here IS the
        # band's target. ``args=("wp",)`` is the positional equivalent and is read too, so a
        # future author choosing the other spelling is not reported as an omission.
        target = _constant_target(keywords.get("target"))
        if target is None and isinstance(keywords.get("args"), ast.Tuple):
            elements = keywords["args"].elts  # type: ignore[union-attr]
            target = _constant_target(elements[0]) if elements else None
        found.append((label, node.lineno, dispatched[0], target))

    return found


def _live_sites() -> list[tuple[str, int, str, str | None]]:
    sites: list[tuple[str, int, str, str | None]] = []
    for relative_path in SCANNED_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        sites.extend(banding_call_sites(source, relative_path))
    return sites


# ---------------------------------------------------------------------------
# Control 1 -- non-vacuity
# ---------------------------------------------------------------------------


def test_the_scan_visits_both_production_consumers_and_finds_six_sites() -> None:
    """Anti-vacuity. "No site omits the target" is trivially true of a scan that found none."""
    assert len(SCANNED_FILES) == 2, SCANNED_FILES
    for relative_path in SCANNED_FILES:
        assert (REPO_ROOT / relative_path).is_file(), relative_path

    sites = _live_sites()
    assert len(sites) == 6, (
        "the edge-band scan found "
        f"{len(sites)} call site(s), not 6: {[(p, line, c) for p, line, c, _t in sites]}. "
        "A site that vanished and a site that arrived are equally a change in the band's "
        "reach (tests.phase33_state.EDGE_TIER_DISPLAY_ONLY_CORRECTED)."
    )


# ---------------------------------------------------------------------------
# Control 2 -- the assertion
# ---------------------------------------------------------------------------


def test_every_production_call_site_supplies_a_target() -> None:
    """T-33-84. An omitted target bands one unit against another unit's thresholds."""
    omissions = [
        f"{path}:{lineno} {callee}(...) supplies no target"
        for path, lineno, callee, target in _live_sites()
        if target is None
    ]
    assert not omissions, (
        "an edge-band call site does not say which unit its edge is in:\n"
        + "\n".join(f"  - {line}" for line in omissions)
    )


def test_the_live_sites_match_the_committed_set_exactly() -> None:
    """A seventh consumer is a finding, not a silent extension."""
    live = sorted(
        (path, callee, target) for path, _line, callee, target in _live_sites()
    )
    assert live == sorted(EDGE_TIER_CALL_SITES), (
        f"live edge-band call sites {live} disagree with the committed set "
        f"{sorted(EDGE_TIER_CALL_SITES)}"
    )


def test_each_committed_target_is_a_known_one() -> None:
    """A site supplying "spread" would pass the omission check and refuse at runtime."""
    from utils.edge_tier import EDGE_TIER_TARGETS

    for path, callee, target in EDGE_TIER_CALL_SITES:
        assert target in EDGE_TIER_TARGETS, (path, callee, target)


# ---------------------------------------------------------------------------
# Control 3 -- a PLANTED omission makes the scan fire
# ---------------------------------------------------------------------------


def test_the_scan_reports_a_planted_omission_in_both_call_shapes() -> None:
    """Fail-closed. A scan observed only passing is indistinguishable from a dead one."""
    planted = (
        'merged["wp_confidence"] = edge_tier_series(merged["wp_edge"])\n'
        'merged["ats_confidence"] = merged["ats_edge"].apply(edge_tier)\n'
    )
    reported = banding_call_sites(planted, "<planted>")
    assert len(reported) == 2, reported
    assert all(target is None for _p, _line, _c, target in reported), reported


def test_the_scan_reports_a_non_constant_target_as_an_omission() -> None:
    """A target computed at runtime cannot be vouched for by a source scan, so it is reported.

    Erring toward reporting is the right direction: the cost is one conversation, and the cost
    of the other direction is a published label banded against the wrong unit.
    """
    planted = (
        'merged["ou_confidence"] = edge_tier_series(merged["ou_edge"], chosen_target)\n'
    )
    reported = banding_call_sites(planted, "<planted>")
    assert reported and reported[0][3] is None, reported


# ---------------------------------------------------------------------------
# Control 4 -- the legitimate forms are NOT reported
# ---------------------------------------------------------------------------


def test_the_scan_does_not_flag_a_correctly_targeted_call() -> None:
    """Fail-open. A scan that reddened on the correct form is one a future author weakens."""
    legitimate = (
        'merged["wp_confidence"] = edge_tier_series(merged["wp_edge"], "wp")\n'
        'merged["ats_confidence"] = edge_tier_series(merged["ats_edge"], target="ats")\n'
        'merged["ou_confidence"] = merged["ou_edge"].apply(edge_tier, target="ou")\n'
        'merged["x"] = merged["x_edge"].apply(edge_tier, args=("wp",))\n'
    )
    reported = banding_call_sites(legitimate, "<legitimate>")
    assert len(reported) == 4, reported
    assert [target for _p, _line, _c, target in reported] == [
        "wp",
        "ats",
        "ou",
        "wp",
    ], reported


def test_the_scan_ignores_unrelated_calls() -> None:
    """An import line and a neighbouring call must not be counted as banding sites."""
    unrelated = (
        "from utils.edge_tier import edge_tier\n"
        'merged["blended_wp"] = blend(merged["wp_edge"], weights)\n'
        "assign_ev_tier(record)\n"
    )
    assert banding_call_sites(unrelated, "<unrelated>") == []
