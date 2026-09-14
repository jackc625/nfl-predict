"""The bye-window ageing assertion is proven CAPABLE OF FAILING (D33-18, T-33-53).

WHY THIS MODULE IS THE DELIVERABLE AND THE PROOF IS NOT
-------------------------------------------------------
``features/team_form._select_dynamic_window`` is already correct: it is GAME-indexed,
so a bye contributes no game and the window it produced last week is the window it
produces this week. The rule holds. That is exactly what makes the sibling proof
dangerous -- an assertion written against a rule that already holds passes on the day
it is written and keeps passing whether or not it is wired to anything. A test that
cannot fail is indistinguishable from a test that is not connected, and this project
treats that as a defect rather than as a green tick.

So the deliverable here is not the assertion. It is the EVIDENCE THAT THE ASSERTION CAN
FAIL: the same callable, unchanged, run against a deliberately WEEK-INDEXED twin of the
selector, where it must and does raise.

BOTH ARMS LIVE IN ONE MODULE, ON PURPOSE
-----------------------------------------
A reader should be able to see, in one file, that the assertion passes against the real
selector and fails against the twin. Splitting the arms across modules is how one of
them quietly stops running.

THE ASSERTION IS IMPORTED, NEVER RE-TYPED
------------------------------------------
``assert_window_ages_not_resets`` comes from ``tests/helpers/window_assertions.py``, the
tier-neutral package. A re-typed copy is not the same assertion: it can be weakened here
without the sibling proof noticing, which would leave the control green and the thing it
controls unguarded. The helper deliberately lives in neither tier, because a unit module
importing from a sibling tier's directory would drag that tier's collection into a
process meant to run alone.

THE CONSTRUCTED SCHEDULES ARE CONSTRUCTED, AND THAT IS STATED RATHER THAN HIDDEN
--------------------------------------------------------------------------------
Two shapes in this module -- a team with TWO byes inside one window, and a team whose
early-season games are spaced unusually far apart -- do NOT occur in the 2026 feed. Each
team gets exactly one bye and plays every other week, so neither can be captured; they
are BUILT. A constructed case is weaker evidence than a measured one and is labelled as
such wherever it appears.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from features.team_form import TeamFormCalculator
from tests.helpers.window_assertions import assert_window_ages_not_resets, observe

REPO_ROOT = Path(__file__).resolve().parents[2]

SEASON = 2026
PRIOR_SEASON = 2025
MAX_PRIOR_GAMES = 8

# Every source tree that is production code. The twin below must be unreachable from
# all of them.
PRODUCTION_TREES = (
    "features",
    "scripts",
    "models",
    "api",
    "backtest",
    "pipeline",
    "ratings",
    "utils",
)

PLACEHOLDER_PREFIX = "WEEKSLOT"


# ---------------------------------------------------------------------------
# The deliberately WEEK-INDEXED twin. It exists only to fail.
# ---------------------------------------------------------------------------


def _week_indexed_window(
    team_group: pd.DataFrame,
    target_season: int,
    target_week: int,
    team: str,
    max_prior_games: int = MAX_PRIOR_GAMES,
) -> pd.DataFrame:
    """A NEGATIVE CONTROL. It must never be imported by production code.

    It is a deliberately WEEK-INDEXED twin of
    ``features/team_form.TeamFormCalculator._select_dynamic_window``, differing from it
    in exactly one dimension: where the real selector enumerates the GAMES a team has
    played, this one enumerates the WEEKS that have passed and emits one observation per
    week, inventing a zero-valued placeholder for any week the team did not play.
    Everything else is kept identical -- the same prior-season top-up to
    ``max_prior_games``, the same tail selection, the same chronological sort -- so that
    a failure can only be attributed to the indexing.

    This is the plausible wrong implementation, not a strawman. Enumerating weeks is
    the natural thing to write if one thinks of a rolling window as "the last eight
    weeks", and the bug it produces is invisible until a bye week arrives: the window
    acquires a row for a game that was never played, which drags every average over the
    window toward the placeholder's value.

    Args:
        team_group: The team's games, already filtered to before the target week.
        target_season: The season being predicted.
        target_week: The week being predicted.
        team: The team, carried onto any placeholder row.
        max_prior_games: The prior-season top-up budget.

    Returns:
        One row per week from week 1 to ``target_week - 1``, plus the prior-season tail.
    """
    current = team_group[team_group["season"] == target_season]
    played_by_week = {int(row["week"]): dict(row) for _index, row in current.iterrows()}

    slots: list[dict[str, object]] = []
    for week in range(1, target_week):
        if week in played_by_week:
            slots.append(played_by_week[week])
        else:
            # THE DEFECT, MADE EXPLICIT: a week with no game still gets a slot.
            slots.append(
                {
                    "game_id": f"{PLACEHOLDER_PREFIX}_{target_season}_{week:02d}_{team}",
                    "season": target_season,
                    "week": week,
                    "team": team,
                }
            )

    weeks_elapsed = target_week - 1
    prior_count = max(0, max_prior_games - weeks_elapsed)
    prior_tail = team_group[team_group["season"] == target_season - 1].tail(prior_count)

    combined = pd.concat([prior_tail, pd.DataFrame(slots)], ignore_index=True)
    return combined.sort_values(["season", "week"])


# ---------------------------------------------------------------------------
# Constructed schedules. Nothing here touches the lake.
# ---------------------------------------------------------------------------


def build_team_group(
    team: str,
    current_weeks: tuple[int, ...],
    prior_weeks: tuple[int, ...] = tuple(range(1, 18)),
) -> pd.DataFrame:
    """A team's game rows for a constructed prior season and current season.

    Args:
        team: The team.
        current_weeks: The weeks the team plays in ``SEASON``.
        prior_weeks: The weeks the team plays in ``PRIOR_SEASON``.

    Returns:
        Long-form rows, sorted chronologically.
    """
    rows = [
        {
            "game_id": f"{PRIOR_SEASON}_{week:02d}_{team}_PRIOR",
            "season": PRIOR_SEASON,
            "week": week,
            "team": team,
        }
        for week in prior_weeks
    ] + [
        {
            "game_id": f"{SEASON}_{week:02d}_{team}_CURRENT",
            "season": SEASON,
            "week": week,
            "team": team,
        }
        for week in current_weeks
    ]
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def before_target_week(group: pd.DataFrame, target_week: int) -> pd.DataFrame:
    """The leakage filter the production caller applies before selecting a window."""
    return group[
        (group["season"] < SEASON)
        | ((group["season"] == SEASON) & (group["week"] < target_week))
    ].sort_values(["season", "week"])


def real_observation(group: pd.DataFrame, team: str, target_week: int):
    """Drive the REAL selector and wrap the result as a ``WindowObservation``."""
    selected = TeamFormCalculator()._select_dynamic_window(
        before_target_week(group, target_week), SEASON, target_week
    )
    return observe(SEASON, target_week, selected.to_dict("records"))


def twin_observation(group: pd.DataFrame, team: str, target_week: int):
    """Drive the WEEK-INDEXED twin and wrap the result the same way."""
    selected = _week_indexed_window(
        before_target_week(group, target_week), SEASON, target_week, team
    )
    return observe(SEASON, target_week, selected.to_dict("records"))


# A single bye in week 11, matching the real 2026 shape the sibling proof measures.
SINGLE_BYE_WEEKS = tuple(week for week in range(1, 19) if week != 11)

# TWO byes inside one window. CONSTRUCTED: no 2026 team has two byes.
CONSECUTIVE_BYE_WEEKS = tuple(week for week in range(1, 19) if week not in (5, 6))

# Unusually wide early-season spacing. CONSTRUCTED: every 2026 team plays every week
# except its single bye, so this shape cannot be captured either.
UNUSUAL_SPACING_WEEKS = (1, 2, 5, 9, 13, 14, 15, 16, 17, 18)


# ---------------------------------------------------------------------------
# ARM ONE: the assertion PASSES against the real selector.
# ---------------------------------------------------------------------------


def test_the_shared_assertion_passes_against_the_real_selector() -> None:
    """The meta-control's positive arm, on a constructed single-bye schedule.

    The sibling proof runs this against the real captured 2026 season. It is repeated
    here, on constructed rows, so that both arms of the control are visible in one
    file: the reader does not have to take on trust that the assertion is satisfiable.
    """
    group = build_team_group("XXX", SINGLE_BYE_WEEKS)
    played = set(group["game_id"])
    assert_window_ages_not_resets(
        real_observation(group, "XXX", 11),
        real_observation(group, "XXX", 12),
        team="XXX",
        played_game_ids=played,
    )


# ---------------------------------------------------------------------------
# ARM TWO: the same assertion FAILS against the week-indexed twin.
# ---------------------------------------------------------------------------


def test_the_shared_assertion_fails_against_the_week_indexed_twin() -> None:
    """D33-18 / T-33-53: the assertion has discriminating power, demonstrated.

    Identical inputs, identical assertion, one substitution -- weeks for games. The
    twin hands the week-12 window an eleventh observation for a game that was never
    played, so the COUNT property raises. This is the whole reason the sibling proof's
    green means anything.
    """
    group = build_team_group("XXX", SINGLE_BYE_WEEKS)
    with pytest.raises(AssertionError) as excinfo:
        assert_window_ages_not_resets(
            twin_observation(group, "XXX", 11),
            twin_observation(group, "XXX", 12),
            team="XXX",
            played_game_ids=set(group["game_id"]),
        )
    assert "observation COUNT moved from 10 to 11" in str(excinfo.value), str(
        excinfo.value
    )


def test_the_twin_manufactures_a_row_for_the_bye_week_and_the_real_selector_does_not() -> (
    None
):
    """The mechanism behind the failure, named rather than left to the message.

    The twin's week-12 window carries a ``WEEKSLOT`` row dated to the bye week. The
    real selector's does not carry it, because it never enumerated the week in the
    first place. Asserting the mechanism -- not only that something raised -- is what
    stops this control from passing for the wrong reason later.
    """
    group = build_team_group("XXX", SINGLE_BYE_WEEKS)
    played = set(group["game_id"])

    twin_ids = {game_id for game_id, _s, _w in twin_observation(group, "XXX", 12).games}
    real_ids = {game_id for game_id, _s, _w in real_observation(group, "XXX", 12).games}

    manufactured = sorted(twin_ids - played)
    assert manufactured == [f"{PLACEHOLDER_PREFIX}_{SEASON}_11_XXX"], manufactured
    assert not real_ids - played, sorted(real_ids - played)


def test_the_manufactured_row_arm_of_the_assertion_fires_on_its_own() -> None:
    """Property 5 is independently live, not only reachable behind the count check.

    A padded window whose count happens to match would sail past the count assertion.
    This drives the ``played_game_ids`` arm directly, with the counts made equal, so
    the padding check is known to fire by itself.
    """
    before = observe(SEASON, 11, [{"game_id": "G1", "season": SEASON, "week": 10}])
    padded = observe(
        SEASON,
        12,
        [{"game_id": f"{PLACEHOLDER_PREFIX}_x", "season": SEASON, "week": 9}],
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_window_ages_not_resets(
            before, padded, team="XXX", played_game_ids={"G1"}
        )
    assert "never played" in str(excinfo.value), str(excinfo.value)


# ---------------------------------------------------------------------------
# The constructed edge shapes (Codex LOW).
# ---------------------------------------------------------------------------


def test_two_byes_inside_one_window_still_age_rather_than_reset() -> None:
    """CONSTRUCTED: a team on bye in consecutive weeks 5 and 6.

    No 2026 team has two byes -- an 18-week season of 17 games gives each team exactly
    one -- so this shape is built rather than captured, and it is weaker evidence than
    the sibling proof's measured week-11 case for exactly that reason. It is included
    because the ageing rule should not depend on byes being isolated, and a GAME-indexed
    selector gives that for free: two missing games are two games that were never
    counted.
    """
    group = build_team_group("XXX", CONSECUTIVE_BYE_WEEKS)
    played = set(group["game_id"])
    for before_week, after_week in ((5, 6), (6, 7), (5, 7)):
        assert_window_ages_not_resets(
            real_observation(group, "XXX", before_week),
            real_observation(group, "XXX", after_week),
            team="XXX",
            played_game_ids=played,
        )


def test_unusually_spaced_early_season_games_still_age_rather_than_reset() -> None:
    """CONSTRUCTED: a team whose early-season games sit weeks apart.

    Generalises the short-rest-versus-post-bye regression Antigravity asked for into a
    controllable fixture: the property at stake is window selection under irregular
    spacing, and a constructed schedule states the spacing outright instead of hunting
    the real feed for a pair that happens to exhibit it. Built, not captured -- every
    2026 team plays every week but one.
    """
    group = build_team_group("XXX", UNUSUAL_SPACING_WEEKS)
    played = set(group["game_id"])
    for before_week, after_week in ((6, 9), (10, 13), (3, 5)):
        assert_window_ages_not_resets(
            real_observation(group, "XXX", before_week),
            real_observation(group, "XXX", after_week),
            team="XXX",
            played_game_ids=played,
        )


def test_the_constructed_shapes_really_are_absent_from_the_single_bye_shape() -> None:
    """Non-vacuity: the constructed schedules differ from the real one they stand in for.

    A "constructed edge case" that is really the ordinary case tests nothing. The two
    shapes are asserted to be what they claim: two missing weeks, and gaps wider than
    one week.
    """
    assert len(set(range(1, 19)) - set(CONSECUTIVE_BYE_WEEKS)) == 2
    assert len(set(range(1, 19)) - set(SINGLE_BYE_WEEKS)) == 1
    gaps = [second - first for first, second in pairwise(UNUSUAL_SPACING_WEEKS)]
    assert max(gaps) >= 3, gaps


# ---------------------------------------------------------------------------
# T-33-56: the twin is unreachable from production code.
# ---------------------------------------------------------------------------


def test_no_production_module_references_the_week_indexed_twin() -> None:
    """A repo-wide search, because a first-line docstring is a request, not a guard.

    The twin is a deliberately wrong implementation living in the test tree. If it ever
    became reachable from ``features/`` or ``scripts/`` it would be a real defect
    shipped under the protection of a name that says it is a control.
    """
    result = subprocess.run(
        ["git", "grep", "-n", "_week_indexed_window", "--", *PRODUCTION_TREES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", (
        "the week-indexed negative control is referenced from production code:\n"
        + result.stdout
    )
    assert result.returncode in (0, 1), result.stderr


def test_the_twin_declares_itself_a_negative_control_on_its_first_docstring_line() -> (
    None
):
    """The declaration is asserted, so it cannot be edited away without a failure."""
    first_line = (_week_indexed_window.__doc__ or "").strip().splitlines()[0]
    assert "negative control" in first_line.lower(), first_line
    assert (
        "never be imported by production code"
        in (_week_indexed_window.__doc__ or "").lower()
    )
