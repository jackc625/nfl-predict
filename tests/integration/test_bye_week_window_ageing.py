"""R12 on the real 2026 season: a bye team's rolling window AGES, it does not RESET.

THE CLAIM, AND WHY IT IS WORTH PROVING ON REAL DATA
---------------------------------------------------
``features/team_form.TeamFormCalculator._select_dynamic_window`` chooses which past
games feed a team's rolling form metrics. Byes are handled by NO branch in it -- there
is no ``if bye`` anywhere -- so what happens in a bye week is whatever falls out of how
the selector indexes. This module measures what actually falls out, on the captured
2026 schedule, for the six teams that really are on bye in week 11.

It falls out CORRECTLY, and the reason is that the selector is GAME-indexed rather than
week-indexed: it asks how many games a team has played, never how many weeks have
passed. A bye contributes no game, so the window it produced last week is the window it
produces this week -- the same rows, one week older.

THE WEEK PAIR IS (11, 12), AND THE PLAN ASKED FOR (10, 11)
-----------------------------------------------------------
The correction is recorded in full beside ``BYE_WEEK_FIXTURE`` in
``tests/phase33_state.py`` and is not repeated here. In one line: for target week W the
selector takes the games of weeks 1..W-1, so a week-11 bye first appears in the window
selected for week 12. Comparing weeks 10 and 11 would have measured an ordinary week of
football and recorded the result as a proof of R12.

WHAT IS ASSERTED, AND WHAT IS DELIBERATELY NOT
-----------------------------------------------
COUNT, SPAN and ROW IDENTITY are three separate claims with three separate messages;
the shared assertion in ``tests/helpers/window_assertions.py`` owns all three plus the
no-reset and no-manufactured-row properties, and the unit tier imports the SAME
function so the two tiers cannot drift apart.

SPAN GROWTH IS NOT THE DISCRIMINATOR, AND SAYING SO IS PART OF THE RECORD. Both a bye
team and its control gain a week of span between week 11 and week 12, because the week
advanced for both. What separates them is COUNT and IDENTITY: the bye team's window
holds the same ten games, the control's gains an eleventh. The control arm below
asserts that the shared assertion FAILS for the control, which is what makes the effect
attributable to the bye rather than to the week.

THE WINDOW IS DRIVEN ON SCHEDULE ROWS, NOT ON PLAY-BY-PLAY
-----------------------------------------------------------
``_select_dynamic_window`` reads only ``season`` and ``week`` and returns whole rows, so
driving it on one row per team per game is the same computation the production caller
performs -- with the metric columns, which the selector never inspects, left off. 2026
has no play-by-play yet; that is the cold-start condition this phase exists for.

``features/team_form.py`` IS NOT MODIFIED BY THIS PLAN. The rule is already correct and
the deliverable is the proof. See ``tests/unit/test_bye_window_negative_control.py`` for
the control that makes the proof capable of failing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.team_form import TeamFormCalculator
from tests import phase33_state
from tests.fixtures.season_2026 import (
    CapturedScheduleUnavailableError,
    bye_teams_by_week,
    load_captured_schedule,
)
from tests.helpers.window_assertions import (
    assert_window_ages_not_resets,
    observe,
    window_span,
)

SEASON = 2026
PRIOR_SEASON = 2025

CAPTURE_MISSING_SKIP = (
    "the bye-window ageing proof needs the captured 2026 schedule, which lives under "
    "the gitignored production store and does not travel with the repository. This is "
    "a control that did NOT run on this checkout, not a control that passed."
)

BYE_TEAMS = tuple(team for team, _ids in phase33_state.BYE_WEEK_EXPECTED_GAME_IDS)
EXPECTED_IDS = dict(phase33_state.BYE_WEEK_EXPECTED_GAME_IDS)


# ---------------------------------------------------------------------------
# Driving the real selector.
# ---------------------------------------------------------------------------


def team_game_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: ``game_id``, ``season``, ``week``, ``team``.

    Args:
        schedule: A schedule frame carrying ``home_team`` and ``away_team``.

    Returns:
        The long-form frame, sorted chronologically, as the selector expects.
    """
    parts = [
        schedule[["game_id", "season", "week", side]].rename(columns={side: "team"})
        for side in ("home_team", "away_team")
    ]
    rows = pd.concat(parts, ignore_index=True)
    rows["season"] = rows["season"].astype(int)
    rows["week"] = rows["week"].astype(int)
    return rows.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def select_window(
    rows: pd.DataFrame,
    team: str,
    target_week: int,
    target_season: int = SEASON,
) -> pd.DataFrame:
    """Run the REAL ``_select_dynamic_window`` for one team at one target week.

    The pre-filter to games strictly before the target week is the caller's job in
    production too (``calculate_rolling_averages`` does it before grouping), so it is
    reproduced here rather than assumed away.

    Args:
        rows: Long-form team-game rows.
        team: The team.
        target_week: The week being predicted.
        target_season: The season being predicted.

    Returns:
        The selected window, exactly as the production selector returns it.
    """
    group = rows[rows["team"] == team]
    group = group[
        (group["season"] < target_season)
        | ((group["season"] == target_season) & (group["week"] < target_week))
    ]
    group = group.sort_values(["season", "week"])
    return TeamFormCalculator()._select_dynamic_window(
        group, target_season, target_week
    )


def observed(rows: pd.DataFrame, team: str, target_week: int):
    """The selected window as a tier-neutral ``WindowObservation``."""
    selected = select_window(rows, team, target_week)
    return observe(SEASON, target_week, selected.to_dict("records"))


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schedule() -> pd.DataFrame:
    """The captured 2026 schedule, or an evidence-backed skip."""
    try:
        return load_captured_schedule()
    except CapturedScheduleUnavailableError:
        pytest.skip(CAPTURE_MISSING_SKIP)


@pytest.fixture(scope="module")
def rows(schedule: pd.DataFrame) -> pd.DataFrame:
    """Long-form team-game rows for the captured 2026 season."""
    return team_game_rows(schedule)


@pytest.fixture(scope="module")
def byes(schedule: pd.DataFrame) -> dict[int, frozenset[str]]:
    """The bye sets DERIVED from the capture, not the recorded copy of them."""
    return bye_teams_by_week()


# ---------------------------------------------------------------------------
# The positive proof, one property at a time.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("team", BYE_TEAMS)
def test_the_observation_count_does_not_move_across_the_bye(
    rows: pd.DataFrame, team: str
) -> None:
    """COUNT: the window holds the same number of games at week 11 and week 12.

    One of the three properties, asserted alone so that a failure names which one
    moved. A bye contributes no game, so a GAME-indexed selector has nothing to add.
    """
    before = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE)
    after = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE_AFTER)
    assert len(after.games) == len(before.games), (
        f"{team} is on bye in week {phase33_state.BYE_WEEK_FIXTURE} and played no game "
        f"that week, yet its window count moved from {len(before.games)} to "
        f"{len(after.games)}."
    )
    assert len(after.games) == len(EXPECTED_IDS[team])


@pytest.mark.parametrize("team", BYE_TEAMS)
def test_the_window_span_grows_across_the_bye(rows: pd.DataFrame, team: str) -> None:
    """SPAN: the distance from the earliest selected game to the current week grows.

    This is the half of "ageing" the count cannot show. The window stood still while
    the calendar did not, so its contents are one week older than they were.
    """
    before = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE)
    after = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE_AFTER)
    span_before = window_span(before)
    span_after = window_span(after)
    assert span_after > span_before, (
        f"{team}: span did not grow across the bye ({span_before} -> {span_after})."
    )
    assert (span_before, span_after) == (
        phase33_state.BYE_WEEK_SPAN_BEFORE,
        phase33_state.BYE_WEEK_SPAN_AFTER,
    ), (
        f"{team}: the measured span pair ({span_before}, {span_after}) is not the "
        f"recorded pair ({phase33_state.BYE_WEEK_SPAN_BEFORE}, "
        f"{phase33_state.BYE_WEEK_SPAN_AFTER})."
    )


@pytest.mark.parametrize("team", BYE_TEAMS)
def test_the_selected_game_ids_are_the_pinned_set_at_both_weeks(
    rows: pd.DataFrame, team: str
) -> None:
    """ROW IDENTITY: the exact games, at BOTH target weeks, against a committed set.

    A window holding the right NUMBER of the WRONG games satisfies a count assertion
    completely. Asserting the set at both weeks is also the sharpest possible statement
    of the claim: the selection did not merely keep its size, it kept its rows.
    """
    expected = set(EXPECTED_IDS[team])
    before_ids = {game_id for game_id, _s, _w in observed(rows, team, 11).games}
    after_ids = {game_id for game_id, _s, _w in observed(rows, team, 12).games}
    assert before_ids == expected, (
        f"{team}: the week-{phase33_state.BYE_WEEK_FIXTURE} window is not the pinned "
        f"set.\n  unexpected: {sorted(before_ids - expected)}\n"
        f"  missing:    {sorted(expected - before_ids)}"
    )
    assert after_ids == expected, (
        f"{team}: the week-{phase33_state.BYE_WEEK_FIXTURE_AFTER} window is not the "
        f"pinned set.\n  unexpected: {sorted(after_ids - expected)}\n"
        f"  missing:    {sorted(expected - after_ids)}"
    )


@pytest.mark.parametrize("team", BYE_TEAMS)
def test_the_window_ages_rather_than_resets(rows: pd.DataFrame, team: str) -> None:
    """The shared assertion, run end to end on the real pair -- R12 itself.

    The same callable the unit tier runs against a deliberately week-indexed twin,
    where it MUST fail. That is what makes this green meaningful rather than merely
    green.
    """
    played = set(rows[rows["team"] == team]["game_id"])
    assert_window_ages_not_resets(
        observed(rows, team, phase33_state.BYE_WEEK_FIXTURE),
        observed(rows, team, phase33_state.BYE_WEEK_FIXTURE_AFTER),
        team=team,
        expected_game_ids=EXPECTED_IDS[team],
        played_game_ids=played,
    )


@pytest.mark.parametrize("team", BYE_TEAMS)
def test_no_window_contains_a_manufactured_bye_week_observation(
    rows: pd.DataFrame, team: str
) -> None:
    """No placeholder row is invented for the week the team did not play.

    This is the property that separates AGEING from RESETTING-WITH-PADDING. A zero-
    valued stand-in for the bye would keep the count steady -- satisfying the count
    assertion above -- while dragging every average in the window toward zero. Every
    selected row must be a real game the team really played, and none of them may sit
    in the bye week.
    """
    played = set(rows[rows["team"] == team]["game_id"])
    window = select_window(rows, team, phase33_state.BYE_WEEK_FIXTURE_AFTER)

    manufactured = sorted(set(window["game_id"]) - played)
    assert not manufactured, (
        f"{team}: the window holds {len(manufactured)} game(s) the team never played: "
        f"{manufactured}."
    )
    in_bye_week = window[window["week"] == phase33_state.BYE_WEEK_FIXTURE]
    assert in_bye_week.empty, (
        f"{team}: the window holds {len(in_bye_week)} row(s) dated to week "
        f"{phase33_state.BYE_WEEK_FIXTURE}, the week it was on bye: "
        f"{sorted(in_bye_week['game_id'])}. A row for a game that was not played is a "
        "placeholder, whatever value it carries."
    )
    assert window["game_id"].is_unique


# ---------------------------------------------------------------------------
# The control, which is what makes the effect attributable to the bye.
# ---------------------------------------------------------------------------


def test_the_control_team_gains_exactly_one_game_over_the_same_week_pair(
    rows: pd.DataFrame, byes: dict[int, frozenset[str]]
) -> None:
    """ARI played in week 11, so its window gains exactly one game -- and WHICH one.

    The control shares the week pair, the season, the selector and -- because it has
    not had its bye either -- the same week-11 count of ten. The single remaining
    difference is that it played. Its window therefore does what a working rolling
    window is supposed to do, which is the thing the bye teams' windows correctly did
    not do.
    """
    control = phase33_state.BYE_CONTROL_TEAM
    assert control not in byes[phase33_state.BYE_WEEK_FIXTURE], (
        f"the control team {control} is itself on bye in week "
        f"{phase33_state.BYE_WEEK_FIXTURE}; it cannot control for the bye."
    )

    before = observed(rows, control, phase33_state.BYE_WEEK_FIXTURE)
    after = observed(rows, control, phase33_state.BYE_WEEK_FIXTURE_AFTER)
    before_ids = {game_id for game_id, _s, _w in before.games}
    after_ids = {game_id for game_id, _s, _w in after.games}

    assert len(before.games) == len(EXPECTED_IDS[BYE_TEAMS[0]]), (
        f"{control}'s week-{phase33_state.BYE_WEEK_FIXTURE} count is "
        f"{len(before.games)}; the control was chosen to match the bye teams' "
        f"{len(EXPECTED_IDS[BYE_TEAMS[0]])} exactly, so that the bye is the only "
        "difference left."
    )
    assert after_ids - before_ids == {phase33_state.BYE_CONTROL_TEAM_GAME_GAINED}, (
        f"{control} gained {sorted(after_ids - before_ids)} rather than the pinned "
        f"{phase33_state.BYE_CONTROL_TEAM_GAME_GAINED!r}."
    )
    assert not before_ids - after_ids, (
        f"{control} LOST {sorted(before_ids - after_ids)} across the pair; the control "
        "is supposed to gain a game, not trade one."
    )


def test_the_shared_assertion_fails_for_the_control_team(rows: pd.DataFrame) -> None:
    """The attribution arm: the same assertion does NOT hold for a team that played.

    An assertion that held for every team would be measuring the week, not the bye.
    This is the second of this plan's two fail-closed controls -- the other being the
    week-indexed twin in the unit tier -- and it is the one that runs against real
    2026 data.
    """
    control = phase33_state.BYE_CONTROL_TEAM
    with pytest.raises(AssertionError) as excinfo:
        assert_window_ages_not_resets(
            observed(rows, control, phase33_state.BYE_WEEK_FIXTURE),
            observed(rows, control, phase33_state.BYE_WEEK_FIXTURE_AFTER),
            team=control,
            played_game_ids=set(rows[rows["team"] == control]["game_id"]),
        )
    assert "observation COUNT moved" in str(excinfo.value), str(excinfo.value)


def test_exactly_the_six_bye_teams_hold_their_count_and_every_other_team_does_not(
    rows: pd.DataFrame, byes: dict[int, frozenset[str]]
) -> None:
    """The discrimination is total across all thirty-two teams, not just the control.

    Six teams move 10 -> 10 and twenty-six move by exactly +1. No team is ambiguous, so
    "the window held its count" and "the team was on bye" are the same predicate on
    this week pair -- which is a far stronger statement than one control can make.
    """
    held: set[str] = set()
    grew: set[str] = set()
    for team in sorted(rows["team"].unique()):
        before = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE)
        after = observed(rows, team, phase33_state.BYE_WEEK_FIXTURE_AFTER)
        delta = len(after.games) - len(before.games)
        assert delta in (0, 1), f"{team}: unexpected window delta {delta}."
        (held if delta == 0 else grew).add(team)

    assert held == set(byes[phase33_state.BYE_WEEK_FIXTURE]), (
        "the teams whose window count held are not exactly the week-"
        f"{phase33_state.BYE_WEEK_FIXTURE} bye teams.\n"
        f"  held but not on bye: {sorted(held - byes[phase33_state.BYE_WEEK_FIXTURE])}\n"
        f"  on bye but grew:     {sorted(byes[phase33_state.BYE_WEEK_FIXTURE] - held)}"
    )
    assert len(held) == 6 and len(grew) == 26, (len(held), len(grew))


# ---------------------------------------------------------------------------
# The zero-bye week, asserted rather than stumbled over.
# ---------------------------------------------------------------------------


def test_a_week_with_no_byes_is_not_an_error(
    rows: pd.DataFrame, byes: dict[int, frozenset[str]]
) -> None:
    """Week 12 has ZERO byes, the run over it is legitimately empty, and that is fine.

    NF-11: 2026's byes fall in weeks 5-11 and 13-14. A test parameterising over weeks
    5-14 hits week 12 and finds nothing, and the natural reading of that -- "the
    derivation broke" -- is wrong. The bye-team loop below runs zero times BY DESIGN,
    and the assertions around it are what stop this from being a vacuous pass.
    """
    zero_bye_teams = byes[phase33_state.ZERO_BYE_WEEK]
    assert zero_bye_teams == frozenset(), (
        f"week {phase33_state.ZERO_BYE_WEEK} was measured as having no byes in 2026, "
        f"but the capture now derives {sorted(zero_bye_teams)}. Re-measure "
        "ZERO_BYE_WEEK rather than loosening this."
    )

    visited = 0
    for team in sorted(zero_bye_teams):
        visited += 1
        assert_window_ages_not_resets(
            observed(rows, team, phase33_state.ZERO_BYE_WEEK),
            observed(rows, team, phase33_state.ZERO_BYE_WEEK + 1),
            team=team,
        )
    assert visited == 0, (
        "the zero-bye loop visited a team; the point of this case is that it visits "
        "none."
    )

    # Non-vacuity: the SAME derivation, on the SAME capture, does find byes elsewhere.
    assert len(byes[phase33_state.BYE_WEEK_FIXTURE]) == 6, (
        "the bye derivation returned nothing for week "
        f"{phase33_state.BYE_WEEK_FIXTURE} either, so the empty week-"
        f"{phase33_state.ZERO_BYE_WEEK} set is evidence of a broken derivation rather "
        "than of the 2026 calendar."
    )
    assert all(team in byes[11] for team in BYE_TEAMS)


def test_every_week_between_five_and_fourteen_is_accounted_for(
    byes: dict[int, frozenset[str]],
) -> None:
    """The recorded bye table and the live derivation agree, week by week, 5 to 14.

    Recorded once in ``tests/fixtures/season_2026.BYE_TEAMS_BY_WEEK`` and checked here
    against the derivation, so a capture that moved is a named disagreement rather
    than a silently different set every downstream test then agrees with.
    """
    from tests.fixtures.season_2026 import BYE_TEAMS_BY_WEEK

    for week in range(5, 15):
        assert byes[week] == BYE_TEAMS_BY_WEEK[week], (
            f"week {week}: derived {sorted(byes[week])}, recorded "
            f"{sorted(BYE_TEAMS_BY_WEEK[week])}."
        )
    assert sum(len(byes[week]) for week in range(1, 19)) == 32, (
        "the 2026 season must hand out exactly 32 byes, one per team, across an "
        "18-week regular season of 17 games."
    )


# ---------------------------------------------------------------------------
# The D33-18 observation: RECORDED, measured, and corrected.
# ---------------------------------------------------------------------------


def _constructed_prior_season(teams: list[str]) -> pd.DataFrame:
    """A deterministic 17-week prior season, one game per team per week.

    CONSTRUCTED, NOT CAPTURED, and that is a statement about the data: the 2026
    capture holds 2026 rows only, and this module deliberately does not reach into
    ``data/silver/`` for a second gitignored table. The prior season's CONTENTS are
    irrelevant to the quantity being measured -- ``prior_count`` is
    ``max(0, 8 - current_count)`` and never looks at a row -- so a synthetic season
    measures the same arithmetic a real one would, without importing a dependency on
    a table Plan 33-12 is about to rewrite.

    The figure WAS independently corroborated against the real 2025 season during
    measurement; see ``BYE_WINDOW_SPAN_OBSERVATION``.
    """
    return pd.DataFrame(
        [
            {
                "game_id": f"{PRIOR_SEASON}_{week:02d}_{team}_PRIOR",
                "season": PRIOR_SEASON,
                "week": week,
                "team": team,
            }
            for team in teams
            for week in range(1, 18)
        ]
    )


def test_the_extra_prior_season_pull_is_exactly_one_game_and_only_in_weeks_six_to_nine(
    rows: pd.DataFrame, byes: dict[int, frozenset[str]]
) -> None:
    """D33-18, MEASURED -- and the decision's stated week range is wrong.

    D33-18 records that a bye team's window pulls in MORE prior-season data than its
    peers, "a second bootstrap-regime effect stacked on weeks 2-4". Measured, the pull
    is real, is bounded at EXACTLY ONE extra game, and spans target weeks 6 to 9 -- not
    2-4, where no bye has happened yet, and emphatically not week 11, where it is ZERO
    because every team's current-season count has already passed ``max_prior_games``.

    The finding is asserted here so the readout carries a measured figure instead of
    an assumed one. The full record, including the roster-continuity caveat this plan
    records rather than fixes, is ``BYE_WINDOW_SPAN_OBSERVATION``.
    """
    teams = sorted(rows["team"].unique())
    combined = pd.concat(
        [_constructed_prior_season(teams), rows], ignore_index=True
    ).sort_values(["season", "week", "game_id"])

    observed_extra: dict[int, int] = {}
    for target_week in range(6, 13):
        already_had_bye = {
            team
            for week in range(1, target_week)
            for team in byes.get(week, frozenset())
        }
        prior_counts: dict[bool, set[int]] = {True: set(), False: set()}
        for team in teams:
            window = select_window(combined, team, target_week)
            prior_counts[team in already_had_bye].add(
                int((window["season"] == PRIOR_SEASON).sum())
            )
        assert len(prior_counts[True]) == 1 and len(prior_counts[False]) == 1, (
            f"week {target_week}: the prior-season pull is not uniform within a group "
            f"-- post-bye {sorted(prior_counts[True])}, peers "
            f"{sorted(prior_counts[False])}."
        )
        observed_extra[target_week] = next(iter(prior_counts[True])) - next(
            iter(prior_counts[False])
        )

    assert observed_extra == {6: 1, 7: 1, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0}, (
        "the extra prior-season pull is not the measured profile recorded in "
        f"BYE_WINDOW_SPAN_OBSERVATION; got {observed_extra}."
    )
    assert observed_extra[phase33_state.BYE_WEEK_FIXTURE] == 0, (
        "D33-18's premise is that a bye team pulls extra prior-season data 'that "
        f"week'. At week {phase33_state.BYE_WEEK_FIXTURE} it pulls none, and the "
        "readout must say so."
    )


def test_the_recorded_observation_states_the_roster_continuity_caveat() -> None:
    """The caveat is IN the record, not only in a reviewer's comment thread.

    Antigravity raised that pulling prior-season games assumes roster continuity,
    which offseason turnover degrades. This plan does not add a decay weight -- that
    is a feature change to a rule the phase deliberately does not modify -- so the
    only honest alternative is to state the assumption where the figure lives.
    """
    observation = phase33_state.BYE_WINDOW_SPAN_OBSERVATION.lower()
    for phrase in (
        "roster-continuity caveat",
        "no longer exists in the same form",
        "decay weight",
        "target weeks 6 to 9",
    ):
        assert phrase in observation, (
            f"BYE_WINDOW_SPAN_OBSERVATION does not mention {phrase!r}; the caveat and "
            "the measured range are the two things it exists to carry."
        )
    assert "measured 2026-09-14" in observation


# ---------------------------------------------------------------------------
# The fail-closed control on this module's own skip.
# ---------------------------------------------------------------------------


def test_the_missing_capture_skip_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard observed only NOT firing is indistinguishable from one that is unwired.

    Drives the exact path a checkout without the gitignored lake would take and
    asserts the skip is issued with the pinned message, so the module is known to
    degrade to an honest "did not run" rather than to a silent pass.
    """

    def _raise() -> pd.DataFrame:
        raise CapturedScheduleUnavailableError("planted")

    monkeypatch.setattr(
        "tests.integration.test_bye_week_window_ageing.load_captured_schedule", _raise
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        schedule.__wrapped__()
    assert str(excinfo.value) == CAPTURE_MISSING_SKIP
