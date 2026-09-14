"""The shared rolling-window ageing assertion, and the one definition of SPAN.

WHAT "AGES RATHER THAN RESETS" MEANS, STATED ONCE
-------------------------------------------------
A rolling form window AGES when the passage of a week moves the window's right-hand
edge without disturbing what the window holds. It RESETS when the same passage of a
week drops observations, re-anchors the window's left-hand edge later, or manufactures
a placeholder row to stand in for a game that was never played.

THE DEFINITION OF SPAN, FIXED HERE SO IT CANNOT BE SUBSTITUTED LATER
---------------------------------------------------------------------
SPAN is the number of NFL weeks between the EARLIEST game the window selected and the
CURRENT (target) week:

    span = week_ordinal(target_season, target_week)
           - min(week_ordinal(game.season, game.week) for game in the window)

``week_ordinal`` places every (season, week) pair on one integer line at
``season * WEEKS_PER_SEASON_ORDINAL + week``, so a window reaching back into the prior
season measures a real distance rather than a negative week number. The multiplier is an
ORDERING DEVICE, not a claim about the calendar; it matches ``utils.date_utils``'s
``NFL_TOTAL_WEEKS`` of 22 so that consecutive seasons never overlap on the line, and it
is not imported from there because this module takes no project imports at all.

THE THREE PROPERTIES ARE ASSERTED SEPARATELY, ON PURPOSE
--------------------------------------------------------
COUNT, SPAN and ROW IDENTITY are three different claims and each gets its own assertion
with its own message. An assertion that conflates them proves none of the three: a
window holding the right NUMBER of the WRONG games satisfies a count check completely,
and a span check that also moves when the count moves cannot say which one moved.

WHY THIS MODULE IS PURE
-----------------------
No fixtures, no I/O, no project imports, and nothing here reaches into any test tier.
Both the tier that reads the real captured season and the tier that runs alone on
constructed schedules import this same function, so they assert the same thing rather
than two things that happen to agree today.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

# The ordering device described above. 22 is ``utils.date_utils.NFL_TOTAL_WEEKS``, the
# full NFL week count including the postseason, so two consecutive seasons can never
# collide on the integer line.
WEEKS_PER_SEASON_ORDINAL = 22


class WindowObservation(NamedTuple):
    """One selected rolling window, together with the week it was selected FOR.

    The target week is part of the observation because SPAN is measured from the
    window's earliest game to the CURRENT week, not to the window's own latest game.
    A window that stops growing while the weeks keep passing is precisely the thing
    this module exists to detect, and only the target week makes that visible.
    """

    target_season: int
    target_week: int
    games: tuple[tuple[str, int, int], ...]
    """Each entry is ``(game_id, season, week)``, in selection order."""


def week_ordinal(
    season: int, week: int, weeks_per_season: int = WEEKS_PER_SEASON_ORDINAL
) -> int:
    """Place a ``(season, week)`` pair on a single monotonically increasing line.

    Args:
        season: The NFL season.
        week: The week within that season.
        weeks_per_season: The stride between consecutive seasons on the line.

    Returns:
        The absolute week ordinal.
    """
    return season * weeks_per_season + week


def observe(
    target_season: int,
    target_week: int,
    rows: Iterable[Mapping[str, Any]],
) -> WindowObservation:
    """Build a :class:`WindowObservation` from row mappings.

    Accepts anything iterable of mappings carrying ``game_id``, ``season`` and ``week``
    -- notably ``DataFrame.to_dict("records")`` -- so that this module never has to
    import a dataframe library to be useful to callers that do.

    Args:
        target_season: The season the window was selected for.
        target_week: The week the window was selected for.
        rows: The selected games.

    Returns:
        The observation, with row order preserved.
    """
    return WindowObservation(
        target_season=int(target_season),
        target_week=int(target_week),
        games=tuple(
            (str(row["game_id"]), int(row["season"]), int(row["week"])) for row in rows
        ),
    )


def window_span(
    observation: WindowObservation,
    weeks_per_season: int = WEEKS_PER_SEASON_ORDINAL,
) -> int:
    """The window's SPAN, exactly as this module's docstring defines it.

    Args:
        observation: A selected window.
        weeks_per_season: The stride passed through to :func:`week_ordinal`.

    Returns:
        The number of weeks from the earliest selected game to the target week.

    Raises:
        ValueError: The window is empty, so it has no earliest game and therefore no
            span. An empty window is a finding, never a span of zero.
    """
    if not observation.games:
        raise ValueError(
            "an empty window has no earliest game and therefore no span. An empty "
            "selection is a finding about the selector, not a span of zero, and "
            "reporting it as zero would make a window that selected nothing "
            "indistinguishable from one that selected this week's game."
        )
    earliest = min(
        week_ordinal(season, week, weeks_per_season)
        for _game_id, season, week in observation.games
    )
    current = week_ordinal(
        observation.target_season, observation.target_week, weeks_per_season
    )
    return current - earliest


def _describe(observation: WindowObservation) -> str:
    """A compact, deterministic rendering of a window, for failure messages."""
    return (
        f"target={observation.target_season}w{observation.target_week} "
        f"n={len(observation.games)} "
        f"games={[game_id for game_id, _season, _week in observation.games]}"
    )


def assert_window_ages_not_resets(
    before: WindowObservation,
    after: WindowObservation,
    *,
    team: str,
    expected_game_ids: Iterable[str] | None = None,
    played_game_ids: Iterable[str] | None = None,
    weeks_per_season: int = WEEKS_PER_SEASON_ORDINAL,
) -> None:
    """Assert a rolling window AGED between two target weeks rather than RESETTING.

    The five properties, each asserted separately with its own message:

    1. COUNT -- the number of selected observations did NOT move.
    2. SPAN -- the distance from the earliest selected game to the current week DID
       move, strictly upward. Time passed and the window did not follow it.
    3. ROW IDENTITY -- the selected ``game_id`` set equals ``expected_game_ids``, when
       one is supplied. A right-sized window over the wrong games satisfies (1)
       completely.
    4. NO RESET -- the earliest selected game did not move LATER. A window that
       re-anchors its left edge has reset, whatever its count says.
    5. NO MANUFACTURED ROW -- every selected game is one the team actually played,
       when ``played_game_ids`` is supplied, and no game is selected twice. This is
       the property that distinguishes ageing from resetting-with-padding: a zero
       placeholder inserted for an unplayed week keeps the count steady and is exactly
       the failure a count assertion alone would wave through.

    Args:
        before: The window selected for the EARLIER target week.
        after: The window selected for the LATER target week.
        team: The team the two windows belong to, for the failure messages.
        expected_game_ids: The exact ``game_id`` set ``after`` must hold, if pinned.
        played_game_ids: The universe of games the team actually played, if known.
        weeks_per_season: The stride passed through to :func:`week_ordinal`.

    Raises:
        AssertionError: Any of the five properties does not hold.
    """
    assert after.target_week > before.target_week, (
        f"{team}: the two observations are not in time order -- 'before' is "
        f"{before.target_season}w{before.target_week} and 'after' is "
        f"{after.target_season}w{after.target_week}. An ageing claim compared "
        "backwards proves nothing."
    )

    # 1. COUNT
    assert len(after.games) == len(before.games), (
        f"{team}: the window's observation COUNT moved from {len(before.games)} to "
        f"{len(after.games)} between week {before.target_week} and week "
        f"{after.target_week}. A window that ages holds the same observations; a "
        "count that moves means a game entered or left.\n"
        f"  before: {_describe(before)}\n  after:  {_describe(after)}"
    )

    # 2. SPAN
    span_before = window_span(before, weeks_per_season)
    span_after = window_span(after, weeks_per_season)
    assert span_after > span_before, (
        f"{team}: the window's SPAN did not grow -- {span_before} weeks at week "
        f"{before.target_week}, {span_after} weeks at week {after.target_week}. Span "
        "is the distance from the earliest selected game to the current week, so a "
        "span that does not grow when the week does means the window re-anchored "
        "itself instead of ageing.\n"
        f"  before: {_describe(before)}\n  after:  {_describe(after)}"
    )

    # 3. ROW IDENTITY
    after_ids = [game_id for game_id, _season, _week in after.games]
    if expected_game_ids is not None:
        expected = set(expected_game_ids)
        actual = set(after_ids)
        assert actual == expected, (
            f"{team}: the window at week {after.target_week} selected a different SET "
            "of games than the pinned expectation. An equal count over a different "
            "set satisfies a count assertion and proves nothing.\n"
            f"  unexpected: {sorted(actual - expected)}\n"
            f"  missing:    {sorted(expected - actual)}"
        )

    # 4. NO RESET
    earliest_before = min(
        week_ordinal(season, week, weeks_per_season)
        for _game_id, season, week in before.games
    )
    earliest_after = min(
        week_ordinal(season, week, weeks_per_season)
        for _game_id, season, week in after.games
    )
    assert earliest_after <= earliest_before, (
        f"{team}: the earliest selected game moved LATER, from ordinal "
        f"{earliest_before} to {earliest_after}. That is a reset -- the window "
        "dropped its oldest observations -- not an ageing.\n"
        f"  before: {_describe(before)}\n  after:  {_describe(after)}"
    )

    # 5. NO MANUFACTURED ROW
    assert len(set(after_ids)) == len(after_ids), (
        f"{team}: the window at week {after.target_week} selected the same game twice: "
        f"{sorted({game_id for game_id in after_ids if after_ids.count(game_id) > 1})}. "
        "A duplicated observation is a padded window wearing the right count."
    )
    if played_game_ids is not None:
        played = set(played_game_ids)
        manufactured = sorted(set(after_ids) - played)
        assert not manufactured, (
            f"{team}: the window at week {after.target_week} contains "
            f"{len(manufactured)} game(s) the team never played: {manufactured}. A "
            "placeholder row inserted for an unplayed week keeps the count steady "
            "while corrupting every average computed over the window."
        )
