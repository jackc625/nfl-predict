"""R2's four canonical checks on the canonical Elo chain (COLD-04).

The chain was RE-DERIVED over 2002-2025 and has since been extended forward by the live
2026 capture; see "WHAT THE LIVE 2026 CAPTURE CHANGED" below for what that moved and what
it did not.


WHAT MAKES THIS CHAIN CANONICAL, AND WHAT DOES NOT
--------------------------------------------------
Not provenance. There is no known-good source: no backup of the Elo artifacts
existed anywhere in the tree before Plan 33-13, the snapshot table was a 2,227-row
2018-2025 file a sandbox-less test wrote during Phase 31, and the other four
carried v2.0-era mtimes. So the operation was a RE-DERIVATION from
``data/silver/games.parquet`` and never a restore, and what makes its output
trustworthy is this module: four independent checks that can each fail for a
different, nameable reason.

    1. The chain STARTS in 2002 and ENDS on the last season silver holds a completed
       game for -- two separate assertions on the two BOUNDARY VALUES, so a table
       starting at 2003 and a table ending a season early fail differently and a
       count or a range containment cannot stand in for either.
    2. Per-season snapshot counts RECONCILE, season by season, against the
       completed games in ``games.parquet``, with every mismatching season named.
    3. Every rating falls inside ``tests.phase33_state.ELO_RATING_BAND_FROZEN``.
    4. Each team's week-1 pre-rating for the season AFTER the chain's terminal one is
       its terminal rating with the season carryover applied EXACTLY ONCE, to four
       decimal places, for all 32 teams.

D33-09 added a fifth: ``games_with_elo`` and ``elo_rating_history`` each reconciled
to the completed-game count, because a side table that did not would mean a SECOND
unexplained divergence. D33.2-22 (Plan 33.2-05, owner ratified 2026-09-21) DELETED
both side tables -- they were written by a legacy per-season pass that learned no
home-field advantage and differed from the chain by up to 9.7 rating points. The
intent survives as a stronger check: there is no second copy left to diverge, and
the fifth check now asserts that none of the four deleted artifacts exists anywhere
under ``data/silver/``, at the root or inside any staged generation.

WHY THE BAND IS IMPORTED AND NEVER MEASURED HERE
------------------------------------------------
``ELO_RATING_BAND_FROZEN`` was derived from the PRE-run chain and committed in a
strictly EARLIER commit than the re-derivation. That ordering is the whole
content of the check. A band taken from the run's own output and then asserted
against that same run's values is TRUE BY CONSTRUCTION for any output whatever --
including a catastrophically wrong one -- since it would assert only that a
maximum is not below a minimum. This module therefore imports exactly one band,
and the post-run endpoints (recorded separately in ``tests/phase33_state.py`` for
the record) are deliberately not imported, not read and not named here.

WHY THE CARRYOVER IS COMPUTED IN MEMORY
---------------------------------------
Because a stored row for the season being carried INTO is a row the carryover has
not produced yet, so reading one would check the arithmetic against its own output.
``build_elo_with_snapshots`` re-derives the chain from ``games.parquet``; this
module is the DETERMINISTIC half -- the carryover arithmetic, checked against the
re-derived TERMINAL state -- and the STATIC half of the carryover-once proof, the
dynamic half being Plan 33-03's rerun-identity suite, which shows that running the
chain twice does not move it.

WHAT THE LIVE 2026 CAPTURE CHANGED, AND WHY NOTHING HERE WAS RELAXED (Plan 33.2-20)
-----------------------------------------------------------------------------------
This module was written when the store held 2002-2025 and nothing else, and it said
so in as many words: "no STORED 2026 row can exist at this point in the phase". That
premise is now FALSE. The live 2026 capture appended 32 rows -- 17 REAL snapshots for
the played games (16 of week 1 and ``2026_W02_DET@BUF``) and 15 PROVISIONAL rows for
the unplayed remainder of week 2 -- and Plan 33.2-20's clean production build consumed
the 17. Five assertions here were pinned to the pre-capture shape and went red for that
reason alone.

Each was re-anchored by making the claim it already made DERIVABLE, never by loosening
it. MEASURED 2026-09-22 on the post-build store, and every number below is arithmetic
the tests recompute rather than a pin:

* 6,531 rows = 6,516 NON-PROVISIONAL + 15 PROVISIONAL;
* the 6,516 non-provisional rows reconcile against the 6,516 completed games in
  ``games.parquet`` EXACTLY, season by season, with zero mismatching seasons;
* all 15 provisional rows are UNPLAYED games -- none has a score -- which is precisely
  what ``is_provisional`` is for;
* the 2002-2025 slice is still 6,499 rows, the figure the re-derivation recorded in
  ``ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION``, so the canonical burn-in did not move.

So the CANONICAL CHAIN is now defined by the property that always defined it -- one
non-provisional snapshot per completed game -- rather than by the season range it
happened to span when it was written. The boundary check asks the SILVER SCHEDULE what
the last completed season is instead of naming 2025, so it cannot go stale again at the
2027 rollover; the provisional check is now STRONGER, because it additionally asserts
that every provisional row is an unplayed game (a provisional row for a PLAYED game
still fails, which is the defect the original wording was reaching for); and the
carryover target is derived from the re-derived chain's own terminal season rather than
pinned to 2026.

THIS MODULE WRITES NOTHING. It reads the production store and asserts about it;
one test additionally requests ``data_boundary_guard`` so the claim is enforced by
content digests rather than by intent.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from tests.phase33_state import (
    ELO_RATING_BAND_FROZEN,
    ELO_SEASON_COVERAGE,
    ELO_SNAPSHOT_COLUMNS_AFTER_REDERIVATION,
    ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION,
)

SILVER = Path("data") / "silver"
SNAPSHOTS_PATH = SILVER / "elo_game_snapshots.parquet"
GAMES_PATH = SILVER / "games.parquet"

# The four Elo artifacts D33.2-22 deleted, by FILENAME, so the absence check covers the
# silver root and every staged generation directory with one walk.
DELETED_ELO_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "games_with_elo.parquet",
    "elo_rating_history.parquet",
    "elo_ratings_current.parquet",
    "elo_ratings.json",
)

# The first season of the canonical burn-in, and the last season the RE-DERIVATION
# recorded. LAST_SEASON is no longer the season the live table ends on: the 2026 capture
# appended real snapshots for the played 2026 games. It is still the boundary of the
# RECORDED burn-in slice, which this module asserts has not moved.
FIRST_SEASON, LAST_SEASON = ELO_SEASON_COVERAGE

# The seasons the recorded re-derivation covered, as a slice predicate. Everything the
# re-derivation's own figures are asserted against is scoped to it.
REDERIVED_SEASONS = range(FIRST_SEASON, LAST_SEASON + 1)

# Decimal places the carryover comparison is made to. Four, as R2 specifies: a
# looser comparison would not distinguish "applied once" from "applied once and
# then nudged", which is the failure mode worth catching.
CARRYOVER_PLACES = 4

# Every team must be covered, not merely the ones that happen to appear.
EXPECTED_TEAM_COUNT = 32

# A skip REASON that tests/conftest.py recognises as EVIDENCE-BACKED ("not present
# at"), so a checkout without the gitignored lake SAYS these controls did not run
# rather than reporting a green suite that quietly excluded them.
_MISSING_SILVER_REASON = (
    f"the silver layer is not present at {SILVER.as_posix()}/ -- data/ is "
    "gitignored, so a checkout that has not built the lake cannot judge the "
    "canonical Elo chain. This is a fact about the checkout, not about the data."
)

pytestmark = pytest.mark.skipif(not SILVER.is_dir(), reason=_MISSING_SILVER_REASON)


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    """The re-derived snapshot table as it stands on the production store."""
    return pd.read_parquet(SNAPSHOTS_PATH, engine="pyarrow")


@pytest.fixture(scope="module")
def completed_games() -> pd.DataFrame:
    """The completed games in silver -- the single input the chain was derived from."""
    games = pd.read_parquet(GAMES_PATH, engine="pyarrow")
    return games.loc[games["home_score"].notna() & games["away_score"].notna()]


@pytest.fixture(scope="module")
def canonical_snapshots(snapshots) -> pd.DataFrame:
    """The CANONICAL chain: the non-provisional rows, one per completed game.

    Plan 33.2-20. The canonical chain was once "the whole table", because the table
    held nothing else. Since the live 2026 capture it also holds PROVISIONAL rows for
    games that have not been played, which the builder writes at serving time and the
    train boundary refuses. The defining property is unchanged and is now applied:
    a canonical row is one the chain DERIVED from a played game.
    """
    return snapshots.loc[~snapshots["is_provisional"].astype(bool)]


@pytest.fixture(scope="module")
def last_completed_season(completed_games) -> int:
    """The last season silver holds a completed game for -- ASKED, never pinned.

    Plan 33.2-20. The chain is derived from exactly this frame, so the season it ends
    on is a fact about the schedule and not a constant to maintain. Pinning it is what
    made the boundary check go red when the 2026 season started, and would make it go
    red again in 2027.
    """
    return int(completed_games["season"].max())


@pytest.fixture(scope="module")
def carryover_target_season(last_completed_season) -> int:
    """The season the carryover is applied FOR: the one after the chain's terminal."""
    return last_completed_season + 1


@pytest.fixture(scope="module")
def rederived_terminal_ratings() -> dict[str, float]:
    """Each team's 2025 FINAL rating, re-derived in memory from ``games.parquet``.

    The chain is rebuilt rather than read back, for the same reason
    ``EloBuilder.build_prior_terminal_state`` rebuilds rather than reading:
    a persisted terminal snapshot is a second piece of state that can drift from
    the ratings it claims to describe. Measured at about a second for 6,499 games.
    """
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    builder.build_elo_with_snapshots(start_season=FIRST_SEASON)
    return {team: rating.rating for team, rating in builder.elo_system.ratings.items()}


@pytest.fixture(scope="module")
def carried_forward_ratings(
    rederived_terminal_ratings, carryover_target_season
) -> dict[str, object]:
    """The 2026 week-1 pre-ratings, and the shrink used, computed IN MEMORY.

    Returns the carried-forward ratings, the shrink factor the system applied, and
    the state AFTER a second carryover call -- which is what makes "exactly once"
    checkable rather than merely asserted.
    """
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    builder.build_elo_with_snapshots(start_season=FIRST_SEASON)
    system = copy.deepcopy(builder.elo_system)

    system.apply_season_carryover(carryover_target_season)
    once = {team: rating.rating for team, rating in system.ratings.items()}

    system.apply_season_carryover(carryover_target_season)
    twice = {team: rating.rating for team, rating in system.ratings.items()}

    return {
        "shrink": builder.elo_system.season_carryover,
        "once": once,
        "twice": twice,
    }


# ---------------------------------------------------------------------------
# R2 check 1 -- the two BOUNDARY seasons, as two separate assertions.
# ---------------------------------------------------------------------------


def test_the_chain_starts_in_2002(snapshots):
    """The snapshot table's FIRST season is 2002 -- the boundary value itself."""
    first = int(snapshots["season"].min())
    assert first == FIRST_SEASON, (
        f"the canonical Elo chain starts in {first}, not {FIRST_SEASON}. The "
        "burn-in exists so that the first backtest season (2018) is preceded by "
        "sixteen seasons of learned ratings; a chain that starts later hands "
        "every model a cold 1500 for seasons it is being trained on. This is "
        "asserted on the MINIMUM season value and never on a count or a range, "
        "because a table holding the right NUMBER of seasons starting in the "
        "wrong one would pass either of those."
    )


def test_the_chain_ends_on_the_last_completed_season(snapshots, last_completed_season):
    """The chain's LAST season is silver's -- a separate, differently-named failure.

    RE-ANCHORED, not relaxed (Plan 33.2-20). This asserted ``== 2025`` and measured
    2026, because the live capture appended real snapshots for the played 2026 games
    -- which is the chain doing its job, not a defect. The claim it was making is that
    the most recent COMPLETED season reached the table the gold Elo columns LEFT JOIN
    against; that is a comparison against the schedule, so the schedule is now what it
    is compared to. Pinning the season was what made it go stale, and would again in
    2027.
    """
    last = int(snapshots["season"].max())
    assert last == last_completed_season, (
        f"the canonical Elo chain ends in {last}, while silver holds a completed "
        f"game as late as {last_completed_season}. A chain ending early means the "
        "most recent completed season never reached the table the gold Elo columns "
        "LEFT JOIN against, so the live cold start would carry ratings that are a "
        "season stale. Deliberately a SECOND assertion from the first-season one: a "
        "table starting at 2003 and a table ending a season early are different "
        "defects and must fail differently."
    )

    assert last_completed_season >= LAST_SEASON, (
        f"silver's last completed season is {last_completed_season}, EARLIER than "
        f"the {LAST_SEASON} the re-derivation recorded. The schedule cannot lose a "
        "season it already held, so this is a store that went backwards -- and "
        "without this floor the assertion above would happily agree with it."
    )


def test_the_chain_covers_every_season_in_between(snapshots, last_completed_season):
    """No season between the boundaries is missing -- the boundaries alone cannot say so."""
    present = {int(value) for value in snapshots["season"].dropna().unique()}
    expected = set(range(FIRST_SEASON, last_completed_season + 1))
    missing = sorted(expected - present)
    assert not missing, (
        f"the chain spans {FIRST_SEASON}-{last_completed_season} at its boundaries "
        f"but is MISSING season(s) {missing}. Both boundary assertions would still "
        "pass; a hole in the middle is exactly what they cannot see."
    )


# ---------------------------------------------------------------------------
# R2 check 2 -- per-season reconciliation against games.parquet.
# ---------------------------------------------------------------------------


def test_per_season_snapshot_counts_reconcile_against_games(
    canonical_snapshots, completed_games
):
    """Season by season, one CANONICAL row per completed game -- naming every mismatch.

    RE-ANCHORED, not relaxed (Plan 33.2-20). This reconciled the WHOLE table, which
    was the same thing until the live capture added 15 PROVISIONAL rows for unplayed
    2026 week-2 games; 2026 then read (17 completed, 32 rows). A provisional row is
    written for a game that has NOT been played, so counting it against completed
    games compares two different populations. The reconciliation now runs over the
    non-provisional rows -- the ones the chain derived from a result -- and MEASURED
    2026-09-22 it is exact: 6,516 against 6,516, zero mismatching seasons. The
    provisional rows are not dropped from the module's coverage; they are asserted in
    ``test_no_row_in_the_canonical_chain_is_provisional`` as what they are.
    """
    snapshot_counts = canonical_snapshots.groupby("season").size().to_dict()
    game_counts = completed_games.groupby("season").size().to_dict()

    seasons = sorted(set(snapshot_counts) | set(game_counts))
    mismatches = [
        (
            int(season),
            int(game_counts.get(season, 0)),
            int(snapshot_counts.get(season, 0)),
        )
        for season in seasons
        if int(snapshot_counts.get(season, 0)) != int(game_counts.get(season, 0))
    ]

    assert not mismatches, (
        "the re-derived chain does not reconcile against "
        f"{GAMES_PATH.as_posix()}. Every mismatching season is named, as "
        "(season, completed games, snapshot rows): "
        f"{mismatches}. The canonical builder emits exactly one snapshot per "
        "COMPLETED game -- it skips any game with a null score by construction -- "
        "so a season with fewer snapshot rows than completed games means the "
        "chain silently dropped results, and a season with more means it "
        "double-counted them. A single boolean here would say only that "
        "something disagreed."
    )


def test_the_chain_holds_one_row_per_completed_game_overall(
    snapshots, canonical_snapshots, completed_games
):
    """The totals reconcile too, and the shape is the twelve-column one.

    RE-ANCHORED, not relaxed (Plan 33.2-20). The recorded
    ``ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION`` is 6,499 -- the 2002-2025 re-derivation's
    own figure, and an APPEND-ONCE record that must not be edited to today's number.
    The live 2026 capture appended 32 rows, so the whole-table comparison measured
    6,531. Both facts are now asserted SEPARATELY, which is strictly more than before:
    the canonical rows reconcile against the completed games (6,516 == 6,516), AND the
    recorded re-derivation slice is asserted UNMOVED at 6,499, so a change to
    2002-2025 still fails here even though the table has grown past it.
    """
    assert len(canonical_snapshots) == len(completed_games), (
        f"the canonical chain holds {len(canonical_snapshots)} non-provisional rows "
        f"against {len(completed_games)} completed games in {GAMES_PATH.as_posix()}."
    )
    rederived = canonical_snapshots.loc[
        canonical_snapshots["season"].isin(REDERIVED_SEASONS)
    ]
    assert len(rederived) == ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION, (
        f"the {FIRST_SEASON}-{LAST_SEASON} slice holds {len(rederived)} rows; the "
        f"re-derivation recorded {ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION}. That record "
        "is append-once and is never edited to match a later store: a slice that "
        "disagrees means the burn-in itself moved, which no live capture can do."
    )
    assert snapshots.shape[1] == ELO_SNAPSHOT_COLUMNS_AFTER_REDERIVATION, (
        f"the chain has {snapshots.shape[1]} columns; the re-derived table has "
        f"{ELO_SNAPSHOT_COLUMNS_AFTER_REDERIVATION}, the twelfth being "
        "is_provisional. An eleven-column table here is the PRE-flag shape, which "
        "would mean the store is the one this plan replaced."
    )


def test_no_row_in_the_canonical_chain_is_provisional(snapshots, completed_games):
    """A provisional row is a row for a game that has NOT been played. Both directions.

    RE-ANCHORED, and STRONGER than what it replaced (Plan 33.2-20). This asserted that
    the table carried ZERO provisional rows, which was true only while the table held
    nothing but the 2002-2025 rebuild; the live capture writes a provisional row for an
    UPCOMING game, which is exactly what the flag exists to mark, so "zero" had become
    a claim that the live path had never run.

    The defect the original was reaching for -- "a serving-time placeholder reached the
    table three deployed models train through" -- is a placeholder for a game that HAS
    been played, or a played game whose row is not marked. Both are now asserted by
    name, so this node catches strictly more than it did: a provisional row for a
    completed game fails, AND a completed game is still required to carry a real row
    through the reconciliation above.
    """
    provisional = snapshots.loc[snapshots["is_provisional"].astype(bool)]
    completed_ids = set(completed_games["game_id"].astype(str))
    played_but_provisional = sorted(
        set(provisional["game_id"].astype(str)) & completed_ids
    )

    assert played_but_provisional == [], (
        f"{len(played_but_provisional)} row(s) carry is_provisional=True for a game "
        f"that HAS been played, e.g. {played_but_provisional[:8]}. The canonical "
        "builder cannot produce one -- it derives a row from a result -- so this is a "
        "serving-time placeholder that was never replaced by the real snapshot, in "
        "the table three deployed models train through."
    )

    assert len(provisional) == len(snapshots) - len(completed_games), (
        f"the table holds {len(snapshots)} rows and {len(completed_games)} completed "
        f"games, so {len(snapshots) - len(completed_games)} rows should be "
        f"provisional; {len(provisional)} are. A row for an unplayed game that is NOT "
        "flagged is the same defect seen from the other side, and the counts are the "
        "only thing that sees it."
    )


# ---------------------------------------------------------------------------
# R2 check 3 -- the rating band, FROZEN before the run.
# ---------------------------------------------------------------------------


def test_every_rating_falls_inside_the_frozen_band(snapshots):
    """No pre-game rating escapes the band frozen from the PRE-run chain.

    The band is imported, never measured here. A band measured from this run's own
    output and then asserted against this run's values would hold for any output
    at all, so it would prove nothing.
    """
    low, high = ELO_RATING_BAND_FROZEN
    values = pd.concat([snapshots["home_elo_pre"], snapshots["away_elo_pre"]]).dropna()

    assert len(values) > 0, "the chain carries no pre-game ratings at all."

    outside = values[(values < low) | (values > high)]
    assert len(outside) == 0, (
        f"{len(outside)} rating(s) fall outside the frozen sanity band "
        f"[{low}, {high}] -- observed min {float(values.min()):.4f}, max "
        f"{float(values.max()):.4f}. The band was derived from the chain as it "
        "stood BEFORE the re-derivation and committed in an earlier commit, so "
        "this assertion can actually fail. Do NOT widen it to accommodate an "
        "output: a band adjusted to fit the numbers it is judging is decoration."
    )


def test_the_frozen_band_is_narrower_than_the_code_s_own_sanity_range():
    """The frozen band constrains something the shipped warning does not.

    ``EloBuilder.validate_ratings`` warns outside (800, 2200). A frozen band wider
    than that would add no information over a check the code already makes.
    """
    low, high = ELO_RATING_BAND_FROZEN
    assert 800.0 < low < high < 2200.0, (
        f"the frozen band [{low}, {high}] is not strictly inside the (800, 2200) "
        "range scripts/build_elo.EloBuilder.validate_ratings already warns "
        "outside of, so it constrains nothing that was not constrained already."
    )


# ---------------------------------------------------------------------------
# R2 check 4 -- the 2025-to-2026 carryover, applied EXACTLY ONCE, in memory.
# ---------------------------------------------------------------------------


def test_the_next_season_week_one_pre_rating_is_the_terminal_final_carried_once(
    rederived_terminal_ratings,
    carried_forward_ratings,
    carryover_target_season,
    last_completed_season,
):
    """For all 32 teams, to four decimal places, with every failing team named.

    RE-ANCHORED, not relaxed (Plan 33.2-20). The pair of seasons was pinned at
    2025 -> 2026. ``build_elo_with_snapshots`` re-derives from silver, and silver now
    holds completed 2026 games, so its terminal state is 2026's and the carryover it
    was being checked against had already been applied. The ARITHMETIC claim -- the
    next season's week-1 pre-rating is the terminal rating shrunk toward 1500 exactly
    once -- is what this node is for, and it holds for any pair of consecutive
    seasons, so the pair is now derived from the chain's own terminal season instead
    of named. Nothing about the comparison, the four-decimal tolerance or the
    all-32-teams requirement changed.
    """
    shrink = carried_forward_ratings["shrink"]
    once = carried_forward_ratings["once"]

    assert len(rederived_terminal_ratings) == EXPECTED_TEAM_COUNT, (
        f"the re-derived {last_completed_season} terminal state rates "
        f"{len(rederived_terminal_ratings)} teams, not {EXPECTED_TEAM_COUNT}."
    )
    assert set(once) == set(rederived_terminal_ratings), (
        "the carried-forward state does not cover the same teams as the "
        f"{last_completed_season} terminal state: "
        f"{sorted(set(once) ^ set(rederived_terminal_ratings))}"
    )

    failures = []
    for team, final in sorted(rederived_terminal_ratings.items()):
        expected = final * shrink + 1500.0 * (1.0 - shrink)
        actual = once[team]
        if round(actual, CARRYOVER_PLACES) != round(expected, CARRYOVER_PLACES):
            failures.append(
                (team, round(final, 4), round(expected, 4), round(actual, 4))
            )

    assert not failures, (
        f"the {carryover_target_season} week-1 pre-rating does not equal the "
        f"{last_completed_season} final with the carryover shrink ({shrink}) applied "
        "exactly once, for: "
        f"{failures} -- each entry (team, {last_completed_season} final, expected, "
        "actual), "
        f"compared to {CARRYOVER_PLACES} decimal places. Every failing team is "
        "named rather than only the first."
    )


def test_a_second_carryover_call_moves_no_rating(
    carried_forward_ratings, carryover_target_season
):
    """ "Exactly once" is the claim, so applying it again must be a no-op.

    Double application is the defect this guards: it would shrink every rating a
    second time toward 1500 and silently flatten the whole league in the column
    three deployed models read through.
    """
    once = carried_forward_ratings["once"]
    twice = carried_forward_ratings["twice"]

    moved = [
        (team, round(once[team], 4), round(twice[team], 4))
        for team in sorted(once)
        if round(once[team], CARRYOVER_PLACES) != round(twice[team], CARRYOVER_PLACES)
    ]
    assert not moved, (
        f"a SECOND apply_season_carryover({carryover_target_season}) call moved "
        f"{len(moved)} rating(s): {moved}. The carryover is supposed to be "
        "idempotent for a season it has already been applied for; a second "
        "application shrinks every rating toward 1500 again."
    )


def test_the_provisional_week_one_snapshot_reads_the_carried_forward_rating(
    carried_forward_ratings, carryover_target_season, data_boundary_guard
):
    """The live path's week-1 pre-rating IS the carried-forward number.

    Driven against an IN-MEMORY schedule frame for a season the chain has not
    reached, never a stored row: a stored row for that season would be the very
    output this checks, and the STORED live rows are asserted in Plan 33-18.
    ``data_boundary_guard`` is requested here because this is the one test in the
    module that drives a writer's code path, so "it wrote nothing" is settled by
    content digests rather than by reading the implementation.
    """
    from scripts.build_elo import SNAPSHOT_COLUMNS, EloBuilder

    once = carried_forward_ratings["once"]
    teams = sorted(once)
    home, away = teams[0], teams[1]

    schedule = pd.DataFrame(
        [
            {
                "game_id": f"{carryover_target_season}_W01_{away}@{home}",
                "season": carryover_target_season,
                "week": 1,
                "kickoff_et": pd.Timestamp(
                    f"{carryover_target_season}-09-10T20:00:00Z"
                ),
                "home_team": home,
                "away_team": away,
                "home_score": None,
                "away_score": None,
            }
        ]
    )

    builder = EloBuilder()
    builder.build_elo_with_snapshots(start_season=FIRST_SEASON)
    builder.elo_system.apply_season_carryover(carryover_target_season)

    frame = builder.snapshot_upcoming_week(carryover_target_season, 1, games=schedule)

    assert list(frame.columns) == list(SNAPSHOT_COLUMNS), (
        "the provisional frame is not in the twelve-column snapshot shape: "
        f"{list(frame.columns)}"
    )
    assert len(frame) == 1, f"expected one provisional row, got {len(frame)}"
    row = frame.iloc[0]
    assert bool(row["is_provisional"]) is True, (
        "the row for an unplayed game is not flagged provisional, so a "
        "serving-time placeholder would be indistinguishable from an observation."
    )
    assert round(float(row["home_elo_pre"]), CARRYOVER_PLACES) == round(
        once[home], CARRYOVER_PLACES
    ), (
        f"the {carryover_target_season} week-1 home pre-rating for {home} is "
        f"{float(row['home_elo_pre']):.4f}, not the carried-forward "
        f"{once[home]:.4f}."
    )
    assert round(float(row["away_elo_pre"]), CARRYOVER_PLACES) == round(
        once[away], CARRYOVER_PLACES
    ), (
        f"the {carryover_target_season} week-1 away pre-rating for {away} is "
        f"{float(row['away_elo_pre']):.4f}, not the carried-forward "
        f"{once[away]:.4f}."
    )


# ---------------------------------------------------------------------------
# D33-09, re-expressed under D33.2-22: no second copy of the chain survives.
# ---------------------------------------------------------------------------


def test_no_deleted_elo_artifact_survives_anywhere_under_silver():
    """The four deleted side stores are absent at the root AND in every generation.

    D33-09 reconciled two of them against the chain because a side table that
    disagreed would be a second, unexplained answer. D33.2-22 removed the pass that
    wrote them, so the right check is that no copy is left at all -- including the
    staged copies inside ``elo_generations/<id>/``, which a root-only check would miss.
    """
    survivors = sorted(
        path.as_posix()
        for path in SILVER.rglob("*")
        if path.name in DELETED_ELO_ARTIFACT_FILENAMES
    )
    assert survivors == [], (
        f"deleted Elo side stores are still on disk: {survivors}. D33.2-22 removed "
        "the legacy pass and its four stores together; a surviving copy is a second "
        "answer that nothing keeps in step with elo_game_snapshots."
    )


def test_the_absence_check_walked_a_real_silver_layer(snapshots):
    """Non-vacuity: the walk above found the snapshot table, so it looked somewhere."""
    assert SNAPSHOTS_PATH.is_file() and len(snapshots) > 0, (
        "the snapshot table is missing or empty, so an absence check over silver "
        "would pass for the wrong reason."
    )
