"""R2's four canonical checks on the re-derived 2002-2025 Elo chain (COLD-04).

WHAT MAKES THIS CHAIN CANONICAL, AND WHAT DOES NOT
--------------------------------------------------
Not provenance. There is no known-good source: no backup of the Elo artifacts
existed anywhere in the tree before Plan 33-13, the snapshot table was a 2,227-row
2018-2025 file a sandbox-less test wrote during Phase 31, and the other four
carried v2.0-era mtimes. So the operation was a RE-DERIVATION from
``data/silver/games.parquet`` and never a restore, and what makes its output
trustworthy is this module: four independent checks that can each fail for a
different, nameable reason.

    1. The chain STARTS in 2002 and ENDS in 2025 -- two separate assertions on the
       two BOUNDARY VALUES, so a table starting at 2003 and a table ending at 2024
       fail differently and a count or a range containment cannot stand in for
       either.
    2. Per-season snapshot counts RECONCILE, season by season, against the
       completed games in ``games.parquet``, with every mismatching season named.
    3. Every rating falls inside ``tests.phase33_state.ELO_RATING_BAND_FROZEN``.
    4. Each team's 2026 week-1 pre-rating is its 2025 final with the season
       carryover applied EXACTLY ONCE, to four decimal places, for all 32 teams.

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
Because no STORED 2026 row can exist at this point in the phase, and reading one
would either find nothing or find a row the phase has not yet written.
``build_elo_with_snapshots`` rebuilds 2002-2025; provisional 2026 rows arrive only
through ``EloBuilder.snapshot_upcoming_week`` on the live path, which Plan 33-18
exercises -- and the STORED 2026 rows are asserted THERE, not here. This module is
the DETERMINISTIC half: the carryover arithmetic, checked against the re-derived
2025 terminal state. It is also the STATIC half of the carryover-once proof, the
dynamic half being Plan 33-03's rerun-identity suite, which shows that running the
chain twice does not move it.

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

# The first season of the canonical burn-in and the season the chain must end on.
FIRST_SEASON, LAST_SEASON = ELO_SEASON_COVERAGE

# The season whose week-1 pre-ratings the carryover check derives in memory.
CARRYOVER_TARGET_SEASON = LAST_SEASON + 1

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
def carried_forward_ratings(rederived_terminal_ratings) -> dict[str, object]:
    """The 2026 week-1 pre-ratings, and the shrink used, computed IN MEMORY.

    Returns the carried-forward ratings, the shrink factor the system applied, and
    the state AFTER a second carryover call -- which is what makes "exactly once"
    checkable rather than merely asserted.
    """
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    builder.build_elo_with_snapshots(start_season=FIRST_SEASON)
    system = copy.deepcopy(builder.elo_system)

    system.apply_season_carryover(CARRYOVER_TARGET_SEASON)
    once = {team: rating.rating for team, rating in system.ratings.items()}

    system.apply_season_carryover(CARRYOVER_TARGET_SEASON)
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


def test_the_chain_ends_in_2025(snapshots):
    """The snapshot table's LAST season is 2025 -- a separate, differently-named failure."""
    last = int(snapshots["season"].max())
    assert last == LAST_SEASON, (
        f"the canonical Elo chain ends in {last}, not {LAST_SEASON}. A chain "
        "ending early means the most recent completed season never reached the "
        "table the gold Elo columns LEFT JOIN against, so the live cold start "
        "would carry ratings that are a season stale. Deliberately a SECOND "
        "assertion from the first-season one: a table starting at 2003 and a "
        "table ending at 2024 are different defects and must fail differently."
    )


def test_the_chain_covers_every_season_in_between(snapshots):
    """No season between the boundaries is missing -- the boundaries alone cannot say so."""
    present = {int(value) for value in snapshots["season"].dropna().unique()}
    expected = set(range(FIRST_SEASON, LAST_SEASON + 1))
    missing = sorted(expected - present)
    assert not missing, (
        f"the chain spans {FIRST_SEASON}-{LAST_SEASON} at its boundaries but is "
        f"MISSING season(s) {missing}. Both boundary assertions would still pass; "
        "a hole in the middle is exactly what they cannot see."
    )


# ---------------------------------------------------------------------------
# R2 check 2 -- per-season reconciliation against games.parquet.
# ---------------------------------------------------------------------------


def test_per_season_snapshot_counts_reconcile_against_games(snapshots, completed_games):
    """Season by season, one snapshot row per completed game -- naming every mismatch."""
    snapshot_counts = snapshots.groupby("season").size().to_dict()
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


def test_the_chain_holds_one_row_per_completed_game_overall(snapshots, completed_games):
    """The totals reconcile too, and the shape is the twelve-column one."""
    assert len(snapshots) == len(completed_games), (
        f"the chain holds {len(snapshots)} rows against "
        f"{len(completed_games)} completed games in {GAMES_PATH.as_posix()}."
    )
    assert len(snapshots) == ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION, (
        f"the chain holds {len(snapshots)} rows; the re-derivation recorded "
        f"{ELO_SNAPSHOT_ROWS_AFTER_REDERIVATION}."
    )
    assert snapshots.shape[1] == ELO_SNAPSHOT_COLUMNS_AFTER_REDERIVATION, (
        f"the chain has {snapshots.shape[1]} columns; the re-derived table has "
        f"{ELO_SNAPSHOT_COLUMNS_AFTER_REDERIVATION}, the twelfth being "
        "is_provisional. An eleven-column table here is the PRE-flag shape, which "
        "would mean the store is the one this plan replaced."
    )


def test_no_row_in_the_canonical_chain_is_provisional(snapshots):
    """Every row came from a played game, so every flag is False."""
    provisional = snapshots.loc[snapshots["is_provisional"]]
    assert len(provisional) == 0, (
        f"{len(provisional)} row(s) in the canonical chain carry "
        "is_provisional=True, e.g. "
        f"{sorted(provisional['game_id'].astype(str))[:8]}. The canonical builder "
        "cannot produce one: it skips every game with a null score. A provisional "
        "row here means a serving-time placeholder reached the table three "
        "deployed models train through."
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


def test_the_2026_week_one_pre_rating_is_the_2025_final_carried_once(
    rederived_terminal_ratings, carried_forward_ratings
):
    """For all 32 teams, to four decimal places, with every failing team named."""
    shrink = carried_forward_ratings["shrink"]
    once = carried_forward_ratings["once"]

    assert len(rederived_terminal_ratings) == EXPECTED_TEAM_COUNT, (
        f"the re-derived {LAST_SEASON} terminal state rates "
        f"{len(rederived_terminal_ratings)} teams, not {EXPECTED_TEAM_COUNT}."
    )
    assert set(once) == set(rederived_terminal_ratings), (
        "the carried-forward state does not cover the same teams as the "
        f"{LAST_SEASON} terminal state: "
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
        f"the {CARRYOVER_TARGET_SEASON} week-1 pre-rating does not equal the "
        f"{LAST_SEASON} final with the carryover shrink ({shrink}) applied "
        "exactly once, for: "
        f"{failures} -- each entry (team, {LAST_SEASON} final, expected, actual), "
        f"compared to {CARRYOVER_PLACES} decimal places. Every failing team is "
        "named rather than only the first."
    )


def test_a_second_carryover_call_moves_no_rating(carried_forward_ratings):
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
        f"a SECOND apply_season_carryover({CARRYOVER_TARGET_SEASON}) call moved "
        f"{len(moved)} rating(s): {moved}. The carryover is supposed to be "
        "idempotent for a season it has already been applied for; a second "
        "application shrinks every rating toward 1500 again."
    )


def test_the_provisional_week_one_snapshot_reads_the_carried_forward_rating(
    carried_forward_ratings, data_boundary_guard
):
    """The live path's week-1 pre-rating IS the carried-forward number.

    Driven against an IN-MEMORY schedule frame, never a stored 2026 row: none
    exists at this point in the phase, and the STORED rows are asserted in Plan
    33-18. ``data_boundary_guard`` is requested here because this is the one test
    in the module that drives a writer's code path, so "it wrote nothing" is
    settled by content digests rather than by reading the implementation.
    """
    from scripts.build_elo import SNAPSHOT_COLUMNS, EloBuilder

    once = carried_forward_ratings["once"]
    teams = sorted(once)
    home, away = teams[0], teams[1]

    schedule = pd.DataFrame(
        [
            {
                "game_id": f"{CARRYOVER_TARGET_SEASON}_W01_{away}@{home}",
                "season": CARRYOVER_TARGET_SEASON,
                "week": 1,
                "kickoff_et": pd.Timestamp(
                    f"{CARRYOVER_TARGET_SEASON}-09-10T20:00:00Z"
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
    builder.elo_system.apply_season_carryover(CARRYOVER_TARGET_SEASON)

    frame = builder.snapshot_upcoming_week(CARRYOVER_TARGET_SEASON, 1, games=schedule)

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
        f"the {CARRYOVER_TARGET_SEASON} week-1 home pre-rating for {home} is "
        f"{float(row['home_elo_pre']):.4f}, not the carried-forward "
        f"{once[home]:.4f}."
    )
    assert round(float(row["away_elo_pre"]), CARRYOVER_PLACES) == round(
        once[away], CARRYOVER_PLACES
    ), (
        f"the {CARRYOVER_TARGET_SEASON} week-1 away pre-rating for {away} is "
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
