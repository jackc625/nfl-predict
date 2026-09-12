"""The shared, READ-ONLY 2026 season fixtures every COLD-09 and COLD-01 test reads.

WHY ONE MODULE AND NOT ONE PER PLAN
-----------------------------------
Sixteen plans in Phase 33 need some view of the 2026 schedule: the neutral-site set, the
bye weeks, a postseason row, an ATS row on a half-point line. Sixteen private derivations
would be sixteen chances to disagree about what 2026 contains, and the disagreement would
surface as a test failure in the plan that derived it LAST. The sets are derived once,
here, and the measured expectations sit beside the derivation so the derivation has
something to be checked against.

THIS MODULE READS. IT NEVER WRITES.
-----------------------------------
``CAPTURED_SCHEDULE_PATH`` points INSIDE the gitignored production store
(``data/bronze/``), which Plan 33-01's autouse write guard now watches by content on
every test in every tier. Nothing here opens a file for writing, calls
``save_dataframe``, or creates a directory, and
``tests/unit/test_season_2026_fixture_schema.py`` proves it with an AST scan rather than
by assertion in prose. A fixture that wrote its own source would be the COLD-05 violation
class, committed by the module whose job is to make COLD-05 checkable.

THE CAPTURE IS VALIDATED THROUGH THE PRODUCTION TRANSFORMER
-----------------------------------------------------------
``transform_captured_schedule()`` runs the captured feed through
``scripts.ingest_games.GameDataIngester.transform_schedule_data`` -- the REAL transform,
called and never reimplemented. A fixture validated only against its own static contents
is self-consistent fiction: it keeps passing while the ingestion schema moves underneath
it, and the drift is discovered by whichever plan first tries to use the fixture against
real production output.

TWO FIXTURES ARE CONSTRUCTED, AND THAT IS A STATEMENT ABOUT THE DATA
---------------------------------------------------------------------
``WEEK_19_POSTSEASON_FIXTURE`` and ``HALF_POINT_ATS_ROWS`` are BUILT, not captured,
because neither case exists to be captured: all 272 rows of the 2026 capture are ``REG``
weeks 1-18 (weeks 19-22 are unseeded at capture time), and 0 of the 1,139 real ATS rows
carry ``abs(spread) <= 0.5``. Both are held-out cases the SPEC's own edge table names, and
saying so here is the difference between a held-out fixture and a stand-in for data
nobody has.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.team_data import ALL_TEAMS

REPO_ROOT = Path(__file__).resolve().parents[2]

# The single live 2026 capture, taken by Plan 32-09 Task 2 on 2026-09-11 at 11:02:52 UTC.
#
# IT LIVES UNDER THE GITIGNORED PRODUCTION STORE AND IS READ, NEVER WRITTEN. The path is
# recorded rather than globbed so that a second capture landing beside it does not
# silently change what every test in this phase means: a new capture is a new constant,
# appended, with its own measured expectations.
CAPTURED_SCHEDULE_PATH = (
    REPO_ROOT
    / "data"
    / "bronze"
    / "schedules_raw_bronze_2026_W01_20260911T110252.parquet"
)

CAPTURED_SCHEDULE_ROWS = 272
CAPTURED_SCHEDULE_COLUMNS = 46

# The captured feed's column order, MEASURED from the parquet on 2026-09-11 and recorded
# here so this module is importable -- and the constructed fixtures buildable -- on a
# checkout that does not carry the gitignored lake.
CAPTURED_FEED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "location",
    "result",
    "total",
    "overtime",
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
    "stadium_id",
    "stadium",
)

# The MEASURED bye weeks, weeks 5-14, recorded so the derivation below has something to
# be checked against rather than only itself.
#
# WEEK 12 IS THE NATURAL NEGATIVE CONTROL AND IT IS EMPTY. All 32 teams play in week 12,
# so a test that parameterises blindly over weeks 5-14 and expects a non-empty bye set
# will fail there for a reason that has nothing to do with the code under test. It is
# recorded as an explicit empty set rather than omitted, because an absent key and a
# measured zero are different facts.
BYE_TEAMS_BY_WEEK: dict[int, frozenset[str]] = {
    5: frozenset({"CAR", "KC"}),
    6: frozenset({"CIN", "DET", "MIA", "MIN"}),
    7: frozenset({"BUF", "JAX", "LAC", "WAS"}),
    8: frozenset({"HOU", "NO", "NYG", "SF"}),
    9: frozenset({"PIT", "TEN"}),
    10: frozenset({"CHI", "DEN", "PHI", "TB"}),
    11: frozenset({"ATL", "CLE", "GB", "LA", "NE", "SEA"}),
    12: frozenset(),
    13: frozenset({"BAL", "IND", "LV", "NYJ"}),
    14: frozenset({"ARI", "DAL"}),
}


class CapturedScheduleUnavailableError(RuntimeError):
    """The 2026 capture is not on this checkout.

    Raised rather than returning an empty frame, so a checkout without the gitignored
    lake produces a NAMED refusal a fixture can turn into an evidence-backed skip --
    never a silently smaller schedule that every derived set then agrees with.
    """


class CapturedScheduleShapeError(RuntimeError):
    """The 2026 capture is not the frame every expectation in this phase was measured on."""


def load_captured_schedule() -> pd.DataFrame:
    """Read the captured 2026 schedule and assert it is the frame we measured.

    Returns:
        The raw feed, 272 rows x 46 columns.

    Raises:
        CapturedScheduleUnavailableError: the parquet is absent on this checkout.
        CapturedScheduleShapeError: the parquet is present but is not 272 x 46.
    """
    if not CAPTURED_SCHEDULE_PATH.is_file():
        raise CapturedScheduleUnavailableError(
            "the captured 2026 schedule is not present at "
            f"{CAPTURED_SCHEDULE_PATH.as_posix()}. It lives under the gitignored "
            "production store, so it does not travel with the repository; re-capture it "
            "with `python -m scripts.capture_live_season` or run this suite on the "
            "machine that took it."
        )

    frame = pd.read_parquet(CAPTURED_SCHEDULE_PATH)
    if frame.shape != (CAPTURED_SCHEDULE_ROWS, CAPTURED_SCHEDULE_COLUMNS):
        raise CapturedScheduleShapeError(
            f"the captured 2026 schedule at {CAPTURED_SCHEDULE_PATH.as_posix()} is "
            f"{frame.shape} and every expectation in Phase 33 was measured on "
            f"({CAPTURED_SCHEDULE_ROWS}, {CAPTURED_SCHEDULE_COLUMNS}). The capture "
            "MOVED. Re-measure the neutral-site set, the bye weeks and the feed column "
            "list against the new capture and append them under the state manifest's "
            "append protocol -- do NOT loosen this check, which would leave every "
            "derived expectation in this phase asserting against a frame nobody looked "
            "at."
        )
    return frame


def transform_captured_schedule() -> pd.DataFrame:
    """Run the captured feed through the PRODUCTION transformer and return the result.

    Calls ``scripts.ingest_games.GameDataIngester.transform_schedule_data`` -- the same
    code path ``ingest_games`` uses before validation and the silver upsert. It is CALLED,
    never reimplemented, and the result is NOT persisted: this is the fixture's link to
    the real ingestion schema, and a reimplementation would be a second schema that
    agrees with the first only until one of them changes.

    Returns:
        The transformed frame, in the shape ``GameSchema`` validation receives.
    """
    from scripts.ingest_games import GameDataIngester

    return GameDataIngester().transform_schedule_data(load_captured_schedule())


def neutral_site_games() -> pd.DataFrame:
    """The 2026 international games -- the rows the feed marks ``location == "Neutral"``.

    Returns:
        The eight neutral-site rows, each carrying a populated ``stadium_id`` and
        ``stadium`` name.
    """
    frame = load_captured_schedule()
    return frame[frame["location"] == "Neutral"].copy()


def bye_teams_by_week() -> dict[int, frozenset[str]]:
    """Derive the bye teams for every week in the capture.

    The derivation is a set difference of the 32 CANONICAL teams (``utils.team_data``)
    against the teams playing that week -- not of the teams appearing anywhere in the
    schedule. Deriving the universe from the schedule itself would make a team missing
    from the whole season invisible, because it would be missing from both sides of the
    subtraction.

    Returns:
        ``{week: frozenset(team)}`` for every week present, including weeks with none.
    """
    frame = load_captured_schedule()
    canonical = frozenset(ALL_TEAMS)
    derived: dict[int, frozenset[str]] = {}
    for week in sorted(frame["week"].unique()):
        played = frame[frame["week"] == week]
        playing = set(played["home_team"]) | set(played["away_team"])
        derived[int(week)] = canonical - playing
    return derived


def _build_week_19_postseason_fixture() -> pd.DataFrame:
    """Construct a week-19 wild-card frame in the captured feed's column shape.

    HELD OUT BY THE SPEC'S OWN EDGE TABLE, NOT A STAND-IN FOR REAL DATA. The 2026 capture
    holds 272 ``REG`` rows across weeks 1-18; weeks 19-22 are unseeded at capture time and
    there is nothing to capture. Any plan that needs a postseason row has to build one,
    and building it once here is the difference between one constructed fixture and four.
    """
    rows = [
        {
            "game_id": "2026_19_KC_BUF",
            "season": 2026,
            "game_type": "WC",
            "week": 19,
            "gameday": "2027-01-09",
            "weekday": "Saturday",
            "gametime": "16:30",
            "away_team": "KC",
            "home_team": "BUF",
            "location": "Home",
            "stadium_id": "BUF00",
            "stadium": "Highmark Stadium",
            "spread_line": -2.5,
            "total_line": 47.5,
            "roof": "outdoors",
            "surface": "a_turf",
            "div_game": 0,
        },
        {
            "game_id": "2026_19_SF_DAL",
            "season": 2026,
            "game_type": "WC",
            "week": 19,
            "gameday": "2027-01-10",
            "weekday": "Sunday",
            "gametime": "20:15",
            "away_team": "SF",
            "home_team": "DAL",
            "location": "Home",
            "stadium_id": "DAL00",
            "stadium": "AT&T Stadium",
            "spread_line": 1.5,
            "total_line": 44.0,
            "roof": "closed",
            "surface": "matrixturf",
            "div_game": 0,
        },
    ]
    frame = pd.DataFrame(rows)
    # Reindex onto the captured feed's exact column order so a consumer cannot tell this
    # frame apart from a captured one by its shape. The columns nobody set read NA, which
    # is what an unplayed postseason game looks like in the feed anyway.
    return frame.reindex(columns=list(CAPTURED_FEED_COLUMNS))


WEEK_19_POSTSEASON_FIXTURE: pd.DataFrame = _build_week_19_postseason_fixture()

# CONSTRUCTED ATS rows covering the edge cases real data cannot supply.
#
# 0 OF 1,139 REAL ATS ROWS CARRY abs(spread) <= 0.5, so the half-point and pick-em cases
# cannot be found and must be built. They are the cases D33-31 turns on: Plan 33-10
# repairs the edge expression to ``ats_edge = ats_prediction - market_spread`` UNIFORMLY,
# including at ``market_spread == 0``, where the old ratio form forced a zero.
#
# THE PICK-EM ROW CARRIES A NON-ZERO ats_prediction ON PURPOSE. A pick-em row whose
# prediction is also zero gives the repaired branch and the old forced zero the same
# answer, so it distinguishes nothing and would let the repair ship untested at exactly
# the point it changes behaviour.
#
# ``market_spread`` and ``spread`` are MIRRORED to the same value in every row, because
# the two spellings both occur downstream (the gold matrices carry ``spread``, the
# selection path reads ``market_spread``) and a fixture that answered differently
# depending on which name a plan reached for would be a fixture that decides the outcome.
HALF_POINT_ATS_ROWS: tuple[dict[str, object], ...] = (
    {
        "game_id": "2026_W19_HALFPOINT_HOME_FAVOURITE",
        "case": "home favourite by a half point",
        "market_spread": -0.5,
        "spread": -0.5,
        "ats_prediction": -3.5,
    },
    {
        "game_id": "2026_W19_HALFPOINT_HOME_UNDERDOG",
        "case": "home underdog by a half point",
        "market_spread": 0.5,
        "spread": 0.5,
        "ats_prediction": 2.5,
    },
    {
        "game_id": "2026_W19_PICKEM",
        "case": "pick-em, with a NON-ZERO prediction so the repaired branch is testable",
        "market_spread": 0.0,
        "spread": 0.0,
        "ats_prediction": -4.0,
    },
    {
        "game_id": "2026_W19_LARGE_FAVOURITE",
        "case": "large favourite",
        "market_spread": -13.5,
        "spread": -13.5,
        "ats_prediction": -7.0,
    },
    {
        "game_id": "2026_W19_LARGE_UNDERDOG",
        "case": "large underdog",
        "market_spread": 11.0,
        "spread": 11.0,
        "ats_prediction": 14.5,
    },
    {
        "game_id": "2026_W19_NO_STORED_SPREAD",
        "case": "no stored spread at all",
        "market_spread": None,
        "spread": None,
        "ats_prediction": 1.5,
    },
)


def build_half_point_ats_frame() -> pd.DataFrame:
    """The constructed ATS edge cases as a frame.

    Returns:
        One row per entry of ``HALF_POINT_ATS_ROWS``, with ``market_spread`` / ``spread``
        as float columns so the null row reads NaN rather than object ``None``.
    """
    frame = pd.DataFrame(list(HALF_POINT_ATS_ROWS))
    for column in ("market_spread", "spread", "ats_prediction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
