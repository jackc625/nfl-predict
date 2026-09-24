"""Ingest the odds-trajectory (line-movement) snapshots into ``odds_timeline``.

This is the shared ingest/capture entry point for the line-movement signal
(SIG-04), with two modes:

* ``--backfill <start> <end>`` -- the PAID historical path. For each season in
  range and each requested timestamp T -- the three week-level D-12 cadence
  samples (open Tue / intraweek Wed / late Thu, noon ET) plus the DISTINCT
  per-game lock instants of that week's games (Plan 33.2-02) -- it calls
  ``OddsAPIClient.get_historical_nfl_odds`` ONCE per timestamp (whole board),
  normalizes each raw API game envelope into a consensus-median row, and writes
  to the additive ``odds_timeline`` silver table via ``upsert_silver_composite``
  plus an append-only bronze snapshot. The actual paid pull runs in Plan 29-05
  under a paid key.
* ``--current-week`` -- the FREE-tier forward-capture path. It captures the
  current week's snapshot using the REGULAR live endpoint (``get_nfl_odds``,
  ``markets x regions`` cost), stamping ``snapshot_ts = now``. This is the entry
  point wired into the Friday orchestrator by Plan 29-08.

CRITICAL invariants:
* ``snapshot_ts`` is stamped from the ENVELOPE's actual ``timestamp`` (the
  closest available snapshot at/earlier than the requested T), NOT the requested
  T -- storing T loses provenance and distorts trajectory ordering / inter-
  snapshot spacing that the late/steam path features depend on (review 29-03
  HIGH).
* Collection CADENCE and ADMISSIBILITY are separate. The cadence is a sampling
  decision and stays week-level; admissibility is per game, and each game's lock
  comes from ``utils.game_lock`` (18:00 ET on the ET day before kickoff), an
  ``America/New_York`` instant, never a UTC-localized 18:00 (the WR-02 lesson).
  The single week-level Friday freeze this module used to request is retired.
* The backfill path HARD-FAILS on mock mode -- only REAL archived odds enter
  ``odds_timeline`` (OUM-06 discipline).
* This script writes ONLY ``odds_timeline`` + bronze; it never writes
  ``odds_snapshot`` (D-11).
* A captured event is KEYED BY THE SCHEDULE (Plan 33.2-24 step 24b): it takes the
  ``game_id`` of the silver ``games`` row with the same canonical home and away teams
  whose kickoff is nearest its ``commence_time``, within
  :data:`SCHEDULE_MATCH_TOLERANCE`. An event with no such row is REFUSED by name.
  The week is never counted from a date: that count filed the two Wednesday 2024
  Christmas games under week-16 ids no game carries, would have filed 2026's
  Thanksgiving-eve game under week 11, and refused the 2026 Wednesday opener and
  every Super Bowl since 2021 outright.
* SPEND SAFETY: the backfill SKIPS the paid call for any requested timestamp
  already covered in ``odds_timeline`` (see :func:`_snapshot_already_stored`).
  ``upsert_silver_composite`` makes the WRITE idempotent, but not the paid CALL
  -- without this guard a crash at 80% of a 360-timestamp backfill would re-buy
  ~288 timestamps of data already on disk. Coverage is recorded from the
  ENVELOPE timestamp, so a paid call that returns an empty board is not re-bought
  either (WR-04).
* SPEND CEILING: the skip guard bounds RE-spend, not spend. The loop also reads
  the ``x-requests-remaining`` credit header on EVERY paid call and aborts on a
  credit floor or a hard call ceiling (:class:`SpendGuardError`, WR-03). Before
  that guard was wired in, the header reader had zero call sites in the repo and
  nothing stood between a typo'd ``--backfill 2015 2024`` and the account
  balance.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, cast

import pandas as pd

import utils.game_lock as lock_rule
from conf.settings import get_settings
from data.schemas import OddsTimelineSchema
from data.storage import save_bronze_snapshot, upsert_silver_composite
from scripts.ingest_odds import OddsAPIClient
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
)
from utils.date_utils import ET, get_nfl_season_start
from utils.game_id_utils import normalize_team_name

logger = get_logger(__name__)

# Totals are primary for the line-movement signal (D-07); spreads are opt-in
# (Tier (a)).
DEFAULT_MARKETS = ["totals"]

# The odds_timeline value column each requested market populates. The spend guard
# needs this to answer "is this timestamp covered FOR THE MARKETS I am about to
# request", which is a different question from "does this timestamp exist" (WR-03).
_VALUE_COLUMN_FOR = {"totals": "total", "spreads": "spread"}

# Provenance label for the per-game consensus row (median across US books).
CONSENSUS_SOURCE = "consensus_median"

# Regular-season weeks captured by a full-season backfill.
_REGULAR_SEASON_WEEKS = 18

# How far a listed ``commence_time`` may sit from its scheduled kickoff and still be
# that game. MEASURED 2026-09-23 over silver ``games``: the shortest gap between two
# meetings of the same ORDERED pairing is 6.16 days (2009_W17/W18 PHI@DAL; 2022's
# BAL@CIN week 18 and wild-card round are 7.3 days apart). Three days is below half of
# that, so the nearest scheduled kickoff inside the window is the only one that can be,
# and a listing six days away from a real meeting -- a speculative playoff pairing
# priced before week 18 settled, say -- is a DIFFERENT game and is refused rather than
# attached to the meeting it happens to share teams with. A flexed kickoff moves by
# hours, well inside it.
SCHEDULE_MATCH_TOLERANCE = timedelta(hours=72)

# The silver ``games`` columns the per-game key match and the per-game lock read.
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff_et",
)

# Lookback window used to decide whether a requested snapshot timestamp T is
# ALREADY covered in odds_timeline (the spend-safety skip guard).
#
# The subtlety this window exists for: the stored ``snapshot_ts`` is the
# ENVELOPE timestamp -- the actual archived snapshot at/EARLIER than the
# requested T (review 29-03 HIGH) -- so an exact-equality check against T would
# NEVER match and the guard would be a silent no-op. Observed 2021 drift is ~5
# minutes (10-minute archive cadence), but older seasons may be coarser, so the
# window is deliberately generous.
#
# The window must stay BELOW the closest spacing of any two requested instants, or a
# snapshot stored for one would falsely count as covering the next. Cadence samples
# are 24h apart, but the requested set now also holds per-game LOCK instants
# (Plan 33.2-02), and the closest pairs are SIX hours apart: a Thursday game locks
# Wednesday 18:00 ET, six hours after the Wednesday-noon sample, and a Friday game
# locks Thursday 18:00 ET, six hours after the Thursday-noon sample. The former 12h
# window would have skipped both of those lock requests as already stored. 3h is
# half the closest spacing, and still 18x the observed ~10-minute archive drift.
_SNAPSHOT_MATCH_LOOKBACK = timedelta(hours=3)

# WR-03 spend ceiling. The skip guard bounds RE-spend; these bound spend itself.
#
# A full single-season pull is 18 weeks x (3 cadence samples + about 3-4 distinct
# locks) = roughly 110-130 paid calls, so 400
# leaves room for a deliberate multi-season run while keeping a typo'd
# `--backfill 2015 2024` (which would otherwise issue ~720 calls with nothing
# between the loop and the account balance) bounded and re-runnable.
DEFAULT_MAX_PAID_CALLS = 400

# Abort when the API reports fewer than this many credits left. A FLOOR, not a
# budget: it preserves headroom for the weekly forward-collect job rather than
# draining the account to zero inside one backfill.
DEFAULT_MIN_CREDITS_REMAINING = 100

# The Odds API returns per-request credit accounting in these response headers.
_CREDITS_REMAINING_HEADER = "x-requests-remaining"

# Per-data-root cache of the games silver game_id set, so the WR-05 orphan check
# costs one parquet read per backfill rather than one per snapshot written.
# ``None`` means "games silver unavailable", which reads as "cannot check".
_GAMES_ID_CACHE: dict[str, set[str] | None] = {}


class MockModeBackfillError(DataIngestionError):
    """Raised when a historical backfill is attempted with a mock-mode client.

    Only REAL archived odds may enter ``odds_timeline``; a mock-mode client
    would synthesize fake lines, contaminating the trajectory table (OUM-06
    discipline).
    """


class SpendGuardError(DataIngestionError):
    """Raised when a paid backfill hits its credit floor or its call ceiling.

    Deliberately a hard stop rather than a warning. Rows already written are
    retained and the skip guard makes a re-run resume without re-buying them, so
    aborting costs nothing but an operator decision -- which is the point.
    """


def _credits_remaining(headers: Any) -> int | None:
    """Read ``x-requests-remaining`` off a response headers mapping.

    Returns ``None`` when the header is absent or unparseable, which the caller
    treats as "unknown" rather than as "zero": a missing header must not abort a
    legitimate backfill, and the call ceiling still bounds the run.
    """
    if headers is None:
        return None
    try:
        raw = headers.get(_CREDITS_REMAINING_HEADER)
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Unparseable Odds API credit header",
            header=_CREDITS_REMAINING_HEADER,
            value=str(raw),
        )
        return None


def weekly_cadence_timestamps(season: int, week: int) -> list[tuple[str, datetime]]:
    """Return the three week-level D-12 trajectory samples for a game week.

    COLLECTION CADENCE IS A SAMPLING DECISION, NOT AN ADMISSIBILITY BOUNDARY. These
    three fixed ET instants sample the open-to-lock path (D-08) for the steam/path
    features and legitimately stay week-level:

    * ``open``      -- Tuesday 12:00 ET
    * ``intraweek`` -- Wednesday 12:00 ET
    * ``late``      -- Thursday 12:00 ET

    There is no fourth, week-level cutoff member. What is admissible for a game is
    decided per game by its own lock -- see :func:`game_lock_instants`, which needs
    the schedule this function deliberately does not take.

    Args:
        season: NFL season year.
        week: NFL week number (1-based).

    Returns:
        A list of ``(label, snapshot_ts_utc)`` tuples, each timestamp tz-aware
        UTC.
    """
    season_start = get_nfl_season_start(season)  # first Thursday, ET
    week_thursday = season_start + timedelta(weeks=week - 1)
    week_tuesday = week_thursday - timedelta(days=2)
    week_wednesday = week_thursday - timedelta(days=1)

    def _et_noon(day: datetime) -> datetime:
        return datetime(day.year, day.month, day.day, 12, 0, tzinfo=ET)

    anchors = [
        ("open", _et_noon(week_tuesday)),
        ("intraweek", _et_noon(week_wednesday)),
        ("late", _et_noon(week_thursday)),
    ]
    return [(label, dt.astimezone(UTC)) for label, dt in anchors]


def game_lock_instants(games_df: pd.DataFrame) -> list[tuple[str, str, datetime]]:
    """The DISTINCT per-game lock instants of *games_df*, through the one rule.

    Every instant is ``utils.game_lock.lock_frame`` over the games' own kickoffs --
    18:00 ET on the ET day before kickoff (D33.2-01). Games that share a lock collapse
    into ONE entry before any caller spends a credit: every Sunday game of a week
    locks at the same Saturday 18:00 ET, so a typical week has three or four distinct
    locks (Wednesday for a Thursday game, Friday for a Saturday game, Saturday for
    the Sunday slate, Sunday for Monday night), not one per game.

    Args:
        games_df: A schedule frame carrying ``game_id`` and a tz-aware
            ``kickoff_et``.

    Returns:
        ``(label, game_ids, lock_utc)`` per distinct lock, sorted by instant, where
        ``label`` is ``"lock"`` and ``game_ids`` names every game sharing that lock,
        comma-joined and sorted.

    Raises:
        utils.game_lock.MissingKickoffError: naming every game with no kickoff -- no
            lock is emitted for the rows that could be computed.
    """
    if games_df.empty:
        return []
    locks = lock_rule.lock_frame(games_df)
    by_instant: dict[datetime, list[str]] = {}
    for game_id, lock in locks.items():
        instant = lock.to_pydatetime().astimezone(UTC)
        by_instant.setdefault(instant, []).append(str(game_id))
    return [
        ("lock", ",".join(sorted(ids)), instant)
        for instant, ids in sorted(by_instant.items())
    ]


def _week_request_instants(
    games: pd.DataFrame, season: int, week: int
) -> list[tuple[str, datetime]]:
    """The union a backfill requests for one week: cadence samples, then locks."""
    week_games = cast(
        "pd.DataFrame", games[(games["season"] == season) & (games["week"] == week)]
    )
    cadence = weekly_cadence_timestamps(season, week)
    locks = [(label, lock) for label, _ids, lock in game_lock_instants(week_games)]
    return [*cadence, *locks]


def load_timeline_schedule(base_path: Path | None) -> pd.DataFrame:
    """The silver ``games`` schedule every captured event is keyed against.

    Both entry points read it: the backfill derives each week's lock instants from it,
    and both the backfill and the live capture key every board event to a game through
    it (:func:`match_event_to_schedule`).

    Raises:
        DataIngestionError: when it cannot be read. No schedule means no per-game lock
            and no game to key an event to, and an ingest that guessed either would
            write a line under the wrong game or buy data against the wrong instant.
    """
    if base_path is None:
        base_path = Path(get_settings().config.data.root_path)
    games_path = base_path / "silver" / "games.parquet"
    try:
        return pd.read_parquet(
            games_path, columns=list(SCHEDULE_COLUMNS), engine="pyarrow"
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        msg = (
            f"cannot ingest odds_timeline without a schedule: {games_path} could not "
            f"be read ({exc}). Each event is keyed to its game and each game's lock is "
            "derived from its kickoff, so a missing schedule is refused rather than "
            "guessed around."
        )
        raise DataIngestionError(msg) from exc


def _require_schedule_columns(schedule: pd.DataFrame) -> None:
    """Refuse a schedule that cannot key an event, naming what it lacks."""
    missing = [c for c in SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing:
        msg = (
            f"the odds_timeline schedule is missing {missing}; an event is keyed to its "
            "game by canonical home team, away team and kickoff, so a schedule without "
            "them cannot key anything"
        )
        raise DataIngestionError(msg)


def _parse_envelope_timestamp(timestamp: str | None) -> datetime:
    """Parse the historical envelope ``timestamp`` to tz-aware UTC.

    The stored ``snapshot_ts`` is this ENVELOPE timestamp -- the actual snapshot
    at/earlier than the requested date -- never the requested date itself
    (review 29-03 HIGH).
    """
    if not timestamp:
        raise ValueError("Historical envelope is missing its 'timestamp' field")
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Envelope timestamp is timezone-naive: {timestamp!r}")
    return parsed.astimezone(UTC)


def _parse_commence_time(value: Any) -> datetime | None:
    """A listed ``commence_time`` as a tz-aware UTC instant, or None when unusable.

    A naive value is None rather than assumed UTC: an instant nobody stated the zone of
    is not an instant this ingest may key a game by.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def match_event_to_schedule(
    *,
    home_team: str,
    away_team: str,
    commence: datetime,
    schedule: pd.DataFrame,
) -> str | None:
    """The ``game_id`` of the scheduled game a board event IS, or None when it is none.

    THE KEY COMES FROM THE SCHEDULE (Plan 33.2-24 step 24b). This replaces a week COUNTED
    from the listed ``commence_time`` -- whole weeks since the season's opening Thursday --
    which put every game played before the Thursday of its own week into the PREVIOUS
    week. Measured over silver ``games`` 2020-2026, ten games disagreed with that count:
    the two Wednesday 2024 Christmas games (stored as week-16 ids no game carries, so their
    owned pre-lock lines joined nothing), 2026's Wednesday Thanksgiving-eve game (week 11
    for week 12), the Wednesday 2026 opener (week 0, refused), the five Super Bowls since
    2021 (week 23, refused) and the 2020 Super Bowl (week 22 where the schedule says 21).

    The match: the rows with the same CANONICAL home and away team, and among them the one
    whose kickoff is nearest ``commence``, accepted only within
    :data:`SCHEDULE_MATCH_TOLERANCE`. Home and away are ordered, so a neutral-site listing
    keyed the other way round names no game and is refused rather than flipped.

    THERE IS NO FALLBACK. An event with no row inside the window is not keyed at all --
    WR-05's rule, that an event which cannot be placed is refused rather than clamped onto a
    valid-looking key, now applied by the schedule instead of by a week range. A fallback
    that derived a week from the date is exactly the mechanism that mis-keyed the Christmas
    games silently.

    Raises:
        utils.DataValidationError: a team name that does not map to a canonical franchise
            (the project's hard-fail on unknown teams).
    """
    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)
    candidates = schedule[
        (schedule["home_team"] == home) & (schedule["away_team"] == away)
    ]
    if candidates.empty:
        return None
    gaps = (pd.to_datetime(candidates["kickoff_et"], utc=True) - commence).abs()
    nearest = gaps.idxmin()
    if gaps.loc[nearest] > SCHEDULE_MATCH_TOLERANCE:
        return None
    return str(candidates.loc[nearest, "game_id"])


def _extract_book_total(bookmaker: dict[str, Any]) -> float | None:
    """Return a single book's totals line (the Over outcome's point)."""
    for market in bookmaker.get("markets", []):
        if market.get("key") == "totals":
            for outcome in market.get("outcomes", []):
                if str(outcome.get("name", "")).lower() == "over":
                    point = outcome.get("point")
                    return float(point) if point is not None else None
    return None


def _extract_book_spread(
    bookmaker: dict[str, Any], home_team_name: str
) -> float | None:
    """Return a single book's spread line from the home team's perspective."""
    for market in bookmaker.get("markets", []):
        if market.get("key") == "spreads":
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_team_name:
                    point = outcome.get("point")
                    return float(point) if point is not None else None
    return None


def normalize_envelope_to_timeline_rows(
    envelope: dict[str, Any],
    markets: list[str],
    *,
    schedule: pd.DataFrame,
    region: str = "us",
) -> list[dict[str, Any]]:
    """Normalize a raw historical API envelope into consensus timeline rows.

    A DEDICATED raw-response normalizer (review 29-03 MED): the upstream (now retired)
    ``create_consensus_lines`` expected already ``opening_``/``snapshot_``-
    prefixed rows, NOT raw API game envelopes. Here each ``envelope["data"]``
    game is KEYED TO ITS SCHEDULED GAME (:func:`match_event_to_schedule`), its
    per-book totals (and spreads when requested) are read, and ONE consensus-median
    row per game is emitted -- stamped with ``snapshot_ts = envelope timestamp``
    (review 29-03 HIGH).

    Args:
        envelope: ``{timestamp, ..., data: [game, ...]}`` historical envelope.
        markets: Markets present (``"totals"`` always; ``"spreads"`` opt-in).
        schedule: The silver ``games`` schedule (:data:`SCHEDULE_COLUMNS`) -- the WHOLE
            schedule, never one week's slice: a Tuesday board lists games of more than
            one week, and a Wednesday game belongs to the week the schedule says.
        region: Odds region recorded as provenance (default ``"us"``).

    Returns:
        A list of dicts ready for ``OddsTimelineSchema`` validation.

    Raises:
        DataIngestionError: when the schedule lacks a column the key match reads.
    """
    _require_schedule_columns(schedule)
    snapshot_ts = _parse_envelope_timestamp(envelope.get("timestamp"))
    want_spreads = "spreads" in markets

    rows: list[dict[str, Any]] = []
    for game in envelope.get("data", []):
        home_name = game.get("home_team")
        away_name = game.get("away_team")

        # WR-05: a listing that cannot be placed is SKIPPED with a warning -- never
        # keyed by a guess and never allowed to discard the rest of a paid snapshot.
        # Mirrors the "No usable book lines" skip below.
        commence = _parse_commence_time(game.get("commence_time"))
        if commence is None:
            logger.warning(
                "Skipping game whose commence_time is missing or unreadable",
                home_team=home_name,
                away_team=away_name,
                commence_time=game.get("commence_time"),
                snapshot_ts=snapshot_ts.isoformat(),
            )
            continue

        # Canonical, hard-fail-on-unknown team mapping (CLAUDE.md constraint) happens
        # inside the match; the KEY is the scheduled game's own id.
        game_id = match_event_to_schedule(
            home_team=home_name,
            away_team=away_name,
            commence=commence,
            schedule=schedule,
        )
        if game_id is None:
            logger.warning(
                "Skipping game: no scheduled game has these teams within the match "
                "tolerance of its commence_time (refused, never keyed by a guess)",
                home_team=home_name,
                away_team=away_name,
                commence_time=game.get("commence_time"),
                tolerance_hours=SCHEDULE_MATCH_TOLERANCE.total_seconds() / 3600,
                snapshot_ts=snapshot_ts.isoformat(),
            )
            continue

        book_totals: list[float] = []
        book_spreads: list[float] = []
        for bookmaker in game.get("bookmakers", []):
            total = _extract_book_total(bookmaker)
            if total is not None:
                book_totals.append(total)
            if want_spreads:
                spread = _extract_book_spread(bookmaker, home_name)
                if spread is not None:
                    book_spreads.append(spread)

        if not book_totals and not book_spreads:
            logger.warning(
                "No usable book lines for game; skipping",
                game_id=game_id,
                snapshot_ts=snapshot_ts.isoformat(),
            )
            continue

        rows.append(
            {
                "game_id": game_id,
                "snapshot_ts": snapshot_ts,
                "total": float(median(book_totals)) if book_totals else None,
                "spread": float(median(book_spreads)) if book_spreads else None,
                "sportsbook": CONSENSUS_SOURCE,
                "region": region,
            }
        )

    return rows


def _validate_timeline_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate consensus rows against ``OddsTimelineSchema`` into a DataFrame."""
    validated = [OddsTimelineSchema(**row).model_dump() for row in rows]
    return pd.DataFrame(validated)


def _games_game_ids(base_path: Path | None = None) -> set[str] | None:
    """Return the ``game_id`` set from ``games`` silver, cached per base_path.

    Returns ``None`` when ``games`` silver is unavailable, which callers treat as
    "cannot check" rather than "everything is an orphan".
    """
    if base_path is None:
        base_path = Path(get_settings().config.data.root_path)

    key = str(base_path)
    if key in _GAMES_ID_CACHE:
        return _GAMES_ID_CACHE[key]

    ids: set[str] | None
    games_path = base_path / "silver" / "games.parquet"
    try:
        games = pd.read_parquet(games_path, columns=["game_id"], engine="pyarrow")
        ids = set(games["game_id"].astype(str))
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        # Read directly rather than via load_dataframe: this module's other
        # silver reads are base_path-parameterized parquet reads (tests point at
        # tmp_path), and load_dataframe resolves the configured root only.
        logger.debug(
            "games silver unavailable for orphan check",
            path=str(games_path),
            error=str(e),
        )
        ids = None

    _GAMES_ID_CACHE[key] = ids
    return ids


def _warn_on_orphaned_game_ids(
    timeline_df: pd.DataFrame, base_path: Path | None = None
) -> int:
    """Log the count of derived ``game_id``s that do NOT join ``games`` silver.

    WR-05. The ``game_id`` written here used to be DERIVED arithmetically from
    ``commence_time`` and never RECONCILED against the games table. That is the
    exact mechanism that orphaned the entire 2020 archive (D29-06-01) and cost a
    re-key of 1,780 paid rows, and -- after ``get_nfl_season_start`` was corrected --
    still filed the two Wednesday 2024 Christmas games under week-16 ids (Plan 33.2-24
    step 24b). Every id is now TAKEN from the schedule the caller passed
    (:func:`match_event_to_schedule`), so an orphan here means that schedule was not
    silver ``games`` -- a stale copy, or a caller's own frame -- and this is the line
    that says so.

    A single WARNING line with the orphan count would have surfaced D29-06-01 on
    the day it happened, so that is what this emits. It WARNS rather than raises:
    the paid call has already been made, and the rows are keyed to real games of
    the schedule that was supplied.

    Returns:
        The number of orphaned ``game_id`` values (0 when the check cannot run).
    """
    known = _games_game_ids(base_path)
    if known is None or not known:
        return 0

    derived = set(timeline_df["game_id"].astype(str))
    orphans = sorted(derived - known)
    if orphans:
        logger.warning(
            "Derived odds_timeline game_ids do NOT join games silver -- the "
            "trajectory for these games will be invisible to the feature builder "
            "(the D29-06-01 failure mode)",
            orphan_count=len(orphans),
            total_rows=len(timeline_df),
            sample=orphans[:5],
        )
    return len(orphans)


def _write_timeline_rows(
    rows: list[dict[str, Any]],
    season: int,
    week: int,
    base_path: Path | None = None,
) -> int:
    """Validate + write consensus rows to bronze and ``odds_timeline`` silver.

    Returns the number of rows written (0 when ``rows`` is empty).
    """
    if not rows:
        return 0

    timeline_df = _validate_timeline_rows(rows)
    _warn_on_orphaned_game_ids(timeline_df, base_path=base_path)
    save_bronze_snapshot(
        timeline_df, "odds_timeline", season=season, week=week, base_path=base_path
    )
    upsert_silver_composite(
        timeline_df,
        "odds_timeline",
        key_columns=["game_id", "snapshot_ts"],
        base_path=base_path,
    )
    return len(timeline_df)


def _load_stored_snapshot_timestamps(
    base_path: Path | None = None,
    markets: list[str] | None = None,
) -> set[pd.Timestamp]:
    """Return the ``snapshot_ts`` values already stored FOR EVERY REQUESTED MARKET.

    WR-03. This previously read only the ``snapshot_ts`` column, so the requested
    markets never entered the coverage question. A ``--markets totals`` run followed
    by ``--markets totals spreads`` would find every timestamp already present, skip
    all of them, make ZERO paid calls, write zero rows, log a large skip count, exit
    zero, and leave ``spread`` null forever -- which is precisely the resumed-backfill
    case this guard's own docstring claims to serve.

    A timestamp now counts as covered only when EVERY requested market has a
    non-null value at it, so a widened market list correctly re-fetches.

    Read ONCE at the start of a backfill so the spend-safety guard costs a single
    parquet read rather than a read per timestamp. Returns an empty set when the
    silver table does not exist yet (the first run).

    Args:
        base_path: Optional data root (for tests); defaults to settings.
        markets: The markets this run will request. Defaults to
            :data:`DEFAULT_MARKETS`.
    """
    if base_path is None:
        base_path = Path(get_settings().config.data.root_path)
    if markets is None:
        markets = DEFAULT_MARKETS

    silver_path = base_path / "silver" / "odds_timeline.parquet"
    if not silver_path.exists():
        return set()

    value_columns = [
        _VALUE_COLUMN_FOR[market] for market in markets if market in _VALUE_COLUMN_FOR
    ]
    stored = pd.read_parquet(
        silver_path, columns=["snapshot_ts", *value_columns], engine="pyarrow"
    )
    if value_columns:
        stored = stored.dropna(subset=value_columns)

    return set(pd.to_datetime(stored["snapshot_ts"], utc=True).unique())


def _snapshot_already_stored(requested_t: datetime, stored: set[pd.Timestamp]) -> bool:
    """Return True when *requested_t* is already covered by a stored snapshot.

    The API returns the closest archived snapshot at/EARLIER than the requested
    T, and that ENVELOPE timestamp is what gets stored -- so coverage is tested
    as "a stored snapshot falls in ``(T - _SNAPSHOT_MATCH_LOOKBACK, T]``", never
    as equality with T (which would never match; see
    :data:`_SNAPSHOT_MATCH_LOOKBACK`).
    """
    upper = pd.Timestamp(requested_t)
    lower = upper - _SNAPSHOT_MATCH_LOOKBACK
    return any(lower < ts <= upper for ts in stored)


def backfill_timeline(
    seasons: list[int],
    client: OddsAPIClient,
    markets: list[str] | None = None,
    weeks: list[int] | None = None,
    base_path: Path | None = None,
    max_paid_calls: int = DEFAULT_MAX_PAID_CALLS,
    min_credits_remaining: int = DEFAULT_MIN_CREDITS_REMAINING,
    games: pd.DataFrame | None = None,
) -> int:
    """Backfill the odds trajectory from the PAID historical endpoint.

    For each season and week it requests the union of the three week-level cadence
    samples and the week's DISTINCT per-game lock instants (Plan 33.2-02): for each
    requested timestamp T it calls ``get_historical_nfl_odds`` ONCE (whole board),
    normalizes the envelope into consensus rows stamped on the ENVELOPE timestamp,
    and writes them idempotently into ``odds_timeline``.

    HARD-FAILS on mock mode -- only real archived odds enter ``odds_timeline``
    (OUM-06). The check runs BEFORE any call so no synthetic data is ever
    fetched.

    SPEND SAFETY: any requested timestamp already covered in ``odds_timeline``
    is SKIPPED without issuing the paid call, so resuming an interrupted
    backfill re-buys nothing (``upsert_silver_composite`` makes the WRITE
    idempotent, not the CALL). Coverage is tested against the stored ENVELOPE
    timestamps via :func:`_snapshot_already_stored`, and is MARKETS-AWARE (WR-03):
    a timestamp counts as covered only when every requested market already has a
    non-null value there, so widening ``markets`` re-fetches instead of silently
    skipping and leaving the new market's column null forever.

    SPEND CEILING (WR-03): the skip guard bounds RE-spend, not spend. Before this
    fix nothing sat between this loop and the account balance: a wrong
    ``--backfill 2015 2024`` issued 18 weeks x 4 cadence points x N seasons paid
    calls, and the credit-header reader that the module's prose credited as "the
    cost guard" had ZERO call sites anywhere in the repo -- it existed only as an
    unused method. The guard is now wired in. Every call reads
    ``x-requests-remaining`` off the response and the loop ABORTS on either a
    credit floor or a hard call ceiling, so an operator typo costs at most
    ``max_paid_calls`` credits' worth of calls instead of the whole balance.

    Args:
        seasons: Seasons to backfill.
        client: A REAL (non-mock) ``OddsAPIClient``.
        markets: Markets to fetch (default ``["totals"]``; add ``"spreads"`` for
            Tier (a)).
        weeks: Optional explicit week list (default weeks 1..18).
        base_path: Optional data root (for tests); defaults to settings.
        max_paid_calls: Hard ceiling on PAID calls in one invocation. Sized above
            a full single-season pull (18 weeks x 3 cadence samples plus about
            3-4 distinct locks, roughly 110-130 calls) with room for a multi-season
            run, and far below a runaway.
        min_credits_remaining: Abort when the API reports fewer remaining credits
            than this. A floor, not a budget: it leaves headroom for the weekly
            forward-collect job rather than draining the account to zero.
        games: The schedule the lock instants are derived from AND every board event
            is keyed against (:data:`SCHEDULE_COLUMNS`). Read from silver ``games``
            under *base_path* when omitted. The WHOLE schedule keys events: a week's
            Tuesday board lists games of later weeks too.

    Returns:
        Total rows written across all snapshots.

    Raises:
        DataIngestionError: If no schedule is supplied and none can be read. Raised
            before any paid call.
        MockModeBackfillError: If ``client`` is in mock mode.
        SpendGuardError: If the credit floor or the call ceiling is hit. Raised
            AFTER the current snapshot's rows are written, so an abort never
            discards a call that was already paid for.
    """
    if client.mock_mode:
        raise MockModeBackfillError(
            "Refusing to backfill odds_timeline with a mock-mode client: only "
            "real archived odds may enter the trajectory table (OUM-06). "
            "Provide a paid ODDS_API_KEY."
        )

    if markets is None:
        markets = DEFAULT_MARKETS

    # The schedule the per-game lock instants come from and every event is keyed
    # against, resolved and checked BEFORE any paid call.
    schedule = load_timeline_schedule(base_path) if games is None else games
    _require_schedule_columns(schedule)

    # Spend-safety guard: one read of what is already on disk FOR THESE MARKETS,
    # kept current as the loop writes so a resumed backfill never re-buys stored
    # timestamps -- and never skips a timestamp that lacks a requested market.
    stored_snapshots = _load_stored_snapshot_timestamps(base_path, markets=markets)

    total_written = 0
    calls_made = 0
    snapshots_skipped = 0
    for season in seasons:
        target_weeks = (
            weeks if weeks is not None else range(1, _REGULAR_SEASON_WEEKS + 1)
        )
        for week in target_weeks:
            for label, snapshot_t in _week_request_instants(schedule, season, week):
                t_iso = snapshot_t.strftime("%Y-%m-%dT%H:%M:%SZ")

                if _snapshot_already_stored(snapshot_t, stored_snapshots):
                    snapshots_skipped += 1
                    logger.info(
                        "Skipping already-stored trajectory snapshot (no paid call)",
                        season=season,
                        week=week,
                        cadence=label,
                        requested_t=t_iso,
                    )
                    continue

                logger.info(
                    "Backfilling trajectory snapshot",
                    season=season,
                    week=week,
                    cadence=label,
                    requested_t=t_iso,
                )
                envelope, headers = client.get_historical_nfl_odds(
                    t_iso, markets, regions="us", return_headers=True
                )
                calls_made += 1

                # WR-04: record coverage from the ENVELOPE timestamp, not from the
                # emitted rows. When ``rows`` is empty -- an envelope with no
                # board, or every game skipped by the "No usable book lines"
                # branch -- nothing used to be added to ``stored_snapshots`` and
                # nothing was written, so the next run re-requested that exact
                # timestamp and PAID for it again, forever. The envelope timestamp
                # is what ``_snapshot_already_stored`` compares against, and it is
                # available whether or not any row survived normalization.
                envelope_ts = _parse_envelope_timestamp(envelope.get("timestamp"))
                stored_snapshots.add(pd.Timestamp(envelope_ts))

                rows = normalize_envelope_to_timeline_rows(
                    envelope, markets, schedule=schedule
                )
                if not rows:
                    # Visible rather than silent: a paid call that bought nothing
                    # is exactly the event an operator needs to see. Note this
                    # in-memory record does NOT survive a process crash -- a
                    # durable fix would write a zero-row marker or a sidecar
                    # "requested timestamps" file, which is recorded as follow-up.
                    logger.warning(
                        "Paid snapshot returned NO usable rows; recording coverage "
                        "from the envelope timestamp so it is not re-bought",
                        season=season,
                        week=week,
                        cadence=label,
                        requested_t=t_iso,
                        envelope_ts=envelope_ts.isoformat(),
                    )
                total_written += _write_timeline_rows(
                    rows, season, week, base_path=base_path
                )
                stored_snapshots.update(
                    pd.Timestamp(row["snapshot_ts"]) for row in rows
                )

                # WR-03: the cost guard, AFTER the write so an abort never
                # discards a call that has already been paid for.
                remaining = _credits_remaining(headers)
                logger.info(
                    "Odds API credit usage",
                    last=headers.get("x-requests-last"),
                    remaining=remaining,
                    calls_made=calls_made,
                )
                if remaining is not None and remaining < min_credits_remaining:
                    raise SpendGuardError(
                        f"Aborting backfill: only {remaining} Odds API credits "
                        f"remain (floor {min_credits_remaining}); {calls_made} "
                        f"paid calls made, {total_written} rows written."
                    )
                if calls_made >= max_paid_calls:
                    raise SpendGuardError(
                        f"Aborting backfill: hit the {max_paid_calls}-paid-call "
                        f"ceiling; {total_written} rows written. Re-run to "
                        f"continue -- stored snapshots are skipped without a "
                        f"paid call."
                    )

    logger.info(
        "Odds-timeline backfill completed",
        seasons=seasons,
        rows_written=total_written,
        paid_calls_made=calls_made,
        snapshots_skipped=snapshots_skipped,
    )
    return total_written


def capture_current_week(
    client: OddsAPIClient,
    markets: list[str] | None = None,
    season: int | None = None,
    week: int | None = None,
    region: str = "us",
    base_path: Path | None = None,
    games: pd.DataFrame | None = None,
) -> int:
    """Capture the current week's trajectory snapshot (FREE-tier forward path).

    Uses the REGULAR live endpoint (``get_nfl_odds``, ``markets x regions``
    cost) -- NOT the paid historical endpoint -- and stamps ``snapshot_ts = now``
    (tz-aware UTC). This is the forward-collect entry point wired into the Friday
    orchestrator by Plan 29-08.

    Every board event is keyed to its scheduled game through the WHOLE schedule,
    exactly as the backfill keys it (Plan 33.2-24 step 24b). The ``season`` / ``week``
    below only label the bronze file; they never key a row. That matters most on this
    path: the 2026 season carries a Wednesday opener and a Wednesday Thanksgiving-eve
    game, which the retired week count refused and mis-filed respectively.

    Args:
        client: An ``OddsAPIClient`` (real key in automation).
        markets: Markets to fetch (default ``["totals"]``).
        season: Override season (default: current). Labels the bronze file only.
        week: Override week (default: current). Labels the bronze file only.
        region: Odds region recorded as provenance (default ``"us"``).
        base_path: Optional data root (for tests); defaults to settings.
        games: The schedule events are keyed against (:data:`SCHEDULE_COLUMNS`). Read
            from silver ``games`` under *base_path* when omitted.

    Returns:
        Rows written for the captured snapshot.

    Raises:
        DataIngestionError: when no schedule is supplied and none can be read. Raised
            BEFORE the API call, so a capture that could key nothing spends nothing.
    """
    if markets is None:
        markets = DEFAULT_MARKETS

    schedule = load_timeline_schedule(base_path) if games is None else games
    _require_schedule_columns(schedule)

    if season is None or week is None:
        current_season, current_week = get_current_nfl_week()
        season = season if season is not None else current_season
        week = week if week is not None else current_week

    raw_games = client.get_nfl_odds(markets=markets)
    snapshot_now = datetime.now(UTC)

    # Reuse the normalizer by wrapping the regular-endpoint game list in the
    # same envelope shape, stamping snapshot_ts = now.
    envelope = {
        "timestamp": snapshot_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": raw_games,
    }
    rows = normalize_envelope_to_timeline_rows(
        envelope, markets, schedule=schedule, region=region
    )
    written = _write_timeline_rows(rows, season, week, base_path=base_path)

    logger.info(
        "Captured current-week trajectory snapshot",
        season=season,
        week=week,
        rows_written=written,
    )
    return written


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description="Ingest NFL odds trajectory snapshots into odds_timeline"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--backfill",
        nargs=2,
        type=int,
        metavar=("START_SEASON", "END_SEASON"),
        help="Backfill the trajectory from the PAID historical endpoint "
        "(inclusive season range)",
    )
    mode.add_argument(
        "--current-week",
        action="store_true",
        help="Capture the current week's snapshot via the regular live endpoint "
        "(free tier)",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["totals", "spreads"],
        default=DEFAULT_MARKETS,
        help="Markets to capture (default: totals; add spreads for Tier (a))",
    )
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        metavar="WEEK",
        help="Explicit week list for --backfill (default: weeks 1-18). Bounds a "
        "paid pull to a few timestamps -- used for the pre-bulk smoke check and "
        "for targeted recovery after an interrupted backfill",
    )
    # WR-03: the spend ceiling is operator-visible and operator-tunable. Defaults
    # are the safe ones; raising them is a deliberate act, which is the point.
    parser.add_argument(
        "--max-paid-calls",
        type=int,
        default=DEFAULT_MAX_PAID_CALLS,
        help=f"Hard ceiling on PAID historical calls in one --backfill run "
        f"(default: {DEFAULT_MAX_PAID_CALLS}). A full season is roughly 110-130 "
        f"calls (18 weeks x 3 cadence samples + each week's distinct game locks). "
        f"Hitting the ceiling aborts; re-run to "
        f"continue, since stored snapshots are skipped without a paid call",
    )
    parser.add_argument(
        "--min-credits-remaining",
        type=int,
        default=DEFAULT_MIN_CREDITS_REMAINING,
        help=f"Abort --backfill when the Odds API reports fewer remaining "
        f"credits than this (default: {DEFAULT_MIN_CREDITS_REMAINING}). A floor, "
        f"not a budget: it leaves headroom for the weekly forward-collect job",
    )
    parser.add_argument("--api-key", type=str, help="Odds API key (overrides config)")
    return parser


def main():
    """CLI entry point for odds-timeline ingest/capture."""
    parser = _build_parser()
    args = parser.parse_args()

    client = None
    try:
        from utils import setup_logging

        setup_logging()

        client = OddsAPIClient(args.api_key)

        if args.backfill:
            start_season, end_season = args.backfill
            seasons = list(range(start_season, end_season + 1))
            logger.info(
                "Starting odds-timeline backfill", seasons=seasons, weeks=args.weeks
            )
            rows = backfill_timeline(
                seasons,
                client,
                markets=args.markets,
                weeks=args.weeks,
                max_paid_calls=args.max_paid_calls,
                min_credits_remaining=args.min_credits_remaining,
            )
            print(f"Backfilled {rows} odds_timeline rows across seasons {seasons}")
        else:
            logger.info("Starting current-week odds-timeline capture")
            rows = capture_current_week(client, markets=args.markets)
            print(f"Captured {rows} odds_timeline rows for the current week")

    except Exception as e:
        logger.error("Odds-timeline ingest failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
