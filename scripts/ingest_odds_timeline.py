"""Ingest the odds-trajectory (line-movement) snapshots into ``odds_timeline``.

This is the shared ingest/capture entry point for the line-movement signal
(SIG-04), with two modes:

* ``--backfill <start> <end>`` -- the PAID historical path. For each season in
  range and each weekly snapshot timestamp T (the D-12 intraweek cadence:
  open ~Tue / Wed / Thu / Fri-18:00-ET freeze), it calls
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
* The Friday-18:00 freeze fence is computed in ``ZoneInfo("America/New_York")``,
  never UTC (a UTC-localized 18:00 is 4-5 hours early and silently drops
  legitimate ET-evening snapshots -- the WR-02 lesson).
* The backfill path HARD-FAILS on mock mode -- only REAL archived odds enter
  ``odds_timeline`` (OUM-06 discipline).
* This script writes ONLY ``odds_timeline`` + bronze; it never writes
  ``odds_snapshot`` (D-11).
* SPEND SAFETY: the backfill SKIPS the paid call for any requested timestamp
  already covered in ``odds_timeline`` (see :func:`_snapshot_already_stored`).
  ``upsert_silver_composite`` makes the WRITE idempotent, but not the paid CALL
  -- without this guard a crash at 80% of a 360-timestamp backfill would re-buy
  ~288 timestamps of data already on disk.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from conf.settings import get_settings
from data.schemas import OddsTimelineSchema
from data.storage import save_bronze_snapshot, upsert_silver_composite
from scripts.ingest_odds import OddsAPIClient
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
    get_snapshot_time,
)
from utils.date_utils import ET, get_nfl_season_start
from utils.game_id_utils import create_standard_game_id

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
# 12 hours is safe against FALSE skips by construction: consecutive D-12 cadence
# timestamps are at least 24h apart (Tue noon -> Wed noon -> Thu noon -> Fri
# 18:00 ET), so a snapshot stored for one cadence point can never fall inside
# the next cadence point's ``(T - 12h, T]`` window.
_SNAPSHOT_MATCH_LOOKBACK = timedelta(hours=12)


class MockModeBackfillError(DataIngestionError):
    """Raised when a historical backfill is attempted with a mock-mode client.

    Only REAL archived odds may enter ``odds_timeline``; a mock-mode client
    would synthesize fake lines, contaminating the trajectory table (OUM-06
    discipline).
    """


def weekly_snapshot_timestamps(season: int, week: int) -> list[tuple[str, datetime]]:
    """Return the four D-12 intraweek capture timestamps for a game week.

    The cadence captures the FULL open->freeze path (D-08) -- the only way to
    support the steam/path features -- as four fixed ET instants anchored to the
    game week:

    * ``open``      -- Tuesday 12:00 ET
    * ``intraweek`` -- Wednesday 12:00 ET
    * ``late``      -- Thursday 12:00 ET
    * ``freeze``    -- Friday 18:00 ET (the freeze fence; resolved via
      ``get_snapshot_time`` in ET, never UTC -- WR-02)

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

    # Resolve the Friday-18:00-ET freeze for this game week in ET (WR-02):
    # passing the week's Tuesday lands get_snapshot_time on the SAME week's
    # Friday.
    freeze_et = get_snapshot_time(week_tuesday, "Friday 18:00")

    def _et_noon(day: datetime) -> datetime:
        return datetime(day.year, day.month, day.day, 12, 0, tzinfo=ET)

    anchors = [
        ("open", _et_noon(week_tuesday)),
        ("intraweek", _et_noon(week_wednesday)),
        ("late", _et_noon(week_thursday)),
        ("freeze", freeze_et),
    ]
    return [(label, dt.astimezone(UTC)) for label, dt in anchors]


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


def _derive_season_week(commence_time: str | None) -> tuple[int, int]:
    """Derive ``(season, week)`` from a game's ``commence_time``.

    The NFL season spans Sept-Feb, so a January/February game belongs to the
    PRIOR calendar year's season. The week is the number of weeks elapsed since
    the season's first Thursday, clamped to a valid range.
    """
    if not commence_time:
        raise ValueError("Game envelope is missing its 'commence_time' field")
    kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).astimezone(
        ET
    )
    season = kickoff.year if kickoff.month >= 8 else kickoff.year - 1
    season_start = get_nfl_season_start(season)
    days_since_start = (kickoff - season_start).days
    week = max(1, min(days_since_start // 7 + 1, 22))
    return season, week


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
    region: str = "us",
) -> list[dict[str, Any]]:
    """Normalize a raw historical API envelope into consensus timeline rows.

    A DEDICATED raw-response normalizer (review 29-03 MED): the upstream
    ``create_consensus_lines`` expects already ``opening_``/``snapshot_``-
    prefixed rows, NOT raw API game envelopes. Here each ``envelope["data"]``
    game is normalized (canonical teams + ``game_id``), its per-book totals (and
    spreads when requested) are read, and ONE consensus-median row per game is
    emitted -- stamped with ``snapshot_ts = envelope timestamp`` (review 29-03
    HIGH).

    Args:
        envelope: ``{timestamp, ..., data: [game, ...]}`` historical envelope.
        markets: Markets present (``"totals"`` always; ``"spreads"`` opt-in).
        region: Odds region recorded as provenance (default ``"us"``).

    Returns:
        A list of dicts ready for ``OddsTimelineSchema`` validation.
    """
    snapshot_ts = _parse_envelope_timestamp(envelope.get("timestamp"))
    want_spreads = "spreads" in markets

    rows: list[dict[str, Any]] = []
    for game in envelope.get("data", []):
        home_name = game.get("home_team")
        away_name = game.get("away_team")
        season, week = _derive_season_week(game.get("commence_time"))

        # Canonical, hard-fail-on-unknown team mapping (CLAUDE.md constraint);
        # create_standard_game_id normalizes the full API team names internally.
        game_id = create_standard_game_id(
            season=season, week=week, away_team=away_name, home_team=home_name
        )

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
) -> int:
    """Backfill the odds trajectory from the PAID historical endpoint.

    For each season and each weekly snapshot timestamp T, calls
    ``get_historical_nfl_odds`` ONCE (whole board), normalizes the envelope into
    consensus rows stamped on the ENVELOPE timestamp, and writes them
    idempotently into ``odds_timeline``.

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

    Args:
        seasons: Seasons to backfill.
        client: A REAL (non-mock) ``OddsAPIClient``.
        markets: Markets to fetch (default ``["totals"]``; add ``"spreads"`` for
            Tier (a)).
        weeks: Optional explicit week list (default weeks 1..18).
        base_path: Optional data root (for tests); defaults to settings.

    Returns:
        Total rows written across all snapshots.

    Raises:
        MockModeBackfillError: If ``client`` is in mock mode.
    """
    if client.mock_mode:
        raise MockModeBackfillError(
            "Refusing to backfill odds_timeline with a mock-mode client: only "
            "real archived odds may enter the trajectory table (OUM-06). "
            "Provide a paid ODDS_API_KEY."
        )

    if markets is None:
        markets = DEFAULT_MARKETS

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
            for label, snapshot_t in weekly_snapshot_timestamps(season, week):
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
                envelope = client.get_historical_nfl_odds(t_iso, markets, regions="us")
                calls_made += 1
                rows = normalize_envelope_to_timeline_rows(envelope, markets)
                total_written += _write_timeline_rows(
                    rows, season, week, base_path=base_path
                )
                stored_snapshots.update(
                    pd.Timestamp(row["snapshot_ts"]) for row in rows
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
) -> int:
    """Capture the current week's trajectory snapshot (FREE-tier forward path).

    Uses the REGULAR live endpoint (``get_nfl_odds``, ``markets x regions``
    cost) -- NOT the paid historical endpoint -- and stamps ``snapshot_ts = now``
    (tz-aware UTC). This is the forward-collect entry point wired into the Friday
    orchestrator by Plan 29-08.

    Args:
        client: An ``OddsAPIClient`` (real key in automation).
        markets: Markets to fetch (default ``["totals"]``).
        season: Override season (default: current).
        week: Override week (default: current).
        region: Odds region recorded as provenance (default ``"us"``).
        base_path: Optional data root (for tests); defaults to settings.

    Returns:
        Rows written for the captured snapshot.
    """
    if markets is None:
        markets = DEFAULT_MARKETS

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
    rows = normalize_envelope_to_timeline_rows(envelope, markets, region=region)
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
                seasons, client, markets=args.markets, weeks=args.weeks
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
