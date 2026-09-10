"""Ingest historical NFL betting odds data from nflreadpy.

Transforms nflreadpy schedule data (which includes closing lines) into standardized
``OddsSchema`` records with per-game Friday 6 PM Eastern freeze timestamps. Stores as timestamped
Bronze snapshots and merges into Silver.

THE WRITE CONTRACT IS PRE-REGISTERED (Plan 31-08)
-------------------------------------------------
Four separately-attributable facts about what this module WRITES are clauses of the FROZEN
Phase-31 pre-registration (``PROFITABILITY-PREREGISTRATION.md`` section 4.3 plus
``backtest/ev_chain_constants.py``), not implementation details anyone may quietly change:

1. **Label.** Every row carries :data:`~backtest.ev_chain_constants.ODDS_SPORTSBOOK_LABEL`, which
   is the label the live silver rows already hold. The literal this module wrote from v1.0
   Phase 02 (commit ``b3f158d``) until Plan 31-08 was a DOCUMENTED LEGACY MISLABEL: no row in live
   silver ever carried it, and the OUM-06 provenance allowlist
   (``backtest.ou_divergence._ALLOWED_SPORTSBOOKS``) would have hard-failed the verdict run at
   step one. The allowlist is NOT widened (D31-12 branch 1); the label is corrected at the source.
2. **Juice is ADDITIVE.** The four ``OddsSchema`` juice columns are written keyed on ``game_id``
   and the stored ``spread``, ``total``, ``ml_home`` and ``ml_away`` are NEVER overwritten
   (D31-15). A missing juice value is a FAILURE TO INVESTIGATE, never a -110 default to fill:
   nflreadpy carries all four on every admitted game for every season this project ingests.
3. **``snapshot_ts`` is RE-DERIVED per game** through :func:`get_synthetic_snapshot_ts`, from that
   game's OWN preceding Friday 6 PM Eastern instant, and stored as a tz-aware datetime (D31-37).
   Every replay row then sits exactly AT its freeze and is FRESH under the rule that at-freeze is
   fresh. The alternative -- exempting replay rows from the freshness check -- was REJECTED
   because it creates a SECOND freshness rule.
4. **Game-type scope.** The admitted set is READ from the frozen constants
   (:func:`admitted_game_types`), never declared here. Playoffs are admitted EVERYWHERE, in both
   the tune window and the hold (D31-38); preseason is excluded.

A TYPE TRAP TRAVELS WITH CLAUSE 3
---------------------------------
The stored ``snapshot_ts`` column is a STRING today, holding one fixed calendar date per season,
and its single non-consensus row uses a different, space-separated UTC format. Any comparison
against a freeze instant must PARSE and must never string-compare, and it must be
Eastern-anchored rather than UTC-anchored. :func:`normalize_snapshot_ts` is the ONE parse path;
:func:`is_fresh_at_freeze` is the ONE comparison built on it. Plan 31-09's selector calls the
same two functions, so the value written and the value compared come from one rule.

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not write silver without first running the pre-ingest gates. The synthetic-id gate and
the SPEC R2 completeness gate live in ``scripts/audit_odds_preingest.py`` (Plan 31-02) precisely
so a production write path can import them, and :func:`write_odds_additively` calls them BEFORE
the merge, never after.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa

from backtest.ev_chain_constants import (
    HOLD_GAME_TYPES,
    HOLD_SEASONS_P31,
    ODDS_SPORTSBOOK_LABEL,
    TUNE_GAME_TYPES,
)
from data import upstream_pin
from data.quality_gates import validate_bronze_to_silver
from data.schemas import OddsSchema
from data.storage import (
    _atomic_write_parquet,
    save_bronze_snapshot,
    upsert_silver,
)
from scripts.audit_odds_preingest import (
    assert_2025_odds_completeness,
    assert_no_synthetic_game_ids,
    find_synthetic_game_ids,
)
from utils import DataIngestionError, get_logger, log_data_operation
from utils.game_id_utils import GAME_ID_PATTERN, create_standard_game_id
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# The freeze instant, and the columns the write contract names.
# ---------------------------------------------------------------------------

# The market's own timezone. EVERY freeze instant and every parsed snapshot value is anchored
# here, never in the process's local zone: a UTC-anchored or local-anchored comparison silently
# moves the freeze by the offset and suppresses or admits the wrong rows (T-31-37).
EASTERN = ZoneInfo("America/New_York")

# 6 PM Eastern on the preceding Friday. This is the project's data-freeze convention.
FREEZE_HOUR_ET = 18

# The four juice columns clause 2 adds. Derived from the schema rather than restated by hand
# would be better still, but these four are named individually in the pre-registration, so they
# are named individually here and asserted against the schema below.
JUICE_COLUMNS: tuple[str, ...] = (
    "spread_ju_home",
    "spread_ju_away",
    "total_over_ju",
    "total_under_ju",
)

# The four stored values clause 2 forbids overwriting.
PROTECTED_LINE_COLUMNS: tuple[str, ...] = ("spread", "total", "ml_home", "ml_away")

# The triple clause 6 requires to be unique after any ingest.
ODDS_KEY_COLUMNS: tuple[str, ...] = ("game_id", "sportsbook", "snapshot_ts")

# Where each juice column is read from on an nflreadpy schedule row.
_SCHEDULE_JUICE_SOURCE: dict[str, str] = {
    "spread_ju_home": "home_spread_odds",
    "spread_ju_away": "away_spread_odds",
    "total_over_ju": "over_odds",
    "total_under_ju": "under_odds",
}

# The historical backfill window. 2025 is DELIBERATELY absent: the single unburned season is
# spent once, by an explicit --seasons 2025, never by a default that quietly includes it.
DEFAULT_HISTORICAL_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

# The gold O/U matrix the synthetic-id gate checks membership against.
DEFAULT_FEATURES_OU_PATH = Path("data/gold/features_ou.parquet")

_SILVER_ODDS_TABLE = "odds_snapshot"


def _assert_juice_columns_match_the_schema() -> None:
    """The four names above must BE the schema's juice fields, not merely resemble them."""
    schema_juice = tuple(
        name
        for name in OddsSchema.model_fields
        if name.endswith("_ju") or "_ju_" in name
    )
    if set(schema_juice) != set(JUICE_COLUMNS):
        msg = (
            f"JUICE_COLUMNS {JUICE_COLUMNS} disagrees with OddsSchema's juice fields "
            f"{schema_juice}. The pre-registration names these four columns explicitly; a "
            "disagreement means the write contract and the schema have drifted apart."
        )
        raise DataIngestionError(msg)


_assert_juice_columns_match_the_schema()


# ---------------------------------------------------------------------------
# Clause 4: the game-type scope, READ from the frozen pre-registration.
# ---------------------------------------------------------------------------


def admitted_game_types(season: int) -> frozenset[str]:
    """The game types admitted for *season*, read from the FROZEN constants.

    D31-38 is an owner decision: playoffs EVERYWHERE, so the tune and hold sets are equal today.
    They are nevertheless selected SEPARATELY here, because they are two separate clauses of the
    pre-registration and a future phase that splits them must not have to discover that this
    module silently collapsed them into one.

    Seasons outside the hold -- including the 2018-2020 bias seed window -- take the tune set.

    Args:
        season: The NFL season being ingested.

    Returns:
        The admitted ``game_type`` values for that season.
    """
    if season in HOLD_SEASONS_P31:
        return frozenset(HOLD_GAME_TYPES)
    return frozenset(TUNE_GAME_TYPES)


# ---------------------------------------------------------------------------
# Clause 3: the per-game freeze instant, and the ONE parse path.
# ---------------------------------------------------------------------------


def get_synthetic_snapshot_ts(gameday: str) -> datetime:
    """The Friday 6 PM Eastern freeze instant preceding *gameday*.

    This is the ONE source of the per-game freeze (D31-18). The rule is PER-GAME, not per-week:
    a Thursday game's preceding Friday is seven days before the Friday preceding that week's
    Sunday games, so a per-week freeze would suppress a correct Thursday snapshot every week as a
    pure calendar artifact.

    Args:
        gameday: Game date as a string (YYYY-MM-DD) or anything ``pd.to_datetime`` accepts.

    Returns:
        The freeze instant as a tz-aware UTC datetime. UTC is a storage convention only; the
        instant is COMPUTED in Eastern, so it survives daylight-saving transitions.
    """
    game_date = pd.to_datetime(gameday)

    # Find the preceding Friday (weekday 4 = Friday).
    days_since_friday = (game_date.weekday() - 4) % 7
    if days_since_friday == 0:
        days_since_friday = 7  # If gameday IS Friday, use the PRIOR Friday.
    friday = game_date - timedelta(days=days_since_friday)

    snapshot = datetime(
        friday.year, friday.month, friday.day, FREEZE_HOUR_ET, 0, tzinfo=EASTERN
    )
    return snapshot.astimezone(UTC)


def _is_null_scalar(value: Any) -> bool:
    """True when *value* is a scalar null. Array-likes are never null for this purpose."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_snapshot_ts(value: Any) -> datetime:
    """Parse ANY stored or derived snapshot value into one Eastern-anchored instant.

    This is the ONE parse path clause 3's type trap requires, shared by this ingest and by Plan
    31-09's selector so a freshness verdict cannot be reached two different ways. It handles, by
    design, every shape the live column is known to hold:

    * the legacy per-season string ``2021-09-19T18:00:00-04:00``;
    * the single non-consensus row's different, space-separated UTC string
      ``2025-09-29 18:44:09.707942+00:00``;
    * a tz-aware ``datetime`` or ``pandas.Timestamp``, which is what this module now writes.

    A NAIVE value is anchored in EASTERN, not UTC. That choice is the point of the function: an
    unqualified wall-clock time in this project's odds data is a market-local time, and reading it
    as UTC moves it by four or five hours -- across the 6 PM freeze in either direction.

    The process's own local timezone is never consulted. There is no bare ``astimezone()`` and no
    naive ``datetime.now()`` anywhere on this path, so the verdict is identical wherever it runs.

    Args:
        value: A snapshot value in any of the shapes above.

    Returns:
        The same instant as a tz-aware datetime expressed in Eastern.

    Raises:
        ValueError: on a null, empty or unparseable value. A snapshot that cannot be parsed is
            never silently treated as fresh.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            msg = "snapshot_ts is an empty string; there is no instant to compare."
            raise ValueError(msg)
        parsed = pd.Timestamp(text)
    else:
        if value is None or _is_null_scalar(value):
            msg = "snapshot_ts is null; a missing freeze instant is never treated as fresh."
            raise ValueError(msg)
        parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        # The Eastern anchor, stated rather than defaulted.
        parsed = parsed.tz_localize(EASTERN)

    return parsed.tz_convert(EASTERN).to_pydatetime()


def is_fresh_at_freeze(snapshot_value: Any, gameday: str) -> bool:
    """True when *snapshot_value* is AT or AFTER that game's own preceding-Friday freeze.

    At-freeze is FRESH (SPEC R6); strictly before the freeze is stale. Both sides of the
    comparison come from this module -- the freeze from :func:`get_synthetic_snapshot_ts` and the
    snapshot from :func:`normalize_snapshot_ts` -- so no caller can compare a parsed value against
    a re-derived freeze, and no caller can string-compare.

    Args:
        snapshot_value: The stored or derived ``snapshot_ts`` in any supported shape.
        gameday: That game's own kickoff date.

    Returns:
        True when the snapshot is fresh.
    """
    return normalize_snapshot_ts(snapshot_value) >= get_synthetic_snapshot_ts(gameday)


# ---------------------------------------------------------------------------
# Clause 5: canonical keys.
# ---------------------------------------------------------------------------


def canonical_game_id(game_id: str) -> str:
    """Re-key *game_id* through :func:`create_standard_game_id` so its teams are canonical.

    The stored odds table carries the non-canonical ``LAR`` in 116 Rams ids, and ``upsert_silver``
    keys on ``game_id``, so ``LAR`` and ``LA`` are DIFFERENT KEYS: a plain merge would silently
    duplicate every Rams game rather than replace it (measured in Plan 31-02, +19 added / 0
    replaced). Normalizing the stored keys BEFORE any merge is the pre-registered resolution;
    replacing the odds table outright was REJECTED because it discards the accumulated juice
    history the PROMOTE branch depends on.

    An id that does not match ``GAME_ID_PATTERN`` is returned UNCHANGED rather than repaired. The
    synthetic-id gate is the named place a malformed id is refused; quietly rewriting one here
    would hide exactly what that gate exists to surface.

    Args:
        game_id: A project-standard game id, or anything at all.

    Returns:
        The canonical id, or the input unchanged when it is not a parseable game id.
    """
    text = str(game_id)
    match = GAME_ID_PATTERN.match(text)
    if match is None:
        return text

    season, week, away, home = match.groups()
    try:
        return create_standard_game_id(
            season=int(season), week=int(week), away_team=away, home_team=home
        )
    except (ValueError, KeyError):
        # An unknown abbreviation is a hard-fail everywhere else in this project; here it means
        # the id is not one this ingest owns, so it is left for the synthetic-id gate to name.
        return text


def assert_canonical_game_ids(odds_df: pd.DataFrame) -> None:
    """Raise unless every ``game_id`` is canonically keyed and well-formed (clause 5).

    Args:
        odds_df: The rows about to be written.

    Raises:
        ValueError: naming every malformed and every non-canonical id.
    """
    if odds_df.empty:
        return

    ids = odds_df["game_id"].astype(str)
    malformed = sorted({gid for gid in ids if not GAME_ID_PATTERN.match(gid)})
    non_canonical = sorted(
        {
            gid
            for gid in ids
            if GAME_ID_PATTERN.match(gid) and canonical_game_id(gid) != gid
        }
    )

    if malformed or non_canonical:
        msg = (
            "non-canonical odds game_id detected before write (clause 5, T-31-35). "
            f"MALFORMED (fail GAME_ID_PATTERN): {malformed[:10]}. "
            f"NON-CANONICAL abbreviation: {non_canonical[:10]}. "
            "upsert_silver keys on game_id, so two spellings of one franchise are two keys and "
            "the merge would duplicate the game rather than replace it."
        )
        raise ValueError(msg)


def assert_one_row_per_key(odds_df: pd.DataFrame, stage: str) -> None:
    """Raise unless exactly one row exists per (game_id, sportsbook, snapshot_ts) (clause 6).

    A violation is a HARD FAILURE naming the offending keys, never a warning: a duplicated key is
    a duplicated market observation, and a duplicated market observation double-counts a bet.

    Args:
        odds_df: The rows to check.
        stage: A human name for where the check ran, quoted in the failure message.

    Raises:
        ValueError: naming the offending key triples and their counts.
    """
    if odds_df.empty:
        return

    missing = [c for c in ODDS_KEY_COLUMNS if c not in odds_df.columns]
    if missing:
        msg = (
            f"cannot check the one-row-per-key invariant at {stage}: the frame is missing "
            f"{missing}. A key column that is not present is not a key that holds."
        )
        raise ValueError(msg)

    # snapshot_ts may be a datetime on one row and a legacy string on another, so the key is
    # built from the PARSED instant wherever it parses. String-keying would call two encodings of
    # one instant two different keys, which is the failure this invariant exists to catch.
    def _key_instant(value: Any) -> Any:
        try:
            return normalize_snapshot_ts(value)
        except (ValueError, TypeError):
            return str(value)

    keys = pd.DataFrame(
        {
            "game_id": odds_df["game_id"].astype(str),
            "sportsbook": odds_df["sportsbook"].astype(str),
            "snapshot_ts": odds_df["snapshot_ts"].map(_key_instant).astype(str),
        }
    )
    counts = keys.groupby(list(ODDS_KEY_COLUMNS), dropna=False).size()
    offenders = counts[counts > 1]

    if not offenders.empty:
        named = [
            f"{gid} / {book} / {ts} x{int(n)}"
            for (gid, book, ts), n in offenders.head(10).items()
        ]
        msg = (
            f"the one-row-per-key invariant FAILED at {stage} (clause 6, T-31-35): "
            f"{len(offenders)} duplicated (game_id, sportsbook, snapshot_ts) triple(s). "
            f"Offending keys: {named}. A duplicated market observation double-counts a bet; "
            "this is a hard failure, never a warning."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# The transform.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OddsTransformReport:
    """What the transform admitted and what it dropped, as COUNTED RETURN VALUES.

    Plan 31-11 asserts these counts as its ingest step's gate. A count that exists only in a log
    line cannot be asserted, so it is not a gate.

    Attributes:
        odds: The transformed rows, in ``OddsSchema`` shape.
        admitted: Rows that survived every filter and were transformed.
        dropped_by_game_type: Rows dropped by the clause-4 scope, keyed by the dropped type.
        dropped_no_betting_data: Admitted-type rows carrying no spread, total or away moneyline.
        dropped_transform_error: Rows whose transform raised (a normalization or id failure).
    """

    odds: pd.DataFrame
    admitted: int
    dropped_by_game_type: dict[str, int] = field(default_factory=dict)
    dropped_no_betting_data: int = 0
    dropped_transform_error: int = 0


def transform_nfl_odds_with_counts(schedules_df: pd.DataFrame) -> OddsTransformReport:
    """Transform nflreadpy schedules into ``OddsSchema`` rows, reporting every drop.

    Applies the four pre-registered write-contract clauses: the frozen label, the frozen
    game-type scope, additive juice with no -110 default, and a per-game re-derived tz-aware
    ``snapshot_ts``.

    Args:
        schedules_df: A frame from ``nflreadpy.load_schedules(...).to_pandas()``.

    Returns:
        The transformed rows plus the admit and drop counts.

    Raises:
        DataIngestionError: when an admitted row is missing any of the four juice values.
            nflreadpy carries all four on every admitted game for every season this project
            ingests, so an absence is a fact to investigate, not a default to fill.
    """
    if schedules_df.empty:
        return OddsTransformReport(odds=pd.DataFrame(), admitted=0)

    seasons_present = {int(s) for s in schedules_df["season"].dropna().unique()}
    admitted_by_season = {s: admitted_game_types(s) for s in seasons_present}

    in_scope = schedules_df.apply(
        lambda row: (
            str(row["game_type"])
            in admitted_by_season.get(int(row["season"]), frozenset())
        ),
        axis=1,
    )
    filtered = schedules_df[in_scope].copy()
    dropped = schedules_df[~in_scope]
    dropped_by_game_type = {
        str(game_type): int(n)
        for game_type, n in dropped["game_type"].astype(str).value_counts().items()
    }

    if filtered.empty:
        logger.warning(
            "No in-scope games found",
            dropped_by_game_type=dropped_by_game_type,
        )
        return OddsTransformReport(
            odds=pd.DataFrame(),
            admitted=0,
            dropped_by_game_type=dropped_by_game_type,
        )

    has_betting_data = (
        filtered["spread_line"].notna()
        | filtered["total_line"].notna()
        | filtered["away_moneyline"].notna()
    )
    betting_games = filtered[has_betting_data].copy()
    dropped_no_betting_data = int((~has_betting_data).sum())

    if betting_games.empty:
        logger.warning("No games with betting odds found")
        return OddsTransformReport(
            odds=pd.DataFrame(),
            admitted=0,
            dropped_by_game_type=dropped_by_game_type,
            dropped_no_betting_data=dropped_no_betting_data,
        )

    odds_records: list[dict[str, Any]] = []
    missing_juice: list[str] = []
    dropped_transform_error = 0

    for _, game in betting_games.iterrows():
        try:
            home_team = normalize_team_abbreviation(game["home_team"])
            away_team = normalize_team_abbreviation(game["away_team"])

            game_id = create_standard_game_id(
                season=int(game["season"]),
                week=int(game["week"]),
                away_team=away_team,
                home_team=home_team,
            )

            # Clause 3: this game's OWN preceding-Friday freeze, as a tz-aware instant.
            snapshot_ts = get_synthetic_snapshot_ts(game["gameday"])

            ml_home = (
                int(game["home_moneyline"])
                if pd.notna(game.get("home_moneyline"))
                else None
            )
            ml_away = (
                int(game["away_moneyline"])
                if pd.notna(game.get("away_moneyline"))
                else None
            )

            # Clause 2: real two-sided prices, and NO -110 default. A missing value is recorded
            # and raised on after the loop, naming every affected game rather than the first.
            juice: dict[str, int] = {}
            for column, source in _SCHEDULE_JUICE_SOURCE.items():
                raw = game.get(source)
                if pd.isna(raw):
                    missing_juice.append(f"{game_id}:{column}")
                else:
                    juice[column] = int(raw)

            odds_record = {
                "game_id": game_id,
                "snapshot_ts": snapshot_ts,
                "sportsbook": ODDS_SPORTSBOOK_LABEL,
                "ml_home": ml_home,
                "ml_away": ml_away,
                "spread": game.get("spread_line")
                if pd.notna(game.get("spread_line"))
                else None,
                "total": game.get("total_line")
                if pd.notna(game.get("total_line"))
                else None,
                "is_live": False,
                "last_update": None,
                "created_at": datetime.now(UTC),
                **juice,
            }

            odds_records.append(odds_record)

        except Exception as e:
            dropped_transform_error += 1
            logger.warning(
                "Failed to transform odds record",
                home_team=game.get("home_team"),
                away_team=game.get("away_team"),
                error=str(e),
            )
            continue

    if missing_juice:
        msg = (
            f"{len(missing_juice)} juice value(s) are MISSING on admitted games: "
            f"{missing_juice[:10]}. Clause 2 of the frozen write contract forbids filling a "
            "missing price with the -110 default: both spread-side and both total-side prices "
            "are complete on every admitted game in every season this project ingests, so an "
            "absence is a source change to investigate, not a value to invent. A fabricated "
            "-110 would move the per-bet breakeven and corrupt the EV floor it feeds."
        )
        raise DataIngestionError(msg)

    odds_df = pd.DataFrame(odds_records)

    logger.info(
        "Transformed odds data",
        input_games=len(betting_games),
        output_records=len(odds_df),
        dropped_by_game_type=dropped_by_game_type,
        dropped_no_betting_data=dropped_no_betting_data,
        dropped_transform_error=dropped_transform_error,
    )

    return OddsTransformReport(
        odds=odds_df,
        admitted=len(odds_df),
        dropped_by_game_type=dropped_by_game_type,
        dropped_no_betting_data=dropped_no_betting_data,
        dropped_transform_error=dropped_transform_error,
    )


def transform_nfl_odds_to_standard_format(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """The transformed rows alone, for callers that do not need the drop counts.

    A thin wrapper over :func:`transform_nfl_odds_with_counts`, kept so the frame-returning
    signature every existing caller uses is unchanged.

    Args:
        schedules_df: A frame from ``nflreadpy.load_schedules(...).to_pandas()``.

    Returns:
        DataFrame in standard ``OddsSchema`` format.
    """
    return transform_nfl_odds_with_counts(schedules_df).odds


# ---------------------------------------------------------------------------
# The write path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OddsWriteReport:
    """What the merge did, as counted return values.

    Attributes:
        path: The silver table written.
        rows_incoming: Rows presented to the merge.
        rows_before: Rows in the destination before the merge.
        rows_after: Rows in the destination after the merge.
        stored_ids_normalized: Stored rows whose ``game_id`` was re-keyed to canonical BEFORE the
            merge (clause 5). A non-zero count here is the Rams resolution doing its job.
        rows_with_lines_preserved: Incoming rows whose ``spread``, ``total``, ``ml_home`` and
            ``ml_away`` were taken from the STORED row rather than the incoming one (clause 2).
        rows_carried_forward: Stored rows re-supplied to the write because ``upsert_silver`` keys
            on ``game_id`` alone and would otherwise have DELETED them (clause 6, CR-01/CR-03).
            A non-zero count is a second sportsbook or a second snapshot for an incoming game
            surviving the merge instead of being silently erased.
    """

    path: Path
    rows_incoming: int
    rows_before: int
    rows_after: int
    stored_ids_normalized: int
    rows_with_lines_preserved: int
    rows_carried_forward: int = 0


def _silver_table_path(base_path: Path, table_name: str) -> Path:
    return base_path / "silver" / f"{table_name}.parquet"


def normalize_stored_game_ids(
    base_path: Path,
    table_name: str = _SILVER_ODDS_TABLE,
) -> int:
    """Re-key the STORED table's ``game_id`` values to canonical, BEFORE any merge (clause 5).

    This is the pre-registered Rams resolution. Outright replacement of the odds table was
    REJECTED: it would discard the accumulated juice history the PROMOTE branch depends on. Key
    normalization is the minimal change that makes ``upsert_silver``'s own keying correct.

    The write is idempotent -- a table that is already canonical is not rewritten at all.

    Args:
        base_path: The data root whose ``silver/`` holds the table.
        table_name: The silver table to normalize.

    Returns:
        The number of rows whose ``game_id`` changed.
    """
    path = _silver_table_path(base_path, table_name)
    if not path.is_file():
        return 0

    stored = pd.read_parquet(path)
    if stored.empty or "game_id" not in stored.columns:
        return 0

    canonical = stored["game_id"].astype(str).map(canonical_game_id)
    n_changed = int((canonical != stored["game_id"].astype(str)).sum())
    if n_changed == 0:
        return 0

    stored = stored.copy()
    stored["game_id"] = canonical
    _atomic_write_parquet(pa.Table.from_pandas(stored), path)

    logger.info(
        "Normalized stored odds game_ids to canonical BEFORE merge",
        table=table_name,
        rows_rekeyed=n_changed,
    )
    return n_changed


@dataclass(frozen=True)
class SyntheticRemovalReport:
    """What the named pre-ingest removal step deleted from the STORED table.

    Attributes:
        path: The silver table read, and written only when something was removed.
        rows_before: Rows in the table before the removal.
        rows_after: Rows in the table after it.
        removed_malformed: The ids deleted for failing ``GAME_ID_PATTERN``.
        removed_orphans: The ids deleted for naming no game in the gold O/U matrix.
    """

    path: Path
    rows_before: int
    rows_after: int
    removed_malformed: tuple[str, ...]
    removed_orphans: tuple[str, ...]

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def remove_synthetic_stored_rows(
    *,
    base_path: Path,
    features_ou_df: pd.DataFrame,
    table_name: str = _SILVER_ODDS_TABLE,
) -> SyntheticRemovalReport:
    """Delete synthetic rows from the STORED odds table. The NAMED pre-ingest step.

    Clause 7's gate runs on the INCOMING frame, so it cannot see -- and the merge cannot
    remove -- a forged row already sitting in the destination. Production silver holds
    exactly one: ``2025_W01_TEST@HOME``, a hand-written fixture carrying the legitimate
    sportsbook ``draftkings``, which the OUM-06 allowlist therefore ADMITS. Plan 31-08
    recorded its survival as a carry-forward and said plainly that removing it "does not
    happen by itself".

    This is that step, and it is deliberately a step rather than a side effect of the
    merge: it DELETES production rows, so it must be invoked, counted and reported on its
    own rather than folded into a write whose report is about something else.

    The predicate is ``find_synthetic_game_ids`` -- the same function clause 7's gate
    refuses on. A row this deletes is exactly a row that gate would reject.

    Args:
        base_path: The data root whose ``silver/`` holds the table.
        features_ou_df: The gold O/U matrix orphan-hood is judged against.
        table_name: The silver table to clean.

    Returns:
        The counted record. Idempotent: a table with nothing synthetic in it is NOT
        rewritten, so a second run reports zero removals and moves no byte.
    """
    path = _silver_table_path(base_path, table_name)
    if not path.is_file():
        return SyntheticRemovalReport(
            path=path,
            rows_before=0,
            rows_after=0,
            removed_malformed=(),
            removed_orphans=(),
        )

    stored = pd.read_parquet(path)
    rows_before = len(stored)
    if stored.empty:
        return SyntheticRemovalReport(
            path=path,
            rows_before=rows_before,
            rows_after=rows_before,
            removed_malformed=(),
            removed_orphans=(),
        )

    # ORDERING GUARD, and it is load-bearing rather than defensive. Measured on
    # production silver on 2026-09-05: run BEFORE the clause-5 key normalization, this
    # step classifies all 116 non-canonical ``LAR`` Rams rows as ORPHANS -- gold keys the
    # Rams as canonical ``LA``, so a well-formed ``2018_W01_LAR@LV`` names no game in
    # features_ou -- and would DELETE sixteen seasons of real accumulated odds history.
    # Run AFTER it, the orphan set is EMPTY and the only row removed is the malformed
    # fixture. The two steps are not commutative, and the destructive order is the one an
    # operator reaches for first, so the wrong order REFUSES rather than proceeding.
    non_canonical = int(
        (
            stored["game_id"].astype(str).map(canonical_game_id)
            != stored["game_id"].astype(str)
        ).sum()
    )
    if non_canonical:
        msg = (
            f"{non_canonical} stored row(s) in '{path}' still carry a NON-CANONICAL "
            "game_id, so orphan-hood cannot be judged against gold yet: a non-canonical "
            "key names no game in features_ou and would be deleted as an orphan. Measured "
            "on production silver, running this step first would remove all 116 'LAR' "
            "Rams rows -- sixteen seasons of real odds history. Call "
            "normalize_stored_game_ids(base_path) FIRST (write-contract clause 5), then "
            "call this."
        )
        raise ValueError(msg)

    malformed, orphans = find_synthetic_game_ids(stored, features_ou_df)
    doomed = set(malformed) | set(orphans)
    if not doomed:
        return SyntheticRemovalReport(
            path=path,
            rows_before=rows_before,
            rows_after=rows_before,
            removed_malformed=(),
            removed_orphans=(),
        )

    kept = stored[~stored["game_id"].astype(str).isin(doomed)].copy()
    _atomic_write_parquet(pa.Table.from_pandas(kept), path)

    report = SyntheticRemovalReport(
        path=path,
        rows_before=rows_before,
        rows_after=len(kept),
        removed_malformed=tuple(malformed),
        removed_orphans=tuple(orphans),
    )
    logger.info(
        "Removed synthetic rows from stored odds BEFORE ingest",
        table=table_name,
        rows_before=report.rows_before,
        rows_after=report.rows_after,
        removed_malformed=list(report.removed_malformed),
        removed_orphans=list(report.removed_orphans),
    )
    return report


def preserve_stored_lines(
    incoming: pd.DataFrame, stored: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Carry the STORED spread, total and both moneylines onto matching incoming rows.

    Clause 2 makes this write ADDITIVE: the four juice columns are added, and the four stored
    line values are never overwritten. Because ``upsert_silver`` replaces a whole row by key, the
    only way to leave a stored value untouched is to carry it INTO the row being written -- which
    is what this does, matching on the canonical ``(game_id, sportsbook)`` pair.

    Args:
        incoming: The transformed rows about to be written.
        stored: The rows already in the destination.

    Returns:
        The incoming rows with stored lines carried over, and the number of rows affected.
    """
    if incoming.empty or stored.empty:
        return incoming, 0

    available = [c for c in PROTECTED_LINE_COLUMNS if c in stored.columns]
    if not available or "sportsbook" not in stored.columns:
        return incoming, 0

    lookup = (
        stored[["game_id", "sportsbook", *available]]
        .assign(
            game_id=lambda d: d["game_id"].astype(str),
            sportsbook=lambda d: d["sportsbook"].astype(str),
        )
        .drop_duplicates(subset=["game_id", "sportsbook"], keep="last")
        .set_index(["game_id", "sportsbook"])
    )

    result = incoming.copy()
    keys = pd.MultiIndex.from_arrays(
        [result["game_id"].astype(str), result["sportsbook"].astype(str)]
    )
    matched = keys.isin(lookup.index)
    n_matched = int(matched.sum())
    if n_matched == 0:
        return result, 0

    for column in available:
        stored_values = pd.Series(
            lookup[column].reindex(keys).to_numpy(), index=result.index
        )
        # ONLY WHERE THE STORED VALUE IS PRESENT (WR-14). Clause 2's intent is that a stored line
        # is never OVERWRITTEN; copying the stored cell unconditionally also overwrote a real
        # incoming line with a stored NULL. Rows carrying a null moneyline exist by construction --
        # ``transform_nfl_odds_with_counts`` writes ``ml_home = None`` when the source is null --
        # so a stored 2019 row with no moneyline would erase the ``-150`` a later nflverse pull
        # supplied, and "never overwritten" would become "never improved, and sometimes
        # destroyed".
        take = matched & stored_values.notna()
        result.loc[take, column] = stored_values[take]

    return result, n_matched


def carry_forward_unmatched_stored_rows(
    incoming: pd.DataFrame, stored: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Re-supply the stored rows a ``game_id``-keyed write would DELETE (clause 6, CR-03).

    ``upsert_silver`` keys on ``game_id`` ALONE --
    ``existing[~existing["game_id"].isin(new["game_id"])]`` -- so it deletes every stored row whose
    game appears anywhere in the incoming frame, regardless of which SPORTSBOOK wrote it. Every
    incoming row carries ``sportsbook = "consensus"``, so a re-ingest of a season erased every
    non-consensus row for those games. The module docstring records that a ``draftkings`` row was
    in production silver as recently as Plan 31-08, and the OUM-06 allowlist admits it.

    ``assert_one_row_per_key`` cannot catch that: the clobber GUARANTEES one row per key, so the
    invariant passes trivially and reports success on a table that just lost rows.

    The resolution keeps the shared ``upsert_silver`` untouched and instead hands it the
    survivors: stored rows for an incoming game whose ``(game_id, sportsbook)`` pair is NOT in the
    incoming frame are appended to what is written, so the by-``game_id`` delete removes them and
    the write puts them straight back.

    WHY THE PAIR AND NOT THE FULL ``ODDS_KEY_COLUMNS`` TRIPLE. Carrying forward on the triple was
    tried and is WRONG here, because clause 3 RE-DERIVES ``snapshot_ts`` per game on every run: a
    stored 2025 row spelled ``2025-08-29 22:00:00+00:00`` and the row this module now writes for
    the same game are the same fact at two different instants, so a triple-keyed carry-forward
    treats the re-derivation as an addition. Measured on the real merge fixture: +285 rows for 285
    incoming games, nothing replaced. The pair is also the grain
    :func:`preserve_stored_lines` already matches and de-duplicates on, so the two halves of the
    clause-2 / clause-6 contract now agree. Per-book TRAJECTORY lives in the separate
    ``odds_timeline`` table (D-11), which is composite-keyed on ``(game_id, snapshot_ts)``
    precisely because ``odds_snapshot`` is a latest-per-(game, book) table.

    Args:
        incoming: The rows about to be written (already line-preserved).
        stored: The rows currently in the destination.

    Returns:
        The frame to write, and the number of stored rows carried forward.
    """
    key = ["game_id", "sportsbook"]
    if incoming.empty or stored.empty:
        return incoming, 0
    if not set(key).issubset(incoming.columns) or not set(key).issubset(stored.columns):
        return incoming, 0

    def _pairs(frame: pd.DataFrame) -> pd.Series:
        return frame["game_id"].astype(str) + "|" + frame["sportsbook"].astype(str)

    clobbered = stored["game_id"].astype(str).isin(incoming["game_id"].astype(str))
    if not clobbered.any():
        return incoming, 0

    at_risk = stored[clobbered]
    survivors = at_risk[~_pairs(at_risk).isin(set(_pairs(incoming)))]
    if survivors.empty:
        return incoming, 0

    logger.info(
        "Carrying stored odds rows forward past the game_id-keyed write",
        rows=len(survivors),
        sportsbooks=sorted(survivors["sportsbook"].astype(str).unique()),
    )
    combined = pd.concat([incoming, survivors], ignore_index=True)
    return combined, len(survivors)


def write_odds_additively(
    incoming: pd.DataFrame,
    *,
    base_path: Path,
    features_ou_df: pd.DataFrame,
    completeness_seasons: tuple[int, ...] = (),
    table_name: str = _SILVER_ODDS_TABLE,
) -> OddsWriteReport:
    """Merge *incoming* into the silver odds table under every pre-registered invariant.

    The ORDER of the steps below is itself part of the contract, and Plan 31-08's tests pin it by
    reading this function's AST: every gate runs BEFORE the write, because a gate that runs after
    a write is a report, not a gate.

    1. The synthetic-id gate (clause 7). A ``game_id`` failing ``GAME_ID_PATTERN``, or a
       well-formed one absent from the O/U feature matrix, stops the run. The OUM-06 sportsbook
       allowlist provably does not catch this: the ``2025_W01_TEST@HOME`` row in production silver
       carries the legitimate ``draftkings`` and an impossible id, and ``assert_real_odds``
       ADMITS it.
    2. The completeness hard stop (clause 8). Zero rows, or fewer than 285 admitted 2025 games
       carrying a total, raises with the measured count.
    3. Canonical keys on the incoming rows (clause 5).
    4. Canonical keys on the STORED rows, so the merge cannot create a second key for one game.
    5. Stored lines carried onto the incoming rows (clause 2), so the merge adds juice without
       overwriting a stored line.
    6. The merge itself.
    7. The one-row-per-key invariant (clause 6), asserted on what is now on disk.

    Args:
        incoming: The transformed rows to merge.
        base_path: The data root to write under. Every test in Plan 31-08 passes a ``tmp_path``;
            production silver is written only by Plan 31-11, under CHECKPOINT 2.
        features_ou_df: The gold O/U matrix the synthetic-id gate checks membership against.
        completeness_seasons: Seasons the SPEC R2 completeness gate must hold for.
        table_name: The silver table to merge into.

    Returns:
        The counted record of what the merge did.

    Raises:
        ValueError: from any gate or invariant, naming the offending rows.
    """
    assert_no_synthetic_game_ids(incoming, features_ou_df)

    for season in completeness_seasons:
        assert_2025_odds_completeness(incoming, season=season)

    assert_canonical_game_ids(incoming)

    stored_ids_normalized = normalize_stored_game_ids(base_path, table_name)

    path = _silver_table_path(base_path, table_name)
    stored = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    rows_before = len(stored)

    rows, rows_with_lines_preserved = preserve_stored_lines(incoming, stored)

    # Clause 6 / CR-03: upsert_silver keys on game_id ALONE, so it would delete every stored row
    # for an incoming game -- other sportsbooks, other snapshots and all. Hand it the survivors so
    # the write puts them back.
    rows_to_write, rows_carried_forward = carry_forward_unmatched_stored_rows(
        rows, stored
    )

    written_path = upsert_silver(
        rows_to_write, table_name, key_column="game_id", base_path=base_path
    )

    merged = pd.read_parquet(written_path)
    assert_one_row_per_key(merged, stage=f"post-merge {table_name}")

    report = OddsWriteReport(
        path=written_path,
        rows_incoming=len(rows),
        rows_before=rows_before,
        rows_after=len(merged),
        stored_ids_normalized=stored_ids_normalized,
        rows_with_lines_preserved=rows_with_lines_preserved,
        rows_carried_forward=rows_carried_forward,
    )
    logger.info(
        "Merged odds into silver",
        table=table_name,
        rows_incoming=report.rows_incoming,
        rows_before=report.rows_before,
        rows_after=report.rows_after,
        stored_ids_normalized=report.stored_ids_normalized,
        rows_with_lines_preserved=report.rows_with_lines_preserved,
    )
    return report


@dataclass(frozen=True)
class OddsIngestReport:
    """The whole ingest run, as counted return values.

    Attributes:
        odds: The validated rows presented to the merge.
        seasons: The seasons ingested, in the order requested.
        admitted: Rows admitted across all seasons.
        dropped_by_game_type: Rows dropped by the clause-4 scope, keyed by the dropped type.
        dropped_no_betting_data: Admitted-type rows carrying no betting data.
        dropped_transform_error: Rows whose transform raised.
        write: The merge record, or None on a dry run that transformed without writing.
    """

    odds: pd.DataFrame
    seasons: tuple[int, ...]
    admitted: int
    dropped_by_game_type: dict[str, int]
    dropped_no_betting_data: int
    dropped_transform_error: int
    write: OddsWriteReport | None


def load_features_ou(path: Path = DEFAULT_FEATURES_OU_PATH) -> pd.DataFrame:
    """The gold O/U ``game_id`` column the synthetic-id gate checks membership against.

    Args:
        path: The gold O/U feature matrix.

    Returns:
        A one-column frame of known game ids.

    Raises:
        DataIngestionError: when the matrix is absent. This is deliberately NOT a soft skip: an
            odds write whose synthetic-id gate could not run is an odds write with no gate, and
            the point of clause 7 is that the gate runs BEFORE the write, every time.
    """
    if not path.is_file():
        msg = (
            f"the gold O/U feature matrix is absent at {path}, so the pre-ingest synthetic-id "
            "gate cannot run. The ingest REFUSES to write without it: a write whose gate could "
            "not run is a write with no gate."
        )
        raise DataIngestionError(msg)
    return pd.read_parquet(path, columns=["game_id"])


def ingest_historical_odds_for_seasons(
    seasons: list[int],
    *,
    base_path: Path | None = None,
    features_ou_df: pd.DataFrame | None = None,
    features_ou_path: Path = DEFAULT_FEATURES_OU_PATH,
    write: bool = True,
    save_bronze: bool = True,
) -> OddsIngestReport:
    """Ingest historical odds for *seasons* under the pre-registered write contract.

    Args:
        seasons: Seasons to ingest. 2025 is ingested only when named explicitly.
        base_path: The data root. Defaults to the configured project root; every Plan 31-08 test
            passes a ``tmp_path`` instead.
        features_ou_df: The gold O/U matrix for the synthetic-id gate. Loaded from
            *features_ou_path* when not supplied.
        features_ou_path: Where to load the O/U matrix from when one is not supplied.
        write: When False, transform and validate but perform no merge. Used to measure a
            population before deciding to write it.
        save_bronze: When False, skip the per-season Bronze snapshot.

    Returns:
        The counted record of the run.

    Raises:
        DataIngestionError: on a season that could not be loaded, or on an empty result.
    """
    if base_path is None:
        from conf.settings import get_settings

        base_path = Path(get_settings().config.data.root_path)

    all_odds: list[pd.DataFrame] = []
    admitted = 0
    dropped_by_game_type: dict[str, int] = {}
    dropped_no_betting_data = 0
    dropped_transform_error = 0

    for season in seasons:
        try:
            logger.info("Ingesting historical odds", season=season)

            # PINNED schedule read (data/upstream_pin.py). The odds rows this
            # ingest writes are derived from spread_line/total_line on the schedule
            # frame, so a live fetch here would make the ingest -- and every gold
            # market anchor built from it -- irreproducible.
            schedules = upstream_pin.load_schedules([season])
            report = transform_nfl_odds_with_counts(schedules)

            admitted += report.admitted
            dropped_no_betting_data += report.dropped_no_betting_data
            dropped_transform_error += report.dropped_transform_error
            for game_type, n in report.dropped_by_game_type.items():
                dropped_by_game_type[game_type] = (
                    dropped_by_game_type.get(game_type, 0) + n
                )

            if not report.odds.empty:
                if save_bronze:
                    save_bronze_snapshot(
                        report.odds,
                        "odds",
                        season=season,
                        week=0,  # 0 indicates a full-season snapshot
                        base_path=base_path,
                    )

                all_odds.append(report.odds)
                logger.info(
                    "Successfully processed season",
                    season=season,
                    odds_records=report.admitted,
                    dropped_by_game_type=report.dropped_by_game_type,
                )
            else:
                logger.warning("No odds data generated", season=season)

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(
                "Failed to ingest odds for season", season=season, error=str(e)
            )
            raise DataIngestionError(f"Season {season} ingestion failed: {e}")

    if not all_odds:
        raise DataIngestionError("No odds data was successfully ingested")

    combined_odds = pd.concat(all_odds, ignore_index=True)
    validated_odds = validate_bronze_to_silver(combined_odds, OddsSchema)

    write_report: OddsWriteReport | None = None
    if write:
        if features_ou_df is None:
            features_ou_df = load_features_ou(features_ou_path)
        write_report = write_odds_additively(
            validated_odds,
            base_path=base_path,
            features_ou_df=features_ou_df,
            completeness_seasons=tuple(s for s in seasons if s in HOLD_SEASONS_P31),
        )

    logger.info(
        "Historical odds ingestion completed",
        seasons=seasons,
        total_records=len(validated_odds),
        unique_games=validated_odds["game_id"].nunique(),
        admitted=admitted,
        dropped_by_game_type=dropped_by_game_type,
    )

    return OddsIngestReport(
        odds=validated_odds,
        seasons=tuple(seasons),
        admitted=admitted,
        dropped_by_game_type=dropped_by_game_type,
        dropped_no_betting_data=dropped_no_betting_data,
        dropped_transform_error=dropped_transform_error,
        write=write_report,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser, built separately so its defaults are directly assertable."""
    parser = argparse.ArgumentParser(
        description="Ingest historical NFL odds data from nflreadpy"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HISTORICAL_SEASONS),
        help=(
            "Seasons to ingest (default: "
            f"{DEFAULT_HISTORICAL_SEASONS[0]}-{DEFAULT_HISTORICAL_SEASONS[-1]}). "
            "The 2025 hold season is ingested only when named explicitly."
        ),
    )
    return parser


def main():
    """CLI entry point for historical odds data ingestion."""
    args = build_parser().parse_args()

    try:
        from utils import setup_logging

        setup_logging()

        logger.info("Starting historical odds ingestion", seasons=args.seasons)

        report = ingest_historical_odds_for_seasons(args.seasons)
        odds_df = report.odds

        log_data_operation(
            operation="ingest_historical",
            table=_SILVER_ODDS_TABLE,
            rows=len(odds_df),
            metadata={
                "seasons": args.seasons,
                "unique_games": odds_df["game_id"].nunique(),
                "data_source": "nflreadpy",
                "sportsbook": ODDS_SPORTSBOOK_LABEL,
                "admitted": report.admitted,
                "dropped_by_game_type": report.dropped_by_game_type,
            },
        )

        print(f"Successfully ingested {len(odds_df)} historical odds records")
        print(f"Covering {odds_df['game_id'].nunique()} unique games")
        print(f"Seasons: {args.seasons}")
        print(f"Admitted: {report.admitted}")
        print(f"Dropped by game type: {report.dropped_by_game_type or 'none'}")
        if report.write is not None:
            print(
                f"Store rows {report.write.rows_before} -> {report.write.rows_after}; "
                f"stored ids re-keyed to canonical: {report.write.stored_ids_normalized}; "
                f"rows whose stored lines were preserved: "
                f"{report.write.rows_with_lines_preserved}"
            )

        print("\nSample historical odds data:")
        sample_cols = ["game_id", "sportsbook", "spread", "total", "ml_home", "ml_away"]
        available_cols = [col for col in sample_cols if col in odds_df.columns]
        print(odds_df[available_cols].head().to_string(index=False))

    except Exception as e:
        logger.error("Historical odds ingestion failed", error=str(e))
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
