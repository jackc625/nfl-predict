"""One-shot, declared-write repair of silver ``odds_snapshot`` (Plan 33.2-08, SPEC R12).

WHY THIS EXISTS
---------------
After D33.2-03 the stored lines no longer train anything -- they GRADE bets and measure CLV.
A wrong closing line silently mis-grades a bet, so the table has to be internally consistent
before any fit or grading run in this phase reads it. ``33.2-CONTEXT.md`` D33.2-08 item 4 names
the defects; D33.2-23 rules how each is settled:

1. **Spread-sign conflicts.** A row whose spread names one team as the favourite while its own
   moneyline names the other. Each is resolved against a cited source -- our own purchased
   ``odds_timeline`` FIRST, then a free public archive -- and a value that no source can settle is
   set to unknown with a recorded reason. The row is never dropped: dropping removes the game from
   grading and CLV, which is its own distortion (D33.2-23).
2. **Disputed totals** (the 28.5 on ``2023_W18_NYJ@NE``): confirmed or corrected with a citation.
3. **The stray ``snapshot_ts=`` partition folders** under ``data/silver/``: every row is examined,
   anything legitimate and missing is recovered by an insert that CANNOT overwrite, and the folders
   are then removed.
4. **The 1970 ``created_at`` family** (Task 3): one mechanical cause, detected by mechanism and by
   symptom over EVERY row, recorded ONCE as a ``[[family]]`` entry with its full membership, each
   value repaired from a defensible source or set to NULL. No row is dropped.

EVERY COUNT IS EMITTED, NEVER CARRIED
-------------------------------------
The CONTEXT's figures were written from an earlier scan and do not all reproduce. This script
prints what its own detectors measured on the table AS FOUND (``SIGN_CONFLICTS_FOUND=`` and the
rest of the output contract) on both ``--dry-run`` and ``--apply``, and every briefing, the
correction record and the rung cause quote those lines. No count below is a plan-time literal.

THE OWNER'S RULING IS A RUNTIME INPUT
-------------------------------------
``--apply`` requires ``--ruling`` (the Task-1 checkpoint answer) and ``--ruling-date``. The
ruling decides whether the stray folders are removed and whether an unsettleable value is nulled
(the locked path) or its row dropped (an amendment). Nothing here presumes the answer.

WRITES
------
``--dry-run`` (the default) writes nothing. ``--apply`` writes exactly: ``silver/odds_snapshot.parquet``
(one atomic full-table write through ``upsert_silver``), the removal of the stray folders (under
the locked ruling), and the correction record ``config/odds_corrections.toml`` (merged, never
truncated, so a later run cannot erase what an earlier run recorded).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.storage import upsert_silver
from scripts.ingest_historical_odds import canonical_game_id

ODDS_TABLE = "odds_snapshot"
TIMELINE_TABLE = "odds_timeline"
GAMES_TABLE = "games"
STRAY_DIR_GLOB = "snapshot_ts=*"
DEFAULT_RECORD_PATH = Path("config/odds_corrections.toml")

MONEYLINE_COLUMNS = ("ml_home", "ml_away")
LINE_VALUE_COLUMNS = ("spread", "total", "ml_home", "ml_away")

RULING_RATIFY_LOCKED = "ratify-locked"
RULING_AMEND_KEEP_STRAYS = "amend-keep-strays"
RULING_AMEND_DROP_ROW = "amend-drop-row"
RULINGS = (RULING_RATIFY_LOCKED, RULING_AMEND_KEEP_STRAYS, RULING_AMEND_DROP_ROW)

DISPOSITION_CORRECTED = "corrected"
DISPOSITION_NULLED = "nulled"
DISPOSITION_CONFIRMED = "confirmed"

ESPN_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event}"
    "/competitions/{event}/odds"
)
ARCHIVE_RETRIEVED_ON = "2026-09-21"
TIMELINE_SOURCE_URL = (
    "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
    "?date={snapshot}&regions=us&markets=spreads,totals (owned purchase, silver odds_timeline)"
)


# ---------------------------------------------------------------------------
# Free public archive quotes, each retrieved and cited individually.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveQuote:
    """One game's pre-game market as a free public archive records it.

    ``spread`` is stored in THIS table's convention (positive = home favoured). ESPN's own
    ``spread`` field is the home team's handicap (negative = home favoured), so every value below
    is ESPN's number with its sign flipped once, here, at the point it enters this module.
    """

    event: str
    provider: str
    game_date: str
    spread: float | None
    ml_home: float | None
    ml_away: float | None
    total: float | None
    note: str = ""

    @property
    def source_url(self) -> str:
        return ESPN_ODDS_URL.format(event=self.event)

    @property
    def detail(self) -> str:
        return f"ESPN public odds archive, provider {self.provider}, retrieved {ARCHIVE_RETRIEVED_ON}"

    def moneyline_favours_home(self) -> bool | None:
        """True / False when the moneyline names a favourite, None when it names neither."""
        if self.ml_home is None or self.ml_away is None or self.ml_home == self.ml_away:
            return None
        return self.ml_home < self.ml_away


# Retrieved 2026-09-21 from ESPN's public odds archive (the URL on each quote). The provider
# named is the one this table's single ``consensus`` book is matched against: ESPN's own
# ``consensus`` provider where the archive carries it, otherwise the only provider it carries.
# A quote is EVIDENCE, looked up by game; it is not a list of games to change. Which games are
# in conflict is decided by the detector at run time.
PUBLIC_ARCHIVE_QUOTES: dict[str, ArchiveQuote] = {
    "2018_W09_LAC@SEA": ArchiveQuote(
        "401030852",
        "consensus",
        "2018-11-04",
        1.0,
        -118.0,
        -102.0,
        48.5,
        "all ten providers quoting it name SEA the spread favourite; seven name SEA the moneyline "
        "favourite and none names LAC",
    ),
    "2020_W09_BAL@IND": ArchiveQuote(
        "401220189",
        "consensus",
        "2020-11-08",
        1.0,
        -111.0,
        -108.0,
        48.0,
    ),
    "2021_W04_CLE@MIN": ArchiveQuote(
        "401326385",
        "consensus",
        "2021-10-03",
        -1.0,
        -107.0,
        -113.0,
        51.5,
    ),
    "2021_W15_NE@IND": ArchiveQuote(
        "401326540",
        "consensus",
        "2021-12-18",
        1.5,
        -122.0,
        102.0,
        46.5,
    ),
    "2022_W08_SF@LA": ArchiveQuote(
        "401437815",
        "consensus",
        "2022-10-30",
        -1.5,
        -113.0,
        -107.0,
        42.0,
        "the archive's own consensus quote contradicts itself (SF the spread favourite, LA the "
        "moneyline favourite); the fourteen providers quoting it split",
    ),
    "2024_W17_TEN@JAX": ArchiveQuote(
        "401671634",
        "ESPN BET",
        "2024-12-29",
        -1.5,
        100.0,
        -120.0,
        38.5,
        "closing quote; the archive carries no consensus provider for this game",
    ),
    "2023_W18_NYJ@NE": ArchiveQuote(
        "401547648",
        "DraftKings",
        "2024-01-07",
        2.5,
        -142.0,
        120.0,
        28.5,
        "closing quote (opened 38); eight of the ten pre-game providers quoting a total close at "
        "28.5, one at 29 and one at 30.5",
    ),
}

# The stored totals named as disputed by D33.2-08 item 4. A total cannot contradict another
# column of its own row, so no detector can find these; they are named, and each is confirmed or
# corrected against the archive quote for its game.
DISPUTED_TOTALS: dict[str, str] = {
    "2023_W18_NYJ@NE": (
        "the owned odds_timeline reads 30.5 at its last pre-kickoff snapshot (Friday 2024-01-05 "
        "22:55 UTC, two days before kickoff); D33.2-08 item 4 asked for an outside confirmation"
    ),
}


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------


@dataclass
class Correction:
    """One individually-resolved value, written as a ``[[correction]]`` entry."""

    game_id: str
    column: str
    old_value: float | None
    new_value: float | None
    disposition: str
    reason: str
    source_url: str | None = None
    source_date: str | None = None
    source_detail: str | None = None

    def key(self) -> tuple[str, str]:
        return (self.game_id, self.column)


@dataclass
class RepairPlan:
    """Everything one run found and would change, before anything is written."""

    sign_conflicts_found: int
    opposite_convention_would_find: int
    spread_zero_rows: int
    corrections: list[Correction] = field(default_factory=list)
    unresolved_without_reason: int = 0
    strays: StrayReport | None = None
    epoch_1970_found: int = 0
    legacy_integer_found: int = 0
    created_at_member_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detection.
# ---------------------------------------------------------------------------


def _comparable(odds: pd.DataFrame) -> pd.Series:
    """Rows on which a sign conflict is decidable at all.

    A spread of exactly 0 is a pick'em: it names no favourite, so it cannot contradict a
    moneyline and is EXCLUDED here. Measured 2026-09-16 and again at every run: the live table
    holds zero such rows, so this exclusion is correct but currently inert -- it must not be
    reported as having excluded anything.
    """
    return (
        odds["spread"].notna()
        & (odds["spread"] != 0)
        & odds["ml_home"].notna()
        & odds["ml_away"].notna()
    )


def sign_conflict_mask(odds: pd.DataFrame, *, inverted: bool = False) -> pd.Series:
    """Rows whose spread sign contradicts their own moneyline.

    The stored convention is "positive spread = home favoured". A home favourite by moneyline
    has the more negative price, so a conflict is ``spread > 0`` with ``ml_home > ml_away``
    (spread says home, price says away) or ``spread < 0`` with ``ml_home < ml_away``.

    ``inverted=True`` reads the column the other way. It exists ONLY as the convention control:
    under the wrong reading almost every comparable row is a "conflict", so the stored reading
    returning far fewer is what proves the detector assumes the right convention.
    """
    agrees_home = (odds["spread"] > 0) & (odds["ml_home"] < odds["ml_away"])
    agrees_away = (odds["spread"] < 0) & (odds["ml_home"] > odds["ml_away"])
    contradicts_home = (odds["spread"] > 0) & (odds["ml_home"] > odds["ml_away"])
    contradicts_away = (odds["spread"] < 0) & (odds["ml_home"] < odds["ml_away"])
    conflict = (
        (agrees_home | agrees_away)
        if inverted
        else (contradicts_home | contradicts_away)
    )
    return _comparable(odds) & conflict


# ---------------------------------------------------------------------------
# The 1970 created_at family (Task 3): one cause, detected by mechanism AND by symptom.
# ---------------------------------------------------------------------------

CREATED_AT_FAMILY_NAME = "odds_snapshot_created_at_legacy_integer_seconds"
EARLIEST_PLAUSIBLE_CREATED_AT = pd.Timestamp("2000-01-01", tz="UTC")
_NANOSECONDS_PER_SECOND = 1_000_000_000

CREATED_AT_FAMILY_CAUSE = (
    "The first historical odds ingest (scripts/ingest_historical_odds.py as added in 7c70abf, "
    'line 71) wrote "created_at": game.get("season") -- the season integer -- and a later '
    "append read that integer column through data/storage.py::_migrate_schema_for_append's "
    'pd.to_datetime(..., unit="s"), so season 2018 became 2,018 seconds after the epoch '
    "(1970-01-01 00:33:38) and 2024 became 00:33:44."
)
CREATED_AT_FAMILY_DETECTOR = (
    "MECHANISM: created_at is non-null, has no sub-second part, and its epoch-seconds value "
    "either equals the season in the row's own game_id or decodes before 2000-01-01 UTC (a real "
    "capture instant from datetime.now carries microseconds and post-dates the project). "
    "SYMPTOM: created_at parses to calendar year 1970. A row is a member when either holds; "
    "both counts are reported separately."
)
CREATED_AT_FAMILY_REASON = (
    "The true write time of these rows is not recoverable. The stored value is the season, not "
    "a time, and no other source records when each row was written: the owned odds_timeline's "
    "timestamps are the instants its own snapshots were taken and ingested (2020-2024 and "
    "2026-08-16), not the instant an odds_snapshot row was created. So each value is set to "
    "NULL -- an honest unknown -- and the row is kept for grading and CLV (D33.2-23). "
    "created_at is lineage only; no feature, grade or model reads it."
)


def epoch_1970_mask(odds: pd.DataFrame) -> pd.Series:
    """SYMPTOM: ``created_at`` values that parse to calendar year 1970."""
    created = pd.to_datetime(odds["created_at"], utc=True, errors="coerce")
    return (created.dt.year == 1970).fillna(False).astype(bool)


def legacy_integer_created_at_mask(odds: pd.DataFrame) -> pd.Series:
    """MECHANISM: ``created_at`` values that are an integer read as seconds since the epoch.

    Tests EVERY row, not only the year-1970 ones: an integer that happens to decode to a later
    date is the same defect in a better disguise, and a year filter cannot see it.
    """
    created = pd.to_datetime(odds["created_at"], utc=True, errors="coerce")
    present = created.notna()
    nanoseconds = created.astype("int64").where(present, 0)
    whole_second = present & (nanoseconds % _NANOSECONDS_PER_SECOND == 0)
    seconds = nanoseconds // _NANOSECONDS_PER_SECOND
    season = pd.to_numeric(odds["game_id"].astype(str).str[:4], errors="coerce")
    is_season = seconds == season
    too_early = created < EARLIEST_PLAUSIBLE_CREATED_AT
    return (whole_second & (is_season | too_early)).fillna(False).astype(bool)


def created_at_family_members(odds: pd.DataFrame) -> pd.Series:
    return epoch_1970_mask(odds) | legacy_integer_created_at_mask(odds)


def build_created_at_family(member_ids: Sequence[str]) -> dict[str, Any]:
    """The single ``[[family]]`` entry: cause, detector, disposition, reason, full membership."""
    ids = sorted(member_ids)
    return {
        "name": CREATED_AT_FAMILY_NAME,
        "cause": CREATED_AT_FAMILY_CAUSE,
        "detector": CREATED_AT_FAMILY_DETECTOR,
        "disposition": "nulled",
        "disposition_counts": f"repaired 0, nulled {len(ids)}",
        "column": "created_at",
        "reason": CREATED_AT_FAMILY_REASON,
        "affected_count": len(ids),
        "affected_game_ids": ids,
    }


# ---------------------------------------------------------------------------
# The owned timeline, read ONCE with its sign flipped.
# ---------------------------------------------------------------------------


def timeline_in_snapshot_convention(timeline: pd.DataFrame) -> pd.DataFrame:
    """Return the owned timeline with its spread in THIS table's convention.

    ``odds_timeline`` stores the spread with the OPPOSITE sign to ``odds_snapshot`` and the live
    ingest (measured 2026-09-15: corr(timeline spread, stored spread) = -0.9867, and
    corr(-timeline spread, home_win) = +0.3922 against -0.3922 as stored; ``33.2-RESEARCH.md``
    sections 7.3 and 9.4). The flip happens HERE, exactly once. No caller flips it again.
    """
    flipped = timeline.copy()
    flipped["spread"] = -flipped["spread"]
    return flipped


def latest_pre_kickoff_timeline(
    timeline_flipped: pd.DataFrame, kickoffs: pd.Series
) -> dict[str, tuple[float, pd.Timestamp]]:
    """Each game's last owned-timeline spread strictly before its kickoff.

    Args:
        timeline_flipped: output of :func:`timeline_in_snapshot_convention`.
        kickoffs: tz-aware kickoff instants indexed by ``game_id``.

    Returns:
        ``game_id -> (spread in this table's convention, snapshot_ts)``; games with no usable
        pre-kickoff snapshot are absent.
    """
    frame = timeline_flipped.dropna(subset=["spread"]).copy()
    frame["kickoff"] = frame["game_id"].map(kickoffs)
    frame = frame[frame["kickoff"].notna()]
    frame = frame[pd.to_datetime(frame["snapshot_ts"], utc=True) < frame["kickoff"]]
    frame = frame.sort_values(["game_id", "snapshot_ts"])
    last = frame.groupby("game_id").tail(1)
    return {
        str(row.game_id): (float(row.spread), pd.Timestamp(row.snapshot_ts))
        for row in last.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------


def _sign(value: float | None) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == 0:
        return 0
    return 1 if value > 0 else -1


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _timeline_citation(snapshot: pd.Timestamp) -> str:
    return TIMELINE_SOURCE_URL.format(snapshot=snapshot.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _null(game_id: str, column: str, old: Any, reason: str) -> Correction:
    return Correction(game_id, column, _as_float(old), None, DISPOSITION_NULLED, reason)


@dataclass(frozen=True)
class _Evidence:
    """Which team one cited source says the market favoured, and how to cite it."""

    sign: int
    value: float
    url: str
    date: str
    detail: str
    words: str


def _timeline_evidence(timeline: tuple[float, pd.Timestamp] | None) -> _Evidence | None:
    if timeline is None or _sign(timeline[0]) == 0:
        return None
    value, snapshot = timeline
    return _Evidence(
        _sign(value),
        value,
        _timeline_citation(snapshot),
        snapshot.strftime("%Y-%m-%d"),
        "odds_timeline consensus_median, last snapshot before kickoff, sign flipped once",
        f"the owned odds_timeline's last pre-kickoff snapshot ({snapshot.isoformat()}) reads "
        f"{value:+g} in this table's convention",
    )


def _archive_evidence(quote: ArchiveQuote | None) -> _Evidence | None:
    if quote is None or quote.spread is None or _sign(quote.spread) == 0:
        return None
    return _Evidence(
        _sign(quote.spread),
        quote.spread,
        quote.source_url,
        quote.game_date,
        quote.detail,
        f"{quote.provider} in the public archive ({quote.source_url}) reads {quote.spread:+g}",
    )


def _null_all_disputed(row: pd.Series, reason: str) -> list[Correction]:
    game_id = str(row["game_id"])
    return [
        _null(game_id, column, row[column], reason)
        for column in ("spread", *MONEYLINE_COLUMNS)
    ]


def _resolve_one_conflict(
    row: pd.Series,
    timeline_latest: dict[str, tuple[float, pd.Timestamp]],
    archive: dict[str, ArchiveQuote],
) -> list[Correction]:
    """Settle one conflicting row, owned timeline first, then the public archive.

    Step 1 decides WHICH field is wrong by asking the sources which team the market favoured.
    The owned timeline is asked first; when the archive also has the game and names the OTHER
    team, the two cited sources disagree and nothing can be settled, so the disputed values are
    nulled rather than one source being silently preferred. Step 2 fixes the wrong field: a
    wrong spread takes the deciding source's value; a wrong moneyline takes the archive's pair
    only when that pair itself names the confirmed favourite, and is nulled otherwise.
    """
    game_id = str(row["game_id"])
    quote = archive.get(game_id)
    owned = _timeline_evidence(timeline_latest.get(game_id))
    public = _archive_evidence(quote)

    if owned is not None and public is not None and owned.sign != public.sign:
        return _null_all_disputed(
            row,
            f"sources disagree on the favourite: {owned.words}, while {public.words}; the stored "
            "spread and moneyline each agree with one of them, so neither can be settled and "
            "both are set to unknown (D33.2-23)",
        )
    deciding = owned if owned is not None else public
    if deciding is None:
        return _null_all_disputed(
            row,
            "no owned timeline snapshot and no public-archive quote names a favourite, so it "
            "cannot be told whether the spread or the moneyline is wrong; both are set to unknown "
            "(D33.2-23)",
        )

    if deciding.sign != _sign(row["spread"]):
        return [
            Correction(
                game_id,
                "spread",
                _as_float(row["spread"]),
                deciding.value,
                DISPOSITION_CORRECTED,
                f"the stored spread names the wrong favourite: {deciding.words}, agreeing with "
                "the stored moneyline",
                deciding.url,
                deciding.date,
                deciding.detail,
            )
        ]

    prefix = (
        f"the stored spread's favourite is confirmed ({deciding.words}); the moneyline is the "
        "disputed value"
    )
    if quote is not None and quote.moneyline_favours_home() is (deciding.sign > 0):
        return [
            Correction(
                game_id,
                column,
                _as_float(row[column]),
                getattr(quote, column),
                DISPOSITION_CORRECTED,
                f"{prefix}; corrected to the archive's {quote.provider} pair, which names the "
                "same favourite",
                quote.source_url,
                quote.game_date,
                quote.detail,
            )
            for column in MONEYLINE_COLUMNS
        ]
    if quote is None:
        why = "no public-archive quote exists for this game"
    else:
        why = (
            f"the archive's {quote.provider} pair does not name the confirmed favourite"
        )
        why += f" ({quote.note})" if quote.note else ""
    return [
        _null(
            game_id,
            column,
            row[column],
            f"{prefix}; {why}, so it cannot be settled and is set to unknown (D33.2-23)",
        )
        for column in MONEYLINE_COLUMNS
    ]


def resolve_sign_conflicts(
    odds: pd.DataFrame,
    timeline_latest: dict[str, tuple[float, pd.Timestamp]],
    archive: dict[str, ArchiveQuote] | None = None,
) -> list[Correction]:
    """One ``Correction`` per value changed or nulled to end every detected sign conflict."""
    archive = PUBLIC_ARCHIVE_QUOTES if archive is None else archive
    corrections: list[Correction] = []
    for _, row in odds[sign_conflict_mask(odds)].iterrows():
        corrections.extend(_resolve_one_conflict(row, timeline_latest, archive))
    return corrections


def resolve_disputed_totals(
    odds: pd.DataFrame,
    archive: dict[str, ArchiveQuote] | None = None,
    disputed: dict[str, str] | None = None,
) -> list[Correction]:
    """Confirm or correct each named disputed total against its archive quote."""
    archive = PUBLIC_ARCHIVE_QUOTES if archive is None else archive
    disputed = DISPUTED_TOTALS if disputed is None else disputed
    by_id = odds.set_index("game_id")
    corrections: list[Correction] = []
    for game_id, context in disputed.items():
        if game_id not in by_id.index:
            continue
        stored = _as_float(by_id.at[game_id, "total"])
        quote = archive.get(game_id)
        if quote is None or quote.total is None:
            corrections.append(
                _null(
                    game_id,
                    "total",
                    stored,
                    f"disputed total ({context}) and no public-archive quote to settle it (D33.2-23)",
                )
            )
            continue
        confirmed = stored == quote.total
        corrections.append(
            Correction(
                game_id,
                "total",
                stored,
                quote.total,
                DISPOSITION_CONFIRMED if confirmed else DISPOSITION_CORRECTED,
                (
                    f"disputed total ({context}); {quote.provider} in the public archive closes at "
                    f"{quote.total:g}"
                    + (f" ({quote.note})" if quote.note else "")
                    + (
                        "; the stored value is confirmed"
                        if confirmed
                        else "; corrected to it"
                    )
                ),
                quote.source_url,
                quote.game_date,
                quote.detail,
            )
        )
    return corrections


def apply_corrections(
    odds: pd.DataFrame,
    corrections: Iterable[Correction],
    *,
    drop_unresolvable: bool = False,
) -> pd.DataFrame:
    """Return a copy of *odds* with every correction applied.

    Under the locked ruling a nulled value is set to NaN and its row KEPT. ``drop_unresolvable``
    exists only for the ``amend-drop-row`` amendment, which removes the row instead.
    """
    repaired = odds.copy()
    position = {
        gid: idx for idx, gid in zip(repaired.index, repaired["game_id"], strict=True)
    }
    dropped: set[str] = set()
    for correction in corrections:
        idx = position[correction.game_id]
        if correction.disposition == DISPOSITION_NULLED and drop_unresolvable:
            dropped.add(correction.game_id)
            continue
        value = math.nan if correction.new_value is None else correction.new_value
        repaired.loc[idx, correction.column] = value
    if dropped:
        repaired = repaired[~repaired["game_id"].isin(dropped)]
    return repaired


# ---------------------------------------------------------------------------
# The stray partition folders.
# ---------------------------------------------------------------------------

STRAY_DUPLICATE = "duplicate_of_main_row"
STRAY_DUPLICATE_AFTER_CANONICAL_ID = "duplicate_after_canonical_id"
STRAY_SAME_KEY_DIFFERENT_VALUE = "same_game_and_book_different_value"
STRAY_BOOK_NOT_CARRIED = "book_the_main_table_does_not_carry"
STRAY_ID_NOT_IN_GAMES = "game_id_not_in_silver_games"
STRAY_RECOVERABLE = "legitimate_and_missing"


@dataclass
class StrayReport:
    """What the stray folders held and what became of each row."""

    dirs: list[str]
    files: int
    rows: int
    categories: dict[str, int]
    book_rows_with_id_not_in_games: int
    recoverable: pd.DataFrame


def load_stray_rows(silver_root: Path) -> tuple[list[Path], pd.DataFrame]:
    """Every row in every file under the stray ``snapshot_ts=`` folders, file by file.

    Files are read one at a time because their schemas differ (one file per folder carries the
    joined ``games`` columns and an integer ``created_at``); a dataset read would fail or
    silently union them.
    """
    dirs = sorted(p for p in silver_root.glob(STRAY_DIR_GLOB) if p.is_dir())
    frames = []
    for directory in dirs:
        for path in sorted(directory.glob("*.parquet")):
            frame = pd.read_parquet(path)
            keep = [
                c for c in ("game_id", "sportsbook", *LINE_VALUE_COLUMNS) if c in frame
            ]
            frame = frame[keep].copy()
            frame["source_dir"] = directory.name
            frame["source_file"] = path.name
            frames.append(frame)
    columns = [
        "game_id",
        "sportsbook",
        *LINE_VALUE_COLUMNS,
        "source_dir",
        "source_file",
    ]
    rows = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )
    return dirs, rows


def _values_equal(left: pd.Series, right: pd.Series) -> bool:
    for column in LINE_VALUE_COLUMNS:
        a, b = _as_float(left.get(column)), _as_float(right.get(column))
        if a != b:
            return False
    return True


def classify_stray_rows(
    strays: pd.DataFrame, main: pd.DataFrame, games_ids: set[str]
) -> pd.DataFrame:
    """Label every stray row with what it is relative to the main table.

    The main table's grain is ONE row per ``(game_id, sportsbook)``. A stray row is RECOVERABLE
    only when its canonical id is a real silver game, its book is one the main table carries, and
    the main table has no row for that key. Anything else is a duplicate, a disagreement (which is
    reported and never written -- the main row is never overwritten) or a row at a different
    grain.
    """
    labelled = strays.copy()
    labelled["canonical_game_id"] = labelled["game_id"].map(canonical_game_id)
    main_by_key = main.set_index(["game_id", "sportsbook"])
    carried_books = set(main["sportsbook"])
    categories = []
    for _, row in labelled.iterrows():
        key = (row["canonical_game_id"], row["sportsbook"])
        if key in main_by_key.index:
            if _values_equal(row, main_by_key.loc[key]):
                categories.append(
                    STRAY_DUPLICATE
                    if row["game_id"] == row["canonical_game_id"]
                    else STRAY_DUPLICATE_AFTER_CANONICAL_ID
                )
            else:
                categories.append(STRAY_SAME_KEY_DIFFERENT_VALUE)
        elif row["sportsbook"] not in carried_books:
            categories.append(STRAY_BOOK_NOT_CARRIED)
        elif row["canonical_game_id"] not in games_ids:
            categories.append(STRAY_ID_NOT_IN_GAMES)
        else:
            categories.append(STRAY_RECOVERABLE)
    labelled["category"] = categories
    return labelled


def build_stray_report(
    silver_root: Path, main: pd.DataFrame, games_ids: set[str]
) -> StrayReport:
    dirs, strays = load_stray_rows(silver_root)
    labelled = (
        classify_stray_rows(strays, main, games_ids)
        if len(strays)
        else strays.assign(canonical_game_id=[], category=[])
    )
    counts = labelled["category"].value_counts().to_dict() if len(labelled) else {}
    book_rows = (
        labelled[labelled["category"] == STRAY_BOOK_NOT_CARRIED]
        if len(labelled)
        else labelled
    )
    recoverable = (
        labelled[labelled["category"] == STRAY_RECOVERABLE].drop_duplicates(
            subset=["canonical_game_id", "sportsbook"]
        )
        if len(labelled)
        else labelled
    )
    return StrayReport(
        dirs=[d.name for d in dirs],
        files=int(
            labelled["source_file"].groupby(labelled["source_dir"]).nunique().sum()
        )
        if len(labelled)
        else 0,
        rows=len(labelled),
        categories={
            name: int(counts.get(name, 0))
            for name in (
                STRAY_DUPLICATE,
                STRAY_DUPLICATE_AFTER_CANONICAL_ID,
                STRAY_SAME_KEY_DIFFERENT_VALUE,
                STRAY_BOOK_NOT_CARRIED,
                STRAY_ID_NOT_IN_GAMES,
                STRAY_RECOVERABLE,
            )
        },
        book_rows_with_id_not_in_games=int(
            (~book_rows["canonical_game_id"].isin(games_ids)).sum()
        )
        if len(book_rows)
        else 0,
        recoverable=recoverable,
    )


def insert_if_absent(
    main: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Append candidate rows whose ``(game_id, sportsbook)`` the main table lacks. Never overwrites.

    Raises:
        AssertionError: if the row count did not grow by exactly the number inserted.
    """
    if candidates.empty:
        return main, 0
    present = set(zip(main["game_id"], main["sportsbook"], strict=True))
    new_rows = candidates[
        [
            (gid, book) not in present
            for gid, book in zip(
                candidates["game_id"], candidates["sportsbook"], strict=True
            )
        ]
    ]
    new_rows = new_rows[[c for c in main.columns if c in new_rows.columns]]
    combined = pd.concat([main, new_rows], ignore_index=True)
    if len(combined) - len(main) != len(new_rows):
        raise AssertionError(
            "insert_if_absent changed the row count by other than the inserted count"
        )
    return combined, len(new_rows)


# ---------------------------------------------------------------------------
# Planning, the record and the run.
# ---------------------------------------------------------------------------


def load_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    silver = data_root / "silver"
    odds = pd.read_parquet(silver / f"{ODDS_TABLE}.parquet")
    timeline = pd.read_parquet(silver / f"{TIMELINE_TABLE}.parquet")
    games = pd.read_parquet(
        silver / f"{GAMES_TABLE}.parquet", columns=["game_id", "kickoff_et"]
    )
    return odds, timeline, games


def plan_repair(data_root: Path) -> tuple[pd.DataFrame, RepairPlan]:
    """Measure the table as found and decide every change, writing nothing."""
    odds, timeline, games = load_tables(data_root)
    kickoffs = pd.to_datetime(games.set_index("game_id")["kickoff_et"], utc=True)
    timeline_latest = latest_pre_kickoff_timeline(
        timeline_in_snapshot_convention(timeline), kickoffs
    )

    plan = RepairPlan(
        sign_conflicts_found=int(sign_conflict_mask(odds).sum()),
        opposite_convention_would_find=int(
            sign_conflict_mask(odds, inverted=True).sum()
        ),
        spread_zero_rows=int((odds["spread"] == 0).sum()),
    )
    plan.corrections = resolve_sign_conflicts(
        odds, timeline_latest
    ) + resolve_disputed_totals(odds)

    after = apply_corrections(odds, plan.corrections)
    unreasoned = sum(1 for c in plan.corrections if not c.reason.strip())
    plan.unresolved_without_reason = int(sign_conflict_mask(after).sum()) + unreasoned
    plan.strays = build_stray_report(data_root / "silver", odds, set(games["game_id"]))

    plan.epoch_1970_found = int(epoch_1970_mask(odds).sum())
    plan.legacy_integer_found = int(legacy_integer_created_at_mask(odds).sum())
    plan.created_at_member_ids = odds.loc[
        created_at_family_members(odds), "game_id"
    ].tolist()
    return odds, plan


def null_created_at(odds: pd.DataFrame, member_ids: Sequence[str]) -> pd.DataFrame:
    """Set each family member's ``created_at`` to NULL. Never drops a row."""
    repaired = odds.copy()
    repaired.loc[repaired["game_id"].isin(set(member_ids)), "created_at"] = pd.NaT
    return repaired


def emit(plan: RepairPlan, *, recovered: int, stray_dirs_removed: int) -> list[str]:
    """The declared output contract, one ``KEY= value`` per line."""
    changed = [c for c in plan.corrections if c.disposition == DISPOSITION_CORRECTED]
    nulled = [c for c in plan.corrections if c.disposition == DISPOSITION_NULLED]
    confirmed = [c for c in plan.corrections if c.disposition == DISPOSITION_CONFIRMED]
    strays = plan.strays
    lines = [
        f"SIGN_CONFLICTS_FOUND= {plan.sign_conflicts_found}",
        f"OPPOSITE_CONVENTION_WOULD_FIND= {plan.opposite_convention_would_find}",
        f"SPREAD_ZERO_ROWS= {plan.spread_zero_rows}",
        f"RESOLVED= {len(changed)}",
        f"CONFIRMED_WITH_CITATION= {len(confirmed)}",
        f"NULLED_WITH_REASON= {len(nulled)}",
        f"UNRESOLVED_WITHOUT_REASON= {plan.unresolved_without_reason}",
        f"EPOCH_1970_FOUND= {plan.epoch_1970_found}",
        f"LEGACY_INTEGER_FOUND= {plan.legacy_integer_found}",
        "CREATED_AT_REPAIRED= 0",
        f"CREATED_AT_NULLED= {len(plan.created_at_member_ids)}",
    ]
    if strays is not None:
        lines += [
            f"STRAY_DIRS_FOUND= {len(strays.dirs)}",
            f"STRAY_FILES_FOUND= {strays.files}",
            f"STRAY_ROWS_FOUND= {strays.rows}",
            *(
                f"STRAY_{name.upper()}= {count}"
                for name, count in strays.categories.items()
            ),
            f"STRAY_BOOK_ROWS_WITH_ID_NOT_IN_GAMES= {strays.book_rows_with_id_not_in_games}",
        ]
    lines += [f"RECOVERED= {recovered}", f"STRAY_DIRS_REMOVED= {stray_dirs_removed}"]
    return lines


_INLINE_ARRAY_LIMIT = 8


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) <= _INLINE_ARRAY_LIMIT:
            return "[" + ", ".join(_toml_value(v) for v in value) + "]"
        return "[\n" + "".join(f"    {_toml_value(v)},\n" for v in value) + "]"
    return json.dumps(str(value))


def _toml_table(header: str, table: dict[str, Any]) -> list[str]:
    """A TOML table; ``None`` values are OMITTED, because TOML has no null."""
    lines = [header]
    lines += [
        f"{key} = {_toml_value(value)}"
        for key, value in table.items()
        if value is not None
    ]
    return [*lines, ""]


RECORD_HEADER = """\
# Silver odds_snapshot correction record -- Plan 33.2-08, SPEC R12, D33.2-08 item 4, D33.2-23.
#
# Written by scripts/repair_odds_snapshot.py --apply. TWO record shapes, chosen by defect size:
#   [[correction]] one entry per individually-resolved value. A corrected or confirmed value
#                  carries source_url; a nulled value carries a reason and no new_value (TOML
#                  has no null, so an absent new_value IS the unknown).
#   [[family]]     one entry per programmatically-identified defect family, with its cause, its
#                  detector in words, its disposition, its reason and its full membership.
# Every count below was EMITTED by the script's own detector on the table as found. It is not a
# plan-time figure: 33.2-CONTEXT.md D33.2-08 item 4's "10 of 2,140" spread-sign conflicts does not
# reproduce against the live table, and the CONTEXT is deliberately left unedited as the owner's
# locked decision text.
"""


def load_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _recorded_new_value(entry: dict[str, Any]) -> float | None:
    value = entry.get("new_value")
    return None if value is None else float(value)


def unrecorded_corrections(
    existing: dict[str, Any], corrections: Sequence[Correction]
) -> list[Correction]:
    """The corrections the record does not yet carry -- what makes a re-run idempotent.

    A re-run re-finds only what is still true of the table (a confirmed total stays confirmed),
    and re-recording it would duplicate the entry. A value already recorded is skipped when the
    re-run agrees with it and REFUSED when it does not: the record is never silently rewritten.

    Raises:
        ValueError: if a recorded ``(game_id, column)`` now resolves to a different value.
    """
    recorded = {
        (entry["game_id"], entry["column"]): entry
        for entry in existing.get("correction", [])
    }
    fresh: list[Correction] = []
    for correction in corrections:
        entry = recorded.get(correction.key())
        if entry is None:
            fresh.append(correction)
            continue
        if _recorded_new_value(entry) != correction.new_value:
            raise ValueError(
                f"{correction.key()} is recorded as {entry.get('new_value')!r} but now resolves "
                f"to {correction.new_value!r}; refusing to rewrite the record"
            )
    return fresh


def merge_record(
    existing: dict[str, Any],
    *,
    ruling: dict[str, Any] | None,
    measurements: dict[str, dict[str, Any]],
    corrections: Sequence[Correction],
    families: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Merge a run's findings into the record without ever erasing an earlier run's.

    A measurement section is replaced only by a run that measured it; a correction or family
    already recorded is kept, and a new one with the same key is refused rather than silently
    replacing it.
    """
    merged: dict[str, Any] = {
        "schema_version": 1,
        "ruling": existing.get("ruling", {}),
        "measurements": dict(existing.get("measurements", {})),
        "correction": list(existing.get("correction", [])),
        "family": list(existing.get("family", [])),
    }
    if ruling:
        merged["ruling"] = ruling
    merged["measurements"].update(measurements)
    known = {(c["game_id"], c["column"]) for c in merged["correction"]}
    for correction in corrections:
        if correction.key() in known:
            raise ValueError(
                f"correction {correction.key()} is already recorded; refusing to overwrite it"
            )
        merged["correction"].append(
            {
                "game_id": correction.game_id,
                "column": correction.column,
                "old_value": correction.old_value,
                "new_value": correction.new_value,
                "disposition": correction.disposition,
                "source_url": correction.source_url,
                "source_date": correction.source_date,
                "source_detail": correction.source_detail,
                "reason": correction.reason,
            }
        )
    names = {f["name"] for f in merged["family"]}
    for family in families:
        if family["name"] in names:
            raise ValueError(
                f"family {family['name']!r} is already recorded; refusing to overwrite it"
            )
        merged["family"].append(dict(family))
    return merged


def render_record(record: dict[str, Any]) -> str:
    lines = [RECORD_HEADER, f"schema_version = {record['schema_version']}", ""]
    if record.get("ruling"):
        lines += _toml_table("[ruling]", record["ruling"])
    for name, table in sorted(record.get("measurements", {}).items()):
        lines += _toml_table(f"[measurements.{name}]", table)
    for correction in record.get("correction", []):
        lines += _toml_table("[[correction]]", correction)
    for family in record.get("family", []):
        lines += _toml_table("[[family]]", family)
    return "\n".join(lines).rstrip() + "\n"


def write_odds_table(data_root: Path, repaired: pd.DataFrame) -> Path:
    """One atomic full-table write. Every stored ``game_id`` is in *repaired*, so the key
    replacement in ``upsert_silver`` replaces the whole table with exactly this frame."""
    return upsert_silver(
        repaired.reset_index(drop=True), ODDS_TABLE, base_path=data_root
    )


def _measurements_for(
    plan: RepairPlan, measured_at: str, recovered: int, removed: list[str]
) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}
    if plan.sign_conflicts_found > 0:
        measurements["sign_conflicts"] = {
            "measured_at": measured_at,
            "sign_conflicts_found": plan.sign_conflicts_found,
            "opposite_convention_would_find": plan.opposite_convention_would_find,
            "spread_zero_rows": plan.spread_zero_rows,
            "context_figure_not_reproduced": "33.2-CONTEXT.md D33.2-08 item 4 records 10; the detector measured the value above",
        }
    strays = plan.strays
    if strays is not None and strays.dirs:
        measurements["strays"] = {
            "measured_at": measured_at,
            "dirs_found": len(strays.dirs),
            "files_found": strays.files,
            "rows_found": strays.rows,
            **strays.categories,
            "book_rows_with_id_not_in_games": strays.book_rows_with_id_not_in_games,
            "recovered": recovered,
            "recovered_reason": (
                "nothing real and missing was found: every stray row is a duplicate of a main "
                "row (directly or after canonicalising LAR to LA), or a per-book 2025 row at a "
                "grain the one-consensus-row-per-game table does not carry"
                if recovered == 0
                else "legitimate missing rows inserted without overwriting"
            ),
            "dirs_removed": removed,
        }
    return measurements


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair silver odds_snapshot (Plan 33.2-08)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="measure and report; write nothing (default)",
    )
    mode.add_argument(
        "--apply", action="store_true", help="perform the declared writes"
    )
    parser.add_argument(
        "--ruling", choices=RULINGS, help="the owner's Task-1 checkpoint answer"
    )
    parser.add_argument(
        "--ruling-date", help="the date the owner gave the ruling (YYYY-MM-DD)"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD_PATH)
    args = parser.parse_args(argv)

    if args.apply and (args.ruling is None or args.ruling_date is None):
        parser.error(
            "--apply requires --ruling and --ruling-date: the owner's answer is an input"
        )

    odds, plan = plan_repair(args.data_root)
    if not args.apply:
        for line in emit(plan, recovered=0, stray_dirs_removed=0):
            print(line)
        print("MODE= dry-run (nothing written)")
        return 0

    drop_unresolvable = args.ruling == RULING_AMEND_DROP_ROW
    remove_strays = args.ruling != RULING_AMEND_KEEP_STRAYS
    repaired = apply_corrections(
        odds, plan.corrections, drop_unresolvable=drop_unresolvable
    )
    recoverable = plan.strays.recoverable if plan.strays is not None else pd.DataFrame()
    candidates = (
        recoverable.assign(game_id=recoverable["canonical_game_id"])
        if len(recoverable)
        else recoverable
    )
    before_insert = len(repaired)
    repaired, recovered = insert_if_absent(repaired, candidates)
    assert len(repaired) == before_insert + recovered
    repaired = null_created_at(repaired, plan.created_at_member_ids)

    existing_record = load_record(args.record)
    new_corrections = unrecorded_corrections(existing_record, plan.corrections)
    if new_corrections or recovered or plan.created_at_member_ids:
        write_odds_table(args.data_root, repaired)

    removed: list[str] = []
    if remove_strays and plan.strays is not None:
        for name in plan.strays.dirs:
            shutil.rmtree(args.data_root / "silver" / name)
            removed.append(name)

    measured_at = datetime.now(UTC).isoformat()
    ruling = {"answer": args.ruling, "date": args.ruling_date}
    measurements = _measurements_for(plan, measured_at, recovered, removed)
    families = []
    if plan.created_at_member_ids:
        families.append(build_created_at_family(plan.created_at_member_ids))
        measurements["created_at_family"] = {
            "measured_at": measured_at,
            "epoch_1970_found": plan.epoch_1970_found,
            "legacy_integer_found": plan.legacy_integer_found,
            "created_at_repaired": 0,
            "created_at_nulled": len(plan.created_at_member_ids),
            "table_rows_before": len(odds),
            "table_rows_after": len(repaired),
            "context_figure_not_reproduced": (
                '33.2-CONTEXT.md D33.2-08 item 4 says "some" created_at values read 1970; the '
                "detector measured the counts above"
            ),
        }
    if new_corrections or measurements:
        record = merge_record(
            existing_record,
            ruling=ruling,
            measurements=measurements,
            corrections=new_corrections,
            families=families,
        )
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(render_record(record), encoding="utf-8")

    for line in emit(plan, recovered=recovered, stray_dirs_removed=len(removed)):
        print(line)
    print(f"MODE= apply (ruling {args.ruling}, {args.ruling_date})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
