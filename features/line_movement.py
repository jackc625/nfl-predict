"""Line-Movement Feature Builder (SIG-04).

This module turns the additive ``odds_timeline`` trajectory silver table (Plan
29-02 storage, keyed ``(game_id, snapshot_ts)``) into leakage-safe, game-level
line-movement features, all fenced strictly to snapshots at/before EACH game's
OWN Friday 6 PM ET freeze:

- ``opening_total`` -- the totals line in the earliest captured pre-freeze
  snapshot (the genuinely-new information; the freeze line is already a model
  feature via ``market_anchors.snapshot_total``, D-02). "Earliest captured" is
  NOT a fixed horizon, and the spread of horizons is wide enough to change what
  the whole family means -- see the WR-09 note below.
- ``total_drift`` / ``total_drift_dir`` -- net drift ``freeze_total -
  opening_total`` and its sign (D-09 i).
- ``total_late_drift`` -- drift over the last ~24-48h before the freeze, the
  steam feature that rests on the FULL trajectory, not a two-anchor capture
  (D-08/D-09 iii).
- ``total_abs_travel`` / ``total_reversals`` / ``total_range`` -- path shape:
  summed absolute movement, count of sign flips, and max-min span (D-09 iv).
- ``line_movement_coverage`` -- 1.0 when >=2 pre-freeze snapshots exist
  (2020-06-06+ coverage), else 0.0 with neutral defaults (D-10).

The load-bearing temporal control is a PER-GAME Friday-6PM-ET freeze (D-15):
``odds_timeline`` spans many game-weeks across 2020-2024, so there is no single
Friday. Each game's freeze is derived from its own kickoff date and the builder
fences to ``snapshot_ts <= min(as_of_datetime, that_game_freeze)``. The cutoff
is localized to ``America/New_York`` (ET), NEVER UTC -- a UTC-localized Friday
18:00 cutoff would be 14:00 ET and wrongly drop the legitimate ET-evening
snapshots (the WR-02 lesson, borrowed from
``market_anchors.identify_snapshot_lines`` but NOT its global-max cutoff).

``as_of_datetime`` is canonicalized to tz-aware UTC FIRST (review 29-04 HIGH),
because comparing a naive datetime against the tz-aware UTC ``snapshot_ts``
would raise ``TypeError``.

CRITICAL (CR-01, second occurrence): a NAIVE ``as_of`` is interpreted as ET, not
as UTC. ``build_features.py`` now defaults to a tz-aware ``datetime.now(ET)``, so
on the canonical path nothing naive arrives here at all -- but the previous
default was a naive LOCAL ``datetime.now()`` that ``ensure_utc_aware``
RE-LABELLED as UTC without shifting the wall clock (its own docstring forbids
exactly this use). On the owner's ET machine the Friday 18:05 ET orchestrator
slot became 18:05Z == 14:05 ET, so the fence ``min(as_of, freeze, kickoff)``
collapsed to 14:05 ET and silently dropped the Friday-6PM-ET freeze snapshot --
the single most important point on every trajectory. East of UTC the same
mislabelling points the other way and admits post-cutoff rows, i.e. it LEAKS.
Training gold was built with an ``as_of`` decades after every freeze, so the
fence never bound there: it was textbook train/serve skew. ET is this module's
one wall clock (kickoffs, the Friday freeze, ``kickoff_wall_clock_et``), so ET is
what a naive ``as_of`` must mean here.

CRITICAL (D-15): the TRUE closing line is NEVER emitted -- only snapshots
strictly ``<= freeze`` are used; the closing total is reserved for CLV grading.

CRITICAL (CR-01): the fence is ``min(as_of, that game's Friday freeze, one second
before kickoff)``. The kickoff term is not redundant. The freeze is derived from the
kickoff DATE, so for a Friday-afternoon kickoff it lands AFTER kickoff and admits an
IN-PLAY line -- which is how three archive games carried post-kickoff values into
gold. For every weekday except Friday the freeze already falls on a prior calendar
day, so the cap does not bind.

DEGRADED-INPUT EDGE, stated rather than hidden: ``_resolve_game_date`` walks
``_KICKOFF_COLUMNS`` in order, so a games frame carrying only a date-only ``gameday``
column yields a MIDNIGHT-ET kickoff and hence a fence tighter than necessary. That is
conservative -- never leaky -- and does not arise on the canonical path, where
``kickoff_et`` is present.

CRITICAL (Pitfall 1, review 29-04 MED): historical odds start 2020-06-06, so the
2018-2019 train window has ZERO trajectory coverage. Uncovered games get
drift/path families = 0.0 and ``line_movement_coverage`` = 0.0, but
``opening_total`` is imputed from a NON-LEAKY in-row anchor (the single available
pre-freeze snapshot total) or a prior-only constant -- never a literal 0.0,
which is out-of-distribution for a ~40-50 totals line and would let the model
learn a coverage/season artifact instead of keying on the coverage flag.

CRITICAL (Pitfall 2): every column name is DISTINCT from the structurally-zero
``total_movement`` / ``spread_movement`` that ``market_anchors`` already emits.

WR-09 -- "OPENING" HAS NO HORIZON CONTROL, AND THE HORIZONS ARE NOT COMPARABLE.
``opening_*`` is simply ``values[0]`` of the admitted trajectory: whatever the
archive happened to capture first. Measured on the live archive (1,347 games
carrying a total): median 8.8 days before kickoff, p75 12.0, p90 68.6, max 124.2;
290 games (22 percent) open more than 14 days out and 166 more than 60 days out.
The horizons can differ WITHIN a single game -- ``2020_W16_MIN@NO`` draws its
opening spread from a 2020-09-01 line and its opening total from a 2020-12-22
line, 16 weeks apart.

So the drift, travel, reversal and range families mean materially different things
for different games: for one game they describe a week of movement, for another a
four-month repricing that spans roster and season-context changes. NO horizon
covariate is emitted, so a model cannot tell those cases apart.

This is stated rather than fixed on purpose. Bounding the trajectory would move
feature values for about 22 percent of covered games, and emitting a lead-days
covariate would widen the ``line_movement`` group by one column -- either would
make a corrected screen non-attributable to the CR-01 leak fix and would re-open a
frozen pre-registration. The choice is recorded as an owner decision for Phase 30
in the quick-task 260817-dyp ``deferred-items.md``.
"""

from datetime import datetime, timedelta

import pandas as pd

from data.storage import load_dataframe
from utils import get_logger
from utils.date_utils import ET, UTC, ensure_utc_aware, kickoff_wall_clock_et
from utils.exceptions import DataIngestionError

logger = get_logger(__name__)

# The late/steam window: drift over the last ~24-48h before the freeze (D-09 iii).
# A CHOSEN, not-tuned span -- 48h is the standard "late money" horizon and is the
# reason a full trajectory (not a two-anchor capture) is worth buying (D-08).
LATE_WINDOW_HOURS = 48

# Prior-only, non-leaky fallback for ``opening_total`` on games with ZERO
# pre-freeze snapshots (the pre-2020 train window, Pitfall 1). A league-average
# NFL game total -- IN-DISTRIBUTION for a ~40-50 line, unlike a literal 0.0 which
# is out-of-distribution and lets the model learn a coverage/season artifact
# (review 29-04 MED). A CHOSEN constant, never tuned.
LEAGUE_AVERAGE_TOTAL = 44.0

# Game-level totals feature columns (the primary family, D-07). Names are DISTINCT
# from the structurally-zero ``total_movement`` market_anchors emits (Pitfall 2).
_TOTAL_FEATURE_COLUMNS = [
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
]

# Conditional spread siblings (Tier (a) only) -- emitted ONLY when odds_timeline
# carries non-null spread data (D-07 keeps totals primary).
_SPREAD_FEATURE_COLUMNS = [
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
]

# The shared coverage flag (one per game; the totals family is primary, D-07).
_COVERAGE_COLUMN = "line_movement_coverage"

# Candidate columns a games row may carry the kickoff date in (ET wall clock).
_KICKOFF_COLUMNS = ("kickoff_et", "gameday", "game_date", "start_time")


def _as_of_to_utc(as_of_datetime: datetime | None) -> datetime:
    """Canonicalize an ``as_of`` fence to tz-aware UTC, reading NAIVE as ET.

    CR-01. ``ensure_utc_aware`` reinterprets a naive datetime as UTC without
    shifting the wall clock, which is correct only when the caller knows the
    source was already UTC. Nothing in this module's world is UTC: kickoffs, the
    Friday 18:00 freeze and ``kickoff_wall_clock_et`` are all ET wall clocks, and
    the historical naive default was a LOCAL ``datetime.now()``. So a naive value
    is localized to ET and then CONVERTED, exactly as the ``ensure_utc_aware``
    docstring instructs callers whose source is a different timezone to do.

    Args:
        as_of_datetime: The fence cutoff, tz-aware or naive, or ``None``.

    Returns:
        The same instant as a tz-aware UTC ``datetime``; ``datetime.now(UTC)``
        when ``as_of_datetime`` is ``None``.
    """
    if as_of_datetime is None:
        return datetime.now(UTC)
    if as_of_datetime.tzinfo is None:
        logger.warning(
            "naive as_of received; interpreting it as ET, not UTC (CR-01)",
            as_of=str(as_of_datetime),
        )
        return as_of_datetime.replace(tzinfo=ET).astimezone(UTC)
    return ensure_utc_aware(as_of_datetime)


class LineMovementBuilder:
    """Build leakage-safe game-level line-movement features from odds_timeline.

    Conforms to the FeatureBuilder Protocol (``build_features`` with the
    ``as_of_datetime`` Friday-freeze fence + ``get_features_for_game``). Emits one
    row per game with the four D-09 totals families + ``line_movement_coverage``
    (and ``spread_*`` siblings only when the trajectory carries spreads).

    The binding temporal control is a PER-GAME Friday-6PM-ET freeze (D-15): each
    game's freeze is derived from its own kickoff date and localized to ET (NOT
    UTC, WR-02); the fence is ``snapshot_ts <= min(as_of_datetime, game_freeze)``
    with ``as_of_datetime`` canonicalized to tz-aware UTC first (review 29-04
    HIGH). The true closing line is never emitted -- only snapshots ``<= freeze``.
    """

    def __init__(self, *, timeline_df: pd.DataFrame | None = None) -> None:
        """Initialize the line-movement builder.

        Args:
            timeline_df: Optional ``odds_timeline`` trajectory frame to use
                directly (the test-injection seam). When ``None`` the builder
                loads ``odds_timeline`` from the silver layer.
        """
        self._timeline_df = timeline_df
        self._games_cache: pd.DataFrame | None = None
        # WR-08: memoize the loaded+coerced trajectory. ``_load_timeline`` was
        # called once via ``_timeline_has_spread()`` and again for the build, and
        # ``get_features_for_game`` did the same PER GAME -- so a 16-game weekly
        # loop performed 32 full parquet reads plus 32 full ``pd.to_datetime``
        # coercions of the ~10k-row archive.
        self._timeline_cache: pd.DataFrame | None = None
        self._emit_spread_logged = False

    # ------------------------------------------------------------------
    # odds_timeline loading
    # ------------------------------------------------------------------

    def _load_timeline(self) -> pd.DataFrame:
        """Load the trajectory silver (or the injected frame), snapshot_ts UTC.

        ``snapshot_ts`` is coerced to a tz-aware UTC datetime so the on-disk
        round-trip (which may surface it as an ISO string) drives the per-game
        fence comparison correctly.

        WR-08: MEMOIZED for the lifetime of the builder. The load is
        deterministic for a given builder, and it used to run at least twice per
        ``build_features`` call and twice PER GAME in ``get_features_for_game``.
        A builder is constructed per build, so the cache cannot serve stale data
        across a rebuild.
        """
        if self._timeline_cache is not None:
            return self._timeline_cache

        if self._timeline_df is not None:
            timeline = self._timeline_df
        else:
            try:
                timeline = load_dataframe("odds_timeline", layer="silver")
            except (DataIngestionError, FileNotFoundError, OSError, ValueError) as exc:
                logger.warning("odds_timeline silver not available", error=str(exc))
                self._timeline_cache = pd.DataFrame(
                    columns=["game_id", "snapshot_ts", "total"]
                )
                return self._timeline_cache

        if timeline is None or len(timeline) == 0:
            self._timeline_cache = pd.DataFrame(
                columns=["game_id", "snapshot_ts", "total"]
            )
            return self._timeline_cache

        timeline = timeline.copy()
        timeline["snapshot_ts"] = pd.to_datetime(
            timeline["snapshot_ts"], utc=True, errors="coerce"
        )
        self._timeline_cache = timeline
        return timeline

    # ------------------------------------------------------------------
    # Per-game Friday 6 PM ET freeze (NOT a global cutoff, review 29-04 HIGH)
    # ------------------------------------------------------------------

    @staticmethod
    def _game_friday_freeze(game_date: datetime) -> datetime:
        """Return THAT game's own Friday-6PM-ET freeze from its kickoff date.

        The freeze is the most recent Friday at/before the game's kickoff date,
        at 18:00 ET. This is a PER-GAME freeze -- it does NOT derive one global
        Friday from ``odds_df['snapshot_ts'].max()`` the way
        ``identify_snapshot_lines`` does (correct for a single game-week, WRONG
        across a multi-season odds_timeline, review 29-04 HIGH).

        The Friday 18:00 cutoff is localized to ``America/New_York`` (ET), NEVER
        UTC (WR-02), then returned -- callers convert to UTC for the comparison so
        the wall-clock instant is preserved (18:00 ET == 22:00/23:00 UTC).

        THIS FUNCTION IS NOT THE WHOLE FENCE. For a Friday kickoff
        ``(4 - 4) % 7 == 0``, so the freeze is that SAME day at 18:00 ET -- after any
        Friday kickoff earlier than 6 PM. ``_compute_game_features`` caps the fence at
        one second before kickoff for exactly that reason (CR-01).

        The Friday derivation here is deliberately UNCHANGED. Shifting it to the
        PRIOR Friday would also close the leak, but it would discard the entire game
        week of legitimate movement for those games, and if both landed the
        prior-Friday shift would dominate and make the kickoff cap dead code for
        precisely the games it was written for.
        """
        # WR-06: the ET wall clock comes from the ONE documented accessor, so this
        # module and every other kickoff reader share a single contract instead of
        # each re-deriving one (the two Phase-29 readers assumed OPPOSITE contracts
        # and agreed only by luck).
        et_date = kickoff_wall_clock_et(game_date)

        # Most recent Friday (weekday 4) at/before the kickoff date.
        days_since_friday = (et_date.weekday() - 4) % 7
        friday = et_date.date() - timedelta(days=days_since_friday)
        return datetime(friday.year, friday.month, friday.day, 18, 0, 0, tzinfo=ET)

    @staticmethod
    def _resolve_game_date(row: pd.Series) -> datetime | None:
        """Resolve a game's kickoff date (ET wall clock) from a games row."""
        for col in _KICKOFF_COLUMNS:
            if col in row.index and pd.notna(row[col]):
                ts = pd.to_datetime(row[col], errors="coerce")
                if pd.notna(ts):
                    return ts.to_pydatetime()
        return None

    # ------------------------------------------------------------------
    # Family derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_family(
        pairs: list[tuple[pd.Timestamp, float]],
        fence_utc: datetime,
        prefix: str,
        uncovered_open: float,
    ) -> tuple[dict[str, float], bool]:
        """Derive one D-09 family (totals or spread) from sorted pre-freeze pairs.

        Args:
            pairs: ``(snapshot_ts, value)`` pairs, ascending, non-null values,
                already fenced to ``<= freeze``.
            fence_utc: The per-game UTC freeze instant (for the late window).
            prefix: ``"total"`` or ``"spread"`` -- drives the column names.
            uncovered_open: The non-leaky prior-only opening level for games with
                ZERO pre-freeze snapshots (never a literal 0.0, review 29-04 MED).

        Returns:
            ``(family_dict, covered)`` where ``covered`` is True iff >=2 pre-freeze
            snapshots exist. Drift/path stay 0.0 when uncovered; ``opening`` is the
            single in-row snapshot anchor (1 snapshot) or ``uncovered_open`` (0).
        """
        if len(pairs) >= 2:
            values = [float(v) for _, v in pairs]
            opening = values[0]
            freeze_value = values[-1]
            drift = freeze_value - opening

            late_start = fence_utc - timedelta(hours=LATE_WINDOW_HOURS)
            late_values = [float(v) for ts, v in pairs if ts >= late_start]
            late_drift = (
                late_values[-1] - late_values[0] if len(late_values) >= 2 else 0.0
            )

            diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
            abs_travel = sum(abs(d) for d in diffs)
            signs = [1 if d > 0 else -1 for d in diffs if d != 0]
            reversals = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
            value_range = max(values) - min(values)

            family = {
                f"opening_{prefix}": opening,
                f"{prefix}_drift": drift,
                f"{prefix}_drift_dir": float((drift > 0) - (drift < 0)),
                f"{prefix}_late_drift": late_drift,
                f"{prefix}_abs_travel": abs_travel,
                f"{prefix}_reversals": float(reversals),
                f"{prefix}_range": value_range,
            }
            return family, True

        # Uncovered: drift/path = 0.0; opening from the single in-row snapshot
        # anchor (non-leaky, <= freeze) or the prior-only constant -- never 0.0.
        opening = float(pairs[0][1]) if len(pairs) == 1 else uncovered_open
        family = {
            f"opening_{prefix}": opening,
            f"{prefix}_drift": 0.0,
            f"{prefix}_drift_dir": 0.0,
            f"{prefix}_late_drift": 0.0,
            f"{prefix}_abs_travel": 0.0,
            f"{prefix}_reversals": 0.0,
            f"{prefix}_range": 0.0,
        }
        return family, False

    @staticmethod
    def _pairs_for(
        game_rows: pd.DataFrame, fence_utc: datetime, value_col: str
    ) -> list[tuple[pd.Timestamp, float]]:
        """Return ascending ``(snapshot_ts, value)`` pairs fenced ``<= freeze``.

        Only snapshots strictly at/before the freeze are kept (D-15 -- the closing
        line is never used); rows with a null value or null snapshot_ts are
        dropped so a covered game's families are never NaN.
        """
        if len(game_rows) == 0 or value_col not in game_rows.columns:
            return []
        fenced = game_rows[
            game_rows["snapshot_ts"].notna()
            & (game_rows["snapshot_ts"] <= fence_utc)
            & game_rows[value_col].notna()
        ].sort_values("snapshot_ts")
        return [
            (ts, float(val))
            for ts, val in zip(fenced["snapshot_ts"], fenced[value_col], strict=True)
        ]

    def _compute_game_features(
        self,
        game_id: str,
        game_date: datetime | None,
        as_of_utc: datetime,
        timeline: pd.DataFrame,
        emit_spread: bool,
    ) -> dict[str, float]:
        """Compute the line-movement feature dict for a single game."""
        if game_date is None:
            return self._neutral_features(emit_spread)

        freeze_utc = self._game_friday_freeze(game_date).astimezone(UTC)

        # CR-01: cap the fence at KICKOFF. The Friday freeze is "the most recent
        # Friday 18:00 ET at/before the kickoff DATE", so for a Friday-afternoon
        # kickoff -- Black Friday, Christmas -- that freeze is AFTER kickoff and the
        # Friday-18:00 cadence snapshot it admits is an IN-PLAY line. Three games in
        # the archive carried one; 2023_W12_MIA@NYJ's spread moved 9.5 -> 20.5 DURING
        # the game. For every other weekday the freeze already lands on a prior
        # calendar day, so this term simply does not bind.
        #
        # The one-second offset is deliberate: _pairs_for filters with <=, and D-15
        # says no snapshot AT OR AFTER kickoff may enter a feature, so the strict form
        # is the one that matches the contract. No archive snapshot lands on an exact
        # kickoff instant, so it changes no real value.
        kickoff_fence_utc = kickoff_wall_clock_et(game_date).astimezone(
            UTC
        ) - timedelta(seconds=1)
        fence_utc = min(as_of_utc, freeze_utc, kickoff_fence_utc)

        game_rows = (
            timeline[timeline["game_id"] == game_id] if len(timeline) > 0 else timeline
        )

        total_pairs = self._pairs_for(game_rows, fence_utc, "total")
        total_family, covered = self._derive_family(
            total_pairs, fence_utc, "total", LEAGUE_AVERAGE_TOTAL
        )

        record: dict[str, float] = dict(total_family)
        record[_COVERAGE_COLUMN] = 1.0 if covered else 0.0

        if emit_spread:
            spread_pairs = self._pairs_for(game_rows, fence_utc, "spread")
            # Impute an uncovered opening spread from a pick'em prior (0.0 IS the
            # in-distribution neutral spread, unlike a 0.0 totals line).
            spread_family, _ = self._derive_family(
                spread_pairs, fence_utc, "spread", 0.0
            )
            record.update(spread_family)

        return record

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build game-level line-movement features for games.

        Conforms to the FeatureBuilder Protocol. ``as_of_datetime`` is
        canonicalized to tz-aware UTC FIRST (review 29-04 HIGH); each game is
        fenced to ``snapshot_ts <= min(as_of_utc, that_game's_Friday_6PM_ET
        freeze)`` (D-15). The closing line is never emitted.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff (capped per-game by the Friday
                freeze). A naive datetime is interpreted as ET and CONVERTED
                (CR-01); ``None`` defaults to ``datetime.now(UTC)``.
            target_season: Optional season to filter games for.
            target_week: Optional week to filter games for.

        Returns:
            One row per game with the line-movement feature columns.
        """
        emit_spread = self._timeline_has_spread()
        cols = ["game_id", *self._feature_columns(emit_spread)]

        if games_df is None or len(games_df) == 0:
            return pd.DataFrame(columns=cols)

        self._games_cache = games_df

        # Canonicalize as_of to tz-aware UTC BEFORE any snapshot comparison
        # (review 29-04 HIGH). A naive value is read as ET, never as UTC (CR-01).
        as_of_utc = _as_of_to_utc(as_of_datetime)

        target_games = games_df
        if target_season is not None and "season" in games_df.columns:
            target_games = target_games[target_games["season"] == target_season]
        if target_week is not None and "week" in target_games.columns:
            target_games = target_games[target_games["week"] == target_week]
        if len(target_games) == 0:
            return pd.DataFrame(columns=cols)

        timeline = self._load_timeline()

        rows: list[dict] = []
        for _, game in target_games.iterrows():
            game_date = self._resolve_game_date(game)
            feats = self._compute_game_features(
                str(game["game_id"]), game_date, as_of_utc, timeline, emit_spread
            )
            rows.append({"game_id": game["game_id"], **feats})

        return pd.DataFrame(rows, columns=cols)

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get line-movement features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff (capped by the game's Friday
                freeze). A naive value is interpreted as ET (CR-01).

        Returns:
            Dict of line-movement feature values (neutral defaults when the game
            is unknown or has no covered trajectory).
        """
        emit_spread = self._timeline_has_spread()
        as_of_utc = _as_of_to_utc(as_of_datetime)

        game_date = self._resolve_game_date_for(game_id)
        if game_date is None:
            return self._neutral_features(emit_spread)

        timeline = self._load_timeline()
        return self._compute_game_features(
            game_id, game_date, as_of_utc, timeline, emit_spread
        )

    # ------------------------------------------------------------------
    # Column / default helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_columns(emit_spread: bool) -> list[str]:
        """The emitted feature column order (totals primary; spread conditional)."""
        cols = [*_TOTAL_FEATURE_COLUMNS, _COVERAGE_COLUMN]
        if emit_spread:
            cols = cols + list(_SPREAD_FEATURE_COLUMNS)
        return cols

    def _timeline_has_spread(self) -> bool:
        """True iff odds_timeline carries non-null spread data (Tier (a), D-07).

        WR-08, stated rather than hidden: this is a GLOBAL predicate over the
        whole multi-season table, so the emitted column SET is data-dependent --
        the seven ``spread_*`` columns appear or vanish according to whether ANY
        row anywhere carries a spread. A totals-only re-pull or a partial archive
        restore therefore FLIPS the gold schema (202 vs 209/210/209 columns,
        which ``scripts/data_qa.py`` hardcodes) rather than raising. The schema
        actually emitted is logged at INFO below so a width change is traceable
        to a cause instead of being discovered by the tripwire. Making the column
        set an unconditional contract would widen gold, so it belongs with the
        deliberate Phase-30 input rebuild, not here.
        """
        timeline = self._load_timeline()
        spread_non_null = (
            int(timeline["spread"].notna().sum())
            if len(timeline) > 0 and "spread" in timeline.columns
            else 0
        )
        emit_spread = spread_non_null > 0

        if not self._emit_spread_logged:
            self._emit_spread_logged = True
            logger.info(
                "line_movement emitted schema resolved",
                emit_spread=emit_spread,
                timeline_rows=len(timeline),
                spread_non_null=spread_non_null,
                emitted_columns=len(self._feature_columns(emit_spread)),
            )

        return emit_spread

    def _neutral_features(self, emit_spread: bool) -> dict[str, float]:
        """Neutral defaults (D-10): non-null, opening imputed from a prior."""
        record: dict[str, float] = {
            "opening_total": LEAGUE_AVERAGE_TOTAL,
            "total_drift": 0.0,
            "total_drift_dir": 0.0,
            "total_late_drift": 0.0,
            "total_abs_travel": 0.0,
            "total_reversals": 0.0,
            "total_range": 0.0,
            _COVERAGE_COLUMN: 0.0,
        }
        if emit_spread:
            record.update(
                {
                    "opening_spread": 0.0,
                    "spread_drift": 0.0,
                    "spread_drift_dir": 0.0,
                    "spread_late_drift": 0.0,
                    "spread_abs_travel": 0.0,
                    "spread_reversals": 0.0,
                    "spread_range": 0.0,
                }
            )
        return record

    def _resolve_game_date_for(self, game_id: str) -> datetime | None:
        """Resolve a game's kickoff date, loading ``games`` silver if needed.

        WR-07: this used to read ``self._games_cache`` and nothing else, and that
        cache is only ever populated by ``build_features``. So a caller using the
        FeatureBuilder Protocol's documented per-game entry point -- which is
        exactly how a serving path wires a builder in -- got all-neutral features
        for EVERY game (``opening_total`` 44.0, coverage 0.0, every drift and
        path 0.0), with no exception, no log line, and no way to tell "this game
        has no trajectory" apart from "you called the wrong method first". The
        same silent answer came back for a genuinely unknown ``game_id``.

        It now loads games itself and DISTINGUISHES the two cases in the log. The
        neutral return is preserved (the degradation contract is deliberate), but
        it is no longer indistinguishable from a correct answer.
        """
        if self._games_cache is None:
            try:
                self._games_cache = load_dataframe("games", layer="silver")
            except (DataIngestionError, FileNotFoundError, OSError, ValueError) as exc:
                logger.warning(
                    "games silver unavailable for per-game line-movement lookup; "
                    "emitting neutral features",
                    game_id=game_id,
                    error=str(exc),
                )
                return None

        match = self._games_cache[self._games_cache["game_id"] == game_id]
        if len(match) == 0:
            logger.warning(
                "game_id not present in games silver; emitting neutral "
                "line-movement features",
                game_id=game_id,
            )
            return None
        return self._resolve_game_date(match.iloc[0])
