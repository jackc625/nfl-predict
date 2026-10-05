"""Completeness of the bet-list record, checkable in ONE query.

Phase 31, plan 31-13 (SPEC R6, D31-19/21, PROD-03). This is the strongest available form of
"never silently dropped", and the reason it can be this strong is a schema decision rather than a
test technique.

WHY ONE QUERY IS POSSIBLE AT ALL (D31-21).

  Suppressed and live rows live in the SAME table, distinguished by a ``status`` column plus a
  ``rejection_reason`` drawn from the selector's own ``REJECTION_REASONS``. One write path, one
  read path, and shared ordering and tie-break by construction. The payoff is that completeness
  is a single GROUP BY over the same table the page reads: for each season and week the row count
  equals the scheduled games times the number of registered targets, every non-live row carries a
  taxonomy reason, and no row carries a status outside the live/suppressed pair.

  The rejected alternatives both lose that. A sibling suppressed table means two schemas and two
  writes, and a row could land in NEITHER with nothing to catch it. Persisting suppressed rows
  only for forward weeks means the replay history cannot answer what was declined and why.

VOLUME IS NOT A CONSTRAINT AND SHOULD NOT BE TRADED AGAINST HONESTY HERE. The universe is roughly
three rows per scheduled game per week -- about 48 rows for a 16-game week, and on the order of a
few thousand for the full 2021-2025 replay. That is a trivial table by any measure, so there is no
efficiency argument for writing only the bets and reconstructing the declines later.

THE INVARIANT IS PROVEN LOAD-BEARING, not merely green: a written row is deliberately dropped and
the same single query is asserted to FAIL. An invariant that has never been seen to fail is a
sentence, not a check.

Run this module:  .venv/Scripts/python.exe -m pytest tests/integration/test_bet_list_completeness.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    GRADING_STATUS_PENDING,
    RUN_MODE_REPLAY,
    materialize_bet_list,
    stamp_bet_list_provenance,
)
from backtest.bet_selector import REJECTION_REASONS, BetSelector
from backtest.selector_strategies import TargetStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"

_SEASON = 2023
_WEEK = 1
_N_GAMES = 16
_FROZEN_SD = 13.0
_SEASON_BIAS = {_SEASON: -1.0}
_BANKROLL = 10_000.0

# A Sunday kickoff and its own preceding Friday 6 PM Eastern freeze. SPEC R6 defines at-freeze as
# fresh, so every fixture row here is fresh and no row is suppressed for staleness by accident.
_SUNDAY_GAMEDAY = "2023-09-10"
_FREEZE_AT_SUNDAY = "2023-09-08T18:00:00-04:00"

# The two statuses a bet-list row may carry. Anything else is a row nobody can render (D31-21).
_STATUS_LIVE = "live"
_STATUS_SUPPRESSED = "suppressed"
_PERMITTED_STATUSES = (_STATUS_LIVE, _STATUS_SUPPRESSED)

# THE ONE QUERY. Every clause of the completeness invariant is a column of this result: the row
# count per season and week, the count of rows with an unrenderable status, the count of non-live
# rows carrying no reason, and the distinct reasons actually used.
_COMPLETENESS_QUERY = """
SELECT
    season,
    week,
    COUNT(*) AS rows_written,
    COUNT(*) FILTER (WHERE status NOT IN ('live', 'suppressed')) AS rows_with_bad_status,
    COUNT(*) FILTER (WHERE status <> 'live' AND rejection_reason IS NULL)
        AS suppressed_without_reason,
    list_distinct(list(rejection_reason) FILTER (WHERE status <> 'live')) AS reasons_used
FROM bet_list
GROUP BY season, week
ORDER BY season, week
"""


# ---------------------------------------------------------------------------
# A test-local, fully implemented target strategy
# ---------------------------------------------------------------------------


class _MiniStrategy:
    """A minimal, fully implemented ``TargetStrategy`` for a test-local target.

    It makes no claim about how any real target prices a bet. Its only job is to put more than one
    target into the registry so the universe identity is proven at a realistic cardinality. Plan
    31-06 forbids a registerable PRODUCTION strategy whose methods are unimplemented; this one is
    test-local, complete, and checked against the Protocol by ``isinstance`` before use.
    """

    def __init__(self, target: str, market_fields: tuple[str, ...]) -> None:
        self.target = target
        self.required_market_fields = market_fields

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        return row.get(f"{self.target}_side", "home")

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        return None if bet_side is not None else "not_subpop"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        return "all"

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float]:
        return float(row.get(f"{self.target}_p", 0.60)), float(
            row[self.required_market_fields[-1]]
        )

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        return {}

    def grade(self, record: dict[str, Any]) -> bool | None:
        return None


_TARGET_MARKET_VALUES: dict[str, dict[str, Any]] = {
    "winner": {"model_win_prob": 0.61, "ml_home": -130.0, "ml_away": 110.0},
    "spread": {"model_spread": -3.5, "closing_spread": -2.5},
    "totals": {"model_total": 41.0, "closing_total": 45.0},
}


def _registry() -> list[Any]:
    strategies = [
        _MiniStrategy("winner", ("model_win_prob", "ml_home", "ml_away")),
        _MiniStrategy("spread", ("model_spread", "closing_spread")),
        _MiniStrategy("totals", ("model_total", "closing_total")),
    ]
    for strategy in strategies:
        assert isinstance(strategy, TargetStrategy), strategy.target
    return strategies


def _selector(strategies: list[Any]) -> BetSelector:
    return BetSelector(
        frozen_sd=_FROZEN_SD,
        season_bias_by_season=_SEASON_BIAS,
        ev_floor_t=0.0,
        bankroll=_BANKROLL,
        strategies=strategies,
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _schedule(n_games: int = _N_GAMES) -> list[dict[str, Any]]:
    return [
        {
            "game_id": f"{_SEASON}_W{_WEEK:02d}_T{index:02d}@H{index:02d}",
            "season": _SEASON,
            "week": _WEEK,
            "gameday": _SUNDAY_GAMEDAY,
        }
        for index in range(n_games)
    ]


def _candidates(
    schedule: list[dict[str, Any]], targets: list[str], *, priced_games: int
) -> list[dict[str, Any]]:
    """Market data for the FIRST *priced_games* games only.

    The remaining games carry no candidate at all, so the selector builds a skeleton for them and
    suppresses the pair. That mix is the realistic one, and it is what makes the count identity a
    claim about the SCHEDULE rather than about the odds join.
    """
    rows: list[dict[str, Any]] = []
    for game in schedule[:priced_games]:
        for target in targets:
            rows.append(
                {
                    "game_id": game["game_id"],
                    "season": _SEASON,
                    "week": _WEEK,
                    "target": target,
                    "snapshot_ts": _FREEZE_AT_SUNDAY,
                    "sportsbook": "consensus",
                    "is_live": False,
                    **_TARGET_MARKET_VALUES[target],
                }
            )
    return rows


def _to_bet_list_row(record: dict[str, Any], *, live: bool) -> dict[str, Any]:
    """Map ONE selector record onto the locked 29-column ``bet_list`` schema.

    Nothing is re-derived. The recommendation facts are carried through as the selector produced
    them and the grading half stays ``pending``: this plan writes the record, it does not grade it.

    ``decided_at_utc`` is left at its ``dict.fromkeys`` NULL (Phase 33, Plan 33-05 Task 3): every
    row this fixture writes is stamped ``backtest_replay`` below, and a replay row is derived and
    fully regenerable, so it carries no observation time and needs none.
    """
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": record["game_id"],
            "season": int(record["season"]),
            "week": int(record["week"]),
            "target": record.get("target"),
            "bet_side": record.get("bet_side"),
            "model_value": record.get("model_value"),
            "market_value": record.get("market_value"),
            "line": record.get("line"),
            "slipped_line": record.get("slipped_line"),
            "calibrated_p_side": record.get("calibrated_p_side"),
            "per_bet_ev": record.get("per_bet_ev"),
            "stake_units": record.get("kelly_stake"),
            "ev_tier": None,
            "status": _STATUS_LIVE if live else _STATUS_SUPPRESSED,
            "rejection_reason": None if live else record["rejection_reason"],
            "eligibility_label": record.get("subpop_label"),
            "snapshot_ts": _FREEZE_AT_SUNDAY,
            "freeze_ts": _FREEZE_AT_SUNDAY,
            "selected_odds": record.get("odds"),
            "flat_stake": 1.0 if live else None,
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
            "clv": record.get("clv"),
            "payout_flat": None,
            "realized_units": None,
            "graded_at": None,
        }
    )
    return row


def _materialize_week(
    schedule: list[dict[str, Any]],
    strategies: list[Any],
    *,
    priced_games: int = 6,
    drop_rows: int = 0,
) -> tuple[duckdb.DuckDBPyConnection, int]:
    """Select over *schedule*, write EVERY resulting record, and return the connection.

    *drop_rows* deliberately discards that many written rows. It is the negative control: the
    invariant must FAIL when the record is incomplete, or it is not checking anything.
    """
    targets = [strategy.target for strategy in strategies]
    result = _selector(strategies).select(
        _candidates(schedule, targets, priced_games=priced_games),
        scheduled_games=schedule,
    )

    rows = [_to_bet_list_row(record, live=True) for record in result.selected]
    rows += [_to_bet_list_row(record, live=False) for record in result.rejected]
    assert rows, "the selector produced no records at all -- the fixture is broken"

    if drop_rows:
        rows = rows[:-drop_rows]

    frame = stamp_bet_list_provenance(pd.DataFrame(rows), RUN_MODE_REPLAY)
    conn = duckdb.connect(":memory:")
    written = materialize_bet_list(conn, frame)
    return conn, written


def _completeness_rows(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    result = conn.execute(_COMPLETENESS_QUERY)
    columns = [desc[0] for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


class TestTheCompletenessInvariantHoldsInOneQuery:
    """T-31-65: a candidate silently absent from the record is a test failure."""

    def test_the_count_identity_holds_per_season_and_week(self) -> None:
        """Rows written == scheduled games x registered targets.

        BOTH factors are read at runtime from the objects under test, so a plan that grows the
        strategy registry cannot break this assertion through wave order alone (REVIEW-REGISTRY).
        """
        schedule = _schedule()
        strategies = _registry()
        conn, _written = _materialize_week(schedule, strategies)
        rows = _completeness_rows(conn)

        assert len(rows) == 1, rows
        row = rows[0]
        assert (row["season"], row["week"]) == (_SEASON, _WEEK)
        assert row["rows_written"] == len(schedule) * len(strategies)

    def test_every_non_live_row_carries_a_reason_from_the_taxonomy(self) -> None:
        """Membership is asserted against the LIVE exported tuple, never a copied list.

        The taxonomy has grown twice already (four members, then eight, then nine). Copying it
        here would make this test assert the taxonomy of the day it was written.
        """
        conn, _written = _materialize_week(_schedule(), _registry())
        row = _completeness_rows(conn)[0]

        assert row["suppressed_without_reason"] == 0
        reasons = set(row["reasons_used"])
        assert reasons, "no row was suppressed, so the reason clause proves nothing"
        assert reasons <= set(REJECTION_REASONS), sorted(
            reasons - set(REJECTION_REASONS)
        )

    def test_no_row_carries_a_status_outside_the_live_suppressed_pair(self) -> None:
        conn, _written = _materialize_week(_schedule(), _registry())
        assert _completeness_rows(conn)[0]["rows_with_bad_status"] == 0

    def test_live_and_suppressed_rows_partition_the_universe(self) -> None:
        """Every written row is one or the other, and the two halves sum to the universe."""
        schedule = _schedule()
        strategies = _registry()
        conn, _written = _materialize_week(schedule, strategies)
        counts = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM bet_list GROUP BY status"
            ).fetchall()
        )
        assert set(counts) <= set(_PERMITTED_STATUSES)
        assert sum(counts.values()) == len(schedule) * len(strategies)

    def test_a_game_with_no_market_data_is_present_and_suppressed_not_absent(
        self,
    ) -> None:
        """The whole point of the universe: an unpriced game appears, saying why it was not bet."""
        schedule = _schedule()
        conn, _written = _materialize_week(schedule, _registry(), priced_games=2)
        unpriced_game = schedule[-1]["game_id"]
        rows = conn.execute(
            "SELECT status, rejection_reason FROM bet_list WHERE game_id = ?",
            [unpriced_game],
        ).fetchall()
        assert len(rows) == len(_registry())
        assert all(status == _STATUS_SUPPRESSED for status, _reason in rows)
        assert all(reason in REJECTION_REASONS for _status, reason in rows)


class TestTheInvariantIsLoadBearing:
    """A guard nobody has seen fail is a sentence, not a check."""

    def test_dropping_one_written_row_breaks_the_count_identity(self) -> None:
        """The negative control, kept as a permanent test rather than a one-off demonstration."""
        schedule = _schedule()
        strategies = _registry()
        expected = len(schedule) * len(strategies)

        conn, _written = _materialize_week(schedule, strategies, drop_rows=1)
        observed = _completeness_rows(conn)[0]["rows_written"]

        assert observed == expected - 1
        assert observed != expected, (
            "one row was dropped and the count identity still held; the invariant is not "
            "measuring what it claims to measure"
        )

    def test_the_universe_is_bounded_and_the_volume_argument_is_arithmetic(
        self,
    ) -> None:
        """Roughly three rows per scheduled game per week -- a few thousand for a full replay.

        Asserted rather than asserted-in-prose so the "volume is not a constraint" claim in
        D31-21 stays true as the registry grows.
        """
        schedule = _schedule()
        strategies = _registry()
        _conn, written = _materialize_week(schedule, strategies)
        assert written == len(schedule) * len(strategies)
        assert written < 100, written
        # Five seasons x ~285 games x the registry, the order D31-21 quotes.
        assert 5 * 285 * len(strategies) < 10_000


# ---------------------------------------------------------------------------
# The same invariant against the REAL schedule
# ---------------------------------------------------------------------------


class TestTheInvariantOnTheRealSchedule:
    """The hand-built week proves the rule; the real week proves the rule meets real data."""

    def test_a_real_scheduled_week_produces_one_row_per_game_and_target(
        self, data_boundary_guard: Any
    ) -> None:
        """Reads the silver schedule and writes only to an in-memory DB.

        ``data_boundary_guard`` digests ``data/`` before and after, so a write this test did not
        intend is a failure it can report -- rather than the vacuous ``git status`` check that
        cannot fail on a gitignored tree.
        """
        if not SILVER_GAMES.exists():
            pytest.skip(
                f"{SILVER_GAMES} is not readable on this checkout; the silver schedule is "
                "gitignored, so the real-week form of the completeness invariant did not run."
            )
        games = pd.read_parquet(SILVER_GAMES)
        week = games[(games["season"] == _SEASON) & (games["week"] == _WEEK)]
        if week.empty:
            pytest.skip(
                f"season {_SEASON} week {_WEEK} is not readable on this checkout's silver "
                "schedule; the real-week completeness invariant did not run."
            )

        schedule = [
            {
                "game_id": str(row.game_id),
                "season": int(row.season),
                "week": int(row.week),
                # The selector needs each game's OWN kickoff date for the per-game freeze fence
                # (D31-18); silver stores it as a kickoff instant rather than a bare date.
                "gameday": pd.Timestamp(row.kickoff_et).strftime("%Y-%m-%d"),
            }
            for row in week.itertuples()
        ]
        strategies = _registry()
        conn, _written = _materialize_week(schedule, strategies, priced_games=4)
        row = _completeness_rows(conn)[0]

        assert row["rows_written"] == len(schedule) * len(strategies)
        assert row["rows_with_bad_status"] == 0
        assert row["suppressed_without_reason"] == 0
        assert set(row["reasons_used"]) <= set(REJECTION_REASONS)


# ---------------------------------------------------------------------------
# The R6 refusal is a TRIPWIRE, not a weekly guillotine
# ---------------------------------------------------------------------------
#
# Phase 33, Plan 33-05 Task 1 (COLD-03, R6, D33-28, T-33-25); re-expressed on the day-before
# lock by Plan 33.2-02 (D33.2-01). A binding refusal that fires in normal operation is not a
# guard, it is an outage. So the whole real 2026 regular season is driven forward through the
# lock-instant selection and the refusal is asserted to fire not once.
#
# THE DECISION INSTANT IS EACH LOCK ITSELF, AND THAT IS THE POINT RATHER THAN A DODGE. At-lock
# information is admissible (``decided_at <= lock``), so the LATEST legitimate decision for a
# game is exactly at its lock -- the hardest ordinary case the daily run can produce. Driving
# every instant at that edge and seeing no refusal is a stronger claim than driving it a second
# early; the one-second-AFTER refusal is covered on a constructed input in
# ``tests/unit/test_freeze_fence_binding.py``.


class TestTheLockRefusalDoesNotFireInNormalWeeklyOperation:
    """T-33-25: every game a season -- Thursday games included -- survives the fence."""

    def _schedule_2026(self) -> pd.DataFrame:
        from tests.fixtures.season_2026 import (
            CapturedScheduleUnavailableError,
            load_captured_schedule,
        )

        try:
            feed = load_captured_schedule()
        except CapturedScheduleUnavailableError as exc:
            pytest.skip(str(exc))
        return feed[["game_id", "season", "week", "gameday", "weekday"]].copy()

    def test_driven_across_the_whole_2026_season_the_refusal_never_fires(self) -> None:
        from backtest.weekly_bet_list import (
            LockPassedError,
            select_games_for_decision_instant,
        )
        from scripts.ingest_historical_odds import gameday_lock

        schedule = self._schedule_2026()
        instants = sorted(
            {
                gameday_lock(str(gameday))
                for gameday in schedule["gameday"].dropna().unique()
            }
        )
        assert len(instants) >= 18, (
            f"only {len(instants)} lock instants resolved from the 2026 capture; a short "
            "list would make the tripwire assertion below cheap"
        )

        refusals: list[str] = []
        covered: set[str] = set()
        for instant in instants:
            try:
                selected = select_games_for_decision_instant(
                    schedule, instant, decided_at=instant
                )
            except LockPassedError as exc:
                refusals.append(f"{instant.isoformat()}: {exc}")
                continue
            covered.update(str(game_id) for game_id in selected["game_id"])

        assert refusals == [], (
            "the R6 refusal fired during ordinary forward operation, which means it is a "
            "guillotine rather than a tripwire:\n" + "\n".join(refusals)
        )
        # And the drive was not vacuous: every scheduled game belongs to exactly one instant,
        # so the union of the selections IS the season.
        assert covered == set(schedule["game_id"].astype(str))

    def test_every_thursday_game_is_carried_by_some_instant(self) -> None:
        """The games a WEEK-scoped fence would have lost, counted rather than described."""
        from backtest.weekly_bet_list import select_games_for_decision_instant
        from scripts.ingest_historical_odds import gameday_lock

        schedule = self._schedule_2026()
        thursdays = schedule[schedule["weekday"] == "Thursday"]
        assert len(thursdays) >= 15, len(thursdays)

        carried: set[str] = set()
        for gameday in sorted(schedule["gameday"].dropna().unique()):
            instant = gameday_lock(str(gameday))
            selected = select_games_for_decision_instant(
                schedule, instant, decided_at=instant
            )
            carried.update(str(game_id) for game_id in selected["game_id"])

        missing = set(thursdays["game_id"].astype(str)) - carried
        assert missing == set(), sorted(missing)


# ---------------------------------------------------------------------------
# THE 234 STORED REPLAY ROWS TAKE NULL, AND NULL IS WHAT THEY KEEP
# ---------------------------------------------------------------------------
#
# Phase 33, Plan 33-05 Task 3 (COLD-03, R7, D33-27, T-33-22). The OWNER RULING of 2026-09-12
# added `decided_at_utc` and ruled that the rows already on disk are NEVER backfilled. Every
# one of them is a `backtest_replay` row already past its freeze, so filling it from
# `snapshot_ts` -- or from the file's mtime, or from anything -- would stamp an observation time
# at which nobody observed anything. That is the prohibition this plan named, and it is the same
# failure class COLD-03 exists to prevent, pointed at history instead of at the future.
#
# THIS BLOCK IS READ-ONLY AND CARRIES NO `writes_production_store` MARKER, deliberately. It reads
# the production artifact and asserts a property of it; it writes nothing anywhere. The
# "a re-run does not fill them" claim is proven by reading TWICE through the shim and asserting
# the file's bytes are unchanged -- which is the actual claim (the shim must not write back), and
# is stronger than re-running a generator whose own write would need the marker.


class TestTheStoredReplayRowsAreNeverBackfilled:
    """T-33-22: the schema bump must not become a licence to stamp history."""

    _ARTIFACT = REPO_ROOT / "outputs" / "bet_list" / "bet_list.parquet"

    def _artifact(self) -> Path:
        if not self._ARTIFACT.exists():
            pytest.skip(
                f"{self._ARTIFACT.as_posix()} is absent (outputs/ is gitignored); "
                "regenerate with `uv run python scripts/generate_bet_list.py`"
            )
        return self._ARTIFACT

    def test_the_stored_artifact_is_all_replay_and_carries_no_observation_time(
        self,
    ) -> None:
        """Measured, not assumed: 234 ``backtest_replay`` rows, every stamp NULL.

        Was (until Plan 33-18 Task 8): the WHOLE file was 234 rows, all replay. Since
        2026-09-26 the scheduled daily run appends guarded ``forward`` rows (D33-38),
        each carrying its own pre-lock ``decided_at_utc`` and asserted by
        ``tests/integration/test_live_2026_prediction_set.py``. The claim here is about
        the stored REPLAY rows, so it is pinned on them.
        """
        from backtest.weekly_bet_list import (
            DECIDED_AT_COLUMN,
            read_bet_list_with_schema_shim,
        )
        from tests.phase33_state import BET_LIST_REPLAY_ROW_COUNT

        frame = read_bet_list_with_schema_shim(self._artifact())
        replay = frame.loc[frame["provenance"] == "backtest_replay"]

        assert len(replay) == BET_LIST_REPLAY_ROW_COUNT
        assert list(frame.columns) == list(BET_LIST_COLUMNS)
        assert replay[DECIDED_AT_COLUMN].isna().all(), (
            "a stored replay row carries an observation time; the 234 rows predate this column "
            "and were ruled un-backfillable on 2026-09-12"
        )

    def test_reading_through_the_shim_does_not_write_the_column_back(self) -> None:
        """The shim is a READ. Two reads, and the file on disk is byte-identical afterwards.

        This is the operative form of "a re-run does not fill them": the risk is not that some
        generator would deliberately backfill, it is that the convenience shim would persist its
        own NULL fill and quietly re-width the artifact.
        """
        from backtest.weekly_bet_list import (
            DECIDED_AT_COLUMN,
            read_bet_list_with_schema_shim,
        )
        from tests.phase33_state import (
            BET_LIST_COLUMN_COUNT_AFTER,
            BET_LIST_COLUMN_COUNT_BEFORE,
        )

        path = self._artifact()
        before = path.read_bytes()

        first = read_bet_list_with_schema_shim(path)
        second = read_bet_list_with_schema_shim(path)

        assert path.read_bytes() == before, (
            "the schema shim wrote to the stored artifact"
        )
        for frame in (first, second):
            replay = frame.loc[frame["provenance"] == "backtest_replay"]
            assert replay[DECIDED_AT_COLUMN].isna().all()

        # Was (until Plan 33-18 Task 8): the file on disk was still the pre-bump width
        # (BET_LIST_COLUMN_COUNT_BEFORE) with no DECIDED_AT_COLUMN. Since 2026-09-26 the
        # scheduled daily run writes the bumped schema when it appends forward rows, so
        # the column is on disk -- and on every stored replay row it is still NULL.
        # Was: ``in (BET_LIST_COLUMN_COUNT_BEFORE, len(BET_LIST_COLUMNS))``. Phase 34 (Plan 34-01
        # Task 2) widened the locked schema to 51 while the stored file is still the 29-column
        # Phase-33 width until the daily run next writes it, so all three widths are honest.
        raw = pd.read_parquet(path)
        assert raw.shape[1] in (
            BET_LIST_COLUMN_COUNT_BEFORE,
            BET_LIST_COLUMN_COUNT_AFTER,
            len(BET_LIST_COLUMNS),
        )
        if DECIDED_AT_COLUMN in raw.columns:
            stored_replay = raw.loc[raw["provenance"] == "backtest_replay"]
            assert stored_replay[DECIDED_AT_COLUMN].isna().all()
