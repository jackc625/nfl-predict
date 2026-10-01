"""``data_qa`` must be able to PASS a correct live Friday run -- and still refuse a broken one.

WHY THIS MODULE EXISTS
----------------------
Plan 33-18's live acceptance run, attempt 1 on 2026-09-15, halted at ``data_qa`` with five
counted failures. Measured read-only afterwards, two were TRUE refusals and three were the
gate being wrong:

* TRUE: ``games`` completeness 0 of 16 for 2026 week 2, and the DuckDB-versus-parquet guard
  (272 rows only in parquet). Both were the N-01 split, which is fixed at the writer
  (``data.storage._keep_duckdb_copy_in_step``). They are what stopped the run safely, so
  ``TestTheTrueRefusalsSurviveTheFix`` proves both still refuse the split and pass once healed.
* WRONG (ii): ``odds_snapshot`` freshness. ``ingest_odds`` is registered AFTER ``data_qa``,
  so at this boundary the odds are always last week's. A gate that demands data a LATER step
  produces refuses every correct Friday.
* WRONG (iii): ``weather_forecast`` freshness. That is a 14-row legacy DuckDB table last
  written 2025-10-07; the live ingest writes silver ``weather``. So the check watched nothing
  the run does -- which is also why it could not see that attempt 1's weather ingest wrote
  NOTHING while reporting success. Pointed at ``weather``, completeness now refuses exactly
  that.

And one the owner named that did not count toward the halt but is a real defect:

* (i) ``validate_temporal_consistency`` compared a tz-AWARE ``kickoff_et`` with tz-NAIVE season
  bounds and raised ``TypeError``, so the ``games`` temporal check never ran. The comparison is
  fixed; the stored data is not touched.

The owner ruled all of this on 2026-09-15 (ruling R1, ``data_qa`` option (a)). The fourth
item the owner named -- one stored odds total outside 30-70 -- is a DATA judgment and is NOT
changed here; it is reported instead.

Sandboxed throughout: the parquet manager and DuckDB connection point at ``tmp_path``.
ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import DuckDBConnection, ParquetManager, upsert_silver
from pipeline.steps import build_step_registry
from scripts import data_qa
from scripts.data_qa import DataQualityMonitor
from utils import validate_temporal_consistency

SEASON, WEEK = 2026, 2
NOW = datetime.now(UTC)
WEEK_2_IDS = [f"2026_W02_T{i:02d}@H{i:02d}" for i in range(16)]


def _games(ids: list[str], *, created_at: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ids,
            "season": [SEASON] * len(ids),
            "week": [WEEK] * len(ids),
            "kickoff_et": [pd.Timestamp("2026-09-20 17:00", tz="UTC")] * len(ids),
            "home_team": ["BUF"] * len(ids),
            "away_team": ["DET"] * len(ids),
            "created_at": [created_at] * len(ids),
        }
    )


@pytest.fixture
def lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "lake"
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(root)))
    monkeypatch.setattr(
        storage_mod, "_db_connection", DuckDBConnection(str(root / "sandbox.duckdb"))
    )
    yield root
    storage_mod._db_connection.close()


def _save_parquet(root: Path, frame: pd.DataFrame, table: str) -> None:
    ParquetManager(str(root)).save(frame, f"silver/{table}.parquet")


def _save_both(root: Path, frame: pd.DataFrame, table: str) -> None:
    _save_parquet(root, frame, table)
    storage_mod.get_db_connection().create_table_from_df(frame, table, "replace")


# ---------------------------------------------------------------------------
# (i) the temporal comparison
# ---------------------------------------------------------------------------


class TestTheTemporalCheckComparesLikeWithLike:
    """(i) A tz-aware kickoff is compared against season bounds in ONE frame of reference."""

    def test_an_aware_in_season_kickoff_is_not_a_violation(self):
        frame = _games(WEEK_2_IDS[:2], created_at=NOW)
        assert validate_temporal_consistency(frame) == []

    def test_an_aware_out_of_season_kickoff_is_still_caught(self):
        frame = _games(WEEK_2_IDS[:1], created_at=NOW)
        frame["kickoff_et"] = [pd.Timestamp("2026-05-01 17:00", tz="UTC")]
        violations = validate_temporal_consistency(frame)
        assert len(violations) == 1 and "inconsistent with season 2026" in violations[0]

    def test_a_naive_kickoff_still_works(self):
        frame = _games(WEEK_2_IDS[:1], created_at=NOW)
        frame["kickoff_et"] = [pd.Timestamp("2026-09-20 13:00")]
        assert validate_temporal_consistency(frame) == []

    def test_the_games_quality_check_no_longer_errors(self, lake):
        _save_both(lake, _games(WEEK_2_IDS, created_at=NOW), "games")
        checks = DataQualityMonitor().check_data_quality("games")["checks"]
        assert "error" not in checks, (
            f"the games quality check errored: {checks.get('error')}"
        )
        assert checks["temporal_consistency"]["status"] == "pass"


# ---------------------------------------------------------------------------
# (ii) a freshness demand for data a LATER step produces
# ---------------------------------------------------------------------------


def _stale_odds(ids: list[str]) -> pd.DataFrame:
    old = NOW - timedelta(days=10)
    return pd.DataFrame(
        {
            "game_id": ids,
            "sportsbook": ["draftkings"] * len(ids),
            "total": [44.5] * len(ids),
            "snapshot_ts": [old] * len(ids),
            "last_update": [old] * len(ids),
            "created_at": [old] * len(ids),
        }
    )


def _report_over(lake: Path, tables: tuple[str, ...], monkeypatch) -> dict:
    """generate_qa_report over ONLY *tables*, with the whole-lake sections stubbed out."""
    monitor = DataQualityMonitor()
    monitor.monitored_tables = {t: monitor.monitored_tables[t] for t in tables}
    empty = {"checks": {}}
    for name in (
        "check_data_consistency",
        "check_gold_integrity",
        "check_team_abbreviations",
        "check_duckdb_parquet_consistency",
    ):
        monkeypatch.setattr(monitor, name, lambda *a, **k: empty)
    monkeypatch.setattr(monitor, "check_data_quality", lambda t: {"checks": {}})
    # Completeness is stubbed as NOT COUNTED so these tests measure freshness alone. The
    # odds completeness check has its own defect -- odds_snapshot carries no season/week
    # columns, so it counts EVERY row in the table -- which is recorded, not fixed here.
    monkeypatch.setattr(
        monitor,
        "check_data_completeness",
        lambda *a, **k: {"status": "unknown_expected"},
    )
    monkeypatch.setattr(data_qa, "get_database_stats", lambda: {})
    return monitor.generate_qa_report(SEASON, WEEK)


class TestOddsFreshnessIsNotDemandedBeforeTheOddsStep:
    """(ii) the odds are produced by a step registered after data_qa."""

    def test_the_odds_producer_really_is_registered_after_data_qa(self):
        names = [step.name for step in build_step_registry()]
        assert names.index("ingest_odds") > names.index("data_qa"), (
            "ingest_odds now runs BEFORE data_qa; the odds freshness exemption no longer "
            "has a reason to exist and must be removed"
        )

    def test_stale_odds_do_not_fail_the_gate(self, lake, monkeypatch):
        _save_both(lake, _games(WEEK_2_IDS, created_at=NOW), "games")
        _save_parquet(lake, _stale_odds(WEEK_2_IDS), "odds_snapshot")
        report = _report_over(lake, ("games", "odds_snapshot"), monkeypatch)
        assert report["summary"]["failed"] == 0, (
            f"week-old odds failed data_qa: {report['table_reports']['odds_snapshot']}. "
            "ingest_odds runs AFTER data_qa, so this refuses every correct Friday."
        )

    def test_the_odds_age_is_still_reported(self, lake):
        _save_parquet(lake, _stale_odds(WEEK_2_IDS), "odds_snapshot")
        result = DataQualityMonitor().check_data_freshness("odds_snapshot")
        assert result["age_hours"] is not None and result["age_hours"] > 24
        assert "ingest_odds" in str(result.get("not_applicable_reason", ""))

    def test_a_stale_games_table_still_fails_the_gate(self, lake, monkeypatch):
        old = NOW - timedelta(days=3)
        _save_both(lake, _games(WEEK_2_IDS, created_at=old), "games")
        _save_parquet(lake, _stale_odds(WEEK_2_IDS), "odds_snapshot")
        report = _report_over(lake, ("games", "odds_snapshot"), monkeypatch)
        assert report["summary"]["failed"] >= 1, "the exemption must not reach games"


# ---------------------------------------------------------------------------
# (iii) watch the table the live weather ingest actually writes
# ---------------------------------------------------------------------------


def _weather(ids: list[str], *, created_at: datetime = NOW) -> pd.DataFrame:
    """Silver ``weather`` in its REAL shape: keyed by ``game_id``, NO season or week.

    Measured on the production parquet on 2026-09-15. The earlier fixture carried season and
    week columns the real table does not have, which let two vacuous checks look sound: a
    completeness count that filters on season/week counts EVERY row of a table without them,
    and a freshness check that maxes over every time-like column reads a FUTURE ``game_time``
    as "fresh" forever. Both are pinned below against the real shape.
    """
    kickoff = pd.Timestamp("2026-09-20 17:00", tz="UTC")
    return pd.DataFrame(
        {
            "game_id": ids,
            "forecast_time": [created_at] * len(ids),
            "game_time": [kickoff] * len(ids),
            "is_outdoor": [True] * len(ids),
            "created_at": [created_at] * len(ids),
        }
    )


class TestWeatherIsWatchedWhereTheLiveIngestWrites:
    """(iii) silver ``weather``, not the legacy DuckDB-only ``weather_forecast``."""

    def test_the_monitor_watches_weather_not_weather_forecast(self):
        tables = DataQualityMonitor().monitored_tables
        assert "weather" in tables and "weather_forecast" not in tables

    def test_weather_written_days_ago_reads_stale_despite_future_kickoffs(self, lake):
        """``game_time`` is a KICKOFF, not a write time; it must not make stale data fresh."""
        old = NOW - timedelta(days=10)
        _save_parquet(lake, _weather(WEEK_2_IDS, created_at=old), "weather")
        result = DataQualityMonitor().check_data_freshness("weather")
        assert result["status"] == "stale", (
            f"weather written 10 days ago reads {result['status']!r} "
            f"(last_update {result['last_update']}); a future kickoff time is being read as "
            "the moment the ingest wrote"
        )

    def test_fresh_weather_reads_fresh(self, lake):
        _save_parquet(lake, _weather(WEEK_2_IDS), "weather")
        assert DataQualityMonitor().check_data_freshness("weather")["status"] == "fresh"

    def test_a_week_with_weather_for_every_game_is_complete(self, lake):
        _save_both(lake, _games(WEEK_2_IDS, created_at=NOW), "games")
        _save_parquet(lake, _weather(WEEK_2_IDS), "weather")
        result = DataQualityMonitor().check_data_completeness("weather", SEASON, WEEK)
        assert (result["status"], result["actual_count"]) == ("complete", 16)

    def test_a_week_the_weather_ingest_wrote_nothing_for_is_refused(self, lake):
        """What attempt 1 needed: a silent 'success' that wrote no rows must fail here."""
        _save_both(lake, _games(WEEK_2_IDS, created_at=NOW), "games")
        _save_parquet(lake, _weather(["2025_W01_DAL@PHI"]), "weather")
        result = DataQualityMonitor().check_data_completeness("weather", SEASON, WEEK)
        assert result["status"] == "incomplete"

    def test_the_daily_run_is_judged_on_its_slate_and_an_empty_slate_still_fails(
        self, lake
    ):
        """The daily run forecasts tomorrow's games only (2026-09-30: 1 of 16 refused)."""
        _save_both(lake, _games(WEEK_2_IDS, created_at=NOW), "games")
        slate = frozenset(WEEK_2_IDS[:1])
        _save_parquet(lake, _weather(WEEK_2_IDS[:1]), "weather")
        monitor = DataQualityMonitor()
        result = monitor.check_data_completeness("weather", SEASON, WEEK, slate)
        assert (result["status"], result["expected_count"]) == ("complete", 1)
        _save_parquet(lake, _weather(["2025_W01_DAL@PHI"]), "weather")
        result = monitor.check_data_completeness("weather", SEASON, WEEK, slate)
        assert (result["status"], result["actual_count"]) == ("incomplete", 0)


# ---------------------------------------------------------------------------
# The two TRUE refusals survive
# ---------------------------------------------------------------------------


class TestTheTrueRefusalsSurviveTheFix:
    """Controls: the checks that stopped attempt 1 safely still refuse the split."""

    @pytest.fixture
    def split(self, lake):
        history = _games(["2025_W01_DAL@PHI"], created_at=NOW).assign(
            season=2025, week=1
        )
        _save_both(lake, history, "games")
        _save_parquet(
            lake, pd.concat([history, _games(WEEK_2_IDS, created_at=NOW)]), "games"
        )
        return lake

    def test_completeness_refuses_the_split(self, split):
        result = DataQualityMonitor().check_data_completeness("games", SEASON, WEEK)
        assert (result["status"], result["actual_count"]) == ("incomplete", 0)

    def test_the_consistency_guard_refuses_the_split(self, split):
        checks = DataQualityMonitor().check_duckdb_parquet_consistency()["checks"]
        assert checks["games"]["status"] == "fail"
        assert checks["games"]["only_in_parquet_count"] == 16

    def test_both_pass_once_the_upsert_heals_the_split(self, split):
        upsert_silver(_games(WEEK_2_IDS, created_at=NOW), "games", base_path=split)
        monitor = DataQualityMonitor()
        completeness = monitor.check_data_completeness("games", SEASON, WEEK)
        consistency = monitor.check_duckdb_parquet_consistency()["checks"]["games"]
        assert (completeness["status"], consistency["status"]) == ("complete", "pass")
