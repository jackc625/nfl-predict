"""Tests for pipeline.staleness -- StalenessGate with season gate and staleness checks."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from utils.date_utils import ET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_resolver(st_mtime: float):
    """Build a get_latest_artifact_path stand-in for the latest.json convention.

    _check_model_age now resolves each target via
    ``get_latest_artifact_path(target, artifacts_dir=...)`` (the real
    artifacts/latest.json convention, CR-01/IN-02 fix) instead of probing
    fixed-name ``artifacts/models/*.pkl`` stubs. The returned artifact dir's
    ``/ "model.pkl"`` resolves to a mock with the given mtime so the age math
    runs against a controlled timestamp.
    """

    def _resolver(target, artifacts_dir=None):
        model_path = MagicMock()
        model_path.exists.return_value = True
        stat = MagicMock()
        stat.st_mtime = st_mtime
        model_path.stat.return_value = stat

        artifact_dir = MagicMock()
        artifact_dir.__truediv__.return_value = model_path
        return artifact_dir

    return _resolver


def _make_settings_mock(overrides: dict | None = None):
    """Create a mock Settings object with PipelineStalenessConfig defaults."""
    defaults = {
        "data_age_hours": 168,
        "odds_age_hours": 168,
        "weather_age_hours": 336,
        "model_age_days": 90,
        "partial_run_log": "logs/friday_pipeline.json",
    }
    if overrides:
        defaults.update(overrides)

    staleness_cfg = MagicMock(**defaults)
    pipeline_cfg = MagicMock(staleness=staleness_cfg)
    config = MagicMock(pipeline=pipeline_cfg)
    settings = MagicMock(config=config)
    return settings


# ---------------------------------------------------------------------------
# Season gate tests
# ---------------------------------------------------------------------------


class TestCheckSeason:
    """Tests for StalenessGate.check_season()."""

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.datetime")
    @patch("pipeline.staleness.get_nfl_season_start")
    def test_check_season_passes_in_season(
        self, mock_season_start, mock_dt, mock_settings
    ):
        """Season gate passes during in-season (October)."""
        mock_settings.return_value = _make_settings_mock()
        # Season 2025 starts September 4
        season_start = datetime(2025, 9, 4, tzinfo=ET)
        mock_season_start.return_value = season_start
        # Current time is October 10 (in-season)
        mock_dt.now.return_value = datetime(2025, 10, 10, 12, 0, tzinfo=ET)

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_season()

        assert result.passed is True
        assert len(result.errors) == 0

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.datetime")
    @patch("pipeline.staleness.get_nfl_season_start")
    def test_check_season_fails_offseason(
        self, mock_season_start, mock_dt, mock_settings
    ):
        """Season gate fails during offseason (April)."""
        mock_settings.return_value = _make_settings_mock()
        season_start = datetime(2025, 9, 4, tzinfo=ET)
        mock_season_start.return_value = season_start
        # Current time is April 15 (offseason -- after season end)
        mock_dt.now.return_value = datetime(2026, 4, 15, 12, 0, tzinfo=ET)

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=1)
        result = gate.check_season()

        assert result.passed is False
        assert any(
            "offseason" in e.lower() or "season" in e.lower() for e in result.errors
        )

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.datetime")
    @patch("pipeline.staleness.get_nfl_season_start")
    def test_force_bypasses_season(self, mock_season_start, mock_dt, mock_settings):
        """force=True during offseason returns passed=True with bypass message."""
        mock_settings.return_value = _make_settings_mock()
        season_start = datetime(2025, 9, 4, tzinfo=ET)
        mock_season_start.return_value = season_start
        mock_dt.now.return_value = datetime(2026, 4, 15, 12, 0, tzinfo=ET)

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=1, force=True)
        result = gate.check_season()

        assert result.passed is True
        assert any("FORCED RUN" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Staleness check tests
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    """Tests for StalenessGate.check_staleness() -- four staleness signals."""

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    def test_check_staleness_all_pass(
        self, mock_time, mock_path_cls, mock_resolver, mock_get_week, mock_settings
    ):
        """All data fresh, no partial run -- staleness passes."""
        mock_settings.return_value = _make_settings_mock()
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        # Models resolved via latest.json, 1 day old (fresh).
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - 86400)

        # games.parquet exists and is fresh (modified 1 hour ago)
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - 3600  # 1 hour ago
        games_path.stat.return_value = games_stat

        # odds exists and is fresh
        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        # weather dir exists and is fresh
        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        # Mock Path() calls in order: 3 data files, partial-run log, then
        # the single Path("artifacts") in _check_model_age (resolver is mocked).
        path_instances = [
            games_path,
            odds_path,
            weather_path,
            MagicMock(exists=MagicMock(return_value=False)),  # partial run log
            MagicMock(),  # Path("artifacts") root
        ]
        mock_path_cls.side_effect = path_instances

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is True
        assert len(result.errors) == 0

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    def test_check_staleness_week_mismatch(self, mock_get_week, mock_settings):
        """Week validation fails when current NFL week does not match expected."""
        mock_settings.return_value = _make_settings_mock()
        mock_get_week.return_value = (2025, 6)  # Different from expected week 5

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is False
        assert any("mismatch" in e.lower() or "Week" in e for e in result.errors)

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    def test_check_staleness_stale_games_data(
        self, mock_time, mock_path_cls, mock_resolver, mock_get_week, mock_settings
    ):
        """games.parquet age > data_age_hours triggers error (passed=False)."""
        mock_settings.return_value = _make_settings_mock({"data_age_hours": 168})
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - 86400)

        # games.parquet exists but is 200 hours old (> 168 threshold)
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - (200 * 3600)  # 200 hours ago
        games_path.stat.return_value = games_stat

        # odds -- fresh
        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        # weather -- fresh
        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        # No partial run log
        log_path = MagicMock()
        log_path.exists.return_value = False

        mock_path_cls.side_effect = [
            games_path,
            odds_path,
            weather_path,
            log_path,
            MagicMock(),  # Path("artifacts") root
        ]

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is False
        assert any("games" in e.lower() or "stale" in e.lower() for e in result.errors)

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    @patch("builtins.open")
    @patch("pipeline.staleness.json")
    def test_check_staleness_partial_run_detected(
        self,
        mock_json,
        mock_open,
        mock_time,
        mock_path_cls,
        mock_resolver,
        mock_get_week,
        mock_settings,
    ):
        """Partial run detection: status='running' in log JSON triggers error."""
        mock_settings.return_value = _make_settings_mock()
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - 86400)

        # Fresh data files
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - 3600
        games_path.stat.return_value = games_stat

        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        # Log file exists
        log_path = MagicMock()
        log_path.exists.return_value = True

        mock_path_cls.side_effect = [
            games_path,
            odds_path,
            weather_path,
            log_path,
            MagicMock(),  # Path("artifacts") root
        ]

        # Log JSON says status=running
        mock_json.load.return_value = {
            "status": "running",
            "pid": 12345,
            "season": 2025,
            "week": 5,
            "steps": [],
        }

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is False
        assert any(
            "running" in e.lower() or "incomplete" in e.lower() or "12345" in e
            for e in result.errors
        )

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    @patch("builtins.open")
    @patch("pipeline.staleness.json")
    def test_check_staleness_partial_run_corrupt_log(
        self,
        mock_json,
        mock_open,
        mock_time,
        mock_path_cls,
        mock_resolver,
        mock_get_week,
        mock_settings,
    ):
        """Corrupt log file returns passed=True with warning."""
        mock_settings.return_value = _make_settings_mock()
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - 86400)

        # Fresh files
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - 3600
        games_path.stat.return_value = games_stat

        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        # Log file exists but corrupt
        log_path = MagicMock()
        log_path.exists.return_value = True

        mock_path_cls.side_effect = [
            games_path,
            odds_path,
            weather_path,
            log_path,
            MagicMock(),  # Path("artifacts") root
        ]

        mock_json.load.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_json.JSONDecodeError = json.JSONDecodeError

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is True
        assert any(
            "corrupt" in w.lower() or "json" in w.lower() for w in result.warnings
        )

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    def test_check_staleness_model_age_warns_only(
        self, mock_time, mock_path_cls, mock_resolver, mock_get_week, mock_settings
    ):
        """Models older than threshold produce warning, NOT error (passed=True)."""
        mock_settings.return_value = _make_settings_mock({"model_age_days": 90})
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        # Models resolved via latest.json are 100 days old (> 90 threshold).
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - (100 * 86400))

        # Fresh data
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - 3600
        games_path.stat.return_value = games_stat

        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        # No partial run
        log_path = MagicMock()
        log_path.exists.return_value = False

        mock_path_cls.side_effect = [
            games_path,
            odds_path,
            weather_path,
            log_path,
            MagicMock(),  # Path("artifacts") root
        ]

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.check_staleness()

        assert result.passed is True  # Warnings don't cause failure
        assert len(result.warnings) > 0
        assert any("model" in w.lower() for w in result.warnings)

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.get_current_nfl_week")
    def test_force_bypasses_staleness(self, mock_get_week, mock_settings):
        """force=True bypasses staleness checks with bypass message."""
        mock_settings.return_value = _make_settings_mock()
        mock_get_week.return_value = (2025, 6)  # Week mismatch would normally fail

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5, force=True)
        result = gate.check_staleness()

        assert result.passed is True
        assert any("FORCED RUN" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# run_all_checks tests
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    """Tests for StalenessGate.run_all_checks()."""

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.datetime")
    @patch("pipeline.staleness.get_nfl_season_start")
    @patch("pipeline.staleness.get_current_nfl_week")
    @patch("pipeline.staleness.get_latest_artifact_path")
    @patch("pipeline.staleness.Path")
    @patch("pipeline.staleness.time")
    def test_run_all_checks_merges_results(
        self,
        mock_time,
        mock_path_cls,
        mock_resolver,
        mock_get_week,
        mock_season_start,
        mock_dt,
        mock_settings,
    ):
        """run_all_checks combines season + staleness results."""
        mock_settings.return_value = _make_settings_mock()
        # In-season
        season_start = datetime(2025, 9, 4, tzinfo=ET)
        mock_season_start.return_value = season_start
        mock_dt.now.return_value = datetime(2025, 10, 10, 12, 0, tzinfo=ET)
        mock_get_week.return_value = (2025, 5)
        mock_time.time.return_value = 1000000.0
        mock_resolver.side_effect = _make_model_resolver(1000000.0 - 86400)

        # Fresh data
        games_path = MagicMock()
        games_path.exists.return_value = True
        games_stat = MagicMock()
        games_stat.st_mtime = 1000000.0 - 3600
        games_path.stat.return_value = games_stat

        odds_path = MagicMock()
        odds_path.exists.return_value = True
        odds_stat = MagicMock()
        odds_stat.st_mtime = 1000000.0 - 3600
        odds_path.stat.return_value = odds_stat

        weather_path = MagicMock()
        weather_path.exists.return_value = True
        weather_stat = MagicMock()
        weather_stat.st_mtime = 1000000.0 - 3600
        weather_path.stat.return_value = weather_stat

        log_path = MagicMock()
        log_path.exists.return_value = False

        mock_path_cls.side_effect = [
            games_path,
            odds_path,
            weather_path,
            log_path,
            MagicMock(),  # Path("artifacts") root
        ]

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=5)
        result = gate.run_all_checks()

        assert result.passed is True

    @patch("pipeline.staleness.get_settings")
    @patch("pipeline.staleness.datetime")
    @patch("pipeline.staleness.get_nfl_season_start")
    def test_run_all_checks_season_fails_skips_staleness(
        self, mock_season_start, mock_dt, mock_settings
    ):
        """If season fails, staleness checks are skipped."""
        mock_settings.return_value = _make_settings_mock()
        season_start = datetime(2025, 9, 4, tzinfo=ET)
        mock_season_start.return_value = season_start
        mock_dt.now.return_value = datetime(2026, 4, 15, 12, 0, tzinfo=ET)

        from pipeline.staleness import StalenessGate

        gate = StalenessGate(season=2025, week=1)
        result = gate.run_all_checks()

        assert result.passed is False
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPipelineStalenessConfig:
    """Tests for PipelineStalenessConfig loaded from settings."""

    def test_config_defaults(self):
        """PipelineStalenessConfig loads with correct defaults."""
        from conf.settings import PipelineStalenessConfig

        cfg = PipelineStalenessConfig()
        assert cfg.data_age_hours == 168
        assert cfg.odds_age_hours == 168
        assert cfg.weather_age_hours == 336
        assert cfg.model_age_days == 90
        assert cfg.partial_run_log == "logs/friday_pipeline.json"

    def test_pipeline_config_has_subsections(self):
        """PipelineConfig has staleness, retry, schedule subsections."""
        from conf.settings import PipelineConfig

        cfg = PipelineConfig()
        assert hasattr(cfg, "staleness")
        assert hasattr(cfg, "retry")
        assert hasattr(cfg, "schedule")
