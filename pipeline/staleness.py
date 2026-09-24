"""Pre-flight staleness gate for the Friday pipeline.

Provides two distinct gate concepts:
1. **Season gate** -- Is it NFL season? Aborts during offseason unless --force.
2. **Staleness checks** -- Are data sources fresh? Any partial runs? Week correct?

These are SEPARATE from health checks (environment/integrity), which are handled
by pipeline.health and always run regardless of --force.

Addresses review concerns:
- Staleness vs health vs season gate semantic overlap (HIGH, Codex)
- --force semantics too broad (MEDIUM-HIGH, consensus)
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from conf.settings import get_settings
from models.artifacts import get_latest_artifact_path
from utils.current_slate import resolve_current_slate
from utils.date_utils import ET, get_current_nfl_week
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StalenessResult:
    """Result of a staleness or season gate check.

    Attributes:
        passed: Whether the check passed (True = OK to proceed).
        errors: Blocking issues that prevent pipeline execution.
        warnings: Advisory issues (e.g., model age) that do not block.
    """

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class StalenessGate:
    """Pre-flight staleness gate with separate season and data freshness checks.

    The gate checks two categories:
    - Season gate: Is it NFL season?
    - Staleness checks: Four data freshness signals.

    Both are bypassed by force=True. Health checks (pipeline.health) are NOT
    part of this gate and always run independently.
    """

    def __init__(self, season: int, week: int, force: bool = False) -> None:
        """Initialize staleness gate.

        Args:
            season: Target NFL season year.
            week: Target NFL week number.
            force: If True, bypass season gate and staleness checks.
        """
        settings = get_settings()
        self.config = settings.config.pipeline.staleness
        # The configured data root (``DATA_ROOT_PATH`` redirects it), the root every silver
        # writer uses; the freshness checks read their files under it (33.2 review B WR-04).
        self.silver_root = f"{settings.config.data.root_path}/silver"
        self.season = season
        self.week = week
        self.force = force

    def check_season(self) -> StalenessResult:
        """Check whether the recorded schedule has a slate open now.

        This is a SEPARATE concept from staleness checks. The season gate
        prevents running the pipeline during the offseason.

        "In season" is the schedule-keyed resolver's answer
        (``utils.current_slate.resolve_current_slate``), the same one the Friday
        CLI's offseason no-op reads, so the two cannot disagree. Step 24c of Plan
        33.2-24 replaced the calendar window ``[computed opener Thursday, + 22
        weeks]``, which opened after the 2026 opener's lock and closed before the
        2026 Super Bowl.

        Returns:
            StalenessResult with passed=False if offseason.

        Raises:
            utils.current_slate.SlateResolutionError: the schedule cannot name the
                slate; never read as "offseason".
        """
        if self.force:
            logger.warning("FORCED RUN -- season check bypassed")
            return StalenessResult(
                passed=True,
                warnings=["FORCED RUN -- season check bypassed"],
            )

        now = datetime.now(ET)
        slate = resolve_current_slate(now)

        if not slate.in_season:
            msg = (
                f"Not in NFL season (offseason): the recorded schedule has no slate "
                f"open on {now.strftime('%Y-%m-%d')} (offseason value "
                f"S{slate.season}W{slate.week}). Use --force to bypass."
            )
            logger.warning(msg)
            return StalenessResult(passed=False, errors=[msg])

        logger.info(
            "Season gate passed",
            season=self.season,
            week=self.week,
        )
        return StalenessResult(passed=True)

    def check_staleness(self) -> StalenessResult:
        """Run four staleness signals (separate from season gate).

        Signals:
        1. Week validation -- does current NFL week match expected?
        2. Source data age -- are data files fresh enough?
        3. Partial run detection -- any incomplete prior runs?
        4. Model artifact age -- are models getting old? (warning only)

        Returns:
            StalenessResult with passed=False if any blocking signal fires.
        """
        if self.force:
            logger.warning("FORCED RUN -- staleness checks bypassed")
            return StalenessResult(
                passed=True,
                warnings=["FORCED RUN -- staleness checks bypassed"],
            )

        errors: list[str] = []
        warnings: list[str] = []

        # 1. Week validation
        week_ok, week_msg = self._check_week_valid()
        if not week_ok and week_msg:
            errors.append(week_msg)

        # 2. Source data freshness
        data_errors, data_warnings = self._check_data_freshness()
        errors.extend(data_errors)
        warnings.extend(data_warnings)

        # 3. Partial run detection
        partial_ok, partial_msg = self._check_partial_run()
        if not partial_ok and partial_msg:
            errors.append(partial_msg)
        elif partial_ok and partial_msg:
            # Corrupt log case -- warning, not error
            warnings.append(partial_msg)

        # 4. Model artifact age (warnings only)
        model_warnings = self._check_model_age()
        warnings.extend(model_warnings)

        passed = len(errors) == 0

        if passed:
            logger.info("Staleness checks passed", warnings=len(warnings))
        else:
            logger.warning(
                "Staleness checks failed",
                errors=errors,
                warnings=warnings,
            )

        return StalenessResult(passed=passed, errors=errors, warnings=warnings)

    def run_all_checks(self) -> StalenessResult:
        """Run season gate first, then staleness checks if season passes.

        Convenience method that merges both results. If the season gate fails,
        staleness checks are skipped (no point checking data freshness during
        offseason).

        Returns:
            Merged StalenessResult.
        """
        season_result = self.check_season()

        if not season_result.passed:
            # Season failed -- skip staleness checks entirely
            return season_result

        staleness_result = self.check_staleness()

        # Merge results
        return StalenessResult(
            passed=staleness_result.passed,
            errors=[*season_result.errors, *staleness_result.errors],
            warnings=[*season_result.warnings, *staleness_result.warnings],
        )

    # ------------------------------------------------------------------
    # Private check methods
    # ------------------------------------------------------------------

    def _check_week_valid(self) -> tuple[bool, str | None]:
        """Validate that current NFL week matches the target week.

        Returns:
            (passed, error_message_or_none)
        """
        current_season, current_week = get_current_nfl_week()

        if current_season != self.season or current_week != self.week:
            msg = (
                f"Week mismatch: expected S{self.season}W{self.week}, "
                f"got S{current_season}W{current_week}"
            )
            logger.warning(msg)
            return (False, msg)

        return (True, None)

    def _check_data_freshness(self) -> tuple[list[str], list[str]]:
        """Check file modification times for source data staleness.

        Returns:
            (errors, warnings) -- errors block execution, warnings are advisory.
        """
        errors: list[str] = []
        warnings: list[str] = []
        now = time.time()

        # games.parquet -- blocking if stale
        games_path = Path(self.silver_root, "games.parquet")
        if games_path.exists():
            age_hours = (now - games_path.stat().st_mtime) / 3600
            if age_hours > self.config.data_age_hours:
                errors.append(
                    f"Games data is stale: {age_hours:.1f}h old "
                    f"(threshold: {self.config.data_age_hours}h)"
                )
        else:
            errors.append(f"Games data file not found: {games_path}")

        # odds_snapshot.parquet -- warning only (may not exist yet for new week)
        odds_path = Path(self.silver_root, "odds_snapshot.parquet")
        if odds_path.exists():
            age_hours = (now - odds_path.stat().st_mtime) / 3600
            if age_hours > self.config.odds_age_hours:
                warnings.append(
                    f"Odds data is stale: {age_hours:.1f}h old "
                    f"(threshold: {self.config.odds_age_hours}h)"
                )
        else:
            warnings.append("Odds snapshot not found (may not exist yet for new week)")

        # silver weather -- warning only. The TABLE FILE, weather.parquet (33.2 review B WR-04):
        # this used to test a "data/silver/weather" DIRECTORY that no writer creates, so every
        # run warned "not found" and a genuinely stale weather table was never detected.
        weather_path = Path(self.silver_root, "weather.parquet")
        if weather_path.exists():
            age_hours = (now - weather_path.stat().st_mtime) / 3600
            if age_hours > self.config.weather_age_hours:
                warnings.append(
                    f"Weather data is stale: {age_hours:.1f}h old "
                    f"(threshold: {self.config.weather_age_hours}h)"
                )
        else:
            warnings.append(f"Weather data file not found: {weather_path}")

        return errors, warnings

    def _check_partial_run(self) -> tuple[bool, str | None]:
        """Check for incomplete prior pipeline runs.

        Reads the execution log JSON file directly (no import dependency on
        pipeline.execution_log) to detect partial runs.

        Returns:
            (passed, message_or_none) -- message is error if failed, warning if corrupt.
        """
        log_path = Path(self.config.partial_run_log)

        if not log_path.exists():
            # No prior run log -- nothing to check
            return (True, None)

        try:
            with open(log_path, encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            # Corrupt log file -- warning, not blocking
            return (
                True,
                f"Execution log is corrupt JSON: {self.config.partial_run_log}",
            )

        status = log_data.get("status", "")
        pid = log_data.get("pid", "unknown")

        # Check for incomplete run (still marked as running)
        if status == "running":
            msg = (
                f"Incomplete prior pipeline run detected (PID: {pid}). "
                f"Previous run still marked as 'running'."
            )
            logger.warning(msg)
            return (False, msg)

        # Completed runs from prior weeks are fine
        return (True, None)

    def _check_model_age(self) -> list[str]:
        """Check age of model artifacts. Produces warnings only, never errors.

        Resolves the ACTIVE deployed models via ``artifacts/latest.json`` (the
        same pointer the prediction pipeline loads), NOT the fixed-name stubs
        under ``artifacts/models/``. The stubs are ~250 days stale, so the
        former convention silently misreported model age (IN-02). A missing or
        unparseable manifest reports the target as absent rather than crashing.

        Returns:
            List of warning strings (may be empty).
        """
        warnings: list[str] = []
        now = time.time()
        threshold_seconds = self.config.model_age_days * 86400
        artifacts_dir = Path("artifacts")

        for target in ("wp", "ats", "ou"):
            try:
                artifact_dir = get_latest_artifact_path(
                    target, artifacts_dir=artifacts_dir
                )
            except (OSError, json.JSONDecodeError):
                artifact_dir = None
            model_path = artifact_dir / "model.pkl" if artifact_dir else None

            if model_path is not None and model_path.exists():
                age_days = (now - model_path.stat().st_mtime) / 86400
                if age_days * 86400 > threshold_seconds:
                    warnings.append(
                        f"Model artifact {target} is {age_days:.0f} days old "
                        f"(threshold: {self.config.model_age_days} days)"
                    )
            else:
                warnings.append(f"Model artifact missing: {target}")

        return warnings
