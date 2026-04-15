"""Shared pytest fixtures for API tests.

Provides:
- test_db: Temporary DuckDB with CACHE_SCHEMA and sample data
- test_client: FastAPI TestClient with overridden DB_PATH
- sample_game_data: List of sample game dicts
- Phase 16 insights data (via _insights_* helpers): multi-season, multi-target
  backtest_predictions, feature_importances, market rows, and chart_cache
  placeholder entries covering edge cases E1..E15.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA
from api.services import clear_cache

# ---------------------------------------------------------------------------
# Phase 16 insights data helpers
# ---------------------------------------------------------------------------
# These helpers return lists of dicts that the `test_db` fixture inserts into
# the corresponding tables. They live at module scope so Phase 16 chart-generator
# and metric-helper tests can consume them directly without spinning up DuckDB.
#
# Edge cases MANDATED by Plan 16-01 (E1..E15) are tagged with inline comments on
# the row that exercises them. Every edge case must be present — test files
# depend on this coverage.

# Authoritative insights chart-id tuple (Plan 16-02 will export from api.charts).
INSIGHTS_CHART_IDS: tuple[str, ...] = (
    "insights_calibration_ats",
    "insights_calibration_ou",
    "insights_feature_importance_wp",
    "insights_feature_importance_ats",
    "insights_feature_importance_ou",
    "insights_accuracy_trend",
    "insights_model_vs_market_wp",
    "insights_model_vs_market_ats",
    "insights_model_vs_market_ou",
)
assert len(INSIGHTS_CHART_IDS) == 9


_TEAM_ABBREVS = ("KC", "BUF", "PHI", "GB", "SF", "DAL", "CIN", "MIA", "BAL", "NYJ")


def _gid(season: int, week: int, away: str, home: str) -> str:
    """Build a deterministic fixture game_id."""
    return f"{season}_W{week:02d}_{away}@{home}"


def _insights_backtest_rows() -> list[dict]:
    """Rows for the `backtest_predictions` table covering WP / ATS / OU across
    2021-2024 with explicit edge-case rows E6, E7, E8, E11-E15.

    Schema columns: game_id, season, week, target, model_prob, actual,
    probability_clv, has_closing_odds.
    """
    rows: list[dict] = []

    # ------------------------------------------------------------------ WP
    # 10 WP rows per season for 2021-2023; 5 for 2024 (E8 small-sample season).
    wp_seasons: dict[int, int] = {2021: 10, 2022: 10, 2023: 10, 2024: 5}  # E8
    for season, count in wp_seasons.items():
        for i in range(count):
            # Walk around the team list; probabilities span 0.05..0.95.
            away = _TEAM_ABBREVS[i % len(_TEAM_ABBREVS)]
            home = _TEAM_ABBREVS[(i + 1) % len(_TEAM_ABBREVS)]
            model_prob = 0.05 + (i / max(count - 1, 1)) * 0.90
            actual = 1.0 if model_prob >= 0.5 else 0.0
            rows.append(
                {
                    "game_id": _gid(season, i + 1, away, home),
                    "season": season,
                    "week": i + 1,
                    "target": "wp",
                    "model_prob": round(model_prob, 4),
                    "actual": actual,
                    "probability_clv": round(0.01 * (i - count / 2), 4),
                    "has_closing_odds": True,
                },
            )

    # E11: WP model_prob = 0.0 exactly (log-loss clipping case).
    rows.append(
        {
            "game_id": _gid(2022, 18, "NYJ", "BAL"),
            "season": 2022,
            "week": 18,
            "target": "wp",
            "model_prob": 0.0,  # E11: WP model_prob at exactly 0.0
            "actual": 1.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E12: WP model_prob = 1.0 exactly (mirror of E11).
    rows.append(
        {
            "game_id": _gid(2022, 19, "MIA", "CIN"),
            "season": 2022,
            "week": 19,
            "target": "wp",
            "model_prob": 1.0,  # E12: WP model_prob at exactly 1.0
            "actual": 0.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )

    # ------------------------------------------------------------------ ATS
    # 10 ATS rows per season 2021-2024 plus overflow + push rows.
    ats_seasons = (2021, 2022, 2023, 2024)
    for season in ats_seasons:
        for i in range(10):
            away = _TEAM_ABBREVS[i % len(_TEAM_ABBREVS)]
            home = _TEAM_ABBREVS[(i + 2) % len(_TEAM_ABBREVS)]
            model_prob = -10.0 + (i * 2.0)  # -10..+8 predicted home margin
            actual = model_prob + (1.5 if i % 2 else -1.5)
            rows.append(
                {
                    "game_id": _gid(season, i + 1, away, home),
                    "season": season,
                    "week": i + 1,
                    "target": "ats",
                    "model_prob": model_prob,
                    "actual": actual,
                    "probability_clv": round(0.01 * i, 4),
                    "has_closing_odds": True,
                },
            )

    # E6: ATS push — predicted home margin equals -market_spread (set below in
    # market rows so this row's actual matches the spread exactly).
    rows.append(
        {
            "game_id": _gid(2023, 15, "DAL", "PHI"),
            "season": 2023,
            "week": 15,
            "target": "ats",
            "model_prob": 3.0,
            "actual": 3.0,  # E6: ATS push (actual == -market_spread == 3.0)
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E13: ATS residual <= -21 (model predicts -25, actual +5 => residual -30).
    rows.append(
        {
            "game_id": _gid(2021, 16, "NYJ", "SF"),
            "season": 2021,
            "week": 16,
            "target": "ats",
            "model_prob": -25.0,  # E13: residual = model - actual = -30 (<= -21 bin)
            "actual": 5.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E14: ATS residual >= +21 (model predicts +25, actual -5 => residual +30).
    rows.append(
        {
            "game_id": _gid(2021, 17, "MIA", "KC"),
            "season": 2021,
            "week": 17,
            "target": "ats",
            "model_prob": 25.0,  # E14: residual = model - actual = +30 (>= +21 bin)
            "actual": -5.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )

    # ------------------------------------------------------------------ OU
    # 10 OU rows per season 2021-2024.
    for season in ats_seasons:
        for i in range(10):
            away = _TEAM_ABBREVS[(i + 3) % len(_TEAM_ABBREVS)]
            home = _TEAM_ABBREVS[(i + 4) % len(_TEAM_ABBREVS)]
            model_prob = 38.0 + (i * 2.0)  # 38..56 predicted total points
            actual = model_prob + (2.5 if i % 2 else -2.5)
            rows.append(
                {
                    "game_id": _gid(season, i + 1, away, home),
                    "season": season,
                    "week": i + 1,
                    "target": "ou",
                    "model_prob": model_prob,
                    "actual": actual,
                    "probability_clv": round(0.01 * i, 4),
                    "has_closing_odds": True,
                },
            )

    # E7: OU push — actual == market_total (45.0 matches market_total below).
    rows.append(
        {
            "game_id": _gid(2023, 14, "BUF", "MIA"),
            "season": 2023,
            "week": 14,
            "target": "ou",
            "model_prob": 47.0,
            "actual": 45.0,  # E7: OU push (actual == market_total == 45.0)
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E15a: OU prediction below the fixed 35-65 bin window.
    rows.append(
        {
            "game_id": _gid(2021, 20, "NYJ", "CIN"),
            "season": 2021,
            "week": 20,
            "target": "ou",
            "model_prob": 32.0,  # E15: OU model_prob < 35 (overflow low)
            "actual": 33.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E15b: OU prediction above the fixed 35-65 bin window.
    rows.append(
        {
            "game_id": _gid(2021, 21, "SF", "KC"),
            "season": 2021,
            "week": 21,
            "target": "ou",
            "model_prob": 68.0,  # E15: OU model_prob > 65 (overflow high)
            "actual": 67.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )

    return rows


def _insights_feature_importance_rows() -> list[dict]:
    """Rows for `feature_importances`, 15 features per target (D-02).

    All rows use game_id='_model_' per the existing snapshot convention.
    Importance values are in [0.01, 0.30] and roughly sum to 1.0 per target.
    """
    feature_names = (
        "elo_diff",
        "rolling_off_epa",
        "rolling_def_epa",
        "snapshot_spread",
        "rolling_total_epa",
        "elo_momentum_home",
        "elo_rank_home",
        "home_rest_days",
        "away_rest_days",
        "weather_severity",
        "is_outdoor",
        "is_divisional",
        "is_primetime",
        "rolling_turnover_diff",
        "rolling_scoring_margin",
    )
    # Importance weights per target (sum ~= 1.0).
    weights = (
        0.18,
        0.14,
        0.12,
        0.10,
        0.09,
        0.07,
        0.06,
        0.05,
        0.04,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.01,
    )
    rows: list[dict] = []
    for target in ("wp", "ats", "ou"):
        for name, w in zip(feature_names, weights, strict=True):
            # Slight per-target perturbation so rows are not identical across targets.
            bump = {"wp": 0.0, "ats": 0.005, "ou": -0.005}[target]
            rows.append(
                {
                    "game_id": "_model_",
                    "target": target,
                    "feature_name": name,
                    "importance": round(max(0.01, w + bump), 4),
                },
            )
    return rows


def _insights_market_rows() -> list[dict]:
    """Rows for the `predictions` table exercising market edge cases E1..E5 and
    the sign-convention edges E9/E10.

    Every row here corresponds to a backtest_predictions row so the downstream
    model-vs-market join has data. Schema matches _PREDICTIONS_COLUMNS.
    """
    # Reference game IDs that exist in _insights_backtest_rows().
    base_game_date = datetime(2023, 10, 1, 17, 0, tzinfo=UTC)

    def _row(
        game_id: str,
        *,
        season: int,
        week: int,
        home: str,
        away: str,
        market_spread: float | None,
        market_total: float | None,
        market_ml_home: int | None,
        market_ml_away: int | None,
        note: str,
    ) -> dict:
        # Embed the edge-case note as a comment key so grep/debug is easy; the
        # `_note` key is stripped before insert (see insert loop in `test_db`).
        return {
            "game_id": game_id,
            "season": season,
            "week": week,
            "game_date": base_game_date,
            "home_team": home,
            "away_team": away,
            "status": "completed",
            "home_score": 24,
            "away_score": 20,
            "wp_prob": 0.55,
            "wp_confidence": "medium",
            "ats_prediction": -market_spread if market_spread is not None else 0.0,
            "ats_confidence": "medium",
            "ou_prediction": market_total if market_total is not None else 45.0,
            "ou_confidence": "medium",
            "market_spread": market_spread,
            "market_total": market_total,
            "market_ml_home": market_ml_home,
            "market_ml_away": market_ml_away,
            "wp_edge": 0.01,
            "ats_edge": 0.01,
            "ou_edge": 0.01,
            "blended_wp": 0.55,
            "blended_ats": -market_spread if market_spread is not None else 0.0,
            "blended_ou": market_total if market_total is not None else 45.0,
            "_note": note,
        }

    rows: list[dict] = [
        # E1: null home_moneyline (away populated).
        _row(
            _gid(2023, 15, "DAL", "PHI"),
            season=2023,
            week=15,
            home="PHI",
            away="DAL",
            market_spread=-3.0,  # E6 backtest row references this (actual == -market_spread => 3.0? -> E6 uses 3.0)
            market_total=47.0,
            market_ml_home=None,  # E1: null home_moneyline
            market_ml_away=140,
            note="E1",
        ),
        # E2: null away_moneyline (home populated).
        _row(
            _gid(2023, 14, "BUF", "MIA"),
            season=2023,
            week=14,
            home="MIA",
            away="BUF",
            market_spread=2.5,
            market_total=45.0,  # E7 OU push depends on this
            market_ml_home=-130,
            market_ml_away=None,  # E2: null away_moneyline
            note="E2",
        ),
        # E3: both moneylines null (row must be excluded from market comparison).
        _row(
            _gid(2021, 16, "NYJ", "SF"),
            season=2021,
            week=16,
            home="SF",
            away="NYJ",
            market_spread=-7.5,  # E9: home favorite (market_spread < 0)
            market_total=42.0,
            market_ml_home=None,  # E3: both moneylines null
            market_ml_away=None,
            note="E3+E9",
        ),
        # E4: null market_spread for a game that has an ATS backtest prediction.
        _row(
            _gid(2021, 17, "MIA", "KC"),
            season=2021,
            week=17,
            home="KC",
            away="MIA",
            market_spread=None,  # E4: null market_spread
            market_total=51.0,
            market_ml_home=-200,
            market_ml_away=170,
            note="E4",
        ),
        # E5: null market_total for a game that has an OU backtest prediction.
        _row(
            _gid(2021, 20, "NYJ", "CIN"),
            season=2021,
            week=20,
            home="CIN",
            away="NYJ",
            market_spread=-4.0,
            market_total=None,  # E5: null market_total
            market_ml_home=-180,
            market_ml_away=160,
            note="E5",
        ),
        # E10: home underdog (market_spread > 0) with a companion OU row.
        _row(
            _gid(2021, 21, "SF", "KC"),
            season=2021,
            week=21,
            home="KC",
            away="SF",
            market_spread=+3.5,  # E10: home underdog (market_spread > 0)
            market_total=55.0,
            market_ml_home=150,
            market_ml_away=-175,
            note="E10",
        ),
    ]
    return rows


@pytest.fixture(autouse=True)
def _isolate_data_service_cache() -> Iterator[None]:
    """Clear the module-level DataService TTLCache before and after every test.

    The ``api.services._cache`` is a module-level :class:`cachetools.TTLCache`
    (plan 15-02). Because Python modules are singletons, cache entries leak
    across tests unless explicitly cleared. This fixture uses ``try/finally``
    so isolation holds even when a test fails mid-way.
    """
    clear_cache()
    try:
        yield
    finally:
        clear_cache()


def _sample_game_data() -> list[dict]:
    """Return a list of 3 sample game dicts with all prediction fields."""
    return [
        {
            "game_id": "2024_W01_BUF@KC",
            "season": 2024,
            "week": 1,
            "game_date": datetime(2024, 9, 5, 20, 15, tzinfo=UTC),
            "home_team": "KC",
            "away_team": "BUF",
            "status": "completed",
            "home_score": 27,
            "away_score": 20,
            "wp_prob": 0.62,
            "wp_confidence": "medium",
            "ats_prediction": -3.5,
            "ats_confidence": "high",
            "ou_prediction": 48.5,
            "ou_confidence": "medium",
            "market_spread": -3.0,
            "market_total": 47.5,
            "market_ml_home": -155,
            "market_ml_away": 135,
            "wp_edge": 0.05,
            "ats_edge": 0.02,
            "ou_edge": 0.01,
            "blended_wp": 0.60,
            "blended_ats": -3.2,
            "blended_ou": 48.0,
        },
        {
            "game_id": "2024_W01_PHI@GB",
            "season": 2024,
            "week": 1,
            "game_date": datetime(2024, 9, 6, 20, 15, tzinfo=UTC),
            "home_team": "GB",
            "away_team": "PHI",
            "status": "completed",
            "home_score": 34,
            "away_score": 29,
            "wp_prob": 0.45,
            "wp_confidence": "low",
            "ats_prediction": 1.5,
            "ats_confidence": "medium",
            "ou_prediction": 50.0,
            "ou_confidence": "high",
            "market_spread": 2.5,
            "market_total": 49.0,
            "market_ml_home": 110,
            "market_ml_away": -130,
            "wp_edge": -0.03,
            "ats_edge": 0.04,
            "ou_edge": 0.03,
            "blended_wp": 0.47,
            "blended_ats": 1.8,
            "blended_ou": 49.5,
        },
        {
            "game_id": "2023_W18_SF@SEA",
            "season": 2023,
            "week": 18,
            "game_date": datetime(2024, 1, 7, 16, 25, tzinfo=UTC),
            "home_team": "SEA",
            "away_team": "SF",
            "status": "completed",
            "home_score": 20,
            "away_score": 24,
            "wp_prob": 0.38,
            "wp_confidence": "medium",
            "ats_prediction": 3.0,
            "ats_confidence": "low",
            "ou_prediction": 45.0,
            "ou_confidence": "medium",
            "market_spread": 3.5,
            "market_total": 44.5,
            "market_ml_home": 150,
            "market_ml_away": -175,
            "wp_edge": 0.01,
            "ats_edge": -0.01,
            "ou_edge": 0.02,
            "blended_wp": 0.40,
            "blended_ats": 3.2,
            "blended_ou": 44.8,
        },
    ]


@pytest.fixture()
def sample_game_data() -> list[dict]:
    """Return sample game data for testing."""
    return _sample_game_data()


@pytest.fixture()
def test_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB with CACHE_SCHEMA and sample data.

    Returns the path to the database file.
    """
    db_path = tmp_path / "test_cache.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create schema
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)

    # Insert sample predictions
    games = _sample_game_data()
    for game in games:
        conn.execute(
            """
            INSERT INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            [
                game["game_id"],
                game["season"],
                game["week"],
                game["game_date"],
                game["home_team"],
                game["away_team"],
                game["status"],
                game["home_score"],
                game["away_score"],
                game["wp_prob"],
                game["wp_confidence"],
                game["ats_prediction"],
                game["ats_confidence"],
                game["ou_prediction"],
                game["ou_confidence"],
                game["market_spread"],
                game["market_total"],
                game["market_ml_home"],
                game["market_ml_away"],
                game["wp_edge"],
                game["ats_edge"],
                game["ou_edge"],
                game["blended_wp"],
                game["blended_ats"],
                game["blended_ou"],
            ],
        )

    # Insert sample feature importances
    importances = [
        ("_model_", "wp", "elo_diff", 0.25),
        ("_model_", "wp", "rolling_off_epa", 0.18),
        ("_model_", "wp", "snapshot_spread", 0.15),
        ("_model_", "ats", "elo_diff", 0.20),
        ("_model_", "ats", "rolling_off_epa", 0.22),
        ("_model_", "ou", "rolling_total_epa", 0.30),
    ]
    conn.executemany(
        "INSERT INTO feature_importances VALUES (?, ?, ?, ?)",
        importances,
    )

    # Insert backtest metrics (season-level for performance page)
    backtest_metrics = [
        (2023, "wp", "accuracy", 0.65),
        (2023, "wp", "brier_score", 0.22),
        (2023, "wp", "ece", 0.035),
        (2023, "ats", "mae", 6.5),
        (2023, "ats", "rmse", 8.2),
        (2024, "wp", "accuracy", 0.68),
        (2024, "wp", "brier_score", 0.21),
        (2024, "wp", "ece", 0.030),
        (2024, "ats", "mae", 6.2),
        (2024, "ats", "rmse", 7.9),
    ]
    conn.executemany(
        "INSERT INTO backtest_metrics VALUES (?, ?, ?, ?)",
        backtest_metrics,
    )

    # Insert backtest predictions (for chart generation)
    backtest_predictions = [
        ("2023_W18_SF@SEA", 2023, 18, "wp", 0.38, 0.0, 0.02, True),
        ("2024_W01_BUF@KC", 2024, 1, "wp", 0.62, 1.0, 0.05, True),
        ("2024_W01_PHI@GB", 2024, 1, "wp", 0.45, 1.0, -0.03, True),
    ]
    conn.executemany(
        "INSERT INTO backtest_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        backtest_predictions,
    )

    # Insert chart cache entries (empty HTML for testing graceful fallback)
    chart_now = datetime.now(tz=UTC)
    chart_entries = [
        (
            "calibration",
            '<div data-chart-id="calibration">test calibration</div>',
            chart_now,
        ),
        ("clv", "<div>test clv</div>", chart_now),
        ("heatmap", "<div>test heatmap</div>", chart_now),
        ("equity", "<div>test equity</div>", chart_now),
    ]
    # Phase 16: marker divs for each insights chart_id so route tests can assert
    # the /insights route consumes exactly this set.
    for chart_id in INSIGHTS_CHART_IDS:
        chart_entries.append(
            (
                chart_id,
                f'<div data-chart-id="{chart_id}">fixture {chart_id}</div>',
                chart_now,
            ),
        )
    conn.executemany(
        "INSERT INTO chart_cache VALUES (?, ?, ?)",
        chart_entries,
    )

    # -------------------------------------------------------------------
    # Phase 16 insights data: multi-season backtest_predictions with edge
    # cases E1..E15, 15-per-target feature_importances, per-season backtest
    # metrics for all three targets, and matching market rows in predictions.
    # -------------------------------------------------------------------
    # Backtest predictions (E6, E7, E8, E11-E15 live here).
    for br in _insights_backtest_rows():
        conn.execute(
            "INSERT OR REPLACE INTO backtest_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                br["game_id"],
                br["season"],
                br["week"],
                br["target"],
                br["model_prob"],
                br["actual"],
                br["probability_clv"],
                br["has_closing_odds"],
            ],
        )

    # Feature importances (15 per target, D-02).
    for fi in _insights_feature_importance_rows():
        conn.execute(
            "INSERT OR REPLACE INTO feature_importances VALUES (?, ?, ?, ?)",
            [fi["game_id"], fi["target"], fi["feature_name"], fi["importance"]],
        )

    # Per-season backtest_metrics for all three targets (2021-2024).
    insight_metrics: list[tuple[int, str, str, float]] = []
    for season in (2021, 2022, 2023, 2024):
        insight_metrics.extend(
            [
                (season, "wp", "accuracy", 0.62 + 0.01 * (season - 2021)),
                (season, "wp", "brier_score", 0.22 - 0.005 * (season - 2021)),
                (season, "wp", "n_games", 256.0),
                (season, "wp", "n_predictions", 256.0),
                (season, "ats", "mae", 6.5 - 0.1 * (season - 2021)),
                (season, "ats", "rmse", 8.1 - 0.05 * (season - 2021)),
                (season, "ats", "r2", 0.12 + 0.01 * (season - 2021)),
                (season, "ats", "n_games", 256.0),
                (season, "ats", "n_predictions", 256.0),
                (season, "ou", "mae", 10.8 - 0.2 * (season - 2021)),
                (season, "ou", "rmse", 13.4 - 0.1 * (season - 2021)),
                (season, "ou", "r2", 0.09 + 0.005 * (season - 2021)),
                (season, "ou", "n_games", 256.0),
                (season, "ou", "n_predictions", 256.0),
            ],
        )
    conn.executemany(
        "INSERT OR REPLACE INTO backtest_metrics VALUES (?, ?, ?, ?)",
        insight_metrics,
    )

    # Market rows — insert into predictions table so model-vs-market join has
    # data. Some game_ids already exist from _sample_game_data(); use INSERT OR
    # REPLACE to avoid PK conflicts.
    for mr in _insights_market_rows():
        # Strip the annotation-only key before binding.
        mr_clean = {k: v for k, v in mr.items() if k != "_note"}
        conn.execute(
            """
            INSERT OR REPLACE INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            [
                mr_clean["game_id"],
                mr_clean["season"],
                mr_clean["week"],
                mr_clean["game_date"],
                mr_clean["home_team"],
                mr_clean["away_team"],
                mr_clean["status"],
                mr_clean["home_score"],
                mr_clean["away_score"],
                mr_clean["wp_prob"],
                mr_clean["wp_confidence"],
                mr_clean["ats_prediction"],
                mr_clean["ats_confidence"],
                mr_clean["ou_prediction"],
                mr_clean["ou_confidence"],
                mr_clean["market_spread"],
                mr_clean["market_total"],
                mr_clean["market_ml_home"],
                mr_clean["market_ml_away"],
                mr_clean["wp_edge"],
                mr_clean["ats_edge"],
                mr_clean["ou_edge"],
                mr_clean["blended_wp"],
                mr_clean["blended_ats"],
                mr_clean["blended_ou"],
            ],
        )

    # Insert game context for one game (game detail page tests)
    game_context_rows = [
        (
            "2024_W01_BUF@KC",  # game_id
            1550.0,  # home_elo
            1520.0,  # away_elo
            '["W","W","L","W","W"]',  # home_last5
            '["W","L","W","W","L"]',  # away_last5
            '{"home_wins": 3, "away_wins": 2}',  # h2h_record
            "GEHA Field at Arrowhead Stadium",  # venue_name
            "Grass",  # surface
            "outdoors",  # roof_type
            2.5,  # weather_severity
            12.0,  # wind_mph
            True,  # is_outdoor
            True,  # is_divisional
            True,  # is_primetime
        ),
    ]
    conn.executemany(
        "INSERT INTO game_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        game_context_rows,
    )

    # Insert cache metadata
    now = datetime.now(tz=UTC)
    conn.executemany(
        "INSERT INTO cache_meta VALUES (?, ?, ?)",
        [
            ("last_updated", now.isoformat(), now),
            ("prediction_count", "3", now),
            ("season_range", "2023-2024", now),
        ],
    )

    conn.close()
    return db_path


@pytest.fixture()
def empty_test_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB with CACHE_SCHEMA but no data.

    Returns the path to the empty database file.
    """
    db_path = tmp_path / "empty_cache.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create schema only, no data
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)

    conn.close()
    return db_path


@pytest.fixture()
def test_client(test_db: Path) -> Iterator[TestClient]:
    """Create a FastAPI TestClient with the test database.

    Uses FastAPI ``app.dependency_overrides`` to inject a test DuckDB
    connection via :func:`api.dependencies.get_db`. This is the canonical
    pattern for unit-testing FastAPI dependencies and replaces the older
    ``deps.DB_PATH`` mutation approach.
    """
    import threading

    from api.dependencies import get_db
    from api.main import app

    test_conn = duckdb.connect(str(test_db), read_only=True)

    def _override_get_db():
        return test_conn

    # Ensure app.state has the lock even when lifespan has not run yet.
    # TestClient triggers lifespan, but dependency overrides bypass the
    # reconnect path inside get_db anyway -- the lock is here so any code
    # that touches app.state.db_lock (e.g. health endpoint) does not break.
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = _override_get_db

    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        test_conn.close()


@pytest.fixture()
def empty_test_client(empty_test_db: Path) -> Iterator[TestClient]:
    """Create a FastAPI TestClient backed by an empty database.

    Useful for testing empty-state UI rendering.
    """
    import threading

    from api.dependencies import get_db
    from api.main import app

    test_conn = duckdb.connect(str(empty_test_db), read_only=True)

    def _override_get_db():
        return test_conn

    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = _override_get_db

    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        test_conn.close()
