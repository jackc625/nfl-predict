"""Unit tests for the blend's data module (models/blending_data.py).

WHAT MOVED, AND WHY (Plan 33.2-24 Task 1)
-----------------------------------------
This module used to test ``load_tuning_period_data`` as a LIVE nflverse schedule fetch that
tuned the blend on 2010-2017 CLOSING lines: column mapping from ``spread_line`` /
``total_line`` / the moneylines, a regular-season filter, NaN dropping, team-abbreviation
normalization and the pandas return type. That loader is RETIRED by D33.2-03 / D33.2-10 --
a closing line did not exist at a game's lock -- and so are those nine tests, each for a
stated reason rather than silently:

* column mapping / moneyline mapping / abbreviation normalization / pandas type: the new
  loader reads the owned ``odds_timeline``, which has no moneyline and already stores
  canonical game ids; its columns are pinned by
  ``tests/integration/test_blend_tuning_prelock_only.py``;
* NaN dropping: carried forward as "no missing line is ever filled" -- a game with no
  pre-lock line is EXCLUDED and returned, pinned there by the prohibition test;
* the regular-season filter: deliberately NOT carried forward. The owned corpus keeps the
  six postseason games that have a pre-lock line; the models predict postseason games too,
  and dropping them would discard honest evidence;
* ``TUNING_SEASONS`` / ``get_tuning_period_games``: the 2010-2017 window and its
  game-info wrapper have no reader left.

The noise-profile tests below are retired with the dynamic blend in Task 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers: Synthetic parquet data for noise profile tests
# ---------------------------------------------------------------------------


def _write_synthetic_parquets(
    baselines_dir: Path,
    n_games_per_week: int = 16,
    weeks: list[int] | None = None,
    season: int = 2022,
    wp_error_mean: float = 0.0,
    wp_error_std: float = 0.10,
    ats_error_mean: float = 0.0,
    ats_error_std: float = 3.0,
    ou_error_mean: float = 0.0,
    ou_error_std: float = 3.0,
    rng_seed: int = 42,
) -> None:
    """Write synthetic prediction parquets mimicking baselines structure.

    Creates predictions_wp.parquet, predictions_ats.parquet, predictions_ou.parquet
    in baselines_dir with known error distributions for testing.
    """
    if weeks is None:
        weeks = list(range(1, 19))

    rng = np.random.default_rng(rng_seed)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    rows_wp = []
    rows_ats = []
    rows_ou = []

    teams = [
        "ATL",
        "PHI",
        "DAL",
        "NYG",
        "TB",
        "NO",
        "CAR",
        "SF",
        "SEA",
        "LA",
        "ARI",
        "GB",
        "MIN",
        "CHI",
        "DET",
        "WAS",
        "KC",
        "LV",
        "DEN",
        "LAC",
        "BUF",
        "MIA",
        "NE",
        "NYJ",
        "BAL",
        "PIT",
        "CLE",
        "CIN",
        "TEN",
        "HOU",
        "IND",
        "JAX",
    ]

    for week in weeks:
        for i in range(n_games_per_week):
            away = teams[(i * 2) % len(teams)]
            home = teams[(i * 2 + 1) % len(teams)]
            game_id = f"{season}_W{week:02d}_{away}@{home}"

            # WP
            fair_prob = rng.uniform(0.30, 0.70)
            error = rng.normal(wp_error_mean, wp_error_std)
            rows_wp.append(
                {
                    "game_id": game_id,
                    "model_prob": np.clip(fair_prob + error, 0.01, 0.99),
                    "fair_closing_prob": fair_prob,
                }
            )

            # ATS
            closing_spread = rng.normal(-2.5, 5.0)
            ats_error = rng.normal(ats_error_mean, ats_error_std)
            rows_ats.append(
                {
                    "game_id": game_id,
                    "model_spread": closing_spread + ats_error,
                    "spread": closing_spread,
                }
            )

            # O/U
            closing_total = rng.normal(45.0, 4.0)
            ou_error = rng.normal(ou_error_mean, ou_error_std)
            rows_ou.append(
                {
                    "game_id": game_id,
                    "model_total": closing_total + ou_error,
                    "total": closing_total,
                }
            )

    pd.DataFrame(rows_wp).to_parquet(baselines_dir / "predictions_wp.parquet")
    pd.DataFrame(rows_ats).to_parquet(baselines_dir / "predictions_ats.parquet")
    pd.DataFrame(rows_ou).to_parquet(baselines_dir / "predictions_ou.parquet")


# ---------------------------------------------------------------------------
# Test class: Noise profile extraction
# ---------------------------------------------------------------------------


class TestNoiseProfile:
    """Tests for extract_noise_profile function."""

    def test_extract_noise_profile_returns_all_targets(self, tmp_path: Path) -> None:
        """extract_noise_profile returns dict with keys 'wp', 'ats', 'ou'."""
        _write_synthetic_parquets(
            tmp_path, n_games_per_week=16, weeks=list(range(1, 19))
        )

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=tmp_path)

        assert set(result.keys()) == {"wp", "ats", "ou"}
        for target in ("wp", "ats", "ou"):
            assert isinstance(result[target], pd.DataFrame)
            expected_cols = {"week", "mean", "std", "count"}
            assert expected_cols.issubset(set(result[target].columns))

    def test_noise_profile_filters_playoff_weeks(self, tmp_path: Path) -> None:
        """Playoff weeks (> 18 for post-2020 era) are filtered out."""
        # Include regular and playoff weeks
        _write_synthetic_parquets(
            tmp_path,
            n_games_per_week=16,
            weeks=[1, 2, 18, 19, 20],
            season=2022,
        )

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=tmp_path)

        for target in ("wp", "ats", "ou"):
            weeks_in_result = result[target]["week"].tolist()
            assert 19 not in weeks_in_result
            assert 20 not in weeks_in_result
            # Regular season weeks should be present
            assert 1 in weeks_in_result
            assert 2 in weeks_in_result
            assert 18 in weeks_in_result

    def test_noise_profile_correct_error_computation(self, tmp_path: Path) -> None:
        """WP error is model_prob - fair_closing_prob (verified with known values)."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        # Create known data: exactly one game per week
        wp_df = pd.DataFrame(
            [
                {
                    "game_id": "2022_W01_ATL@PHI",
                    "model_prob": 0.70,
                    "fair_closing_prob": 0.60,
                },
                {
                    "game_id": "2022_W02_DAL@NYG",
                    "model_prob": 0.50,
                    "fair_closing_prob": 0.55,
                },
            ]
        )
        ats_df = pd.DataFrame(
            [
                {"game_id": "2022_W01_ATL@PHI", "model_spread": -3.0, "spread": -5.0},
                {"game_id": "2022_W02_DAL@NYG", "model_spread": 2.0, "spread": 1.0},
            ]
        )
        ou_df = pd.DataFrame(
            [
                {"game_id": "2022_W01_ATL@PHI", "model_total": 48.0, "total": 45.0},
                {"game_id": "2022_W02_DAL@NYG", "model_total": 42.0, "total": 44.0},
            ]
        )
        wp_df.to_parquet(baselines_dir / "predictions_wp.parquet")
        ats_df.to_parquet(baselines_dir / "predictions_ats.parquet")
        ou_df.to_parquet(baselines_dir / "predictions_ou.parquet")

        from models.blending_data import extract_noise_profile

        # With only 1 game per week, sparse blending will apply, but mean
        # should still reflect the actual errors blended with overall.
        # Week 1 WP error = 0.70 - 0.60 = 0.10
        # Week 2 WP error = 0.50 - 0.55 = -0.05
        # Season mean = (0.10 + -0.05) / 2 = 0.025
        # With 1 game and min_sample_count=30: blend_weight = 1/30
        # Week 1 blended mean = (1/30)*0.10 + (29/30)*0.025 = ~0.0275
        result = extract_noise_profile(baselines_dir=baselines_dir, min_sample_count=30)

        # Verify WP: week 1 raw error = 0.10, blended toward season mean 0.025
        wp_week1 = result["wp"][result["wp"]["week"] == 1].iloc[0]
        blend_w = 1.0 / 30.0
        expected_mean = blend_w * 0.10 + (1 - blend_w) * 0.025
        assert wp_week1["mean"] == pytest.approx(expected_mean, abs=1e-6)

        # Verify ATS: week 1 raw error = -3.0 - (-5.0) = 2.0
        # Week 2 raw error = 2.0 - 1.0 = 1.0
        # Season mean = (2.0 + 1.0) / 2 = 1.5
        ats_week1 = result["ats"][result["ats"]["week"] == 1].iloc[0]
        expected_ats_mean = blend_w * 2.0 + (1 - blend_w) * 1.5
        assert ats_week1["mean"] == pytest.approx(expected_ats_mean, abs=1e-6)

        # Verify O/U: week 1 raw error = 48.0 - 45.0 = 3.0
        # Week 2 raw error = 42.0 - 44.0 = -2.0
        # Season mean = (3.0 + -2.0) / 2 = 0.5
        ou_week1 = result["ou"][result["ou"]["week"] == 1].iloc[0]
        expected_ou_mean = blend_w * 3.0 + (1 - blend_w) * 0.5
        assert ou_week1["mean"] == pytest.approx(expected_ou_mean, abs=1e-6)

    def test_noise_profile_missing_baselines_dir(self) -> None:
        """Raises FileNotFoundError with descriptive message if baselines_dir does not exist."""
        from models.blending_data import extract_noise_profile

        with pytest.raises(FileNotFoundError, match=r"(?i)baselines.*not found"):
            extract_noise_profile(baselines_dir=Path("/nonexistent/baselines"))

    def test_noise_profile_missing_parquet(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if predictions_wp.parquet missing."""
        baselines_dir = tmp_path / "empty_baselines"
        baselines_dir.mkdir()

        from models.blending_data import extract_noise_profile

        with pytest.raises(FileNotFoundError, match=r"predictions_wp\.parquet"):
            extract_noise_profile(baselines_dir=baselines_dir)

    def test_noise_profile_per_week_stats(self, tmp_path: Path) -> None:
        """Per-week stats are correct for known input with enough samples."""
        # Create 50 games in week 1 with known error distribution
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rng = np.random.default_rng(123)
        n_games = 50
        known_errors = rng.normal(0.05, 0.08, size=n_games)

        wp_rows = []
        ats_rows = []
        ou_rows = []
        teams = [
            "ATL",
            "PHI",
            "DAL",
            "NYG",
            "TB",
            "NO",
            "CAR",
            "SF",
            "SEA",
            "LA",
            "ARI",
            "GB",
            "KC",
            "LV",
            "DEN",
            "LAC",
            "BUF",
            "MIA",
            "NE",
            "NYJ",
            "BAL",
            "PIT",
            "CLE",
            "CIN",
            "TEN",
            "HOU",
            "IND",
            "JAX",
            "MIN",
            "CHI",
            "DET",
            "WAS",
            "GB",
            "MIN",
            "CHI",
            "DET",
            "WAS",
            "NYG",
            "PHI",
            "DAL",
            "TB",
            "NO",
            "CAR",
            "SF",
            "SEA",
            "LA",
            "ARI",
            "KC",
            "LV",
            "DEN",
        ]

        for i in range(n_games):
            away = teams[i % len(teams)]
            home = teams[(i + 1) % len(teams)]
            game_id = f"2022_W01_{away}@{home}_{i:03d}"
            fair_prob = 0.50
            wp_rows.append(
                {
                    "game_id": game_id,
                    "model_prob": fair_prob + known_errors[i],
                    "fair_closing_prob": fair_prob,
                }
            )
            ats_rows.append(
                {
                    "game_id": game_id,
                    "model_spread": -3.0 + known_errors[i] * 30,
                    "spread": -3.0,
                }
            )
            ou_rows.append(
                {
                    "game_id": game_id,
                    "model_total": 45.0 + known_errors[i] * 30,
                    "total": 45.0,
                }
            )

        pd.DataFrame(wp_rows).to_parquet(baselines_dir / "predictions_wp.parquet")
        pd.DataFrame(ats_rows).to_parquet(baselines_dir / "predictions_ats.parquet")
        pd.DataFrame(ou_rows).to_parquet(baselines_dir / "predictions_ou.parquet")

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=baselines_dir)

        wp_week1 = result["wp"][result["wp"]["week"] == 1].iloc[0]
        expected_mean = float(np.mean(known_errors))
        expected_std = float(np.std(known_errors, ddof=1))

        # 50 games > min_sample_count of 30, so no blending applied
        assert wp_week1["mean"] == pytest.approx(expected_mean, abs=1e-4)
        assert wp_week1["std"] == pytest.approx(expected_std, abs=1e-4)
        assert wp_week1["count"] == 50

    def test_noise_profile_sparse_week_blends_to_season(self, tmp_path: Path) -> None:
        """Weeks with < 30 games blend stats toward season-wide mean/std."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rng = np.random.default_rng(99)

        # 100 games across weeks 1-10 (10 per week, below threshold individually)
        # Plus 5 games in week 18 (sparse -- should be blended)
        wp_rows = []
        ats_rows = []
        ou_rows = []

        all_errors = []

        # Weeks 1-10: 10 games each
        for week in range(1, 11):
            for i in range(10):
                game_id = f"2022_W{week:02d}_T{i}@T{i + 16}"
                error = rng.normal(0.02, 0.10)
                all_errors.append(error)
                wp_rows.append(
                    {
                        "game_id": game_id,
                        "model_prob": 0.50 + error,
                        "fair_closing_prob": 0.50,
                    }
                )
                ats_rows.append(
                    {
                        "game_id": game_id,
                        "model_spread": -3.0 + error * 30,
                        "spread": -3.0,
                    }
                )
                ou_rows.append(
                    {
                        "game_id": game_id,
                        "model_total": 45.0 + error * 30,
                        "total": 45.0,
                    }
                )

        # Week 18: 5 games (sparse)
        week18_errors = []
        for i in range(5):
            game_id = f"2022_W18_T{i}@T{i + 16}"
            error = rng.normal(0.15, 0.05)
            all_errors.append(error)
            week18_errors.append(error)
            wp_rows.append(
                {
                    "game_id": game_id,
                    "model_prob": 0.50 + error,
                    "fair_closing_prob": 0.50,
                }
            )
            ats_rows.append(
                {
                    "game_id": game_id,
                    "model_spread": -3.0 + error * 30,
                    "spread": -3.0,
                }
            )
            ou_rows.append(
                {
                    "game_id": game_id,
                    "model_total": 45.0 + error * 30,
                    "total": 45.0,
                }
            )

        pd.DataFrame(wp_rows).to_parquet(baselines_dir / "predictions_wp.parquet")
        pd.DataFrame(ats_rows).to_parquet(baselines_dir / "predictions_ats.parquet")
        pd.DataFrame(ou_rows).to_parquet(baselines_dir / "predictions_ou.parquet")

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=baselines_dir, min_sample_count=30)

        wp_stats = result["wp"]
        week18_row = wp_stats[wp_stats["week"] == 18].iloc[0]

        # Week 18 has 5 games (< 30), should be blended with weight 5/30
        season_mean = float(np.mean(all_errors))
        week18_raw_mean = float(np.mean(week18_errors))
        blend_w = 5.0 / 30.0
        expected_blended_mean = blend_w * week18_raw_mean + (1 - blend_w) * season_mean

        assert week18_row["mean"] == pytest.approx(expected_blended_mean, abs=1e-4)
        assert week18_row["count"] == 5

        # Weeks 1-10 also have < 30 games (10 each), so they should also be blended
        week1_row = wp_stats[wp_stats["week"] == 1].iloc[0]
        assert week1_row["count"] == 10
        # Blended since 10 < 30

    def test_noise_profile_configurable_path(self, tmp_path: Path) -> None:
        """extract_noise_profile works with custom baselines_dir (not hardcoded to v2.0)."""
        custom_dir = tmp_path / "custom_baselines" / "my_version"
        _write_synthetic_parquets(custom_dir, n_games_per_week=16, weeks=[1, 2, 3])

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=custom_dir)

        assert set(result.keys()) == {"wp", "ats", "ou"}
        for target in ("wp", "ats", "ou"):
            assert len(result[target]) > 0

    def test_noise_profile_game_id_format_variants(self, tmp_path: Path) -> None:
        """Both W01 (zero-padded) and W1 (unpadded) game_id formats extract correctly."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        # Mix of zero-padded and unpadded week formats
        wp_rows = [
            {
                "game_id": "2022_W01_ATL@PHI",
                "model_prob": 0.60,
                "fair_closing_prob": 0.50,
            },
            {
                "game_id": "2022_W1_DAL@NYG",
                "model_prob": 0.65,
                "fair_closing_prob": 0.55,
            },
            {
                "game_id": "2022_W10_KC@BUF",
                "model_prob": 0.55,
                "fair_closing_prob": 0.50,
            },
        ]
        ats_rows = [
            {"game_id": "2022_W01_ATL@PHI", "model_spread": -3.0, "spread": -5.0},
            {"game_id": "2022_W1_DAL@NYG", "model_spread": 2.0, "spread": 1.0},
            {"game_id": "2022_W10_KC@BUF", "model_spread": -1.0, "spread": -2.0},
        ]
        ou_rows = [
            {"game_id": "2022_W01_ATL@PHI", "model_total": 48.0, "total": 45.0},
            {"game_id": "2022_W1_DAL@NYG", "model_total": 42.0, "total": 44.0},
            {"game_id": "2022_W10_KC@BUF", "model_total": 46.0, "total": 45.0},
        ]

        pd.DataFrame(wp_rows).to_parquet(baselines_dir / "predictions_wp.parquet")
        pd.DataFrame(ats_rows).to_parquet(baselines_dir / "predictions_ats.parquet")
        pd.DataFrame(ou_rows).to_parquet(baselines_dir / "predictions_ou.parquet")

        from models.blending_data import extract_noise_profile

        result = extract_noise_profile(baselines_dir=baselines_dir)

        # W01 and W1 should both extract to week 1
        wp_stats = result["wp"]
        week1_row = wp_stats[wp_stats["week"] == 1]
        assert len(week1_row) == 1  # Both W01 and W1 merged into week 1
        assert week1_row.iloc[0]["count"] == 2  # Two games in week 1

        week10_row = wp_stats[wp_stats["week"] == 10]
        assert len(week10_row) == 1
        assert week10_row.iloc[0]["count"] == 1
