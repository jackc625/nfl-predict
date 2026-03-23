"""Tests for era normalization utilities (BACK-06).

Validates 16-vs-17 game week normalization and COVID annotation.
"""

import pytest

from backtest.era import (
    ERA_TRANSITION_SEASON,
    get_covid_hfa_annotation,
    get_season_total_weeks,
    normalize_week_to_progress,
)


class TestGetSeasonTotalWeeks:
    """Tests for get_season_total_weeks()."""

    def test_pre_2021_total_weeks_2018(self) -> None:
        assert get_season_total_weeks(2018) == 17

    def test_pre_2021_total_weeks_2020(self) -> None:
        assert get_season_total_weeks(2020) == 17

    def test_post_2021_total_weeks_2021(self) -> None:
        assert get_season_total_weeks(2021) == 18

    def test_post_2021_total_weeks_2024(self) -> None:
        assert get_season_total_weeks(2024) == 18


class TestNormalizeWeekToProgress:
    """Tests for normalize_week_to_progress()."""

    def test_normalize_week_pre_2021(self) -> None:
        assert normalize_week_to_progress(12, 2020) == pytest.approx(12 / 17)

    def test_normalize_week_post_2021(self) -> None:
        assert normalize_week_to_progress(12, 2022) == pytest.approx(12 / 18)

    def test_week_1_always_positive(self) -> None:
        assert normalize_week_to_progress(1, 2020) > 0

    def test_week_1_pre_2021(self) -> None:
        assert normalize_week_to_progress(1, 2019) == pytest.approx(1 / 17)

    def test_week_1_post_2021(self) -> None:
        assert normalize_week_to_progress(1, 2023) == pytest.approx(1 / 18)

    def test_final_week_pre_2021(self) -> None:
        """Week 17 in a 17-week season should be 1.0."""
        assert normalize_week_to_progress(17, 2020) == pytest.approx(1.0)

    def test_final_week_post_2021(self) -> None:
        """Week 18 in an 18-week season should be 1.0."""
        assert normalize_week_to_progress(18, 2022) == pytest.approx(1.0)


class TestEraTransitionSeason:
    """Tests for ERA_TRANSITION_SEASON constant."""

    def test_era_transition_is_2021(self) -> None:
        assert ERA_TRANSITION_SEASON == 2021


class TestCovidAnnotation:
    """Tests for get_covid_hfa_annotation()."""

    def test_covid_annotation_has_required_fields(self) -> None:
        ann = get_covid_hfa_annotation()
        assert "home_win_pct" in ann
        assert "seasons" in ann
        assert "normal_home_win_pct" in ann
        assert "note" in ann
        assert "impact_on_model" in ann

    def test_covid_annotation_seasons(self) -> None:
        ann = get_covid_hfa_annotation()
        assert ann["seasons"] == [2020]

    def test_covid_annotation_home_win_pct(self) -> None:
        ann = get_covid_hfa_annotation()
        assert ann["home_win_pct"] == pytest.approx(0.496, abs=0.01)

    def test_covid_annotation_normal_pct(self) -> None:
        ann = get_covid_hfa_annotation()
        assert ann["normal_home_win_pct"] == pytest.approx(0.57, abs=0.02)
