"""Integration tests for the baseline comparison and gating logic (Gap 12-04-01, ACCU-06).

Verifies:
1. load_baseline_metrics reads metrics_*.json from a directory
2. _gate_wp correctly identifies PASS/FAIL for WP metric deltas
3. _gate_regression_target correctly identifies PASS/FAIL for ATS/O/U deltas
4. gate_targets returns per-target results with the expected structure
5. fill_comparison_template produces valid markdown with v2.0 values and deltas

Does NOT run any actual backtest. All metric dicts are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.retrain_models import (
    _extract_avg_metric,
    _format_delta,
    _gate_regression_target,
    _gate_wp,
    fill_comparison_template,
    gate_targets,
    load_baseline_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(directory: Path, target: str, metrics: dict) -> None:
    """Write a metrics JSON file for the given target into directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"metrics_{target}.json"
    path.write_text(json.dumps(metrics))


def _make_wp_metrics(
    headline_clv: float,
    accuracy: float = 0.55,
    n_seasons: int = 2,
) -> dict:
    """Synthetic WP metrics dict mimicking the real baseline format."""
    per_season = [
        {
            "season": 2021 + i,
            "metrics": {
                "accuracy": accuracy,
                "mae": 0.45,
            },
        }
        for i in range(n_seasons)
    ]
    return {
        "headline_clv": headline_clv,
        "per_season_results": per_season,
    }


def _make_regression_metrics(
    headline_clv: float,
    mae: float,
    n_seasons: int = 2,
) -> dict:
    """Synthetic ATS/O/U metrics dict mimicking the real baseline format."""
    per_season = [
        {
            "season": 2021 + i,
            "metrics": {
                "mae": mae,
                "rmse": mae * 1.3,
                "r2": 0.05,
            },
        }
        for i in range(n_seasons)
    ]
    return {
        "headline_clv": headline_clv,
        "per_season_results": per_season,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v1_dir(tmp_path):
    """v1.0 baseline directory with all three metrics files."""
    d = tmp_path / "v1.0"
    _write_metrics(d, "wp", _make_wp_metrics(headline_clv=-0.0164, accuracy=0.57))
    _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.8701, mae=9.5))
    _write_metrics(d, "ou", _make_regression_metrics(headline_clv=45.77, mae=10.2))
    return d


@pytest.fixture
def v2_better_dir(tmp_path):
    """v2.0 baseline where all targets improved (should all PASS gating)."""
    d = tmp_path / "v2_better"
    _write_metrics(d, "wp", _make_wp_metrics(headline_clv=-0.0100, accuracy=0.58))
    _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.9000, mae=9.3))
    _write_metrics(d, "ou", _make_regression_metrics(headline_clv=46.00, mae=10.0))
    return d


@pytest.fixture
def v2_worse_dir(tmp_path):
    """v2.0 baseline where all targets regressed (should all FAIL gating)."""
    d = tmp_path / "v2_worse"
    _write_metrics(d, "wp", _make_wp_metrics(headline_clv=-0.0400, accuracy=0.53))
    _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.7000, mae=9.8))
    _write_metrics(d, "ou", _make_regression_metrics(headline_clv=44.00, mae=10.5))
    return d


# ---------------------------------------------------------------------------
# load_baseline_metrics
# ---------------------------------------------------------------------------


class TestLoadBaselineMetrics:
    """load_baseline_metrics reads all three targets from JSON files."""

    def test_loads_all_three_targets(self, v1_dir):
        """load_baseline_metrics returns dict with wp, ats, ou keys."""
        metrics = load_baseline_metrics(v1_dir)

        assert "wp" in metrics
        assert "ats" in metrics
        assert "ou" in metrics

    def test_loaded_values_match_written_values(self, v1_dir):
        """Loaded headline_clv matches what was written to disk."""
        metrics = load_baseline_metrics(v1_dir)

        assert abs(metrics["wp"]["headline_clv"] - (-0.0164)) < 1e-6
        assert abs(metrics["ats"]["headline_clv"] - 0.8701) < 1e-4
        assert abs(metrics["ou"]["headline_clv"] - 45.77) < 1e-2

    def test_raises_on_missing_file(self, tmp_path):
        """load_baseline_metrics raises FileNotFoundError when a file is missing."""
        d = tmp_path / "incomplete"
        d.mkdir()
        # Only write wp and ats, not ou
        _write_metrics(d, "wp", _make_wp_metrics(headline_clv=0.0))
        _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.0, mae=10.0))

        with pytest.raises(FileNotFoundError):
            load_baseline_metrics(d)


# ---------------------------------------------------------------------------
# _gate_wp
# ---------------------------------------------------------------------------


class TestGateWP:
    """_gate_wp: CLV regression and accuracy drop gating."""

    def test_wp_passes_when_clv_improves(self):
        """WP gates as PASS when CLV improves (delta > 0)."""
        v1 = _make_wp_metrics(headline_clv=-0.0164, accuracy=0.56)
        v2 = _make_wp_metrics(headline_clv=-0.0100, accuracy=0.57)

        result = _gate_wp(v1, v2)

        assert result["passed"] is True

    def test_wp_passes_when_clv_small_regression(self):
        """WP gates as PASS when CLV regresses within the 0.005 threshold."""
        v1 = _make_wp_metrics(headline_clv=-0.0164, accuracy=0.56)
        # delta = -0.003 (within threshold of 0.005)
        v2 = _make_wp_metrics(headline_clv=-0.0194, accuracy=0.56)

        result = _gate_wp(v1, v2)

        assert result["passed"] is True

    def test_wp_fails_when_clv_regresses_beyond_threshold(self):
        """WP gates as FAIL when CLV regresses by more than 0.005."""
        v1 = _make_wp_metrics(headline_clv=-0.0164, accuracy=0.56)
        # delta = -0.010 (beyond threshold)
        v2 = _make_wp_metrics(headline_clv=-0.0264, accuracy=0.56)

        result = _gate_wp(v1, v2)

        assert result["passed"] is False

    def test_wp_fails_when_accuracy_drops_more_than_1pct(self):
        """WP gates as FAIL when accuracy drops by more than 1%."""
        v1 = _make_wp_metrics(headline_clv=-0.0164, accuracy=0.57)
        # accuracy drops by 0.02 (> 0.01 threshold)
        v2 = _make_wp_metrics(headline_clv=-0.0164, accuracy=0.55)

        result = _gate_wp(v1, v2)

        assert result["passed"] is False

    def test_wp_result_contains_expected_keys(self):
        """_gate_wp result contains passed, reasons, v1_metrics, v2_metrics."""
        v1 = _make_wp_metrics(headline_clv=-0.01, accuracy=0.55)
        v2 = _make_wp_metrics(headline_clv=-0.01, accuracy=0.55)

        result = _gate_wp(v1, v2)

        assert "passed" in result
        assert "reasons" in result
        assert "v1_metrics" in result
        assert "v2_metrics" in result
        assert isinstance(result["reasons"], list)
        assert len(result["reasons"]) > 0


# ---------------------------------------------------------------------------
# _gate_regression_target
# ---------------------------------------------------------------------------


class TestGateRegressionTarget:
    """_gate_regression_target: MAE and CLV gating for ATS/O/U."""

    def test_ats_passes_when_clv_and_mae_improve(self):
        """ATS passes when CLV increases and MAE decreases."""
        v1 = _make_regression_metrics(headline_clv=0.87, mae=9.5)
        v2 = _make_regression_metrics(headline_clv=0.90, mae=9.3)

        result = _gate_regression_target(v1, v2, "ats")

        assert result["passed"] is True

    def test_ats_fails_when_clv_decreases(self):
        """ATS fails when CLV decreases."""
        v1 = _make_regression_metrics(headline_clv=0.87, mae=9.5)
        v2 = _make_regression_metrics(headline_clv=0.70, mae=9.3)

        result = _gate_regression_target(v1, v2, "ats")

        assert result["passed"] is False

    def test_ats_fails_when_mae_increases(self):
        """ATS fails when MAE increases (worse performance)."""
        v1 = _make_regression_metrics(headline_clv=0.87, mae=9.5)
        v2 = _make_regression_metrics(headline_clv=0.90, mae=9.8)

        result = _gate_regression_target(v1, v2, "ats")

        assert result["passed"] is False

    def test_ou_passes_when_clv_and_mae_improve(self):
        """O/U passes when CLV increases and MAE decreases."""
        v1 = _make_regression_metrics(headline_clv=45.0, mae=10.2)
        v2 = _make_regression_metrics(headline_clv=46.0, mae=10.0)

        result = _gate_regression_target(v1, v2, "ou")

        assert result["passed"] is True

    def test_regression_result_contains_expected_keys(self):
        """_gate_regression_target result contains passed, reasons, v1_metrics, v2_metrics."""
        v1 = _make_regression_metrics(headline_clv=0.87, mae=9.5)
        v2 = _make_regression_metrics(headline_clv=0.87, mae=9.5)

        result = _gate_regression_target(v1, v2, "ats")

        assert "passed" in result
        assert "reasons" in result
        assert "v1_metrics" in result
        assert "v2_metrics" in result


# ---------------------------------------------------------------------------
# Tests: gate_targets (full three-target gating)
# ---------------------------------------------------------------------------


class TestGateTargets:
    """gate_targets: full three-target gating logic with file I/O."""

    def test_all_pass_when_v2_improves(self, v1_dir, v2_better_dir):
        """All three targets PASS when v2.0 metrics improve over v1.0."""
        results = gate_targets(v1_dir, v2_better_dir)

        assert results["wp"]["passed"] is True
        assert results["ats"]["passed"] is True
        assert results["ou"]["passed"] is True

    def test_all_fail_when_v2_regresses(self, v1_dir, v2_worse_dir):
        """All three targets FAIL when v2.0 metrics regress vs v1.0."""
        results = gate_targets(v1_dir, v2_worse_dir)

        assert results["wp"]["passed"] is False
        assert results["ats"]["passed"] is False
        assert results["ou"]["passed"] is False

    def test_gate_targets_returns_three_targets(self, v1_dir, v2_better_dir):
        """gate_targets returns a dict with exactly the wp/ats/ou keys."""
        results = gate_targets(v1_dir, v2_better_dir)

        assert set(results.keys()) == {"wp", "ats", "ou"}

    def test_per_target_gating_is_independent(self, tmp_path, v1_dir):
        """A target that improves PASSes even when another FAILs."""
        # wp improves, ats/ou regress
        v2_mixed = tmp_path / "v2_mixed"
        _write_metrics(
            v2_mixed, "wp", _make_wp_metrics(headline_clv=-0.005, accuracy=0.60)
        )
        _write_metrics(
            v2_mixed, "ats", _make_regression_metrics(headline_clv=0.60, mae=10.0)
        )
        _write_metrics(
            v2_mixed, "ou", _make_regression_metrics(headline_clv=44.00, mae=11.0)
        )

        results = gate_targets(v1_dir, v2_mixed)

        assert results["wp"]["passed"] is True
        assert results["ats"]["passed"] is False
        assert results["ou"]["passed"] is False


# ---------------------------------------------------------------------------
# fill_comparison_template
# ---------------------------------------------------------------------------


class TestFillComparisonTemplate:
    """fill_comparison_template produces valid markdown with v2.0 numbers."""

    def test_template_contains_headline_clv_row(self, v1_dir, v2_better_dir):
        """The filled template markdown contains a Headline CLV row."""
        md = fill_comparison_template(v1_dir, v2_better_dir)

        assert "Headline CLV" in md

    def test_template_contains_v2_metric_values(self, v1_dir, v2_better_dir):
        """The filled template contains actual v2.0 metric values (not '---')."""
        md = fill_comparison_template(v1_dir, v2_better_dir)

        # v2.0 WP CLV is -0.0100; string representation should appear
        assert "-0.01" in md

    def test_template_is_markdown_table(self, v1_dir, v2_better_dir):
        """The output starts with a markdown heading and contains table pipes."""
        md = fill_comparison_template(v1_dir, v2_better_dir)

        assert md.startswith("#")
        assert "|" in md

    def test_template_contains_delta_column(self, v1_dir, v2_better_dir):
        """The filled template contains at least one formatted delta value."""
        md = fill_comparison_template(v1_dir, v2_better_dir)

        # Deltas are formatted with a sign prefix
        has_positive_delta = "+" in md
        has_negative_delta = "-0." in md
        assert has_positive_delta or has_negative_delta


class TestHelpers:
    """Unit tests for _format_delta and _extract_avg_metric."""

    def test_format_delta_positive(self):
        """_format_delta returns '+' prefixed string for positive delta."""
        result = _format_delta(v2_val=1.0, v1_val=0.9)
        assert result.startswith("+")

    def test_format_delta_negative(self):
        """_format_delta returns '-' prefixed string for negative delta."""
        result = _format_delta(v2_val=0.9, v1_val=1.0)
        assert result.startswith("-")

    def test_format_delta_zero(self):
        """_format_delta returns '+' prefixed for zero delta."""
        result = _format_delta(v2_val=1.0, v1_val=1.0)
        assert result.startswith("+")

    def test_extract_avg_metric_returns_mean(self):
        """_extract_avg_metric returns mean across seasons."""
        metrics = {
            "per_season_results": [
                {"season": 2021, "metrics": {"mae": 9.0}},
                {"season": 2022, "metrics": {"mae": 11.0}},
            ]
        }
        avg = _extract_avg_metric(metrics, "mae")
        assert avg is not None
        assert abs(avg - 10.0) < 1e-6

    def test_extract_avg_metric_returns_none_when_key_absent(self):
        """_extract_avg_metric returns None when metric key is not in per_season_results."""
        metrics = {
            "per_season_results": [
                {"season": 2021, "metrics": {"accuracy": 0.55}},
            ]
        }
        result = _extract_avg_metric(metrics, "mae")
        assert result is None
