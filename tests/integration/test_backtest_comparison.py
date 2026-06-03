"""Integration tests for the baseline comparison + the rewired gating logic (ACCU-06).

Verifies, after the Phase 24-04 rewire of ``scripts/retrain_models.py`` onto the shared
``models.deploy_gate``:
1. load_baseline_metrics reads metrics_*.json from a directory (retained for the comparison
   template).
2. gate_targets delegates the per-target deploy decision to the shared
   deploy_gate.build_candidate_bundle + evaluate_target (against the FROZEN config/gate.toml
   baseline) -- the legacy headline_clv path (_gate_wp / _gate_regression_target) is GONE and
   its logic is covered by tests/unit/test_deploy_gate.py.
3. fill_comparison_template produces valid markdown with v2.0 values and deltas (retained).

Does NOT run any actual backtest. The gate_targets candidate frames are synthetic, mirroring
the tiny-scored-frame idiom from tests/unit/test_deploy_gate.py.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestResults
from scripts.retrain_models import (
    _extract_avg_metric,
    _format_delta,
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


def _results_like(all_predictions: dict[str, pd.DataFrame]) -> BacktestResults:
    """Wrap a {target -> predictions} dict in a real BacktestResults (mirrors diagnose).

    The rewired ``gate_targets`` reads only ``results.all_predictions[target]`` (the scored
    candidate frame fed to the shared bundle builder), so the other fields are empty -- but a
    real ``BacktestResults`` (not a duck-typed shim) keeps the call type-correct, exactly as
    ``backtest.diagnose._results_like`` does.
    """
    return BacktestResults(
        config=BacktestConfig(),
        season_results=[],
        all_predictions=all_predictions,
        all_clv={},
        headline_clv={},
        odds_coverage={},
        covid_annotation={},
        era_info={},
        is_blended=False,
    )


def _candidate_frame_and_odds(
    clv_center: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build synthetic per-target scored frames + matching odds for gate_targets.

    Each target gets a per-game scored frame across the four holdout seasons (so per-season
    CLV slices clear MIN_CLV_SAMPLE), and the model outputs are shifted by ``clv_center`` so the
    candidate's CLV is deterministically positive (passes the floor) or negative (fails).

    Args:
        clv_center: Positive nudges model edges to beat the line (positive CLV -> pass);
            negative makes the candidate systematically worse than the line (fail).

    Returns:
        ``(all_predictions, odds_df)`` ready for ``gate_targets``.
    """
    rng = np.random.default_rng(17)
    seasons = [2021, 2022, 2023, 2024]
    per_season_n = 80
    n = per_season_n * len(seasons)
    game_ids = [f"G{i:04d}" for i in range(n)]
    season_col = [s for s in seasons for _ in range(per_season_n)]

    base_spread = rng.normal(-2.5, 4.0, n)
    base_total = rng.normal(45.0, 4.0, n)

    odds = pd.DataFrame(
        {
            "game_id": game_ids,
            "ml_home": [-150] * n,
            "ml_away": [130] * n,
            "spread": base_spread,
            "total": base_total,
        }
    )

    preds: dict[str, pd.DataFrame] = {}

    # WP: a probability whose implied edge over the (devigged) closing line is shifted by
    # clv_center; actual outcomes track that probability so accuracy stays near baseline.
    wp_prob = np.clip(0.58 + clv_center + rng.normal(0, 0.02, n), 0.05, 0.95)
    preds["wp"] = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": season_col,
            "week": list(range(1, n + 1)),
            "model_prob": wp_prob,
            "actual": (rng.random(n) < wp_prob).astype(int),
        }
    )

    # ATS: model_spread vs the closing spread; shift by clv_center * 3 (points) to move CLV.
    ats_spread = base_spread + clv_center * 3.0 + rng.normal(0, 0.5, n)
    preds["ats"] = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": season_col,
            "week": list(range(1, n + 1)),
            "model_prob": ats_spread,
            "model_spread": ats_spread,
            "actual": base_spread + rng.normal(0, 7.0, n),
        }
    )

    # OU: model_total vs the closing total; shift by clv_center * 3 (points) to move CLV.
    ou_total = base_total + clv_center * 3.0 + rng.normal(0, 0.5, n)
    preds["ou"] = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": season_col,
            "week": list(range(1, n + 1)),
            "model_prob": ou_total,
            "model_total": ou_total,
            "actual": base_total + rng.normal(0, 8.0, n),
        }
    )

    return preds, odds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v1_dir(tmp_path):
    """v1.0 baseline directory with all three metrics files (comparison template)."""
    d = tmp_path / "v1.0"
    _write_metrics(d, "wp", _make_wp_metrics(headline_clv=-0.0164, accuracy=0.57))
    _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.8701, mae=9.5))
    _write_metrics(d, "ou", _make_regression_metrics(headline_clv=45.77, mae=10.2))
    return d


@pytest.fixture
def v2_better_dir(tmp_path):
    """v2.0 baseline where all targets improved (comparison template)."""
    d = tmp_path / "v2_better"
    _write_metrics(d, "wp", _make_wp_metrics(headline_clv=-0.0100, accuracy=0.58))
    _write_metrics(d, "ats", _make_regression_metrics(headline_clv=0.9000, mae=9.3))
    _write_metrics(d, "ou", _make_regression_metrics(headline_clv=46.00, mae=10.0))
    return d


# ---------------------------------------------------------------------------
# load_baseline_metrics (retained for the comparison template)
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
# gate_targets: rewired onto the shared deploy_gate (build_candidate_bundle + evaluate_target)
# ---------------------------------------------------------------------------


class TestGateTargetsSharedGate:
    """gate_targets delegates to the shared deploy_gate against the frozen baseline.

    The per-target gate LOGIC (CLV floor, per-season-must-pass, secondary non-regression) is
    exhaustively covered by tests/unit/test_deploy_gate.py. Here we assert only the rewiring
    contract: gate_targets returns the evaluate_target reason-dict shape for all three targets,
    and a systematically-negative candidate CLV fails the gate.
    """

    def test_returns_three_targets_with_reason_dict_shape(self):
        """gate_targets returns wp/ats/ou each with the evaluate_target reason-dict shape."""
        preds, odds = _candidate_frame_and_odds(clv_center=0.05)
        results = gate_targets(_results_like(preds), odds)

        assert set(results.keys()) == {"wp", "ats", "ou"}
        for target in ("wp", "ats", "ou"):
            result = results[target]
            assert "passed" in result
            assert "reasons" in result
            assert isinstance(result["reasons"], list)
            assert len(result["reasons"]) > 0
            # evaluate_target aliases (kept for print_gating_summary compatibility).
            assert "v1_metrics" in result
            assert "v2_metrics" in result

    def test_systematically_negative_candidate_fails(self):
        """A candidate whose CLV is systematically worse than the line FAILS the floor."""
        preds, odds = _candidate_frame_and_odds(clv_center=-0.20)
        results = gate_targets(_results_like(preds), odds)

        # With a strongly-negative shift, at least one target trips the significance-tested
        # CLV floor (the hard block) rather than every target slipping through on noise.
        assert any(not results[t]["passed"] for t in ("wp", "ats", "ou")), results

    def test_empty_candidate_frame_fails_closed(self):
        """A target with no candidate predictions fails closed (no silent pass)."""
        preds, odds = _candidate_frame_and_odds(clv_center=0.05)
        preds["ats"] = pd.DataFrame()  # simulate an empty backtest frame for one target
        results = gate_targets(_results_like(preds), odds)

        assert results["ats"]["passed"] is False
        assert any("ats" in r.lower() for r in results["ats"]["reasons"])


# ---------------------------------------------------------------------------
# fill_comparison_template (retained for the human-readable comparison)
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
    """Unit tests for _format_delta and _extract_avg_metric (comparison-template helpers)."""

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
        metrics: dict[str, Any] = {
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
        metrics: dict[str, Any] = {
            "per_season_results": [
                {"season": 2021, "metrics": {"accuracy": 0.55}},
            ]
        }
        result = _extract_avg_metric(metrics, "mae")
        assert result is None
