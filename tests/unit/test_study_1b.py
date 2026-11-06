"""Tests for Study 1b: the line-move tests, the two-comparison verdict and the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.market_info import InfoResult
from backtest.study_1b import (
    COMPARISONS,
    TARGETS,
    MoveTest,
    Study1bResult,
    TargetVerdict,
    move_test,
    passes,
    summary_lines,
    target_verdict,
    wp_market_log_odds,
    write_report,
)
from models.market_probability import NoPriorFoldError
from scripts.run_study_1b import main


def _moves(seed=11, n=1300):
    rng = np.random.default_rng(seed)
    early, edge = rng.normal(0, 6, n), rng.normal(0, 2.5, n)
    return early + 0.2 * edge + rng.normal(0, 1.0, n), early, early + edge


_OK = MoveTest(InfoResult(0.05, 0.01, 1e-6, 1300), "", 1300, 1.0, 2.0)
_FLAT = MoveTest(InfoResult(0.01, 0.02, 0.31, 1300), "", 1300, 0.6, 2.0)


def fake_result(*, passed: dict[str, bool]) -> Study1bResult:
    verdicts = {
        target: TargetVerdict(
            close=_OK if passed[target] else _FLAT,
            day_before=_OK if passed[target] else _FLAT,
            passed=passed[target],
            share_clearing_slippage=None if target == "wp" else 0.41,
        )
        for target in TARGETS
    }
    counts = {
        "study_season_games": 1408,
        "lines_excluded_no_early_capture": 65,
        "games_with_lines": 1343,
        "inputs_excluded_site_may_postdate_early_capture": 22,
        "qb_values_clipped": 59,
        "wp_no_prior_fold_converter": 260,
    }
    return Study1bResult(
        generation_cutoff_utc="2026-11-06T20:00:00+00:00",
        verdicts=verdicts,
        counts=counts,
        code_commit="abc1234",
        v2_gold_manifest={"rows": 6499},
    )


def test_a_planted_move_toward_the_model_is_found():
    later, early, model = _moves()
    r = move_test(later, early, model)
    assert r.info is not None
    assert (
        r.info.coef == pytest.approx(0.2, abs=0.05)
        and r.info.p_one_sided < 0.05
        and r.games == 1300
    )


def test_a_missing_early_line_drops_only_that_row():
    later, early, model = _moves()
    early[3] = np.nan
    assert move_test(later, early, model).games == 1299


def test_a_fit_that_cannot_run_becomes_a_note():
    r = move_test([1.0, 2.0], [0.0, 0.0], [1.0, 1.5])
    assert r.info is None and "at least 3" in r.note and r.games == 2


def test_both_comparisons_must_pass():
    ok = MoveTest(InfoResult(0.05, 0.01, 1e-6, 1300), "", 1300, 1.0, 2.0)
    flat = MoveTest(InfoResult(0.01, 0.02, 0.31, 1300), "", 1300, 0.6, 2.0)
    wrong_way = MoveTest(InfoResult(-0.05, 0.01, 1.0, 1300), "", 1300, 1.0, 2.0)
    broken = MoveTest(None, "degenerate fit", 1300, None, None)
    assert passes(ok, ok)
    assert not passes(ok, flat) and not passes(flat, ok)
    assert not passes(ok, broken) and not passes(wrong_way, ok)


def test_wp_market_log_odds_are_the_season_slope_times_the_spread():
    frame = pd.DataFrame({"season": [2021, 2021], "early_spread": [3.0, -7.0]})
    np.testing.assert_allclose(
        wp_market_log_odds(frame, "early_spread", {2021: 0.15}), [0.45, -1.05]
    )
    with pytest.raises(NoPriorFoldError):
        wp_market_log_odds(frame.assign(season=2020), "early_spread", {2021: 0.15})


def _joined(n=60, seed=5):
    """Predictions joined to lines: 2 rows of 2020, one missing early total, one no close."""
    rng = np.random.default_rng(seed)
    early = rng.normal(0, 6, n)
    edge = rng.normal(0, 2.5, n)
    total = 45 + rng.normal(0, 4, n)
    frame = pd.DataFrame(
        {
            "game_id": [f"g{i:03d}" for i in range(n)],
            "season": [2020, 2020, *([2021] * (n - 2))],
            "early_spread": early,
            "day_before_spread": early + 0.3 * edge + rng.normal(0, 0.5, n),
            "close_spread": early + 0.4 * edge + rng.normal(0, 0.8, n),
            "early_total": total,
            "day_before_total": total + rng.normal(0, 0.5, n),
            "close_total": total + rng.normal(0, 0.8, n),
        }
    )
    frame.loc[10, "early_total"] = np.nan
    frame.loc[20, "close_spread"] = np.nan
    return frame, early + edge, total + rng.normal(0, 2, n)


def test_target_verdict_counts_every_drop_and_measures_the_slippage_share():
    frame, margin, total = _joined()
    slopes = {2021: 0.15}

    ats, ats_counts = target_verdict("ats", frame.assign(prediction=margin), slopes)
    assert ats_counts == {
        "ats_close_missing_number": 1,
        "ats_day_before_missing_number": 0,
    }
    assert (ats.close.games, ats.day_before.games) == (59, 60)
    assert ats.close.info is not None
    keep = frame["close_spread"].notna().to_numpy()
    departure = margin[keep] - frame["early_spread"].to_numpy()[keep]
    expected = np.mean(np.abs(ats.close.info.coef * departure) >= 0.5)
    assert ats.share_clearing_slippage == pytest.approx(expected)

    _, ou_counts = target_verdict("ou", frame.assign(prediction=total), slopes)
    assert ou_counts == {
        "ou_close_missing_number": 1,
        "ou_day_before_missing_number": 1,
    }

    probability = 1.0 / (1.0 + np.exp(-0.15 * margin))
    wp, wp_counts = target_verdict("wp", frame.assign(prediction=probability), slopes)
    assert wp_counts == {
        "wp_no_prior_fold_converter": 2,
        "wp_close_missing_number": 1,
        "wp_day_before_missing_number": 0,
    }
    assert (wp.close.games, wp.day_before.games) == (57, 58)
    assert wp.share_clearing_slippage is None


def test_report_is_ascii_and_states_each_verdict(tmp_path):
    write_report(fake_result(passed={"wp": False, "ats": True, "ou": False}), tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.isascii()
    assert "ATS: passes" in text and "WP: does NOT pass" in text
    assert (
        json.loads((tmp_path / "study_1b.json").read_text())["verdicts"]["ats"][
            "passed"
        ]
        is True
    )


def test_the_report_states_the_counted_caveats(tmp_path: Path):
    write_report(fake_result(passed={"wp": False, "ats": False, "ou": False}), tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="ascii")
    assert "22 neutral-site games" in text
    assert "59 rebuilt QB values" in text
    assert "indoor" in text and "Prediction Tracker" in text
    assert "Thursday" in text
    for comparison in COMPARISONS:
        assert (
            comparison
            in json.loads((tmp_path / "study_1b.json").read_text())["verdicts"]["ou"]
        )


def test_summary_names_the_arms_plan_4_may_build():
    lines = summary_lines(fake_result(passed={"wp": False, "ats": True, "ou": False}))
    text = "\n".join(lines)
    assert len(lines) <= 8 and text.isascii()
    assert "41%" in text and "early-week arm for: ATS" in text
    assert "keep recording" in text

    none = "\n".join(
        summary_lines(fake_result(passed={"wp": False, "ats": False, "ou": False}))
    )
    assert "no early-week arm" in none and "41%" not in none


def test_the_command_refuses_a_data_out_dir_before_the_run():
    with pytest.raises(ValueError, match="Refusing to write the Study 1b report"):
        main(["--out-dir", "data/study_1b"])
