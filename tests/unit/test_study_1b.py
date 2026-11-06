"""Tests for Study 1b: the line-move tests, the two-comparison verdict and the report."""

from __future__ import annotations

import json
import math
from dataclasses import replace
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
    postseason_counts,
    render_report,
    summary_lines,
    target_verdict,
    wp_market_log_odds,
    write_report,
)
from backtest.study_1b_lines import NO_DAY_BEFORE, NO_EARLY_CAPTURE, StudyLines
from models.market_probability import NoPriorFoldError
from scripts.run_study_1b import main


def _moves(seed=11, n=1300):
    rng = np.random.default_rng(seed)
    early, edge = rng.normal(0, 6, n), rng.normal(0, 2.5, n)
    return early + 0.2 * edge + rng.normal(0, 1.0, n), early, early + edge


def _move(
    coef: float, p: float, *, robust_p: float | None = None, move: float = 1.0
) -> MoveTest:
    """A move test whose classical and robust results share *coef*.

    *robust_p* defaults to *p*; the robust SE is set larger than the classical one, as on
    heavy-tailed moves.
    """
    return MoveTest(
        info=InfoResult(coef=coef, se=0.01, p_one_sided=p, n=1300),
        robust_info=InfoResult(
            coef=coef, se=0.015, p_one_sided=p if robust_p is None else robust_p, n=1300
        ),
        note="",
        games=1300,
        mean_abs_move=move,
        mean_abs_departure=2.0,
    )


_OK = _move(0.05, 1e-6)
_FLAT = _move(0.01, 0.31, move=0.6)


def fake_result(
    *, passed: dict[str, bool], postseason_with_lines: int = 0
) -> Study1bResult:
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
        "postseason_games": 65,
        "postseason_games_with_lines": postseason_with_lines,
        "postseason_no_early_capture": 65 - postseason_with_lines,
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
    assert r.info is not None and r.robust_info is not None
    assert (
        r.info.coef == pytest.approx(0.2, abs=0.05)
        and r.info.p_one_sided < 0.05
        and r.robust_info.p_one_sided < 0.05
        and r.games == 1300
    )


def test_a_missing_early_line_drops_only_that_row():
    later, early, model = _moves()
    early[3] = np.nan
    assert move_test(later, early, model).games == 1299


def test_a_fit_that_cannot_run_becomes_a_note():
    r = move_test([1.0, 2.0], [0.0, 0.0], [1.0, 1.5])
    assert r.info is None and r.robust_info is None
    assert "at least 3" in r.note and r.games == 2


def test_both_comparisons_must_pass():
    ok = _move(0.05, 1e-6)
    flat = _move(0.01, 0.31, move=0.6)
    wrong_way = _move(-0.05, 1.0)
    broken = MoveTest(
        info=None,
        robust_info=None,
        note="degenerate fit",
        games=1300,
        mean_abs_move=None,
        mean_abs_departure=None,
    )
    assert passes(ok, ok)
    assert not passes(ok, flat) and not passes(flat, ok)
    assert not passes(ok, broken) and not passes(wrong_way, ok)


def test_the_robust_se_is_hc1_by_hand():
    """HC1 on seven fixed rows, against the slope's sandwich written out by hand."""
    later = [1.0, 2.5, 0.5, 4.0, 3.0, -1.0, 2.0]
    early = [0.0, 1.0, 0.5, 2.0, 1.0, 0.0, 1.5]
    model = [1.5, 2.0, -0.5, 4.5, 3.5, -2.0, 1.0]
    y = [a - b for a, b in zip(later, early, strict=True)]
    x = [m - b for m, b in zip(model, early, strict=True)]
    n = len(y)
    x_bar, y_bar = sum(x) / n, sum(y) / n
    sxx = sum((xi - x_bar) ** 2 for xi in x)
    slope = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y, strict=True)) / sxx
    intercept = y_bar - slope * x_bar
    resid = [yi - intercept - slope * xi for xi, yi in zip(x, y, strict=True)]
    # The slope row of (X'X)^-1 X' is (x_i - x_bar) / Sxx, so its sandwich entry is the sum
    # of that weight squared times e_i^2; HC1 scales it by n / (n - 2).
    variance = (
        n
        / (n - 2)
        * sum(
            ((xi - x_bar) / sxx) ** 2 * ei**2 for xi, ei in zip(x, resid, strict=True)
        )
    )
    se = math.sqrt(variance)
    p_one_sided = 0.5 * math.erfc(slope / se / math.sqrt(2.0))

    r = move_test(later, early, model)
    assert r.info is not None and r.robust_info is not None
    assert r.robust_info.n == n == r.games
    assert abs(r.robust_info.coef - slope) < 1e-10
    assert abs(r.robust_info.se - se) < 1e-10
    assert abs(r.robust_info.p_one_sided - p_one_sided) < 1e-10
    assert r.robust_info.coef == r.info.coef  # one slope, two standard errors


def test_a_classical_pass_with_a_robust_miss_does_not_clear():
    robust_miss = _move(0.05, 0.01, robust_p=0.08)
    robust_only = _move(0.05, 0.08, robust_p=0.01)
    no_robust = replace(_OK, robust_info=None, note="degenerate robust fit")
    assert not passes(robust_miss, _OK) and not passes(_OK, robust_miss)
    assert not passes(no_robust, _OK)
    assert passes(robust_only, _OK)  # the robust p decides, the classical p does not

    result = fake_result(passed={"wp": False, "ats": False, "ou": False})
    verdict = TargetVerdict(
        close=robust_miss,
        day_before=_OK,
        passed=passes(robust_miss, _OK),
        share_clearing_slippage=0.41,
    )
    result = replace(result, verdicts={**result.verdicts, "ats": verdict})
    summary = "\n".join(summary_lines(result))
    assert "ATS: does NOT pass (it fails against the nflverse close)." in summary
    rows = {
        line.split(" | ")[1]: line
        for line in render_report(result).splitlines()
        if line.startswith("| ATS | ")
    }
    assert "| no |" in rows["the nflverse close"]
    assert "| yes |" in rows["the day-before line"]


def test_noise_growing_with_the_disagreement_widens_the_robust_se():
    rng = np.random.default_rng(7)
    n = 2000
    early, edge = rng.normal(0, 6, n), rng.normal(0, 2.5, n)
    later = early + 0.2 * edge + rng.normal(0, 1.0, n) * 0.5 * np.abs(edge)
    r = move_test(later, early, early + edge)
    assert r.info is not None and r.robust_info is not None
    assert r.robust_info.coef == r.info.coef
    assert r.robust_info.se > 1.4 * r.info.se


def test_a_robust_se_that_is_not_finite_becomes_a_note():
    """Rows near 1e78 overflow the HC1 meat (e^2 x^2) while the classical fit stays finite."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * 1e78
    later = 0.5 * x + np.array([1.0, -1.0, 0.5, 1.0, -1.0]) * 1e78
    r = move_test(later, np.zeros(5), x)
    assert r.info is not None and r.robust_info is None
    assert "robust" in r.note
    assert not passes(r, _OK)


def test_p_values_either_side_of_the_bar_print_differently():
    result = fake_result(passed={"wp": False, "ats": False, "ou": False})
    below = _move(0.05, 0.04996, robust_p=0.04996)
    above = _move(0.05, 0.05004, robust_p=0.05004)
    hair = _move(0.05, 0.0499996, robust_p=0.0499996)
    verdicts = {
        "wp": TargetVerdict(below, below, passes(below, below), None),
        "ats": TargetVerdict(above, above, passes(above, above), 0.41),
        "ou": TargetVerdict(hair, _OK, passes(hair, _OK), 0.41),
    }
    result = replace(result, verdicts=verdicts)
    for text in ("\n".join(summary_lines(result)), render_report(result)):
        assert "0.04996" in text and "0.05004" in text
        assert "0.0499996" in text  # never rounded up onto the bar
        assert "1e-06" in text  # a tiny p stays readable


def test_wp_market_log_odds_are_the_season_slope_times_the_spread():
    frame = pd.DataFrame({"season": [2021, 2021], "early_spread": [3.0, -7.0]})
    np.testing.assert_allclose(
        wp_market_log_odds(frame, "early_spread", {2021: 0.15}), [0.45, -1.05]
    )
    with pytest.raises(NoPriorFoldError):
        wp_market_log_odds(frame.assign(season=2020), "early_spread", {2021: 0.15})


def _joined(n=60, seed=5):
    """Predictions joined to lines: 2 rows of 2020, one missing early total, one no close.

    The spreads carry a planted move: with the model margin at early + edge, the close sits
    at early + 0.4 x edge and the day-before line at early + 0.3 x edge. The totals carry none.
    """
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


@pytest.mark.parametrize(
    ("target", "close_slope", "day_before_slope"),
    [("ats", 0.4, 0.3), ("wp", 0.4, 0.3), ("ou", 0.0, 0.0)],
)
def test_target_verdict_recovers_the_planted_slopes(
    target: str, close_slope: float, day_before_slope: float
):
    """Pins the direction and scale of each target's regression, not only its counts.

    WP's probability is sigmoid(0.15 x margin) and its lines are 0.15 x the spreads, so in
    log-odds its slopes equal the spreads' 0.4 and 0.3. A swapped later and early line, a raw
    WP probability, or later WP lines left in points all land far outside the tolerance.
    """
    frame, margin, total = _joined()
    predictions = {
        "ats": margin,
        "ou": total,
        "wp": 1.0 / (1.0 + np.exp(-0.15 * margin)),
    }
    verdict, _ = target_verdict(
        target, frame.assign(prediction=predictions[target]), {2021: 0.15}
    )
    assert verdict.close.info is not None and verdict.day_before.info is not None
    assert verdict.close.info.coef == pytest.approx(close_slope, abs=0.1)
    assert verdict.day_before.info.coef == pytest.approx(day_before_slope, abs=0.1)


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
    assert "59 rebuilt QB values" in text and "WHOLE season" in text
    assert "robust" in text and "HC1" in text and "owner ruling 3" in text
    assert "indoor" in text and "Prediction Tracker" in text
    assert "Thursday" in text
    for comparison in COMPARISONS:
        assert (
            comparison
            in json.loads((tmp_path / "study_1b.json").read_text())["verdicts"]["ou"]
        )


def test_postseason_counts_come_from_the_games():
    games = pd.DataFrame(
        {
            "game_id": ["r1", "r2", "r3", "p1", "p2", "p3"],
            "game_type": ["REG", "REG", "REG", "WC", "SB", "DIV"],
        }
    )
    lines = StudyLines(
        frame=pd.DataFrame({"game_id": ["r1", "r2", "p3"]}),
        excluded=pd.DataFrame(
            {
                "game_id": ["r3", "p1", "p2"],
                "reason": [NO_EARLY_CAPTURE, NO_EARLY_CAPTURE, NO_DAY_BEFORE],
            }
        ),
    )
    assert postseason_counts(lines, games) == {
        "postseason_games": 3,
        "postseason_games_with_lines": 1,
        "postseason_no_early_capture": 1,
    }


def test_the_report_says_the_verdict_covers_only_the_regular_season():
    none = render_report(fake_result(passed={"wp": False, "ats": False, "ou": False}))
    assert "postseason included" not in none
    assert "the regular season only" in none
    assert "none of the 65 postseason games has lines" in none
    assert "65 of them have no early capture" in none

    some = render_report(
        fake_result(
            passed={"wp": False, "ats": False, "ou": False}, postseason_with_lines=3
        )
    )
    assert "the regular season only" not in some
    assert "3 of the 65 postseason games have lines" in some


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
