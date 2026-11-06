"""Tests for Study 1a: the install bar, the weight bar and the written report."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from backtest.market_info import InfoResult
from backtest.study_1a import (
    TARGETS,
    VARIANTS,
    LossScores,
    MarketTest,
    Study1aResult,
    VariantScores,
    context_notes,
    earned_weight,
    install_verdict,
    summary_lines,
    write_report,
)

_LOSS = {"wp": 0.640, "ats": 10.50, "ou": 11.07}


def _market_test(info: InfoResult | None) -> MarketTest:
    return MarketTest(
        info=info,
        note=""
        if info is not None
        else "degenerate fit: residuals are numerically zero",
        games=1300,
        market_loss=0.6,
        model_loss=0.7,
    )


def _scores(target: str, weight: float) -> VariantScores:
    info = InfoResult(0.2, 0.08, 0.006, 1300) if weight > 0 else None
    return VariantScores(
        losses=LossScores(
            pooled=_LOSS[target],
            pooled_games=2209,
            by_season={2017: _LOSS[target], 2026: _LOSS[target]},
            residual_by_season={2017: 0.7, 2026: -0.2},
            games_by_season={2017: 267, 2026: 121},
            weeks_2026=_LOSS[target],
            weeks_2026_games=121,
        ),
        decision=_market_test(info),
        tuned_weight=weight,
        earned_weight=weight,
        close=_market_test(InfoResult(-0.1, 0.03, 0.999, 2208)),
    )


def fake_result(*, install: bool, weights: dict[str, float]) -> Study1aResult:
    return Study1aResult(
        generation_cutoff_utc="2026-11-05T22:00:00+00:00",
        scores={
            target: {variant: _scores(target, weights[target]) for variant in VARIANTS}
            for target in TARGETS
        },
        install=install,
        weights=dict(weights),
        current_weights=dict(weights),
        counts={"decision_corpus_games": 1421},
    )


def test_v2_installs_only_if_it_is_at_least_as_good_on_every_target():
    cur = {"wp": 0.640, "ats": 10.50, "ou": 11.07}
    assert install_verdict(cur, {"wp": 0.630, "ats": 10.30, "ou": 11.07})
    assert not install_verdict(cur, {"wp": 0.630, "ats": 10.30, "ou": 11.08})


def test_weight_is_zero_unless_the_information_test_passes():
    assert earned_weight(0.4, InfoResult(0.2, 0.08, 0.006, 1300)) == 0.4
    assert earned_weight(0.4, InfoResult(0.2, 0.15, 0.09, 1300)) == 0.0
    assert earned_weight(0.4, InfoResult(-0.1, 0.03, 0.999, 1300)) == 0.0


def test_report_is_ascii_and_states_the_verdict(tmp_path: Path):
    write_report(
        fake_result(install=False, weights={"wp": 0.0, "ats": 0.0, "ou": 0.12}),
        tmp_path,
    )
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.isascii()
    assert "v2 does NOT go in" in text
    assert json.loads((tmp_path / "study_1a.json").read_text())["weights"]["ou"] == 0.12


def test_summary_stays_within_eight_lines_and_keeps_the_verdict_phrase():
    for install in (True, False):
        lines = summary_lines(
            fake_result(install=install, weights={"wp": 0.0, "ats": 0.0, "ou": 0.0})
        )
        assert len(lines) <= 8
        assert all(line.isascii() for line in lines)
        assert ("v2 goes in" in lines[0]) is install
        assert ("v2 does NOT go in" in lines[0]) is not install


def test_summary_reports_the_2026_games_and_the_line_comparison():
    lines = summary_lines(
        fake_result(install=False, weights={"wp": 0.0, "ats": 0.0, "ou": 0.0})
    )
    text = "\n".join(lines)
    assert "On the 121 finished 2026 games (reported only, never a veto)" in text
    assert "WP level (0.6400 vs 0.6400)" in text
    assert "for v2 vs" in text and "for the line (trails)" in text


def test_notes_report_passing_tests_the_lean_and_the_optimism_caveat():
    notes = "\n".join(
        context_notes(
            fake_result(install=True, weights={"wp": 0.0, "ats": 0.0, "ou": 0.0})
        )
    )
    assert "small optimistic bias" in notes
    assert "None of the 6 information tests passes" in notes
    assert "after-the-fact lean correction must not be dropped" in notes
    assert "v2 +0.70 points vs current +0.70" in notes

    passing = fake_result(install=True, weights={"wp": 0.0, "ats": 0.0, "ou": 0.4})
    text = "\n".join(context_notes(passing))
    assert "2 of the 6 information tests passes" in text
    assert "weak evidence (2 of 6 tests)" in text


def test_the_fitted_weight_zero_is_named_when_a_test_passes():
    result = fake_result(install=True, weights={"wp": 0.0, "ats": 0.0, "ou": 0.0})
    scores = result.scores["ou"]["current"]
    passing = VariantScores(
        losses=scores.losses,
        decision=_market_test(InfoResult(0.131, 0.07, 0.034, 1406)),
        tuned_weight=0.0,
        earned_weight=0.0,
        close=scores.close,
    )
    result.scores["ou"]["current"] = passing
    text = "\n".join(context_notes(result))
    assert "1 of the 6 information tests passes" in text
    assert "the blend fit chose 0" in text


def test_the_json_carries_the_applied_weights_and_provenance(tmp_path: Path):
    base = fake_result(install=True, weights={"wp": 0.0, "ats": 0.0, "ou": 0.12})
    result = replace(
        base,
        applied_weights=dict(base.weights),
        code_commit="abc1234",
        v2_gold_manifest={"rows": 6499},
    )
    write_report(result, tmp_path)
    data = json.loads((tmp_path / "study_1a.json").read_text())
    assert (
        data["applied_weights"]
        == data["weights"]
        == {"wp": 0.0, "ats": 0.0, "ou": 0.12}
    )
    assert data["code_commit"] == "abc1234"
    assert data["v2_gold_manifest"] == {"rows": 6499}
