"""Tests for Study 1a: the install bar, the weight bar and the written report."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.market_info import InfoResult
from backtest.study_1a import (
    TARGETS,
    VARIANTS,
    LossScores,
    MarketTest,
    Study1aResult,
    VariantScores,
    earned_weight,
    install_verdict,
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
