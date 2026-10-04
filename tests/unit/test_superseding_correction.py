"""The 2026 cold-start rule is superseded VISIBLY, never edited in place (Plan 33.2-26 Task 2, R14).

WHAT THIS MODULE PINS
---------------------
Commit ``11761c7`` froze the 2026 edge-band thresholds and the 2026 chain-fit bias in
``backtest/cold_start_constants.py`` + ``COLD-START-PREREGISTRATION.md``. Both were derived from
models and closing lines this phase replaced. A pre-registration is only worth something if it is
never edited, so the correction is a NEW module and a LATER commit that names the original by sha:

* the frozen pair of files is byte-unchanged since ``11761c7``;
* ``backtest/corrected_cold_start_constants.py`` names ``11761c7``, carries every symbol the
  original carries, and the commit that added it names ``11761c7`` too;
* a target with too little honest data gets NO threshold -- a named refusal and ``None`` -- never a
  default, a zero or the old value;
* the WP market side of every HISTORICAL row is converted OUT OF FOLD, never with the serving
  slope, and 2020 (no prior fold) leaves the WP derivation as ``no_prior_fold_converter`` while
  staying in the ATS and O/U derivations; no 2025 row enters.

What this module does NOT pin: that a ``None`` threshold places no bets through the selection path.
That is the selection path's behaviour, it lands with the repoint (Task 3), and it is asserted in
``tests/unit/test_two_test_win_bet_rule.py`` by behaviour.

Run this module:  uv run python -m pytest tests/unit/test_superseding_correction.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.derive_cold_start_constants as derive
from models.market_probability import (
    market_home_win_probability,
    oof_market_probability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SUPERSEDED_SHA = "11761c7"
FROZEN_PATHS: tuple[str, str] = (
    "COLD-START-PREREGISTRATION.md",
    "backtest/cold_start_constants.py",
)
CORRECTED_MODULE = "backtest/corrected_cold_start_constants.py"
CORRECTION_DOCUMENT = "COLD-START-CORRECTION.md"

# The fixture converter. Its per-season prior-only slopes differ SHARPLY from the serving slope,
# so a conversion through the wrong one cannot agree with the right one by accident.
_WALK_FORWARD_SLOPES: dict[str, float] = {
    "2021": 0.30,
    "2022": 0.02,
    "2023": 0.30,
    "2024": 0.02,
}
_SERVING_SLOPE = 0.15
_GAMES_PER_SEASON = 40
_SEASONS = (2020, 2021, 2022, 2023, 2024)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


# ---------------------------------------------------------------------------
# Fixture frames (in memory; nothing under data/, artifacts/ or outputs/ is read)
# ---------------------------------------------------------------------------


def _corpus(seasons: tuple[int, ...] = _SEASONS) -> pd.DataFrame:
    """An owned pre-lock corpus: one row per game, spread on the home-margin scale."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for index in range(_GAMES_PER_SEASON):
            rows.append(
                {
                    "game_id": f"{season}_{index:03d}",
                    "season": season,
                    "week": 1 + index % 17,
                    "market_spread": float(rng.normal(0.0, 6.0)),
                    "market_total": float(rng.normal(45.0, 4.0)),
                }
            )
    return pd.DataFrame(rows)


def _predictions(corpus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Walk-forward-shaped predictions (game_id, season, prediction, actual) per target."""
    rng = np.random.default_rng(11)
    n = len(corpus)
    base = corpus[["game_id", "season"]].reset_index(drop=True)
    return {
        "wp": base.assign(
            prediction=np.clip(rng.uniform(0.2, 0.8, n), 0.01, 0.99),
            actual=rng.integers(0, 2, n).astype(float),
        ),
        "ats": base.assign(
            prediction=corpus["market_spread"].to_numpy() + rng.normal(0.0, 2.0, n),
            actual=rng.normal(0.0, 13.0, n),
        ),
        "ou": base.assign(
            prediction=corpus["market_total"].to_numpy() + rng.normal(0.0, 2.0, n),
            actual=rng.normal(45.0, 13.0, n),
        ),
    }


def _edges() -> Any:
    corpus = _corpus()
    return derive.build_prelock_edge_frame(
        corpus, _predictions(corpus), _WALK_FORWARD_SLOPES
    )


# ---------------------------------------------------------------------------
# 1. The frozen record is byte-unchanged, and the correction names it
# ---------------------------------------------------------------------------


def test_the_frozen_record_is_byte_unchanged_since_11761c7() -> None:
    """``git diff 11761c7 HEAD`` and the working-tree diff are both EMPTY over both files.

    The ``ls-tree`` control stops a sha that never carried the files from making both diffs
    trivially empty.
    """
    carried = _git("ls-tree", "--name-only", SUPERSEDED_SHA, "--", *FROZEN_PATHS)
    assert sorted(carried.stdout.split()) == sorted(FROZEN_PATHS), carried.stderr
    committed = _git("diff", SUPERSEDED_SHA, "HEAD", "--", *FROZEN_PATHS)
    working = _git("diff", SUPERSEDED_SHA, "--", *FROZEN_PATHS)
    assert committed.returncode == 0 and working.returncode == 0
    assert committed.stdout == "", "the frozen 11761c7 record was edited in a commit"
    assert working.stdout == "", (
        "the frozen 11761c7 record is edited in the working tree"
    )


def test_the_corrected_module_names_the_record_it_supersedes() -> None:
    """The NEW module exists, names 11761c7 by sha, and resolves to the same full commit."""
    path = REPO_ROOT / CORRECTED_MODULE
    assert path.exists(), f"{CORRECTED_MODULE} has not been written"
    assert SUPERSEDED_SHA in path.read_text(encoding="utf-8")

    import backtest.corrected_cold_start_constants as corrected

    full = _git("rev-parse", SUPERSEDED_SHA).stdout.strip()
    assert full == corrected.SUPERSEDED_PREREGISTRATION_COMMIT
    assert corrected.PREREGISTRATION_PATHS == (CORRECTION_DOCUMENT, CORRECTED_MODULE)


def test_the_commit_that_added_the_corrected_module_names_11761c7() -> None:
    """The staging commit of the supersession names the record it supersedes."""
    added = _git("log", "--diff-filter=A", "--format=%B", "--", CORRECTED_MODULE)
    assert added.stdout.strip(), f"no commit has added {CORRECTED_MODULE} yet"
    assert SUPERSEDED_SHA in added.stdout


def test_the_corrected_module_carries_every_symbol_the_original_carries() -> None:
    """Nothing silently dropped in the re-render (T-33.2-26-06)."""
    import backtest.cold_start_constants as original
    import backtest.corrected_cold_start_constants as corrected

    original_symbols = {name for name in dir(original) if name.isupper()}
    corrected_symbols = {name for name in dir(corrected) if name.isupper()}
    assert sorted(original_symbols - corrected_symbols) == []


def test_the_corrected_values_are_not_the_superseded_ones() -> None:
    """Non-vacuity: a correction whose values equal the originals would supersede nothing."""
    import backtest.cold_start_constants as original
    import backtest.corrected_cold_start_constants as corrected

    assert corrected.CHAIN_FIT_BIAS_2026 != original.CHAIN_FIT_BIAS_2026
    assert (
        corrected.EDGE_TIER_THRESHOLDS_BY_TARGET["ats"]
        != original.EDGE_TIER_THRESHOLDS_BY_TARGET["ats"]
    )
    assert (
        dict(original.EDGE_TIER_THRESHOLDS_BY_TARGET)
        == corrected.SUPERSEDED_EDGE_TIER_THRESHOLDS_BY_TARGET
    )
    assert (
        dict(original.CHAIN_FIT_BIAS_2026) == corrected.SUPERSEDED_CHAIN_FIT_BIAS_2026
    )


# ---------------------------------------------------------------------------
# 2. Too little honest data: a named refusal and None, never a borrowed value
# ---------------------------------------------------------------------------


def test_a_too_small_pool_raises_the_named_refusal() -> None:
    too_small = np.ones(derive.MIN_HONEST_THRESHOLD_ROWS - 1)
    with pytest.raises(derive.InsufficientHonestDataForThresholdError, match="'ats'"):
        derive.require_honest_pool("ats", too_small)
    derive.require_honest_pool("ats", np.ones(derive.MIN_HONEST_THRESHOLD_ROWS))


def test_the_refusal_dodges_the_repository_s_degrade_quietly_handlers() -> None:
    """A refusal a ``(ValueError, KeyError, RuntimeError, LookupError)`` catch could swallow
    would turn "no threshold" back into "use a default" somewhere downstream."""
    assert not issubclass(
        derive.InsufficientHonestDataForThresholdError,
        (ValueError, KeyError, RuntimeError, LookupError),
    )


def test_a_planted_small_pool_yields_none_and_never_the_old_value() -> None:
    edges = _edges()
    frame = edges.frame.copy()
    keep = frame.index[:5]
    frame.loc[~frame.index.isin(keep), "ats_edge"] = np.nan

    result = derive.derive_threshold_pairs(frame)

    assert result.thresholds["ats"] is None
    assert "ats" in result.refusals
    assert str(derive.MIN_HONEST_THRESHOLD_ROWS) in result.refusals["ats"]
    assert result.thresholds["ou"] is not None
    assert result.thresholds["wp"] == (
        derive.WP_HIGH_THRESHOLD,
        derive.WP_MEDIUM_THRESHOLD,
    )


def test_a_refused_wp_anchor_leaves_every_target_without_a_threshold() -> None:
    """ATS and O/U reproduce WP's band shares; with no WP anchor there is nothing to reproduce."""
    frame = _edges().frame.copy()
    frame.loc[frame.index[5:], "wp_edge"] = np.nan

    result = derive.derive_threshold_pairs(frame)

    assert result.thresholds == {"ats": None, "ou": None, "wp": None}
    assert sorted(result.refusals) == ["ats", "ou", "wp"]


# ---------------------------------------------------------------------------
# 3. The WP market side is OUT OF FOLD, never the serving slope
# ---------------------------------------------------------------------------


def test_the_wp_market_side_equals_the_out_of_fold_conversion_row_for_row() -> None:
    edges = _edges()
    wp = edges.frame[edges.frame["wp_edge"].notna()]
    expected = oof_market_probability(
        wp.assign(home_fav_margin=wp["market_spread"]), _WALK_FORWARD_SLOPES
    )
    np.testing.assert_array_equal(wp["wp_market_prob"].to_numpy(), expected)


def test_a_planted_serving_slope_conversion_differs() -> None:
    """The control that makes the equality above load-bearing."""
    wp = _edges().frame
    wp = wp[wp["wp_edge"].notna()]
    planted = market_home_win_probability(
        wp["market_spread"].to_numpy(), _SERVING_SLOPE
    )
    assert not np.allclose(wp["wp_market_prob"].to_numpy(), planted)


class ServingSlopeUsedOnHistoryError(Exception):
    """Raised by the patched serving-slope paths: history must never reach them."""


def test_the_derivation_completes_with_the_serving_slope_paths_patched_to_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioural half of the parsed-tree scan: a helper the scan cannot see into is caught."""
    from models import market_probability
    from models.blending import MarketBlender

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ServingSlopeUsedOnHistoryError

    monkeypatch.setattr(market_probability, "market_home_win_probability", _refuse)
    monkeypatch.setattr(MarketBlender, "_blend_wp_predictions", _refuse)

    result = derive.derive_threshold_pairs(_edges().frame)
    assert result.thresholds["ats"] is not None


def test_2020_leaves_the_wp_derivation_and_stays_in_ats_and_ou() -> None:
    edges = _edges()
    frame = edges.frame
    first = frame[frame["season"] == 2020]
    assert len(first) == _GAMES_PER_SEASON
    assert first["wp_edge"].isna().all()
    assert first["ats_edge"].notna().all()
    assert first["ou_edge"].notna().all()
    assert sorted(edges.wp_excluded["game_id"]) == sorted(first["game_id"])
    assert set(edges.wp_excluded["reason"]) == {"no_prior_fold_converter"}

    result = derive.derive_threshold_pairs(frame)
    assert result.eligible_rows["wp"] == 4 * _GAMES_PER_SEASON
    assert result.eligible_rows["ats"] == 5 * _GAMES_PER_SEASON
    assert result.eligible_rows["ou"] == 5 * _GAMES_PER_SEASON


def test_a_2025_row_is_refused_by_name() -> None:
    """The single-use 2025 hold is spent and no row of it enters the derivation."""
    from scripts.derive_corrected_ev_chain import SpentHoldSeasonError

    corpus = _corpus((2024, 2025))
    with pytest.raises(SpentHoldSeasonError):
        derive.build_prelock_edge_frame(
            corpus, _predictions(corpus), _WALK_FORWARD_SLOPES
        )


def test_the_derivation_source_calls_only_the_out_of_fold_converter() -> None:
    """Parsed CALL nodes, so a comment naming a forbidden function cannot trip it."""
    tree = ast.parse(
        (REPO_ROOT / "scripts/derive_cold_start_constants.py").read_text(
            encoding="utf-8"
        )
    )

    def _name(call: ast.Call) -> str:
        func = call.func
        if isinstance(func, ast.Attribute):
            return func.attr
        return getattr(func, "id", "")

    calls = [_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls
    assert calls.count("oof_market_probability") >= 1
    forbidden = {
        "market_home_win_probability",
        "blend_predictions",
        "_blend_wp_predictions",
    }
    assert sorted(set(calls) & forbidden) == []


# ---------------------------------------------------------------------------
# 4. The chain-fit read path and the 2026 bias
# ---------------------------------------------------------------------------


def test_the_derivation_reads_the_corrected_chain_fit_not_the_phase31_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserted on the path the loader is actually HANDED, not on a constant's intent."""
    import backtest.weekly_bet_list as weekly

    seen: list[Path] = []

    def _record_path(path: Path | str = weekly.DEFAULT_CHAIN_FIT_PATH) -> Any:
        seen.append(Path(path))
        return {}

    monkeypatch.setattr(weekly, "load_frozen_chain_fit", _record_path)
    derive.load_corrected_chain_fit(Path("repo"))

    assert [p.as_posix() for p in seen] == [
        "repo/outputs/p332/corrected_chain_fit.json"
    ]
    assert "profitability_2025_verdict" not in derive.CHAIN_FIT_SOURCE.as_posix()


def test_the_2026_bias_continues_the_record_s_own_series() -> None:
    """The record's per-season biases are REPRODUCED from the same residuals, then extended."""
    from backtest.ou_ev_chain import estimate_prior_season_bias

    residuals = {
        2017: np.array([1.0, -1.0, 3.0]),
        2018: np.array([0.5]),
        2019: np.array([-2.0, 2.0]),
        2020: np.array([4.0, 0.0]),
    }
    record = {
        season: estimate_prior_season_bias(residuals, season) for season in (2019, 2020)
    }
    bias = derive.extend_prior_season_bias(residuals, record, 2026)
    assert bias == estimate_prior_season_bias(residuals, 2026)

    with pytest.raises(derive.ChainFitReproductionError, match="2020"):
        derive.extend_prior_season_bias(residuals, {**record, 2020: 9.0}, 2026)


# ---------------------------------------------------------------------------
# 5. The published record
# ---------------------------------------------------------------------------


def test_the_correction_document_covers_the_whole_2026_rule() -> None:
    path = REPO_ROOT / CORRECTION_DOCUMENT
    assert path.exists(), f"{CORRECTION_DOCUMENT} has not been written"
    text = path.read_text(encoding="utf-8")
    low = text.lower()
    assert text.isascii()
    assert "proven" not in low
    for token in ("11761c7", "ee20773", "EV-CHAIN-CORRECTION.md", "33.2-29"):
        assert token in text, token
    for phrase in (
        "ev floor",
        "residual sd",
        "no_prior_fold_converter",
        "not clean evidence",
    ):
        assert phrase in low, phrase
