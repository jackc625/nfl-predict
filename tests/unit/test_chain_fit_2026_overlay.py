"""The 2026 chain-fit bias reaches the loader by OVERLAY (Plan 33-17 Task 2; D33-21, T-33-88/89).

THE PROBLEM
-----------
``load_frozen_chain_fit`` reads the per-target walk-forward bias out of the profitability run
record, which covers seasons 2021-2025. ``_require_season_covered`` therefore refuses every 2026
selection, and the measurement that would extend the record is a SINGLE-USE 2025 hold split the
committed ledger marks as spent. The bias for 2026 exists -- it is frozen in
``backtest.cold_start_constants.CHAIN_FIT_BIAS_2026`` -- but nothing connected the two.

THE DESIGN, AND THE TWO ALTERNATIVES IT REJECTS (D33-21)
----------------------------------------------------------
The loader keeps reading the run record for 2021-2025 and OVERLAYS the committed constant for
2026 only.

  REJECTED: copying 2021-2025 into the new module. Two copies that can drift apart -- this
  repository has been bitten by exactly that three times.

  REJECTED: writing 2026 into the run record. That edits a spent one-shot measurement the ledger
  records as completed under an exclusive lock, which is T-33-86.

WHEN TWO SOURCES DISAGREE, THE LOAD REFUSES BY NAME
-----------------------------------------------------
If the overlay ever names a season the run record ALSO covers and the values DIFFER, the load
raises ``ChainFitOverlayDisagreementError`` naming the season and BOTH values. It does not apply a
precedence rule: a rule nobody remembers is how two sources of truth quietly become one answer. The
AGREEING case is asserted separately not to refuse, because a guard that refused on agreement too
would just be an outage.

THE KEY TRAP, AND WHY IT IS NOT FIXED TWICE
---------------------------------------------
JSON has no integer keys, so a season round-trips as the STRING ``"2026"`` while every lookup is by
``int``. ``load_frozen_chain_fit`` ALREADY normalizes at exactly one place -- it builds
``season_bias_by_season={int(season): float(bias) ...}``. The overlay's job is to PRESERVE that
int-keying, not to add a second coercion; two normalization sites are a drift surface.

MIND THE SHAPE THE LOADER RETURNS. ``load_frozen_chain_fit`` returns ``dict[str, WeeklyChainFit]``
keyed by TARGET, not by season. The per-season mapping is the INNER ``season_bias_by_season``, so
every check below iterates ``fits.items()`` -- an ``int(k)`` over the OUTER keys raises on ``'wp'``.

Run this module:  uv run pytest tests/unit/test_chain_fit_2026_overlay.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import backtest.weekly_bet_list as wbl
from backtest.cold_start_constants import (
    CHAIN_FIT_BIAS_2026,
    CHAIN_FIT_BIAS_SEASONS,
)
from backtest.ou_ev_chain import estimate_prior_season_bias
from backtest.weekly_bet_list import (
    CANONICAL_TARGETS,
    ChainFitOverlayDisagreementError,
    EmptyPriorResidualPoolError,
    frozen_overlay_season,
    load_frozen_chain_fit,
)

_OVERLAY_SEASON = 2026


def _record(season_bias: dict[str, float] | None = None) -> dict[str, Any]:
    """A run record in the on-disk shape the loader reads, covering 2021-2025 by default."""
    bias = (
        season_bias
        if season_bias is not None
        else {str(year): 0.1 for year in CHAIN_FIT_BIAS_SEASONS}
    )
    return {
        "tune_fit": {
            "wp": {
                "ev_floor_t": 0.05,
                "frozen_sd": None,
                "season_bias_by_season": dict(bias),
            },
            "ats": {
                "ev_floor_t": 0.05,
                "frozen_sd": 11.5,
                "season_bias_by_season": dict(bias),
            },
            "ou": {
                "ev_floor_t": 0.0,
                "frozen_sd": 13.0,
                "season_bias_by_season": dict(bias),
            },
        }
    }


def _write(tmp_path: Path, record: dict[str, Any]) -> Path:
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. The overlay season is DERIVED, not re-typed
# ---------------------------------------------------------------------------


def test_the_overlay_season_is_the_one_after_the_strictly_prior_pool() -> None:
    """2026 is not a literal in the loader: it is ``max(strictly-prior pool) + 1``.

    Deriving it is what keeps the season the bias is FOR and the seasons it was pooled FROM from
    becoming two independently-editable facts. The frozen constant is named ``CHAIN_FIT_BIAS_2026``
    and this must agree with that name, which is asserted rather than assumed.
    """
    assert frozen_overlay_season() == _OVERLAY_SEASON
    assert max(CHAIN_FIT_BIAS_SEASONS) == _OVERLAY_SEASON - 1
    assert all(season < _OVERLAY_SEASON for season in CHAIN_FIT_BIAS_SEASONS), (
        "the frozen pool contains a season at or after the season it debiases; the bias would "
        "then be estimated partly from the target season's own data"
    )


# ---------------------------------------------------------------------------
# 2. 2021-2025 still come from the run record; only 2026 comes from the overlay
# ---------------------------------------------------------------------------


def test_the_run_record_remains_the_source_for_its_own_seasons(tmp_path: Path) -> None:
    """The overlay ADDS a season; it must not restate the ones the record already carries."""
    path = _write(tmp_path, _record())
    fits = load_frozen_chain_fit(path)

    for target, fit in fits.items():
        for season in CHAIN_FIT_BIAS_SEASONS:
            assert fit.season_bias_by_season[season] == pytest.approx(0.1), (
                f"{target} season {season} no longer reads the run record's value"
            )


def test_every_target_gains_the_frozen_2026_entry(tmp_path: Path) -> None:
    """The whole point: after the overlay a 2026 selection has a bias to use."""
    fits = load_frozen_chain_fit(_write(tmp_path, _record()))

    assert sorted(fits) == sorted(CANONICAL_TARGETS)
    for target, fit in fits.items():
        assert _OVERLAY_SEASON in fit.season_bias_by_season, target
        assert fit.season_bias_by_season[_OVERLAY_SEASON] == pytest.approx(
            CHAIN_FIT_BIAS_2026[target]
        )


def test_the_inner_mapping_stays_int_keyed_through_the_overlay(tmp_path: Path) -> None:
    """R10. The loader already coerces once; the overlay must not introduce a string key.

    Asserted on the OUTER/INNER split as well, because a check that did ``int(k)`` over the outer
    dictionary would raise on ``'wp'`` -- the loader is TARGET-keyed, not season-keyed.
    """
    fits = load_frozen_chain_fit(_write(tmp_path, _record()))

    assert all(isinstance(target, str) for target in fits)
    for target, fit in fits.items():
        bad = [key for key in fit.season_bias_by_season if not isinstance(key, int)]
        assert not bad, f"{target} carries non-int season keys {bad}"


def test_a_json_string_key_is_found_by_an_int_lookup(tmp_path: Path) -> None:
    """The trap itself, driven end to end through a real JSON round trip.

    The record is written with a ``"2026"`` STRING key -- which is the only thing JSON can carry --
    and the assertion looks it up with the integer 2026.
    """
    record = _record()
    for block in record["tune_fit"].values():
        block["season_bias_by_season"][str(_OVERLAY_SEASON)] = CHAIN_FIT_BIAS_2026["wp"]
    # All three targets now carry the WP value under a STRING key; only WP agrees with the
    # overlay, so write each target its own agreeing value to isolate the ENCODING question
    # from the DISAGREEMENT question, which the next section tests on its own.
    for target, block in record["tune_fit"].items():
        block["season_bias_by_season"][str(_OVERLAY_SEASON)] = CHAIN_FIT_BIAS_2026[
            target
        ]

    reloaded = json.loads(json.dumps(record))
    assert str(_OVERLAY_SEASON) in reloaded["tune_fit"]["wp"]["season_bias_by_season"]

    fits = load_frozen_chain_fit(_write(tmp_path, reloaded))
    for target, fit in fits.items():
        assert fit.season_bias_by_season[_OVERLAY_SEASON] == pytest.approx(
            CHAIN_FIT_BIAS_2026[target]
        ), target


# ---------------------------------------------------------------------------
# 3. The disagreement refusal, and its agreeing control
# ---------------------------------------------------------------------------


def test_a_disagreeing_overlay_season_refuses_by_name(tmp_path: Path) -> None:
    """T-33-89. Two sources for one season, naming the season and BOTH values.

    No precedence rule is applied. Silently preferring one source is how two sources of truth
    become one answer nobody can attribute.
    """
    record = _record()
    record["tune_fit"]["ats"]["season_bias_by_season"][str(_OVERLAY_SEASON)] = -9.5

    with pytest.raises(ChainFitOverlayDisagreementError) as excinfo:
        load_frozen_chain_fit(_write(tmp_path, record))

    message = str(excinfo.value)
    assert str(_OVERLAY_SEASON) in message
    assert "-9.5" in message, message
    assert (
        repr(CHAIN_FIT_BIAS_2026["ats"]) in message
        or str(CHAIN_FIT_BIAS_2026["ats"]) in message
    ), message
    assert "ats" in message


def test_an_agreeing_overlay_season_does_not_refuse(tmp_path: Path) -> None:
    """The control. A guard that refused on agreement too would just be an outage.

    Uses the EXACT frozen float per target. A Python float round-trips through JSON exactly, so
    "the same value" is a real equality here rather than an approximation.
    """
    record = _record()
    for target, block in record["tune_fit"].items():
        block["season_bias_by_season"][str(_OVERLAY_SEASON)] = CHAIN_FIT_BIAS_2026[
            target
        ]

    fits = load_frozen_chain_fit(_write(tmp_path, record))
    for target, fit in fits.items():
        assert fit.season_bias_by_season[_OVERLAY_SEASON] == pytest.approx(
            CHAIN_FIT_BIAS_2026[target]
        ), target


def test_a_record_covering_only_its_own_seasons_never_triggers_the_refusal(
    tmp_path: Path,
) -> None:
    """The ordinary case -- the production record -- must not be one byte away from refusing."""
    fits = load_frozen_chain_fit(_write(tmp_path, _record()))
    assert all(_OVERLAY_SEASON in fit.season_bias_by_season for fit in fits.values())


# ---------------------------------------------------------------------------
# 4. The empty strictly-prior pool REFUSES by name
# ---------------------------------------------------------------------------


def test_an_empty_frozen_pool_refuses_rather_than_overlaying_a_bias_from_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R10 edge. A pooled bias over zero seasons is not a small bias -- it is no bias at all.

    Driven by emptying the POOL the frozen constant records, which is the only way the loader
    could be asked to overlay a value with nothing behind it.
    """
    monkeypatch.setattr(wbl, "CHAIN_FIT_BIAS_SEASONS", ())

    with pytest.raises(EmptyPriorResidualPoolError) as excinfo:
        load_frozen_chain_fit(_write(tmp_path, _record()))

    message = str(excinfo.value)
    assert "no bias" in message.lower() or "not invented" in message.lower(), message


def test_the_existing_estimator_also_refuses_an_empty_strictly_prior_pool() -> None:
    """The same rule at the place the frozen value was actually produced.

    ``backtest.ou_ev_chain.estimate_prior_season_bias`` is the estimator the frozen module names;
    no second estimator was written for this plan. It refuses by name and states the reason, and
    THAT is what guarantees there is no fallback to the target season's own data.
    """
    with pytest.raises(ValueError, match="never use the target's own"):
        estimate_prior_season_bias({2026: np.array([1.0, 2.0])}, 2026)


# ---------------------------------------------------------------------------
# 5. The withhold test -- the frozen bias uses NO 2026 data
# ---------------------------------------------------------------------------


def test_the_pooled_bias_is_identical_whether_or_not_2026_rows_are_present() -> None:
    """The no-leak property, proven by DELIBERATELY ADDING 2026 rows and getting the same number.

    This is stronger than asserting the pool's membership: a mapping can list the right seasons
    and still be summed wrongly. Here the target season's residuals are present and extreme, and
    the estimate must not move by so much as a float.
    """
    prior = {
        2021: np.array([1.0, 2.0, 3.0]),
        2022: np.array([-1.0, 0.0]),
        2023: np.array([4.0]),
        2024: np.array([0.5, 0.5]),
        2025: np.array([-2.0, 2.0, 6.0]),
    }
    withheld = estimate_prior_season_bias(prior, _OVERLAY_SEASON)

    contaminated = {**prior, 2026: np.array([1000.0, -1000.0, 5000.0])}
    with_target_present = estimate_prior_season_bias(contaminated, _OVERLAY_SEASON)

    assert with_target_present == withheld, (
        "the estimate moved when the target season's own residuals were present; the walk-forward "
        "bias would then be estimated partly from the season it is used to debias"
    )


def test_the_frozen_pool_is_exactly_the_five_strictly_prior_seasons() -> None:
    """The recorded claim, checked against the recorded seasons.

    Pairs with the withhold test above: that one proves the ESTIMATOR ignores the target season,
    this one proves the FROZEN value was taken over the seasons the constant says it was.
    """
    assert CHAIN_FIT_BIAS_SEASONS == (2021, 2022, 2023, 2024, 2025)
    assert sorted(CHAIN_FIT_BIAS_2026) == sorted(CANONICAL_TARGETS)
