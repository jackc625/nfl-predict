"""The key-number converter prices a line fairly and knows which final numbers are special.

``models/key_numbers.py`` turns an expected closing line into win, push and lose chances using a
table fitted once on past seasons. These tests build small tables by hand, so each rule of the
conversion can be seen working, then check the committed table against what is known about NFL
finals: 3 and 7 are common margins, and a tie is rare.

Writes only under ``tmp_path``. The one committed file read is the fitted table.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import scripts.fit_key_numbers as fit_script
from backtest.early_arm_2026 import KEY_NUMBER_FIT_SEASONS, KEY_NUMBER_TABLES_PATH
from models.key_numbers import (
    KeyNumberTable,
    OutcomeChance,
    fit_key_number_table,
    load_key_number_tables,
    outcome_chance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

LOW = -40
HIGH = 100
SD = 13.0

# A flat table: no final number is special, so the chances are the plain normal ones.
TABLE = KeyNumberTable(sd=SD, low=LOW, factors=(1.0,) * (HIGH - LOW + 1))


def table_with_factor(number: int, factor: float) -> KeyNumberTable:
    """The flat table, with one final number made ``factor`` times as likely."""
    factors = list(TABLE.factors)
    factors[number - LOW] = factor
    return KeyNumberTable(sd=SD, low=LOW, factors=tuple(factors))


def normal_mass(center: float, sd: float, outcomes: np.ndarray) -> np.ndarray:
    """The normal curve's mass on each whole number, written out independently of the module."""
    return norm.cdf((outcomes + 0.5 - center) / sd) - norm.cdf(
        (outcomes - 0.5 - center) / sd
    )


def expected_games(closing: np.ndarray, sd: float, outcomes: np.ndarray) -> np.ndarray:
    """How many games the normal curve expects on each whole number."""
    return np.sum([normal_mass(float(line), sd, outcomes) for line in closing], axis=0)


def schedule_row(
    *,
    game_type: str = "REG",
    result: float = 3.0,
    total: float = 47.0,
    spread_line: float = 2.5,
    total_line: float = 44.5,
) -> dict[str, object]:
    return {
        "game_type": game_type,
        "result": result,
        "total": total,
        "spread_line": spread_line,
        "total_line": total_line,
    }


COMPLETE_GAMES = [
    schedule_row(),
    schedule_row(result=-7.0, total=38.0, spread_line=-3.0, total_line=41.5),
    schedule_row(result=10.0, total=51.0, spread_line=6.5, total_line=48.0),
]

GAMES_THE_FIT_DROPS = [
    schedule_row(game_type="POST", result=30.0, total=90.0),
    schedule_row(result=float("nan")),
    schedule_row(total=float("nan")),
    schedule_row(spread_line=float("nan")),
    schedule_row(total_line=float("nan")),
]


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def test_the_market_line_is_fair() -> None:
    chance = outcome_chance(TABLE, market_line=44.5, center=44.5, threshold=44.5)
    assert chance.above == pytest.approx(chance.below, abs=1e-9) and chance.at == 0


def test_a_whole_number_line_has_a_push() -> None:
    chance = outcome_chance(TABLE, market_line=3.0, center=3.0, threshold=3.0)
    assert chance.above == pytest.approx(chance.below, abs=1e-9) and chance.at > 0


def test_a_key_number_beside_the_line_does_not_tilt_a_fair_line() -> None:
    table = table_with_factor(3, 3.0)
    chance = outcome_chance(table, market_line=2.5, center=2.5, threshold=2.5)
    assert chance.above == pytest.approx(chance.below, abs=1e-9)


@pytest.mark.parametrize("table", [TABLE, table_with_factor(3, 3.0)])
@pytest.mark.parametrize(
    ("market_line", "center", "threshold"),
    [(3.0, 3.0, 3.0), (2.5, 4.25, 3.0), (44.5, 41.0, 44.5), (-6.5, -9.0, -7.0)],
)
def test_the_three_chances_sum_to_one(
    table: KeyNumberTable, market_line: float, center: float, threshold: float
) -> None:
    chance = outcome_chance(
        table, market_line=market_line, center=center, threshold=threshold
    )
    assert chance.above + chance.at + chance.below == pytest.approx(1.0, abs=1e-12)


def test_moving_the_center_up_raises_above() -> None:
    at_the_line = outcome_chance(TABLE, market_line=44.5, center=44.5, threshold=44.5)
    moved_up = outcome_chance(TABLE, market_line=44.5, center=45.5, threshold=44.5)
    assert moved_up.above > at_the_line.above
    assert moved_up.below < at_the_line.below


def test_a_half_point_across_a_key_number_is_worth_more() -> None:
    table = table_with_factor(3, 3.0)

    def chance(threshold: float) -> OutcomeChance:
        return outcome_chance(table, market_line=4.0, center=4.0, threshold=threshold)

    across = chance(2.5).above - chance(3.5).above
    plain = chance(4.5).above - chance(5.5).above
    assert across > plain


def test_levelling_the_two_sides_keeps_the_push_share() -> None:
    table = table_with_factor(33, 3.0)
    chance = outcome_chance(table, market_line=30.0, center=30.0, threshold=30.0)

    outcomes = np.arange(LOW, HIGH + 1)
    weighted = normal_mass(30.0, SD, outcomes) * np.array(table.factors)
    assert chance.at == pytest.approx(weighted[30 - LOW] / weighted.sum(), abs=1e-12)


@pytest.mark.parametrize(
    "market_line", [LOW, HIGH, LOW - 1.5, HIGH + 1.5, float("nan")]
)
def test_a_line_the_table_cannot_price_is_refused(market_line: float) -> None:
    with pytest.raises(ValueError, match="market_line"):
        outcome_chance(TABLE, market_line=market_line, center=3.0, threshold=3.0)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_a_fit_smooths_the_counts_against_the_normal_curve() -> None:
    actual = np.array([3.0, 3.0, 7.0, -4.0, 10.0])
    closing = np.array([1.0, 2.0, 3.0, 0.0, 4.0])

    table = fit_key_number_table(actual, closing, low=-30, high=30, mirror=False)

    sd = float(np.std(actual - closing, ddof=1))
    expected = expected_games(closing, sd, np.arange(-30, 31))
    assert table.sd == pytest.approx(sd)
    assert table.low == -30
    assert len(table.factors) == 61
    assert table.factors[3 + 30] == pytest.approx((2 + 10) / (expected[3 + 30] + 10))
    assert table.factors[8 + 30] == pytest.approx(10 / (expected[8 + 30] + 10))


def test_a_mirrored_fit_is_symmetric() -> None:
    actual = np.array([3.0, 3.0, 3.0, 7.0, -1.0, 14.0])
    closing = np.array([2.5, 1.0, 3.5, 6.0, -3.0, 9.5])

    mirrored = fit_key_number_table(actual, closing, low=-30, high=30, mirror=True)
    lopsided = fit_key_number_table(actual, closing, low=-30, high=30, mirror=False)

    assert mirrored.factors == mirrored.factors[::-1]
    assert lopsided.factors != lopsided.factors[::-1], "the sample was already even"

    sd = float(np.std(actual - closing, ddof=1))
    expected = expected_games(closing, sd, np.arange(-30, 31))
    pooled_expected = expected[3 + 30] + expected[-3 + 30]
    assert mirrored.factors[3 + 30] == pytest.approx((3 + 10) / (pooled_expected + 10))
    assert mirrored.factors[0 + 30] == pytest.approx(10 / (2 * expected[0 + 30] + 10))


def test_a_fit_refuses_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_key_number_table([3.0, 7.0], [2.5], low=-30, high=30, mirror=False)


def test_a_mirrored_fit_refuses_a_range_that_is_not_centered_on_zero() -> None:
    with pytest.raises(ValueError, match="mirror"):
        fit_key_number_table([3.0, 7.0], [2.5, 6.0], low=-30, high=40, mirror=True)


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


def test_the_fit_uses_only_complete_regular_season_games() -> None:
    tables = fit_script.fit_tables(pd.DataFrame(COMPLETE_GAMES + GAMES_THE_FIT_DROPS))

    kept = pd.DataFrame(COMPLETE_GAMES)
    assert tables["ats"] == fit_key_number_table(
        kept["result"], kept["spread_line"], low=-70, high=70, mirror=True
    )
    assert tables["ou"] == fit_key_number_table(
        kept["total"], kept["total_line"], low=0, high=120, mirror=False
    )


def test_the_script_fits_the_recorded_seasons_and_writes_the_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: list[list[int]] = []

    def fake_load_schedules(seasons: list[int]) -> pd.DataFrame:
        requested.append(seasons)
        return pd.DataFrame(COMPLETE_GAMES)

    output = tmp_path / "tables.json"
    monkeypatch.setattr(fit_script, "load_schedules", fake_load_schedules)
    monkeypatch.setattr(fit_script, "KEY_NUMBER_TABLES_PATH", str(output))

    assert fit_script.main() == 0

    first, last = KEY_NUMBER_FIT_SEASONS
    assert requested == [list(range(first, last + 1))]

    text = output.read_text(encoding="ascii")
    payload = json.loads(text)
    assert payload["fit_seasons"] == [first, last]
    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert load_key_number_tables(output) == fit_script.fit_tables(
        pd.DataFrame(COMPLETE_GAMES)
    )

    printed = capsys.readouterr().out.splitlines()
    assert [line.split("=")[0] for line in printed] == ["ats sd", "ou sd"]


# ---------------------------------------------------------------------------
# The committed tables
# ---------------------------------------------------------------------------


def test_the_committed_tables_know_the_key_numbers() -> None:
    tables = load_key_number_tables()
    margin = tables["ats"]
    assert margin.factors[3 - margin.low] > 2.0
    assert margin.factors[7 - margin.low] > 1.4
    assert margin.factors[0 - margin.low] < 0.5
    assert 12.0 < margin.sd < 15.0 and 12.0 < tables["ou"].sd < 15.0


def test_the_committed_file_names_its_seasons_and_ranges() -> None:
    path = REPO_ROOT / KEY_NUMBER_TABLES_PATH
    assert json.loads(path.read_text(encoding="ascii"))["fit_seasons"] == list(
        KEY_NUMBER_FIT_SEASONS
    )

    tables = load_key_number_tables(path)
    assert (tables["ats"].low, len(tables["ats"].factors)) == (-70, 141)
    assert (tables["ou"].low, len(tables["ou"].factors)) == (0, 121)
    assert tables["ats"].factors == tables["ats"].factors[::-1]
