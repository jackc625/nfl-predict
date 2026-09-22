"""p332_ extra step 8c: normalization statistics are ordered by LOCK INSTANT.

OWNER RULING 2026-09-22 ("Order by lock time", deferred-items.md). A row's expanding
normalization mean and standard deviation are computed over every row of its season whose
game LOCK is at or before that row's lock. At-lock is admissible (D33.2-01), so every game
sharing one lock -- a whole Sunday slate sharing its Saturday 18:00 ET lock -- sees the
others and they share ONE statistic.

WHAT WAS WRONG. ``expanding_normalize`` sorted each season by ``(season, week)``, which is
an UNSTABLE sort with no tie-break inside a week, and then took an expanding window over
that order. Two consequences, both leaks of a kind:

* a same-week game whose lock is LATER entered an earlier-locking game's statistic -- a
  Monday night game's day-before forecast, issued on the Sunday, reaching a Sunday game
  whose lock was Saturday 18:00;
* two games played the same afternoon got DIFFERENT statistics depending on the arbitrary
  order their rows happened to arrive in.

WHAT MUST NOT CHANGE, and is asserted here beside the fix: the row set and the row ORDER of
the returned frame, and the meaning of ``min_periods``, the prior-season bootstrap, the
neutral-0.0 fallback, ``preserve_missing_cols`` and ``preserve_level_cols``. Only WHICH rows
form each statistic changes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from features import normalization
from features.normalization import expanding_normalize


def _refusal() -> type[Exception]:
    """The named refusal, resolved at CALL time so this module still imports without it.

    An import-time failure is not a red test, it is a broken module: it would take the
    whole file down and say nothing about the behaviour under test.
    """
    error = getattr(normalization, "LockOrderedStatisticUnavailableError", None)
    assert error is not None, (
        "features.normalization does not declare "
        "LockOrderedStatisticUnavailableError, so a caller that cannot supply a lock "
        "has no named refusal to fail closed with"
    )
    return error


#: Two locks one day apart, in the nanoseconds the builder hands the normalizer.
SATURDAY = 1_700_000_000_000_000_000
SUNDAY = SATURDAY + 86_400_000_000_000

#: Six games at each lock, INTERLEAVED in the input frame so that a later-locking row
#: physically precedes an earlier-locking one. Under the retired week-order window that
#: interleaving is exactly what let the Monday game into the Sunday game's statistic.
_LOCKS = [SUNDAY, SATURDAY] * 6


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"2024_W01_{i:02d}" for i in range(len(values))],
            "season": 2024,
            "week": 1,
            "metric": values,
        }
    )


def _locks(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(_LOCKS[: len(frame)], index=frame.index, name="lock_ns")


def _accepts_locks() -> bool:
    """Whether ``expanding_normalize`` takes a lock yet.

    The behavioural tests below must fail on their OWN assertions before the fix, not on
    a TypeError from an argument the function does not yet have -- a red test that never
    reached the behaviour proves nothing about it. So the lock is passed only when the
    parameter exists; without it the call exercises the retired week-order window, which
    is exactly what the target assertion is about.
    """
    return "row_locks" in inspect.signature(expanding_normalize).parameters


def _normalize(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    options: dict[str, object] = {"feature_cols": ["metric"], "min_periods": 2}
    if _accepts_locks():
        options["row_locks"] = _locks(frame)
    options.update(kwargs)
    return expanding_normalize(frame, **options)


def _by_game(out: pd.DataFrame) -> pd.Series:
    return out.set_index("game_id")["metric"].sort_index()


_BASE = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]


def _early_rows(out: pd.DataFrame) -> pd.Series:
    """The SATURDAY-locking games, keyed by game id (odd positions in the input)."""
    early = {f"2024_W01_{i:02d}" for i in range(len(_LOCKS)) if _LOCKS[i] == SATURDAY}
    return _by_game(out).loc[sorted(early)]


def _late_rows(out: pd.DataFrame) -> pd.Series:
    late = {f"2024_W01_{i:02d}" for i in range(len(_LOCKS)) if _LOCKS[i] == SUNDAY}
    return _by_game(out).loc[sorted(late)]


class TestASameWeekLaterLockCannotReachAnEarlierLock:
    def test_planting_a_later_locking_games_value_moves_no_earlier_locking_row(
        self,
    ) -> None:
        """The target assertion.

        Position 0 is a SUNDAY-locking game and position 1 a SATURDAY-locking one, so
        under the week-order window position 0's value was inside position 1's expanding
        mean. It must not be inside it now: a Sunday game's information did not exist at
        Saturday 18:00.
        """
        calm = _normalize(_frame(list(_BASE)))
        planted_values = list(_BASE)
        planted_values[0] = 5000.0
        planted = _normalize(_frame(planted_values))

        pd.testing.assert_series_equal(_early_rows(calm), _early_rows(planted))

    def test_planting_an_earlier_locking_games_value_does_move_a_later_locking_row(
        self,
    ) -> None:
        """The positive control: the fix must not amount to isolating every row."""
        calm = _normalize(_frame(list(_BASE)))
        planted_values = list(_BASE)
        planted_values[1] = 5000.0
        planted = _normalize(_frame(planted_values))

        assert not np.allclose(
            _late_rows(calm).to_numpy(), _late_rows(planted).to_numpy()
        ), (
            "a Saturday-locking game's value did not reach the Sunday-locking games' "
            "statistic -- lock ordering must ADMIT earlier locks, not isolate rows"
        )


class TestGamesSharingALockShareOneStatistic:
    def test_the_result_does_not_depend_on_the_input_row_order(self) -> None:
        """Two games played the same afternoon must not differ by row order."""
        frame = _frame(list(_BASE))
        shuffled_positions = [11, 3, 6, 0, 9, 2, 5, 8, 1, 10, 4, 7]
        shuffled = frame.iloc[shuffled_positions].reset_index(drop=True)
        shuffled_locks = pd.Series(
            [_LOCKS[p] for p in shuffled_positions],
            index=shuffled.index,
            name="lock_ns",
        )

        straight = _normalize(frame)
        rotated_options: dict[str, object] = {
            "feature_cols": ["metric"],
            "min_periods": 2,
        }
        if _accepts_locks():
            rotated_options["row_locks"] = shuffled_locks
        rotated = expanding_normalize(shuffled, **rotated_options)

        pd.testing.assert_series_equal(_by_game(straight), _by_game(rotated))

    def test_every_row_at_one_lock_took_the_same_mean_and_standard_deviation(
        self,
    ) -> None:
        """Recovered from the transform itself: z = (x - m) / s is affine in x.

        Two rows sharing a statistic satisfy ``(x1 - x2) / (z1 - z2) == s`` for the same
        ``s``, so a single implied ``s`` across the whole tie group is the assertion.
        """
        frame = _frame(list(_BASE))
        out = _normalize(frame)
        merged = frame.merge(out, on="game_id", suffixes=("_raw", "_norm"))
        merged["lock"] = _LOCKS

        for lock, group in merged.groupby("lock"):
            raw = group["metric_raw"].to_numpy()
            norm = group["metric_norm"].to_numpy()
            implied = np.diff(raw) / np.diff(norm)
            assert np.allclose(implied, implied[0]), (lock, implied)


class TestTheLockIsRequiredAndFailsClosedByName:
    def test_a_caller_that_supplies_no_lock_is_refused(self) -> None:
        with pytest.raises(_refusal()):
            expanding_normalize(_frame(list(_BASE)), feature_cols=["metric"])

    def test_a_null_lock_is_refused(self) -> None:
        frame = _frame(list(_BASE))
        locks = _locks(frame).astype("float64")
        locks.iloc[3] = np.nan
        with pytest.raises(_refusal()):
            expanding_normalize(
                frame, feature_cols=["metric"], min_periods=2, row_locks=locks
            )


class TestTheUnchangedContract:
    def test_the_row_set_and_row_order_are_preserved(self) -> None:
        frame = _frame(list(_BASE))
        out = _normalize(frame)

        assert out["game_id"].tolist() == frame["game_id"].tolist()
        assert len(out) == len(frame)

    def test_min_periods_still_means_what_it_meant(self) -> None:
        """Below min_periods with no prior-season bootstrap, the answer is the neutral 0.0."""
        frame = _frame(list(_BASE))
        out = _normalize(frame, min_periods=99)

        assert (out["metric"] == 0.0).all()

    def test_the_prior_season_bootstrap_is_still_used_below_min_periods(self) -> None:
        frame = _frame(list(_BASE))
        out = _normalize(
            frame, min_periods=99, prior_season_stats={"metric": (10.0, 2.0)}
        )

        assert not (out["metric"] == 0.0).all()
        assert out.loc[out["game_id"] == "2024_W01_00", "metric"].iloc[0] == (
            pytest.approx(0.0)
        )

    def test_a_level_preserved_column_is_not_transformed_at_all(self) -> None:
        frame = _frame(list(_BASE))
        out = _normalize(frame, preserve_level_cols=("metric",))

        pd.testing.assert_series_equal(out["metric"], frame["metric"])

    def test_an_absent_value_in_a_preserved_column_comes_back_absent(self) -> None:
        values = list(_BASE)
        values[4] = float("nan")
        frame = _frame(values)
        out = _normalize(frame, preserve_missing_cols=("metric",))

        assert bool(out.loc[out["game_id"] == "2024_W01_04", "metric"].isna().iloc[0])
