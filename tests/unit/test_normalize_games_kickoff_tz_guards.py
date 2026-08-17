"""WR-09 regression: the kickoff-tz normalization tool's guard rails.

``scripts/normalize_games_kickoff_tz.py`` repairs 1,926 stale ``games.kickoff_et``
rows whose ET wall clock was stored under a UTC label. It is a one-shot data
surgery tool whose entire value is that its invariants hold, so a guard rail that
CANNOT fire, or that raises before it can check, is worse than no guard rail --
it reads as assurance.

The Phase-29 review found three:

1. The tz-NAIVE branch of ``normalize_kickoffs`` was broken. It read the naive
   column as if it were UTC and then assigned a tz-AWARE corrected Series back
   into a tz-NAIVE column; pandas upcasts to object or raises, so the frame that
   would then be WRITTEN is not the frame the invariants assume. The branch
   exists precisely for a copy typed differently from today's two -- i.e. it
   fires the first time it is actually needed.

2. ``dst_correlation_holds`` assumed a unique index: it looked values up with
   ``.loc[label]`` using labels from ``groupby(...).groups``. On a duplicated
   index -- any ``pd.concat`` without ``ignore_index=True`` -- ``.loc`` returns a
   Series and ``Series in (9, 10)`` raises. This function is the SOLE staleness
   discriminator and is re-run on the reloaded copy inside ``apply()``, i.e.
   after the write.

3. The post-write cohort-size assertion was an ``or`` over both copies' expected
   counts. Both are 1,926 today, so the disjunction was vacuous, and it could not
   say which copy ``load_dataframe`` resolved -- the very N-01 ambiguity the
   module docstring cites as the reason single-copy fixes are dangerous.

These tests build synthetic frames; they never touch the real silver.
"""

import pandas as pd
import pytest

from scripts.normalize_games_kickoff_tz import (
    SHIFTED_COHORT_CREATED_AT,
    NormalizeInvariantError,
    dst_correlation_holds,
    normalize_kickoffs,
)

# The tool identifies the affected rows by their ingest ``created_at`` stamp, so
# a fixture must carry the REAL cohort stamp to be processed at all.
_STALE_CREATED_AT = SHIFTED_COHORT_CREATED_AT


def _season_rows(season: int, tz: str | None = "UTC") -> pd.DataFrame:
    """A season with 1 PM ET kickoffs on both sides of the November DST boundary.

    Stored as TRUE instants (the corrected shape), so the DST correlation holds:
    13:00 EDT is 17:00Z in October and 13:00 EST is 18:00Z in December.
    """
    kickoffs = [
        f"{season}-09-14T17:00:00Z",
        f"{season}-10-12T17:00:00Z",
        f"{season}-12-14T18:00:00Z",
        f"{season + 1}-01-04T18:00:00Z",
    ]
    frame = pd.DataFrame(
        {
            "game_id": [f"{season}_{i}" for i in range(len(kickoffs))],
            "season": season,
            "week": [1, 5, 15, 18],
            "kickoff_et": pd.to_datetime(kickoffs, utc=True),
            "created_at": _STALE_CREATED_AT,
        }
    )
    if tz is None:
        frame["kickoff_et"] = frame["kickoff_et"].dt.tz_localize(None)
    elif tz != "UTC":
        frame["kickoff_et"] = frame["kickoff_et"].dt.tz_convert(tz)
    return frame


def _stale_season_rows(season: int, tz: str | None = "UTC") -> pd.DataFrame:
    """The BROKEN shape: a 13:00 ET wall clock stored under a UTC label.

    Stored hour is 13:00 on both sides of the November boundary, so the DST
    correlation fails and ``cohort_is_stale`` is True -- which is what makes
    ``normalize_kickoffs`` proceed past its idempotency early-return and reach
    the timezone handling under test.
    """
    frame = _season_rows(season)
    frame["kickoff_et"] = pd.to_datetime(
        [
            f"{season}-09-14T13:00:00Z",
            f"{season}-10-12T13:00:00Z",
            f"{season}-12-14T13:00:00Z",
            f"{season + 1}-01-04T13:00:00Z",
        ],
        utc=True,
    )
    if tz is None:
        frame["kickoff_et"] = frame["kickoff_et"].dt.tz_localize(None)
    elif tz != "UTC":
        frame["kickoff_et"] = frame["kickoff_et"].dt.tz_convert(tz)
    return frame


class TestNaiveColumnIsRefused:
    """(1) Refusing beats half-handling when the answer cannot be known."""

    def test_a_timezone_naive_kickoff_column_raises(self) -> None:
        """The correction turns on WHICH label the stale wall clock carries.

        An unlabelled column does not say, so guessing would silently write a
        wrong instant into the silver both feature builders read. The fixture is
        the STALE shape, so the idempotency early-return does not mask the branch.
        """
        naive = _stale_season_rows(2023, tz=None)

        with pytest.raises(NormalizeInvariantError, match="timezone-naive"):
            normalize_kickoffs(naive)

    def test_the_refusal_names_the_two_known_shapes(self) -> None:
        """An operator hitting this must know what to do next."""
        naive = _stale_season_rows(2023, tz=None)

        with pytest.raises(NormalizeInvariantError) as exc:
            normalize_kickoffs(naive)

        assert "UTC" in str(exc.value)
        assert "America/New_York" in str(exc.value)

    @pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
    def test_a_labelled_column_is_still_corrected(self, tz: str) -> None:
        """Positive control: both real shapes still go through and are FIXED.

        A refusal that also blocked the two live shapes would be useless, so this
        asserts the correction actually lands: the 13:00 wall clock becomes a true
        instant (17:00Z in October, 18:00Z in December) and the frame stops
        reading as stale.
        """
        frame = _stale_season_rows(2023, tz=tz)

        out = normalize_kickoffs(frame)

        assert len(out) == len(frame)
        assert out["kickoff_et"].dt.tz is not None
        assert str(out["kickoff_et"].dt.tz) == str(frame["kickoff_et"].dt.tz)

        stored = pd.to_datetime(out["kickoff_et"], utc=True)
        assert stored.dt.hour.tolist() == [17, 17, 18, 18]

    def test_the_correction_is_idempotent(self) -> None:
        """Re-running must not shift the same rows a SECOND time."""
        once = normalize_kickoffs(_stale_season_rows(2023))
        twice = normalize_kickoffs(once)

        pd.testing.assert_frame_equal(once, twice)


class TestDstCorrelationSurvivesADuplicatedIndex:
    """(2) The sole staleness discriminator must not raise on frame shape."""

    def test_a_duplicated_index_does_not_raise(self) -> None:
        """``pd.concat`` without ``ignore_index=True`` is the everyday producer.

        Pre-fix this raised ``TypeError`` from ``Series in (9, 10)``, and it does
        so inside ``apply()`` AFTER the write has already landed.
        """
        duplicated = pd.concat([_season_rows(2022), _season_rows(2023)])
        assert not duplicated.index.is_unique, "fixture must have a duplicate index"

        result = dst_correlation_holds(duplicated)

        assert set(result) == {"2022", "2023"}

    def test_the_verdict_matches_the_unique_index_case(self) -> None:
        """Index shape must not change the ANSWER, only stop it from crashing."""
        rows = pd.concat([_season_rows(2022), _season_rows(2023)])

        assert dst_correlation_holds(rows) == dst_correlation_holds(
            rows.reset_index(drop=True)
        )

    def test_true_instants_pass_the_correlation(self) -> None:
        """Positive control: 13:00 ET shifts 17:00Z -> 18:00Z across the boundary."""
        assert dst_correlation_holds(_season_rows(2023)) == {"2023": True}

    def test_a_stale_wall_clock_fails_the_correlation(self) -> None:
        """The discriminator must still discriminate.

        An ET wall clock mislabelled UTC stores 18:00Z all year, so the stored
        hour does NOT shift across the DST boundary -- which is the whole tell.
        """
        stale = _season_rows(2023)
        stale["kickoff_et"] = pd.to_datetime(
            [
                "2023-09-14T13:00:00Z",
                "2023-10-12T13:00:00Z",
                "2023-12-14T13:00:00Z",
                "2024-01-04T13:00:00Z",
            ],
            utc=True,
        )

        assert dst_correlation_holds(stale) == {"2023": False}
