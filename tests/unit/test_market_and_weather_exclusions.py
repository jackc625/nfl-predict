"""Both registry-driven gold exclusions, in one module: the betting line and the weather.

SPEC R6 and R7. Two families leave gold because the ONE registry says so, and they leave
through the SAME mechanism -- ``backtest.signal_lift._GROUP_PREDICATE`` names them and
``scripts.build_features._enforce_groups_dropped`` removes them before imputation. They are
tested together because a second drop mechanism, or a second list of names, is the defect
both are guarding against.

* R7, the BETTING LINE (Plan 33.2-19, p332_ rung 9, D33.2-03): ``snapshot_spread``,
  ``snapshot_total``, ``snapshot_ml_prob_home_fair``, ``spread_movement``,
  ``total_movement``. All three deployed models selected one, and for 2018-2025 those are
  CLOSING lines, which did not exist at the lock. No target needs one: WP predicts the
  winner, ATS the margin, O/U total points. The line is used afterwards, to price a bet.
* R6, the WEATHER INPUTS NO FORECAST SUPPLIES (Plan 33.2-12, rung 4): ``precip_mm`` and
  ``raw_precip_mm``. The archived day-before bulletins carry a probability and an ordinal
  category, never a millimetre amount.

THE GOLD-ABSENCE NODE IS EVALUATED AT TASK 3, not here: ``test_no_excluded_column_reaches_gold``
is authored in this module and DESELECTED by Task 2's own gate, because the rung-9 rebuild that
takes the market columns out of gold is Task 3. Its own docstring says so, so a reader who runs
it alone before the rebuild knows why it fails.

FOUR STRUCTURAL CONTROLS, per ``tests/unit/test_freeze_parse_single_source.py:197-233``:
non-vacuity (the declared sets are non-empty and their lengths asserted), the assertion
itself, a planted violation (``home_snapshot_spread`` IS matched -- the suffix rule works
through a prefix), and no false positive (``rolling_total_epa``, ``total_points`` are NOT).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import (
    _GROUP_PREDICATE,
    ALL_REGISTERED_GROUPS,
    GROUPS,
    group_columns,
)
from scripts.build_features import (
    GOLD_DROPPED_GROUPS,
    FeatureMatrixBuilder,
    drop_feature_group,
)

GOLD_DIR = Path("data/gold")
GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")

#: The five betting-line columns (SPEC R7).
MARKET_COLUMNS: tuple[str, ...] = (
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
)

#: The two weather inputs no forecast supplies (SPEC R6).
WEATHER_UNSUPPLIED_COLUMNS: tuple[str, ...] = ("precip_mm", "raw_precip_mm")

#: Columns that must NEVER be matched: a legitimate target, a legitimate feature, and the
#: two near-misses a bare substring rule would sweep up.
NEAR_MISSES: tuple[str, ...] = (
    "total_points",
    "rolling_total_epa",
    "precip_prob",
    "precip_impact_score",
)


def _header(*columns: str) -> pd.DataFrame:
    """A header-only frame: the predicate is asked about NAMES, never about data."""
    return pd.DataFrame(columns=pd.Index(columns))


class TestTheDeclaredSetsAreNonVacuous:
    """CONTROL 1: the sets this module asserts about are real and the right size."""

    def test_the_market_set_has_five_names(self) -> None:
        assert len(MARKET_COLUMNS) == 5
        assert len(set(MARKET_COLUMNS)) == 5

    def test_the_weather_set_has_two_names(self) -> None:
        assert len(WEATHER_UNSUPPLIED_COLUMNS) == 2
        assert len(set(WEATHER_UNSUPPLIED_COLUMNS)) == 2


class TestThePredicatesMatchExactlyTheirDeclaredSets:
    """CONTROL 2: the assertion. Exact membership, so a silent join or leave fails."""

    def test_the_market_predicate_matches_exactly_the_five(self) -> None:
        frame = _header(*MARKET_COLUMNS, *WEATHER_UNSUPPLIED_COLUMNS, *NEAR_MISSES)
        assert set(group_columns(frame, "market")) == set(MARKET_COLUMNS)

    def test_the_weather_predicate_matches_exactly_the_two(self) -> None:
        frame = _header(*MARKET_COLUMNS, *WEATHER_UNSUPPLIED_COLUMNS, *NEAR_MISSES)
        assert set(group_columns(frame, "weather_unsupplied")) == set(
            WEATHER_UNSUPPLIED_COLUMNS
        )

    def test_the_two_families_are_disjoint(self) -> None:
        assert not set(MARKET_COLUMNS) & set(WEATHER_UNSUPPLIED_COLUMNS)


class TestTheSuffixRuleWorksThroughAPrefixAndNotThroughASubstring:
    """CONTROLS 3 and 4: a planted violation, and no false positive."""

    @pytest.mark.parametrize(
        "column",
        ["home_snapshot_spread", "away_snapshot_total", "home_raw_precip_mm"],
    )
    def test_a_prefixed_column_is_still_matched(self, column: str) -> None:
        frame = _header(column)
        matched = set(group_columns(frame, "market")) | set(
            group_columns(frame, "weather_unsupplied")
        )
        assert matched == {column}

    @pytest.mark.parametrize("column", NEAR_MISSES)
    def test_a_near_miss_is_never_matched(self, column: str) -> None:
        frame = _header(column)
        assert group_columns(frame, "market") == []
        assert group_columns(frame, "weather_unsupplied") == []


class TestTheRegistryIsTheOneAnswer:
    """``market`` is registered for DROPPING, not admitted to ``GROUPS`` for screening."""

    def test_groups_is_exactly_the_phase_28_triple(self) -> None:
        """EXACT TUPLE EQUALITY. A set assertion proves membership, not the invariant.

        Adding a group to ``GROUPS`` silently changes the Phase-28 baseline, which that
        module's own docstring explains at length: the baseline was never "gold minus
        these three names", it was "gold minus every signal column we know about".
        """
        assert tuple(GROUPS) == ("injury", "snap", "situational")

    def test_market_is_registered_but_not_screened(self) -> None:
        assert "market" in _GROUP_PREDICATE
        assert "market" not in GROUPS

    def test_market_joins_the_derived_baseline_pin_automatically(self) -> None:
        """``ALL_REGISTERED_GROUPS`` is DERIVED, which is why nothing else had to change."""
        assert "market" in ALL_REGISTERED_GROUPS
        assert set(ALL_REGISTERED_GROUPS) == set(_GROUP_PREDICATE)

    def test_one_drop_mechanism_serves_all_three_removed_groups(self) -> None:
        assert tuple(GOLD_DROPPED_GROUPS) == (
            "line_movement",
            "weather_unsupplied",
            "market",
        )


class TestTheProductionPathRemovesARestoredMergeSeam:
    """A market column that came back is removed BEFORE imputation, on the real path.

    The end state of gold is clean whether or not ``market`` is wired into the drop --
    the merge block is deleted either way -- so only pushing a market-BEARING frame
    through the build's own enforcement step, with the build's OWN group argument, shows
    the restoration guard is live. The tuple is read from the build module rather than
    re-typed here: a hand-typed tuple would pass with ``market`` missing from the real
    call, which is the gap this case exists to close.
    """

    @staticmethod
    def _restored_merge_frame() -> pd.DataFrame:
        """The shape a restored merge block would produce, plus the two near-misses."""
        return pd.DataFrame(
            {
                "game_id": ["2024_W01_DET@KC", "2024_W02_BUF@NYJ"],
                **{column: [1.0, 2.0] for column in MARKET_COLUMNS},
                "total_points": [45.0, 38.0],
                "rolling_total_epa": [0.1, -0.2],
            }
        )

    def test_the_five_are_removed_and_the_near_misses_survive(self) -> None:
        frame = self._restored_merge_frame()

        out = FeatureMatrixBuilder._enforce_groups_dropped(frame, GOLD_DROPPED_GROUPS)

        assert not set(MARKET_COLUMNS) & set(out.columns)
        assert {"total_points", "rolling_total_epa", "game_id"} <= set(out.columns)
        assert len(out) == len(frame)

    def test_the_empty_seam_returns_the_frame_unchanged_and_does_not_raise(
        self,
    ) -> None:
        """The deliberate asymmetry, both halves together.

        A build whose seams are gone was never asking for a drop, so the enforcement step
        must NOT raise -- while ``drop_feature_group`` itself refuses a zero match, because
        a drop asked to remove something and removing nothing is indistinguishable
        downstream from one that worked.
        """
        frame = pd.DataFrame(
            {
                "game_id": ["2024_W01_DET@KC"],
                "total_points": [45.0],
                "rolling_total_epa": [0.1],
            }
        )

        out = FeatureMatrixBuilder._enforce_groups_dropped(frame, GOLD_DROPPED_GROUPS)

        pd.testing.assert_frame_equal(out, frame)
        with pytest.raises(ValueError):
            drop_feature_group(frame, "market")


class TestAddingALineToASelectableGroupFails:
    """SPEC R7's acceptance line, as a REAL failing assertion rather than a comment.

    A synthetic gold frame carrying ``snapshot_spread`` inside a SELECTABLE feature group
    -- one of the three in ``GROUPS``, the set a screen may put into a model -- is FLAGGED
    here. The flag is the group registry itself: a column that a selectable group's
    predicate matches AND the market predicate matches is a betting line about to be
    screened into a model.
    """

    @staticmethod
    def _lines_inside_selectable_groups(
        frame: pd.DataFrame, groups: tuple[str, ...] = GROUPS
    ) -> list[str]:
        """Columns a SELECTABLE group claims that are also betting lines."""
        market = set(group_columns(frame, "market"))
        selectable = {
            column for group in groups for column in group_columns(frame, group)
        }
        return sorted(market & selectable)

    def test_today_no_selectable_group_claims_a_line(self) -> None:
        frame = _header(
            *MARKET_COLUMNS,
            "home_qb_out_flag",
            "home_off_bye",
            "home_snap_continuity",
        )
        # Non-vacuity in BOTH directions: the frame really carries lines, and really
        # carries columns the selectable groups claim.
        assert group_columns(frame, "market")
        assert any(group_columns(frame, group) for group in GROUPS)

        assert self._lines_inside_selectable_groups(frame) == []

    def test_a_planted_line_inside_a_selectable_group_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PLANTED VIOLATION: teach a selectable group to claim a line; the check fails.

        This is the acceptance line as a real failing assertion. The families are
        suffix-disjoint today, so the only way a line gets INTO a selectable group is for
        someone to widen that group's predicate -- which is exactly what is planted here.
        """
        monkeypatch.setitem(
            _GROUP_PREDICATE,
            "situational",
            lambda col: col.lower().endswith(("off_bye", "snapshot_spread")),
        )
        frame = _header("snapshot_spread", "home_off_bye")

        assert self._lines_inside_selectable_groups(frame) == ["snapshot_spread"], (
            "a betting line sitting inside a selectable feature group was not flagged; "
            "the screen would put it into a model"
        )

    def test_making_the_market_group_itself_selectable_is_flagged(self) -> None:
        """The other way a line becomes selectable: ``GROUPS`` gains ``market``."""
        frame = _header(*MARKET_COLUMNS)

        flagged = self._lines_inside_selectable_groups(
            frame, groups=(*GROUPS, "market")
        )

        assert flagged == sorted(MARKET_COLUMNS)


@pytest.mark.skipif(
    not all((GOLD_DIR / f"{m}.parquet").is_file() for m in GOLD_MATRICES),
    reason="the gold matrices are absent on this checkout (data/ is gitignored)",
)
def test_no_excluded_column_reaches_gold() -> None:
    """None of the seven excluded columns appears in any of the three gold matrices.

    EVALUATED AT TASK 3, NOT AT TASK 2, and deselected by name in Task 2's gate. The two
    weather columns left gold at rung 4; the five market columns leave at RUNG 9, which is
    Task 3's rebuild. MEASURED before it: all three matrices still carried
    ``snapshot_spread``, ``snapshot_total``, ``snapshot_ml_prob_home_fair``,
    ``spread_movement`` and ``total_movement``. Running this node alone before that rebuild
    therefore FAILS, and the cheapest way to make it pass early would be to weaken the very
    assertion SPEC R7 requires -- which is why the split is stated here rather than left to
    be discovered.
    """
    excluded = {*MARKET_COLUMNS, *WEATHER_UNSUPPLIED_COLUMNS}
    assert len(excluded) == 7, "non-vacuity"
    for matrix in GOLD_MATRICES:
        columns = set(pd.read_parquet(GOLD_DIR / f"{matrix}.parquet").columns)
        assert not columns & excluded, (matrix, sorted(columns & excluded))
        # No false negative: the near-misses are still there to be confused with.
        assert "total_points" in columns or matrix != "features_ou", matrix


@pytest.mark.skipif(
    not all((GOLD_DIR / f"{m}.parquet").is_file() for m in GOLD_MATRICES),
    reason="the gold matrices are absent on this checkout (data/ is gitignored)",
)
def test_the_predicate_finds_nothing_left_to_drop_in_gold() -> None:
    """The registry's own answer on the live matrices, not a hand-written name list.

    Evaluated with the node above, at Task 3. ``group_columns`` is what performed the
    removal, so asking it about the built matrices is the check that cannot drift from
    the mechanism.
    """
    for matrix in GOLD_MATRICES:
        frame = pd.read_parquet(GOLD_DIR / f"{matrix}.parquet")
        assert group_columns(frame, "market") == [], matrix
        assert group_columns(frame, "weather_unsupplied") == [], matrix
