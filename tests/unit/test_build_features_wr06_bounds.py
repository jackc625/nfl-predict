"""WR-06: imputation medians and winsorization bounds are fitted on PRIOR seasons only.

``handle_missing_data_and_outliers`` used to fit every distributional statistic on
the WHOLE multi-season frame:

* the game-level imputation median (``processed_df[col].median()``),
* the q01/q99 winsorization bounds (``processed_df[col].quantile(...)``),
* and -- named by neither the SPEC nor CONTEXT -- the two last-resort whole-frame
  medians inside ``_impute_team_features``, which is the branch every ``home_*`` /
  ``away_*`` column takes and therefore the large majority of features.

All three let a FUTURE season change a PAST feature value. That is the temporal
boundary this whole phase turns on, and the third surface is load-bearing for
SPEC R2 specifically: any 2021-2024 row that reaches those fallbacks WILL move when
the N-01 re-sync adds 2025 rows, failing the byte-identity control for a cause that
has nothing to do with the two surfaces D30-16 names.

The contract these tests pin:

* For season Y, every fitted statistic comes from seasons strictly before Y.
* The earliest data-bearing season self-fits, under a machine-readable flag
  (``FeatureMatrixBuilder.self_fit_seasons``), and so does any season whose
  strictly-prior slice holds too few non-null values to support a bound.
* The CR-02 discrete-indicator exemption is evaluated ONCE PER COLUMN, over the
  whole frame, so a globally-continuous column that happens to be constant within
  one season is not misclassified as an indicator in that season.
* Missing-handling and outlier-handling stay independent.
* The WR-10 neutral-default guard is untouched: it is removed at rung 3, not here.

The residual this does NOT close, deliberately: within-season lookahead. Season Y's
week-1 bound still sees season Y's week 18 whenever Y self-fits, and
``_impute_team_features``' team mean and season mean remain within-season. D30-16
accepts both; they cannot be moved by adding a LATER season's rows, which is what
SPEC R2 asserts.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import FeatureMatrixBuilder

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The committed pre-Phase-30 gold fixture (Plan 30-03). Read for its SEASON SPAN
# only -- never written, never regenerated. It is used here because it is TRACKED,
# so the "real gold season span" assertion still asserts something on a checkout
# that has no data/ lake.
_GOLD_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "gold" / "features_ats_pre_phase30.parquet"
)


@pytest.fixture(scope="module")
def builder() -> FeatureMatrixBuilder:
    return FeatureMatrixBuilder()


def _season_block(season: int, n: int, **columns) -> pd.DataFrame:
    """One season's worth of rows, shaped like the real combined matrix."""
    frame = pd.DataFrame(
        {
            "game_id": [f"{season}_{i:04d}" for i in range(n)],
            "season": season,
            "week": (np.arange(n) % 18) + 1,
            "home_team": "KC",
            "away_team": "DET",
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def _stack(*blocks: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(blocks, ignore_index=True)


def _self_fit_seasons(builder: FeatureMatrixBuilder) -> list[int]:
    """The distinct seasons recorded in the builder's machine-readable self-fit log."""
    return sorted({s for seasons in builder.self_fit_seasons.values() for s in seasons})


def _method_calls(method, receiver: str, attr: str) -> list[str]:
    """Return every ``{receiver}[...].{attr}(...)`` call parsed out of *method*.

    Structural, never textual: a guard that greps a method's own source matches the
    docstring in which that method explains what it removed.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Subscript)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == receiver
    ]


def _real_gold_seasons() -> list[int]:
    frame = pd.read_parquet(_GOLD_FIXTURE, columns=["season"])
    return sorted(int(season) for season in frame["season"].dropna().unique())


class TestBoundsComeFromStrictlyPriorSeasons:
    """The named WR-06 contract, on the two surfaces D30-16 calls out."""

    def test_a_later_seasons_bounds_are_fitted_on_the_earlier_season(
        self, builder
    ) -> None:
        """Season 2's clip lands on season 1's scale, not on its own and not on the frame's.

        Season 1 spans [0, 1] and season 2 spans [100, 200]. Three candidate fit
        sources give three unmistakably different answers: the whole frame gives an
        upper bound near 198, season 2's own values give one near 199, and season 1
        gives one just under 1. Only the last is WR-06.
        """
        n = 285
        frame = _stack(
            _season_block(2001, n, metric=np.linspace(0.0, 1.0, n)),
            _season_block(2002, n, metric=np.linspace(100.0, 200.0, n)),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        later = out.loc[out["season"] == 2002, "metric"]

        assert later.max() <= 1.0, (
            "season 2002's upper winsorization bound did not come from season 2001 "
            f"-- observed max {later.max()}"
        )
        assert later.min() >= 0.0

    def test_appending_rows_to_the_latest_season_moves_no_earlier_value(
        self, builder
    ) -> None:
        """SPEC R2's positive control in miniature, at unit tier.

        Build three seasons, process them, then append rows to the LATEST season and
        re-process. Every value in the two earlier seasons must be byte-identical.
        This is the exact property the rung-4 re-sync asserts over the real matrices
        after a multi-hour rebuild; catching a regression here costs milliseconds.

        The appended rows are deliberately extreme in BOTH tails so that a
        whole-frame quantile would move a long way -- an append that could not move
        the old bounds would make this test vacuous.
        """
        rng = np.random.default_rng(3016)
        base = _stack(
            *[
                _season_block(
                    season,
                    120,
                    metric=rng.normal(10.0, 2.0, 120),
                    tempo=rng.normal(-3.0, 1.0, 120),
                )
                for season in (2001, 2002, 2003)
            ]
        )

        extra = _season_block(
            2003,
            40,
            metric=np.full(40, -1.0e6),
            tempo=np.full(40, 1.0e6),
        )
        extra["game_id"] = [f"2003_extra_{i:04d}" for i in range(40)]
        widened = _stack(base, extra)

        before = builder.handle_missing_data_and_outliers(base)
        after = builder.handle_missing_data_and_outliers(widened)

        columns = ["metric", "tempo"]
        earlier_before = (
            before.loc[before["season"] < 2003]
            .set_index("game_id")[columns]
            .sort_index()
        )
        earlier_after = (
            after.loc[after["season"] < 2003].set_index("game_id")[columns].sort_index()
        )

        pd.testing.assert_frame_equal(earlier_before, earlier_after)


class TestANeutralConstantPrehistoryDoesNotEraseTheFeature:
    """The bound is fitted on a PRE-CLIP snapshot, and a degenerate bound is skipped.

    Both guards exist because of a measured failure. A first implementation of
    WR-06 read the LIVE frame inside the season loop, so season Y's bound was
    fitted on the ALREADY-CLIPPED values of the seasons before it. For a column
    whose early history is a neutral constant -- the Phase-28 injury family's
    ``availability_fraction`` default of 1.0, the Phase-29 line-movement family's
    WR-10 zeros over the seventeen seasons ``odds_timeline`` does not cover -- that
    cascade is fatal:

      * the first season whose prior slice is dominated by the neutral value gets
        q01 == q99 == that value,
      * clipping to ``[c, c]`` overwrites every genuine observation in the season,
      * which makes the NEXT season's prior slice even more constant,
      * so the column can never recover, even after real data arrives.

    On the real matrices that form destroyed 18 columns outright, including all 15
    line-movement columns and all four injury availability columns. The fix is two
    parts, and this class pins both: fit on the pre-clip snapshot, and treat a
    degenerate bound as "no informative prior", not as a clip.
    """

    _N = 120

    def _frame(self) -> pd.DataFrame:
        """Neutral 1.0 for 2002-2012; real, varying values from 2013 on."""
        rng = np.random.default_rng(2806)
        blocks = []
        for season in range(2002, 2020):
            if season < 2013:
                values = np.ones(self._N)
            else:
                values = rng.uniform(0.55, 1.0, self._N)
            blocks.append(_season_block(season, self._N, availability=values))
        return _stack(*blocks)

    def test_the_first_covered_season_is_not_erased_by_a_degenerate_bound(
        self, builder
    ) -> None:
        out = builder.handle_missing_data_and_outliers(self._frame())
        first = out.loc[out["season"] == 2013, "availability"]

        assert first.nunique() > 1, (
            "2013's prior slice is a wall of the neutral 1.0, so q01 == q99 == 1.0. "
            "Clipping to that degenerate bound erases the first season of real "
            "data -- CR-02's finding in continuous clothing"
        )

    def test_every_covered_season_after_the_first_keeps_its_variance(
        self, builder
    ) -> None:
        out = builder.handle_missing_data_and_outliers(self._frame())

        flattened = [
            season
            for season in range(2013, 2020)
            if out.loc[out["season"] == season, "availability"].nunique() <= 1
        ]

        assert not flattened, (
            f"seasons {flattened} were flattened to a constant. The bound is being "
            "fitted on already-clipped prior seasons, so the neutral prehistory "
            "cascades forward and the feature never recovers"
        )

    def test_the_neutral_prehistory_itself_is_left_alone(self, builder) -> None:
        """A season with no informative prior bound is not clipped, not erased."""
        out = builder.handle_missing_data_and_outliers(self._frame())
        prehistory = out.loc[out["season"] < 2013, "availability"]

        assert (prehistory == 1.0).all()


class TestTheSelfFitLog:
    """The earliest season self-fits, and the flag that says so is machine-readable."""

    def test_only_the_earliest_season_self_fits_on_a_fully_populated_frame(
        self, builder
    ) -> None:
        rng = np.random.default_rng(11)
        frame = _stack(
            *[
                _season_block(
                    season,
                    60,
                    metric=rng.normal(0.0, 1.0, 60),
                    tempo=rng.normal(5.0, 2.0, 60),
                )
                for season in range(2001, 2006)
            ]
        )

        builder.handle_missing_data_and_outliers(frame)

        assert _self_fit_seasons(builder) == [2001]

    def test_over_the_real_gold_season_span_only_the_earliest_season_self_fits(
        self, builder
    ) -> None:
        """The same claim, over the season span the real gold matrices actually carry.

        The span is read from the COMMITTED pre-Phase-30 fixture rather than from
        ``data/gold``, so this asserts on a fresh checkout instead of skipping. The
        columns are synthetic and fully populated on purpose: this test is about the
        RULE (only the first season has no prior), not about any particular column's
        upstream coverage floor. A column whose upstream source starts late has an
        EMPTY prior slice in its first populated season and self-fits there by
        design -- that behaviour is pinned separately by
        ``TestALateArrivingColumn`` below.
        """
        seasons = _real_gold_seasons()
        assert len(seasons) > 1, "the fixture must span more than one season"

        rng = np.random.default_rng(2002)
        frame = _stack(
            *[
                _season_block(
                    season,
                    40,
                    metric=rng.normal(44.0, 3.0, 40),
                    tempo=rng.normal(0.0, 1.0, 40),
                )
                for season in seasons
            ]
        )

        builder.handle_missing_data_and_outliers(frame)

        assert _self_fit_seasons(builder) == [seasons[0]], (
            "a season other than the earliest self-fitted its own bounds. On a fully "
            "populated frame that is a coverage surprise, not a rounding detail"
        )


class TestALateArrivingColumn:
    """A column whose upstream source starts mid-history (T-30-55).

    WHY THIS TEST EXISTS, rather than being settled by inspection: on FINAL gold
    there are zero numeric columns that are all-NaN before 2013, because the
    whole-frame median imputation filled them -- so the case is not reproducible
    after imputation. But WR-06 fits its bounds on the PRE-IMPUTATION frame, where
    the case is unverified in either direction. A naive prior-only rewrite would
    hand such a column an EMPTY strictly-prior slice in its first populated season
    and produce NaN bounds, or raise. The insufficient-prior-data fallback covers
    the case by construction, and nothing else proves the construction is right.

    This frame is deliberately SYNTHETIC, and deliberately distinct from
    ``test_over_the_real_gold_season_span_only_the_earliest_season_self_fits``. The
    two are not in tension: that one is a coverage claim about a fully populated
    frame, this one is a behavioural claim about a shape gold does not currently
    contain in its post-imputation form. A reader who conflates them will "fix" one
    of them wrongly.
    """

    _SEASONS = tuple(range(2002, 2022))
    _N = 30

    def _frame(self, values_2020: np.ndarray, values_2021: np.ndarray) -> pd.DataFrame:
        blocks = []
        for season in self._SEASONS:
            if season == 2020:
                values = values_2020
            elif season == 2021:
                values = values_2021
            else:
                values = np.full(self._N, np.nan)
            blocks.append(_season_block(season, self._N, late_metric=values))
        return _stack(*blocks)

    def _run(self, builder, values_2020, values_2021):
        out = builder.handle_missing_data_and_outliers(
            self._frame(values_2020, values_2021)
        )
        return out, dict(builder.self_fit_seasons)

    def test_it_self_fits_in_its_first_populated_season_then_uses_prior_bounds(
        self, builder
    ) -> None:
        rng = np.random.default_rng(2055)
        base_2020 = rng.uniform(10.0, 20.0, self._N)
        base_2021 = rng.uniform(10.0, 20.0, self._N)
        base_2021[0] = 1000.0

        out, log = self._run(builder, base_2020, base_2021)

        # 2020 -- the first populated season -- has an EMPTY prior slice and self-fits.
        assert 2020 in log.get("late_metric", []), (
            "the first populated season of a late-arriving column must be recorded as "
            f"self-fitting; log was {log.get('late_metric')}"
        )
        # 2021 has a usable prior slice and must NOT self-fit.
        assert 2021 not in log.get("late_metric", [])

        # Nothing raised, nothing is a NaN bound: 2021's outlier was actually clipped
        # to a finite value drawn from 2020's range.
        clipped_2021 = out.loc[out["season"] == 2021, "late_metric"]
        assert clipped_2021.notna().all()
        assert clipped_2021.max() < 25.0, (
            "2021's upper bound did not come from 2020's [10, 20] range -- observed "
            f"max {clipped_2021.max()}"
        )

        # The all-NaN early seasons are left alone rather than filled from the future.
        early = out.loc[out["season"] < 2020, "late_metric"]
        assert early.isna().all(), (
            "an all-NaN early season was filled from a LATER season's statistic -- "
            "that is precisely the leak WR-06 removes"
        )

    def test_changing_2020_moves_the_2021_bound_and_changing_2021_does_not(
        self, builder
    ) -> None:
        rng = np.random.default_rng(2056)
        base_2020 = rng.uniform(10.0, 20.0, self._N)
        shifted_2020 = base_2020 + 100.0
        base_2021 = rng.uniform(10.0, 20.0, self._N)
        base_2021[0] = 1000.0
        shifted_2021 = base_2021 + 500.0
        shifted_2021[0] = 1000.0

        baseline, _ = self._run(builder, base_2020, base_2021)
        moved_source, _ = self._run(builder, shifted_2020, base_2021)
        moved_own, _ = self._run(builder, base_2020, shifted_2021)

        def bound(frame: pd.DataFrame) -> float:
            return float(frame.loc[frame["season"] == 2021, "late_metric"].max())

        assert bound(moved_source) > bound(baseline) + 50.0, (
            "moving 2020's values did not move 2021's bound -- 2021 is not fitting on "
            "its prior season"
        )
        assert bound(moved_own) == pytest.approx(bound(baseline)), (
            "moving 2021's OWN values moved 2021's bound -- the bound is still being "
            "fitted within-season"
        )


class TestCr02IsEvaluatedOncePerColumn:
    """The discreteness predicate must see the WHOLE column, never one season."""

    def test_a_globally_continuous_but_locally_constant_column_is_still_winsorized(
        self, builder
    ) -> None:
        """Season 2002's values are all exactly 1.0 -- an indicator level by accident.

        Evaluated per season, ``{1.0}`` is a subset of ``{-1, 0, 1}`` and the column
        would be exempted from winsorization for that season. Evaluated once over the
        whole column it is plainly a continuous measurement, and season 2002's rows
        must be pulled up to season 2001's lower bound.
        """
        n = 200
        rng = np.random.default_rng(202)
        early = rng.uniform(40.0, 50.0, n)
        early[0] = 500.0

        frame = _stack(
            _season_block(2001, n, pace=early),
            _season_block(2002, n, pace=np.full(n, 1.0)),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        later = out.loc[out["season"] == 2002, "pace"]

        assert (later > 35.0).all(), (
            "a globally continuous column that happens to be constant at 1.0 within "
            "one season was misclassified as a discrete indicator and escaped "
            f"winsorization -- observed values {sorted(set(later.tolist()))[:3]}"
        )

    def test_a_real_indicator_is_exempt_in_every_season(self, builder) -> None:
        """Including a season in which only one of its levels appears."""
        n = 200
        early = np.ones(n)
        early[:2] = 0.0

        frame = _stack(
            _season_block(2001, n, saturday_game=early),
            _season_block(2002, n, saturday_game=np.ones(n)),
        )

        out = builder.handle_missing_data_and_outliers(frame)

        assert (out.loc[out["season"] == 2001, "saturday_game"] == 0.0).sum() == 2
        assert (out.loc[out["season"] == 2002, "saturday_game"] == 1.0).all()


class TestMissingAndOutlierHandlingStayIndependent:
    def test_a_column_with_gaps_is_both_imputed_and_winsorized(self, builder) -> None:
        n = 200
        rng = np.random.default_rng(77)
        early = rng.normal(10.0, 1.0, n)
        later = rng.normal(10.0, 1.0, n)
        later[0] = 900.0
        later[1:4] = np.nan

        frame = _stack(
            _season_block(2001, n, tempo=early),
            _season_block(2002, n, tempo=later),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        processed = out.loc[out["season"] == 2002, "tempo"]

        assert processed.isna().sum() == 0, "the gap was not imputed"
        assert processed.max() < 20.0, "the outlier was not winsorized"


class TestTheWr10GuardIsPreserved:
    """Removed at rung 3, not bundled into the WR-06 commit."""

    def test_the_line_movement_family_is_filled_from_neutral_defaults(
        self, builder
    ) -> None:
        """A median fill would stamp uncovered games COVERED; the neutral fill is 0.0."""
        n = 200
        coverage_early = np.ones(n)
        coverage_late = np.ones(n)
        coverage_late[:3] = np.nan

        frame = _stack(
            _season_block(2001, n, line_movement_coverage=coverage_early),
            _season_block(2002, n, line_movement_coverage=coverage_late),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        later = out.loc[out["season"] == 2002, "line_movement_coverage"]

        assert (later == 0.0).sum() == 3, (
            "the WR-10 neutral-default guard no longer fills line_movement_coverage "
            "from the builder's own defaults"
        )
        assert later.isna().sum() == 0

    def test_the_neutral_opening_total_default_still_wins_over_a_median(
        self, builder
    ) -> None:
        n = 200
        rng = np.random.default_rng(44)
        early = rng.normal(60.0, 1.0, n)
        later = rng.normal(60.0, 1.0, n)
        later[:3] = np.nan

        frame = _stack(
            _season_block(2001, n, opening_total=early),
            _season_block(2002, n, opening_total=later),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        filled = out.loc[out["season"] == 2002, "opening_total"].iloc[:3]

        # LEAGUE_AVERAGE_TOTAL is 44.0; the prior-season median is ~60. The neutral
        # default must win, then survive winsorization against the prior season's
        # lower bound.
        assert (filled < 58.0).all(), (
            "opening_total's gap was filled from a fitted median rather than from the "
            f"builder's neutral default -- observed {filled.tolist()}"
        )


class TestImputeTeamFeaturesFallbacks:
    """Surface 3: the two whole-frame medians D30-16 does not name."""

    def test_the_no_data_for_team_fallback_uses_a_prior_season_median(
        self, builder
    ) -> None:
        """Season 2002 has NO data at all for this team feature.

        Its within-season team mean and season mean are both NaN, so it falls
        through to the last-resort median. The whole-frame median here is ~505
        (season 2001 sits at 10, season 2003 at 1000); the prior-seasons-only median
        is ~10. Only the latter is point-in-time.
        """
        n = 60
        rng = np.random.default_rng(798)
        frame = _stack(
            _season_block(2001, n, home_metric=rng.normal(10.0, 0.1, n)),
            _season_block(2002, n, home_metric=np.full(n, np.nan)),
            _season_block(2003, n, home_metric=rng.normal(1000.0, 1.0, n)),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        filled = out.loc[out["season"] == 2002, "home_metric"]

        assert filled.notna().all()
        assert filled.max() < 20.0, (
            "the no-data-for-team fallback filled season 2002 from a statistic that "
            f"includes season 2003 -- observed max {filled.max()}"
        )

    def test_the_non_team_feature_fallback_uses_a_prior_season_median(
        self, builder
    ) -> None:
        """The other whole-frame median: a column that DISPATCHES to the team branch
        but is not itself prefixed.

        ``handle_missing_data_and_outliers`` routes on ``"home_" in col`` while
        ``_impute_team_features`` routes on ``col.startswith("home_")``, so a column
        carrying the substring anywhere but the front lands in the non-team-feature
        branch. It is a narrow shape, but it is a live code path and it held the
        second whole-frame median.
        """
        n = 60
        rng = np.random.default_rng(817)
        later = np.full(n, np.nan)
        frame = _stack(
            _season_block(2001, n, at_home_pace=rng.normal(10.0, 0.1, n)),
            _season_block(2002, n, at_home_pace=later),
            _season_block(2003, n, at_home_pace=rng.normal(1000.0, 1.0, n)),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        filled = out.loc[out["season"] == 2002, "at_home_pace"]

        assert filled.notna().all()
        assert filled.max() < 20.0, (
            "the non-team-feature fallback filled season 2002 from a statistic that "
            f"includes season 2003 -- observed max {filled.max()}"
        )

    def test_the_method_computes_no_whole_frame_median(self) -> None:
        """A source guard beside the two behavioural ones.

        The behavioural tests prove the observable outcome; this one names the exact
        expression that used to sit at lines 798 and 817, so a re-introduction is
        caught even in a shape whose effect a future frame happens to hide.

        It asserts on STRUCTURE (the parsed call), never on the source text. A
        textual guard here matches the method's own docstring, which necessarily
        quotes the expression it removed -- the self-referential guard hazard Plan
        30-04 hit and recorded.
        """
        assert not _method_calls(
            FeatureMatrixBuilder._impute_team_features, "df", "median"
        ), (
            "_impute_team_features computes a median over the WHOLE frame again -- "
            "that is the third WR-06 surface (T-30-28)"
        )

    def test_the_within_season_means_are_deliberately_retained(self) -> None:
        """D30-16 accepts the within-season residual; removing it silently would be a
        larger behavioural change than the phase authorises in this file.

        Structural for the same reason as the guard above, and additionally because
        this is a PRESENCE assertion: a textual form would pass vacuously the moment
        the expression survived only in a comment.
        """
        source = inspect.getsource(FeatureMatrixBuilder._impute_team_features)
        tree = ast.parse(textwrap.dedent(source))

        means = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mean"
        }

        assert "team_values.mean" in means, "the within-season TEAM mean was removed"
        assert "season_data[col].mean" in means, (
            "the within-season SEASON mean was removed"
        )
