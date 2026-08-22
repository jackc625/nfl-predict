"""A missing ``odds_timeline`` must degrade to neutral defaults, not abort the build.

CR-03 of the Phase-29 deep code review. ``DataIngestionError`` derives from
``NFLPredictException`` and therefore from ``Exception`` directly -- it is NOT a
subclass of ``ValueError``, ``OSError`` or ``FileNotFoundError``. ``load_dataframe``
raises exactly that (``data/storage.py:923``) when a table is absent from both
DuckDB and parquet. So before this fix every ``except`` tuple on the degradation
path was a guard that could not fire, and a fresh clone could not rebuild gold at
all -- which falsified the graceful-degradation contract
``scripts/build_features.py`` asserts in prose, and that claim is the reason nothing
else guards the construction.

The suite had no such test, which is exactly why the contract could be false while
everything passed.
"""

from datetime import datetime

import pandas as pd
import pytest

import features.line_movement as line_movement_module
import scripts.build_features as build_features_module
from features.line_movement import LEAGUE_AVERAGE_TOTAL, LineMovementBuilder
from utils.exceptions import DataIngestionError

_MISSING_TABLE_MESSAGE = "Table odds_timeline not found in DB or Parquet"


@pytest.fixture
def missing_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``load_dataframe`` raise the real absent-table error."""

    def _raise(*args, **kwargs):
        raise DataIngestionError(_MISSING_TABLE_MESSAGE)

    monkeypatch.setattr(line_movement_module, "load_dataframe", _raise)


def _games(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"2023_W1{i}_A@B" for i in range(n)],
            "season": [2023] * n,
            "week": list(range(1, n + 1)),
            "home_team": ["NYJ"] * n,
            "away_team": ["MIA"] * n,
            "kickoff_et": [pd.Timestamp("2023-09-17 13:00")] * n,
        }
    )


class TestDataIngestionErrorIsCaught:
    """The exception the review found escaping."""

    def test_the_error_is_not_a_valueerror_or_oserror(self) -> None:
        """The MRO fact that made every guard inert. Pinned so it cannot be
        'simplified' back into the old tuple."""
        assert not issubclass(
            DataIngestionError, (ValueError, OSError, FileNotFoundError, KeyError)
        )

    def test_populated_games_degrade_to_neutral_defaults(
        self, missing_timeline: None
    ) -> None:
        out = LineMovementBuilder().build_features(_games(3), datetime(2024, 1, 1, 12))

        assert len(out) == 3
        assert (out["line_movement_coverage"] == 0.0).all()

    def test_opening_total_is_the_league_average_not_a_literal_zero(
        self, missing_timeline: None
    ) -> None:
        """0.0 is out of distribution for a 40-to-50 totals line; the neutral
        default has to sit inside the distribution the model has seen."""
        out = LineMovementBuilder().build_features(_games(1), datetime(2024, 1, 1, 12))

        assert out["opening_total"].iloc[0] == LEAGUE_AVERAGE_TOTAL
        assert LEAGUE_AVERAGE_TOTAL != 0.0

    def test_empty_games_frame_also_survives_the_load(
        self, missing_timeline: None
    ) -> None:
        """Load-bearing: ``build_features`` calls ``_timeline_has_spread()``, which
        loads, BEFORE the ``len(games_df) == 0`` early return -- so even a no-op
        build hits the load and would have raised."""
        empty = _games(0)
        out = LineMovementBuilder().build_features(empty, datetime(2024, 1, 1, 12))

        assert len(out) == 0
        assert "line_movement_coverage" in out.columns

    def test_none_games_frame_also_survives_the_load(
        self, missing_timeline: None
    ) -> None:
        out = LineMovementBuilder().build_features(None, datetime(2024, 1, 1, 12))
        assert len(out) == 0


class TestLoadAllFeatureSourcesDegrades:
    """The shared build_features guard, exercised through a surviving source.

    This class used to drive the guard through the LINE-MOVEMENT source. Plan
    30-07 (rung 3) removed that registration seam from ``scripts/build_features.py``
    entirely -- the family no longer reaches gold (SPEC R3, D29-07-01) -- so there
    is no line-movement source left to make raise there.

    The CR-03 contract itself is unchanged and is what is under test: ONE shared
    ``_SOURCE_LOAD_ERRORS`` tuple guards every optional source, so the guards cannot
    drift apart. Re-scoping onto ``snap_builder`` keeps that claim exercised against
    a source that still exists. The builder half of this module -- every test in
    ``TestDataIngestionErrorIsCaught`` -- is untouched: ``features/line_movement.py``
    stays in the tree and still degrades to neutral defaults, because what left gold
    is the columns, not the builder or the paid archive behind it.
    """

    def test_a_failing_source_becomes_an_empty_frame_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = build_features_module.FeatureMatrixBuilder()

        def _raise(*args, **kwargs):
            raise DataIngestionError(_MISSING_TABLE_MESSAGE)

        monkeypatch.setattr(builder.snap_builder, "build_features", _raise)
        monkeypatch.setattr(
            build_features_module, "load_dataframe", lambda *a, **k: _games(2)
        )
        # Keep the other on-the-fly builders cheap: each returns an empty frame.
        for calc_attr in (
            "elo_calc",
            "contextual_calc",
            "market_calc",
            "qb_tracker",
            "injury_builder",
        ):
            monkeypatch.setattr(
                getattr(builder, calc_attr),
                "build_features",
                lambda *a, **k: pd.DataFrame(),
            )

        sources = builder.load_all_feature_sources()

        assert sources["snaps"].empty
        assert len(sources["games"]) == 2

    def test_no_line_movement_source_is_registered_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The seam is gone, so there is nothing to degrade (SPEC R3).

        Asserted here as well as in ``test_build_features_gold_write.py`` because
        this module is where a future reader looks for the line-movement source's
        behaviour, and "it degrades gracefully" would be a false answer.
        """
        builder = build_features_module.FeatureMatrixBuilder()
        monkeypatch.setattr(
            build_features_module, "load_dataframe", lambda *a, **k: _games(2)
        )
        for calc_attr in (
            "elo_calc",
            "contextual_calc",
            "market_calc",
            "qb_tracker",
            "snap_builder",
            "injury_builder",
        ):
            monkeypatch.setattr(
                getattr(builder, calc_attr),
                "build_features",
                lambda *a, **k: pd.DataFrame(),
            )

        assert "line_movement" not in builder.load_all_feature_sources()

    def test_the_shared_error_tuple_includes_dataingestionerror(self) -> None:
        """One constant guards all twelve sites, so they cannot drift apart."""
        assert DataIngestionError in build_features_module._SOURCE_LOAD_ERRORS
