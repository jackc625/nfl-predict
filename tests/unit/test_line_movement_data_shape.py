"""Data-shape edge cases the REAL 2020-2024 archive contains (SIG-04, Plan 29-06).

The paid backfill (Plan 29-05) produced two shapes the synthetic 29-04 fixtures
never exercised, both of which would silently corrupt gold if mishandled:

1. **1,435 rows carry a NULL ``total``** -- sportsbooks post a spread for a game
   weeks ahead of putting a total up. A null must be DROPPED from the trajectory,
   never propagated as NaN into a gold feature and never silently read as ``0.0``
   (which would fabricate a fake ~0 line level and an enormous fake drift).

2. **``2020_W06_BUF@TEN`` (the COVID-postponed Titans game) has exactly ONE
   snapshot**, so it has no trajectory at all. It must score
   ``line_movement_coverage = 0.0``, NOT a degenerate zero-drift value that is
   indistinguishable from a game whose line genuinely never moved.

The distinction in (2) is the whole reason ``line_movement_coverage`` exists: a
MEASURED zero drift and an UNMEASURABLE trajectory must not be encoded
identically. ``TestCoverageSeparatesMeasuredFromUnmeasurable`` pins that contrast
directly.
"""

from unittest.mock import patch

import pandas as pd

from features.line_movement import LEAGUE_AVERAGE_TOTAL, LineMovementBuilder

# All fixture games kick off Sunday 2023-09-17, so each derives the SAME per-game
# Friday-6PM-ET freeze: Fri 2023-09-15 18:00 ET == 2023-09-15 22:00 UTC.
_KICKOFF = pd.Timestamp("2023-09-17 13:00")

_GAME_NULL_HEAD = "2023_02_NULLHEAD"
_GAME_NO_TOTALS = "2023_02_NOTOTALS"
_GAME_ONE_SNAPSHOT = "2023_02_ONESNAP"
_GAME_FLAT_MEASURED = "2023_02_FLATLINE"

# An as_of far after every fixture freeze, so the PER-GAME freeze is the only
# binding fence.
_AS_OF = pd.Timestamp("2023-09-20 18:00").to_pydatetime()


def _ts(spec: str) -> pd.Timestamp:
    """A tz-aware UTC snapshot timestamp."""
    return pd.Timestamp(spec, tz="UTC")


def _games(*game_ids: str) -> pd.DataFrame:
    """A minimal games frame -- all games share one kickoff (one freeze)."""
    return pd.DataFrame(
        [
            {
                "game_id": gid,
                "season": 2023,
                "week": 2,
                "home_team": "BUF",
                "away_team": "LV",
                "kickoff_et": _KICKOFF,
            }
            for gid in game_ids
        ]
    )


def _timeline() -> pd.DataFrame:
    """The four real-archive shapes, all with pre-freeze snapshots only.

    * ``_GAME_NULL_HEAD``  -- spread posted first (total NULL), then 44.0 -> 45.0.
      A correct build reads opening 44.0 / drift +1.0; reading the NULL as 0.0
      would give opening 0.0 / drift +45.0.
    * ``_GAME_NO_TOTALS``  -- spreads on every snapshot, NO total ever posted.
    * ``_GAME_ONE_SNAPSHOT`` -- exactly one snapshot (the BUF@TEN shape).
    * ``_GAME_FLAT_MEASURED`` -- three snapshots, line genuinely never moved.
    """
    rows = [
        # Null-total head, then a real two-point trajectory.
        (_GAME_NULL_HEAD, "2023-09-12 16:00", None, -2.5),
        (_GAME_NULL_HEAD, "2023-09-13 16:00", 44.0, -3.0),
        (_GAME_NULL_HEAD, "2023-09-15 22:00", 45.0, -3.5),
        # Spread-only game: a total was never posted before the freeze.
        (_GAME_NO_TOTALS, "2023-09-12 16:00", None, -1.0),
        (_GAME_NO_TOTALS, "2023-09-13 16:00", None, -1.5),
        (_GAME_NO_TOTALS, "2023-09-15 22:00", None, -2.0),
        # One snapshot only -- no trajectory exists (the BUF@TEN shape).
        (_GAME_ONE_SNAPSHOT, "2023-09-13 16:00", 41.5, -6.0),
        # Three snapshots, line genuinely flat -- a MEASURED zero drift.
        (_GAME_FLAT_MEASURED, "2023-09-12 16:00", 47.0, -7.0),
        (_GAME_FLAT_MEASURED, "2023-09-13 16:00", 47.0, -7.0),
        (_GAME_FLAT_MEASURED, "2023-09-15 22:00", 47.0, -7.0),
    ]
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "snapshot_ts": [_ts(r[1]) for r in rows],
            "total": [r[2] for r in rows],
            "spread": [r[3] for r in rows],
        }
    )


def _build(*game_ids: str) -> pd.DataFrame:
    """Build line-movement features for the named fixture games, indexed by id."""
    builder = LineMovementBuilder(timeline_df=_timeline())
    return builder.build_features(_games(*game_ids), _AS_OF).set_index("game_id")


class TestNullTotals:
    """1,435 archive rows carry a NULL ``total`` (Plan 29-05 finding)."""

    def test_null_total_row_is_dropped_not_read_as_zero(self):
        """A NULL total is skipped; the trajectory starts at the first real line.

        Opening is the first NON-NULL total (44.0) and drift is +1.0. Reading the
        leading NULL as ``0.0`` would instead produce opening 0.0, drift +45.0 and
        range 45.0 -- a fabricated line level and a fabricated steam signal. All
        three are asserted against explicitly.
        """
        out = _build(_GAME_NULL_HEAD)

        assert out.loc[_GAME_NULL_HEAD, "opening_total"] == 44.0
        assert out.loc[_GAME_NULL_HEAD, "total_drift"] == 1.0
        assert out.loc[_GAME_NULL_HEAD, "total_range"] == 1.0
        # The fabricated-zero readings, named so a regression is unambiguous.
        assert out.loc[_GAME_NULL_HEAD, "opening_total"] != 0.0
        assert out.loc[_GAME_NULL_HEAD, "total_drift"] != 45.0

    def test_null_total_row_still_yields_full_coverage(self):
        """Dropping the NULL leaves >=2 real snapshots, so the game IS covered."""
        out = _build(_GAME_NULL_HEAD)

        assert out.loc[_GAME_NULL_HEAD, "line_movement_coverage"] == 1.0

    def test_null_total_does_not_propagate_nan_into_features(self):
        """No emitted value is NaN -- the data_qa all-null check stays satisfied."""
        out = _build(_GAME_NULL_HEAD, _GAME_NO_TOTALS, _GAME_ONE_SNAPSHOT)

        assert out.notna().all().all(), out[out.isna().any(axis=1)]

    def test_game_with_no_total_ever_posted_gets_non_ood_default(self):
        """A game whose totals are ALL null is uncovered with a non-OOD opening.

        ``opening_total`` falls back to the prior-only ``LEAGUE_AVERAGE_TOTAL``
        constant -- never a literal 0.0, which is out-of-distribution for a ~40-50
        line and would let the model learn a coverage artifact instead of keying
        on the coverage flag.
        """
        out = _build(_GAME_NO_TOTALS)

        assert out.loc[_GAME_NO_TOTALS, "line_movement_coverage"] == 0.0
        assert out.loc[_GAME_NO_TOTALS, "opening_total"] == LEAGUE_AVERAGE_TOTAL
        assert out.loc[_GAME_NO_TOTALS, "opening_total"] != 0.0
        assert out.loc[_GAME_NO_TOTALS, "total_drift"] == 0.0

    def test_spread_family_still_derives_when_totals_are_all_null(self):
        """Totals-primary (D-07): the coverage flag tracks TOTALS, and the spread
        siblings are derived independently from the non-null spread trajectory.

        This pins the documented asymmetry rather than leaving it implicit. In the
        real 2021-2024 archive it is unobservable -- zero joined games are
        totals-uncovered while carrying non-zero spread movement -- but the
        semantics must not drift silently.
        """
        out = _build(_GAME_NO_TOTALS)

        assert out.loc[_GAME_NO_TOTALS, "opening_spread"] == -1.0
        assert out.loc[_GAME_NO_TOTALS, "spread_drift"] == -1.0
        assert out.loc[_GAME_NO_TOTALS, "line_movement_coverage"] == 0.0


class TestSingleSnapshotGame:
    """``2020_W06_BUF@TEN`` -- one snapshot, hence no trajectory (Plan 29-05)."""

    def test_single_snapshot_game_is_uncovered(self):
        """One snapshot cannot form a trajectory -> coverage 0.0."""
        out = _build(_GAME_ONE_SNAPSHOT)

        assert out.loc[_GAME_ONE_SNAPSHOT, "line_movement_coverage"] == 0.0

    def test_single_snapshot_opening_uses_the_in_row_anchor(self):
        """The lone pre-freeze snapshot IS a legitimate, non-leaky opening anchor.

        It is at/before the freeze, so using it is temporally safe and strictly
        better than the league-average fallback -- but the drift/path families
        stay 0.0 because no movement was observed.
        """
        out = _build(_GAME_ONE_SNAPSHOT)

        assert out.loc[_GAME_ONE_SNAPSHOT, "opening_total"] == 41.5
        for col in (
            "total_drift",
            "total_drift_dir",
            "total_late_drift",
            "total_abs_travel",
            "total_reversals",
            "total_range",
        ):
            assert out.loc[_GAME_ONE_SNAPSHOT, col] == 0.0, col


class TestCoverageSeparatesMeasuredFromUnmeasurable:
    """The load-bearing distinction ``line_movement_coverage`` exists to make."""

    def test_measured_zero_drift_and_no_trajectory_are_not_encoded_identically(self):
        """A flat MEASURED line and an UNMEASURABLE one differ in the flag.

        Both games emit ``total_drift == 0.0``. Without the coverage flag they
        would be indistinguishable to the model -- "the market had no opinion" and
        "we never observed the market" collapsed into one value. The flag is the
        only thing separating them, so it must differ.
        """
        out = _build(_GAME_FLAT_MEASURED, _GAME_ONE_SNAPSHOT)

        assert out.loc[_GAME_FLAT_MEASURED, "total_drift"] == 0.0
        assert out.loc[_GAME_ONE_SNAPSHOT, "total_drift"] == 0.0

        assert out.loc[_GAME_FLAT_MEASURED, "line_movement_coverage"] == 1.0
        assert out.loc[_GAME_ONE_SNAPSHOT, "line_movement_coverage"] == 0.0
        assert (
            out.loc[_GAME_FLAT_MEASURED, "line_movement_coverage"]
            != out.loc[_GAME_ONE_SNAPSHOT, "line_movement_coverage"]
        )

    def test_measured_flat_line_keeps_its_real_opening_level(self):
        """A covered flat game keeps its OWN opening level, not the fallback."""
        out = _build(_GAME_FLAT_MEASURED)

        assert out.loc[_GAME_FLAT_MEASURED, "opening_total"] == 47.0
        assert out.loc[_GAME_FLAT_MEASURED, "opening_total"] != LEAGUE_AVERAGE_TOTAL


# ---------------------------------------------------------------------------
# WR-07: the per-game accessor must not depend on call ordering
#
# _resolve_game_date_for read self._games_cache and nothing else, and that cache
# is only ever populated by build_features. So a caller using the FeatureBuilder
# Protocol's documented per-game entry point -- exactly how a serving path wires
# a builder in -- got all-neutral features for EVERY game, with no exception, no
# log line, and no way to tell "this game has no trajectory" apart from "you
# called the wrong method first".
# ---------------------------------------------------------------------------


class TestPerGameAccessorLoadsGamesItself:
    """``get_features_for_game`` works as the FIRST call on a fresh builder."""

    @staticmethod
    def _games_frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": _GAME_FLAT_MEASURED,
                    "season": 2023,
                    "week": 2,
                    "home_team": "KC",
                    "away_team": "DET",
                    "kickoff_et": _KICKOFF,
                }
            ]
        )

    def test_it_returns_real_features_without_a_prior_build_call(self, monkeypatch):
        """No build_features first, yet the real trajectory is used.

        Pre-fix this returned the neutral dict -- opening_total 44.0, coverage
        0.0 -- for every game, silently.
        """
        monkeypatch.setattr(
            "features.line_movement.load_dataframe",
            lambda *_a, **_k: self._games_frame(),
        )
        builder = LineMovementBuilder(timeline_df=_timeline())

        feats = builder.get_features_for_game(_GAME_FLAT_MEASURED, _AS_OF)

        assert feats["line_movement_coverage"] == 1.0
        assert feats["opening_total"] == 47.0
        assert feats["opening_total"] != LEAGUE_AVERAGE_TOTAL

    def test_an_unknown_game_id_is_logged_not_silently_neutral(self, monkeypatch):
        """Neutral is still returned -- but it is no longer indistinguishable."""
        monkeypatch.setattr(
            "features.line_movement.load_dataframe",
            lambda *_a, **_k: self._games_frame(),
        )
        builder = LineMovementBuilder(timeline_df=_timeline())

        with patch("features.line_movement.logger") as mock_logger:
            feats = builder.get_features_for_game("2023_02_NOT_A_GAME", _AS_OF)

        assert feats["line_movement_coverage"] == 0.0
        assert feats["opening_total"] == LEAGUE_AVERAGE_TOTAL
        assert "not present in games silver" in str(mock_logger.warning.call_args_list)

    def test_unavailable_games_silver_degrades_with_a_warning(self, monkeypatch):
        """A missing games table must warn, not raise, and not read as data."""

        def _boom(*_a, **_k):
            raise FileNotFoundError("no games silver in this environment")

        monkeypatch.setattr("features.line_movement.load_dataframe", _boom)
        builder = LineMovementBuilder(timeline_df=_timeline())

        with patch("features.line_movement.logger") as mock_logger:
            feats = builder.get_features_for_game(_GAME_FLAT_MEASURED, _AS_OF)

        assert feats["line_movement_coverage"] == 0.0
        assert "games silver unavailable" in str(mock_logger.warning.call_args_list)


# ---------------------------------------------------------------------------
# WR-08: the trajectory load is memoized, and the emitted schema is logged
# ---------------------------------------------------------------------------


class TestTimelineLoadIsMemoized:
    """One load per builder, not one per predicate call and one per game."""

    def test_the_archive_is_loaded_once_per_builder(self, monkeypatch):
        """Pre-fix a 16-game weekly loop cost 32 full parquet reads + coercions."""
        calls: list[str] = []

        def _counting_load(table, **_kwargs):
            calls.append(table)
            return _timeline() if table == "odds_timeline" else pd.DataFrame()

        monkeypatch.setattr("features.line_movement.load_dataframe", _counting_load)
        builder = LineMovementBuilder()

        builder._load_timeline()
        builder._timeline_has_spread()
        builder._load_timeline()

        assert calls.count("odds_timeline") == 1, (
            f"odds_timeline was loaded {calls.count('odds_timeline')} times; the "
            f"memoization is not in place (WR-08)"
        )

    def test_the_resolved_schema_is_logged_once(self):
        """A gold-width change must be traceable to a cause, not discovered."""
        builder = LineMovementBuilder(timeline_df=_timeline())

        with patch("features.line_movement.logger") as mock_logger:
            builder._timeline_has_spread()
            builder._timeline_has_spread()

        info_calls = [
            c
            for c in mock_logger.info.call_args_list
            if "emitted schema resolved" in str(c)
        ]
        assert len(info_calls) == 1
        assert "emit_spread" in str(info_calls[0])
