"""CR-01 regression: the ``as_of`` leakage fence must never be a naive LOCAL clock.

The Phase-29 review found that both ``as_of_datetime`` defaults in
``scripts/build_features.py`` were a bare ``datetime.now()`` -- a naive LOCAL wall
clock -- and that every downstream consumer RE-LABELS a naive value as UTC rather
than converting it (``ensure_utc_aware`` reinterprets, and
``LeakageGate.check_time_fence`` used to ``tz_localize`` the column's zone onto
it). ``pipeline/steps.py:235`` calls ``generate_feature_matrices()`` with no
arguments, so that default IS the live orchestrator fence.

The magnitude of the resulting error is exactly the host machine's UTC offset, and
nothing in the repo pins the host timezone:

- West of UTC (the owner's ET machine) the fence lands EARLY. At the Friday 18:05
  ET orchestrator slot the fence became 18:05Z == 14:05 ET, so the Friday-6PM-ET
  freeze snapshot -- the single most important point on every trajectory, and the
  reason the whole D-12 cadence exists -- was silently dropped from the current
  week's features. Training gold was built with an ``as_of`` decades after every
  freeze, so the fence never bound there: textbook train/serve skew.
- East of UTC the fence lands LATE, i.e. it admits post-cutoff rows. That is a
  leak, and temporal safety is this project's paramount constraint.

These tests assert the corrected contract at all three seams: the two defaults,
the CLI ``--as-of`` parse, and the two consumers.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from features.line_movement import _as_of_to_utc
from features.validation import LeakageGate, LeakageViolation
from scripts.build_features import FeatureMatrixBuilder
from utils.date_utils import ET, UTC


class _AsOfCaptured(Exception):
    """Sentinel raised by the stub loader once it has recorded ``as_of``.

    Deliberately NOT a member of ``build_features._SOURCE_LOAD_ERRORS``, so it
    propagates through ``generate_feature_matrices``' outer guard instead of being
    swallowed and re-raised as a load failure.
    """

    def __init__(self, as_of: datetime) -> None:
        super().__init__("as_of captured")
        self.as_of = as_of


def _capture_default_as_of(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Run ``generate_feature_matrices()`` with NO as_of and return the default."""
    builder = FeatureMatrixBuilder()

    def _stub(_self, _season=None, _week=None, *, as_of_datetime=None):
        raise _AsOfCaptured(as_of_datetime)

    monkeypatch.setattr(FeatureMatrixBuilder, "load_all_feature_sources", _stub)
    with pytest.raises(_AsOfCaptured) as exc:
        builder.generate_feature_matrices()
    return exc.value.as_of


class TestDefaultsAreTimezoneAware:
    """The two ``as_of`` defaults must be tz-aware, and must denote ET."""

    def test_generate_feature_matrices_default_as_of_is_tz_aware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live orchestrator path (``pipeline/steps.py:235``) passes no as_of.

        A naive default here is the whole of CR-01: it is re-labelled UTC further
        down, shifting the fence by the host's UTC offset.
        """
        as_of = _capture_default_as_of(monkeypatch)

        assert as_of is not None
        assert as_of.tzinfo is not None, (
            "generate_feature_matrices defaulted to a NAIVE datetime; it will be "
            "re-labelled as UTC downstream and shift the leakage fence by the "
            "host's UTC offset (CR-01)"
        )

    def test_generate_feature_matrices_default_denotes_the_et_wall_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default must be ET 'now', not UTC 'now' relabelled.

        Compared against a genuine ET now(): the same instant, within a minute.
        """
        as_of = _capture_default_as_of(monkeypatch)
        drift = abs(as_of.astimezone(UTC) - datetime.now(UTC))

        assert drift < timedelta(minutes=1)
        assert as_of.astimezone(ET).hour == datetime.now(ET).hour

    def test_load_all_feature_sources_default_as_of_is_tz_aware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second default site must be aware for the same reason.

        This default is what every per-builder fence (QBTracker, OpponentAdjuster,
        LineMovementBuilder, ...) receives when the method is called directly.
        ``load_dataframe`` is stubbed to a one-row games frame so the first builder
        runs, and that builder records the as_of it was handed.
        """
        builder = FeatureMatrixBuilder()
        games = pd.DataFrame(
            [
                {
                    "game_id": "2026_W02_DET@KC",
                    "season": 2026,
                    "week": 2,
                    "home_team": "KC",
                    "away_team": "DET",
                    "kickoff_et": datetime(2026, 9, 13, 13, 0),
                }
            ]
        )

        monkeypatch.setattr(
            "scripts.build_features.load_dataframe",
            lambda *_a, **_k: games.copy(),
        )

        def _capture(_games, as_of, **_kwargs):
            raise _AsOfCaptured(as_of)

        monkeypatch.setattr(builder.elo_calc, "build_features", _capture)
        with pytest.raises(_AsOfCaptured) as exc:
            builder.load_all_feature_sources()

        assert exc.value.as_of.tzinfo is not None, (
            "load_all_feature_sources defaulted to a NAIVE datetime (CR-01)"
        )


class TestNaiveAsOfIsReadAsEasternNotUtc:
    """A naive ``as_of`` denotes ET everywhere in this system, never UTC."""

    def test_as_of_to_utc_converts_a_naive_value_from_et(self) -> None:
        """18:05 naive is 18:05 ET == 22:05 UTC (EDT), not 18:05 UTC.

        Under the pre-fix behaviour (``ensure_utc_aware``) this returned 18:05Z,
        i.e. 14:05 ET -- four hours early.
        """
        got = _as_of_to_utc(datetime(2026, 9, 11, 18, 5))

        assert got == datetime(2026, 9, 11, 18, 5, tzinfo=ET).astimezone(UTC)
        assert got.astimezone(ET).hour == 18
        assert got.hour != 18, "a naive ET wall clock was re-labelled as UTC (CR-01)"

    def test_as_of_to_utc_leaves_an_aware_value_alone(self) -> None:
        """An explicitly-zoned value is converted, never re-stamped."""
        aware = datetime(2026, 9, 11, 22, 5, tzinfo=UTC)

        assert _as_of_to_utc(aware) == aware


class TestTheFridayFreezeSnapshotSurvivesTheFence:
    """The concrete CR-01 symptom, on the trajectory the fence exists to protect."""

    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
        """One Sunday game whose LAST pre-freeze snapshot is at Friday 18:00 ET.

        Friday 2026-09-11 18:00 ET == 22:00 UTC (EDT). A fence mistakenly set to
        18:05 *UTC* (== 14:05 ET) admits only the Tuesday row, so the game reads as
        UNCOVERED and its drift collapses to 0.0.
        """
        games = pd.DataFrame(
            [
                {
                    "game_id": "2026_W02_DET@KC",
                    "season": 2026,
                    "week": 2,
                    "home_team": "KC",
                    "away_team": "DET",
                    "kickoff_et": datetime(2026, 9, 13, 13, 0),
                }
            ]
        )
        timeline = pd.DataFrame(
            {
                "game_id": ["2026_W02_DET@KC", "2026_W02_DET@KC"],
                "snapshot_ts": [
                    pd.Timestamp("2026-09-08 16:00", tz="UTC"),  # Tuesday
                    pd.Timestamp("2026-09-11 22:00", tz="UTC"),  # Fri 18:00 ET FREEZE
                ],
                "total": [44.0, 46.5],
            }
        )
        return games, timeline

    @pytest.mark.parametrize(
        "as_of",
        [
            pytest.param(datetime(2026, 9, 11, 18, 5), id="naive-et-wall-clock"),
            pytest.param(
                datetime(2026, 9, 11, 18, 5, tzinfo=ET), id="explicitly-aware-et"
            ),
        ],
    )
    def test_the_friday_6pm_et_freeze_snapshot_is_admitted(self, as_of) -> None:
        """At the Friday 18:05 ET orchestrator slot the 18:00 ET freeze is IN.

        This is the test that fails on the pre-fix code for the naive case: the
        fence became 14:05 ET, the 22:00 UTC freeze row was dropped, only one
        snapshot remained, and the game was stamped ``line_movement_coverage ==
        0.0`` with every drift and path value at 0.0 -- for the current week only,
        which is precisely why no training-gold test could see it.
        """
        from features.line_movement import LineMovementBuilder

        games, timeline = self._fixture()
        out = (
            LineMovementBuilder(timeline_df=timeline)
            .build_features(games, as_of)
            .set_index("game_id")
        )
        row = out.loc["2026_W02_DET@KC"]

        assert row["line_movement_coverage"] == 1.0, (
            "the Friday 6 PM ET freeze snapshot was fenced OUT -- the as_of clock "
            "is being read as UTC instead of ET (CR-01)"
        )
        assert row["opening_total"] == 44.0
        assert row["total_drift"] == 2.5, "the freeze value is not the last admitted"

    def test_a_genuinely_early_fence_still_binds(self) -> None:
        """Positive control: an as_of BEFORE the freeze must still exclude it.

        Without this, the assertion above could be satisfied by removing the
        as_of term from the fence altogether.
        """
        from features.line_movement import LineMovementBuilder

        games, timeline = self._fixture()
        out = (
            LineMovementBuilder(timeline_df=timeline)
            .build_features(games, datetime(2026, 9, 11, 12, 0, tzinfo=ET))
            .set_index("game_id")
        )

        assert out.loc["2026_W02_DET@KC", "line_movement_coverage"] == 0.0


class TestLeakageGateAlignsTimezonesByConverting:
    """``check_time_fence`` must convert a naive cutoff from ET, not stamp it."""

    @staticmethod
    def _snapshot_frame(when: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["G1"],
                "snapshot_ts": [pd.Timestamp(when, tz="UTC")],
                "opening_total": [44.0],
            }
        )

    def test_a_naive_cutoff_admits_the_et_evening_snapshot_it_covers(self) -> None:
        """Fri 18:00 ET (22:00 UTC) is BEFORE a naive Fri 18:05 ET cutoff.

        Pre-fix, the naive cutoff was ``tz_localize('UTC')``-ed to 18:05Z, so a
        legitimate 22:00 UTC snapshot read as four hours in the FUTURE and the
        gate hard-failed the whole build (or, on a host east of UTC, silently let
        genuinely-future rows through).
        """
        gate = LeakageGate()

        gate.check_time_fence(
            self._snapshot_frame("2026-09-11 22:00"),
            datetime(2026, 9, 11, 18, 5),
            "line_movement",
        )

    def test_a_naive_cutoff_still_rejects_a_genuinely_future_snapshot(self) -> None:
        """Positive control: a snapshot after the ET cutoff must still raise."""
        gate = LeakageGate()

        with pytest.raises(LeakageViolation):
            gate.check_time_fence(
                self._snapshot_frame("2026-09-12 22:00"),
                datetime(2026, 9, 11, 18, 5),
                "line_movement",
            )

    def test_an_aware_cutoff_is_compared_in_et_against_naive_columns(self) -> None:
        """Naive ``kickoff_et`` columns carry an ET wall clock, so compare in ET.

        A UTC-aware cutoff previously had its tz simply dropped, leaving a UTC
        wall clock to be compared against ET wall clocks -- four hours too
        permissive.
        """
        gate = LeakageGate()
        frame = pd.DataFrame(
            {
                "game_id": ["G1"],
                "kickoff_et": [datetime(2026, 9, 11, 20, 0)],  # 20:00 ET
                "elo_home": [1500.0],
            }
        )

        # 18:05 ET == 22:05 UTC. The 20:00 ET kickoff is in the FUTURE of it.
        with pytest.raises(LeakageViolation):
            gate.check_time_fence(
                frame, datetime(2026, 9, 11, 22, 5, tzinfo=UTC), "elo"
            )


class TestCliAsOfIsInterpretedAsEastern:
    """``--as-of 2024-10-04T18:00:00`` means 18:00 ET, the freeze the repo fences on."""

    def test_a_bare_iso_string_is_localized_to_et(self) -> None:
        """Mirrors the parse in ``main()``; a bare string must not mean UTC."""
        parsed = datetime.fromisoformat("2024-10-04T18:00:00")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET)

        assert parsed.astimezone(UTC) == datetime(2024, 10, 4, 22, 0, tzinfo=UTC)

    def test_an_explicit_offset_is_honoured_as_given(self) -> None:
        """An operator who writes an offset means it."""
        parsed = datetime.fromisoformat("2024-10-04T18:00:00+00:00")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET)

        assert parsed.astimezone(UTC) == datetime(2024, 10, 4, 18, 0, tzinfo=UTC)
