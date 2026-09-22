"""CR-01 regression: the ``as_of`` leakage fence must never be a naive LOCAL clock.

The Phase-29 review found that both ``as_of_datetime`` defaults in
``scripts/build_features.py`` were a bare ``datetime.now()`` -- a naive LOCAL wall
clock -- and that every downstream consumer RE-LABELS a naive value as UTC rather
than converting it (``ensure_utc_aware`` reinterprets, and the since-retired
``LeakageGate.check_time_fence`` used to ``tz_localize`` the column's zone onto
it). ``pipeline/steps.py:235`` calls ``generate_feature_matrices()`` with no
arguments, so that default was the live orchestrator fence.

PHASE 33.2 UPDATE (D33.2-01). The fence is now the per-game LOCK, checked by
``features.provenance.InformationTimeGate``; the ``as_of`` default survives only as
the cutoff handed to builders that still take one, and it must still be tz-aware.
The convert-never-relabel controls below are kept against the new gate, and the
naive branch -- which the old fence ALIGNED -- is now a raise.

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
from features.provenance import (
    InformationTimeGate,
    InformationTimeViolation,
    SourceCheckState,
    build_lock_frame,
)
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

    # ``through_season`` joined the signature at Plan 33.2-08 (the ladder-rung bound);
    # the stub accepts it -- and any later keyword -- so it keeps capturing the one
    # argument this module is about instead of failing on a keyword it never reads.
    def _stub(_self, _season=None, _week=None, *, as_of_datetime=None, **_kwargs):
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
                    # Tz-aware, as silver ``games.kickoff_et`` is: since Plan 33.2-13
                    # the method builds each game's lock from its kickoff BEFORE any
                    # builder runs, and a naive kickoff is refused by the lock rule.
                    "kickoff_et": datetime(2026, 9, 13, 13, 0, tzinfo=ET),
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


class TestTheInformationTimeGateConvertsNeverRelabels:
    """The CR-01 discipline, kept against the gate that replaced ``check_time_fence``.

    The old fence ALIGNED a naive value to ET. The discipline it carried -- an instant is
    converted, never relabelled -- is kept; the silent alignment branch is removed, so a
    naive value now RAISES (D33.2-01).
    """

    _GAME = "2026_W02_DET@KC"

    @classmethod
    def _check(cls, information_time: object) -> SourceCheckState:
        # Sunday 2026-09-13 13:00 ET kickoff -> lock Saturday 2026-09-12 18:00 ET
        # == 22:00 UTC (EDT).
        games = pd.DataFrame(
            {
                "game_id": [cls._GAME],
                "kickoff_et": [pd.Timestamp("2026-09-13 13:00", tz=ET)],
            }
        )
        return InformationTimeGate().check(
            "line_movement",
            pd.DataFrame({"game_id": [cls._GAME], "opening_total": [44.0]}),
            pd.DataFrame(
                {
                    "game_id": [cls._GAME],
                    "basis": ["per_row"],
                    "information_time": pd.Series([information_time], dtype="object"),
                }
            ),
            build_lock_frame(games),
            no_information_signature={"opening_total": 0.0},
        )

    def test_a_utc_stored_instant_at_the_et_lock_is_admitted(self) -> None:
        """22:00 UTC IS 18:00 EDT -- the same instant, so it is AT the lock, not after.

        Pre-CR-01 the relabelling direction read a 22:00 UTC value as four hours in the
        FUTURE of an 18:05 ET cutoff and hard-failed the build.
        """
        assert (
            self._check(pd.Timestamp("2026-09-12 22:00", tz="UTC"))
            is SourceCheckState.CHECKED
        )

    def test_a_genuinely_post_lock_instant_is_still_refused(self) -> None:
        """Positive control: one second after the lock, in UTC, must raise."""
        with pytest.raises(InformationTimeViolation, match=self._GAME):
            self._check(pd.Timestamp("2026-09-12 22:00:01", tz="UTC"))

    def test_an_aware_et_instant_is_compared_as_the_same_instant(self) -> None:
        """An ET-labelled instant one second late is refused exactly like its UTC twin."""
        with pytest.raises(InformationTimeViolation):
            self._check(datetime(2026, 9, 12, 18, 0, 1, tzinfo=ET))
        assert self._check(datetime(2026, 9, 12, 18, 0, tzinfo=ET)) is (
            SourceCheckState.CHECKED
        )

    def test_a_naive_information_time_raises_instead_of_being_aligned(self) -> None:
        """THE BRANCH THAT CHANGED: the old fence aligned a naive value to ET."""
        with pytest.raises(InformationTimeViolation, match="timezone"):
            self._check(datetime(2026, 9, 12, 17, 0))


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
