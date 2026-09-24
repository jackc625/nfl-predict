"""The capture-time injury basis, exercised on PRODUCTION-SHAPED data (Plan 33.2-15 Task 3).

Plan 33.2-13 WROTE the capture-stamp selection rule in ``features/injury.py`` against a frame
column (``features.injury.UPSTREAM_CAPTURE_COLUMN``) and could only exercise it on synthetic
frames, because nothing wrote that column yet. Plan 33.2-15's ingest now writes it, so this
module is the rule's first run on what production actually stores: every frame below is the
silver table the REAL ``scripts.ingest_injuries.ingest_injuries_season`` writes, run into a
temporary data root with a 2025-shaped upstream payload (no ``date_modified`` column) and a
stubbed release-asset stamp.

THE SELECTION RULE IS ASSERTED BY NAME AND THE PROVENANCE BY VALUE -- never a third basis.
``features.provenance.InformationBasis`` has exactly two members (Plan 33.2-01's owned contract):
"capture-time" is a SELECTION rule inside the builder (``capture_stamp``), and what the gate sees
for it is ``basis="per_row"`` with the stamp as ``information_time``.

The six cases, and the two that are the point:

1. POSITIVE: a stamp at or before the lock is admitted on ``capture_stamp``, reported
   ``per_row`` with ``information_time`` EQUAL to the stamp, values populated.
2. THE FALL-THROUGH CONTROL: the same frame with the stamp column REMOVED is ``none_admitted`` /
   ``no_information`` -- and the two outcomes are asserted DIFFERENT, so a builder that always
   reports the unknown state cannot satisfy case 1.
3. THE LOCK BOUNDARY on the stamp: at the lock admitted; one second later not, with features
   byte-identical to the build without that row.
4. NO FABRICATION: the capture path writes no ``date_modified`` anywhere.
5. THE BACKFILL CASE, which is what makes the stamp PROSPECTIVE: a 2025 row fetched in 2026 --
   the shape of every 2025 row the ingest actually wrote -- is NOT admitted, and no provenance
   row's ``information_time`` is later than its game's lock.
6. THE NAME TIE: ``InjurySchema`` declares the column the builder reads.

WHETHER features/injury.py NEEDED AN EDIT: the Plan 33.2-13 interface matched the ingest (the
same column name, a tz-aware UTC instant), so the capture branch was NOT changed. Plan 33.2-15
added one public constant there (``INJURY_FEATURE_COLUMNS``) for the gold build's empty-family
guard, which touches no selection. This module RUNS the branch, which is what distinguishes "no
edit was needed" from "the branch was never run".

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from data.schemas import InjurySchema, SnapCountSchema
from features.injury import UPSTREAM_CAPTURE_COLUMN, InjuryBuilder, SelectionRule
from features.provenance import InformationBasis, InformationTimeGate, SourceCheckState
from features.snaps import SnapCountBuilder

ET = ZoneInfo("America/New_York")

_QB1_ID = "00-0033873"  # KC QB1
_QB2_ID = "00-0000002"  # KC QB2

# A THURSDAY 2025 opener, so the lock is the Wednesday 18:00 ET before it.
_GAME_ID = "2025_W01_BUF@KC"
_KICKOFF = datetime(2025, 9, 4, 20, 20, tzinfo=ET)
_LOCK = pd.Timestamp(lock_rule.game_lock(_KICKOFF))
# A SUNDAY game the same week, with its own later lock.
_SUNDAY_ID = "2025_W01_MIA@LAC"
_SUNDAY_KICKOFF = datetime(2025, 9, 7, 16, 25, tzinfo=ET)
_SUNDAY_LOCK = pd.Timestamp(lock_rule.game_lock(_SUNDAY_KICKOFF))

_ONE_SECOND = timedelta(seconds=1)
# Carried for the FeatureBuilder Protocol only; it is not a fence.
_AS_OF = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)

#: The stamp the 2025 injuries asset actually carried when Plan 33.2-15 ingested it.
_BACKFILL_STAMP = pd.Timestamp("2026-09-07T12:23:41Z")


def _games() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "game_id": _GAME_ID,
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": _KICKOFF,
            },
            {
                "game_id": _SUNDAY_ID,
                "season": 2025,
                "week": 1,
                "home_team": "LAC",
                "away_team": "MIA",
                "kickoff_et": _SUNDAY_KICKOFF,
            },
        ]
    )
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _raw_2025_payload() -> pd.DataFrame:
    """The 2025+ nflreadpy shape (RESEARCH 5.1): NO date_modified column, season_type present."""
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "game_type": ["REG", "REG"],
            "team": ["KC", "LAC"],
            "week": [1, 1],
            "gsis_id": [_QB1_ID, "00-0099999"],
            "position": ["QB", "WR"],
            "full_name": ["Patrick Mahomes", "LAC Receiver"],
            "first_name": ["Patrick", "LAC"],
            "last_name": ["Mahomes", "Receiver"],
            "report_primary_injury": ["Ankle", "Knee"],
            "report_secondary_injury": [None, None],
            "report_status": ["Out", "Questionable"],
            "practice_primary_injury": ["Ankle", "Knee"],
            "practice_secondary_injury": [None, None],
            "practice_status": ["Did Not Participate", "Limited"],
        }
    )


def _ingested_silver(tmp_path: Path, stamp: pd.Timestamp) -> pd.DataFrame:
    """Run the REAL injury ingest into *tmp_path* and read back the silver it wrote."""
    from scripts.ingest_injuries import ingest_injuries_season

    ingest_injuries_season(
        2025,
        loader=lambda _season: _raw_2025_payload(),
        stamp_reader=lambda _asset, *, fresh=False: stamp.to_pydatetime(),
        games=_games(),
        base_path=tmp_path,
    )
    return pd.read_parquet(tmp_path / "silver" / "injuries.parquet")


def _depth_charts() -> pd.DataFrame:
    """2025+ depth charts carry a publication time ``dt``; published well before the lock."""
    published = pd.Timestamp("2025-09-01T12:00:00Z")
    return pd.DataFrame(
        {
            "season": [2025, 2025],
            "club_code": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_team": ["1", "2"],
            "gsis_id": [_QB1_ID, _QB2_ID],
            "full_name": ["Patrick Mahomes", "Backup QB"],
            "dt": [published, published],
        }
    )


def _builder(injuries: pd.DataFrame) -> InjuryBuilder:
    snaps = SnapCountBuilder(snaps_df=pd.DataFrame(), schedule_df=_games())
    return InjuryBuilder(
        snap_builder=snaps,
        injuries_df=injuries,
        depth_charts_df=_depth_charts(),
        pbp_df=pd.DataFrame(),
    )


def _outcome(injuries: pd.DataFrame) -> dict:
    """Everything a case asserts on: the selection, the features and the provenance."""
    games = _games()
    builder = _builder(injuries)
    locks = lock_rule.lock_frame(games)
    game = games.loc[games["game_id"] == _GAME_ID].iloc[0].to_dict()
    selection = builder.fenced_injuries(game, locks[_GAME_ID])
    features = builder.build_features(games, _AS_OF, target_season=2025, target_week=1)
    provenance = builder.information_times(games, target_season=2025, target_week=1)
    gate = InformationTimeGate().check(
        "injury",
        features,
        provenance,
        locks,
        no_information_signature=builder.no_information_signature(),
    )
    return {
        "selection": selection,
        "features": features.set_index("game_id"),
        "provenance": provenance.set_index("game_id"),
        "gate": gate,
        "locks": locks,
    }


def _without_date_modified(frame: pd.DataFrame) -> pd.DataFrame:
    """The ingest carries date_modified as NaT; the raw 2025+ upstream has no column at all."""
    return frame.drop(columns=["date_modified"])


# ---------------------------------------------------------------------------
# 1. The positive case
# ---------------------------------------------------------------------------


class TestAPreLockCaptureIsAdmittedOnTheCaptureRule:
    @pytest.mark.parametrize(
        "shape", ["no_date_modified_column", "as_the_ingest_stores_it"]
    )
    def test_admitted_on_capture_stamp_per_row_at_the_stamp_values_populated(
        self, tmp_path: Path, shape: str
    ) -> None:
        stamp = _LOCK - pd.Timedelta(hours=5)
        silver = _ingested_silver(tmp_path, stamp)
        frame = (
            _without_date_modified(silver)
            if shape == "no_date_modified_column"
            else silver
        )
        assert len(frame) > 0, "non-vacuity: the production-shaped fixture has rows"
        if shape == "as_the_ingest_stores_it":
            assert frame["date_modified"].isna().all()

        outcome = _outcome(frame)
        selection = outcome["selection"]
        assert selection.rule is SelectionRule.CAPTURE_STAMP
        assert len(selection.rows) >= 1, "non-vacuity: at least one row admitted"
        assert selection.information_time == stamp

        row = outcome["provenance"].loc[_GAME_ID]
        assert row["basis"] == InformationBasis.PER_ROW.value
        assert pd.Timestamp(row["information_time"]) == stamp

        values = outcome["features"].loc[_GAME_ID]
        assert values["home_qb_out_flag"] == 1.0  # Mahomes listed Out, admitted
        assert values["home_injury_coverage"] == 1.0
        assert outcome["gate"] is SourceCheckState.CHECKED


# ---------------------------------------------------------------------------
# 2. THE FALL-THROUGH CONTROL
# ---------------------------------------------------------------------------


class TestTheFallThroughControl:
    def test_fall_through_control_without_the_stamp_is_no_information_and_differs(
        self, tmp_path: Path
    ) -> None:
        stamp = _LOCK - pd.Timedelta(hours=5)
        stamped = _without_date_modified(_ingested_silver(tmp_path, stamp))
        unstamped = stamped.drop(columns=[UPSTREAM_CAPTURE_COLUMN])

        with_stamp = _outcome(stamped)
        without = _outcome(unstamped)

        assert without["selection"].rule is SelectionRule.NONE_ADMITTED
        assert without["selection"].information_time is None
        row = without["provenance"].loc[_GAME_ID]
        assert row["basis"] == InformationBasis.NO_INFORMATION.value
        assert pd.isna(row["information_time"])
        assert without["features"].loc[_GAME_ID, "home_injury_coverage"] == 0.0

        # THE POINT: the stamped outcome is not the unknown state under another name.
        assert with_stamp["selection"].rule is not without["selection"].rule
        assert (
            with_stamp["provenance"].loc[_GAME_ID, "basis"]
            != without["provenance"].loc[_GAME_ID, "basis"]
        )
        assert (
            not with_stamp["features"]
            .loc[_GAME_ID]
            .equals(without["features"].loc[_GAME_ID])
        )


# ---------------------------------------------------------------------------
# 3. The lock boundary on the capture stamp
# ---------------------------------------------------------------------------


class TestTheLockBoundaryOnTheStamp:
    def test_a_stamp_exactly_at_the_lock_is_admitted(self, tmp_path: Path) -> None:
        frame = _without_date_modified(_ingested_silver(tmp_path, _LOCK))
        outcome = _outcome(frame)
        assert outcome["selection"].rule is SelectionRule.CAPTURE_STAMP
        assert outcome["selection"].information_time == _LOCK

    def test_one_second_later_is_not_admitted_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        late = _without_date_modified(_ingested_silver(tmp_path, _LOCK + _ONE_SECOND))
        outcome = _outcome(late)
        assert outcome["selection"].rule is SelectionRule.NONE_ADMITTED
        # Byte-identical to the build WITHOUT that game's rows.
        no_rows = _outcome(late.loc[late["team"] != "KC"])
        assert (
            outcome["features"].loc[_GAME_ID].equals(no_rows["features"].loc[_GAME_ID])
        )
        assert (
            outcome["provenance"]
            .loc[_GAME_ID]
            .equals(no_rows["provenance"].loc[_GAME_ID])
        )

    def test_each_game_is_fenced_at_its_own_lock(self, tmp_path: Path) -> None:
        # A stamp between the Thursday game's lock and the Sunday game's lock: admitted for
        # the Sunday game, not for the Thursday one.
        between = _LOCK + pd.Timedelta(hours=12)
        assert _LOCK < between < _SUNDAY_LOCK
        outcome = _outcome(_without_date_modified(_ingested_silver(tmp_path, between)))
        provenance = outcome["provenance"]
        assert (
            provenance.loc[_GAME_ID, "basis"] == InformationBasis.NO_INFORMATION.value
        )
        assert provenance.loc[_SUNDAY_ID, "basis"] == InformationBasis.PER_ROW.value
        assert pd.Timestamp(provenance.loc[_SUNDAY_ID, "information_time"]) == between


# ---------------------------------------------------------------------------
# 4. No fabrication
# ---------------------------------------------------------------------------


class TestTheCapturePathFabricatesNoDateModified:
    def test_no_date_modified_value_is_written_anywhere(self, tmp_path: Path) -> None:
        stamp = _LOCK - pd.Timedelta(hours=5)
        silver = _ingested_silver(tmp_path, stamp)
        assert silver["date_modified"].isna().all(), (
            "the ingest invents no date_modified"
        )

        frame = _without_date_modified(silver)
        before = frame.copy(deep=True)
        outcome = _outcome(frame)
        assert frame.equals(before), "the builder mutated its input frame"
        assert "date_modified" not in frame.columns
        rows = outcome["selection"].rows
        if "date_modified" in rows.columns:
            assert rows["date_modified"].isna().all()


# ---------------------------------------------------------------------------
# 5. THE BACKFILL CASE: a stamp after the lock admits nothing
# ---------------------------------------------------------------------------


class TestABackfillIsExcludedNotPresentedAsHistoricallyKnown:
    def test_backfill_a_2025_row_fetched_in_2026_is_not_admitted(
        self, tmp_path: Path
    ) -> None:
        silver = _ingested_silver(tmp_path, _BACKFILL_STAMP)
        outcome = _outcome(silver)  # exactly as the ingest stores it
        assert outcome["selection"].rule is SelectionRule.NONE_ADMITTED
        provenance = outcome["provenance"]
        assert set(provenance["basis"]) == {InformationBasis.NO_INFORMATION.value}
        assert outcome["features"]["home_injury_coverage"].eq(0.0).all()
        assert outcome["gate"] is SourceCheckState.CHECKED

    def test_backfill_no_provenance_time_is_later_than_its_games_lock(
        self, tmp_path: Path
    ) -> None:
        # Mixed: one game's stamp would be pre-lock for the Sunday game only. Over every case,
        # no reported information time exceeds that game's lock.
        for stamp in (_BACKFILL_STAMP, _LOCK + pd.Timedelta(hours=12), _LOCK):
            root = tmp_path / stamp.strftime("%Y%m%dT%H%M%S")
            outcome = _outcome(_ingested_silver(root, stamp))
            provenance, locks = outcome["provenance"], outcome["locks"]
            for game_id, row in provenance.iterrows():
                if pd.notna(row["information_time"]):
                    assert pd.Timestamp(row["information_time"]) <= pd.Timestamp(
                        locks[game_id]
                    ), (game_id, stamp)

    def test_no_provenance_row_reports_a_third_basis(self, tmp_path: Path) -> None:
        allowed = {
            InformationBasis.PER_ROW.value,
            InformationBasis.NO_INFORMATION.value,
        }
        assert {member.value for member in InformationBasis} == allowed
        for stamp in (_BACKFILL_STAMP, _LOCK):
            root = tmp_path / stamp.strftime("%Y%m%dT%H%M%S")
            outcome = _outcome(_ingested_silver(root, stamp))
            assert set(outcome["provenance"]["basis"]) <= allowed


# ---------------------------------------------------------------------------
# 6. THE NAME TIE
# ---------------------------------------------------------------------------


class TestTheNameTie:
    def test_the_injury_schema_declares_the_column_the_builder_reads(self) -> None:
        assert UPSTREAM_CAPTURE_COLUMN in InjurySchema.model_fields
        assert InjurySchema.model_fields[UPSTREAM_CAPTURE_COLUMN].is_required()

    def test_the_snap_schema_and_both_ingesters_use_the_same_literal(self) -> None:
        import scripts.ingest_injuries as injuries
        import scripts.ingest_snaps as snaps

        assert UPSTREAM_CAPTURE_COLUMN in SnapCountSchema.model_fields
        assert injuries.CAPTURE_COLUMN == UPSTREAM_CAPTURE_COLUMN
        assert snaps.CAPTURE_COLUMN == UPSTREAM_CAPTURE_COLUMN


# ---------------------------------------------------------------------------
# 7. 33.2 review C1 CR-04 = B CR-01: a later capture never erases an earlier one
# ---------------------------------------------------------------------------


def _capture_into(root: Path, stamp: pd.Timestamp, payload: pd.DataFrame) -> None:
    from scripts.ingest_injuries import ingest_injuries_season

    ingest_injuries_season(
        2025,
        loader=lambda _season: payload,
        stamp_reader=lambda _asset, *, fresh=False: stamp.to_pydatetime(),
        games=_games(),
        base_path=root,
    )


class TestCapturesAccumulateAndTheLatestPreLockOneIsRead:
    """The nightly whole-season re-capture used to REPLACE every game's rows latest-wins.

    The night after a game, its pre-lock capture was overwritten by the same reports carrying
    a post-lock stamp, so for 2025+ the game lost the only admissible record it had.
    """

    @pytest.fixture(autouse=True)
    def _distinct_bronze_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two captures inside one second would (correctly) collide on the bronze name."""
        from itertools import count

        from data import storage

        seconds = count()

        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return datetime(2026, 9, 22, 6, 0, tzinfo=UTC) + timedelta(
                    seconds=next(seconds)
                )

        monkeypatch.setattr(storage, "datetime", _Clock)

    def test_a_post_lock_recapture_keeps_the_pre_lock_capture_admitted(
        self, tmp_path: Path
    ) -> None:
        before = _LOCK - pd.Timedelta(hours=5)
        _capture_into(tmp_path, before, _raw_2025_payload())
        _capture_into(tmp_path, _LOCK + pd.Timedelta(days=1), _raw_2025_payload())

        silver = pd.read_parquet(tmp_path / "silver" / "injuries.parquet")
        stamps = set(pd.to_datetime(silver[UPSTREAM_CAPTURE_COLUMN], utc=True))
        assert stamps == {before, _LOCK + pd.Timedelta(days=1)}, "a capture was erased"

        outcome = _outcome(silver)
        assert outcome["selection"].rule is SelectionRule.CAPTURE_STAMP
        assert outcome["selection"].information_time == before
        assert outcome["features"].loc[_GAME_ID, "home_qb_out_flag"] == 1.0

    def test_an_identical_recapture_is_idempotent(self, tmp_path: Path) -> None:
        stamp = _LOCK - pd.Timedelta(hours=5)
        _capture_into(tmp_path, stamp, _raw_2025_payload())
        first = pd.read_parquet(tmp_path / "silver" / "injuries.parquet")
        _capture_into(tmp_path, stamp, _raw_2025_payload())
        again = pd.read_parquet(tmp_path / "silver" / "injuries.parquet")
        assert len(again) == len(first)

    def test_the_game_reads_one_capture_never_a_mix_of_captures(
        self, tmp_path: Path
    ) -> None:
        """Mahomes is Out in the earlier capture and cleared from the later one."""
        early, late = _LOCK - pd.Timedelta(days=2), _LOCK - pd.Timedelta(hours=2)
        _capture_into(tmp_path, early, _raw_2025_payload())
        cleared = _raw_2025_payload()
        kc = cleared["gsis_id"] == _QB1_ID
        cleared.loc[kc, ["gsis_id", "position", "full_name"]] = [
            "00-0077777",
            "WR",
            "KC Receiver",
        ]
        cleared.loc[kc, "report_status"] = "Questionable"
        _capture_into(tmp_path, late, cleared)

        outcome = _outcome(pd.read_parquet(tmp_path / "silver" / "injuries.parquet"))
        assert outcome["selection"].information_time == late
        assert _QB1_ID not in set(outcome["selection"].rows["gsis_id"]), (
            "a player the latest pre-lock capture no longer lists was carried forward"
        )
        assert outcome["features"].loc[_GAME_ID, "home_qb_out_flag"] == 0.0
