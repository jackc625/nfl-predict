"""Postseason injury reports are ingested and timed like any other (Plan 33.2-15 extra step 6b).

Plan 33.2-13 recorded that silver ``injuries`` held regular-season rows only, so every postseason
game (11-13 a season) resolved to "no report admitted", and routed the question to this plan:
can the postseason reports be fetched from the free source, honestly timed? Measured 2026-09-22:
YES -- upstream publishes 3,544 postseason rows over 2009-2025, with a per-row ``date_modified``
for 2010-2024. The ingest's own ``game_type == "REG"`` filter was what dropped them.

Asserted here, each in a temporary data root:

* the default keeps every game type (REG and the four postseason rounds);
* ``game_types=POSTSEASON_GAME_TYPES`` adds postseason reports WITHOUT touching a stored
  regular-season row (the backfill's shape);
* an upstream game type outside the known vocabulary is REFUSED by name, not silently dropped;
* a postseason report is admitted for its game exactly when its ``date_modified`` is at or
  before that game's lock -- the same fence a regular-season report meets.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.injury import InjuryBuilder, SelectionRule
from features.provenance import InformationBasis
from features.snaps import SnapCountBuilder
from scripts.ingest_injuries import (
    INJURY_GAME_TYPES,
    POSTSEASON_GAME_TYPES,
    ingest_injuries_season,
)
from utils.exceptions import DataValidationError

ET = ZoneInfo("America/New_York")
STAMP = datetime(2025, 2, 13, 7, 16, 35, tzinfo=UTC)

# The 2024 Wild Card: KC hosts on a Saturday; its lock is Friday 18:00 ET.
_REG_ID = "2024_W18_KC@DEN"
_WC_ID = "2024_W19_HOU@KC"
_WC_KICKOFF = datetime(2025, 1, 18, 16, 30, tzinfo=ET)
_WC_LOCK = pd.Timestamp(lock_rule.game_lock(_WC_KICKOFF))


def _games() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "game_id": _REG_ID,
                "season": 2024,
                "week": 18,
                "home_team": "DEN",
                "away_team": "KC",
                "kickoff_et": datetime(2025, 1, 5, 16, 25, tzinfo=ET),
                "game_type": "REG",
            },
            {
                "game_id": _WC_ID,
                "season": 2024,
                "week": 19,
                "home_team": "KC",
                "away_team": "HOU",
                "kickoff_et": _WC_KICKOFF,
                "game_type": "WC",
            },
        ]
    )
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _raw(wc_modified: pd.Timestamp, extra_type: str | None = None) -> pd.DataFrame:
    rows = [
        {"week": 18, "game_type": "REG", "team": "KC", "gsis_id": "00-0033873",
         "report_status": "Questionable", "date_modified": pd.Timestamp("2025-01-03T20:00Z")},
        {"week": 19, "game_type": "WC", "team": "KC", "gsis_id": "00-0033873",
         "report_status": "Out", "date_modified": wc_modified},
    ]  # fmt: skip
    if extra_type is not None:
        rows.append(dict(rows[1], game_type=extra_type))
    frame = pd.DataFrame(rows)
    frame["season"] = 2024
    frame["position"] = "QB"
    frame["full_name"] = "Patrick Mahomes"
    for column in (
        "report_primary_injury",
        "report_secondary_injury",
        "practice_status",
        "practice_primary_injury",
        "practice_secondary_injury",
    ):
        frame[column] = None
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


def _ingest(root: Path, raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ingest_injuries_season(
        2024,
        loader=lambda _season: raw,
        stamp_reader=lambda _asset, *, fresh=False: STAMP,
        games=_games(),
        base_path=root,
        **kwargs,
    )
    return pd.read_parquet(root / "silver" / "injuries.parquet")


class TestTheVocabulary:
    def test_regular_season_plus_the_four_rounds(self) -> None:
        assert INJURY_GAME_TYPES == ("REG", "WC", "DIV", "CON", "SB")
        assert POSTSEASON_GAME_TYPES == ("WC", "DIV", "CON", "SB")


class TestPostseasonReportsAreKept:
    def test_the_default_keeps_regular_season_and_postseason(
        self, tmp_path: Path
    ) -> None:
        silver = _ingest(tmp_path, _raw(_WC_LOCK - pd.Timedelta(hours=3)))
        assert sorted(silver["game_type"]) == ["REG", "WC"]
        assert set(silver["game_id"]) == {_REG_ID, _WC_ID}

    def test_the_backfill_shape_adds_postseason_without_touching_regular_season(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from data import storage

        # Two captures of one season inside a second would (correctly) collide on the bronze
        # filename; pin the filename clock to two distinct instants.
        instants = [
            datetime(2026, 9, 22, 6, 0, 0, tzinfo=UTC),
            datetime(2026, 9, 22, 6, 0, 5, tzinfo=UTC),
        ]

        class _Clock(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return instants.pop(0)

        monkeypatch.setattr(storage, "datetime", _Clock)
        raw = _raw(_WC_LOCK - pd.Timedelta(hours=3))
        _ingest(tmp_path, raw, game_types=("REG",))
        before = pd.read_parquet(tmp_path / "silver" / "injuries.parquet")
        after = _ingest(tmp_path, raw, game_types=POSTSEASON_GAME_TYPES)
        regular_before = before.loc[before["game_type"] == "REG"].reset_index(drop=True)
        regular_after = after.loc[after["game_type"] == "REG"].reset_index(drop=True)
        assert regular_after.equals(regular_before)
        assert (after["game_type"] == "WC").sum() == 1

    def test_an_unknown_upstream_game_type_is_refused(self, tmp_path: Path) -> None:
        raw = _raw(_WC_LOCK - pd.Timedelta(hours=3), extra_type="PRO")
        with pytest.raises(DataValidationError, match="PRO"):
            _ingest(tmp_path, raw)
        assert not (tmp_path / "silver" / "injuries.parquet").exists()

    def test_an_unknown_requested_game_type_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="XX"):
            _ingest(tmp_path, _raw(_WC_LOCK), game_types=("XX",))


class TestAPostseasonReportMeetsTheSameFence:
    @staticmethod
    def _selection(silver: pd.DataFrame):
        games = _games()
        builder = InjuryBuilder(
            snap_builder=SnapCountBuilder(snaps_df=pd.DataFrame(), schedule_df=games),
            injuries_df=silver,
            depth_charts_df=pd.DataFrame(),
            pbp_df=pd.DataFrame(),
        )
        game = games.loc[games["game_id"] == _WC_ID].iloc[0].to_dict()
        selection = builder.fenced_injuries(game, _WC_LOCK)
        provenance = builder.information_times(games).set_index("game_id")
        return selection, provenance.loc[_WC_ID]

    def test_a_report_at_the_lock_is_admitted_on_date_modified(
        self, tmp_path: Path
    ) -> None:
        selection, provenance = self._selection(_ingest(tmp_path, _raw(_WC_LOCK)))
        assert selection.rule is SelectionRule.DATE_MODIFIED
        assert selection.information_time == _WC_LOCK
        assert provenance["basis"] == InformationBasis.PER_ROW.value

    def test_one_second_after_the_lock_is_not(self, tmp_path: Path) -> None:
        late = _WC_LOCK + pd.Timedelta(seconds=1)
        selection, provenance = self._selection(_ingest(tmp_path, _raw(late)))
        assert selection.rule is SelectionRule.NONE_ADMITTED
        assert provenance["basis"] == InformationBasis.NO_INFORMATION.value
