"""Silver ``odds_snapshot`` is internally consistent and every change is on the record (Plan 33.2-08).

SPEC R12: zero rows whose spread sign contradicts their own moneyline (a pick'em spread of 0 is
not a conflict), no stray ``snapshot_ts=`` folder under ``data/silver/``, and every corrected or
nulled value named in ``config/odds_corrections.toml`` with a citation or a reason.

Two controls keep a clean result from being a vacuous one: a planted conflicting row IS flagged
(so "zero conflicts" cannot come from a detector that finds nothing), and the detector run under
the INVERTED convention flags far more rows than under the stored one (so the detector provably
assumes the convention the column is actually stored in). Only the inequality is asserted: the
inverted count has been measured as 2,125 and as 2,134 depending on how equal moneylines are
counted, and a pinned literal would make a correct run fail.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts.repair_odds_snapshot import (
    DISPOSITION_CORRECTED,
    DISPOSITION_NULLED,
    STRAY_BOOK_NOT_CARRIED,
    STRAY_DUPLICATE,
    STRAY_DUPLICATE_AFTER_CANONICAL_ID,
    STRAY_SAME_KEY_DIFFERENT_VALUE,
    ArchiveQuote,
    classify_stray_rows,
    insert_if_absent,
    latest_pre_kickoff_timeline,
    resolve_sign_conflicts,
    sign_conflict_mask,
    timeline_in_snapshot_convention,
)

SILVER = Path("data/silver")
ODDS_PATH = SILVER / "odds_snapshot.parquet"
TIMELINE_PATH = SILVER / "odds_timeline.parquet"
GAMES_PATH = SILVER / "games.parquet"
RECORD_PATH = Path("config/odds_corrections.toml")

needs_lake = pytest.mark.skipif(
    not ODDS_PATH.exists(), reason="silver odds_snapshot not built"
)


def _row(game_id: str, spread: float, ml_home: float, ml_away: float) -> dict:
    return {
        "game_id": game_id,
        "sportsbook": "consensus",
        "spread": spread,
        "total": 44.0,
        "ml_home": ml_home,
        "ml_away": ml_away,
    }


class TestTheDetector:
    def test_a_planted_conflict_is_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _row("home_by_spread_away_by_price", 3.0, 120.0, -140.0),
                _row("away_by_spread_home_by_price", -3.0, -140.0, 120.0),
            ]
        )
        assert sign_conflict_mask(frame).tolist() == [True, True]

    def test_agreeing_rows_are_not_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _row("home_favoured", 3.0, -140.0, 120.0),
                _row("away_favoured", -3.0, 120.0, -140.0),
            ]
        )
        assert not sign_conflict_mask(frame).any()

    def test_a_pick_em_spread_is_never_a_conflict(self) -> None:
        frame = pd.DataFrame(
            [
                _row("pick_em_home_price", 0.0, -120.0, 100.0),
                _row("pick_em_away_price", 0.0, 100.0, -120.0),
            ]
        )
        assert not sign_conflict_mask(frame).any()
        assert not sign_conflict_mask(frame, inverted=True).any()

    def test_a_missing_price_is_not_decidable(self) -> None:
        frame = pd.DataFrame([_row("no_price", 3.0, math.nan, -140.0)])
        assert not sign_conflict_mask(frame).any()


class TestTheOwnedTimelineIsFlippedOnceAtTheReader:
    def test_the_reader_flips_the_sign(self) -> None:
        raw = pd.DataFrame(
            {
                "game_id": ["g"],
                "snapshot_ts": [pd.Timestamp("2021-01-01", tz="UTC")],
                "spread": [-2.5],
                "total": [45.0],
            }
        )
        assert timeline_in_snapshot_convention(raw)["spread"].tolist() == [2.5]
        assert raw["spread"].tolist() == [-2.5], "the reader must not mutate its input"

    @pytest.mark.skipif(not TIMELINE_PATH.exists(), reason="odds_timeline not built")
    def test_a_known_game_reads_home_favoured_after_the_flip(self) -> None:
        # 2021_W15_NE@IND: every archive provider names IND the favourite. The timeline STORES
        # -2.5 (its inverted convention); read through the reader it is +2.5 = home favoured.
        timeline = pd.read_parquet(TIMELINE_PATH)
        games = pd.read_parquet(GAMES_PATH, columns=["game_id", "kickoff_et"])
        kickoffs = pd.to_datetime(games.set_index("game_id")["kickoff_et"], utc=True)
        latest = latest_pre_kickoff_timeline(
            timeline_in_snapshot_convention(timeline), kickoffs
        )
        spread, snapshot = latest["2021_W15_NE@IND"]
        assert spread == 2.5
        assert snapshot < kickoffs["2021_W15_NE@IND"]


class TestResolution:
    ARCHIVE = {
        "consistent": ArchiveQuote(
            "1", "consensus", "2020-01-01", 1.0, -120.0, 100.0, 44.0
        ),
        "inconsistent": ArchiveQuote(
            "2", "consensus", "2020-01-01", 1.0, 100.0, -120.0, 44.0
        ),
        "disagrees": ArchiveQuote(
            "3", "consensus", "2020-01-01", -1.5, 100.0, -120.0, 44.0
        ),
    }
    SNAP = pd.Timestamp("2020-01-01 12:00", tz="UTC")

    def _resolve(self, game_id: str, timeline: dict) -> list:
        frame = pd.DataFrame([_row(game_id, 1.0, 100.0, -110.0)])
        return resolve_sign_conflicts(frame, timeline, self.ARCHIVE)

    def test_a_confirmed_spread_takes_a_consistent_archive_moneyline(self) -> None:
        fixes = self._resolve("consistent", {"consistent": (1.0, self.SNAP)})
        assert {(f.column, f.new_value, f.disposition) for f in fixes} == {
            ("ml_home", -120.0, DISPOSITION_CORRECTED),
            ("ml_away", 100.0, DISPOSITION_CORRECTED),
        }
        assert all(f.source_url for f in fixes)

    def test_an_inconsistent_archive_moneyline_is_nulled_with_a_reason(self) -> None:
        fixes = self._resolve("inconsistent", {"inconsistent": (1.0, self.SNAP)})
        assert {f.column for f in fixes} == {"ml_home", "ml_away"}
        assert all(
            f.disposition == DISPOSITION_NULLED and f.new_value is None and f.reason
            for f in fixes
        )

    def test_a_wrong_spread_takes_the_timeline_value(self) -> None:
        fixes = self._resolve("consistent", {"consistent": (-2.0, self.SNAP)})
        # The timeline and the archive DISAGREE here (-2.0 against +1.0): both are nulled.
        assert {f.disposition for f in fixes} == {DISPOSITION_NULLED}
        fixes = self._resolve("no_archive", {"no_archive": (-2.0, self.SNAP)})
        assert [(f.column, f.new_value) for f in fixes] == [("spread", -2.0)]
        assert "odds_timeline" in fixes[0].source_url

    def test_disagreeing_sources_null_both_disputed_values(self) -> None:
        fixes = self._resolve("disagrees", {"disagrees": (1.0, self.SNAP)})
        assert {f.column for f in fixes} == {"spread", "ml_home", "ml_away"}
        assert all(
            f.disposition == DISPOSITION_NULLED and "disagree" in f.reason
            for f in fixes
        )

    def test_no_evidence_nulls_rather_than_guesses(self) -> None:
        fixes = self._resolve("nothing", {})
        assert all(f.disposition == DISPOSITION_NULLED and f.reason for f in fixes)


class TestStrayRecoveryNeverOverwrites:
    MAIN = pd.DataFrame([_row("2023_W01_ARI@WAS", -7.0, 250.0, -300.0)])

    def test_an_existing_key_is_never_overwritten(self) -> None:
        candidate = pd.DataFrame([_row("2023_W01_ARI@WAS", 99.0, 1.0, 1.0)])
        combined, inserted = insert_if_absent(self.MAIN, candidate)
        assert inserted == 0
        assert combined["spread"].tolist() == [-7.0]

    def test_an_absent_key_is_inserted_and_counted(self) -> None:
        candidate = pd.DataFrame([_row("2023_W01_BUF@NYJ", 2.5, -130.0, 110.0)])
        combined, inserted = insert_if_absent(self.MAIN, candidate)
        assert inserted == 1 and len(combined) == len(self.MAIN) + 1

    def test_the_classifier_separates_duplicates_disagreements_and_other_books(
        self,
    ) -> None:
        main = pd.DataFrame(
            [
                _row("2023_W01_ARI@WAS", -7.0, 250.0, -300.0),
                _row("2023_W01_SEA@LA", 3.0, -150.0, 130.0),
            ]
        )
        strays = pd.DataFrame(
            [
                _row("2023_W01_ARI@WAS", -7.0, 250.0, -300.0),
                _row("2023_W01_SEA@LAR", 3.0, -150.0, 130.0),
                _row("2023_W01_ARI@WAS", -6.5, 250.0, -300.0),
                {
                    **_row("2023_W01_ARI@WAS", -7.0, 250.0, -300.0),
                    "sportsbook": "fanduel",
                },
            ]
        )
        labelled = classify_stray_rows(strays, main, set(main["game_id"]))
        assert labelled["category"].tolist() == [
            STRAY_DUPLICATE,
            STRAY_DUPLICATE_AFTER_CANONICAL_ID,
            STRAY_SAME_KEY_DIFFERENT_VALUE,
            STRAY_BOOK_NOT_CARRIED,
        ]


@needs_lake
class TestTheLiveTableIsClean:
    def test_zero_sign_conflicts_remain_on_comparable_rows(self) -> None:
        odds = pd.read_parquet(ODDS_PATH)
        comparable = (
            odds["spread"].notna()
            & (odds["spread"] != 0)
            & odds["ml_home"].notna()
            & odds["ml_away"].notna()
        )
        assert int(comparable.sum()) > 0, (
            "a zero-conflict result over zero comparable rows is vacuous"
        )
        conflicts = odds.loc[sign_conflict_mask(odds), "game_id"].tolist()
        assert conflicts == []

    def test_the_inverted_convention_flags_far_more_rows(self) -> None:
        odds = pd.read_parquet(ODDS_PATH)
        stored = int(sign_conflict_mask(odds).sum())
        inverted = int(sign_conflict_mask(odds, inverted=True).sum())
        comparable = int((odds["spread"].notna() & (odds["spread"] != 0)).sum())
        assert inverted > stored
        assert inverted > comparable / 2, (
            "under the wrong reading most rows must look like conflicts"
        )

    def test_no_stray_partition_folder_remains(self) -> None:
        assert [p.name for p in SILVER.glob("snapshot_ts=*")] == []


@needs_lake
class TestTheCorrectionRecord:
    @pytest.fixture(scope="class")
    def record(self) -> dict:
        assert RECORD_PATH.exists(), f"{RECORD_PATH} is the committed record R12 reads"
        return tomllib.loads(RECORD_PATH.read_text(encoding="utf-8"))

    def test_every_recorded_value_is_in_the_table(self, record: dict) -> None:
        corrections = record.get("correction", [])
        assert corrections, "the record carries no [[correction]] entries"
        odds = pd.read_parquet(ODDS_PATH).set_index("game_id")
        for entry in corrections:
            stored = odds.at[entry["game_id"], entry["column"]]
            if "new_value" in entry:
                assert stored == entry["new_value"], entry
            else:
                assert pd.isna(stored), entry

    def test_every_non_null_value_carries_a_citation(self, record: dict) -> None:
        for entry in record.get("correction", []):
            if "new_value" in entry:
                assert entry.get("source_url"), entry
            assert entry.get("reason"), entry

    def test_the_disputed_total_is_confirmed_or_corrected_with_a_citation(
        self, record: dict
    ) -> None:
        totals = [
            e
            for e in record.get("correction", [])
            if e["game_id"] == "2023_W18_NYJ@NE" and e["column"] == "total"
        ]
        assert len(totals) == 1 and totals[0].get("source_url")
