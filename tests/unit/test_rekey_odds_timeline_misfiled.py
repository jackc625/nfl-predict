"""The misfiled-row repair re-keys ONLY rows the retired week count mis-filed.

Plan 33.2-24 step 24b. ``scripts/rekey_odds_timeline_misfiled.py`` tells a MISFILED stored
id (a real game filed a week early) from an id that NAMES NO GAME (a suspended game, a
speculative playoff pairing) by replaying the retired count forward over the schedule --
an exact re-derivation, never a guess from team names. These tests pin that split, the
in-place rewrite that moves only ``game_id`` on only the misfiled rows, and the refusals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.rekey_odds_timeline_misfiled as repair

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER = REPO_ROOT / "data" / "silver"

GAMES = pd.DataFrame(
    {
        "game_id": ["2024_W16_DET@CHI", "2024_W17_KC@PIT", "2024_W17_BAL@HOU"],
        "season": [2024, 2024, 2024],
        "week": [16, 17, 17],
        "home_team": ["CHI", "PIT", "HOU"],
        "away_team": ["DET", "KC", "BAL"],
        "kickoff_et": pd.to_datetime(
            ["2024-12-22T18:00:00Z", "2024-12-25T18:00:00Z", "2024-12-25T21:30:00Z"],
            utc=True,
        ),
    }
)


def _timeline() -> pd.DataFrame:
    rows = [
        ("2024_W16_DET@CHI", "2024-12-20T22:55:38Z", 44.0, -3.0),
        ("2024_W16_KC@PIT", "2024-12-20T22:55:38Z", 42.5, 2.0),
        ("2024_W16_KC@PIT", "2024-12-24T16:55:38Z", 44.0, 2.75),
        ("2024_W16_BAL@HOU", "2024-12-24T16:55:38Z", 46.5, 5.5),
        ("2024_W19_DET@LA", "2025-01-01T16:55:38Z", 51.0, -3.0),
    ]
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "snapshot_ts": pd.to_datetime([r[1] for r in rows], utc=True),
            "total": [r[2] for r in rows],
            "spread": [r[3] for r in rows],
            "sportsbook": ["consensus_median"] * len(rows),
            "region": ["us"] * len(rows),
            "created_at": pd.to_datetime(
                ["2026-08-16T18:10:33Z"] * len(rows), utc=True
            ),
        }
    )


def test_the_frozen_replay_reproduces_the_stored_christmas_keys():
    """The replay is only worth anything if it produces the ids actually stored."""
    assert repair.retired_week_count(pd.Timestamp("2024-12-25T18:00:00Z")) == (2024, 16)
    assert repair.retired_week_count(pd.Timestamp("2024-12-22T18:00:00Z")) == (2024, 16)


def test_misfiled_and_no_game_ids_are_told_apart():
    classification = repair.classify_unjoined_ids(_timeline(), GAMES)

    assert classification.rekey_map == {
        "2024_W16_BAL@HOU": "2024_W17_BAL@HOU",
        "2024_W16_KC@PIT": "2024_W17_KC@PIT",
    }
    assert classification.no_game_ids == ("2024_W19_DET@LA",)


def test_only_game_id_moves_and_only_on_the_misfiled_rows():
    before = _timeline()
    report = repair.prepare_repair(before, GAMES)

    assert report.rows_rekeyed == 3
    changed = report.after["game_id"] != before["game_id"]
    assert changed.tolist() == [False, True, True, True, False]
    others = [c for c in before.columns if c != "game_id"]
    pd.testing.assert_frame_equal(report.after[others], before[others])
    assert report.after.loc[changed, "game_id"].tolist() == [
        "2024_W17_KC@PIT",
        "2024_W17_KC@PIT",
        "2024_W17_BAL@HOU",
    ]
    assert report.pair_list_sha256_before != report.pair_list_sha256_after


def test_a_target_that_already_carries_rows_is_refused():
    timeline = pd.concat(
        [_timeline(), _timeline().iloc[[1]].assign(game_id="2024_W17_KC@PIT")],
        ignore_index=True,
    )
    with pytest.raises(repair.MisfiledRekeyError, match="already carry rows"):
        repair.prepare_repair(timeline, GAMES)


def test_a_row_captured_after_its_games_kickoff_is_refused():
    timeline = _timeline()
    timeline.loc[3, "snapshot_ts"] = pd.Timestamp("2024-12-26T00:00:00Z")
    with pytest.raises(repair.MisfiledRekeyError, match="at or after their game"):
        repair.prepare_repair(timeline, GAMES)


def test_apply_rewrites_once_and_a_second_apply_is_a_no_op(tmp_path):
    silver = tmp_path / "silver"
    silver.mkdir()
    _timeline().to_parquet(silver / "odds_timeline.parquet", index=False)
    GAMES.to_parquet(silver / "games.parquet", index=False)
    backup = tmp_path / "backup.parquet"

    report, written = repair.apply_repair(
        tmp_path, backup_to=backup, enforce_planned_state=False
    )

    assert written is True
    assert backup.exists()
    stored = pd.read_parquet(silver / "odds_timeline.parquet")
    assert sorted(set(stored["game_id"])) == [
        "2024_W16_DET@CHI",
        "2024_W17_BAL@HOU",
        "2024_W17_KC@PIT",
        "2024_W19_DET@LA",
    ]
    assert repair.pair_list_sha256(stored) == report.pair_list_sha256_after

    again, written_again = repair.apply_repair(
        tmp_path, backup_to=None, enforce_planned_state=False
    )
    assert written_again is False
    assert again.already_repaired


def test_the_planned_state_guard_refuses_a_different_archive():
    report = repair.prepare_repair(_timeline(), GAMES)
    with pytest.raises(repair.MisfiledRekeyError):
        repair.assert_planned_state(report)
