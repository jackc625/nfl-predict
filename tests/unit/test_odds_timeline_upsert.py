"""Composite-key idempotency + naive-reject proof for the odds_timeline table.

Covers Phase 29 Plan 02 (D-11, SIG-04): ``upsert_silver_composite`` writes the
``odds_timeline`` trajectory table keyed ``(game_id, snapshot_ts)`` so distinct
snapshots of one game coexist (no ``game_id`` latest-wins clobber) and re-runs
are idempotent. Also proves a timezone-naive ``snapshot_ts`` is REJECTED at
BOTH seams -- the ``OddsTimelineSchema`` validator and the storage writer --
never silently stored as UTC (review 29-02 HIGH).
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from data.schemas import OddsTimelineSchema
from data.storage import upsert_silver_composite

GAME_ID = "2024_W01_KC@BUF"


def _aware(year: int, month: int, day: int, hour: int = 18) -> datetime:
    """A tz-aware UTC datetime."""
    return datetime(year, month, day, hour, 0, tzinfo=UTC)


def _row(game_id: str, snapshot_ts, total: float) -> dict:
    return {
        "game_id": game_id,
        "snapshot_ts": snapshot_ts,
        "total": total,
        "spread": -3.0,
        "sportsbook": "consensus",
        "region": "us",
    }


def _read(path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["snapshot_ts"] = pd.to_datetime(out["snapshot_ts"], utc=True)
    return out


def test_double_run_is_a_noop(tmp_path):
    """Writing the same (game_id, snapshot_ts) rows twice yields one row per pair."""
    df = pd.DataFrame(
        [
            _row(GAME_ID, _aware(2024, 9, 6), 47.5),
            _row(GAME_ID, _aware(2024, 9, 5), 48.0),
        ]
    )
    upsert_silver_composite(df, "odds_timeline", base_path=tmp_path)
    path = upsert_silver_composite(df, "odds_timeline", base_path=tmp_path)

    out = _read(path)
    assert len(out) == 2
    pairs = set(zip(out["game_id"], out["snapshot_ts"], strict=True))
    assert len(pairs) == 2


def test_distinct_snapshots_for_one_game_coexist(tmp_path):
    """Two distinct snapshot_ts for one game coexist -- the D-11 no-clobber property."""
    df = pd.DataFrame(
        [
            _row(GAME_ID, _aware(2024, 9, 3), 49.0),  # opening
            _row(GAME_ID, _aware(2024, 9, 6), 47.0),  # freeze
        ]
    )
    path = upsert_silver_composite(df, "odds_timeline", base_path=tmp_path)

    out = _read(path)
    assert len(out) == 2
    assert (out["game_id"] == GAME_ID).all()
    assert out["snapshot_ts"].nunique() == 2


def test_rewriting_a_pair_keeps_latest_value(tmp_path):
    """Re-writing one (game_id, snapshot_ts) with a changed value keeps='last'."""
    ts = _aware(2024, 9, 6)
    upsert_silver_composite(
        pd.DataFrame([_row(GAME_ID, ts, 47.5)]), "odds_timeline", base_path=tmp_path
    )
    path = upsert_silver_composite(
        pd.DataFrame([_row(GAME_ID, ts, 50.0)]), "odds_timeline", base_path=tmp_path
    )

    out = _read(path)
    assert len(out) == 1
    assert out.iloc[0]["total"] == 50.0


def test_naive_snapshot_ts_raises_at_writer(tmp_path):
    """A timezone-naive snapshot_ts RAISES at the storage writer (not stored as UTC)."""
    df = pd.DataFrame([_row(GAME_ID, datetime(2024, 9, 6, 18, 0), 47.5)])  # naive
    with pytest.raises(ValueError, match="naive"):
        upsert_silver_composite(df, "odds_timeline", base_path=tmp_path)

    assert not (tmp_path / "silver" / "odds_timeline.parquet").exists()


def test_naive_snapshot_ts_raises_at_schema():
    """The OddsTimelineSchema validator RAISES on a naive snapshot_ts (the other seam)."""
    with pytest.raises(ValueError):
        OddsTimelineSchema(
            game_id=GAME_ID,
            snapshot_ts=datetime(2024, 9, 6, 18, 0),  # naive
            total=47.5,
        )
