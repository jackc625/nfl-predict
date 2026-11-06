"""The v2 gold build: derived columns, the builder's three seams, the families and the write."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data.write_sink import current_sink
from features.opponent_adj_windowed import WindowedOpponentAdjuster
from features.provenance import InformationTimeViolation, build_lock_frame
from features.qb_v2 import QBTrackerV2
from scripts import build_gold_v2 as v2
from scripts.build_features import FeatureMatrixBuilder
from scripts.build_gold_v2 import (
    RAW_WINDOWED_EPA_COLUMNS,
    V2_DERIVED_COLUMNS,
    FeatureMatrixBuilderV2,
    add_v2_derived_columns,
    add_v2_families,
    build_gold_v2,
)

ET = "America/New_York"
GAME_ID = "2024_W01_BAL@KC"

# One game's inputs, chosen so every derived column is a distinct round number.
_DERIVED_INPUTS = {
    "home_off_rolling_success_rate": 0.50,
    "home_def_rolling_success_rate": 0.45,
    "away_off_rolling_success_rate": 0.40,
    "away_def_rolling_success_rate": 0.45,
    "home_off_rolling_opp_adj_epa_per_play": 0.20,
    "home_def_rolling_opp_adj_epa_per_play": -0.10,
    "away_off_rolling_opp_adj_epa_per_play": 0.10,
    "away_def_rolling_opp_adj_epa_per_play": 0.05,
    "home_qb_adjustment": 1.0,
    "away_qb_adjustment": -0.5,
    "home_off_rolling_neutral_pace": 66.0,
    "away_off_rolling_neutral_pace": 64.0,
}


def _games() -> pd.DataFrame:
    """One silver game: kickoff Thursday 20:20 ET, so the lock is Wednesday 18:00 ET."""
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "season": [2024],
            "week": [1],
            "kickoff_et": [pd.Timestamp("2024-09-05 20:20", tz=ET)],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "neutral_site": [False],
            "home_score": [27.0],
            "away_score": [20.0],
        }
    )


def _matrix(labels: dict[str, float]) -> pd.DataFrame:
    """A one-row gold matrix: meta, scores, the given labels, and the derived inputs."""
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "season": [2024],
            "week": [1],
            "feature_timestamp": [pd.Timestamp("2026-11-05 12:00", tz="UTC")],
            "home_score": [27.0],
            "away_score": [20.0],
            **{name: [value] for name, value in labels.items()},
            "elo_diff": [87.0],
            "hfa_used": [48.0],
            **{name: [value] for name, value in _DERIVED_INPUTS.items()},
        }
    )


def _three_matrices() -> dict[str, pd.DataFrame]:
    return {
        "wp": _matrix({"target_wp": 1.0, "home_win": 1.0}),
        "ats": _matrix({"home_margin": 7.0, "point_differential": 7.0}),
        "ou": _matrix({"total_points": 47.0}),
    }


def _market_family(information_time: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_mkt_rating": [3.0],
            "away_mkt_rating": [1.0],
            "mkt_rating_diff": [2.0],
            "mkt_total_sum": [46.5],
            "mkt_rating_coverage": [1.0],
            "information_time": [information_time],
        }
    )


def _league_family() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "league_points_level": [44.0],
            "information_time": [pd.Timestamp("2024-02-12 03:30", tz="UTC")],
        }
    )


@pytest.fixture
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny frames in place of the real build, the silver/nflverse loads and the families."""

    def fake_generate(self, **_kwargs):
        return _three_matrices()

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", fake_generate
    )
    monkeypatch.setattr(v2, "load_dataframe", lambda *_args, **_kwargs: _games())
    monkeypatch.setattr(v2, "load_schedules", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        v2,
        "market_ratings",
        lambda *_args: _market_family(pd.Timestamp("2024-02-12 03:30", tz="UTC")),
    )
    monkeypatch.setattr(v2, "league_points_level", lambda *_args: _league_family())


def test_derived_columns_have_the_stated_values():
    out = add_v2_derived_columns(
        pd.DataFrame({k: [v] for k, v in _DERIVED_INPUTS.items()})
    )
    assert out.loc[0, "success_rate_net"] == pytest.approx(0.10)
    assert out.loc[0, "opp_adj_epa_net"] == pytest.approx(0.25)
    assert out.loc[0, "qb_value_diff"] == pytest.approx(1.5)
    assert out.loc[0, "off_epa_sum"] == pytest.approx(0.30)
    assert out.loc[0, "def_epa_sum"] == pytest.approx(-0.05)
    assert out.loc[0, "pace_sum"] == pytest.approx(130.0)
    assert set(V2_DERIVED_COLUMNS) <= set(out.columns)


def test_v2_builder_swaps_in_the_repaired_pieces_and_skips_normalization():
    b = FeatureMatrixBuilderV2()
    assert isinstance(b.opponent_adj, WindowedOpponentAdjuster)
    assert b.opponent_adj.window == 10
    assert b.opponent_adj.min_opponent_games == 4
    assert isinstance(b.qb_tracker, QBTrackerV2)
    frame = pd.DataFrame({"elo_diff": [87.0]})
    assert b.normalize_combined_features(frame).equals(frame)


def test_v2_builder_keeps_the_raw_windowed_epa_the_live_merge_drops(monkeypatch):
    def live_merge(self, combined, *_args):
        dropped = combined.drop(columns=list(RAW_WINDOWED_EPA_COLUMNS))
        return dropped.assign(home_off_rolling_opp_adj_epa_per_play=0.3)

    monkeypatch.setattr(FeatureMatrixBuilder, "_merge_opponent_adjusted", live_merge)
    raw = {name: float(i) for i, name in enumerate(RAW_WINDOWED_EPA_COLUMNS, start=1)}
    combined = pd.DataFrame({"game_id": [GAME_ID], **{k: [v] for k, v in raw.items()}})

    out = FeatureMatrixBuilderV2()._merge_opponent_adjusted(
        combined, _games(), build_lock_frame(_games()), datetime.now(UTC), None, None
    )

    assert out.loc[0, "home_off_rolling_opp_adj_epa_per_play"] == 0.3
    for name, value in raw.items():
        assert out.loc[0, name] == value


def test_families_are_refused_when_their_information_postdates_the_lock(monkeypatch):
    after_lock = pd.Timestamp(
        "2024-09-05 12:00", tz="UTC"
    )  # lock: 2024-09-04 22:00 UTC
    monkeypatch.setattr(v2, "market_ratings", lambda *_args: _market_family(after_lock))
    monkeypatch.setattr(v2, "league_points_level", lambda *_args: _league_family())
    with pytest.raises(InformationTimeViolation):
        add_v2_families(_three_matrices(), _games(), pd.DataFrame())


def test_build_writes_only_three_parquet_files_and_a_manifest(tmp_path, fake_sources):
    matrices = build_gold_v2(tmp_path / "gold_v2")
    assert sorted(p.name for p in (tmp_path / "gold_v2").iterdir()) == [
        "features_ats.parquet",
        "features_ou.parquet",
        "features_wp.parquet",
        "manifest.json",
    ]
    written = pd.read_parquet(tmp_path / "gold_v2" / "features_ats.parquet")
    assert written.loc[0, "mkt_rating_diff"] == 2.0
    assert written.loc[0, "league_points_level"] == 44.0
    assert written.loc[0, "pace_sum"] == pytest.approx(130.0)
    assert not any("information_time" in c for c in written.columns)
    assert set(matrices) == {"wp", "ats", "ou"}


def test_build_refuses_when_the_builder_asked_to_write(
    tmp_path, fake_sources, monkeypatch
):
    def writing_generate(self, **_kwargs):
        current_sink().authorize("gold/features_wp", "save_dataframe", 1)
        return _three_matrices()

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", writing_generate
    )
    with pytest.raises(RuntimeError, match="write"):
        build_gold_v2(tmp_path / "gold_v2")
    assert not (tmp_path / "gold_v2").exists()


@pytest.mark.parametrize("out_dir", ["data/gold_v2", "data/gold/v2"])
def test_build_refuses_an_output_directory_under_data(
    out_dir, fake_sources, monkeypatch
):
    # Every source is faked and the build itself raises, so even a broken guard can never
    # reach real data or write into the live lake; it fails here instead.
    def unreachable_build(self, **_kwargs):
        raise AssertionError("the data/ refusal must come before any build work")

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", unreachable_build
    )
    with pytest.raises(ValueError, match="data"):
        build_gold_v2(Path(out_dir))
