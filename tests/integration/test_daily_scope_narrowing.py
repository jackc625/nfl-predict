"""The gold build's scope, as the daily run uses it (Plan 33.2-27).

Two facts are pinned here:

1. SEASON-ALONE NARROWING. ``load_all_feature_sources`` narrowed ``team_form_features`` and
   ``weather_features`` only when BOTH a season and a week were given, so a season-only build left
   both silver sources unfiltered, left-merged and median-filled (the WR-10 defect). Both now
   narrow on the season alone, and on the week as well when one is given.
2. THE DAILY BUILD IS THE FULL HISTORY, AND IT IS SAVED. Owner ruling 2026-09-23 ("Rebuild
   everything nightly") superseded D33.2-19's scoped build: a one-season build diverges from the
   full build on 139 of 186 numeric columns. So the daily build step passes NO season or week, and
   it writes what it builds (it used to discard it).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.build_features as bf_mod

SEASONS = (2024, 2025)


def _games() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        for week in (1, 2, 3):
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_NE@BUF",
                    "season": season,
                    "week": week,
                    "kickoff_et": pd.Timestamp(f"{season}-10-0{week} 17:00", tz="UTC"),
                    "home_team": "BUF",
                    "away_team": "NE",
                }
            )
    return pd.DataFrame(rows)


def _team_form() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": team,
                "target_season": season,
                "target_week": week,
                "rolling_epa": 0.1,
            }
            for season in SEASONS
            for week in (1, 2, 3)
            for team in ("NE", "BUF")
        ]
    )


def _weather() -> pd.DataFrame:
    games = _games()
    return games[["game_id", "season", "week"]].assign(temperature=60.0)


class _EmptyBuilder:
    """A feature builder that contributes nothing: this test is about the two silver sources."""

    def build_features(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


class _PassThroughWeather:
    def enforce_fence_on_feature_frame(self, weather_df, games_df) -> pd.DataFrame:
        return weather_df


@pytest.fixture
def builder(monkeypatch):
    """A FeatureMatrixBuilder wired to the fixture silver tables, recording team form's input."""
    tables = {
        "games": _games(),
        "team_form_features": _team_form(),
        "weather_features": _weather(),
    }
    monkeypatch.setattr(
        bf_mod,
        "load_dataframe",
        lambda name, *a, **k: tables[name].copy(),
    )
    instance = object.__new__(bf_mod.FeatureMatrixBuilder)
    for attr in (
        "elo_calc",
        "contextual_calc",
        "market_calc",
        "qb_tracker",
        "snap_builder",
        "injury_builder",
    ):
        setattr(instance, attr, _EmptyBuilder())
    instance.weather_calc = _PassThroughWeather()
    seen: dict[str, pd.DataFrame] = {}

    def _record_team_form(team_form_df, games_df):
        seen["team_form"] = team_form_df
        return pd.DataFrame({"game_id": games_df["game_id"]})

    instance._team_form_per_game = _record_team_form
    instance.seen = seen
    return instance


def seasons_outside(frame: pd.DataFrame, column: str, allowed: set[int]) -> set[int]:
    """The seasons in *frame* that a build scoped to *allowed* must not carry."""
    return set(frame[column].astype(int)) - allowed


def test_a_season_only_build_narrows_team_form_and_weather(builder) -> None:
    sources = builder.load_all_feature_sources(target_season=2025)
    assert len(builder.seen["team_form"]) > 0
    assert len(sources["weather"]) > 0
    assert not seasons_outside(builder.seen["team_form"], "target_season", {2025})
    assert not seasons_outside(sources["weather"], "season", {2025})


def test_a_season_and_week_build_narrows_the_week_as_well(builder) -> None:
    sources = builder.load_all_feature_sources(target_season=2025, target_week=2)
    assert set(builder.seen["team_form"]["target_week"]) == {2}
    assert set(sources["weather"]["week"]) == {2}


def test_the_check_flags_an_unnarrowed_source() -> None:
    """Control: the unfiltered silver frame IS reported as carrying a foreign season."""
    assert seasons_outside(_team_form(), "target_season", {2025}) == {2024}
    assert seasons_outside(_weather(), "season", {2025}) == {2024}


def test_the_daily_build_step_is_full_history_and_saves(monkeypatch) -> None:
    """The step passes no season or week, and writes the matrices it built."""
    from pipeline import steps

    calls: dict[str, object] = {}
    matrices = {"wp": pd.DataFrame({"game_id": ["g"]})}

    class _RecordingBuilder:
        def generate_feature_matrices(self, *args, **kwargs):
            calls["generate"] = (args, kwargs)
            return matrices

        def save_feature_matrices(self, feature_matrices, *args, **kwargs):
            calls["save"] = (feature_matrices, args, kwargs)

    monkeypatch.setattr(bf_mod, "FeatureMatrixBuilder", _RecordingBuilder)
    steps.step_build_features()

    args, kwargs = calls["generate"]
    assert args == ()
    assert "target_season" not in kwargs and "target_week" not in kwargs
    saved, save_args, save_kwargs = calls["save"]
    assert saved is matrices
    # No scope, and on a clean night no refused history (33.2 review C1 CR-06): the write is
    # the full table, exactly as before.
    assert save_args == ()
    assert save_kwargs == {
        "history_refused": frozenset(),
        "serve_game_ids": frozenset(),
    }


def test_an_unplayed_game_survives_target_creation_with_blank_labels() -> None:
    """Tomorrow's games have no score; target creation used to DELETE them from gold.

    Played games keep their labels and their dtypes; an unplayed game keeps its row, with every
    label blank, so it can be served (and is refused by every trainer through its provisional Elo).
    """
    instance = object.__new__(bf_mod.FeatureMatrixBuilder)
    frame = pd.DataFrame(
        {
            "game_id": ["played", "unplayed"],
            "home_score": [24.0, None],
            "away_score": [17.0, None],
            "elo_diff": [1.0, 2.0],
        }
    )
    targets = instance.create_target_variables(frame)

    assert list(targets["game_id"]) == ["played", "unplayed"]
    played = targets.set_index("game_id").loc["played"]
    assert played["home_margin"] == 7 and played["total_points"] == 41
    assert played["home_win"] == 1
    unplayed = targets.set_index("game_id").loc["unplayed"]
    for label in ("target_wp", "home_win", "home_margin", "total_points"):
        assert pd.isna(unplayed[label]), label
    assert unplayed["elo_diff"] == 2.0


def test_a_history_build_keeps_its_label_dtypes() -> None:
    """With every game played, target creation is exactly what it was."""
    instance = object.__new__(bf_mod.FeatureMatrixBuilder)
    frame = pd.DataFrame(
        {"game_id": ["a", "b"], "home_score": [24.0, 10.0], "away_score": [17.0, 13.0]}
    )
    targets = instance.create_target_variables(frame)
    assert pd.api.types.is_integer_dtype(targets["home_win"])
    assert list(targets["home_win"]) == [1, 0]


def test_a_refused_historical_game_is_passed_as_history_not_served(monkeypatch) -> None:
    """33.2 review C1 CR-06: excluded games outside tonight's slate are HISTORY refusals."""
    from pipeline import live_skip, steps

    calls: dict[str, object] = {}

    class _RecordingBuilder:
        def generate_feature_matrices(self, *args, **kwargs):
            calls["excluded"] = kwargs["excluded_game_ids"]
            return {"wp": pd.DataFrame({"game_id": ["g"]})}

        def save_feature_matrices(self, feature_matrices, *args, **kwargs):
            calls["save"] = kwargs

    monkeypatch.setattr(bf_mod, "FeatureMatrixBuilder", _RecordingBuilder)
    live_skip.reset_excluded_games()
    live_skip.exclude_games({"2011_W02_BUF@KC", "2026_W03_LA@DEN"})
    try:
        steps.build_and_save_gold(frozenset({"2026_W03_LA@DEN", "2026_W03_LAC@BUF"}))
    finally:
        live_skip.reset_excluded_games()

    assert calls["save"] == {
        "history_refused": frozenset({"2011_W02_BUF@KC"}),
        "serve_game_ids": frozenset({"2026_W03_LAC@BUF"}),
    }
