"""``--through-season``: the p332_ ladder-rung full rebuild that stops at a season.

OWNER RULING 2026-09-21 (Plan 33.2-08 Task 4 checkpoint, Option A): every rung of the
Phase-33.2 ``p332_`` gold ladder (Plans 33.2-08 .. 33.2-19) rebuilds gold over 2002-2025
ONLY, so the unplayed 2026 season cannot enter gold mid-ladder as a second cause
(D33.2-20). Plan 33.2-20 is the one build that adds 2026.

What the ruling requires the option to be, and what each class below proves:

(a) it carries EVERY season up to the bound -- nothing inside the range is lost;
(b) it carries NOTHING after the bound;
(c) it is otherwise the full rebuild: the same builder call, the same REPLACE write
    path, and the information-time gate still runs over every game that remains;
and it is never the default, and it refuses to combine with a scoped build.

Hermetic: no real data lake is read or written. Calculators are replaced by recorders,
and the gate is intercepted after it is reached rather than executed on real sources.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.build_features as bf
from scripts.build_features import (
    FeatureMatrixBuilder,
    parse_args,
    scope_games_through_season,
)

LADDER_BOUND = 2025
FIRST_SEASON = 2002
SILVER_SEASONS = list(range(FIRST_SEASON, 2027))  # 2002..2026, as silver holds today


def _games(seasons: list[int] = SILVER_SEASONS, per_season: int = 2) -> pd.DataFrame:
    """A minimal silver-games-shaped frame: two games per season, ET kickoffs."""
    rows = []
    for season in seasons:
        for week in range(1, per_season + 1):
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_BUF@KC",
                    "season": season,
                    "week": week,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "kickoff_et": pd.Timestamp(
                        f"{season}-09-{7 * week:02d} 13:00", tz="America/New_York"
                    ),
                }
            )
    return pd.DataFrame(rows)


class TestTheBoundItself:
    def test_every_season_through_the_bound_is_kept(self) -> None:
        scoped = scope_games_through_season(_games(), LADDER_BOUND)
        assert sorted(scoped["season"].unique()) == list(
            range(FIRST_SEASON, LADDER_BOUND + 1)
        )

    def test_every_game_through_the_bound_is_kept(self) -> None:
        games = _games()
        scoped = scope_games_through_season(games, LADDER_BOUND)
        expected = games.loc[games["season"] <= LADDER_BOUND, "game_id"]
        assert list(scoped["game_id"]) == list(expected)

    def test_nothing_after_the_bound_survives(self) -> None:
        scoped = scope_games_through_season(_games(), LADDER_BOUND)
        assert 2026 not in set(scoped["season"])

    def test_no_bound_is_the_unbounded_rebuild_unchanged(self) -> None:
        games = _games()
        assert scope_games_through_season(games, None) is games

    @pytest.mark.parametrize(
        ("target_season", "target_week"), [(2024, None), (None, 3), (2024, 3)]
    )
    def test_it_refuses_to_bound_a_scoped_build(
        self, target_season: int | None, target_week: int | None
    ) -> None:
        with pytest.raises(ValueError, match="bounds the FULL rebuild"):
            scope_games_through_season(
                _games(),
                LADDER_BOUND,
                target_season=target_season,
                target_week=target_week,
            )

    def test_a_bound_that_leaves_no_game_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leaves no game"):
            scope_games_through_season(_games(), FIRST_SEASON - 1)


class TestTheCommandLine:
    def test_it_is_not_the_default(self) -> None:
        assert parse_args([]).through_season is None
        assert parse_args(["--all-seasons"]).through_season is None

    def test_it_parses_as_a_full_rebuild(self) -> None:
        args = parse_args(["--through-season", str(LADDER_BOUND)])
        assert args.through_season == LADDER_BOUND
        assert args.season is None
        assert args.week is None

    def test_it_may_name_the_full_rebuild_explicitly(self) -> None:
        args = parse_args(["--all-seasons", "--through-season", str(LADDER_BOUND)])
        assert args.all_seasons is True
        assert args.through_season == LADDER_BOUND

    @pytest.mark.parametrize(
        "scope",
        [["--season", "2024"], ["--week", "3"], ["--season", "2024", "--week", "3"]],
    )
    def test_it_refuses_a_scoped_build(self, scope: list[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            parse_args(["--through-season", str(LADDER_BOUND), *scope])
        assert excinfo.value.code == 2

    def test_the_help_documents_it(self) -> None:
        text = " ".join(bf.build_parser().format_help().split())  # argparse wraps
        assert "--through-season" in text
        assert "REPLACES gold exactly like --all-seasons" in text


class _RecordingBuilder:
    """Stands in for FeatureMatrixBuilder in ``main``: records, never builds."""

    generate_calls: list[dict] = []
    save_calls: list[tuple] = []

    def generate_feature_matrices(self, **kwargs):
        type(self).generate_calls.append(kwargs)
        return {"wp": pd.DataFrame({"game_id": ["g"], "target_wp": [1.0]})}

    def save_feature_matrices(self, matrices, target_season=None, target_week=None):
        type(self).save_calls.append((target_season, target_week))


class TestMainIsTheFullRebuildOtherwise:
    @pytest.fixture
    def recorder(self, monkeypatch):
        _RecordingBuilder.generate_calls = []
        _RecordingBuilder.save_calls = []
        monkeypatch.setattr(bf, "FeatureMatrixBuilder", _RecordingBuilder)
        return _RecordingBuilder

    def test_the_only_difference_from_all_seasons_is_the_bound(self, recorder) -> None:
        bf.main(["--all-seasons"])
        bf.main(["--through-season", str(LADDER_BOUND)])
        unbounded, bounded = recorder.generate_calls
        assert unbounded.pop("through_season") is None
        assert bounded.pop("through_season") == LADDER_BOUND
        assert bounded == unbounded

    def test_it_takes_the_same_replace_write_path(self, recorder) -> None:
        bf.main(["--all-seasons"])
        bf.main(["--through-season", str(LADDER_BOUND)])
        # (None, None) is the scope save_feature_matrices maps to replace_mode=True.
        assert recorder.save_calls == [(None, None), (None, None)]


def _recording_calculator(seen: list[set[int]]):
    class _Calculator:
        def build_features(self, games_df, as_of_datetime, **kwargs):
            seen.append(set(games_df["season"]))
            return pd.DataFrame()

    return _Calculator()


@pytest.fixture
def hermetic_builder(monkeypatch):
    """A real FeatureMatrixBuilder whose silver reads and calculators are sandboxed."""
    games = _games()

    def fake_load(table_name, layer="silver", *args, **kwargs):
        if table_name == "games":
            return games.copy()
        raise FileNotFoundError(f"{layer}/{table_name} is not part of this sandbox")

    monkeypatch.setattr(bf, "load_dataframe", fake_load)
    builder = FeatureMatrixBuilder()
    seen: list[set[int]] = []
    for name in (
        "elo_calc",
        "contextual_calc",
        "market_calc",
        "qb_tracker",
        "snap_builder",
        "injury_builder",
    ):
        setattr(builder, name, _recording_calculator(seen))
    return builder, seen


class TestTheLoaderAndTheGate:
    def test_every_builder_sees_only_seasons_through_the_bound(
        self, hermetic_builder
    ) -> None:
        builder, seen = hermetic_builder
        sources = builder.load_all_feature_sources(through_season=LADDER_BOUND)
        expected = set(range(FIRST_SEASON, LADDER_BOUND + 1))
        assert set(sources["games"]["season"]) == expected
        assert seen, "no calculator was reached -- the sandbox is vacuous"
        assert all(s == expected for s in seen)

    def test_without_the_bound_the_same_loader_carries_2026(
        self, hermetic_builder
    ) -> None:
        """Non-vacuity: the sandbox really holds 2026, so its absence above is the bound."""
        builder, _ = hermetic_builder
        sources = builder.load_all_feature_sources()
        assert 2026 in set(sources["games"]["season"])

    def test_the_information_time_gate_still_runs_over_every_kept_game(
        self, hermetic_builder, monkeypatch
    ) -> None:
        builder, _ = hermetic_builder
        reached: dict[str, object] = {}

        class _GateReached(Exception):
            pass

        def intercept(self, feature_sources, lock_frame, target_season, target_week):
            reached["games"] = set(feature_sources["games"]["game_id"])
            reached["locks"] = set(lock_frame.index)
            raise _GateReached

        monkeypatch.setattr(FeatureMatrixBuilder, "_check_information_times", intercept)
        with pytest.raises(_GateReached):
            builder.generate_feature_matrices(through_season=LADDER_BOUND)

        games = _games()
        kept = set(games.loc[games["season"] <= LADDER_BOUND, "game_id"])
        assert reached["games"] == kept
        assert reached["locks"] == kept
