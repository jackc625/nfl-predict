"""Silver team form is computed for every season from 2002, week 1 included (Plan 33.2-17 Task 1).

D33.2-08 item 2: before team form's first covered season every gold team-form value was a flat
0.0 with no flag, and silver ``team_form_features`` held 2020-2026 only because
``scripts/build_team_form.py`` defaulted to 2020. The pinned play-by-play starts at 2001, so
every season from the corpus's first one can be COMPUTED rather than defaulted -- 2002's week 1
rolling over 2001's games.

WHAT THIS MODULE PINS:

* the corpus build's first season is ``conf.season_partition.CORPUS_FIRST_SEASON`` -- no season
  literal in ``scripts/build_team_form.py``;
* THE PRIOR SEASON IS A BOOTSTRAP, NOT A TARGET: its play-by-play is fetched and TIMED so the
  first target season's week 1 has a window, but no rolling row and no ``team_game_stats`` row
  is persisted for it;
* the bootstrap season is timed by the SAME ingest path silver ``games`` is built by (its pinned
  schedule through ``GameDataIngester.transform_schedule_data`` and ``GameSchema``), and a game
  that fails to time is a refusal, never a silently dropped window;
* the named check that the pin carries the prior season fails LOUDLY when it does not.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pandas as pd
import pytest

import features.team_form as team_form_module
from conf.season_partition import CORPUS_FIRST_SEASON
from features.team_form import ROLLING_COLUMNS, TeamFormCalculator
from scripts import build_team_form

ET = "America/New_York"


def _schedule(season: int, weeks: range) -> pd.DataFrame:
    first = pd.Timestamp(f"{season}-09-08 13:00", tz=ET)
    return pd.DataFrame(
        [
            {
                "game_id": f"{season}_W{week:02d}_BUF@MIA",
                "season": season,
                "week": week,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": first + pd.Timedelta(days=7 * (week - 1)),
            }
            for week in weeks
        ]
    )


def _stats(season: int, weeks: range) -> pd.DataFrame:
    """Synthetic per-team-game stats: every metric equals season + week / 100."""
    metrics = {
        "epa_per_play": 0.0,
        "pass_epa_per_play": 0.0,
        "rush_epa_per_play": 0.0,
        "success_rate": 0.0,
        "pass_success_rate": 0.0,
        "rush_success_rate": 0.0,
        "neutral_pass_rate": 0.0,
        "red_zone_td_rate": 0.0,
        "third_down_conversion_rate": 0.0,
        "team_cpoe": 0.0,
        "avg_drive_start_yardline": 0.0,
        "neutral_pace": 0.0,
    }
    rows = []
    for week in weeks:
        for team in ("BUF", "MIA"):
            for side in ("offense", "defense"):
                value = season + week / 100.0
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_BUF_MIA",
                        "season": season,
                        "week": week,
                        "team": team,
                        "side": side,
                        **dict.fromkeys(metrics, value),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def corpus(monkeypatch):
    """A calculator over a 2001 bootstrap plus 2002, with fetch / stats / save stubbed."""
    schedule = pd.concat(
        [_schedule(2001, range(15, 18)), _schedule(2002, range(1, 4))],
        ignore_index=True,
    )
    fetched: list[list[int]] = []
    saved: dict[str, pd.DataFrame] = {}
    calculator = TeamFormCalculator(max_prior_games=4, schedule_df=schedule)
    monkeypatch.setattr(
        calculator,
        "fetch_pbp_data",
        lambda seasons: fetched.append(list(seasons)) or pd.DataFrame({"s": seasons}),
    )
    monkeypatch.setattr(
        calculator,
        "calculate_team_game_stats",
        # Stats exist only for the seasons actually FETCHED (the stub pbp carries them).
        lambda pbp: pd.concat(
            [
                frame
                for season, frame in (
                    (2001, _stats(2001, range(15, 18))),
                    (2002, _stats(2002, range(1, 4))),
                )
                if season in set(pbp["s"])
            ],
            ignore_index=True,
        ),
    )
    monkeypatch.setattr(
        team_form_module,
        "save_dataframe",
        lambda frame, name, **_kw: saved.__setitem__(name, frame.copy()),
    )
    return calculator, fetched, saved


def _build(calculator, **kwargs) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return calculator.build_team_form_features([2002], **kwargs)


class TestThePriorSeasonIsABootstrapNotATarget:
    def test_the_bootstrap_seasons_play_by_play_is_fetched(self, corpus) -> None:
        calculator, fetched, _saved = corpus
        _build(calculator, bootstrap_seasons=[2001])
        assert fetched == [[2001, 2002]]

    def test_week_one_of_the_first_target_season_rolls_over_the_bootstrap(
        self, corpus
    ) -> None:
        calculator, _fetched, _saved = corpus
        rolling = _build(calculator, bootstrap_seasons=[2001])
        week1 = rolling[
            (rolling["target_season"] == 2002) & (rolling["target_week"] == 1)
        ]
        assert len(week1) == 4, "both teams, both sides"
        assert week1[list(ROLLING_COLUMNS)].notna().any(axis=None)
        # Every value comes from 2001's weeks 15-17 (2001.15 .. 2001.17), never from 2002.
        values = week1["rolling_epa_per_play"]
        assert values.between(2001.15, 2001.17).all()

    def test_without_the_bootstrap_week_one_has_no_window(self, corpus) -> None:
        calculator, _fetched, _saved = corpus
        rolling = _build(calculator)
        week1 = rolling[
            (rolling["target_season"] == 2002) & (rolling["target_week"] == 1)
        ]
        assert len(week1) == 0

    def test_no_rolling_row_is_persisted_for_the_bootstrap_season(self, corpus) -> None:
        calculator, _fetched, saved = corpus
        _build(calculator, bootstrap_seasons=[2001])
        assert set(saved["team_form_features"]["target_season"]) == {2002}

    def test_no_team_game_stats_row_is_persisted_for_the_bootstrap_season(
        self, corpus
    ) -> None:
        calculator, _fetched, saved = corpus
        _build(calculator, bootstrap_seasons=[2001])
        assert set(saved["team_game_stats"]["season"]) == {2002}


class TestTheCorpusStartsAtTheRuleModulesFirstSeason:
    def test_the_all_seasons_default_is_the_corpus_first_season(self) -> None:
        args = build_team_form.build_parser().parse_args(["--all-seasons"])
        assert args.start_season == CORPUS_FIRST_SEASON

    def test_the_script_carries_no_season_literal(self) -> None:
        tree = ast.parse(Path(build_team_form.__file__).read_text(encoding="utf-8"))
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and 1990 <= node.value <= 2100
        ]
        assert literals == []


class TestThePriorSeasonMustBePinned:
    def test_a_pin_without_the_prior_season_is_refused_by_name(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            build_team_form.upstream_pin,
            "pinned_seasons",
            lambda dataset, _m: [2002, 2003] if dataset == "pbp" else [2001, 2002],
        )
        monkeypatch.setattr(build_team_form.upstream_pin, "load_manifest", lambda: {})
        with pytest.raises(build_team_form.PriorSeasonNotPinnedError, match="2001"):
            build_team_form.require_prior_season_pinned(2002)

    def test_the_schedule_is_required_too(self, monkeypatch) -> None:
        monkeypatch.setattr(
            build_team_form.upstream_pin,
            "pinned_seasons",
            lambda dataset, _m: [2001, 2002] if dataset == "pbp" else [2002],
        )
        monkeypatch.setattr(build_team_form.upstream_pin, "load_manifest", lambda: {})
        with pytest.raises(
            build_team_form.PriorSeasonNotPinnedError, match="schedules"
        ):
            build_team_form.require_prior_season_pinned(2002)

    def test_a_pinned_prior_season_passes(self, monkeypatch) -> None:
        monkeypatch.setattr(
            build_team_form.upstream_pin, "pinned_seasons", lambda _d, _m: [2001, 2002]
        )
        monkeypatch.setattr(build_team_form.upstream_pin, "load_manifest", lambda: {})
        assert build_team_form.require_prior_season_pinned(2002) == 2001

    def test_the_refusal_escapes_the_scripts_error_handler(self) -> None:
        # main() catches (ValueError, KeyError, TypeError, FileNotFoundError, OSError) and
        # exits 1 quietly; the refusal must not be one of them.
        assert issubclass(build_team_form.PriorSeasonNotPinnedError, RuntimeError)
        assert not issubclass(
            build_team_form.PriorSeasonNotPinnedError,
            (ValueError, KeyError, TypeError, FileNotFoundError, OSError),
        )


class TestTheBootstrapSeasonIsTimedByTheIngestPath:
    def test_a_game_the_transform_drops_is_a_refusal(self, monkeypatch) -> None:
        schedule = pd.DataFrame({"game_id": ["a", "b"]})
        monkeypatch.setattr(
            build_team_form.upstream_pin, "load_schedules", lambda _s: schedule
        )
        monkeypatch.setattr(
            build_team_form.GameDataIngester,
            "transform_schedule_data",
            lambda _self, frame: frame.iloc[:1],
        )
        with pytest.raises(build_team_form.BootstrapTimingError, match="1 of 2"):
            build_team_form.bootstrap_timing_games(2001)

    def test_the_real_pinned_2001_schedule_times_every_game(self) -> None:
        # Read-only against the committed pin: every 2001 game gets a tz-aware kickoff.
        try:
            games = build_team_form.bootstrap_timing_games(CORPUS_FIRST_SEASON - 1)
        except (
            build_team_form.upstream_pin.UpstreamPinError
        ) as error:  # pragma: no cover
            pytest.skip(f"the upstream pin is not present on this checkout: {error}")
        assert len(games) > 250
        assert str(games["kickoff_et"].dt.tz) == ET
        assert set(games["season"]) == {CORPUS_FIRST_SEASON - 1}


class TestTheProvenanceTimesTheBootstrapWindow:
    """Plan 33.2-18 (found by rung 8's preview): the gold provenance times a window exactly as
    the silver corpus build did.

    The corpus build times the bootstrap season's team-games from its pinned schedule, so the
    first target season's week 1 carries a rolling row drawn from the season before. The
    information-time provenance used to time windows against silver ``games`` alone, which
    does not carry that season -- so it declared those rows ``no_information`` beside real
    values, and the gate (rightly) refused the build. It now reads the same corpus timing.
    """

    @staticmethod
    def _calculator(monkeypatch, *, pinned: bool) -> tuple[TeamFormCalculator, list]:
        calls: list[int] = []
        form = pd.DataFrame(
            {
                "team": ["BUF", "MIA"],
                "side": ["offense", "offense"],
                "target_season": [2002, 2002],
                "target_week": [1, 1],
            }
        )
        calculator = TeamFormCalculator(
            max_prior_games=4, schedule_df=_schedule(2002, range(1, 4))
        )
        calculator._form_df = form
        monkeypatch.setattr(
            team_form_module.upstream_pin,
            "pinned_seasons",
            lambda dataset, _m: [2001, 2002] if pinned else [2002],
        )
        monkeypatch.setattr(team_form_module.upstream_pin, "load_manifest", lambda: {})

        def _bootstrap(season: int) -> pd.DataFrame:
            calls.append(season)
            return _schedule(season, range(15, 18))

        monkeypatch.setattr(build_team_form, "bootstrap_timing_games", _bootstrap)
        return calculator, calls

    def test_week_one_of_the_first_season_is_timed_from_the_bootstrap(
        self, monkeypatch
    ) -> None:
        calculator, calls = self._calculator(monkeypatch, pinned=True)
        provenance = calculator.information_times(_schedule(2002, range(1, 2)))
        row = provenance.iloc[0]
        assert row["basis"] == "per_row"
        # The latest admitted game is 2001 week 17: its kickoff plus the declared duration.
        kickoff = pd.Timestamp("2001-09-08 13:00", tz=ET) + pd.Timedelta(days=7 * 16)
        expected = kickoff.tz_convert("UTC") + team_form_module.DECLARED_GAME_DURATION
        assert pd.Timestamp(row["information_time"]) == expected
        assert calls == [2001]

    def test_an_unpinned_bootstrap_leaves_the_row_untimed(self, monkeypatch) -> None:
        """CONTROL: no second timing source is invented; the gate then refuses loudly."""
        calculator, calls = self._calculator(monkeypatch, pinned=False)
        provenance = calculator.information_times(_schedule(2002, range(1, 2)))
        assert provenance.iloc[0]["basis"] == "no_information"
        assert calls == []

    def test_a_game_after_the_first_season_does_not_read_the_bootstrap(
        self, monkeypatch
    ) -> None:
        """CONTROL: only the first target season's windows can reach the season before it."""
        calculator, calls = self._calculator(monkeypatch, pinned=True)
        calculator._form_df = pd.DataFrame(
            {
                "team": ["BUF", "MIA", "BUF"],
                "side": ["offense", "offense", "offense"],
                "target_season": [2002, 2002, 2003],
                "target_week": [1, 1, 1],
            }
        )
        calculator.information_times(_schedule(2003, range(1, 2)))
        assert calls == []
