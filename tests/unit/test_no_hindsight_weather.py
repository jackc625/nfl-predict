"""STANDING PROHIBITION: no observed (hindsight) weather stands in for a missing forecast.

Plan 33.2-12 Task 1 (SPEC R6 prohibition, T-33.2-12-01 / T-33.2-12-08), p332_ rung 4.

TWO INSTRUMENTS, AND WHICH ONE IS AUTHORITATIVE
-----------------------------------------------
A hindsight fallback RETURNS A VALUE, so it leaves a green suite behind it -- the argument
``scripts/backfill_historical_weather.py``'s quarantine docstring makes. Two instruments
answer it:

* THE CHEAP GUARD, a SOURCE SCAN of the production weather path. It reads the PARSED TREE
  and matches by NODE SHAPE, never raw text. Its subject is ``ast.Name`` ids,
  ``ast.Attribute`` attrs, ``ast.arg`` names, call keyword names, ``def``/``class`` names,
  and string constants ONLY where the constant is a ``Subscript`` slice or an
  ``Assign``/``AnnAssign`` value -- so a column or table name actually being READ is caught
  while narration is not. Docstrings and ``#`` comments are outside the subject by
  construction: a comment is absent from the AST, and a docstring is an ``ast.Expr``
  statement, neither a subscript slice nor an assignment value. That match mode is pinned
  here, not left to intent, because this same task REQUIRES prose in ``features/weather.py``
  naming the ERA5 observation replacement and the NWS TYP-line fallback: a raw-text gate
  would force a choice between that explanation and a green run.
* THE AUTHORITATIVE CASE, a BEHAVIOURAL one. A node-shape scan cannot see through a helper:
  ``_best_available_temp()`` reading ERA5 internally is invisible to it. So a helper that
  returns an ERA5 observation for a forecast-less game is INJECTED into the production
  selection, and the emitted row must still be NULL-plus-flag. A green scan is not a proof of
  behaviour; this case is.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import features.weather as weather_module
from features.weather import WeatherFeaturesCalculator

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The production weather path: the builder, the history regeneration and its entry point.
SCANNED_FILES: tuple[Path, ...] = (
    REPO_ROOT / "features" / "weather.py",
    REPO_ROOT / "scripts" / "weather_from_mos.py",
    REPO_ROOT / "scripts" / "build_weather.py",
)

#: Lower-cased fragments that mark an OBSERVED value being read where a forecast belongs.
HINDSIGHT_TOKENS: tuple[str, ...] = (
    "era5",
    "reanalysis",
    "hindsight",
    "observed_temp",
    "observed_wind",
    "observed_precip",
    "actual_temp",
    "actual_wind",
    "actual_precip",
    "archive_temp",
    "archive_wind",
    "archive_precip",
)
HINDSIGHT_TOKEN_COUNT = 12


def _subject_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string in the scan's subject, with its line: names, and read-position constants."""
    found: list[tuple[int, str]] = []

    def add(node: ast.AST, text: str) -> None:
        found.append((getattr(node, "lineno", 0), text))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            add(node, node.id)
        elif isinstance(node, ast.Attribute):
            add(node, node.attr)
        elif isinstance(node, ast.arg) or (
            isinstance(node, ast.keyword) and node.arg is not None
        ):
            add(node, node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            add(node, node.name)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                add(node, node.slice.value)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                add(node, value.value)
    return found


def hindsight_hits(source: str) -> list[tuple[int, str]]:
    """``(line, text)`` for every subject string carrying a hindsight token."""
    return [
        (line, text)
        for line, text in _subject_strings(ast.parse(source))
        if any(token in text.lower() for token in HINDSIGHT_TOKENS)
    ]


class TestTheNodeShapeScan:
    def test_the_scan_is_not_vacuous(self):
        assert len(HINDSIGHT_TOKENS) == HINDSIGHT_TOKEN_COUNT
        assert SCANNED_FILES and all(path.is_file() for path in SCANNED_FILES)

    def test_no_production_weather_path_reads_an_observation_for_a_forecast(self):
        hits = {
            path.name: hindsight_hits(path.read_text(encoding="utf-8"))
            for path in SCANNED_FILES
        }
        assert all(not found for found in hits.values()), hits

    def test_a_planted_era5_fallback_written_as_a_subscript_is_flagged(self):
        planted = (
            "def weather_for(row):\n"
            "    if row.get('temp_f') is None:\n"
            "        return row['era5_temp_c'] * 9 / 5 + 32\n"
            "    return row['temp_f']\n"
        )
        assert hindsight_hits(planted) == [(3, "era5_temp_c")]

    def test_narration_of_the_deleted_fallback_is_not_flagged(self):
        """The match mode, pinned: docstrings and comments are outside the subject.

        This is the shape ``features/weather.py`` takes after this plan -- a module
        docstring, a method docstring and a ``#`` comment all describing the deleted ERA5
        observation fallback in as many words.
        """
        narrated = (
            '"""The ERA5 reanalysis observation used to stand in for a missing forecast.\n'
            "\n"
            "That hindsight fallback is deleted; observed_temp is never read.\n"
            '"""\n'
            "\n"
            "\n"
            "def weather_for(row):\n"
            '    """No ERA5 observation, no reanalysis, no actual_temp: a forecast or NULL."""\n'
            "    # The ERA5 archive_temp fallback that lived here was removed (hindsight).\n"
            "    return row['temp_f']\n"
        )
        assert hindsight_hits(narrated) == []


# ---------------------------------------------------------------------------
# THE AUTHORITATIVE CASE.
# ---------------------------------------------------------------------------

KICKOFF = pd.Timestamp("2016-09-11 17:00", tz="UTC")
BULLETIN = pd.Timestamp("2016-09-10 12:00", tz="UTC")
BUILD = datetime(2016, 9, 20, tzinfo=UTC)
GAME = "2016_W01_ABS@ABS"

ERA5_OBSERVATION = {
    "game_id": GAME,
    "forecast_time": BULLETIN,
    "forecast_issue_time": None,
    "weather_source": "archive",
    "is_outdoor": True,
    "weather_coverage": True,
    "temp_f": 71.0,
    "wind_mph": 11.0,
    "humidity_pct": 40.0,
    "precip_prob": None,
    "precip_mm": 0.0,
}
ABSENCE = {
    "game_id": GAME,
    "forecast_time": BULLETIN,
    "forecast_issue_time": None,
    "weather_source": "historical_forecast",
    "is_outdoor": True,
    "weather_coverage": False,
}


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [{"game_id": GAME, "season": 2016, "week": 1, "kickoff_et": KICKOFF}]
    )


def _full_row(weather: pd.DataFrame) -> pd.Series:
    with pytest.warns(DeprecationWarning):
        built = WeatherFeaturesCalculator().build_weather_features(
            _games(), weather_df=weather, build_instant=BUILD
        )
    return built.set_index("game_id").loc[GAME]


def _assert_null_plus_flag(row: pd.Series) -> None:
    assert row["weather_coverage"] == 0.0
    assert row["weather_affects_game"] == 1.0
    for column in ("temp_f", "raw_temp_f", "wind_mph", "raw_wind_mph"):
        assert math.isnan(row[column]), f"{column} carries {row[column]!r}"


class TestAnInjectedObservationNeverReachesTheFrame:
    def test_a_helper_returning_an_era5_observation_is_refused(self, monkeypatch):
        def _best_available_weather(game_weather, kickoff, build_instant, *, game_id):
            return dict(ERA5_OBSERVATION), None

        monkeypatch.setattr(
            weather_module, "select_weather_row", _best_available_weather
        )
        _assert_null_plus_flag(_full_row(pd.DataFrame([ABSENCE])))

    def test_the_compressed_builder_refuses_it_too(self, monkeypatch):
        def _best_available_weather(game_weather, kickoff, build_instant, *, game_id):
            return dict(ERA5_OBSERVATION), None

        monkeypatch.setattr(
            weather_module, "select_weather_row", _best_available_weather
        )
        built = WeatherFeaturesCalculator().build_features(
            _games(), BUILD, weather_df=pd.DataFrame([ABSENCE])
        )
        row = built.set_index("game_id").loc[GAME]
        assert math.isnan(row["wind_mph"]) and math.isnan(row["weather_severity_score"])

    def test_an_observation_with_a_claimed_forecast_time_is_still_refused(
        self, monkeypatch
    ):
        """A helper that labels the observation with a time does not launder it."""

        def _best_available_weather(game_weather, kickoff, build_instant, *, game_id):
            return dict(ERA5_OBSERVATION), BULLETIN

        monkeypatch.setattr(
            weather_module, "select_weather_row", _best_available_weather
        )
        _assert_null_plus_flag(_full_row(pd.DataFrame([ABSENCE])))

    def test_an_observation_blended_into_silver_is_never_selected(self):
        blended = pd.DataFrame([ABSENCE, ERA5_OBSERVATION])
        _assert_null_plus_flag(_full_row(blended))

    def test_an_observation_alone_is_never_selected(self):
        _assert_null_plus_flag(_full_row(pd.DataFrame([ERA5_OBSERVATION])))
