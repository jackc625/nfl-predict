"""WEATHER-NULL-LIST.md is tied to the live table, and the forecast-less inputs stay out.

Plan 33.2-12 Task 2 (SPEC R6), p332_ rung 4. On the ``test_signal_lift_readout_md.py``
pattern: the document exists at the repo root, is ASCII, carries its required sections, and
-- the deeper anti-rot guard -- the count, the per-reason split and the game list it records
EQUAL what the live silver ``weather`` table produces. It asserts the RULING (NULL plus a flag,
never a stand-in) and the COUNT RELATIONSHIP, never a pinned point estimate divorced from the
data.

It also guards the registry route by which the two inputs no forecast can supply --
``precip_mm`` and ``raw_precip_mm`` -- leave the model-input candidate set: a planted addition
is flagged by the group predicate and dropped by the one drop mechanism.

``test_forecastless_columns_absent_from_gold`` is AUTHORED at Task 2 and EVALUATED at Task 3:
see its own docstring.

SINCE p332_ EXTRA STEP 4b (Plan 33.2-14, owner ruling 2026-09-21) a retractable roof's
open/closed state is not known at the lock, so every game at a retractable stadium carries the
day-before forecast and only a FIXED-roof dome is indoor. ``2019_W18_BUF@HOU``, whose 12 UTC
run is a confirmed archive gap, is therefore a LISTED absence under "US venue, no resolved
bulletin" rather than a closed-roof game named in the prose, and the US list is tied to the
live table exactly as the list abroad is.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import _GROUP_PREDICATE, GROUPS, group_columns

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "records" / "WEATHER-NULL-LIST.md"
SILVER = REPO_ROOT / "data" / "silver"
GOLD = REPO_ROOT / "data" / "gold"
VENUES = REPO_ROOT / "data" / "venues.json"
GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")
FORECASTLESS_COLUMNS = ("precip_mm", "raw_precip_mm")

REQUIRED_SECTIONS = (
    "## Current count",
    "## By reason",
    "## By season",
    "## Games outside the United States",
    "## US games with no resolved bulletin",
)


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _live_null_plus_flag() -> pd.DataFrame:
    """2002-2025 silver rows where weather applies but no forecast exists, with a reason."""
    path = SILVER / "weather.parquet"
    if not path.exists():
        pytest.skip("silver weather is not built on this checkout")
    weather = pd.read_parquet(path)
    weather = weather[weather["game_id"].str[:4].astype(int).between(2002, 2025)]
    absent = weather[
        ~weather["weather_coverage"].astype(bool) & weather["is_outdoor"].astype(bool)
    ].copy()
    games = pd.read_parquet(SILVER / "games.parquet").set_index("game_id")
    countries = {
        v["stadium_id"]: v["country"]
        for v in json.loads(VENUES.read_text(encoding="utf-8"))["venues"]
    }
    absent["abroad"] = absent["game_id"].map(
        lambda g: countries[games.loc[g, "stadium_id"]] != "USA"
    )
    return absent


class TestTheDocument:
    def test_it_exists_at_the_repo_root_and_is_ascii(self):
        assert DOC.is_file()
        _doc().encode("ascii")

    @pytest.mark.parametrize("section", REQUIRED_SECTIONS)
    def test_every_required_section_is_present(self, section):
        assert section in _doc()

    def test_it_states_the_ruling_and_the_reason_abroad(self):
        text = _doc()
        assert "nothing observed afterwards is put in its place" in text
        assert "United States, Puerto Rico and the US Virgin Islands" in text
        assert "No stand-in station is ever used" in text

    def test_it_states_the_retractable_roof_ruling(self):
        """Step 4b: the open/closed state is post-lock information, never an input."""
        text = _doc()
        assert "retractable roof" in text
        assert "not known at the lock" in text

    def test_the_over_claim_word_never_appears(self):
        assert "proven" not in _doc().lower()


class TestTheDocumentMatchesTheLiveTable:
    def test_the_recorded_count_is_the_live_count(self):
        recorded = re.search(r"NULL-plus-flag games, 2002-2025: (\d+)", _doc())
        assert recorded is not None
        assert int(recorded.group(1)) == len(_live_null_plus_flag())

    def test_the_per_reason_split_is_the_live_split(self):
        live = _live_null_plus_flag()
        text = _doc()
        abroad = re.search(r"\| Venue outside the United States \| (\d+) \|", text)
        us = re.search(r"\| US venue, no resolved bulletin \| (\d+) \|", text)
        assert abroad is not None and us is not None
        assert int(abroad.group(1)) == int(live["abroad"].sum())
        assert int(us.group(1)) == int((~live["abroad"]).sum())

    def test_the_listed_games_abroad_are_exactly_the_live_ones(self):
        live = _live_null_plus_flag()
        section = (
            _doc().split("## Games outside the United States", 1)[1].split("## ", 1)[0]
        )
        listed = set(re.findall(r"^\| (\d{4}_W\d{2}_\w+@\w+) \|", section, flags=re.M))
        assert listed == set(live.loc[live["abroad"], "game_id"])
        assert len(listed) > 0, "non-vacuity: the games abroad are listed"

    def test_the_listed_us_games_are_exactly_the_live_ones(self):
        """The US absences are tied to the table exactly as the games abroad are."""
        live = _live_null_plus_flag()
        section = (
            _doc()
            .split("## US games with no resolved bulletin", 1)[1]
            .split("## ", 1)[0]
        )
        listed = set(re.findall(r"^\| (\d{4}_W\d{2}_\w+@\w+) \|", section, flags=re.M))
        assert listed == set(live.loc[~live["abroad"], "game_id"])

    def test_no_fixed_roof_dome_is_listed(self):
        """Only a FIXED-roof dome is indoor since step 4b, and none is on the list."""
        weather = pd.read_parquet(SILVER / "weather.parquet").set_index("game_id")
        listed = set(re.findall(r"(\d{4}_W\d{2}_\w+@\w+)", _doc()))
        indoor = set(weather.index[~weather["is_outdoor"].astype(bool)])
        assert not listed & indoor, sorted(listed & indoor)


class TestTheForecastlessInputsLeaveThroughTheRegistry:
    def test_the_group_is_registered_and_is_not_a_screened_group(self):
        assert "weather_unsupplied" in _GROUP_PREDICATE
        assert "weather_unsupplied" not in GROUPS

    def test_a_planted_forecastless_column_is_flagged_and_its_lookalikes_are_not(self):
        planted = pd.DataFrame(
            columns=[
                "game_id",
                *FORECASTLESS_COLUMNS,
                "precip_prob",
                "raw_precip_prob",
                "precip_heavy",
                "precip_impact_score",
            ]
        )
        assert group_columns(planted, "weather_unsupplied") == sorted(
            FORECASTLESS_COLUMNS
        )

    def test_the_one_drop_mechanism_removes_a_planted_addition(self):
        from scripts.build_features import GOLD_DROPPED_GROUPS, FeatureMatrixBuilder

        assert "weather_unsupplied" in GOLD_DROPPED_GROUPS
        planted = pd.DataFrame(
            {"game_id": ["G"], "precip_mm": [None], "precip_prob": [0.4]}
        )
        out = FeatureMatrixBuilder()._enforce_groups_dropped(
            planted, GOLD_DROPPED_GROUPS
        )
        assert "precip_mm" not in out.columns and "precip_prob" in out.columns


@pytest.mark.parametrize("matrix", GOLD_MATRICES)
def test_forecastless_columns_absent_from_gold(matrix):
    """Neither forecast-less input is in any gold matrix -- EVALUATED AT TASK 3.

    Authored at Task 2 and deselected there by name, because the rung-4 gold rebuild that
    drops the two columns is Task 3 (measured 2026-09-16 and again at Task 2: both were
    still present in all three matrices). Running it at Task 2 would leave the gate red
    until Task 3, and the cheapest route to green would be weakening the assertion SPEC R6
    requires. The rung's predicted width change is exactly -2 per matrix -- these two
    columns and nothing else -- so a matrix of 195 / 196 / 195 columns becomes 193 / 194 /
    193.
    """
    path = GOLD / f"{matrix}.parquet"
    if not path.exists():
        pytest.skip("gold is not built on this checkout")
    columns = set(pd.read_parquet(path).columns)
    assert not columns & set(FORECASTLESS_COLUMNS), sorted(
        columns & set(FORECASTLESS_COLUMNS)
    )
