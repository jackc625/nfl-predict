"""A predicted game with no market line and no bet still reaches the site with the model's numbers.

The owner's requirement (2026-09-23): every game and every model prediction is visible, whether or
not it is a good bet. The blend's WP and ATS weights are 0.00, so the published numbers are the
market's; the model's own WP, margin and total must still be shown, and a game with no pre-lock
line must say so rather than show a zero.

The row is written in the published CSV header (``PREDICTION_OUTPUT_COLUMNS``), loaded by the cache
loader the population step calls, read back through ``DataService`` and rendered by the game card.
READ-ONLY: everything lives under ``tmp_path`` and an in-memory database.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader

from api.cache import CACHE_SCHEMA, _load_current_week_predictions
from api.services import DataService
from scripts.generate_current_week_predictions import PREDICTION_OUTPUT_COLUMNS

TEMPLATES = Path(__file__).resolve().parents[2] / "web" / "templates"
GAME_ID = "2026_W01_DEN@KC"


def test_a_no_market_non_bet_game_carries_the_models_own_three_numbers(
    tmp_path: Path,
) -> None:
    # The CSV row of a game the models scored, with no owned line (every market cell empty) and
    # therefore no edge, no band and no bet.
    row = dict.fromkeys(PREDICTION_OUTPUT_COLUMNS)
    row.update(
        game_id=GAME_ID,
        season=2026,
        week=1,
        home_team="KC",
        away_team="DEN",
        wp_prob=0.64,
        ats_prediction=4.2,
        ou_prediction=45.1,
    )
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pd.DataFrame([row], columns=PREDICTION_OUTPUT_COLUMNS).to_csv(
        predictions_dir / "predictions_2026_week1.csv", index=False
    )
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_score": [None],
            "away_score": [None],
            "kickoff_et": [pd.Timestamp("2026-09-13 16:25", tz="America/New_York")],
        }
    ).to_parquet(silver_dir / "games.parquet")

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    assert _load_current_week_predictions(conn, predictions_dir, silver_dir) == 1

    game = DataService(conn).get_game_detail(GAME_ID)
    assert game is not None
    assert (game["wp_prob"], game["ats_prediction"], game["ou_prediction"]) == (
        0.64,
        4.2,
        45.1,
    )
    assert game["status"] == "scheduled"
    for absent in (
        "market_spread",
        "market_total",
        "market_wp",
        "wp_confidence",
        "blended_wp",
    ):
        assert game[absent] is None, (
            f"{absent} is {game[absent]!r}, not an absent value"
        )

    card = (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=game)
    )
    for shown in ("KC 64.0%", "KC by 4.2", "45.1", "No line"):
        assert shown in card, f"the game card does not show {shown!r}"
    assert "nan" not in card.lower()


def test_a_game_with_no_edge_carries_no_band_even_when_an_older_csv_wrote_low(
    tmp_path: Path,
) -> None:
    """33.2 review C2 CR-04 = B WR-11: an absent edge has NO band in the cache or the exports.

    A CSV written before the fix carries ``low`` for a game with no line. The loader re-derives
    every band from its stored edge, so the row is stored with NULL bands -- which is what the
    CSV/JSON exports read -- and the card renders no badge.
    """
    row = dict.fromkeys(PREDICTION_OUTPUT_COLUMNS)
    row.update(
        game_id=GAME_ID,
        season=2026,
        week=1,
        home_team="KC",
        away_team="DEN",
        wp_prob=0.64,
        ats_prediction=4.2,
        ou_prediction=45.1,
        wp_confidence="low",
        ats_confidence="low",
        ou_confidence="low",
    )
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pd.DataFrame([row], columns=PREDICTION_OUTPUT_COLUMNS).to_csv(
        predictions_dir / "predictions_2026_week1.csv", index=False
    )
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_score": [None],
            "away_score": [None],
            "kickoff_et": [pd.Timestamp("2026-09-13 16:25", tz="America/New_York")],
        }
    ).to_parquet(silver_dir / "games.parquet")

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    assert _load_current_week_predictions(conn, predictions_dir, silver_dir) == 1

    exported = DataService(conn).export_predictions(season=2026, week=1)
    assert len(exported) == 1
    for band in ("wp_confidence", "ats_confidence", "ou_confidence"):
        assert exported[0][band] is None, (
            f"{band} is {exported[0][band]!r} for a game with no edge; the export publishes a "
            "band for an edge that does not exist"
        )

    card = (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=exported[0])
    )
    assert "Low" not in card


def test_the_wp_edge_is_measured_against_the_market_wp_shown_beside_it(
    tmp_path: Path,
) -> None:
    """33.2 review C2 WR-05: the WP "Edge" is the model's WP minus the market WP on the page.

    The market WP shown is the pre-lock spread through the blend's converter; the edge beside it
    used to be taken against the de-vigged moneyline, so "Model 61.0% / Market 58.0% / Edge
    -1.2%" was a normal display. A CSV written before the fix carries that old edge; the cache
    re-takes it against the row's own ``market_wp``.
    """
    row = dict.fromkeys(PREDICTION_OUTPUT_COLUMNS)
    row.update(
        game_id=GAME_ID,
        season=2026,
        week=1,
        home_team="KC",
        away_team="DEN",
        wp_prob=0.61,
        ats_prediction=4.2,
        ou_prediction=45.1,
        market_spread=2.5,
        market_wp=0.58,
        wp_edge=-0.012,  # the old moneyline-based edge
    )
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pd.DataFrame([row], columns=PREDICTION_OUTPUT_COLUMNS).to_csv(
        predictions_dir / "predictions_2026_week1.csv", index=False
    )
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_score": [None],
            "away_score": [None],
            "kickoff_et": [pd.Timestamp("2026-09-13 16:25", tz="America/New_York")],
        }
    ).to_parquet(silver_dir / "games.parquet")

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    assert _load_current_week_predictions(conn, predictions_dir, silver_dir) == 1

    game = DataService(conn).get_game_detail(GAME_ID)
    assert game is not None
    assert game["wp_edge"] == pytest.approx(0.61 - 0.58)
    assert game["wp_confidence"] == "medium"  # 0.03 on WP's 0.05 / 0.02 pair

    card = (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=game)
    )
    assert "KC 61.0%" in card and "KC 58.0%" in card
    assert "Edge: +3.0%" in card


def test_the_prediction_script_takes_the_wp_edge_against_market_wp() -> None:
    """The CSV writer uses the same yardstick; no market WP means no edge and no band."""
    from scripts.generate_current_week_predictions import compute_wp_edge

    frame = pd.DataFrame(
        {
            "wp_prob": [0.61, 0.70],
            "market_wp": [0.58, float("nan")],
        }
    )
    result = compute_wp_edge(frame)
    assert result.loc[0, "wp_edge"] == pytest.approx(0.03)
    assert result.loc[0, "wp_confidence"] == "medium"
    assert pd.isna(result.loc[1, "wp_edge"])
    assert result.loc[1, "wp_confidence"] is None


def test_a_game_in_two_week_files_keeps_its_most_recent_prediction(
    tmp_path: Path,
) -> None:
    """33.2 review C2 IN-01: the NEWER file wins, not the file whose name sorts last.

    ``predictions_2026_week10.csv`` sorts before ``predictions_2026_week2.csv``, so the old
    lexicographic order kept week 2's row for a game present in both, whichever was written last.
    """
    import os

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    for week, wp_prob, age_seconds in ((2, 0.30, 0), (10, 0.70, 3600)):
        row = dict.fromkeys(PREDICTION_OUTPUT_COLUMNS)
        row.update(
            game_id=GAME_ID,
            season=2026,
            week=week,
            home_team="KC",
            away_team="DEN",
            wp_prob=wp_prob,
            ats_prediction=4.2,
            ou_prediction=45.1,
        )
        path = predictions_dir / f"predictions_2026_week{week}.csv"
        pd.DataFrame([row], columns=PREDICTION_OUTPUT_COLUMNS).to_csv(path, index=False)
        # week 2 is the NEWER file: week 10 was written an hour earlier.
        stamp = 1_800_000_000 - age_seconds
        os.utime(path, (stamp, stamp))
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_score": [None],
            "away_score": [None],
            "kickoff_et": [pd.Timestamp("2026-09-13 16:25", tz="America/New_York")],
        }
    ).to_parquet(silver_dir / "games.parquet")

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    assert _load_current_week_predictions(conn, predictions_dir, silver_dir) == 1

    game = DataService(conn).get_game_detail(GAME_ID)
    assert game is not None
    assert game["wp_prob"] == pytest.approx(0.30), "the older week-10 row survived"
    assert game["week"] == 2


@pytest.mark.parametrize("session_tz", ["UTC", "America/Los_Angeles", "Asia/Tokyo"])
def test_game_date_is_the_eastern_kickoff_whatever_the_server_zone(
    tmp_path: Path, session_tz: str
) -> None:
    """33.2 review C2 IN-03: ``game_date`` is the Eastern wall clock, pinned by the loader.

    Silver ``kickoff_et`` is tz-aware; written into the naive TIMESTAMP column it was cast through
    the DuckDB SESSION zone, so it read as Eastern only on an Eastern machine.
    """
    row = dict.fromkeys(PREDICTION_OUTPUT_COLUMNS)
    row.update(
        game_id=GAME_ID,
        season=2026,
        week=1,
        home_team="KC",
        away_team="DEN",
        wp_prob=0.64,
        ats_prediction=4.2,
        ou_prediction=45.1,
    )
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pd.DataFrame([row], columns=PREDICTION_OUTPUT_COLUMNS).to_csv(
        predictions_dir / "predictions_2026_week1.csv", index=False
    )
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    pd.DataFrame(
        {
            "game_id": [GAME_ID],
            "home_score": [None],
            "away_score": [None],
            # 16:25 ET, stored UTC exactly as silver stores it.
            "kickoff_et": [pd.Timestamp("2026-09-13 20:25", tz="UTC")],
        }
    ).to_parquet(silver_dir / "games.parquet")

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET TimeZone='{session_tz}'")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    assert _load_current_week_predictions(conn, predictions_dir, silver_dir) == 1

    game = DataService(conn).get_game_detail(GAME_ID)
    assert game is not None
    assert str(game["game_date"]) == "2026-09-13 16:25:00"


def test_equal_numbers_render_no_pick_and_an_even_wp_names_one_team() -> None:
    """33.2 review C2 IN-05: the labels agree with the grading on every equality case.

    ``_ats_outcome`` / ``_ou_outcome`` exclude a model number equal to the line as "no pick", but
    the card said "{away} covers" and "Under"; and "Predicted:" named the away team at exactly
    0.5 while the WP figure named the home team.
    """
    game = {
        **dict.fromkeys(PREDICTION_OUTPUT_COLUMNS),
        "game_id": GAME_ID,
        "home_team": "KC",
        "away_team": "DEN",
        "status": "completed",
        "home_score": 24,
        "away_score": 20,
        "wp_prob": 0.5,
        "ats_prediction": 3.0,
        "market_spread": 3.0,
        "ou_prediction": 44.5,
        "market_total": 44.5,
    }
    card = (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=game)
    )
    assert card.count("No pick") == 2
    assert "DEN covers" not in card
    assert "Under" not in card
    assert "Predicted: DEN 50% WP" in " ".join(card.split())
    assert "DEN 50.0%" in card and "KC 50.0%" not in card
