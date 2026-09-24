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
