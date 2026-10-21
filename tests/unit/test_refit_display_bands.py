"""The website's edge bands follow the in-force re-fit generation (Plan 37-12, D-36, RFIT-06).

WHAT IS PROVEN
--------------
The ``*_confidence`` columns of the cache's ``predictions`` table are DISPLAY labels: each
prediction's edge banded by ``utils.edge_tier``. Owner ruling D-36: after a re-fit swap they are
banded with the in-force generation's re-measured edge-tier thresholds, not the row-19 constants.

* :func:`refit.generations.in_force_edge_thresholds` resolves the thresholds OUTSIDE ``api/``:
  ``None`` for generation 0 (``latest.json`` names the pin) and for no ``latest.json`` at all, and
  the generation's bet-rule record thresholds once a generation is installed;
* ``utils.edge_tier`` keeps ONE band implementation and only gains an optional ``thresholds``
  mapping (``None`` = the row-19 constants);
* ``api.cache.populate_cache(edge_thresholds_by_target=...)`` stores the generation's band, and
  with ``None`` stores exactly today's band, for both prediction loaders;
* ``scripts/populate_cache.py`` resolves the thresholds from its OWN ``--artifacts-dir``, never
  the production default (review round 1, T-37-71).

Every store lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/`` and ``ledger/`` are never read or written (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pandas as pd
import pytest

import scripts.populate_cache as populate_cache_script
from api.cache import populate_cache
from backtest.neutral_hfa_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from refit import generations
from tests.unit.test_refit_decision_path import build_stores, install_generation_one
from utils.edge_tier import edge_tier, edge_tier_series

# A generation-1 mapping whose ATS pair bands a 1.0-point edge "high"; the row-19 ATS pair
# (1.6229 / 0.6211) bands the same edge "medium". WP and O/U keep their row-19 pairs.
GENERATION_ONE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "ats": (0.9, 0.5),
    "ou": (0.0452, 0.0162),
    "wp": (0.05, 0.02),
}

# The edge whose band differs between the two rulers, and the two bands.
_ATS_EDGE = 1.0
_ROW19_BAND = "medium"
_GENERATION_ONE_BAND = "high"

_BACKTEST_GAME = "2025_W01_DAL@PHI"
_CURRENT_WEEK_GAME = "2026_W08_NYG@DAL"


# ---------------------------------------------------------------------------
# The resolver (refit/generations.py)
# ---------------------------------------------------------------------------


def test_generation_zero_resolves_to_none(tmp_path: Path) -> None:
    """While latest.json names the pin the display keeps the row-19 bands; no manifest is None too."""
    stores = build_stores(tmp_path / "pinned")
    ledger = tmp_path / "ledger"

    assert generations.in_force_edge_thresholds(stores.artifacts, ledger) is None
    assert not ledger.exists(), (
        "generation 0 must not create or read the re-fit records"
    )

    empty_artifacts = tmp_path / "no_manifest"
    empty_artifacts.mkdir()
    assert generations.in_force_edge_thresholds(empty_artifacts, ledger) is None


def test_generation_one_resolves_to_its_record(tmp_path: Path) -> None:
    """After an install the mapping is the generation record's own thresholds, as tuples."""
    installed = install_generation_one(
        tmp_path, edge_tier_thresholds=GENERATION_ONE_THRESHOLDS
    )

    resolved = generations.in_force_edge_thresholds(
        installed.stores.artifacts, installed.ledger
    )

    assert resolved == GENERATION_ONE_THRESHOLDS
    assert resolved is not None
    assert all(isinstance(pair, tuple) for pair in resolved.values())


# ---------------------------------------------------------------------------
# The one band implementation (utils/edge_tier.py)
# ---------------------------------------------------------------------------


def test_explicit_thresholds_change_the_band() -> None:
    """The keyword selects the ruler; omitting it (or None) is today's row-19 ruler."""
    assert EDGE_TIER_THRESHOLDS_BY_TARGET["ats"] == (1.6229, 0.6211)
    assert edge_tier(_ATS_EDGE, "ats") == _ROW19_BAND
    assert edge_tier(_ATS_EDGE, "ats", thresholds=None) == _ROW19_BAND
    assert (
        edge_tier(_ATS_EDGE, "ats", thresholds=GENERATION_ONE_THRESHOLDS)
        == _GENERATION_ONE_BAND
    )

    values = pd.Series([_ATS_EDGE, 0.6, 0.4, None])
    scalar = [
        edge_tier(value, "ats", thresholds=GENERATION_ONE_THRESHOLDS)
        for value in values
    ]
    series = edge_tier_series(values, "ats", thresholds=GENERATION_ONE_THRESHOLDS)
    assert list(series) == scalar == ["high", "medium", "low", None]

    # A given mapping is used as-is: an unknown target is still refused by name.
    from utils.edge_tier import UnknownEdgeTargetError

    with pytest.raises(UnknownEdgeTargetError):
        edge_tier_series(
            pd.Series([], dtype=float), "spread", thresholds=GENERATION_ONE_THRESHOLDS
        )


# ---------------------------------------------------------------------------
# The cached band (api/cache.py)
# ---------------------------------------------------------------------------


def _write_prediction_inputs(root: Path) -> tuple[Path, Path, Path]:
    """One backtest row and one current-week row, each with a 1.0-point ATS edge."""
    outputs = root / "outputs"
    silver = root / "silver"
    predictions = root / "predictions"
    for directory in (outputs, silver, predictions):
        directory.mkdir(parents=True)

    shared = {
        "game_id": _BACKTEST_GAME,
        "season": 2025,
        "ml_home": -150.0,
        "ml_away": 130.0,
        "spread": 3.0,
        "total": 47.5,
        "probability_clv": 0.0,
    }
    pd.DataFrame(
        [
            {**shared, "target": "wp", "model_value": 0.60},
            {**shared, "target": "ats", "model_spread": 3.0 + _ATS_EDGE},
            {**shared, "target": "ou", "model_total": 47.5},
        ]
    ).to_csv(outputs / "predictions_all.csv", index=False)

    pd.DataFrame(
        [
            {
                "game_id": _CURRENT_WEEK_GAME,
                "season": 2026,
                "week": 8,
                "home_team": "DAL",
                "away_team": "NYG",
                "wp_prob": 0.55,
                "ats_prediction": 2.5 + _ATS_EDGE,
                "ou_prediction": 44.0,
                "market_spread": 2.5,
                "market_total": 44.0,
                "ats_edge": _ATS_EDGE,
                "ou_edge": 0.0,
            }
        ]
    ).to_csv(predictions / "predictions_2026_week8.csv", index=False)

    pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_score": None,
                "away_score": None,
                "kickoff_et": pd.Timestamp(kickoff, tz="UTC"),
            }
            for game_id, season, week, home, away, kickoff in (
                (_BACKTEST_GAME, 2025, 1, "PHI", "DAL", "2025-09-05 00:20:00"),
                (_CURRENT_WEEK_GAME, 2026, 8, "DAL", "NYG", "2026-10-30 00:15:00"),
            )
        ]
    ).to_parquet(silver / "games.parquet")
    return outputs, silver, predictions


def _cached_ats_bands(
    root: Path, thresholds: dict[str, tuple[float, float]] | None
) -> dict[str, str | None]:
    outputs, silver, predictions = _write_prediction_inputs(root)
    db_path = root / "cache.duckdb"
    populate_cache(
        db_path=db_path,
        # An EMPTY artifacts root: no blend, no latest.json, never production's.
        artifacts_dir=root / "artifacts",
        outputs_dir=outputs,
        gold_dir=root / "gold",
        silver_dir=silver,
        predictions_dir=predictions,
        edge_thresholds_by_target=thresholds,
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT game_id, ats_edge, ats_confidence FROM predictions"
        ).fetchall()
    finally:
        conn.close()
    assert {row[0] for row in rows} == {_BACKTEST_GAME, _CURRENT_WEEK_GAME}
    assert all(row[1] == pytest.approx(_ATS_EDGE) for row in rows), rows
    return {row[0]: row[2] for row in rows}


def test_cached_band_follows_the_generation(tmp_path: Path) -> None:
    """Both prediction loaders band with the generation's pair; None stores today's band."""
    generation_one = _cached_ats_bands(tmp_path / "gen1", GENERATION_ONE_THRESHOLDS)
    today = _cached_ats_bands(tmp_path / "gen0", None)

    assert generation_one == dict.fromkeys(
        (_BACKTEST_GAME, _CURRENT_WEEK_GAME), _GENERATION_ONE_BAND
    )
    assert today == dict.fromkeys((_BACKTEST_GAME, _CURRENT_WEEK_GAME), _ROW19_BAND)


# ---------------------------------------------------------------------------
# The manual rebuild resolves from its own --artifacts-dir (review round 1)
# ---------------------------------------------------------------------------


def test_the_manual_rebuild_resolves_from_its_own_artifacts_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom-directory rebuild can never band with production's generation (T-37-71)."""
    resolved_with: list[tuple[Path, Path]] = []
    populated: list[dict[str, Any]] = []
    sentinel = {"ats": (0.9, 0.5), "ou": None, "wp": (0.05, 0.02)}

    def _resolver(artifacts_dir: Path, ledger_dir: Path) -> dict[str, Any]:
        resolved_with.append((artifacts_dir, ledger_dir))
        return sentinel

    sources = SimpleNamespace(
        bet_list=None,
        tracker=None,
        schedule=None,
        corrections=None,
        graded_outcomes=None,
        verdict_context=None,
        week_results=None,
    )
    monkeypatch.setattr(
        populate_cache_script, "read_bet_cache_sources", lambda *a, **kw: sources
    )
    monkeypatch.setattr(
        populate_cache_script, "populate_cache", lambda **kw: populated.append(kw)
    )
    monkeypatch.setattr(populate_cache_script, "in_force_edge_thresholds", _resolver)

    custom = tmp_path / "custom_artifacts"
    args = argparse.Namespace(
        db_path=tmp_path / "web_cache.duckdb",
        artifacts_dir=custom,
        outputs_dir=tmp_path / "outputs",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
        predictions_dir=tmp_path / "predictions",
        bet_list_dir=tmp_path / "bet_list",
    )
    populate_cache_script._rebuild(args)

    assert [artifacts for artifacts, _ledger in resolved_with] == [custom]
    assert resolved_with[0][0] != Path("artifacts")
    assert len(populated) == 1
    assert populated[0]["edge_thresholds_by_target"] is sentinel
    assert populated[0]["artifacts_dir"] == custom
