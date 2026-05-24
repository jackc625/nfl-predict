"""Regression tests for game_context divisional-flag recomputation.

Guards the fix for the corrupted ``is_divisional`` flag in the web cache.
Previously ``api/cache.py`` applied ``.astype(bool)`` to the gold
``is_divisional`` column, which is expanding-window z-score normalized
(float). Any non-zero z-score became ``True``, so ~99.8% of games were
mislabeled "Divisional Game" in the UI.

The fix recomputes the flag from the canonical
``ratings.elo.is_divisional_game`` via the module-level helper
``api.cache._compute_is_divisional``. These tests assert known
divisional / non-divisional pairs, a genuine bool dtype, and a plausible
divisional-rate band that explicitly catches the ~100%-True regression.
"""

from __future__ import annotations

import pandas as pd

from api.cache import _compute_is_divisional


def test_same_division_pairs_resolve_true() -> None:
    """Known same-division matchups must resolve to True."""
    home = pd.Series(["BUF", "KC", "DAL", "KC"])
    away = pd.Series(["NYJ", "LV", "PHI", "DEN"])

    result = _compute_is_divisional(home, away)

    assert result.tolist() == [True, True, True, True]


def test_cross_division_pairs_resolve_false() -> None:
    """Known cross-division / cross-conference matchups must resolve to False."""
    home = pd.Series(["BUF", "DAL", "GB", "WAS", "KC"])
    away = pd.Series(["KC", "SF", "MIA", "LAC", "CLE"])

    result = _compute_is_divisional(home, away)

    assert result.tolist() == [False, False, False, False, False]


def test_result_is_real_bool_dtype() -> None:
    """The schema column is BOOLEAN, so the Series must be real bool dtype.

    A float dtype here is the exact shape of the original bug.
    """
    home = pd.Series(["BUF", "BUF"])
    away = pd.Series(["NYJ", "KC"])

    result = _compute_is_divisional(home, away)

    assert result.dtype == bool


def test_divisional_rate_in_plausible_band() -> None:
    """Over a representative mixed set, the True-rate stays in a plausible band.

    Each team plays 6 of 17 divisional games (~35% league-wide), so a
    realistic rate sits well under 0.5. A rate near 1.0 reproduces the
    normalized-float-to-bool regression and must fail this assertion.
    """
    # Mixed set: 4 divisional pairs, 8 non-divisional pairs -> rate = 1/3.
    home = pd.Series(
        [
            "BUF",  # divisional (AFC East)
            "KC",  # divisional (AFC West)
            "DAL",  # divisional (NFC East)
            "GB",  # divisional (NFC North)
            "BUF",  # cross-division (AFC)
            "DAL",  # cross-division (NFC)
            "GB",  # cross-conference
            "WAS",  # cross-conference
            "KC",  # cross-division (AFC)
            "SEA",  # cross-division (NFC)
            "NE",  # cross-division (AFC)
            "PIT",  # cross-conference
        ]
    )
    away = pd.Series(
        [
            "NYJ",
            "LV",
            "PHI",
            "MIN",
            "KC",
            "SF",
            "MIA",
            "LAC",
            "CLE",
            "NYG",
            "DEN",
            "ATL",
        ]
    )

    result = _compute_is_divisional(home, away)
    rate = float(result.mean())

    assert 0.2 <= rate <= 0.5, f"divisional rate {rate:.3f} outside plausible band"
