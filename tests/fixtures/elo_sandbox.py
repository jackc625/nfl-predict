"""Shared, sandbox-only helpers for the Phase-33 Elo write-path tests.

WHY THIS MODULE EXISTS
----------------------
Five Phase-33 modules drive the Elo write path -- Plan 33-03's two write verbs, the
generation publisher, the persist refusal, the per-season append isolation and the
rerun-identity suite. Every one of them needs the SAME three things: ``data.storage``'s
module singletons pointed at a ``tmp_path`` sandbox, a small but real-shaped games
frame, and an ``EloBuilder`` whose ``data_root`` is that sandbox. Copying that setup
into five modules would produce five subtly different sandboxes, and the one that
drifted would be the one that wrote production -- which is the Phase-31 defect
``tests/integration/test_elo_integration.py``'s docstring records at length.

THE TWO SEAMS ARE NOT ONE SEAM, AND BOTH ARE NEEDED.
``data.storage.save_dataframe`` resolves its destination through the module-level
``_parquet_manager`` / ``_db_connection`` singletons and takes no root argument, so
``monkeypatch.setattr`` on those globals is its only injectable seam.
``data.storage.upsert_silver`` -- the verb the live append uses -- takes an explicit
``base_path`` instead and ignores the singletons entirely. A test that redirected only
the singletons would still upsert into ``data/silver/``. Both are set here, from one
sandbox root, so they cannot disagree.

NOTHING HERE WRITES ``data/`` OR ``artifacts/``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scripts.build_elo import EloBuilder

# Eight teams spanning four divisions, so ``is_divisional_game`` resolves True for some
# pairings and False for others without the frame having to name all 32.
SANDBOX_TEAMS: tuple[str, ...] = (
    "BUF",
    "MIA",
    "NE",
    "NYJ",
    "KC",
    "DEN",
    "LAC",
    "LV",
)

# The four fixed matchups played every week of a sandbox season. Two are divisional
# (BUF/MIA, KC/DEN) and two are not, so the divisional HFA reduction is exercised.
# Kickoffs are tz-aware Eastern, matching the real silver ``games`` table. A naive
# datetime is rejected outright by ``DuckDBConnection._normalize_datetime_columns``,
# so a naive fixture would not even reach the code under test.
_ET = ZoneInfo("America/New_York")

_WEEKLY_MATCHUPS: tuple[tuple[str, str], ...] = (
    ("BUF", "MIA"),
    ("KC", "DEN"),
    ("NE", "LAC"),
    ("NYJ", "LV"),
)


def make_season_games(
    season: int,
    *,
    weeks: int = 3,
    graded_weeks: tuple[int, ...] | None = None,
    home_points: int = 27,
    away_points: int = 17,
) -> pd.DataFrame:
    """Build a small season frame in the shape the Elo path actually reads.

    Args:
        season: Season year stamped on every row.
        weeks: How many weeks to generate (four games each).
        graded_weeks: Weeks that carry scores. ``None`` means every week is graded;
            an empty tuple produces a season with ZERO completed games, which is R3's
            explicit edge case and must NOT be a hard failure anywhere downstream.
        home_points: Score the home team records in a graded game.
        away_points: Score the away team records in a graded game.

    Returns:
        DataFrame with ``game_id``, ``season``, ``week``, ``home_team``, ``away_team``,
        ``home_score``, ``away_score`` and ``kickoff_et``.
    """
    graded = tuple(range(1, weeks + 1)) if graded_weeks is None else graded_weeks

    rows: list[dict] = []
    for week in range(1, weeks + 1):
        for index, (home, away) in enumerate(_WEEKLY_MATCHUPS):
            is_graded = week in graded
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_{away}@{home}",
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(home_points) if is_graded else None,
                    "away_score": float(away_points) if is_graded else None,
                    "kickoff_et": datetime(
                        season, 9, 6 + (week * 7), 13 + index, 0, tzinfo=_ET
                    ),
                }
            )

    return pd.DataFrame(rows)


def redirect_storage_to_sandbox(monkeypatch, tmp_path: Path) -> Path:
    """Point every ``data.storage`` write seam at a sandbox and return its data root.

    Returns:
        The sandbox data root (the analogue of the repo's ``data/``), suitable for
        passing to ``EloBuilder(data_root=...)`` and to ``upsert_silver(base_path=...)``.
    """
    import data.storage as storage_mod
    from data.storage import DuckDBConnection, ParquetManager

    sandbox = tmp_path / "data"
    (sandbox / "silver").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(sandbox)))
    monkeypatch.setattr(
        storage_mod,
        "_db_connection",
        DuckDBConnection(str(sandbox / "sandbox.duckdb")),
    )
    return sandbox


def seed_sandbox_games(games: pd.DataFrame) -> None:
    """Write *games* into the (already redirected) sandbox silver layer."""
    from data.storage import save_dataframe

    save_dataframe(games, "games", layer="silver", replace_mode=True)


def sandbox_builder(sandbox: Path, games: pd.DataFrame) -> EloBuilder:
    """Seed *games* and return an ``EloBuilder`` rooted at *sandbox*."""
    from scripts.build_elo import EloBuilder

    seed_sandbox_games(games)
    return EloBuilder(data_root=sandbox)


def read_sandbox_table(sandbox: Path, table_name: str) -> pd.DataFrame:
    """Read a silver table's PARQUET bytes straight off the sandbox.

    Deliberately NOT ``load_dataframe``: that resolves DuckDB first, so it would answer
    from the database copy and could not tell a stale parquet from a fresh one. The
    parquet file is the artifact this plan's write verbs are judged on.
    """
    path = sandbox / "silver" / f"{table_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def per_season_row_digests(frame: pd.DataFrame) -> dict[str, str]:
    """Per-season ROW digests of *frame*, invariant to row and column order.

    D33-08's "byte-identical" is read as PER-SEASON ROW DIGESTS rather than a file
    sha256, because a parquet or DuckDB rewrite moves file bytes on a logical no-op --
    demonstrated live in this repository by
    ``tests/integration/test_n01_resync_control.py``'s resync-idempotency arm. Digesting
    the ROWS is the only reading a writer can actually satisfy.

    The shape is ``scripts/fingerprint_gold._per_season_digests``'s -- sort by
    ``game_id``, group by ``season``, sha256, truncate to 16 hex characters -- widened
    from one column to the whole row, because R3's criterion is about rows.
    """
    if len(frame) == 0 or "season" not in frame.columns:
        return {}

    ordered = frame.sort_values("game_id").reset_index(drop=True)
    ordered = ordered[sorted(ordered.columns)]

    digests: dict[str, str] = {}
    for season in sorted(int(value) for value in frame["season"].dropna().unique()):
        block = ordered[ordered["season"] == season]
        payload = block.to_csv(index=False).encode("utf-8")
        digests[str(season)] = hashlib.sha256(payload).hexdigest()[:16]
    return digests
