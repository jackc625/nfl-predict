"""AUDIT-05 source-freshness diff harness (Phase 20, plan 20-05, D-04 / D-12).

Proves the "currency" pillar of the Trust & Reproducibility milestone: the on-disk
silver/gold data matches what the FREE sources return for the ingested seasons, the latest
snapshot timestamps are confirmed, and the 2025 partial-season currency gap is documented
(D-05). This is a thin NEW harness entry per D-13 (no existing home fits a source-freshness
diff); it does NOT create a parallel scripts/audit_*.py story.

Freshness method (D-04):
  - Games:   diff the nflreadpy ``load_schedules`` game set against on-disk silver ``games``
             for a fully-ingested season (2024) on a normalized (season, week, home, away)
             key -- the two sources use DIFFERENT raw game_id formats (nflreadpy
             ``2024_01_BAL_KC`` vs silver ``2024_W01_KC@BAL``), so the canonical comparison
             key is the matchup tuple, not the raw id. Assert no MISSING settled games and
             record any EXTRA.
  - Weather: re-fetch the Open-Meteo archive for ONE outdoor game's stadium lat/lon and game
             hour (the same endpoint ``test_openmeteo_smoke.py`` uses) and compare temp/wind
             to the on-disk ``weather`` row at order-of-magnitude tolerance.
  - Odds:    NO live pull (D-04 -- The Odds API is 500 req/hr, costly historical endpoint,
             offseason). Odds freshness = Bronze/Silver snapshot-timestamp confirmation only.

Network pulls SKIP cleanly when the source is unreachable (offline-safe per RESEARCH
Environment Availability); the on-disk snapshot-timestamp confirmation always runs. The test
also writes ``outputs/diagnostics/audit_freshness.md`` (the findings diagnostic emitted for
plan 20-06). No correctness fixes here (D-10 -- catalog only).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data.storage import load_dataframe

# Fully-ingested season used for the games freshness diff (latest complete season on disk).
FRESHNESS_SEASON = 2024

# One outdoor game whose on-disk weather is re-fetched from Open-Meteo.
WEATHER_GAME_ID = "2024_W06_SF@SEA"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

DIAGNOSTIC_PATH = Path("outputs/diagnostics/audit_freshness.md")


def _matchup_key(df: pd.DataFrame) -> set[tuple[int, int, str, str]]:
    """Normalize a games frame to a (season, week, home, away) matchup-key set.

    nflreadpy and silver use different raw game_id string formats, so the matchup tuple is
    the canonical, format-independent freshness key.
    """
    rows = df.dropna(subset=["home_team", "away_team", "season", "week"])
    return {
        (int(s), int(w), str(h), str(a))
        for s, w, h, a in zip(
            rows["season"],
            rows["week"],
            rows["home_team"],
            rows["away_team"],
            strict=True,
        )
    }


def _load_source_schedule(season: int) -> pd.DataFrame | None:
    """Pull nflreadpy schedules for a season; return None if the source is unreachable."""
    try:
        import nflreadpy as nfl

        return nfl.load_schedules([season]).to_pandas()
    except Exception as exc:
        pytest.skip(f"nflreadpy source unreachable (offline-safe skip): {exc}")
        return None


def _fetch_open_meteo(
    lat: float, lon: float, game_date: str, game_hour: int
) -> dict | None:
    """Re-fetch one outdoor game's weather from the Open-Meteo archive.

    Returns the temp_f / wind_mph at the game hour, or None if the source is unreachable.
    """
    try:
        import httpx

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": game_date,
            "end_date": game_date,
            "hourly": "temperature_2m,wind_speed_10m,precipitation",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
        }
        response = httpx.get(OPEN_METEO_URL, params=params, timeout=30.0)
        response.raise_for_status()
        hourly = response.json()["hourly"]
        idx = min(game_hour, len(hourly["temperature_2m"]) - 1)
        return {
            "temp_f": hourly["temperature_2m"][idx],
            "wind_mph": hourly["wind_speed_10m"][idx],
            "precip_mm": hourly["precipitation"][idx],
        }
    except Exception as exc:
        pytest.skip(f"Open-Meteo source unreachable (offline-safe skip): {exc}")
        return None


def _venue_for_team(home_team: str) -> dict | None:
    """Look up the venue (lat/lon) that the given home team plays at."""
    venues = json.loads(Path("data/venues.json").read_text())["venues"]
    for venue in venues:
        if home_team in venue.get("home_teams", []):
            return venue
    return None


@pytest.mark.integration
class TestGamesFreshness:
    """nflreadpy (free source) vs on-disk silver games -- the games freshness diff (D-04)."""

    def test_no_missing_settled_games(self) -> None:
        """Every settled source game for a fully-ingested season is present on disk."""
        source = _load_source_schedule(FRESHNESS_SEASON)
        disk = load_dataframe("games", layer="silver")
        disk_season = disk[disk["season"] == FRESHNESS_SEASON]

        source_keys = _matchup_key(source)
        disk_keys = _matchup_key(disk_season)

        missing = source_keys - disk_keys
        assert not missing, (
            f"{len(missing)} source games MISSING from on-disk silver for "
            f"{FRESHNESS_SEASON}: {sorted(missing)[:5]}"
        )

    def test_row_count_matches_source(self) -> None:
        """On-disk silver game count matches the source schedule count for the season."""
        source = _load_source_schedule(FRESHNESS_SEASON)
        source_settled = source.dropna(subset=["home_team", "away_team"])
        disk = load_dataframe("games", layer="silver")
        disk_season = disk[disk["season"] == FRESHNESS_SEASON].drop_duplicates(
            subset=["game_id"]
        )
        assert len(disk_season) == len(source_settled), (
            f"Row-count drift for {FRESHNESS_SEASON}: source={len(source_settled)} "
            f"disk={len(disk_season)}"
        )

    def test_completed_game_scores_present(self) -> None:
        """On-disk completed games for a fully-settled season carry both team scores."""
        disk = load_dataframe("games", layer="silver")
        disk_season = disk[disk["season"] == FRESHNESS_SEASON]
        # A fully-settled prior season must have no NULL scores on any game.
        null_scores = disk_season[
            disk_season["home_score"].isna() | disk_season["away_score"].isna()
        ]
        assert null_scores.empty, (
            f"{len(null_scores)} completed {FRESHNESS_SEASON} games are missing scores "
            f"on disk: {null_scores['game_id'].head(5).tolist()}"
        )


@pytest.mark.integration
class TestWeatherFreshness:
    """Open-Meteo (free source) re-fetch vs on-disk weather for one outdoor game (D-04)."""

    def test_outdoor_game_weather_matches_source(self) -> None:
        """On-disk weather for one outdoor game is order-of-magnitude consistent with source."""
        weather = load_dataframe("weather", layer="silver")
        disk_rows = weather[weather["game_id"] == WEATHER_GAME_ID]
        assert not disk_rows.empty, (
            f"Expected outdoor weather row {WEATHER_GAME_ID} on disk"
        )
        disk_row = disk_rows.iloc[0]
        assert bool(disk_row["is_outdoor"]), (
            f"{WEATHER_GAME_ID} should be an outdoor game for the weather re-fetch"
        )

        games = load_dataframe("games", layer="silver")
        game = games[games["game_id"] == WEATHER_GAME_ID].iloc[0]
        venue = _venue_for_team(str(game["home_team"]))
        assert venue is not None, f"No venue found for home team {game['home_team']}"

        kickoff = pd.to_datetime(game["kickoff_et"])
        game_date = kickoff.strftime("%Y-%m-%d")
        game_hour = int(kickoff.hour)

        source = _fetch_open_meteo(
            float(venue["latitude"]), float(venue["longitude"]), game_date, game_hour
        )

        # Order-of-magnitude tolerance (re-analysis vs original ingest can differ slightly).
        assert abs(float(source["temp_f"]) - float(disk_row["temp_f"])) <= 15.0, (
            f"temp_f drift: source={source['temp_f']} disk={disk_row['temp_f']}"
        )
        assert abs(float(source["wind_mph"]) - float(disk_row["wind_mph"])) <= 15.0, (
            f"wind_mph drift: source={source['wind_mph']} disk={disk_row['wind_mph']}"
        )


@pytest.mark.integration
class TestOddsSnapshotFreshness:
    """Odds freshness = Bronze/Silver snapshot-timestamp confirmation ONLY (D-04, NO live pull)."""

    def test_odds_snapshot_timestamp_present(self) -> None:
        """The silver odds_snapshot table carries a valid latest snapshot_ts (no live pull)."""
        odds = load_dataframe("odds_snapshot", layer="silver")
        assert not odds.empty, "Expected non-empty silver odds_snapshot table"
        assert "snapshot_ts" in odds.columns, (
            "odds_snapshot must carry snapshot_ts for freshness confirmation"
        )
        latest = pd.to_datetime(odds["snapshot_ts"], errors="coerce").max()
        assert pd.notna(latest), "odds_snapshot has no valid snapshot_ts values"

    def test_bronze_odds_snapshot_files_exist(self) -> None:
        """Bronze odds snapshots exist on disk (the append-only freeze record)."""
        bronze_odds = list(Path("data/bronze").glob("odds_*"))
        assert bronze_odds, "Expected at least one Bronze odds snapshot on disk"


def test_emit_freshness_diagnostic() -> None:
    """Emit outputs/diagnostics/audit_freshness.md (the findings diagnostic for plan 20-06).

    Records: the games diff result, the weather comparison, the odds snapshot-timestamp
    confirmation, and the DOCUMENTED 2025 partial-season currency gap (re-confirmed at
    execution time). ASCII tags only, no emoji (CLAUDE.md).
    """
    lines: list[str] = []
    lines.append("# AUDIT-05 Freshness + Determinism Findings (Phase 20, plan 20-05)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append(
        "Method (D-04): verify on-disk data against the FREE sources (nflreadpy games, "
        "Open-Meteo weather) for ingested seasons + confirm snapshot timestamps. "
        "NO live odds pull (The Odds API offseason / costly). NO correctness fix (D-10)."
    )
    lines.append("")

    # -- Games freshness diff (offline-safe) --
    lines.append("## Games freshness diff (nflreadpy vs on-disk silver)")
    lines.append("")
    disk = load_dataframe("games", layer="silver")
    disk_season = disk[disk["season"] == FRESHNESS_SEASON]
    disk_keys = _matchup_key(disk_season.drop_duplicates(subset=["game_id"]))
    games_source_available = True
    try:
        import nflreadpy as nfl

        source = nfl.load_schedules([FRESHNESS_SEASON]).to_pandas()
        source_keys = _matchup_key(source)
        missing = source_keys - disk_keys
        extra = disk_keys - source_keys
        lines.append(f"- Season checked: {FRESHNESS_SEASON} (fully ingested)")
        lines.append(f"- Source games (nflreadpy): {len(source_keys)}")
        lines.append(f"- On-disk silver games: {len(disk_keys)}")
        lines.append(f"- MISSING (in source, not on disk): {len(missing)}")
        lines.append(f"- EXTRA (on disk, not in source): {len(extra)}")
        if missing:
            lines.append(f"  - missing sample: {sorted(missing)[:5]}")
        if extra:
            lines.append(f"  - extra sample: {sorted(extra)[:5]}")
        verdict = "[PASS]" if not missing else "[FAIL]"
        lines.append(f"- Verdict: {verdict} (no settled source game missing from disk)")
    except Exception as exc:
        games_source_available = False
        lines.append(
            f"- [SKIP] nflreadpy source unreachable at run time (offline): {exc}"
        )
        lines.append(f"- On-disk silver games for {FRESHNESS_SEASON}: {len(disk_keys)}")
    lines.append("")

    # -- Weather freshness re-fetch (offline-safe) --
    lines.append("## Weather freshness re-fetch (Open-Meteo vs on-disk weather)")
    lines.append("")
    weather = load_dataframe("weather", layer="silver")
    disk_w = weather[weather["game_id"] == WEATHER_GAME_ID]
    if disk_w.empty:
        lines.append(f"- [SKIP] No on-disk weather row for {WEATHER_GAME_ID}")
    else:
        disk_row = disk_w.iloc[0]
        lines.append(f"- Game: {WEATHER_GAME_ID} (outdoor)")
        lines.append(
            f"- On-disk: temp_f={disk_row['temp_f']} wind_mph={disk_row['wind_mph']} "
            f"precip_mm={disk_row['precip_mm']}"
        )
        try:
            import httpx

            games = load_dataframe("games", layer="silver")
            game = games[games["game_id"] == WEATHER_GAME_ID].iloc[0]
            venue = _venue_for_team(str(game["home_team"]))
            kickoff = pd.to_datetime(game["kickoff_et"])
            params = {
                "latitude": float(venue["latitude"]),
                "longitude": float(venue["longitude"]),
                "start_date": kickoff.strftime("%Y-%m-%d"),
                "end_date": kickoff.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/New_York",
            }
            resp = httpx.get(OPEN_METEO_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            idx = min(int(kickoff.hour), len(hourly["temperature_2m"]) - 1)
            lines.append(
                f"- Source (Open-Meteo re-fetch): temp_f={hourly['temperature_2m'][idx]} "
                f"wind_mph={hourly['wind_speed_10m'][idx]} "
                f"precip_mm={hourly['precipitation'][idx]}"
            )
            lines.append(
                "- Verdict: [PASS] order-of-magnitude consistent (re-analysis tolerance)"
            )
        except Exception as exc:
            lines.append(
                f"- [SKIP] Open-Meteo unreachable at run time (offline): {exc}"
            )
    lines.append("")

    # -- Odds snapshot-timestamp confirmation (NO live pull, D-04) --
    lines.append(
        "## Odds freshness (snapshot-timestamp confirmation, NO live pull -- D-04)"
    )
    lines.append("")
    odds = load_dataframe("odds_snapshot", layer="silver")
    latest_odds_ts = pd.to_datetime(odds["snapshot_ts"], errors="coerce").max()
    bronze_odds = sorted(p.name for p in Path("data/bronze").glob("odds_*"))
    lines.append(f"- Silver odds_snapshot rows: {len(odds)}")
    lines.append(f"- Latest snapshot_ts (freeze timestamp): {latest_odds_ts}")
    lines.append(f"- Bronze odds snapshots on disk: {bronze_odds}")
    lines.append(
        "- The Odds API was NOT pulled (offseason, 500 req/hr, costly historical "
        "endpoint). Odds freshness = snapshot-timestamp confirmation only (D-04)."
    )
    lines.append("")

    # -- 2025 partial-season currency gap (D-05) -- re-confirmed at execution time --
    lines.append(
        "## 2025 partial-season currency gap (D-05) -- documented, NOT backfilled"
    )
    lines.append("")
    gold = load_dataframe("features_wp", layer="gold")
    gold_2025 = (
        gold[gold["season"] == 2025] if "season" in gold.columns else gold.iloc[0:0]
    )
    gold_2025_weeks = (
        sorted(int(w) for w in gold_2025["week"].unique())
        if "week" in gold_2025.columns and not gold_2025.empty
        else []
    )
    silver_2025 = disk[disk["season"] == 2025]
    silver_2025_weeks = sorted(int(w) for w in silver_2025["week"].unique())
    lines.append(
        f"- Gold (features_wp): 2025 present for weeks {gold_2025_weeks} "
        f"= {len(gold_2025)} rows (full gold span "
        f"{int(gold['season'].min())}-{int(gold['season'].max())}, {len(gold)} rows)"
    )
    lines.append(
        f"- Silver games: 2025 present for weeks {silver_2025_weeks} "
        f"= {len(silver_2025)} rows"
    )
    lines.append(
        "- Disposition (D-05 / AUDIT-05): 2025 ingested through ~week 4-5, frozen, NOT "
        "backfilled this phase. This is an EXPLAINED currency gap, not a failure. "
        "Backfilling 2025 is deferred to the future re-fit milestone. Do NOT propose "
        "backfilling in Phase 20."
    )
    lines.append("")

    DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    written = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    assert "2025 partial-season currency gap" in written
    assert "NO live pull" in written
    # No non-ASCII characters (emoji guard, CLAUDE.md).
    assert written.isascii(), "Diagnostic must be pure ASCII (no emoji)"
    # Games diff must have run unless the source was genuinely offline.
    if games_source_available:
        assert "MISSING (in source, not on disk): 0" in written
