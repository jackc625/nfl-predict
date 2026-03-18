---
phase: 01-foundation-hardening
plan: 02
subsystem: data-ingestion
tags: [nflreadpy, open-meteo, httpx, zoneinfo, weather, data-migration]

# Dependency graph
requires:
  - phase: 01-foundation-hardening/01
    provides: "uv-based Python dependency management, Ruff linting, WeatherDataError exception class"
provides:
  - "nflreadpy replacing nfl_data_py across all call sites (3 files)"
  - "Open-Meteo via httpx replacing Meteostat for weather ingestion"
  - "zoneinfo replacing pytz in weather ingestion"
  - "Integration smoke tests for nflreadpy and Open-Meteo"
  - "Full weather field set: temp, dewpoint, wind, precip, snow, gusts, cloud cover, weather code"
affects: [02-data-pipeline, 03-feature-engineering, 05-new-features]

# Tech tracking
tech-stack:
  added: []
  removed_usage: [nfl_data_py, meteostat, pytz]
  patterns:
    - "nflreadpy: nfl.load_*().to_pandas() for all data loading"
    - "Open-Meteo: httpx async GET to archive-api.open-meteo.com/v1/archive"
    - "tenacity retry with exponential backoff for weather API calls"
    - "zoneinfo.ZoneInfo replacing pytz.timezone for timezone handling"

key-files:
  created:
    - "tests/integration/test_nflreadpy_smoke.py"
    - "tests/integration/test_openmeteo_smoke.py"
  modified:
    - "scripts/ingest_games.py"
    - "scripts/ingest_historical_odds.py"
    - "features/team_form.py"
    - "scripts/ingest_weather.py"
    - "scripts/load_venues.py"

key-decisions:
  - "Used httpx directly for Open-Meteo instead of openmeteo-requests library (simpler, already in stack)"
  - "Kept asyncio.run() for weather fetches within synchronous ingestion loop (pragmatic, avoids full async refactor)"
  - "Preserved mock weather generation for testing and indoor games (behavioral compatibility)"

patterns-established:
  - "nflreadpy load pattern: nfl.load_schedules(seasons).to_pandas(), nfl.load_pbp(seasons).to_pandas()"
  - "Open-Meteo fetch pattern: async httpx GET with tenacity retry, WeatherDataError on failure"
  - "Weather output schema: temp_f, temp_c, wind_mph, wind_direction, humidity_pct, precip_mm + dew_point_f, apparent_temp_f, snowfall_cm, wind_gusts_mph, cloud_cover_pct, weather_code"

requirements-completed: [FOUN-01, FOUN-02]

# Metrics
duration: 8min
completed: 2026-03-18
---

# Phase 01 Plan 02: Data Library Migration Summary

**Replaced nfl_data_py with nflreadpy and Meteostat with Open-Meteo httpx calls, with integration smoke tests confirming both return real data**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-18T21:47:44Z
- **Completed:** 2026-03-18T21:56:46Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Migrated all 3 files importing nfl_data_py to nflreadpy (load_schedules, load_pbp with .to_pandas())
- Rewrote weather ingestion to use Open-Meteo Historical API via httpx with full weather field coverage
- Replaced pytz with stdlib zoneinfo in weather ingestion
- Added 4 integration smoke tests (2 nflreadpy, 2 Open-Meteo) all passing
- Zero remaining references to nfl_data_py, meteostat, or pytz in touched files

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate nfl_data_py to nflreadpy across all call sites** - `7c70abf` (feat)
2. **Task 2: Migrate Meteostat to Open-Meteo via httpx** - `6a46d26` (feat)

## Files Created/Modified
- `scripts/ingest_games.py` - nfl.import_schedules -> nfl.load_schedules().to_pandas(), narrowed except blocks
- `scripts/ingest_historical_odds.py` - Same nflreadpy migration, updated docstrings and metadata
- `features/team_form.py` - nfl.import_pbp_data -> nfl.load_pbp().to_pandas(), removed include_participation
- `scripts/ingest_weather.py` - Full rewrite: Meteostat -> Open-Meteo httpx, pytz -> zoneinfo, async fetch with retry
- `scripts/load_venues.py` - Fixed bare except to catch (FileNotFoundError, json.JSONDecodeError)
- `tests/integration/test_nflreadpy_smoke.py` - Smoke test proving nflreadpy returns 2024 schedule data
- `tests/integration/test_openmeteo_smoke.py` - Smoke test proving Open-Meteo returns weather for MetLife Stadium

## Decisions Made
- Used httpx directly for Open-Meteo API rather than the openmeteo-requests library -- httpx is already a dependency, the REST API is simple JSON, and adding openmeteo-requests would introduce FlatBuffers complexity for no benefit
- Kept asyncio.run() within the synchronous game iteration loop rather than converting the entire ingestion pipeline to async -- pragmatic choice that avoids a full async refactor while still getting retry benefits
- Preserved the mock weather generation for testing and indoor games -- behavioral compatibility with existing callers

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed bare except in load_venues.py**
- **Found during:** Task 2 (auditing load_venues.py)
- **Issue:** Line 196 had a bare `except:` clause catching all exceptions silently
- **Fix:** Changed to `except (FileNotFoundError, json.JSONDecodeError):` for specific types
- **Files modified:** scripts/load_venues.py
- **Verification:** ruff check passes, behavior preserved
- **Committed in:** 6a46d26 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug via fix-what-you-touch policy)
**Impact on plan:** Minimal -- fixed pre-existing code quality issue in touched file. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. Open-Meteo requires no API key (free tier: 10,000 requests/day).

## Next Phase Readiness
- All data library migrations complete -- subsequent phases can use nflreadpy and Open-Meteo
- Weather ingestion now returns expanded field set (dew_point_f, snowfall_cm, wind_gusts_mph, cloud_cover_pct, weather_code) available for feature engineering in Phase 3
- Remaining pytz references in untouched files (features/contextual.py, scripts/data_qa.py, etc.) can be cleaned up via Plan 03's error handling sweep or the fix-what-you-touch policy
- Venue data verified: all 32 NFL teams covered, coordinates accurate, includes SoFi and Allegiant stadiums

## Self-Check: PASSED

All claimed files exist. All commit hashes verified in git log.

---
*Phase: 01-foundation-hardening*
*Completed: 2026-03-18*
