---
phase: 01-foundation-hardening
plan: 04
subsystem: frontend-toolchain
tags: [tailwind-v4, htmx, cleanup, zoneinfo, standalone-css]

# Dependency graph
requires:
  - phase: 01-foundation-hardening/02
    provides: "pytz replaced with zoneinfo in production code"
  - phase: 01-foundation-hardening/03
    provides: "BLE ruff rule enforced, exception hierarchy in place"
provides:
  - "Tailwind v4 standalone CLI (no Node.js dependency)"
  - "HTMX 2.0 CDN smoke test template proving stack works"
  - "Makefile build-css target for Tailwind v4 standalone"
  - "Zero pytz imports remaining anywhere in the project"
  - "27 dead test scripts removed (~13k lines of dead code)"
  - "Clean .gitignore for modernized toolchain (tools/, node_modules/)"
affects: [08-api-web-ui]

# Tech tracking
tech-stack:
  added: [tailwind-v4-standalone, htmx-2.0]
  removed: [node_modules, package.json, tailwind.config.js, 27-dead-test-scripts]
  patterns:
    - "Tailwind v4 CSS-only config: @import 'tailwindcss' in input.css (no tailwind.config.js)"
    - "Tailwind build via standalone binary: ./tools/tailwindcss -i input.css -o output.css --minify"
    - "HTMX 2.0 loaded via CDN script tag (no npm)"

key-files:
  created:
    - "web/static/input.css"
    - "web/templates/smoke_test.html"
  modified:
    - ".gitignore"
    - "Makefile"
    - "tests/conftest.py"
    - "tests/integration/test_historical_week_pipeline.py"
  deleted:
    - "27 scripts/test_*.py files (~13k lines)"
    - "package.json"
    - "tailwind.config.js"
    - "node_modules/"

key-decisions:
  - "Tailwind v4 standalone binary replaces Node.js npm-based build -- zero JS toolchain dependency"
  - "HTMX 2.0 via CDN rather than local bundle -- keeps frontend dependency-free"
  - "Pre-existing test failures (46 tests) confirmed as interface mismatches, not Phase 1 regressions"

patterns-established:
  - "Frontend CSS: Tailwind v4 standalone CLI processes web/static/input.css to web/static/css/tailwind-compiled.css"
  - "Frontend JS: HTMX 2.0 via CDN (cdn.jsdelivr.net/npm/htmx.org@2.0.8)"
  - "No Node.js in the project -- Python + standalone binaries only"

requirements-completed: [FOUN-09, FOUN-10]

# Metrics
duration: 12min
completed: 2026-03-18
---

# Phase 01 Plan 04: Frontend Toolchain and Cleanup Summary

**Installed Tailwind v4 standalone CLI and HTMX 2.0 smoke test, removed 27 dead test scripts (~13k lines), eliminated all pytz imports project-wide**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-18T22:10:00Z
- **Completed:** 2026-03-18T22:22:00Z
- **Tasks:** 2
- **Files modified:** 29

## Accomplishments
- Tailwind v4 standalone binary installed at tools/tailwindcss.exe, producing CSS output from CSS-only config (no tailwind.config.js, no Node.js)
- HTMX 2.0 smoke test template created at web/templates/smoke_test.html, loading HTMX via CDN and using Tailwind-compiled CSS
- Removed 27 dead test scripts from scripts/ directory (~13,158 lines of dead code removed)
- Eliminated all remaining pytz imports in test files (tests/conftest.py, tests/integration/test_historical_week_pipeline.py) -- zero pytz references remain project-wide
- Updated .gitignore with tools/ and node_modules/ entries for modernized toolchain
- Updated Makefile with build-css target using Tailwind v4 standalone

## Task Commits

Each task was committed atomically:

1. **Task 1: Set up Tailwind v4 standalone, HTMX 2.0 smoke test, and clean up dead files** - `2e290e0` (feat)
2. **Task 2: Verify complete Phase 1 Foundation Hardening** - checkpoint:human-verify (user approved -- 46 test failures confirmed as pre-existing interface mismatches, not Phase 1 regressions)

## Files Created/Modified
- `web/static/input.css` - Tailwind v4 CSS-only config (@import "tailwindcss")
- `web/templates/smoke_test.html` - HTMX 2.0 + Tailwind v4 smoke test page
- `.gitignore` - Added node_modules/ and tools/ entries
- `Makefile` - Added build-css target, removed dead script references
- `tests/conftest.py` - Replaced pytz with zoneinfo
- `tests/integration/test_historical_week_pipeline.py` - Replaced pytz with zoneinfo
- 27 `scripts/test_*.py` files deleted (~13k lines of dead test scripts)

## Decisions Made
- Used Tailwind v4 standalone binary instead of npm-based toolchain -- eliminates Node.js as a project dependency entirely
- HTMX 2.0 loaded via CDN script tag rather than local bundle -- no build step needed for JS
- 46 test failures confirmed by user as pre-existing interface mismatches (tests written against expected interfaces like EloRatingSystem.get_rating() that don't exist yet, API root returning HTML not JSON) -- not caused by Phase 1 changes; 31 tests pass including new smoke tests

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- 46 of 77 pytest tests fail, but user confirmed these are pre-existing interface mismatches from tests written against planned APIs that haven't been implemented yet (e.g., EloRatingSystem.get_rating() method doesn't exist, API root returns HTML not JSON). The 31 passing tests include the new smoke tests added in Phase 1. These failures will be resolved as the planned interfaces are implemented in subsequent phases.

## User Setup Required
None - no external service configuration required. Tailwind v4 standalone binary is auto-downloaded; HTMX is loaded via CDN.

## Next Phase Readiness
- Phase 1 Foundation Hardening is complete -- all 12 FOUN requirements satisfied
- Modern toolchain ready: uv + Ruff + pyright (Plan 01)
- Data libraries migrated: nflreadpy + Open-Meteo (Plan 02)
- Error handling hardened: specific exceptions, fail-loud DuckDB, no HTML fallback (Plan 03)
- Frontend tooling ready for Phase 8: Tailwind v4 standalone + HTMX 2.0 (Plan 04)
- No deprecated dependencies remain (nfl_data_py, meteostat, pytz all eliminated)
- Phase 2 (Data Pipeline Audit) can begin building on this clean foundation

## Self-Check: PASSED

All claimed files exist. Commit hash 2e290e0 verified in git log.

---
*Phase: 01-foundation-hardening*
*Completed: 2026-03-18*
