# Stored GitHub Releases payloads (Plan 32-05, Task 1a)

These three files are a **verbatim snapshot of a live third-party API**, captured once and
committed. They are not hand-written, and nothing in the test suite regenerates them.

## What was captured

| File | URL | Bytes | `assets[]` | `.parquet` assets |
|---|---|---|---|---|
| `pbp.json` | `https://api.github.com/repos/nflverse/nflverse-data/releases/tags/pbp` | 235,963 | 164 | 28 (1999-2026) |
| `schedules.json` | `https://api.github.com/repos/nflverse/nflverse-data/releases/tags/schedules` | 12,386 | 7 | 1 (`games.parquet`) |
| `depth_charts.json` | `https://api.github.com/repos/nflverse/nflverse-data/releases/tags/depth_charts` | 163,958 | 109 | 26 (2001-2026) |

**Captured at (UTC):** 2026-09-11T06:55:21Z, 06:55:22Z and 06:55:22Z respectively, by one
unauthenticated `requests.get` per tag with no headers beyond the `requests` defaults.

**Rate-limit headers observed on the capture run:** `x-ratelimit-limit: 60`,
`x-ratelimit-remaining: 59 -> 58 -> 57` across the three calls, `x-ratelimit-reset:
1789113323`. The limit is 60 requests per hour per IP for unauthenticated access.

## Why they are stored rather than fetched

The GitHub API rate limit above is a hard per-IP ceiling. A suite that probed the live
endpoint would both exhaust it and make its own result depend on what nflverse published
that morning -- a test that fails because upstream re-released a season is a test nobody
can read. **No automated test in Phase 32 makes a network call.** Every detector test in
`tests/unit/test_revision_detector.py` reads these files.

## They are a snapshot, not a truth

`schedules.json` in particular records the fact `32-RESEARCH.md` Finding 5 measured: the
one monolithic `games.parquet` asset carries a single `updated_at` covering 1999-2026, so
a metadata probe over it reports every pinned schedules season as moved. That is why
`data/sealed_probe.py::SEALED_PROBE_STRATEGY` routes `schedules` to a CONTENT probe.

These files will drift from the live API as nflverse publishes. That is fine and expected:
their job is to be a FIXED input, so a detector test measures the detector rather than
upstream. Re-capture them only with a recorded reason, and expect the metadata values in
`tests/unit/test_revision_detector.py` to be read from the file rather than transcribed.

## They are committed despite `.gitignore`

`.gitignore:254` blankets `*.json`. These three files and this README are tracked by an
explicit `git add -f`, the same way `config/upstream_pin.json` is. Verify with:

```
git ls-files tests/fixtures/github_releases/
```
