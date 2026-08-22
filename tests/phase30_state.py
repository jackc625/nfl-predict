"""The git-TRACKED Phase-30 state manifest -- measured facts the phase's controls assert against.

WHY THIS MODULE EXISTS
----------------------
Phase 30's single most load-bearing proof is SPEC R2's fail-closed positive control for the
WR-06 leakage fix: after Plan 30-08's N-01 re-sync, every 2021-2024 gold feature value must be
byte-identical, and the re-sync must actually have re-synced something. That control is only a
control if it compares against a divergence measured BEFORE the re-sync -- afterwards the
number is unrecoverable -- and if it still asserts that number on a checkout that never ran the
phase.

The natural home for such a measurement is the run record, and Plan 30-04 Task 2 does write
one: ``outputs/n01/divergence_before.json`` and ``outputs/n01/odds_timeline_baseline.json``.
But ``outputs/`` is gitignored at ``.gitignore:26`` with only ``!outputs/.gitkeep`` at ``:27``
(``git check-ignore -v outputs/n01/divergence_before.json`` resolves to
``.gitignore:26:outputs/``), and the repository-wide ``*.json`` rule at ``.gitignore:250``
excludes them a second time. Those documents do NOT survive a fresh checkout.

For most generated state that is fine, and this repository has a deliberate precedent for
reading it behind a clean skip guard --
``tests/integration/test_activation_parity.py::_require_cache_and_backtest``, whose docstring
says it skips "so the suite is offseason / fresh-checkout safe". Those reads stay exactly as
they are.

The N-01 control is the one case where that pattern breaks, because the gitignored file is not
an INPUT to the proof -- it IS the proof's EXPECTED VALUE. Skip-guarding it would make Phase
30's most load-bearing assertion silently stop asserting anything on every checkout that did
not just run the phase, which is precisely the anti-pattern D30-06 rejects in Plan 30-03's own
words: a skipped test silently stops proving anything. Skip-guarding the control is not an
option; making its expected values DURABLE is.

``.planning/`` is gitignored too (``.gitignore:229``, and ``commit_docs`` is false), so a plan
SUMMARY is not a durable home either. The two homes that actually work in this repository are a
tracked module under ``tests/`` -- this file -- and the repo-root ``GATED-REFIT-READOUT.md``
that Plan 30-13 publishes. This module is the machine-readable half; the readout is the
human-readable half, and Plan 30-13 publishes the whole manifest into it.

The ``outputs/`` scratch documents are still written exactly as before. They are the run record
and carry the full per-season, per-week and per-id detail that does not belong in a constants
module. What changed is only that the committed controls no longer DEPEND on them: a control
reads the tracked constant here, and cross-checks the scratch document against it when the
scratch document happens to be present. A disagreement between the two is itself a failure --
a regenerated scratch file must never silently supersede the committed expectation (T-30-52).

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can import it
without cost or side effects. Every value below was MEASURED, never transcribed, and carries
the plan, task and date that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later plan that
needs a new durable value appends a new slot here rather than inventing a second home:

* Plan 30-04 (this file's author) -- ``N01_PARQUET_ROWS_BEFORE``, ``N01_DB_ROWS_BEFORE``,
  ``N01_DIVERGENCE_BEFORE``, ``N01_MISSING_GAME_IDS_SHA256``, ``ODDS_TIMELINE_ROWS``,
  ``ODDS_TIMELINE_PAIRS``, ``ODDS_TIMELINE_SEASON_COUNTS``,
  ``ODDS_TIMELINE_PAIR_LIST_SHA256``, ``GOLD_2025_ROWS_BEFORE_RESYNC``.
* Plan 30-08 -- ``SLICE_DIGESTS_2021_2024`` (the per-matrix, per-column 2021-2024 slice digests
  measured BEFORE the re-sync) and ``ACCEPTED_RUNG4_FINGERPRINT_SHA256``.
* Plan 30-10 -- ``PRE_REGISTRATION_COMMIT``, ``MEASUREMENT_COMMIT`` and
  ``GROUP_VERDICT_FILE_SHA256``.
* Plan 30-11 -- ``MANIFEST_SHA256_BEFORE`` and ``MANIFEST_SHA256_AFTER``.

Plan 30-13 publishes the whole manifest into ``GATED-REFIT-READOUT.md``, so every value has a
second, human-readable tracked home.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# N-01: the DuckDB mirror of silver `games` that fell behind its parquet.
#
# MEASURED by Plan 30-04 Task 2 on 2026-08-22, BEFORE Plan 30-08's re-sync, via
#   load_dataframe("games", layer="silver", source="parquet")   -> N01_PARQUET_ROWS_BEFORE
#   load_dataframe("games", layer="silver", source="db")        -> N01_DB_ROWS_BEFORE
# (note the source literal is "db"; "duckdb" raises ValueError at data/storage.py:970).
#
# The divergence is STRICTLY POSITIVE and the capture asserted so at capture time. A zero
# divergence would mean someone had already re-synced, leaving the Plan 30-08 positive control
# with nothing to prove -- a control that can be satisfied by doing nothing is not a control.
# ---------------------------------------------------------------------------
N01_PARQUET_ROWS_BEFORE: Final[int] = 6499
N01_DB_ROWS_BEFORE: Final[int] = 6292
N01_DIVERGENCE_BEFORE: Final[int] = 207

# sha256 over the newline-joined, sorted list of the game_ids present in the parquet copy and
# absent from the DuckDB copy. All 207 are season 2025. The full list lives in
# outputs/n01/divergence_before.json; this digest is what makes that list checkable after the
# scratch document is gone.
N01_MISSING_GAME_IDS_SHA256: Final[str] = (
    "40d871d75da81616744e95cb5fd82da77273f1e08c17f3cfa5e6f4c9a08bd9bd"
)

# Gold's 2025 slice is BEHIND BOTH copies of silver `games` -- gold carries weeks 1-4 while the
# DuckDB table has weeks 1-5 and the parquet has weeks 1-22. This is a FINDING, not a problem,
# and it is why SPEC R2's two clauses are about two DIFFERENT objects: gold's 2025 coverage is
# bounded by the other silver sources in the join, not by `games` alone. Plan 30-08's control
# must therefore NOT be written as "gold grew by 207" -- empirically it will not.
GOLD_2025_ROWS_BEFORE_RESYNC: Final[int] = 49
GOLD_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = (1, 2, 3, 4)
SILVER_PARQUET_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = tuple(range(1, 23))
SILVER_DB_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# The paid odds_timeline archive (T-30-12, SPEC R3).
#
# MEASURED by Plan 30-04 Task 2 on 2026-08-22 through
# load_dataframe("odds_timeline", layer="silver") -- deliberately NOT by reading the parquet
# path, because load_dataframe(source="auto") resolves DuckDB first and only falls back to
# parquet, so this is a statement about what the PIPELINE sees.
#
# 9,957 rows bought with real Odds API credits (Plan 29-05) and re-keyed in place by quick task
# 260816-u0e. Not re-derivable from anything in this repository.
#
# NOTE the integrity gate is asserted from CONTENT, never from `git status --porcelain data/`,
# which is vacuous: .gitignore:22 is `data/`, so porcelain returns empty whether the archive is
# intact or destroyed (N-03).
# ---------------------------------------------------------------------------
ODDS_TIMELINE_ROWS: Final[int] = 9957
ODDS_TIMELINE_PAIRS: Final[int] = 9957
ODDS_TIMELINE_SEASON_COUNTS: Final[dict[int, int]] = {
    2020: 1780,
    2021: 1219,
    2022: 1452,
    2023: 2759,
    2024: 2747,
}

# sha256 over the newline-joined, sorted "game_id\x1fsnapshot_ts" pair list. The counts can all
# match while the CONTENT has moved; this digest is what catches that.
ODDS_TIMELINE_PAIR_LIST_SHA256: Final[str] = (
    "c4e313a788799b8bfbae5b85169c190101684b155b44e3ea28c9b2c4a0e82a2f"
)
