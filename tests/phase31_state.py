"""The git-TRACKED Phase-31 state manifest -- measured facts the phase's controls assert against.

WHY THIS MODULE EXISTS
----------------------
Phase 31 spends the single unburned 2025 season exactly once, under a pre-registration frozen
in Plan 31-05. Several of the numbers that pre-registration binds to are MEASURED in Wave 1 and
then have to stay checkable for the rest of the phase and after it -- otherwise a later
disagreement degenerates into an argument about which number was right, which is precisely what
a one-shot verdict cannot survive.

The natural home for such a measurement is the run record, and Plan 31-02 does write one:
``outputs/p31/odds_store_audit.json``. But ``outputs/`` is gitignored (``.gitignore:26``, with
only ``!outputs/.gitkeep`` at ``:27``) and the repository-wide ``*.json`` rule excludes it a
second time, so that document does NOT survive a fresh checkout. ``.planning/`` is gitignored
too (``commit_docs`` is false), so a plan SUMMARY is not a durable home either.

This module is the same answer Phase 30 reached in ``tests/phase30_state.py``, applied to Phase
31: a tracked constants module under ``tests/`` is the machine-readable durable home, and the
repo-root readout the phase publishes is the human-readable one. The ``outputs/`` scratch
document is still written exactly as before -- it carries the per-file and per-partition detail
that does not belong in a constants module. What changed is only that the committed controls no
longer DEPEND on it: a control reads the tracked constant here and treats a disagreement with a
regenerated scratch document as a FAILURE, never as a silent supersession.

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can import it
without cost or side effects. Every value below was MEASURED, never transcribed, and carries the
plan, task, date and scored artifact id that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later plan that
needs a new durable value appends a new slot here rather than inventing a second home:

* Plan 31-02 (this file's author) -- ``ATS_RESIDUAL_BY_SEASON`` and ``ATS_RESIDUAL_POOLED``,
  with their ``*_PROVENANCE`` siblings.
* Plan 31-05 -- ``PRE_REGISTRATION_COMMIT`` and ``PRE_REGISTRATION_FILE_SHA256``, the
  git-ancestry anchor for the frozen pre-registration.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# The ATS residual bias, RE-DERIVED from the DEPLOYED artifact (Plan 31-02 Task 2).
#
# MEASURED by scoring the deployed ATS artifact through
# ``backtest.diagnose.score_deployed_artifacts("ats")`` over canonical gold, under the contract
#   residual = actual home margin - predicted home spread
# and NOT by reading ``data/web_cache.duckdb``. RESEARCH measured the pooled figure from that
# cache and flagged it as assumption A1 precisely because the cache can be stale relative to the
# deployed artifact. It IS stale: the cache reports a pooled mean of +0.714049 (sd 13.021757)
# with 2022 at +0.118713, and the re-score reports the values below with 2022 NEGATIVE. The
# cache figures MUST NOT be used.
#
# The 2021-2024 window is INTERNAL to ``_load_gold_holdout`` via ``HOLDOUT_FIRST_SEASON`` /
# ``HOLDOUT_LAST_SEASON``; the measuring script ASSERTS those constants are 2021/2024 rather
# than assuming it, so a future change to them surfaces as a failure instead of silently
# widening the window these numbers are measured over.
#
# THE DIRECTION GUARD IS ON THE POOLED SIGN ONLY (REVIEW-ATS). 2022 is negative and that is
# simply true; a per-season sign gate would hard-stop the phase on a real fact, and a magnitude
# TOLERANCE chosen after seeing these values would be the post-hoc threshold selection this
# phase forbids everywhere else. The per-season means are recorded here in full, so a reader of
# the pre-registration sees the negative season rather than a gate that hid it.
# ---------------------------------------------------------------------------

# Field order of every tuple in ATS_RESIDUAL_BY_SEASON and of ATS_RESIDUAL_POOLED.
ATS_RESIDUAL_FIELDS: Final[tuple[str, ...]] = ("n", "mean", "sd", "t", "p")

ATS_RESIDUAL_BY_SEASON: Final[dict[int, tuple[int, float, float, float, float]]] = {
    2021: (
        285,
        0.68147702779163399,
        11.054154663839386,
        1.040754060336891,
        0.29887489269038414,
    ),
    2022: (
        284,
        -0.03539119799896865,
        9.7166175756444702,
        -0.061381758141627787,
        0.95109850647921679,
    ),
    2023: (
        285,
        0.58219338768864415,
        10.57502548958826,
        0.92941200047286621,
        0.35346489722496455,
    ),
    2024: (
        285,
        1.1270376943443952,
        12.856635704530254,
        1.479903963231449,
        0.14000729531088796,
    ),
}

ATS_RESIDUAL_BY_SEASON_PROVENANCE: Final[dict[str, str]] = {
    "plan": "31-02",
    "task": "Task 2: Re-derive the ATS residual bias from the deployed artifact (closes A1)",
    "date": "2026-09-03",
    "artifact_id": "ats_20260605_220128",
    "measured_by": "scripts.audit_odds_preingest.measure_ats_residual_bias",
    "source": "backtest.diagnose.score_deployed_artifacts('ats') over canonical gold",
    "float_format": "{:.17g}",
}

ATS_RESIDUAL_POOLED: Final[tuple[int, float, float, float, float]] = (
    1139,
    0.58937727047262922,
    11.104272279463643,
    1.7912868714554835,
    0.073512927737913666,
)

ATS_RESIDUAL_POOLED_PROVENANCE: Final[dict[str, str]] = {
    "plan": "31-02",
    "task": "Task 2: Re-derive the ATS residual bias from the deployed artifact (closes A1)",
    "date": "2026-09-03",
    "artifact_id": "ats_20260605_220128",
    "measured_by": "scripts.audit_odds_preingest.measure_ats_residual_bias",
    "source": "backtest.diagnose.score_deployed_artifacts('ats') over canonical gold",
    "float_format": "{:.17g}",
    "direction_claim": "pooled mean residual is strictly positive (model under-predicts)",
    "per_season_sign_asserted": "no -- 2022 is negative and is REPORTED, never gated against",
}

# Phi(pooled_mean / pooled_sd) - 0.5, against the flat -110 breakeven edge (110/210 - 0.5).
# The bias is ~89 percent of the entire house edge, which is the magnitude argument the
# pre-registration makes -- and the reason omitting the correction is not a neutral
# simplification.
ATS_IMPLIED_COVER_PROBABILITY_SHIFT: Final[float] = 0.021164571224560169
ATS_MINUS_110_BREAKEVEN_EDGE: Final[float] = 0.023809523809523836


# ---------------------------------------------------------------------------
# The git-ancestry anchor for the FROZEN Phase-31 pre-registration (Plan 31-05 Task 1).
#
# APPENDED on 2026-09-04, in a SEPARATE, LATER commit than the pre-registration itself. That
# separation is the whole point.
#
# WHY THE ANCHOR LIVES HERE AND NOT INSIDE THE FILES IT WITNESSES (REVIEW-CIRCULAR). A document
# that must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential: writing the
# hash changes the bytes the hash was computed over, so no fixed point exists without a canonical
# exclusion rule nobody has defined. A test written against such a file would have to be relaxed
# into meaninglessness. So the witness lives OUTSIDE the witnessed files, exactly as Phase 30
# proved with PRE_REGISTRATION_COMMIT / MEASUREMENT_COMMIT / GROUP_VERDICT_FILE_SHA256 in
# tests/phase30_state.py. backtest/ev_chain_constants.py declares PREREGISTRATION_PATHS so the
# resolving code names the paths from ONE place and hardcodes no hash of its own.
#
# WHAT PRE_REGISTRATION_COMMIT IS. The last commit to touch EITHER pre-registration path, resolved
# by `git log -1 --format=%H -- PROFITABILITY-PREREGISTRATION.md backtest/ev_chain_constants.py`.
# That commit contains EXACTLY those two paths and nothing else, so its message's claim to be the
# pre-registration is checkable rather than merely asserted. The two files are ONE rule in two
# artifacts -- a constants module the code reads and a prose document the owner ratifies -- which
# is why the anchor is their COMBINED last-modifying commit rather than either file's alone.
#
# THE RELATION PLAN 31-14 ASSERTS, reproducible from any non-shallow checkout:
# `git merge-base --is-ancestor <PRE_REGISTRATION_COMMIT> <the 2025 measurement commit>` exits 0,
# and the two SHAs are NOT equal. The inequality is not pedantry. A rule and the results it
# produced landing in one commit is not a pre-registration; it is only a claim of one.
# ---------------------------------------------------------------------------
PRE_REGISTRATION_COMMIT: Final[str] = "ee20773b58c3a59de2450d56c64992e240282820"

# sha256 of each pre-registration file's NEWLINE-NORMALIZED bytes -- every CRLF folded to LF
# before hashing.
#
# The normalization is load-bearing, not tidiness. This repository has core.autocrlf=true and no
# .gitattributes, so these files are LF in the git blob and CRLF in a fresh Windows working tree.
# A digest over raw working-tree bytes would pin a value that holds on the machine that measured
# it and fails on every other checkout -- the exact opposite of what a tracked constant is for.
# Normalized, each value equals the sha256 of
# `git cat-file blob <PRE_REGISTRATION_COMMIT>:<path>` and is reproducible on any platform. Both
# were verified equal by both routes at append time.
#
# Keys are repo-root-relative POSIX paths and MUST match
# backtest.ev_chain_constants.PREREGISTRATION_PATHS exactly; the ancestry test asserts that the
# two sets agree, so a path added to the rule without a hash appended here is a failure rather
# than a silently unwitnessed file.
PRE_REGISTRATION_FILE_SHA256: Final[dict[str, str]] = {
    "PROFITABILITY-PREREGISTRATION.md": (
        "5e788d646f6948e903b7769bbad55d9a09002ea9e1a6c7d569d5e5e5203a149c"
    ),
    "backtest/ev_chain_constants.py": (
        "a80bf5c1c9ac558cb9074e40316115c773f7352ad2a5bbfa527ec43d06f78220"
    ),
}

PRE_REGISTRATION_PROVENANCE: Final[dict[str, str]] = {
    "plan": "31-05",
    "task": "Task 1: The FROZEN pre-registration -- constants module plus document, one commit",
    "date": "2026-09-04",
    "resolved_by": (
        "git log -1 --format=%H -- PROFITABILITY-PREREGISTRATION.md "
        "backtest/ev_chain_constants.py"
    ),
    "hash_basis": "newline-normalized file bytes (CRLF folded to LF); equals the git blob sha256",
    "commit_contents": "exactly the two pre-registration paths and nothing else",
}


# ---------------------------------------------------------------------------
# THE ATS RESIDUAL CONSTANT DRIFT -- a DISCLOSURE record, not a re-freeze
# (owner ruling B, 2026-09-05; register entry DEF-31-06)
# ---------------------------------------------------------------------------
#
# ``ATS_RESIDUAL_BY_SEASON`` and ``ATS_RESIDUAL_POOLED`` above are UNCHANGED and stay
# unchanged. They are the ratified rule, carried verbatim into
# ``PROFITABILITY-PREREGISTRATION.md``, and ``backtest/ev_chain_constants.py`` reads the
# frozen values -- so the Phase-31 verdict is computed with the numbers the owner ratified,
# whatever gold says today. That is what pre-registration MEANS.
#
# WHAT WAS LOST IS THE RE-DERIVABILITY OF THOSE NUMBERS, NOT THE NUMBERS THEMSELVES.
# Re-scoring the same deployed artifact ``ats_20260605_220128`` over gold no longer returns
# them. Every per-season sample size ``n`` is UNCHANGED (285/284/285/285, pooled 1139), so
# the population is identical; only the values moved. The owner ruled on 2026-09-05 that
# this is a DISCLOSURE item rather than a tampering one, and that it must be stated in
# ``PROFITABILITY-READOUT.md`` (Plan 31-19) with BOTH value sets and the reason they differ.
#
# THE DRIFT HAS TWO SEPARATELY-ATTRIBUTABLE CAUSES, and conflating them would misreport it:
#
#   1. UPSTREAM. nflverse re-released play-by-play and depth-chart data between the
#      2026-09-03 measurement and 2026-09-05. Sixteen gold columns moved -- the twelve
#      ``{home,away}_{off,def}_rolling_opp_adj_*``, ``{home,away}_qb_adjustment`` and
#      ``{home,away}_backup_quality_delta`` -- as the Phase-30 control
#      ``tests/integration/test_n01_resync_control.py`` independently reports. Plan 31-11's
#      upstream pin (``data/upstream_pin.py``) stops this recurring; it cannot undo it,
#      because the 2026-08-22 revision the constants were measured on is gone from upstream.
#      This cause was present BEFORE this plan wrote anything.
#
#   2. THE RATIFIED CLAUSE-5 KEY NORMALIZATION REACHING GOLD. ``normalize_stored_game_ids``
#      re-keyed 116 stored ``LAR`` Rams rows to the canonical ``LA``. Those rows had been
#      ORPHANS against gold, so 68 games in the protected 2021-2024 window carried a
#      FABRICATED zero market line and therefore a wrong ``target_ats``. The 2026-09-05
#      full-scope rebuild is the first build to re-derive those seasons since, so it is the
#      build in which the correction reached gold. This cause is a data CORRECTION, and it
#      arrived under the owner's own ruling A.
#
# The three columns below are recorded so a reader can see WHICH cause moved WHICH number.
# No assertion is written against ``ATS_RESIDUAL_LIVE_*``: pinning a drifting value as a
# green test would convert a disclosure into a moving target, and the ratified constants
# stay asserted (as expected failures) in
# ``tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold``
# so the drift stays visible and measurable in the terminal summary.

# Measured 2026-09-05, BEFORE the ruling-A full rebuild: upstream cause alone.
ATS_RESIDUAL_LIVE_UPSTREAM_ONLY: Final[dict[str, float]] = {
    "2021": 0.68443905065457022,
    "2022": -0.03170638450119697,
    "2023": 0.59053647537437970,
    "2024": 1.12758861518742750,
    "pooled": 0.59326265763681096,
}

# Measured 2026-09-05, AFTER the ruling-A full-scope rebuild: upstream PLUS the clause-5
# correction reaching the 2021-2024 slice. This is the figure a re-score returns today.
ATS_RESIDUAL_LIVE_AFTER_FULL_REBUILD: Final[dict[str, float]] = {
    "2021": 0.63968358671194625,
    "2022": 0.046538869338765949,
    "2023": 0.62408175513867226,
    "2024": 1.1071011105258213,
    "pooled": 0.60484106920061009,
}

ATS_RESIDUAL_DRIFT_PROVENANCE: Final[dict[str, str]] = {
    "register_entry": "DEF-31-06",
    "ruled_by": "owner",
    "ruled_on": "2026-09-05",
    "ruling": (
        "ACCEPTED AS A DISCLOSED FACT. Do not re-derive, do not re-freeze. The frozen "
        "constants ARE the rule and the chain reads them from the frozen module."
    ),
    "artifact_id": "ats_20260605_220128",
    "measured_by": "scripts.audit_odds_preingest.measure_ats_residual_bias",
    "population_unchanged": "per-season n 285/284/285/285, pooled n 1139 -- identical on all three",
    "readout_obligation": (
        "PROFITABILITY-READOUT.md (Plan 31-19) MUST state BOTH the ratified set and the "
        "live set, and the reason they differ."
    ),
    "sign_change_to_disclose": (
        "the ratified 2022 season mean is NEGATIVE (-0.03539119799896865) and the "
        "pre-registration's prose says so explicitly; after the full rebuild the live 2022 "
        "mean is POSITIVE (+0.046538869338765949), so measure_ats_residual_bias now reports "
        "seasons_with_negative_mean == [] where it reported [2022]. The pooled direction "
        "claim (strictly positive) still HOLDS and is in fact stronger."
    ),
}
