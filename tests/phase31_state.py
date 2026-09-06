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


# ---------------------------------------------------------------------------
# CHECKPOINT 2 -- the owner's ACCEPTANCE of the Plan 31-11 rebuild
# (owner decision 2026-09-05; register entries DEF-31-09, DEF-31-10, DEF-31-11)
# ---------------------------------------------------------------------------
#
# Plan 31-11 stopped at CHECKPOINT 2, the blocking owner gate on the only write to
# ``data/`` in this phase. The automated HARD STOP had already fired: the ruling-A
# full-scope rebuild MOVED the protected 2021-2024 slice, which the pre-registration
# binds, so the executor refused to proceed and put the measured attribution to the owner
# as a blocking finding rather than accepting it.
#
# The owner ACCEPTED. This slot records that decision where it survives a fresh checkout,
# because ``.planning/`` is gitignored and the SUMMARY that carries the narrative does not
# travel. The obligations this acceptance places on ``PROFITABILITY-READOUT.md`` (Plan
# 31-19) are named in ``CHECKPOINT_2_READOUT_OBLIGATIONS`` below; a readout that discharges
# none of them is a readout that hid what the owner accepted in the open.
#
# NOTHING WAS RE-FROZEN. ``config/gate.toml`` is byte-unchanged -- git blob
# ``e56b7d08628ce86578f6d74921d584382cfdf825`` at the ``ee20773`` anchor, at HEAD and in
# the working tree. The owner explicitly declined option 3 (accept and re-freeze). The gate
# reads the COMMITTED baseline block rather than re-scoring, so the verdict is still judged
# against the frozen non-regression reference; the divergence between that block and what a
# re-score returns today is DISCLOSED (DEF-31-10), not erased.

CHECKPOINT_2_DECISION: Final[dict[str, str]] = {
    "plan": "31-11",
    "task": "Task 3: Gate baseline byte-identity proof, then CHECKPOINT 2 (rebuild acceptance)",
    "gate": "blocking-human",
    "question": "Is this rebuild acceptable?",
    "decision": "ACCEPT",
    "decided_on": "2026-09-05",
    "decided_by": "owner",
    "options_offered": (
        "1 accept; 2 reject and revert to a gold in which 68 protected-window games carry "
        "fabricated zero market lines; 3 accept and RE-FREEZE the gate baseline against the "
        "corrected gold. Option 1 was chosen. Option 3 was listed only so the option space "
        "was complete and was NOT taken."
    ),
    "rationale": (
        "The corrected labels are the true ones, and grading a holdout against fabricated "
        "zero lines was never defensible."
    ),
    "gate_toml_re_frozen": (
        "NO -- config/gate.toml is byte-unchanged and the baseline was not re-frozen"
    ),
    "accepted_scope": (
        "(a) the ruling-A full-scope rebuild from the upstream pin, which restored 2025's "
        "prior-season context; (b) the resulting move of the protected 2021-2024 slice, "
        "whose cause is the ratified clause-5 LAR to LA normalization reaching gold; (c) the "
        "consequent divergence between the frozen gate-baseline block and a re-score; and "
        "(d) the 2022 ATS residual sign flip, accepted AS A DISCLOSURE on the same basis as "
        "DEF-31-06."
    ),
    "still_pending_and_NOT_covered": (
        "The Phase-28 situational group flipping KEEP to DROP on the rebuilt gold is a "
        "SEPARATE ruling the owner has not made. "
        "tests/unit/test_signal_lift_readout_md.py::TestReadoutMatchesHarness::"
        "test_situational_ou_keep_ruling_reproduces_from_harness is left FAILING and "
        "untouched, deliberately, pending that ruling."
    ),
}

# The measured cause of the protected-slice move, as accepted. Every figure here was
# measured rather than inferred; the verification line was re-measured independently
# against live gold on 2026-09-05, after the acceptance.
PROTECTED_SLICE_CORRECTION: Final[dict[str, str]] = {
    "register_entry": "DEF-31-09",
    "cause": (
        "the ratified clause-5 normalize_stored_game_ids LAR to LA re-keying of 116 stored "
        "odds rows reaching gold for the first time"
    ),
    "mechanism": (
        "gold keys the Rams canonically as LA; silver stored 116 of their odds rows as LAR. "
        "Those rows were ORPHANS against gold, so those games fell through to "
        "_default_compressed_market_features and gold's imputation wrote a literal 0.0 "
        "market line. scripts/build_features.py then computed target_ats = "
        "point_differential - snapshot_spread and target_ou = total_points - snapshot_total "
        "against that fabricated zero, producing WRONG labels."
    ),
    "games_corrected_in_protected_window": "68 -- 17 per season, 2021 through 2024",
    "games_corrected_all_seasons": (
        "136, every one of them a Rams game: 16/16/16 (2018-2020), 17/17/17/17 (2021-2024), "
        "20 (2025)"
    ),
    "column_slots_moved": (
        "11 -- snapshot_spread, snapshot_total and snapshot_ml_prob_home_fair in all three "
        "matrices, plus target_ats in features_ats and target_ou in features_ou"
    ),
    "what_they_were_graded_against_before": (
        "a fabricated 0.0 market line, and therefore an ATS and an O/U label derived from it"
    ),
    "verification_2026_09_05": (
        "live data/gold/features_ats.parquet: 75 LA-involved rows in 2021-2024 "
        "(21/17/18/19), ZERO of them at snapshot_spread == 0.0; ZERO LAR-keyed rows remain "
        "in gold; only 2 rows in the entire 1,139-row protected window are still at 0.0 and "
        "neither is a Rams game (2024_W01_JAX@MIA, 2024_W01_CAR@NO)"
    ),
    "corroborating_instrument": (
        "tests/integration/test_n01_resync_control.py, a Phase-30 control this phase did not "
        "write, independently reports 20 moved columns in 2021-2024 for features_ats where "
        "it reported 16; the four additions are exactly snapshot_spread, snapshot_total, "
        "snapshot_ml_prob_home_fair and target_ats"
    ),
    "per_season_row_counts_unchanged": (
        "2021=285, 2022=284, 2023=285, 2024=285 across rungs 0, 1, 2 and 3"
    ),
}

# The gate baseline diverges from a re-score, and that divergence is DISCLOSED rather than
# re-frozen. This is the distinction the readout has to carry: the committed block is the
# rule the gate reads, and a re-score is a measurement of today's gold, not a correction to
# the rule.
GATE_BASELINE_DIVERGENCE: Final[dict[str, str]] = {
    "register_entry": "DEF-31-10",
    "fields_differing": (
        "47 of 68, with 21 identical; it was 44 of 68 before the ruling-A full rebuild"
    ),
    "population_unchanged": (
        "all twelve per-season sample sizes are IDENTICAL -- ats/ou/wp 2021=272, 2022=271, "
        "2023=272, 2024=272 -- so the scored population did not move, only the values"
    ),
    "pooled_means": (
        "ats.pooled.mean committed -0.00149507 against regenerated +0.01362779; "
        "ou.pooled.mean committed 1.09908061 against regenerated 1.09091962; "
        "wp.pooled.mean committed -0.03800034 against regenerated -0.03892902"
    ),
    "why_this_is_not_a_re_freeze": (
        "the gate reads the COMMITTED baseline block, never a re-score, so the frozen block "
        "remains the non-regression reference the verdict is judged against. Re-freezing "
        "would move the reference to match the data being judged, which is the one thing a "
        "non-regression baseline must never do. config/gate.toml is byte-unchanged, the "
        "holdout stays exactly 2021-2024, and 2025 is absent from it."
    ),
    "standing_red_controls": (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::"
        "test_the_generated_block_equals_the_committed_block_byte_for_byte and "
        "tests/unit/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields "
        "are EXPECTED red and are deliberately not made green"
    ),
}

# What the acceptance obliges PROFITABILITY-READOUT.md (Plan 31-19) to state. These are
# ADDITIVE to the six-point obligation already recorded in ATS_RESIDUAL_DRIFT_PROVENANCE
# and DEF-31-06; none of them replaces it.
CHECKPOINT_2_READOUT_OBLIGATIONS: Final[tuple[str, ...]] = (
    "State that 68 games inside the protected 2021-2024 holdout were previously graded "
    "against a FABRICATED zero market line and now carry the real one -- naming what they "
    "were graded against before, not merely that a correction happened (DEF-31-09).",
    "State the 2022 ATS residual SIGN FLIP against the pre-registration's own prose: the "
    "ratified 2022 mean is negative (-0.03539119799896865) and the document says so in as "
    "many words, while live gold returns +0.046538869338765949 and "
    "seasons_with_negative_mean == [] where it returned [2022]. Disclose it; do not "
    "re-derive it, and do not quietly drop the sentence (DEF-31-06, DEF-31-11).",
    "State that the frozen gate-baseline block diverges from a re-score in 47 of 68 fields "
    "with every per-season n unchanged, and WHY that is a disclosure rather than a "
    "re-freeze: the gate reads the committed block, and moving the reference to match the "
    "data it judges would destroy the non-regression property (DEF-31-10).",
    "State that the owner ACCEPTED this at CHECKPOINT 2 on 2026-09-05, with the rationale "
    "recorded, so a reader knows the divergence was ruled on rather than discovered later.",
)


# ---------------------------------------------------------------------------
# The 2025 MEASUREMENT COMMIT and the verdict artifact's digest (Plan 31-14 Task 2).
#
# APPENDED on 2026-09-05, in a SEPARATE, LATER commit than the verdict artifact itself. The
# separation is not bookkeeping: a commit hash is a FUNCTION of the committed bytes, so an
# artifact that carried its own commit hash would have no fixed point and the assertion built on
# it could never be satisfied (REVIEW-CIRCULAR). The witness therefore lives OUTSIDE the artifact
# it witnesses, exactly as PRE_REGISTRATION_COMMIT does above and exactly as
# tests/phase30_state.py resolved the same problem for Phase 30.
#
# APPENDING AFTER THE MEASUREMENT IS NOT A POST-HOC EDIT OF THE RULE. This module records FACTS
# ABOUT the frozen artifacts and never changes them: PROFITABILITY-PREREGISTRATION.md and
# backtest/ev_chain_constants.py are byte-unchanged from the ee20773 anchor, and the ancestry
# relation the assertion rests on holds either way -- the rule commit strictly precedes the
# measurement commit, and the two are different commits. What could not exist before the run is
# the measurement commit itself; nothing about the rule was decided here.
#
# THE RELATION, reproducible from any non-shallow checkout. Four claims, all verified at append
# time: `git merge-base --is-ancestor <PRE_REGISTRATION_COMMIT> <MEASUREMENT_COMMIT>` exits 0;
# the two SHAs are NOT equal; the measurement SHA below equals what
# `git log -1 --format=%H -- config/profitability_2025_verdict.toml` resolves; and the digest
# below equals the sha256 of that file's committed bytes.
#
# WHAT THE MEASUREMENT COMMIT CONTAINS. Exactly two paths -- the verdict artifact and the
# COMPLETED run ledger -- so the record of what was spent travels in the same commit as what was
# measured, and neither can be produced without the other.
# ---------------------------------------------------------------------------
MEASUREMENT_COMMIT: Final[str] = "01b246468f2f330c35d54e051be248f0f8994376"

# sha256 of config/profitability_2025_verdict.toml's NEWLINE-NORMALIZED bytes -- every CRLF
# folded to LF before hashing, on the same basis as PRE_REGISTRATION_FILE_SHA256 above and for
# the same reason: core.autocrlf=true with no .gitattributes means a raw working-tree digest
# would hold only on the machine that measured it. Normalized, this equals the sha256 of
# `git cat-file blob <MEASUREMENT_COMMIT>:config/profitability_2025_verdict.toml`, and both
# routes were computed and compared equal at append time.
VERDICT_FILE_SHA256: Final[str] = (
    "4befbcb73dd5697e75052f49d4d97091d1085dac74af794e1a4ec9916c9ead2e"
)

VERDICT_PATH: Final[str] = "config/profitability_2025_verdict.toml"
RUN_LEDGER_COMMITTED_PATH: Final[str] = "config/profitability_2025_run_ledger.toml"

MEASUREMENT_PROVENANCE: Final[dict[str, str]] = {
    "plan": "31-14",
    "task": "Task 2: CHECKPOINT 3 -- arm the one-shot 2025 run",
    "date": "2026-09-05",
    "checkpoint_3_decision": "ARM",
    "attempt_id": "3f1bcdc127434540a656f4ae554df398",
    "prior_armed_attempt_id": "24edd3860ce14114b91c576bbf61dcbc",
    "hold_seasons": "2025 ONLY -- 285 games, playoffs included",
    "tune_seasons": "2021-2024, with 2018-2020 as the strictly-prior bias seed",
    "resolved_by": "git log -1 --format=%H -- config/profitability_2025_verdict.toml",
    "hash_basis": "newline-normalized file bytes (CRLF folded to LF); equals the git blob sha256",
    "commit_contents": (
        "exactly the verdict artifact and the COMPLETED run ledger, and nothing else"
    ),
    "tokens": "wp INCONCLUSIVE_CLEAN, ats UNPROFITABLE_CLEAN, ou INCONCLUSIVE_CLEAN",
    "bh_denominator_used": "6 -- 3 primary plus 3 robustness cuts; NO calibration fallback fired",
    "clv_disposition": "REPORT-ONLY; carried beside the verdict, absent from the family",
}


# ---------------------------------------------------------------------------
# The READOUT ACCEPTANCE -- the owner's ruling on PROFITABILITY-READOUT.md
# (owner decision 2026-09-06; Plan 31-19 Task 3, gate blocking-human)
# ---------------------------------------------------------------------------
#
# This is NOT a fourth numbered checkpoint. D31-16 fixes the phase's blocking
# checkpoint count at THREE -- pre-registration ratification (31-05), rebuild
# acceptance (31-11) and arming the one-shot run (31-14) -- and this slot does
# not add to that count. It is the plan-level human-verify gate on Plan 31-19
# Task 3, which asks the ONE question the drift guard cannot: the guard pins
# structure, verdict tokens, prior-document content hashes and every 2025
# figure, all mechanical, but whether the prose is free of hype is read by a
# person.
#
# It is recorded HERE, in the tracked tree, for the same reason CHECKPOINT 2's
# acceptance was (commit b480c3a): ``.planning/`` is gitignored, so
# ``31-19-SUMMARY.md`` does not survive a fresh checkout. An acceptance that
# lives only in the gitignored planning tree is an acceptance a later reader
# cannot find.
#
# NOTHING WAS DEPLOYED BY THE PLAN THAT TOOK THIS ACCEPTANCE. Plan 31-19
# changed twelve tracked paths, all of them documents and their guards;
# ``artifacts/latest.json`` is sha256
# 7ff78a506b1cb06e206705c5900438a5388be64e963bcc8a39c7ed6a8d0f66f1, unchanged,
# and every frozen artifact named in the plan is byte-identical to its state
# before the plan began.

READOUT_ACCEPTANCE: Final[dict[str, str]] = {
    "plan": "31-19",
    "task": (
        "Task 3: Reconcile the remaining documents, run the full suite, and "
        "take the readout acceptance"
    ),
    "gate": "blocking-human",
    "document": "PROFITABILITY-READOUT.md",
    "question": "Is this framing honest?",
    "decision": "ACCEPT",
    "decided_on": "2026-09-06",
    "decided_by": "owner",
    # VERBATIM. The owner's words, unedited and unsummarised.
    "verbatim": (
        "ACCEPT. The readout's framing is honest and free of hype. It states "
        "the result before anything else, refuses to let the win-probability "
        "closing-line finding soften the spread target's measured loss, "
        "explains why five tests are deliberately red rather than hiding "
        "them, and closes by stating plainly that nothing in it establishes "
        "an edge that survives its own significance test."
    ),
    # What an independent read of the document confirmed BEFORE the owner
    # ruled. Recorded as corroboration of the acceptance, not as a substitute
    # for it -- the judgment is the owner's.
    "independent_review_findings": (
        'the opening states "No target is `PROFITABLE_CLEAN`" before any '
        "other content; section 0a exists to prevent the CLV/profitability "
        "conflation; section 2 states the divergence as a finding while "
        "explicitly refusing to let it soften the ATS loss; section 8 "
        "explains the five reds as tripwires that fired on accepted events; "
        'section 10 ends with "It is not a statement that the system should '
        'be bet." A scan for hype language found every occurrence of '
        '"profitable" to be either a negation or a token name.'
    ),
    "document_unchanged_by_the_acceptance": (
        "YES -- the owner accepted the document AS WRITTEN. "
        "PROFITABILITY-READOUT.md was not restructured, and "
        "tests/unit/test_profitability_readout_md.py pins it."
    ),
}

# The ONE prohibition the requirements themselves rule JUDGMENT-TIER, carried
# forward verbatim from 31-SPEC.md and recorded as the owner's ATTESTATION.
#
# It is recorded as an attestation and NOT as a verified claim, deliberately.
# Two of the three clauses are checkable and ARE checked: the commit-order
# assertion and the content-hash lock on the pre-registration
# (PRE_REGISTRATION_COMMIT / PRE_REGISTRATION_FILE_SHA256 above, and
# MEASUREMENT_COMMIT's ancestry relation). The third clause -- that no
# threshold was tuned after seeing the 2025 results -- is about INTENT, and no
# test can prove intent. Claiming it as tested would be exactly the kind of
# overstatement this milestone's readout exists to refuse.
READOUT_JUDGMENT_TIER_ATTESTATION: Final[dict[str, str]] = {
    "source": "31-SPEC.md prohibition table, row R3",
    "tier": "judgment (owner-ruled), NOT test",
    # VERBATIM, from the SPEC's prohibition table.
    "prohibition_verbatim": (
        "MUST NOT tune any threshold after seeing 2025 results, re-run the "
        "single-use split, or publish a 2025 verdict without its "
        "pre-registration committed first"
    ),
    # VERBATIM, from the same row's verification column.
    "why_judgment_tier_verbatim": (
        'the "did not tune after seeing results" clause is judgment-tier, '
        "since no test can prove intent (owner-ruled)"
    ),
    "attested_by": "owner",
    "attested_on": "2026-09-06",
    "attestation": (
        "No threshold was tuned after seeing the 2025 results. This is the "
        "owner's attestation, recorded as such; it is NOT claimed as tested."
    ),
    "what_IS_tested_alongside_it": (
        "the other two clauses of the same prohibition: the pre-registration "
        "commit strictly precedes the measurement commit and the two are "
        "distinct (tests/unit/test_preregistration_ancestry.py), and "
        "PROFITABILITY-PREREGISTRATION.md is content-hash locked to "
        "PRE_REGISTRATION_FILE_SHA256. The single-use split was spent exactly "
        "once, on the attempt id recorded in MEASUREMENT_PROVENANCE."
    ),
}
