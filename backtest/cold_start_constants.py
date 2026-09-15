"""The FROZEN Phase-33 cold-start pre-registration (COLD-07, CLEAN-01, D33-19/D33-20).

GENERATOR OUTPUT. Emitted by ``python -m scripts.derive_cold_start_constants`` from the
END-STATE artifacts, in ONE operation (D24-07). Do NOT hand-edit any value below.

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py`` (D31-05). This
module IS the Phase-33 cold-start pre-registration, not a description of one. Its
LAST-MODIFYING COMMIT is the git-ancestry anchor: the ancestry guard requires that commit to be
a strict git ANCESTOR of the commit recording the Phase-33 readout.

Stated plainly because it is easy to forget two plans later: EDITING THIS FILE AFTER THE ANCHOR
COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. There is no honest repair path. A value
that is wrong here is wrong for the remainder of the phase, and the only legitimate response is
a NEW, VISIBLY-LATER CORRECTIVE COMMIT that explicitly INVALIDATES this pre-registration by
naming its commit sha. Never an edit in place, and never a quiet supersession.

IT IS ONE PRE-REGISTRATION IN TWO FILES. ``COLD-START-PREREGISTRATION.md`` at the repo root is
the human-readable half the owner ratifies. Their COMBINED last-modifying commit is the anchor;
``PREREGISTRATION_PATHS`` below names both so every guard resolves them from one place.

NEITHER FILE RECORDS ITS OWN CONTENT HASH. A document that must CONTAIN and exactly REPRODUCE
its own whole-file hash is self-referential: writing the hash changes the bytes the hash was
computed over, so no fixed point exists without a canonical exclusion rule nobody has defined.
The witness lives OUTSIDE the witnessed files: ``tests/phase33_state.py`` records
``PRE_REGISTRATION_COMMIT``, ``PRE_REGISTRATION_FILE_SHA256`` and
``PRE_REGISTRATION_AUTHOR_DATE`` in a LATER commit under its APPEND PROTOCOL, and
``tests/unit/test_phase33_preregistration_ancestry.py`` recomputes and compares.

THE ANCHORS, AND WHAT EACH ONE CAN ACTUALLY CARRY
--------------------------------------------------
(i) GIT ANCESTRY against the committed Phase-33 readout commit. This proves ordering INSIDE the
repository and nothing about wall-clock time.

(ii) AN EXTERNAL TIME ANCHOR -- a signed tag pushed to the remote, a remote push receipt, or a
CI attestation. Only something produced by a system OTHER than this working tree can establish
that the rule existed before kickoff, which is the property a pre-registration exists to have.
Which kind was obtained, or that none was, is recorded in
``tests.phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND``.

(iii) THE COMMIT'S AUTHOR DATE, against 2026-09-17T20:15:00-04:00. This is CORROBORATION ONLY.
A git author date is LOCALLY SETTABLE -- ``GIT_AUTHOR_DATE`` and ``git commit --date`` both set
it -- so it cannot prove pre-kickoff existence on its own. It is asserted because a recorded
date that disagrees with the record is still a finding, not because it carries the claim.

THE TWO QUANTITIES, AND THE ONE PROPERTY THEY SHARE
-----------------------------------------------------
They live in ONE module because they share the single property that defines them: FROZEN BEFORE
WEEK 2, NEVER RECOMPUTED IN-SEASON. Splitting them would give the phase two anchors to keep
consistent instead of one.

THE DERIVATION IS MILDLY CIRCULAR, AND IT IS STATED RATHER THAN DISCOVERED (D33-20)
------------------------------------------------------------------------------------
The thresholds are derived on the pinned 2021-2024
backtest population, and that is ALSO the population the label movement is measured against.
Measuring the movement on a population the thresholds were not derived from would trade a
stated caveat for an unstated mismatch, so the circularity is kept and named here.

THE BIAS POOL IS PARTLY IN-SAMPLE, AND THAT IS ALSO STATED
------------------------------------------------------------
The deployed artifacts were fitted through the final-fit entry point over every completed
season 2002-2025, which INCLUDES the 2021-2025 residual pool below.
So these residuals are in-sample and the bias they produce is ATTENUATED -- the real
out-of-sample bias is likely larger in magnitude. That direction is the conservative one to
know about and it is recorded rather than left to be inferred.

THE EMPTY-POOL REFUSAL (R10 edge)
-----------------------------------
An EMPTY strictly-prior residual pool REFUSES BY NAME. No bias is invented and there is NO
fallback to the target season's own data. The refusal is
``scripts.derive_cold_start_constants.EmptyResidualPoolError``, wrapping the existing
``backtest.ou_ev_chain.estimate_prior_season_bias`` refusal.

JSON ROUND-TRIPS SEASON KEYS AS STRINGS WHILE THE LOOKUP IS BY INT
-------------------------------------------------------------------
Every season-keyed mapping here uses INT keys in Python. A JSON-facing representation of the
same mapping carries STRING keys, because JSON has no integer keys at all. A consumer that
serializes and reloads one of these mappings must therefore look up ``"2026"`` and not ``2026``
-- stated here so the mismatch is a known conversion rather than a silent miss.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# THE TWO FILES THAT ARE THE PRE-REGISTRATION. Named here so every guard resolves them from ONE
# place: the ancestry test's `git log -1 --format=%H -- <paths>`, the content-hash comparison
# against tests/phase33_state.PRE_REGISTRATION_FILE_SHA256, and the Plan 33-18 assertion that
# this commit strictly precedes the readout commit. Repo-root-relative POSIX paths.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "COLD-START-PREREGISTRATION.md",
    "backtest/cold_start_constants.py",
)

# THE SIX EDGE THRESHOLDS, ONE PAIR PER TARGET, as (HIGH, MEDIUM) and on EACH TARGET'S OWN
# UNIT. Stated to FOUR DECIMAL PLACES in the committed text, which is the pre-registered
# precision and not merely an upper bound on it.
#
#   ats  POINTS          -- signed model-minus-market home margin (R13 / D33-05, Plan 33-10)
#   ou   RATIO           -- (model total - market total) / max(market total, 30)
#   wp   PROBABILITY     -- model probability minus the devigged fair closing probability
#
# WP's pair is UNCHANGED at 0.05 / 0.02 by design (D33-20): WP labels do not move at all, so
# the 23-point `_EDGE_TIER_SNAPSHOT` recorded before the Phase-31 collapse survives unchanged
# and keeps its evidentiary value instead of being rewritten with a new expectation.
EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float]] = {
    "ats": (1.9493, 0.8359),
    "ou": (0.0546, 0.0220),
    "wp": (0.0500, 0.0200),
}

# The UNROUNDED quantiles the four-decimal values above were rounded from. Recorded so the
# rounding is visible rather than implied, and so a reviewer re-running the derivation can
# compare against the full-precision figure rather than against an already-rounded one.
EDGE_TIER_THRESHOLDS_UNROUNDED: dict[str, tuple[float, float]] = {
    "ats": (1.9493383058409353, 0.8359444713855734),
    "ou": (0.054562252442456764, 0.022044224810953677),
    "wp": (0.05, 0.02),
}

# The unit each pair belongs to. A threshold without a stated unit is a number nobody can
# check, and applying ONE pair to three incompatible units is the defect (DEF-31-17) this
# pre-registration exists to retire.
EDGE_TIER_THRESHOLD_UNITS: dict[str, str] = {
    "ats": "points (signed model-minus-market home margin)",
    "ou": "ratio of the market total, floored at 30",
    "wp": "probability (model minus devigged fair closing probability)",
}

# The EXACT quantile convention, NAMED rather than left to a library default that could change
# on an upgrade. ATS's and O/U's thresholds are the values reproducing WP's OWN band shares on
# their own |edge| distributions: the medium threshold is the quantile at WP's cumulative "low"
# share and the high threshold is the quantile at WP's cumulative "low + medium" share, under
# `utils.edge_tier`'s STRICT `>` comparisons.
THRESHOLD_QUANTILE_CONVENTION: str = (
    'numpy.quantile(magnitudes, q, method="linear"); q taken from WP band shares under its '
    "unchanged 0.05 / 0.02 pair; bands assigned with STRICT > so a value exactly at a "
    "threshold falls in the LOWER band"
)

# WP's measured band shares under its unchanged pair -- the ANCHOR the other two reproduce.
# RE-DERIVED on the end-state artifacts rather than copied. The pre-rebuild reference recorded
# in tests.phase33_state.ATS_BAND_SHARES_BEFORE was wp low 0.2677 / medium 0.2999 / high
# 0.4324; the divergence is REAL and is the re-fit's doing, not a measurement error.
WP_ANCHOR_BAND_SHARES: dict[str, float] = {
    "low": 0.2171,
    "medium": 0.2649,
    "high": 0.5179,
}

# The population the thresholds were derived on -- ALSO the population the label movement is
# measured against (the circularity named in the docstring).
THRESHOLD_DERIVATION_POPULATION: str = (
    "gold seasons 2021-2024 inclusive, scored with "
    "the END-STATE deployed artifacts and joined to data/silver/odds_snapshot.parquet "
    "closing lines; shares taken over the rows carrying a COMPUTABLE edge, which is also "
    "the D31-04 pinned population the label movement is measured against"
)

# The 2026 CHAIN-FIT BIAS, per target, pooled mean residual (actual - predicted) over the
# STRICTLY-PRIOR seasons below. Computed by the EXISTING
# backtest.ou_ev_chain.estimate_prior_season_bias; no second estimator was written.
CHAIN_FIT_BIAS_2026: dict[str, float] = {
    "ats": 0.257407648096468,
    "ou": -0.35080281804116925,
    "wp": -0.03377244391544111,
}

CHAIN_FIT_BIAS_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

# The per-season mean residual each pooled bias was estimated from. INT season keys; see the
# docstring's JSON note.
CHAIN_FIT_BIAS_BY_SEASON: dict[str, dict[int, float]] = {
    "ats": {
        2021: 0.33270674627764446,
        2022: -0.13196612944872393,
        2023: 0.4689554050434054,
        2024: 0.5486763704894928,
        2025: 0.06729962433966105,
    },
    "ou": {
        2021: 0.8403306793748286,
        2022: -1.156911849975586,
        2023: -0.7878977390757779,
        2024: -0.517684106659471,
        2025: -0.1346795266134697,
    },
    "wp": {
        2021: -0.05890535003559801,
        2022: -0.027205781020481202,
        2023: -0.016465509806907815,
        2024: -0.020979506032376267,
        2025: -0.0452830317594038,
    },
}

# WHERE EACH TARGET'S RESIDUALS CAME FROM, read from config/phase33_gate_verdict.toml and
# artifacts/latest.json rather than assumed. `kind` is the gate disposition; `verdict` is the
# verdict as MEASURED and left standing. A `promoted_refit` carrying a non-PASS `verdict` was
# shipped under a recorded owner override -- the verdict was never softened to match the
# ruling, and this field is where that is visible.
CHAIN_FIT_BIAS_SOURCE_BY_TARGET: dict[str, dict[str, object]] = {
    "ats": {
        "artifact": "ats_20260914_221751",
        "kind": "promoted_refit",
        "promoted_against_verdict": True,
        "verdict": "FAIL",
    },
    "ou": {
        "artifact": "ou_20260914_221756",
        "kind": "promoted_refit",
        "promoted_against_verdict": True,
        "verdict": "FAIL",
    },
    "wp": {
        "artifact": "wp_20260914_221745",
        "kind": "promoted_refit",
        "promoted_against_verdict": False,
        "verdict": "PASS",
    },
}

# The END-STATE artifact ids every number above was derived from.
DERIVATION_ARTIFACTS: dict[str, str] = {
    "ats": "ats_20260914_221751",
    "ou": "ou_20260914_221756",
    "wp": "wp_20260914_221745",
}

# EVERY input this derivation read, with its digest. GIT-TRACKED TEXT inputs -- here only
# config/phase33_gate_verdict.toml -- are digested over NEWLINE-NORMALIZED bytes, because this
# repository has core.autocrlf=true and no .gitattributes. Everything under artifacts/ and
# data/ is a GITIGNORED PRODUCTION STORE and is digested over RAW bytes, which is the same
# instrument tests.data_boundary.digest_file uses: the artifacts/latest.json digest below is
# therefore literally equal to tests.phase33_state.POST_GATE_MANIFEST_DIGEST. An `artifact:<id>`
# key is the sha256 over a sorted `<relpath>\0<file sha256>` manifest of the whole directory.
DERIVATION_INPUT_DIGESTS: dict[str, str] = {
    "artifact:ats_20260914_221751": "9ff7f5f8a886d5720afcb99d04f86b8a302c348965a59edaa69eb960231f624b",
    "artifact:ou_20260914_221756": "d591edc12428dd09b1ec1b594f31afa4f59227df3bfb07324771930399f36523",
    "artifact:wp_20260914_221745": "372ca1bb4fb461aa8c39074aa53cbaf99f72416a30ac6ff2b57e25f65321666b",
    "artifacts/latest.json": "9115c8d76532820e6b77dfecc7903f10c911bfc30e02a15cb6c3a33dbc602cfb",
    "config/phase33_gate_verdict.toml": "ace5d8b23757299a61c11e2bfbb779bb4792a990a2e921388f5334aaebf96e67",
    "data/gold/features_ats.parquet": "38353668fd12faa7926dd1ebe025a1ea28932cf659616a8a1966c9053e101245",
    "data/gold/features_ou.parquet": "3f5cb75b7c4ff93ddcd0266e0b0bedbf05688392f78714030c411cfb4cf81a35",
    "data/gold/features_wp.parquet": "9232fb488f0853290c9014ba45e7e48c999175e88671182e0342f43602a5c5a1",
    "data/silver/odds_snapshot.parquet": "218d359910affbb923993bb4ca31eef1b88675c9346303221cee1571514d74cb",
}

# The FORMATTER this module's bytes were produced under. Recorded because this file is
# ruff-formatted as the last step of emission, so that the pre-commit hook has nothing left to
# rewrite in a FROZEN rule. A ruff upgrade that changed the formatting would fail the
# reproduce-check in tests/unit/test_cold_start_derivation_cli.py loudly, and this constant is
# what tells the reader why.
DERIVATION_FORMATTER: str = "ruff 0.15.6"

# The row count behind each derived number, per target. `threshold_rows` is the rows with a
# COMPUTABLE edge; `threshold_population_rows` is the whole population including rows with no
# stored market line; `bias_rows` is the residual pool.
DERIVATION_ELIGIBLE_COUNTS: dict[str, dict[str, int]] = {
    "ats": {
        "bias_rows": 1424,
        "bias_seasons": 5,
        "threshold_population_rows": 1139,
        "threshold_rows": 1087,
    },
    "ou": {
        "bias_rows": 1424,
        "bias_seasons": 5,
        "threshold_population_rows": 1139,
        "threshold_rows": 1087,
    },
    "wp": {
        "bias_rows": 1424,
        "bias_seasons": 5,
        "threshold_population_rows": 1139,
        "threshold_rows": 1087,
    },
}

# THE LABEL MOVEMENT, held at ONE model and ONE edge definition so the thresholds are the only
# thing that varies. `_UNDER_CURRENT` bands the SAME end-state edges with the 0.05 / 0.02 pair
# in force today; `_AFTER` bands them with the frozen pairs above. WP's two rows are identical
# by construction and zero WP games change band.
BAND_SHARES_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, float]] = {
    "ats": {"low": 0.0018, "medium": 0.0101, "high": 0.988},
    "ou": {"low": 0.195, "medium": 0.2374, "high": 0.5676},
    "wp": {"low": 0.2171, "medium": 0.2649, "high": 0.5179},
}

BAND_COUNTS_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, int]] = {
    "ats": {"low": 2, "medium": 11, "high": 1074},
    "ou": {"low": 212, "medium": 258, "high": 617},
    "wp": {"low": 236, "medium": 288, "high": 563},
}

ATS_BAND_SHARES_AFTER: dict[str, dict[str, float]] = {
    "ats": {"low": 0.2171, "medium": 0.2649, "high": 0.5179},
    "ou": {"low": 0.2144, "medium": 0.2695, "high": 0.5161},
    "wp": {"low": 0.2171, "medium": 0.2649, "high": 0.5179},
}

ATS_BAND_COUNTS_AFTER: dict[str, dict[str, int]] = {
    "ats": {"low": 236, "medium": 288, "high": 563},
    "ou": {"low": 233, "medium": 293, "high": 561},
    "wp": {"low": 236, "medium": 288, "high": 563},
}

GAMES_CHANGING_BAND: dict[str, int] = {
    "ats": 522,
    "ou": 77,
    "wp": 0,
}

# Plan 33-08's declared allowance, RESTATED here as part of the pre-registration record. It was
# declared in scripts/run_phase33_gate.py BEFORE any verdict existed; this is not a second
# declaration but the same one, carried into the frozen rule so a reader of the rule meets it.
FIX_CYCLE_ALLOWANCE: int = 0
