"""The Phase 37 witness: the 2026 rolling re-fit pre-registration, the shadow pin and the gate files.

Plan 37-05 (RFIT-01, RFIT-06, D-23). The expected values live HERE, outside the files they witness
and in a commit strictly after the record commit, for the reason ``tests/phase34_state.py`` and
``tests/phase33_state.py`` record: a file carrying its own hash or its own commit sha has no fixed
point. ``tests/unit/test_phase37_preregistration_ancestry.py`` recomputes every value and compares.

The production code cannot import this module (``tests.*`` is never imported by production,
A33.2-review WR-06, research Pitfall 9), so ``refit/witness.py`` restates the record's digests,
commit and committed-at as literals, and the ancestry test asserts the two agree.

Hashes are over NEWLINE-NORMALIZED bytes (CRLF -> LF): ``core.autocrlf`` is true with no
``.gitattributes``, so a raw working-tree hash would disagree with the blob on a Windows checkout.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plan 37-05 Task 1 -- the pre-registration witness, MEASURED 2026-10-20 at
# 15:48 ET (read via PowerShell Get-Date), after Plan 37-04 committed the record
# ALONE as e25884d and pushed it to origin/master at 15:41:32 ET.
#
# The commit is `git log -1 --format=%H -- <path>`, the same for both paths, and
# `git show --name-only --format= <commit>` lists exactly these two paths. Each
# digest is sha256 of the file's bytes with CRLF -> LF (the working tree was
# clean for both paths, so it equals the committed blob). The committed-at value
# is `git show -s --format=%cI <commit>`. These slots land in a LATER commit
# than the record, so no record carries its own witness.
# ---------------------------------------------------------------------------

P37_PREREGISTRATION_WITNESS: dict[str, dict[str, str]] = {
    "ROLLING-REFIT-PREREGISTRATION.md": {
        "commit": "e25884da8356b9ec76970c4c14f87a797d2a32d7",
        "normalized_sha256": "8972e617c0dee1e5cc86c2b127985c5092cb59a6eaf0c626b408989a21ea0add",
    },
    "backtest/refit_preregistration_2026.py": {
        "commit": "e25884da8356b9ec76970c4c14f87a797d2a32d7",
        "normalized_sha256": "174510d7ec74785a02ccc00cab75b4f9a54af9d7056385ebd5041e1fd0f5ea19",
    },
}

P37_PREREGISTRATION_COMMITTED_AT: str = "2026-10-20T15:41:23-04:00"

# D-03: week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (EDT), in UTC. Equal
# to backtest.refit_preregistration_2026.PREREGISTRATION_DEADLINE_UTC (asserted, not imported).
P37_DEADLINE_UTC: str = "2026-10-28T21:00:00+00:00"

# ---------------------------------------------------------------------------
# The frozen shadow arm's pin (Plan 37-01), MEASURED at the same instant: its
# last commit (`git log -1 --format=%H -- backtest/shadow_pin_2026.py`, the
# commit that first tracked it) and its newline-normalized sha256.
# ---------------------------------------------------------------------------

P37_SHADOW_PIN_WITNESS: dict[str, dict[str, str]] = {
    "backtest/shadow_pin_2026.py": {
        "commit": "66c99337b5940786368b532dd92b87bdcec422e7",
        "normalized_sha256": "89f082834427a40c2a4ce2fd6a09fb658205d5e1f1a2d9135a1d54390ff6b3f4",
    },
}

# ---------------------------------------------------------------------------
# RFIT-06 / R5: the old deploy-gate path, byte-unchanged by Phase 37. Each value
# is the newline-normalized sha256 MEASURED at the same instant, with no
# uncommitted change to any of the three files.
# ---------------------------------------------------------------------------

P37_UNTOUCHED_GATE_FILES: dict[str, str] = {
    "config/gate.toml": "ae24a1085d3d2e5f39cbf0d46a61a3be0dbcebeb2d481a02a301ded3ce4c4db2",
    "models/deploy_gate.py": "cd9ff969c1933b7627c4c1ead9cc1bad9e76fc77bce5f6dab7c5397ea0ca4a0e",
    "scripts/promote_models.py": "d6dc1d6da62ed2b58b179aa8fd6a7d6c1bc4f07e71284f89b8611a02080587c3",
}
