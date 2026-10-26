"""The Phase 38 witness: the 2026 verdict pre-registration.

Plan 38-03 (CLEAN-04, D-11, D-12). The expected values live HERE, outside the files they witness
and in a commit strictly after the record commit, for the reason ``tests/phase34_state.py`` and
``tests/phase33_state.py`` record: a file carrying its own hash or its own commit sha has no fixed
point. ``tests/unit/test_phase38_preregistration_ancestry.py`` recomputes every value and compares.

Constants only: NO project imports, NO I/O and NO logic, so any test at any tier can import it
without cost or side effects.

Hashes are over NEWLINE-NORMALIZED bytes (CRLF -> LF): ``core.autocrlf`` is true with no
``.gitattributes``, so a raw working-tree hash would disagree with the blob on a Windows checkout.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later
correction is a NEW slot placed beside the old one, under its own separator naming the
plan, the task and the measurement date; the old slot stays as the record of what was
measured when it was written.

* Plan 38-03 Task 1 (this file's author) -- ``P38_PREREGISTRATION_WITNESS``,
  ``P38_PREREGISTRATION_COMMITTED_AT``, ``P38_DEADLINE_UTC``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plan 38-03 Task 1 -- the pre-registration witness, MEASURED 2026-10-21 at
# 19:37 ET (read via PowerShell Get-Date), after Plan 38-01 committed the record
# ALONE as 922b010 and pushed it to origin/master at 19:22:45 ET.
#
# The commit is `git log -1 --format=%H -- <path>`, the same for both paths, and
# `git show --name-only --format= <commit>` lists exactly these two paths. Each
# digest is sha256 of the file's bytes with CRLF -> LF (`uv run python -c` with
# hashlib; the working tree was clean for both paths, and the digest equals the
# sha256 of `git show <commit>:<path>`). The committed-at value is
# `git show -s --format=%cI <commit>`. These slots land in a LATER commit than
# the record, so no record carries its own witness.
# ---------------------------------------------------------------------------

P38_PREREGISTRATION_WITNESS: dict[str, dict[str, str]] = {
    "SEASON-VERDICT-PREREGISTRATION.md": {
        "commit": "922b010b70d973a21945c15ca88067dae4149ad5",
        "normalized_sha256": "4f1cddc3571aafaceb92ea0a00a3d14738e6c5db96bd820892ee0192864a6fbf",
    },
    "backtest/verdict_test_2026.py": {
        "commit": "922b010b70d973a21945c15ca88067dae4149ad5",
        "normalized_sha256": "c6607f881bc817ff4610d229d3940e398326612324d9d1e9db0a494ac2caa20c",
    },
}

P38_PREREGISTRATION_COMMITTED_AT: str = "2026-10-21T19:19:13-04:00"

# D-11: week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (EDT), in UTC. Equal
# to backtest.verdict_test_2026.PREREGISTRATION_DEADLINE_UTC (asserted, not imported).
P38_DEADLINE_UTC: str = "2026-10-28T21:00:00+00:00"
