"""The mid-season switch witness: the 2026 switch record and the v2 recipe it installs.

Task 11 of the 2026-11-06 switch plan. The expected values live HERE, outside the files they
witness and in a commit strictly after the record commit, for the reason ``tests/phase38_state.py``
records: a file carrying its own hash or its own commit sha has no fixed point.
``tests/unit/test_midseason_switch_ancestry.py`` recomputes every value and compares.

Constants only: NO imports of any kind, NO I/O and NO logic, so any test at any tier can import it
without cost or side effects.

Hashes are over NEWLINE-NORMALIZED bytes (CRLF -> LF): ``core.autocrlf`` is true with no
``.gitattributes``, so a raw working-tree hash would disagree with the blob on a Windows checkout.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later
correction is a NEW slot placed beside the old one, under its own separator naming the
plan, the task and the measurement date; the old slot stays as the record of what was
measured when it was written.

* Switch plan 2026-11-06 Task 11 (this file's author) -- ``SWITCH_RECORD_WITNESS``,
  ``SWITCH_RECORD_COMMITTED_AT``, ``SWITCH_RECORD_DEADLINE_UTC``, ``V2_RECIPE_FILE_WITNESS``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

# ---------------------------------------------------------------------------
# Switch plan 2026-11-06 Task 11 -- the switch record witness, MEASURED
# 2026-11-06 at 16:48 ET (read via PowerShell Get-Date), on branch
# switch-witness cut from origin/master f81f3a7.
#
# The record was committed alone as a27898a (switch at week 13) and corrected
# once, by owner ruling, as baa2d33 (switch at week 10; its message INVALIDATES
# a27898aa1c1369ade3d8b47be8ab983cf16572b6 in part). The witnessed commit is the
# correction: `git log -1 --format=%H -- <path>` gives baa2d33 for both paths,
# and `git show --name-only --format= <commit>` lists exactly these two paths.
# Each digest is sha256 of the file's bytes with CRLF -> LF (`uv run python`
# with hashlib, the same fold as refit/witness.py `_normalized_sha256`); the
# working tree was clean for both paths, and each digest equals the sha256 of
# `git show <commit>:<path>`. The committed-at value is
# `git show -s --format=%cI <commit>`. The commit is reachable from
# refs/remotes/origin/master. These slots land in a LATER commit than the
# record, so no record carries its own witness.
# ---------------------------------------------------------------------------

SWITCH_RECORD_WITNESS: dict[str, dict[str, str]] = {
    "MIDSEASON-SWITCH-RECORD.md": {
        "commit": "baa2d333256f0d6df1c0cc10a5ff1c863b604d0c",
        "normalized_sha256": "22c49e140637cdb03aa18bce9ea23a9312ab60757f98476bc159b70ca0e4912c",
    },
    "backtest/midseason_switch_2026.py": {
        "commit": "baa2d333256f0d6df1c0cc10a5ff1c863b604d0c",
        "normalized_sha256": "e3e5a99289b78f1d5f3004b502573163bcc51d4c389e79e22b639a511f1b4343",
    },
}

SWITCH_RECORD_COMMITTED_AT: str = "2026-11-06T16:13:22-05:00"

# Tue 2026-11-10 17:00 America/New_York (EST), in UTC: after week 9's last lock and before
# week 10's first (record section 8). Equal to
# backtest.midseason_switch_2026.PREREGISTRATION_DEADLINE_UTC (asserted, not imported).
SWITCH_RECORD_DEADLINE_UTC: str = "2026-11-10T22:00:00+00:00"

# models/v2_recipe.py, the recipe the record installs (record section 4), measured the same
# way at the same time: `git log -1 --format=%H -- models/v2_recipe.py` gives 6652baf, the
# commit record section 4 names, and the digest is the CRLF -> LF sha256 of the clean working
# tree, equal to the sha256 of `git show 6652baf:models/v2_recipe.py`.
V2_RECIPE_FILE_WITNESS: dict[str, dict[str, str]] = {
    "models/v2_recipe.py": {
        "commit": "6652baf60baec2934b345a0a6e08848ec1e1ab1c",
        "normalized_sha256": "af68c6a341c81d8df545cbb633fcd468582032fd1e8540cebfab695b2a425fa8",
    },
}
