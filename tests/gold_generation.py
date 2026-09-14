"""The gold GENERATION seam: how a published reading survives a gold rebuild.

THE PROBLEM THIS EXISTS TO SOLVE
--------------------------------
Several repo-root readouts in this project are guarded by permanent doc-drift
tests, and some of those guards RE-RUN their analysis harness against live gold
and assert that the committed point estimates still reproduce. That works exactly
as long as gold never moves. When gold is rebuilt -- Phase 33.1's weather rung,
Phase 33's Wave 14 Elo rung -- every such pinned estimate becomes false, and
there are only three honest responses:

  1. rewrite the readout so it agrees with the new gold. FORBIDDEN. It overwrites
     a published reading. ``GATED-REFIT-READOUT.md`` states the governing rule in
     its own header: nothing published earlier is overwritten, and superseded
     readings stay where they were written, with their dates and their reasons.
  2. delete the guard. FORBIDDEN in principle -- it removes the anti-rot control
     that is the whole point of the guard.
  3. make the harness-reproduction half state WHICH GOLD it was measured against,
     and refuse to compare across generations.

This module is (3). It is the seam, and Plan 33-14's Elo rung is expected to
reuse it rather than rediscover the problem.

Note on (2): four harness-reproduction classes WERE deleted on 2026-09-12 by
owner instruction, for a second reason this seam does not address -- they were
not deterministic (a situational-OU delta measured four different values at four
BLAS thread counts). Non-determinism is a defect in the measurement itself;
generation drift is not. This seam is for the second problem only.

WHY A CONTENT DIGEST IS THE RIGHT KEY, AND A COLUMN-NAME DIGEST IS NOT
----------------------------------------------------------------------
The thing that invalidates a pinned point estimate is a change in VALUES. A gold
rebuild that changes no column name at all -- which is exactly what Plan
33.1-07's rung 3 did to 32 of its columns -- still moves every number a harness
computes from those columns. A schema-shaped key would call that gold the same
generation and let a stale anchor keep asserting. So the key is taken over the
BYTES of the three matrices.

The cost of that choice is recorded rather than hidden: the key is maximally
sensitive. Any rebuild changes it, including one that changes nothing a given
reading depends on. The failure direction is the safe one -- a needless skip
costs a check, a missed skip asserts a falsehood.

A STAT SIGNATURE IS REFUSED, NOT ACCEPTED (D33-32)
--------------------------------------------------
``tests.data_boundary.digest_file`` falls back to a ``stat-size-mtime:`` string
when a file cannot be read. That fallback is right for its own purpose and wrong
here: an mtime moves when nothing about the data does, so a key built on one
would report a new generation after a harmless touch and skip guards that should
have run. An unprovable key is refused by name instead.

CONSTRAINTS
-----------
ASCII only, no emoji (CLAUDE.md hard constraint). Importable from any tier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.data_boundary import digest_file

# Repo root resolved from this file: tests/gold_generation.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# The three gold feature matrices, repo-relative. Every reading this seam guards
# is computed from one or more of them, so the generation is a property of all
# three together rather than of whichever one a given harness happens to open.
GOLD_MATRIX_PATHS: tuple[str, ...] = (
    "data/gold/features_wp.parquet",
    "data/gold/features_ats.parquet",
    "data/gold/features_ou.parquet",
)

# The prefix tests.data_boundary.digest_file uses when it could not read bytes
# and fell back to size and mtime.
_STAT_SIGNATURE_PREFIX = "stat-size-mtime:"


class GoldMatrixMissingError(RuntimeError):
    """A gold matrix named in GOLD_MATRIX_PATHS is not on disk.

    Raised rather than skipped so a missing matrix cannot silently produce a
    stable key over the two that remain -- which would be a key that says "same
    generation" about gold that is missing a third of itself.
    """


class GoldGenerationUnprovableError(RuntimeError):
    """A gold matrix's bytes could not be read, so the generation cannot be proven.

    D33-32's rule, applied here: an unprovable comparison is a refusal, never a
    pass and never a weaker instrument quietly substituted.
    """


def gold_generation_key() -> str:
    """Return a 64-character hex key identifying the CONTENT of today's gold.

    Returns:
        The sha256 over the per-matrix content digests, sorted by path so the key
        does not depend on the order GOLD_MATRIX_PATHS happens to be written in.

    Raises:
        GoldMatrixMissingError: A matrix named in GOLD_MATRIX_PATHS is absent.
        GoldGenerationUnprovableError: A matrix's bytes could not be read and
            ``digest_file`` degraded to a stat signature.
    """
    lines: list[str] = []
    for relative in GOLD_MATRIX_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            msg = (
                f"gold matrix {relative} is absent from {REPO_ROOT}. A generation "
                "key over the remaining matrices would be a key that claims to "
                "describe gold it never read."
            )
            raise GoldMatrixMissingError(msg)
        digest = digest_file(path)
        if digest.startswith(_STAT_SIGNATURE_PREFIX):
            msg = (
                f"gold matrix {relative} could not be read as bytes; digest_file "
                f"degraded to {_STAT_SIGNATURE_PREFIX!r}. A generation key built "
                "on size and mtime would move when nothing about the data did "
                "(D33-32: an unprovable comparison is a refusal, never a pass)."
            )
            raise GoldGenerationUnprovableError(msg)
        lines.append(f"{relative}={digest}")
    joined = "\n".join(sorted(lines))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def require_gold_generation(
    expected_key: str,
    *,
    reading: str,
    moved_by: str,
    recorded_in: str,
) -> None:
    """Skip the calling test unless live gold is the generation *reading* was measured on.

    Args:
        expected_key: The generation key the reading was MEASURED against.
        reading: What the caller is guarding, in words a human can act on --
            e.g. "OU-DIVERGENCE-DIAGNOSIS.md's pooled over-share of 0.727".
        moved_by: The phase or plan that moved gold out from under the reading.
        recorded_in: Where the supersession is recorded, so the skip message
            points at the document that explains it rather than at nothing.

    Returns:
        None, when the live generation equals *expected_key*.

    Raises:
        Skipped: pytest's skip exception, when the generations differ or gold is
            absent. It SKIPS rather than fails on purpose: a reading measured on
            other gold is not a wrong reading, and a guard that failed here would
            be reporting a defect where there is only a different generation.
    """
    try:
        live_key = gold_generation_key()
    except GoldMatrixMissingError as absent:
        pytest.skip(
            f"gold is not present in this checkout, so the generation "
            f"{reading!r} was measured against cannot be compared: {absent}"
        )
    if live_key == expected_key:
        return
    pytest.skip(
        f"SUPERSEDED READING, NOT A FAILURE. {reading} was measured against gold "
        f"generation {expected_key}. Live gold is generation {live_key}, because "
        f"{moved_by} rebuilt it. The reading is NOT rewritten to agree with the "
        f"new gold: it stays where it was written, with its date and its reason, "
        f"and the supersession is recorded in {recorded_in}. Only this "
        f"harness-reproduction half is skipped -- the document-level assertions "
        f"that actually guard the document still run."
    )
