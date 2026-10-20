"""The production witness of the 2026 rolling re-fit pre-registration (Phase 37; RFIT-01, P4).

WHY IT EXISTS
-------------
The runtime fingerprint refusal (the re-fit's first integrity check, ``recipe_fingerprint``) must
know what the frozen record hashes to, and production code must not import ``tests.*``
(A33.2-review WR-06, research Pitfall 9). So this module restates, as literals, the record's
digests, its commit and its committed-at; ``tests/unit/test_phase37_preregistration_ancestry.py``
asserts they equal the test witness in ``tests/phase37_state.py`` and what git reports.

It lands in a commit strictly AFTER the record commit, outside the two files it covers: a file
carrying its own hash or its own commit sha has no fixed point.

Hashes are over NEWLINE-NORMALIZED bytes (CRLF -> LF): ``core.autocrlf`` is true with no
``.gitattributes``, so a raw working-tree hash would disagree with the blob on a Windows checkout.

Stdlib only; no project imports.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "PREREGISTRATION_COMMIT",
    "PREREGISTRATION_COMMITTED_AT",
    "PREREGISTRATION_PATHS",
    "PREREGISTRATION_SHA256",
    "REPO_ROOT",
    "committed_at",
    "fingerprint_mismatches",
]

# The repository root, anchored to this file: refit/witness.py -> repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# The two files of the one pre-registration, repo-relative.
PREREGISTRATION_PATHS: tuple[str, str] = (
    "ROLLING-REFIT-PREREGISTRATION.md",
    "backtest/refit_preregistration_2026.py",
)

# Each file's newline-normalized sha256 at the record commit.
PREREGISTRATION_SHA256: dict[str, str] = {
    "ROLLING-REFIT-PREREGISTRATION.md": (
        "8972e617c0dee1e5cc86c2b127985c5092cb59a6eaf0c626b408989a21ea0add"
    ),
    "backtest/refit_preregistration_2026.py": (
        "174510d7ec74785a02ccc00cab75b4f9a54af9d7056385ebd5041e1fd0f5ea19"
    ),
}

# The commit that landed both files alone, pushed to origin/master 2026-10-20 15:41:32 ET.
PREREGISTRATION_COMMIT: str = "e25884da8356b9ec76970c4c14f87a797d2a32d7"

# That commit's committer date, as ``git show -s --format=%cI`` prints it.
PREREGISTRATION_COMMITTED_AT: str = "2026-10-20T15:41:23-04:00"


def committed_at() -> datetime:
    """The record commit's committer date as an aware UTC datetime."""
    return datetime.fromisoformat(PREREGISTRATION_COMMITTED_AT).astimezone(UTC)


def _normalized_sha256(path: Path) -> str:
    """sha256 of a file's bytes with CRLF folded to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint_mismatches(repo_root: Path = REPO_ROOT) -> list[str]:
    """The record paths that are missing or no longer hash to the witness, in record order.

    An empty list is the only state in which a re-fit may proceed.
    """
    return [
        relative_path
        for relative_path in PREREGISTRATION_PATHS
        if not (repo_root / relative_path).is_file()
        or _normalized_sha256(repo_root / relative_path)
        != PREREGISTRATION_SHA256[relative_path]
    ]
