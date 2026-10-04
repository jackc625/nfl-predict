"""Where the repository's reports live, looked up by bare file name.

On 2026-10-04 the public-repo facelift moved most repo-root reports into two folders:
``docs/guides/`` (how the system works and how to run it) and ``docs/records/`` (the paper
trail of readouts, audits and diagnoses). Six history-anchored records stay at the repo root
and must never move: their tamper evidence asks git for the last commit that touched each file
AT ITS ROOT PATH, and a move would make the move commit that last commit.

Most tests open their one report through a path constant. The tests that look reports up by
NAME -- because the same name is also searched for as a token inside another document -- call
``doc_path`` so the token stays bare while the file is found wherever it lives.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

# Repo root resolved from this file: tests/doc_locations.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

GUIDES_DIR = REPO_ROOT / "docs" / "guides"
RECORDS_DIR = REPO_ROOT / "docs" / "records"

GUIDES: tuple[str, ...] = (
    "AUTOMATION.md",
    "METHODOLOGY.md",
    "PIPELINE.md",
    "RUNBOOK.md",
    "STATE-OF-SYSTEM.md",
)

RECORDS: tuple[str, ...] = (
    "ACTIVATION-READOUT.md",
    "AUDIT-REPORT.md",
    "BLEND-TUNING-READOUT.md",
    "CLOSING-LINE-AUDIT.md",
    "GATED-REFIT-READOUT.md",
    "GROUP-VERDICT-READOUT.md",
    "HISTORICAL-WEATHER-READOUT.md",
    "LINE-MOVEMENT-READOUT.md",
    "LIVE-COLD-START-READOUT.md",
    "MODEL-DIAGNOSIS.md",
    "OU-DIVERGENCE-DIAGNOSIS.md",
    "PRECOVERAGE-SCAN.md",
    "PROFITABILITY-READOUT.md",
    "REFIT-READOUT.md",
    "SELECTION-CENSUS.md",
    "SIGNAL-LIFT-READOUT.md",
    "WEATHER-NULL-LIST.md",
)

# Never moved: each one's git history at its root path is its proof of having been frozen
# before the results it governs existed.
ROOT_RECORDS: tuple[str, ...] = (
    "COLD-START-CORRECTION.md",
    "COLD-START-PREREGISTRATION.md",
    "EV-CHAIN-CORRECTION.md",
    "MOS-DECODE-COMPARISON.md",
    "NEUTRAL-HFA-BET-RULE-CORRECTION.md",
    "PROFITABILITY-PREREGISTRATION.md",
)


def doc_path(name: str) -> Path:
    """Return the current location of a report from its bare file name.

    Args:
        name: A bare report name such as ``"PIPELINE.md"``, or any repo-relative path.

    Returns:
        ``docs/guides/<name>`` or ``docs/records/<name>`` for a moved report, otherwise
        ``<repo root>/<name>`` (README.md, CLAUDE.md, the six ROOT_RECORDS, and any
        repo-relative path such as ``config/gate.toml``).
    """
    if name in GUIDES:
        return GUIDES_DIR / name
    if name in RECORDS:
        return RECORDS_DIR / name
    return REPO_ROOT / name
