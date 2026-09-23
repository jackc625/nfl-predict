"""Doc-drift guard for the repo-root EV-CHAIN-CORRECTION.md (Plan 33.2-29 Task 3, D33.2-25).

Pattern: ``tests/unit/test_signal_lift_readout_md.py``. The document is checked as PARSED ROWS,
and the anti-rot guard compares the RULING it records -- which targets have a floor and which
have none -- against ``outputs/p332/corrected_chain_fit.json``. It never pins a point estimate:
pinning a number to a re-derivable fit is the mistake this repository already made (D29-06-02).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRECTION_MD = REPO_ROOT / "EV-CHAIN-CORRECTION.md"
CORRECTED_RECORD = REPO_ROOT / "outputs" / "p332" / "corrected_chain_fit.json"

TARGETS = ("wp", "ats", "ou")
NO_THRESHOLD_TOKENS = ("none", "no threshold")

_REQUIRED_SECTION_MARKERS = (
    "What changed, in plain English",
    "What was superseded",
    "Old and new values",
    "The window",
    "How win bets are priced",
    "The owner ruling on a target with no floor",
    "Staged, not live",
)
_REQUIRED_PHRASES = (
    "not clean evidence",
    "ee20773",
    "byte-unchanged",
    "run ledger",
    "was NOT re-spent",
    "no-bets-for-that-target",
)


def _text() -> str:
    return CORRECTION_MD.read_text(encoding="utf-8")


def parse_correction_rows(text: str) -> list[dict[str, str]]:
    """The per-target rows of the old/new table, as ``{target, old_floor, new_floor, old_sd,
    new_sd}``. Only rows whose first cell is a canonical target are returned."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in TARGETS:
            continue
        target, old_floor, new_floor, old_sd, new_sd = cells
        rows.append(
            {
                "target": target,
                "old_floor": old_floor,
                "new_floor": new_floor,
                "old_sd": old_sd,
                "new_sd": new_sd,
            }
        )
    return rows


def test_the_document_exists_at_the_repo_root_and_is_ascii() -> None:
    assert CORRECTION_MD.is_file()
    assert _text().isascii()


@pytest.mark.parametrize("marker", _REQUIRED_SECTION_MARKERS)
def test_every_required_section_is_present(marker: str) -> None:
    assert marker in _text()


@pytest.mark.parametrize("phrase", _REQUIRED_PHRASES)
def test_every_required_phrase_is_present(phrase: str) -> None:
    assert phrase.lower() in _text().lower()


def test_the_over_claim_word_is_absent() -> None:
    assert "proven" not in _text().lower()


def test_every_target_has_one_row_with_old_and_new_floors() -> None:
    rows = parse_correction_rows(_text())
    assert sorted(row["target"] for row in rows) == sorted(TARGETS)
    for row in rows:
        assert row["old_floor"], row
        assert row["new_floor"], row


def test_the_parser_ignores_non_target_rows() -> None:
    """Control: the header and separator rows are not parsed as targets."""
    table = (
        "| Target | Old floor | New floor | Old SD | New SD |\n|---|---|---|---|---|\n"
    )
    assert parse_correction_rows(table) == []


def test_the_documented_ruling_matches_the_record() -> None:
    """Which targets have NO threshold -- a ruling, not a number -- agrees with the record."""
    if not CORRECTED_RECORD.exists():
        pytest.skip(
            f"{CORRECTED_RECORD} is absent (outputs/ is gitignored); regenerate it with "
            "`uv run python -m scripts.derive_corrected_ev_chain`."
        )
    record = json.loads(CORRECTED_RECORD.read_text(encoding="utf-8"))["tune_fit"]
    documented = sorted(
        row["target"]
        for row in parse_correction_rows(_text())
        if row["new_floor"].lower() in NO_THRESHOLD_TOKENS
    )
    recorded = sorted(t for t, block in record.items() if block["ev_floor_t"] is None)
    assert documented == recorded
