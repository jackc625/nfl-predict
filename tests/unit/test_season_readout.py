"""What the 2026 season-close readout must show (Phase 38, Plan 38-04; SPEC R7, R9).

Every scheduled game of the verdict scope lands in exactly one coverage state, rolled up per week,
so a partly missed week can never read as covered. The preview is labelled "season to date -- not
the verdict" and shows no test result.

COLD-05: fixture tests write only under ``tmp_path``. The real-store tests READ the production
ledger and silver games (never write them) and step aside with a message containing "absent" when
a store is missing on the machine running them.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forward_ledger.settle import DEFAULT_SILVER_DIR, SILVER_GAMES_FILENAME
from forward_ledger.store import LEDGER_DIR, ledger_path
from scripts import season_readout

REAL_LEDGER = ledger_path(LEDGER_DIR)
REAL_SILVER_GAMES = DEFAULT_SILVER_DIR / SILVER_GAMES_FILENAME


def _require_real_stores() -> None:
    """Step aside, naming the store, when the real ledger or silver games is absent."""
    for path in (REAL_LEDGER, REAL_SILVER_GAMES):
        if not Path(path).exists():
            pytest.skip(
                f"the real store {Path(path).as_posix()} is absent on this machine"
            )


def _line(lines: list[str], prefix: str) -> str:
    """The one output line starting with *prefix*."""
    matches = [line for line in lines if line.startswith(prefix)]
    assert len(matches) == 1, f"expected one line starting {prefix!r}, got {matches}"
    return matches[0]


def test_the_preview_accounts_for_weeks_5_and_6_on_the_real_stores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SPEC AC 19: week 5 lost 13 of 15 games with the laptop off; week 6 missed WAS@SF only."""
    _require_real_stores()

    assert season_readout.main([]) == 0
    lines = capsys.readouterr().out.splitlines()

    week6 = _line(lines, "COVERAGE= week=6 ")
    assert week6.startswith("COVERAGE= week=6 scheduled=14 decided=13 bet=11 no_bet=2")
    assert week6.endswith("missed=1")
    week5 = _line(lines, "COVERAGE= week=5 ")
    assert week5.startswith("COVERAGE= week=5 scheduled=15 decided=2 bet=2 no_bet=0")
