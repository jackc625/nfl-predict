"""Install the v2 models once, recorded (MIDSEASON-SWITCH-RECORD.md section 4).

The owner command of the 2026 mid-season switch. It rebuilds the shared gold as v2, trains the
three v2 models once at this instant, saves them, tunes their blend, writes their bet-rule record,
records the install in ``ledger/refit/owner_installs.jsonl`` and swaps ``artifacts/latest.json``
through ``models.artifacts.replace_manifest`` -- or refuses, leaving ``latest.json`` unchanged and
the v1 gold rebuilt (``refit.owner_install.run_owner_install``).

It refuses before week 9's last lock (Sun 2026-11-08 18:00 ET), from a tree with uncommitted
code, and once v2 is already in force. The gold build alone takes about 20 minutes, so the owner
launches it detached (RUNBOOK.md, "Switch day: install v2").

Usage:
    uv run python -m scripts.install_v2

Prints ``INSTALLED=`` with the five ids and exits 0, or ``INSTALL_REFUSED=`` with the reason and
exits 2. Any other failure is a crash (a traceback, exit 1) after the same v1-gold restore.

ASCII only, no emoji.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

#: The production stores, relative to the repository root the owner runs from.
ARTIFACTS_DIR = Path("artifacts")
GOLD_DIR = Path("data/gold")
SILVER_DIR = Path("data/silver")

#: The exit code of a refusal (0 is installed; a crash exits 1 through its traceback).
EXIT_REFUSED: int = 2


def build_parser() -> argparse.ArgumentParser:
    """The CLI: no options; the install instant is the moment it runs."""
    return argparse.ArgumentParser(
        description=(
            "Install the v2 models once (MIDSEASON-SWITCH-RECORD.md section 4): v2 gold, one "
            "training, the blend, the bet rule, the recorded swap of artifacts/latest.json."
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the owner install; print what it installed or why it refused."""
    build_parser().parse_args(argv)
    # Imported here, as scripts.rebuild_gold imports its step: --help loads no training stack.
    from forward_ledger.store import LEDGER_DIR
    from pipeline import steps
    from refit import owner_install

    def build_gold(builder_class: type) -> None:
        steps._build_and_save_gold_with(
            builder_class, frozenset(), unplayed_through_lock=datetime.now(UTC)
        )

    try:
        outcome = owner_install.run_owner_install(
            artifacts_dir=ARTIFACTS_DIR,
            ledger_dir=LEDGER_DIR,
            gold_dir=GOLD_DIR,
            silver_dir=SILVER_DIR,
            now_utc=datetime.now(UTC),
            build_gold=build_gold,
        )
    except owner_install.InstallRefused as refusal:
        print(f"INSTALL_REFUSED= {refusal}")
        return EXIT_REFUSED

    ids = " ".join(f"{key}={value}" for key, value in outcome.ids.items())
    weights = " ".join(f"{key}={value}" for key, value in outcome.blend_weights.items())
    print(f"INSTALLED= {ids}")
    print(f"BET_RULE_SHA256= {outcome.bet_rule_sha256}")
    print(f"BLEND_WEIGHTS= {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
