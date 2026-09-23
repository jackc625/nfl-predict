"""Install the four corrected artifacts into production in ONE atomic write (Plan 33.2-25).

WHAT THIS DOES. It moves the four pointers in ``artifacts/latest.json`` -- ``wp``, ``ats``,
``ou`` and ``blend`` -- to the corrected artifacts, through
``models.artifacts.replace_manifest``, called exactly once. That function LOADS every
artifact through the serving path's own loaders, cross-checks that all four record the same
gold and that the blend was tuned on these three models, and only then writes the manifest
once through the single atomic-write helper. A refusal leaves the manifest byte-unchanged.

WHERE THE IDS COME FROM. They are read from ``tests.phase33_state.P332_25B_SWAP_ARTIFACT_IDS``,
the slot recorded BEFORE the swap. They are never re-derived by scanning ``artifacts/`` for
the newest directories: a scan would pick up anything else written since. The slot supersedes
``P332_25_SWAP_ARTIFACT_IDS``, which named the models fitted before the snap coverage flag was
found in the candidate pool (see the step 25b comment in ``tests/phase33_state.py``).

WHY ``scripts/promote_models --promote`` IS NOT REUSED. It runs the per-target deploy gate,
and SPEC R13 removes that gate for this swap by owner ruling: the corrected artifacts replace
today's UNCONDITIONALLY, with no pre-correction model or gate baseline as a comparator. The
owner accepted the four artifacts as production on 2026-09-23 (Plan 33.2-25 Task 3, "Swap").

THE PRE-STATE GUARD. The swap is reversible only by hand, from the pre-swap manifest recorded
in ``tests.phase33_state.P332_25_PRE_SWAP_LATEST_JSON_TEXT``. So ``--apply`` refuses unless
the live manifest's bytes still hash to ``P332_25_PRE_SWAP_LATEST_JSON_SHA256``: if production
moved after the record was written, the record no longer describes what would be replaced.

Usage::

    uv run python -m scripts.swap_corrected_artifacts --dry-run   # validate; write nothing
    uv run python -m scripts.swap_corrected_artifacts --apply     # the production swap

Declared write: ``artifacts/latest.json`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from models.artifacts import (
    ARTIFACT_VALIDATORS,
    ArtifactBundleInvalidError,
    replace_manifest,
)
from tests.phase33_state import (
    P332_25_PRE_SWAP_LATEST_JSON_SHA256,
    P332_25B_SWAP_ARTIFACT_IDS,
)

ARTIFACTS_DIR = Path("artifacts")
LATEST_PATH = ARTIFACTS_DIR / "latest.json"


class SwapRefusedError(Exception):
    """The live manifest is not the one the recorded pre-swap state describes."""


def _sha256(path: Path) -> str:
    """Hex sha256 of a file's raw bytes (CRLF as stored, never normalized)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def swap_mapping() -> dict[str, str]:
    """The four pointers the swap installs, exactly as recorded in the state manifest."""
    return dict(P332_25B_SWAP_ARTIFACT_IDS)


def _current_manifest() -> dict[str, str]:
    return json.loads(LATEST_PATH.read_text(encoding="utf-8"))


def assert_pre_swap_state() -> None:
    """Refuse unless ``latest.json`` is byte-for-byte the recorded pre-swap manifest."""
    if not LATEST_PATH.exists():
        msg = (
            f"{LATEST_PATH} does not exist; there is no production manifest to replace."
        )
        raise SwapRefusedError(msg)
    actual = _sha256(LATEST_PATH)
    if actual != P332_25_PRE_SWAP_LATEST_JSON_SHA256:
        msg = (
            f"{LATEST_PATH} hashes to {actual}, not the recorded pre-swap "
            f"{P332_25_PRE_SWAP_LATEST_JSON_SHA256}: production moved after the record was "
            "written, so the recorded manifest would not restore what this swap replaces."
        )
        raise SwapRefusedError(msg)


def validate_only(mapping: dict[str, str]) -> None:
    """Load each artifact through its registered validator; raises on the first failure.

    The dry run's check. ``--apply`` does not rely on it: ``replace_manifest`` repeats the
    same validation, collects every failure, and adds the cross-artifact checks.
    """
    for key, version in mapping.items():
        ARTIFACT_VALIDATORS[key](version, ARTIFACTS_DIR)


def _print_plan(before: dict[str, str], mapping: dict[str, str]) -> None:
    for key, new in mapping.items():
        print(f"POINTER {key}: {before.get(key)} -> {new}")


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Install the four corrected artifacts into artifacts/latest.json in one "
            "atomic write. Runs no deploy gate (SPEC R13)."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate all four; write nothing."
    )
    mode.add_argument("--apply", action="store_true", help="Swap production.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    mapping = swap_mapping()
    try:
        assert_pre_swap_state()
        before = _current_manifest()
        _print_plan(before, mapping)
        print(f"LATEST_SHA256_BEFORE= {_sha256(LATEST_PATH)}")
        if args.dry_run:
            validate_only(mapping)
            print("VALIDATED= 4 (dry run -- nothing written)")
            sys.exit(0)
        replace_manifest(mapping, ARTIFACTS_DIR)
    except (SwapRefusedError, ArtifactBundleInvalidError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    after = _current_manifest()
    swapped = sum(after.get(key) == value for key, value in mapping.items())
    print(f"LATEST_SHA256_AFTER= {_sha256(LATEST_PATH)}")
    print(f"LATEST= {after}")
    print(f"SWAPPED= {swapped}")
    sys.exit(0 if swapped == len(mapping) else 1)


if __name__ == "__main__":
    main()
