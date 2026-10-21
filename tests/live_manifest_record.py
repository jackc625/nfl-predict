"""What the live ``artifacts/latest.json`` must hold, read from the records (Phase 37, Plan 37-20).

WHY THE LIVE-MANIFEST TESTS FOLLOW A RECORD AND NOT A HARD-CODED SWAP
----------------------------------------------------------------------
Until Phase 37 the live manifest moved only by a hand-run, recorded swap, and each swap re-pointed
by hand the tests that pin it (row 19 last, 2026-10-04). From Tue 2026-11-03 the daily run installs
re-fit generations unattended (D-01, RFIT-06), recording each install in
``ledger/refit/dates.jsonl`` BEFORE the swap writes ``latest.json`` (the install_intent, Plan 37-11,
research Pitfall 2). So the recorded end state is the newest generation
``refit.generations.installed_generations`` derives -- the one derivation; nothing here re-decides
which intents count -- with ONE exception: a newest install whose date has no ``closed`` event and
whose four ids ``latest.json`` does not serve is not yet in force, because its run died between the
intent and the swap. Before the first install the end state is generation 0, the row-19 swap's
record (``tests.phase33_state.ROW19_SWAP_ARTIFACT_IDS`` and
``ROW19_POST_SWAP_LATEST_JSON_SHA256``).

The owner's standing ruling that old models and baselines are dead is why row 19 is accepted only
as the record in force until the first recorded install, and never held as "the" live manifest.
A refused, unrecorded or rolled-back swap still makes every comparing test fail: a refused intent
does not count (installed_generations), and a closed install that ``latest.json`` no longer serves
stays the expected state.

Reads production stores (``ledger/`` and ``artifacts/latest.json``) READ-ONLY; every argument can
point at a temporary directory, which is how the controls in
``tests/unit/test_live_manifest_record.py`` exercise each branch.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backtest.refit_preregistration_2026 import (
    CUTOFF_LOCAL_TIME,
    CUTOFF_TIMEZONE,
    REFIT_DATE_LISTS,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    TRAINING_START_SEASON,
)
from refit.generations import Generation, installed_generations
from refit.records import read_date_events
from refit.witness import PREREGISTRATION_SHA256
from tests.phase33_state import ROW19_SWAP_ARTIFACT_IDS

__all__ = [
    "LIVE_LATEST_JSON",
    "LIVE_LEDGER_DIR",
    "MANIFEST_KEYS",
    "REPO_ROOT",
    "expected_manifest_text",
    "recorded_live_manifest",
    "refit_metadata_problems",
]

# The repository root, anchored to this file: tests/live_manifest_record.py -> repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
LIVE_LEDGER_DIR: Path = REPO_ROOT / "ledger"
LIVE_LATEST_JSON: Path = REPO_ROOT / "artifacts" / "latest.json"

# The four pointers latest.json holds, in the order replace_manifest keeps them.
MANIFEST_KEYS: tuple[str, ...] = ("wp", "ats", "ou", "blend")


def _manifest_ids(generation: Generation) -> dict[str, str]:
    """The four manifest pointers of *generation* (the converter is bound by the blend)."""
    return {key: str(getattr(generation.ids, key)) for key in MANIFEST_KEYS}


def _serves(latest_path: Path, ids: Mapping[str, str]) -> bool:
    """Whether *latest_path* names exactly *ids* under every manifest key (absent = serves nothing)."""
    if not latest_path.exists():
        return False
    manifest = json.loads(latest_path.read_text(encoding="utf-8"))
    return all(manifest.get(key) == ids[key] for key in MANIFEST_KEYS)


def _date_closed(events: list[dict[str, Any]], generation: Generation) -> bool:
    """Whether the cadence date that installed *generation* has a ``closed`` event."""
    return any(
        event["event"] == "closed"
        and int(event["list_week"]) == generation.list_week
        and int(event["date_index"]) == generation.date_index
        for event in events
    )


def recorded_live_manifest(
    ledger_dir: Path = LIVE_LEDGER_DIR, latest_path: Path = LIVE_LATEST_JSON
) -> tuple[int, dict[str, str]]:
    """The generation in force by the records, and the four pointers latest.json must hold.

    Args:
        ledger_dir: The ledger directory holding ``refit/dates.jsonl`` (read-only).
        latest_path: The manifest consulted only to tell a pending install from a served one
            (read-only; an absent file serves nothing).

    Returns:
        ``(generation number, {wp, ats, ou, blend: artifact id})``; generation 0 is the row-19
        swap's record.
    """
    generations = installed_generations(ledger_dir)
    newest = max(generations, key=lambda generation: generation.number)
    if newest.number >= 1 and not _date_closed(read_date_events(ledger_dir), newest):
        if not _serves(latest_path, _manifest_ids(newest)):
            # The intent is written before replace_manifest (Plan 37-11): a run that died
            # between them leaves the intent counted but never served, so it is not in force.
            newest = next(g for g in generations if g.number == newest.number - 1)
    if newest.number == 0:
        return 0, dict(ROW19_SWAP_ARTIFACT_IDS)
    return newest.number, _manifest_ids(newest)


def expected_manifest_text(ids: Mapping[str, str]) -> str:
    """The text ``models.artifacts._atomic_write_json`` writes for *ids*, before newline translation."""
    return json.dumps({key: ids[key] for key in MANIFEST_KEYS}, indent=2)


def _cadence_window() -> tuple[datetime, datetime]:
    """The first and last pre-registered re-fit dates, each at the cutoff time in the record's zone."""
    zone = ZoneInfo(CUTOFF_TIMEZONE)
    cutoff_time = time.fromisoformat(CUTOFF_LOCAL_TIME)
    dates = sorted({date for dates in REFIT_DATE_LISTS.values() for date, _ in dates})
    first, last = (
        datetime.combine(datetime.fromisoformat(day).date(), cutoff_time, tzinfo=zone)
        for day in (dates[0], dates[-1])
    )
    return first, last


def _cutoff_problem(value: object) -> str | None:
    """Why *value* is not an aware cutoff inside the 2026 cadence dates, or None."""
    if not isinstance(value, str):
        return f"cutoff_utc is missing or not text: {value!r}"
    try:
        cutoff = datetime.fromisoformat(value)
    except ValueError:
        return f"cutoff_utc {value!r} does not parse as an ISO instant"
    if cutoff.tzinfo is None:
        return f"cutoff_utc {value!r} is naive; a cutoff is an instant"
    first, last = _cadence_window()
    if not first <= cutoff <= last:
        return (
            f"cutoff_utc {value!r} is outside the pre-registered re-fit dates "
            f"[{first.isoformat()}, {last.isoformat()}]"
        )
    return None


def refit_metadata_problems(metadata: Mapping[str, Any]) -> list[str]:
    """Every way a served model's metadata.json breaks the pre-registered re-fit contract.

    Checks, in order: the recipe id; the cutoff (aware, inside the 2026 cadence dates); the
    calibration season (REFIT_SEASON - 1, D-26); the helper window (TRAINING_START_SEASON to the
    calibration season, exclusive); the record digest (refit.witness). An empty list is a pass.
    """
    problems: list[str] = []
    recipe = metadata.get("refit_recipe")
    if recipe != REFIT_RECIPE_ID:
        problems.append(f"refit_recipe {recipe!r} is not {REFIT_RECIPE_ID!r}")
    cutoff = _cutoff_problem(metadata.get("cutoff_utc"))
    if cutoff is not None:
        problems.append(cutoff)
    calibration = metadata.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    season = calibration.get("season")
    if season != REFIT_SEASON - 1:
        problems.append(f"calibration season {season!r} is not {REFIT_SEASON - 1}")
    helper = calibration.get("helper_seasons")
    expected_helper = list(range(TRAINING_START_SEASON, REFIT_SEASON - 1))
    if helper != expected_helper:
        problems.append(
            f"calibration helper_seasons {helper!r} are not "
            f"{TRAINING_START_SEASON}..{REFIT_SEASON - 2}"
        )
    digest = metadata.get("preregistration_sha256")
    if digest != dict(PREREGISTRATION_SHA256):
        problems.append(
            f"preregistration_sha256 {digest!r} is not the witness's record digest"
        )
    return problems
