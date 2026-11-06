"""What the live ``artifacts/latest.json`` must hold, read from the records (Phase 37, Plan 37-20).

WHY THE LIVE-MANIFEST TESTS FOLLOW A RECORD AND NOT A HARD-CODED SWAP
----------------------------------------------------------------------
Until Phase 37 the live manifest moved only by a hand-run, recorded swap, and each swap re-pointed
by hand the tests that pin it (row 19 last, 2026-10-04). From Tue 2026-11-03 the daily run installs
re-fit generations unattended (D-01, RFIT-06), recording each install in
``ledger/refit/dates.jsonl`` BEFORE the swap writes ``latest.json`` (the install_intent, Plan 37-11,
research Pitfall 2). So the recorded end state is the newest generation
``refit.generations.installed_generations`` derives -- the one derivation; nothing here re-decides
which intents count -- with ONE exception: a newest install that recorded no end and whose four ids
``latest.json`` does not serve is not yet in force, because its run died between the intent and the
swap. A cadence install records its end with its date's ``closed`` event; the owner install of the
v2 models (MIDSEASON-SWITCH-RECORD.md section 4, ``ledger/refit/owner_installs.jsonl``) with its
``installed`` event naming its ids, so a hand roll-back away from v2 after that event is never read
as the previous generation being in force. Before the first install the end state is generation 0,
the row-19 swap's record (``tests.phase33_state.ROW19_SWAP_ARTIFACT_IDS`` and
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

import dataclasses
import json
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backtest.midseason_switch_2026 import INSTALLED_RECIPE
from backtest.refit_preregistration_2026 import (
    CUTOFF_LOCAL_TIME,
    CUTOFF_TIMEZONE,
    REFIT_DATE_LISTS,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    TRAINING_START_SEASON,
)
from models.v2_recipe import V2_INPUTS
from refit.generations import Generation, installed_generations
from refit.owner_install import switch_record_sha256
from refit.records import read_date_events, read_owner_installs
from refit.witness import PREREGISTRATION_SHA256
from tests.phase33_state import ROW19_SWAP_ARTIFACT_IDS

__all__ = [
    "LIVE_LATEST_JSON",
    "LIVE_LEDGER_DIR",
    "MANIFEST_KEYS",
    "REPO_ROOT",
    "expected_cutoff",
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


def _owner_installed(events: list[dict[str, Any]], generation: Generation) -> bool:
    """Whether an owner ``installed`` event names exactly *generation*'s five ids."""
    ids = dataclasses.asdict(generation.ids)
    return any(
        event["event"] == "installed" and event["after_ids"] == ids for event in events
    )


def _install_ended(ledger_dir: Path, generation: Generation) -> bool:
    """Whether *generation*'s install recorded its end (generation >= 1).

    An owner install (it carries its own cutoff) ends with its ``installed`` event; a cadence
    install with its date's ``closed`` event.
    """
    if generation.cutoff_utc is not None:
        return _owner_installed(read_owner_installs(ledger_dir), generation)
    return _date_closed(read_date_events(ledger_dir), generation)


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
    if newest.number >= 1 and not _install_ended(ledger_dir, newest):
        if not _serves(latest_path, _manifest_ids(newest)):
            # The intent is written before replace_manifest (Plan 37-11, and the owner command
            # alike): a run that died between them leaves the intent counted but never served,
            # so it is not in force.
            newest = next(g for g in generations if g.number == newest.number - 1)
    if newest.number == 0:
        return 0, dict(ROW19_SWAP_ARTIFACT_IDS)
    return newest.number, _manifest_ids(newest)


def expected_manifest_text(ids: Mapping[str, str]) -> str:
    """The text ``models.artifacts._atomic_write_json`` writes for *ids*, before newline translation."""
    return json.dumps({key: ids[key] for key in MANIFEST_KEYS}, indent=2)


def expected_cutoff(generation: Generation) -> datetime:
    """The cutoff *generation*'s models must record: its cadence date at the record's cutoff time.

    Read from the pre-registered date list the generation's install names (list week, date
    index), at ``CUTOFF_LOCAL_TIME`` in ``CUTOFF_TIMEZONE`` -- never derived from a model. An
    owner install has no cadence date: its cutoff is the one its ``install_intent`` records
    (``Generation.cutoff_utc``).
    """
    if generation.cutoff_utc is not None:
        return generation.cutoff_utc
    if generation.list_week is None or generation.date_index is None:
        msg = f"generation {generation.number} names no cadence date; it has no re-fit cutoff"
        raise ValueError(msg)
    day, _nfl_week = REFIT_DATE_LISTS[generation.list_week][generation.date_index - 1]
    return datetime.combine(
        date.fromisoformat(day),
        time.fromisoformat(CUTOFF_LOCAL_TIME),
        tzinfo=ZoneInfo(CUTOFF_TIMEZONE),
    )


def _cutoff_problem(value: object, expected: datetime) -> str | None:
    """Why *value* is not exactly the *expected* cutoff instant, or None."""
    if not isinstance(value, str):
        return f"cutoff_utc is missing or not text: {value!r}"
    try:
        cutoff = datetime.fromisoformat(value)
    except ValueError:
        return f"cutoff_utc {value!r} does not parse as an ISO instant"
    if cutoff.tzinfo is None:
        return f"cutoff_utc {value!r} is naive; a cutoff is an instant"
    if cutoff != expected:
        return (
            f"cutoff_utc {value!r} is not the served generation's recorded cutoff "
            f"{expected.isoformat()}"
        )
    return None


def _v2_metadata_problems(
    metadata: Mapping[str, Any], generation: Generation
) -> list[str]:
    """Every way a served v2 model's metadata.json breaks the switch record's install contract.

    Checks, in order: the recipe (``v2_recipe``); the cutoff (exactly the one the owner install
    recorded); the inputs (exactly ``V2_INPUTS`` of the metadata's target); the switch record's
    two-file digest (``refit.owner_install.switch_record_sha256``).
    """
    problems: list[str] = []
    recipe = metadata.get("v2_recipe")
    if recipe != INSTALLED_RECIPE:
        problems.append(f"v2_recipe {recipe!r} is not {INSTALLED_RECIPE!r}")
    cutoff = _cutoff_problem(metadata.get("cutoff_utc"), expected_cutoff(generation))
    if cutoff is not None:
        problems.append(cutoff)
    target = metadata.get("target")
    names = metadata.get("feature_names")
    inputs = V2_INPUTS.get(target) if isinstance(target, str) else None
    if inputs is None:
        problems.append(f"target {target!r} is not one of {list(V2_INPUTS)}")
    elif names != list(inputs):
        problems.append(
            f"feature_names {names!r} are not the v2 inputs of {target!r} {list(inputs)}"
        )
    digest = metadata.get("switch_record_sha256")
    expected_digest = switch_record_sha256(REPO_ROOT)
    if digest != expected_digest:
        problems.append(
            f"switch_record_sha256 {digest!r} is not the switch record's digest "
            f"{expected_digest!r}"
        )
    return problems


def refit_metadata_problems(
    metadata: Mapping[str, Any], generation: Generation
) -> list[str]:
    """Every way a served model's metadata.json breaks the contract of *generation*'s recipe.

    A generation serving the v2 recipe (``INSTALLED_RECIPE``, the owner install of
    MIDSEASON-SWITCH-RECORD.md section 4) is checked by :func:`_v2_metadata_problems`. Every other
    generation by the pre-registered re-fit contract, in order: the recipe id; the cutoff (exactly
    *generation*'s pre-registered cutoff, :func:`expected_cutoff`, so a re-install of an earlier
    generation's models is caught); the calibration season (REFIT_SEASON - 1, D-26); the helper
    window (TRAINING_START_SEASON to the calibration season, exclusive); the record digest
    (refit.witness). An empty list is a pass.
    """
    if generation.recipe == INSTALLED_RECIPE:
        return _v2_metadata_problems(metadata, generation)
    problems: list[str] = []
    recipe = metadata.get("refit_recipe")
    if recipe != REFIT_RECIPE_ID:
        problems.append(f"refit_recipe {recipe!r} is not {REFIT_RECIPE_ID!r}")
    cutoff = _cutoff_problem(metadata.get("cutoff_utc"), expected_cutoff(generation))
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
