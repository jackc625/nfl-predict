"""The ONE place the re-fit writes ``artifacts/latest.json`` (RFIT-06, SPEC R5, research Pitfall 2).

THE SWAP SURFACE
----------------
A generation that passed its integrity checks (``refit.integrity``) goes live through
``models.artifacts.replace_manifest`` and nothing else: it validates every artifact through the
serving loaders and the blend's own provenance (its source ids must equal the three new model ids)
and then writes the four pointers -- wp, ats, ou, blend -- in one atomic, fsynced write. Gate-free
by owner ruling: no deploy-gate module, gate file or promotion script is called here.

THE INTENT IS RECORDED FIRST
----------------------------
Before the swap, the ``install_intent`` (the five ids, the bet-rule record's path and sha, the
attempt's code commit) is appended DURABLY to ``ledger/refit/dates.jsonl``. A crash between the
swap and the date's closing record then leaves ``latest.json`` serving ids an intent names, so the
next decision run can stamp them and the next hook run can close the date without re-training
(:func:`pending_install`). A declined intent write refuses the install: a served manifest no
record explains would be unexplainable after a power cut.

HOW THE HOOK CLOSES THE DATE (Plan 37-13)
-----------------------------------------
:func:`install_generation` returning normally IS the ``installed`` outcome: the swap wrote. The
hook then records the ``installed`` attempt and the date's ``closed`` event (``ran`` or
``ran late``) in the same run, because a date with an installed attempt but no closed event is
still open (``refit.cadence``). :func:`pending_install` finds the date a crash left between those
writes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.artifacts import (
    BUNDLE_MANIFEST_KEYS,
    ArtifactBundleInvalidError,
    replace_manifest,
    resolve_production_artifacts,
)
from refit.cadence import CadenceDate
from refit.integrity import IntegrityRefusal
from refit.records import (
    append_date_event,
    bet_rule_record_path,
    bet_rule_relpath,
    read_bet_rule_record,
    read_date_events,
)

__all__ = [
    "InstallIntentNotRecordedError",
    "install_generation",
    "pending_install",
]

# The five ids an install names: the four manifest pointers and the converter the blend is bound to.
_ID_KEYS: tuple[str, ...] = ("wp", "ats", "ou", "blend", "converter")

_CHECK_CLEAN_CODE_COMMIT: str = "clean_code_commit"
_CHECK_BUNDLE_VALIDATION: str = "bundle_validation"

_DIRTY_SUFFIX: str = "+dirty"
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class InstallIntentNotRecordedError(Exception):
    """The write sink declined the ``install_intent``, so the install never proceeds.

    An install whose intent is not on disk would be unexplainable after a power cut: ``latest.json``
    would serve ids no record names. Inherits bare ``Exception`` so no broad handler degrades it.
    """


def _code_commit_problem(code_commit: str | None) -> str | None:
    """Why *code_commit* cannot be written on the permanent install intent, or None."""
    if code_commit is None:
        return "the attempt's code commit is unknown (git could not name it)"
    if code_commit.endswith(_DIRTY_SUFFIX):
        return (
            f"the attempt's code commit {code_commit!r} is dirty: the tracked code has "
            "uncommitted changes, so the code that would install is in no commit"
        )
    if not _FULL_COMMIT.match(code_commit):
        return f"the attempt's code commit {code_commit!r} is not a full 40-hex commit"
    return None


def _install_ids(ids: Mapping[str, str]) -> dict[str, str]:
    """The five ids as text; a missing or empty one is refused before anything is written."""
    missing = [key for key in _ID_KEYS if not ids.get(key)]
    if missing:
        msg = f"an install names all of {list(_ID_KEYS)}; {missing} are missing"
        raise ValueError(msg)
    return {key: str(ids[key]) for key in _ID_KEYS}


def _bound_converter(artifacts_dir: Path, blend: str) -> str | None:
    """The converter the blend *blend* is bound to, read from its own payload."""
    # Imported lazily, as models.artifacts does: the blender pulls in pandas and scipy.
    from models.blending import MarketBlender

    return MarketBlender.from_artifacts(
        artifacts_dir, version=blend
    ).market_probability_artifact_id


def install_generation(
    ledger_dir: Path | str,
    artifacts_dir: Path | str,
    *,
    date: CadenceDate,
    ids: Mapping[str, str],
    bet_rule_sha256: str,
    code_commit: str | None,
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Record the install intent, then install the four pointers through ``replace_manifest``.

    The order is the point of the function:

    1. ``clean_code_commit`` (pre-registered): an unknown or dirty code commit is refused before
       anything is read or written, so the permanent intent -- whose code commit the first
       re-fit's ancestry test reads -- always names a real commit. The date stays open and the
       next daily run retries it with the same cutoff.
    2. The bet-rule record is read through the sha-verifying reader, so a missing or tampered
       record raises before anything is written; it must name these ids.
    3. The new blend's bound converter must be the converter the install names, or the served ids
       would name no generation (``bundle_validation``).
    4. The served ids are read once (the ``before`` ids).
    5. The ``install_intent`` is appended durably; a declined write raises
       :class:`InstallIntentNotRecordedError`.
    6. ``replace_manifest`` validates and writes the four pointers; its refusal is
       ``bundle_validation``.

    Args:
        ledger_dir: The ledger directory (``ledger/refit/`` holds the records).
        artifacts_dir: The artifacts root holding ``latest.json`` and the new artifacts.
        date: The cadence date being installed.
        ids: wp, ats, ou, blend and converter of the new generation.
        bet_rule_sha256: The sha256 naming the generation's bet-rule record (never a path).
        code_commit: The code commit the hook captured once for this attempt.

    Returns:
        ``(before, after)``: the five ids served before, and the four new pointers plus the
        bound converter.

    Raises:
        IntegrityRefusal: ``clean_code_commit`` or ``bundle_validation``.
        InstallIntentNotRecordedError: the write sink declined the intent.
        FileNotFoundError: the bet-rule record does not exist.
        refit.records.RefitRecordFormatError: the bet-rule record no longer hashes to its name.
        ValueError: *ids* lacks an id, the sha is malformed, or the record names other ids.
    """
    problem = _code_commit_problem(code_commit)
    if problem is not None:
        raise IntegrityRefusal(_CHECK_CLEAN_CODE_COMMIT, problem)

    ledger = Path(ledger_dir)
    artifacts = Path(artifacts_dir)
    new_ids = _install_ids(ids)
    record = read_bet_rule_record(bet_rule_record_path(ledger, bet_rule_sha256))
    if record.get("artifact_ids") != new_ids:
        msg = (
            f"the bet-rule record {bet_rule_sha256} was measured for "
            f"{record.get('artifact_ids')}, not for the ids being installed {new_ids}"
        )
        raise ValueError(msg)

    try:
        bound = _bound_converter(artifacts, new_ids["blend"])
    except Exception as exc:  # re-raised by name below, never swallowed
        msg = f"the new blend {new_ids['blend']!r} does not load: {type(exc).__name__}: {exc}"
        raise IntegrityRefusal(_CHECK_BUNDLE_VALIDATION, msg) from exc
    if bound != new_ids["converter"]:
        msg = (
            f"the new blend {new_ids['blend']!r} is bound to converter {bound!r}, but the "
            f"install names {new_ids['converter']!r}: the served ids would name no generation"
        )
        raise IntegrityRefusal(_CHECK_BUNDLE_VALIDATION, msg)

    before: dict[str, str | None] = dataclasses.asdict(
        resolve_production_artifacts(artifacts)
    )
    intent: dict[str, Any] = {
        "event": "install_intent",
        "list_week": date.list_week,
        "date_index": date.index,
        "date": date.date.isoformat(),
        "nfl_week": date.nfl_week,
        "ids": new_ids,
        "bet_rule_path": bet_rule_relpath(bet_rule_sha256),
        "bet_rule_sha256": bet_rule_sha256,
        "code_commit": code_commit,
    }
    if not append_date_event(ledger, intent):
        msg = (
            f"the install intent of cadence date {date.date.isoformat()} was not written (the "
            "write sink declined it), so latest.json was not touched"
        )
        raise InstallIntentNotRecordedError(msg)

    pointers = {key: new_ids[key] for key in BUNDLE_MANIFEST_KEYS}
    try:
        replace_manifest(pointers, artifacts)
    except ArtifactBundleInvalidError as exc:
        raise IntegrityRefusal(_CHECK_BUNDLE_VALIDATION, str(exc)) from exc

    after: dict[str, str | None] = {**pointers, "converter": bound}
    return before, after


def pending_install(ledger_dir: Path | str, artifacts_dir: Path | str) -> int | None:
    """The date index of an open date whose last install intent ``latest.json`` serves, or None.

    "Open" is a date with no ``closed`` event. Whatever attempt follows the intent -- none,
    ``installed`` or ``crashed`` -- is NOT asked about: a crash can come after
    ``replace_manifest`` wrote, and ``latest.json`` naming the intent's freshly trained ids is the
    proof the swap happened (research Pitfall 2). ``latest.json`` is read only when an open date
    has an intent.

    Raises:
        refit.records.RefitRecordFormatError: ``dates.jsonl`` holds a malformed line.
    """
    by_date: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for event in read_date_events(ledger_dir):
        key = (int(event["list_week"]), int(event["date_index"]))
        by_date.setdefault(key, []).append(event)

    candidates: list[tuple[int, dict[str, Any]]] = []
    for (_list_week, date_index), events in by_date.items():
        if any(event["event"] == "closed" for event in events):
            continue
        intents = [event for event in events if event["event"] == "install_intent"]
        if intents:
            candidates.append((date_index, intents[-1]))
    if not candidates:
        return None

    served = dataclasses.asdict(resolve_production_artifacts(Path(artifacts_dir)))
    for date_index, intent in candidates:
        named = {key: intent["ids"].get(key) for key in _ID_KEYS}
        if named == served:
            return date_index
    return None
