"""Replay a ledger row from what was stored when it was decided (LDGR-09, D-02).

WHY REPLAY READS ONLY THE LEDGER
--------------------------------
Gold is rebuilt nightly, so a decision re-run from the lake is a different decision (NF-08). Every
ledger row instead names a content-addressed decision-input snapshot (``forward_ledger.snapshots``)
holding exactly what the decision scored and selected over, and the ledger keeps its own copies of
every artifact the decision was scored by plus the chain-fit record the bet rule read
(``forward_ledger.artifacts_copy``). Replay reads THOSE and nothing else -- never ``artifacts/``,
``outputs/`` or ``data/`` -- so the private backup alone can replay (D-02).

WHAT ONE SNAPSHOT'S REPLAY DOES
-------------------------------
  1. load and verify the snapshot: a changed byte is refused naming the part;
  2. require a ledger copy of every id the rows were stamped with (and the blend's converter), and
     the chain-fit copy whose sha256 the snapshot recorded -- each missing one named;
  3. re-score each target's stored gold rows with the stamped model id from the ledger's copies and
     compare the model output with the stored candidates (``REPLAY_TOLERANCE``), then substitute it;
  4. re-attach the spread-derived market probability WP's second test reads, from the stored
     spread and the ledger's blend copy, compare and substitute it;
  5. run the ONE selection path (``select_weekly_bets`` then ``records_to_bet_list_frame``) over
     the stored candidates and schedule with the bet rule the row's own generation recorded: the
     chain-fit copy is read through ``load_bet_rule_record``, so a row-19 record supplies its
     overlaid fits and the committed edge thresholds, and a re-fit generation's record (Phase 37,
     research Pitfall 10) its own fits -- no row-19 overlay -- and its own edge thresholds;
  6. compare every decision field of every stored row: categoricals exactly, numerics within
     ``REPLAY_TOLERANCE``.

Steps 3 and 4 are ``forward_ledger.rescore``, the one re-score routine the frozen shadow arm also
calls. Selection reads only the stored decision inputs. The closing columns live in a row's
mutable half, which replay never reads, so a closing value cannot move a replayed decision (CLV
report-only).

THE ROW KEY
-----------
Stored rows and every :class:`ReplayResult` are keyed by ``LEDGER_ROW_KEY``, arm included, so a
live and a shadow row of the same game, week and target are two results (34-RESEARCH Pitfall 7).
A re-derived decision has no arm -- the arm belongs to the stored row -- so each stored row is
compared with the re-derived row of its ``(game_id, season, week, target)`` from ITS OWN snapshot.

A row with no snapshot (a migrated pre-verdict row) is ``not_replayable``, never a pass; a VERDICT
row with no snapshot fails. Replay is a CLI (``scripts/replay_ledger.py``); no ``api/`` module may
import it (D-18).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from api.cache import RUN_MODE_FORWARD
from backtest.diagnose import score_deployed_artifacts
from backtest.weekly_bet_list import (
    CANONICAL_TARGETS,
    FrozenChainFitError,
    build_strategies,
    load_bet_rule_record,
    records_to_bet_list_frame,
    select_weekly_bets,
)
from forward_ledger.artifacts_copy import (
    ARTIFACT_COPIES_DIRNAME,
    RECIPE_COPIES_DIRNAME,
)
from forward_ledger.canonical import (
    ENTRY_KIND_ROW,
    IMMUTABLE_COLUMN_TYPES_V1,
    coerce_canonical_value,
)
from forward_ledger.rescore import (
    REPLAY_TOLERANCE,
    Notes,
    Scorer,
    numbers_agree,
    reattach_spread_probability,
    rescore_candidates,
)
from forward_ledger.schema import LEDGER_ROW_KEY, VERDICT_SCOPE_VERDICT
from forward_ledger.snapshots import (
    DecisionSnapshot,
    SnapshotDigestMismatchError,
    SnapshotMissingError,
    load_decision_snapshot,
)
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    read_entries,
    verify_chain,
)
from refit.records import RefitRecordFormatError

__all__ = [
    "CATEGORICAL_FIELDS",
    "NUMERIC_FIELDS",
    "REPLAY_FAIL",
    "REPLAY_NOT_REPLAYABLE",
    "REPLAY_PASS",
    "REPLAY_STATUSES",
    "REPLAY_TOLERANCE",
    "ReplayInputError",
    "ReplayResult",
    "replay_ledger",
    "replay_snapshot_rows",
]

# The decision fields a replay compares. Categoricals must be equal; numerics within tolerance.
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "bet_side",
    "status",
    "rejection_reason",
    "ev_tier",
    "eligibility_label",
    "snapshot_ts",
    "freeze_ts",
)
NUMERIC_FIELDS: tuple[str, ...] = (
    "model_value",
    "market_value",
    "line",
    "slipped_line",
    "calibrated_p_side",
    "per_bet_ev",
    "stake_units",
    "selected_odds",
    "flat_stake",
)

REPLAY_PASS = "pass"
REPLAY_FAIL = "fail"
REPLAY_NOT_REPLAYABLE = "not_replayable"
REPLAY_STATUSES: tuple[str, ...] = (REPLAY_PASS, REPLAY_FAIL, REPLAY_NOT_REPLAYABLE)

# The identity of a RE-DERIVED decision row: ``LEDGER_ROW_KEY`` without the arm, which a decision
# does not carry (the arm belongs to the stored row). Used only to look a stored row's decision up.
_DECISION_KEY: tuple[str, ...] = tuple(name for name in LEDGER_ROW_KEY if name != "arm")


class ReplayInputError(Exception):
    """A stored input replay needs is absent from the ledger or does not match its record.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a missing input must never be
    degraded into a pass.
    """


@dataclass(frozen=True)
class ReplayResult:
    """One stored row's replay verdict.

    Attributes:
        key: The row's ``LEDGER_ROW_KEY`` values.
        status: ``pass``, ``fail`` or ``not_replayable``.
        mismatches: The decision fields whose replayed value differs from the stored one.
        reason: Why the row did not pass (None on a pass).
    """

    key: tuple[Any, ...]
    status: str
    mismatches: tuple[str, ...] = ()
    reason: str | None = None


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in LEDGER_ROW_KEY)


def _single_id(values: set[Any], fallback: Any, what: str) -> str:
    """The one id the rows were stamped with, else the snapshot's resolved id."""
    stamped = {value for value in values if value is not None}
    if len(stamped) > 1:
        msg = f"the rows of one snapshot carry several {what} stamps {sorted(stamped)}"
        raise ReplayInputError(msg)
    chosen = next(iter(stamped), fallback)
    if not chosen:
        msg = f"no {what} is stamped on the rows or recorded by the snapshot"
        raise ReplayInputError(msg)
    return str(chosen)


def _scoring_ids(
    rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> dict[str, str]:
    """The model id per target, the blend id and the converter id replay must load.

    The rows' stamped ids are used. A target with no row among *rows* (a filtered replay) is still
    scored, because the week's selection pools every target's stakes; its id then comes from the
    snapshot's own resolved record.
    """
    resolved = meta.get("resolved") or {}
    ids = {
        target: _single_id(
            {row["model_artifact_id"] for row in rows if row["target"] == target},
            resolved.get(target),
            f"{target} model id",
        )
        for target in CANONICAL_TARGETS
    }
    ids["blend"] = _single_id(
        {row["blend_id"] for row in rows}, resolved.get("blend"), "blend id"
    )
    converter = resolved.get("converter")
    if converter:
        ids["converter"] = str(converter)
    return ids


def _require_copies(copies: Path, ids: Iterable[str]) -> None:
    missing = sorted(
        {artifact_id for artifact_id in ids if not (copies / artifact_id).is_dir()}
    )
    if missing:
        msg = (
            f"the ledger holds no copy of artifact(s) {missing} under {copies.as_posix()}; "
            "replay reads only the ledger's own copies (D-02)"
        )
        raise ReplayInputError(msg)


def _chain_fit_copy(ledger: Path, meta: Mapping[str, Any]) -> Path:
    """The ledger's copy of the chain-fit record the snapshot recorded, verified by sha256."""
    sha = str(meta.get("chain_fit_sha256") or "")
    path = ledger / RECIPE_COPIES_DIRNAME / f"{sha}.json"
    if not sha or not path.is_file():
        msg = (
            f"the ledger holds no copy of the chain-fit record {sha or '<unrecorded>'} under "
            f"{(ledger / RECIPE_COPIES_DIRNAME).as_posix()}"
        )
        raise ReplayInputError(msg)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        msg = f"the ledger's chain-fit copy {path.name} no longer hashes to its recorded sha256"
        raise ReplayInputError(msg)
    return path


def _rederive(
    ledger: Path,
    snapshot: DecisionSnapshot,
    rows: Sequence[Mapping[str, Any]],
    scorer: Scorer,
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], Notes]:
    """The snapshot's decision re-derived from the ledger alone, keyed by ``_DECISION_KEY``.

    Returns:
        ``(decision rows by (game_id, season, week, target), candidate notes by (game_id,
        target))`` -- a note records a stored model input the re-scoring did not reproduce.
    """
    copies = ledger / ARTIFACT_COPIES_DIRNAME
    ids = _scoring_ids(rows, snapshot.meta)
    _require_copies(copies, ids.values())
    fits, thresholds = load_bet_rule_record(_chain_fit_copy(ledger, snapshot.meta))

    notes: Notes = {}
    candidates = snapshot.frames["candidates"].copy()
    if not candidates.empty:
        rescore_candidates(
            candidates,
            {target: snapshot.frames[f"gold_{target}"] for target in CANONICAL_TARGETS},
            copies,
            ids,
            scorer,
            notes,
        )
        candidates = reattach_spread_probability(
            candidates, copies, ids["blend"], notes
        )

    result = select_weekly_bets(
        candidates,
        snapshot.frames["schedule"],
        fits,
        strategies=build_strategies(fits),
        edge_thresholds=thresholds,
    )
    # The rows' own instant, so the re-derived frame stamps what the stored rows carry. It feeds
    # only ``decided_at_utc``, which no compared field depends on.
    decided_at = next(
        (row["decided_at_utc"] for row in rows if row["decided_at_utc"]), None
    )
    frame = records_to_bet_list_frame(
        result,
        fits,
        run_mode=RUN_MODE_FORWARD,
        decided_at=None if decided_at is None else datetime.fromisoformat(decided_at),
    )
    derived = {
        tuple(
            coerce_canonical_value(name, IMMUTABLE_COLUMN_TYPES_V1[name], record[name])
            for name in _DECISION_KEY
        ): record
        for record in frame.to_dict(orient="records")
    }
    return derived, notes


def _compare_row(
    row: Mapping[str, Any],
    derived: Mapping[tuple[Any, ...], Mapping[str, Any]],
    notes: Mapping[tuple[str, str], list[str]],
) -> ReplayResult:
    key = _row_key(row)
    replayed = derived.get(tuple(row[name] for name in _DECISION_KEY))
    if replayed is None:
        return ReplayResult(
            key,
            REPLAY_FAIL,
            reason="the re-derived decision has no row for this game, week and target",
        )

    mismatches: list[str] = []
    details: list[str] = []
    for name in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS):
        stored = row[name]
        again = coerce_canonical_value(
            name, IMMUTABLE_COLUMN_TYPES_V1[name], replayed[name]
        )
        agree = (
            numbers_agree(stored, again) if name in NUMERIC_FIELDS else stored == again
        )
        if not agree:
            mismatches.append(name)
            details.append(f"{name} stored={stored!r} replayed={again!r}")
    details += notes.get((str(row["game_id"]), str(row["target"])), [])
    if not details:
        return ReplayResult(key, REPLAY_PASS)
    return ReplayResult(key, REPLAY_FAIL, tuple(mismatches), "; ".join(details))


def replay_snapshot_rows(
    ledger_dir: Path | str,
    digest: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    scorer: Scorer = score_deployed_artifacts,
) -> list[ReplayResult]:
    """Replay *rows* (stored immutable halves) that all reference the snapshot *digest*.

    Args:
        ledger_dir: The ledger directory -- the only store read.
        digest: The decision-snapshot digest every row names.
        rows: The rows' canonical immutable values.
        scorer: The model scorer (``backtest.diagnose.score_deployed_artifacts``).

    Returns:
        One result per row, in *rows* order. A missing or tampered snapshot, a missing artifact or
        chain-fit copy fails every row with the reason named.
    """
    ledger = Path(ledger_dir)
    try:
        snapshot = load_decision_snapshot(ledger, digest)
        derived, notes = _rederive(ledger, snapshot, rows, scorer)
    except SnapshotDigestMismatchError as error:
        reason = f"snapshot digest mismatch in {digest}: part {error.part!r}: {error}"
    except SnapshotMissingError as error:
        reason = f"decision snapshot {digest} is missing: {error}"
    except (ReplayInputError, FrozenChainFitError, RefitRecordFormatError) as error:
        reason = str(error)
    else:
        return [_compare_row(row, derived, notes) for row in rows]
    return [ReplayResult(_row_key(row), REPLAY_FAIL, reason=reason) for row in rows]


def _unreplayable(row: Mapping[str, Any]) -> ReplayResult:
    """A row with no decision snapshot: never a pass, and a failure for a verdict row."""
    if row["verdict_scope"] == VERDICT_SCOPE_VERDICT:
        return ReplayResult(
            _row_key(row),
            REPLAY_FAIL,
            reason="a verdict row carries no decision snapshot, so it cannot be replayed",
        )
    return ReplayResult(
        _row_key(row),
        REPLAY_NOT_REPLAYABLE,
        reason="no decision snapshot was stored for this pre-verdict row",
    )


def replay_ledger(
    ledger_dir: Path | str = LEDGER_DIR,
    *,
    season: int | None = None,
    week: int | None = None,
    game_ids: Iterable[str] | None = None,
) -> list[ReplayResult]:
    """Replay every ledger row (optionally one season, week or set of games), in file order.

    Raises:
        LedgerFormatError: the store cannot be read.
        LedgerChainBrokenError: the chain does not verify -- replay never runs on an unverified
            ledger.
    """
    ledger = Path(ledger_dir)
    entries = read_entries(ledger)
    verdict = verify_chain(entries)
    if not verdict.ok:
        msg = (
            f"the ledger's chain is broken at seq {verdict.first_broken_seq} (key "
            f"{verdict.first_broken_key}): {verdict.reason}. Replay never runs on an unverified "
            "ledger."
        )
        raise LedgerChainBrokenError(msg)

    wanted_games = None if game_ids is None else {str(game_id) for game_id in game_ids}
    rows = [
        entry.immutable
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW
        and (season is None or entry.immutable["season"] == season)
        and (week is None or entry.immutable["week"] == week)
        and (wanted_games is None or entry.immutable["game_id"] in wanted_games)
    ]

    by_digest: dict[str, list[Mapping[str, Any]]] = {}
    results: dict[tuple[Any, ...], ReplayResult] = {}
    for row in rows:
        digest = row["decision_snapshot_digest"]
        if digest is None:
            results[_row_key(row)] = _unreplayable(row)
        else:
            by_digest.setdefault(digest, []).append(row)
    for digest, group in by_digest.items():
        for result in replay_snapshot_rows(ledger, digest, group):
            results[result.key] = result
    return [results[_row_key(row)] for row in rows]
