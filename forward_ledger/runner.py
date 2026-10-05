"""The ONE place the daily run talks to the ledger (Phase 34, Plan 34-15).

Three entry points, each called by ``scripts/daily_lock_pipeline.py`` / ``pipeline/daily_steps.py``
once the committed cutover switch (``forward_ledger.cutover``) sends forward rows here:

:func:`record_forward_slate` -- the forward write, replacing ``generate_weekly_bet_list`` on the
daily run's critical recommend step. In order:

  1. resolve the production artifact ids ONCE (``models.artifacts.resolve_production_artifacts``)
     and build the week's decision bundle with them, so a ``latest.json`` swap mid-run cannot
     mis-stamp a row (Plan 34-05);
  2. compute the reproduction key and the decision-snapshot digest IN MEMORY (Plan 34-08);
  3. stamp the rows at the one stamping site (``forward_ledger.stamps``);
  4. first pick stands (D-13, SPEC LDGR-01): an identical repeat is skipped; a row that reached a
     DIFFERENT pick for a stored key is refused, logged by name (``first_pick_refusal``) and
     printed (``LEDGER_REFUSED=``) -- the run does not fail, and new keys still append;
  5. nothing new -> NOTHING is written: no snapshot, no copy, no ledger write, no anchor commit;
  6. otherwise store the snapshot and the artifact and chain-fit copies (written BEFORE the row
     that references them, and idempotent), then ONE ``commit_changes`` with the slate's
     ``publish_by`` deadline, which re-reads the clock immediately before the replace.

:func:`settle_ledger` -- every day the schedule refresh returns a season: grade pending rows from
the refreshed silver scores, append the corrections owed by the live manifest's
``live_revision_graded`` verdicts, and finalize closing columns -- all in ONE ``commit_changes``,
so the three halves land together or not at all (34-RESEARCH Q2, Pitfall 11).

:func:`sync_after_run` -- after the run, anchor the head and back up the directory through
``forward_ledger.sync.publish_ledger_state``. It NEVER raises (LDGR-05, D-01).

Every event goes through ``forward_ledger.run_log.record_event`` (or an injected ``log``), so each
carries the daily run's bound ``run_id``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from backtest.weekly_bet_list import build_weekly_decision_bundle
from data.upstream_live import load_live_manifest
from forward_ledger.artifacts_copy import (
    ensure_artifact_copies,
    ensure_recipe_record_copy,
)
from forward_ledger.canonical import ENTRY_KIND_ROW, IMMUTABLE_COLUMNS_V1
from forward_ledger.closing import finalize_closing
from forward_ledger.corrections import correction_payloads, owed_correction_events
from forward_ledger.declarations import (
    VerdictScope,
    VerdictScopeUndeclaredError,
    load_verdict_scope,
)
from forward_ledger.repro_key import build_repro_key
from forward_ledger.run_log import read_events, record_event
from forward_ledger.schema import VERDICT_SCOPE_VERDICT
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    grading_updates,
    live_strategies,
    load_silver_games,
    realized_values_from_scores,
)
from forward_ledger.snapshots import compute_snapshot_digest, write_decision_snapshot
from forward_ledger.stamps import stamp_ledger_rows
from forward_ledger.store import (
    LEDGER_DIR,
    LEDGER_SEASON,
    LedgerEntry,
    MissingStampError,
    classify_incoming,
    commit_changes,
    read_entries,
)
from forward_ledger.sync import SyncOutcome, publish_ledger_state
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from pipeline.daily_steps import DailySlate

__all__ = [
    "SILVER_ODDS_FILENAME",
    "RecordOutcome",
    "SettleOutcome",
    "record_forward_slate",
    "settle_ledger",
    "sync_after_run",
]

logger = get_logger(__name__)

SILVER_ODDS_FILENAME: str = "odds_snapshot.parquet"

LogCallable = Callable[..., Any]


@dataclass(frozen=True)
class RecordOutcome:
    """What one :func:`record_forward_slate` call did.

    Attributes:
        decided: Rows the decision produced (live and suppressed).
        appended: New keys written to the ledger.
        identical: Repeats of stored rows, skipped.
        refused: ``(key, differing identity columns)`` for every conflicting row (first pick
            stands); the stored rows are unchanged.
        wrote: Whether the ledger file was replaced.
        head_hash: The ledger head after the call (None when nothing was decided).
        entry_count: The ledger entry count after the call (None when nothing was decided).
    """

    decided: int
    appended: int
    identical: int
    refused: tuple[tuple[tuple[Any, ...], tuple[str, ...]], ...]
    wrote: bool
    head_hash: str | None
    entry_count: int | None


@dataclass(frozen=True)
class SettleOutcome:
    """What one :func:`settle_ledger` pass did. ``changed`` is True when the ledger was written."""

    changed: bool
    graded: int
    corrections: int
    closing_finalized: int
    observations: int


def _silver_odds(silver_dir: Path | str) -> pd.DataFrame:
    """The accumulating silver odds store, READ ONLY.

    Raises:
        FileNotFoundError: the store is absent. Never an empty frame: "no odds" read as "no
            capture" would stamp an odds digest over nothing and finalize every closing column
            as ``capture_missed`` -- a one-way write.
    """
    path = Path(silver_dir) / SILVER_ODDS_FILENAME
    if not path.exists():
        msg = f"the silver odds store {path.as_posix()} does not exist"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def _resolve_once(artifacts_dir: Path) -> Any:
    """The production ids, read ONCE from ``latest.json``; None when there is no manifest."""
    if not (artifacts_dir / "latest.json").exists():
        return None
    from models.artifacts import resolve_production_artifacts

    return resolve_production_artifacts(artifacts_dir)


def _verdict_scope_or_none() -> VerdictScope | None:
    """The committed declaration, or None while none exists (a malformed one still raises)."""
    try:
        return load_verdict_scope()
    except VerdictScopeUndeclaredError:
        return None


def _has_verdict_rows(entries: Sequence[LedgerEntry]) -> bool:
    return any(
        entry.kind == ENTRY_KIND_ROW
        and entry.immutable.get("verdict_scope") == VERDICT_SCOPE_VERDICT
        for entry in entries
    )


def _key_text(key: Sequence[Any]) -> str:
    return " ".join(str(value) for value in key)


def record_forward_slate(
    slate: DailySlate,
    *,
    decided_at: datetime,
    excluded_game_ids: frozenset[str],
    publish_by: datetime | None,
    ledger_dir: Path | str = LEDGER_DIR,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    manifest_dir: Path | str | None = None,
    bundle_builder: Callable[..., Any] = build_weekly_decision_bundle,
    log: LogCallable | None = None,
) -> RecordOutcome:
    """Decide the slate's week and write its new forward rows to the ledger, first pick standing.

    Args:
        slate: The daily slate (its season and week are decided).
        decided_at: The decision instant -- the rows' ``decided_at_utc``.
        excluded_game_ids: Games the run will not bet (the live skip and every week game outside
            tonight's slate), dropped before selection.
        publish_by: Nothing is written once the clock passes it (the slate's lock); None when
            the slate has no game in scope.
        ledger_dir, artifacts_dir, gold_dir, silver_dir: The stores read and written.
        manifest_dir: The live-manifest directory (``config/upstream_live`` when None).
        bundle_builder: The decision seam (``build_weekly_decision_bundle``).
        log: ``(event, **fields)``; ``forward_ledger.run_log.record_event`` when None.

    Returns:
        What was decided, appended, skipped and refused.

    Raises:
        backtest.weekly_bet_list.PublishDeadlinePassedError: the write was ready after
            *publish_by*; nothing was written to the ledger.
        Everything the decision, stamping and ``commit_changes`` refuse, by name.
    """
    emit = log or record_event
    ledger = Path(ledger_dir)
    bundle = bundle_builder(
        slate.season,
        slate.week,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        now=decided_at,
        excluded_game_ids=frozenset(excluded_game_ids),
        resolved=_resolve_once(Path(artifacts_dir)),
    )
    frame: pd.DataFrame = bundle.frame
    if frame.empty:
        return RecordOutcome(0, 0, 0, (), False, None, None)
    if bundle.resolved is None:
        msg = (
            "the decision was made without resolved artifact ids (no latest.json under "
            f"{Path(artifacts_dir).as_posix()}); a ledger row is never stamped without them"
        )
        raise MissingStampError(msg)

    repro = build_repro_key(
        manifest=load_live_manifest(slate.season, manifest_dir=manifest_dir),
        gold_dir=gold_dir,
        odds=_silver_odds(silver_dir),
        game_ids=sorted(frame["game_id"].astype(str).unique()),
        decided_at=decided_at,
    )
    entries = read_entries(ledger)
    stamped = stamp_ledger_rows(
        frame,
        resolved=bundle.resolved,
        repro=repro,
        snapshot_digest=compute_snapshot_digest(bundle),
        scope=_verdict_scope_or_none(),
        ledger_has_verdict_rows=_has_verdict_rows(entries),
    )
    decided = cast("pd.DataFrame", stamped[list(IMMUTABLE_COLUMNS_V1)])
    rows = decided.to_dict(orient="records")
    classification = classify_incoming(entries, rows)

    for key, fields in classification.conflicting:
        print(  # noqa: T201 -- the daily run's console contract names every refused row
            f"LEDGER_REFUSED= {_key_text(key)}: first pick stands "
            f"(differing {', '.join(fields)})"
        )
        emit("first_pick_refusal", key=list(key), fields=list(fields))

    if not classification.new:
        return RecordOutcome(
            decided=len(rows),
            appended=0,
            identical=len(classification.identical),
            refused=classification.conflicting,
            wrote=False,
            head_hash=entries[-1].chain_hash if entries else None,
            entry_count=len(entries),
        )

    snapshot_digest = write_decision_snapshot(ledger, bundle)
    ensure_artifact_copies(ledger, bundle.resolved, Path(artifacts_dir))
    ensure_recipe_record_copy(ledger, bundle.chain_fit_path)
    result = commit_changes(
        ledger, new_rows=list(classification.new), publish_by=publish_by
    )
    if result.wrote:
        emit(
            "append",
            season=slate.season,
            week=slate.week,
            appended_rows=result.appended_rows,
            identical_rows=len(classification.identical),
            refused_rows=len(classification.conflicting),
            snapshot_digest=snapshot_digest,
            head_hash=result.head_hash,
            entry_count=result.entry_count,
        )
    return RecordOutcome(
        decided=len(rows),
        appended=result.appended_rows if result.wrote else 0,
        identical=len(classification.identical),
        refused=classification.conflicting,
        wrote=result.wrote,
        head_hash=result.head_hash,
        entry_count=result.entry_count,
    )


def settle_ledger(
    *,
    now: datetime,
    ledger_dir: Path | str = LEDGER_DIR,
    silver_dir: Path | str = DEFAULT_SILVER_DIR,
    manifest_dir: Path | str | None = None,
    strategies: Mapping[str, Any] | None = None,
    log: LogCallable | None = None,
) -> SettleOutcome:
    """Grade, correct and finalize the ledger from refreshed silver data, in ONE atomic write.

    Args:
        now: The settlement instant (``graded_at`` and ``corrected_at_utc``); tz-aware.
        ledger_dir: The ledger directory.
        silver_dir: The silver store holding ``games.parquet`` and ``odds_snapshot.parquet``.
        manifest_dir: The live-manifest directory (``config/upstream_live`` when None).
        strategies: ``target -> strategy``; the live registry
            (``forward_ledger.settle.live_strategies``) when None.
        log: ``(event, **fields)``; ``forward_ledger.run_log.record_event`` when None.

    Returns:
        What changed. ``changed`` is False -- and the file untouched -- when nothing was owed.

    Raises:
        Every refusal of the loaders and of ``commit_changes``, by name. The daily run records
        the failure and goes on (it never stops the decision run).
    """
    emit = log or record_event
    ledger = Path(ledger_dir)
    entries = read_entries(ledger)
    if not entries:
        return SettleOutcome(False, 0, 0, 0, 0)

    games = load_silver_games(silver_dir)
    odds = _silver_odds(silver_dir)
    realized = realized_values_from_scores(games)
    registry = dict(strategies) if strategies is not None else live_strategies()
    manifest = load_live_manifest(LEDGER_SEASON, manifest_dir=manifest_dir) or {}

    grading = grading_updates(entries, registry, realized, now)
    corrections = correction_payloads(
        entries, owed_correction_events(manifest), realized, registry, now
    )
    closing = finalize_closing(
        entries, games, odds, read_events(event="closing_capture"), registry
    )

    for observation in corrections.observations:
        emit(
            "correction_label_withdrawn",
            key=list(observation["key"]),
            week=observation["week"],
            source=observation["source"],
        )
    observed = len(corrections.observations)
    if not (grading or corrections.payloads or closing):
        return SettleOutcome(False, 0, 0, 0, observed)

    result = commit_changes(
        ledger,
        grading_updates=grading,
        new_corrections=corrections.payloads,
        closing_updates=closing,
    )
    if result.wrote:
        emit(
            "settle",
            graded=result.grading_changed,
            corrections=result.appended_corrections,
            closing_finalized=result.closing_changed,
            head_hash=result.head_hash,
            entry_count=result.entry_count,
        )
        for payload in corrections.payloads:
            emit(
                "correction_appended",
                key=[
                    payload["game_id"],
                    payload["season"],
                    payload["week"],
                    payload["target"],
                    payload["arm"],
                ],
                prior_grading_status=payload["prior_grading_status"],
                corrected_grading_status=payload["corrected_grading_status"],
                source_sequence=payload["source_sequence"],
            )
        for key, update in closing.items():
            emit(
                "closing_finalized",
                key=list(key),
                closing_null_reason=update.get("closing_null_reason"),
                forward_clv=update.get("forward_clv"),
            )
    return SettleOutcome(
        changed=result.wrote,
        graded=result.grading_changed if result.wrote else 0,
        corrections=result.appended_corrections if result.wrote else 0,
        closing_finalized=result.closing_changed if result.wrote else 0,
        observations=observed,
    )


def sync_after_run(
    *,
    ledger_dir: Path | str = LEDGER_DIR,
    repo_dir: Path | str = Path(),
    log: LogCallable | None = None,
    publisher: Callable[..., SyncOutcome] | None = None,
) -> SyncOutcome | None:
    """Anchor the head and back up the ledger after the run. NEVER raises (LDGR-05, D-01).

    Returns:
        The publisher's outcome, or None when it raised (recorded as ``sync_refused``).
    """
    emit = log or record_event
    publish = publisher or publish_ledger_state
    try:
        return publish(ledger_dir=Path(ledger_dir), repo_dir=Path(repo_dir), log=emit)
    except Exception as error:  # noqa: BLE001 - a sync failure never fails the run
        reason = f"the sync raised {type(error).__name__}: {error}"
        logger.error("Ledger sync failed", error=reason)
        try:
            emit("sync_refused", ok=False, reason=reason)
        except Exception as log_error:  # noqa: BLE001 - nor does a failing run log
            logger.error(
                "Ledger run log refused the sync failure", error=str(log_error)
            )
        return None
