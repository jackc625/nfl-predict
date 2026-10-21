"""The ONE place the daily run talks to the ledger (Phase 34, Plan 34-15).

Three entry points, each called by ``scripts/daily_lock_pipeline.py`` / ``pipeline/daily_steps.py``
once the committed cutover switch (``forward_ledger.cutover``) sends forward rows here:

:func:`record_forward_slate` -- the forward write, replacing ``generate_weekly_bet_list`` on the
daily run's critical recommend step. In order:

  1. resolve the production artifact ids ONCE (``models.artifacts.resolve_production_artifacts``)
     and build the week's decision bundle with them, so a ``latest.json`` swap mid-run cannot
     mis-stamp a row (Plan 34-05). The ids name the generation in force
     (``refit.generations.in_force_generation``, D-19): the pin's ids are generation 0 and the
     builder is called exactly as before Phase 37 (D-04); a re-fit generation's ids decide with
     that generation's own bet-rule record (its fits, no row-19 overlay, and its edge thresholds,
     ``backtest.weekly_bet_list.load_bet_rule_record``), and ids no install records are refused
     by name before anything is decided or written (research Pitfall 1);
  2. compute the reproduction key and the decision-snapshot digest IN MEMORY (Plan 34-08);
  3. stamp the rows at the one stamping site (``forward_ledger.stamps``) with the generation's
     recipe -- ``recipe-2026-row19-v1`` for generation 0, ``recipe-2026-refit-v1`` with every
     installed generation's ids for a re-fit generation (D-20);
  4. first pick stands (D-13, SPEC LDGR-01): an identical repeat is skipped; a row that reached a
     DIFFERENT pick for a stored key is refused, logged by name (``first_pick_refusal``) and
     printed (``LEDGER_REFUSED=``) -- the run does not fail, and new keys still append;
  5. nothing new -> NOTHING is written: no snapshot, no copy, no ledger write, no anchor commit;
  6. otherwise check the slate's ``publish_by`` deadline (a late write leaves NOTHING in
     ``ledger/``, not even a snapshot or copy), then store the snapshot and the artifact and
     chain-fit copies (written BEFORE the row that references them, and idempotent) -- refusing
     when the snapshot written is not the one the rows were stamped with;
  7. the frozen shadow arm (Phase 37, D-04): the live bundle re-decided with the pinned row-19
     models and bet rule (``forward_ledger.shadow``), stamped ``arm = shadow`` with the SAME
     reproduction key, decision instant and verdict scope, kept only for keys whose live row is
     new, pre-validated, then given its own snapshot and copies. ANY shadow failure -- including
     too little time left before the deadline (``SHADOW_COMMIT_HEADROOM``, checked before the
     shadow build starts, so a build is never begun without it, and again after the build) --
     drops every shadow row and records each new live key as a hole named by the exception
     class; it never blocks, delays or changes a live row;
  8. ONE ``commit_changes`` carrying the live rows first and then the shadow rows, with the same
     deadline, which re-reads the clock immediately before the replace -- so an interrupted write
     leaves both arms' rows of the run or neither. Once it wrote, every hole is appended to
     ``ledger/refit/shadow_holes.jsonl`` (``refit.records``); a hole record that cannot be written
     is logged by class and never fails the call.

:func:`settle_ledger` -- every day the schedule refresh returns a season: grade pending rows from
the refreshed silver scores, append the corrections owed by the live manifest's
``live_revision_graded`` verdicts, and finalize closing columns -- all in ONE ``commit_changes``,
so the three halves land together or not at all (34-RESEARCH Q2, Pitfall 11). Only the closing
finalization reads the run log; a run log that cannot be parsed DEFERS it -- those rows stay NULL
and are retried by a later run, and the outcome names why -- while grading and corrections still
land (34-REVIEW WR-01). A malformed log is never read as "no events".

:func:`sync_after_run` -- after the run, anchor the head and back up the directory through
``forward_ledger.sync.publish_ledger_state``. It NEVER raises (LDGR-05, D-01).

Every event goes through ``forward_ledger.run_log.record_event`` (or an injected ``log``), so each
carries the daily run's bound ``run_id``.

NEVER ONTO A LEDGER SHORTER THAN ITS ANCHOR (34-REVIEW CR-01): when the caller names the repository
carrying ``ledger-anchor`` (``repo_dir``; the daily run passes its working directory), the record and
settle writes are refused unless the store still extends the anchored head. An absent, emptied or
shortened store is then a named refusal, never a fresh genesis-rooted ledger.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from backtest.weekly_bet_list import (
    DecisionBundle,
    build_weekly_decision_bundle,
    load_bet_rule_record,
)
from data.upstream_live import load_live_manifest
from forward_ledger.anchor import read_local_anchor
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
from forward_ledger.schema import ARM_SHADOW, VERDICT_SCOPE_VERDICT
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    grading_updates,
    live_strategies,
    load_silver_games,
    realized_values_from_scores,
)
from forward_ledger.shadow import (
    SHADOW_COMMIT_HEADROOM,
    SHADOW_PIN,
    ShadowDeadlineHeadroomError,
    build_shadow_bundle,
)
from forward_ledger.snapshots import (
    SnapshotDigestMismatchError,
    compute_snapshot_digest,
    write_decision_snapshot,
)
from forward_ledger.stamps import IN_FORCE_RECIPE_ID, stamp_ledger_rows
from forward_ledger.store import (
    LEDGER_DIR,
    LEDGER_SEASON,
    FirstPickStandsError,
    LedgerEntry,
    MissingStampError,
    _validate_new_row,
    assert_before_publish_deadline,
    assert_extends_anchor,
    classify_incoming,
    commit_changes,
    read_entries,
)
from forward_ledger.sync import SyncOutcome, publish_ledger_state
from refit.generations import (
    GENERATION_ZERO,
    Generation,
    in_force_generation,
    installed_generations,
)
from refit.records import record_shadow_holes
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from models.artifacts import ResolvedArtifacts
    from pipeline.daily_steps import DailySlate

__all__ = [
    "NO_SHADOW_ROW",
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

# The hole reason for a new live row whose shadow decision produced no twin.
NO_SHADOW_ROW: str = "no_shadow_row"

# ``(game_id, season, week, target)`` -- a live row's key without its arm -- and why it has no
# shadow twin.
ShadowHole = tuple[tuple[str, int, int, str], str]


@dataclass(frozen=True)
class RecordOutcome:
    """What one :func:`record_forward_slate` call did.

    Attributes:
        decided: Rows the decision produced (live and suppressed).
        appended: New LIVE rows written to the ledger (never the shadow rows).
        identical: Repeats of stored rows, skipped.
        refused: ``(key, differing identity columns)`` for every conflicting row (first pick
            stands); the stored rows are unchanged.
        wrote: Whether the ledger file was replaced.
        head_hash: The ledger head after the call (None when nothing was decided).
        entry_count: The ledger entry count after the call (None when nothing was decided).
        appended_rows: The canonical row values written by this call, live rows first and then
            the shadow rows; empty when nothing was written. The notifier reads these, never a
            recomputation (Phase 35, D-14), and keeps the live arm only.
        shadow_appended: New shadow rows written to the ledger.
        shadow_holes: ``((game_id, season, week, target), reason)`` for every new live row
            written without a shadow twin; the reason is an exception class name or
            :data:`NO_SHADOW_ROW`.
    """

    decided: int
    appended: int
    identical: int
    refused: tuple[tuple[tuple[Any, ...], tuple[str, ...]], ...]
    wrote: bool
    head_hash: str | None
    entry_count: int | None
    appended_rows: tuple[dict[str, Any], ...] = ()
    shadow_appended: int = 0
    shadow_holes: tuple[ShadowHole, ...] = ()


@dataclass(frozen=True)
class SettleOutcome:
    """What one :func:`settle_ledger` pass did. ``changed`` is True when the ledger was written.

    ``closing_deferred`` names why closing finalization was deferred (the run log could not be
    parsed); None when it ran.
    """

    changed: bool
    graded: int
    corrections: int
    closing_finalized: int
    observations: int
    closing_deferred: str | None = None


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


def _is_refit(generation: Generation | None) -> bool:
    """True for a re-fit generation; False for generation 0 and for no resolved ids."""
    return generation is not None and generation.number != GENERATION_ZERO


def _generation_bet_rule(generation: Generation | None) -> dict[str, Any]:
    """The builder keywords of a re-fit generation's own bet rule; none for generation 0 (D-04).

    The record is loaded (and its bytes checked against its name) BEFORE anything is decided, so
    a bet-rule file edited after install leaves nothing written.
    """
    if generation is None or not _is_refit(generation):
        return {}
    fits, thresholds = load_bet_rule_record(generation.bet_rule_path)
    return {
        "chain_fit_path": generation.bet_rule_path,
        "fits": fits,
        "edge_thresholds": thresholds,
    }


def _live_recipe(generation: Generation | None) -> str:
    """The recipe a live row is stamped with: the generation's for a re-fit, else today's."""
    if generation is None or not _is_refit(generation):
        return IN_FORCE_RECIPE_ID
    return generation.recipe


def _registered_ids(
    generation: Generation | None, ledger: Path
) -> tuple[ResolvedArtifacts, ...]:
    """Every installed re-fit generation's ids (the stamping check of a re-fit row); none else.

    Read as ``in_force_generation`` reads it: a torn last line of ``dates.jsonl`` is left out.
    """
    if not _is_refit(generation):
        return ()
    return tuple(
        installed.ids
        for installed in installed_generations(ledger, skip_torn_tail=True)
        if installed.number != GENERATION_ZERO
    )


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


def _twin_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    """A row's ``(game_id, season, week, target)``: its ledger key without the arm."""
    return (
        str(row["game_id"]),
        int(row["season"]),
        int(row["week"]),
        str(row["target"]),
    )


def _require_shadow_headroom(
    publish_by: datetime | None, clock: Callable[[], datetime] | None
) -> None:
    """Refuse the shadow rows when less than ``SHADOW_COMMIT_HEADROOM`` is left before *publish_by*.

    The ledger store re-checks the deadline immediately before it writes, so shadow work that ran
    late would otherwise turn into a refused LIVE write.
    """
    if publish_by is None:
        return
    now = clock() if clock is not None else datetime.now(tz=UTC)
    if now + SHADOW_COMMIT_HEADROOM > publish_by:
        msg = (
            f"only {(publish_by - now).total_seconds():.0f} s are left before the publish "
            f"deadline {publish_by.isoformat()}; the shadow rows need "
            f"{SHADOW_COMMIT_HEADROOM.total_seconds():.0f} s and are dropped"
        )
        raise ShadowDeadlineHeadroomError(msg)


def _shadow_rows(
    shadow_builder: Callable[..., DecisionBundle],
    bundle: DecisionBundle,
    *,
    live_new: Sequence[Mapping[str, Any]],
    entries: Sequence[LedgerEntry],
    repro: Any,
    scope: VerdictScope | None,
    ledger_has_verdict_rows: bool,
    ledger: Path,
    artifacts_dir: Path,
    publish_by: datetime | None,
    clock: Callable[[], datetime] | None,
) -> tuple[list[dict[str, Any]], tuple[ShadowHole, ...]]:
    """The shadow twins of the new live rows, ready to commit, and the holes. NEVER raises.

    Every step that can fail runs before anything of the shadow arm is written: check the deadline
    headroom (no build is started without it), build, digest, stamp (the SAME reproduction key,
    scope and decision instant as the live rows), keep the twins of NEW live keys only, validate
    them as the store will, check the deadline headroom again, and only then store the shadow
    snapshot and copies. Any exception drops every shadow row and makes
    every new live key a hole named by the exception CLASS only (D-10), so a shadow failure never
    costs a live row.
    """
    live_keys = [_twin_key(row) for row in live_new]
    try:
        # Before the build as well as after it: a build started with too little time left could
        # itself carry the clock past the deadline, and the store would then refuse the LIVE rows.
        _require_shadow_headroom(publish_by, clock)
        shadow = shadow_builder(bundle, artifacts_dir=artifacts_dir)
        if shadow.resolved is None:
            msg = (
                "the shadow decision carries no resolved ids; a stamp is never invented"
            )
            raise MissingStampError(msg)
        shadow_digest = compute_snapshot_digest(shadow)
        stamped = stamp_ledger_rows(
            shadow.frame,
            resolved=shadow.resolved,
            repro=repro,
            snapshot_digest=shadow_digest,
            scope=scope,
            ledger_has_verdict_rows=ledger_has_verdict_rows,
            recipe_id=SHADOW_PIN.recipe,
            arm=ARM_SHADOW,
        )
        wanted = set(live_keys)
        twins = [
            _validate_new_row(row)
            for row in cast(
                "pd.DataFrame", stamped[list(IMMUTABLE_COLUMNS_V1)]
            ).to_dict(orient="records")
            if _twin_key(row) in wanted
        ]
        classification = classify_incoming(entries, twins)
        if classification.conflicting:
            key, fields = classification.conflicting[0]
            raise FirstPickStandsError(key, fields, classification.conflicting)
        _require_shadow_headroom(publish_by, clock)
        written = write_decision_snapshot(ledger, shadow)
        if written != shadow_digest:
            raise SnapshotDigestMismatchError(
                "meta",
                f"the shadow rows were stamped with snapshot {shadow_digest} but the snapshot "
                f"written is {written}",
            )
        ensure_artifact_copies(ledger, shadow.resolved, artifacts_dir)
        ensure_recipe_record_copy(ledger, shadow.chain_fit_path)
    except Exception as error:  # noqa: BLE001 - a shadow failure never costs a live row (R8)
        reason = type(error).__name__
        logger.error("Shadow arm failed; the live rows are written alone", error=reason)
        return [], tuple((key, reason) for key in live_keys)

    covered = {
        _twin_key(row) for row in (*classification.new, *classification.identical)
    }
    holes = tuple((key, NO_SHADOW_ROW) for key in live_keys if key not in covered)
    return list(classification.new), holes


def _record_holes(
    ledger: Path, holes: tuple[ShadowHole, ...], decided_at: datetime
) -> None:
    """Append the holes to the ledger's refit records. NEVER raises: the live rows are committed."""
    try:
        record_shadow_holes(ledger, holes, decided_at=decided_at)
    except Exception as error:  # noqa: BLE001 - a hole record never fails the forward write
        logger.error("Shadow hole record refused", error=type(error).__name__)


def _anchored_head(repo_dir: Path | str | None) -> tuple[str, int] | None:
    """The local ``ledger-anchor`` head, or None when no repository is named or no anchor exists.

    Raises:
        forward_ledger.anchor.AnchorFormatError, forward_ledger.transport.GitCommandError: the
            anchor exists but cannot be read -- a write is refused rather than made unchecked.
    """
    if repo_dir is None:
        return None
    return read_local_anchor(Path(repo_dir))


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
    repo_dir: Path | str | None = None,
    clock: Callable[[], datetime] | None = None,
    shadow_builder: Callable[..., DecisionBundle] | None = build_shadow_bundle,
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
        repo_dir: The repository carrying ``ledger-anchor``; when given, the write is refused
            unless the store extends the anchored head. None checks no anchor (a ``tmp_path``
            ledger in a test).
        clock: The instant source for *publish_by*, shared by the check before the side files,
            the shadow headroom checks (before and after the shadow build) and the one inside
            ``commit_changes``;
            ``datetime.now(UTC)`` when None.
        shadow_builder: The frozen shadow arm (``forward_ledger.shadow.build_shadow_bundle``),
            called as ``shadow_builder(bundle, artifacts_dir=...)``. None runs no shadow arm; it
            exists only for tests whose synthetic bundles hold no scorable artifacts, and
            production callers never pass it.

    Returns:
        What was decided, appended, skipped and refused.

    Raises:
        backtest.weekly_bet_list.PublishDeadlinePassedError: the write was ready after
            *publish_by*; nothing was written to ``ledger/`` -- no row, snapshot or copy.
        refit.generations.UnknownGenerationError: ``latest.json`` names ids that are neither the
            pin's nor an installed generation's; nothing was decided or written.
        refit.records.RefitRecordFormatError: the generation records, or the generation's
            bet-rule file, cannot be read as written; nothing was decided or written.
        Everything the decision, stamping and ``commit_changes`` refuse, by name.
    """
    emit = log or record_event
    ledger = Path(ledger_dir)
    resolved = _resolve_once(Path(artifacts_dir))
    # No ids (no latest.json) keeps today's path, which refuses the stamp by name below.
    generation = None if resolved is None else in_force_generation(resolved, ledger)
    bundle = bundle_builder(
        slate.season,
        slate.week,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
        now=decided_at,
        excluded_game_ids=frozenset(excluded_game_ids),
        resolved=resolved,
        **_generation_bet_rule(generation),
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
    # Computed ONCE and handed to both arms, so a live row and its shadow twin cannot differ in
    # their reproduction key or verdict scope (37-RESEARCH Pitfall 5).
    scope = _verdict_scope_or_none()
    ledger_has_verdict_rows = _has_verdict_rows(entries)
    stamped_digest = compute_snapshot_digest(bundle)
    stamped = stamp_ledger_rows(
        frame,
        resolved=bundle.resolved,
        repro=repro,
        snapshot_digest=stamped_digest,
        scope=scope,
        ledger_has_verdict_rows=ledger_has_verdict_rows,
        recipe_id=_live_recipe(generation),
        registered_ids=_registered_ids(generation, ledger),
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

    # The deadline BEFORE any side file: a late decision must leave no snapshot, artifact copy or
    # recipe copy behind for the next backup to publish as an orphan (34-REVIEW WR-25).
    # ``commit_changes`` checks it again immediately before the replace.
    assert_before_publish_deadline(
        publish_by, clock, ready="the forward slate's ledger write"
    )
    snapshot_digest = write_decision_snapshot(ledger, bundle)
    if snapshot_digest != stamped_digest:
        # The writer re-reads uv.lock, the chain-fit file and the library versions; if one moved
        # since stamping, the rows would name a snapshot that does not exist (34-REVIEW WR-05).
        raise SnapshotDigestMismatchError(
            "meta",
            f"the rows were stamped with snapshot {stamped_digest} but the snapshot written is "
            f"{snapshot_digest}; nothing was written to the ledger",
        )
    ensure_artifact_copies(ledger, bundle.resolved, Path(artifacts_dir))
    ensure_recipe_record_copy(ledger, bundle.chain_fit_path)

    shadow_rows: list[dict[str, Any]] = []
    holes: tuple[ShadowHole, ...] = ()
    if shadow_builder is not None:
        shadow_rows, holes = _shadow_rows(
            shadow_builder,
            bundle,
            live_new=classification.new,
            entries=entries,
            repro=repro,
            scope=scope,
            ledger_has_verdict_rows=ledger_has_verdict_rows,
            ledger=ledger,
            artifacts_dir=Path(artifacts_dir),
            publish_by=publish_by,
            clock=clock,
        )
    # ONE write, live rows first and then their shadow twins (the documented fixed order).
    written_rows = [*classification.new, *shadow_rows]
    result = commit_changes(
        ledger,
        new_rows=written_rows,
        publish_by=publish_by,
        clock=clock,
        anchored_head=_anchored_head(repo_dir),
    )
    if result.wrote:
        emit(
            "append",
            season=slate.season,
            week=slate.week,
            appended_rows=len(classification.new),
            shadow_appended_rows=len(shadow_rows),
            identical_rows=len(classification.identical),
            refused_rows=len(classification.conflicting),
            snapshot_digest=snapshot_digest,
            head_hash=result.head_hash,
            entry_count=result.entry_count,
        )
        if holes:
            _record_holes(ledger, holes, decided_at)
    return RecordOutcome(
        decided=len(rows),
        appended=len(classification.new) if result.wrote else 0,
        identical=len(classification.identical),
        refused=classification.conflicting,
        wrote=result.wrote,
        head_hash=result.head_hash,
        entry_count=result.entry_count,
        appended_rows=tuple(written_rows) if result.wrote else (),
        shadow_appended=len(shadow_rows) if result.wrote else 0,
        shadow_holes=holes,
    )


def settle_ledger(
    *,
    now: datetime,
    ledger_dir: Path | str = LEDGER_DIR,
    silver_dir: Path | str = DEFAULT_SILVER_DIR,
    manifest_dir: Path | str | None = None,
    strategies: Mapping[str, Any] | None = None,
    log: LogCallable | None = None,
    repo_dir: Path | str | None = None,
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
        repo_dir: The repository carrying ``ledger-anchor``; when given, a store that does not
            extend the anchored head (absent and emptied included) is refused by name.

    Returns:
        What changed. ``changed`` is False -- and the file untouched -- when nothing was owed.

    Raises:
        Every refusal of the loaders and of ``commit_changes``, by name. The daily run records
        the failure and goes on (it never stops the decision run).
    """
    emit = log or record_event
    ledger = Path(ledger_dir)
    entries = read_entries(ledger)
    anchored = _anchored_head(repo_dir)
    assert_extends_anchor(entries, anchored)
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
    closing_deferred: str | None = None
    try:
        capture_events = read_events(event="closing_capture")
    except ValueError as error:
        # The closing NULL reasons come from these records, so no closing column is finalized
        # without them; grading and corrections do not read the log and still land.
        closing_deferred = str(error)
        logger.error("Closing finalization deferred", error=closing_deferred)
        capture_events = None
    closing = (
        {}
        if capture_events is None
        else finalize_closing(entries, games, odds, capture_events, registry)
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
        return SettleOutcome(False, 0, 0, 0, observed, closing_deferred)

    result = commit_changes(
        ledger,
        grading_updates=grading,
        new_corrections=corrections.payloads,
        closing_updates=closing,
        anchored_head=anchored,
    )
    if result.wrote:
        emit(
            "settle",
            graded=result.grading_changed,
            corrections=result.appended_corrections,
            closing_finalized=result.closing_changed,
            closing_deferred=closing_deferred,
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
        closing_deferred=closing_deferred,
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
        # The class name only (D-10): an exception message can carry a URL or a secret, and
        # this reason is logged and written to logs/ledger_runs.jsonl.
        reason = f"the sync raised {type(error).__name__}"
        logger.error("Ledger sync failed", error=type(error).__name__)
        try:
            emit("sync_refused", ok=False, reason=reason)
        except Exception as log_error:  # noqa: BLE001 - nor does a failing run log
            logger.error(
                "Ledger run log refused the sync failure",
                error=type(log_error).__name__,
            )
        return None
