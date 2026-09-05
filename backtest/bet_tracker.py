"""The realized-versus-expected tracker: a PURE aggregation over the stored bet-list blob.

Phase 31, plan 31-13 (SPEC R8, D31-21/22, PROD-03).

WHAT THIS MODULE IS
-------------------
A thin harness in the shape ``backtest/signal_lift.py`` documents: it COMPOSES, it does not
re-derive. Every figure it reports is a count or a sum over columns that another module already
wrote to the ``bet_list`` table. It prices nothing, computes no expected value, re-grades no bet
and opens no connection. Given the frame, its output is a function of the stored numbers alone.

That is deliberate and it is load-bearing. The per-target price a bet was struck at is an OPEN
question in this phase; a tracker that re-derived a payout would have to take a side, and would
publish a different number the moment the ruling landed. Reading ``payout_flat`` and ``flat_stake``
off the row means a ruling changes the values flowing IN and changes nothing here.

WHY THE TRACKER IS NOT UNDER ``api/``
-------------------------------------
It looks like a page feature, which invites an API home. But aggregation IS computation, and
computation in the request path breaks the no-request-path-math rule (UIAP-01) -- while passing the
shipped ``tests/api/test_import_guard.py``, which forbids only ``models`` / ``features`` /
``ratings``. So it lives in the analytics package, ``pipeline/steps.py`` calls it at population
time, and ``api/cache.py`` exposes only the pure-persistence writer
``materialize_bet_tracker_blocks(conn, tracker_df)`` which inserts a PRECOMPUTED frame.

THE CALL DIRECTION IS PART OF THE CONTRACT (REVIEW-IMPORT). ``api/cache.py`` must NOT import
``backtest``: the Plan 31-01 sibling guard walks every file under ``api/`` with ``ast.walk``,
covers lazy function-scoped imports, and carries an EXACT-count allow-list naming only two
inherited ``api/charts/core.py`` sites. Phase 31 adds ZERO entries to it. The dependency that DOES
exist points the other way -- this module imports the LABEL VOCABULARY and the persistence column
order from ``api.cache``, constants only, no behaviour -- because re-typing the vocabulary here is
the second-list failure this repository has already paid for once.

THE THREE DISTINCTIONS THAT ARE EASY TO COLLAPSE
------------------------------------------------
1. **A push versus an ungraded bet.** Both store ``outcome`` as SQL NULL, so the discriminator is
   ``grading_status`` and never the outcome. A push is COUNTED (it is a settled result) and an
   ungraded row is EXCLUDED from every figure INCLUDING pushes (nothing about it is known yet).

2. **The hit-rate denominator versus the return denominator.** They are different on purpose. A
   push is not a contest won or lost, so it is outside the hit-rate denominator. It DOES turn over
   money at zero profit, so its stake is inside the return denominator. Collapsing the two would be
   wrong in one direction or the other.

3. **Not measured versus measured zero.** A block with no graded rows returns
   :class:`EmptyTrackerBlock`, which carries no rate field at all -- the rate is not computed in
   that branch, rather than computed and guarded. A block whose graded rows carry no stake reports
   ``flat_return_units = None``. Publishing either as ``0.0`` would read as "we broke even", which
   is a claim nobody made. This mirrors ``BetSelector._clv_report`` defaulting ``clv`` to ``None``.

Run the tests:  .venv/Scripts/python.exe -m pytest tests/unit/test_bet_tracker.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, cast

import pandas as pd

from api.cache import (
    BET_TRACKER_BLOCK_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    GRADING_STATUSES,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
    VALIDATION_TYPE_CLEAN_HOLDOUT,
    VALIDATION_TYPE_CONTAMINATED,
    VALIDATION_TYPE_FORWARD_REALIZED,
)
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "TRACKER_BLOCK_FIGURES",
    "TRACKER_BLOCK_ORDER",
    "TRACKER_REQUIRED_COLUMNS",
    "EmptyTrackerBlock",
    "TrackerBlock",
    "aggregate_all_blocks",
    "aggregate_by_provenance",
    "to_tracker_frame",
]


# ---------------------------------------------------------------------------
# Vocabulary and contracts
# ---------------------------------------------------------------------------

# The bet-list columns this module reads. ``status`` is here because a SUPPRESSED candidate was
# never bet, and a candidate that was never bet cannot be part of a betting record (D31-21). It is
# present in the same table by design; it is excluded from every figure by rule.
TRACKER_REQUIRED_COLUMNS: tuple[str, ...] = (
    "status",
    "provenance",
    "validation_type",
    "grading_status",
    "flat_stake",
    "payout_flat",
)

# The per-block figure set, EXACTLY. Six figures, no more: an extra figure on the page is an extra
# claim, and the UI-SPEC's block is drawn against this list.
TRACKER_BLOCK_FIGURES: tuple[str, ...] = (
    "bets_graded",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "flat_return_units",
)

# The row status that means a bet was actually placed.
_STATUS_LIVE = "live"

# The grading statuses that mean the result is KNOWN. ``pending`` is deliberately absent.
_GRADED_STATUSES: frozenset[str] = frozenset(
    {GRADING_STATUS_WIN, GRADING_STATUS_LOSS, GRADING_STATUS_PUSH}
)

# The statuses that CONTEST a hit rate. A push settled without either side winning, so it is not a
# member -- and it is not a loss either, which is the coercion this set exists to prevent.
_CONTESTED_STATUSES: frozenset[str] = frozenset(
    {GRADING_STATUS_WIN, GRADING_STATUS_LOSS}
)

_PROVENANCES: frozenset[str] = frozenset(
    {PROVENANCE_BACKTEST_REPLAY, PROVENANCE_FORWARD}
)
_VALIDATION_TYPES: frozenset[str] = frozenset(
    {
        VALIDATION_TYPE_CONTAMINATED,
        VALIDATION_TYPE_CLEAN_HOLDOUT,
        VALIDATION_TYPE_FORWARD_REALIZED,
    }
)

# The DECLARED display order of the blocks. Contaminated replay first, then the single clean
# holdout, then the live forward record -- weakest evidence to strongest, which is the order the
# reader should meet them in. A pair absent from the frame simply does not appear.
TRACKER_BLOCK_ORDER: tuple[tuple[str, str], ...] = (
    (PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CONTAMINATED),
    (PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CLEAN_HOLDOUT),
    (PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED),
)


@dataclass(frozen=True)
class TrackerBlock:
    """One provenance-and-validation-type class with AT LEAST ONE graded bet in it.

    The six figures are exactly ``TRACKER_BLOCK_FIGURES``. ``flat_return_units`` is the flat-stake
    return per unit staked -- ``sum(payout_flat) / sum(flat_stake)`` over the graded rows, the same
    quantity ``backtest.ou_monetization._flat_roi_from_records`` computes -- and it is ``None``
    when the graded rows carry no stake at all, because an unmeasured return published as zero
    reads as a break-even result.
    """

    provenance: str
    validation_type: str
    bets_graded: int
    wins: int
    losses: int
    pushes: int
    hit_rate: float
    flat_return_units: float | None

    is_empty: ClassVar[bool] = False


@dataclass(frozen=True)
class EmptyTrackerBlock:
    """A class with ZERO graded bets: the marker the template renders an empty state from.

    It carries NO ``hit_rate`` and NO ``flat_return_units`` field. That absence is the point.
    Guarding a division still performs the division conceptually and still puts a number on the
    page; a block of zeros asserts that the record was measured and came out zero. Nothing was
    measured here, so there is nothing to report but the fact that nothing was measured.
    """

    provenance: str
    validation_type: str

    bets_graded: ClassVar[int] = 0
    is_empty: ClassVar[bool] = True


TrackerBlockResult = TrackerBlock | EmptyTrackerBlock


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_columns(bet_list_df: pd.DataFrame) -> None:
    """Raise a NAMED ``KeyError`` listing exactly which required columns are absent."""
    missing = [c for c in TRACKER_REQUIRED_COLUMNS if c not in bet_list_df.columns]
    if missing:
        msg = (
            f"aggregate_by_provenance: bet_list frame missing required column(s) {missing}; "
            "the tracker reads stored values and cannot substitute for an absent one."
        )
        raise KeyError(msg)


def _validate_vocabularies(bet_list_df: pd.DataFrame) -> None:
    """Refuse a frame carrying a label outside the closed vocabularies.

    Refusing rather than dropping: an unrecognised provenance is a row nobody can classify, and
    quietly excluding it from every block is the silent-drop failure the whole phase is built
    against.
    """
    for column, permitted in (
        ("provenance", _PROVENANCES),
        ("validation_type", _VALIDATION_TYPES),
        ("grading_status", frozenset(GRADING_STATUSES)),
    ):
        values = cast(pd.Series, bet_list_df[column]).dropna().unique()
        offending = sorted({str(v) for v in values if str(v) not in permitted})
        if offending:
            msg = (
                f"aggregate_by_provenance: {column} carries out-of-vocabulary value(s) "
                f"{offending}; the closed vocabulary is {sorted(permitted)}."
            )
            raise ValueError(msg)


def _validate_suppressed_rows_are_ungraded(bet_list_df: pd.DataFrame) -> None:
    """Refuse a suppressed row carrying a terminal grade.

    A candidate the selector declined was never staked, so it cannot have won, lost or pushed.
    Such a row is a contradiction in the ledger rather than a figure to report, and silently
    excluding it would hide a real defect in whatever wrote it.
    """
    # pandas-stubs widens boolean-mask __getitem__ to DataFrame | Series, so the .empty and
    # column accesses below lose their overload match at type-check time though every value is a
    # DataFrame at runtime (the same stub gap api/cache.py annotates elsewhere).
    suppressed_graded = cast(
        pd.DataFrame,
        bet_list_df[
            (bet_list_df["status"] != _STATUS_LIVE)
            & (bet_list_df["grading_status"].isin(sorted(_GRADED_STATUSES)))
        ],
    )
    if not suppressed_graded.empty:
        statuses = sorted({str(v) for v in suppressed_graded["status"].unique()})
        msg = (
            f"aggregate_by_provenance: {len(suppressed_graded)} row(s) with status {statuses} "
            "carry a terminal grading_status. A candidate that was never bet cannot have won, "
            "lost or pushed; this is a contradiction in the ledger, not a figure."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _flat_return_units(graded: pd.DataFrame) -> float | None:
    """``sum(payout_flat) / sum(flat_stake)`` over *graded*, or ``None`` when no stake was made.

    Read from the STORED prices, never derived from ``outcome``. Under asymmetric American prices
    a win at -110 returns 0.909 units and a win at +140 returns 1.400 units on the same stake, so
    an outcome-derived return is wrong for every bet not struck at the reference juice -- which is
    most of them. Mirrors ``backtest.ou_monetization._flat_roi_from_records``, including its
    ``None`` for a zero denominator.
    """
    total_stake = float(graded["flat_stake"].fillna(0.0).sum())
    if total_stake <= 0:
        return None
    return float(graded["payout_flat"].fillna(0.0).sum() / total_stake)


def aggregate_by_provenance(
    bet_list_df: pd.DataFrame,
    *,
    provenance: str,
    validation_type: str,
) -> TrackerBlockResult:
    """Aggregate the bet-list rows of ONE honesty class into one block.

    The partition is applied BEFORE any arithmetic and there is no default to fall into: both
    partition arguments are keyword-only and required, so a caller states which block it is asking
    for. There is deliberately no entry point that returns a figure spanning two classes.

    The partition is on the PAIR, not on ``provenance`` alone. The replay class contains both the
    burned 2021-2024 rows and the single clean 2025 holdout, and pooling those would publish the
    one unspent split's result inside a contaminated figure -- which is exactly why D31-22 carries
    two orthogonal columns rather than one label.

    Args:
        bet_list_df: Bet-list rows carrying at least ``TRACKER_REQUIRED_COLUMNS``. Rows outside
            the requested class, suppressed rows and ungraded rows are all excluded.
        provenance: The class's ``provenance`` label. Required, keyword-only.
        validation_type: The class's ``validation_type`` label. Required, keyword-only.

    Returns:
        A :class:`TrackerBlock` when at least one row in the class is graded, otherwise an
        :class:`EmptyTrackerBlock` -- and in that branch no rate is computed at all.

    Raises:
        KeyError: on an absent required column.
        ValueError: on an out-of-vocabulary label, or a suppressed row carrying a terminal grade.
    """
    if provenance not in _PROVENANCES:
        msg = (
            f"aggregate_by_provenance: provenance {provenance!r} is outside the vocabulary "
            f"{sorted(_PROVENANCES)}."
        )
        raise ValueError(msg)
    if validation_type not in _VALIDATION_TYPES:
        msg = (
            f"aggregate_by_provenance: validation_type {validation_type!r} is outside the "
            f"vocabulary {sorted(_VALIDATION_TYPES)}."
        )
        raise ValueError(msg)

    _require_columns(bet_list_df)
    _validate_vocabularies(bet_list_df)
    _validate_suppressed_rows_are_ungraded(bet_list_df)

    # (1) The partition, BEFORE any arithmetic.
    in_class = cast(
        pd.DataFrame,
        bet_list_df[
            (bet_list_df["provenance"] == provenance)
            & (bet_list_df["validation_type"] == validation_type)
            & (bet_list_df["status"] == _STATUS_LIVE)
        ],
    )

    # (2) Graded versus ungraded, decided on grading_status and NEVER on a null outcome. A push
    # (status push, outcome NULL) is graded; an ungraded forward bet (status pending, outcome
    # NULL) is not. The two are indistinguishable by outcome, which is why this line reads status.
    graded = cast(
        pd.DataFrame,
        in_class[in_class["grading_status"].isin(sorted(_GRADED_STATUSES))],
    )

    if graded.empty:
        # No rate is computed in this branch. See EmptyTrackerBlock's docstring.
        return EmptyTrackerBlock(provenance=provenance, validation_type=validation_type)

    statuses = cast(pd.Series, graded["grading_status"])
    wins = int((statuses == GRADING_STATUS_WIN).sum())
    losses = int((statuses == GRADING_STATUS_LOSS).sum())
    pushes = int((statuses == GRADING_STATUS_PUSH).sum())

    # The hit-rate denominator is the CONTESTED rows only, read from the same status column rather
    # than inferred as "graded minus pushes" -- so the push count sits outside this group by
    # construction, never added to the numerator and never to the denominator.
    contested = int(statuses.isin(sorted(_CONTESTED_STATUSES)).sum())
    # ``graded`` is non-empty here, so ``contested`` can only be zero when every graded row is a
    # push -- a week that settled without a single contest. Reporting 0.0 for that is a genuine
    # measurement (no bet was won), unlike the zero-graded branch above where nothing was measured.
    hit_rate = float(wins / contested) if contested else 0.0

    return TrackerBlock(
        provenance=provenance,
        validation_type=validation_type,
        bets_graded=len(graded),
        wins=wins,
        losses=losses,
        pushes=pushes,
        hit_rate=hit_rate,
        flat_return_units=_flat_return_units(graded),
    )


def aggregate_all_blocks(bet_list_df: pd.DataFrame) -> list[TrackerBlockResult]:
    """One block per honesty class PRESENT in *bet_list_df*, in the declared display order.

    Every block is computed by :func:`aggregate_by_provenance` within its own class, so no figure
    returned here spans two classes. A class absent from the frame is absent from the result --
    this function reports what the ledger contains and does not manufacture an empty block for a
    class that was never written.
    """
    _require_columns(bet_list_df)
    _validate_vocabularies(bet_list_df)
    _validate_suppressed_rows_are_ungraded(bet_list_df)

    if bet_list_df.empty:
        return []

    present = {
        (str(row_provenance), str(row_validation_type))
        for row_provenance, row_validation_type in zip(
            bet_list_df["provenance"], bet_list_df["validation_type"], strict=True
        )
    }

    blocks: list[TrackerBlockResult] = []
    for pair in TRACKER_BLOCK_ORDER:
        if pair not in present:
            continue
        blocks.append(
            aggregate_by_provenance(
                bet_list_df, provenance=pair[0], validation_type=pair[1]
            )
        )

    logger.info(
        "Aggregated realized-vs-expected tracker blocks",
        blocks=len(blocks),
        classes=sorted(present),
    )
    return blocks


# ---------------------------------------------------------------------------
# The persistence handoff
# ---------------------------------------------------------------------------


def to_tracker_frame(blocks: list[TrackerBlockResult]) -> pd.DataFrame:
    """Render *blocks* as the frame ``api.cache.materialize_bet_tracker_blocks`` persists.

    The column order is TAKEN from ``api.cache.BET_TRACKER_BLOCK_COLUMNS`` rather than re-typed,
    so the producer here cannot drift from the writer there.

    SQL has no way to express an ABSENT column, so an :class:`EmptyTrackerBlock` persists its two
    rate fields as NULL and its four counts as 0. The empty-state marker on the read side is
    ``bets_graded == 0``; a NULL rate beside it says the rate was not computed, which is a
    different statement from a stored ``0.0``.
    """
    records: list[dict[str, Any]] = []
    for block in blocks:
        record: dict[str, Any] = {
            "provenance": block.provenance,
            "validation_type": block.validation_type,
            "bets_graded": block.bets_graded,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": None,
            "flat_return_units": None,
        }
        if isinstance(block, TrackerBlock):
            for figure in TRACKER_BLOCK_FIGURES:
                record[figure] = getattr(block, figure)
        records.append(record)

    frame = pd.DataFrame(records, columns=pd.Index(BET_TRACKER_BLOCK_COLUMNS))
    if frame.empty:
        return frame
    # Object dtype keeps a None rate as None rather than promoting it to NaN, so the
    # not-measured / measured-zero distinction survives the round trip into DuckDB.
    for column in ("hit_rate", "flat_return_units"):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def block_field_names() -> tuple[str, ...]:
    """The populated block's field names, for a caller that wants them without importing the class."""
    return tuple(field.name for field in fields(TrackerBlock))
