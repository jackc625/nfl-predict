"""The ONE canonical serialization the forward ledger's hash chain covers (LDGR-05).

WHAT IS HASHED, AND WHY IT IS SCHEMA-DRIVEN RATHER THAN DTYPE-DRIVEN
-------------------------------------------------------------------
A chain hash is only evidence if re-reading and re-writing the store without changing any value
reproduces it exactly. So the bytes hashed are never file bytes and never whatever pandas happened
to infer: each value is coerced by the column's DECLARED type (the DDL type in
``api.cache.BET_LIST_SCHEMA``), and the entry is rendered as one compact JSON list of
``[name, value]`` pairs in the frozen v1 column order. Binding the names into the payload means a
value moved to a different column changes the hash, not just a value that changed.

  * INTEGER -> builtin ``int``. numpy ints convert; an integral float (pandas upcasts an int
    column holding a NULL to float64) converts; a non-integral float and a ``bool`` are refused.
  * DOUBLE -> builtin ``float``, whose JSON form is ``float.__repr__`` -- the shortest string that
    round-trips to the identical IEEE-754 double. ``+/-inf`` is refused by name; ``-0.0`` and
    ``0.0`` stay distinct (both are exact, and both survive a round trip).
  * VARCHAR -> the ``str`` exactly as stored. Timestamps here are already offset-carrying strings
    (``decided_at_utc``, ``snapshot_ts``, ``freeze_ts``); parsing and re-rendering one would let a
    formatting change move a hash, so a non-string value is refused rather than rendered.
  * BOOLEAN -> builtin ``bool`` (numpy bools convert); anything else is refused.
  * Every NULL form -- ``None``, float NaN, ``pd.NA``, ``pd.NaT``, ``numpy.nan`` -- becomes the ONE
    JSON ``null``. Without that, a value read back from parquet as NaN would hash differently
    from the same value written as ``None``.

VERSIONED, BECAUSE THE SCHEMA WILL GROW
---------------------------------------
``canon_v`` is INSIDE the hashed payload, and each version owns a frozen column list. A column a
later phase adds creates v2; every row written under v1 keeps hashing its v1 list forever, so a
schema bump can never change an old row's recomputed hash. ``IMMUTABLE_COLUMNS_V1`` and
``CORRECTION_COLUMNS_V1`` are therefore LITERALS and must never be edited once an entry is written.

THE CHAIN
---------
``h_i = sha256(CHAIN_DOMAIN + h_{i-1} + "\\n" + canonical_bytes_i)``, with ``h_{-1}`` the published
``GENESIS_HASH``. The head of the ledger is ``(h_n, n)``; an empty ledger's head is
``(GENESIS_HASH, 0)``. The genesis hash is a literal in this source so it is published by the
commit that introduces it; the tests re-derive it from ``GENESIS_SEED``.

THE SAME DISCIPLINE FOR A WHOLE FRAME (LDGR-09, Plan 34-08)
-----------------------------------------------------------
:func:`frame_digest` gives a decision-input frame (a snapshot part, the admissible odds rows) one
content digest that a parquet round trip, a row shuffle or a column reorder cannot move. A frame
has no declared schema, so each column is coerced by its pandas dtype KIND instead, and the kind
is bound into the payload: columns sorted by name, rows sorted by a stated key that must identify
every row UNIQUELY (a tie would leave the digest dependent on the order the tied rows arrived in),
the one null form, floats by exact ``repr``, tz-aware instants as UTC ISO 8601 at nanosecond
precision. A naive datetime, an infinity or an object it cannot render exactly is refused by name.

Constants and pure functions only: no I/O, no clock, no state.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CANON_VERSION",
    "CHAIN_DOMAIN",
    "CORRECTION_COLUMNS_V1",
    "CORRECTION_COLUMN_TYPES_V1",
    "ENTRY_KIND_CORRECTION",
    "ENTRY_KIND_ROW",
    "FRAME_DOMAIN",
    "GENESIS_HASH",
    "GENESIS_SEED",
    "IMMUTABLE_COLUMNS_V1",
    "IMMUTABLE_COLUMN_TYPES_V1",
    "CanonicalValueError",
    "canonical_entry_bytes",
    "canonical_values",
    "chain_hash",
    "coerce_canonical_value",
    "frame_digest",
]

CANON_VERSION = 1

# The two entry kinds the ledger holds. A ROW is one bet decision (its immutable half is hashed);
# a CORRECTION records a re-grade after a score was revised post-grading (D-05). Both are chained.
ENTRY_KIND_ROW = "row"
ENTRY_KIND_CORRECTION = "correction"

# The immutable half of a ledger row, FROZEN FOREVER FOR v1. The first 23 names are
# ``api.cache.BET_LIST_IMMUTABLE_COLUMNS`` as they stood before Phase 34, in their order; the
# last 11 are the Phase-34 stamps. A literal, not an import: ``api.cache`` may grow its list in a
# later phase, and that must create v2 here rather than silently re-hash every v1 row.
# ``forward_ledger.schema`` proves at import that the live list still equals this one.
IMMUTABLE_COLUMNS_V1: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "target",
    "bet_side",
    "model_value",
    "market_value",
    "line",
    "slipped_line",
    "calibrated_p_side",
    "per_bet_ev",
    "stake_units",
    "ev_tier",
    "status",
    "rejection_reason",
    "eligibility_label",
    "snapshot_ts",
    "freeze_ts",
    "selected_odds",
    "flat_stake",
    "provenance",
    "validation_type",
    "decided_at_utc",
    "arm",
    "model_artifact_id",
    "blend_id",
    "recipe_id",
    "fill_convention_id",
    "upstream_capture_key",
    "gold_generation_key",
    "odds_snapshot_digest",
    "decision_snapshot_digest",
    "verdict_scope",
    "regime_label",
)

# Each v1 immutable column's DECLARED type: the first 23 copied from the ``BET_LIST_SCHEMA`` DDL,
# the 11 Phase-34 stamps all VARCHAR. Insertion order equals ``IMMUTABLE_COLUMNS_V1``.
IMMUTABLE_COLUMN_TYPES_V1: dict[str, str] = {
    "game_id": "VARCHAR",
    "season": "INTEGER",
    "week": "INTEGER",
    "target": "VARCHAR",
    "bet_side": "VARCHAR",
    "model_value": "DOUBLE",
    "market_value": "DOUBLE",
    "line": "DOUBLE",
    "slipped_line": "DOUBLE",
    "calibrated_p_side": "DOUBLE",
    "per_bet_ev": "DOUBLE",
    "stake_units": "DOUBLE",
    "ev_tier": "VARCHAR",
    "status": "VARCHAR",
    "rejection_reason": "VARCHAR",
    "eligibility_label": "VARCHAR",
    "snapshot_ts": "VARCHAR",
    "freeze_ts": "VARCHAR",
    "selected_odds": "DOUBLE",
    "flat_stake": "DOUBLE",
    "provenance": "VARCHAR",
    "validation_type": "VARCHAR",
    "decided_at_utc": "VARCHAR",
    "arm": "VARCHAR",
    "model_artifact_id": "VARCHAR",
    "blend_id": "VARCHAR",
    "recipe_id": "VARCHAR",
    "fill_convention_id": "VARCHAR",
    "upstream_capture_key": "VARCHAR",
    "gold_generation_key": "VARCHAR",
    "odds_snapshot_digest": "VARCHAR",
    "decision_snapshot_digest": "VARCHAR",
    "verdict_scope": "VARCHAR",
    "regime_label": "VARCHAR",
}

# A correction entry (D-05), FROZEN FOREVER ONCE THE FIRST ONE IS WRITTEN. Three outcome states
# are recorded side by side, and the middle one is what makes a history of corrections readable:
#   * ``original_*``  -- the row's OWN grade, which never changes (the grader is one-way);
#   * ``prior_*``     -- the outcome IN FORCE immediately before this entry: the row's own grade for
#     a first correction, else the previous correction's corrected values. ``prior_correction_seq``
#     is the ledger seq of the correction entry that was in force, NULL when the row's own grade
#     was. Without these, a win -> loss -> win history would record its second entry as
#     win -> win (34-REVIEWS.md Consensus 3);
#   * ``corrected_*`` -- the outcome this entry puts in force.
CORRECTION_COLUMNS_V1: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "target",
    "arm",
    "original_grading_status",
    "original_payout_flat",
    "prior_grading_status",
    "prior_payout_flat",
    "prior_realized_units",
    "prior_correction_seq",
    "corrected_grading_status",
    "corrected_outcome",
    "corrected_payout_flat",
    "corrected_realized_units",
    "realized_value",
    "source_dataset",
    "source_season",
    "source_week",
    "source_sequence",
    "detected_at_utc",
    "corrected_at_utc",
)

CORRECTION_COLUMN_TYPES_V1: dict[str, str] = {
    "game_id": "VARCHAR",
    "season": "INTEGER",
    "week": "INTEGER",
    "target": "VARCHAR",
    "arm": "VARCHAR",
    "original_grading_status": "VARCHAR",
    "original_payout_flat": "DOUBLE",
    "prior_grading_status": "VARCHAR",
    "prior_payout_flat": "DOUBLE",
    "prior_realized_units": "DOUBLE",
    "prior_correction_seq": "INTEGER",
    "corrected_grading_status": "VARCHAR",
    "corrected_outcome": "BOOLEAN",
    "corrected_payout_flat": "DOUBLE",
    "corrected_realized_units": "DOUBLE",
    "realized_value": "DOUBLE",
    "source_dataset": "VARCHAR",
    "source_season": "INTEGER",
    "source_week": "INTEGER",
    "source_sequence": "INTEGER",
    "detected_at_utc": "VARCHAR",
    "corrected_at_utc": "VARCHAR",
}

# The published genesis: the hash the first ledger entry chains from. Written as a literal so the
# commit introducing it is the publication; ``tests/unit/test_forward_ledger_chain.py`` re-derives
# it from the seed.
GENESIS_SEED = b"nfl-predict forward ledger 2026 genesis v1"
GENESIS_HASH = "1962f73e60f847b4867c95ddfbcf7de42651b4d56814a17d0569dc2358f6e6d7"

# Domain separation: no other sha256 in this repository can produce a chain link by accident.
CHAIN_DOMAIN = b"nfl-ledger-chain-v1\n"

# The same separation for a frame digest (LDGR-09): a frame digest can never equal a chain link.
FRAME_DOMAIN = b"nfl-ledger-frame-v1\n"

# The pandas dtype kinds a frame digest can render exactly: signed and unsigned integers, floats,
# booleans, datetimes (tz-aware only) and objects (strings, categories, the string dtype).
_FRAME_KINDS: frozenset[str] = frozenset("iufbMO")
_NANOS_PER_SECOND = 1_000_000_000

_COLUMNS_BY_KIND: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    ENTRY_KIND_ROW: (IMMUTABLE_COLUMNS_V1, IMMUTABLE_COLUMN_TYPES_V1),
    ENTRY_KIND_CORRECTION: (CORRECTION_COLUMNS_V1, CORRECTION_COLUMN_TYPES_V1),
}


class CanonicalValueError(Exception):
    """A value cannot be put into the canonical form, so it cannot be chained.

    Inherits ``Exception`` and deliberately NOT ``ValueError`` / ``RuntimeError``: several loaders
    in this repository catch those broadly and degrade to an empty result, and a refusal to chain
    a value must never be converted into "nothing to record" (the ``data.graded_weeks`` rule).
    """


def _is_null(value: Any) -> bool:
    """True for every NULL form the store can meet: None, NaN, pd.NA, pd.NaT, numpy NaT."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return math.isnan(value)
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def _refuse(
    column: str, declared_type: str, value: Any, why: str
) -> CanonicalValueError:
    return CanonicalValueError(
        f"column {column!r} ({declared_type}) cannot canonicalize {value!r} "
        f"({type(value).__name__}): {why}"
    )


def coerce_canonical_value(column: str, declared_type: str, value: Any) -> Any:
    """Coerce *value* to the one canonical Python form of *declared_type*.

    Args:
        column: The column name, carried into every refusal so the offending field is named.
        declared_type: ``VARCHAR``, ``INTEGER``, ``DOUBLE`` or ``BOOLEAN``.
        value: The stored or incoming value.

    Returns:
        ``None`` for every null form, else a builtin ``str`` / ``int`` / ``float`` / ``bool``.

    Raises:
        CanonicalValueError: on a value the declared type cannot hold exactly.
    """
    if _is_null(value):
        return None

    if declared_type == "VARCHAR":
        if isinstance(value, str):
            return str(value)
        raise _refuse(
            column, declared_type, value, "a VARCHAR is stored exactly as a string"
        )

    if declared_type == "INTEGER":
        if isinstance(value, bool | np.bool_):
            raise _refuse(column, declared_type, value, "a boolean is not an integer")
        if isinstance(value, int | np.integer):
            return int(value)
        if isinstance(value, float | np.floating):
            if float(value).is_integer():
                return int(value)
            raise _refuse(column, declared_type, value, "a non-integral float")
        raise _refuse(column, declared_type, value, "not a number")

    if declared_type == "DOUBLE":
        if isinstance(value, bool | np.bool_):
            raise _refuse(column, declared_type, value, "a boolean is not a double")
        if isinstance(value, int | np.integer | float | np.floating):
            number = float(value)
            if math.isinf(number):
                raise _refuse(
                    column, declared_type, value, "infinity has no canonical form"
                )
            return number
        raise _refuse(column, declared_type, value, "not a number")

    if declared_type == "BOOLEAN":
        if isinstance(value, bool | np.bool_):
            return bool(value)
        raise _refuse(column, declared_type, value, "only a boolean is a BOOLEAN")

    raise _refuse(column, declared_type, value, "unknown declared type")


def canonical_values(kind: str, immutable: Mapping[str, Any]) -> dict[str, Any]:
    """The coerced v1 values of *immutable*, in the kind's frozen column order.

    Args:
        kind: ``ENTRY_KIND_ROW`` or ``ENTRY_KIND_CORRECTION``.
        immutable: Exactly the kind's v1 columns -- no more, no fewer.

    Raises:
        CanonicalValueError: on an unknown kind, a missing or extra column, or a value its
            column's declared type cannot hold.
    """
    if kind not in _COLUMNS_BY_KIND:
        msg = f"entry kind {kind!r} is outside {tuple(_COLUMNS_BY_KIND)}"
        raise CanonicalValueError(msg)
    columns, types = _COLUMNS_BY_KIND[kind]

    missing = [name for name in columns if name not in immutable]
    extra = sorted(str(name) for name in immutable if name not in types)
    if missing or extra:
        msg = (
            f"a {kind!r} entry must carry exactly its v{CANON_VERSION} columns; "
            f"missing {missing}, extra {extra}"
        )
        raise CanonicalValueError(msg)

    return {
        name: coerce_canonical_value(name, types[name], immutable[name])
        for name in columns
    }


def canonical_entry_bytes(kind: str, immutable: Mapping[str, Any]) -> bytes:
    """The exact bytes a ledger entry's chain hash covers.

    ``[["canon_v", 1], ["kind", kind], [name, value], ...]`` as compact, ASCII-only JSON with
    ``allow_nan=False`` -- so an accidental NaN or infinity is a hard error, never a silent token.
    """
    values = canonical_values(kind, immutable)
    payload: list[list[Any]] = [["canon_v", CANON_VERSION], ["kind", kind]]
    payload += [[name, value] for name, value in values.items()]
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def chain_hash(prev_hash: str, payload: bytes) -> str:
    """``sha256(CHAIN_DOMAIN + prev_hash + "\\n" + payload)`` as lowercase hex."""
    return hashlib.sha256(
        CHAIN_DOMAIN + prev_hash.encode("ascii") + b"\n" + payload
    ).hexdigest()


# ---------------------------------------------------------------------------
# frame_digest: one content digest for a whole frame (LDGR-09)
# ---------------------------------------------------------------------------


def _refuse_cell(column: str, kind: str, value: Any, why: str) -> CanonicalValueError:
    return CanonicalValueError(
        f"frame column {column!r} (dtype kind {kind!r}) cannot canonicalize {value!r} "
        f"({type(value).__name__}): {why}"
    )


def _instant_text(column: str, kind: str, value: Any) -> str:
    """A tz-aware instant as UTC ISO 8601 at nanosecond precision.

    Rendered from the nanosecond count since the epoch, so a ``[us]`` and a ``[ns]`` column holding
    the same instant -- parquet writers and readers disagree about the unit -- give the same text.
    """
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise _refuse_cell(column, kind, value, "a naive datetime names no instant")
    seconds, nanos = divmod(stamp.value, _NANOS_PER_SECOND)
    whole = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{whole}.{nanos:09d}+00:00"


def _frame_cell(column: str, kind: str, value: Any) -> Any:
    """*value* of a column of dtype *kind* in its one canonical JSON form."""
    if _is_null(value):
        return None
    if kind in "iu":
        if isinstance(value, bool | np.bool_):
            raise _refuse_cell(column, kind, value, "a boolean is not an integer")
        return int(value)
    if kind == "f":
        number = float(value)
        if math.isinf(number):
            raise _refuse_cell(column, kind, value, "infinity has no canonical form")
        return number
    if kind == "b":
        return bool(value)
    if kind == "M":
        return _instant_text(column, kind, value)
    # Object kind: strings (object, category and the string dtype all report "O"), and the
    # booleans a left join leaves in an object column beside its NaN.
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise _refuse_cell(
        column, kind, value, "an object column may hold only strings and booleans"
    )


def _frame_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """``(name, dtype kind)`` for every column, sorted by name, or a named refusal."""
    names = list(frame.columns)
    if any(not isinstance(name, str) for name in names):
        msg = f"frame column names must be strings; got {names!r}"
        raise CanonicalValueError(msg)
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        msg = f"frame column name(s) {duplicated} appear more than once"
        raise CanonicalValueError(msg)

    columns: list[tuple[str, str]] = []
    for name in sorted(names):
        dtype = frame[name].dtype
        kind = dtype.kind
        if kind not in _FRAME_KINDS:
            msg = f"frame column {name!r} has dtype {dtype} (kind {kind!r}), which has no canonical form"
            raise CanonicalValueError(msg)
        if kind == "M" and not isinstance(dtype, pd.DatetimeTZDtype):
            msg = f"frame column {name!r} is a naive datetime ({dtype}); it names no instant"
            raise CanonicalValueError(msg)
        columns.append((name, kind))
    return columns


def frame_digest(frame: pd.DataFrame, sort_by: Sequence[str]) -> str:
    """One sha256 over *frame*'s content, independent of row order, column order and dtype unit.

    Args:
        frame: The frame to digest. Its index is ignored.
        sort_by: Columns whose values identify every row UNIQUELY; rows are ordered by them.

    Returns:
        ``sha256(FRAME_DOMAIN + payload)`` as lowercase hex, where the payload is compact ASCII
        JSON binding the ``(name, dtype kind)`` column list, the sort key and the sorted rows.

    Raises:
        CanonicalValueError: a key column is missing or holds a null, a key tuple appears twice
            (named with its columns and value), a column name is not a unique string, a dtype
            has no canonical form, a datetime is naive, a float is infinite, or an object cell is
            neither a string nor a boolean.
    """
    key = tuple(sort_by)
    if not key:
        msg = "frame_digest needs a non-empty sort key"
        raise CanonicalValueError(msg)
    missing = [name for name in key if name not in frame.columns]
    if missing:
        msg = f"frame_digest sort key {list(key)} names column(s) {missing} the frame lacks"
        raise CanonicalValueError(msg)

    columns = _frame_columns(frame)
    names = [name for name, _ in columns]
    rendered = [
        [_frame_cell(name, kind, value) for value in frame[name].tolist()]
        for name, kind in columns
    ]
    rows = [list(row) for row in zip(*rendered, strict=True)] if rendered else []

    key_positions = [names.index(name) for name in key]
    keyed: dict[str, list[Any]] = {}
    for row in rows:
        key_values = [row[position] for position in key_positions]
        if any(value is None for value in key_values):
            msg = (
                f"frame_digest sort key {list(key)} holds a null in row {key_values!r}"
            )
            raise CanonicalValueError(msg)
        key_text = json.dumps(key_values, ensure_ascii=True, separators=(",", ":"))
        if key_text in keyed:
            msg = (
                f"frame_digest sort key {list(key)} is not unique: {tuple(key_values)!r} "
                "appears more than once, so row order would move the digest"
            )
            raise CanonicalValueError(msg)
        keyed[key_text] = row

    payload = [
        ["columns", [[name, kind] for name, kind in columns]],
        ["sort_by", list(key)],
        ["rows", [keyed[text] for text in sorted(keyed)]],
    ]
    body = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(FRAME_DOMAIN + body).hexdigest()
